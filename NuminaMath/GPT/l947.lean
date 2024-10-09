import Mathlib

namespace solve_system_of_equations_l947_94756

theorem solve_system_of_equations :
  (∃ x y : ℝ, (x / y + y / x = 173 / 26) ∧ (1 / x + 1 / y = 15 / 26) ∧ ((x = 13 ∧ y = 2) ∨ (x = 2 ∧ y = 13))) :=
by
  sorry

end solve_system_of_equations_l947_94756


namespace find_A_l947_94720

theorem find_A (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 2 * x ^ 2 - 13 * x + 10 ≠ 0 → 1 / (x ^ 3 - 2 * x ^ 2 - 13 * x + 10) = A / (x + 2) + B / (x - 1) + C / (x - 1) ^ 2)
  → A = 1 / 9 := 
sorry

end find_A_l947_94720


namespace y_intercept_of_line_l947_94774

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by sorry

end y_intercept_of_line_l947_94774


namespace tenth_term_is_correct_l947_94762

-- Definitions corresponding to the problem conditions
def sequence_term (n : ℕ) : ℚ := (-1)^n * (2 * n + 1) / (n^2 + 1)

-- Theorem statement for the equivalent proof problem
theorem tenth_term_is_correct : sequence_term 10 = 21 / 101 := by sorry

end tenth_term_is_correct_l947_94762


namespace find_possible_values_l947_94788

def real_number_y (y : ℝ) := (3 < y ∧ y < 4)

theorem find_possible_values (y : ℝ) (h : real_number_y y) : 
  42 < (y^2 + 7*y + 12) ∧ (y^2 + 7*y + 12) < 56 := 
sorry

end find_possible_values_l947_94788


namespace cafeteria_extra_fruits_l947_94732

theorem cafeteria_extra_fruits (red_apples green_apples bananas oranges students : ℕ) (fruits_per_student : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : bananas = 17)
  (h4 : oranges = 12)
  (h5 : students = 21)
  (h6 : fruits_per_student = 2) :
  (red_apples + green_apples + bananas + oranges) - (students * fruits_per_student) = 43 :=
by
  sorry

end cafeteria_extra_fruits_l947_94732


namespace find_m_n_l947_94750

theorem find_m_n :
  ∀ (m n : ℤ), (∀ x : ℤ, (x - 4) * (x + 8) = x^2 + m * x + n) → 
  (m = 4 ∧ n = -32) :=
by
  intros m n h
  let x := 0
  sorry

end find_m_n_l947_94750


namespace total_cost_shoes_and_jerseys_l947_94754

theorem total_cost_shoes_and_jerseys 
  (shoes : ℕ) (jerseys : ℕ) (cost_shoes : ℕ) (cost_jersey : ℕ) 
  (cost_total_shoes : ℕ) (cost_per_shoe : ℕ) (cost_per_jersey : ℕ) 
  (h1 : shoes = 6)
  (h2 : jerseys = 4) 
  (h3 : cost_per_jersey = cost_per_shoe / 4)
  (h4 : cost_total_shoes = 480)
  (h5 : cost_per_shoe = cost_total_shoes / shoes)
  (h6 : cost_per_jersey = cost_per_shoe / 4)
  (total_cost : ℕ) 
  (h7 : total_cost = cost_total_shoes + cost_per_jersey * jerseys) :
  total_cost = 560 :=
sorry

end total_cost_shoes_and_jerseys_l947_94754


namespace tangent_points_l947_94758

noncomputable def f (x : ℝ) : ℝ := x^3 + 1
def P : ℝ × ℝ := (-2, 1)
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_points (x0 : ℝ) (y0 : ℝ) (hP : P = (-2, 1)) (hf : y0 = f x0) :
  (3 * x0^2 = (y0 - 1) / (x0 + 2)) → (x0 = 0 ∨ x0 = -3) :=
by
  sorry

end tangent_points_l947_94758


namespace bounded_expression_l947_94796

theorem bounded_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := 
sorry

end bounded_expression_l947_94796


namespace geometric_seq_common_ratio_l947_94769

theorem geometric_seq_common_ratio (a_n : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (hS3 : S 3 = a_n 1 * (1 - q ^ 3) / (1 - q))
  (hS2 : S 2 = a_n 1 * (1 - q ^ 2) / (1 - q))
  (h : S 3 + 3 * S 2 = 0) 
  (hq_not_one : q ≠ 1) :
  q = -2 :=
by sorry

end geometric_seq_common_ratio_l947_94769


namespace sum_of_midpoint_coordinates_l947_94781

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := -1
  let x2 := 11
  let y2 := 21
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 17 := by
  sorry

end sum_of_midpoint_coordinates_l947_94781


namespace values_of_x0_l947_94772

noncomputable def x_seq (x_0 : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => x_0
  | n + 1 => if 3 * (x_seq x_0 n) < 1 then 3 * (x_seq x_0 n)
             else if 3 * (x_seq x_0 n) < 2 then 3 * (x_seq x_0 n) - 1
             else 3 * (x_seq x_0 n) - 2

theorem values_of_x0 (x_0 : ℝ) (h : 0 ≤ x_0 ∧ x_0 < 1) :
  (∃! x_0, x_0 = x_seq x_0 6) → (x_seq x_0 6 = x_0) :=
  sorry

end values_of_x0_l947_94772


namespace max_log_sum_value_l947_94735

noncomputable def max_log_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4 * y = 40) : ℝ :=
  Real.log x + Real.log y

theorem max_log_sum_value : ∀ (x y : ℝ), x > 0 → y > 0 → x + 4 * y = 40 → max_log_sum x y sorry sorry sorry = 2 :=
by
  intro x y h1 h2 h3
  sorry

end max_log_sum_value_l947_94735


namespace total_workers_count_l947_94797

theorem total_workers_count 
  (W N : ℕ)
  (h1 : (W : ℝ) * 9000 = 7 * 12000 + N * 6000)
  (h2 : W = 7 + N) 
  : W = 14 :=
sorry

end total_workers_count_l947_94797


namespace find_c_l947_94714

theorem find_c (x y c : ℝ) (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = c^x * 27^y)
  (h2 : x + y = 4) : c = 49 :=
by
  sorry

end find_c_l947_94714


namespace fraction_to_terminating_decimal_l947_94700

theorem fraction_to_terminating_decimal :
  (45 : ℚ) / 64 = (703125 : ℚ) / 1000000 := by
  sorry

end fraction_to_terminating_decimal_l947_94700


namespace find_x_l947_94768

-- Definitions based directly on conditions
def vec_a : ℝ × ℝ := (2, 4)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 3)
def vec_c (x : ℝ) : ℝ × ℝ := (2 - x, 1)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematically equivalent proof problem statement
theorem find_x (x : ℝ) : dot_product (vec_c x) (vec_b x) = 0 → (x = -1 ∨ x = 3) :=
by
  -- Placeholder for the proof
  sorry

end find_x_l947_94768


namespace pat_earns_per_photo_l947_94794

-- Defining conditions
def minutes_per_shark := 10
def fuel_cost_per_hour := 50
def hunting_hours := 5
def expected_profit := 200

-- Defining intermediate calculations based on the conditions
def sharks_per_hour := 60 / minutes_per_shark
def total_sharks := sharks_per_hour * hunting_hours
def total_fuel_cost := fuel_cost_per_hour * hunting_hours
def total_earnings := expected_profit + total_fuel_cost
def earnings_per_photo := total_earnings / total_sharks

-- Main theorem: Prove that Pat earns $15 for each photo
theorem pat_earns_per_photo : earnings_per_photo = 15 := by
  -- The proof would be here
  sorry

end pat_earns_per_photo_l947_94794


namespace largest_prime_factor_5985_l947_94745

theorem largest_prime_factor_5985 : ∃ p, Nat.Prime p ∧ p ∣ 5985 ∧ ∀ q, Nat.Prime q ∧ q ∣ 5985 → q ≤ p :=
sorry

end largest_prime_factor_5985_l947_94745


namespace trigonometric_identity_l947_94761

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.sin (α + π / 3) = 12 / 13) 
  : Real.cos (π / 6 - α) = 12 / 13 := 
sorry

end trigonometric_identity_l947_94761


namespace num_valid_arrangements_l947_94789

-- Define the people and the days of the week
inductive Person := | A | B | C | D | E
inductive DayOfWeek := | Monday | Tuesday | Wednesday | Thursday | Friday

-- Define the arrangement function type
def Arrangement := DayOfWeek → Person

/-- The total number of valid arrangements for 5 people
    (A, B, C, D, E) on duty from Monday to Friday such that:
    - A and B are not on duty on adjacent days,
    - B and C are on duty on adjacent days,
    is 36.
-/
theorem num_valid_arrangements : 
  ∃ (arrangements : Finset (Arrangement)), arrangements.card = 36 ∧
  (∀ (x : Arrangement), x ∈ arrangements →
    (∀ (d1 d2 : DayOfWeek), 
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) →
      ¬(x d1 = Person.A ∧ x d2 = Person.B)) ∧
    (∃ (d1 d2 : DayOfWeek),
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) ∧
      (x d1 = Person.B ∧ x d2 = Person.C)))
  := sorry

end num_valid_arrangements_l947_94789


namespace distribution_centers_count_l947_94711

theorem distribution_centers_count (n : ℕ) (h : n = 5) : n + (n * (n - 1)) / 2 = 15 :=
by
  subst h -- replace n with 5
  show 5 + (5 * (5 - 1)) / 2 = 15
  have : (5 * 4) / 2 = 10 := by norm_num
  show 5 + 10 = 15
  norm_num

end distribution_centers_count_l947_94711


namespace equilateral_triangle_area_percentage_l947_94744

noncomputable def percentage_area_of_triangle_in_pentagon (s : ℝ) : ℝ :=
  ((4 * Real.sqrt 3 - 3) / 13) * 100

theorem equilateral_triangle_area_percentage
  (s : ℝ) :
  let pentagon_area := s^2 * (1 + Real.sqrt 3 / 4)
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  (triangle_area / pentagon_area) * 100 = percentage_area_of_triangle_in_pentagon s :=
by
  sorry

end equilateral_triangle_area_percentage_l947_94744


namespace geometric_progression_coincides_arithmetic_l947_94765

variables (a d q : ℝ)
variables (ap : ℕ → ℝ) (gp : ℕ → ℝ)

-- Define the N-th term of the arithmetic progression
def nth_term_ap (n : ℕ) : ℝ := a + n * d

-- Define the N-th term of the geometric progression
def nth_term_gp (n : ℕ) : ℝ := a * q^n

theorem geometric_progression_coincides_arithmetic :
  gp 3 = ap 10 →
  gp 4 = ap 74 :=
by
  intro h
  sorry

end geometric_progression_coincides_arithmetic_l947_94765


namespace payal_finished_fraction_l947_94739

-- Define the conditions
variables (x : ℕ)

-- Given conditions
-- 1. Total pages in the book
def total_pages : ℕ := 60
-- 2. Payal has finished 20 more pages than she has yet to read.
def pages_yet_to_read (x : ℕ) : ℕ := x - 20

-- Main statement to prove: the fraction of the pages finished is 2/3
theorem payal_finished_fraction (h : x + (x - 20) = 60) : (x : ℚ) / 60 = 2 / 3 :=
sorry

end payal_finished_fraction_l947_94739


namespace problem_l947_94705

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1) * d) / 2

theorem problem (a1 S3 : ℕ) (a1_eq : a1 = 2) (S3_eq : S3 = 12) : 
  ∃ a6 : ℕ, a6 = 12 := by
  let a2 := (S3 - a1) / 2
  let d := a2 - a1
  let a6 := a1 + 5 * d
  use a6
  sorry

end problem_l947_94705


namespace part1_part2_l947_94717

noncomputable def triangle_area (A B C : ℝ) (a b c : ℝ) : ℝ :=
  1/2 * a * c * Real.sin B

theorem part1 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  triangle_area A B C a b c = Real.sqrt 3 :=
sorry

theorem part2 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  ∃ B, 
  (B = 2 * π / 3) ∧ (4 * Real.sin C^2 + 3 * Real.sin A^2 + 2) / (Real.sin B^2) = 5 :=
sorry

end part1_part2_l947_94717


namespace linear_function_through_origin_l947_94741

theorem linear_function_through_origin (k : ℝ) (h : ∃ x y : ℝ, (x = 0 ∧ y = 0) ∧ y = (k - 2) * x + (k^2 - 4)) : k = -2 :=
by
  sorry

end linear_function_through_origin_l947_94741


namespace canoe_stream_speed_l947_94784

theorem canoe_stream_speed (C S : ℝ) (h1 : C - S = 9) (h2 : C + S = 12) : S = 1.5 :=
by
  sorry

end canoe_stream_speed_l947_94784


namespace sheep_count_l947_94798

theorem sheep_count (S H : ℕ) (h1 : S / H = 3 / 7) (h2 : H * 230 = 12880) : S = 24 :=
by
  sorry

end sheep_count_l947_94798


namespace inequality_solution_set_l947_94709

theorem inequality_solution_set (a : ℝ) : (-16 < a ∧ a ≤ 0) ↔ (∀ x : ℝ, a * x^2 + a * x - 4 < 0) :=
by
  sorry

end inequality_solution_set_l947_94709


namespace same_solution_set_l947_94748

theorem same_solution_set :
  (∀ x : ℝ, (x - 1) / (x - 2) ≤ 0 ↔ (x^3 - x^2 + x - 1) / (x - 2) ≤ 0) :=
sorry

end same_solution_set_l947_94748


namespace solve_equation_one_solve_equation_two_l947_94719

theorem solve_equation_one (x : ℝ) : 3 * x + 7 = 32 - 2 * x → x = 5 :=
by
  intro h
  sorry

theorem solve_equation_two (x : ℝ) : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1 → x = -1 :=
by
  intro h
  sorry

end solve_equation_one_solve_equation_two_l947_94719


namespace total_amount_spent_l947_94751

-- Define the prices of the CDs
def price_life_journey : ℕ := 100
def price_day_life : ℕ := 50
def price_when_rescind : ℕ := 85

-- Define the discounted price for The Life Journey CD
def discount_life_journey : ℕ := 20 -- 20% discount equivalent to $20
def discounted_price_life_journey : ℕ := price_life_journey - discount_life_journey

-- Define the number of CDs bought
def num_life_journey : ℕ := 3
def num_day_life : ℕ := 4
def num_when_rescind : ℕ := 2

-- Define the function to calculate money spent on each type with offers in consideration
def cost_life_journey : ℕ := num_life_journey * discounted_price_life_journey
def cost_day_life : ℕ := (num_day_life / 2) * price_day_life -- Buy one get one free offer
def cost_when_rescind : ℕ := num_when_rescind * price_when_rescind

-- Calculate the total cost
def total_cost := cost_life_journey + cost_day_life + cost_when_rescind

-- Define Lean theorem to prove the total cost
theorem total_amount_spent : total_cost = 510 :=
  by
    -- Skipping the actual proof as the prompt specifies
    sorry

end total_amount_spent_l947_94751


namespace mandy_yoga_time_l947_94792

theorem mandy_yoga_time (G B Y : ℕ) (h1 : 2 * B = 3 * G) (h2 : 3 * Y = 2 * (G + B)) (h3 : Y = 30) : Y = 30 := by
  sorry

end mandy_yoga_time_l947_94792


namespace spaceship_travel_distance_l947_94710

-- Define each leg of the journey
def distance1 := 0.5
def distance2 := 0.1
def distance3 := 0.1

-- Define the total distance traveled
def total_distance := distance1 + distance2 + distance3

-- The statement to prove
theorem spaceship_travel_distance : total_distance = 0.7 := sorry

end spaceship_travel_distance_l947_94710


namespace first_place_friend_distance_friend_running_distance_l947_94793

theorem first_place_friend_distance (distance_mina_finish : ℕ) (halfway_condition : ∀ x, x = distance_mina_finish / 2) :
  (∃ y, y = distance_mina_finish / 2) :=
by
  sorry

-- Given conditions
def distance_mina_finish : ℕ := 200
noncomputable def first_place_friend_position := distance_mina_finish / 2

-- The theorem we need to prove
theorem friend_running_distance : first_place_friend_position = 100 :=
by
  sorry

end first_place_friend_distance_friend_running_distance_l947_94793


namespace max_plus_min_l947_94757

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂ - 2016
axiom condition2 (x : ℝ) : x > 0 → f x > 2016

theorem max_plus_min (M N : ℝ) (hM : M = f 2016) (hN : N = f (-2016)) : M + N = 4032 :=
by
  sorry

end max_plus_min_l947_94757


namespace Cathy_wins_l947_94787

theorem Cathy_wins (n k : ℕ) (hn : n > 0) (hk : k > 0) : (∃ box_count : ℕ, box_count = 1) :=
  if h : n ≤ 2^(k-1) then
    sorry
  else
    sorry

end Cathy_wins_l947_94787


namespace modulo_arithmetic_l947_94726

theorem modulo_arithmetic :
  (222 * 15 - 35 * 9 + 2^3) % 18 = 17 :=
by
  sorry

end modulo_arithmetic_l947_94726


namespace weighted_valid_votes_l947_94766

theorem weighted_valid_votes :
  let total_votes := 10000
  let invalid_vote_rate := 0.25
  let valid_votes := total_votes * (1 - invalid_vote_rate)
  let v_b := (valid_votes - 2 * (valid_votes * 0.15 + valid_votes * 0.07) + valid_votes * 0.05) / 4
  let v_a := v_b + valid_votes * 0.15
  let v_c := v_a + valid_votes * 0.07
  let v_d := v_b - valid_votes * 0.05
  let weighted_votes_A := v_a * 3.0
  let weighted_votes_B := v_b * 2.5
  let weighted_votes_C := v_c * 2.75
  let weighted_votes_D := v_d * 2.25
  weighted_votes_A = 7200 ∧
  weighted_votes_B = 3187.5 ∧
  weighted_votes_C = 8043.75 ∧
  weighted_votes_D = 2025 :=
by
  sorry

end weighted_valid_votes_l947_94766


namespace radius_of_circle_l947_94736

theorem radius_of_circle (r : ℝ) (h : 6 * Real.pi * r + 6 = 2 * Real.pi * r^2) : 
  r = (3 + Real.sqrt 21) / 2 :=
by
  sorry

end radius_of_circle_l947_94736


namespace exists_permutation_with_large_neighbor_difference_l947_94746

theorem exists_permutation_with_large_neighbor_difference :
  ∃ (σ : Fin 100 → Fin 100), 
    (∀ (i : Fin 99), (|σ i.succ - σ i| ≥ 50)) :=
sorry

end exists_permutation_with_large_neighbor_difference_l947_94746


namespace find_number_l947_94718

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l947_94718


namespace find_z_l947_94724

theorem find_z (z : ℝ) (v : ℝ × ℝ × ℝ) (u : ℝ × ℝ × ℝ)
  (h_v : v = (4, 1, z)) (h_u : u = (2, -3, 4))
  (h_eq : (4 * 2 + 1 * -3 + z * 4) / (2 * 2 + -3 * -3 + 4 * 4) = 5 / 29) :
  z = 0 :=
by
  sorry

end find_z_l947_94724


namespace goods_train_passes_man_in_10_seconds_l947_94730

def goods_train_pass_time (man_speed_kmph goods_speed_kmph goods_length_m : ℕ) : ℕ :=
  let relative_speed_mps := (man_speed_kmph + goods_speed_kmph) * 1000 / 3600
  goods_length_m / relative_speed_mps

theorem goods_train_passes_man_in_10_seconds :
  goods_train_pass_time 55 60 320 = 10 := sorry

end goods_train_passes_man_in_10_seconds_l947_94730


namespace room_length_l947_94770

theorem room_length (L : ℝ) (width : ℝ := 4) (total_cost : ℝ := 20900) (rate : ℝ := 950) :
  L * width = total_cost / rate → L = 5.5 :=
by
  sorry

end room_length_l947_94770


namespace sin_cos_from_tan_l947_94738

variable {α : Real} (hα : α > 0) -- Assume α is an acute angle

theorem sin_cos_from_tan (h : Real.tan α = 2) : 
  Real.sin α = 2 / Real.sqrt 5 ∧ Real.cos α = 1 / Real.sqrt 5 := 
by sorry

end sin_cos_from_tan_l947_94738


namespace constant_term_equality_l947_94790

theorem constant_term_equality (a : ℝ) 
  (h1 : ∃ T, T = (x : ℝ)^2 + 2 / x ∧ T^9 = 64 * ↑(Nat.choose 9 6)) 
  (h2 : ∃ T, T = (x : ℝ) + a / (x^2) ∧ T^9 = a^3 * ↑(Nat.choose 9 3)):
  a = 4 := 
sorry

end constant_term_equality_l947_94790


namespace monomial_2015_l947_94727

def a (n : ℕ) : ℤ := (-1 : ℤ)^n * (2 * n - 1)

theorem monomial_2015 :
  a 2015 * (x : ℤ) ^ 2015 = -4029 * (x : ℤ) ^ 2015 :=
by
  sorry

end monomial_2015_l947_94727


namespace smallest_solution_of_equation_l947_94786

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (9 * x^2 - 45 * x + 50 = 0) ∧ (∀ y : ℝ, 9 * y^2 - 45 * y + 50 = 0 → x ≤ y) :=
sorry

end smallest_solution_of_equation_l947_94786


namespace verify_quadratic_solution_l947_94708

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def quadratic_roots : Prop :=
  ∃ (p q : ℕ) (x1 x2 : ℤ), is_prime p ∧ is_prime q ∧ 
  (x1 + x2 = -(p : ℤ)) ∧ (x1 * x2 = (3 * q : ℤ)) ∧ x1 < 0 ∧ x2 < 0 ∧ 
  ((p = 7 ∧ q = 2) ∨ (p = 5 ∧ q = 2))

theorem verify_quadratic_solution : quadratic_roots :=
  by {
    sorry
  }

end verify_quadratic_solution_l947_94708


namespace swapped_two_digit_number_l947_94777

variable (a : ℕ)

theorem swapped_two_digit_number (h : a < 10) (sum_digits : ∃ t : ℕ, t + a = 13) : 
    ∃ n : ℕ, n = 9 * a + 13 :=
by
  sorry

end swapped_two_digit_number_l947_94777


namespace josef_game_l947_94764

theorem josef_game : 
  ∃ S : Finset ℕ, 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 1440 ∧ 1440 % n = 0 ∧ n % 5 = 0) ∧ 
    S.card = 18 := sorry

end josef_game_l947_94764


namespace inequality_pow_gt_linear_l947_94725

theorem inequality_pow_gt_linear {a : ℝ} (n : ℕ) (h₁ : a > -1) (h₂ : a ≠ 0) (h₃ : n ≥ 2) :
  (1 + a:ℝ)^n > 1 + n * a :=
sorry

end inequality_pow_gt_linear_l947_94725


namespace buttons_ratio_l947_94767

theorem buttons_ratio
  (initial_buttons : ℕ)
  (shane_multiplier : ℕ)
  (final_buttons : ℕ)
  (total_buttons_after_shane : ℕ) :
  initial_buttons = 14 →
  shane_multiplier = 3 →
  final_buttons = 28 →
  total_buttons_after_shane = initial_buttons + shane_multiplier * initial_buttons →
  (total_buttons_after_shane - final_buttons) / total_buttons_after_shane = 1 / 2 :=
by
  intros
  sorry

end buttons_ratio_l947_94767


namespace max_comic_books_l947_94775

namespace JasmineComicBooks

-- Conditions
def total_money : ℝ := 12.50
def comic_book_cost : ℝ := 1.15

-- Statement of the theorem
theorem max_comic_books (n : ℕ) (h : n * comic_book_cost ≤ total_money) : n ≤ 10 := by
  sorry

end JasmineComicBooks

end max_comic_books_l947_94775


namespace initial_milk_quantity_l947_94799

theorem initial_milk_quantity 
  (milk_left_in_tank : ℕ) -- the remaining milk in the tank
  (pumping_rate : ℕ) -- the rate at which milk was pumped out
  (pumping_hours : ℕ) -- hours during which milk was pumped out
  (adding_rate : ℕ) -- the rate at which milk was added
  (adding_hours : ℕ) -- hours during which milk was added 
  (initial_milk : ℕ) -- initial milk collected
  (h1 : milk_left_in_tank = 28980) -- condition 3
  (h2 : pumping_rate = 2880) -- condition 1 (rate)
  (h3 : pumping_hours = 4) -- condition 1 (hours)
  (h4 : adding_rate = 1500) -- condition 2 (rate)
  (h5 : adding_hours = 7) -- condition 2 (hours)
  : initial_milk = 30000 :=
by
  sorry

end initial_milk_quantity_l947_94799


namespace exists_multiple_of_prime_with_all_nines_digits_l947_94771

theorem exists_multiple_of_prime_with_all_nines_digits (p : ℕ) (hp_prime : Nat.Prime p) (h2 : p ≠ 2) (h5 : p ≠ 5) :
  ∃ n : ℕ, (∀ d ∈ (n.digits 10), d = 9) ∧ p ∣ n :=
by
  sorry

end exists_multiple_of_prime_with_all_nines_digits_l947_94771


namespace price_reduction_equation_l947_94707

variable (x : ℝ)

theorem price_reduction_equation :
    (58 * (1 - x)^2 = 43) :=
sorry

end price_reduction_equation_l947_94707


namespace neg_abs_nonneg_l947_94740

theorem neg_abs_nonneg :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by
  sorry

end neg_abs_nonneg_l947_94740


namespace solve_for_x_l947_94776

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = 1 / 3 * (6 * x + 45)) : x = 0 :=
sorry

end solve_for_x_l947_94776


namespace min_k_l947_94713

noncomputable 
def f (k : ℕ) (x : ℝ) : ℝ := 
  (Real.sin (k * x / 10)) ^ 4 + (Real.cos (k * x / 10)) ^ 4

theorem min_k (k : ℕ) 
    (h : (∀ a : ℝ, {y | ∃ x : ℝ, a < x ∧ x < a+1 ∧ y = f k x} = 
                  {y | ∃ x : ℝ, y = f k x})) 
    : k ≥ 16 :=
by
  sorry

end min_k_l947_94713


namespace number_of_clients_l947_94742

theorem number_of_clients (num_cars num_selections_per_car num_cars_per_client total_selections num_clients : ℕ)
  (h1 : num_cars = 15)
  (h2 : num_selections_per_car = 3)
  (h3 : num_cars_per_client = 3)
  (h4 : total_selections = num_cars * num_selections_per_car)
  (h5 : num_clients = total_selections / num_cars_per_client) :
  num_clients = 15 := 
by
  sorry

end number_of_clients_l947_94742


namespace intersection_complement_l947_94729

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {1, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 3} := by
  sorry

end intersection_complement_l947_94729


namespace trevor_eggs_l947_94722

theorem trevor_eggs :
  let gertrude := 4
  let blanche := 3
  let nancy := 2
  let martha := 2
  let ophelia := 5
  let penelope := 1
  let quinny := 3
  let dropped := 2
  let gifted := 3
  let total_collected := gertrude + blanche + nancy + martha + ophelia + penelope + quinny
  let remaining_after_drop := total_collected - dropped
  let final_eggs := remaining_after_drop - gifted
  final_eggs = 15 := by
    sorry

end trevor_eggs_l947_94722


namespace general_formula_arithmetic_sequence_l947_94752

variable (a : ℕ → ℤ)

def isArithmeticSequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_formula_arithmetic_sequence :
  isArithmeticSequence a →
  a 5 = 9 →
  a 1 + a 7 = 14 →
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  intros h_seq h_a5 h_a17
  sorry

end general_formula_arithmetic_sequence_l947_94752


namespace alpha_range_in_first_quadrant_l947_94731

open Real

theorem alpha_range_in_first_quadrant (k : ℤ) (α : ℝ) 
  (h1 : cos α ≤ sin α) : 
  (2 * k * π + π / 4) ≤ α ∧ α < (2 * k * π + π / 2) :=
sorry

end alpha_range_in_first_quadrant_l947_94731


namespace rectangular_solid_surface_area_l947_94779

-- Definitions based on conditions
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rectangular_solid (a b c : ℕ) :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a * b * c = 231

noncomputable def surface_area (a b c : ℕ) := 2 * (a * b + b * c + c * a)

-- Main theorem based on question and answer
theorem rectangular_solid_surface_area :
  ∃ (a b c : ℕ), rectangular_solid a b c ∧ surface_area a b c = 262 := by
  sorry

end rectangular_solid_surface_area_l947_94779


namespace sequence_type_l947_94783

-- Definitions based on the conditions
def Sn (a : ℝ) (n : ℕ) : ℝ := a^n - 1

def sequence_an (a : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a - 1 else (Sn a n - Sn a (n - 1))

-- Proving the mathematical statement
theorem sequence_type (a : ℝ) (h : a ≠ 0) : 
  (∀ n > 1, (sequence_an a n = sequence_an a 1 + (n - 1) * (sequence_an a 2 - sequence_an a 1)) ∨
  (∀ n > 2, sequence_an a n / sequence_an a (n-1) = a)) :=
sorry

end sequence_type_l947_94783


namespace surface_area_of_resulting_solid_l947_94704

-- Define the original cube dimensions
def original_cube_surface_area (s : ℕ) := 6 * s * s

-- Define the smaller cube dimensions to be cut
def small_cube_surface_area (s : ℕ) := 3 * s * s

-- Define the proof problem
theorem surface_area_of_resulting_solid :
  original_cube_surface_area 3 - small_cube_surface_area 1 - small_cube_surface_area 2 + (3 * 1 + 3 * 4) = 54 :=
by
  -- The actual proof is to be filled in here
  sorry

end surface_area_of_resulting_solid_l947_94704


namespace surface_area_correct_l947_94759

def w := 3 -- width in cm
def l := 4 -- length in cm
def h := 5 -- height in cm

def surface_area : Nat := 
  2 * (h * w) + 2 * (l * w) + 2 * (l * h)

theorem surface_area_correct : surface_area = 94 := 
  by
    sorry

end surface_area_correct_l947_94759


namespace sum_of_interior_angles_quadrilateral_l947_94721

-- Define the function for the sum of the interior angles
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Theorem that the sum of the interior angles of a quadrilateral is 360 degrees
theorem sum_of_interior_angles_quadrilateral : sum_of_interior_angles 4 = 360 :=
by
  sorry

end sum_of_interior_angles_quadrilateral_l947_94721


namespace solve_y_minus_x_l947_94702

theorem solve_y_minus_x (x y : ℝ) (h1 : x + y = 399) (h2 : x / y = 0.9) : y - x = 21 :=
sorry

end solve_y_minus_x_l947_94702


namespace maximum_special_points_l947_94716

theorem maximum_special_points (n : ℕ) (h : n = 11) : 
  ∃ p : ℕ, p = 91 := 
sorry

end maximum_special_points_l947_94716


namespace max_min_diff_half_dollars_l947_94701

-- Definitions based only on conditions
variables (a c d : ℕ)

-- Conditions:
def condition1 : Prop := a + c + d = 60
def condition2 : Prop := 5 * a + 25 * c + 50 * d = 1000

-- The mathematically equivalent proof statement
theorem max_min_diff_half_dollars : condition1 a c d → condition2 a c d → (∃ d_min d_max : ℕ, d_min = 0 ∧ d_max = 15 ∧ d_max - d_min = 15) :=
by
  intros
  sorry

end max_min_diff_half_dollars_l947_94701


namespace current_algae_plants_l947_94703

def original_algae_plants : ℕ := 809
def additional_algae_plants : ℕ := 2454

theorem current_algae_plants :
  original_algae_plants + additional_algae_plants = 3263 := by
  sorry

end current_algae_plants_l947_94703


namespace time_to_park_l947_94728

-- distance from house to market in miles
def d_market : ℝ := 5

-- distance from house to park in miles
def d_park : ℝ := 3

-- time to market in minutes
def t_market : ℝ := 30

-- assuming constant speed, calculate time to park
theorem time_to_park : (3 / 5) * 30 = 18 := by
  sorry

end time_to_park_l947_94728


namespace opposite_of_negative_2023_l947_94785

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l947_94785


namespace calculate_sum_and_difference_l947_94749

theorem calculate_sum_and_difference : 0.5 - 0.03 + 0.007 = 0.477 := sorry

end calculate_sum_and_difference_l947_94749


namespace sum_of_five_integers_l947_94733

-- Definitions of the five integers based on the conditions given in the problem
def a := 12345
def b := 23451
def c := 34512
def d := 45123
def e := 51234

-- Statement of the proof problem
theorem sum_of_five_integers :
  a + b + c + d + e = 166665 :=
by
  -- The proof is omitted
  sorry

end sum_of_five_integers_l947_94733


namespace tom_cheaper_than_jane_l947_94715

-- Define constants for Store A
def store_a_full_price : ℝ := 125
def store_a_discount_one : ℝ := 0.08
def store_a_discount_two : ℝ := 0.12
def store_a_tax : ℝ := 0.07

-- Define constants for Store B
def store_b_full_price : ℝ := 130
def store_b_discount_one : ℝ := 0.10
def store_b_discount_three : ℝ := 0.15
def store_b_tax : ℝ := 0.05

-- Define the number of smartphones bought by Tom and Jane
def tom_quantity : ℕ := 2
def jane_quantity : ℕ := 3

-- Define the final amount Tom pays
def final_amount_tom : ℝ :=
  let full_price := tom_quantity * store_a_full_price
  let discount := store_a_discount_two * full_price
  let discounted_price := full_price - discount
  let tax := store_a_tax * discounted_price
  discounted_price + tax

-- Define the final amount Jane pays
def final_amount_jane : ℝ :=
  let full_price := jane_quantity * store_b_full_price
  let discount := store_b_discount_three * full_price
  let discounted_price := full_price - discount
  let tax := store_b_tax * discounted_price
  discounted_price + tax

-- Prove that Tom's total cost is $112.68 cheaper than Jane's total cost
theorem tom_cheaper_than_jane : final_amount_jane - final_amount_tom = 112.68 :=
by
  have tom := final_amount_tom
  have jane := final_amount_jane
  sorry

end tom_cheaper_than_jane_l947_94715


namespace leaks_drain_time_l947_94755

-- Definitions from conditions
def pump_rate : ℚ := 1 / 2 -- tanks per hour
def leak1_rate : ℚ := 1 / 6 -- tanks per hour
def leak2_rate : ℚ := 1 / 9 -- tanks per hour

-- Proof statement
theorem leaks_drain_time : (leak1_rate + leak2_rate)⁻¹ = 3.6 :=
by
  sorry

end leaks_drain_time_l947_94755


namespace problem_statement_l947_94737

variable {a b c d : ℚ}

-- Conditions
axiom h1 : a / b = 3
axiom h2 : b / c = 3 / 4
axiom h3 : c / d = 2 / 3

-- Goal
theorem problem_statement : d / a = 2 / 3 := by
  sorry

end problem_statement_l947_94737


namespace hyperbola_eccentricity_l947_94753

theorem hyperbola_eccentricity (a b : ℝ) (h : a^2 = 4 ∧ b^2 = 3) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 7 / 2 :=
    by
  sorry

end hyperbola_eccentricity_l947_94753


namespace jodi_walked_miles_per_day_l947_94760

theorem jodi_walked_miles_per_day (x : ℕ) 
  (h1 : 6 * x + 12 + 18 + 24 = 60) : 
  x = 1 :=
by
  sorry

end jodi_walked_miles_per_day_l947_94760


namespace distance_from_origin_to_line_l947_94706

theorem distance_from_origin_to_line : 
  let A := 1
  let B := 2
  let C := -5
  let x_0 := 0
  let y_0 := 0
  let distance := |A * x_0 + B * y_0 + C| / (Real.sqrt (A ^ 2 + B ^ 2))
  distance = Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_line_l947_94706


namespace velocity_at_3_velocity_at_4_l947_94778

-- Define the distance as a function of time
def s (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Define the velocity as the derivative of the distance
noncomputable def v (t : ℝ) : ℝ := deriv s t

theorem velocity_at_3 : v 3 = 20 :=
by
  sorry

theorem velocity_at_4 : v 4 = 26 :=
by
  sorry

end velocity_at_3_velocity_at_4_l947_94778


namespace karlsson_weight_l947_94763

variable {F K M : ℕ}

theorem karlsson_weight (h1 : F + K = M + 120) (h2 : K + M = F + 60) : K = 90 := by
  sorry

end karlsson_weight_l947_94763


namespace recording_time_is_one_hour_l947_94791

-- Define the recording interval and number of instances
def recording_interval : ℕ := 5 -- The device records data every 5 seconds
def number_of_instances : ℕ := 720 -- The device recorded 720 instances of data

-- Prove that the total recording time is 1 hour
theorem recording_time_is_one_hour : (recording_interval * number_of_instances) / 3600 = 1 := by
  sorry

end recording_time_is_one_hour_l947_94791


namespace max_beds_120_l947_94743

/-- The dimensions of the park. --/
def park_length : ℕ := 60
def park_width : ℕ := 30

/-- The dimensions of each flower bed. --/
def bed_length : ℕ := 3
def bed_width : ℕ := 5

/-- The available fencing length. --/
def total_fencing : ℕ := 2400

/-- Calculate the largest number of flower beds that can be created. --/
def max_flower_beds (park_length park_width bed_length bed_width total_fencing : ℕ) : ℕ := 
  let n := park_width / bed_width  -- number of beds per column
  let m := park_length / bed_length  -- number of beds per row
  let vertical_fencing := bed_width * (n - 1) * m
  let horizontal_fencing := bed_length * (m - 1) * n
  if vertical_fencing + horizontal_fencing <= total_fencing then n * m else 0

theorem max_beds_120 : max_flower_beds 60 30 3 5 2400 = 120 := by
  unfold max_flower_beds
  rfl

end max_beds_120_l947_94743


namespace tricycles_count_l947_94782

theorem tricycles_count {s t : Nat} (h1 : s + t = 10) (h2 : 2 * s + 3 * t = 26) : t = 6 :=
sorry

end tricycles_count_l947_94782


namespace Bing_max_games_l947_94712

/-- 
  Jia, Yi, and Bing play table tennis with the following rules: each game is played between two 
  people, and the loser gives way to the third person. If Jia played 10 games and Yi played 
  7 games, then Bing can play at most 13 games; and can win at most 10 games.
-/
theorem Bing_max_games 
  (games_played_Jia : ℕ)
  (games_played_Yi : ℕ)
  (games_played_Bing : ℕ)
  (games_won_Bing  : ℕ)
  (hJia : games_played_Jia = 10)
  (hYi : games_played_Yi = 7) :
  (games_played_Bing ≤ 13) ∧ (games_won_Bing ≤ 10) := 
sorry

end Bing_max_games_l947_94712


namespace cost_price_of_computer_table_l947_94723

variable (C : ℝ) (SP : ℝ)
variable (h1 : SP = 5400)
variable (h2 : SP = C * 1.32)

theorem cost_price_of_computer_table : C = 5400 / 1.32 :=
by
  -- We are required to prove C = 5400 / 1.32
  sorry

end cost_price_of_computer_table_l947_94723


namespace number_of_free_ranging_chickens_is_105_l947_94747

namespace ChickenProblem

-- Conditions as definitions
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def free_ranging_chickens : ℕ := 2 * run_chickens - 4
def total_coop_run_chickens : ℕ := coop_chickens + run_chickens

-- The ratio condition
def ratio_condition : Prop :=
  (coop_chickens + run_chickens) * 5 = free_ranging_chickens * 2

-- Proof Statement
theorem number_of_free_ranging_chickens_is_105 :
  free_ranging_chickens = 105 :=
by {
  sorry
}

end ChickenProblem

end number_of_free_ranging_chickens_is_105_l947_94747


namespace streetlights_each_square_l947_94734

-- Define the conditions
def total_streetlights : Nat := 200
def total_squares : Nat := 15
def unused_streetlights : Nat := 20

-- State the question mathematically
def streetlights_installed := total_streetlights - unused_streetlights
def streetlights_per_square := streetlights_installed / total_squares

-- The theorem we need to prove
theorem streetlights_each_square : streetlights_per_square = 12 := sorry

end streetlights_each_square_l947_94734


namespace area_of_triangle_from_squares_l947_94795

theorem area_of_triangle_from_squares :
  ∃ (a b c : ℕ), (a = 15 ∧ b = 15 ∧ c = 6 ∧ (1/2 : ℚ) * a * c = 45) :=
by
  let a := 15
  let b := 15
  let c := 6
  have h1 : (1/2 : ℚ) * a * c = 45 := sorry
  exact ⟨a, b, c, ⟨rfl, rfl, rfl, h1⟩⟩

end area_of_triangle_from_squares_l947_94795


namespace cos_double_angle_l947_94773

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 1/4) : Real.cos (2 * theta) = -7/8 :=
by
  sorry

end cos_double_angle_l947_94773


namespace number_of_ways_to_choose_a_pair_of_socks_l947_94780

-- Define the number of socks of each color
def white_socks := 5
def brown_socks := 5
def blue_socks := 5
def green_socks := 5

-- Define the total number of socks
def total_socks := white_socks + brown_socks + blue_socks + green_socks

-- Define the number of ways to choose 2 blue socks from 5 blue socks
def num_ways_choose_two_blue_socks : ℕ := Nat.choose blue_socks 2

-- The proof statement
theorem number_of_ways_to_choose_a_pair_of_socks :
  num_ways_choose_two_blue_socks = 10 :=
sorry

end number_of_ways_to_choose_a_pair_of_socks_l947_94780
