import Mathlib

namespace solve_fractional_equation_l753_75355

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) : 
  (2 * x) / (x - 1) = x / (3 * (x - 1)) + 1 ↔ x = -3 / 2 :=
by sorry

end solve_fractional_equation_l753_75355


namespace LCM_20_45_75_is_900_l753_75394

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l753_75394


namespace problem_l753_75305

theorem problem (x y z : ℝ) 
  (h1 : (x - 4)^2 + (y - 3)^2 + (z - 2)^2 = 0)
  (h2 : 3 * x + 2 * y - z = 12) :
  x + y + z = 9 := 
  sorry

end problem_l753_75305


namespace dot_product_vec_a_vec_b_l753_75347

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem dot_product_vec_a_vec_b : dot_product vec_a vec_b = 1 := by
  sorry

end dot_product_vec_a_vec_b_l753_75347


namespace area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l753_75384

noncomputable def area_enclosed_by_sine_and_line : ℝ :=
  (∫ x in (Real.pi / 6)..(5 * Real.pi / 6), (Real.sin x - 1 / 2))

theorem area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3 :
  area_enclosed_by_sine_and_line = Real.sqrt 3 - Real.pi / 3 := by
  sorry

end area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l753_75384


namespace select_female_athletes_l753_75391

theorem select_female_athletes (males females sample_size total_size : ℕ)
    (h1 : males = 56) (h2 : females = 42) (h3 : sample_size = 28)
    (h4 : total_size = males + females) : 
    (females * sample_size / total_size = 12) := 
by
  sorry

end select_female_athletes_l753_75391


namespace round_trip_time_correct_l753_75342

variables (river_current_speed boat_speed_still_water distance_upstream_distance : ℕ)

def upstream_speed := boat_speed_still_water - river_current_speed
def downstream_speed := boat_speed_still_water + river_current_speed

def time_upstream := distance_upstream_distance / upstream_speed
def time_downstream := distance_upstream_distance / downstream_speed

def round_trip_time := time_upstream + time_downstream

theorem round_trip_time_correct :
  river_current_speed = 10 →
  boat_speed_still_water = 50 →
  distance_upstream_distance = 120 →
  round_trip_time river_current_speed boat_speed_still_water distance_upstream_distance = 5 :=
by
  intros rc bs d
  sorry

end round_trip_time_correct_l753_75342


namespace impossible_to_get_100_pieces_l753_75361

/-- We start with 1 piece of paper. Each time a piece of paper is torn into 3 parts,
it increases the total number of pieces by 2.
Therefore, the number of pieces remains odd through any sequence of tears.
Prove that it is impossible to obtain exactly 100 pieces. -/
theorem impossible_to_get_100_pieces : 
  ∀ n, n = 1 ∨ (∃ k, n = 1 + 2 * k) → n ≠ 100 :=
by
  sorry

end impossible_to_get_100_pieces_l753_75361


namespace range_of_m_l753_75317

variable {R : Type*} [LinearOrderedField R]

def discriminant (a b c : R) := b * b - 4 * a * c

theorem range_of_m (m : R) : (∀ x : R, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by sorry

end range_of_m_l753_75317


namespace positive_difference_of_solutions_is_14_l753_75311

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 5 * x + 15 = x + 55

-- Define the positive difference between solutions of the quadratic equation
def positive_difference (a b : ℝ) : ℝ := |a - b|

-- State the theorem
theorem positive_difference_of_solutions_is_14 : 
  ∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ positive_difference a b = 14 :=
by
  sorry

end positive_difference_of_solutions_is_14_l753_75311


namespace triangle_perimeter_l753_75351

/-- Given the lengths of two sides of a triangle are 1 and 4,
    and the length of the third side is an integer, 
    prove that the perimeter of the triangle is 9 -/
theorem triangle_perimeter
  (a b : ℕ)
  (c : ℤ)
  (h₁ : a = 1)
  (h₂ : b = 4)
  (h₃ : 3 < c ∧ c < 5) :
  a + b + c = 9 :=
by sorry

end triangle_perimeter_l753_75351


namespace draw_at_least_one_even_ball_l753_75324

theorem draw_at_least_one_even_ball:
  -- Let the total number of ordered draws of 4 balls from 15 balls
  let total_draws := 15 * 14 * 13 * 12
  -- Let the total number of ordered draws of 4 balls where all balls are odd (balls 1, 3, ..., 15)
  let odd_draws := 8 * 7 * 6 * 5
  -- The number of valid draws containing at least one even ball
  total_draws - odd_draws = 31080 :=
by
  sorry

end draw_at_least_one_even_ball_l753_75324


namespace ratio_difference_l753_75378

theorem ratio_difference (x : ℕ) (h : (2 * x + 4) * 7 = (3 * x + 4) * 5) : 3 * x - 2 * x = 8 := 
by sorry

end ratio_difference_l753_75378


namespace effective_simple_interest_rate_proof_l753_75326

noncomputable def effective_simple_interest_rate : ℝ :=
  let P := 1
  let r1 := 0.10 / 2 -- Half-yearly rate for year 1
  let t1 := 2 -- number of compounding periods semi-annual
  let A1 := P * (1 + r1) ^ t1

  let r2 := 0.12 / 2 -- Half-yearly rate for year 2
  let t2 := 2
  let A2 := A1 * (1 + r2) ^ t2

  let r3 := 0.14 / 2 -- Half-yearly rate for year 3
  let t3 := 2
  let A3 := A2 * (1 + r3) ^ t3

  let r4 := 0.16 / 2 -- Half-yearly rate for year 4
  let t4 := 2
  let A4 := A3 * (1 + r4) ^ t4

  let CI := 993
  let P_actual := CI / (A4 - P)
  let effective_simple_interest := (CI / P_actual) * 100
  effective_simple_interest

theorem effective_simple_interest_rate_proof :
  effective_simple_interest_rate = 65.48 := by
  sorry

end effective_simple_interest_rate_proof_l753_75326


namespace three_digit_numbers_satisfy_condition_l753_75350

theorem three_digit_numbers_satisfy_condition : 
  ∃ (x y z : ℕ), 
    1 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 ∧ 
    0 ≤ z ∧ z ≤ 9 ∧ 
    x + y + z = (10 * x + y) - (10 * y + z) ∧ 
    (100 * x + 10 * y + z = 209 ∨ 
     100 * x + 10 * y + z = 428 ∨ 
     100 * x + 10 * y + z = 647 ∨ 
     100 * x + 10 * y + z = 866 ∨ 
     100 * x + 10 * y + z = 214 ∨ 
     100 * x + 10 * y + z = 433 ∨ 
     100 * x + 10 * y + z = 652 ∨ 
     100 * x + 10 * y + z = 871) := sorry

end three_digit_numbers_satisfy_condition_l753_75350


namespace shelby_scooter_drive_l753_75392

/-- 
Let y be the time (in minutes) Shelby drove when it was not raining.
Speed when not raining is 25 miles per hour, which is 5/12 mile per minute.
Speed when raining is 15 miles per hour, which is 1/4 mile per minute.
Total distance covered is 18 miles.
Total time taken is 36 minutes.
Prove that Shelby drove for 6 minutes when it was not raining.
-/
theorem shelby_scooter_drive
  (y : ℝ)
  (h_not_raining_speed : ∀ t (h : t = (25/60 : ℝ)), t = (5/12 : ℝ))
  (h_raining_speed : ∀ t (h : t = (15/60 : ℝ)), t = (1/4 : ℝ))
  (h_total_distance : ∀ d (h : d = ((5/12 : ℝ) * y + (1/4 : ℝ) * (36 - y))), d = 18)
  (h_total_time : ∀ t (h : t = 36), t = 36) :
  y = 6 :=
sorry

end shelby_scooter_drive_l753_75392


namespace nine_consecutive_arithmetic_mean_divisible_1111_l753_75396

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l753_75396


namespace bracelet_price_l753_75328

theorem bracelet_price 
  (B : ℝ) -- price of each bracelet
  (H1 : B > 0) 
  (H2 : 3 * B + 2 * 10 + 20 = 100 - 15) : 
  B = 15 :=
by
  sorry

end bracelet_price_l753_75328


namespace selection_methods_l753_75348

/-- Type definition for the workers -/
inductive Worker
  | PliersOnly  : Worker
  | CarOnly     : Worker
  | Both        : Worker

/-- Conditions -/
def num_workers : ℕ := 11
def num_pliers_only : ℕ := 5
def num_car_only : ℕ := 4
def num_both : ℕ := 2
def pliers_needed : ℕ := 4
def car_needed : ℕ := 4

/-- Main statement -/
theorem selection_methods : 
  (num_pliers_only + num_car_only + num_both = num_workers) → 
  (num_pliers_only = 5) → 
  (num_car_only = 4) → 
  (num_both = 2) → 
  (pliers_needed = 4) → 
  (car_needed = 4) → 
  ∃ n : ℕ, n = 185 := 
by 
  sorry -- Proof Skipped

end selection_methods_l753_75348


namespace sum_of_interior_angles_divisible_by_360_l753_75303

theorem sum_of_interior_angles_divisible_by_360
  (n : ℕ)
  (h : n > 0) :
  ∃ k : ℤ, ((2 * n - 2) * 180) = 360 * k :=
by
  sorry

end sum_of_interior_angles_divisible_by_360_l753_75303


namespace wilsons_theorem_l753_75373

theorem wilsons_theorem (p : ℕ) (hp : Nat.Prime p) : (Nat.factorial (p - 1)) % p = p - 1 :=
by
  sorry

end wilsons_theorem_l753_75373


namespace sours_total_l753_75388

variable (c l o T : ℕ)

axiom cherry_sours : c = 32
axiom ratio_cherry_lemon : 4 * l = 5 * c
axiom orange_sours_ratio : o = 25 * T / 100
axiom total_sours : T = c + l + o

theorem sours_total :
  T = 96 :=
by
  sorry

end sours_total_l753_75388


namespace inequality_solution_set_l753_75316

theorem inequality_solution_set :
  { x : ℝ | (10 * x^2 + 20 * x - 68) / ((2 * x - 3) * (x + 4) * (x - 2)) < 3 } =
  { x : ℝ | (-4 < x ∧ x < -2) ∨ (-1 / 3 < x ∧ x < 3 / 2) } :=
by
  sorry

end inequality_solution_set_l753_75316


namespace polyhedron_faces_l753_75379

theorem polyhedron_faces (V E F T P t p : ℕ)
  (hF : F = 20)
  (hFaces : t + p = 20)
  (hTriangles : t = 2 * p)
  (hVertex : T = 2 ∧ P = 2)
  (hEdges : E = (3 * t + 5 * p) / 2)
  (hEuler : V - E + F = 2) :
  100 * P + 10 * T + V = 238 :=
by
  sorry

end polyhedron_faces_l753_75379


namespace right_triangle_angles_l753_75385

theorem right_triangle_angles (a b S : ℝ) (hS : S = 1 / 2 * a * b) (h : (a + b) ^ 2 = 8 * S) :
  ∃ θ₁ θ₂ θ₃ : ℝ, θ₁ = 45 ∧ θ₂ = 45 ∧ θ₃ = 90 :=
by {
  sorry
}

end right_triangle_angles_l753_75385


namespace percentage_of_girl_scouts_with_slips_l753_75374

-- Define the proposition that captures the problem
theorem percentage_of_girl_scouts_with_slips 
    (total_scouts : ℕ)
    (scouts_with_slips : ℕ := total_scouts * 60 / 100)
    (boy_scouts : ℕ := total_scouts * 45 / 100)
    (boy_scouts_with_slips : ℕ := boy_scouts * 50 / 100)
    (girl_scouts : ℕ := total_scouts - boy_scouts)
    (girl_scouts_with_slips : ℕ := scouts_with_slips - boy_scouts_with_slips) :
  (girl_scouts_with_slips * 100 / girl_scouts) = 68 :=
by 
  -- The proof goes here
  sorry

end percentage_of_girl_scouts_with_slips_l753_75374


namespace mrs_sheridan_initial_cats_l753_75383

theorem mrs_sheridan_initial_cats (bought_cats total_cats : ℝ) (h_bought : bought_cats = 43.0) (h_total : total_cats = 54) : total_cats - bought_cats = 11 :=
by
  rw [h_bought, h_total]
  norm_num

end mrs_sheridan_initial_cats_l753_75383


namespace white_truck_percentage_is_17_l753_75382

-- Define the conditions
def total_trucks : ℕ := 50
def total_cars : ℕ := 40
def total_vehicles : ℕ := total_trucks + total_cars

def red_trucks : ℕ := total_trucks / 2
def black_trucks : ℕ := (total_trucks * 20) / 100
def white_trucks : ℕ := total_trucks - red_trucks - black_trucks

def percentage_white_trucks : ℕ := (white_trucks * 100) / total_vehicles

theorem white_truck_percentage_is_17 :
  percentage_white_trucks = 17 :=
  by sorry

end white_truck_percentage_is_17_l753_75382


namespace total_money_shared_l753_75322

-- Let us define the conditions
def ratio (a b c : ℕ) : Prop := ∃ k : ℕ, (2 * k = a) ∧ (3 * k = b) ∧ (8 * k = c)

def olivia_share := 30

-- Our goal is to prove the total amount of money shared
theorem total_money_shared (a b c : ℕ) (h_ratio : ratio a b c) (h_olivia : a = olivia_share) :
    a + b + c = 195 :=
by
  sorry

end total_money_shared_l753_75322


namespace largest_perimeter_regular_polygons_l753_75309

theorem largest_perimeter_regular_polygons :
  ∃ (p q r : ℕ), 
    (p ≥ 3 ∧ q ≥ 3 ∧ r >= 3) ∧
    (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧
    (180 * (p - 2)/p + 180 * (q - 2)/q + 180 * (r - 2)/r = 360) ∧
    ((p + q + r - 6) = 9) :=
sorry

end largest_perimeter_regular_polygons_l753_75309


namespace remainder_2_pow_2015_mod_20_l753_75310

/-- 
  Given that powers of 2 modulo 20 follow a repeating cycle every 4 terms:
  2, 4, 8, 16, 12
  
  Prove that the remainder when \(2^{2015}\) is divided by 20 is 8.
-/
theorem remainder_2_pow_2015_mod_20 : (2 ^ 2015) % 20 = 8 :=
by
  -- The proof is to be filled in.
  sorry

end remainder_2_pow_2015_mod_20_l753_75310


namespace tech_gadgets_components_total_l753_75377

theorem tech_gadgets_components_total (a₁ r n : ℕ) (h₁ : a₁ = 8) (h₂ : r = 3) (h₃ : n = 4) :
  a₁ * (r^n - 1) / (r - 1) = 320 := by
  sorry

end tech_gadgets_components_total_l753_75377


namespace real_roots_quadratic_range_l753_75370

theorem real_roots_quadratic_range (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end real_roots_quadratic_range_l753_75370


namespace exception_to_roots_l753_75343

theorem exception_to_roots (x : ℝ) :
    ¬ (∃ x₀, (x₀ ∈ ({x | x = x} ∩ {x | x = x - 2}))) :=
by sorry

end exception_to_roots_l753_75343


namespace amount_saved_percent_l753_75312

variable (S : ℝ)

theorem amount_saved_percent :
  (0.165 * S) / (0.10 * S) * 100 = 165 := sorry

end amount_saved_percent_l753_75312


namespace student_A_more_stable_l753_75356

-- Define the variances for students A and B
def variance_A : ℝ := 0.05
def variance_B : ℝ := 0.06

-- The theorem to prove that student A has more stable performance
theorem student_A_more_stable : variance_A < variance_B :=
by {
  -- proof goes here
  sorry
}

end student_A_more_stable_l753_75356


namespace find_original_number_l753_75301

theorem find_original_number :
  ∃ x : ℚ, (5 * (3 * x + 15) = 245) ∧ x = 34 / 3 := by
  sorry

end find_original_number_l753_75301


namespace heartsuit_calc_l753_75334

def heartsuit (u v : ℝ) : ℝ := (u + 2*v) * (u - v)

theorem heartsuit_calc : heartsuit 2 (heartsuit 3 4) = -260 := by
  sorry

end heartsuit_calc_l753_75334


namespace ladder_length_difference_l753_75331

theorem ladder_length_difference :
  ∀ (flights : ℕ) (flight_height rope ladder_total_height : ℕ),
    flights = 3 →
    flight_height = 10 →
    rope = (flights * flight_height) / 2 →
    ladder_total_height = 70 →
    ladder_total_height - (flights * flight_height + rope) = 25 →
    ladder_total_height - (flights * flight_height) - rope = 10 :=
by
  intros
  sorry

end ladder_length_difference_l753_75331


namespace lines_region_division_l753_75304

theorem lines_region_division (f : ℕ → ℕ) (k : ℕ) (h : k ≥ 2) : 
  (∀ m, f m = m * (m + 1) / 2 + 1) → f (k + 1) = f k + (k + 1) :=
by
  intro h_f
  have h_base : f 1 = 2 := by sorry
  have h_ih : ∀ n, n ≥ 2 → f (n + 1) = f n + (n + 1) := by sorry
  exact h_ih k h

end lines_region_division_l753_75304


namespace max_piles_660_stones_l753_75307

-- Define the conditions in Lean
def initial_stones := 660

def valid_pile_sizes (piles : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → a ≤ b → b < 2 * a

-- Define the goal statement in Lean
theorem max_piles_660_stones :
  ∃ (piles : List ℕ), (piles.length = 30) ∧ (piles.sum = initial_stones) ∧ valid_pile_sizes piles :=
sorry

end max_piles_660_stones_l753_75307


namespace race_course_length_to_finish_at_same_time_l753_75386

variable (v : ℝ) -- speed of B
variable (d : ℝ) -- length of the race course

-- A's speed is 4 times B's speed and A gives B a 75-meter head start.
theorem race_course_length_to_finish_at_same_time (h1 : v > 0) (h2 : d > 75) : 
  (1 : ℝ) / 4 * (d / v) = ((d - 75) / v) ↔ d = 100 := 
sorry

end race_course_length_to_finish_at_same_time_l753_75386


namespace min_value_am_hm_l753_75390

theorem min_value_am_hm (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end min_value_am_hm_l753_75390


namespace solve_for_square_l753_75344

theorem solve_for_square (x : ℝ) 
  (h : 10 + 9 + 8 * 7 / x + 6 - 5 * 4 - 3 * 2 = 1) : 
  x = 28 := 
by 
  sorry

end solve_for_square_l753_75344


namespace first_term_of_geometric_series_l753_75359

theorem first_term_of_geometric_series (r a S : ℚ) (h_common_ratio : r = -1/5) (h_sum : S = 16) :
  a = 96 / 5 :=
by
  sorry

end first_term_of_geometric_series_l753_75359


namespace sufficient_but_not_necessary_l753_75308

theorem sufficient_but_not_necessary (a : ℝ) : (a = 1 → a^2 = 1) ∧ ¬(a^2 = 1 → a = 1) :=
by
  sorry

end sufficient_but_not_necessary_l753_75308


namespace fraction_walk_home_l753_75365

theorem fraction_walk_home : 
  (1 - ((1 / 2) + (1 / 4) + (1 / 10) + (1 / 8))) = (1 / 40) :=
by 
  sorry

end fraction_walk_home_l753_75365


namespace fat_content_whole_milk_l753_75323

open Real

theorem fat_content_whole_milk :
  ∃ (s w : ℝ), 0 < s ∧ 0 < w ∧
  3 / 100 = 0.75 * s / 100 ∧
  s / 100 = 0.8 * w / 100 ∧
  w = 5 :=
by
  sorry

end fat_content_whole_milk_l753_75323


namespace uncle_kahn_total_cost_l753_75389

noncomputable def base_price : ℝ := 10
noncomputable def child_discount : ℝ := 0.3
noncomputable def senior_discount : ℝ := 0.1
noncomputable def handling_fee : ℝ := 5
noncomputable def discounted_senior_ticket_price : ℝ := 14
noncomputable def num_child_tickets : ℝ := 2
noncomputable def num_senior_tickets : ℝ := 2

theorem uncle_kahn_total_cost :
  let child_ticket_cost := (1 - child_discount) * base_price + handling_fee
  let senior_ticket_cost := discounted_senior_ticket_price
  num_child_tickets * child_ticket_cost + num_senior_tickets * senior_ticket_cost = 52 :=
by
  sorry

end uncle_kahn_total_cost_l753_75389


namespace cube_diagonal_length_l753_75358

theorem cube_diagonal_length
  (side_length : ℝ)
  (h_side_length : side_length = 15) :
  ∃ d : ℝ, d = side_length * Real.sqrt 3 :=
by
  sorry

end cube_diagonal_length_l753_75358


namespace compound_interest_principal_l753_75332

theorem compound_interest_principal 
    (CI : Real)
    (r : Real)
    (n : Nat)
    (t : Nat)
    (A : Real)
    (P : Real) :
  CI = 945.0000000000009 →
  r = 0.10 →
  n = 1 →
  t = 2 →
  A = P * (1 + r / n) ^ (n * t) →
  CI = A - P →
  P = 4500.0000000000045 :=
by intros
   sorry

end compound_interest_principal_l753_75332


namespace primes_solution_l753_75353

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_solution (p : ℕ) (hp : is_prime p) :
  is_prime (p^2 + 2007 * p - 1) ↔ p = 3 :=
by
  sorry

end primes_solution_l753_75353


namespace magnification_factor_is_correct_l753_75387

theorem magnification_factor_is_correct
    (diameter_magnified_image : ℝ)
    (actual_diameter_tissue : ℝ)
    (diameter_magnified_image_eq : diameter_magnified_image = 2)
    (actual_diameter_tissue_eq : actual_diameter_tissue = 0.002) :
  diameter_magnified_image / actual_diameter_tissue = 1000 := by
  -- Theorem and goal statement
  sorry

end magnification_factor_is_correct_l753_75387


namespace intersection_M_N_l753_75349

def M := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def N := { y : ℝ | y > 0 }

theorem intersection_M_N : (M ∩ N) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l753_75349


namespace calculate_expression_l753_75306

theorem calculate_expression :
  500 * 996 * 0.0996 * 20 + 5000 = 997016 :=
by
  sorry

end calculate_expression_l753_75306


namespace intersection_of_A_and_B_l753_75367

open Set

variable (A : Set ℕ) (B : Set ℕ)

theorem intersection_of_A_and_B (hA : A = {0, 1, 2}) (hB : B = {0, 2, 4}) :
  A ∩ B = {0, 2} := by
  sorry

end intersection_of_A_and_B_l753_75367


namespace age_of_teacher_l753_75375

theorem age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) (inc_avg_with_teacher : ℕ) (num_people_with_teacher : ℕ) :
  avg_age_students = 21 →
  num_students = 20 →
  inc_avg_with_teacher = 22 →
  num_people_with_teacher = 21 →
  let total_age_students := num_students * avg_age_students
  let total_age_with_teacher := num_people_with_teacher * inc_avg_with_teacher
  total_age_with_teacher - total_age_students = 42 :=
by
  intros
  sorry

end age_of_teacher_l753_75375


namespace mean_value_of_pentagon_interior_angles_l753_75341

theorem mean_value_of_pentagon_interior_angles :
  let n := 5
  let sum_of_interior_angles := (n - 2) * 180
  let mean_value := sum_of_interior_angles / n
  mean_value = 108 :=
by
  sorry

end mean_value_of_pentagon_interior_angles_l753_75341


namespace sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l753_75369

noncomputable def sin_pi_div_two_plus_2alpha (α : ℝ) : ℝ :=
  Real.sin ((Real.pi / 2) + 2 * α)

def cos_alpha (α : ℝ) := Real.cos α = - (Real.sqrt 2) / 3

theorem sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth (α : ℝ) (h : cos_alpha α) :
  sin_pi_div_two_plus_2alpha α = -5 / 9 :=
sorry

end sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l753_75369


namespace find_some_number_l753_75354

theorem find_some_number : 
  ∃ (some_number : ℝ), (∃ (n : ℝ), n = 54 ∧ (n / some_number) * (n / 162) = 1) → some_number = 18 :=
by
  sorry

end find_some_number_l753_75354


namespace find_x_l753_75357

theorem find_x (x : ℝ) (h : (2 * x) / 16 = 25) : x = 200 :=
sorry

end find_x_l753_75357


namespace unique_elements_set_l753_75345

theorem unique_elements_set (x : ℝ) : x ≠ 3 ∧ x ≠ -1 ∧ x ≠ 0 ↔ 3 ≠ x ∧ x ≠ (x ^ 2 - 2 * x) ∧ (x ^ 2 - 2 * x) ≠ 3 := by
  sorry

end unique_elements_set_l753_75345


namespace liquid_x_percentage_l753_75395

theorem liquid_x_percentage (a_weight b_weight : ℝ) (a_percentage b_percentage : ℝ)
  (result_weight : ℝ) (x_weight_result : ℝ) (x_percentage_result : ℝ) :
  a_weight = 500 → b_weight = 700 → a_percentage = 0.8 / 100 →
  b_percentage = 1.8 / 100 → result_weight = a_weight + b_weight →
  x_weight_result = a_weight * a_percentage + b_weight * b_percentage →
  x_percentage_result = (x_weight_result / result_weight) * 100 →
  x_percentage_result = 1.3833 :=
by sorry

end liquid_x_percentage_l753_75395


namespace value_at_4_value_of_x_when_y_is_0_l753_75380

-- Problem statement
def f (x : ℝ) : ℝ := 2 * x - 3

-- Proof statement 1: When x = 4, y = 5
theorem value_at_4 : f 4 = 5 := sorry

-- Proof statement 2: When y = 0, x = 3/2
theorem value_of_x_when_y_is_0 : (∃ x : ℝ, f x = 0) → (∃ x : ℝ, x = 3 / 2) := sorry

end value_at_4_value_of_x_when_y_is_0_l753_75380


namespace expand_polynomial_l753_75321

theorem expand_polynomial :
  (2 * t^2 - 3 * t + 2) * (-3 * t^2 + t - 5) = -6 * t^4 + 11 * t^3 - 19 * t^2 + 17 * t - 10 :=
by sorry

end expand_polynomial_l753_75321


namespace no_solution_l753_75333

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x)))

theorem no_solution : problem_statement :=
by
  intro x
  have h₁ : ¬(85 + x = 3.5 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  have h₂ : ¬(55 + x = 2 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  exact sorry

end no_solution_l753_75333


namespace M_value_l753_75314

noncomputable def x : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)

noncomputable def y : ℝ := Real.sqrt (4 - 2 * Real.sqrt 3)

noncomputable def M : ℝ := x - y

theorem M_value :
  M = (5 / 2) * Real.sqrt 2 - Real.sqrt 3 + (3 / 2) :=
sorry

end M_value_l753_75314


namespace initial_volume_of_mixture_l753_75318

theorem initial_volume_of_mixture (M W : ℕ) (h1 : 2 * M = 3 * W) (h2 : 4 * M = 3 * (W + 46)) : M + W = 115 := 
sorry

end initial_volume_of_mixture_l753_75318


namespace six_digit_number_divisible_by_eleven_l753_75319

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def reverse_digits (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a

def concatenate_reverse (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a

theorem six_digit_number_divisible_by_eleven (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
  (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) :
  11 ∣ concatenate_reverse a b c :=
by
  sorry

end six_digit_number_divisible_by_eleven_l753_75319


namespace circle_symmetric_point_l753_75339

theorem circle_symmetric_point (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x - 2 * y + b = 0 → x = 2 ∧ y = 1) ∧
  (∀ x y : ℝ, (x, y) ∈ { (px, py) | px = 2 ∧ py = 1 ∨ x + y - 1 = 0 } → x^2 + y^2 + a * x - 2 * y + b = 0) →
  a = 0 ∧ b = -3 := 
by {
  sorry
}

end circle_symmetric_point_l753_75339


namespace modified_expression_range_l753_75372

open Int

theorem modified_expression_range (m : ℤ) :
  ∃ n_min n_max : ℤ, 1 < 4 * n_max + 7 ∧ 4 * n_min + 7 < 60 ∧ (n_max - n_min + 1 = 15) →
  ∃ k_min k_max : ℤ, 1 < m * k_max + 7 ∧ m * k_min + 7 < 60 ∧ (k_max - k_min + 1 ≥ 15) := 
sorry

end modified_expression_range_l753_75372


namespace factorization_correct_l753_75397

theorem factorization_correct (a b : ℝ) : 
  a^2 + 2 * b - b^2 - 1 = (a - b + 1) * (a + b - 1) :=
by
  sorry

end factorization_correct_l753_75397


namespace sine_tangent_coincide_3_decimal_places_l753_75362

open Real

noncomputable def deg_to_rad (d : ℝ) : ℝ := d * (π / 180)

theorem sine_tangent_coincide_3_decimal_places :
  ∀ θ : ℝ,
    0 ≤ θ ∧ θ ≤ deg_to_rad (4 + 20 / 60) →
    |sin θ - tan θ| < 0.0005 :=
by
  intros θ hθ
  sorry

end sine_tangent_coincide_3_decimal_places_l753_75362


namespace LindasTrip_l753_75352

theorem LindasTrip (x : ℝ) :
    (1 / 4) * x + 30 + (1 / 6) * x = x →
    x = 360 / 7 :=
by
  intros h
  sorry

end LindasTrip_l753_75352


namespace tracy_initial_candies_l753_75363

theorem tracy_initial_candies (y : ℕ) 
  (condition1 : y - y / 4 = y * 3 / 4)
  (condition2 : y * 3 / 4 - (y * 3 / 4) / 3 = y / 2)
  (condition3 : y / 2 - 24 = y / 2 - 12 - 12)
  (condition4 : y / 2 - 24 - 4 = 2) : 
  y = 60 :=
by sorry

end tracy_initial_candies_l753_75363


namespace totalNumberOfPeople_l753_75381

def numGirls := 542
def numBoys := 387
def numTeachers := 45
def numStaff := 27

theorem totalNumberOfPeople : numGirls + numBoys + numTeachers + numStaff = 1001 := by
  sorry

end totalNumberOfPeople_l753_75381


namespace total_veg_eaters_l753_75327

def people_eat_only_veg : ℕ := 16
def people_eat_only_nonveg : ℕ := 9
def people_eat_both_veg_and_nonveg : ℕ := 12

theorem total_veg_eaters : people_eat_only_veg + people_eat_both_veg_and_nonveg = 28 := 
by
  sorry

end total_veg_eaters_l753_75327


namespace average_of_remaining_numbers_l753_75302

theorem average_of_remaining_numbers 
  (S S' : ℝ)
  (h1 : S / 12 = 90)
  (h2 : S' = S - 80 - 82) :
  S' / 10 = 91.8 :=
sorry

end average_of_remaining_numbers_l753_75302


namespace max_abs_diff_f_l753_75340

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f (k : ℝ) (h₁ : -3 ≤ k) (h₂ : k ≤ -1) (x₁ x₂ : ℝ) (h₃ : k ≤ x₁) (h₄ : x₁ ≤ k + 2) (h₅ : k ≤ x₂) (h₆ : x₂ ≤ k + 2) :
  |f x₁ - f x₂| ≤ 4 * Real.exp 1 := sorry

end max_abs_diff_f_l753_75340


namespace mean_of_five_numbers_is_correct_l753_75366

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l753_75366


namespace proof_problem_l753_75325

-- Define the problem:
def problem := ∀ (a : Fin 100 → ℝ), 
  (∀ i j, i ≠ j → a i ≠ a j) →  -- All numbers are distinct
  ∃ i : Fin 100, a i + a (⟨i.val + 3, sorry⟩) > a (⟨i.val + 1, sorry⟩) + a (⟨i.val + 2, sorry⟩)
-- Summarize: there exists four consecutive points on the circle such that 
-- the sum of the numbers at the ends is greater than the sum of the numbers in the middle.

theorem proof_problem : problem := sorry

end proof_problem_l753_75325


namespace cos_ratio_l753_75315

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (angle_A angle_B angle_C : ℝ)
variable (bc_coeff : 2 * c = 3 * b)
variable (sin_coeff : Real.sin angle_A = 2 * Real.sin angle_B)

theorem cos_ratio :
  (2 * c = 3 * b) →
  (Real.sin angle_A = 2 * Real.sin angle_B) →
  let cos_A := (b^2 + c^2 - a^2) / (2 * b * c)
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  (Real.cos angle_A / Real.cos angle_B = -2 / 7) :=
by
  intros bc_coeff sin_coeff
  sorry

end cos_ratio_l753_75315


namespace sharmila_hourly_wage_l753_75346

-- Sharmila works 10 hours per day on Monday, Wednesday, and Friday.
def hours_worked_mwf : ℕ := 3 * 10

-- Sharmila works 8 hours per day on Tuesday and Thursday.
def hours_worked_tt : ℕ := 2 * 8

-- Total hours worked in a week.
def total_hours_worked : ℕ := hours_worked_mwf + hours_worked_tt

-- Sharmila earns $460 per week.
def weekly_earnings : ℕ := 460

-- Calculate and prove her hourly wage is $10 per hour.
theorem sharmila_hourly_wage : (weekly_earnings / total_hours_worked) = 10 :=
by sorry

end sharmila_hourly_wage_l753_75346


namespace ellipse_reflection_symmetry_l753_75371

theorem ellipse_reflection_symmetry :
  (∀ x y, (x = -y ∧ y = -x) →
  (∀ a b : ℝ, 
    (a - 3)^2 / 9 + (b - 2)^2 / 4 = 1 ↔
    (b - 3)^2 / 4 + (a - 2)^2 / 9 = 1)
  )
  →
  (∀ x y, 
    ((x + 2)^2 / 9 + (y + 3)^2 / 4 = 1) = 
    (∃ a b : ℝ, 
      (a - 3)^2 / 9 + (b - 2)^2 / 4 = 1 ∧ 
      (a = -y ∧ b = -x))
  ) :=
by
  intros
  sorry

end ellipse_reflection_symmetry_l753_75371


namespace percent_formula_l753_75398

theorem percent_formula (x y p : ℝ) (h : x = (p / 100) * y) : p = 100 * x / y :=
by
    sorry

end percent_formula_l753_75398


namespace find_a_l753_75368

noncomputable def tangent_condition (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 : ℝ) = (1 / (x₀ + a))

theorem find_a : ∃ a : ℝ, tangent_condition a ∧ a = 2 :=
by
  sorry

end find_a_l753_75368


namespace math_problem_l753_75393

theorem math_problem :
  (-1 : ℤ) ^ 49 + 2 ^ (4 ^ 3 + 3 ^ 2 - 7 ^ 2) = 16777215 := by
  sorry

end math_problem_l753_75393


namespace find_integer_k_l753_75320

theorem find_integer_k (k x : ℤ) (h : (k^2 - 1) * x^2 - 6 * (3 * k - 1) * x + 72 = 0) (hx : x > 0) :
  k = 1 ∨ k = 2 ∨ k = 3 :=
sorry

end find_integer_k_l753_75320


namespace probability_distribution_correct_l753_75329

noncomputable def X_possible_scores : Set ℤ := {-90, -30, 30, 90}

def prob_correct : ℚ := 0.8
def prob_incorrect : ℚ := 1 - prob_correct

def P_X_neg90 : ℚ := prob_incorrect ^ 3
def P_X_neg30 : ℚ := 3 * prob_correct * prob_incorrect ^ 2
def P_X_30 : ℚ := 3 * prob_correct ^ 2 * prob_incorrect
def P_X_90 : ℚ := prob_correct ^ 3

def P_advance : ℚ := P_X_30 + P_X_90

theorem probability_distribution_correct :
  (P_X_neg90 = (1/125) ∧ P_X_neg30 = (12/125) ∧ P_X_30 = (48/125) ∧ P_X_90 = (64/125)) ∧ 
  P_advance = (112/125) := 
by
  sorry

end probability_distribution_correct_l753_75329


namespace export_volume_scientific_notation_l753_75364

theorem export_volume_scientific_notation :
  (234.1 * 10^6) = (2.341 * 10^8) := 
sorry

end export_volume_scientific_notation_l753_75364


namespace prob_at_least_one_2_in_two_8_sided_dice_l753_75335

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end prob_at_least_one_2_in_two_8_sided_dice_l753_75335


namespace distance_correct_l753_75399

-- Define geometry entities and properties
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Sphere where
  center : Point
  radius : ℝ

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define conditions
def sphere_center : Point := { x := 0, y := 0, z := 0 }
def sphere : Sphere := { center := sphere_center, radius := 5 }
def triangle : Triangle := { a := 13, b := 13, c := 10 }

-- Define the distance calculation
noncomputable def distance_from_sphere_center_to_plane (O : Point) (T : Triangle) : ℝ :=
  let h := 12  -- height calculation based on given triangle sides
  let A := 60  -- area of the triangle
  let s := 18  -- semiperimeter
  let r := 10 / 3  -- inradius calculation
  let x := 5 * (Real.sqrt 5) / 3  -- final distance calculation
  x

-- Prove the obtained distance matches expected value
theorem distance_correct :
  distance_from_sphere_center_to_plane sphere_center triangle = 5 * (Real.sqrt 5) / 3 :=
by
  sorry

end distance_correct_l753_75399


namespace find_second_number_l753_75313

theorem find_second_number
  (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x = (3 : ℚ) / 4 * y)
  (h3 : z = (7 : ℚ) / 5 * y) :
  y = 800 / 21 :=
by
  sorry

end find_second_number_l753_75313


namespace water_consumption_per_hour_l753_75338

theorem water_consumption_per_hour 
  (W : ℝ) 
  (initial_water : ℝ := 20) 
  (initial_food : ℝ := 10) 
  (initial_gear : ℝ := 20) 
  (food_consumption_rate : ℝ := 1 / 3) 
  (hours : ℝ := 6) 
  (remaining_weight : ℝ := 34)
  (initial_weight := initial_water + initial_food + initial_gear)
  (consumed_water := W * hours)
  (consumed_food := food_consumption_rate * W * hours)
  (consumed_weight := consumed_water + consumed_food)
  (final_equation := initial_weight - consumed_weight)
  (correct_answer := 2) :
  final_equation = remaining_weight → W = correct_answer := 
by 
  sorry

end water_consumption_per_hour_l753_75338


namespace consecutive_integers_sum_l753_75330

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 504) : n + n+1 + n+2 = 24 :=
sorry

end consecutive_integers_sum_l753_75330


namespace number_of_cars_on_street_l753_75337

-- Definitions based on conditions
def cars_equally_spaced (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

def distance_between_first_and_last_car (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 242

def distance_between_cars (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

-- Given all conditions, prove n = 45
theorem number_of_cars_on_street (n : ℕ) :
  cars_equally_spaced n →
  distance_between_first_and_last_car n →
  distance_between_cars n →
  n = 45 :=
sorry

end number_of_cars_on_street_l753_75337


namespace bananas_per_box_l753_75336

def total_bananas : ℕ := 40
def number_of_boxes : ℕ := 10

theorem bananas_per_box : total_bananas / number_of_boxes = 4 := by
  sorry

end bananas_per_box_l753_75336


namespace coefficient_and_degree_of_monomial_l753_75300

variable (x y : ℝ)

def monomial : ℝ := -2 * x * y^3

theorem coefficient_and_degree_of_monomial :
  ( ∃ c : ℝ, ∃ d : ℤ, monomial x y = c * x * y^d ∧ c = -2 ∧ d = 4 ) :=
by
  sorry

end coefficient_and_degree_of_monomial_l753_75300


namespace no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l753_75376

theorem no_positive_integer_solutions_m2_m3 (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (∃ m, m = 2 ∨ m = 3 → (x / y + y / z + z / t + t / x = m) → false) :=
sorry

theorem positive_integer_solutions_m4 (x y z t : ℕ) :
  x / y + y / z + z / t + t / x = 4 ↔ ∃ k : ℕ, k > 0 ∧ (x = k ∧ y = k ∧ z = k ∧ t = k) :=
sorry

end no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l753_75376


namespace value_of_y_at_x_3_l753_75360

theorem value_of_y_at_x_3 (a b c : ℝ) (h : a * (-3 : ℝ)^5 + b * (-3)^3 + c * (-3) - 5 = 7) :
  a * (3 : ℝ)^5 + b * 3^3 + c * 3 - 5 = -17 :=
by
  sorry

end value_of_y_at_x_3_l753_75360
