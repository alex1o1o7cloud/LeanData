import Mathlib

namespace domain_of_log_x_squared_sub_2x_l1853_185368

theorem domain_of_log_x_squared_sub_2x (x : ℝ) : x^2 - 2 * x > 0 ↔ x < 0 ∨ x > 2 :=
by
  sorry

end domain_of_log_x_squared_sub_2x_l1853_185368


namespace triangle_split_points_l1853_185398

noncomputable def smallest_n_for_split (AB BC CA : ℕ) : ℕ := 
  if AB = 13 ∧ BC = 14 ∧ CA = 15 then 27 else sorry

theorem triangle_split_points (AB BC CA : ℕ) (h : AB = 13 ∧ BC = 14 ∧ CA = 15) :
  smallest_n_for_split AB BC CA = 27 :=
by
  cases h with | intro h1 h23 => sorry

-- Assertions for the explicit values provided in the conditions
example : smallest_n_for_split 13 14 15 = 27 :=
  triangle_split_points 13 14 15 ⟨rfl, rfl, rfl⟩

end triangle_split_points_l1853_185398


namespace coeff_of_linear_term_l1853_185337

def quadratic_eqn (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem coeff_of_linear_term :
  ∀ (x : ℝ), (quadratic_eqn x = 0) → (∃ c_b : ℝ, quadratic_eqn x = x^2 + c_b * x + 3 ∧ c_b = -2) :=
by
  sorry

end coeff_of_linear_term_l1853_185337


namespace temperature_difference_l1853_185394

theorem temperature_difference (T_south T_north : ℝ) (h_south : T_south = 6) (h_north : T_north = -3) :
  T_south - T_north = 9 :=
by 
  -- Proof goes here
  sorry

end temperature_difference_l1853_185394


namespace buses_needed_for_trip_l1853_185329

theorem buses_needed_for_trip :
  ∀ (total_students students_in_vans bus_capacity : ℕ),
  total_students = 500 →
  students_in_vans = 56 →
  bus_capacity = 45 →
  ⌈(total_students - students_in_vans : ℝ) / bus_capacity⌉ = 10 :=
by
  sorry

end buses_needed_for_trip_l1853_185329


namespace common_ratio_of_geometric_sequence_l1853_185322

theorem common_ratio_of_geometric_sequence (a₁ : ℝ) (S : ℕ → ℝ) (q : ℝ) (h₁ : ∀ n, S (n + 1) = S n + a₁ * q ^ n) (h₂ : 2 * S n = S (n + 1) + S (n + 2)) :
  q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l1853_185322


namespace solve_equation_l1853_185317

theorem solve_equation :
  let lhs := ((4 - 3.5 * (15/7 - 6/5)) / 0.16)
  let rhs := ((23/7 - (3/14) / (1/6)) / (3467/84 - 2449/60))
  lhs / 1 = rhs :=
by
  sorry

end solve_equation_l1853_185317


namespace least_positive_a_exists_l1853_185336

noncomputable def f (x a : ℤ) : ℤ := 5 * x ^ 13 + 13 * x ^ 5 + 9 * a * x

theorem least_positive_a_exists :
  ∃ a : ℕ, (∀ x : ℤ, 65 ∣ f x a) ∧ ∀ b : ℕ, (∀ x : ℤ, 65 ∣ f x b) → a ≤ b :=
sorry

end least_positive_a_exists_l1853_185336


namespace find_sum_lent_l1853_185353

theorem find_sum_lent (r t : ℝ) (I : ℝ) (P : ℝ) (h1: r = 0.06) (h2 : t = 8) (h3 : I = P - 520) (h4: I = P * r * t) : P = 1000 := by
  sorry

end find_sum_lent_l1853_185353


namespace decreasing_function_condition_l1853_185361

theorem decreasing_function_condition :
  (∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * ((1 / x1 - x1) - (1 / x2 - x2)) < 0) :=
by
  -- Proof outline goes here
  sorry

end decreasing_function_condition_l1853_185361


namespace oliver_total_money_l1853_185340

-- Define the initial conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def cost_frisbee : ℕ := 4
def cost_puzzle : ℕ := 3
def birthday_money : ℕ := 8

-- Define the problem statement in Lean
theorem oliver_total_money : 
  (initial_amount + allowance_saved - (cost_frisbee + cost_puzzle) + birthday_money) = 15 := 
by 
  sorry

end oliver_total_money_l1853_185340


namespace middle_pile_cards_l1853_185300

theorem middle_pile_cards (x : Nat) (h : x ≥ 2) : 
    let left := x - 2
    let middle := x + 2
    let right := x
    let middle_after_step3 := middle + 1
    let final_middle := middle_after_step3 - left
    final_middle = 5 := 
by
  sorry

end middle_pile_cards_l1853_185300


namespace largest_result_among_expressions_l1853_185365

def E1 : ℕ := 992 * 999 + 999
def E2 : ℕ := 993 * 998 + 998
def E3 : ℕ := 994 * 997 + 997
def E4 : ℕ := 995 * 996 + 996

theorem largest_result_among_expressions : E4 > E1 ∧ E4 > E2 ∧ E4 > E3 :=
by sorry

end largest_result_among_expressions_l1853_185365


namespace closest_point_on_line_to_target_l1853_185308

noncomputable def parametricPoint (s : ℝ) : ℝ × ℝ × ℝ :=
  (6 + 3 * s, 2 - 9 * s, 0 + 6 * s)

noncomputable def closestPoint : ℝ × ℝ × ℝ :=
  (249/42, 95/42, -1/7)

theorem closest_point_on_line_to_target :
  ∃ s : ℝ, parametricPoint s = closestPoint :=
by
  sorry

end closest_point_on_line_to_target_l1853_185308


namespace weightlifter_one_hand_l1853_185312

theorem weightlifter_one_hand (total_weight : ℕ) (h : total_weight = 20) (even_distribution : total_weight % 2 = 0) : total_weight / 2 = 10 :=
by
  sorry

end weightlifter_one_hand_l1853_185312


namespace rocky_first_round_knockouts_l1853_185355

theorem rocky_first_round_knockouts
  (total_fights : ℕ)
  (knockout_percentage : ℝ)
  (first_round_knockout_percentage : ℝ)
  (h1 : total_fights = 190)
  (h2 : knockout_percentage = 0.50)
  (h3 : first_round_knockout_percentage = 0.20) :
  (total_fights * knockout_percentage * first_round_knockout_percentage = 19) := 
by
  sorry

end rocky_first_round_knockouts_l1853_185355


namespace sqrt_sine_tan_domain_l1853_185311

open Real

noncomputable def domain_sqrt_sine_tan : Set ℝ :=
  {x | ∃ (k : ℤ), (-π / 2 + 2 * k * π < x ∧ x < π / 2 + 2 * k * π) ∨ x = k * π}

theorem sqrt_sine_tan_domain (x : ℝ) :
  (sin x * tan x ≥ 0) ↔ x ∈ domain_sqrt_sine_tan :=
by
  sorry

end sqrt_sine_tan_domain_l1853_185311


namespace car_total_travel_time_l1853_185372

-- Define the given conditions
def travel_time_ngapara_zipra : ℝ := 60
def travel_time_ningi_zipra : ℝ := 0.8 * travel_time_ngapara_zipra
def speed_limit_zone_fraction : ℝ := 0.25
def speed_reduction_factor : ℝ := 0.5
def travel_time_zipra_varnasi : ℝ := 0.75 * travel_time_ningi_zipra

-- Total adjusted travel time from Ningi to Zipra including speed limit delay
def adjusted_travel_time_ningi_zipra : ℝ :=
  let delayed_time := speed_limit_zone_fraction * travel_time_ningi_zipra * (2 - speed_reduction_factor)
  travel_time_ningi_zipra + delayed_time

-- Total travel time in the day
def total_travel_time : ℝ :=
  travel_time_ngapara_zipra + adjusted_travel_time_ningi_zipra + travel_time_zipra_varnasi

-- Proposition to prove
theorem car_total_travel_time : total_travel_time = 156 :=
by
  -- We skip the proof for now
  sorry

end car_total_travel_time_l1853_185372


namespace num_positive_integers_which_make_polynomial_prime_l1853_185335

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem num_positive_integers_which_make_polynomial_prime :
  (∃! n : ℕ, n > 0 ∧ is_prime (n^3 - 7 * n^2 + 18 * n - 10)) :=
sorry

end num_positive_integers_which_make_polynomial_prime_l1853_185335


namespace solve_for_s_l1853_185384

theorem solve_for_s (r s : ℝ) (h1 : 1 < r) (h2 : r < s) (h3 : 1 / r + 1 / s = 3 / 4) (h4 : r * s = 8) : s = 4 :=
sorry

end solve_for_s_l1853_185384


namespace circumscribed_circle_radius_l1853_185351

noncomputable def radius_of_circumscribed_circle 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) : ℝ :=
2

theorem circumscribed_circle_radius 
  {a b c A B C : ℝ} 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) :
  radius_of_circumscribed_circle a b c A B C h1 h2 h3 = 2 :=
sorry

end circumscribed_circle_radius_l1853_185351


namespace wolf_and_nobel_prize_laureates_l1853_185390

-- Definitions from the conditions
def num_total_scientists : ℕ := 50
def num_wolf_prize_laureates : ℕ := 31
def num_nobel_prize_laureates : ℕ := 29
def num_no_wolf_prize_and_yes_nobel := 3 -- N_W = N_W'
def num_without_wolf_or_nobel : ℕ := num_total_scientists - num_wolf_prize_laureates - 11 -- Derived from N_W' 

-- The statement to be proved
theorem wolf_and_nobel_prize_laureates :
  ∃ W_N, W_N = num_nobel_prize_laureates - (19 - 3) ∧ W_N = 18 :=
  by
    sorry

end wolf_and_nobel_prize_laureates_l1853_185390


namespace uma_fraction_part_l1853_185350

theorem uma_fraction_part (r s t u : ℕ) 
  (hr : r = 6) 
  (hs : s = 5) 
  (ht : t = 7) 
  (hu : u = 8) 
  (shared_amount: ℕ)
  (hr_amount: shared_amount = r / 6)
  (hs_amount: shared_amount = s / 5)
  (ht_amount: shared_amount = t / 7)
  (hu_amount: shared_amount = u / 8) :
  ∃ total : ℕ, ∃ uma_total : ℕ, uma_total * 13 = 2 * total :=
sorry

end uma_fraction_part_l1853_185350


namespace ratio_of_areas_l1853_185381

noncomputable def side_length_WXYZ : ℝ := 16

noncomputable def WJ : ℝ := (3/4) * side_length_WXYZ
noncomputable def JX : ℝ := (1/4) * side_length_WXYZ

noncomputable def side_length_JKLM := 4 * Real.sqrt 2

noncomputable def area_JKLM := (side_length_JKLM)^2
noncomputable def area_WXYZ := (side_length_WXYZ)^2

theorem ratio_of_areas : area_JKLM / area_WXYZ = 1 / 8 :=
by
  sorry

end ratio_of_areas_l1853_185381


namespace compute_expression_l1853_185338

open Real

theorem compute_expression : 
  sqrt (1 / 4) * sqrt 16 - (sqrt (1 / 9))⁻¹ - sqrt 0 + sqrt (45 / 5) = 2 := 
by
  -- The proof details would go here, but they are omitted.
  sorry

end compute_expression_l1853_185338


namespace find_fifth_day_income_l1853_185345

-- Define the incomes for the first four days
def income_day1 := 45
def income_day2 := 50
def income_day3 := 60
def income_day4 := 65

-- Define the average income over five days
def average_income := 58

-- Expressing the question in terms of a function to determine the fifth day's income
theorem find_fifth_day_income : 
  ∃ (income_day5 : ℕ), 
    (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = average_income 
    ∧ income_day5 = 70 :=
sorry

end find_fifth_day_income_l1853_185345


namespace largest_sum_of_two_largest_angles_of_EFGH_l1853_185391

theorem largest_sum_of_two_largest_angles_of_EFGH (x d : ℝ) (y z : ℝ) :
  (∃ a b : ℝ, a + 2 * b = x + 70 ∧ a + b = 70 ∧ 2 * a + 3 * b = 180) ∧
  (2 * x + 3 * d = 180) ∧ (x = 30) ∧ (y = 70) ∧ (z = 100) ∧ (z + 70 = x + d) ∧
  x + d + x + 2 * d + x + 3 * d + x = 360 →
  max (70 + y) (70 + z) + max (y + 70) (z + 70) = 210 := 
sorry

end largest_sum_of_two_largest_angles_of_EFGH_l1853_185391


namespace least_months_for_tripling_debt_l1853_185331

theorem least_months_for_tripling_debt (P : ℝ) (r : ℝ) (t : ℕ) : P = 1500 → r = 0.06 → (3 * P < P * (1 + r) ^ t) → t ≥ 20 :=
by
  intros hP hr hI
  rw [hP, hr] at hI
  norm_num at hI
  sorry

end least_months_for_tripling_debt_l1853_185331


namespace joshua_miles_ratio_l1853_185321

-- Definitions corresponding to conditions
def mitch_macarons : ℕ := 20
def joshua_extra : ℕ := 6
def total_kids : ℕ := 68
def macarons_per_kid : ℕ := 2

-- Variables for unspecified amounts
variable (M : ℕ) -- number of macarons Miles made

-- Calculations for Joshua and Renz's macarons based on given conditions
def joshua_macarons := mitch_macarons + joshua_extra
def renz_macarons := (3 * M) / 4 - 1

-- Total macarons calculation
def total_macarons := mitch_macarons + joshua_macarons + renz_macarons + M

-- Proof statement: Showing the ratio of number of macarons Joshua made to the number of macarons Miles made
theorem joshua_miles_ratio : (total_macarons = total_kids * macarons_per_kid) → (joshua_macarons : ℚ) / (M : ℚ) = 1 / 2 :=
by
  sorry

end joshua_miles_ratio_l1853_185321


namespace perimeter_of_face_given_volume_l1853_185399

-- Definitions based on conditions
def volume_of_cube (v : ℝ) := v = 512

def side_of_cube (s : ℝ) := s^3 = 512

def perimeter_of_face (p s : ℝ) := p = 4 * s

-- Lean 4 statement: prove that the perimeter of one face of the cube is 32 cm given the volume is 512 cm³.
theorem perimeter_of_face_given_volume :
  ∃ s : ℝ, volume_of_cube (s^3) ∧ perimeter_of_face 32 s :=
by sorry

end perimeter_of_face_given_volume_l1853_185399


namespace seedlings_total_l1853_185397

theorem seedlings_total (seeds_per_packet : ℕ) (packets : ℕ) (total_seedlings : ℕ) 
  (h1 : seeds_per_packet = 7) (h2 : packets = 60) : total_seedlings = 420 :=
by {
  sorry
}

end seedlings_total_l1853_185397


namespace dist_between_centers_l1853_185358

noncomputable def dist_centers_tangent_circles : ℝ :=
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  Real.sqrt 2 * (a₁ - a₂)

theorem dist_between_centers :
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  let C₁ := (a₁, a₁)
  let C₂ := (a₂, a₂)
  dist_centers_tangent_circles = 8 :=
by
  sorry

end dist_between_centers_l1853_185358


namespace human_height_weight_correlated_l1853_185374

-- Define the relationships as types
def taxiFareDistanceRelated : Prop := ∀ x y : ℕ, x = y → True
def houseSizePriceRelated : Prop := ∀ x y : ℕ, x = y → True
def humanHeightWeightCorrelated : Prop := ∃ k : ℕ, ∀ x y : ℕ, x / k = y
def ironBlockMassRelated : Prop := ∀ x y : ℕ, x = y → True

-- Main theorem statement
theorem human_height_weight_correlated : humanHeightWeightCorrelated :=
  sorry

end human_height_weight_correlated_l1853_185374


namespace at_least_one_less_than_equal_one_l1853_185357

theorem at_least_one_less_than_equal_one
  (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 := 
by 
  sorry

end at_least_one_less_than_equal_one_l1853_185357


namespace average_waiting_time_l1853_185377

/-- 
A traffic light at a pedestrian crossing allows pedestrians to cross the street 
for one minute and prohibits crossing for two minutes. Prove that the average 
waiting time for a pedestrian who arrives at the intersection is 40 seconds.
-/ 
theorem average_waiting_time (pG : ℝ) (pR : ℝ) (eTG : ℝ) (eTR : ℝ) (cycle : ℝ) :
  pG = 1 / 3 ∧ pR = 2 / 3 ∧ eTG = 0 ∧ eTR = 1 ∧ cycle = 3 → 
  (eTG * pG + eTR * pR) * (60 / cycle) = 40 :=
by
  sorry

end average_waiting_time_l1853_185377


namespace complement_intersection_l1853_185320

open Set

variable (I : Set ℕ) (A B : Set ℕ)

-- Given the universal set and specific sets A and B
def universal_set : Set ℕ := {1,2,3,4,5}
def set_A : Set ℕ := {2,3,5}
def set_B : Set ℕ := {1,2}

-- To prove that the complement of B in I intersects A to be {3,5}
theorem complement_intersection :
  (universal_set \ set_B) ∩ set_A = {3,5} :=
sorry

end complement_intersection_l1853_185320


namespace find_a2_b2_c2_l1853_185310

theorem find_a2_b2_c2 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = 1) (h5 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5 + 1) : 
  a^2 + b^2 + c^2 = 7 / 5 := 
sorry

end find_a2_b2_c2_l1853_185310


namespace cost_per_can_of_tuna_l1853_185363

theorem cost_per_can_of_tuna
  (num_cans : ℕ) -- condition 1
  (num_coupons : ℕ) -- condition 2
  (coupon_discount_cents : ℕ) -- condition 2 detail
  (amount_paid_dollars : ℚ) -- condition 3
  (change_received_dollars : ℚ) -- condition 3 detail
  (cost_per_can_cents: ℚ) : -- the quantity we want to prove
  num_cans = 9 →
  num_coupons = 5 →
  coupon_discount_cents = 25 →
  amount_paid_dollars = 20 →
  change_received_dollars = 5.5 →
  cost_per_can_cents = 175 :=
by
  intros hn hc hcd hap hcr
  sorry

end cost_per_can_of_tuna_l1853_185363


namespace kabulek_four_digits_l1853_185379

def isKabulekNumber (N: ℕ) : Prop :=
  let a := N / 100
  let b := N % 100
  (a + b) ^ 2 = N

theorem kabulek_four_digits :
  {N : ℕ | 1000 ≤ N ∧ N < 10000 ∧ isKabulekNumber N} = {2025, 3025, 9801} :=
by sorry

end kabulek_four_digits_l1853_185379


namespace shepherd_initial_sheep_l1853_185330

def sheep_pass_gate (sheep : ℕ) : ℕ :=
  sheep / 2 + 1

noncomputable def shepherd_sheep (initial_sheep : ℕ) : ℕ :=
  (sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate) initial_sheep

theorem shepherd_initial_sheep (initial_sheep : ℕ) (h : shepherd_sheep initial_sheep = 2) :
  initial_sheep = 2 :=
sorry

end shepherd_initial_sheep_l1853_185330


namespace positive_polynomial_l1853_185359

theorem positive_polynomial (x : ℝ) : 3 * x ^ 2 - 6 * x + 3.5 > 0 := 
by sorry

end positive_polynomial_l1853_185359


namespace sum_of_three_squares_not_divisible_by_3_l1853_185347

theorem sum_of_three_squares_not_divisible_by_3
    (N : ℕ) (n : ℕ) (a b c : ℤ) 
    (h1 : N = 9^n * (a^2 + b^2 + c^2))
    (h2 : ∃ (a1 b1 c1 : ℤ), a = 3 * a1 ∧ b = 3 * b1 ∧ c = 3 * c1) :
    ∃ (k m n : ℤ), N = k^2 + m^2 + n^2 ∧ (¬ (3 ∣ k ∧ 3 ∣ m ∧ 3 ∣ n)) :=
sorry

end sum_of_three_squares_not_divisible_by_3_l1853_185347


namespace product_divisible_by_3_or_5_l1853_185371

theorem product_divisible_by_3_or_5 {a b c d : ℕ} (h : Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = a + b + c + d) :
  (a * b * c * d) % 3 = 0 ∨ (a * b * c * d) % 5 = 0 :=
by
  sorry

end product_divisible_by_3_or_5_l1853_185371


namespace hole_depth_l1853_185318

theorem hole_depth (height : ℝ) (half_depth : ℝ) (total_depth : ℝ) 
    (h_height : height = 90) 
    (h_half_depth : half_depth = total_depth / 2)
    (h_position : height + half_depth = total_depth - height) : 
    total_depth = 120 := 
by
    sorry

end hole_depth_l1853_185318


namespace no_fractions_satisfy_condition_l1853_185375

theorem no_fractions_satisfy_condition :
  ∀ (x y : ℕ), 
    x > 0 → y > 0 → Nat.gcd x y = 1 →
    (1.2 : ℚ) * (x : ℚ) / (y : ℚ) = (x + 2 : ℚ) / (y + 2 : ℚ) →
    False :=
by
  intros x y hx hy hrel hcond
  sorry

end no_fractions_satisfy_condition_l1853_185375


namespace binom_9_5_eq_126_l1853_185380

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l1853_185380


namespace minimum_choir_members_l1853_185386

theorem minimum_choir_members:
  ∃ n : ℕ, (n % 9 = 0) ∧ (n % 10 = 0) ∧ (n % 11 = 0) ∧ (∀ m : ℕ, (m % 9 = 0) ∧ (m % 10 = 0) ∧ (m % 11 = 0) → n ≤ m) → n = 990 :=
by
  sorry

end minimum_choir_members_l1853_185386


namespace perpendicular_lines_l1853_185395

theorem perpendicular_lines (a : ℝ) :
  (∃ (m₁ m₂ : ℝ), ((a + 1) * m₁ + a * m₂ = 0) ∧ 
                  (a * m₁ + 2 * m₂ = 1) ∧ 
                  m₁ * m₂ = -1) ↔ (a = 0 ∨ a = -3) := 
sorry

end perpendicular_lines_l1853_185395


namespace michelle_initial_crayons_l1853_185325

variable (m j : Nat)

axiom janet_crayons : j = 2
axiom michelle_has_after_gift : m + j = 4

theorem michelle_initial_crayons : m = 2 :=
by
  sorry

end michelle_initial_crayons_l1853_185325


namespace temperature_reading_l1853_185378

theorem temperature_reading (scale_min scale_max : ℝ) (arrow : ℝ) (h1 : scale_min = -6.0) (h2 : scale_max = -5.5) (h3 : scale_min < arrow) (h4 : arrow < scale_max) : arrow = -5.7 :=
sorry

end temperature_reading_l1853_185378


namespace min_value_expression_l1853_185342

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem min_value_expression : (1 + b / a) * (4 * a / b) ≥ 9 :=
sorry

end min_value_expression_l1853_185342


namespace container_capacity_l1853_185324

theorem container_capacity 
  (C : ℝ)
  (h1 : 0.75 * C - 0.30 * C = 45) :
  C = 100 := by
  sorry

end container_capacity_l1853_185324


namespace unique_solution_of_system_l1853_185306

theorem unique_solution_of_system (n k m : ℕ) (hnk : n + k = Nat.gcd n k ^ 2) (hkm : k + m = Nat.gcd k m ^ 2) (hmn : m + n = Nat.gcd m n ^ 2) : 
  n = 2 ∧ k = 2 ∧ m = 2 :=
by
  sorry

end unique_solution_of_system_l1853_185306


namespace prove_area_and_sum_l1853_185315

-- Define the coordinates of the vertices of the quadrilateral.
variables (a b : ℤ)

-- Define the non-computable requirements related to the problem.
noncomputable def problem_statement : Prop :=
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a > b ∧ (4 * a * b = 32) ∧ (a + b = 5)

theorem prove_area_and_sum : problem_statement := 
sorry

end prove_area_and_sum_l1853_185315


namespace chiquita_height_l1853_185387

theorem chiquita_height (C : ℝ) :
  (C + (C + 2) = 12) → (C = 5) :=
by
  intro h
  sorry

end chiquita_height_l1853_185387


namespace product_of_digits_l1853_185356

theorem product_of_digits (A B : ℕ) (h1 : A + B = 13) (h2 : (10 * A + B) % 4 = 0) : A * B = 42 :=
by
  sorry

end product_of_digits_l1853_185356


namespace range_of_m_l1853_185369

open Real Set

def P (m : ℝ) := |m + 1| ≤ 2
def Q (m : ℝ) := ∃ x : ℝ, x^2 - m*x + 1 = 0 ∧ (m^2 - 4 ≥ 0)

theorem range_of_m (m : ℝ) :
  (¬¬ P m ∧ ¬ (P m ∧ Q m)) → -2 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l1853_185369


namespace find_quotient_l1853_185354

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![4, 5]]

noncomputable def matrix_b (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![c, d]]

theorem find_quotient (a b c d : ℝ) (H1 : matrix_a * (matrix_b a b c d) = (matrix_b a b c d) * matrix_a)
  (H2 : 2*b ≠ 3*c) : ((a - d) / (c - 2*b)) = 3 / 2 :=
  sorry

end find_quotient_l1853_185354


namespace part1_positive_root_part2_negative_solution_l1853_185344

theorem part1_positive_root (x k : ℝ) (hx1 : x > 0)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k = 6 ∨ k = -8 := 
sorry

theorem part2_negative_solution (x k : ℝ) (hx2 : x < 0)
  (hx_ne1 : x ≠ 1) (hx_ne_neg1 : x ≠ -1)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k < -1 ∧ k ≠ -8 := 
sorry

end part1_positive_root_part2_negative_solution_l1853_185344


namespace ages_proof_l1853_185305

def hans_now : ℕ := 8

def sum_ages (annika_now emil_now frida_now : ℕ) :=
  hans_now + annika_now + emil_now + frida_now = 58

def annika_age_in_4_years (annika_now : ℕ) : ℕ :=
  3 * (hans_now + 4)

def emil_age_in_4_years (emil_now : ℕ) : ℕ :=
  2 * (hans_now + 4)

def frida_age_in_4_years (frida_now : ℕ) :=
  2 * 12

def annika_frida_age_difference (annika_now frida_now : ℕ) : Prop :=
  annika_now = frida_now + 5

theorem ages_proof :
  ∃ (annika_now emil_now frida_now : ℕ),
    sum_ages annika_now emil_now frida_now ∧
    annika_age_in_4_years annika_now = 36 ∧
    emil_age_in_4_years emil_now = 24 ∧
    frida_age_in_4_years frida_now = 24 ∧
    annika_frida_age_difference annika_now frida_now :=
by
  sorry

end ages_proof_l1853_185305


namespace sum_squares_bound_l1853_185382

theorem sum_squares_bound (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) : 
  ab + bc + cd + da ≤ 4 :=
sorry

end sum_squares_bound_l1853_185382


namespace longer_trip_due_to_red_lights_l1853_185366

theorem longer_trip_due_to_red_lights :
  ∀ (num_lights : ℕ) (green_time first_route_base_time red_time_per_light second_route_time : ℕ),
  num_lights = 3 →
  first_route_base_time = 10 →
  red_time_per_light = 3 →
  second_route_time = 14 →
  (first_route_base_time + num_lights * red_time_per_light) - second_route_time = 5 :=
by
  intros num_lights green_time first_route_base_time red_time_per_light second_route_time
  sorry

end longer_trip_due_to_red_lights_l1853_185366


namespace arthur_num_hamburgers_on_first_day_l1853_185376

theorem arthur_num_hamburgers_on_first_day (H D : ℕ) (hamburgers_1 hamburgers_2 : ℕ) (hotdogs_1 hotdogs_2 : ℕ)
  (h1 : hamburgers_1 * H + hotdogs_1 * D = 10)
  (h2 : hamburgers_2 * H + hotdogs_2 * D = 7)
  (hprice : D = 1)
  (h1_hotdogs : hotdogs_1 = 4)
  (h2_hotdogs : hotdogs_2 = 3) : 
  hamburgers_1 = 1 := 
by
  sorry

end arthur_num_hamburgers_on_first_day_l1853_185376


namespace train_crossing_time_l1853_185304

theorem train_crossing_time
  (train_length : ℕ)           -- length of the train in meters
  (train_speed_kmh : ℕ)        -- speed of the train in kilometers per hour
  (conversion_factor : ℕ)      -- conversion factor from km/hr to m/s
  (train_speed_ms : ℕ)         -- speed of the train in meters per second
  (time_to_cross : ℚ)          -- time to cross in seconds
  (h1 : train_length = 60)
  (h2 : train_speed_kmh = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : train_speed_ms = train_speed_kmh * conversion_factor)
  (h5 : time_to_cross = train_length / train_speed_ms) :
  time_to_cross = 1.5 :=
by sorry

end train_crossing_time_l1853_185304


namespace gain_percent_of_cost_selling_relation_l1853_185346

theorem gain_percent_of_cost_selling_relation (C S : ℕ) (h : 50 * C = 45 * S) : 
  (S > C) ∧ ((S - C) / C * 100 = 100 / 9) :=
by
  sorry

end gain_percent_of_cost_selling_relation_l1853_185346


namespace searchlight_probability_l1853_185326

theorem searchlight_probability (revolutions_per_minute : ℕ) (D : ℝ) (prob : ℝ)
  (h1 : revolutions_per_minute = 4)
  (h2 : prob = 0.6666666666666667) :
  D = (2 / 3) * (60 / revolutions_per_minute) :=
by
  -- To complete the proof, we will use the conditions given.
  sorry

end searchlight_probability_l1853_185326


namespace min_value_of_function_l1853_185388

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x^2

theorem min_value_of_function :
  ∀ x > 0, f x ≥ 9 :=
by
  intro x hx_pos
  sorry

end min_value_of_function_l1853_185388


namespace correct_total_count_l1853_185323

variable (x : ℕ)

-- Define the miscalculation values
def value_of_quarter := 25
def value_of_dime := 10
def value_of_half_dollar := 50
def value_of_nickel := 5

-- Calculate the individual overestimations and underestimations
def overestimation_from_quarters := (value_of_quarter - value_of_dime) * (2 * x)
def underestimation_from_half_dollars := (value_of_half_dollar - value_of_nickel) * x

-- Calculate the net correction needed
def net_correction := overestimation_from_quarters - underestimation_from_half_dollars

theorem correct_total_count :
  net_correction x = 15 * x :=
by
  sorry

end correct_total_count_l1853_185323


namespace smallest_feared_sequence_l1853_185332

def is_feared (n : ℕ) : Prop :=
  -- This function checks if a number contains '13' as a contiguous substring.
  sorry

def is_fearless (n : ℕ) : Prop := ¬is_feared n

theorem smallest_feared_sequence : ∃ (n : ℕ) (a : ℕ), 0 < n ∧ a < 100 ∧ is_fearless n ∧ is_fearless (n + 10 * a) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → is_feared (n + k * a)) ∧ n = 1287 := 
by
  sorry

end smallest_feared_sequence_l1853_185332


namespace volume_small_pyramid_eq_27_60_l1853_185327

noncomputable def volume_of_smaller_pyramid (base_edge : ℝ) (slant_edge : ℝ) (height_above_base : ℝ) : ℝ :=
  let total_height := Real.sqrt ((slant_edge ^ 2) - ((base_edge / (2 * Real.sqrt 2)) ^ 2))
  let smaller_pyramid_height := total_height - height_above_base
  let scale_factor := (smaller_pyramid_height / total_height)
  let new_base_edge := base_edge * scale_factor
  let new_base_area := (new_base_edge ^ 2) * 2
  (1 / 3) * new_base_area * smaller_pyramid_height

theorem volume_small_pyramid_eq_27_60 :
  volume_of_smaller_pyramid (10 * Real.sqrt 2) 12 4 = 27.6 :=
by
  sorry

end volume_small_pyramid_eq_27_60_l1853_185327


namespace cloaks_always_short_l1853_185339

-- Define the problem parameters
variables (Knights Cloaks : Type)
variables [Fintype Knights] [Fintype Cloaks]
variables (h_knights : Fintype.card Knights = 20) (h_cloaks : Fintype.card Cloaks = 20)

-- Assume every knight initially found their cloak too short
variable (too_short : Knights -> Prop)

-- Height order for knights
variable (height_order : LinearOrder Knights)
-- Length order for cloaks
variable (length_order : LinearOrder Cloaks)

-- Sorting function
noncomputable def sorted_cloaks (kn : Knights) : Cloaks := sorry

-- State that after redistribution, every knight's cloak is still too short
theorem cloaks_always_short : 
  ∀ (kn : Knights), too_short kn :=
by sorry

end cloaks_always_short_l1853_185339


namespace sum_of_heights_less_than_perimeter_l1853_185307

theorem sum_of_heights_less_than_perimeter
  (a b c h1 h2 h3 : ℝ) 
  (H1 : h1 ≤ b) 
  (H2 : h2 ≤ c) 
  (H3 : h3 ≤ a) 
  (H4 : h1 < b ∨ h2 < c ∨ h3 < a) : 
  h1 + h2 + h3 < a + b + c :=
by {
  sorry
}

end sum_of_heights_less_than_perimeter_l1853_185307


namespace find_p_q_r_sum_l1853_185360

noncomputable def Q (p q r : ℝ) (v : ℂ) : Polynomial ℂ :=
  (Polynomial.C v + 2 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C v + 8 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C (3 * v - 5)).comp Polynomial.X

theorem find_p_q_r_sum (p q r : ℝ) (v : ℂ)
  (h_roots : ∃ v : ℂ, Polynomial.roots (Q p q r v) = {v + 2 * Complex.I, v + 8 * Complex.I, 3 * v - 5}) :
  (p + q + r) = -82 :=
by
  sorry

end find_p_q_r_sum_l1853_185360


namespace factorize_expression_l1853_185314

theorem factorize_expression (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) :=
by sorry

end factorize_expression_l1853_185314


namespace necessary_not_sufficient_condition_l1853_185389
-- Import the necessary libraries

-- Define the real number condition
def real_number (a : ℝ) : Prop := true

-- Define line l1
def line_l1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define line l2
def line_l2 (a y x: ℝ) : Prop := a * x + 4 * y + 6 = 0

-- Define the parallel condition
def parallel_lines (a : ℝ) : Prop :=
  (a = 2 ∨ a = -2) ∧ 
  ∀ x y : ℝ, line_l1 a x y ∧ line_l2 a x y → a * x + 4 * x + 6 = 3

-- State the main theorem to prove
theorem necessary_not_sufficient_condition (a : ℝ) : 
  real_number a → (a = 2 ∨ a = -2) ↔ (parallel_lines a) := 
by
  sorry

end necessary_not_sufficient_condition_l1853_185389


namespace max_points_of_intersection_l1853_185370

open Set

def Point := ℝ × ℝ

structure Circle :=
(center : Point)
(radius : ℝ)

structure Line :=
(coeffs : ℝ × ℝ × ℝ) -- Assume line equation in the form Ax + By + C = 0

def max_intersection_points (circle : Circle) (lines : List Line) : ℕ :=
  let circle_line_intersect_count := 2
  let line_line_intersect_count := 1
  
  let number_of_lines := lines.length
  let pairwise_line_intersections := number_of_lines.choose 2
  
  let circle_and_lines_intersections := circle_line_intersect_count * number_of_lines
  let total_intersections := circle_and_lines_intersections + pairwise_line_intersections

  total_intersections

theorem max_points_of_intersection (c : Circle) (l1 l2 l3 : Line) :
  max_intersection_points c [l1, l2, l3] = 9 :=
by
  sorry

end max_points_of_intersection_l1853_185370


namespace solution_set_for_inequality_l1853_185373

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

noncomputable def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

theorem solution_set_for_inequality
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_decreasing : decreasing_on f (Set.Iio 0))
  (h_f1 : f 1 = 0) :
  {x : ℝ | x^3 * f x > 0} = {x : ℝ | x > 1 ∨ x < -1} :=
by
  sorry

end solution_set_for_inequality_l1853_185373


namespace probability_three_fair_coins_l1853_185334

noncomputable def probability_one_head_two_tails (n : ℕ) : ℚ :=
  if n = 3 then 3 / 8 else 0

theorem probability_three_fair_coins :
  probability_one_head_two_tails 3 = 3 / 8 :=
by
  sorry

end probability_three_fair_coins_l1853_185334


namespace horse_running_time_l1853_185333

def area_of_square_field : Real := 625
def speed_of_horse_around_field : Real := 25

theorem horse_running_time : (4 : Real) = 
  let side_length := Real.sqrt area_of_square_field
  let perimeter := 4 * side_length
  perimeter / speed_of_horse_around_field :=
by
  sorry

end horse_running_time_l1853_185333


namespace pandas_increase_l1853_185393

theorem pandas_increase 
  (C P : ℕ) -- C: Number of cheetahs 5 years ago, P: Number of pandas 5 years ago
  (h_ratio_5_years_ago : C / P = 1 / 3)
  (h_cheetahs_increase : ∃ z : ℕ, z = 2)
  (h_ratio_now : ∃ k : ℕ, (C + k) / (P + x) = 1 / 3) :
  x = 6 :=
by
  sorry

end pandas_increase_l1853_185393


namespace equivalent_single_discount_l1853_185341

theorem equivalent_single_discount (p : ℝ) :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  let final_price := (1 - discount1) * (1 - discount2) * (1 - discount3) * p
  (1 - final_price / p) = 0.27325 :=
by
  sorry

end equivalent_single_discount_l1853_185341


namespace radius_of_circle_through_points_l1853_185367

theorem radius_of_circle_through_points : 
  ∃ (x : ℝ), 
  (dist (x, 0) (2, 5) = dist (x, 0) (3, 4)) →
  (∃ (r : ℝ), r = dist (x, 0) (2, 5) ∧ r = 5) :=
by
  sorry

end radius_of_circle_through_points_l1853_185367


namespace print_pages_500_l1853_185348

theorem print_pages_500 (cost_per_page cents total_dollars) : 
  cost_per_page = 3 → 
  total_dollars = 15 → 
  cents = 100 * total_dollars → 
  (cents / cost_per_page) = 500 :=
by 
  intros h1 h2 h3
  sorry

end print_pages_500_l1853_185348


namespace sally_credit_card_balance_l1853_185309

theorem sally_credit_card_balance (G P : ℝ) (X : ℝ)  
  (h1 : P = 2 * G)  
  (h2 : XP = X * P)  
  (h3 : G / 3 + XP = (5 / 12) * P) : 
  X = 1 / 4 :=
by
  sorry

end sally_credit_card_balance_l1853_185309


namespace find_x_eq_3_plus_sqrt7_l1853_185316

variable (x y : ℝ)
variable (h1 : x > y)
variable (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40)
variable (h3 : x * y + x + y = 8)

theorem find_x_eq_3_plus_sqrt7 (h1 : x > y) (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40) (h3 : x * y + x + y = 8) : 
  x = 3 + Real.sqrt 7 :=
sorry

end find_x_eq_3_plus_sqrt7_l1853_185316


namespace all_rationals_on_number_line_l1853_185302

theorem all_rationals_on_number_line :
  ∀ q : ℚ, ∃ p : ℝ, p = ↑q :=
by
  sorry

end all_rationals_on_number_line_l1853_185302


namespace line_equation_problem_l1853_185392

theorem line_equation_problem
  (P : ℝ × ℝ)
  (h1 : (P.1 + P.2 - 2 = 0) ∧ (P.1 - P.2 + 4 = 0))
  (l : ℝ × ℝ → Prop)
  (h2 : ∀ A B : ℝ × ℝ, l A → l B → (∃ k, B.2 - A.2 = k * (B.1 - A.1)))
  (h3 : ∀ Q : ℝ × ℝ, l Q → (3 * Q.1 - 2 * Q.2 + 4 = 0)) :
  l P ↔ 3 * P.1 - 2 * P.2 + 9 = 0 := 
sorry

end line_equation_problem_l1853_185392


namespace problem_lean_statement_l1853_185349

def P (x : ℝ) : ℝ := x^2 - 3*x - 9

theorem problem_lean_statement :
  let a := 61
  let b := 109
  let c := 621
  let d := 39
  let e := 20
  a + b + c + d + e = 850 := 
by
  sorry

end problem_lean_statement_l1853_185349


namespace cos_5theta_l1853_185343

theorem cos_5theta (theta : ℝ) (h : Real.cos theta = 2 / 5) : Real.cos (5 * theta) = 2762 / 3125 := 
sorry

end cos_5theta_l1853_185343


namespace sum_of_box_dimensions_l1853_185383

theorem sum_of_box_dimensions (X Y Z : ℝ) (h1 : X * Y = 32) (h2 : X * Z = 50) (h3 : Y * Z = 80) :
  X + Y + Z = 25.5 * Real.sqrt 2 :=
by sorry

end sum_of_box_dimensions_l1853_185383


namespace fifth_term_geometric_sequence_l1853_185364

theorem fifth_term_geometric_sequence (x y : ℚ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x + y
    let a2 := x - y
    let a3 := x / y
    let a4 := x * y
    let r := (x - y)/(x + y)
    (a4 * r = (2 / 3)) :=
by
  -- Proof omitted
  sorry

end fifth_term_geometric_sequence_l1853_185364


namespace explicit_x_n_formula_l1853_185396

theorem explicit_x_n_formula (x y : ℕ → ℕ) (n : ℕ) :
  x 0 = 2 ∧ y 0 = 1 ∧
  (∀ n, x (n + 1) = x n ^ 2 + y n ^ 2) ∧
  (∀ n, y (n + 1) = 2 * x n * y n) →
  x n = (3 ^ (2 ^ n) + 1) / 2 :=
by
  sorry

end explicit_x_n_formula_l1853_185396


namespace trig_identity_l1853_185385

variable (α : ℝ)
variable (h : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (h₁ : Real.sin α = 4 / 5)

theorem trig_identity : Real.sin (α + Real.pi / 4) + Real.cos (α + Real.pi / 4) = -3 * Real.sqrt 2 / 5 := 
by 
  sorry

end trig_identity_l1853_185385


namespace no_negative_roots_of_polynomial_l1853_185313

def polynomial (x : ℝ) := x^4 - 5 * x^3 - 4 * x^2 - 7 * x + 4

theorem no_negative_roots_of_polynomial :
  ¬ ∃ (x : ℝ), x < 0 ∧ polynomial x = 0 :=
by
  sorry

end no_negative_roots_of_polynomial_l1853_185313


namespace folding_hexagon_quadrilateral_folding_hexagon_pentagon_l1853_185352

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem folding_hexagon_quadrilateral :
  (sum_of_interior_angles 4 = 360) :=
by
  sorry

theorem folding_hexagon_pentagon :
  (sum_of_interior_angles 5 = 540) :=
by
  sorry

end folding_hexagon_quadrilateral_folding_hexagon_pentagon_l1853_185352


namespace find_coordinates_of_P_l1853_185301

theorem find_coordinates_of_P (P : ℝ × ℝ) (hx : abs P.2 = 5) (hy : abs P.1 = 3) (hq : P.1 < 0 ∧ P.2 > 0) : 
  P = (-3, 5) := 
  sorry

end find_coordinates_of_P_l1853_185301


namespace leo_total_points_l1853_185319

theorem leo_total_points (x y : ℕ) (h1 : x + y = 50) :
  0.4 * (x : ℝ) * 3 + 0.5 * (y : ℝ) * 2 = 0.2 * (x : ℝ) + 50 :=
by sorry

end leo_total_points_l1853_185319


namespace sample_size_proportion_l1853_185303

theorem sample_size_proportion (n : ℕ) (ratio_A B C : ℕ) (A_sample : ℕ) (ratio_A_val : ratio_A = 5) (ratio_B_val : ratio_B = 2) (ratio_C_val : ratio_C = 3) (A_sample_val : A_sample = 15) (total_ratio : ratio_A + ratio_B + ratio_C = 10) : 
  15 / n = 5 / 10 → n = 30 :=
sorry

end sample_size_proportion_l1853_185303


namespace proof_of_acdb_l1853_185362

theorem proof_of_acdb
  (x a b c d : ℤ)
  (hx_eq : 7 * x - 8 * x = 20)
  (hx_form : (a + b * Real.sqrt c) / d = x)
  (hints : x = (4 + 2 * Real.sqrt 39) / 7)
  (int_cond : a = 4 ∧ b = 2 ∧ c = 39 ∧ d = 7) :
  a * c * d / b = 546 := by
sorry

end proof_of_acdb_l1853_185362


namespace find_a_8_l1853_185328

variable {α : Type*} [LinearOrderedField α]
variables (a : ℕ → α) (n : ℕ)

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) := ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition
def given_condition (a : ℕ → α) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- Main theorem to prove
theorem find_a_8 (h_arith : is_arithmetic_seq a) (h_cond : given_condition a) : a 8 = 24 :=
  sorry

end find_a_8_l1853_185328
