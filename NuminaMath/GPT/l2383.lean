import Mathlib

namespace product_of_primes_sum_101_l2383_238393

theorem product_of_primes_sum_101 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 101) : p * q = 194 := by
  sorry

end product_of_primes_sum_101_l2383_238393


namespace find_T_shirts_l2383_238350

variable (T S : ℕ)

-- Given conditions
def condition1 : S = 2 * T := sorry
def condition2 : T + S - (T + 3) = 15 := sorry

-- Prove that number of T-shirts T Norma left in the washer is 9
theorem find_T_shirts (h1 : S = 2 * T) (h2 : T + S - (T + 3) = 15) : T = 9 :=
  by
    sorry

end find_T_shirts_l2383_238350


namespace area_of_wrapping_paper_l2383_238397

theorem area_of_wrapping_paper (l w h: ℝ) (l_pos: 0 < l) (w_pos: 0 < w) (h_pos: 0 < h) :
  ∃ s: ℝ, s = l + w ∧ s^2 = (l + w)^2 :=
by 
  sorry

end area_of_wrapping_paper_l2383_238397


namespace range_of_a_l2383_238351

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a-1) * x^2 + 2 * (a-1) * x - 4 ≥ 0 -> false) ↔ -3 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l2383_238351


namespace new_average_after_multiplication_l2383_238309

theorem new_average_after_multiplication
  (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 7)
  (h2 : a = 25)
  (h3 : m = 5):
  (n * a * m / n) = 125 :=
by
  sorry


end new_average_after_multiplication_l2383_238309


namespace coordinates_of_N_l2383_238370

-- Define the given conditions
def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)
def minusThreeA : ℝ × ℝ := (-3, 6)
def vectorMN (N : ℝ × ℝ) : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define the required goal
theorem coordinates_of_N (N : ℝ × ℝ) : vectorMN N = minusThreeA → N = (2, 0) :=
by
  sorry

end coordinates_of_N_l2383_238370


namespace point_not_on_transformed_plane_l2383_238390

def point_A : ℝ × ℝ × ℝ := (4, 0, -3)

def plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - 1

def scale_factor : ℝ := 3

def transformed_plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - (scale_factor * 1)

theorem point_not_on_transformed_plane :
  transformed_plane_eq 4 0 (-3) ≠ 0 :=
by
  sorry

end point_not_on_transformed_plane_l2383_238390


namespace tetrahedron_edge_length_of_tangent_spheres_l2383_238376

theorem tetrahedron_edge_length_of_tangent_spheres (r : ℝ) (h₁ : r = 2) :
  ∃ s : ℝ, s = 4 :=
by
  sorry

end tetrahedron_edge_length_of_tangent_spheres_l2383_238376


namespace units_digit_of_eight_consecutive_odd_numbers_is_zero_l2383_238320

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem units_digit_of_eight_consecutive_odd_numbers_is_zero (n : ℤ)
  (h₀ : is_odd n) :
  ((n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) * (n + 14)) % 10 = 0) :=
sorry

end units_digit_of_eight_consecutive_odd_numbers_is_zero_l2383_238320


namespace solve_system_l2383_238384

theorem solve_system :
  ∃ x y : ℝ, (x + y = 5) ∧ (x + 2 * y = 8) ∧ (x = 2) ∧ (y = 3) :=
by
  sorry

end solve_system_l2383_238384


namespace point_in_quadrant_l2383_238394

theorem point_in_quadrant (a b : ℝ) (h1 : a - b > 0) (h2 : a * b < 0) : 
  (a > 0 ∧ b < 0) ∧ ¬(a > 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b < 0) := 
by 
  sorry

end point_in_quadrant_l2383_238394


namespace average_first_18_even_numbers_l2383_238372

theorem average_first_18_even_numbers : 
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  average = 19 :=
by
  -- Definitions
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  -- The claim
  show average = 19
  sorry

end average_first_18_even_numbers_l2383_238372


namespace solution_to_quadratic_inequality_l2383_238355

theorem solution_to_quadratic_inequality :
  {x : ℝ | x^2 + 3*x < 10} = {x : ℝ | -5 < x ∧ x < 2} :=
sorry

end solution_to_quadratic_inequality_l2383_238355


namespace total_surface_area_of_cylinder_l2383_238315

noncomputable def rectangle_length : ℝ := 4 * Real.pi
noncomputable def rectangle_width : ℝ := 2

noncomputable def cylinder_radius (length : ℝ) : ℝ := length / (2 * Real.pi)
noncomputable def cylinder_height (width : ℝ) : ℝ := width

noncomputable def cylinder_surface_area (radius height : ℝ) : ℝ :=
  2 * Real.pi * radius^2 + 2 * Real.pi * radius * height

theorem total_surface_area_of_cylinder :
  cylinder_surface_area (cylinder_radius rectangle_length) (cylinder_height rectangle_width) = 16 * Real.pi :=
by
  sorry

end total_surface_area_of_cylinder_l2383_238315


namespace droneSystemEquations_l2383_238306

-- Definitions based on conditions
def typeADrones (x y : ℕ) : Prop := x = (1/2 : ℝ) * (x + y) + 11
def typeBDrones (x y : ℕ) : Prop := y = (1/3 : ℝ) * (x + y) - 2

-- Theorem statement
theorem droneSystemEquations (x y : ℕ) :
  typeADrones x y ∧ typeBDrones x y ↔
  (x = (1/2 : ℝ) * (x + y) + 11 ∧ y = (1/3 : ℝ) * (x + y) - 2) :=
by sorry

end droneSystemEquations_l2383_238306


namespace A_speed_is_10_l2383_238380

noncomputable def A_walking_speed (v t : ℝ) := 
  v * (t + 7) = 140 ∧ v * (t + 7) = 20 * t

theorem A_speed_is_10 (v t : ℝ) 
  (h1 : v * (t + 7) = 140)
  (h2 : v * (t + 7) = 20 * t) :
  v = 10 :=
sorry

end A_speed_is_10_l2383_238380


namespace problem_l2383_238364

-- Define the problem
theorem problem {a b c : ℤ} (h1 : a = c + 1) (h2 : b - 1 = a) :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 = 6 := 
sorry

end problem_l2383_238364


namespace determine_time_Toronto_l2383_238329

noncomputable def timeDifferenceBeijingToronto: ℤ := -12

def timeBeijing: ℕ × ℕ := (1, 8) -- (day, hour) format for simplicity: October 1st, 8:00

def timeToronto: ℕ × ℕ := (30, 20) -- Expected result in (day, hour): September 30th, 20:00

theorem determine_time_Toronto :
  timeDifferenceBeijingToronto = -12 →
  timeBeijing = (1, 8) →
  timeToronto = (30, 20) :=
by
  -- proof to be written 
  sorry

end determine_time_Toronto_l2383_238329


namespace sum_m_n_l2383_238314

open Real

noncomputable def f (x : ℝ) : ℝ := |log x / log 2|

theorem sum_m_n (m n : ℝ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_mn : m < n) 
  (h_f_eq : f m = f n) (h_max_f : ∀ x : ℝ, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
  m + n = 5 / 2 :=
sorry

end sum_m_n_l2383_238314


namespace not_always_greater_quotient_l2383_238336

theorem not_always_greater_quotient (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : 0 < b) : ¬ (∀ b < 1, a / b > a) ∧ ¬ (∀ b > 1, a / b > a) :=
by sorry

end not_always_greater_quotient_l2383_238336


namespace rebecca_gemstones_needed_l2383_238321

-- Definitions for the conditions
def magnets_per_earring : Nat := 2
def buttons_per_magnet : Nat := 1 / 2
def gemstones_per_button : Nat := 3
def earrings_per_set : Nat := 2
def sets : Nat := 4

-- Statement to be proved
theorem rebecca_gemstones_needed : 
  gemstones_per_button * (buttons_per_magnet * (magnets_per_earring * (earrings_per_set * sets))) = 24 :=
by
  sorry

end rebecca_gemstones_needed_l2383_238321


namespace circles_intersect_at_two_points_l2383_238382

theorem circles_intersect_at_two_points : 
  let C1 := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9}
  let C2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 36}
  ∃ pts : Finset (ℝ × ℝ), pts.card = 2 ∧ ∀ p ∈ pts, p ∈ C1 ∧ p ∈ C2 := 
sorry

end circles_intersect_at_two_points_l2383_238382


namespace ryan_sandwiches_l2383_238333

theorem ryan_sandwiches (sandwich_slices : ℕ) (total_slices : ℕ) (h1 : sandwich_slices = 3) (h2 : total_slices = 15) :
  total_slices / sandwich_slices = 5 :=
by
  sorry

end ryan_sandwiches_l2383_238333


namespace probability_computation_l2383_238395

noncomputable def probability_inside_sphere : ℝ :=
  let volume_of_cube : ℝ := 64
  let volume_of_sphere : ℝ := (4/3) * Real.pi * (2^3)
  volume_of_sphere / volume_of_cube

theorem probability_computation :
  probability_inside_sphere = Real.pi / 6 :=
by
  sorry

end probability_computation_l2383_238395


namespace combination_of_15_3_l2383_238365

open Nat

theorem combination_of_15_3 : choose 15 3 = 455 :=
by
  -- The statement describes that the number of ways to choose 3 books out of 15 is 455
  sorry

end combination_of_15_3_l2383_238365


namespace fill_time_eight_faucets_l2383_238326

theorem fill_time_eight_faucets (r : ℝ) (h1 : 4 * r * 8 = 150) :
  8 * r * (50 / (8 * r)) * 60 = 80 := by
  sorry

end fill_time_eight_faucets_l2383_238326


namespace min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l2383_238310

theorem min_questions_to_find_phone_number : 
  ∃ n : ℕ, ∀ (N : ℕ), (N = 100000 → 2 ^ n ≥ N) ∧ (2 ^ (n - 1) < N) := sorry

-- In simpler form, since log_2(100000) ≈ 16.60965, we have:
theorem min_questions_to_find_phone_number_is_17 : 
  ∀ (N : ℕ), (N = 100000 → 17 = Nat.ceil (Real.logb 2 100000)) := sorry

end min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l2383_238310


namespace count_lines_in_2008_cube_l2383_238359

def num_lines_through_centers_of_unit_cubes (n : ℕ) : ℕ :=
  n * n * 3 + n * 2 * 3 + 4

theorem count_lines_in_2008_cube :
  num_lines_through_centers_of_unit_cubes 2008 = 12115300 :=
by
  -- The actual proof would go here
  sorry

end count_lines_in_2008_cube_l2383_238359


namespace solution_exists_l2383_238327

theorem solution_exists (x y z u v : ℕ) (hx : x > 2000) (hy : y > 2000) (hz : z > 2000) (hu : u > 2000) (hv : v > 2000) : 
  x^2 + y^2 + z^2 + u^2 + v^2 = x * y * z * u * v - 65 :=
sorry

end solution_exists_l2383_238327


namespace num_non_divisible_by_3_divisors_l2383_238361

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l2383_238361


namespace cookie_distribution_l2383_238311

def trays := 4
def cookies_per_tray := 24
def total_cookies := trays * cookies_per_tray
def packs := 8
def cookies_per_pack := total_cookies / packs

theorem cookie_distribution : cookies_per_pack = 12 := by
  sorry

end cookie_distribution_l2383_238311


namespace even_function_order_l2383_238346

noncomputable def f (m : ℝ) (x : ℝ) := (m - 1) * x^2 + 6 * m * x + 2

theorem even_function_order (m : ℝ) (h_even : ∀ x : ℝ, f m (-x) = f m x) : 
  m = 0 ∧ f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
sorry

end even_function_order_l2383_238346


namespace cages_needed_l2383_238377

theorem cages_needed (initial_puppies sold_puppies puppies_per_cage : ℕ) (h1 : initial_puppies = 13) (h2 : sold_puppies = 7) (h3 : puppies_per_cage = 2) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := 
by
  sorry

end cages_needed_l2383_238377


namespace neg_one_quadratic_residue_iff_l2383_238332

theorem neg_one_quadratic_residue_iff (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) : 
  (∃ x : ℤ, x^2 ≡ -1 [ZMOD p]) ↔ p % 4 = 1 :=
sorry

end neg_one_quadratic_residue_iff_l2383_238332


namespace find_s_l2383_238325

theorem find_s (k s : ℝ) (h1 : 5 = k * 2^s) (h2 : 45 = k * 8^s) : s = (Real.log 9) / (2 * Real.log 2) :=
by
  sorry

end find_s_l2383_238325


namespace number_of_cats_adopted_l2383_238373

theorem number_of_cats_adopted (c : ℕ) 
  (h1 : 50 * c + 3 * 100 + 2 * 150 = 700) :
  c = 2 :=
by
  sorry

end number_of_cats_adopted_l2383_238373


namespace glen_animals_total_impossible_l2383_238307

theorem glen_animals_total_impossible (t : ℕ) :
  ¬ (∃ t : ℕ, 41 * t = 108) := sorry

end glen_animals_total_impossible_l2383_238307


namespace even_function_derivative_l2383_238362

theorem even_function_derivative (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_deriv_pos : ∀ x > 0, deriv f x = (x - 1) * (x - 2)) : f (-2) < f 1 :=
sorry

end even_function_derivative_l2383_238362


namespace value_of_a_l2383_238341

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem value_of_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 6 * x + 3 * a * x^2) →
  deriv (f a) (-1) = 6 → a = 4 :=
by
  -- Proof will be filled in here
  sorry

end value_of_a_l2383_238341


namespace base_6_to_base_10_exact_value_l2383_238308

def base_6_to_base_10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_6_to_base_10_exact_value : base_6_to_base_10 154 = 70 := by
  rfl

end base_6_to_base_10_exact_value_l2383_238308


namespace minimum_apples_to_guarantee_18_one_color_l2383_238389

theorem minimum_apples_to_guarantee_18_one_color :
  let red := 32
  let green := 24
  let yellow := 22
  let blue := 15
  let orange := 14
  ∀ n, (n >= 81) →
  (∃ red_picked green_picked yellow_picked blue_picked orange_picked : ℕ,
    red_picked + green_picked + yellow_picked + blue_picked + orange_picked = n
    ∧ red_picked ≤ red ∧ green_picked ≤ green ∧ yellow_picked ≤ yellow ∧ blue_picked ≤ blue ∧ orange_picked ≤ orange
    ∧ (red_picked = 18 ∨ green_picked = 18 ∨ yellow_picked = 18 ∨ blue_picked = 18 ∨ orange_picked = 18)) :=
by {
  -- The proof is omitted for now.
  sorry
}

end minimum_apples_to_guarantee_18_one_color_l2383_238389


namespace bacon_strips_needed_l2383_238388

theorem bacon_strips_needed (plates : ℕ) (eggs_per_plate : ℕ) (bacon_per_plate : ℕ) (customers : ℕ) :
  eggs_per_plate = 2 →
  bacon_per_plate = 2 * eggs_per_plate →
  customers = 14 →
  plates = customers →
  plates * bacon_per_plate = 56 := by
  sorry

end bacon_strips_needed_l2383_238388


namespace john_unanswered_problems_is_9_l2383_238312

variables (x y z : ℕ)

theorem john_unanswered_problems_is_9 (h1 : 5 * x + 2 * z = 93)
                                      (h2 : 4 * x - y = 54)
                                      (h3 : x + y + z = 30) : 
  z = 9 :=
by 
  sorry

end john_unanswered_problems_is_9_l2383_238312


namespace Greg_gold_amount_l2383_238323

noncomputable def gold_amounts (G K : ℕ) : Prop :=
  G = K / 4 ∧ G + K = 100

theorem Greg_gold_amount (G K : ℕ) (h : gold_amounts G K) : G = 20 := 
by
  sorry

end Greg_gold_amount_l2383_238323


namespace smallest_possible_value_of_other_integer_l2383_238378

theorem smallest_possible_value_of_other_integer (x : ℕ) (x_pos : 0 < x) (a b : ℕ) (h1 : a = 77) 
    (h2 : gcd a b = x + 7) (h3 : lcm a b = x * (x + 7)) : b = 22 :=
sorry

end smallest_possible_value_of_other_integer_l2383_238378


namespace sequence_an_value_l2383_238331

theorem sequence_an_value (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hS : ∀ n, 4 * S n = (a n - 1) * (a n + 3))
  (h_pos : ∀ n, 0 < a n)
  (n_nondec : ∀ n, a (n + 1) - a n = 2) :
  a 1005 = 2011 := 
sorry

end sequence_an_value_l2383_238331


namespace page_number_added_twice_l2383_238347

-- Define the sum of natural numbers from 1 to n
def sum_nat (n: ℕ): ℕ := n * (n + 1) / 2

-- Incorrect sum due to one page number being counted twice
def incorrect_sum (n p: ℕ): ℕ := sum_nat n + p

-- Declaring the known conditions as Lean definitions
def n : ℕ := 70
def incorrect_sum_val : ℕ := 2550

-- Lean theorem statement to be proven
theorem page_number_added_twice :
  ∃ p, incorrect_sum n p = incorrect_sum_val ∧ p = 65 := by
  sorry

end page_number_added_twice_l2383_238347


namespace arithmetic_seq_b3_b6_l2383_238374

theorem arithmetic_seq_b3_b6 (b : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, b n = b 1 + n * d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_b4_b5 : b 4 * b 5 = 30) :
  b 3 * b 6 = 28 := 
sorry

end arithmetic_seq_b3_b6_l2383_238374


namespace cj_more_stamps_than_twice_kj_l2383_238304

variable (C K A : ℕ) (x : ℕ)

theorem cj_more_stamps_than_twice_kj :
  (C = 2 * K + x) →
  (K = A / 2) →
  (C + K + A = 930) →
  (A = 370) →
  (x = 25) →
  (C - 2 * K = 5) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cj_more_stamps_than_twice_kj_l2383_238304


namespace average_riding_speed_l2383_238339

theorem average_riding_speed
  (initial_reading : ℕ) (final_reading : ℕ) (time_day1 : ℕ) (time_day2 : ℕ)
  (h_initial : initial_reading = 2332)
  (h_final : final_reading = 2552)
  (h_time_day1 : time_day1 = 5)
  (h_time_day2 : time_day2 = 4) :
  (final_reading - initial_reading) / (time_day1 + time_day2) = 220 / 9 :=
by
  sorry

end average_riding_speed_l2383_238339


namespace quadratic_decreasing_l2383_238348

theorem quadratic_decreasing (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 ≤ x2 → x2 ≤ 4 → (x1^2 + 4*a*x1 - 2) ≥ (x2^2 + 4*a*x2 - 2)) : a ≤ -2 := 
by
  sorry

end quadratic_decreasing_l2383_238348


namespace kevin_feeds_each_toad_3_worms_l2383_238302

theorem kevin_feeds_each_toad_3_worms
  (num_toads : ℕ) (minutes_per_worm : ℕ) (hours_to_minutes : ℕ) (total_minutes : ℕ)
  (H1 : num_toads = 8)
  (H2 : minutes_per_worm = 15)
  (H3 : hours_to_minutes = 60)
  (H4 : total_minutes = 6 * hours_to_minutes)
  :
  total_minutes / minutes_per_worm / num_toads = 3 :=
sorry

end kevin_feeds_each_toad_3_worms_l2383_238302


namespace swim_time_CBA_l2383_238392

theorem swim_time_CBA (d t_down t_still t_upstream: ℝ) 
  (h1 : d = 1) 
  (h2 : t_down = 1 / (6 / 5))
  (h3 : t_still = 1)
  (h4 : t_upstream = (4 / 5) / 2)
  (total_time_down : (t_down + t_still) = 1)
  (total_time_up : (t_still + t_down) = 2) :
  (t_upstream * (d - (d / 5))) / 2 = 5 / 2 :=
by sorry

end swim_time_CBA_l2383_238392


namespace solution_set_eq_l2383_238354

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def decreasing_condition (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x1 * f (x1) - x2 * f (x2)) / (x1 - x2) < 0

variable (f : ℝ → ℝ)
variable (h_odd : odd_function f)
variable (h_minus_2_zero : f (-2) = 0)
variable (h_decreasing : decreasing_condition f)

theorem solution_set_eq :
  {x : ℝ | f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
sorry

end solution_set_eq_l2383_238354


namespace percent_of_b_l2383_238399

theorem percent_of_b (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 2.5 * a) : c = 0.1 * b := 
by
  sorry

end percent_of_b_l2383_238399


namespace intersection_point_of_planes_l2383_238343

theorem intersection_point_of_planes :
  ∃ (x y z : ℚ), 
    3 * x - y + 4 * z = 2 ∧ 
    -3 * x + 4 * y - 3 * z = 4 ∧ 
    -x + y - z = 5 ∧ 
    x = -55 ∧ 
    y = -11 ∧ 
    z = 39 := 
by
  sorry

end intersection_point_of_planes_l2383_238343


namespace LaKeisha_needs_to_mow_more_sqft_l2383_238366

noncomputable def LaKeisha_price_per_sqft : ℝ := 0.10
noncomputable def LaKeisha_book_cost : ℝ := 150
noncomputable def LaKeisha_mowed_sqft : ℕ := 3 * 20 * 15
noncomputable def LaKeisha_earnings_so_far : ℝ := LaKeisha_mowed_sqft * LaKeisha_price_per_sqft

theorem LaKeisha_needs_to_mow_more_sqft (additional_sqft_needed : ℝ) :
  additional_sqft_needed = (LaKeisha_book_cost - LaKeisha_earnings_so_far) / LaKeisha_price_per_sqft → 
  additional_sqft_needed = 600 :=
by
  sorry

end LaKeisha_needs_to_mow_more_sqft_l2383_238366


namespace quadratic_roots_l2383_238367

theorem quadratic_roots : ∀ x : ℝ, (x^2 - 6 * x + 5 = 0) ↔ (x = 5 ∨ x = 1) :=
by sorry

end quadratic_roots_l2383_238367


namespace ab_necessary_but_not_sufficient_l2383_238301

theorem ab_necessary_but_not_sufficient (a b : ℝ) (i : ℂ) (hi : i^2 = -1) : 
  ab < 0 → ¬ (ab >= 0) ∧ (¬ (ab <= 0)) → (z = i * (a + b * i)) ∧ a > 0 ∧ -b > 0 := 
  sorry

end ab_necessary_but_not_sufficient_l2383_238301


namespace find_a_and_b_min_value_expression_l2383_238379

universe u

-- Part (1): Prove the values of a and b
theorem find_a_and_b :
    (∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) →
    a = 1 ∧ b = 2 :=
sorry

-- Part (2): Given a = 1 and b = 2 prove the minimum value of 2x + y + 3
theorem min_value_expression :
    (1 / (x + 1) + 2 / (y + 1) = 1) →
    (x > 0) →
    (y > 0) →
    ∀ x y : ℝ, 2 * x + y + 3 ≥ 8 :=
sorry

end find_a_and_b_min_value_expression_l2383_238379


namespace speed_ratio_bus_meets_Vasya_first_back_trip_time_l2383_238391

namespace TransportProblem

variable (d : ℝ) -- distance from point A to B
variable (v_bus : ℝ) -- bus speed
variable (v_Vasya : ℝ) -- Vasya's speed
variable (v_Petya : ℝ) -- Petya's speed

-- Conditions
axiom bus_speed : v_bus * 3 = d
axiom bus_meet_Vasya_second_trip : 7.5 * v_Vasya = 0.5 * d
axiom bus_meet_Petya_at_B : 9 * v_Petya = d
axiom bus_start_time : d / v_bus = 3

theorem speed_ratio: (v_Vasya / v_Petya) = (3 / 5) :=
  sorry

theorem bus_meets_Vasya_first_back_trip_time: ∃ (x: ℕ), x = 11 :=
  sorry

end TransportProblem

end speed_ratio_bus_meets_Vasya_first_back_trip_time_l2383_238391


namespace Kim_sales_on_Friday_l2383_238357

theorem Kim_sales_on_Friday (tuesday_sales : ℕ) (tuesday_discount_rate : ℝ) 
    (monday_increase_rate : ℝ) (wednesday_increase_rate : ℝ) 
    (thursday_decrease_rate : ℝ) (friday_increase_rate : ℝ) 
    (final_friday_sales : ℕ) :
    tuesday_sales = 800 →
    tuesday_discount_rate = 0.05 →
    monday_increase_rate = 0.50 →
    wednesday_increase_rate = 1.5 →
    thursday_decrease_rate = 0.20 →
    friday_increase_rate = 1.3 →
    final_friday_sales = 1310 :=
by
  sorry

end Kim_sales_on_Friday_l2383_238357


namespace evaluate_expression_l2383_238386

theorem evaluate_expression : 3 + 5 * 2^3 - 4 / 2 + 7 * 3 = 62 := 
  by sorry

end evaluate_expression_l2383_238386


namespace rebus_solution_l2383_238385

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l2383_238385


namespace min_value_of_function_l2383_238305

theorem min_value_of_function : 
  ∃ x > 2, ∀ y > 2, (y + 1 / (y - 2)) ≥ 4 ∧ (x + 1 / (x - 2)) = 4 := 
by sorry

end min_value_of_function_l2383_238305


namespace remainder_div_3973_28_l2383_238387

theorem remainder_div_3973_28 : (3973 % 28) = 9 := by
  sorry

end remainder_div_3973_28_l2383_238387


namespace graph_not_pass_through_second_quadrant_l2383_238368

theorem graph_not_pass_through_second_quadrant 
    (k : ℝ) (b : ℝ) (h1 : k = 1) (h2 : b = -2) : 
    ¬ ∃ (x y : ℝ), y = k * x + b ∧ x < 0 ∧ y > 0 := 
by
  sorry

end graph_not_pass_through_second_quadrant_l2383_238368


namespace square_of_fourth_power_of_fourth_smallest_prime_l2383_238334

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- Define the square of the fourth power of that number
def square_of_fourth_power (n : ℕ) : ℕ := (n^4)^2

-- Prove the main statement
theorem square_of_fourth_power_of_fourth_smallest_prime : square_of_fourth_power fourth_smallest_prime = 5764801 :=
by
  sorry

end square_of_fourth_power_of_fourth_smallest_prime_l2383_238334


namespace kara_water_intake_l2383_238344

-- Definitions based on the conditions
def daily_doses := 3
def week1_days := 7
def week2_days := 7
def forgot_doses_day := 2
def total_weeks := 2
def total_water := 160

-- The statement to prove
theorem kara_water_intake :
  let total_doses := (daily_doses * week1_days) + (daily_doses * week2_days - forgot_doses_day)
  ∃ (water_per_dose : ℕ), water_per_dose * total_doses = total_water ∧ water_per_dose = 4 :=
by
  sorry

end kara_water_intake_l2383_238344


namespace polygon_sides_eq_eight_l2383_238330

theorem polygon_sides_eq_eight (n : ℕ) (h : (n - 2) * 180 = 3 * 360) : n = 8 := by 
  sorry

end polygon_sides_eq_eight_l2383_238330


namespace finite_transformation_l2383_238356

-- Define the function representing the number transformation
def transform (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 5

-- Define the predicate stating that the process terminates
def process_terminates (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ transform^[k] n = 1

-- Lean 4 statement for the theorem
theorem finite_transformation (n : ℕ) (h : n > 1) : process_terminates n ↔ ¬ (∃ m : ℕ, m > 0 ∧ n = 5 * m) :=
by
  sorry

end finite_transformation_l2383_238356


namespace simplify_expression_l2383_238300

theorem simplify_expression (w : ℝ) :
  2 * w^2 + 3 - 4 * w^2 + 2 * w - 6 * w + 4 = -2 * w^2 - 4 * w + 7 :=
by
  sorry

end simplify_expression_l2383_238300


namespace min_value_of_one_over_a_plus_one_over_b_l2383_238317

theorem min_value_of_one_over_a_plus_one_over_b (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 12) :
  (∃ c : ℝ, (c = 1/ a + 1 / b) ∧ c = 1 / 3) :=
sorry

end min_value_of_one_over_a_plus_one_over_b_l2383_238317


namespace alpha_values_perpendicular_l2383_238360

theorem alpha_values_perpendicular
  (α : ℝ)
  (h1 : α ∈ Set.Ico 0 (2 * Real.pi))
  (h2 : ∀ (x y : ℝ), x * Real.cos α - y - 1 = 0 → x + y * Real.sin α + 1 = 0 → false):
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
by
  sorry

end alpha_values_perpendicular_l2383_238360


namespace part1_ABC_inquality_part2_ABCD_inquality_l2383_238358

theorem part1_ABC_inquality (a b c ABC : ℝ) : 
  (ABC <= (a^2 + b^2) / 4) -> 
  (ABC <= (b^2 + c^2) / 4) -> 
  (ABC <= (a^2 + c^2) / 4) -> 
    (ABC < (a^2 + b^2 + c^2) / 6) :=
sorry

theorem part2_ABCD_inquality (a b c d ABC BCD CDA DAB ABCD : ℝ) :
  (ABCD = 1/2 * ((ABC) + (BCD) + (CDA) + (DAB))) -> 
  (ABC < (a^2 + b^2 + c^2) / 6) -> 
  (BCD < (b^2 + c^2 + d^2) / 6) -> 
  (CDA < (c^2 + d^2 + a^2) / 6) -> 
  (DAB < (d^2 + a^2 + b^2) / 6) -> 
    (ABCD < (a^2 + b^2 + c^2 + d^2) / 6) :=
sorry

end part1_ABC_inquality_part2_ABCD_inquality_l2383_238358


namespace min_value_of_reciprocal_sum_l2383_238375

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y = 2) : 
  ∃ (z : ℝ), z = (1 / x + 1 / y) ∧ z = (3 / 2 + Real.sqrt 2) :=
sorry

end min_value_of_reciprocal_sum_l2383_238375


namespace gcd_117_182_evaluate_polynomial_l2383_238338

-- Problem 1: Prove that GCD of 117 and 182 is 13
theorem gcd_117_182 : Int.gcd 117 182 = 13 := 
by
  sorry

-- Problem 2: Prove that evaluating the polynomial at x = -1 results in 12
noncomputable def f : ℤ → ℤ := λ x => 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

theorem evaluate_polynomial : f (-1) = 12 := 
by
  sorry

end gcd_117_182_evaluate_polynomial_l2383_238338


namespace james_running_increase_l2383_238349

theorem james_running_increase (initial_miles_per_week : ℕ) (percent_increase : ℝ) (total_days : ℕ) (days_in_week : ℕ) :
  initial_miles_per_week = 100 →
  percent_increase = 0.2 →
  total_days = 280 →
  days_in_week = 7 →
  ∃ miles_per_week_to_add : ℝ, miles_per_week_to_add = 3 :=
by
  intros
  sorry

end james_running_increase_l2383_238349


namespace perfect_square_l2383_238318

theorem perfect_square (a b : ℝ) : a^2 + 2 * a * b + b^2 = (a + b)^2 := by
  sorry

end perfect_square_l2383_238318


namespace Katya_saves_enough_l2383_238342

theorem Katya_saves_enough {h c_pool_sauna x y : ℕ} (hc : h = 275) (hcs : c_pool_sauna = 250)
  (hx : x = y + 200) (heq : x + y = c_pool_sauna) : (h / (c_pool_sauna - x)) = 11 :=
by
  sorry

end Katya_saves_enough_l2383_238342


namespace original_denominator_is_nine_l2383_238340

theorem original_denominator_is_nine (d : ℕ) : 
  (2 + 5) / (d + 5) = 1 / 2 → d = 9 := 
by sorry

end original_denominator_is_nine_l2383_238340


namespace fish_minimum_catch_l2383_238319

theorem fish_minimum_catch (a1 a2 a3 a4 a5 : ℕ) (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
  (h_non_increasing : a1 ≥ a2 ∧ a2 ≥ a3 ∧ a3 ≥ a4 ∧ a4 ≥ a5) : 
  a1 + a3 + a5 ≥ 50 :=
sorry

end fish_minimum_catch_l2383_238319


namespace smallest_n_for_congruence_l2383_238335

theorem smallest_n_for_congruence : ∃ n : ℕ, 0 < n ∧ 7^n % 5 = n^4 % 5 ∧ (∀ m : ℕ, 0 < m ∧ 7^m % 5 = m^4 % 5 → n ≤ m) ∧ n = 4 :=
by
  sorry

end smallest_n_for_congruence_l2383_238335


namespace magnitude_v_l2383_238381

open Complex

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * Complex.I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 5 := by
  sorry

end magnitude_v_l2383_238381


namespace unique_natural_in_sequences_l2383_238353

def seq_x (n : ℕ) : ℤ := if n = 0 then 10 else if n = 1 then 10 else seq_x (n - 2) * (seq_x (n - 1) + 1) + 1
def seq_y (n : ℕ) : ℤ := if n = 0 then -10 else if n = 1 then -10 else (seq_y (n - 1) + 1) * seq_y (n - 2) + 1

theorem unique_natural_in_sequences (k : ℕ) (i j : ℕ) :
  seq_x i = k → seq_y j ≠ k :=
by
  sorry

end unique_natural_in_sequences_l2383_238353


namespace geometric_sequence_fifth_term_l2383_238303

theorem geometric_sequence_fifth_term
  (a : ℕ) (r : ℕ)
  (h₁ : a = 3)
  (h₂ : a * r^3 = 243) :
  a * r^4 = 243 :=
by
  sorry

end geometric_sequence_fifth_term_l2383_238303


namespace volume_of_prism_is_429_l2383_238328

theorem volume_of_prism_is_429 (x y z : ℝ) (h1 : x * y = 56) (h2 : y * z = 57) (h3 : z * x = 58) : 
  x * y * z = 429 :=
by
  sorry

end volume_of_prism_is_429_l2383_238328


namespace find_m_l2383_238363

theorem find_m 
  (m : ℤ) 
  (h1 : ∀ x y : ℤ, -3 * x + y = m → 2 * x + y = 28 → x = -6) : 
  m = 58 :=
by 
  sorry

end find_m_l2383_238363


namespace not_necessarily_divisor_l2383_238396

def consecutive_product (k : ℤ) : ℤ := k * (k + 1) * (k + 2) * (k + 3)

theorem not_necessarily_divisor (k : ℤ) (hk : 8 ∣ consecutive_product k) : ¬ (48 ∣ consecutive_product k) :=
sorry

end not_necessarily_divisor_l2383_238396


namespace second_term_is_4_l2383_238322

-- Define the arithmetic sequence conditions
variables (a d : ℝ) -- first term a, common difference d

-- The condition given in the problem
def sum_first_and_third_term (a d : ℝ) : Prop :=
  a + (a + 2 * d) = 8

-- What we need to prove
theorem second_term_is_4 (a d : ℝ) (h : sum_first_and_third_term a d) : a + d = 4 :=
sorry

end second_term_is_4_l2383_238322


namespace positive_difference_is_correct_l2383_238371

/-- Angela's compounded interest parameters -/
def angela_initial_deposit : ℝ := 9000
def angela_interest_rate : ℝ := 0.08
def years : ℕ := 25

/-- Bob's simple interest parameters -/
def bob_initial_deposit : ℝ := 11000
def bob_interest_rate : ℝ := 0.09

/-- Compound interest calculation for Angela -/
def angela_balance : ℝ := angela_initial_deposit * (1 + angela_interest_rate) ^ years

/-- Simple interest calculation for Bob -/
def bob_balance : ℝ := bob_initial_deposit * (1 + bob_interest_rate * years)

/-- Difference calculation -/
def balance_difference : ℝ := angela_balance - bob_balance

/-- The positive difference between their balances to the nearest dollar -/
theorem positive_difference_is_correct :
  abs (round balance_difference) = 25890 :=
by
  sorry

end positive_difference_is_correct_l2383_238371


namespace average_runs_in_second_set_l2383_238316

theorem average_runs_in_second_set
  (avg_first_set : ℕ → ℕ → ℕ)
  (avg_all_matches : ℕ → ℕ → ℕ)
  (avg1 : ℕ := avg_first_set 20 30)
  (avg2 : ℕ := avg_all_matches 30 25) :
  ∃ (A : ℕ), A = 15 := by
  sorry

end average_runs_in_second_set_l2383_238316


namespace a_three_equals_35_l2383_238345

-- Define the mathematical sequences and functions
def S (n : ℕ) : ℕ := 5 * n^2 + 10 * n

def a (n : ℕ) : ℕ := S (n + 1) - S n

-- The proposition we want to prove
theorem a_three_equals_35 : a 2 = 35 := by 
  sorry

end a_three_equals_35_l2383_238345


namespace smallest_positive_period_one_increasing_interval_l2383_238398

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

def is_periodic_with_period (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem smallest_positive_period :
  is_periodic_with_period f Real.pi :=
sorry

theorem one_increasing_interval :
  is_increasing_on f (-(Real.pi / 8)) (3 * Real.pi / 8) :=
sorry

end smallest_positive_period_one_increasing_interval_l2383_238398


namespace meal_total_l2383_238369

noncomputable def meal_price (appetizer entree dessert drink sales_tax tip : ℝ) : ℝ :=
  let total_before_tax := appetizer + (2 * entree) + dessert + (2 * drink)
  let tax_amount := (sales_tax / 100) * total_before_tax
  let subtotal := total_before_tax + tax_amount
  let tip_amount := (tip / 100) * subtotal
  subtotal + tip_amount

theorem meal_total : 
  meal_price 9 20 11 6.5 7.5 22 = 95.75 :=
by
  sorry

end meal_total_l2383_238369


namespace find_b_l2383_238313

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
variables (h_area : (1/2) * a * c * (Real.sin B) = sqrt 3)
variables (h_B : B = Real.pi / 3)
variables (h_relation : a^2 + c^2 = 3 * a * c)

-- Claim
theorem find_b :
    b = 2 * Real.sqrt 2 :=
  sorry

end find_b_l2383_238313


namespace find_constant_term_l2383_238324

theorem find_constant_term (c : ℤ) (y : ℤ) (h1 : y = 2) (h2 : 5 * y^2 - 8 * y + c = 59) : c = 55 :=
by
  sorry

end find_constant_term_l2383_238324


namespace Jacob_age_is_3_l2383_238383

def Phoebe_age : ℕ := sorry
def Rehana_age : ℕ := 25
def Jacob_age (P : ℕ) : ℕ := 3 * P / 5

theorem Jacob_age_is_3 (P : ℕ) (h1 : Rehana_age + 5 = 3 * (P + 5)) (h2 : Rehana_age = 25) (h3 : Jacob_age P = 3) : Jacob_age P = 3 := by {
  sorry
}

end Jacob_age_is_3_l2383_238383


namespace nell_gave_cards_l2383_238352

theorem nell_gave_cards (c_original : ℕ) (c_left : ℕ) (cards_given : ℕ) :
  c_original = 528 → c_left = 252 → cards_given = c_original - c_left → cards_given = 276 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end nell_gave_cards_l2383_238352


namespace attendants_both_tools_l2383_238337

theorem attendants_both_tools (pencil_users pen_users only_one_type total_attendants both_types : ℕ)
  (h1 : pencil_users = 25) 
  (h2 : pen_users = 15) 
  (h3 : only_one_type = 20) 
  (h4 : total_attendants = only_one_type + both_types) 
  (h5 : total_attendants = pencil_users + pen_users - both_types) 
  : both_types = 10 :=
by
  -- Fill in the proof sub-steps here if needed
  sorry

end attendants_both_tools_l2383_238337
