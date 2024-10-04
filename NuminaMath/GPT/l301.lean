import Mathlib

namespace cottonwood_fiber_diameter_in_scientific_notation_l301_301469

theorem cottonwood_fiber_diameter_in_scientific_notation:
  (∃ (a : ℝ) (n : ℤ), 0.0000108 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10) → (0.0000108 = 1.08 * 10 ^ (-5)) :=
by
  sorry

end cottonwood_fiber_diameter_in_scientific_notation_l301_301469


namespace minimum_value_inequality_l301_301031

theorem minimum_value_inequality (m n : ℝ) (h₁ : m > n) (h₂ : n > 0) : m + (n^2 - mn + 4)/(m - n) ≥ 4 :=
  sorry

end minimum_value_inequality_l301_301031


namespace find_theta_ratio_l301_301300

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem find_theta_ratio (θ : ℝ) 
  (h : det2x2 (Real.sin θ) 2 (Real.cos θ) 3 = 0) : 
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 := 
by 
  sorry

end find_theta_ratio_l301_301300


namespace prime_factor_of_sum_l301_301159

theorem prime_factor_of_sum (n : ℤ) : ∃ p : ℕ, Nat.Prime p ∧ p = 2 ∧ (2 * n + 1 + 2 * n + 3 + 2 * n + 5 + 2 * n + 7) % p = 0 :=
by
  sorry

end prime_factor_of_sum_l301_301159


namespace find_m_l301_301327

-- Define the vector
def vec2 := (ℝ × ℝ)

-- Given vectors
def a : vec2 := (2, -1)
def c : vec2 := (-1, 2)

-- Definition of parallel vectors
def parallel (v1 v2 : vec2) := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

-- Problem Statement
theorem find_m (m : ℝ) (b : vec2 := (-1, m)) (h : parallel (a.1 + b.1, a.2 + b.2) c) : m = -1 :=
sorry

end find_m_l301_301327


namespace probability_one_die_shows_4_given_sum_7_l301_301395

def outcomes_with_sum_7 : List (ℕ × ℕ) := [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)]

def outcome_has_4 (outcome : ℕ × ℕ) : Bool :=
  outcome.fst = 4 ∨ outcome.snd = 4

def favorable_outcomes : List (ℕ × ℕ) :=
  outcomes_with_sum_7.filter outcome_has_4

theorem probability_one_die_shows_4_given_sum_7 :
  (favorable_outcomes.length : ℚ) / (outcomes_with_sum_7.length : ℚ) = 1 / 3 := sorry

end probability_one_die_shows_4_given_sum_7_l301_301395


namespace least_number_to_add_l301_301260

theorem least_number_to_add (n : ℕ) (divisor : ℕ) (modulus : ℕ) (h1 : n = 1076) (h2 : divisor = 23) (h3 : n % divisor = 18) :
  modulus = divisor - (n % divisor) ∧ modulus = 5 := 
sorry

end least_number_to_add_l301_301260


namespace heather_bicycled_distance_l301_301659

def speed : ℕ := 8
def time : ℕ := 5
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

theorem heather_bicycled_distance : distance speed time = 40 := by
  sorry

end heather_bicycled_distance_l301_301659


namespace sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l301_301859

-- Prove that the square root of 36 equals ±6
theorem sqrt_36_eq_pm6 : ∃ y : ℤ, y * y = 36 ∧ y = 6 ∨ y = -6 :=
by
  sorry

-- Prove that the arithmetic square root of sqrt(16) equals 2
theorem arith_sqrt_sqrt_16_eq_2 : ∃ z : ℝ, z * z = 16 ∧ z = 4 ∧ 2 * 2 = z :=
by
  sorry

-- Prove that the cube root of -27 equals -3
theorem cube_root_minus_27_eq_minus_3 : ∃ x : ℝ, x * x * x = -27 ∧ x = -3 :=
by
  sorry

end sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l301_301859


namespace money_lent_to_C_l301_301606

theorem money_lent_to_C (X : ℝ) (interest_rate : ℝ) (P_b : ℝ) (T_b : ℝ) (T_c : ℝ) (total_interest : ℝ) :
  interest_rate = 0.09 →
  P_b = 5000 →
  T_b = 2 →
  T_c = 4 →
  total_interest = 1980 →
  (P_b * interest_rate * T_b + X * interest_rate * T_c = total_interest) →
  X = 500 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end money_lent_to_C_l301_301606


namespace range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l301_301040

open Real

theorem range_a_of_abs_2x_minus_a_eq_1_two_real_solutions :
  {a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (abs (2^x1 - a) = 1) ∧ (abs (2^x2 - a) = 1)} = {a : ℝ | 1 < a} :=
by
  sorry

end range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l301_301040


namespace find_y_l301_301184

theorem find_y (x y : ℝ) (h₁ : x^2 - 2 * x + 5 = y + 3) (h₂ : x = 5) : y = 17 :=
by
  sorry

end find_y_l301_301184


namespace intersection_points_l301_301158

noncomputable def parabola (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 4
noncomputable def line (x : ℝ) : ℝ := -x + 2

theorem intersection_points :
  (parabola (-1 / 3) = line (-1 / 3) ∧ parabola (-2) = line (-2)) ∧
  (parabola (-1 / 3) = 7 / 3) ∧ (parabola (-2) = 4) :=
by
  sorry

end intersection_points_l301_301158


namespace apple_harvest_l301_301233

theorem apple_harvest (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 →
  num_sections = 8 →
  total_sacks = sacks_per_section * num_sections →
  total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end apple_harvest_l301_301233


namespace number_of_tens_in_sum_l301_301943

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l301_301943


namespace solve_for_x_l301_301365

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l301_301365


namespace value_of_expression_l301_301111

theorem value_of_expression (x : ℤ) (h : x = 3) : x^6 - 3 * x = 720 := by
  sorry

end value_of_expression_l301_301111


namespace tan_alpha_value_sin_cos_expression_l301_301908

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_value (α : ℝ) (h1 : Real.tan (α + Real.pi / 4) = 2) : tan_alpha α = 1 / 3 :=
by
  sorry

theorem sin_cos_expression (α : ℝ) (h2 : tan_alpha α = 1 / 3) :
  (Real.sin (2 * α) - Real.sin α ^ 2) / (1 + Real.cos (2 * α)) = 5 / 18 :=
by
  sorry

end tan_alpha_value_sin_cos_expression_l301_301908


namespace travel_time_correct_l301_301586

noncomputable def timeSpentOnRoad : Nat :=
  let startTime := 7  -- 7:00 AM in hours
  let endTime := 20   -- 8:00 PM in hours
  let totalJourneyTime := endTime - startTime
  let stopTimes := [25, 10, 25]  -- minutes
  let totalStopTime := stopTimes.foldl (· + ·) 0
  let stopTimeInHours := totalStopTime / 60
  totalJourneyTime - stopTimeInHours

theorem travel_time_correct : timeSpentOnRoad = 12 :=
by
  sorry

end travel_time_correct_l301_301586


namespace ellipse_range_k_l301_301492

theorem ellipse_range_k (k : ℝ) : 
  (∃ (x y : ℝ) (hk : \(\frac{x^2}{3+k} + \frac{y^2}{2-k} = 1\)), (3 + k > 0) ∧ (2 - k > 0) ∧ (3+k ≠ 2-k)) ↔ 
  k ∈ set.Ioo (-3) (-1/2) ∪ set.Ioo (-1/2) 2 := 
sorry

end ellipse_range_k_l301_301492


namespace movie_theater_people_l301_301276

def totalSeats : ℕ := 750
def emptySeats : ℕ := 218
def peopleWatching := totalSeats - emptySeats

theorem movie_theater_people :
  peopleWatching = 532 := by
  sorry

end movie_theater_people_l301_301276


namespace arrangement_possible_32_arrangement_possible_100_l301_301600

-- Problem (1)
theorem arrangement_possible_32 : 
  ∃ (f : Fin 32 → Fin 32), ∀ (a b : Fin 32), ∀ (i : Fin 32), 
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry

-- Problem (2)
theorem arrangement_possible_100 : 
  ∃ (f : Fin 100 → Fin 100), ∀ (a b : Fin 100), ∀ (i : Fin 100),
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry


end arrangement_possible_32_arrangement_possible_100_l301_301600


namespace gain_percent_of_50C_eq_25S_l301_301933

variable {C S : ℝ}

theorem gain_percent_of_50C_eq_25S (h : 50 * C = 25 * S) : 
  ((S - C) / C) * 100 = 100 :=
by
  sorry

end gain_percent_of_50C_eq_25S_l301_301933


namespace greatest_possible_multiple_of_4_l301_301709

theorem greatest_possible_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x^2 < 400) : x ≤ 16 :=
by 
sorry

end greatest_possible_multiple_of_4_l301_301709


namespace score_below_mean_l301_301630

theorem score_below_mean :
  ∃ (σ : ℝ), (74 - 2 * σ = 58) ∧ (98 - 74 = 3 * σ) :=
sorry

end score_below_mean_l301_301630


namespace correct_operation_l301_301854

theorem correct_operation (x : ℝ) (h : x ≠ 0) : 1 / x^(-2) = x^2 :=
by {
  -- Proof can go here
  sorry
}

end correct_operation_l301_301854


namespace range_of_a_l301_301919

noncomputable def f (a x : ℝ) : ℝ :=
  if x > 1 then x + a / x + 1 else -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f a x ≤ f a y) : -1 ≤ a ∧ a ≤ 1 := 
by
  sorry

end range_of_a_l301_301919


namespace no_real_solutions_for_equation_l301_301422

theorem no_real_solutions_for_equation :
  ¬ (∃ x : ℝ, (2 * x - 3 * x + 7)^2 + 2 = -|2 * x|) :=
by 
-- proof will go here
sorry

end no_real_solutions_for_equation_l301_301422


namespace find_larger_number_l301_301309

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := 
by 
  sorry

end find_larger_number_l301_301309


namespace volume_tetrahedron_l301_301096

def A1 := 4^2
def A2 := 3^2
def h := 1

theorem volume_tetrahedron:
  (h / 3 * (A1 + A2 + Real.sqrt (A1 * A2))) = 37 / 3 := by
  sorry

end volume_tetrahedron_l301_301096


namespace sum_of_tens_l301_301951

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l301_301951


namespace sarah_socks_l301_301705

theorem sarah_socks :
  ∃ (a b c : ℕ), a + b + c = 15 ∧ 2 * a + 4 * b + 5 * c = 45 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ (a = 8 ∨ a = 9) :=
by {
  sorry
}

end sarah_socks_l301_301705


namespace decreasing_by_25_l301_301399

theorem decreasing_by_25 (n : ℕ) (k : ℕ) (y : ℕ) (hy : 0 ≤ y ∧ y < 10^k) : 
  (n = 6 * 10^k + y → n / 10 = y / 25) → (∃ m, n = 625 * 10^m) := 
sorry

end decreasing_by_25_l301_301399


namespace number_of_poles_l301_301210

theorem number_of_poles (side_length : ℝ) (distance_between_poles : ℝ) 
  (h1 : side_length = 150) (h2 : distance_between_poles = 30) : 
  ((4 * side_length) / distance_between_poles) = 20 :=
by 
  -- Placeholder to indicate missing proof
  sorry

end number_of_poles_l301_301210


namespace problem1_problem2_problem3_problem4_l301_301218

-- Problem 1
theorem problem1 (x : ℤ) (h : 4 * x = 20) : x = 5 :=
sorry

-- Problem 2
theorem problem2 (x : ℤ) (h : x - 18 = 40) : x = 58 :=
sorry

-- Problem 3
theorem problem3 (x : ℤ) (h : x / 7 = 12) : x = 84 :=
sorry

-- Problem 4
theorem problem4 (n : ℚ) (h : 8 * n / 2 = 15) : n = 15 / 4 :=
sorry

end problem1_problem2_problem3_problem4_l301_301218


namespace sum_cos_to_tan_l301_301036

theorem sum_cos_to_tan (p q : ℕ) (h_rel_prime : Nat.gcd p q = 1) (h_pos : 0 < p ∧ 0 < q) (h_lt : p < 90 * q) 
  (h_sum : ∑ k in Finset.range 50, Real.cos (3 * (k : ℝ)) = Real.tan (p / q)) :
  p + q = 76 := by
  sorry

end sum_cos_to_tan_l301_301036


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301762

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301762


namespace locus_of_intersection_l301_301512

theorem locus_of_intersection
  (a b : ℝ) (h_a_nonzero : a ≠ 0) (h_b_nonzero : b ≠ 0) (h_neq : a ≠ b) :
  ∃ (x y : ℝ), 
    (∃ c : ℝ, y = (a/c)*x ∧ (x/b + y/c = 1)) 
    ∧ 
    ( (x - b/2)^2 / (b^2/4) + y^2 / (ab/4) = 1 ) :=
sorry

end locus_of_intersection_l301_301512


namespace box_surface_area_correct_l301_301089

-- Define the dimensions of the original cardboard.
def original_length : ℕ := 25
def original_width : ℕ := 40

-- Define the size of the squares removed from each corner.
def square_side : ℕ := 8

-- Define the surface area function.
def surface_area (length width : ℕ) (square_side : ℕ) : ℕ :=
  let area_remaining := (length * width) - 4 * (square_side * square_side)
  area_remaining

-- The theorem statement to prove
theorem box_surface_area_correct : surface_area original_length original_width square_side = 744 :=
by
  sorry

end box_surface_area_correct_l301_301089


namespace integral_abs_sin_l301_301017

theorem integral_abs_sin (I : ℤ := 0) : 
  ∫ x in 0..(2 * real.pi), |real.sin x| = 4 :=
by
  sorry

end integral_abs_sin_l301_301017


namespace cells_after_9_days_l301_301189

noncomputable def remaining_cells (initial : ℕ) (days : ℕ) : ℕ :=
  let rec divide_and_decay (cells: ℕ) (remaining_days: ℕ) : ℕ :=
    if remaining_days = 0 then cells
    else
      let divided := cells * 2
      let decayed := (divided * 9) / 10
      divide_and_decay decayed (remaining_days - 3)
  divide_and_decay initial days

theorem cells_after_9_days :
  remaining_cells 5 9 = 28 := by
  sorry

end cells_after_9_days_l301_301189


namespace total_votes_is_5000_l301_301057

theorem total_votes_is_5000 :
  ∃ (V : ℝ), 0.45 * V - 0.35 * V = 500 ∧ 0.35 * V - 0.20 * V = 350 ∧ V = 5000 :=
by
  sorry

end total_votes_is_5000_l301_301057


namespace remainder_first_six_primes_div_seventh_l301_301847

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301847


namespace min_value_of_f_l301_301312

noncomputable def f (x : ℝ) : ℝ := max (2 * x + 1) (5 - x)

theorem min_value_of_f : ∃ y, (∀ x : ℝ, f x ≥ y) ∧ y = 11 / 3 :=
by 
  sorry

end min_value_of_f_l301_301312


namespace exp_monotonic_iff_l301_301905

theorem exp_monotonic_iff (a b : ℝ) : (a > b) ↔ (Real.exp a > Real.exp b) :=
sorry

end exp_monotonic_iff_l301_301905


namespace michael_will_meet_two_times_l301_301208

noncomputable def michael_meetings : ℕ :=
  let michael_speed := 6 -- feet per second
  let pail_distance := 300 -- feet
  let truck_speed := 12 -- feet per second
  let truck_stop_time := 20 -- seconds
  let initial_distance := pail_distance -- feet
  let michael_position (t: ℕ) := michael_speed * t
  let truck_position (cycle: ℕ) := pail_distance * cycle
  let truck_cycle_time := pail_distance / truck_speed + truck_stop_time -- seconds per cycle
  let truck_position_at_time (t: ℕ) := 
    let cycle := t / truck_cycle_time
    let remaining_time := t % truck_cycle_time
    if remaining_time < (pail_distance / truck_speed) then 
      truck_position cycle + truck_speed * remaining_time
    else 
      truck_position cycle + pail_distance
  let distance_between := 
    λ (t: ℕ) => truck_position_at_time t - michael_position t
  let meet_time := 
    λ (t: ℕ) => if distance_between t = 0 then 1 else 0
  let total_meetings := 
    (List.range 300).map meet_time -- estimating within 300 seconds
    |> List.sum
  total_meetings

theorem michael_will_meet_two_times : michael_meetings = 2 :=
  sorry

end michael_will_meet_two_times_l301_301208


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301791

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301791


namespace card_arrangement_bound_l301_301019

theorem card_arrangement_bound : 
  ∀ (cards : ℕ) (cells : ℕ), cards = 1000 → cells = 1994 → 
  ∃ arrangements : ℕ, arrangements = cells - cards + 1 ∧ arrangements < 500000 :=
by {
  sorry
}

end card_arrangement_bound_l301_301019


namespace simplify_root_exponentiation_l301_301361

theorem simplify_root_exponentiation : (7 ^ (1 / 3) : ℝ) ^ 6 = 49 := by
  sorry

end simplify_root_exponentiation_l301_301361


namespace margie_drive_distance_l301_301532

-- Conditions
def car_mpg : ℝ := 45  -- miles per gallon
def gas_price : ℝ := 5 -- dollars per gallon
def money_spent : ℝ := 25 -- dollars

-- Question: Prove that Margie can drive 225 miles with $25 worth of gas.
theorem margie_drive_distance (h1 : car_mpg = 45) (h2 : gas_price = 5) (h3 : money_spent = 25) :
  money_spent / gas_price * car_mpg = 225 := by
  sorry

end margie_drive_distance_l301_301532


namespace coefficient_of_x3_in_expansion_l301_301520

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_x3_in_expansion :
  let T_r := λ r : ℕ, binom 4 r * (-2)^r in
  T_r 4 + T_r 1 * (-2)^0 = 8 :=
by
  sorry

end coefficient_of_x3_in_expansion_l301_301520


namespace understanding_related_to_gender_probability_of_understanding_l301_301711

noncomputable def chi_squared 
  (a b c d n : ℝ) : ℝ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem understanding_related_to_gender (a b c d n : ℝ) (alpha : ℝ) :
  a = 140 ∧ b = 60 ∧ c = 180 ∧ d = 20 ∧ n = 400 ∧ alpha = 0.001 →
  chi_squared a b c d n = 25 ∧ 25 > 10.828 :=
by sorry

noncomputable def binomial_prob 
  (n k : ℕ) (p : ℝ) : ℝ :=
  (finset.card (finset.filter (λ x, x = k) (finset.range n))) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_of_understanding (total_understanding : ℝ) 
  (total_population : ℝ) (selected_residents : ℕ) (k : ℕ) :
  total_understanding = 320 ∧ total_population = 400 ∧ selected_residents = 5 ∧ k = 3 →
  binomial_prob selected_residents k (total_understanding / total_population) = 128 / 625 :=
by sorry

end understanding_related_to_gender_probability_of_understanding_l301_301711


namespace find_integers_with_sum_and_gcd_l301_301173

theorem find_integers_with_sum_and_gcd {a b : ℕ} (h_sum : a + b = 104055) (h_gcd : Nat.gcd a b = 6937) :
  (a = 6937 ∧ b = 79118) ∨ (a = 13874 ∧ b = 90181) ∨ (a = 27748 ∧ b = 76307) ∨ (a = 48559 ∧ b = 55496) :=
sorry

end find_integers_with_sum_and_gcd_l301_301173


namespace solve_quadratic_l301_301088

theorem solve_quadratic (y : ℝ) :
  3 * y * (y - 1) = 2 * (y - 1) → y = 2 / 3 ∨ y = 1 :=
by
  sorry

end solve_quadratic_l301_301088


namespace polynomial_identity_l301_301686

theorem polynomial_identity
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)
  (h : (2*x + 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) :
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 729)
  ∧ (a_1 + a_3 + a_5 = 364)
  ∧ (a_2 + a_4 = 300) := sorry

end polynomial_identity_l301_301686


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301816

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301816


namespace num_terms_100_pow_10_as_sum_of_tens_l301_301961

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l301_301961


namespace prob_qualified_bulb_factory_a_l301_301343

-- Define the given probability of a light bulb being produced by Factory A
def prob_factory_a : ℝ := 0.7

-- Define the given pass rate (conditional probability) of Factory A's light bulbs
def pass_rate_factory_a : ℝ := 0.95

-- The goal is to prove that the probability of getting a qualified light bulb produced by Factory A is 0.665
theorem prob_qualified_bulb_factory_a : prob_factory_a * pass_rate_factory_a = 0.665 :=
by
  -- This is where the proof would be, but we'll use sorry to skip the proof
  sorry

end prob_qualified_bulb_factory_a_l301_301343


namespace lesser_solution_of_quadratic_l301_301580

theorem lesser_solution_of_quadratic :
  (∃ x y: ℝ, x ≠ y ∧ x^2 + 10*x - 24 = 0 ∧ y^2 + 10*y - 24 = 0 ∧ min x y = -12) :=
by {
  sorry
}

end lesser_solution_of_quadratic_l301_301580


namespace tan_sub_eq_minus_2sqrt3_l301_301542

theorem tan_sub_eq_minus_2sqrt3 
  (h1 : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3)
  (h2 : Real.tan (5 * Real.pi / 12) = 2 + Real.sqrt 3) : 
  Real.tan (Real.pi / 12) - Real.tan (5 * Real.pi / 12) = -2 * Real.sqrt 3 :=
by
  sorry

end tan_sub_eq_minus_2sqrt3_l301_301542


namespace student_ratio_l301_301967

theorem student_ratio (total_students below_eight eight_years above_eight : ℕ) 
  (h1 : below_eight = total_students * 20 / 100) 
  (h2 : eight_years = 72) 
  (h3 : total_students = 150) 
  (h4 : total_students = below_eight + eight_years + above_eight) :
  (above_eight / eight_years) = 2 / 3 :=
by
  sorry

end student_ratio_l301_301967


namespace LitterPatrol_pickup_l301_301550

theorem LitterPatrol_pickup :
  ∃ n : ℕ, n = 10 + 8 :=
sorry

end LitterPatrol_pickup_l301_301550


namespace ratio_A_to_B_l301_301010

noncomputable def A_annual_income : ℝ := 436800.0000000001
noncomputable def B_increase_rate : ℝ := 0.12
noncomputable def C_monthly_income : ℝ := 13000

noncomputable def A_monthly_income : ℝ := A_annual_income / 12
noncomputable def B_monthly_income : ℝ := C_monthly_income + (B_increase_rate * C_monthly_income)

theorem ratio_A_to_B :
  ((A_monthly_income / 80) : ℝ) = 455 ∧
  ((B_monthly_income / 80) : ℝ) = 182 :=
by
  sorry

end ratio_A_to_B_l301_301010


namespace tan_cot_theta_l301_301636

theorem tan_cot_theta 
  (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = (Real.sqrt 2) / 3) 
  (h2 : Real.pi / 2 < θ ∧ θ < Real.pi) : 
  Real.tan θ - (1 / Real.tan θ) = - (8 * Real.sqrt 2) / 7 := 
sorry

end tan_cot_theta_l301_301636


namespace problem_sol_max_distance_from_circle_to_line_l301_301973

noncomputable def max_distance_circle_line : ℝ :=
  let ρ (θ : ℝ) : ℝ := 8 * Real.sin θ
  let line (θ : ℝ) : Prop := θ = Real.pi / 3
  let circle_center := (0, 4)
  let line_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x
  let shortest_distance := 2  -- Already calculated in solution
  let radius := 4
  shortest_distance + radius

theorem problem_sol_max_distance_from_circle_to_line :
  max_distance_circle_line = 6 :=
by
  unfold max_distance_circle_line
  sorry

end problem_sol_max_distance_from_circle_to_line_l301_301973


namespace unique_solution_m_l301_301900

theorem unique_solution_m (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0 ∧ (∀ y₁ y₂ : ℝ, 3 * y₁^2 - 6 * y₂ + m = 0 → y₁ = y₂)) → m = 3 :=
by
  sorry

end unique_solution_m_l301_301900


namespace shaded_area_percentage_l301_301254

theorem shaded_area_percentage (side : ℕ) (total_shaded_area : ℕ) (expected_percentage : ℕ)
  (h1 : side = 5)
  (h2 : total_shaded_area = 15)
  (h3 : expected_percentage = 60) :
  ((total_shaded_area : ℚ) / (side * side) * 100) = expected_percentage :=
by
  sorry

end shaded_area_percentage_l301_301254


namespace marie_gift_boxes_l301_301694

theorem marie_gift_boxes
  (total_eggs : ℕ)
  (weight_per_egg : ℕ)
  (remaining_weight : ℕ)
  (melted_eggs_weight : ℕ)
  (eggs_per_box : ℕ)
  (total_boxes : ℕ)
  (H1 : total_eggs = 12)
  (H2 : weight_per_egg = 10)
  (H3 : remaining_weight = 90)
  (H4 : melted_eggs_weight = total_eggs * weight_per_egg - remaining_weight)
  (H5 : melted_eggs_weight / weight_per_egg = eggs_per_box)
  (H6 : total_eggs / eggs_per_box = total_boxes) :
  total_boxes = 4 := 
sorry

end marie_gift_boxes_l301_301694


namespace correct_expansion_l301_301000

variables {x y : ℝ}

theorem correct_expansion : 
  (-x + y)^2 = x^2 - 2 * x * y + y^2 := sorry

end correct_expansion_l301_301000


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301761

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301761


namespace solve_cubic_diophantine_l301_301026

theorem solve_cubic_diophantine :
  (∃ x y z : ℤ, x^3 + y^3 + z^3 - 3 * x * y * z = 2003) ↔ 
  (x = 667 ∧ y = 668 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 668 ∧ z = 667) :=
sorry

end solve_cubic_diophantine_l301_301026


namespace least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l301_301853

-- The numbers involved and the requirements described
def num : ℕ := 427398

def least_to_subtract_10 : ℕ := 8
def least_to_subtract_100 : ℕ := 98
def least_to_subtract_1000 : ℕ := 398

-- Proving the conditions:
-- 1. (num - least_to_subtract_10) is divisible by 10
-- 2. (num - least_to_subtract_100) is divisible by 100
-- 3. (num - least_to_subtract_1000) is divisible by 1000

theorem least_subtract_divisible_by_10 : (num - least_to_subtract_10) % 10 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_100 : (num - least_to_subtract_100) % 100 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_1000 : (num - least_to_subtract_1000) % 1000 = 0 := 
by 
  sorry

end least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l301_301853


namespace dwarfs_truthful_count_l301_301447

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l301_301447


namespace find_a_plus_2b_l301_301487

variable (a b : ℝ)

theorem find_a_plus_2b (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : 
  a + 2 * b = 0 := 
sorry

end find_a_plus_2b_l301_301487


namespace prime_sum_remainder_l301_301777

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301777


namespace part1_part2_l301_301920

-- Part 1: Proving the value of a given f(x) = a/x + 1 and f(-2) = 0
theorem part1 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a / x + 1) (h2 : f (-2) = 0) : a = 2 := 
by 
-- Placeholder for the proof
sorry

-- Part 2: Proving the value of f(4) given f(x) = 6/x + 1
theorem part2 (f : ℝ → ℝ) (h1 : ∀ x, f x = 6 / x + 1) : f 4 = 5 / 2 := 
by 
-- Placeholder for the proof
sorry

end part1_part2_l301_301920


namespace total_amount_paid_l301_301869

theorem total_amount_paid (num_sets : ℕ) (cost_per_set : ℕ) (tax_rate : ℝ) 
  (h1 : num_sets = 5) (h2 : cost_per_set = 6) (h3 : tax_rate = 0.1) : 
  let cost_before_tax := num_sets * cost_per_set
  let tax_amount := cost_before_tax * tax_rate
  let total_cost := cost_before_tax + tax_amount
  in total_cost = 33 :=
by
  sorry

end total_amount_paid_l301_301869


namespace equal_cost_at_20_minutes_l301_301106

/-- Define the cost functions for each telephone company -/
def united_cost (m : ℝ) : ℝ := 11 + 0.25 * m
def atlantic_cost (m : ℝ) : ℝ := 12 + 0.20 * m
def global_cost (m : ℝ) : ℝ := 13 + 0.15 * m

/-- Prove that at 20 minutes, the cost is the same for all three companies -/
theorem equal_cost_at_20_minutes : 
  united_cost 20 = atlantic_cost 20 ∧ atlantic_cost 20 = global_cost 20 :=
by
  sorry

end equal_cost_at_20_minutes_l301_301106


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301736

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301736


namespace determine_h_l301_301423

variable {R : Type*} [CommRing R]

def h_poly (x : R) : R := -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2

theorem determine_h (x : R) :
  (8*x^4 - 4*x^2 + 2 + h_poly x = 2*x^3 - 6*x + 4) ->
  h_poly x = -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2 :=
by
  intro h
  sorry

end determine_h_l301_301423


namespace combined_total_difference_l301_301895

theorem combined_total_difference :
  let Chris_cards := 18
  let Charlie_cards := 32
  let Diana_cards := 25
  let Ethan_cards := 40
  (Charlie_cards - Chris_cards) + (Diana_cards - Chris_cards) + (Ethan_cards - Chris_cards) = 43 :=
by
  let Chris_cards := 18
  let Charlie_cards := 32
  let Diana_cards := 25
  let Ethan_cards := 40
  have h1 : Charlie_cards - Chris_cards = 14 := by sorry
  have h2 : Diana_cards - Chris_cards = 7 := by sorry
  have h3 : Ethan_cards - Chris_cards = 22 := by sorry
  show (Charlie_cards - Chris_cards) + (Diana_cards - Chris_cards) + (Ethan_cards - Chris_cards) = 43 from
    by rw [h1, h2, h3]; exact (14 + 7 + 22).symm

end combined_total_difference_l301_301895


namespace volleyball_match_probabilities_l301_301225

noncomputable def probability_of_team_A_winning : ℚ := (2 / 3) ^ 3
noncomputable def probability_of_team_B_winning_3_0 : ℚ := 1 / 3
noncomputable def probability_of_team_B_winning_3_1 : ℚ := (2 / 3) * (1 / 3)
noncomputable def probability_of_team_B_winning_3_2 : ℚ := (2 / 3) ^ 2 * (1 / 3)

theorem volleyball_match_probabilities :
  probability_of_team_A_winning = 8 / 27 ∧
  probability_of_team_B_winning_3_0 = 1 / 3 ∧
  probability_of_team_B_winning_3_1 ≠ 1 / 9 ∧
  probability_of_team_B_winning_3_2 ≠ 4 / 9 :=
by
  sorry

end volleyball_match_probabilities_l301_301225


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301815

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301815


namespace trig_identity_l301_301245

theorem trig_identity : 
  Real.sin (600 * Real.pi / 180) + Real.tan (240 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l301_301245


namespace dwarfs_truthful_count_l301_301440

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l301_301440


namespace min_value_correct_l301_301169

noncomputable def min_value (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)] : ℝ :=
  if x + y = 1 then (a / x + b / y) else 0

theorem min_value_correct (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)]
  (h : x + y = 1) : min_value a b x y = (Real.sqrt a + Real.sqrt b)^2 :=
by
  sorry

end min_value_correct_l301_301169


namespace range_j_l301_301204

def h (x : ℝ) : ℝ := 4 * x - 3
def j (x : ℝ) : ℝ := h (h (h x))

theorem range_j : ∀ x, 0 ≤ x ∧ x ≤ 3 → -63 ≤ j x ∧ j x ≤ 129 :=
by
  intro x
  intro hx
  sorry

end range_j_l301_301204


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301769

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301769


namespace coin_difference_l301_301350

theorem coin_difference : ∀ (p : ℕ), 1 ≤ p ∧ p ≤ 999 → (10000 - 9 * 1) - (10000 - 9 * 999) = 8982 :=
by
  intro p
  intro hp
  sorry

end coin_difference_l301_301350


namespace intersection_M_N_l301_301166

def M : Set ℝ := {y | ∃ x : ℝ, y = x - |x|}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {0} :=
  sorry

end intersection_M_N_l301_301166


namespace eight_digit_product_1400_l301_301627

def eight_digit_numbers_count : Nat :=
  sorry

theorem eight_digit_product_1400 : eight_digit_numbers_count = 5880 :=
  sorry

end eight_digit_product_1400_l301_301627


namespace intersection_of_M_and_N_l301_301689

-- Defining our sets M and N based on the conditions provided
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x^2 < 4 }

-- The statement we want to prove
theorem intersection_of_M_and_N :
  M ∩ N = { x | -2 < x ∧ x < 1 } :=
sorry

end intersection_of_M_and_N_l301_301689


namespace minimum_expenses_for_Nikifor_to_win_maximum_F_value_l301_301594

noncomputable def number_of_voters := 35
noncomputable def sellable_voters := 14 -- 40% of 35
noncomputable def preference_voters := 21 -- 60% of 35
noncomputable def minimum_votes_to_win := 18 -- 50% of 35 + 1
noncomputable def cost_per_vote := 9

def vote_supply_function (P : ℕ) : ℕ :=
  if P = 0 then 10
  else if 1 ≤ P ∧ P ≤ 14 then 10 + P
  else 24


theorem minimum_expenses_for_Nikifor_to_win :
  ∃ P : ℕ, P * cost_per_vote = 162 ∧ vote_supply_function P ≥ minimum_votes_to_win := 
sorry

theorem maximum_F_value (F : ℕ) : 
  F = 3 :=
sorry

end minimum_expenses_for_Nikifor_to_win_maximum_F_value_l301_301594


namespace johns_watermelon_weight_l301_301696

-- Michael's largest watermelon weighs 8 pounds
def michael_weight : ℕ := 8

-- Clay's watermelon weighs three times the size of Michael's watermelon
def clay_weight : ℕ := 3 * michael_weight

-- John's watermelon weighs half the size of Clay's watermelon
def john_weight : ℕ := clay_weight / 2

-- Prove that John's watermelon weighs 12 pounds
theorem johns_watermelon_weight : john_weight = 12 := by
  sorry

end johns_watermelon_weight_l301_301696


namespace probability_at_least_half_correct_l301_301708

/--
Steve guesses randomly on a 20-question multiple choice test where each question has three choices
(one correct and two incorrect). Prove that the probability that he gets at least half of the
questions correct is 1/2.
-/
theorem probability_at_least_half_correct :
  ∑ k in finset.range (21) \ finset.range (10), (nat.choose 20 k) * (1/3:NNReal)^k * (2/3:NNReal)^(20 - k) = 1/2 := by
  sorry

end probability_at_least_half_correct_l301_301708


namespace number_of_truthful_dwarfs_l301_301451

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l301_301451


namespace geom_seq_sum_eqn_l301_301725

theorem geom_seq_sum_eqn (n : ℕ) (a : ℚ) (r : ℚ) (S_n : ℚ) : 
  a = 1/3 → r = 1/3 → S_n = 80/243 → S_n = a * (1 - r^n) / (1 - r) → n = 5 :=
by
  intros ha hr hSn hSum
  sorry

end geom_seq_sum_eqn_l301_301725


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301739

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301739


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301811

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301811


namespace sum_of_tens_l301_301938

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l301_301938


namespace no_rational_xyz_satisfies_l301_301153

theorem no_rational_xyz_satisfies:
  ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
  (1 / (x - y) ^ 2 + 1 / (y - z) ^ 2 + 1 / (z - x) ^ 2 = 2014) :=
by
  -- The proof will go here
  sorry

end no_rational_xyz_satisfies_l301_301153


namespace hyperbola_asymptotes_angle_l301_301631

theorem hyperbola_asymptotes_angle {a b : ℝ} (h₁ : a > b) 
  (h₂ : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h₃ : ∀ θ : ℝ, θ = Real.pi / 4) : a / b = Real.sqrt 2 :=
by
  sorry

end hyperbola_asymptotes_angle_l301_301631


namespace McKenna_stuffed_animals_count_l301_301534

def stuffed_animals (M K T : ℕ) : Prop :=
  M + K + T = 175 ∧ K = 2 * M ∧ T = K + 5

theorem McKenna_stuffed_animals_count (M K T : ℕ) (h : stuffed_animals M K T) : M = 34 :=
by
  sorry

end McKenna_stuffed_animals_count_l301_301534


namespace vec_eq_l301_301501

def a : ℝ × ℝ := (-1, 0)
def b : ℝ × ℝ := (0, 2)

theorem vec_eq : (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2) = (-2, -6) := by
  sorry

end vec_eq_l301_301501


namespace extended_pattern_ratio_l301_301023

def original_black_tiles : ℕ := 13
def original_white_tiles : ℕ := 12
def original_total_tiles : ℕ := 5 * 5

def new_side_length : ℕ := 7
def new_total_tiles : ℕ := new_side_length * new_side_length
def added_white_tiles : ℕ := new_total_tiles - original_total_tiles

def new_black_tiles : ℕ := original_black_tiles
def new_white_tiles : ℕ := original_white_tiles + added_white_tiles

def ratio_black_to_white : ℚ := new_black_tiles / new_white_tiles

theorem extended_pattern_ratio :
  ratio_black_to_white = 13 / 36 :=
by
  sorry

end extended_pattern_ratio_l301_301023


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301802

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301802


namespace increase_fraction_l301_301902

theorem increase_fraction (A F : ℝ) 
  (h₁ : A = 83200) 
  (h₂ : A * (1 + F) ^ 2 = 105300) : 
  F = 0.125 :=
by
  sorry

end increase_fraction_l301_301902


namespace apple_harvest_l301_301232

theorem apple_harvest (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 →
  num_sections = 8 →
  total_sacks = sacks_per_section * num_sections →
  total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end apple_harvest_l301_301232


namespace martin_goldfish_count_l301_301207

-- Define the initial number of goldfish
def initial_goldfish := 18

-- Define the number of goldfish that die each week
def goldfish_die_per_week := 5

-- Define the number of goldfish purchased each week
def goldfish_purchased_per_week := 3

-- Define the number of weeks
def weeks := 7

-- Calculate the expected number of goldfish after 7 weeks
noncomputable def final_goldfish := initial_goldfish - (goldfish_die_per_week * weeks) + (goldfish_purchased_per_week * weeks)

-- State the theorem and the proof target
theorem martin_goldfish_count : final_goldfish = 4 := 
sorry

end martin_goldfish_count_l301_301207


namespace rowing_probability_l301_301275

open ProbabilityTheory

theorem rowing_probability
  (P_left_works : ℚ := 3 / 5)
  (P_right_works : ℚ := 3 / 5) :
  let P_left_breaks := 1 - P_left_works
  let P_right_breaks := 1 - P_right_works
  let P_left_works_and_right_works := P_left_works * P_right_works
  let P_left_works_and_right_breaks := P_left_works * P_right_breaks
  let P_left_breaks_and_right_works := P_left_breaks * P_right_works
  P_left_works_and_right_works + P_left_works_and_right_breaks + P_left_breaks_and_right_works = 21 / 25 := by
  sorry

end rowing_probability_l301_301275


namespace no_food_dogs_l301_301671

theorem no_food_dogs (total_dogs watermelon_liking salmon_liking chicken_liking ws_liking sc_liking wc_liking wsp_liking : ℕ) 
    (h_total : total_dogs = 100)
    (h_watermelon : watermelon_liking = 20) 
    (h_salmon : salmon_liking = 70) 
    (h_chicken : chicken_liking = 10) 
    (h_ws : ws_liking = 10) 
    (h_sc : sc_liking = 5) 
    (h_wc : wc_liking = 3) 
    (h_wsp : wsp_liking = 2) :
    (total_dogs - ((watermelon_liking - ws_liking - wc_liking + wsp_liking) + 
    (salmon_liking - ws_liking - sc_liking + wsp_liking) + 
    (chicken_liking - sc_liking - wc_liking + wsp_liking) + 
    (ws_liking - wsp_liking) + 
    (sc_liking - wsp_liking) + 
    (wc_liking - wsp_liking) + wsp_liking)) = 28 :=
  by sorry

end no_food_dogs_l301_301671


namespace coordinate_plane_line_l301_301401

theorem coordinate_plane_line (m n p : ℝ) (h1 : m = n / 5 - 2 / 5) (h2 : m + p = (n + 15) / 5 - 2 / 5) : p = 3 := by
  sorry

end coordinate_plane_line_l301_301401


namespace belts_count_l301_301150

-- Definitions based on conditions
variable (shoes belts hats : ℕ)

-- Conditions from the problem
axiom shoes_eq_14 : shoes = 14
axiom hat_count : hats = 5
axiom shoes_double_of_belts : shoes = 2 * belts

-- Definition of the theorem to prove the number of belts
theorem belts_count : belts = 7 :=
by
  sorry

end belts_count_l301_301150


namespace remainder_first_six_primes_div_seventh_l301_301848

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301848


namespace correct_option_is_c_l301_301417

variable {x y : ℕ}

theorem correct_option_is_c (hx : (x^2)^3 = x^6) :
  (∀ x : ℕ, x * x^2 ≠ x^2) →
  (∀ x y : ℕ, (x + y)^2 ≠ x^2 + y^2) →
  (∃ x : ℕ, x^2 + x^2 ≠ x^4) →
  (x^2)^3 = x^6 :=
by
  intros h1 h2 h3
  exact hx

end correct_option_is_c_l301_301417


namespace PQ_sum_l301_301930

theorem PQ_sum (P Q : ℕ) (h1 : 5 / 7 = P / 63) (h2 : 5 / 7 = 70 / Q) : P + Q = 143 :=
by
  sorry

end PQ_sum_l301_301930


namespace scientific_notation_of_634000000_l301_301886

theorem scientific_notation_of_634000000 :
  634000000 = 6.34 * 10 ^ 8 := 
sorry

end scientific_notation_of_634000000_l301_301886


namespace area_ratio_trapezoid_abm_abcd_l301_301344

-- Definitions based on conditions
variables {A B C D M : Type} [Zero A] [Zero B] [Zero C] [Zero D] [Zero M]
variables (BC AD : ℝ)

-- Condition: ABCD is a trapezoid with BC parallel to AD and diagonals AC and BD intersect M
-- Given BC = b and AD = a

-- Theorem statement
theorem area_ratio_trapezoid_abm_abcd (a b : ℝ) (h1 : BC = b) (h2 : AD = a) : 
  ∃ S_ABM S_ABCD : ℝ,
  (S_ABM / S_ABCD = a * b / (a + b)^2) :=
sorry

end area_ratio_trapezoid_abm_abcd_l301_301344


namespace total_votes_cast_l301_301013

-- Problem statement and conditions
variable (V : ℝ) (candidateVotes : ℝ) (rivalVotes : ℝ)
variable (h1 : candidateVotes = 0.35 * V)
variable (h2 : rivalVotes = candidateVotes + 1350)

-- Target to prove
theorem total_votes_cast : V = 4500 := by
  -- pseudo code proof would be filled here in real Lean environment
  sorry

end total_votes_cast_l301_301013


namespace pens_in_shop_l301_301379

theorem pens_in_shop (P Pe E : ℕ) (h_ratio : 14 * Pe = 4 * P) (h_ratio2 : 14 * E = 14 * 3 + 11) (h_P : P = 140) (h_E : E = 30) : Pe = 40 :=
sorry

end pens_in_shop_l301_301379


namespace yellow_balls_count_l301_301603

theorem yellow_balls_count (total_balls white_balls green_balls red_balls purple_balls : ℕ)
  (h_total : total_balls = 100)
  (h_white : white_balls = 20)
  (h_green : green_balls = 30)
  (h_red : red_balls = 37)
  (h_purple : purple_balls = 3)
  (h_prob : ((white_balls + green_balls + (total_balls - white_balls - green_balls - red_balls - purple_balls)) / total_balls : ℝ) = 0.6) :
  (total_balls - white_balls - green_balls - red_balls - purple_balls = 10) :=
by {
  sorry
}

end yellow_balls_count_l301_301603


namespace c_ge_a_plus_b_sin_half_C_l301_301880

-- Define a triangle with sides a, b, and c opposite to angles A, B, and C respectively, with C being the angle at vertex C
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a)
  (angles_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  (angles_sum : A + B + C = π)

namespace TriangleProveInequality

open Triangle

theorem c_ge_a_plus_b_sin_half_C (t : Triangle) :
  t.c ≥ (t.a + t.b) * Real.sin (t.C / 2) := sorry

end TriangleProveInequality

end c_ge_a_plus_b_sin_half_C_l301_301880


namespace books_returned_percentage_l301_301871

theorem books_returned_percentage :
  ∃ (L R : ℕ) (p : ℚ),
  L = 40 ∧
  R = 28 ∧
  p = (R / L) * 100 ∧
  p = 70 :=
by
  let L := 40
  let R := 28
  let p := (R / L) * 100
  have L_def : L = 40 := by rfl
  have R_def : R = 28 := by rfl
  have p_def : p = 70 := by
    calc
      p = (R / L) * 100 : by rfl
      ... = (28 / 40) * 100 : by rw [R_def, L_def]
      ... = 0.7 * 100 : by norm_num
      ... = 70 : by norm_num
  exact ⟨L, R, p, L_def, R_def, p_def, rfl⟩

end books_returned_percentage_l301_301871


namespace John_has_22_quarters_l301_301681

variable (q d n : ℕ)

-- Conditions
axiom cond1 : d = q + 3
axiom cond2 : n = q - 6
axiom cond3 : q + d + n = 63

theorem John_has_22_quarters : q = 22 := by
  sorry

end John_has_22_quarters_l301_301681


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301800

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301800


namespace all_roots_are_nth_roots_of_unity_l301_301110

noncomputable def smallest_positive_integer_n : ℕ :=
  5
  
theorem all_roots_are_nth_roots_of_unity :
  (∀ z : ℂ, (z^4 + z^3 + z^2 + z + 1 = 0) → z^(smallest_positive_integer_n) = 1) :=
  by
    sorry

end all_roots_are_nth_roots_of_unity_l301_301110


namespace cricket_target_l301_301521

theorem cricket_target (run_rate_first_10overs run_rate_next_40overs : ℝ) (overs_first_10 next_40_overs : ℕ)
    (h_first : run_rate_first_10overs = 3.2) 
    (h_next : run_rate_next_40overs = 6.25) 
    (h_overs_first : overs_first_10 = 10) 
    (h_overs_next : next_40_overs = 40) 
    : (overs_first_10 * run_rate_first_10overs + next_40_overs * run_rate_next_40overs) = 282 :=
by
  sorry

end cricket_target_l301_301521


namespace Tim_pays_correct_amount_l301_301390

def pays_in_a_week (hourly_rate : ℕ) (num_bodyguards : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * num_bodyguards * hours_per_day * days_per_week

theorem Tim_pays_correct_amount :
  pays_in_a_week 20 2 8 7 = 2240 := by
  sorry

end Tim_pays_correct_amount_l301_301390


namespace min_board_size_l301_301707

theorem min_board_size (n : ℕ) (total_area : ℕ) (domino_area : ℕ) 
  (h1 : total_area = 2008) 
  (h2 : domino_area = 2) 
  (h3 : ∀ domino_count : ℕ, domino_count = total_area / domino_area → (∃ m : ℕ, (m+1) * (m+1) ≥ domino_count * (2 + 4) → n = m)) :
  n = 77 :=
by
  sorry

end min_board_size_l301_301707


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301760

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301760


namespace remainder_of_sum_of_primes_l301_301822

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301822


namespace initial_sum_invested_l301_301604

-- Definition of the initial problem setup and given conditions
def compound_interest (P r n : ℝ) : ℝ :=
  P * (1 + r) ^ n

def interest_difference (P r1 r2 n : ℝ) : ℝ :=
  (compound_interest P r1 n) - (compound_interest P r2 n)

theorem initial_sum_invested (P : ℝ) :
  let r1 := 0.15
  let r2 := 0.10
  let n := 3
  interest_difference P r1 r2 n = 1500 →
  P ≈ 7899.47 :=
by
  sorry

end initial_sum_invested_l301_301604


namespace middle_aged_participating_l301_301278

-- Definitions of the given conditions
def total_employees : Nat := 1200
def ratio (elderly middle_aged young : Nat) := elderly = 1 ∧ middle_aged = 5 ∧ young = 6
def selected_employees : Nat := 36

-- The stratified sampling condition implies
def stratified_sampling (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) :=
  (elderly + middle_aged + young = total) ∧
  (selected = 36)

-- The proof statement
theorem middle_aged_participating (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) 
  (h_ratio : ratio elderly middle_aged young) 
  (h_total : total = total_employees)
  (h_sampled : stratified_sampling elderly middle_aged young (elderly + middle_aged + young) selected) : 
  selected * middle_aged / (elderly + middle_aged + young) = 15 := 
by sorry

end middle_aged_participating_l301_301278


namespace largest_b_value_l301_301246

/- Definitions -/
def conditions (a b c : ℕ) : Prop :=
  1 < c ∧ c < b ∧ b < a ∧ a * b * c = 360

noncomputable def largest_b : ℕ :=
  Nat.find_max' (λ b, ∃ a c, conditions a b c) sorry

/- Theorem -/
theorem largest_b_value : largest_b = 12 :=
sorry

end largest_b_value_l301_301246


namespace rhombus_division_ratio_l301_301672

-- Defining the conditions
variable (α : ℝ)

-- Statement of the problem in Lean 4
theorem rhombus_division_ratio (hα_pos : 0 < α) (hα_lt_pi2 : α < π/2) :
  ∃ r : ℝ, r = (cos (α / 6)) / (cos (α / 2)) :=
begin
  sorry
end

end rhombus_division_ratio_l301_301672


namespace find_range_of_k_l301_301494

-- Define the conditions and the theorem
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

theorem find_range_of_k :
  {k : ℝ | is_ellipse k} = {k : ℝ | (-3 < k ∧ k < -1/2) ∨ (-1/2 < k ∧ k < 2)} :=
by
  sorry

end find_range_of_k_l301_301494


namespace gcd_282_470_l301_301250

theorem gcd_282_470 : Int.gcd 282 470 = 94 := by
  sorry

end gcd_282_470_l301_301250


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301753

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301753


namespace number_of_truthful_dwarfs_l301_301454

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l301_301454


namespace dwarfs_truthful_count_l301_301443

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l301_301443


namespace exists_obtuse_triangle_l301_301064

section
variables {a b c : ℝ} (ABC : Triangle ℝ) (h₁ : ABC.side1 = a)
  (h₂ : ABC.side2 = a + 1) (h₃ : ABC.side3 = a + 2)
  (h₄ : 2 * sin ABC.γ = 3 * sin ABC.α)

noncomputable def area_triangle_proof : ABC.area = (15 * real.sqrt 7) / 4 :=
sorry

theorem exists_obtuse_triangle : ∃ (a : ℝ), a = 2 ∧ ∀ (h : Triangle ℝ), 
  h.side1 = a → h.side2 = a + 1 → h.side3 = a + 2 → obtuse h :=
sorry
end

end exists_obtuse_triangle_l301_301064


namespace sum_of_three_numbers_l301_301112

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 42) (h3 : c + a = 58) :
  a + b + c = 67.5 :=
by
  sorry

end sum_of_three_numbers_l301_301112


namespace set_complement_union_eq_l301_301992

open Set

variable (U : Set ℕ) (P : Set ℕ) (Q : Set ℕ)

theorem set_complement_union_eq :
  U = {1, 2, 3, 4, 5, 6} →
  P = {1, 3, 5} →
  Q = {1, 2, 4} →
  (U \ P) ∪ Q = {1, 2, 4, 6} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end set_complement_union_eq_l301_301992


namespace inequality_holds_for_positive_y_l301_301873

theorem inequality_holds_for_positive_y (y : ℝ) (hy : y > 0) : y^2 ≥ 2 * y - 1 :=
by
  sorry

end inequality_holds_for_positive_y_l301_301873


namespace max_boxes_in_warehouse_l301_301911

def warehouse_length : ℕ := 50
def warehouse_width : ℕ := 30
def warehouse_height : ℕ := 5
def box_edge_length : ℕ := 2

theorem max_boxes_in_warehouse : (warehouse_length / box_edge_length) * (warehouse_width / box_edge_length) * (warehouse_height / box_edge_length) = 750 := 
by
  sorry

end max_boxes_in_warehouse_l301_301911


namespace positive_number_decreased_by_4_is_21_times_reciprocal_l301_301874

theorem positive_number_decreased_by_4_is_21_times_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x - 4 = 21 * (1 / x)) : x = 7 := 
sorry

end positive_number_decreased_by_4_is_21_times_reciprocal_l301_301874


namespace solve_for_x_l301_301363

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
sorry

end solve_for_x_l301_301363


namespace prime_sum_remainder_l301_301774

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301774


namespace sum_first_n_natural_numbers_l301_301560

theorem sum_first_n_natural_numbers (n : ℕ) (h : (n * (n + 1)) / 2 = 1035) : n = 46 :=
sorry

end sum_first_n_natural_numbers_l301_301560


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301734

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301734


namespace jamie_dimes_l301_301345

theorem jamie_dimes (y : ℕ) (h : 5 * y + 10 * y + 25 * y = 1440) : y = 36 :=
by 
  sorry

end jamie_dimes_l301_301345


namespace new_average_after_exclusion_l301_301551

theorem new_average_after_exclusion (S : ℕ) (h1 : S = 27 * 5) (excluded : ℕ) (h2 : excluded = 35) : (S - excluded) / 4 = 25 :=
by
  sorry

end new_average_after_exclusion_l301_301551


namespace packs_of_yellow_bouncy_balls_l301_301993

/-- Maggie bought 4 packs of red bouncy balls, some packs of yellow bouncy balls (denoted as Y), and 4 packs of green bouncy balls. -/
theorem packs_of_yellow_bouncy_balls (Y : ℕ) : 
  (4 + Y + 4) * 10 = 160 -> Y = 8 := 
by 
  sorry

end packs_of_yellow_bouncy_balls_l301_301993


namespace mod_sum_l301_301310

theorem mod_sum : 
  (5432 + 5433 + 5434 + 5435) % 7 = 2 := 
by
  sorry

end mod_sum_l301_301310


namespace Robe_savings_l301_301084

-- Define the conditions and question in Lean 4
theorem Robe_savings 
  (repair_fee : ℕ)
  (corner_light_cost : ℕ)
  (brake_disk_cost : ℕ)
  (total_remaining_savings : ℕ)
  (total_savings_before : ℕ)
  (h1 : repair_fee = 10)
  (h2 : corner_light_cost = 2 * repair_fee)
  (h3 : brake_disk_cost = 3 * corner_light_cost)
  (h4 : total_remaining_savings = 480)
  (h5 : total_savings_before = total_remaining_savings + (repair_fee + corner_light_cost + 2 * brake_disk_cost)) :
  total_savings_before = 630 :=
by
  -- Proof steps to be filled
  sorry

end Robe_savings_l301_301084


namespace original_avg_expenditure_correct_l301_301564

variables (A B C a b c X Y Z : ℝ)
variables (hA : A > 0) (hB : B > 0) (hC : C > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem original_avg_expenditure_correct
    (h_orig_exp : (A * X + B * Y + C * Z) / (A + B + C) - 1 
    = ((A + a) * X + (B + b) * Y + (C + c) * Z + 42) / 42):
    True := 
sorry

end original_avg_expenditure_correct_l301_301564


namespace jessica_current_age_l301_301979

theorem jessica_current_age : 
  ∃ J M_d M_c : ℕ, 
    J = (M_d / 2) ∧ 
    M_d = M_c - 10 ∧ 
    M_c = 70 ∧ 
    J + 10 = 40 := 
sorry

end jessica_current_age_l301_301979


namespace unclaimed_books_fraction_l301_301881

noncomputable def fraction_unclaimed (total_books : ℝ) : ℝ :=
  let al_share := (2 / 5) * total_books
  let bert_share := (9 / 50) * total_books
  let carl_share := (21 / 250) * total_books
  let dan_share := (189 / 2500) * total_books
  total_books - (al_share + bert_share + carl_share + dan_share)

theorem unclaimed_books_fraction (total_books : ℝ) : fraction_unclaimed total_books / total_books = 1701 / 2500 := 
begin
  sorry
end

end unclaimed_books_fraction_l301_301881


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301772

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301772


namespace differences_multiple_of_nine_l301_301896

theorem differences_multiple_of_nine (S : Finset ℕ) (hS : S.card = 10) (h_unique : ∀ {x y : ℕ}, x ∈ S → y ∈ S → x ≠ y → x ≠ y) : 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b) % 9 = 0 :=
by
  sorry

end differences_multiple_of_nine_l301_301896


namespace total_guitars_sold_l301_301966

theorem total_guitars_sold (total_revenue : ℕ) (price_electric : ℕ) (price_acoustic : ℕ)
  (num_electric_sold : ℕ) (num_acoustic_sold : ℕ) 
  (h1 : total_revenue = 3611) (h2 : price_electric = 479) 
  (h3 : price_acoustic = 339) (h4 : num_electric_sold = 4) 
  (h5 : num_acoustic_sold * price_acoustic + num_electric_sold * price_electric = total_revenue) :
  num_electric_sold + num_acoustic_sold = 9 :=
sorry

end total_guitars_sold_l301_301966


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301740

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301740


namespace find_b_perpendicular_l301_301555

theorem find_b_perpendicular (b : ℝ) : (∀ x y : ℝ, 4 * y - 2 * x = 6 → 5 * y + b * x - 2 = 0 → (1 / 2 : ℝ) * (-(b / 5) : ℝ) = -1) → b = 10 :=
by
  intro h
  sorry

end find_b_perpendicular_l301_301555


namespace find_all_real_solutions_l301_301025

theorem find_all_real_solutions (x : ℝ) :
    (1 / ((x - 1) * (x - 2))) + (1 / ((x - 2) * (x - 3))) + (1 / ((x - 3) * (x - 4))) + (1 / ((x - 4) * (x - 5))) = 1 / 4 →
    x = 1 ∨ x = 5 :=
by
  sorry

end find_all_real_solutions_l301_301025


namespace exhibit_special_13_digit_integer_l301_301624

open Nat 

def thirteenDigitInteger (N : ℕ) : Prop :=
  N ≥ 10^12 ∧ N < 10^13

def isMultipleOf8192 (N : ℕ) : Prop :=
  8192 ∣ N

def hasOnlyEightOrNineDigits (N : ℕ) : Prop :=
  ∀ d ∈ digits 10 N, d = 8 ∨ d = 9

theorem exhibit_special_13_digit_integer : 
  ∃ N : ℕ, thirteenDigitInteger N ∧ isMultipleOf8192 N ∧ hasOnlyEightOrNineDigits N ∧ N = 8888888888888 := 
by
  sorry 

end exhibit_special_13_digit_integer_l301_301624


namespace trailing_zeros_310_factorial_l301_301557

def count_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem trailing_zeros_310_factorial :
  count_trailing_zeros 310 = 76 := by
sorry

end trailing_zeros_310_factorial_l301_301557


namespace John_has_22_quarters_l301_301680

variable (q d n : ℕ)

-- Conditions
axiom cond1 : d = q + 3
axiom cond2 : n = q - 6
axiom cond3 : q + d + n = 63

theorem John_has_22_quarters : q = 22 := by
  sorry

end John_has_22_quarters_l301_301680


namespace necessary_but_not_sufficient_condition_l301_301038

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x < 5) : (x < 2 → x < 5) ∧ ¬(x < 5 → x < 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l301_301038


namespace megan_broke_3_eggs_l301_301995

variables (total_eggs B C P : ℕ)

theorem megan_broke_3_eggs (h1 : total_eggs = 24) (h2 : C = 2 * B) (h3 : P = 24 - (B + C)) (h4 : P - C = 9) : B = 3 := by
  sorry

end megan_broke_3_eggs_l301_301995


namespace Tim_bodyguards_weekly_pay_l301_301387

theorem Tim_bodyguards_weekly_pay :
  let hourly_rate := 20
  let num_bodyguards := 2
  let daily_hours := 8
  let weekly_days := 7
  Tim pays $2240 in a week := (hourly_rate * num_bodyguards * daily_hours * weekly_days = 2240) :=
begin
  sorry
end

end Tim_bodyguards_weekly_pay_l301_301387


namespace cottonwood_fiber_scientific_notation_l301_301467

theorem cottonwood_fiber_scientific_notation :
  0.0000108 = 1.08 * 10^(-5)
:= by
  sorry

end cottonwood_fiber_scientific_notation_l301_301467


namespace tangent_line_to_circle_l301_301237

theorem tangent_line_to_circle (c : ℝ) (h : 0 < c) : 
  (∃ (x y : ℝ), x^2 + y^2 = 8 ∧ x + y = c) ↔ c = 4 :=
by sorry

end tangent_line_to_circle_l301_301237


namespace tim_weekly_payment_l301_301389

-- Define the given conditions
def hourly_rate_bodyguard : ℕ := 20
def number_bodyguards : ℕ := 2
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7

-- Define the total weekly payment calculation
def weekly_payment : ℕ := (hourly_rate_bodyguard * number_bodyguards) * hours_per_day * days_per_week

-- The proof statement
theorem tim_weekly_payment : weekly_payment = 2240 := by
  sorry

end tim_weekly_payment_l301_301389


namespace sam_found_pennies_l301_301541

-- Define the function that computes the number of pennies Sam found given the initial and current amounts of pennies
def find_pennies (initial_pennies current_pennies : Nat) : Nat :=
  current_pennies - initial_pennies

-- Define the main proof problem
theorem sam_found_pennies : find_pennies 98 191 = 93 := by
  -- Proof steps would go here
  sorry

end sam_found_pennies_l301_301541


namespace ferris_wheel_cost_per_child_l301_301281

namespace AmusementPark

def num_children := 5
def daring_children := 3
def merry_go_round_cost_per_child := 3
def ice_cream_cones_per_child := 2
def ice_cream_cost_per_cone := 8
def total_spent := 110

theorem ferris_wheel_cost_per_child (F : ℝ) :
  (daring_children * F + num_children * merry_go_round_cost_per_child +
   num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone = total_spent) →
  F = 5 :=
by
  -- Here we would proceed with the proof steps, but adding sorry to skip it.
  sorry

end AmusementPark

end ferris_wheel_cost_per_child_l301_301281


namespace complex_div_symmetry_l301_301333

open Complex

-- Definitions based on conditions
def z1 : ℂ := 1 + I
def z2 : ℂ := -1 + I

-- Theorem to prove
theorem complex_div_symmetry : z2 / z1 = I := by
  sorry

end complex_div_symmetry_l301_301333


namespace solve_for_x_l301_301364

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
sorry

end solve_for_x_l301_301364


namespace number_of_truthful_dwarfs_is_correct_l301_301456

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l301_301456


namespace discount_percentage_l301_301723

theorem discount_percentage (x : ℝ) : 
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  (marked_price * (1 - x / 100) * (1 - second_discount) * (1 - third_discount) = final_price) ↔ x = 20 := 
by
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  sorry

end discount_percentage_l301_301723


namespace two_f_x_eq_8_over_4_plus_x_l301_301045

variable (f : ℝ → ℝ)
variable (x : ℝ)
variables (hx : 0 < x)
variable (h : ∀ x, 0 < x → f (2 * x) = 2 / (2 + x))

theorem two_f_x_eq_8_over_4_plus_x : 2 * f x = 8 / (4 + x) :=
by sorry

end two_f_x_eq_8_over_4_plus_x_l301_301045


namespace total_bill_l301_301372

variable (B : ℝ)
variable (h1 : 9 * (B / 10 + 3) = B)

theorem total_bill : B = 270 :=
by
  -- proof would go here
  sorry

end total_bill_l301_301372


namespace calculate_expression_l301_301851

def x : Float := 3.241
def y : Float := 14
def z : Float := 100
def expected_result : Float := 0.45374

theorem calculate_expression : (x * y) / z = expected_result := by
  sorry

end calculate_expression_l301_301851


namespace find_m_l301_301236

def f (x m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

theorem find_m :
  let m := 10 / 7
  3 * f 5 m = 2 * g 5 m :=
by
  sorry

end find_m_l301_301236


namespace apothem_comparison_l301_301607

noncomputable def pentagon_side_length : ℝ := 4 / Real.tan (54 * Real.pi / 180)

noncomputable def pentagon_apothem : ℝ := pentagon_side_length / (2 * Real.tan (54 * Real.pi / 180))

noncomputable def hexagon_side_length : ℝ := 4 / Real.sqrt 3

noncomputable def hexagon_apothem : ℝ := (Real.sqrt 3 / 2) * hexagon_side_length

theorem apothem_comparison : pentagon_apothem = 1.06 * hexagon_apothem :=
by
  sorry

end apothem_comparison_l301_301607


namespace journey_speed_second_half_l301_301410

theorem journey_speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (v : ℝ) : 
  total_time = 10 ∧ first_half_speed = 21 ∧ total_distance = 224 →
  v = 24 :=
by
  intro h
  sorry

end journey_speed_second_half_l301_301410


namespace tan_15_simplification_l301_301217

theorem tan_15_simplification :
  (1 + Real.tan (Real.pi / 12)) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end tan_15_simplification_l301_301217


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301813

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301813


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301829

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301829


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301798

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301798


namespace TotalGenuineItems_l301_301567

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem TotalGenuineItems : GenuinePurses + GenuineHandbags = 31 :=
  by
    -- proof
    sorry

end TotalGenuineItems_l301_301567


namespace bottles_stolen_at_dance_l301_301622

-- Define the initial conditions
def initial_bottles := 10
def bottles_lost_at_school := 2
def total_stickers := 21
def stickers_per_bottle := 3

-- Calculate remaining bottles after loss at school
def remaining_bottles_after_school := initial_bottles - bottles_lost_at_school

-- Calculate the remaining bottles after the theft
def remaining_bottles_after_theft := total_stickers / stickers_per_bottle

-- Prove the number of bottles stolen
theorem bottles_stolen_at_dance : remaining_bottles_after_school - remaining_bottles_after_theft = 1 :=
by
  sorry

end bottles_stolen_at_dance_l301_301622


namespace num_pos_pairs_l301_301927

theorem num_pos_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 + 3 * n < 40) :
  ∃ k : ℕ, k = 45 :=
by {
  -- Additional setup and configuration if needed
  -- ...
  sorry
}

end num_pos_pairs_l301_301927


namespace floor_abs_neg_45_7_l301_301463

theorem floor_abs_neg_45_7 : (Int.floor (Real.abs (-45.7))) = 45 :=
by
  sorry

end floor_abs_neg_45_7_l301_301463


namespace coprime_condition_exists_l301_301728

theorem coprime_condition_exists : ∃ (A B C : ℕ), (A > 0 ∧ B > 0 ∧ C > 0) ∧ (Nat.gcd (Nat.gcd A B) C = 1) ∧ 
  (A * Real.log 5 / Real.log 50 + B * Real.log 2 / Real.log 50 = C) ∧ (A + B + C = 4) :=
by {
  sorry
}

end coprime_condition_exists_l301_301728


namespace paul_can_buy_toys_l301_301079

-- Definitions of the given conditions
def initial_dollars : ℕ := 3
def allowance : ℕ := 7
def toy_cost : ℕ := 5

-- Required proof statement
theorem paul_can_buy_toys : (initial_dollars + allowance) / toy_cost = 2 := by
  sorry

end paul_can_buy_toys_l301_301079


namespace number_of_diagonals_intersections_l301_301670

theorem number_of_diagonals_intersections (n : ℕ) (h : n ≥ 4) : 
  (∃ (I : ℕ), I = (n * (n - 1) * (n - 2) * (n - 3)) / 24) :=
by {
  sorry
}

end number_of_diagonals_intersections_l301_301670


namespace sum_area_triangles_lt_total_area_l301_301860

noncomputable def G : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def A_k (k : ℕ+) : ℝ := sorry -- Assume we've defined A_k's expression correctly
noncomputable def S (S1 S2 : ℝ) : ℝ := 2 * S1 - S2

theorem sum_area_triangles_lt_total_area (k : ℕ+) (S1 S2 : ℝ) :
  (A_k k < S S1 S2) :=
sorry

end sum_area_triangles_lt_total_area_l301_301860


namespace second_tray_holds_l301_301998

-- The conditions and the given constants
variables (x : ℕ) (h1 : 2 * x - 20 = 500)

-- The theorem proving the number of cups the second tray holds is 240 
theorem second_tray_holds (h2 : x = 260) : x - 20 = 240 := by
  sorry

end second_tray_holds_l301_301998


namespace jessica_current_age_l301_301978

-- Definitions and conditions from the problem
def J (M : ℕ) : ℕ := M / 2
def M : ℕ := 60

-- Lean statement for the proof problem
theorem jessica_current_age : J M + 10 = 40 :=
by
  sorry

end jessica_current_age_l301_301978


namespace sum_of_roots_l301_301506

theorem sum_of_roots (p q : ℝ) (h_eq : 2 * p + 3 * q = 6) (h_roots : ∀ x : ℝ, x ^ 2 - p * x + q = 0) : p = 2 := by
sorry

end sum_of_roots_l301_301506


namespace min_positive_period_f_max_value_f_decreasing_intervals_g_l301_301499

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem min_positive_period_f : 
  ∃ (p : ℝ), p > 0 ∧ (∀ x : ℝ, f (x + 2*Real.pi) = f x) :=
sorry

theorem max_value_f : 
  ∃ (M : ℝ), (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) ∧ M = 2 :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (-x)

theorem decreasing_intervals_g :
  ∀ (k : ℤ), ∀ x : ℝ, (5 * Real.pi / 4 + 2 * ↑k * Real.pi ≤ x ∧ x ≤ 9 * Real.pi / 4 + 2 * ↑k * Real.pi) →
  ∀ (h : x ≤ Real.pi * 2 * (↑k+1)), g x ≥ g (x + Real.pi) :=
sorry

end min_positive_period_f_max_value_f_decreasing_intervals_g_l301_301499


namespace total_toucans_l301_301128

def initial_toucans : Nat := 2

def new_toucans : Nat := 1

theorem total_toucans : initial_toucans + new_toucans = 3 := by
  sorry

end total_toucans_l301_301128


namespace prime_sum_remainder_l301_301781

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301781


namespace MaximMethod_CorrectNumber_l301_301533

theorem MaximMethod_CorrectNumber (x y : ℕ) (N : ℕ) (h_digit_x : 0 ≤ x ∧ x ≤ 9) (h_digit_y : 1 ≤ y ∧ y ≤ 9)
  (h_N : N = 10 * x + y)
  (h_condition : 1 / (10 * x + y : ℚ) = 1 / (x + y : ℚ) - 1 / (x * y : ℚ)) :
  N = 24 :=
sorry

end MaximMethod_CorrectNumber_l301_301533


namespace base_circumference_cone_l301_301867

theorem base_circumference_cone (r : ℝ) (h : r = 5) (θ : ℝ) (k : θ = 180) : 
  ∃ c : ℝ, c = 5 * π :=
by
  sorry

end base_circumference_cone_l301_301867


namespace double_acute_angle_is_positive_and_less_than_180_l301_301163

variable (α : ℝ) (h : 0 < α ∧ α < π / 2)

theorem double_acute_angle_is_positive_and_less_than_180 :
  0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end double_acute_angle_is_positive_and_less_than_180_l301_301163


namespace point_in_second_quadrant_l301_301862

theorem point_in_second_quadrant (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so its x-coordinate is negative
  (h2 : 0 < P.2) -- Point P is in the second quadrant, so its y-coordinate is positive
  (h3 : |P.2| = 3) -- The distance from P to the x-axis is 3
  (h4 : |P.1| = 4) -- The distance from P to the y-axis is 4
  : P = (-4, 3) := 
  sorry

end point_in_second_quadrant_l301_301862


namespace sphere_surface_area_l301_301661

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : V = 36 * π) 
  (h2 : V = (4 / 3) * π * r^3) 
  (h3 : A = 4 * π * r^2) 
  : A = 36 * π :=
by
  sorry

end sphere_surface_area_l301_301661


namespace max_non_triangulated_segments_correct_l301_301143

open Classical

/-
Problem description:
Given an equilateral triangle divided into smaller equilateral triangles with side length 1, 
we need to define the maximum number of 1-unit segments that can be marked such that no 
triangular subregion has all its sides marked.
-/

def total_segments (n : ℕ) : ℕ :=
  (3 * n * (n + 1)) / 2

def max_non_triangular_segments (n : ℕ) : ℕ :=
  n * (n + 1)

theorem max_non_triangulated_segments_correct (n : ℕ) :
  max_non_triangular_segments n = n * (n + 1) := by sorry

end max_non_triangulated_segments_correct_l301_301143


namespace exists_x0_l301_301215

noncomputable def f (x : Real) (a : Real) : Real :=
  Real.exp x - a * Real.sin x

theorem exists_x0 (a : Real) (h : a = 1) :
  ∃ x0 ∈ Set.Ioo (-Real.pi / 2) 0, 1 < f x0 a ∧ f x0 a < Real.sqrt 2 :=
  sorry

end exists_x0_l301_301215


namespace nonnegative_poly_sum_of_squares_l301_301692

open Polynomial

theorem nonnegative_poly_sum_of_squares (P : Polynomial ℝ) 
    (hP : ∀ x : ℝ, 0 ≤ P.eval x) 
    : ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 := 
by
  sorry

end nonnegative_poly_sum_of_squares_l301_301692


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301807

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301807


namespace find_k_l301_301632

noncomputable def repeating_representation_base_k (k: ℕ) : Prop := 
  ((3 * k + 5 : ℚ) / (k^2 - 1)) = (11 / 85)

theorem find_k (k: ℕ) (hk : 1 < k) : repeating_representation_base_k k → k = 25 :=
by
  sorry

end find_k_l301_301632


namespace coplanar_values_l301_301701

namespace CoplanarLines

-- Define parametric equations of the lines
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (3 + 2 * t, 2 - t, 5 + m * t)
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (4 - m * u, 5 + 3 * u, 6 + 2 * u)

-- Define coplanarity condition
def coplanar_condition (m : ℝ) : Prop :=
  ∃ t u : ℝ, line1 t m = line2 u m

-- Theorem to prove the specific values of m for coplanarity
theorem coplanar_values (m : ℝ) : coplanar_condition m ↔ (m = -13/9 ∨ m = 1) :=
sorry

end CoplanarLines

end coplanar_values_l301_301701


namespace imaginary_unit_div_l301_301988

open Complex

theorem imaginary_unit_div (i : ℂ) (hi : i * i = -1) : (i / (1 + i) = (1 / 2) + (1 / 2) * i) :=
by
  sorry

end imaginary_unit_div_l301_301988


namespace normal_complaints_calculation_l301_301710

-- Define the normal number of complaints
def normal_complaints (C : ℕ) : ℕ := C

-- Define the complaints when short-staffed
def short_staffed_complaints (C : ℕ) : ℕ := (4 * C) / 3

-- Define the complaints when both conditions are met
def both_conditions_complaints (C : ℕ) : ℕ := (4 * C) / 3 + (4 * C) / 15

-- Main statement to prove
theorem normal_complaints_calculation (C : ℕ) (h : 3 * (both_conditions_complaints C) = 576) : C = 120 :=
by sorry

end normal_complaints_calculation_l301_301710


namespace simplification_evaluation_l301_301544

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 5 + 2) :
  (x + 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = Real.sqrt 5 / 5 :=
by
  sorry

end simplification_evaluation_l301_301544


namespace total_participants_l301_301101

theorem total_participants (F M : ℕ)
  (h1 : F / 2 = 130)
  (h2 : F / 2 + M / 4 = (F + M) / 3) : 
  F + M = 780 := 
by 
  sorry

end total_participants_l301_301101


namespace tank_capacity_l301_301130

-- Define the conditions given in the problem.
def tank_full_capacity (x : ℝ) : Prop :=
  (0.25 * x = 60) ∧ (0.15 * x = 36)

-- State the theorem that needs to be proved.
theorem tank_capacity : ∃ x : ℝ, tank_full_capacity x ∧ x = 240 := 
by 
  sorry

end tank_capacity_l301_301130


namespace smallest_positive_period_l301_301936

noncomputable def tan_period (a b x : ℝ) : ℝ := 
  Real.tan ((a + b) * x / 2)

theorem smallest_positive_period 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ p > 0, ∀ x, tan_period a b (x + p) = tan_period a b x ∧ p = 2 * Real.pi :=
by
  sorry

end smallest_positive_period_l301_301936


namespace value_of_a_l301_301647

-- Define the quadratic function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

-- Define the condition f(1) = f(2)
def condition (a b : ℝ) : Prop := f 1 a b = f 2 a b

-- The proof problem statement
theorem value_of_a (a b : ℝ) (h : condition a b) : a = -3 :=
by sorry

end value_of_a_l301_301647


namespace cube_decomposition_smallest_number_91_l301_301645

theorem cube_decomposition_smallest_number_91 (m : ℕ) (h1 : 0 < m) (h2 : (91 - 1) / 2 + 2 = m * m - m + 1) : m = 10 := by {
  sorry
}

end cube_decomposition_smallest_number_91_l301_301645


namespace truthful_dwarfs_count_l301_301427

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l301_301427


namespace atomic_weight_O_l301_301157

-- We define the atomic weights of sodium and chlorine
def atomic_weight_Na : ℝ := 22.99
def atomic_weight_Cl : ℝ := 35.45

-- We define the molecular weight of the compound
def molecular_weight_compound : ℝ := 74.0

-- We want to prove that the atomic weight of oxygen (O) is 15.56 given the above conditions
theorem atomic_weight_O : 
  (molecular_weight_compound = atomic_weight_Na + atomic_weight_Cl + w -> w = 15.56) :=
by
  sorry

end atomic_weight_O_l301_301157


namespace eunseo_change_correct_l301_301154

-- Define the given values
def r : ℕ := 3
def p_r : ℕ := 350
def b : ℕ := 2
def p_b : ℕ := 180
def P : ℕ := 2000

-- Define the total cost of candies and the change
def total_cost := r * p_r + b * p_b
def change := P - total_cost

-- Theorem statement
theorem eunseo_change_correct : change = 590 := by
  -- proof not required, so using sorry
  sorry

end eunseo_change_correct_l301_301154


namespace find_k_l301_301131

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n + 5 else (n + 1) / 2

theorem find_k (k : ℤ) (h1 : k % 2 = 0) (h2 : g (g (g k)) = 61) : k = 236 :=
by
  sorry

end find_k_l301_301131


namespace total_marbles_in_bowls_l301_301104

theorem total_marbles_in_bowls :
  let second_bowl := 600
  let first_bowl := 3 / 4 * second_bowl
  let third_bowl := 1 / 2 * first_bowl
  let fourth_bowl := 1 / 3 * second_bowl
  first_bowl + second_bowl + third_bowl + fourth_bowl = 1475 :=
by
  sorry

end total_marbles_in_bowls_l301_301104


namespace william_time_on_road_l301_301583

-- Define departure and arrival times
def departure_time := 7 -- 7:00 AM
def arrival_time := 20 -- 8:00 PM in 24-hour format

-- Define stop times in minutes
def stop1 := 25
def stop2 := 10
def stop3 := 25

-- Define total journey time in hours
def total_travel_time := arrival_time - departure_time

-- Define total stop time in hours
def total_stop_time := (stop1 + stop2 + stop3) / 60

-- Define time spent on the road
def time_on_road := total_travel_time - total_stop_time

-- The theorem to prove
theorem william_time_on_road : time_on_road = 12 := by
  sorry

end william_time_on_road_l301_301583


namespace parametric_curve_length_correct_l301_301018

noncomputable def parametric_curve_length : ℝ :=
  ∫ t in 0..2*Real.pi, Real.sqrt (9 + 7 * Real.sin t ^ 2)

theorem parametric_curve_length_correct :
  parametric_curve_length = 4 * Real.pi * Real.sqrt ((9 + 7) / 2) :=
sorry

end parametric_curve_length_correct_l301_301018


namespace price_difference_is_correct_l301_301284

-- Define the conditions
def original_price : ℝ := 1200
def increase_percentage : ℝ := 0.10
def decrease_percentage : ℝ := 0.15

-- Define the intermediate values
def increased_price : ℝ := original_price * (1 + increase_percentage)
def final_price : ℝ := increased_price * (1 - decrease_percentage)
def price_difference : ℝ := original_price - final_price

-- State the theorem to prove
theorem price_difference_is_correct : price_difference = 78 := 
by 
  sorry

end price_difference_is_correct_l301_301284


namespace hyperbola_standard_equation_l301_301904

theorem hyperbola_standard_equation (a b : ℝ) :
  (∃ (P Q : ℝ × ℝ), P = (-3, 2 * Real.sqrt 7) ∧ Q = (-6 * Real.sqrt 2, -7) ∧
    (∀ x y b, y^2 / b^2 - x^2 / a^2 = 1 ∧ (2 * Real.sqrt 7)^2 / b^2 - (-3)^2 / a^2 = 1
    ∧ (-7)^2 / b^2 - (-6 * Real.sqrt 2)^2 / a^2 = 1)) →
  b^2 = 25 → a^2 = 75 →
  (∀ x y, y^2 / (25:ℝ) - x^2 / (75:ℝ) = 1) :=
sorry

end hyperbola_standard_equation_l301_301904


namespace meet_time_correct_l301_301257

variable (circumference : ℕ) (speed_yeonjeong speed_donghun : ℕ)

def meet_time (circumference speed_yeonjeong speed_donghun : ℕ) : ℕ :=
  circumference / (speed_yeonjeong + speed_donghun)

theorem meet_time_correct
  (h_circumference : circumference = 3000)
  (h_speed_yeonjeong : speed_yeonjeong = 100)
  (h_speed_donghun : speed_donghun = 150) :
  meet_time circumference speed_yeonjeong speed_donghun = 12 :=
by
  rw [h_circumference, h_speed_yeonjeong, h_speed_donghun]
  norm_num
  sorry

end meet_time_correct_l301_301257


namespace length_first_train_correct_l301_301409

noncomputable def length_first_train 
    (speed_train1_kmph : ℕ := 120)
    (speed_train2_kmph : ℕ := 80)
    (length_train2_m : ℝ := 290.04)
    (time_sec : ℕ := 9) 
    (conversion_factor : ℝ := (5 / 18)) : ℝ :=
  let relative_speed_kmph := speed_train1_kmph + speed_train2_kmph
  let relative_speed_mps := relative_speed_kmph * conversion_factor
  let total_distance_m := relative_speed_mps * time_sec
  let length_train1_m := total_distance_m - length_train2_m
  length_train1_m

theorem length_first_train_correct 
    (L1_approx : ℝ := 210) :
    length_first_train = L1_approx :=
  by
  sorry

end length_first_train_correct_l301_301409


namespace complex_multiplication_l301_301595

theorem complex_multiplication:
  (2 + 2 * complex.I) * (1 - 2 * complex.I) = 6 - 2 * complex.I := by
  sorry

end complex_multiplication_l301_301595


namespace mike_pull_ups_per_week_l301_301537

theorem mike_pull_ups_per_week (pull_ups_per_entry entries_per_day days_per_week : ℕ)
  (h1 : pull_ups_per_entry = 2)
  (h2 : entries_per_day = 5)
  (h3 : days_per_week = 7)
  : pull_ups_per_entry * entries_per_day * days_per_week = 70 := 
by
  sorry

end mike_pull_ups_per_week_l301_301537


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301773

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301773


namespace g_g_g_3_equals_107_l301_301181

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_g_g_3_equals_107 : g (g (g 3)) = 107 := 
by 
  sorry

end g_g_g_3_equals_107_l301_301181


namespace sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l301_301252

theorem sum_of_last_three_digits_9_pow_15_plus_15_pow_15 :
  (9 ^ 15 + 15 ^ 15) % 1000 = 24 :=
by
  sorry

end sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l301_301252


namespace rectangle_length_increase_decrease_l301_301135

theorem rectangle_length_increase_decrease
  (L : ℝ)
  (width : ℝ)
  (increase_percentage : ℝ)
  (decrease_percentage : ℝ)
  (new_width : ℝ)
  (initial_area : ℝ)
  (new_length : ℝ)
  (new_area : ℝ)
  (HLW : width = 40)
  (Hinc : increase_percentage = 0.30)
  (Hdec : decrease_percentage = 0.17692307692307693)
  (Hnew_width : new_width = 40 - (decrease_percentage * 40))
  (Hinitial_area : initial_area = L * 40)
  (Hnew_length : new_length = 1.30 * L)
  (Hequal_area : new_length * new_width = L * 40) :
  L = 30.76923076923077 :=
by
  sorry

end rectangle_length_increase_decrease_l301_301135


namespace find_positive_number_l301_301332

theorem find_positive_number (m : ℝ) 
  (h : (m - 1)^2 = (3 * m - 5)^2) : 
  (m - 1)^2 = 1 ∨ (m - 1)^2 = 1 / 4 :=
by sorry

end find_positive_number_l301_301332


namespace num_distinct_intersections_l301_301022

def linear_eq1 (x y : ℝ) := x + 2 * y - 10
def linear_eq2 (x y : ℝ) := x - 4 * y + 8
def linear_eq3 (x y : ℝ) := 2 * x - y - 1
def linear_eq4 (x y : ℝ) := 5 * x + 3 * y - 15

theorem num_distinct_intersections (n : ℕ) :
  (∀ x y : ℝ, linear_eq1 x y = 0 ∨ linear_eq2 x y = 0) ∧ 
  (∀ x y : ℝ, linear_eq3 x y = 0 ∨ linear_eq4 x y = 0) →
  n = 3 :=
  sorry

end num_distinct_intersections_l301_301022


namespace negation_of_proposition_l301_301720

theorem negation_of_proposition (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by
  sorry

end negation_of_proposition_l301_301720


namespace tribe_leadership_organization_l301_301341

theorem tribe_leadership_organization (n : ℕ) (m : ℕ) (k : ℕ) (total : ℕ)
  (h1 : n = 11)  -- total members in the tribe
  (h2 : m = 1)   -- one chief
  (h3 : k = 2)   -- number of supporting chiefs
  (h4 : total = 11 * (Nat.choose 10 2) * (Nat.choose 8 2) * (Nat.choose 6 2)) :
  total = 207900 :=
by {
  rw [h1, h2, h3, h4],
  simp,
  sorry
}

end tribe_leadership_organization_l301_301341


namespace bake_sale_total_money_l301_301147

def dozens_to_pieces (dozens : Nat) : Nat :=
  dozens * 12

def total_money_raised
  (betty_chocolate_chip_dozen : Nat)
  (betty_oatmeal_raisin_dozen : Nat)
  (betty_brownies_dozen : Nat)
  (paige_sugar_cookies_dozen : Nat)
  (paige_blondies_dozen : Nat)
  (paige_cream_cheese_brownies_dozen : Nat)
  (price_per_cookie : Rat)
  (price_per_brownie_blondie : Rat) : Rat :=
let betty_cookies := dozens_to_pieces betty_chocolate_chip_dozen + dozens_to_pieces betty_oatmeal_raisin_dozen
let paige_cookies := dozens_to_pieces paige_sugar_cookies_dozen
let total_cookies := betty_cookies + paige_cookies
let betty_brownies := dozens_to_pieces betty_brownies_dozen
let paige_brownies_blondies := dozens_to_pieces paige_blondies_dozen + dozens_to_pieces paige_cream_cheese_brownies_dozen
let total_brownies_blondies := betty_brownies + paige_brownies_blondies
(total_cookies * price_per_cookie) + (total_brownies_blondies * price_per_brownie_blondie)

theorem bake_sale_total_money :
  total_money_raised 4 6 2 6 3 5 1 2 = 432 :=
by
  sorry

end bake_sale_total_money_l301_301147


namespace num_terms_100_pow_10_as_sum_of_tens_l301_301958

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l301_301958


namespace remainder_of_sum_of_primes_l301_301820

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301820


namespace percentage_increase_of_x_l301_301099

theorem percentage_increase_of_x 
  (x1 y1 : ℝ) 
  (h1 : ∀ x2 y2, (x1 * y1 = x2 * y2) → (y2 = 0.7692307692307693 * y1) → x2 = x1 * 1.3) : 
  ∃ P : ℝ, P = 30 :=
by 
  have P := 30 
  use P 
  sorry

end percentage_increase_of_x_l301_301099


namespace range_of_a_l301_301053

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * x ^ 2 - 2 * x

noncomputable def y' (x : ℝ) (a : ℝ) : ℝ := 1 / x + 2 * a * x - 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → y' x a ≥ 0) ↔ a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l301_301053


namespace dwarfs_truthful_count_l301_301442

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l301_301442


namespace sqrt_neg9_sq_l301_301263

theorem sqrt_neg9_sq : Real.sqrt ((-9 : Real)^2) = 9 := 
by 
  sorry

end sqrt_neg9_sq_l301_301263


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301799

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301799


namespace b_2018_eq_5043_l301_301913

def b (n : Nat) : Nat :=
  if n % 2 = 1 then 5 * ((n + 1) / 2) - 3 else 5 * (n / 2) - 2

theorem b_2018_eq_5043 : b 2018 = 5043 := by
  sorry

end b_2018_eq_5043_l301_301913


namespace cuboid_diagonal_length_l301_301407

theorem cuboid_diagonal_length (x y z : ℝ) 
  (h1 : y * z = Real.sqrt 2) 
  (h2 : z * x = Real.sqrt 3)
  (h3 : x * y = Real.sqrt 6) : 
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 6 :=
sorry

end cuboid_diagonal_length_l301_301407


namespace winning_majority_vote_l301_301195

def total_votes : ℕ := 600

def winning_percentage : ℝ := 0.70

def losing_percentage : ℝ := 0.30

theorem winning_majority_vote : (0.70 * (total_votes : ℝ) - 0.30 * (total_votes : ℝ)) = 240 := 
by
  sorry

end winning_majority_vote_l301_301195


namespace paid_more_than_free_l301_301726

def num_men : ℕ := 194
def num_women : ℕ := 235
def free_admission : ℕ := 68
def total_people (num_men num_women : ℕ) : ℕ := num_men + num_women
def paid_admission (total_people free_admission : ℕ) : ℕ := total_people - free_admission
def paid_over_free (paid_admission free_admission : ℕ) : ℕ := paid_admission - free_admission

theorem paid_more_than_free :
  paid_over_free (paid_admission (total_people num_men num_women) free_admission) free_admission = 293 := 
by
  sorry

end paid_more_than_free_l301_301726


namespace train_meets_john_l301_301292

open MeasureTheory

noncomputable def john_meets_train_probability : ℝ :=
  let train_arrival := measure_space.mk (Set.Icc 0 60) (by apply_instance)
  let john_arrival := measure_space.mk (Set.Icc 0 120) (by apply_instance)
  let train_waits := 10
  
  let total_area := (120:ℝ) * 60
  let intersection_area := 0.5 * 60 * 10
  (intersection_area / total_area)

theorem train_meets_john :
  john_meets_train_probability = 1 / 24 :=
by 
  unfold john_meets_train_probability
  simp
  norm_num
  sorry

end train_meets_john_l301_301292


namespace number_of_tens_in_sum_l301_301952

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l301_301952


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301805

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301805


namespace tim_weekly_payment_l301_301388

-- Define the given conditions
def hourly_rate_bodyguard : ℕ := 20
def number_bodyguards : ℕ := 2
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7

-- Define the total weekly payment calculation
def weekly_payment : ℕ := (hourly_rate_bodyguard * number_bodyguards) * hours_per_day * days_per_week

-- The proof statement
theorem tim_weekly_payment : weekly_payment = 2240 := by
  sorry

end tim_weekly_payment_l301_301388


namespace sqrt_17_irrational_l301_301357

theorem sqrt_17_irrational : ¬ ∃ (q : ℚ), q * q = 17 := sorry

end sqrt_17_irrational_l301_301357


namespace prime_of_two_pow_sub_one_prime_l301_301087

theorem prime_of_two_pow_sub_one_prime {n : ℕ} (h : Nat.Prime (2^n - 1)) : Nat.Prime n :=
sorry

end prime_of_two_pow_sub_one_prime_l301_301087


namespace smallest_number_divisible_by_618_3648_60_l301_301593

theorem smallest_number_divisible_by_618_3648_60 :
  ∃ n : ℕ, (∀ m, (m + 1) % 618 = 0 ∧ (m + 1) % 3648 = 0 ∧ (m + 1) % 60 = 0 → m = 1038239) :=
by
  use 1038239
  sorry

end smallest_number_divisible_by_618_3648_60_l301_301593


namespace calculate_xy_l301_301167

theorem calculate_xy (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x * y = 32 :=
by
  sorry

end calculate_xy_l301_301167


namespace correctLikeTermsPair_l301_301855

def areLikeTerms (term1 term2 : String) : Bool :=
  -- Define the criteria for like terms (variables and their respective powers)
  sorry

def pairA : (String × String) := ("-2x^3", "-2x")
def pairB : (String × String) := ("-1/2ab", "18ba")
def pairC : (String × String) := ("x^2y", "-xy^2")
def pairD : (String × String) := ("4m", "4mn")

theorem correctLikeTermsPair :
  areLikeTerms pairA.1 pairA.2 = false ∧
  areLikeTerms pairB.1 pairB.2 = true ∧
  areLikeTerms pairC.1 pairC.2 = false ∧
  areLikeTerms pairD.1 pairD.2 = false :=
sorry

end correctLikeTermsPair_l301_301855


namespace problem_statement_l301_301336

/-
If x is equal to the sum of the even integers from 40 to 60 inclusive,
y is the number of even integers from 40 to 60 inclusive,
and z is the sum of the odd integers from 41 to 59 inclusive,
prove that x + y + z = 1061.
-/
theorem problem_statement :
  let x := (11 / 2) * (40 + 60)
  let y := 11
  let z := (10 / 2) * (41 + 59)
  x + y + z = 1061 :=
by
  sorry

end problem_statement_l301_301336


namespace james_and_lisa_pizzas_l301_301975

theorem james_and_lisa_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) :
  slices_per_pizza = 6 →
  2 * total_slices = 3 * 8 →
  total_slices / slices_per_pizza = 2 :=
by
  intros h1 h2
  sorry

end james_and_lisa_pizzas_l301_301975


namespace num_terms_100_pow_10_as_sum_of_tens_l301_301957

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l301_301957


namespace problem_l301_301164

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 4
noncomputable def c : ℝ := Real.log 9 / Real.log 4

theorem problem : a = c ∧ a > b :=
by
  sorry

end problem_l301_301164


namespace minimum_omega_is_4_l301_301648

noncomputable def min_omega (ω : ℝ) : ℝ := 
  if ω > 0 
  ∧ abs (arcsin (1/2)) < π / 2
  ∧ ∀ x : ℝ, (sin (ω * x + arcsin (1/2)) <= sin (ω * π / 12 + arcsin (1/2)))
  then ω else 0

theorem minimum_omega_is_4 : ∃ ω > 0, (ω = 4 ∧ min_omega ω = ω) :=
by
  use 4
  sorry

end minimum_omega_is_4_l301_301648


namespace fraction_of_august_tips_l301_301981

variable {A : ℝ} -- A denotes the average monthly tips for the other months.
variable {total_tips_6_months : ℝ} (h1 : total_tips_6_months = 6 * A)
variable {august_tips : ℝ} (h2 : august_tips = 6 * A)
variable {total_tips : ℝ} (h3 : total_tips = total_tips_6_months + august_tips)

theorem fraction_of_august_tips (h1 : total_tips_6_months = 6 * A)
                                (h2 : august_tips = 6 * A)
                                (h3 : total_tips = total_tips_6_months + august_tips) :
    (august_tips / total_tips) = 1 / 2 :=
by
    sorry

end fraction_of_august_tips_l301_301981


namespace symmetric_curve_eq_l301_301715

theorem symmetric_curve_eq : 
  (∃ x' y', (x' - 3)^2 + 4*(y' - 5)^2 = 4 ∧ (x' - 6 = x' + x) ∧ (y' - 10 = y' + y)) ->
  (∃ x y, (x - 6) ^ 2 + 4 * (y - 10) ^ 2 = 4) :=
by
  sorry

end symmetric_curve_eq_l301_301715


namespace sum_seven_consecutive_integers_l301_301223

theorem sum_seven_consecutive_integers (m : ℕ) :
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) + (m + 6) = 7 * m + 21 :=
by
  -- Sorry to skip the actual proof steps.
  sorry

end sum_seven_consecutive_integers_l301_301223


namespace tetrahedron_volume_l301_301371

theorem tetrahedron_volume (S R V : ℝ) (h : V = (1/3) * S * R) : 
  V = (1/3) * S * R := 
by 
  sorry

end tetrahedron_volume_l301_301371


namespace b_investment_l301_301411

theorem b_investment (x : ℝ) (total_profit A_investment B_investment C_investment A_profit: ℝ)
  (h1 : A_investment = 6300)
  (h2 : B_investment = x)
  (h3 : C_investment = 10500)
  (h4 : total_profit = 12600)
  (h5 : A_profit = 3780)
  (ratio_eq : (A_investment / (A_investment + B_investment + C_investment)) = (A_profit / total_profit)) :
  B_investment = 13700 :=
  sorry

end b_investment_l301_301411


namespace sum_of_tens_l301_301950

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l301_301950


namespace dwarfs_truthful_count_l301_301438

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l301_301438


namespace quadratic_inequality_solution_l301_301509

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 + 4 * m * x - 4 < 0) ↔ -1 < m ∧ m < 0 :=
by
  sorry

end quadratic_inequality_solution_l301_301509


namespace parkway_school_students_l301_301342

theorem parkway_school_students (total_boys total_soccer soccer_boys_percentage girls_not_playing_soccer : ℕ)
  (h1 : total_boys = 320)
  (h2 : total_soccer = 250)
  (h3 : soccer_boys_percentage = 86)
  (h4 : girls_not_playing_soccer = 95)
  (h5 : total_soccer * soccer_boys_percentage / 100 = 215) :
  total_boys + total_soccer - (total_soccer * soccer_boys_percentage / 100) + girls_not_playing_soccer = 450 :=
by
  sorry

end parkway_school_students_l301_301342


namespace tire_mileage_l301_301865

theorem tire_mileage (total_miles_driven : ℕ) (x : ℕ) (spare_tire_miles : ℕ):
  total_miles_driven = 40000 →
  spare_tire_miles = 2 * x →
  4 * x + spare_tire_miles = total_miles_driven →
  x = 6667 := 
by
  intros h_total h_spare h_eq
  sorry

end tire_mileage_l301_301865


namespace average_chemistry_mathematics_l301_301243

variable {P C M : ℝ}

theorem average_chemistry_mathematics (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  sorry

end average_chemistry_mathematics_l301_301243


namespace remainder_first_six_primes_div_seventh_l301_301840

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301840


namespace solve_equation_l301_301242

theorem solve_equation 
  (x : ℝ) 
  (h : (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1)) : 
  x = 5 / 2 := 
sorry

end solve_equation_l301_301242


namespace truthful_dwarfs_count_l301_301426

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l301_301426


namespace largest_M_in_base_7_l301_301202

-- Define the base and the bounds for M^2
def base : ℕ := 7
def lower_bound : ℕ := base^3
def upper_bound : ℕ := base^4

-- Define M and its maximum value.
def M : ℕ := 48

-- Define a function to convert a number to its base 7 representation
def to_base_7 (n : ℕ) : List ℕ :=
  if n == 0 then [0] else
  let rec digits (n : ℕ) : List ℕ :=
    if n == 0 then [] else (n % 7) :: digits (n / 7)
  digits n |>.reverse

-- Define the base 7 representation of 48
def M_base_7 : List ℕ := to_base_7 M

-- The main statement asserting the conditions and the solution
theorem largest_M_in_base_7 :
  lower_bound ≤ M^2 ∧ M^2 < upper_bound ∧ M_base_7 = [6, 6] :=
by
  sorry

end largest_M_in_base_7_l301_301202


namespace allison_marbles_l301_301413

theorem allison_marbles (A B C : ℕ) (h1 : B = A + 8) (h2 : C = 3 * B) (h3 : C + A = 136) : 
  A = 28 :=
by
  sorry

end allison_marbles_l301_301413


namespace find_fraction_of_difference_eq_halves_l301_301654

theorem find_fraction_of_difference_eq_halves (x : ℚ) (h : 9 - x = 2.25) : x = 27 / 4 :=
by sorry

end find_fraction_of_difference_eq_halves_l301_301654


namespace fuel_tank_ethanol_l301_301004

theorem fuel_tank_ethanol (x : ℝ) (H : 0.12 * x + 0.16 * (208 - x) = 30) : x = 82 := 
by
  sorry

end fuel_tank_ethanol_l301_301004


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301789

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301789


namespace vertex_y_coord_of_h_l301_301897

def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 4 * x - 1
def h (x : ℝ) : ℝ := f x - g x

theorem vertex_y_coord_of_h : h (-1 / 10) = 79 / 20 := by
  sorry

end vertex_y_coord_of_h_l301_301897


namespace first_inequality_system_of_inequalities_l301_301548

-- First inequality problem
theorem first_inequality (x : ℝ) : 
  1 - (x - 3) / 6 > x / 3 → x < 3 := 
sorry

-- System of inequalities problem
theorem system_of_inequalities (x : ℝ) : 
  (x + 1 ≥ 3 * (x - 3)) ∧ ((x + 2) / 3 - (x - 1) / 4 > 1) → (1 < x ∧ x ≤ 5) := 
sorry

end first_inequality_system_of_inequalities_l301_301548


namespace terminal_side_in_quadrant_l301_301646

theorem terminal_side_in_quadrant (k : ℤ) (α : ℝ)
  (h: π + 2 * k * π < α ∧ α < (3 / 2) * π + 2 * k * π) :
  (π / 2) + k * π < α / 2 ∧ α / 2 < (3 / 4) * π + k * π :=
sorry

end terminal_side_in_quadrant_l301_301646


namespace fg_of_1_eq_15_l301_301187

def f (x : ℝ) := 2 * x - 3
def g (x : ℝ) := (x + 2) ^ 2

theorem fg_of_1_eq_15 : f (g 1) = 15 :=
by
  sorry

end fg_of_1_eq_15_l301_301187


namespace percent_profit_l301_301291

theorem percent_profit (CP LP SP Profit : ℝ) 
  (hCP : CP = 100) 
  (hLP : LP = CP + 0.30 * CP)
  (hSP : SP = LP - 0.10 * LP) 
  (hProfit : Profit = SP - CP) : 
  (Profit / CP) * 100 = 17 :=
by
  sorry

end percent_profit_l301_301291


namespace at_least_one_not_less_than_two_l301_301639

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) :=
sorry

end at_least_one_not_less_than_two_l301_301639


namespace Lisa_goal_achievable_l301_301144

open Nat

theorem Lisa_goal_achievable :
  ∀ (total_quizzes quizzes_with_A goal_percentage : ℕ),
  total_quizzes = 60 →
  quizzes_with_A = 25 →
  goal_percentage = 85 →
  (quizzes_with_A < goal_percentage * total_quizzes / 100) →
  (∃ remaining_quizzes, goal_percentage * total_quizzes / 100 - quizzes_with_A > remaining_quizzes) :=
by
  intros total_quizzes quizzes_with_A goal_percentage h_total h_A h_goal h_lack
  let needed_quizzes := goal_percentage * total_quizzes / 100
  let remaining_quizzes := total_quizzes - 35
  have h_needed := needed_quizzes - quizzes_with_A
  use remaining_quizzes
  sorry

end Lisa_goal_achievable_l301_301144


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301792

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301792


namespace hulk_jump_exceeds_2000_l301_301549

theorem hulk_jump_exceeds_2000 {n : ℕ} (h : n ≥ 1) :
  2^(n - 1) > 2000 → n = 12 :=
by
  sorry

end hulk_jump_exceeds_2000_l301_301549


namespace jessica_current_age_l301_301980

theorem jessica_current_age : 
  ∃ J M_d M_c : ℕ, 
    J = (M_d / 2) ∧ 
    M_d = M_c - 10 ∧ 
    M_c = 70 ∧ 
    J + 10 = 40 := 
sorry

end jessica_current_age_l301_301980


namespace cars_each_remaining_day_l301_301339

theorem cars_each_remaining_day (total_cars : ℕ) (monday_cars : ℕ) (tuesday_cars : ℕ)
  (wednesday_cars : ℕ) (thursday_cars : ℕ) (remaining_days : ℕ)
  (h_total : total_cars = 450)
  (h_mon : monday_cars = 50)
  (h_tue : tuesday_cars = 50)
  (h_wed : wednesday_cars = 2 * monday_cars)
  (h_thu : thursday_cars = 2 * monday_cars)
  (h_remaining : remaining_days = (total_cars - (monday_cars + tuesday_cars + wednesday_cars + thursday_cars)) / 3)
  :
  remaining_days = 50 := sorry

end cars_each_remaining_day_l301_301339


namespace trigonometric_translation_l301_301554

theorem trigonometric_translation :
  ∀ (x : ℝ), 3 * sin (2 * x - π / 6) = 3 * sin (2 * (x - π / 12)) :=
by sorry

end trigonometric_translation_l301_301554


namespace probability_region_C_l301_301864

theorem probability_region_C :
  let A := 1/5
  let B := 1/3
  let C := x
  let D := C
  let E := 2 * C
  let total := 1
  1 + (1/15) = 1 /\ x = 7/60 

end probability_region_C_l301_301864


namespace unique_function_l301_301903

theorem unique_function (f : ℕ → ℕ) (h : ∀ m n : ℕ, m > 0 → n > 0 → (f(m) + f(n) - m * n ≠ 0) ∧ (f(m) + f(n) - m * n) ∣ (m * f(m) + n * f(n))) : 
  ∀ n : ℕ, n > 0 → f(n) = n * n :=
sorry

end unique_function_l301_301903


namespace total_profit_l301_301392

-- Definitions based on the conditions
def tom_investment : ℝ := 30000
def tom_duration : ℝ := 12
def jose_investment : ℝ := 45000
def jose_duration : ℝ := 10
def jose_share_profit : ℝ := 25000

-- Theorem statement
theorem total_profit (tom_investment tom_duration jose_investment jose_duration jose_share_profit : ℝ) :
  (jose_share_profit / (jose_investment * jose_duration / (tom_investment * tom_duration + jose_investment * jose_duration)) = 5 / 9) →
  ∃ P : ℝ, P = 45000 :=
by
  sorry

end total_profit_l301_301392


namespace find_b_l301_301072

theorem find_b (g : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∃ x, g x ≠ 0) 
               (h3 : a > 0) (h4 : a ≠ 1) (h5 : ∀ x, (1 / (a ^ x - 1) - 1 / b) * g x = (1 / (a ^ (-x) - 1) - 1 / b) * g (-x)) :
    b = -2 :=
sorry

end find_b_l301_301072


namespace remainder_first_six_primes_div_seventh_l301_301850

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301850


namespace price_difference_is_correct_l301_301283

-- Define the conditions
def original_price : ℝ := 1200
def increase_percentage : ℝ := 0.10
def decrease_percentage : ℝ := 0.15

-- Define the intermediate values
def increased_price : ℝ := original_price * (1 + increase_percentage)
def final_price : ℝ := increased_price * (1 - decrease_percentage)
def price_difference : ℝ := original_price - final_price

-- State the theorem to prove
theorem price_difference_is_correct : price_difference = 78 := 
by 
  sorry

end price_difference_is_correct_l301_301283


namespace sum_of_tens_l301_301937

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l301_301937


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301748

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301748


namespace find_expression_for_a_n_l301_301912

-- Definitions for conditions in the problem
variable (a : ℕ → ℝ) -- Sequence is of positive real numbers
variable (S : ℕ → ℝ) -- Sum of the first n terms of the sequence

-- Condition that all terms in the sequence a_n are positive and indexed by natural numbers starting from 1
axiom pos_seq : ∀ n : ℕ, 0 < a (n + 1)
-- Condition for the sum of the terms: 4S_n = a_n^2 + 2a_n for n ∈ ℕ*
axiom sum_condition : ∀ n : ℕ, 4 * S (n + 1) = (a (n + 1))^2 + 2 * a (n + 1)

-- Theorem stating that sequence a_n = 2n given the above conditions
theorem find_expression_for_a_n : ∀ n : ℕ, a (n + 1) = 2 * (n + 1) := by
  sorry

end find_expression_for_a_n_l301_301912


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301797

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301797


namespace birds_on_branch_l301_301085

theorem birds_on_branch (initial_parrots remaining_parrots remaining_crows total_birds : ℕ) (h₁ : initial_parrots = 7) (h₂ : remaining_parrots = 2) (h₃ : remaining_crows = 1) (h₄ : initial_parrots - remaining_parrots = total_birds - remaining_crows - initial_parrots) : total_birds = 13 :=
sorry

end birds_on_branch_l301_301085


namespace range_of_slope_angle_proof_l301_301133

noncomputable def range_of_slope_angle (k : ℝ) : Prop :=
  ∃ A : ℝ × ℝ, A = (sqrt 3, 1) ∧ ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ y - 1 = k * (x - sqrt 3)

theorem range_of_slope_angle_proof :
  (∀ (k : ℝ), ∃ θ : ℝ, θ = real.arctan k ∧ 0 ≤ θ ∧ θ ≤ real.pi / 3) ↔
  range_of_slope_angle :=
sorry

end range_of_slope_angle_proof_l301_301133


namespace evaluate_floor_abs_neg_l301_301462

theorem evaluate_floor_abs_neg (x : ℝ) (h₁ : x = -45.7) : 
  floor (|x|) = 45 :=
by
  sorry

end evaluate_floor_abs_neg_l301_301462


namespace find_y_l301_301098

theorem find_y (x : ℝ) (y : ℝ) (h : (3 + y)^5 = (1 + 3 * y)^4) (hx : x = 1.5) : y = 1.5 :=
by
  -- Proof steps go here
  sorry

end find_y_l301_301098


namespace exactly_four_horses_meet_at_210_l301_301249

theorem exactly_four_horses_meet_at_210 (horses : List ℕ) 
  (h₁ : horses = [2, 3, 5, 7, 11, 13, 17]) : 
  ∃ (T : ℕ), T = 210 ∧ 
  ((horses.filter (λ n, T % n = 0)).length = 4) ∧ 
  ∀ T' > 0, ((horses.filter (λ n, T' % n = 0)).length = 4) → T' ≥ 210 :=
by
  sorry

end exactly_four_horses_meet_at_210_l301_301249


namespace total_difference_in_cards_l301_301894

theorem total_difference_in_cards (cards_chris : ℕ) (cards_charlie : ℕ) (cards_diana : ℕ) (cards_ethan : ℕ)
  (h_chris : cards_chris = 18)
  (h_charlie : cards_charlie = 32)
  (h_diana : cards_diana = 25)
  (h_ethan : cards_ethan = 40) :
  (cards_charlie - cards_chris) + (cards_diana - cards_chris) + (cards_ethan - cards_chris) = 43 := by
  sorry

end total_difference_in_cards_l301_301894


namespace remainder_first_six_primes_div_seventh_l301_301842

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301842


namespace magic_square_base_l301_301021

theorem magic_square_base :
  ∃ b : ℕ, (b + 1 + (b + 5) + 2 = 9 + (b + 3)) ∧ b = 3 :=
by
  use 3
  -- Proof in Lean goes here
  sorry

end magic_square_base_l301_301021


namespace greatest_a_no_integral_solution_l301_301529

theorem greatest_a_no_integral_solution (a : ℤ) :
  (∀ x : ℤ, |x + 1| ≥ a - 3 / 2) → a = 1 :=
by
  sorry

end greatest_a_no_integral_solution_l301_301529


namespace article_filling_correct_l301_301009

-- definitions based on conditions provided
def Gottlieb_Daimler := "Gottlieb Daimler was a German engineer."
def Invented_Car := "Daimler is normally believed to have invented the car."

-- Statement we want to prove
theorem article_filling_correct : 
  (Gottlieb_Daimler = "Gottlieb Daimler was a German engineer.") ∧ 
  (Invented_Car = "Daimler is normally believed to have invented the car.") →
  ("Gottlieb Daimler, a German engineer, is normally believed to have invented the car." = 
   "Gottlieb Daimler, a German engineer, is normally believed to have invented the car.") :=
by
  sorry

end article_filling_correct_l301_301009


namespace width_to_length_ratio_l301_301197

variables {w l P : ℕ}

theorem width_to_length_ratio :
  l = 10 → P = 30 → P = 2 * (l + w) → (w : ℚ) / l = 1 / 2 :=
by
  intro h1 h2 h3
  -- Noncomputable definition for rational division
  -- (ℚ is used for exact rational division)
  sorry

#check width_to_length_ratio

end width_to_length_ratio_l301_301197


namespace quadratic_coefficients_l301_301664

theorem quadratic_coefficients : 
  ∀ (b k : ℝ), (∀ x : ℝ, x^2 + b * x + 5 = (x - 2)^2 + k) → b = -4 ∧ k = 1 :=
by
  intro b k h
  have h1 := h 0
  have h2 := h 1
  sorry

end quadratic_coefficients_l301_301664


namespace determine_d_l301_301334

theorem determine_d (d c : ℕ) (hlcm : Nat.lcm 76 d = 456) (hhcf : Nat.gcd 76 d = c) : d = 24 :=
by
  sorry

end determine_d_l301_301334


namespace min_value_of_squares_l301_301354

theorem min_value_of_squares (a b s t : ℝ) (h1 : a + b = t) (h2 : a - b = s) :
  a^2 + b^2 = (t^2 + s^2) / 2 :=
sorry

end min_value_of_squares_l301_301354


namespace cost_of_1500_pieces_of_gum_in_dollars_l301_301552

theorem cost_of_1500_pieces_of_gum_in_dollars :
  (2 * 1500 * (1 - 0.10) / 100) = 27 := sorry

end cost_of_1500_pieces_of_gum_in_dollars_l301_301552


namespace Willey_Farm_Available_Capital_l301_301227

theorem Willey_Farm_Available_Capital 
  (total_acres : ℕ)
  (cost_per_acre_corn : ℕ)
  (cost_per_acre_wheat : ℕ)
  (acres_wheat : ℕ)
  (available_capital : ℕ) :
  total_acres = 4500 →
  cost_per_acre_corn = 42 →
  cost_per_acre_wheat = 35 →
  acres_wheat = 3400 →
  available_capital = (acres_wheat * cost_per_acre_wheat) + 
                      ((total_acres - acres_wheat) * cost_per_acre_corn) →
  available_capital = 165200 := sorry

end Willey_Farm_Available_Capital_l301_301227


namespace arithmetic_sequence_and_sum_properties_l301_301206

noncomputable def a_n (n : ℕ) : ℤ := 30 - 2 * n
noncomputable def S_n (n : ℕ) : ℤ := -n^2 + 29 * n

theorem arithmetic_sequence_and_sum_properties :
  (a_n 3 = 24 ∧ a_n 6 = 18) ∧
  (∀ n : ℕ, (S_n n = (n * (a_n 1 + a_n n)) / 2) ∧ ((a_n 3 = 24 ∧ a_n 6 = 18) → ∀ n : ℕ, a_n n = 30 - 2 * n)) ∧
  (S_n 14 = 210) :=
by 
  -- Proof omitted.
  sorry

end arithmetic_sequence_and_sum_properties_l301_301206


namespace denominator_of_fractions_l301_301965

theorem denominator_of_fractions (y a : ℝ) (hy : y > 0) 
  (h : (2 * y) / a + (3 * y) / a = 0.5 * y) : a = 10 :=
by
  sorry

end denominator_of_fractions_l301_301965


namespace geometric_sequence_arith_condition_l301_301969

-- Definitions of geometric sequence and arithmetic sequence condition
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions: \( \{a_n\} \) is a geometric sequence with \( a_2 \), \( \frac{1}{2}a_3 \), \( a_1 \) forming an arithmetic sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n + 1) = q * a n

def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop := a 2 = (1 / 2) * a 3 + a 1

-- Final theorem to prove
theorem geometric_sequence_arith_condition (hq : q^2 - q - 1 = 0) 
  (hgeo : is_geometric_sequence a q) 
  (harith : arithmetic_sequence_condition a) : 
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end geometric_sequence_arith_condition_l301_301969


namespace arithmetic_sequence_fifth_term_l301_301561

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15) 
  (h2 : a + 11 * d = 21) :
  a + 4 * d = 0 :=
by
  sorry

end arithmetic_sequence_fifth_term_l301_301561


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301741

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301741


namespace sequence_property_l301_301530

def Sn (n : ℕ) (a : ℕ → ℕ) : ℕ := (Finset.range (n + 1)).sum a

theorem sequence_property (a : ℕ → ℕ) (h : ∀ n : ℕ, Sn (n + 1) a = 2 * a n + 1) : a 3 = 2 :=
sorry

end sequence_property_l301_301530


namespace no_consecutive_beeches_probability_l301_301280

theorem no_consecutive_beeches_probability :
  let total_trees := 12
  let oaks := 3
  let holm_oaks := 4
  let beeches := 5
  let total_arrangements := (Nat.factorial total_trees) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks) * (Nat.factorial beeches))
  let favorable_arrangements :=
    let slots := oaks + holm_oaks + 1
    Nat.choose slots beeches * ((Nat.factorial (oaks + holm_oaks)) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks)))
  let probability := favorable_arrangements / total_arrangements
  probability = 7 / 99 :=
by
  sorry

end no_consecutive_beeches_probability_l301_301280


namespace treaty_signed_on_friday_l301_301373

def days_between (start_date : Nat) (end_date : Nat) : Nat := sorry

def day_of_week (start_day : Nat) (days_elapsed : Nat) : Nat :=
  (start_day + days_elapsed) % 7

def is_leap_year (year : Nat) : Bool :=
  if year % 4 = 0 then
    if year % 100 = 0 then
      if year % 400 = 0 then true else false
    else true
  else false

noncomputable def days_from_1802_to_1814 : Nat :=
  let leap_years := [1804, 1808, 1812]
  let normal_year_days := 365 * 9
  let leap_year_days := 366 * 3
  normal_year_days + leap_year_days

noncomputable def days_from_feb_5_to_apr_11_1814 : Nat :=
  24 + 31 + 11 -- days in February, March, and April 11

noncomputable def total_days_elapsed : Nat :=
  days_from_1802_to_1814 + days_from_feb_5_to_apr_11_1814

noncomputable def start_day : Nat := 5 -- Friday (0 = Sunday, ..., 5 = Friday, 6 = Saturday)

theorem treaty_signed_on_friday : day_of_week start_day total_days_elapsed = 5 := sorry

end treaty_signed_on_friday_l301_301373


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301767

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301767


namespace sin_double_alpha_trig_expression_l301_301914

theorem sin_double_alpha (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin (2 * α) = 4 * Real.sqrt 2 / 9 :=
sorry

theorem trig_expression (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin (α - 2 * π) * Real.cos (2 * π - α)) / (Real.sin (α + π / 2) ^ 2) = Real.sqrt 2 / 4 :=
sorry

end sin_double_alpha_trig_expression_l301_301914


namespace number_of_truthful_dwarfs_is_correct_l301_301460

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l301_301460


namespace min_sum_ab_l301_301203

theorem min_sum_ab (a b : ℤ) (h : a * b = 196) : a + b = -197 :=
sorry

end min_sum_ab_l301_301203


namespace f_of_2_l301_301314

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end f_of_2_l301_301314


namespace pizza_slices_left_l301_301348

theorem pizza_slices_left (total_slices john_ate : ℕ) 
  (initial_slices : total_slices = 12) 
  (john_slices : john_ate = 3) 
  (sam_ate : ¬¬(2 * john_ate = 6)) : 
  ∃ slices_left, slices_left = 3 :=
by
  sorry

end pizza_slices_left_l301_301348


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301745

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301745


namespace age_of_25th_student_l301_301244

theorem age_of_25th_student 
(A : ℤ) (B : ℤ) (C : ℤ) (D : ℤ)
(total_students : ℤ)
(total_age : ℤ)
(age_all_students : ℤ)
(avg_age_all_students : ℤ)
(avg_age_7_students : ℤ)
(avg_age_12_students : ℤ)
(avg_age_5_students : ℤ)
:
total_students = 25 →
avg_age_all_students = 18 →
avg_age_7_students = 20 →
avg_age_12_students = 16 →
avg_age_5_students = 19 →
total_age = total_students * avg_age_all_students →
age_all_students = total_age - (7 * avg_age_7_students + 12 * avg_age_12_students + 5 * avg_age_5_students) →
A = 7 * avg_age_7_students →
B = 12 * avg_age_12_students →
C = 5 * avg_age_5_students →
D = total_age - (A + B + C) →
D = 23 :=
by {
  sorry
}

end age_of_25th_student_l301_301244


namespace non_congruent_triangles_count_l301_301319

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def count_non_congruent_triangles : ℕ :=
  let a_values := [1, 2]
  let b_values := [2, 3]
  let triangles := [(1, 2, 2), (2, 2, 2), (2, 2, 3)]
  triangles.length

theorem non_congruent_triangles_count : count_non_congruent_triangles = 3 :=
  by
    -- Proof would go here
    sorry

end non_congruent_triangles_count_l301_301319


namespace no_real_solutions_sufficient_not_necessary_l301_301113

theorem no_real_solutions_sufficient_not_necessary (m : ℝ) : 
  (|m| < 1) → (m^2 < 4) :=
by
  sorry

end no_real_solutions_sufficient_not_necessary_l301_301113


namespace sum_first_5_terms_arithmetic_l301_301518

variable {a : ℕ → ℝ} -- Defining a sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_eq_1 : a 2 = 1
axiom a4_eq_5 : a 4 = 5

-- Theorem statement
theorem sum_first_5_terms_arithmetic (h_arith : is_arithmetic_sequence a) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end sum_first_5_terms_arithmetic_l301_301518


namespace dwarfs_truthful_count_l301_301439

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l301_301439


namespace cloth_total_selling_price_l301_301609

theorem cloth_total_selling_price
    (meters : ℕ) (profit_per_meter cost_price_per_meter : ℝ) :
    meters = 92 →
    profit_per_meter = 24 →
    cost_price_per_meter = 83.5 →
    (cost_price_per_meter + profit_per_meter) * meters = 9890 :=
by
  intros
  sorry

end cloth_total_selling_price_l301_301609


namespace increase_in_sides_of_polygon_l301_301665

theorem increase_in_sides_of_polygon (n n' : ℕ) (h : (n' - 2) * 180 - (n - 2) * 180 = 180) : n' = n + 1 :=
by
  sorry

end increase_in_sides_of_polygon_l301_301665


namespace insurance_plan_percentage_l301_301524

theorem insurance_plan_percentage
(MSRP : ℝ) (I : ℝ) (total_cost : ℝ) (state_tax_rate : ℝ)
(hMSRP : MSRP = 30)
(htotal_cost : total_cost = 54)
(hstate_tax_rate : state_tax_rate = 0.5)
(h_total_cost_eq : MSRP + I + state_tax_rate * (MSRP + I) = total_cost) :
(I / MSRP) * 100 = 20 :=
by
  -- You can leave the proof as sorry, as it's not needed for the problem
  sorry

end insurance_plan_percentage_l301_301524


namespace remainder_of_sum_of_primes_l301_301823

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301823


namespace find_y_l301_301062

theorem find_y
  (XYZ_is_straight_line : XYZ_is_straight_line)
  (angle_XYZ : ℝ)
  (angle_YWZ : ℝ)
  (y : ℝ)
  (exterior_angle_theorem : angle_XYZ = y + angle_YWZ)
  (h1 : angle_XYZ = 150)
  (h2 : angle_YWZ = 58) :
  y = 92 :=
by
  sorry

end find_y_l301_301062


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301796

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301796


namespace quiz_sum_correct_l301_301193

theorem quiz_sum_correct (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (h_sub : x - y = 4) (h_mul : x * y = 104) :
  x + y = 20 := by
  sorry

end quiz_sum_correct_l301_301193


namespace difference_in_floors_l301_301618

-- Given conditions
variable (FA FB FC : ℕ)
variable (h1 : FA = 4)
variable (h2 : FC = 5 * FB - 6)
variable (h3 : FC = 59)

-- The statement to prove
theorem difference_in_floors : FB - FA = 9 :=
by 
  -- Placeholder proof
  sorry

end difference_in_floors_l301_301618


namespace fourth_term_expansion_l301_301027

def binomial_term (n r : ℕ) (a b : ℚ) : ℚ :=
  (Nat.descFactorial n r) / (Nat.factorial r) * a^(n - r) * b^r

theorem fourth_term_expansion (x : ℚ) (hx : x ≠ 0) : 
  binomial_term 6 3 2 (-(1 / (x^(1/3)))) = (-160 / x) :=
by
  sorry

end fourth_term_expansion_l301_301027


namespace avg_of_other_two_l301_301228

-- Definitions and conditions from the problem
def avg (l : List ℕ) : ℕ := l.sum / l.length

variables {A B C D E : ℕ}
variables (h_avg_five : avg [A, B, C, D, E] = 20)
variables (h_sum_three : A + B + C = 48)
variables (h_twice : A = 2 * B)

-- Theorem to prove
theorem avg_of_other_two (A B C D E : ℕ) 
  (h_avg_five : avg [A, B, C, D, E] = 20)
  (h_sum_three : A + B + C = 48)
  (h_twice : A = 2 * B) :
  avg [D, E] = 26 := 
  sorry

end avg_of_other_two_l301_301228


namespace prime_sum_remainder_l301_301780

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301780


namespace fraction_of_married_men_l301_301418

-- We start by defining the conditions given in the problem.
def only_single_women_and_married_couples (total_women total_married_women : ℕ) :=
  total_women - total_married_women + total_married_women * 2

def probability_single_woman_single (total_women total_single_women : ℕ) :=
  total_single_women / total_women = 3 / 7

-- The main theorem we need to prove under the given conditions.
theorem fraction_of_married_men (total_women total_married_women : ℕ)
  (h1 : probability_single_woman_single total_women (total_women - total_married_women))
  : (total_married_women * 2) / (total_women + total_married_women) = 4 / 11 := sorry

end fraction_of_married_men_l301_301418


namespace find_A_l301_301384

theorem find_A (A B : ℕ) (h1 : 10 * A + 7 + (30 + B) = 73) : A = 3 := by
  sorry

end find_A_l301_301384


namespace suff_not_necessary_no_real_solutions_l301_301118

theorem suff_not_necessary_no_real_solutions :
  ∀ m : ℝ, |m| < 1 → (m : ℝ)^2 < 4 ∧ ∃ x, x^2 - m * x + 1 = 0 →
  ∀ a b : ℝ, (a = 1) ∧ (b = -m) ∧ (c = 1) → (b^2 - 4 * a * c) < 0 ∧ (m > -2) ∧ (m < 2) :=
by
  sorry

end suff_not_necessary_no_real_solutions_l301_301118


namespace no_closed_loop_after_replacement_l301_301006

theorem no_closed_loop_after_replacement (N M : ℕ) 
  (h1 : N = M) 
  (h2 : (N + M) % 4 = 0) :
  ¬((N - 1) - (M + 1)) % 4 = 0 :=
by
  sorry

end no_closed_loop_after_replacement_l301_301006


namespace vector_arithmetic_l301_301651

theorem vector_arithmetic (a b : ℝ × ℝ)
    (h₀ : a = (3, 5))
    (h₁ : b = (-2, 1)) :
    a - (2 : ℝ) • b = (7, 3) :=
sorry

end vector_arithmetic_l301_301651


namespace min_value_frac_sum_l301_301074

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  ∃ (a b : ℝ), (a + 3 * b = 1) ∧ (a > 0) ∧ (b > 0) ∧ (∀ (a b : ℝ), (a + 3 * b = 1) → 0 < a → 0 < b → (1 / a + 3 / b) ≥ 16) :=
sorry

end min_value_frac_sum_l301_301074


namespace Perimeter_PQR_leq_half_Perimeter_ABC_l301_301374
  
variable {α : Type}
variable (A B C : α)

-- Assume triangle and its properties are adequately defined in the context
axiom triangle_ABC : Triangle α
axiom angle_bisectors_intersect_D_E_F : IntersectAngleBisectors α triangle_ABC A B C

-- Definition of points and corresponding lengths to use for perimeter calculatons
noncomputable def perimeter_triangle_PQR : ℝ :=
  sorry -- Assuming calculation based on lengths from P, Q, R positions

noncomputable def perimeter_triangle_ABC : ℝ :=
  sorry -- Calculation using lengths of sides A, B, C

/-- Prove that the perimeter of triangle PQR is at most half the perimeter of triangle ABC -/
theorem Perimeter_PQR_leq_half_Perimeter_ABC :
  perimeter_triangle_PQR A B C ≤ 0.5 * perimeter_triangle_ABC A B C :=
sorry

end Perimeter_PQR_leq_half_Perimeter_ABC_l301_301374


namespace sum_of_fractions_l301_301892

theorem sum_of_fractions :
  (3 / 12 : Real) + (6 / 120) + (9 / 1200) = 0.3075 :=
by
  sorry

end sum_of_fractions_l301_301892


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301732

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301732


namespace sum_of_other_endpoint_coordinates_l301_301721

/-- 
  Given that (9, -15) is the midpoint of the segment with one endpoint (7, 4),
  find the sum of the coordinates of the other endpoint.
-/
theorem sum_of_other_endpoint_coordinates : 
  ∃ x y : ℤ, ((7 + x) / 2 = 9 ∧ (4 + y) / 2 = -15) ∧ (x + y = -23) :=
by
  sorry

end sum_of_other_endpoint_coordinates_l301_301721


namespace ratio_of_second_to_first_show_l301_301347

-- Definitions based on conditions
def first_show_length : ℕ := 30
def total_show_time : ℕ := 150
def second_show_length := total_show_time - first_show_length

-- Proof problem in Lean 4 statement
theorem ratio_of_second_to_first_show : 
  (second_show_length / first_show_length) = 4 := by
  sorry

end ratio_of_second_to_first_show_l301_301347


namespace AM_GM_inequality_l301_301086

theorem AM_GM_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_not_all_eq : x ≠ y ∨ y ≠ z ∨ z ≠ x) :
  (x + y) * (y + z) * (z + x) > 8 * x * y * z :=
by
  sorry

end AM_GM_inequality_l301_301086


namespace find_m_and_union_A_B_l301_301325

variable (m : ℝ)
noncomputable def A := ({3, 4, m^2 - 3 * m - 1} : Set ℝ)
noncomputable def B := ({2 * m, -3} : Set ℝ)

theorem find_m_and_union_A_B (h : A m ∩ B m = ({-3} : Set ℝ)) :
  m = 1 ∧ A m ∪ B m = ({-3, 2, 3, 4} : Set ℝ) :=
sorry

end find_m_and_union_A_B_l301_301325


namespace student_weight_loss_l301_301051

theorem student_weight_loss {S R L : ℕ} (h1 : S = 90) (h2 : S + R = 132) (h3 : S - L = 2 * R) : L = 6 := by
  sorry

end student_weight_loss_l301_301051


namespace perimeter_of_original_rectangle_l301_301136

theorem perimeter_of_original_rectangle
  (s : ℕ)
  (h1 : 4 * s = 24)
  (l w : ℕ)
  (h2 : l = 3 * s)
  (h3 : w = s) :
  2 * (l + w) = 48 :=
by
  sorry

end perimeter_of_original_rectangle_l301_301136


namespace percentage_markup_l301_301097

theorem percentage_markup (CP SP : ℕ) (hCP : CP = 800) (hSP : SP = 1000) :
  let Markup := SP - CP
  let PercentageMarkup := (Markup : ℚ) / CP * 100
  PercentageMarkup = 25 := by
  sorry

end percentage_markup_l301_301097


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301731

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301731


namespace percentage_students_50_59_is_10_71_l301_301668

theorem percentage_students_50_59_is_10_71 :
  let n_90_100 := 3
  let n_80_89 := 6
  let n_70_79 := 8
  let n_60_69 := 4
  let n_50_59 := 3
  let n_below_50 := 4
  let total_students := n_90_100 + n_80_89 + n_70_79 + n_60_69 + n_50_59 + n_below_50
  let fraction := (n_50_59 : ℚ) / total_students
  let percentage := (fraction * 100)
  percentage = 10.71 := by sorry

end percentage_students_50_59_is_10_71_l301_301668


namespace inequality_proof_l301_301644

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (1 + a)^2) + (1 / (1 + b)^2) + (1 / (1 + c)^2) + (1 / (1 + d)^2) ≥ 1 :=
by
  sorry

end inequality_proof_l301_301644


namespace remainder_first_six_primes_div_seventh_l301_301843

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301843


namespace distance_between_stations_proof_l301_301573

-- Definitions from conditions
def train_rate_1 : ℝ := 16
def train_rate_2 : ℝ := 21
def extra_distance : ℝ := 60

-- Let D be the distance traveled by the slower train when they meet
-- The distance traveled by faster train will be D + 60
def distance_by_slower_train (D : ℝ) : Prop := (D / train_rate_1) = ((D + extra_distance) / train_rate_2)

-- Now, based on the condition, we need the actual distance between the two stations
def distance_between_stations : ℝ := λ D : ℝ, 2 * D + extra_distance

theorem distance_between_stations_proof : ∃ D : ℝ, distance_by_slower_train D ∧ distance_between_stations D = 444 :=
by
  use 192
  split
  sorry -- The actual proof goes here

end distance_between_stations_proof_l301_301573


namespace number_of_tens_in_sum_l301_301942

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l301_301942


namespace total_insects_eaten_l301_301270

-- Definitions from the conditions
def numGeckos : Nat := 5
def insectsPerGecko : Nat := 6
def numLizards : Nat := 3
def insectsPerLizard : Nat := insectsPerGecko * 2

-- Theorem statement, proving total insects eaten is 66
theorem total_insects_eaten : numGeckos * insectsPerGecko + numLizards * insectsPerLizard = 66 := by
  sorry

end total_insects_eaten_l301_301270


namespace simplify_expression_l301_301545

theorem simplify_expression :
  (16 / 54) * (27 / 8) * (64 / 81) = 64 / 9 :=
by sorry

end simplify_expression_l301_301545


namespace sequence_transformation_possible_l301_301613

theorem sequence_transformation_possible 
  (a1 a2 : ℕ) (h1 : a1 ≤ 100) (h2 : a2 ≤ 100) (h3 : a1 ≥ a2) : 
  ∃ (operations : ℕ), operations ≤ 51 :=
by
  sorry

end sequence_transformation_possible_l301_301613


namespace johns_watermelon_weight_l301_301695

-- Michael's largest watermelon weighs 8 pounds
def michael_weight : ℕ := 8

-- Clay's watermelon weighs three times the size of Michael's watermelon
def clay_weight : ℕ := 3 * michael_weight

-- John's watermelon weighs half the size of Clay's watermelon
def john_weight : ℕ := clay_weight / 2

-- Prove that John's watermelon weighs 12 pounds
theorem johns_watermelon_weight : john_weight = 12 := by
  sorry

end johns_watermelon_weight_l301_301695


namespace tangent_slope_at_1_l301_301240

def f (x : ℝ) : ℝ := x^3 + x^2 + 1

theorem tangent_slope_at_1 : (deriv f 1) = 5 := by
  sorry

end tangent_slope_at_1_l301_301240


namespace two_pow_15000_mod_1250_l301_301258

theorem two_pow_15000_mod_1250 (h : 2 ^ 500 ≡ 1 [MOD 1250]) :
  2 ^ 15000 ≡ 1 [MOD 1250] :=
sorry

end two_pow_15000_mod_1250_l301_301258


namespace option_B_correct_l301_301256

theorem option_B_correct : 1 ∈ ({0, 1} : Set ℕ) := 
by
  sorry

end option_B_correct_l301_301256


namespace nominal_rate_of_interest_annual_l301_301714

theorem nominal_rate_of_interest_annual (EAR nominal_rate : ℝ) (n : ℕ) (h1 : EAR = 0.0816) (h2 : n = 2) : 
  nominal_rate = 0.0796 :=
by 
  sorry

end nominal_rate_of_interest_annual_l301_301714


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301766

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301766


namespace compound_proposition_C_l301_301483

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x 
def q : Prop := ∀ x : ℝ, sin x < x

theorem compound_proposition_C : p ∧ ¬q :=
by sorry

end compound_proposition_C_l301_301483


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301809

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301809


namespace octahedron_sum_l301_301346

-- Define the properties of an octahedron
def octahedron_edges := 12
def octahedron_vertices := 6
def octahedron_faces := 8

theorem octahedron_sum : octahedron_edges + octahedron_vertices + octahedron_faces = 26 := by
  -- Here we state that the sum of edges, vertices, and faces equals 26
  sorry

end octahedron_sum_l301_301346


namespace factorial_not_multiple_of_57_l301_301964

theorem factorial_not_multiple_of_57 (n : ℕ) (h : ¬ (57 ∣ n!)) : n < 19 := 
sorry

end factorial_not_multiple_of_57_l301_301964


namespace count_ways_to_choose_4_cards_l301_301656

-- A standard deck has 4 suits
def suits : Finset ℕ := {1, 2, 3, 4}

-- Each suit has 6 even cards: 2, 4, 6, 8, 10, and Queen (12)
def even_cards_per_suit : Finset ℕ := {2, 4, 6, 8, 10, 12}

-- Define the problem in Lean: 
-- Total number of ways to choose 4 cards such that all cards are of different suits and each is an even card.
theorem count_ways_to_choose_4_cards : (suits.card = 4 ∧ even_cards_per_suit.card = 6) → (1 * 6^4 = 1296) :=
by
  intros h
  have suits_distinct : suits.card = 4 := h.1
  have even_cards_count : even_cards_per_suit.card = 6 := h.2
  sorry

end count_ways_to_choose_4_cards_l301_301656


namespace ann_top_cost_l301_301882

noncomputable def cost_per_top (T : ℝ) := 75 = (5 * 7) + (2 * 10) + (4 * T)

theorem ann_top_cost : cost_per_top 5 :=
by {
  -- statement: prove cost per top given conditions
  sorry
}

end ann_top_cost_l301_301882


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301751

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301751


namespace non_congruent_triangle_classes_count_l301_301318

theorem non_congruent_triangle_classes_count :
  ∃ (q : ℕ),
  (∀ (a b : ℕ), a ≤ 2 ∧ 2 ≤ b ∧ a + 2 > b ∧ a + b > 2 ∧ b + 2 > a → 
  (a = 1 ∧ b = 2 ∨ a = 2 ∧ (b = 2 ∨ b = 3))) ∧ q = 3 := 
by
  use 3
  intro a b
  rintro ⟨ha, hb, h1, h2, h3⟩
  split
  { intro h,
    cases h with h1 h2,
    { exact or.inl ⟨h1, hb.eq_of_le⟩ },
    cases h2 with h2 h3,
    { exact or.inr ⟨hb.eq_of_le, or.inl rfl⟩ },
    { exact or.inr ⟨hb.eq_of_le, or.inr rfl⟩ } },
  { rintro (⟨ha1, rfl⟩ | ⟨rfl, hb1⟩),
    { refine ⟨le_refl _, le_add_of_lt hb, _, _, _⟩,
      { linarith },
      { linarith },
      { linarith } },
    cases hb1 with rfl rfl,
    { refine ⟨le_refl _, le_refl _, _, _, _⟩;
      linarith },
    { refine ⟨le_refl _, le_add_of_lt _, _, _, _⟩,
      { exact nat.zero_lt_one.trans one_lt_two },
      { linarith },
      { linarith },
      { linarith } } }
  sorry

end non_congruent_triangle_classes_count_l301_301318


namespace length_of_plot_l301_301556

open Real

variable (breadth : ℝ) (length : ℝ)
variable (b : ℝ)

axiom H1 : length = b + 40
axiom H2 : 26.5 * (4 * b + 80) = 5300

theorem length_of_plot : length = 70 :=
by
  -- To prove: The length of the plot is 70 meters.
  exact sorry

end length_of_plot_l301_301556


namespace mean_score_is_74_l301_301591

theorem mean_score_is_74 (σ q : ℝ)
  (h1 : 58 = q - 2 * σ)
  (h2 : 98 = q + 3 * σ) :
  q = 74 :=
by
  sorry

end mean_score_is_74_l301_301591


namespace tan_alpha_eq_4_over_3_expression_value_eq_4_l301_301030

-- Conditions
variable (α : ℝ) (hα1 : 0 < α) (hα2 : α < (Real.pi / 2)) (h_sin : Real.sin α = 4 / 5)

-- Prove: tan α = 4 / 3
theorem tan_alpha_eq_4_over_3 : Real.tan α = 4 / 3 :=
by
  sorry

-- Prove: the value of the given expression is 4
theorem expression_value_eq_4 : 
  (Real.sin (α + Real.pi) - 2 * Real.cos ((Real.pi / 2) + α)) / 
  (- Real.sin (-α) + Real.cos (Real.pi + α)) = 4 :=
by
  sorry

end tan_alpha_eq_4_over_3_expression_value_eq_4_l301_301030


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301737

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301737


namespace TotalGenuineItems_l301_301568

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem TotalGenuineItems : GenuinePurses + GenuineHandbags = 31 :=
  by
    -- proof
    sorry

end TotalGenuineItems_l301_301568


namespace part_a_l301_301123

theorem part_a (n : ℤ) (m : ℤ) (h : m = n + 2) : 
  n * m + 1 = (n + 1) ^ 2 := by
  sorry

end part_a_l301_301123


namespace complex_mul_eq_l301_301597

theorem complex_mul_eq :
  (2 + 2 * Complex.i) * (1 - 2 * Complex.i) = 6 - 2 * Complex.i := 
by
  intros
  sorry

end complex_mul_eq_l301_301597


namespace orchard_harvest_l301_301234

theorem orchard_harvest (sacks_per_section : ℕ) (sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 → sections = 8 → total_sacks = sacks_per_section * sections → total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end orchard_harvest_l301_301234


namespace transform_sequence_l301_301614

theorem transform_sequence (a1 a2 : ℕ) (h_a1a2 : a1 ≥ a2 ∧ a1 + a2 ≤ 100) :
  if |a1 - a2| ≤ 50 then
    true 
  else 
    false :=
by
  intro h_a1a2
  let b1 := 100 - a1
  let b2 := 100 - a2
  have h1 : |a1 - a2| ≤ 50 := sorry
  -- Assumption or further lemmas could be used here if necessary.
  sorry

end transform_sequence_l301_301614


namespace lily_milk_remaining_l301_301355

def lilyInitialMilk : ℚ := 4
def milkGivenAway : ℚ := 7 / 3
def milkLeft : ℚ := 5 / 3

theorem lily_milk_remaining : lilyInitialMilk - milkGivenAway = milkLeft := by
  sorry

end lily_milk_remaining_l301_301355


namespace smallest_integer_expression_l301_301852

theorem smallest_integer_expression :
  ∃ m n : ℤ, 1237 * m + 78653 * n = 1 :=
sorry

end smallest_integer_expression_l301_301852


namespace snickers_cost_l301_301192

variable (S : ℝ)

def cost_of_snickers (n : ℝ) : Prop :=
  2 * n + 3 * (2 * n) = 12

theorem snickers_cost (h : cost_of_snickers S) : S = 1.50 :=
by
  sorry

end snickers_cost_l301_301192


namespace total_money_raised_l301_301146

-- Define the baked goods quantities
def betty_chocolate_chip_cookies := 4
def betty_oatmeal_raisin_cookies := 6
def betty_regular_brownies := 2
def paige_sugar_cookies := 6
def paige_blondies := 3
def paige_cream_cheese_swirled_brownies := 5

-- Define the price of goods
def cookie_price := 1
def brownie_price := 2

-- State the total money raised
theorem total_money_raised :
  let total_cookies := 12 * (betty_chocolate_chip_cookies + betty_oatmeal_raisin_cookies + paige_sugar_cookies),
      total_brownies := 12 * (betty_regular_brownies + paige_blondies + paige_cream_cheese_swirled_brownies),
      money_from_cookies := total_cookies * cookie_price,
      money_from_brownies := total_brownies * brownie_price
  in money_from_cookies + money_from_brownies = 432 := by
  sorry

end total_money_raised_l301_301146


namespace x729_minus_inverse_l301_301915

theorem x729_minus_inverse (x : ℂ) (h : x - x⁻¹ = 2 * Complex.I) : x ^ 729 - x⁻¹ ^ 729 = 2 * Complex.I := 
by 
  sorry

end x729_minus_inverse_l301_301915


namespace number_of_women_in_first_class_l301_301700

-- Definitions for the conditions
def total_passengers : ℕ := 180
def percentage_women : ℝ := 0.65
def percentage_women_first_class : ℝ := 0.15

-- The desired proof statement
theorem number_of_women_in_first_class :
  (round (total_passengers * percentage_women * percentage_women_first_class) = 18) :=
by
  sorry  

end number_of_women_in_first_class_l301_301700


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301836

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301836


namespace length_of_ae_l301_301590

theorem length_of_ae
  (a b c d e : ℝ)
  (bc : ℝ)
  (cd : ℝ)
  (de : ℝ := 8)
  (ab : ℝ := 5)
  (ac : ℝ := 11)
  (h1 : bc = 2 * cd)
  (h2 : bc = ac - ab)
  : ab + bc + cd + de = 22 := 
by
  sorry

end length_of_ae_l301_301590


namespace smallest_possible_value_l301_301505

theorem smallest_possible_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^2 + b^2) / (a * b) + (a * b) / (a^2 + b^2) ≥ 2 :=
sorry

end smallest_possible_value_l301_301505


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301814

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301814


namespace maze_paths_unique_l301_301677

-- Define the conditions and branching points
def maze_structure (x : ℕ) (b : ℕ) : Prop :=
  x > 0 ∧ b > 0 ∧
  -- This represents the structure and unfolding paths at each point
  ∀ (i : ℕ), i < x → ∃ j < b, True

-- Define a function to count the number of unique paths given the number of branching points
noncomputable def count_paths (x : ℕ) (b : ℕ) : ℕ :=
  x * (2 ^ b)

-- State the main theorem
theorem maze_paths_unique : ∃ x b, maze_structure x b ∧ count_paths x b = 16 :=
by
  -- The proof contents are skipped for now
  sorry

end maze_paths_unique_l301_301677


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301794

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301794


namespace total_insects_eaten_l301_301268

theorem total_insects_eaten
  (geckos : ℕ)
  (insects_per_gecko : ℕ)
  (lizards : ℕ)
  (multiplier : ℕ)
  (h_geckos : geckos = 5)
  (h_insects_per_gecko : insects_per_gecko = 6)
  (h_lizards : lizards = 3)
  (h_multiplier : multiplier = 2) :
  geckos * insects_per_gecko + lizards * (insects_per_gecko * multiplier) = 66 :=
by
  rw [h_geckos, h_insects_per_gecko, h_lizards, h_multiplier]
  norm_num
  sorry

end total_insects_eaten_l301_301268


namespace genuine_items_count_l301_301570

def total_purses : ℕ := 26
def total_handbags : ℕ := 24
def fake_purses : ℕ := total_purses / 2
def fake_handbags : ℕ := total_handbags / 4
def genuine_purses : ℕ := total_purses - fake_purses
def genuine_handbags : ℕ := total_handbags - fake_handbags

theorem genuine_items_count : genuine_purses + genuine_handbags = 31 := by
  sorry

end genuine_items_count_l301_301570


namespace geometry_problem_l301_301261

variables {α : Type*} [EuclideanGeometry α]

-- Definitions of Points and Projections
variable (ABC P A' B' C' : α)
variable (A B C : Triangle α)

-- Assumptions: Orthogonality and Projections
def orthogonal_projections : Prop :=
  (P ⟂ BC ∧ P ⟂ CA ∧ P ⟂ AB)

-- Definitions of Points Intersection
variable (C₁ A₁ B₁ : α)
def is_parallel (ℓ₁ ℓ₂ : Line α) : Prop :=
  ∃ d : ℝ, ∀ p : α, p ∈ ℓ₁ → p + d ∈ ℓ₂

-- Assumptions: Projections Properties and Intersections
def projections_intersections : Prop :=
  (is_parallel (line_through P AB) (circumcircle PA'B' ) ∧
   is_parallel (line_through P BC) (circumcircle PB'C') ∧
   is_parallel (line_through P CA) (circumcircle PC'A'))

-- Goal 1: Intersection at a single point
def proof_goal_1 : Prop :=
  ∃ H : α, collinear ({A, A₁, H}) ∧ collinear ({B, B₁, H}) ∧ collinear ({C, C₁, H})

-- Goal 2: Similarity
def proof_goal_2 : Prop :=
  similar (triangle ABC) (triangle A₁ B₁ C₁)

-- Main Theorem Statement combining both goals
theorem geometry_problem :
  orthogonal_projections ABC P A' B' C' →
  projections_intersections ABC P A' B' C' C₁ A₁ B₁ →
  proof_goal_1 ABC P A' B' C₁ A₁ B₁ ∧
  proof_goal_2 ABC P A' B' C₁ A₁ B₁ :=
sorry

end geometry_problem_l301_301261


namespace identify_conic_section_is_hyperbola_l301_301301

theorem identify_conic_section_is_hyperbola :
  ∀ x y : ℝ, x^2 - 16 * y^2 - 10 * x + 4 * y + 36 = 0 →
  (∃ a b h c d k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ h = 0 ∧ (x - c)^2 / a^2 - (y - d)^2 / b^2 = k) :=
by
  sorry

end identify_conic_section_is_hyperbola_l301_301301


namespace time_differences_l301_301693

def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 8 -- minutes per mile
def lila_speed := 7 -- minutes per mile
def race_distance := 12 -- miles

noncomputable def malcolm_time := malcolm_speed * race_distance
noncomputable def joshua_time := joshua_speed * race_distance
noncomputable def lila_time := lila_speed * race_distance

theorem time_differences :
  joshua_time - malcolm_time = 24 ∧
  lila_time - malcolm_time = 12 :=
by
  sorry

end time_differences_l301_301693


namespace smallest_even_consecutive_sum_l301_301380

theorem smallest_even_consecutive_sum (n : ℕ) (h_even : n % 2 = 0) (h_sum : n + (n + 2) + (n + 4) = 162) : n = 52 :=
sorry

end smallest_even_consecutive_sum_l301_301380


namespace count_possible_x_values_l301_301415

theorem count_possible_x_values (x y : ℕ) (H : (x + 2) * (y + 2) - x * y = x * y) :
  (∃! x, ∃ y, (x - 2) * (y - 2) = 8) :=
by {
  sorry
}

end count_possible_x_values_l301_301415


namespace problem_part_1_problem_part_2_l301_301165

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f (x + 1) a + g x

-- Problem Part (1)
theorem problem_part_1 (a : ℝ) (h_pos : 0 < a) :
  (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 :=
sorry

-- Problem Part (2)
theorem problem_part_2 (a : ℝ) (h_cond : ∀ x, 0 ≤ x → h x a ≥ 1) :
  a ≤ 2 :=
sorry

end problem_part_1_problem_part_2_l301_301165


namespace johns_quarters_l301_301682

variable (x : ℕ)  -- Number of quarters John has

def number_of_dimes : ℕ := x + 3  -- Number of dimes
def number_of_nickels : ℕ := x - 6  -- Number of nickels

theorem johns_quarters (h : x + (x + 3) + (x - 6) = 63) : x = 22 :=
by
  sorry

end johns_quarters_l301_301682


namespace pool_capacity_l301_301608

-- Define the total capacity of the pool as a variable
variable (C : ℝ)

-- Define the conditions
def additional_water_needed (x : ℝ) : Prop :=
  x = 300

def increases_by_25_percent (x : ℝ) (y : ℝ) : Prop :=
  y = x * 0.25

-- State the proof problem
theorem pool_capacity :
  ∃ C : ℝ, additional_water_needed 300 ∧ increases_by_25_percent (0.75 * C) 300 ∧ C = 1200 :=
sorry

end pool_capacity_l301_301608


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301750

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301750


namespace probability_A_l301_301125

open ProbabilityTheory MeasureTheory

noncomputable def fair_coin : MeasureTheory.Measure Ω := 
  MeasureTheory.Measure.dirac (λ ω : Ω, (ω = 1/2))

def flip_n_times (n : Nat) : PMF (Fin n → Bool) :=
  PMF.finRange n >>= λ i, PMF.coin (1/2)

def event_A (n m : Nat) : Event (flip_n_times n × flip_n_times m) :=
  {ω | (ω.2.toFinset.filter id).card > (ω.1.toFinset.filter id).card}

open ProbabilityTheory.ProbabilityMeasure Event

theorem probability_A (n : Nat) (m : Nat) (hnm : n = 10) (hmm : m = 11):
  (probability (flip_n_times n) (event_A n m)) = 1 / 2 :=
by
  sorry

end probability_A_l301_301125


namespace equal_number_of_digits_l301_301405

noncomputable def probability_equal_digits : ℚ := (20 * (9/16)^3 * (7/16)^3)

theorem equal_number_of_digits :
  probability_equal_digits = 3115125 / 10485760 := by
  sorry

end equal_number_of_digits_l301_301405


namespace circle_center_radius_l301_301687

-- Define the necessary parameters and let Lean solve the equivalent proof problem
theorem circle_center_radius:
  (∃ a b r : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 2 * y = 1 ↔ (x + 4)^2 + (y - 1)^2 = 18) 
  ∧ a = -4 
  ∧ b = 1 
  ∧ r = 3 * Real.sqrt 2
  ∧ a + b + r = -3 + 3 * Real.sqrt 2) :=
by {
  sorry
}

end circle_center_radius_l301_301687


namespace password_guess_probability_l301_301876

def probability_correct_digit_within_two_attempts : Prop :=
  let total_digits := 10
  let prob_first_attempt := 1 / total_digits
  let prob_second_attempt := (9 / total_digits) * (1 / (total_digits - 1))
  (prob_first_attempt + prob_second_attempt) = 1 / 5

theorem password_guess_probability :
  probability_correct_digit_within_two_attempts :=
by
  -- proof goes here
  sorry

end password_guess_probability_l301_301876


namespace smallest_x_for_square_l301_301629

theorem smallest_x_for_square (N : ℕ) (h1 : ∃ x : ℕ, x > 0 ∧ 1260 * x = N^2) : ∃ x : ℕ, x = 35 :=
by
  sorry

end smallest_x_for_square_l301_301629


namespace gg_of_3_is_107_l301_301180

-- Define the function g
def g (x : ℕ) : ℕ := 3 * x + 2

-- State that g(g(g(3))) equals 107
theorem gg_of_3_is_107 : g (g (g 3)) = 107 := by
  sorry

end gg_of_3_is_107_l301_301180


namespace tangent_line_at_point_l301_301716

noncomputable def tangent_line_eq (x y : ℝ) : Prop := x^3 - y = 0

theorem tangent_line_at_point :
  tangent_line_eq (-2) (-8) →
  ∃ (k : ℝ), (k = 12) ∧ (12 * x - y + 16 = 0) :=
sorry

end tangent_line_at_point_l301_301716


namespace original_number_of_people_l301_301926

theorem original_number_of_people (x : ℕ) 
  (h1 : (x / 2) - ((x / 2) / 3) = 12) : 
  x = 36 :=
sorry

end original_number_of_people_l301_301926


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301733

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301733


namespace refrigerator_cost_is_15000_l301_301214

theorem refrigerator_cost_is_15000 (R : ℝ) 
  (phone_cost : ℝ := 8000)
  (phone_profit : ℝ := 0.10) 
  (fridge_loss : ℝ := 0.03) 
  (overall_profit : ℝ := 350) :
  (0.97 * R + phone_cost * (1 + phone_profit) = (R + phone_cost) + overall_profit) →
  (R = 15000) :=
by
  sorry

end refrigerator_cost_is_15000_l301_301214


namespace sum_of_roots_l301_301528

variable (x1 x2 k m : ℝ)
variable (h1 : x1 ≠ x2)
variable (h2 : 4 * x1^2 - k * x1 = m)
variable (h3 : 4 * x2^2 - k * x2 = m)

theorem sum_of_roots (x1 x2 k m : ℝ) (h1 : x1 ≠ x2)
  (h2 : 4 * x1 ^ 2 - k * x1 = m) (h3 : 4 * x2 ^ 2 - k * x2 = m) :
  x1 + x2 = k / 4 := sorry

end sum_of_roots_l301_301528


namespace total_ribbon_length_l301_301991

theorem total_ribbon_length (a b c d e f g h i : ℝ) 
  (H : a + b + c + d + e + f + g + h + i = 62) : 
  1.5 * (a + b + c + d + e + f + g + h + i) = 93 :=
by
  sorry

end total_ribbon_length_l301_301991


namespace range_of_x_l301_301183

theorem range_of_x 
  (x : ℝ)
  (h1 : 1 / x < 4) 
  (h2 : 1 / x > -6) 
  (h3 : x < 0) : 
  -1 / 6 < x ∧ x < 0 := 
by 
  sorry

end range_of_x_l301_301183


namespace find_x_l301_301129

theorem find_x (x : ℝ) (h : x * 1.6 - (2 * 1.4) / 1.3 = 4) : x = 3.846154 :=
sorry

end find_x_l301_301129


namespace oranges_per_store_visit_l301_301221

theorem oranges_per_store_visit (total_oranges : ℕ) (store_visits : ℕ) (h1 : total_oranges = 16) (h2 : store_visits = 8) :
  (total_oranges / store_visits) = 2 :=
by
  rw [h1, h2],
  exact Nat.div_eq_of_eq_mul_right (by norm_num) (by norm_num : 16 = 8 * 2)

end oranges_per_store_visit_l301_301221


namespace road_time_l301_301584

theorem road_time (departure : ℕ) (arrival : ℕ) (stops : List ℕ) : 
  departure = 7 ∧ arrival = 20 ∧ stops = [25, 10, 25] → 
  ((arrival - departure) * 60 - stops.sum) / 60 = 12 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 hstops
  have h_duration : (20 - 7) * 60 = 780 := rfl
  have h_stops : stops.sum = 60 := by
    simp [hstops]
  have h_total_time_on_road : (780 - 60) / 60 = 12 := rfl
  exact h_total_time_on_road

end road_time_l301_301584


namespace community_children_count_l301_301230

theorem community_children_count (total_members : ℕ) (pct_adult_men : ℝ) (ratio_adult_women : ℝ) :
  total_members = 2000 → pct_adult_men = 0.3 → ratio_adult_women = 2 →
  let num_adult_men := (pct_adult_men * total_members).to_nat in
  let num_adult_women := (ratio_adult_women * num_adult_men).to_nat in
  let total_adults := num_adult_men + num_adult_women in
  let num_children := total_members - total_adults in
  num_children = 200 :=
by
  intro h1 h2 h3
  let num_adult_men := (0.3 * 2000).to_nat
  let num_adult_women := (2 * num_adult_men).to_nat
  let total_adults := num_adult_men + num_adult_women
  let num_children := 2000 - total_adults
  -- we need to skip the proof part
  sorry

end community_children_count_l301_301230


namespace find_y_l301_301005

theorem find_y (y : ℕ) : (8000 * 6000 = 480 * 10 ^ y) → y = 5 :=
by
  intro h
  sorry

end find_y_l301_301005


namespace correct_quotient_l301_301190

theorem correct_quotient (Q : ℤ) (D : ℤ) (h1 : D = 21 * Q) (h2 : D = 12 * 35) : Q = 20 :=
by {
  sorry
}

end correct_quotient_l301_301190


namespace jack_jill_speed_l301_301068

-- Define conditions for Jack and Jill's speeds.
def jill_distance (x : ℝ) : ℝ := x^2 + x - 72
def jill_time (x : ℝ) : ℝ := x + 8
def jack_speed (x : ℝ) : ℝ := x^2 - 7x - 18
def jill_speed (x : ℝ) : ℝ := (x^2 + x - 72) / (x + 8)

-- Define the main theorem to prove
theorem jack_jill_speed : (jack_speed 10 = 2) ∧ (jill_speed 10 = 2) :=
by
  sorry

end jack_jill_speed_l301_301068


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301756

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301756


namespace polygonal_pyramid_faces_l301_301872

/-- A polygonal pyramid is a three-dimensional solid. Its base is a regular polygon. Each of the vertices of the polygonal base is connected to a single point, called the apex. The sum of the number of edges and the number of vertices of a particular polygonal pyramid is 1915. This theorem states that the number of faces of this pyramid is 639. -/
theorem polygonal_pyramid_faces (n : ℕ) (hn : 2 * n + (n + 1) = 1915) : n + 1 = 639 :=
by
  sorry

end polygonal_pyramid_faces_l301_301872


namespace number_of_fence_panels_is_10_l301_301868

def metal_rods_per_sheet := 10
def metal_rods_per_beam := 4
def sheets_per_panel := 3
def beams_per_panel := 2
def total_metal_rods := 380

theorem number_of_fence_panels_is_10 :
  (total_metal_rods = 380) →
  (metal_rods_per_sheet = 10) →
  (metal_rods_per_beam = 4) →
  (sheets_per_panel = 3) →
  (beams_per_panel = 2) →
  380 / (3 * 10 + 2 * 4) = 10 := 
by 
  sorry

end number_of_fence_panels_is_10_l301_301868


namespace measurable_length_l301_301141

-- Definitions of lines, rays, and line segments

-- A line is infinitely long with no endpoints.
def isLine (l : Type) : Prop := ∀ x y : l, (x ≠ y)

-- A line segment has two endpoints and a finite length.
def isLineSegment (ls : Type) : Prop := ∃ a b : ls, a ≠ b ∧ ∃ d : ℝ, d > 0

-- A ray has one endpoint and is infinitely long.
def isRay (r : Type) : Prop := ∃ e : r, ∀ x : r, x ≠ e

-- Problem statement
theorem measurable_length (x : Type) : isLineSegment x → (∃ d : ℝ, d > 0) :=
by
  -- Proof is not required
  sorry

end measurable_length_l301_301141


namespace non_adjacent_boys_arrangements_l301_301360

-- We define the number of boys and girls
def boys := 4
def girls := 6

-- The function to compute combinations C(n, k)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- The function to compute permutations P(n, k)
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- The total arrangements where 2 selected boys are not adjacent
def total_non_adjacent_arrangements : ℕ :=
  (combinations boys 2) * (combinations girls 3) * (permutations 3 3) * (permutations (3 + 1) 2)

theorem non_adjacent_boys_arrangements :
  total_non_adjacent_arrangements = 8640 := by
  sorry

end non_adjacent_boys_arrangements_l301_301360


namespace suff_not_necessary_no_real_solutions_l301_301117

theorem suff_not_necessary_no_real_solutions :
  ∀ m : ℝ, |m| < 1 → (m : ℝ)^2 < 4 ∧ ∃ x, x^2 - m * x + 1 = 0 →
  ∀ a b : ℝ, (a = 1) ∧ (b = -m) ∧ (c = 1) → (b^2 - 4 * a * c) < 0 ∧ (m > -2) ∧ (m < 2) :=
by
  sorry

end suff_not_necessary_no_real_solutions_l301_301117


namespace tan_six_theta_eq_l301_301049

theorem tan_six_theta_eq (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (6 * θ) = 21 / 8 :=
by
  sorry

end tan_six_theta_eq_l301_301049


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301801

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301801


namespace additional_money_needed_l301_301997

/-- Mrs. Smith needs to calculate the additional money required after a discount -/
theorem additional_money_needed
  (initial_amount : ℝ) (ratio_more : ℝ) (discount_rate : ℝ) (final_amount_needed : ℝ) (additional_needed : ℝ)
  (h_initial : initial_amount = 500)
  (h_ratio : ratio_more = 2/5)
  (h_discount : discount_rate = 15/100)
  (h_total_needed : final_amount_needed = initial_amount * (1 + ratio_more) * (1 - discount_rate))
  (h_additional : additional_needed = final_amount_needed - initial_amount) :
  additional_needed = 95 :=
by 
  sorry

end additional_money_needed_l301_301997


namespace length_of_BC_l301_301394

-- Define the given conditions and the theorem using Lean
theorem length_of_BC 
  (A B C : ℝ × ℝ) 
  (hA : A = (0, 0)) 
  (hB : ∃ b : ℝ, B = (-b, -b^2)) 
  (hC : ∃ b : ℝ, C = (b, -b^2)) 
  (hBC_parallel_x_axis : ∀ b : ℝ, C.2 = B.2)
  (hArea : ∀ b : ℝ, b^3 = 72) 
  : ∀ b : ℝ, (BC : ℝ) = 2 * b := 
by
  sorry

end length_of_BC_l301_301394


namespace find_n_satisfies_equation_l301_301475

-- Definition of the problem:
def satisfies_equation (n : ℝ) : Prop := 
  (2 / (n + 1)) + (3 / (n + 1)) + (n / (n + 1)) = 4

-- The statement of the proof problem:
theorem find_n_satisfies_equation : 
  ∃ n : ℝ, satisfies_equation n ∧ n = 1/3 :=
by
  sorry

end find_n_satisfies_equation_l301_301475


namespace kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l301_301983

/-- Conditions: -/
def kitchen_clock_gain_rate : ℝ := 1.5 -- minutes per hour
def bedroom_clock_lose_rate : ℝ := 0.5 -- minutes per hour
def synchronization_time : ℝ := 0 -- time in hours when both clocks were correct

/-- Problem 1: -/
theorem kitchen_clock_correct_again :
  ∃ t : ℝ, 1.5 * t = 720 :=
by {
  sorry
}

/-- Problem 2: -/
theorem bedroom_clock_correct_again :
  ∃ t : ℝ, 0.5 * t = 720 :=
by {
  sorry
}

/-- Problem 3: -/
theorem both_clocks_same_time_again :
  ∃ t : ℝ, 2 * t = 720 :=
by {
  sorry
}

end kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l301_301983


namespace bacteria_growth_rate_l301_301199

theorem bacteria_growth_rate
  (r : ℝ) 
  (h1 : ∃ B D : ℝ, B * r^30 = D) 
  (h2 : ∃ B D : ℝ, B * r^25 = D / 32) :
  r = 2 := 
by 
  sorry

end bacteria_growth_rate_l301_301199


namespace inequality_always_holds_l301_301504

theorem inequality_always_holds (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end inequality_always_holds_l301_301504


namespace remainder_of_sum_of_primes_l301_301819

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301819


namespace prime_sum_remainder_l301_301779

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301779


namespace remainder_when_divided_by_18_l301_301076

theorem remainder_when_divided_by_18 (n : ℕ) (r3 r6 r9 : ℕ)
  (hr3 : r3 = n % 3)
  (hr6 : r6 = n % 6)
  (hr9 : r9 = n % 9)
  (h_sum : r3 + r6 + r9 = 15) :
  n % 18 = 17 := sorry

end remainder_when_divided_by_18_l301_301076


namespace has_propertyT_f1_no_propertyT_f2_no_propertyT_f3_no_propertyT_f4_l301_301511

-- Definitions
def PropertyT (f : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), deriv f x1 * deriv f x2 = -1

-- Functions
def f1 (x : ℝ) := Real.sin x
def f2 (x : ℝ) := Real.log x
def f3 (x : ℝ) := Real.exp x
def f4 (x : ℝ) := x^3

-- Prove PropertyT for f1, f2, f3, and f4
theorem has_propertyT_f1 : PropertyT f1 :=
sorry

theorem no_propertyT_f2 : ¬ PropertyT f2 :=
sorry

theorem no_propertyT_f3 : ¬ PropertyT f3 :=
sorry

theorem no_propertyT_f4 : ¬ PropertyT f4 :=
sorry

end has_propertyT_f1_no_propertyT_f2_no_propertyT_f3_no_propertyT_f4_l301_301511


namespace range_of_a_l301_301317

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → 3 * x - a ≥ 0) → a ≤ 6 :=
by
  intros h
  sorry

end range_of_a_l301_301317


namespace find_smallest_even_number_l301_301383

theorem find_smallest_even_number (n : ℕ) (h : n + (n + 2) + (n + 4) = 162) : n = 52 :=
by
  sorry

end find_smallest_even_number_l301_301383


namespace two_trains_meet_at_distance_l301_301574

theorem two_trains_meet_at_distance 
  (D_slow D_fast : ℕ)  -- Distances traveled by the slower and faster trains
  (T : ℕ)  -- Time taken to meet
  (h0 : 16 * T = D_slow)  -- Distance formula for slower train
  (h1 : 21 * T = D_fast)  -- Distance formula for faster train
  (h2 : D_fast = D_slow + 60)  -- Faster train travels 60 km more than slower train
  : (D_slow + D_fast = 444) := sorry

end two_trains_meet_at_distance_l301_301574


namespace total_insects_eaten_l301_301267

theorem total_insects_eaten
  (geckos : ℕ)
  (insects_per_gecko : ℕ)
  (lizards : ℕ)
  (multiplier : ℕ)
  (h_geckos : geckos = 5)
  (h_insects_per_gecko : insects_per_gecko = 6)
  (h_lizards : lizards = 3)
  (h_multiplier : multiplier = 2) :
  geckos * insects_per_gecko + lizards * (insects_per_gecko * multiplier) = 66 :=
by
  rw [h_geckos, h_insects_per_gecko, h_lizards, h_multiplier]
  norm_num
  sorry

end total_insects_eaten_l301_301267


namespace find_a_in_terms_of_y_l301_301176

theorem find_a_in_terms_of_y (a b y : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * y^3) (h3 : a - b = 3 * y) :
  a = 3 * y :=
sorry

end find_a_in_terms_of_y_l301_301176


namespace prime_base_representation_of_360_l301_301729

theorem prime_base_representation_of_360 :
  ∃ (exponents : List ℕ), exponents = [3, 2, 1, 0]
  ∧ (2^exponents.head! * 3^(exponents.tail!.head!) * 5^(exponents.tail!.tail!.head!) * 7^(exponents.tail!.tail!.tail!.head!)) = 360 := by
sorry

end prime_base_representation_of_360_l301_301729


namespace find_a_and_x_range_l301_301599

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a_and_x_range :
  (∃ a, (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3)) →
  (∀ x, ∃ a, f x a ≤ 5 → 
    ((a = 1 → (0 ≤ x ∧ x ≤ 5)) ∧
     (a = 7 → (3 ≤ x ∧ x ≤ 8)))) :=
by sorry

end find_a_and_x_range_l301_301599


namespace remainder_of_sum_of_primes_l301_301818

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301818


namespace algebra_minimum_value_l301_301916

theorem algebra_minimum_value :
  ∀ x y : ℝ, ∃ m : ℝ, (∀ x y : ℝ, x^2 + y^2 + 6*x - 2*y + 12 ≥ m) ∧ m = 2 :=
by
  sorry

end algebra_minimum_value_l301_301916


namespace largest_square_perimeter_l301_301875

-- Define the conditions
def rectangle_length : ℕ := 80
def rectangle_width : ℕ := 60

-- Define the theorem to prove
theorem largest_square_perimeter : 4 * rectangle_width = 240 := by
  -- The proof steps are omitted
  sorry

end largest_square_perimeter_l301_301875


namespace find_initial_jellybeans_l301_301424

-- Definitions of the initial conditions
def jellybeans_initial (x : ℝ) (days : ℕ) (remaining : ℝ) := 
  days = 4 ∧ remaining = 48 ∧ (0.7 ^ days) * x = remaining

-- The theorem to prove
theorem find_initial_jellybeans (x : ℝ) : 
  jellybeans_initial x 4 48 → x = 200 :=
sorry

end find_initial_jellybeans_l301_301424


namespace number_of_truthful_dwarfs_l301_301452

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l301_301452


namespace dwarfs_truthful_count_l301_301446

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l301_301446


namespace xiaoli_estimate_smaller_l301_301588

variable (x y z : ℝ)
variable (hx : x > y) (hz : z > 0)

theorem xiaoli_estimate_smaller :
  (x - z) - (y + z) < x - y := 
by
  sorry

end xiaoli_estimate_smaller_l301_301588


namespace max_jogs_possible_l301_301891

theorem max_jogs_possible :
  ∃ (x y z : ℕ), (3 * x + 4 * y + 10 * z = 100) ∧ (x + y + z ≥ 20) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ (z ≥ 1) ∧
  (∀ (x' y' z' : ℕ), (3 * x' + 4 * y' + 10 * z' = 100) ∧ (x' + y' + z' ≥ 20) ∧ (x' ≥ 1) ∧ (y' ≥ 1) ∧ (z' ≥ 1) → z' ≤ z) :=
by
  sorry

end max_jogs_possible_l301_301891


namespace find_x_l301_301124

-- Define the known values
def a := 6
def b := 16
def c := 8
def desired_average := 13

-- Define the target number we need to find
def target_x := 22

-- Prove that the number we need to add to get the desired average is 22
theorem find_x : (a + b + c + target_x) / 4 = desired_average :=
by
  -- The proof itself is omitted as per instructions
  sorry

end find_x_l301_301124


namespace no_family_of_lines_exists_l301_301519

theorem no_family_of_lines_exists (k : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n, (1 : ℝ) = k n * (1 : ℝ) + (1 - k n)) ∧
  (∀ n, k (n + 1) = a n - b n ∧ a n = 1 - 1 / k n ∧ b n = 1 - k n) ∧
  (∀ n, k n * k (n + 1) ≥ 0) →
  False :=
by
  sorry

end no_family_of_lines_exists_l301_301519


namespace factor_expression_l301_301156

theorem factor_expression (a b c : ℝ) :
  3*a^3*(b^2 - c^2) - 2*b^3*(c^2 - a^2) + c^3*(a^2 - b^2) =
  (a - b)*(b - c)*(c - a)*(3*a^2 - 2*b^2 - 3*a^3/c + c) :=
sorry

end factor_expression_l301_301156


namespace roots_cubic_reciprocal_sum_l301_301498

theorem roots_cubic_reciprocal_sum (a b c : ℝ) 
(h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 27) (h₃ : a * b * c = 18) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 13 / 24 :=
by
  sorry

end roots_cubic_reciprocal_sum_l301_301498


namespace problem_statement_l301_301514

-- Mathematical Definitions
def num_students : ℕ := 6
def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_selected : ℕ := 3

def event_A : Prop := ∃ (boyA : ℕ), boyA < num_boys
def event_B : Prop := ∃ (girlB : ℕ), girlB < num_girls

def C (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select 3 out of 6 students
def total_ways : ℕ := C num_students num_selected

-- Probability of event A
def P_A : ℚ := C (num_students - 1) (num_selected - 1) / total_ways

-- Probability of events A and B
def P_AB : ℚ := C (num_students - 2) (num_selected - 2) / total_ways

-- Conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem problem_statement : P_B_given_A = 2 / 5 := sorry

end problem_statement_l301_301514


namespace find_expression_for_f_l301_301642

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem find_expression_for_f (x : ℝ) (h : x ≠ -1) 
    (hf : f ((1 - x) / (1 + x)) = x) : 
    f x = (1 - x) / (1 + x) :=
sorry

end find_expression_for_f_l301_301642


namespace find_points_l301_301174

def acute_triangle (A B C : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the triangle formed by A, B, and C is an acute-angled triangle.
  sorry -- This would be formalized ensuring all angles are less than 90 degrees.

def no_three_collinear (A B C D E : ℝ × ℝ × ℝ) : Prop :=
  -- Definition that ensures no three points among A, B, C, D, and E are collinear.
  sorry

def line_normal_to_plane (P Q R S : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the line through any two points P, Q is normal to the plane containing R, S, and the other point.
  sorry

theorem find_points (A B C : ℝ × ℝ × ℝ) (h_acute : acute_triangle A B C) :
  ∃ (D E : ℝ × ℝ × ℝ), no_three_collinear A B C D E ∧
    (∀ (P Q R R' : ℝ × ℝ × ℝ), 
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E) →
      (R' = A ∨ R' = B ∨ R' = C ∨ R' = D ∨ R' = E) →
      P ≠ Q → Q ≠ R → R ≠ R' →
      line_normal_to_plane P Q R R') :=
sorry

end find_points_l301_301174


namespace total_songs_l301_301162

variable (H : String) (M : String) (A : String) (T : String)

def num_songs (s : String) : ℕ :=
  if s = H then 9 else
  if s = M then 5 else
  if s = A ∨ s = T then 
    if H ≠ s ∧ M ≠ s then 6 else 7 
  else 0

theorem total_songs 
  (hH : num_songs H = 9)
  (hM : num_songs M = 5)
  (hA : 5 < num_songs A ∧ num_songs A < 9)
  (hT : 5 < num_songs T ∧ num_songs T < 9) :
  (num_songs H + num_songs M + num_songs A + num_songs T) / 3 = 10 :=
sorry

end total_songs_l301_301162


namespace truncated_cone_volume_l301_301014

theorem truncated_cone_volume :
  let R := 10
  let r := 5
  let h_t := 10
  let V_large := (1/3:Real) * Real.pi * (R^2) * (20)
  let V_small := (1/3:Real) * Real.pi * (r^2) * (10)
  (V_large - V_small) = (1750/3) * Real.pi :=
by
  sorry

end truncated_cone_volume_l301_301014


namespace standard_equation_line_BC_fixed_point_l301_301033

section EllipseProof

open Real

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions from the problem
axiom a_gt_b_gt_0 : ∀ (a b : ℝ), a > b → b > 0
axiom passes_through_point : ∀ (a b x y : ℝ), ellipse a b x y → (x = 1 ∧ y = sqrt 2 / 2)
axiom has_eccentricity : ∀ (a b c : ℝ), c / a = sqrt 2 / 2 → c^2 = a^2 - b^2 → b = 1

-- The standard equation of the ellipse
theorem standard_equation (a b : ℝ) (x y : ℝ) :
  a = sqrt 2 → b = 1 → ellipse a b x y → ellipse (sqrt 2) 1 x y :=
sorry

-- Prove that BC always passes through a fixed point
theorem line_BC_fixed_point (a b x1 x2 y1 y2 : ℝ) :
  a = sqrt 2 → b = 1 → 
  ellipse a b x1 y1 → ellipse a b x2 y2 →
  y1 = -y2 → x1 ≠ x2 → (-1, 0) = (-1, 0) →
  ∃ (k : ℝ) (x : ℝ), x = -2 ∧ y = 0 :=
sorry

end EllipseProof

end standard_equation_line_BC_fixed_point_l301_301033


namespace neg_p_equiv_l301_301042

-- The proposition p
def p : Prop := ∀ x : ℝ, x^2 - 1 < 0

-- Equivalent Lean theorem statement
theorem neg_p_equiv : ¬ p ↔ ∃ x₀ : ℝ, x₀^2 - 1 ≥ 0 :=
by
  sorry

end neg_p_equiv_l301_301042


namespace crickets_total_l301_301121

noncomputable def initial_amount : ℝ := 7.5
noncomputable def additional_amount : ℝ := 11.25
noncomputable def total_amount : ℝ := 18.75

theorem crickets_total : initial_amount + additional_amount = total_amount :=
by
  sorry

end crickets_total_l301_301121


namespace rihanna_money_left_l301_301359

theorem rihanna_money_left :
  ∀ (initial_amount mangoes apple_juice mango_cost juice_cost : ℕ),
    initial_amount = 50 →
    mangoes = 6 →
    apple_juice = 6 →
    mango_cost = 3 →
    juice_cost = 3 →
    initial_amount - (mangoes * mango_cost + apple_juice * juice_cost) = 14 :=
begin
  intros,
  sorry
end

end rihanna_money_left_l301_301359


namespace min_value_112_l301_301370

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  20 * (a^2 + b^2 + c^2 + d^2) - (a^3 * b + a^3 * c + a^3 * d + b^3 * a + b^3 * c + b^3 * d +
                                c^3 * a + c^3 * b + c^3 * d + d^3 * a + d^3 * b + d^3 * c)

theorem min_value_112 (a b c d : ℝ) (h : a + b + c + d = 8) : min_value_expr a b c d = 112 :=
  sorry

end min_value_112_l301_301370


namespace Lewis_found_20_items_l301_301531

-- Define the number of items Tanya found
def Tanya_items : ℕ := 4

-- Define the number of items Samantha found
def Samantha_items : ℕ := 4 * Tanya_items

-- Define the number of items Lewis found
def Lewis_items : ℕ := Samantha_items + 4

-- Theorem to prove the number of items Lewis found
theorem Lewis_found_20_items : Lewis_items = 20 := by
  sorry

end Lewis_found_20_items_l301_301531


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301757

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301757


namespace remainder_of_sum_of_primes_l301_301824

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301824


namespace no_consecutive_squares_of_arithmetic_progression_l301_301229

theorem no_consecutive_squares_of_arithmetic_progression (d : ℕ):
  (d % 10000 = 2019) →
  (∀ a b c : ℕ, a < b ∧ b < c → b^2 - a^2 = d ∧ c^2 - b^2 = d →
  false) :=
sorry

end no_consecutive_squares_of_arithmetic_progression_l301_301229


namespace square_side_increase_l301_301724

variable (s : ℝ)  -- original side length of the square.
variable (p : ℝ)  -- percentage increase of the side length.

theorem square_side_increase (h1 : (s * (1 + p / 100))^2 = 1.21 * s^2) : p = 10 := 
by
  sorry

end square_side_increase_l301_301724


namespace months_rent_in_advance_required_l301_301679

def janet_savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def deposit : ℕ := 500
def additional_needed : ℕ := 775

theorem months_rent_in_advance_required : 
  (janet_savings + additional_needed - deposit) / rent_per_month = 2 :=
by
  sorry

end months_rent_in_advance_required_l301_301679


namespace Randy_drew_pictures_l301_301540

variable (P Q R: ℕ)

def Peter_drew_pictures (P : ℕ) : Prop := P = 8
def Quincy_drew_pictures (Q P : ℕ) : Prop := Q = P + 20
def Total_drawing (R P Q : ℕ) : Prop := R + P + Q = 41

theorem Randy_drew_pictures
  (P_eq : Peter_drew_pictures P)
  (Q_eq : Quincy_drew_pictures Q P)
  (Total_eq : Total_drawing R P Q) :
  R = 5 :=
by 
  sorry

end Randy_drew_pictures_l301_301540


namespace angle_between_line_and_plane_l301_301662

-- Define the conditions
def angle_direct_vector_normal_vector (direction_vector_angle : ℝ) := direction_vector_angle = 120

-- Define the goal to prove
theorem angle_between_line_and_plane (direction_vector_angle : ℝ) :
  angle_direct_vector_normal_vector direction_vector_angle → direction_vector_angle = 120 → 90 - (180 - direction_vector_angle) = 30 :=
by
  intros h_angle_eq angle_120
  sorry

end angle_between_line_and_plane_l301_301662


namespace train_stop_time_per_hour_l301_301304

theorem train_stop_time_per_hour
    (v1 : ℕ) (v2 : ℕ)
    (h1 : v1 = 45)
    (h2 : v2 = 33) : ∃ (t : ℕ), t = 16 := by
  -- including the proof steps here is unnecessary, so we use sorry
  sorry

end train_stop_time_per_hour_l301_301304


namespace find_percentage_l301_301406

theorem find_percentage (P : ℝ) : 100 * (P / 100) + 20 = 100 → P = 80 :=
by
  sorry

end find_percentage_l301_301406


namespace number_of_tens_in_sum_l301_301946

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l301_301946


namespace sum_of_tens_l301_301939

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l301_301939


namespace slope_of_line_l301_301152

theorem slope_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : (- (4 : ℝ) / 7) = -4 / 7 :=
by
  -- Sorry for the proof for completeness
  sorry

end slope_of_line_l301_301152


namespace trigonometric_identity_solution_l301_301589

open Real

noncomputable def x_sol1 (k : ℤ) : ℝ := (π / 2) * (4 * k - 1)
noncomputable def x_sol2 (l : ℤ) : ℝ := (π / 3) * (6 * l + 1)
noncomputable def x_sol2_neg (l : ℤ) : ℝ := (π / 3) * (6 * l - 1)

theorem trigonometric_identity_solution (x : ℝ) :
    (3 * sin (x / 2) ^ 2 * cos (3 * π / 2 + x / 2) +
    3 * sin (x / 2) ^ 2 * cos (x / 2) -
    sin (x / 2) * cos (x / 2) ^ 2 =
    sin (π / 2 + x / 2) ^ 2 * cos (x / 2)) →
    (∃ k : ℤ, x = x_sol1 k) ∨
    (∃ l : ℤ, x = x_sol2 l ∨ x = x_sol2_neg l) :=
by
  sorry

end trigonometric_identity_solution_l301_301589


namespace faster_train_speed_is_45_l301_301393

noncomputable def speedOfFasterTrain (V_s : ℝ) (length_train : ℝ) (time : ℝ) : ℝ :=
  let V_r : ℝ := (length_train * 2) / (time / 3600)
  V_r - V_s

theorem faster_train_speed_is_45 
  (length_train : ℝ := 0.5)
  (V_s : ℝ := 30)
  (time : ℝ := 47.99616030717543) :
  speedOfFasterTrain V_s length_train time = 45 :=
sorry

end faster_train_speed_is_45_l301_301393


namespace lesser_solution_of_quadratic_eq_l301_301578

theorem lesser_solution_of_quadratic_eq : ∃ x ∈ {x | x^2 + 10*x - 24 = 0}, x = -12 :=
by 
  sorry

end lesser_solution_of_quadratic_eq_l301_301578


namespace proof_l301_301255

-- Define the expression
def expr : ℕ :=
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128)

-- Define the conjectured result
def result : ℕ := 5^128 - 4^128

-- Assert their equality
theorem proof : expr = result :=
by
    sorry

end proof_l301_301255


namespace measure_of_theta_l301_301575

theorem measure_of_theta 
  (ACB FEG DCE DEC : ℝ)
  (h1 : ACB = 10)
  (h2 : FEG = 26)
  (h3 : DCE = 14)
  (h4 : DEC = 33) : θ = 11 :=
by
  sorry

end measure_of_theta_l301_301575


namespace min_value_of_expression_l301_301488

-- Define the conditions in the problem
def conditions (m n : ℝ) : Prop :=
  (2 * m + n = 2) ∧ (m > 0) ∧ (n > 0)

-- Define the problem statement
theorem min_value_of_expression (m n : ℝ) (h : conditions m n) : 
  (∀ m n, conditions m n → (1 / m + 2 / n) ≥ 4) :=
by 
  sorry

end min_value_of_expression_l301_301488


namespace solution_set_of_inequality_af_neg2x_pos_l301_301667

-- Given conditions:
-- f(x) = x^2 + ax + b has roots -1 and 2
-- We need to prove that the solution set for af(-2x) > 0 is -1 < x < 1/2
theorem solution_set_of_inequality_af_neg2x_pos (a b : ℝ) (x : ℝ) 
  (h1 : -1 + 2 = -a) 
  (h2 : -1 * 2 = b) : 
  (a * ((-2 * x)^2 + a * (-2 * x) + b) > 0) = (-1 < x ∧ x < 1/2) :=
by
  sorry

end solution_set_of_inequality_af_neg2x_pos_l301_301667


namespace sum_of_roots_l301_301311

theorem sum_of_roots (x : ℝ) (h : x^2 = 10 * x + 16) : x = 10 :=
by 
  -- Rearrange the equation to standard form: x^2 - 10x - 16 = 0
  have eqn : x^2 - 10 * x - 16 = 0 := by sorry
  -- Use the formula for the sum of the roots of a quadratic equation
  -- Prove the sum of the roots is 10
  sorry

end sum_of_roots_l301_301311


namespace number_of_subsets_of_intersection_l301_301923

open Set

variable (A : Set ℕ) (B : Set ℕ)

noncomputable def setA : Set ℕ := {0, 2, 4, 6}
noncomputable def setB : Set ℕ := {n ∈ Set.Iio 3 | n ∈ (Set.range (λ x, 2^x))}

theorem number_of_subsets_of_intersection (hA : A = setA) (hB : B = setB) : 
  (A ∩ B).toFinset.powerset.card = 4 :=
by {
  rw [hA, hB],
  show (setA ∩ setB).toFinset.powerset.card = 4,
  sorry
}

end number_of_subsets_of_intersection_l301_301923


namespace total_amount_is_33_l301_301870

variable (n : ℕ) (c t : ℝ)

def total_amount_paid (n : ℕ) (c t : ℝ) : ℝ :=
  let cost_before_tax := n * c
  let tax := t * cost_before_tax
  cost_before_tax + tax

theorem total_amount_is_33
  (h1 : n = 5)
  (h2 : c = 6)
  (h3 : t = 0.10) :
  total_amount_paid n c t = 33 :=
by
  rw [h1, h2, h3]
  sorry

end total_amount_is_33_l301_301870


namespace line_does_not_pass_second_quadrant_l301_301335

theorem line_does_not_pass_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0 → ¬(x < 0 ∧ y > 0)) ↔ a ≤ -1 :=
by
  sorry

end line_does_not_pass_second_quadrant_l301_301335


namespace eccentricity_of_hyperbola_l301_301507

open Real

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ x y : ℝ, (x^2 + y^2 - 3 * x - 4 * y - 5 = 0) ∧ (ax - by = 0))
  (h4 : b = (3 / 4) * a) :
  let e := (sqrt (a^2 + b^2)) / a in e = 5 / 4 :=
by
  sorry

end eccentricity_of_hyperbola_l301_301507


namespace remainder_of_sum_of_primes_l301_301827

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301827


namespace farmer_owned_land_l301_301605

theorem farmer_owned_land (T : ℝ) (h : 0.10 * T = 720) : 0.80 * T = 5760 :=
by
  sorry

end farmer_owned_land_l301_301605


namespace log_simplification_l301_301216

theorem log_simplification :
  (1 / (Real.log 3 / Real.log 12 + 2))
  + (1 / (Real.log 2 / Real.log 8 + 2))
  + (1 / (Real.log 3 / Real.log 9 + 2)) = 2 :=
  sorry

end log_simplification_l301_301216


namespace eval_power_imaginary_unit_l301_301155

noncomputable def i : ℂ := Complex.I

theorem eval_power_imaginary_unit :
  i^20 + i^39 = 1 - i := by
  -- Skipping the proof itself, indicating it with "sorry"
  sorry

end eval_power_imaginary_unit_l301_301155


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301839

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301839


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301742

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301742


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301730

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301730


namespace sum_of_tens_l301_301949

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l301_301949


namespace pebbles_divisibility_impossibility_l301_301313

def initial_pebbles (K A P D : Nat) := K + A + P + D

theorem pebbles_divisibility_impossibility 
  (K A P D : Nat)
  (hK : K = 70)
  (hA : A = 30)
  (hP : P = 21)
  (hD : D = 45) :
  ¬ (∃ n : Nat, initial_pebbles K A P D = 4 * n) :=
by
  sorry

end pebbles_divisibility_impossibility_l301_301313


namespace james_net_income_correct_l301_301976

def regular_price_per_hour : ℝ := 20
def discount_percent : ℝ := 0.10
def rental_hours_per_day_monday : ℝ := 8
def rental_hours_per_day_wednesday : ℝ := 8
def rental_hours_per_day_friday : ℝ := 6
def rental_hours_per_day_sunday : ℝ := 5
def sales_tax_percent : ℝ := 0.05
def car_maintenance_cost_per_week : ℝ := 35
def insurance_fee_per_day : ℝ := 15

-- Total rental hours
def total_rental_hours : ℝ :=
  rental_hours_per_day_monday + rental_hours_per_day_wednesday + rental_hours_per_day_friday + rental_hours_per_day_sunday

-- Total rental income before discount
def total_rental_income : ℝ := total_rental_hours * regular_price_per_hour

-- Discounted rental income
def discounted_rental_income : ℝ := total_rental_income * (1 - discount_percent)

-- Total income with tax
def total_income_with_tax : ℝ := discounted_rental_income * (1 + sales_tax_percent)

-- Total expenses
def total_expenses : ℝ := car_maintenance_cost_per_week + (insurance_fee_per_day * 4)

-- Net income
def net_income : ℝ := total_income_with_tax - total_expenses

theorem james_net_income_correct : net_income = 415.30 :=
  by
    -- proof omitted
    sorry

end james_net_income_correct_l301_301976


namespace ellipse_range_k_l301_301496

theorem ellipse_range_k (k : ℝ) (h1 : 3 + k > 0) (h2 : 2 - k > 0) (h3 : k ≠ -1 / 2) :
  k ∈ Set.Ioo (-3 : ℝ) (-1 / 2) ∪ Set.Ioo (-1 / 2) (2 : ℝ) :=
sorry

end ellipse_range_k_l301_301496


namespace dwarfs_truthful_count_l301_301434

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l301_301434


namespace ratio_of_capitals_l301_301887

noncomputable def Ashok_loss (total_loss : ℝ) (Pyarelal_loss : ℝ) : ℝ := total_loss - Pyarelal_loss

theorem ratio_of_capitals (total_loss : ℝ) (Pyarelal_loss : ℝ) (Ashok_capital Pyarelal_capital : ℝ) 
    (h_total_loss : total_loss = 1200)
    (h_Pyarelal_loss : Pyarelal_loss = 1080)
    (h_Ashok_capital : Ashok_capital = 120)
    (h_Pyarelal_capital : Pyarelal_capital = 1080) :
    Ashok_capital / Pyarelal_capital = 1 / 9 :=
by
  sorry

end ratio_of_capitals_l301_301887


namespace intersection_complement_equivalence_l301_301500

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_equivalence :
  ((U \ M) ∩ N) = {3} := by
  sorry

end intersection_complement_equivalence_l301_301500


namespace find_smallest_even_number_l301_301382

theorem find_smallest_even_number (n : ℕ) (h : n + (n + 2) + (n + 4) = 162) : n = 52 :=
by
  sorry

end find_smallest_even_number_l301_301382


namespace shape_of_r_eq_c_in_cylindrical_coords_l301_301477

variable {c : ℝ}

theorem shape_of_r_eq_c_in_cylindrical_coords (h : c > 0) :
  ∀ (r θ z : ℝ), (r = c) ↔ ∃ (cylinder : ℝ), cylinder = r ∧ cylinder = c :=
by
  sorry

end shape_of_r_eq_c_in_cylindrical_coords_l301_301477


namespace total_balls_l301_301674

theorem total_balls (black_balls : ℕ) (prob_pick_black : ℚ) (total_balls : ℕ) :
  black_balls = 4 → prob_pick_black = 1 / 3 → total_balls = 12 :=
by
  intros h1 h2
  sorry

end total_balls_l301_301674


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301835

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301835


namespace arithmetic_seq_20th_term_l301_301643

variable (a : ℕ → ℤ) -- a_n is an arithmetic sequence
variable (d : ℤ) -- common difference of the arithmetic sequence

-- Condition for arithmetic sequence
variable (h_seq : ∀ n, a (n+1) = a n + d)

-- Given conditions
axiom h1 : a 1 + a 3 + a 5 = 105
axiom h2 : a 2 + a 4 + a 6 = 99

-- Goal: prove that a 20 = 1
theorem arithmetic_seq_20th_term :
  a 20 = 1 :=
sorry

end arithmetic_seq_20th_term_l301_301643


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301747

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301747


namespace pencils_per_box_l301_301329

theorem pencils_per_box (total_pencils : ℝ) (num_boxes : ℝ) (pencils_per_box : ℝ) 
  (h1 : total_pencils = 2592) 
  (h2 : num_boxes = 4.0) 
  (h3 : pencils_per_box = total_pencils / num_boxes) : 
  pencils_per_box = 648 :=
by
  sorry

end pencils_per_box_l301_301329


namespace orchard_harvest_l301_301235

theorem orchard_harvest (sacks_per_section : ℕ) (sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 → sections = 8 → total_sacks = sacks_per_section * sections → total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end orchard_harvest_l301_301235


namespace tan_subtraction_simplify_l301_301543

theorem tan_subtraction_simplify :
  tan (Real.pi / 12) - tan (5 * Real.pi / 12) = -4 * Real.sqrt 3 := by
sorry

end tan_subtraction_simplify_l301_301543


namespace probability_complement_B_probability_union_A_B_l301_301929

variable (Ω : Type) [MeasurableSpace Ω] {P : ProbabilityMeasure Ω}
variable (A B : Set Ω)

theorem probability_complement_B
  (hB : P B = 1 / 3) : P Bᶜ = 2 / 3 :=
by
  sorry

theorem probability_union_A_B
  (hA : P A = 1 / 2) (hB : P B = 1 / 3) : P (A ∪ B) ≤ 5 / 6 :=
by
  sorry

end probability_complement_B_probability_union_A_B_l301_301929


namespace triple_composition_g_eq_107_l301_301177

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_g_eq_107 : g(g(g(3))) = 107 := by
  sorry

end triple_composition_g_eq_107_l301_301177


namespace quadratic_one_solution_l301_301901

theorem quadratic_one_solution (p : ℝ) : (3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0) 
  → ((-6) ^ 2 - 4 * 3 * p = 0) 
  → p = 3 :=
by
  intro h1 h2
  have h1' : 3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0 := h1
  have h2' : (-6) ^ 2 - 4 * 3 * p = 0 := h2
  sorry

end quadratic_one_solution_l301_301901


namespace total_insects_eaten_l301_301271

theorem total_insects_eaten :
  let geckos := 5
  let insects_per_gecko := 6
  let lizards := 3
  let insects_per_lizard := 2 * insects_per_gecko
  let total_insects := geckos * insects_per_gecko + lizards * insects_per_lizard
  total_insects = 66 := by
  sorry

end total_insects_eaten_l301_301271


namespace least_number_condition_l301_301109

-- Define the set of divisors as a constant
def divisors : Set ℕ := {1, 2, 3, 4, 5, 6, 8, 15}

-- Define the least number that satisfies the condition
def least_number : ℕ := 125

-- The theorem stating that the least number 125 leaves a remainder of 5 when divided by the given set of numbers
theorem least_number_condition : ∀ d ∈ divisors, least_number % d = 5 :=
by
  sorry

end least_number_condition_l301_301109


namespace consecutive_even_number_difference_l301_301508

theorem consecutive_even_number_difference (x : ℤ) (h : x^2 - (x - 2)^2 = 2012) : x = 504 :=
sorry

end consecutive_even_number_difference_l301_301508


namespace factorize_polynomial_l301_301024
   
   -- Define the polynomial
   def polynomial (x : ℝ) : ℝ :=
     x^3 + 3 * x^2 - 4
   
   -- Define the factorized form
   def factorized_form (x : ℝ) : ℝ :=
     (x - 1) * (x + 2)^2
   
   -- The theorem statement
   theorem factorize_polynomial (x : ℝ) : polynomial x = factorized_form x := 
   by
     sorry
   
end factorize_polynomial_l301_301024


namespace no_equal_black_white_shards_l301_301727

def initial_glass_cups := 25
def initial_porcelain_cups := 35
def glass_pieces := 17
def porcelain_pieces := 18

theorem no_equal_black_white_shards (x y : ℕ) :
  (glass_pieces * x + porcelain_pieces * (initial_porcelain_cups - y)
   = glass_pieces * (initial_glass_cups - x) + porcelain_pieces * y) →
  false :=
by sorry

end no_equal_black_white_shards_l301_301727


namespace largest_integer_base_7_l301_301200

theorem largest_integer_base_7 :
  let M := 66 in
  M ^ 2 = 48 ^ 2 :=
by
  let M := (6 * 7 + 6) in
  have h : M ^ 2 = 48 ^ 2 := rfl
  sorry -- Proof not required.

end largest_integer_base_7_l301_301200


namespace bike_price_l301_301994

theorem bike_price (x : ℝ) (h1 : 0.1 * x = 150) : x = 1500 := 
by sorry

end bike_price_l301_301994


namespace calculate_f_17_69_l301_301717

noncomputable def f (x y: ℕ) : ℚ := sorry

axiom f_self : ∀ x, f x x = x
axiom f_symm : ∀ x y, f x y = f y x
axiom f_add : ∀ x y, (x + y) * f x y = y * f x (x + y)

theorem calculate_f_17_69 : f 17 69 = 73.3125 := sorry

end calculate_f_17_69_l301_301717


namespace perfume_price_reduction_l301_301285

theorem perfume_price_reduction : 
  let original_price := 1200
  let increased_price := original_price * (1 + 0.10)
  let final_price := increased_price * (1 - 0.15)
  original_price - final_price = 78 := 
by
  sorry

end perfume_price_reduction_l301_301285


namespace sqrt_cosine_identity_l301_301302

theorem sqrt_cosine_identity :
  Real.sqrt ((3 - Real.cos (Real.pi / 8)^2) * (3 - Real.cos (3 * Real.pi / 8)^2)) = (3 * Real.sqrt 5) / 4 :=
by
  sorry

end sqrt_cosine_identity_l301_301302


namespace div_by_prime_power_l301_301082

theorem div_by_prime_power (p α x : ℕ) (hp : Nat.Prime p) (hpg : p > 2) (hα : α > 0) (t : ℤ) :
  (∃ k : ℤ, x^2 - 1 = k * p^α) ↔ (∃ t : ℤ, x = t * p^α + 1 ∨ x = t * p^α - 1) :=
sorry

end div_by_prime_power_l301_301082


namespace prime_sum_remainder_l301_301775

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301775


namespace pearls_problem_l301_301866

theorem pearls_problem :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 7 = 5) ∧ (n = 54) ∧ (n % 9 = 0) :=
by sorry

end pearls_problem_l301_301866


namespace vector_on_line_l301_301282

noncomputable def k_value (a b : Vector ℝ 3) (m : ℝ) : ℝ :=
  if h : m = 5 / 7 then
    (5 / 7 : ℝ)
  else
    0 -- This branch will never be taken because we will assume m = 5 / 7 as a hypothesis.


theorem vector_on_line (a b : Vector ℝ 3) (m k : ℝ) (h : m = 5 / 7) :
  k = k_value a b m :=
by
  sorry

end vector_on_line_l301_301282


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301837

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301837


namespace net_percentage_gain_approx_l301_301289

noncomputable def netPercentageGain : ℝ :=
  let costGlassBowls := 250 * 18
  let costCeramicPlates := 150 * 25
  let totalCostBeforeDiscount := costGlassBowls + costCeramicPlates
  let discount := 0.05 * totalCostBeforeDiscount
  let totalCostAfterDiscount := totalCostBeforeDiscount - discount
  let revenueGlassBowls := 200 * 25
  let revenueCeramicPlates := 120 * 32
  let totalRevenue := revenueGlassBowls + revenueCeramicPlates
  let costBrokenGlassBowls := 30 * 18
  let costBrokenCeramicPlates := 10 * 25
  let totalCostBrokenItems := costBrokenGlassBowls + costBrokenCeramicPlates
  let netGain := totalRevenue - (totalCostAfterDiscount + totalCostBrokenItems)
  let netPercentageGain := (netGain / totalCostAfterDiscount) * 100
  netPercentageGain

theorem net_percentage_gain_approx :
  abs (netPercentageGain - 2.71) < 0.01 := sorry

end net_percentage_gain_approx_l301_301289


namespace worm_length_l301_301224

theorem worm_length (l1 l2 : ℝ) (h1 : l1 = 0.8) (h2 : l2 = l1 + 0.7) : l1 = 0.8 :=
by
  exact h1

end worm_length_l301_301224


namespace find_C_plus_D_l301_301722

noncomputable def polynomial_divisible (x : ℝ) (C : ℝ) (D : ℝ) : Prop := 
  ∃ (ω : ℝ), ω^2 + ω + 1 = 0 ∧ ω^104 + C*ω + D = 0

theorem find_C_plus_D (C D : ℝ) : 
  (∃ x : ℝ, polynomial_divisible x C D) → C + D = 2 :=
by
  sorry

end find_C_plus_D_l301_301722


namespace train_travel_time_change_l301_301878

theorem train_travel_time_change 
  (t1 t2 : ℕ) (s1 s2 d : ℕ) 
  (h1 : t1 = 4) 
  (h2 : s1 = 50) 
  (h3 : s2 = 100) 
  (h4 : d = t1 * s1) :
  t2 = d / s2 → t2 = 2 :=
by
  intros
  sorry

end train_travel_time_change_l301_301878


namespace infinite_polynomial_pairs_l301_301083

open Polynomial

theorem infinite_polynomial_pairs :
  ∀ n : ℕ, ∃ (fn gn : ℤ[X]), fn^2 - (X^4 - 2 * X) * gn^2 = 1 :=
sorry

end infinite_polynomial_pairs_l301_301083


namespace fraction_equality_l301_301907

theorem fraction_equality (a b : ℝ) (h : (1 / a) - (1 / b) = 4) :
  (a - 2 * a * b - b) / (2 * a - 2 * b + 7 * a * b) = 6 :=
by
  sorry

end fraction_equality_l301_301907


namespace num_socks_in_machine_l301_301414

-- Definition of the number of people who played the match
def num_players : ℕ := 11

-- Definition of the number of socks per player
def socks_per_player : ℕ := 2

-- The goal is to prove that the total number of socks in the washing machine is 22
theorem num_socks_in_machine : num_players * socks_per_player = 22 :=
by
  sorry

end num_socks_in_machine_l301_301414


namespace num_positive_k_for_solution_to_kx_minus_18_eq_3k_l301_301473

theorem num_positive_k_for_solution_to_kx_minus_18_eq_3k : 
  ∃ (k_vals : Finset ℕ), 
  (∀ k ∈ k_vals, ∃ x : ℤ, k * x - 18 = 3 * k) ∧ 
  k_vals.card = 6 :=
by
  sorry

end num_positive_k_for_solution_to_kx_minus_18_eq_3k_l301_301473


namespace repeating_decimal_sum_l301_301149

theorem repeating_decimal_sum :
  (0.6666666666 : ℝ) + (0.7777777777 : ℝ) = (13 : ℚ) / 9 := by
  sorry

end repeating_decimal_sum_l301_301149


namespace count_color_patterns_l301_301698

def regions := 6
def colors := 3

theorem count_color_patterns (h1 : regions = 6) (h2 : colors = 3) :
  3^6 - 3 * 2^6 + 3 * 1^6 = 540 := by
  sorry

end count_color_patterns_l301_301698


namespace jack_jill_same_speed_l301_301067

theorem jack_jill_same_speed (x : ℝ) (h : x^2 - 8*x - 10 = 0) :
  (x^2 - 7*x - 18) = 2 := 
sorry

end jack_jill_same_speed_l301_301067


namespace orthocenter_of_ABC_is_correct_l301_301628

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

def A : Point3D := {x := 2, y := 3, z := -1}
def B : Point3D := {x := 6, y := -1, z := 2}
def C : Point3D := {x := 4, y := 5, z := 4}

def orthocenter (A B C : Point3D) : Point3D := {
  x := 101 / 33,
  y := 95 / 33,
  z := 47 / 33
}

theorem orthocenter_of_ABC_is_correct : orthocenter A B C = {x := 101 / 33, y := 95 / 33, z := 47 / 33} :=
  sorry

end orthocenter_of_ABC_is_correct_l301_301628


namespace ellipse_condition_necessary_not_sufficient_l301_301861

theorem ellipse_condition_necessary_not_sufficient {a b : ℝ} (h : a * b > 0):
  (∀ x y : ℝ, a * x^2 + b * y^2 = 1 → a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0) ∧ 
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) → a * b > 0) :=
sorry

end ellipse_condition_necessary_not_sufficient_l301_301861


namespace Tim_bodyguards_weekly_pay_l301_301386

theorem Tim_bodyguards_weekly_pay :
  let hourly_rate := 20
  let num_bodyguards := 2
  let daily_hours := 8
  let weekly_days := 7
  Tim pays $2240 in a week := (hourly_rate * num_bodyguards * daily_hours * weekly_days = 2240) :=
begin
  sorry
end

end Tim_bodyguards_weekly_pay_l301_301386


namespace radio_show_length_l301_301620

theorem radio_show_length :
  let s3 := 10
  let s2 := s3 + 5
  let s4 := s2 / 2
  let s5 := 2 * s4
  let s1 := 2 * (s2 + s3 + s4 + s5)
  s1 + s2 + s3 + s4 + s5 = 142.5 :=
by
  sorry

end radio_show_length_l301_301620


namespace ellipse_range_k_l301_301497

theorem ellipse_range_k (k : ℝ) (h1 : 3 + k > 0) (h2 : 2 - k > 0) (h3 : k ≠ -1 / 2) :
  k ∈ Set.Ioo (-3 : ℝ) (-1 / 2) ∪ Set.Ioo (-1 / 2) (2 : ℝ) :=
sorry

end ellipse_range_k_l301_301497


namespace first_set_cost_l301_301404

theorem first_set_cost (F S : ℕ) (hS : S = 50) (h_equation : 2 * F + 3 * S = 220) 
: 3 * F + S = 155 := 
sorry

end first_set_cost_l301_301404


namespace number_of_tens_in_sum_l301_301945

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l301_301945


namespace vectors_parallel_x_squared_eq_two_l301_301328

theorem vectors_parallel_x_squared_eq_two (x : ℝ) 
  (a : ℝ × ℝ := (x+2, 1+x)) 
  (b : ℝ × ℝ := (x-2, 1-x)) 
  (parallel : (a.1 * b.2 - a.2 * b.1) = 0) : x^2 = 2 :=
sorry

end vectors_parallel_x_squared_eq_two_l301_301328


namespace cost_of_song_book_l301_301069

-- Define the given constants: cost of trumpet, cost of music tool, and total spent at the music store.
def cost_of_trumpet : ℝ := 149.16
def cost_of_music_tool : ℝ := 9.98
def total_spent_at_store : ℝ := 163.28

-- The goal is to prove that the cost of the song book is $4.14.
theorem cost_of_song_book :
  total_spent_at_store - (cost_of_trumpet + cost_of_music_tool) = 4.14 :=
by
  sorry

end cost_of_song_book_l301_301069


namespace round_robin_games_l301_301103

theorem round_robin_games (x : ℕ) (h : ∃ (n : ℕ), n = 15) : (x * (x - 1)) / 2 = 15 :=
sorry

end round_robin_games_l301_301103


namespace arc_length_solution_l301_301297

open Set Filter Real

noncomputable def arc_length_parabola (x_0 : ℝ) : ℝ :=
  ∫ x in (0 : ℝ)..x_0, sqrt(1 + x^2)

theorem arc_length_solution (x_0 : ℝ) : 
  arc_length_parabola x_0 = (x_0 * sqrt(1 + x_0^2) / 2) + (1 / 2) * ln (x_0 + sqrt(1 + x_0^2)) :=
by
  sorry

end arc_length_solution_l301_301297


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301838

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301838


namespace bacteria_growth_time_l301_301091

-- Define the conditions and the final proof statement
theorem bacteria_growth_time (n0 n1 : ℕ) (t : ℕ) :
  (∀ (k : ℕ), k > 0 → n1 = n0 * 3 ^ k) →
  (∀ (h : ℕ), t = 5 * h) →
  n0 = 200 →
  n1 = 145800 →
  t = 30 :=
by
  sorry

end bacteria_growth_time_l301_301091


namespace average_price_of_pig_l301_301601

theorem average_price_of_pig :
  ∀ (total_cost : ℕ) (num_pigs num_hens : ℕ) (avg_hen_price avg_pig_price : ℕ),
    total_cost = 2100 →
    num_pigs = 5 →
    num_hens = 15 →
    avg_hen_price = 30 →
    avg_pig_price * num_pigs + avg_hen_price * num_hens = total_cost →
    avg_pig_price = 330 :=
by
  intros total_cost num_pigs num_hens avg_hen_price avg_pig_price
  intros h_total_cost h_num_pigs h_num_hens h_avg_hen_price h_eq
  rw [h_total_cost, h_num_pigs, h_num_hens, h_avg_hen_price] at h_eq
  sorry

end average_price_of_pig_l301_301601


namespace truthful_dwarfs_count_l301_301425

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l301_301425


namespace sqrt_neg9_squared_l301_301266

theorem sqrt_neg9_squared : Real.sqrt ((-9: ℝ)^2) = 9 := by
  sorry

end sqrt_neg9_squared_l301_301266


namespace problem_solution_l301_301718

noncomputable def expr := 
  (Real.tan (Real.pi / 15) - Real.sqrt 3) / ((4 * (Real.cos (Real.pi / 15))^2 - 2) * Real.sin (Real.pi / 15))

theorem problem_solution : expr = -4 :=
by
  sorry

end problem_solution_l301_301718


namespace water_cost_is_1_l301_301669

-- Define the conditions
def cost_cola : ℝ := 3
def cost_juice : ℝ := 1.5
def bottles_sold_cola : ℝ := 15
def bottles_sold_juice : ℝ := 12
def bottles_sold_water : ℝ := 25
def total_revenue : ℝ := 88

-- Compute the revenue from cola and juice
def revenue_cola : ℝ := bottles_sold_cola * cost_cola
def revenue_juice : ℝ := bottles_sold_juice * cost_juice

-- Define a proof that the cost of a bottle of water is $1
theorem water_cost_is_1 : (total_revenue - revenue_cola - revenue_juice) / bottles_sold_water = 1 :=
by
  -- Proof is omitted
  sorry

end water_cost_is_1_l301_301669


namespace num_terms_100_pow_10_as_sum_of_tens_l301_301960

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l301_301960


namespace dwarfs_truthful_count_l301_301431

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l301_301431


namespace marble_probability_l301_301602

theorem marble_probability :
  let red_marbles := 6
      white_marbles := 4
      blue_marbles := 8
      total_marbles := red_marbles + white_marbles + blue_marbles
      draw_count := 3 in
  (nat.choose red_marbles draw_count + nat.choose white_marbles draw_count + nat.choose blue_marbles draw_count) /
  (nat.choose total_marbles draw_count : ℚ) = 5 / 51 :=
by repeat { sorry }

end marble_probability_l301_301602


namespace difference_of_squares_l301_301963

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : |x^2 - y^2| = 108 :=
  sorry

end difference_of_squares_l301_301963


namespace Tim_pays_correct_amount_l301_301391

def pays_in_a_week (hourly_rate : ℕ) (num_bodyguards : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * num_bodyguards * hours_per_day * days_per_week

theorem Tim_pays_correct_amount :
  pays_in_a_week 20 2 8 7 = 2240 := by
  sorry

end Tim_pays_correct_amount_l301_301391


namespace find_power_of_7_l301_301474

theorem find_power_of_7 :
  (7^(1/4)) / (7^(1/6)) = 7^(1/12) :=
by
  sorry

end find_power_of_7_l301_301474


namespace yen_to_usd_conversion_l301_301617

theorem yen_to_usd_conversion
  (cost_of_souvenir : ℕ)
  (service_charge : ℕ)
  (conversion_rate : ℕ)
  (total_cost_in_yen : ℕ)
  (usd_equivalent : ℚ)
  (h1 : cost_of_souvenir = 340)
  (h2 : service_charge = 25)
  (h3 : conversion_rate = 115)
  (h4 : total_cost_in_yen = cost_of_souvenir + service_charge)
  (h5 : usd_equivalent = (total_cost_in_yen : ℚ) / conversion_rate) :
  total_cost_in_yen = 365 ∧ usd_equivalent = 3.17 :=
by
  sorry

end yen_to_usd_conversion_l301_301617


namespace dive_has_five_judges_l301_301055

noncomputable def number_of_judges 
  (scores : List ℝ)
  (difficulty : ℝ)
  (point_value : ℝ) : ℕ := sorry

theorem dive_has_five_judges :
  number_of_judges [7.5, 8.0, 9.0, 6.0, 8.8] 3.2 77.76 = 5 :=
by
  sorry

end dive_has_five_judges_l301_301055


namespace remainder_first_six_primes_div_seventh_l301_301844

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301844


namespace apple_picking_ratio_l301_301526

theorem apple_picking_ratio (a b c : ℕ) 
  (h1 : a = 66) 
  (h2 : b = 2 * 66) 
  (h3 : a + b + c = 220) :
  c = 22 → a = 66 → c / a = 1 / 3 := by
    intros
    sorry

end apple_picking_ratio_l301_301526


namespace total_cost_correct_l301_301420

-- Definitions of the constants based on given problem conditions
def cost_burger : ℕ := 5
def cost_pack_of_fries : ℕ := 2
def num_packs_of_fries : ℕ := 2
def cost_salad : ℕ := 3 * cost_pack_of_fries

-- The total cost calculation based on the conditions
def total_cost : ℕ := cost_burger + num_packs_of_fries * cost_pack_of_fries + cost_salad

-- The statement to prove that the total cost Benjamin paid is $15
theorem total_cost_correct : total_cost = 15 := by
  -- This is where the proof would go, but we're omitting it for now.
  sorry

end total_cost_correct_l301_301420


namespace union_of_A_and_B_l301_301188

open Set

variable (A B : Set ℤ)

theorem union_of_A_and_B (hA : A = {0, 1}) (hB : B = {0, -1}) : A ∪ B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l301_301188


namespace travel_time_correct_l301_301587

noncomputable def timeSpentOnRoad : Nat :=
  let startTime := 7  -- 7:00 AM in hours
  let endTime := 20   -- 8:00 PM in hours
  let totalJourneyTime := endTime - startTime
  let stopTimes := [25, 10, 25]  -- minutes
  let totalStopTime := stopTimes.foldl (· + ·) 0
  let stopTimeInHours := totalStopTime / 60
  totalJourneyTime - stopTimeInHours

theorem travel_time_correct : timeSpentOnRoad = 12 :=
by
  sorry

end travel_time_correct_l301_301587


namespace max_dominoes_l301_301989

theorem max_dominoes (m n : ℕ) (h : n ≥ m) :
  ∃ k, k = m * n - (m / 2 : ℕ) :=
by sorry

end max_dominoes_l301_301989


namespace dwarfs_truthful_count_l301_301432

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l301_301432


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301738

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301738


namespace index_cards_per_pack_l301_301893

-- Definitions of the conditions
def students_per_period := 30
def periods_per_day := 6
def index_cards_per_student := 10
def total_spent := 108
def pack_cost := 3

-- Helper Definitions
def total_students := periods_per_day * students_per_period
def total_index_cards_needed := total_students * index_cards_per_student
def packs_bought := total_spent / pack_cost

-- Theorem to prove
theorem index_cards_per_pack :
  total_index_cards_needed / packs_bought = 50 := by
  sorry

end index_cards_per_pack_l301_301893


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301785

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301785


namespace triple_composition_g_eq_107_l301_301178

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_g_eq_107 : g(g(g(3))) = 107 := by
  sorry

end triple_composition_g_eq_107_l301_301178


namespace jessica_current_age_l301_301977

-- Definitions and conditions from the problem
def J (M : ℕ) : ℕ := M / 2
def M : ℕ := 60

-- Lean statement for the proof problem
theorem jessica_current_age : J M + 10 = 40 :=
by
  sorry

end jessica_current_age_l301_301977


namespace battery_lasts_12_more_hours_l301_301982

-- Define initial conditions
def standby_battery_life : ℕ := 36
def active_battery_life : ℕ := 4
def total_time_on : ℕ := 12
def active_usage_time : ℕ := 90  -- in minutes

-- Conversion and calculation functions
def active_usage_hours : ℚ := active_usage_time / 60
def standby_consumption_rate : ℚ := 1 / standby_battery_life
def active_consumption_rate : ℚ := 1 / active_battery_life
def battery_used_standby : ℚ := (total_time_on - active_usage_hours) * standby_consumption_rate
def battery_used_active : ℚ := active_usage_hours * active_consumption_rate
def total_battery_used : ℚ := battery_used_standby + battery_used_active
def remaining_battery : ℚ := 1 - total_battery_used
def additional_hours_standby : ℚ := remaining_battery / standby_consumption_rate

-- Proof statement
theorem battery_lasts_12_more_hours : additional_hours_standby = 12 := by
  sorry

end battery_lasts_12_more_hours_l301_301982


namespace average_speed_is_correct_l301_301294

-- Define the conditions
def initial_odometer : ℕ := 2552
def final_odometer : ℕ := 2882
def time_first_day : ℕ := 5
def time_second_day : ℕ := 7

-- Calculate total time and distance
def total_time : ℕ := time_first_day + time_second_day
def total_distance : ℕ := final_odometer - initial_odometer

-- Prove that the average speed is 27.5 miles per hour
theorem average_speed_is_correct : (total_distance : ℚ) / (total_time : ℚ) = 27.5 :=
by
  sorry

end average_speed_is_correct_l301_301294


namespace fraction_of_students_with_buddy_l301_301340

theorem fraction_of_students_with_buddy (s n : ℕ) (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s : ℚ) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l301_301340


namespace angle_between_lines_l301_301712

theorem angle_between_lines :
  let L1 := {p : ℝ × ℝ | p.1 = -3}  -- Line x+3=0
  let L2 := {p: ℝ × ℝ | p.1 + p.2 - 3 = 0}  -- Line x+y-3=0
  ∃ θ : ℝ, 0 < θ ∧ θ < 180 ∧ θ = 45 :=
sorry

end angle_between_lines_l301_301712


namespace boat_distance_downstream_l301_301011

theorem boat_distance_downstream (speed_boat_still: ℕ) (speed_stream: ℕ) (time: ℕ)
    (h1: speed_boat_still = 25)
    (h2: speed_stream = 5)
    (h3: time = 4) :
    (speed_boat_still + speed_stream) * time = 120 := 
sorry

end boat_distance_downstream_l301_301011


namespace will_new_cards_count_l301_301120

-- Definitions based on conditions
def cards_per_page := 3
def pages_used := 6
def old_cards := 10

-- Proof statement (no proof, only the statement)
theorem will_new_cards_count : (pages_used * cards_per_page) - old_cards = 8 :=
by sorry

end will_new_cards_count_l301_301120


namespace derivative_sqrt_l301_301376

/-- The derivative of the function y = sqrt x is 1 / (2 * sqrt x) -/
theorem derivative_sqrt (x : ℝ) (h : 0 < x) : (deriv (fun x => Real.sqrt x) x) = 1 / (2 * Real.sqrt x) :=
sorry

end derivative_sqrt_l301_301376


namespace five_equal_angles_72_degrees_l301_301476

theorem five_equal_angles_72_degrees
  (five_rays : ℝ)
  (equal_angles : ℝ) 
  (sum_angles : five_rays * equal_angles = 360) :
  equal_angles = 72 :=
by
  sorry

end five_equal_angles_72_degrees_l301_301476


namespace min_value_x_plus_one_over_2x_l301_301910

theorem min_value_x_plus_one_over_2x (x : ℝ) (hx : x > 0) : 
  x + 1 / (2 * x) ≥ Real.sqrt 2 := sorry

end min_value_x_plus_one_over_2x_l301_301910


namespace possible_rectangle_areas_l301_301351

def is_valid_pair (a b : ℕ) := 
  a + b = 12 ∧ a > 0 ∧ b > 0

def rectangle_area (a b : ℕ) := a * b

theorem possible_rectangle_areas :
  {area | ∃ (a b : ℕ), is_valid_pair a b ∧ area = rectangle_area a b} 
  = {11, 20, 27, 32, 35, 36} := 
by 
  sorry

end possible_rectangle_areas_l301_301351


namespace remainder_of_sum_of_primes_l301_301828

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301828


namespace sqrt10_solution_l301_301039

theorem sqrt10_solution (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : (1/a) + (1/b) = 2) :
  m = Real.sqrt 10 :=
sorry

end sqrt10_solution_l301_301039


namespace simplify_polynomial_l301_301108

theorem simplify_polynomial (x : ℝ) : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2) = (-x^2 + 23 * x - 3) := 
by
  sorry

end simplify_polynomial_l301_301108


namespace sufficient_but_not_necessary_condition_l301_301116

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (x^2 - m * x + 1) = 0 → (m^2 - 4 < 0) = ∀ m : ℝ, -2 < m ∧ m < 2 :=
sufficient_but_not_necessary_condition sorry

end sufficient_but_not_necessary_condition_l301_301116


namespace sin_diff_l301_301634

theorem sin_diff (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1 / 3) 
  (h2 : Real.sin β - Real.cos α = 1 / 2) : 
  Real.sin (α - β) = -59 / 72 := 
sorry

end sin_diff_l301_301634


namespace num_terms_100_pow_10_as_sum_of_tens_l301_301959

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l301_301959


namespace gg_of_3_is_107_l301_301179

-- Define the function g
def g (x : ℕ) : ℕ := 3 * x + 2

-- State that g(g(g(3))) equals 107
theorem gg_of_3_is_107 : g (g (g 3)) = 107 := by
  sorry

end gg_of_3_is_107_l301_301179


namespace least_possible_value_of_b_l301_301369

theorem least_possible_value_of_b (a b : ℕ) 
  (ha : ∃ p, (∀ q, p ∣ q ↔ q = 1 ∨ q = p ∨ q = p*p ∨ q = a))
  (hb : ∃ k, (∀ l, k ∣ l ↔ (l = 1 ∨ l = b)))
  (hdiv : a ∣ b) : 
  b = 12 :=
sorry

end least_possible_value_of_b_l301_301369


namespace arithmetic_sequence_problem_l301_301482

-- Define sequence and sum properties
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

/- Theorem Statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) 
  (h_seq : arithmetic_sequence a d) 
  (h_initial : a 1 = 31) 
  (h_S_eq : S 10 = S 22) :
  -- Part 1: Find S_n
  (∀ n, S n = 32 * n - n ^ 2) ∧
  -- Part 2: Maximum sum occurs at n = 16 and is 256
  (∀ n, S n ≤ 256 ∧ (S 16 = 256 → ∀ m ≠ 16, S m < 256)) :=
by
  -- proof to be provided here
  sorry

end arithmetic_sequence_problem_l301_301482


namespace pairwise_sums_modulo_l301_301478

theorem pairwise_sums_modulo (n : ℕ) (h : n = 2011) :
  ∃ (sums_div_3 sums_rem_1 : ℕ),
  (sums_div_3 = (n * (n - 1)) / 6) ∧
  (sums_rem_1 = (n * (n - 1)) / 6) := by
  sorry

end pairwise_sums_modulo_l301_301478


namespace garden_area_l301_301090

-- Given that the garden is a square with certain properties
variables (s A P : ℕ)

-- Conditions:
-- The perimeter of the square garden is 28 feet
def perimeter_condition : Prop := P = 28

-- The area of the garden is equal to the perimeter plus 21
def area_condition : Prop := A = P + 21

-- The perimeter of a square garden with side length s
def perimeter_def : Prop := P = 4 * s

-- The area of a square garden with side length s
def area_def : Prop := A = s * s

-- Prove that the area A is 49 square feet
theorem garden_area : perimeter_condition P → area_condition P A → perimeter_def s P → area_def s A → A = 49 :=
by 
  sorry

end garden_area_l301_301090


namespace initial_distance_between_stations_l301_301571

theorem initial_distance_between_stations
  (speedA speedB distanceA : ℝ)
  (rateA rateB : speedA = 40 ∧ speedB = 30)
  (dist_travelled : distanceA = 200) :
  (distanceA / speedA) * speedB + distanceA = 350 := by
  sorry

end initial_distance_between_stations_l301_301571


namespace trapezoid_area_l301_301191

-- Geometry setup
variable (outer_area : ℝ) (inner_height_ratio : ℝ)

-- Conditions
def outer_triangle_area := outer_area = 36
def inner_height_to_outer_height := inner_height_ratio = 2 / 3

-- Conclusion: Area of one trapezoid
theorem trapezoid_area (outer_area inner_height_ratio : ℝ) 
  (h_outer : outer_triangle_area outer_area) 
  (h_inner : inner_height_to_outer_height inner_height_ratio) : 
  (outer_area - 16 * Real.sqrt 3) / 3 = (36 - 16 * Real.sqrt 3) / 3 := 
sorry

end trapezoid_area_l301_301191


namespace near_square_qoutient_l301_301107

def is_near_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem near_square_qoutient (n : ℕ) (hn : is_near_square n) : 
  ∃ a b : ℕ, is_near_square a ∧ is_near_square b ∧ n = a / b := 
sorry

end near_square_qoutient_l301_301107


namespace remainder_of_sum_of_primes_l301_301821

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301821


namespace problem_l301_301635

theorem problem (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - x) = x^2 + 1) : f (-1) = 5 := 
  sorry

end problem_l301_301635


namespace perfume_price_reduction_l301_301286

theorem perfume_price_reduction : 
  let original_price := 1200
  let increased_price := original_price * (1 + 0.10)
  let final_price := increased_price * (1 - 0.15)
  original_price - final_price = 78 := 
by
  sorry

end perfume_price_reduction_l301_301286


namespace m_coins_can_collect_k_rubles_l301_301400

theorem m_coins_can_collect_k_rubles
  (a1 a2 a3 a4 a5 a6 a7 m k : ℕ)
  (h1 : a1 + 2 * a2 + 5 * a3 + 10 * a4 + 20 * a5 + 50 * a6 + 100 * a7 = m)
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = k) :
  ∃ (b1 b2 b3 b4 b5 b6 b7 : ℕ), 
    100 * (b1 + 2 * b2 + 5 * b3 + 10 * b4 + 20 * b5 + 50 * b6 + 100 * b7) = 100 * k ∧ 
    b1 + b2 + b3 + b4 + b5 + b6 + b7 = m := 
sorry

end m_coins_can_collect_k_rubles_l301_301400


namespace expand_and_simplify_l301_301471

theorem expand_and_simplify (x y : ℝ) : 
  (x + 6) * (x + 8 + y) = x^2 + 14 * x + x * y + 48 + 6 * y :=
by sorry

end expand_and_simplify_l301_301471


namespace truthful_dwarfs_count_l301_301429

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l301_301429


namespace number_of_truthful_dwarfs_l301_301453

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l301_301453


namespace number_of_tens_in_sum_l301_301954

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l301_301954


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301771

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301771


namespace no_third_quadrant_l301_301172

def quadratic_no_real_roots (b : ℝ) : Prop :=
  16 - 4 * b < 0

def passes_through_third_quadrant (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = -2 * x + b ∧ x < 0 ∧ y < 0

theorem no_third_quadrant (b : ℝ) (h : quadratic_no_real_roots b) : ¬ passes_through_third_quadrant b := 
by {
  sorry
}

end no_third_quadrant_l301_301172


namespace arcsin_sqrt_one_half_l301_301299

theorem arcsin_sqrt_one_half : Real.arcsin (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  -- TODO: provide proof
  sorry

end arcsin_sqrt_one_half_l301_301299


namespace correct_calculation_l301_301397

theorem correct_calculation (a b : ℝ) :
  ¬(a^2 + 2 * a^2 = 3 * a^4) ∧
  ¬(a^6 / a^3 = a^2) ∧
  ¬((a^2)^3 = a^5) ∧
  (ab)^2 = a^2 * b^2 := by
  sorry

end correct_calculation_l301_301397


namespace simplify_fraction_l301_301262

theorem simplify_fraction (a b : ℕ) (h1 : a = 252) (h2 : b = 248) :
  (1000 ^ 2 : ℤ) / ((a ^ 2 - b ^ 2) : ℤ) = 500 := by
  sorry

end simplify_fraction_l301_301262


namespace prime_sum_remainder_l301_301776

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301776


namespace wooden_block_even_blue_faces_l301_301015

theorem wooden_block_even_blue_faces :
  let length := 6
  let width := 6
  let height := 2
  let total_cubes := length * width * height
  let corners := 8
  let edges_not_corners := 24
  let faces_not_edges := 24
  let interior := 16
  let even_blue_faces := edges_not_corners + interior
  total_cubes = 72 →
  even_blue_faces = 40 :=
by
  sorry

end wooden_block_even_blue_faces_l301_301015


namespace count_solutions_l301_301168

theorem count_solutions : 
  (∃ (n : ℕ), ∀ (x : ℕ), (x + 17) % 43 = 71 % 43 ∧ x < 150 → n = 4) := 
sorry

end count_solutions_l301_301168


namespace find_common_difference_find_minimum_sum_minimum_sum_value_l301_301675

-- Defining the arithmetic sequence and its properties
def a (n : ℕ) (d : ℚ) := (-3 : ℚ) + n * d

-- Given conditions
def condition_1 : ℚ := -3
def condition_2 (d : ℚ) := 11 * a 4 d = 5 * a 7 d - 13
def common_difference : ℚ := 31 / 9

-- Sum of the first n terms of an arithmetic sequence
def S (n : ℕ) (d : ℚ) := n * (-3 + (n - 1) * d / 2)

-- Defining the necessary theorems
theorem find_common_difference (d : ℚ) : condition_2 d → d = common_difference := by
  sorry

theorem find_minimum_sum (n : ℕ) : S n common_difference ≥ S 2 common_difference := by
  sorry

theorem minimum_sum_value : S 2 common_difference = -23 / 9 := by
  sorry

end find_common_difference_find_minimum_sum_minimum_sum_value_l301_301675


namespace sum_arith_seq_elems_l301_301489

noncomputable def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem sum_arith_seq_elems (a d : ℝ) 
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 48) :
  arithmetic_seq a d 6 + arithmetic_seq a d 7 = 24 := 
by 
  sorry

end sum_arith_seq_elems_l301_301489


namespace william_time_on_road_l301_301582

-- Define departure and arrival times
def departure_time := 7 -- 7:00 AM
def arrival_time := 20 -- 8:00 PM in 24-hour format

-- Define stop times in minutes
def stop1 := 25
def stop2 := 10
def stop3 := 25

-- Define total journey time in hours
def total_travel_time := arrival_time - departure_time

-- Define total stop time in hours
def total_stop_time := (stop1 + stop2 + stop3) / 60

-- Define time spent on the road
def time_on_road := total_travel_time - total_stop_time

-- The theorem to prove
theorem william_time_on_road : time_on_road = 12 := by
  sorry

end william_time_on_road_l301_301582


namespace time_for_B_is_24_days_l301_301002

noncomputable def A_work : ℝ := (1 / 2) / (3 / 4)
noncomputable def B_work : ℝ := 1 -- assume B does 1 unit of work in 1 day
noncomputable def total_work : ℝ := (A_work + B_work) * 18

theorem time_for_B_is_24_days : 
  ((A_work + B_work) * 18) / B_work = 24 := by
  sorry

end time_for_B_is_24_days_l301_301002


namespace number_of_truthful_dwarfs_is_correct_l301_301455

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l301_301455


namespace part_1_part_2_l301_301316

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

end part_1_part_2_l301_301316


namespace g_g_g_3_equals_107_l301_301182

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_g_g_3_equals_107 : g (g (g 3)) = 107 := 
by 
  sorry

end g_g_g_3_equals_107_l301_301182


namespace percentage_to_pass_is_correct_l301_301615

-- Define the conditions
def marks_obtained : ℕ := 130
def marks_failed_by : ℕ := 14
def max_marks : ℕ := 400

-- Define the function to calculate the passing percentage
def passing_percentage (obtained : ℕ) (failed_by : ℕ) (max : ℕ) : ℚ :=
  ((obtained + failed_by : ℕ) / (max : ℚ)) * 100

-- Statement of the problem
theorem percentage_to_pass_is_correct :
  passing_percentage marks_obtained marks_failed_by max_marks = 36 := 
sorry

end percentage_to_pass_is_correct_l301_301615


namespace total_people_100_l301_301056

noncomputable def total_people (P : ℕ) : Prop :=
  (2 / 5 : ℚ) * P = 40 ∧ (1 / 4 : ℚ) * P ≤ P ∧ P ≥ 40 

theorem total_people_100 {P : ℕ} (h : total_people P) : P = 100 := 
by 
  sorry -- proof would go here

end total_people_100_l301_301056


namespace operations_on_S_l301_301650

def is_element_of_S (x : ℤ) : Prop :=
  x = 0 ∨ ∃ n : ℤ, x = 2 * n

theorem operations_on_S (a b : ℤ) (ha : is_element_of_S a) (hb : is_element_of_S b) :
  (is_element_of_S (a + b)) ∧
  (is_element_of_S (a - b)) ∧
  (is_element_of_S (a * b)) ∧
  (¬ is_element_of_S (a / b)) ∧
  (¬ is_element_of_S ((a + b) / 2)) :=
by
  sorry

end operations_on_S_l301_301650


namespace mike_pull_ups_per_week_l301_301538

theorem mike_pull_ups_per_week (pull_ups_per_entry entries_per_day days_per_week : ℕ)
  (h1 : pull_ups_per_entry = 2)
  (h2 : entries_per_day = 5)
  (h3 : days_per_week = 7)
  : pull_ups_per_entry * entries_per_day * days_per_week = 70 := 
by
  sorry

end mike_pull_ups_per_week_l301_301538


namespace number_of_truthful_dwarfs_is_correct_l301_301458

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l301_301458


namespace average_eq_16_l301_301102

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 12

theorem average_eq_16 (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : (x + y) / 2 = 16 := by
  sorry

end average_eq_16_l301_301102


namespace total_insects_eaten_l301_301272

theorem total_insects_eaten :
  let geckos := 5
  let insects_per_gecko := 6
  let lizards := 3
  let insects_per_lizard := 2 * insects_per_gecko
  let total_insects := geckos * insects_per_gecko + lizards * insects_per_lizard
  total_insects = 66 := by
  sorry

end total_insects_eaten_l301_301272


namespace knocks_to_knicks_l301_301186

variable (knicks knacks knocks : ℝ)

def knicks_eq_knacks : Prop := 
  8 * knicks = 3 * knacks

def knacks_eq_knocks : Prop := 
  4 * knacks = 5 * knocks

theorem knocks_to_knicks
  (h1 : knicks_eq_knacks knicks knacks)
  (h2 : knacks_eq_knocks knacks knocks) :
  20 * knocks = 320 / 15 * knicks :=
  sorry

end knocks_to_knicks_l301_301186


namespace dwarfs_truthful_count_l301_301437

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l301_301437


namespace number_of_truthful_dwarfs_is_correct_l301_301457

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l301_301457


namespace proportion_solution_l301_301046

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 :=
by
  sorry

end proportion_solution_l301_301046


namespace find_particular_number_l301_301175

theorem find_particular_number (x : ℤ) (h : x - 29 + 64 = 76) : x = 41 :=
by
  sorry

end find_particular_number_l301_301175


namespace cos_double_angle_l301_301658

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 1/4) : Real.cos (2 * theta) = -7/8 :=
by
  sorry

end cos_double_angle_l301_301658


namespace find_simple_interest_rate_l301_301889

theorem find_simple_interest_rate (P A T SI R : ℝ)
  (hP : P = 750)
  (hA : A = 1125)
  (hT : T = 5)
  (hSI : SI = A - P)
  (hSI_def : SI = (P * R * T) / 100) : R = 10 :=
by
  -- Proof would go here
  sorry

end find_simple_interest_rate_l301_301889


namespace point_P_coordinates_l301_301517

theorem point_P_coordinates :
  ∃ (x y : ℝ), (y = (x^3 - 10 * x + 3)) ∧ (x < 0) ∧ (3 * x^2 - 10 = 2) ∧ (x = -2 ∧ y = 15) := by
sorry

end point_P_coordinates_l301_301517


namespace nonagon_isosceles_triangle_count_l301_301481

theorem nonagon_isosceles_triangle_count (N : ℕ) (hN : N = 9) : 
  ∃(k : ℕ), k = 30 := 
by 
  have h := hN
  sorry      -- Solution steps would go here if we were proving it

end nonagon_isosceles_triangle_count_l301_301481


namespace find_initial_investment_l301_301996

open Real

noncomputable def initial_investment (x : ℝ) (years : ℕ) (final_value : ℝ) : ℝ := 
  final_value / (3 ^ (years / (112 / x)))

theorem find_initial_investment :
  let x := 8
  let years := 28
  let final_value := 31500
  initial_investment x years final_value = 3500 := 
by 
  sorry

end find_initial_investment_l301_301996


namespace movie_theatre_total_seats_l301_301408

theorem movie_theatre_total_seats (A C : ℕ) 
  (hC : C = 188) 
  (hRevenue : 6 * A + 4 * C = 1124) 
  : A + C = 250 :=
by
  sorry

end movie_theatre_total_seats_l301_301408


namespace numPythagoreanTriples_l301_301577

def isPythagoreanTriple (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ x^2 + y^2 = z^2

theorem numPythagoreanTriples (n : ℕ) : ∃! T : (ℕ × ℕ × ℕ) → Prop, 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (T (2^(n+1))) :=
sorry

end numPythagoreanTriples_l301_301577


namespace y_run_time_l301_301858

theorem y_run_time (t : ℕ) (h_avg : (t + 26) / 2 = 42) : t = 58 :=
by
  sorry

end y_run_time_l301_301858


namespace remainder_first_six_primes_div_seventh_l301_301849

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301849


namespace largest_possible_cylindrical_tank_radius_in_crate_l301_301122

theorem largest_possible_cylindrical_tank_radius_in_crate
  (crate_length : ℝ) (crate_width : ℝ) (crate_height : ℝ)
  (cylinder_height : ℝ) (cylinder_radius : ℝ)
  (h_cube : crate_length = 20 ∧ crate_width = 20 ∧ crate_height = 20)
  (h_cylinder_in_cube : cylinder_height = 20 ∧ 2 * cylinder_radius ≤ 20) :
  cylinder_radius = 10 :=
sorry

end largest_possible_cylindrical_tank_radius_in_crate_l301_301122


namespace Wilsons_number_l301_301001

theorem Wilsons_number (N : ℝ) (h : N - N / 3 = 16 / 3) : N = 8 := sorry

end Wilsons_number_l301_301001


namespace triangle_existence_l301_301044

theorem triangle_existence 
  (h_a h_b m_a : ℝ) :
  (m_a ≥ h_a) → 
  ((h_a > 1/2 * h_b ∧ m_a > h_a → true ∨ false) ∧ 
  (m_a = h_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b < m_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b = m_a → false ∨ true) ∧ 
  (1/2 * h_b > m_a → false)) :=
by
  intro
  sorry

end triangle_existence_l301_301044


namespace dwarfs_truthful_count_l301_301441

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l301_301441


namespace problem_statement_l301_301922

noncomputable def polar_to_cartesian_eq (ρ θ : ℝ) : Prop :=
  let x := ρ * cos θ
  let y := ρ * sin θ
  (x - 1)^2 + (y - 2)^2 = 5

theorem problem_statement (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ) :
  P = (0, 3) → 
  (∀ t, l(t) = (1/2 * t, 3 + sqrt 3 / 2 * t)) → 
  polar_to_cartesian_eq ρ θ →
  let d1 := (dist P A)
  let d2 := (dist P B)
  ∃ t1 t2 : ℝ, (l t1 = A) ∧ (l t2 = B) →
  ρ = 2 * cos θ + 4 * sin θ →
  (1 / d1 + 1 / d2 = sqrt (16 - 2 * sqrt 3) / 3) := by sorry

end problem_statement_l301_301922


namespace exists_sequence_of_ten_numbers_l301_301678

theorem exists_sequence_of_ten_numbers :
  ∃ a : Fin 10 → ℝ,
    (∀ i : Fin 6,    a i + a ⟨i.1 + 1, sorry⟩ + a ⟨i.1 + 2, sorry⟩ + a ⟨i.1 + 3, sorry⟩ + a ⟨i.1 + 4, sorry⟩ > 0) ∧
    (∀ j : Fin 4, a j + a ⟨j.1 + 1, sorry⟩ + a ⟨j.1 + 2, sorry⟩ + a ⟨j.1 + 3, sorry⟩ + a ⟨j.1 + 4, sorry⟩ + a ⟨j.1 + 5, sorry⟩ + a ⟨j.1 + 6, sorry⟩ < 0) :=
sorry

end exists_sequence_of_ten_numbers_l301_301678


namespace increase_speed_to_pass_correctly_l301_301058

theorem increase_speed_to_pass_correctly
  (x a : ℝ)
  (ha1 : 50 < a)
  (hx1 : (a - 40) * x = 30)
  (hx2 : (a + 50) * x = 210) :
  a - 50 = 5 :=
by
  sorry

end increase_speed_to_pass_correctly_l301_301058


namespace base_length_of_isosceles_triangle_l301_301719

noncomputable def isosceles_triangle_base_length (height : ℝ) (radius : ℝ) : ℝ :=
  if height = 25 ∧ radius = 8 then 80 / 3 else 0

theorem base_length_of_isosceles_triangle :
  isosceles_triangle_base_length 25 8 = 80 / 3 :=
by
  -- skipping the proof
  sorry

end base_length_of_isosceles_triangle_l301_301719


namespace chip_final_balance_l301_301020

noncomputable def finalBalance : ℝ := 
  let initialBalance := 50.0
  let month1InterestRate := 0.20
  let month2NewCharges := 20.0
  let month2InterestRate := 0.20
  let month3NewCharges := 30.0
  let month3Payment := 10.0
  let month3InterestRate := 0.25
  let month4NewCharges := 40.0
  let month4Payment := 20.0
  let month4InterestRate := 0.15

  -- Month 1
  let month1InterestFee := initialBalance * month1InterestRate
  let balanceMonth1 := initialBalance + month1InterestFee

  -- Month 2
  let balanceMonth2BeforeInterest := balanceMonth1 + month2NewCharges
  let month2InterestFee := balanceMonth2BeforeInterest * month2InterestRate
  let balanceMonth2 := balanceMonth2BeforeInterest + month2InterestFee

  -- Month 3
  let balanceMonth3BeforeInterest := balanceMonth2 + month3NewCharges
  let balanceMonth3AfterPayment := balanceMonth3BeforeInterest - month3Payment
  let month3InterestFee := balanceMonth3AfterPayment * month3InterestRate
  let balanceMonth3 := balanceMonth3AfterPayment + month3InterestFee

  -- Month 4
  let balanceMonth4BeforeInterest := balanceMonth3 + month4NewCharges
  let balanceMonth4AfterPayment := balanceMonth4BeforeInterest - month4Payment
  let month4InterestFee := balanceMonth4AfterPayment * month4InterestRate
  let balanceMonth4 := balanceMonth4AfterPayment + month4InterestFee

  balanceMonth4

theorem chip_final_balance : finalBalance = 189.75 := by sorry

end chip_final_balance_l301_301020


namespace range_of_m_l301_301638

def p (m : ℝ) : Prop := m > 3
def q (m : ℝ) : Prop := m > (1 / 4)

theorem range_of_m (m : ℝ) (h1 : ¬p m) (h2 : p m ∨ q m) : (1 / 4) < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l301_301638


namespace divide_P_Q_l301_301171

noncomputable def sequence_of_ones (n : ℕ) : ℕ := (10 ^ n - 1) / 9

theorem divide_P_Q (n : ℕ) (h : 1997 ∣ sequence_of_ones n) :
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*n) + 9 * 10^(2*n) + 9 * 10^n + 7)) ∧
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*(n + 1)) + 9 * 10^(2*(n + 1)) + 9 * 10^(n + 1) + 7)) := 
by
  sorry

end divide_P_Q_l301_301171


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301758

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301758


namespace find_r_l301_301633

theorem find_r : ∃ r : ℕ, (5 + 7 * 8 + 1 * 8^2) = 120 + r ∧ r = 5 := 
by
  use 5
  sorry

end find_r_l301_301633


namespace equal_chords_divide_equally_l301_301968

theorem equal_chords_divide_equally 
  {A B C D M : ℝ} 
  (in_circle : ∃ (O : ℝ), (dist O A = dist O B) ∧ (dist O C = dist O D) ∧ (dist O M < dist O A))
  (chords_equal : dist A B = dist C D)
  (intersection_M : dist A M + dist M B = dist C M + dist M D ∧ dist A M = dist C M ∧ dist B M = dist D M) :
  dist A M = dist M B ∧ dist C M = dist M D := 
sorry

end equal_chords_divide_equally_l301_301968


namespace prime_sum_remainder_l301_301784

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301784


namespace pizza_slices_left_l301_301349

theorem pizza_slices_left (total_slices john_ate : ℕ) 
  (initial_slices : total_slices = 12) 
  (john_slices : john_ate = 3) 
  (sam_ate : ¬¬(2 * john_ate = 6)) : 
  ∃ slices_left, slices_left = 3 :=
by
  sorry

end pizza_slices_left_l301_301349


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301833

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301833


namespace triangle_PZQ_area_is_50_l301_301516

noncomputable def area_triangle_PZQ (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) : ℝ :=
  let RS := PQ -- since PQRS is a rectangle, RS = PQ
  let XY := RS - RX - SY
  let height := 2 * QR -- height is doubled due to triangle similarity ratio
  let area := 0.5 * PQ * height
  area

theorem triangle_PZQ_area_is_50 (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) :
  area_triangle_PZQ PQ QR RX SY hPQ hQR hRX hSY = 50 :=
  sorry

end triangle_PZQ_area_is_50_l301_301516


namespace least_sub_to_make_div_by_10_l301_301857

theorem least_sub_to_make_div_by_10 : 
  ∃ n, n = 8 ∧ ∀ k, 427398 - k = 10 * m → k ≥ n ∧ k = 8 :=
sorry

end least_sub_to_make_div_by_10_l301_301857


namespace tan_subtraction_modified_l301_301503

theorem tan_subtraction_modified (α β : ℝ) (h1 : Real.tan α = 9) (h2 : Real.tan β = 6) :
  Real.tan (α - β) = (3 : ℝ) / (157465 : ℝ) := by
  have h3 : Real.tan (α - β) = (Real.tan α - Real.tan β) / (1 + (Real.tan α * Real.tan β)^3) :=
    sorry -- this is assumed as given in the conditions
  sorry -- rest of the proof

end tan_subtraction_modified_l301_301503


namespace problem_l301_301052

variable {x y : ℝ}

theorem problem (hx : 0 < x) (hy : 0 < y) (h : x^2 - y^2 = 3 * x * y) :
  (x^2 / y^2) + (y^2 / x^2) - 2 = 9 :=
sorry

end problem_l301_301052


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301746

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301746


namespace total_pull_ups_per_week_l301_301535

-- Definitions from the conditions
def pull_ups_per_time := 2
def visits_per_day := 5
def days_per_week := 7

-- The Math proof problem statement
theorem total_pull_ups_per_week :
  pull_ups_per_time * visits_per_day * days_per_week = 70 := by
  sorry

end total_pull_ups_per_week_l301_301535


namespace johns_avg_speed_l301_301070

/-
John cycled 40 miles at 8 miles per hour and 20 miles at 40 miles per hour.
We want to prove that his average speed for the entire trip is 10.91 miles per hour.
-/

theorem johns_avg_speed :
  let distance1 := 40
  let speed1 := 8
  let distance2 := 20
  let speed2 := 40
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  avg_speed = 10.91 :=
by
  sorry

end johns_avg_speed_l301_301070


namespace number_of_solutions_l301_301655

theorem number_of_solutions :
  (∃ x : ℝ, -25 < x ∧ x < 120 ∧ 2 * Real.cos x ^ 2 - 3 * Real.sin x ^ 2 = 1) → 
  -- There are 24 distinct values of x that satisfy the conditions 
  ∃ count : ℕ, count = 24 := by
  sorry

end number_of_solutions_l301_301655


namespace probability_X_Y_Z_problems_l301_301398

-- Define the success probabilities for Problem A
def P_X_A : ℚ := 1 / 5
def P_Y_A : ℚ := 1 / 2

-- Define the success probabilities for Problem B
def P_Y_B : ℚ := 3 / 5

-- Define the negation of success probabilities for Problem C
def P_Y_not_C : ℚ := 5 / 8
def P_X_not_C : ℚ := 3 / 4
def P_Z_not_C : ℚ := 7 / 16

-- State the final probability theorem
theorem probability_X_Y_Z_problems :
  P_X_A * P_Y_A * P_Y_B * P_Y_not_C * P_X_not_C * P_Z_not_C = 63 / 2048 := 
sorry

end probability_X_Y_Z_problems_l301_301398


namespace even_odd_decomposition_exp_l301_301935

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x
def decomposition (f g : ℝ → ℝ) := ∀ x, f x + g x = Real.exp x

-- Main statement to prove
theorem even_odd_decomposition_exp (hf : is_even f) (hg : is_odd g) (hfg : decomposition f g) :
  f (Real.log 2) + g (Real.log (1 / 2)) = 1 / 2 := 
sorry

end even_odd_decomposition_exp_l301_301935


namespace triangle_area_triangle_obtuse_exists_l301_301063

variables {a b c : ℝ}
variables {A B C : ℝ}

-- Given conditions
def condition_b : b = a + 1 := by sorry
def condition_c : c = a + 2 := by sorry
def condition_sin : 2 * Real.sin C = 3 * Real.sin A := by sorry

-- Proof statement for Part (1)
theorem triangle_area (h1 : condition_b) (h2 : condition_c) (h3 : condition_sin) :
  (1 / 2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 := 
sorry

-- Proof statement for Part (2)
theorem triangle_obtuse_exists (h1 : condition_b) (h2 : condition_c) :
  ∃ a : ℝ, a = 2 ∧ a > 0 ∧
  (let b := a + 1 in let c := a + 2 in
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) > Real.pi / 2) :=
sorry

end triangle_area_triangle_obtuse_exists_l301_301063


namespace community_cleaning_children_l301_301231

theorem community_cleaning_children (total_members adult_men_ratio adult_women_ratio : ℕ) 
(h_total : total_members = 2000)
(h_men_ratio : adult_men_ratio = 30) 
(h_women_ratio : adult_women_ratio = 2) :
  (total_members - (adult_men_ratio * total_members / 100 + 
  adult_women_ratio * (adult_men_ratio * total_members / 100))) = 200 :=
by
  sorry

end community_cleaning_children_l301_301231


namespace determine_moles_Al2O3_formed_l301_301151

noncomputable def initial_moles_Al : ℝ := 10
noncomputable def initial_moles_Fe2O3 : ℝ := 6
noncomputable def balanced_eq (moles_Al moles_Fe2O3 moles_Al2O3 moles_Fe : ℝ) : Prop :=
  2 * moles_Al + moles_Fe2O3 = moles_Al2O3 + 2 * moles_Fe

theorem determine_moles_Al2O3_formed :
  ∃ moles_Al2O3 : ℝ, balanced_eq 10 6 moles_Al2O3 (moles_Al2O3 * 2) ∧ moles_Al2O3 = 5 := 
  by 
  sorry

end determine_moles_Al2O3_formed_l301_301151


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301768

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301768


namespace quadratic_root_range_l301_301962

theorem quadratic_root_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, (x₁ > 0) ∧ (x₂ < 0) ∧ (x₁^2 + 2 * (a - 1) * x₁ + 2 * a + 6 = 0) ∧ (x₂^2 + 2 * (a - 1) * x₂ + 2 * a + 6 = 0)) → a < -3 :=
by
  sorry

end quadratic_root_range_l301_301962


namespace number_of_truthful_dwarfs_l301_301450

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l301_301450


namespace increasing_interval_of_f_l301_301095

def f (x : ℝ) : ℝ := (x - 1) ^ 2 - 2

theorem increasing_interval_of_f : ∀ x, 1 < x → f x > f 1 := 
sorry

end increasing_interval_of_f_l301_301095


namespace sum_of_tens_l301_301940

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l301_301940


namespace prime_sum_remainder_l301_301783

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301783


namespace current_balance_after_deduction_l301_301898

theorem current_balance_after_deduction :
  ∀ (original_balance deduction_percent : ℕ), 
  original_balance = 100000 →
  deduction_percent = 10 →
  original_balance - (deduction_percent * original_balance / 100) = 90000 :=
by
  intros original_balance deduction_percent h1 h2
  sorry

end current_balance_after_deduction_l301_301898


namespace prime_sum_remainder_l301_301782

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301782


namespace range_of_x_for_fx1_positive_l301_301637

-- Define the conditions
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_monotonic_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x
def f_at_2_eq_zero (f : ℝ → ℝ) := f 2 = 0

-- Define the problem statement that needs to be proven
theorem range_of_x_for_fx1_positive (f : ℝ → ℝ) :
  is_even f →
  is_monotonic_decreasing_on_nonneg f →
  f_at_2_eq_zero f →
  ∀ x, f (x - 1) > 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end range_of_x_for_fx1_positive_l301_301637


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301817

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301817


namespace wrenches_in_comparison_group_l301_301653

theorem wrenches_in_comparison_group (H W : ℝ) (x : ℕ) 
  (h1 : W = 2 * H)
  (h2 : 2 * H + 2 * W = (1 / 3) * (8 * H + x * W)) : x = 5 :=
by
  sorry

end wrenches_in_comparison_group_l301_301653


namespace probability_at_least_one_card_each_cousin_correct_l301_301697

noncomputable def probability_at_least_one_card_each_cousin : ℚ :=
  let total_cards := 16
  let cards_per_cousin := 8
  let selections := 3
  let total_ways := Nat.choose total_cards selections
  let ways_all_from_one_cousin := Nat.choose cards_per_cousin selections * 2  -- twice: once for each cousin
  let prob_all_from_one_cousin := (ways_all_from_one_cousin : ℚ) / total_ways
  1 - prob_all_from_one_cousin

theorem probability_at_least_one_card_each_cousin_correct :
  probability_at_least_one_card_each_cousin = 4 / 5 :=
by
  -- Proof would go here
  sorry

end probability_at_least_one_card_each_cousin_correct_l301_301697


namespace problem_solution_l301_301909

theorem problem_solution
  (a b : ℝ)
  (h1 : a * b = 2)
  (h2 : a - b = 3) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 18 :=
by
  sorry

end problem_solution_l301_301909


namespace dwarfs_truthful_count_l301_301448

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l301_301448


namespace captain_and_vicecaptain_pair_boys_and_girls_l301_301673

-- Problem A
theorem captain_and_vicecaptain (n : ℕ) (h : n = 11) : ∃ ways : ℕ, ways = 110 :=
by
  sorry

-- Problem B
theorem pair_boys_and_girls (N : ℕ) : ∃ ways : ℕ, ways = Nat.factorial N :=
by
  sorry

end captain_and_vicecaptain_pair_boys_and_girls_l301_301673


namespace remainder_first_six_primes_div_seventh_l301_301841

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301841


namespace probability_theorem_l301_301161

noncomputable def probability_odd_sum_given_even_product : ℚ :=
  let total_ways := 8^5
  let ways_no_even := 4^5
  let ways_at_least_one_even := total_ways - ways_no_even
  let ways_odd_sum_given_even (n : ℕ) : ℕ :=
    match n with 
    | 1 => 5 * 4^1 * 4^4
    | 3 => 10 * 4^3 * 4^2
    | _ => 0
  let favorable_outcomes := ways_odd_sum_given_even 1 + ways_odd_sum_given_even 3
  favorable_outcomes / ways_at_least_one_even

theorem probability_theorem : probability_odd_sum_given_even_product = 15 / 31 := 
  sorry

end probability_theorem_l301_301161


namespace stamp_total_cost_l301_301145

theorem stamp_total_cost :
  let price_A := 2
  let price_B := 3
  let price_C := 5
  let num_A := 150
  let num_B := 90
  let num_C := 60
  let discount_A := if num_A > 100 then 0.20 else 0
  let discount_B := if num_B > 50 then 0.15 else 0
  let discount_C := if num_C > 30 then 0.10 else 0
  let cost_A := num_A * price_A * (1 - discount_A)
  let cost_B := num_B * price_B * (1 - discount_B)
  let cost_C := num_C * price_C * (1 - discount_C)
  cost_A + cost_B + cost_C = 739.50 := sorry

end stamp_total_cost_l301_301145


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301812

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301812


namespace Q_coordinates_l301_301315

def P : (ℝ × ℝ) := (2, -6)

def Q (x : ℝ) : (ℝ × ℝ) := (x, -6)

axiom PQ_parallel_to_x_axis : ∀ x, Q x = (x, -6)

axiom PQ_length : dist (Q 0) P = 2 ∨ dist (Q 4) P = 2

theorem Q_coordinates : Q 0 = (0, -6) ∨ Q 4 = (4, -6) :=
by {
  sorry
}

end Q_coordinates_l301_301315


namespace amanda_tickets_l301_301016

theorem amanda_tickets (F : ℕ) (h : 4 * F + 32 + 28 = 80) : F = 5 :=
by
  sorry

end amanda_tickets_l301_301016


namespace spinner_probability_l301_301132

theorem spinner_probability :
  let p_A := (1 / 4)
  let p_B := (1 / 3)
  let p_C := (5 / 12)
  let p_D := 1 - (p_A + p_B + p_C)
  p_D = 0 :=
by
  sorry

end spinner_probability_l301_301132


namespace inequality_holds_l301_301985

theorem inequality_holds (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 :=
by sorry

end inequality_holds_l301_301985


namespace negation_of_exists_log3_nonnegative_l301_301034

variable (x : ℝ)

theorem negation_of_exists_log3_nonnegative :
  (¬ (∃ x : ℝ, Real.logb 3 x ≥ 0)) ↔ (∀ x : ℝ, Real.logb 3 x < 0) :=
by
  sorry

end negation_of_exists_log3_nonnegative_l301_301034


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301759

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301759


namespace find_f2_l301_301041

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem find_f2 (a b : ℝ) (h : (∃ x : ℝ, f x a b = 10 ∧ x = 1)):
  f 2 a b = 18 ∨ f 2 a b = 11 :=
sorry

end find_f2_l301_301041


namespace solve_log_equation_l301_301368

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solve_log_equation (x : ℝ) (hx : 2 * log_base 5 x - 3 * log_base 5 4 = 1) :
  x = 4 * Real.sqrt 5 ∨ x = -4 * Real.sqrt 5 :=
sorry

end solve_log_equation_l301_301368


namespace find_a_plus_2b_l301_301486

variable (a b : ℝ)

theorem find_a_plus_2b (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : 
  a + 2 * b = 0 := 
sorry

end find_a_plus_2b_l301_301486


namespace complement_intersection_l301_301598

open Set

theorem complement_intersection
  (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = univ) 
  (hA : A = { x : ℝ | x ≤ -2 }) 
  (hB : B = { x : ℝ | x < 1 }) :
  (U \ A) ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_l301_301598


namespace loan_interest_rate_l301_301592

theorem loan_interest_rate (P SI T R : ℕ) (h1 : P = 900) (h2 : SI = 729) (h3 : T = R) :
  (SI = (P * R * T) / 100) -> R = 9 :=
by
  sorry

end loan_interest_rate_l301_301592


namespace intersecting_points_sum_l301_301378

theorem intersecting_points_sum (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1 : ∀ x y, (y = x ^ 3 - 4 * x + 3) ∧ (x + 3 * y = 3) → 
        (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3)) :
  (x1 + x2 + x3 = 0) ∧ (y1 + y2 + y3 = 3) :=
by
  sorry

end intersecting_points_sum_l301_301378


namespace find_odd_natural_numbers_l301_301134

-- Definition of a friendly number
def is_friendly (n : ℕ) : Prop :=
  ∀ i, (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 + 1 ∨ (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 - 1

-- Given condition: n is divisible by 64m
def is_divisible_by_64m (n m : ℕ) : Prop :=
  64 * m ∣ n

-- Proof problem statement
theorem find_odd_natural_numbers (m : ℕ) (hm1 : m % 2 = 1) :
  (5 ∣ m → ¬ ∃ n, is_friendly n ∧ is_divisible_by_64m n m) ∧ 
  (¬ 5 ∣ m → ∃ n, is_friendly n ∧ is_divisible_by_64m n m) :=
by
  sorry

end find_odd_natural_numbers_l301_301134


namespace intersection_A_B_l301_301924

def A : Set ℝ := {1, 3, 9, 27}
def B : Set ℝ := {y : ℝ | ∃ x ∈ A, y = Real.log x / Real.log 3}
theorem intersection_A_B : A ∩ B = {1, 3} := 
by
  sorry

end intersection_A_B_l301_301924


namespace probability_of_drawing_two_black_two_white_l301_301273

noncomputable def probability_two_black_two_white : ℚ :=
  let total_ways := (Nat.choose 18 4)
  let ways_black := (Nat.choose 10 2)
  let ways_white := (Nat.choose 8 2)
  let favorable_ways := ways_black * ways_white
  favorable_ways / total_ways

theorem probability_of_drawing_two_black_two_white :
  probability_two_black_two_white = 7 / 17 := sorry

end probability_of_drawing_two_black_two_white_l301_301273


namespace solve_quadratic_1_solve_quadratic_2_l301_301219

-- 1. Prove that the solutions to the equation x^2 - 4x - 1 = 0 are x = 2 + sqrt(5) and x = 2 - sqrt(5)
theorem solve_quadratic_1 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

-- 2. Prove that the solutions to the equation 3(x - 1)^2 = 2(x - 1) are x = 1 and x = 5/3
theorem solve_quadratic_2 (x : ℝ) : 3 * (x - 1) ^ 2 = 2 * (x - 1) ↔ x = 1 ∨ x = 5 / 3 :=
sorry

end solve_quadratic_1_solve_quadratic_2_l301_301219


namespace full_time_score_l301_301616

variables (x : ℕ)

def half_time_score_visitors := 14
def half_time_score_home := 9
def visitors_full_time_score := half_time_score_visitors + x
def home_full_time_score := half_time_score_home + 2 * x
def home_team_win_by_one := home_full_time_score = visitors_full_time_score + 1

theorem full_time_score 
  (h : home_team_win_by_one) : 
  visitors_full_time_score = 20 ∧ home_full_time_score = 21 :=
by
  sorry

end full_time_score_l301_301616


namespace acute_triangle_inequality_l301_301059

theorem acute_triangle_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = Real.pi)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
  (Real.sin A + Real.sin B + Real.sin C) * (1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C) ≤
    Real.pi * (1 / A + 1 / B + 1 / C) :=
sorry

end acute_triangle_inequality_l301_301059


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301793

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301793


namespace find_triangle_angles_l301_301239

theorem find_triangle_angles (α β γ : ℝ)
  (h1 : (180 - α) / (180 - β) = 13 / 9)
  (h2 : β - α = 45)
  (h3 : α + β + γ = 180) :
  (α = 33.75) ∧ (β = 78.75) ∧ (γ = 67.5) :=
by
  sorry

end find_triangle_angles_l301_301239


namespace total_cost_jello_l301_301522

def total_cost_james_spent : Real := 259.20

theorem total_cost_jello 
  (pounds_per_cubic_foot : ℝ := 8)
  (gallons_per_cubic_foot : ℝ := 7.5)
  (tablespoons_per_pound : ℝ := 1.5)
  (cost_red_jello : ℝ := 0.50)
  (cost_blue_jello : ℝ := 0.40)
  (cost_green_jello : ℝ := 0.60)
  (percentage_red_jello : ℝ := 0.60)
  (percentage_blue_jello : ℝ := 0.30)
  (percentage_green_jello : ℝ := 0.10)
  (volume_cubic_feet : ℝ := 6) :
  (volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_red_jello * cost_red_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_blue_jello * cost_blue_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_green_jello * cost_green_jello) = total_cost_james_spent :=
by
  sorry

end total_cost_jello_l301_301522


namespace johns_quarters_l301_301683

variable (x : ℕ)  -- Number of quarters John has

def number_of_dimes : ℕ := x + 3  -- Number of dimes
def number_of_nickels : ℕ := x - 6  -- Number of nickels

theorem johns_quarters (h : x + (x + 3) + (x - 6) = 63) : x = 22 :=
by
  sorry

end johns_quarters_l301_301683


namespace sum_m_n_is_55_l301_301984

theorem sum_m_n_is_55 (a b c : ℝ) (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1)
  (h1 : 5 / a = b + c) (h2 : 10 / b = c + a) (h3 : 13 / c = a + b) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : (a + b + c) = m / n) : m + n = 55 :=
  sorry

end sum_m_n_is_55_l301_301984


namespace next_meeting_time_l301_301007

noncomputable def perimeter (AB BC CD DA : ℝ) : ℝ :=
  AB + BC + CD + DA

theorem next_meeting_time 
  (AB BC CD AD : ℝ) 
  (v_human v_dog : ℝ) 
  (initial_meeting_time : ℝ) :
  AB = 100 → BC = 200 → CD = 100 → AD = 200 →
  initial_meeting_time = 2 →
  v_human + v_dog = 300 →
  ∃ next_time : ℝ, next_time = 14 := 
by
  sorry

end next_meeting_time_l301_301007


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301804

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301804


namespace customers_left_l301_301293

-- Given conditions:
def initial_customers : ℕ := 21
def remaining_customers : ℕ := 12

-- Prove that the number of customers who left is 9
theorem customers_left : initial_customers - remaining_customers = 9 := by
  sorry

end customers_left_l301_301293


namespace union_A_B_l301_301035

noncomputable def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
noncomputable def B : Set ℝ := {x | x^2 - 1 < 0}

theorem union_A_B : A ∪ B = {x : ℝ | -1 < x} := by
  sorry

end union_A_B_l301_301035


namespace simplify_expression_l301_301546

variable {a b c : ℝ}

-- Assuming the conditions specified in the problem
def valid_conditions (a b c : ℝ) : Prop := (1 - a * b ≠ 0) ∧ (1 + c * a ≠ 0)

theorem simplify_expression (h : valid_conditions a b c) :
  (a + b) / (1 - a * b) + (c - a) / (1 + c * a) / 
  (1 - ((a + b) / (1 - a * b) * (c - a) / (1 + c * a))) = 
  (b + c) / (1 - b * c) := 
sorry

end simplify_expression_l301_301546


namespace line_equations_l301_301308

theorem line_equations (x y : ℝ) 
  (h1 : (-4, 0) ∈ {p : ℝ × ℝ | p.2 = (Real.sin (Real.arctan (1 / 3))) * (p.1 + 4)}) 
  (h2 : (-2, 1) ∈ {p : ℝ × ℝ | p.2 = 0 ∨ 2 * |3 / 4| / Real.sqrt((3 / 4)^2 + 1) = 2}) : 
  (x + 2 = 0 ∨ 3*x - 4*y + 10 = 0) :=
by {
  sorry
}

end line_equations_l301_301308


namespace square_possible_length_l301_301562

theorem square_possible_length (sticks : Finset ℕ) (H : sticks = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∃ s, s = 9 ∧
  ∃ (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧ a + b + c = 9 :=
by
  sorry

end square_possible_length_l301_301562


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301831

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301831


namespace diff_of_cubes_divisible_by_9_l301_301899

theorem diff_of_cubes_divisible_by_9 (a b : ℤ) : 9 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3) := 
sorry

end diff_of_cubes_divisible_by_9_l301_301899


namespace number_of_pairs_l301_301029

noncomputable def number_of_ordered_pairs (n : ℕ) : ℕ :=
  if n = 5 then 8 else 0

theorem number_of_pairs (f m: ℕ) : f ≥ 0 ∧ m ≥ 0 → number_of_ordered_pairs 5 = 8 :=
by
  intro h
  sorry

end number_of_pairs_l301_301029


namespace family_total_cost_l301_301061

def cost_of_entrance_ticket : ℕ := 5
def cost_of_child_attraction_ticket : ℕ := 2
def cost_of_adult_attraction_ticket : ℕ := 4
def number_of_children : ℕ := 4
def number_of_parents : ℕ := 2
def number_of_grandmothers : ℕ := 1

def number_of_family_members : ℕ := number_of_children + number_of_parents + number_of_grandmothers
def cost_of_entrance_tickets : ℕ := number_of_family_members * cost_of_entrance_ticket
def cost_of_children_attraction_tickets : ℕ := number_of_children * cost_of_child_attraction_ticket
def number_of_adults : ℕ := number_of_parents + number_of_grandmothers
def cost_of_adults_attraction_tickets : ℕ := number_of_adults * cost_of_adult_attraction_ticket

def total_cost : ℕ := cost_of_entrance_tickets + cost_of_children_attraction_tickets + cost_of_adults_attraction_tickets

theorem family_total_cost : total_cost = 55 := 
by 
  -- Calculation of number_of_family_members: 4 children + 2 parents + 1 grandmother = 7
  have h1 : number_of_family_members = 4 + 2 + 1 := by rfl
  -- Calculation of cost_of_entrance_tickets: 7 people * $5 = $35
  have h2 : cost_of_entrance_tickets = 7 * 5 := by rw h1 
  -- Calculation of cost_of_children_attraction_tickets: 4 children * $2 = $8
  have h3 : cost_of_children_attraction_tickets = 4 * 2 := by rfl
  -- Calculation of number_of_adults: 2 parents + 1 grandmother = 3
  have h4 : number_of_adults = 2 + 1 := by rfl
  -- Calculation of cost_of_adults_attraction_tickets: 3 adults * $4 = $12
  have h5 : cost_of_adults_attraction_tickets = 3 * 4 := by rw h4
  -- Total cost calculation: $35 (entrance) + $8 (children) + $12 (adults) = $55
  have h6 : total_cost = 35 + 8 + 12 := 
    by rw [h2, h3, h5]
  exact h6


end family_total_cost_l301_301061


namespace find_a_values_l301_301625

noncomputable def system_has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((|y - 4| + |x + 12| - 3) * (x^2 + y^2 - 12) = 0) ∧ 
    ((x + 5)^2 + (y - 4)^2 = a)

theorem find_a_values : system_has_exactly_three_solutions 16 ∨ 
                        system_has_exactly_three_solutions (41 + 4 * Real.sqrt 123) :=
  by sorry

end find_a_values_l301_301625


namespace number_of_truthful_dwarfs_is_correct_l301_301459

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l301_301459


namespace socks_combinations_correct_l301_301248

noncomputable def num_socks_combinations (colors patterns pairs : ℕ) : ℕ :=
  colors * (colors - 1) * patterns * (patterns - 1)

theorem socks_combinations_correct :
  num_socks_combinations 5 4 20 = 240 :=
by
  sorry

end socks_combinations_correct_l301_301248


namespace cost_of_plastering_l301_301856

def length := 25
def width := 12
def depth := 6
def cost_per_sq_meter_paise := 75

def surface_area_of_two_walls_one := 2 * (length * depth)
def surface_area_of_two_walls_two := 2 * (width * depth)
def surface_area_of_bottom := length * width

def total_surface_area := surface_area_of_two_walls_one + surface_area_of_two_walls_two + surface_area_of_bottom

def cost_per_sq_meter_rupees := cost_per_sq_meter_paise / 100
def total_cost := total_surface_area * cost_per_sq_meter_rupees

theorem cost_of_plastering : total_cost = 558 := by
  sorry

end cost_of_plastering_l301_301856


namespace largest_angle_in_pentagon_l301_301196

-- Define the angles and sum condition
variables (x : ℝ) {P Q R S T : ℝ}

-- Conditions
def angle_P : P = 90 := sorry
def angle_Q : Q = 70 := sorry
def angle_R : R = x := sorry
def angle_S : S = x := sorry
def angle_T : T = 2*x + 20 := sorry
def sum_of_angles : P + Q + R + S + T = 540 := sorry

-- Prove the largest angle
theorem largest_angle_in_pentagon (hP : P = 90) (hQ : Q = 70)
    (hR : R = x) (hS : S = x) (hT : T = 2*x + 20) 
    (h_sum : P + Q + R + S + T = 540) : T = 200 :=
by
  sorry

end largest_angle_in_pentagon_l301_301196


namespace find_sol_y_pct_l301_301362

-- Define the conditions
def sol_x_vol : ℕ := 200            -- Volume of solution x in milliliters
def sol_y_vol : ℕ := 600            -- Volume of solution y in milliliters
def sol_x_pct : ℕ := 10             -- Percentage of alcohol in solution x
def final_sol_pct : ℕ := 25         -- Percentage of alcohol in the final solution
def final_sol_vol := sol_x_vol + sol_y_vol -- Total volume of the final solution

-- Define the problem statement
theorem find_sol_y_pct (sol_x_vol sol_y_vol final_sol_vol : ℕ) 
  (sol_x_pct final_sol_pct : ℕ) : 
  (600 * 10 + sol_y_vol * 30) / 800 = 25 :=
by
  sorry

end find_sol_y_pct_l301_301362


namespace hexagon_angle_D_135_l301_301971

theorem hexagon_angle_D_135 
  (A B C D E F : ℝ)
  (h1 : A = B ∧ B = C)
  (h2 : D = E ∧ E = F)
  (h3 : A = D - 30)
  (h4 : A + B + C + D + E + F = 720) :
  D = 135 :=
by {
  sorry
}

end hexagon_angle_D_135_l301_301971


namespace max_m_divides_f_l301_301480

noncomputable def f (n : ℕ) : ℤ :=
  (2 * n + 7) * 3^n + 9

theorem max_m_divides_f (m n : ℕ) (h1 : n > 0) (h2 : ∀ n : ℕ, n > 0 → m ∣ ((2 * n + 7) * 3^n + 9)) : m = 36 :=
sorry

end max_m_divides_f_l301_301480


namespace percentage_saved_l301_301877

theorem percentage_saved (amount_saved : ℝ) (amount_spent : ℝ) (h1 : amount_saved = 5) (h2 : amount_spent = 45) : 
  (amount_saved / (amount_spent + amount_saved)) * 100 = 10 :=
by 
  sorry

end percentage_saved_l301_301877


namespace piglet_weight_l301_301012

variable (C K P L : ℝ)

theorem piglet_weight (h1 : C = K + P) (h2 : P + C = L + K) (h3 : L = 30) : P = 15 := by
  sorry

end piglet_weight_l301_301012


namespace lesser_solution_of_quadratic_l301_301581

theorem lesser_solution_of_quadratic :
  (∃ x y: ℝ, x ≠ y ∧ x^2 + 10*x - 24 = 0 ∧ y^2 + 10*y - 24 = 0 ∧ min x y = -12) :=
by {
  sorry
}

end lesser_solution_of_quadratic_l301_301581


namespace remainder_when_12_plus_a_div_by_31_l301_301353

open Int

theorem remainder_when_12_plus_a_div_by_31 (a : ℤ) (ha : 0 < a) (h : 17 * a % 31 = 1) : (12 + a) % 31 = 23 := by
  sorry

end remainder_when_12_plus_a_div_by_31_l301_301353


namespace negation_of_sin_le_one_l301_301324

theorem negation_of_sin_le_one : (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end negation_of_sin_le_one_l301_301324


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301787

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301787


namespace gcd_9009_14014_l301_301626

-- Given conditions
def decompose_9009 : 9009 = 9 * 1001 := by sorry
def decompose_14014 : 14014 = 14 * 1001 := by sorry
def coprime_9_14 : Nat.gcd 9 14 = 1 := by sorry

-- Proof problem statement
theorem gcd_9009_14014 : Nat.gcd 9009 14014 = 1001 := by
  have h1 : 9009 = 9 * 1001 := decompose_9009
  have h2 : 14014 = 14 * 1001 := decompose_14014
  have h3 : Nat.gcd 9 14 = 1 := coprime_9_14
  sorry

end gcd_9009_14014_l301_301626


namespace spending_on_other_items_is_30_percent_l301_301212

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

end spending_on_other_items_is_30_percent_l301_301212


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301749

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301749


namespace sqrt_neg9_sq_l301_301264

theorem sqrt_neg9_sq : Real.sqrt ((-9 : Real)^2) = 9 := 
by 
  sorry

end sqrt_neg9_sq_l301_301264


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301803

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301803


namespace exist_alpha_beta_l301_301352

variables {a b : ℝ} {f : ℝ → ℝ}

-- Assume that f has the Intermediate Value Property (for simplicity, define it as a predicate)
def intermediate_value_property (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ k ∈ Set.Icc (min (f a) (f b)) (max (f a) (f b)),
    ∃ c ∈ Set.Ioo a b, f c = k

-- Assume the conditions from the problem
variables (h_ivp : intermediate_value_property f a b) (h_sign_change : f a * f b < 0)

-- The theorem we need to prove
theorem exist_alpha_beta (hivp : intermediate_value_property f a b) (hsign : f a * f b < 0) :
  ∃ α β, a < α ∧ α < β ∧ β < b ∧ f α + f β = f α * f β :=
sorry

end exist_alpha_beta_l301_301352


namespace hexagon_area_correct_m_plus_n_l301_301986

noncomputable def hexagon_area (b : ℝ) : ℝ :=
  let A := (0, 0)
  let B := (b, 3)
  let F := (-3 * (3 + b) / 2, 9)  -- derived from complex numbers and angle conversion
  let hexagon_height := 12  -- height difference between the y-coordinates
  let hexagon_base := 3 * (b + 3) / 2  -- distance between parallel lines AB and DE
  36 / 2 * (b + 3) + 6 * (6 + b * Real.sqrt 3)

theorem hexagon_area_correct (b : ℝ) :
  hexagon_area b = 72 * Real.sqrt 3 :=
sorry

theorem m_plus_n : 72 + 3 = 75 := rfl

end hexagon_area_correct_m_plus_n_l301_301986


namespace smoothie_supplements_combinations_l301_301137

/-- The number of combinations of one type of smoothie and three different supplements -/
theorem smoothie_supplements_combinations : 
  let smoothies := 7
  let supplements := 8
  smoothies * Nat.choose supplements 3 = 392 :=
by
  intros
  have h := Nat.choose_eq_factorial_div_factorial (m := supplements) 3
  rw [h]
  simp
  sorry

end smoothie_supplements_combinations_l301_301137


namespace family_visit_cost_is_55_l301_301060

def num_children := 4
def num_parents := 2
def num_grandmother := 1
def num_people := num_children + num_parents + num_grandmother

def entrance_ticket_cost := 5
def attraction_ticket_cost_kid := 2
def attraction_ticket_cost_adult := 4

def entrance_total_cost := num_people * entrance_ticket_cost
def attraction_total_cost_kids := num_children * attraction_ticket_cost_kid
def adults := num_parents + num_grandmother
def attraction_total_cost_adults := adults * attraction_ticket_cost_adult

def total_cost := entrance_total_cost + attraction_total_cost_kids + attraction_total_cost_adults

theorem family_visit_cost_is_55 : total_cost = 55 := by
  sorry

end family_visit_cost_is_55_l301_301060


namespace keiko_speed_l301_301071

theorem keiko_speed (wA wB tA tB : ℝ) (v : ℝ)
    (h1: wA = 4)
    (h2: wB = 8)
    (h3: tA = 48)
    (h4: tB = 72)
    (h5: v = (24 * π) / 60) :
    v = 2 * π / 5 :=
by
  sorry

end keiko_speed_l301_301071


namespace probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l301_301148

-- Defining the conditions
def p : ℚ := 4 / 5
def n : ℕ := 5
def k1 : ℕ := 2
def k2 : ℕ := 1

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Binomial probability function
def binom_prob (k n : ℕ) (p : ℚ) : ℚ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- The first proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate is 0.05 given the accuracy rate
theorem probability_of_2_out_of_5_accurate :
  binom_prob k1 n p = 0.05 := by
  sorry

-- The second proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate, with the third forecast being one of the accurate ones, is 0.02 given the accuracy rate
theorem probability_of_2_out_of_5_with_third_accurate :
  binom_prob k2 (n - 1) p = 0.02 := by
  sorry

end probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l301_301148


namespace eval_floor_abs_neg_45_7_l301_301465

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l301_301465


namespace remainder_first_six_primes_div_seventh_l301_301845

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301845


namespace B_months_grazing_eq_five_l301_301412

-- Define the conditions in the problem
def A_oxen : ℕ := 10
def A_months : ℕ := 7
def B_oxen : ℕ := 12
def C_oxen : ℕ := 15
def C_months : ℕ := 3
def total_rent : ℝ := 175
def C_rent_share : ℝ := 45

-- Total ox-units function
def total_ox_units (x : ℕ) : ℕ :=
  A_oxen * A_months + B_oxen * x + C_oxen * C_months

-- Prove that the number of months B's oxen grazed is 5
theorem B_months_grazing_eq_five (x : ℕ) :
  total_ox_units x = 70 + 12 * x + 45 →
  (C_rent_share / total_rent = 45 / total_ox_units x) →
  x = 5 :=
by
  intros h1 h2
  sorry

end B_months_grazing_eq_five_l301_301412


namespace tangent_line_computation_l301_301323

variables (f : ℝ → ℝ)

theorem tangent_line_computation (h_tangent : ∀ x, (f x = -x + 8) ∧ (∃ y, y = -x + 8 → (f y) = -x + 8 → deriv f x = -1)) :
    f 5 + deriv f 5 = 2 :=
sorry

end tangent_line_computation_l301_301323


namespace sqrt_neg9_squared_l301_301265

theorem sqrt_neg9_squared : Real.sqrt ((-9: ℝ)^2) = 9 := by
  sorry

end sqrt_neg9_squared_l301_301265


namespace paco_initial_sweet_cookies_l301_301702

theorem paco_initial_sweet_cookies (S : ℕ) (h1 : S - 15 = 7) : S = 22 :=
by
  sorry

end paco_initial_sweet_cookies_l301_301702


namespace dwarfs_truthful_count_l301_301435

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l301_301435


namespace complement_B_range_a_l301_301043

open Set

variable (A B : Set ℝ) (a : ℝ)

def mySetA : Set ℝ := {x | 2 * a - 2 < x ∧ x < a}
def mySetB : Set ℝ := {x | 3 / (x - 1) ≥ 1}

theorem complement_B_range_a (h : mySetA a ⊆ compl mySetB) : 
  compl mySetB = {x | x ≤ 1} ∪ {x | x > 4} ∧ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end complement_B_range_a_l301_301043


namespace total_pull_ups_per_week_l301_301536

-- Definitions from the conditions
def pull_ups_per_time := 2
def visits_per_day := 5
def days_per_week := 7

-- The Math proof problem statement
theorem total_pull_ups_per_week :
  pull_ups_per_time * visits_per_day * days_per_week = 70 := by
  sorry

end total_pull_ups_per_week_l301_301536


namespace largest_x_eq_neg5_l301_301303

theorem largest_x_eq_neg5 (x : ℝ) (h : x ≠ 7) : (x^2 - 5*x - 84)/(x - 7) = 2/(x + 6) → x ≤ -5 := 
sorry

end largest_x_eq_neg5_l301_301303


namespace num_non_congruent_tris_l301_301320

theorem num_non_congruent_tris (a b : ℕ) (h1 : a ≤ 2) (h2 : 2 ≤ b) (h3 : a + 2 > b) (h4 : a + b > 2) (h5 : b + 2 > a) : 
  ∃ q, q = 3 := 
by 
  use 3 
  sorry

end num_non_congruent_tris_l301_301320


namespace find_k_for_tangent_graph_l301_301619

theorem find_k_for_tangent_graph (k : ℝ) (h : (∀ x : ℝ, x^2 - 6 * x + k = 0 → (x = 3))) : k = 9 :=
sorry

end find_k_for_tangent_graph_l301_301619


namespace connected_graph_partitions_l301_301688

noncomputable def partitions_count (G : Graph) : Nat := -- Definition which counts valid partitions
  sorry

theorem connected_graph_partitions (G : Graph) (k : Nat) (h_connected : G.connected) (h_edges : G.edges = k) :
  partitions_count G ≥ k :=
by
  sorry

end connected_graph_partitions_l301_301688


namespace angle_A_measure_triangle_area_l301_301037

variable {a b c : ℝ} 
variable {A B C : ℝ} 
variable (triangle : a^2 = b^2 + c^2 - 2 * b * c * (Real.cos A))

theorem angle_A_measure (h : (b - c)^2 = a^2 - b * c) : A = Real.pi / 3 :=
sorry

theorem triangle_area 
  (h1 : a = 3) 
  (h2 : Real.sin C = 2 * Real.sin B) 
  (h3 : A = Real.pi / 3) 
  (hb : b = Real.sqrt 3)
  (hc : c = 2 * Real.sqrt 3) : 
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end angle_A_measure_triangle_area_l301_301037


namespace polynomial_root_l301_301974

theorem polynomial_root (x0 : ℝ) (z : ℝ) 
  (h1 : x0^3 - x0 - 1 = 0) 
  (h2 : z = x0^2 + 3 * x0 + 1) : 
  z^3 - 5 * z^2 - 10 * z - 11 = 0 := 
sorry

end polynomial_root_l301_301974


namespace guaranteed_winning_strategy_l301_301563

variable (a b : ℝ)

theorem guaranteed_winning_strategy (h : a ≠ b) : (a^3 + b^3) > (a^2 * b + a * b^2) :=
by 
  sorry

end guaranteed_winning_strategy_l301_301563


namespace largest_integer_with_4_digit_square_in_base_7_l301_301201

theorem largest_integer_with_4_digit_square_in_base_7 (M : ℕ) :
  (∀ m : ℕ, m < 240 ∧ 49 ≤ m → m ≤ 239) ∧ nat.to_digits 7 239 = [4, 6, 1] :=
begin
  sorry
end

end largest_integer_with_4_digit_square_in_base_7_l301_301201


namespace genuine_items_count_l301_301569

def total_purses : ℕ := 26
def total_handbags : ℕ := 24
def fake_purses : ℕ := total_purses / 2
def fake_handbags : ℕ := total_handbags / 4
def genuine_purses : ℕ := total_purses - fake_purses
def genuine_handbags : ℕ := total_handbags - fake_handbags

theorem genuine_items_count : genuine_purses + genuine_handbags = 31 := by
  sorry

end genuine_items_count_l301_301569


namespace pair_divisibility_l301_301307

theorem pair_divisibility (m n : ℕ) : 
  (m * n ∣ m ^ 2019 + n) ↔ ((m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 2 ^ 2019)) := sorry

end pair_divisibility_l301_301307


namespace equation_solution_l301_301367

theorem equation_solution (x y : ℕ) (h : x^3 - y^3 = x * y + 61) : x = 6 ∧ y = 5 :=
by
  sorry

end equation_solution_l301_301367


namespace triangle_ratio_l301_301198

-- We define the geometrical configurations and properties
noncomputable def length_AB (AC BC : ℝ) : ℝ := Real.sqrt (AC^2 + BC^2)
def is_right_triangle (A B C : ℝ × ℝ) : Prop := A.x^2 + B.x^2 = C.x^2

theorem triangle_ratio 
    (AC BC AD DE DB: ℝ)
    (h1 : AC = 5) 
    (h2 : BC = 12) 
    (h3 : length_AB AC BC = 13) 
    (h4 : AD = 20) 
    (h5 : is_right_triangle (5, 12) (12, 5) (13, 0))
    (h6 : DE / DB = (12 : ℝ) / 13) :
    (12 + 13 = 25) :=
    by
    sorry

end triangle_ratio_l301_301198


namespace range_of_a_l301_301917

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 
  a * x + 1 - 4 * a 
else 
  x ^ 2 - 3 * a * x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) → 
  a ∈ (Set.Ioi (2/3) ∪ Set.Iic 0) :=
sorry

end range_of_a_l301_301917


namespace largest_possible_b_l301_301247

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
sorry

end largest_possible_b_l301_301247


namespace number_of_tens_in_sum_l301_301953

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l301_301953


namespace totalBalls_l301_301525

def jungkookBalls : Nat := 3
def yoongiBalls : Nat := 2

theorem totalBalls : jungkookBalls + yoongiBalls = 5 := by
  sorry

end totalBalls_l301_301525


namespace sum_of_tens_l301_301947

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l301_301947


namespace slant_height_base_plane_angle_l301_301238

noncomputable def angle_between_slant_height_and_base_plane (R : ℝ) : ℝ :=
  Real.arcsin ((Real.sqrt 13 - 1) / 3)

theorem slant_height_base_plane_angle (R : ℝ) (h : R = R) : angle_between_slant_height_and_base_plane R = Real.arcsin ((Real.sqrt 13 - 1) / 3) :=
by
  -- Here we assume that the mathematical conditions and transformations hold true.
  -- According to the solution steps provided:
  -- We found that γ = arcsin ((sqrt(13) - 1) / 3)
  sorry

end slant_height_base_plane_angle_l301_301238


namespace secant_line_slope_positive_l301_301375

theorem secant_line_slope_positive (f : ℝ → ℝ) (h_deriv : ∀ x : ℝ, 0 < (deriv f x)) :
  ∀ (x1 x2 : ℝ), x1 ≠ x2 → 0 < (f x1 - f x2) / (x1 - x2) :=
by
  intros x1 x2 h_ne
  sorry

end secant_line_slope_positive_l301_301375


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301808

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301808


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301788

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301788


namespace sin_inequality_in_triangle_l301_301054

theorem sin_inequality_in_triangle (A B C : ℝ) (h_sum : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  Real.sin A * Real.sin (A / 2) + Real.sin B * Real.sin (B / 2) + Real.sin C * Real.sin (C / 2) ≤ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end sin_inequality_in_triangle_l301_301054


namespace xyz_inequality_l301_301081

theorem xyz_inequality (x y z : ℝ) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x := 
  sorry

end xyz_inequality_l301_301081


namespace number_of_tens_in_sum_l301_301956

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l301_301956


namespace sequence_an_expression_l301_301641

theorem sequence_an_expression (a : ℕ → ℕ) : 
  a 1 = 1 ∧ (∀ n : ℕ, n ≥ 1 → (a n / n - a (n - 1) / (n - 1)) = 2) → (∀ n : ℕ, a n = 2 * n * n - n) :=
by
  sorry

end sequence_an_expression_l301_301641


namespace serving_calculation_correct_l301_301140

def prepared_orange_juice_servings (cans_of_concentrate : ℕ) 
                                  (oz_per_concentrate_can : ℕ) 
                                  (water_ratio : ℕ) 
                                  (oz_per_serving : ℕ) : ℕ :=
  let total_concentrate := cans_of_concentrate * oz_per_concentrate_can
  let total_water := cans_of_concentrate * water_ratio * oz_per_concentrate_can
  let total_juice := total_concentrate + total_water
  total_juice / oz_per_serving

theorem serving_calculation_correct :
  prepared_orange_juice_servings 60 5 3 6 = 200 := by
  sorry

end serving_calculation_correct_l301_301140


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301764

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301764


namespace value_of_a_minus_b_l301_301185

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a + b > 0) :
  (a - b = -1) ∨ (a - b = -7) :=
by
  sorry

end value_of_a_minus_b_l301_301185


namespace apartments_decrease_l301_301142

theorem apartments_decrease (p_initial e_initial p e q : ℕ) (h1: p_initial = 5) (h2: e_initial = 2) (h3: q = 1)
    (first_mod: p = p_initial - 2) (e_first_mod: e = e_initial + 3) (q_eq: q = 1)
    (second_mod: p = p - 2) (e_second_mod: e = e + 3) :
    p_initial * e_initial * q > p * e * q := by
  sorry

end apartments_decrease_l301_301142


namespace truthful_dwarfs_count_l301_301430

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l301_301430


namespace shuttle_speed_conversion_l301_301138

-- Define the speed of the space shuttle in kilometers per second
def shuttle_speed_km_per_sec : ℕ := 6

-- Define the number of seconds in an hour
def seconds_per_hour : ℕ := 3600

-- Define the expected speed in kilometers per hour
def expected_speed_km_per_hour : ℕ := 21600

-- Prove that the speed converted to kilometers per hour is equal to the expected speed
theorem shuttle_speed_conversion : shuttle_speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hour :=
by
    sorry

end shuttle_speed_conversion_l301_301138


namespace find_d_l301_301419

theorem find_d 
    (a b c d : ℝ) 
    (h_a_pos : 0 < a)
    (h_b_pos : 0 < b)
    (h_c_pos : 0 < c)
    (h_d_pos : 0 < d)
    (max_val : d + a = 7)
    (min_val : d - a = 1) :
    d = 4 :=
by
  sorry

end find_d_l301_301419


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301832

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301832


namespace find_range_of_k_l301_301495

-- Define the conditions and the theorem
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

theorem find_range_of_k :
  {k : ℝ | is_ellipse k} = {k : ℝ | (-3 < k ∧ k < -1/2) ∨ (-1/2 < k ∧ k < 2)} :=
by
  sorry

end find_range_of_k_l301_301495


namespace percentage_died_by_bombardment_l301_301277

noncomputable def initial_population : ℕ := 8515
noncomputable def final_population : ℕ := 6514

theorem percentage_died_by_bombardment :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ 100) ∧
  8515 - ((x / 100) * 8515) - (15 / 100) * (8515 - ((x / 100) * 8515)) = 6514 ∧
  x = 10 :=
by
  sorry

end percentage_died_by_bombardment_l301_301277


namespace find_x_l301_301691

theorem find_x (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z)
(h₄ : x^2 / y = 3) (h₅ : y^2 / z = 4) (h₆ : z^2 / x = 5) : 
  x = (6480 : ℝ)^(1/7 : ℝ) :=
by 
  sorry

end find_x_l301_301691


namespace part1_part2_l301_301065

noncomputable def area_of_triangle (a b c : ℝ) (sinC : ℝ) : ℝ :=
  1 / 2 * a * b * sinC

-- Part 1
theorem part1 (a : ℝ) (b : ℝ) (c : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : 2 * Real.sin C = 3 * Real.sin A)
  (a_val : a = 4) (b_val : b = 5) (c_val : c = 6) : 
  area_of_triangle 4 5 (3 * Real.sqrt 7 / 8) = 15 * Real.sqrt 7 / 4 := by 
  sorry

-- Part 2
theorem part2 (a : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : ∃ (a' : ℕ), a = a') :
  a = 2 := by
  sorry

end part1_part2_l301_301065


namespace floor_abs_neg_45_7_l301_301464

theorem floor_abs_neg_45_7 : (Int.floor (Real.abs (-45.7))) = 45 :=
by
  sorry

end floor_abs_neg_45_7_l301_301464


namespace solve_for_x_l301_301931

theorem solve_for_x (x : ℝ) (h : |2000 * x + 2000| = 20 * 2000) : x = 19 ∨ x = -21 := 
by
  sorry

end solve_for_x_l301_301931


namespace vasya_wins_l301_301080

-- Define the grid size and initial setup
def grid_size : ℕ := 13
def initial_stones : ℕ := 2023

-- Define a condition that checks if a move can put a stone on the 13th cell
def can_win (position : ℕ) : Prop :=
  position = grid_size

-- Define the game logic for Petya and Vasya
def next_position (pos : ℕ) (move : ℕ) : ℕ :=
  pos + move

-- Ensure a win by always ensuring the next move does not leave Petya on positions 4, 7, 10, 13
def winning_strategy_for_vasya (current_pos : ℕ) (move : ℕ) : Prop :=
  (next_position current_pos move) ≠ 4 ∧
  (next_position current_pos move) ≠ 7 ∧
  (next_position current_pos move) ≠ 10 ∧
  (next_position current_pos move) ≠ 13

theorem vasya_wins : ∃ strategy : ℕ → ℕ → Prop,
  ∀ current_pos move, winning_strategy_for_vasya current_pos move → can_win (next_position current_pos move) :=
by
  sorry -- To be provided

end vasya_wins_l301_301080


namespace overtakes_in_16_minutes_l301_301078

def number_of_overtakes (track_length : ℕ) (speed_a : ℕ) (speed_b : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let relative_speed := speed_a - speed_b
  let time_per_overtake := track_length / relative_speed
  time_seconds / time_per_overtake

theorem overtakes_in_16_minutes :
  number_of_overtakes 200 6 4 16 = 9 :=
by
  -- We will insert calculations or detailed proof steps if needed
  sorry

end overtakes_in_16_minutes_l301_301078


namespace number_of_truthful_dwarfs_l301_301449

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l301_301449


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301834

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301834


namespace fishmonger_total_sales_l301_301566

theorem fishmonger_total_sales (first_week_sales : ℕ) (multiplier : ℕ) : 
  first_week_sales = 50 → multiplier = 3 → first_week_sales + first_week_sales * multiplier = 200 :=
by
  intros h_first h_mult
  rw [h_first, h_mult]
  simp
  sorry

end fishmonger_total_sales_l301_301566


namespace one_is_sum_of_others_l301_301928

theorem one_is_sum_of_others {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : |a - b| ≥ c) (h2 : |b - c| ≥ a) (h3 : |c - a| ≥ b) :
    a = b + c ∨ b = a + c ∨ c = a + b :=
sorry

end one_is_sum_of_others_l301_301928


namespace polynomial_factorization_l301_301306

noncomputable def polynomial_equivalence : Prop :=
  ∀ x : ℂ, (x^12 - 3*x^9 + 3*x^3 + 1) = (x + 1)^4 * (x^2 - x + 1)^4

theorem polynomial_factorization : polynomial_equivalence := by
  sorry

end polynomial_factorization_l301_301306


namespace abs_diff_x_plus_1_x_minus_2_l301_301987

theorem abs_diff_x_plus_1_x_minus_2 (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : |x + 1| - |x - 2| = -3 :=
by
  sorry

end abs_diff_x_plus_1_x_minus_2_l301_301987


namespace geometric_sequence_b_l301_301490

theorem geometric_sequence_b (a b c : Real) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : ∃ r, b = r * a ∧ c = r * b) :
  b = 1 ∨ b = -1 :=
by
  sorry

end geometric_sequence_b_l301_301490


namespace football_players_count_l301_301513

-- Define the given conditions
def total_students : ℕ := 39
def long_tennis_players : ℕ := 20
def both_sports : ℕ := 17
def play_neither : ℕ := 10

-- Define a theorem to prove the number of football players is 26
theorem football_players_count : 
  ∃ (F : ℕ), F = 26 ∧ 
  (total_students - play_neither) = (F - both_sports) + (long_tennis_players - both_sports) + both_sports :=
by {
  sorry
}

end football_players_count_l301_301513


namespace proof_problem_l301_301241

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∀ x, (a * x^2 + b * x + 2 > 0) ↔ (x ∈ Set.Ioo (-1/2 : ℝ) (1/3 : ℝ))) 

theorem proof_problem (a b : ℝ) (h : problem_statement a b) : a + b = -14 :=
sorry

end proof_problem_l301_301241


namespace pie_eating_contest_l301_301385

theorem pie_eating_contest :
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  max (max first_student second_student) third_student - 
  min (min first_student second_student) third_student = 1 / 6 :=
by
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  sorry

end pie_eating_contest_l301_301385


namespace n_plus_floor_sqrt2_plus1_pow_n_is_odd_l301_301539

theorem n_plus_floor_sqrt2_plus1_pow_n_is_odd (n : ℕ) (h : n > 0) : 
  Odd (n + ⌊(Real.sqrt 2 + 1) ^ n⌋) :=
by sorry

end n_plus_floor_sqrt2_plus1_pow_n_is_odd_l301_301539


namespace items_from_B_l301_301170

noncomputable def totalItems : ℕ := 1200
noncomputable def ratioA : ℕ := 3
noncomputable def ratioB : ℕ := 4
noncomputable def ratioC : ℕ := 5
noncomputable def totalRatio : ℕ := ratioA + ratioB + ratioC
noncomputable def sampledItems : ℕ := 60
noncomputable def numberB := sampledItems * ratioB / totalRatio

theorem items_from_B :
  numberB = 20 :=
by
  sorry

end items_from_B_l301_301170


namespace number_is_correct_l301_301048

theorem number_is_correct (x : ℝ) (h : 0.35 * x = 0.25 * 50) : x = 35.7143 :=
by 
  sorry

end number_is_correct_l301_301048


namespace solve_for_x_l301_301366

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l301_301366


namespace stratified_sampling_employees_over_50_l301_301611

theorem stratified_sampling_employees_over_50 :
  let total_employees := 500
  let employees_under_35 := 125
  let employees_35_to_50 := 280
  let employees_over_50 := 95
  let total_samples := 100
  (employees_over_50 / total_employees * total_samples) = 19 := by
  sorry

end stratified_sampling_employees_over_50_l301_301611


namespace smallest_even_consecutive_sum_l301_301381

theorem smallest_even_consecutive_sum (n : ℕ) (h_even : n % 2 = 0) (h_sum : n + (n + 2) + (n + 4) = 162) : n = 52 :=
sorry

end smallest_even_consecutive_sum_l301_301381


namespace area_of_triangle_obtuse_triangle_exists_l301_301066

-- Define the mathematical conditions given in the problem
variables {a b c : ℝ}
axiom triangle_inequality : ∀ {x y z : ℝ}, x + y > z ∧ y + z > x ∧ z + x > y
axiom sine_relation : 2 * Real.sin c = 3 * Real.sin a
axiom side_b : b = a + 1
axiom side_c : c = a + 2

-- Part 1: Prove the area of ΔABC equals the provided solution
theorem area_of_triangle : 
  2 * Real.sin c = 3 * Real.sin a → 
  b = a + 1 → 
  c = a + 2 → 
  a = 4 → 
  b = 5 → 
  c = 6 → 
  let ab_sin_c := (1 / 2) * 4 * 5 * ((3 * Real.sqrt 7) / 8) in
  ab_sin_c = 15 * Real.sqrt 7 / 4 :=
by {
  sorry, -- The proof steps are to be provided here
}

-- Part 2: Prove there exists a positive integer a such that ΔABC is obtuse with a = 2
theorem obtuse_triangle_exists :
  (∃ (a : ℕ) (h2 : 1 < a ∧ a < 3), 2 * Real.sin c = 3 * Real.sin a ∧ b = a + 1 ∧ c = a + 2) →
  ∃ a = 2 ∧ (b = a + 1 ∧ c = a + 2) :=
by {
  sorry, -- The proof steps are to be provided here
}

end area_of_triangle_obtuse_triangle_exists_l301_301066


namespace financial_outcome_l301_301209

theorem financial_outcome :
  let initial_value : ℝ := 12000
  let selling_price : ℝ := initial_value * 1.20
  let buying_price : ℝ := selling_price * 0.85
  let financial_outcome : ℝ := buying_price - initial_value
  financial_outcome = 240 :=
by
  sorry

end financial_outcome_l301_301209


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301755

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301755


namespace mrs_martin_pays_l301_301226

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def mr_martin_scoops : ℕ := 1
def mrs_martin_scoops : ℕ := 1
def children_scoops : ℕ := 2
def teenage_children_scoops : ℕ := 3

def total_cost : ℕ :=
  (mr_martin_scoops + mrs_martin_scoops) * regular_scoop_cost +
  children_scoops * kiddie_scoop_cost +
  teenage_children_scoops * double_scoop_cost

theorem mrs_martin_pays : total_cost = 32 :=
  by sorry

end mrs_martin_pays_l301_301226


namespace truthful_dwarfs_count_l301_301428

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l301_301428


namespace dwarfs_truthful_count_l301_301445

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l301_301445


namespace seeds_in_big_garden_l301_301623

-- Definitions based on conditions
def total_seeds : ℕ := 42
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 2
def seeds_planted_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden

-- Proof statement
theorem seeds_in_big_garden : total_seeds - seeds_planted_in_small_gardens = 36 :=
sorry

end seeds_in_big_garden_l301_301623


namespace prime_sum_remainder_l301_301778

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301778


namespace sufficient_but_not_necessary_condition_for_negative_root_l301_301403

def quadratic_equation (a x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem sufficient_but_not_necessary_condition_for_negative_root 
  (a : ℝ) (h : a < 0) : 
  (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) ∧ 
  (∀ a : ℝ, (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) → a ≤ 0) :=
sorry

end sufficient_but_not_necessary_condition_for_negative_root_l301_301403


namespace minimum_value_of_polynomial_l301_301472

def polynomial (x : ℝ) : ℝ := (12 - x) * (10 - x) * (12 + x) * (10 + x)

theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial x = -484 :=
by
  sorry

end minimum_value_of_polynomial_l301_301472


namespace value_of_x_plus_y_l301_301321

theorem value_of_x_plus_y (x y : ℤ) (h1 : x - y = 36) (h2 : x = 20) : x + y = 4 :=
by
  sorry

end value_of_x_plus_y_l301_301321


namespace sum_of_tens_l301_301941

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l301_301941


namespace calculate_f_at_8_l301_301690

def f (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 27 * x^2 - 24 * x - 72

theorem calculate_f_at_8 : f 8 = 952 :=
by sorry

end calculate_f_at_8_l301_301690


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301770

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301770


namespace prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l301_301105

-- Definitions of probabilities of making a shot
def p_A : ℚ := 1 / 3
def p_B : ℚ := 1 / 2

-- Number of attempts
def num_attempts : ℕ := 3

-- Probability that B makes at most 2 shots
theorem prob_B_at_most_2_shots : 
  (1 - (num_attempts.choose 3) * (p_B ^ 3) * ((1 - p_B) ^ (num_attempts - 3))) = 7 / 8 :=
by 
  sorry

-- Probability that B makes exactly 2 more shots than A
theorem prob_B_exactly_2_more_than_A : 
  (num_attempts.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ 1) * (num_attempts.choose 0) * ((1 - p_A) ^ num_attempts) +
  (num_attempts.choose 3) * (p_B ^ 3) * (num_attempts.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (num_attempts - 1)) = 1 / 6 :=
by 
  sorry

end prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l301_301105


namespace time_with_DE_only_l301_301194

theorem time_with_DE_only (d e f : ℚ) 
  (h₁ : d + e + f = 1/2) 
  (h₂ : d + f = 1/3) 
  (h₃ : e + f = 1/4) : 
  1 / (d + e) = 12 / 5 :=
begin
  sorry
end

end time_with_DE_only_l301_301194


namespace greater_number_l301_301558

theorem greater_number (x: ℕ) (h1 : 3 * x + 4 * x = 21) : 4 * x = 12 := by
  sorry

end greater_number_l301_301558


namespace polygon_sides_l301_301666

-- Define the given condition formally
def sum_of_internal_and_external_angle (n : ℕ) : ℕ :=
  (n - 2) * 180 + (1) -- This represents the sum of internal angles plus an external angle

theorem polygon_sides (n : ℕ) : 
  sum_of_internal_and_external_angle n = 1350 → n = 9 :=
by
  sorry

end polygon_sides_l301_301666


namespace min_payment_proof_max_payment_proof_expected_payment_proof_l301_301999

noncomputable def items : List ℕ := List.range 1 11 |>.map (λ n => n * 100)

def min_amount_paid : ℕ :=
  (1000 + 900 + 700 + 600 + 400 + 300 + 100)

def max_amount_paid : ℕ :=
  (1000 + 900 + 800 + 700 + 600 + 500 + 400)

def expected_amount_paid : ℚ :=
  4583 + 33 / 100

theorem min_payment_proof :
  (∑ x in (List.range 15).filter (λ x => x % 3 ≠ 0), (items.get! x : ℕ)) = min_amount_paid := by
  sorry

theorem max_payment_proof :
  (∑ x in List.range 10, if x % 3 = 0 then 0 else (items.get! x : ℕ)) = max_amount_paid := by
  sorry

theorem expected_payment_proof :
  ∑ k in items, ((k : ℚ) * (∏ m in List.range 9, (10 - m) * (9 - m) / 72)) = expected_amount_paid := by
  sorry

end min_payment_proof_max_payment_proof_expected_payment_proof_l301_301999


namespace area_of_hexagon_l301_301713

-- Definitions of the angles and side lengths
def angle_A := 120
def angle_B := 120
def angle_C := 120
def angle_D := 150

def FA := 2
def AB := 2
def BC := 2
def CD := 3
def DE := 3
def EF := 3

-- Theorem statement for the area of hexagon ABCDEF
theorem area_of_hexagon : 
  (angle_A = 120 ∧ angle_B = 120 ∧ angle_C = 120 ∧ angle_D = 150 ∧
   FA = 2 ∧ AB = 2 ∧ BC = 2 ∧ CD = 3 ∧ DE = 3 ∧ EF = 3) →
  (∃ area : ℝ, area = 7.5 * Real.sqrt 3) :=
by
  sorry

end area_of_hexagon_l301_301713


namespace Rihanna_money_left_l301_301358

theorem Rihanna_money_left (initial_money mango_count juice_count mango_price juice_price : ℕ)
  (h_initial : initial_money = 50)
  (h_mango_count : mango_count = 6)
  (h_juice_count : juice_count = 6)
  (h_mango_price : mango_price = 3)
  (h_juice_price : juice_price = 3) :
  initial_money - (mango_count * mango_price + juice_count * juice_price) = 14 :=
sorry

end Rihanna_money_left_l301_301358


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301806

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301806


namespace winning_candidate_votes_l301_301888

theorem winning_candidate_votes (T W : ℕ) (d1 d2 d3 : ℕ) 
  (hT : T = 963)
  (hd1 : d1 = 53) 
  (hd2 : d2 = 79) 
  (hd3 : d3 = 105) 
  (h_sum : T = W + (W - d1) + (W - d2) + (W - d3)) :
  W = 300 := 
by
  sorry

end winning_candidate_votes_l301_301888


namespace necessary_but_not_sufficient_l301_301126

def condition1 (a b : ℝ) : Prop :=
  a > b

def statement (a b : ℝ) : Prop :=
  a > b + 1

theorem necessary_but_not_sufficient (a b : ℝ) (h : condition1 a b) : 
  (∀ a b : ℝ, statement a b → condition1 a b) ∧ ¬ (∀ a b : ℝ, condition1 a b → statement a b) :=
by 
  -- Proof skipped
  sorry

end necessary_but_not_sufficient_l301_301126


namespace canoe_prob_calc_l301_301274

theorem canoe_prob_calc : 
  let p_left_works := 3 / 5
  let p_right_works := 3 / 5
  let p_left_breaks := 1 - p_left_works
  let p_right_breaks := 1 - p_right_works
  let p_both_work := p_left_works * p_right_works
  let p_left_works_right_breaks := p_left_works * p_right_breaks
  let p_left_breaks_right_works := p_left_breaks * p_right_works
  let p_can_row := p_both_work + p_left_works_right_breaks + p_left_breaks_right_works
  p_left_works = 3 / 5 → 
  p_right_works = 3 / 5 → 
  p_can_row = 21 / 25 :=
by
  intros
  unfold p_left_works p_right_works p_left_breaks p_right_breaks p_both_work p_left_works_right_breaks p_left_breaks_right_works p_can_row
  sorry

end canoe_prob_calc_l301_301274


namespace f_nonnegative_when_a_ge_one_l301_301921

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

noncomputable def h (a : ℝ) : ℝ := Real.log a + 1 - (1 / a)

theorem f_nonnegative_when_a_ge_one (a : ℝ) (x : ℝ) (h_a : a ≥ 1) : f a x ≥ 0 := by
  sorry  -- Placeholder for the proof.

end f_nonnegative_when_a_ge_one_l301_301921


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301743

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301743


namespace mary_pizza_order_l301_301510

theorem mary_pizza_order (p e r n : ℕ) (h1 : p = 8) (h2 : e = 7) (h3 : r = 9) :
  n = (r + e) / p → n = 2 :=
by
  sorry

end mary_pizza_order_l301_301510


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301795

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301795


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301786

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301786


namespace equations_not_equivalent_l301_301119

variable {X : Type} [Field X]
variable (A B : X → X)

theorem equations_not_equivalent (h1 : ∀ x, A x ^ 2 = B x ^ 2) (h2 : ¬∀ x, A x = B x) :
  (∃ x, A x ≠ B x ∨ A x ≠ -B x) := 
sorry

end equations_not_equivalent_l301_301119


namespace problem1_problem2_l301_301326

-- Definitions for sets A and S
def setA (x : ℝ) : Prop := -7 ≤ 2 * x - 5 ∧ 2 * x - 5 ≤ 9
def setS (x k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

-- Preliminary ranges for x
lemma range_A : ∀ x, setA x ↔ -1 ≤ x ∧ x ≤ 7 := sorry

noncomputable def k_range1 (k : ℝ) : Prop := 2 ≤ k ∧ k ≤ 4
noncomputable def k_range2 (k : ℝ) : Prop := k < 2 ∨ k > 6

-- Proof problems in Lean 4

-- First problem statement
theorem problem1 (k : ℝ) : (∀ x, setS x k → setA x) ∧ (∃ x, setS x k) → k_range1 k := sorry

-- Second problem statement
theorem problem2 (k : ℝ) : (∀ x, ¬(setA x ∧ setS x k)) → k_range2 k := sorry

end problem1_problem2_l301_301326


namespace find_B_l301_301222

noncomputable def A : ℝ := 1 / 49
noncomputable def C : ℝ := -(1 / 7)

theorem find_B :
  (∀ x : ℝ, 1 / (x^3 + 2 * x^2 - 25 * x - 50) 
            = (A / (x - 2)) + (B / (x + 5)) + (C / ((x + 5)^2))) 
    → B = - (11 / 490) :=
sorry

end find_B_l301_301222


namespace most_likely_wins_l301_301572

theorem most_likely_wins {N : ℕ} (h : N > 0) :
  let p := 1 / 2
  let n := 2 * N
  let E := n * p
  E = N := 
by
  sorry

end most_likely_wins_l301_301572


namespace cottonwood_fiber_scientific_notation_l301_301468

theorem cottonwood_fiber_scientific_notation :
  0.0000108 = 1.08 * 10^(-5)
:= by
  sorry

end cottonwood_fiber_scientific_notation_l301_301468


namespace dwarfs_truthful_count_l301_301433

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l301_301433


namespace compare_abc_l301_301640

noncomputable def a : ℝ := ∫ x in (0:ℝ)..1, x ^ (-1/3 : ℝ)
noncomputable def b : ℝ := 1 - ∫ x in (0:ℝ)..1, x ^ (1/2 : ℝ)
noncomputable def c : ℝ := ∫ x in (0:ℝ)..1, x ^ (3 : ℝ)

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l301_301640


namespace bob_makes_weekly_profit_l301_301296

def weekly_profit (p_cost p_sell : ℝ) (m_daily d_week : ℕ) : ℝ :=
  (p_sell - p_cost) * m_daily * (d_week : ℝ)

theorem bob_makes_weekly_profit :
  weekly_profit 0.75 1.5 12 7 = 63 := 
by
  sorry

end bob_makes_weekly_profit_l301_301296


namespace dwarfs_truthful_count_l301_301444

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l301_301444


namespace bridge_length_l301_301402

theorem bridge_length (length_of_train : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : 
  length_of_train = 110 → train_speed_kmph = 45 → time_seconds = 30 → 
  ∃ length_of_bridge : ℕ, length_of_bridge = 265 := by
  intros h1 h2 h3
  sorry

end bridge_length_l301_301402


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301810

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301810


namespace independence_with_replacement_not_independence_without_replacement_l301_301337

-- Definitions of the events and the probability space
def Ω : Type := {ball : ℕ // ball < 10}
def P : MeasureTheory.ProbabilityMeasure Ω := sorry
def A : Set Ω := {ball | ball < 5}
def B_with_replacement : Set Ω := {ball | ball < 5}  -- Same setup for with replacement
def B_without_replacement (a : Ω) : Set Ω :=
  if a ∈ A then {ball | ball < 4} else {ball | ball < 5}

-- Independence definitions
def independent (P : MeasureTheory.ProbabilityMeasure Ω) (A B : Set Ω) : Prop :=
  P (A ∩ B) = P A * P B

-- Problem Statement
theorem independence_with_replacement :
  independent P A B_with_replacement := sorry

theorem not_independence_without_replacement (a : Ω) :
  ¬independent P A (B_without_replacement a) := sorry

end independence_with_replacement_not_independence_without_replacement_l301_301337


namespace m_over_n_eq_l301_301377

variables (m n : ℝ)
variables (x y x1 y1 x2 y2 x0 y0 : ℝ)

-- Ellipse equation
axiom ellipse_eq : m * x^2 + n * y^2 = 1

-- Line equation
axiom line_eq : x + y = 1

-- Points M and N on the ellipse
axiom M_point : m * x1^2 + n * y1^2 = 1
axiom N_point : m * x2^2 + n * y2^2 = 1

-- Midpoint of MN is P
axiom P_midpoint : x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2

-- Slope of OP
axiom slope_OP : y0 / x0 = (Real.sqrt 2) / 2

theorem m_over_n_eq : m / n = (Real.sqrt 2) / 2 :=
sorry

end m_over_n_eq_l301_301377


namespace commodities_price_difference_l301_301553

theorem commodities_price_difference : 
  ∀ (C1 C2 : ℕ), 
    C1 = 477 → 
    C1 + C2 = 827 → 
    C1 - C2 = 127 :=
by
  intros C1 C2 h1 h2
  sorry

end commodities_price_difference_l301_301553


namespace there_exists_l_l301_301205

theorem there_exists_l (m n : ℕ) (h1 : m ≠ 0) (h2 : n ≠ 0) 
  (h3 : ∀ k : ℕ, 0 < k → Nat.gcd (17 * k - 1) m = Nat.gcd (17 * k - 1) n) :
  ∃ l : ℤ, m = (17 : ℕ) ^ l.natAbs * n := 
sorry

end there_exists_l_l301_301205


namespace add_base8_l301_301612

-- Define x and y in base 8 and their sum in base 8
def x := 24 -- base 8
def y := 157 -- base 8
def result := 203 -- base 8

theorem add_base8 : (x + y) = result := 
by sorry

end add_base8_l301_301612


namespace number_of_tens_in_sum_l301_301944

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l301_301944


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301830

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301830


namespace johns_number_is_thirteen_l301_301523

theorem johns_number_is_thirteen (x : ℕ) (h1 : 10 ≤ x) (h2 : x < 100) (h3 : ∃ a b : ℕ, 10 * a + b = 4 * x + 17 ∧ 92 ≤ 10 * b + a ∧ 10 * b + a ≤ 96) : x = 13 :=
sorry

end johns_number_is_thirteen_l301_301523


namespace division_result_l301_301396

theorem division_result (k q : ℕ) (h₁ : k % 81 = 11) (h₂ : 81 > 0) : k / 81 = q + 11 / 81 :=
  sorry

end division_result_l301_301396


namespace compute_expression_l301_301298

theorem compute_expression :
  2 * 2^5 - 8^58 / 8^56 = 0 := by
  sorry

end compute_expression_l301_301298


namespace dwarfs_truthful_count_l301_301436

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l301_301436


namespace find_sum_a_b_l301_301485

theorem find_sum_a_b (a b : ℝ) 
  (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : a + 2 * b = 0 := 
sorry

end find_sum_a_b_l301_301485


namespace factorial_divides_l301_301330

theorem factorial_divides (a : ℕ) :
  (a = 0 ∨ a = 3) ↔ (a! + (a + 2)! ∣ (a + 4)!) :=
by
  sorry

end factorial_divides_l301_301330


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301752

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301752


namespace find_digit_e_l301_301322

theorem find_digit_e (A B C D E F : ℕ) (h1 : A * 10 + B + (C * 10 + D) = A * 10 + E) (h2 : A * 10 + B - (D * 10 + C) = A * 10 + F) : E = 9 :=
sorry

end find_digit_e_l301_301322


namespace range_of_a_l301_301918

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x < 3) → (4 * a * x + 4 * (a - 3)) ≤ 0) ↔ (0 ≤ a ∧ a ≤ 3 / 4) :=
by
  sorry

end range_of_a_l301_301918


namespace archie_sod_needed_l301_301884

theorem archie_sod_needed 
  (backyard_length : ℝ) (backyard_width : ℝ) (shed_length : ℝ) (shed_width : ℝ)
  (backyard_area : backyard_length = 20 ∧ backyard_width = 13)
  (shed_area : shed_length = 3 ∧ shed_width = 5)
  : backyard_length * backyard_width - shed_length * shed_width = 245 := 
by
  unfold backyard_length backyard_width shed_length shed_width
  sorry

end archie_sod_needed_l301_301884


namespace range_of_f_l301_301008

theorem range_of_f (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 2) : -3 ≤ (3^x - 6/x) ∧ (3^x - 6/x) ≤ 6 :=
by
  sorry

end range_of_f_l301_301008


namespace car_travels_more_l301_301610

theorem car_travels_more (train_speed : ℕ) (car_speed : ℕ) (time : ℕ)
  (h1 : train_speed = 60)
  (h2 : car_speed = 2 * train_speed)
  (h3 : time = 3) :
  car_speed * time - train_speed * time = 180 :=
by
  sorry

end car_travels_more_l301_301610


namespace remainder_of_sum_of_primes_l301_301825

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301825


namespace sufficient_but_not_necessary_condition_l301_301115

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (x^2 - m * x + 1) = 0 → (m^2 - 4 < 0) = ∀ m : ℝ, -2 < m ∧ m < 2 :=
sufficient_but_not_necessary_condition sorry

end sufficient_but_not_necessary_condition_l301_301115


namespace arrangement_probability_l301_301885

theorem arrangement_probability :
  let total_arrangements := 720
  let valid_arrangements := 288
  valid_arrangements / total_arrangements = 2 / 5 :=
by sorry

end arrangement_probability_l301_301885


namespace diet_equivalence_l301_301890

variable (B E L D A : ℕ)

theorem diet_equivalence :
  (17 * B = 170 * L) →
  (100000 * A = 50 * L) →
  (10 * B = 4 * E) →
  12 * E = 600000 * A :=
sorry

end diet_equivalence_l301_301890


namespace female_participation_fraction_l301_301479

noncomputable def fraction_of_females (males_last_year : ℕ) (females_last_year : ℕ) : ℚ :=
  let males_this_year := (1.10 * males_last_year : ℚ)
  let females_this_year := (1.25 * females_last_year : ℚ)
  females_this_year / (males_this_year + females_this_year)

theorem female_participation_fraction
  (males_last_year : ℕ) (participation_increase : ℚ)
  (males_increase : ℚ) (females_increase : ℚ)
  (h_males_last_year : males_last_year = 30)
  (h_participation_increase : participation_increase = 1.15)
  (h_males_increase : males_increase = 1.10)
  (h_females_increase : females_increase = 1.25)
  (h_females_last_year : females_last_year = 15) :
  fraction_of_females males_last_year females_last_year = 19 / 52 := by
  sorry

end female_participation_fraction_l301_301479


namespace find_tangent_line_equation_l301_301032

noncomputable def tangent_line_equation (f : ℝ → ℝ) (perp_line : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  let y₀ := f x₀
  let slope_perp_to_tangent := -2
  let slope_tangent := -1 / 2
  slope_perp_to_tangent = -1 / (deriv f x₀) ∧
  x₀ = 1 ∧ y₀ = 1 ∧
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3

theorem find_tangent_line_equation :
  tangent_line_equation (fun (x : ℝ) => Real.sqrt x) (fun (x : ℝ) => -2 * x - 4) 1 := by
  sorry

end find_tangent_line_equation_l301_301032


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301765

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301765


namespace length_of_ae_l301_301003

def consecutive_points_on_line (a b c d e : ℝ) : Prop :=
  ∃ (ab bc cd de : ℝ), 
  ab = 5 ∧ 
  bc = 2 * cd ∧ 
  de = 4 ∧ 
  a + ab = b ∧ 
  b + bc = c ∧ 
  c + cd = d ∧ 
  d + de = e ∧
  a + ab + bc = c -- ensuring ac = 11

theorem length_of_ae (a b c d e : ℝ) 
  (h1 : consecutive_points_on_line a b c d e) 
  (h2 : a + 5 = b)
  (h3 : b + 2 * (c - b) = c)
  (h4 : d - c = 3)
  (h5 : d + 4 = e)
  (h6 : a + 5 + 2 * (c - b) = c) :
  e - a = 18 :=
sorry

end length_of_ae_l301_301003


namespace determine_n_l301_301863

noncomputable def polynomial (n : ℕ) : ℕ → ℕ := sorry  -- Placeholder for the actual polynomial function

theorem determine_n (n : ℕ) 
  (h_deg : ∀ a, polynomial n a = 2 → (3 ∣ a) ∨ a = 0)
  (h_deg' : ∀ a, polynomial n a = 1 → (3 ∣ (a + 2)))
  (h_deg'' : ∀ a, polynomial n a = 0 → (3 ∣ (a + 1)))
  (h_val : polynomial n (3*n+1) = 730) :
  n = 4 :=
sorry

end determine_n_l301_301863


namespace increasing_function_geq_25_l301_301663

theorem increasing_function_geq_25 {m : ℝ} 
  (h : ∀ x y : ℝ, x ≥ -2 ∧ x ≤ y → (4 * x^2 - m * x + 5) ≤ (4 * y^2 - m * y + 5)) :
  (4 * 1^2 - m * 1 + 5) ≥ 25 :=
by {
  -- Proof is omitted
  sorry
}

end increasing_function_geq_25_l301_301663


namespace remaining_fish_l301_301932

theorem remaining_fish (initial_fish : ℝ) (moved_fish : ℝ) (remaining_fish : ℝ) : initial_fish = 212.0 → moved_fish = 68.0 → remaining_fish = 144.0 → initial_fish - moved_fish = remaining_fish := by sorry

end remaining_fish_l301_301932


namespace sum_of_first_6033_terms_l301_301559

noncomputable def geometric_series_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms
  (a r : ℝ)  
  (h1 : geometric_series_sum a r 2011 = 200)
  (h2 : geometric_series_sum a r 4022 = 380) :
  geometric_series_sum a r 6033 = 542 := 
sorry

end sum_of_first_6033_terms_l301_301559


namespace largest_number_is_A_l301_301416

theorem largest_number_is_A (x y z w: ℕ):
  x = (8 * 9 + 5) → -- 85 in base 9 to decimal
  y = (2 * 6 * 6) → -- 200 in base 6 to decimal
  z = ((6 * 11) + 8) → -- 68 in base 11 to decimal
  w = 70 → -- 70 in base 10 remains 70
  max (max x y) (max z w) = x := -- 77 is the maximum
by
  sorry

end largest_number_is_A_l301_301416


namespace evaluate_floor_abs_neg_l301_301461

theorem evaluate_floor_abs_neg (x : ℝ) (h₁ : x = -45.7) : 
  floor (|x|) = 45 :=
by
  sorry

end evaluate_floor_abs_neg_l301_301461


namespace soja_book_page_count_l301_301259

theorem soja_book_page_count (P : ℕ) (h1 : P > 0) (h2 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 100) : P = 300 :=
by
  -- The Lean proof is not required, so we just add sorry to skip the proof
  sorry

end soja_book_page_count_l301_301259


namespace remainder_of_sum_of_primes_l301_301826

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301826


namespace road_time_l301_301585

theorem road_time (departure : ℕ) (arrival : ℕ) (stops : List ℕ) : 
  departure = 7 ∧ arrival = 20 ∧ stops = [25, 10, 25] → 
  ((arrival - departure) * 60 - stops.sum) / 60 = 12 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 hstops
  have h_duration : (20 - 7) * 60 = 780 := rfl
  have h_stops : stops.sum = 60 := by
    simp [hstops]
  have h_total_time_on_road : (780 - 60) / 60 = 12 := rfl
  exact h_total_time_on_road

end road_time_l301_301585


namespace solution_set_of_inequality_l301_301160

theorem solution_set_of_inequality :
  {x : ℝ | (x^2 - 2*x - 3) * (x^2 + 1) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end solution_set_of_inequality_l301_301160


namespace archie_needs_sod_l301_301883

-- Define the dimensions of the backyard
def backyard_length : ℕ := 20
def backyard_width : ℕ := 13

-- Define the dimensions of the shed
def shed_length : ℕ := 3
def shed_width : ℕ := 5

-- Statement: Prove that the area of the backyard minus the area of the shed equals 245 square yards
theorem archie_needs_sod : 
  backyard_length * backyard_width - shed_length * shed_width = 245 := 
by sorry

end archie_needs_sod_l301_301883


namespace probability_event_occurs_l301_301213

def in_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 * Real.pi

def event_occurs (x : ℝ) : Prop :=
  Real.cos (x + Real.pi / 3) + Real.sqrt 3 * Real.sin (x + Real.pi / 3) ≥ 1

theorem probability_event_occurs : 
  (∀ x, in_interval x → event_occurs x) → 
  (∃ p, p = 1/3) :=
by
  intros h
  sorry

end probability_event_occurs_l301_301213


namespace remainder_first_six_primes_div_seventh_l301_301846

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301846


namespace units_digit_3_pow_2005_l301_301305

theorem units_digit_3_pow_2005 : 
  let units_digit (n : ℕ) : ℕ := n % 10
  units_digit (3^2005) = 3 :=
by
  sorry

end units_digit_3_pow_2005_l301_301305


namespace sofia_initial_floor_l301_301706

theorem sofia_initial_floor (x : ℤ) (h1 : x + 7 - 6 + 5 = 20) : x = 14 := 
sorry

end sofia_initial_floor_l301_301706


namespace eval_floor_abs_neg_45_7_l301_301466

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l301_301466


namespace translated_graph_pass_through_origin_l301_301972

theorem translated_graph_pass_through_origin 
    (φ : ℝ) (h : 0 < φ ∧ φ < π / 2) 
    (passes_through_origin : 0 = Real.sin (-2 * φ + π / 3)) : 
    φ = π / 6 := 
sorry

end translated_graph_pass_through_origin_l301_301972


namespace min_value_l301_301906

theorem min_value (x : ℝ) (h : x > 2) : ∃ y, y = 22 ∧ 
  ∀ z, (z > 2) → (y ≤ (z^2 + 8) / (Real.sqrt (z - 2))) := 
sorry

end min_value_l301_301906


namespace scientific_notation_of_3900000000_l301_301676

theorem scientific_notation_of_3900000000 : 3900000000 = 3.9 * 10^9 :=
by 
  sorry

end scientific_notation_of_3900000000_l301_301676


namespace can_cabinet_be_moved_out_through_door_l301_301515

/-
Definitions for the problem:
- Length, width, and height of the room
- Width, height, and depth of the cabinet
- Width and height of the door
-/

structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

def room : Dimensions := { length := 4, width := 2.5, height := 2.3 }
def cabinet : Dimensions := { length := 0.6, width := 1.8, height := 2.1 }
def door : Dimensions := { length := 0.8, height := 1.9, width := 0 }

theorem can_cabinet_be_moved_out_through_door : 
  (cabinet.length ≤ door.length ∧ cabinet.width ≤ door.height) ∨ 
  (cabinet.width ≤ door.length ∧ cabinet.length ≤ door.height) 
∧ 
cabinet.height ≤ room.height ∧ cabinet.width ≤ room.width ∧ 
cabinet.length ≤ room.length → True :=
by
  sorry

end can_cabinet_be_moved_out_through_door_l301_301515


namespace number_of_tens_in_sum_l301_301955

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l301_301955


namespace no_real_solutions_sufficient_not_necessary_l301_301114

theorem no_real_solutions_sufficient_not_necessary (m : ℝ) : 
  (|m| < 1) → (m^2 < 4) :=
by
  sorry

end no_real_solutions_sufficient_not_necessary_l301_301114


namespace complex_mul_example_l301_301596

theorem complex_mul_example (i : ℝ) (h : i^2 = -1) : (⟨2, 2 * i⟩ : ℂ) * (⟨1, -2 * i⟩) = ⟨6, -2 * i⟩ :=
by
  sorry

end complex_mul_example_l301_301596


namespace worker_surveys_per_week_l301_301139

theorem worker_surveys_per_week :
  let regular_rate := 30
  let cellphone_rate := regular_rate + 0.20 * regular_rate
  let surveys_with_cellphone := 50
  let earnings := 3300
  cellphone_rate = regular_rate + 0.20 * regular_rate →
  earnings = surveys_with_cellphone * cellphone_rate →
  regular_rate = 30 →
  surveys_with_cellphone = 50 →
  earnings = 3300 →
  surveys_with_cellphone = 50 := sorry

end worker_surveys_per_week_l301_301139


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301744

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301744


namespace find_sum_a_b_l301_301484

theorem find_sum_a_b (a b : ℝ) 
  (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : a + 2 * b = 0 := 
sorry

end find_sum_a_b_l301_301484


namespace solve_fractional_equation_l301_301100

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) :
  1 / x = 2 / (x + 1) → x = 1 :=
by
  sorry

end solve_fractional_equation_l301_301100


namespace compute_pqr_l301_301660

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 30) 
  (h_equation : 1 / p + 1 / q + 1 / r + 420 / (p * q * r) = 1) : 
  p * q * r = 576 :=
sorry

end compute_pqr_l301_301660


namespace prob_bigger_number_correct_l301_301295

def bernardo_picks := {n | 1 ≤ n ∧ n ≤ 10}
def silvia_picks := {n | 1 ≤ n ∧ n ≤ 8}

noncomputable def prob_bigger_number : ℚ :=
  let prob_bern_picks_10 : ℚ := 3 / 10
  let prob_bern_not_10_larger_silvia : ℚ := 55 / 112
  let prob_bern_not_picks_10 : ℚ := 7 / 10
  prob_bern_picks_10 + prob_bern_not_10_larger_silvia * prob_bern_not_picks_10

theorem prob_bigger_number_correct :
  prob_bigger_number = 9 / 14 := by
  sorry

end prob_bigger_number_correct_l301_301295


namespace quadratic_sum_l301_301251

theorem quadratic_sum (x : ℝ) (h : x^2 = 16*x - 9) : x = 8 ∨ x = 9 := sorry

end quadratic_sum_l301_301251


namespace cottonwood_fiber_diameter_in_scientific_notation_l301_301470

theorem cottonwood_fiber_diameter_in_scientific_notation:
  (∃ (a : ℝ) (n : ℤ), 0.0000108 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10) → (0.0000108 = 1.08 * 10 ^ (-5)) :=
by
  sorry

end cottonwood_fiber_diameter_in_scientific_notation_l301_301470


namespace ellipse_range_k_l301_301493

theorem ellipse_range_k (k : ℝ) : 
  (∃ (x y : ℝ) (hk : \(\frac{x^2}{3+k} + \frac{y^2}{2-k} = 1\)), (3 + k > 0) ∧ (2 - k > 0) ∧ (3+k ≠ 2-k)) ↔ 
  k ∈ set.Ioo (-3) (-1/2) ∪ set.Ioo (-1/2) 2 := 
sorry

end ellipse_range_k_l301_301493


namespace acute_angle_l301_301502

variables (x : ℝ)

def a : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, 3)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem acute_angle (x : ℝ) : 
  (-2 / 3 < x) → x ≠ -2 / 3 → dot_product (2, x) (1, 3) > 0 :=
by
  intros h1 h2
  sorry

end acute_angle_l301_301502


namespace area_of_large_rectangle_l301_301093

-- Define the given areas for the sub-shapes
def shaded_square_area : ℝ := 4
def bottom_rectangle_area : ℝ := 2
def right_rectangle_area : ℝ := 6

-- Prove the total area of the large rectangle EFGH is 12 square inches
theorem area_of_large_rectangle : shaded_square_area + bottom_rectangle_area + right_rectangle_area = 12 := 
by 
sorry

end area_of_large_rectangle_l301_301093


namespace A_inter_complement_B_eq_01_l301_301925

open Set

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | x ≥ 1}
def complement_B : Set ℝ := U \ B

theorem A_inter_complement_B_eq_01 : A ∩ complement_B = (Set.Ioo 0 1) := 
by 
  sorry

end A_inter_complement_B_eq_01_l301_301925


namespace ratio_of_functions_l301_301331

def f (x : ℕ) : ℕ := 3 * x + 4
def g (x : ℕ) : ℕ := 4 * x - 3

theorem ratio_of_functions :
  f (g (f 3)) * 121 = 151 * g (f (g 3)) :=
by
  sorry

end ratio_of_functions_l301_301331


namespace books_count_l301_301576

theorem books_count (books_per_box : ℕ) (boxes : ℕ) (total_books : ℕ) 
  (h1 : books_per_box = 3)
  (h2 : boxes = 8)
  (h3 : total_books = books_per_box * boxes) : 
  total_books = 24 := 
by 
  rw [h1, h2] at h3
  exact h3

end books_count_l301_301576


namespace perfume_price_decrease_l301_301288

theorem perfume_price_decrease :
  let original_price := 1200
  let increased_price := original_price * (1 + 10 / 100)
  let final_price := increased_price * (1 - 15 / 100)
  original_price - final_price = 78 := by
  calc
  original_price - final_price = ...
  sorry

end perfume_price_decrease_l301_301288


namespace matrix_identity_l301_301527

noncomputable def N : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![-2, 1]]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  N * N = 4 • N + -11 • I :=
by
  sorry

end matrix_identity_l301_301527


namespace loan_amounts_l301_301279

theorem loan_amounts (x y : ℝ) (h1 : x + y = 50) (h2 : 0.1 * x + 0.08 * y = 4.4) : x = 20 ∧ y = 30 := by
  sorry

end loan_amounts_l301_301279


namespace lesser_solution_of_quadratic_eq_l301_301579

theorem lesser_solution_of_quadratic_eq : ∃ x ∈ {x | x^2 + 10*x - 24 = 0}, x = -12 :=
by 
  sorry

end lesser_solution_of_quadratic_eq_l301_301579


namespace value_of_y_for_absolute_value_eq_zero_l301_301253

theorem value_of_y_for_absolute_value_eq_zero :
  ∃ (y : ℚ), |(2:ℚ) * y - 3| ≤ 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_for_absolute_value_eq_zero_l301_301253


namespace perfume_price_decrease_l301_301287

theorem perfume_price_decrease :
  let original_price := 1200
  let increased_price := original_price * (1 + 10 / 100)
  let final_price := increased_price * (1 - 15 / 100)
  original_price - final_price = 78 := by
  calc
  original_price - final_price = ...
  sorry

end perfume_price_decrease_l301_301287


namespace weight_first_watermelon_l301_301684

-- We define the total weight and the weight of the second watermelon
def total_weight := 14.02
def second_watermelon := 4.11

-- We need to prove that the weight of the first watermelon is 9.91 pounds
theorem weight_first_watermelon : total_weight - second_watermelon = 9.91 := by
  -- Insert mathematical steps here (omitted in this case)
  sorry

end weight_first_watermelon_l301_301684


namespace total_insects_eaten_l301_301269

-- Definitions from the conditions
def numGeckos : Nat := 5
def insectsPerGecko : Nat := 6
def numLizards : Nat := 3
def insectsPerLizard : Nat := insectsPerGecko * 2

-- Theorem statement, proving total insects eaten is 66
theorem total_insects_eaten : numGeckos * insectsPerGecko + numLizards * insectsPerLizard = 66 := by
  sorry

end total_insects_eaten_l301_301269


namespace right_heavier_combinations_l301_301077

open Finset

def weights : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def six_combinations := weights.powerset.filter (λ s, s.card = 6)
def three_combinations := weights.powerset.filter (λ s, s.card = 3)

def sum_weights (w : Finset ℕ) : ℕ := w.sum id

def valid_combination (left right : Finset ℕ) : Prop :=
  (weights = left ∪ right) ∧ (left.disjoint right) ∧ (sum_weights right > sum_weights left)

theorem right_heavier_combinations : (∃ (right : Finset ℕ), right ∈ three_combinations ∧
  ∀ left, left = weights \ right → valid_combination left right) ↔ 2 :=
by
  sorry

end right_heavier_combinations_l301_301077


namespace chocolates_bought_l301_301934

theorem chocolates_bought (cost_price selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : cost_price * 24 = selling_price)
  (h2 : gain_percent = 83.33333333333334)
  (h3 : selling_price = cost_price * 24 * (1 + gain_percent / 100)) :
  cost_price * 44 = selling_price :=
by
  sorry

end chocolates_bought_l301_301934


namespace contestant_advancing_probability_l301_301970

noncomputable def probability_correct : ℝ := 0.8
noncomputable def probability_incorrect : ℝ := 1 - probability_correct

def sequence_pattern (q1 q2 q3 q4 : Bool) : Bool :=
  -- Pattern INCORRECT, CORRECT, CORRECT, CORRECT
  q1 == false ∧ q2 == true ∧ q3 == true ∧ q4 == true

def probability_pattern (p_corr p_incorr : ℝ) : ℝ :=
  p_incorr * p_corr * p_corr * p_corr

theorem contestant_advancing_probability :
  (probability_pattern probability_correct probability_incorrect = 0.1024) :=
by
  -- Proof required here
  sorry

end contestant_advancing_probability_l301_301970


namespace cost_price_is_1500_l301_301092

-- Definitions for the given conditions
def selling_price : ℝ := 1200
def loss_percentage : ℝ := 20

-- Define the cost price such that the loss percentage condition is satisfied
def cost_price (C : ℝ) : Prop :=
  loss_percentage = ((C - selling_price) / C) * 100

-- The proof problem to be solved: 
-- Prove that the cost price of the radio is Rs. 1500
theorem cost_price_is_1500 : ∃ C, cost_price C ∧ C = 1500 :=
by
  sorry

end cost_price_is_1500_l301_301092


namespace lcm_45_75_l301_301028

theorem lcm_45_75 : Nat.lcm 45 75 = 225 :=
by
  sorry

end lcm_45_75_l301_301028


namespace part_one_part_two_l301_301990

variable {x : ℝ} {m : ℝ}

-- Question 1
theorem part_one (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) : -4 < m ∧ m <= 0 :=
sorry

-- Question 2
theorem part_two (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → mx^2 - mx - 1 > -m + x - 1) : m > 1 :=
sorry

end part_one_part_two_l301_301990


namespace photocopy_distribution_l301_301657

-- Define the problem setting
variables {n k : ℕ}

-- Define the theorem stating the problem
theorem photocopy_distribution :
  ∀ n k : ℕ, (n > 0) → 
  (k + n).choose (n - 1) = (k + n - 1).choose (n - 1) :=
by sorry

end photocopy_distribution_l301_301657


namespace cubes_squares_problem_l301_301047

theorem cubes_squares_problem (h1 : 2^3 - 7^2 = 1) (h2 : 3^3 - 6^2 = 9) (h3 : 5^3 - 9^2 = 16) : 4^3 - 8^2 = 0 := 
by
  sorry

end cubes_squares_problem_l301_301047


namespace find_ellipse_l301_301491

-- Define the ellipse and conditions
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the focus points
def focus (a b c : ℝ) : Prop :=
  c^2 = a^2 - b^2

-- Define the range condition
def range_condition (a b c : ℝ) : Prop :=
  let min_val := b^2 - c^2;
  let max_val := a^2 - c^2;
  min_val = -3 ∧ max_val = 3

-- Prove the equation of the ellipse
theorem find_ellipse (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (ellipse a b a_pos b_pos ∧ focus a b c ∧ range_condition a b c) →
  (a^2 = 9 ∧ b^2 = 3) :=
by
  sorry

end find_ellipse_l301_301491


namespace mrs_hilt_apples_l301_301356

theorem mrs_hilt_apples (hours : ℕ := 3) (rate : ℕ := 5) : 
  (rate * hours) = 15 := 
by sorry

end mrs_hilt_apples_l301_301356


namespace least_number_of_stamps_is_6_l301_301421

noncomputable def exist_stamps : Prop :=
∃ (c f : ℕ), 5 * c + 7 * f = 40 ∧ c + f = 6

theorem least_number_of_stamps_is_6 : exist_stamps :=
sorry

end least_number_of_stamps_is_6_l301_301421


namespace f_equals_one_l301_301094

-- Define the functions f, g, h with the given properties

def f : ℕ → ℕ := sorry
def g : ℕ → ℕ := sorry
def h : ℕ → ℕ := sorry

-- Condition 1: h is injective
axiom h_injective : ∀ {a b : ℕ}, h a = h b → a = b

-- Condition 2: g is surjective
axiom g_surjective : ∀ n : ℕ, ∃ m : ℕ, g m = n

-- Condition 3: Definition of f in terms of g and h
axiom f_def : ∀ n : ℕ, f n = g n - h n + 1

-- Prove that f(n) = 1 for all n ∈ ℕ
theorem f_equals_one : ∀ n : ℕ, f n = 1 := by
  sorry

end f_equals_one_l301_301094


namespace find_a16_l301_301649

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n ≥ 1, a (n + 1) = 1 - 1 / a n

theorem find_a16 (a : ℕ → ℝ) (h : seq a) : a 16 = 1 / 2 :=
sorry

end find_a16_l301_301649


namespace multiply_1546_by_100_l301_301211

theorem multiply_1546_by_100 : 15.46 * 100 = 1546 :=
by
  sorry

end multiply_1546_by_100_l301_301211


namespace votes_cast_l301_301338

theorem votes_cast (total_votes : ℕ) 
  (h1 : (3/8 : ℚ) * total_votes = 45)
  (h2 : (1/4 : ℚ) * total_votes = (1/4 : ℚ) * 120) : 
  total_votes = 120 := 
by
  sorry

end votes_cast_l301_301338


namespace min_value_z_l301_301652

theorem min_value_z (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  ∃ z_min, z_min = (x + 1 / x) * (y + 1 / y) ∧ z_min = 33 / 4 :=
sorry

end min_value_z_l301_301652


namespace nicky_cards_value_l301_301699

theorem nicky_cards_value 
  (x : ℝ)
  (h : 21 = 2 * x + 5) : 
  x = 8 := by
  sorry

end nicky_cards_value_l301_301699


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301763

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301763


namespace positive_integer_solutions_l301_301220

theorem positive_integer_solutions
  (x : ℤ) :
  (5 + 3 * x < 13) ∧ ((x + 2) / 3 - (x - 1) / 2 <= 2) →
  (x = 1 ∨ x = 2) :=
by
  sorry

end positive_integer_solutions_l301_301220


namespace students_between_min_and_hos_l301_301127

theorem students_between_min_and_hos
  (total_students : ℕ)
  (minyoung_left_position : ℕ)
  (hoseok_right_position : ℕ)
  (total_students_eq : total_students = 13)
  (minyoung_left_position_eq : minyoung_left_position = 8)
  (hoseok_right_position_eq : hoseok_right_position = 9) :
  (minyoung_left_position - (total_students - hoseok_right_position + 1) - 1) = 2 := 
by
  sorry

end students_between_min_and_hos_l301_301127


namespace samantha_exam_score_l301_301704

theorem samantha_exam_score :
  ∀ (q1 q2 q3 : ℕ) (s1 s2 s3 : ℚ),
  q1 = 30 → q2 = 50 → q3 = 20 →
  s1 = 0.75 → s2 = 0.8 → s3 = 0.65 →
  (22.5 + 40 + 2 * (0.65 * 20)) / (30 + 50 + 2 * 20) = 0.7375 :=
by
  intros q1 q2 q3 s1 s2 s3 hq1 hq2 hq3 hs1 hs2 hs3
  sorry

end samantha_exam_score_l301_301704


namespace sum_of_tens_l301_301948

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l301_301948


namespace floor_sqrt_sum_eq_floor_sqrt_expr_l301_301703

-- Proof problem definition
theorem floor_sqrt_sum_eq_floor_sqrt_expr (n : ℕ) : 
  (Int.floor (Real.sqrt n + Real.sqrt (n + 1))) = (Int.floor (Real.sqrt (4 * n + 2))) := 
sorry

end floor_sqrt_sum_eq_floor_sqrt_expr_l301_301703


namespace little_john_friends_share_l301_301075

-- Noncomputable definition for dealing with reals
noncomputable def amount_given_to_each_friend :=
  let total_initial := 7.10
  let total_left := 4.05
  let spent_on_sweets := 1.05
  let total_given_away := total_initial - total_left
  let total_given_to_friends := total_given_away - spent_on_sweets
  total_given_to_friends / 2

-- The theorem stating the result
theorem little_john_friends_share :
  amount_given_to_each_friend = 1.00 :=
by
  sorry

end little_john_friends_share_l301_301075


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301754

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301754


namespace square_side_length_l301_301290

theorem square_side_length (a b : ℕ) (h : a = 9) (h' : b = 16) (A : ℕ) (h1: A = a * b) :
  ∃ (s : ℕ), s * s = A ∧ s = 12 :=
by
  sorry

end square_side_length_l301_301290


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l301_301547

theorem solve_eq1 : ∀ (x : ℝ), x^2 - 5 * x = 0 ↔ x = 0 ∨ x = 5 :=
by sorry

theorem solve_eq2 : ∀ (x : ℝ), (2 * x + 1)^2 = 4 ↔ x = -3 / 2 ∨ x = 1 / 2 :=
by sorry

theorem solve_eq3 : ∀ (x : ℝ), x * (x - 1) + 3 * (x - 1) = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem solve_eq4 : ∀ (x : ℝ), x^2 - 2 * x - 8 = 0 ↔ x = -2 ∨ x = 4 :=
by sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l301_301547


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301790

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301790


namespace train_speed_l301_301879

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) (total_distance : ℝ) 
    (speed_mps : ℝ) (speed_kmph : ℝ) 
    (h1 : train_length = 360) 
    (h2 : bridge_length = 140) 
    (h3 : time = 34.61538461538461) 
    (h4 : total_distance = train_length + bridge_length) 
    (h5 : speed_mps = total_distance / time) 
    (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 52 := 
by 
  sorry

end train_speed_l301_301879


namespace base6_addition_problem_l301_301621

-- Definitions to capture the base-6 addition problem components.
def base6₀ := 0
def base6₁ := 1
def base6₂ := 2
def base6₃ := 3
def base6₄ := 4
def base6₅ := 5

-- The main hypothesis about the base-6 addition
theorem base6_addition_problem (diamond : ℕ) (h : diamond ∈ [base6₀, base6₁, base6₂, base6₃, base6₄, base6₅]) :
  ((diamond + base6₅) % 6 = base6₃ ∨ (diamond + base6₅) % 6 = (base6₃ + 6 * 1 % 6)) ∧
  (diamond + base6₂ + base6₂ = diamond % 6) →
  diamond = base6₄ :=
sorry

end base6_addition_problem_l301_301621


namespace total_fish_sold_l301_301565

-- Define the conditions
def w1 : ℕ := 50
def w2 : ℕ := 3 * w1

-- Define the statement to prove
theorem total_fish_sold : w1 + w2 = 200 := by
  -- Insert the proof here 
  -- (proof omitted as per the instructions)
  sorry

end total_fish_sold_l301_301565


namespace lassie_original_bones_l301_301685

variable (B : ℕ) -- B is the number of bones Lassie started with

-- Conditions translated into Lean statements
def eats_half_on_saturday (B : ℕ) : ℕ := B / 2
def receives_ten_more_on_sunday (B : ℕ) : ℕ := eats_half_on_saturday B + 10
def total_bones_after_sunday (B : ℕ) : Prop := receives_ten_more_on_sunday B = 35

-- Proof goal: B is equal to 50 given the conditions
theorem lassie_original_bones :
  total_bones_after_sunday B → B = 50 :=
sorry

end lassie_original_bones_l301_301685


namespace number_of_subsets_including_1_and_10_l301_301073

def A : Set ℕ := {a : ℕ | ∃ x y z : ℕ, a = 2^x * 3^y * 5^z}
def B : Set ℕ := {b : ℕ | b ∈ A ∧ 1 ≤ b ∧ b ≤ 10}

theorem number_of_subsets_including_1_and_10 :
  ∃ (s : Finset (Finset ℕ)), (∀ x ∈ s, 1 ∈ x ∧ 10 ∈ x) ∧ s.card = 128 := by
  sorry

end number_of_subsets_including_1_and_10_l301_301073


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301735

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301735


namespace smallest_a_mod_remainders_l301_301050

theorem smallest_a_mod_remainders:
  (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], 2521 % d = 1) ∧
  (∀ n : ℕ, ∃ a : ℕ, a = 2520 * n + 1 ∧ (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], a % d = 1)) :=
by
  sorry

end smallest_a_mod_remainders_l301_301050
