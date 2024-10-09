import Mathlib

namespace larger_number_is_17_l2251_225196

noncomputable def x : ℤ := 17
noncomputable def y : ℤ := 12

def sum_condition : Prop := x + y = 29
def diff_condition : Prop := x - y = 5

theorem larger_number_is_17 (h_sum : sum_condition) (h_diff : diff_condition) : x = 17 :=
by {
  sorry
}

end larger_number_is_17_l2251_225196


namespace daniel_fraction_l2251_225177

theorem daniel_fraction (A B C D : Type) (money : A → ℝ) 
  (adriano bruno cesar daniel : A)
  (h1 : money daniel = 0)
  (given_amount : ℝ)
  (h2 : money adriano = 5 * given_amount)
  (h3 : money bruno = 4 * given_amount)
  (h4 : money cesar = 3 * given_amount)
  (h5 : money daniel = (1 / 5) * money adriano + (1 / 4) * money bruno + (1 / 3) * money cesar) :
  money daniel / (money adriano + money bruno + money cesar) = 1 / 4 := 
by
  sorry

end daniel_fraction_l2251_225177


namespace bounds_for_f3_l2251_225156

variable (a c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 - c

theorem bounds_for_f3 (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
                      (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end bounds_for_f3_l2251_225156


namespace thirteen_power_1997_tens_digit_l2251_225185

def tens_digit (n : ℕ) := (n / 10) % 10

theorem thirteen_power_1997_tens_digit :
  tens_digit (13 ^ 1997 % 100) = 5 := by
  sorry

end thirteen_power_1997_tens_digit_l2251_225185


namespace tan_simplify_l2251_225198

theorem tan_simplify (α : ℝ) (h : Real.tan α = 1 / 2) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = - 3 / 4 :=
by
  sorry

end tan_simplify_l2251_225198


namespace balloon_altitude_l2251_225188

theorem balloon_altitude 
  (temp_diff_per_1000m : ℝ)
  (altitude_temp : ℝ) 
  (ground_temp : ℝ)
  (altitude : ℝ) 
  (h1 : temp_diff_per_1000m = 6) 
  (h2 : altitude_temp = -2)
  (h3 : ground_temp = 5) :
  altitude = 7/6 :=
by sorry

end balloon_altitude_l2251_225188


namespace time_to_fill_pool_l2251_225160

noncomputable def slower_pump_rate : ℝ := 1 / 12.5
noncomputable def faster_pump_rate : ℝ := 1.5 * slower_pump_rate
noncomputable def combined_rate : ℝ := slower_pump_rate + faster_pump_rate

theorem time_to_fill_pool : (1 / combined_rate) = 5 := 
by
  sorry

end time_to_fill_pool_l2251_225160


namespace eighth_term_geometric_seq_l2251_225191

theorem eighth_term_geometric_seq (a1 a2 : ℚ) (a1_val : a1 = 3) (a2_val : a2 = 9 / 2) :
  (a1 * (a2 / a1)^(7) = 6561 / 128) :=
  by
    sorry

end eighth_term_geometric_seq_l2251_225191


namespace height_of_smaller_cone_is_18_l2251_225165

theorem height_of_smaller_cone_is_18
  (height_frustum : ℝ)
  (area_larger_base : ℝ)
  (area_smaller_base : ℝ) :
  let R := (area_larger_base / π).sqrt
  let r := (area_smaller_base / π).sqrt
  let ratio := r / R
  let H := height_frustum / (1 - ratio)
  let h := ratio * H
  height_frustum = 18 ∧ area_larger_base = 400 * π ∧ area_smaller_base = 100 * π
  → h = 18 := by
  sorry

end height_of_smaller_cone_is_18_l2251_225165


namespace fixed_point_coordinates_l2251_225112

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  2 * a^(x + 1) - 3

theorem fixed_point_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  sorry

end fixed_point_coordinates_l2251_225112


namespace compare_fractions_l2251_225130

variables {a b : ℝ}

theorem compare_fractions (h : a + b > 0) : 
  (a / (b^2)) + (b / (a^2)) ≥ (1 / a) + (1 / b) :=
sorry

end compare_fractions_l2251_225130


namespace josh_bought_6_CDs_l2251_225120

theorem josh_bought_6_CDs 
  (numFilms : ℕ)   (numBooks : ℕ) (numCDs : ℕ)
  (costFilm : ℕ)   (costBook : ℕ) (costCD : ℕ)
  (totalSpent : ℕ) :
  numFilms = 9 → 
  numBooks = 4 → 
  costFilm = 5 → 
  costBook = 4 → 
  costCD = 3 → 
  totalSpent = 79 → 
  numCDs = (totalSpent - numFilms * costFilm - numBooks * costBook) / costCD → 
  numCDs = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  exact h7

end josh_bought_6_CDs_l2251_225120


namespace larger_integer_is_21_l2251_225144

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l2251_225144


namespace smallest_integer_l2251_225133

theorem smallest_integer {x y z : ℕ} (h1 : 2*y = x) (h2 : 3*y = z) (h3 : x + y + z = 60) : y = 6 :=
by
  sorry

end smallest_integer_l2251_225133


namespace winning_strategy_l2251_225141

theorem winning_strategy (n : ℕ) (take_stones : ℕ → Prop) :
  n = 13 ∧ (∀ k, (k = 1 ∨ k = 2) → take_stones k) →
  (take_stones 12 ∨ take_stones 9 ∨ take_stones 6 ∨ take_stones 3) :=
by sorry

end winning_strategy_l2251_225141


namespace triangle_area_decrease_l2251_225134

theorem triangle_area_decrease (B H : ℝ) : 
  let A_original := (B * H) / 2
  let H_new := 0.60 * H
  let B_new := 1.40 * B
  let A_new := (B_new * H_new) / 2
  A_new = 0.42 * A_original :=
by
  sorry

end triangle_area_decrease_l2251_225134


namespace remainder_when_divided_by_6_l2251_225179

theorem remainder_when_divided_by_6 :
  ∃ (n : ℕ), (∃ k : ℕ, n = 3 * k + 2 ∧ ∃ m : ℕ, k = 4 * m + 3) → n % 6 = 5 :=
by
  sorry

end remainder_when_divided_by_6_l2251_225179


namespace advertisement_broadcasting_methods_l2251_225127

/-- A TV station is broadcasting 5 different advertisements.
There are 3 different commercial advertisements.
There are 2 different Olympic promotional advertisements.
The last advertisement must be an Olympic promotional advertisement.
The two Olympic promotional advertisements cannot be broadcast consecutively.
Prove that the total number of different broadcasting methods is 18. -/
theorem advertisement_broadcasting_methods : 
  ∃ (arrangements : ℕ), arrangements = 18 := sorry

end advertisement_broadcasting_methods_l2251_225127


namespace f_2015_l2251_225135

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 3) = f x
axiom f_interval : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = 2^x

theorem f_2015 : f 2015 = -2 := sorry

end f_2015_l2251_225135


namespace y_value_is_32_l2251_225118

-- Define the conditions
variables (y : ℝ) (hy_pos : y > 0) (hy_eq : y^2 = 1024)

-- State the theorem
theorem y_value_is_32 : y = 32 :=
by
  -- The proof will be written here
  sorry

end y_value_is_32_l2251_225118


namespace scrap_cookie_radius_is_correct_l2251_225174

noncomputable def radius_of_scrap_cookie (large_radius small_radius : ℝ) (number_of_cookies : ℕ) : ℝ :=
  have large_area : ℝ := Real.pi * large_radius^2
  have small_area : ℝ := Real.pi * small_radius^2
  have total_small_area : ℝ := small_area * number_of_cookies
  have scrap_area : ℝ := large_area - total_small_area
  Real.sqrt (scrap_area / Real.pi)

theorem scrap_cookie_radius_is_correct :
  radius_of_scrap_cookie 8 2 9 = 2 * Real.sqrt 7 :=
sorry

end scrap_cookie_radius_is_correct_l2251_225174


namespace percentage_increase_l2251_225162

theorem percentage_increase (N : ℝ) (P : ℝ) (h1 : N + (P / 100) * N - (N - 25 / 100 * N) = 30) (h2 : N = 80) : P = 12.5 :=
by
  sorry

end percentage_increase_l2251_225162


namespace ronald_laundry_frequency_l2251_225103

variable (Tim_laundry_frequency Ronald_laundry_frequency : ℕ)

theorem ronald_laundry_frequency :
  (Tim_laundry_frequency = 9) →
  (18 % Ronald_laundry_frequency = 0) →
  (18 % Tim_laundry_frequency = 0) →
  (Ronald_laundry_frequency ≠ 1) →
  (Ronald_laundry_frequency ≠ 18) →
  (Ronald_laundry_frequency ≠ 9) →
  (Ronald_laundry_frequency = 3) :=
by
  intros hTim hRonaldMultiple hTimMultiple hNot1 hNot18 hNot9
  sorry

end ronald_laundry_frequency_l2251_225103


namespace find_x_plus_y_l2251_225142

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := 
sorry

end find_x_plus_y_l2251_225142


namespace unique_two_digit_integer_l2251_225157

theorem unique_two_digit_integer (s : ℕ) (hs : s > 9 ∧ s < 100) (h : 13 * s ≡ 42 [MOD 100]) : s = 34 :=
by sorry

end unique_two_digit_integer_l2251_225157


namespace inequality_not_always_true_l2251_225143

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) : ¬ (∀ c, (a - b) / c > 0) := 
sorry

end inequality_not_always_true_l2251_225143


namespace cubical_pyramidal_segment_volume_and_area_l2251_225186

noncomputable def volume_and_area_sum (a : ℝ) : ℝ :=
  (1/4 * (9 + 27 * Real.sqrt 13))

theorem cubical_pyramidal_segment_volume_and_area :
  ∀ a : ℝ, a = 3 → volume_and_area_sum a = (9/2 + 27 * Real.sqrt 13 / 8) := by
  intro a ha
  sorry

end cubical_pyramidal_segment_volume_and_area_l2251_225186


namespace value_of_a7_minus_a8_l2251_225173

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem value_of_a7_minus_a8
  (h_seq: arithmetic_sequence a d)
  (h_sum: a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - a 8 = d :=
sorry

end value_of_a7_minus_a8_l2251_225173


namespace imo_34_l2251_225111

-- Define the input conditions
variables (R r ρ : ℝ)

-- The main theorem we need to prove
theorem imo_34 { R r ρ : ℝ } (hR : R = 1) : 
  ρ ≤ 1 - (1/3) * (1 + r)^2 :=
sorry

end imo_34_l2251_225111


namespace new_songs_added_l2251_225122

-- Define the initial, deleted, and final total number of songs as constants
def initial_songs : ℕ := 8
def deleted_songs : ℕ := 5
def total_songs_now : ℕ := 33

-- Define and prove the number of new songs added
theorem new_songs_added : total_songs_now - (initial_songs - deleted_songs) = 30 :=
by
  sorry

end new_songs_added_l2251_225122


namespace entry_cost_proof_l2251_225152

variable (hitting_rate : ℕ → ℝ)
variable (entry_cost : ℝ)
variable (total_hits : ℕ)
variable (money_lost : ℝ)

-- Conditions
axiom hitting_rate_condition : hitting_rate 200 = 0.025
axiom total_hits_condition : total_hits = 300
axiom money_lost_condition : money_lost = 7.5

-- Question: Prove that the cost to enter the contest equals $10.00
theorem entry_cost_proof : entry_cost = 10 := by
  sorry

end entry_cost_proof_l2251_225152


namespace solve_linear_eq_l2251_225158

theorem solve_linear_eq (x y : ℤ) : 2 * x + 3 * y = 0 ↔ (x, y) = (3, -2) := sorry

end solve_linear_eq_l2251_225158


namespace product_of_numbers_l2251_225181

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 :=
by
  sorry

end product_of_numbers_l2251_225181


namespace number_of_points_marked_l2251_225151

theorem number_of_points_marked (a₁ a₂ b₁ b₂ : ℕ) 
  (h₁ : a₁ * a₂ = 50) (h₂ : b₁ * b₂ = 56) (h₃ : a₁ + a₂ = b₁ + b₂) : 
  (a₁ + a₂ + 1 = 16) :=
sorry

end number_of_points_marked_l2251_225151


namespace snacks_displayed_at_dawn_l2251_225101

variable (S : ℝ)
variable (SoldMorning : ℝ)
variable (SoldAfternoon : ℝ)

axiom cond1 : SoldMorning = (3 / 5) * S
axiom cond2 : SoldAfternoon = 180
axiom cond3 : SoldMorning = SoldAfternoon

theorem snacks_displayed_at_dawn : S = 300 :=
by
  sorry

end snacks_displayed_at_dawn_l2251_225101


namespace sum_divisible_by_17_l2251_225178

theorem sum_divisible_by_17 :
    (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 17 = 0 := 
by 
  sorry

end sum_divisible_by_17_l2251_225178


namespace ratio_of_girls_with_long_hair_l2251_225180

theorem ratio_of_girls_with_long_hair (total_people boys girls short_hair long_hair : ℕ)
  (h1 : total_people = 55)
  (h2 : boys = 30)
  (h3 : girls = total_people - boys)
  (h4 : short_hair = 10)
  (h5 : long_hair = girls - short_hhair) :
  long_hair / gcd long_hair girls = 3 ∧ girls / gcd long_hair girls = 5 := 
by {
  -- This placeholder indicates where the proof should be.
  sorry
}

end ratio_of_girls_with_long_hair_l2251_225180


namespace eval_expression_l2251_225187

theorem eval_expression : 
  3000^3 - 2998 * 3000^2 - 2998^2 * 3000 + 2998^3 = 23992 := 
by 
  sorry

end eval_expression_l2251_225187


namespace binary_addition_l2251_225102

theorem binary_addition (a b : ℕ) :
  (a = (2^0 + 2^2 + 2^4 + 2^6)) → (b = (2^0 + 2^3 + 2^6)) →
  (a + b = 158) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end binary_addition_l2251_225102


namespace smallest_possible_n_l2251_225182

theorem smallest_possible_n (x n : ℤ) (hx : 0 < x) (m : ℤ) (hm : m = 30) (h1 : m.gcd n = x + 1) (h2 : m.lcm n = x * (x + 1)) : n = 6 := sorry

end smallest_possible_n_l2251_225182


namespace arrival_time_difference_l2251_225117

theorem arrival_time_difference
  (d : ℝ) (r_H : ℝ) (r_A : ℝ) (h₁ : d = 2) (h₂ : r_H = 12) (h₃ : r_A = 6) :
  (d / r_A * 60) - (d / r_H * 60) = 10 :=
by
  sorry

end arrival_time_difference_l2251_225117


namespace calc1_calc2_calc3_calc4_l2251_225189

-- Proof problem definitions
theorem calc1 : 15 + (-22) = -7 := sorry

theorem calc2 : (-13) + (-8) = -21 := sorry

theorem calc3 : (-0.9) + 1.5 = 0.6 := sorry

theorem calc4 : (1 / 2) + (-2 / 3) = -1 / 6 := sorry

end calc1_calc2_calc3_calc4_l2251_225189


namespace hyperbola_distance_condition_l2251_225183

open Real

theorem hyperbola_distance_condition (a b c x: ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
    (h_dist : abs (b^4 / a^2 / (a - c)) < a + sqrt (a^2 + b^2)) :
    0 < b / a ∧ b / a < 1 :=
by
  sorry

end hyperbola_distance_condition_l2251_225183


namespace product_gcd_lcm_4000_l2251_225159

-- Definitions of gcd and lcm for the given numbers
def gcd_40_100 := Nat.gcd 40 100
def lcm_40_100 := Nat.lcm 40 100

-- Problem: Prove that the product of the gcd and lcm of 40 and 100 equals 4000
theorem product_gcd_lcm_4000 : gcd_40_100 * lcm_40_100 = 4000 := by
  sorry

end product_gcd_lcm_4000_l2251_225159


namespace calculate_value_of_squares_difference_l2251_225140

theorem calculate_value_of_squares_difference : 305^2 - 301^2 = 2424 :=
by {
  sorry
}

end calculate_value_of_squares_difference_l2251_225140


namespace problem1_problem2_l2251_225139

theorem problem1 (a b : ℤ) (h : Even (5 * b + a)) : Even (a - 3 * b) :=
sorry

theorem problem2 (a b : ℤ) (h : Odd (5 * b + a)) : Odd (a - 3 * b) :=
sorry

end problem1_problem2_l2251_225139


namespace gear_rotations_l2251_225199

-- Definitions from the conditions
def gearA_teeth : ℕ := 12
def gearB_teeth : ℕ := 54

-- The main problem: prove that gear A needs 9 rotations and gear B needs 2 rotations
theorem gear_rotations :
  ∃ x y : ℕ, 12 * x = 54 * y ∧ x = 9 ∧ y = 2 := by
  sorry

end gear_rotations_l2251_225199


namespace four_digit_numbers_divisible_by_17_l2251_225129

theorem four_digit_numbers_divisible_by_17 :
  ∃ n, (∀ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 17 = 0 ↔ ∃ k, x = 17 * k ∧ 59 ≤ k ∧ k ≤ 588) ∧ n = 530 := 
sorry

end four_digit_numbers_divisible_by_17_l2251_225129


namespace correct_speed_to_reach_on_time_l2251_225138

theorem correct_speed_to_reach_on_time
  (d : ℝ)
  (t : ℝ)
  (h1 : d = 50 * (t + 1 / 12))
  (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 := 
by
  sorry

end correct_speed_to_reach_on_time_l2251_225138


namespace ratio_diff_squares_eq_16_l2251_225176

theorem ratio_diff_squares_eq_16 (x y : ℕ) (h1 : x + y = 16) (h2 : x ≠ y) :
  (x^2 - y^2) / (x - y) = 16 :=
by
  sorry

end ratio_diff_squares_eq_16_l2251_225176


namespace inequality_1_inequality_2_l2251_225195

noncomputable def f (x : ℝ) : ℝ := |x - 2| - 3
noncomputable def g (x : ℝ) : ℝ := |x + 3|

theorem inequality_1 (x : ℝ) : f x < g x ↔ x > -2 := 
by sorry

theorem inequality_2 (a : ℝ) : (∀ x : ℝ, f x < g x + a) ↔ a > 2 := 
by sorry

end inequality_1_inequality_2_l2251_225195


namespace difference_of_digits_l2251_225161

theorem difference_of_digits (p q : ℕ) (h1 : ∀ n, n < 100 → n ≥ 10 → ∀ m, m < 100 → m ≥ 10 → 9 * (p - q) = 9) : 
  p - q = 1 :=
sorry

end difference_of_digits_l2251_225161


namespace remaining_episodes_l2251_225149

theorem remaining_episodes (seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (h_seasons : seasons = 12) (h_episodes_per_season : episodes_per_season = 20) 
  (h_fraction_watched : fraction_watched = 1/3) : 
  (seasons * episodes_per_season - fraction_watched * (seasons * episodes_per_season) = 160) := 
by
  sorry

end remaining_episodes_l2251_225149


namespace find_f_nine_l2251_225184

-- Define the function f that satisfies the conditions
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x + y) = f(x) * f(y) for all real x and y
axiom functional_equation : ∀ (x y : ℝ), f (x + y) = f x * f y

-- Define the condition that f(3) = 4
axiom f_three : f 3 = 4

-- State the main theorem to prove that f(9) = 64
theorem find_f_nine : f 9 = 64 := by
  sorry

end find_f_nine_l2251_225184


namespace find_abc_l2251_225123

theorem find_abc (a b c : ℝ)
  (h1 : ∀ x : ℝ, (x < -6 ∨ (|x - 31| ≤ 1)) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h2 : a < b) :
  a + 2 * b + 3 * c = 76 :=
sorry

end find_abc_l2251_225123


namespace find_t_l2251_225114

-- Define the utility function based on hours of reading and playing basketball
def utility (reading_hours : ℝ) (basketball_hours : ℝ) : ℝ :=
  reading_hours * basketball_hours

-- Define the conditions for Wednesday and Thursday utilities
def wednesday_utility (t : ℝ) : ℝ :=
  t * (10 - t)

def thursday_utility (t : ℝ) : ℝ :=
  (3 - t) * (t + 4)

-- The main theorem stating the equivalence of utilities implies t = 3
theorem find_t (t : ℝ) (h : wednesday_utility t = thursday_utility t) : t = 3 :=
by
  -- Skip proof with sorry
  sorry

end find_t_l2251_225114


namespace sin_identity_l2251_225113

theorem sin_identity (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) :
  Real.sin ((3 * π / 4) - α) = 3 / 5 :=
by
  sorry

end sin_identity_l2251_225113


namespace distance_from_integer_l2251_225109

theorem distance_from_integer (a : ℝ) (h : a > 0) (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ abs (m * a - k) ≤ (1 / n) :=
by
  sorry

end distance_from_integer_l2251_225109


namespace jill_sod_area_l2251_225164

noncomputable def area_of_sod (yard_width yard_length sidewalk_width sidewalk_length flower_bed1_depth flower_bed1_length flower_bed2_depth flower_bed2_length flower_bed3_width flower_bed3_length flower_bed4_width flower_bed4_length : ℝ) : ℝ :=
  let yard_area := yard_width * yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flower_bed1_area := flower_bed1_depth * flower_bed1_length
  let flower_bed2_area := flower_bed2_depth * flower_bed2_length
  let flower_bed3_area := flower_bed3_width * flower_bed3_length
  let flower_bed4_area := flower_bed4_width * flower_bed4_length
  let total_non_sod_area := sidewalk_area + 2 * flower_bed1_area + flower_bed2_area + flower_bed3_area + flower_bed4_area
  yard_area - total_non_sod_area

theorem jill_sod_area : 
  area_of_sod 200 50 3 50 4 25 4 25 10 12 7 8 = 9474 := by sorry

end jill_sod_area_l2251_225164


namespace area_of_ground_l2251_225197

def height_of_rain : ℝ := 0.05
def volume_of_water : ℝ := 750

theorem area_of_ground : ∃ A : ℝ, A = (volume_of_water / height_of_rain) ∧ A = 15000 := by
  sorry

end area_of_ground_l2251_225197


namespace Harold_speed_is_one_more_l2251_225125

variable (Adrienne_speed Harold_speed : ℝ)
variable (distance_when_Harold_catches_Adr : ℝ)
variable (time_difference : ℝ)

axiom Adrienne_speed_def : Adrienne_speed = 3
axiom Harold_catches_distance : distance_when_Harold_catches_Adr = 12
axiom time_difference_def : time_difference = 1

theorem Harold_speed_is_one_more :
  Harold_speed - Adrienne_speed = 1 :=
by 
  have Adrienne_time := (distance_when_Harold_catches_Adr - Adrienne_speed * time_difference) / Adrienne_speed 
  have Harold_time := distance_when_Harold_catches_Adr / Harold_speed
  have := Adrienne_time = Harold_time - time_difference
  sorry

end Harold_speed_is_one_more_l2251_225125


namespace problem1_problem2_l2251_225154

variable {n : ℕ}
variable {a b : ℝ}

-- Part 1
theorem problem1 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(2 * n^2 / (n + 2) - n * a) - b| < ε) :
  a = 2 ∧ b = 4 := sorry

-- Part 2
theorem problem2 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(3^n / (3^(n + 1) + (a + 1)^n) - 1/3)| < ε) :
  -4 < a ∧ a < 2 := sorry

end problem1_problem2_l2251_225154


namespace incorrect_expression_l2251_225124

theorem incorrect_expression : ¬ (5 = (Real.sqrt (-5))^2) :=
by
  sorry

end incorrect_expression_l2251_225124


namespace required_number_of_shirts_l2251_225168

/-
In a shop, there is a sale of clothes. Every shirt costs $5, every hat $4, and a pair of jeans $10.
You need to pay $51 for a certain number of shirts, two pairs of jeans, and four hats.
Prove that the number of shirts you need to buy is 3.
-/

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_payment : ℕ := 51
def number_of_jeans : ℕ := 2
def number_of_hats : ℕ := 4

theorem required_number_of_shirts (S : ℕ) (h : 5 * S + 2 * jeans_cost + 4 * hat_cost = total_payment) : S = 3 :=
by
  -- This statement asserts that given the defined conditions, the number of shirts that satisfies the equation is 3.
  sorry

end required_number_of_shirts_l2251_225168


namespace fraction_evaluation_l2251_225126

theorem fraction_evaluation :
  ( (1 / 2 * 1 / 3 * 1 / 4 * 1 / 5 + 3 / 2 * 3 / 4 * 3 / 5) / 
    (1 / 2 * 2 / 3 * 2 / 5) ) = 41 / 8 :=
by
  sorry

end fraction_evaluation_l2251_225126


namespace find_number_l2251_225121

theorem find_number (n : ℕ) : gcd 30 n = 10 ∧ 70 ≤ n ∧ n ≤ 80 ∧ 200 ≤ lcm 30 n ∧ lcm 30 n ≤ 300 → (n = 70 ∨ n = 80) :=
sorry

end find_number_l2251_225121


namespace solve_problem_l2251_225148

-- Conditions from the problem
def is_prime (p : ℕ) : Prop := Nat.Prime p

def satisfies_conditions (n p : ℕ) : Prop := 
  (p > 1) ∧ is_prime p ∧ (n > 0) ∧ (n ≤ 2 * p)

-- Main proof statement
theorem solve_problem (n p : ℕ) (h1 : satisfies_conditions n p)
    (h2 : (p - 1) ^ n + 1 ∣ n ^ (p - 1)) :
    (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1 ∧ is_prime p) :=
sorry

end solve_problem_l2251_225148


namespace scientific_notation_110_billion_l2251_225146

theorem scientific_notation_110_billion :
  ∃ (n : ℝ) (e : ℤ), 110000000000 = n * 10 ^ e ∧ 1 ≤ n ∧ n < 10 ∧ n = 1.1 ∧ e = 11 :=
by
  sorry

end scientific_notation_110_billion_l2251_225146


namespace n_greater_than_sqrt_p_sub_1_l2251_225150

theorem n_greater_than_sqrt_p_sub_1 {p n : ℕ} (hp : Nat.Prime p) (hn : n ≥ 2) (hdiv : p ∣ (n^6 - 1)) : n > Nat.sqrt p - 1 := 
by
  sorry

end n_greater_than_sqrt_p_sub_1_l2251_225150


namespace part1_part2_l2251_225194

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (a x : ℝ) := a * |x - 1|

theorem part1 (a : ℝ) :
  (∀ x : ℝ, |f x| = g a x → x = 1) → a < 0 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 :=
sorry

end part1_part2_l2251_225194


namespace equation_zero_solution_l2251_225119

-- Define the conditions and the answer
def equation_zero (x : ℝ) : Prop := (x^2 + x - 2) / (x - 1) = 0
def non_zero_denominator (x : ℝ) : Prop := x - 1 ≠ 0
def solution_x (x : ℝ) : Prop := x = -2

-- The main theorem
theorem equation_zero_solution (x : ℝ) (h1 : equation_zero x) (h2 : non_zero_denominator x) : solution_x x := 
sorry

end equation_zero_solution_l2251_225119


namespace probability_not_within_B_l2251_225137

-- Definition representing the problem context
structure Squares where
  areaA : ℝ
  areaA_pos : areaA = 65
  perimeterB : ℝ
  perimeterB_pos : perimeterB = 16

-- The theorem to be proved
theorem probability_not_within_B (s : Squares) : 
  let sideA := Real.sqrt s.areaA
  let sideB := s.perimeterB / 4
  let areaB := sideB^2
  let area_not_covered := s.areaA - areaB
  let probability := area_not_covered / s.areaA
  probability = 49 / 65 := 
by
  sorry

end probability_not_within_B_l2251_225137


namespace fermat_coprime_l2251_225166

theorem fermat_coprime (m n : ℕ) (hmn : m ≠ n) (hm_pos : m > 0) (hn_pos : n > 0) :
  gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 :=
sorry

end fermat_coprime_l2251_225166


namespace average_boxes_per_day_by_third_day_l2251_225131

theorem average_boxes_per_day_by_third_day (day1 day2 day3_part1 day3_part2 : ℕ) :
  day1 = 318 →
  day2 = 312 →
  day3_part1 = 180 →
  day3_part2 = 162 →
  ((day1 + day2 + (day3_part1 + day3_part2)) / 3) = 324 :=
by
  intros h1 h2 h3 h4
  sorry

end average_boxes_per_day_by_third_day_l2251_225131


namespace closest_integer_to_2_plus_sqrt_6_l2251_225193

theorem closest_integer_to_2_plus_sqrt_6 (sqrt6_lower : 2 < Real.sqrt 6) (sqrt6_upper : Real.sqrt 6 < 2.5) : 
  abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 3) ∧ abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 5) :=
by
  sorry

end closest_integer_to_2_plus_sqrt_6_l2251_225193


namespace james_tip_percentage_l2251_225106

theorem james_tip_percentage :
  let ticket_cost : ℝ := 100
  let dinner_cost : ℝ := 120
  let limo_cost_per_hour : ℝ := 80
  let limo_hours : ℕ := 6
  let total_cost_with_tip : ℝ := 836
  let total_cost_without_tip : ℝ := 2 * ticket_cost + limo_hours * limo_cost_per_hour + dinner_cost
  let tip : ℝ := total_cost_with_tip - total_cost_without_tip
  let percentage_tip : ℝ := (tip / dinner_cost) * 100
  percentage_tip = 30 :=
by
  sorry

end james_tip_percentage_l2251_225106


namespace ratio_d_c_l2251_225104

theorem ratio_d_c (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hc : c ≠ 0) 
  (h1 : 8 * x - 5 * y = c) (h2 : 10 * y - 16 * x = d) : d / c = -2 :=
by
  sorry

end ratio_d_c_l2251_225104


namespace Suzanna_rides_8_miles_in_40_minutes_l2251_225132

theorem Suzanna_rides_8_miles_in_40_minutes :
  (∀ n : ℕ, Suzanna_distance_in_n_minutes = (n / 10) * 2) → Suzanna_distance_in_40_minutes = 8 :=
by
  sorry

-- Definitions for Suzanna's distance conditions
def Suzanna_distance_in_n_minutes (n : ℕ) : ℕ := (n / 10) * 2

noncomputable def Suzanna_distance_in_40_minutes := Suzanna_distance_in_n_minutes 40

#check Suzanna_rides_8_miles_in_40_minutes

end Suzanna_rides_8_miles_in_40_minutes_l2251_225132


namespace number_base_addition_l2251_225105

theorem number_base_addition (A B : ℕ) (h1: A = 2 * B) (h2: 2 * B^2 + 2 * B + 4 + 10 * B + 5 = (3 * B)^2 + 3 * (3 * B) + 4) : 
  A + B = 9 := 
by 
  sorry

end number_base_addition_l2251_225105


namespace geometric_sequence_arith_condition_l2251_225163

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

end geometric_sequence_arith_condition_l2251_225163


namespace constant_function_on_chessboard_l2251_225192

theorem constant_function_on_chessboard
  (f : ℤ × ℤ → ℝ)
  (h_nonneg : ∀ (m n : ℤ), 0 ≤ f (m, n))
  (h_mean : ∀ (m n : ℤ), f (m, n) = (f (m + 1, n) + f (m - 1, n) + f (m, n + 1) + f (m, n - 1)) / 4) :
  ∃ c : ℝ, ∀ (m n : ℤ), f (m, n) = c :=
sorry

end constant_function_on_chessboard_l2251_225192


namespace number_of_workers_l2251_225170

-- Definitions for conditions
def initial_contribution (W C : ℕ) : Prop := W * C = 300000
def additional_contribution (W C : ℕ) : Prop := W * (C + 50) = 350000

-- Proof statement
theorem number_of_workers (W C : ℕ) (h1 : initial_contribution W C) (h2 : additional_contribution W C) : W = 1000 :=
by
  sorry

end number_of_workers_l2251_225170


namespace popsicle_sticks_left_correct_l2251_225172

noncomputable def popsicle_sticks_left (initial : ℝ) (given : ℝ) : ℝ :=
  initial - given

theorem popsicle_sticks_left_correct :
  popsicle_sticks_left 63 50 = 13 :=
by
  sorry

end popsicle_sticks_left_correct_l2251_225172


namespace problem_acd_div_b_l2251_225116

theorem problem_acd_div_b (a b c d : ℤ) (x : ℝ)
    (h1 : x = (a + b * Real.sqrt c) / d)
    (h2 : (7 * x) / 4 + 2 = 6 / x) :
    (a * c * d) / b = -322 := sorry

end problem_acd_div_b_l2251_225116


namespace smallest_sum_minimum_l2251_225147

noncomputable def smallest_sum (x y : ℕ) : ℕ :=
if h₁ : x ≠ y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) then x + y else 0

theorem smallest_sum_minimum (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  smallest_sum x y = 96 := sorry

end smallest_sum_minimum_l2251_225147


namespace magician_draws_two_cards_l2251_225175

-- Define the total number of unique cards
def total_cards : ℕ := 15^2

-- Define the number of duplicate cards
def duplicate_cards : ℕ := 15

-- Define the number of ways to choose 2 cards from the duplicate cards
def choose_two_duplicates : ℕ := Nat.choose 15 2

-- Define the number of ways to choose 1 duplicate card and 1 non-duplicate card
def choose_one_duplicate_one_nonduplicate : ℕ := (15 * (total_cards - 15 - 14 - 14))

-- The main theorem to prove
theorem magician_draws_two_cards : choose_two_duplicates + choose_one_duplicate_one_nonduplicate = 2835 := by
  sorry

end magician_draws_two_cards_l2251_225175


namespace quadratic_polynomial_fourth_power_l2251_225153

theorem quadratic_polynomial_fourth_power {a b c : ℤ} (h : ∀ x : ℤ, ∃ k : ℤ, ax^2 + bx + c = k^4) : a = 0 ∧ b = 0 :=
sorry

end quadratic_polynomial_fourth_power_l2251_225153


namespace number_of_divisors_30_l2251_225107

theorem number_of_divisors_30 : 
  ∃ (d : ℕ), d = 2 * 2 * 2 ∧ d = 8 :=
  by sorry

end number_of_divisors_30_l2251_225107


namespace triangle_angle_side_inequality_l2251_225108

variable {A B C : Type} -- Variables for points in the triangle
variable {a b : ℝ} -- Variables for the lengths of sides opposite to angles A and B
variable {A_angle B_angle : ℝ} -- Variables for the angles at A and B in triangle ABC

-- Define that we are in a triangle setting
def is_triangle (A B C : Type) := True

-- Define the assumption for the proof by contradiction
def assumption (a b : ℝ) := a ≤ b

theorem triangle_angle_side_inequality (h_triangle : is_triangle A B C)
(h_angle : A_angle > B_angle) 
(h_assumption : assumption a b) : a > b := 
sorry

end triangle_angle_side_inequality_l2251_225108


namespace professor_has_to_grade_405_more_problems_l2251_225128

theorem professor_has_to_grade_405_more_problems
  (problems_per_paper : ℕ)
  (total_papers : ℕ)
  (graded_papers : ℕ)
  (remaining_papers := total_papers - graded_papers)
  (p : ℕ := remaining_papers * problems_per_paper) :
  problems_per_paper = 15 ∧ total_papers = 45 ∧ graded_papers = 18 → p = 405 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end professor_has_to_grade_405_more_problems_l2251_225128


namespace sodas_per_pack_l2251_225171

theorem sodas_per_pack 
  (packs : ℕ) (initial_sodas : ℕ) (days_in_a_week : ℕ) (sodas_per_day : ℕ) 
  (total_sodas_consumed : ℕ) (sodas_per_pack : ℕ) :
  packs = 5 →
  initial_sodas = 10 →
  days_in_a_week = 7 →
  sodas_per_day = 10 →
  total_sodas_consumed = 70 →
  total_sodas_consumed - initial_sodas = packs * sodas_per_pack →
  sodas_per_pack = 12 :=
by
  intros hpacks hinitial hsodas hdaws htpd htcs
  sorry

end sodas_per_pack_l2251_225171


namespace average_of_angles_l2251_225145

theorem average_of_angles (p q r s t : ℝ) (h : p + q + r + s + t = 180) : 
  (p + q + r + s + t) / 5 = 36 :=
by
  sorry

end average_of_angles_l2251_225145


namespace trajectory_is_eight_rays_l2251_225136

open Real

def trajectory_of_point (x y : ℝ) : Prop :=
  abs (abs x - abs y) = 2

theorem trajectory_is_eight_rays :
  ∃ (x y : ℝ), trajectory_of_point x y :=
sorry

end trajectory_is_eight_rays_l2251_225136


namespace sum_gcf_lcm_36_56_84_l2251_225167

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem sum_gcf_lcm_36_56_84 :
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  gcf_36_56_84 + lcm_36_56_84 = 516 :=
by
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  show gcf_36_56_84 + lcm_36_56_84 = 516
  sorry

end sum_gcf_lcm_36_56_84_l2251_225167


namespace abs_inequality_solution_rational_inequality_solution_l2251_225110

theorem abs_inequality_solution (x : ℝ) : (|x - 2| + |2 * x - 3| < 4) ↔ (1 / 3 < x ∧ x < 3) :=
sorry

theorem rational_inequality_solution (x : ℝ) : 
  (x^2 - 3 * x) / (x^2 - x - 2) ≤ x ↔ (x ∈ Set.Icc (-1) 0 ∪ {1} ∪ Set.Ioi 2) := 
sorry

#check abs_inequality_solution
#check rational_inequality_solution

end abs_inequality_solution_rational_inequality_solution_l2251_225110


namespace tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l2251_225169

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x) - x * Real.exp x

theorem tangent_line_a_zero (x : ℝ) (y : ℝ) : 
  a = 0 ∧ x = 1 → (2 * Real.exp 1) * x + y - Real.exp 1 = 0 :=
sorry

theorem range_a_if_fx_neg (a : ℝ) : 
  (∀ x ≥ 1, f a x < 0) → a < Real.exp 1 :=
sorry

theorem max_value_a_one (x : ℝ) : 
  a = 1 → x = (Real.exp 1)⁻¹ → f 1 x = -1 :=
sorry

end tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l2251_225169


namespace median_price_l2251_225190

-- Definitions from conditions
def price1 : ℝ := 10
def price2 : ℝ := 12
def price3 : ℝ := 15

def sales1 : ℝ := 0.50
def sales2 : ℝ := 0.30
def sales3 : ℝ := 0.20

-- Statement of the problem
theorem median_price : (price1 * sales1 + price2 * sales2 + price3 * sales3) / 2 = 11 := by
  sorry

end median_price_l2251_225190


namespace washing_time_per_cycle_l2251_225155

theorem washing_time_per_cycle
    (shirts pants sweaters jeans : ℕ)
    (items_per_cycle total_hours : ℕ)
    (h1 : shirts = 18)
    (h2 : pants = 12)
    (h3 : sweaters = 17)
    (h4 : jeans = 13)
    (h5 : items_per_cycle = 15)
    (h6 : total_hours = 3) :
    ((shirts + pants + sweaters + jeans) / items_per_cycle) * (total_hours * 60) / ((shirts + pants + sweaters + jeans) / items_per_cycle) = 45 := 
by
  sorry

end washing_time_per_cycle_l2251_225155


namespace pulley_distance_l2251_225100

theorem pulley_distance (r₁ r₂ d l : ℝ):
    r₁ = 10 →
    r₂ = 6 →
    l = 30 →
    (d = 2 * Real.sqrt 229) :=
by
    intros h₁ h₂ h₃
    sorry

end pulley_distance_l2251_225100


namespace buzz_waiter_ratio_l2251_225115

def total_slices : Nat := 78
def waiter_condition (W : Nat) : Prop := W - 20 = 28

theorem buzz_waiter_ratio (W : Nat) (h : waiter_condition W) : 
  let buzz_slices := total_slices - W
  let ratio_buzz_waiter := buzz_slices / W
  ratio_buzz_waiter = 5 / 8 :=
by
  sorry

end buzz_waiter_ratio_l2251_225115
