import Mathlib

namespace number_of_roses_sold_l231_231178

def initial_roses : ℕ := 50
def picked_roses : ℕ := 21
def final_roses : ℕ := 56

theorem number_of_roses_sold : ∃ x : ℕ, initial_roses - x + picked_roses = final_roses ∧ x = 15 :=
by {
  sorry
}

end number_of_roses_sold_l231_231178


namespace sin_2pi_minus_theta_l231_231462

theorem sin_2pi_minus_theta (theta : ℝ) (k : ℤ) 
  (h1 : 3 * Real.cos theta ^ 2 = Real.tan theta + 3)
  (h2 : theta ≠ k * Real.pi) :
  Real.sin (2 * (Real.pi - theta)) = 2 / 3 := by
  sorry

end sin_2pi_minus_theta_l231_231462


namespace angle_420_mod_360_eq_60_l231_231442

def angle_mod_equiv (a b : ℕ) : Prop := a % 360 = b

theorem angle_420_mod_360_eq_60 : angle_mod_equiv 420 60 := 
by
  sorry

end angle_420_mod_360_eq_60_l231_231442


namespace toy_problem_l231_231787

theorem toy_problem :
  ∃ (n m : ℕ), 
    1500 ≤ n ∧ n ≤ 2000 ∧ 
    n % 15 = 5 ∧ n % 20 = 5 ∧ n % 30 = 5 ∧ 
    (n + m) % 12 = 0 ∧ (n + m) % 18 = 0 ∧ 
    n + m ≤ 2100 ∧ m = 31 := 
sorry

end toy_problem_l231_231787


namespace range_of_k_l231_231752

theorem range_of_k {x y k : ℝ} :
  (∀ x y, 2 * x - y ≤ 1 ∧ x + y ≥ 2 ∧ y - x ≤ 2) →
  (z = k * x + 2 * y) →
  (∀ (x y : ℝ), z = k * x + 2 * y → (x = 1) ∧ (y = 1)) →
  -4 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l231_231752


namespace find_unknown_number_l231_231569

theorem find_unknown_number (x : ℕ) (h₁ : (20 + 40 + 60) / 3 = 5 + (10 + 50 + x) / 3) : x = 45 :=
by sorry

end find_unknown_number_l231_231569


namespace find_x_l231_231428

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 210) : x = 14 := by
  sorry

end find_x_l231_231428


namespace distinct_real_roots_l231_231943

def operation (a b : ℝ) : ℝ := a^2 - a * b + b

theorem distinct_real_roots {x : ℝ} : 
  (operation x 3 = 5) → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation x1 3 = 5 ∧ operation x2 3 = 5) :=
by 
  -- Add your proof here
  sorry

end distinct_real_roots_l231_231943


namespace find_ab_l231_231029

theorem find_ab (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 30) : a * b = 32 :=
by
  -- We will complete the proof in this space
  sorry

end find_ab_l231_231029


namespace solution_set_I_range_of_a_II_l231_231551

def f (x a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2*x - 1|

theorem solution_set_I (x : ℝ) (a : ℝ) (h : a = 2) :
  f x a ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem range_of_a_II (a : ℝ) :
  (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end solution_set_I_range_of_a_II_l231_231551


namespace sum_of_coefficients_l231_231803

def f (x : ℝ) : ℝ := (1 + 2 * x)^4

theorem sum_of_coefficients : f 1 = 81 :=
by
  -- New goal is immediately achieved since the given is precisely ensured.
  sorry

end sum_of_coefficients_l231_231803


namespace total_area_of_union_of_six_triangles_l231_231467

theorem total_area_of_union_of_six_triangles :
  let s := 2 * Real.sqrt 2
  let area_one_triangle := (Real.sqrt 3 / 4) * s^2
  let total_area_without_overlaps := 6 * area_one_triangle
  let side_overlap := Real.sqrt 2
  let area_one_overlap := (Real.sqrt 3 / 4) * side_overlap ^ 2
  let total_overlap_area := 5 * area_one_overlap
  let net_area := total_area_without_overlaps - total_overlap_area
  net_area = 9.5 * Real.sqrt 3 := 
by
  sorry

end total_area_of_union_of_six_triangles_l231_231467


namespace fourth_term_of_sequence_l231_231936

-- Given conditions
def first_term : ℕ := 5
def fifth_term : ℕ := 1280

-- Definition of the common ratio
def common_ratio (a : ℕ) (b : ℕ) : ℕ := (b / a)^(1 / 4)

-- Function to calculate the nth term of a geometric sequence
def nth_term (a r n : ℕ) : ℕ := a * r^(n - 1)

-- Prove the fourth term of the geometric sequence is 320
theorem fourth_term_of_sequence 
    (a : ℕ) (b : ℕ) (a_pos : a = first_term) (b_eq : nth_term a (common_ratio a b) 5 = b) : 
    nth_term a (common_ratio a b) 4 = 320 := by
  sorry

end fourth_term_of_sequence_l231_231936


namespace sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l231_231454

-- Definitions for vertices of pyramids
variables (A B C D E : ℝ)

-- Assuming E is inside pyramid ABCD
variable (inside : E ∈ convex_hull ℝ {A, B, C, D})

-- Assertion 1
theorem sum_of_edges_not_always_smaller
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : D ≠ E):
  ¬ (abs A - E + abs B - E + abs C - E < abs A - D + abs B - D + abs C - D) :=
sorry

-- Assertion 2
theorem at_least_one_edge_shorter
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A)
  (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D)
  (h7 : D ≠ E):
  abs A - E < abs A - D ∨ abs B - E < abs B - D ∨ abs C - E < abs C - D :=
sorry

end sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l231_231454


namespace number_of_paths_in_MATHEMATICIAN_diagram_l231_231965

theorem number_of_paths_in_MATHEMATICIAN_diagram : ∃ n : ℕ, n = 8191 :=
by
  -- Define necessary structure
  -- Number of rows and binary choices
  let rows : ℕ := 12
  let choices_per_position : ℕ := 2
  -- Total paths calculation
  let total_paths := choices_per_position ^ rows
  -- Including symmetry and subtracting duplicate
  let final_paths := 2 * total_paths - 1
  use final_paths
  have : final_paths = 8191 :=
    by norm_num
  exact this

end number_of_paths_in_MATHEMATICIAN_diagram_l231_231965


namespace speed_of_second_car_l231_231868

theorem speed_of_second_car
  (t : ℝ)
  (distance_apart : ℝ)
  (speed_first_car : ℝ)
  (speed_second_car : ℝ)
  (h_total_distance : distance_apart = t * speed_first_car + t * speed_second_car)
  (h_time : t = 2.5)
  (h_distance_apart : distance_apart = 310)
  (h_speed_first_car : speed_first_car = 60) :
  speed_second_car = 64 := by
  sorry

end speed_of_second_car_l231_231868


namespace days_to_learn_all_vowels_l231_231423

-- Defining the number of vowels
def number_of_vowels : Nat := 5

-- Defining the days Charles takes to learn one alphabet
def days_per_vowel : Nat := 7

-- Prove that Charles needs 35 days to learn all the vowels
theorem days_to_learn_all_vowels : number_of_vowels * days_per_vowel = 35 := by
  sorry

end days_to_learn_all_vowels_l231_231423


namespace largest_multiple_of_15_less_than_400_l231_231364

theorem largest_multiple_of_15_less_than_400 (x : ℕ) (k : ℕ) (h : x = 15 * k) (h1 : x < 400) (h2 : ∀ m : ℕ, (15 * m < 400) → m ≤ k) : x = 390 :=
by
  sorry

end largest_multiple_of_15_less_than_400_l231_231364


namespace ninth_graders_only_science_not_history_l231_231800

-- Conditions
def total_students : ℕ := 120
def students_science : ℕ := 85
def students_history : ℕ := 75

-- Statement: Determine the number of students enrolled only in the science class
theorem ninth_graders_only_science_not_history : 
  (students_science - (students_science + students_history - total_students)) = 45 := by
  sorry

end ninth_graders_only_science_not_history_l231_231800


namespace eric_return_home_time_l231_231615

-- Definitions based on conditions
def time_running_to_park : ℕ := 20
def time_jogging_to_park : ℕ := 10
def trip_to_park_time : ℕ := time_running_to_park + time_jogging_to_park
def return_time_multiplier : ℕ := 3

-- Statement of the problem
theorem eric_return_home_time : 
  return_time_multiplier * trip_to_park_time = 90 :=
by 
  -- Skipping proof steps
  sorry

end eric_return_home_time_l231_231615


namespace min_distance_feasible_region_line_l231_231111

def point (x y : ℝ) : Type := ℝ × ℝ 

theorem min_distance_feasible_region_line :
  ∃ (M N : ℝ × ℝ),
    (2 * M.1 + M.2 - 4 >= 0) ∧
    (M.1 - M.2 - 2 <= 0) ∧
    (M.2 - 3 <= 0) ∧
    (N.2 = -2 * N.1 + 2) ∧
    (dist M N = (2 * Real.sqrt 5)/5) :=
by 
  sorry

end min_distance_feasible_region_line_l231_231111


namespace min_trips_calculation_l231_231716

noncomputable def min_trips (total_weight : ℝ) (truck_capacity : ℝ) : ℕ :=
  ⌈total_weight / truck_capacity⌉₊

theorem min_trips_calculation : min_trips 18.5 3.9 = 5 :=
by
  -- Proof goes here
  sorry

end min_trips_calculation_l231_231716


namespace calc_probability_10_or_9_ring_calc_probability_less_than_9_ring_l231_231593

def probability_10_ring : ℝ := 0.13
def probability_9_ring : ℝ := 0.28
def probability_8_ring : ℝ := 0.31

def probability_10_or_9_ring : ℝ := probability_10_ring + probability_9_ring

def probability_less_than_9_ring : ℝ := 1 - probability_10_or_9_ring

theorem calc_probability_10_or_9_ring :
  probability_10_or_9_ring = 0.41 :=
by
  sorry

theorem calc_probability_less_than_9_ring :
  probability_less_than_9_ring = 0.59 :=
by
  sorry

end calc_probability_10_or_9_ring_calc_probability_less_than_9_ring_l231_231593


namespace available_floor_space_equals_110_sqft_l231_231268

-- Definitions for the conditions
def tile_side_in_feet : ℝ := 0.5
def width_main_section_tiles : ℕ := 15
def length_main_section_tiles : ℕ := 25
def width_alcove_tiles : ℕ := 10
def depth_alcove_tiles : ℕ := 8
def width_pillar_tiles : ℕ := 3
def length_pillar_tiles : ℕ := 5

-- Conversion of tiles to feet
def width_main_section_feet : ℝ := width_main_section_tiles * tile_side_in_feet
def length_main_section_feet : ℝ := length_main_section_tiles * tile_side_in_feet
def width_alcove_feet : ℝ := width_alcove_tiles * tile_side_in_feet
def depth_alcove_feet : ℝ := depth_alcove_tiles * tile_side_in_feet
def width_pillar_feet : ℝ := width_pillar_tiles * tile_side_in_feet
def length_pillar_feet : ℝ := length_pillar_tiles * tile_side_in_feet

-- Area calculations
def area_main_section : ℝ := width_main_section_feet * length_main_section_feet
def area_alcove : ℝ := width_alcove_feet * depth_alcove_feet
def total_area : ℝ := area_main_section + area_alcove
def area_pillar : ℝ := width_pillar_feet * length_pillar_feet
def available_floor_space : ℝ := total_area - area_pillar

-- Proof statement
theorem available_floor_space_equals_110_sqft 
  (h1 : width_main_section_feet = width_main_section_tiles * tile_side_in_feet)
  (h2 : length_main_section_feet = length_main_section_tiles * tile_side_in_feet)
  (h3 : width_alcove_feet = width_alcove_tiles * tile_side_in_feet)
  (h4 : depth_alcove_feet = depth_alcove_tiles * tile_side_in_feet)
  (h5 : width_pillar_feet = width_pillar_tiles * tile_side_in_feet)
  (h6 : length_pillar_feet = length_pillar_tiles * tile_side_in_feet) 
  (h7 : area_main_section = width_main_section_feet * length_main_section_feet)
  (h8 : area_alcove = width_alcove_feet * depth_alcove_feet)
  (h9 : total_area = area_main_section + area_alcove)
  (h10 : area_pillar = width_pillar_feet * length_pillar_feet)
  (h11 : available_floor_space = total_area - area_pillar) : 
  available_floor_space = 110 := 
by 
  sorry

end available_floor_space_equals_110_sqft_l231_231268


namespace lowest_test_score_dropped_is_35_l231_231541

theorem lowest_test_score_dropped_is_35 
  (A B C D : ℕ) 
  (h1 : (A + B + C + D) / 4 = 50)
  (h2 : min A (min B (min C D)) = D)
  (h3 : (A + B + C) / 3 = 55) : 
  D = 35 := by
  sorry

end lowest_test_score_dropped_is_35_l231_231541


namespace factorize_x_cubed_minus_9x_l231_231629

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l231_231629


namespace sum_of_common_divisors_36_48_l231_231191

-- Definitions based on the conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- List of divisors for 36 and 48
def divisors_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]
def divisors_48 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]

-- Definition of common divisors
def common_divisors_36_48 : List ℕ := [1, 2, 3, 4, 6, 12]

-- Sum of common divisors
def sum_common_divisors_36_48 := common_divisors_36_48.sum

-- The statement of the theorem
theorem sum_of_common_divisors_36_48 : sum_common_divisors_36_48 = 28 := by
  sorry

end sum_of_common_divisors_36_48_l231_231191


namespace tank_capacity_l231_231819

theorem tank_capacity (C : ℕ) 
  (h : 0.9 * (C : ℝ) - 0.4 * (C : ℝ) = 63) : C = 126 := 
by
  sorry

end tank_capacity_l231_231819


namespace find_876_last_three_digits_l231_231382

noncomputable def has_same_last_three_digits (N : ℕ) : Prop :=
  (N^2 - N) % 1000 = 0

theorem find_876_last_three_digits (N : ℕ) (h1 : has_same_last_three_digits N) (h2 : N > 99) (h3 : N < 1000) : 
  N % 1000 = 876 :=
sorry

end find_876_last_three_digits_l231_231382


namespace find_initial_passengers_l231_231565

def initial_passengers_found (P : ℕ) : Prop :=
  let after_first_station := (2 / 3 : ℚ) * P + 280
  let after_second_station := (1 / 2 : ℚ) * after_first_station + 12
  after_second_station = 242

theorem find_initial_passengers :
  ∃ P : ℕ, initial_passengers_found P ∧ P = 270 :=
by
  sorry

end find_initial_passengers_l231_231565


namespace S_is_line_l231_231511

open Complex

noncomputable def S : Set ℂ := { z : ℂ | ∃ (x y : ℝ), z = x + y * Complex.I ∧ 3 * y + 4 * x = 0 }

theorem S_is_line :
  ∃ (m b : ℝ), S = { z : ℂ | ∃ (x y : ℝ), z = x + y * Complex.I ∧ x = m * y + b } :=
sorry

end S_is_line_l231_231511


namespace find_a_for_tangency_l231_231951

-- Definitions of line and parabola
def line (x y : ℝ) : Prop := x - y - 1 = 0
def parabola (x y : ℝ) (a : ℝ) : Prop := y = a * x^2

-- The tangency condition for quadratic equations
def tangency_condition (a : ℝ) : Prop := 1 - 4 * a = 0

theorem find_a_for_tangency (a : ℝ) :
  (∀ x y, line x y → parabola x y a → tangency_condition a) → a = 1/4 :=
by
  -- Proof omitted
  sorry

end find_a_for_tangency_l231_231951


namespace manuscript_typing_cost_l231_231083

-- Defining the conditions as per our problem
def first_time_typing_rate : ℕ := 5 -- $5 per page for first-time typing
def revision_rate : ℕ := 3 -- $3 per page per revision

def num_pages : ℕ := 100 -- total number of pages
def revised_once : ℕ := 30 -- number of pages revised once
def revised_twice : ℕ := 20 -- number of pages revised twice
def no_revision := num_pages - (revised_once + revised_twice) -- pages with no revisions

-- Defining the cost function to calculate the total cost of typing
noncomputable def total_typing_cost : ℕ :=
  (num_pages * first_time_typing_rate) + (revised_once * revision_rate) + (revised_twice * revision_rate * 2)

-- Lean theorem statement to prove the total cost is $710
theorem manuscript_typing_cost :
  total_typing_cost = 710 := by
  sorry

end manuscript_typing_cost_l231_231083


namespace essay_count_problem_l231_231223

noncomputable def eighth_essays : ℕ := sorry
noncomputable def seventh_essays : ℕ := sorry

theorem essay_count_problem (x : ℕ) (h1 : eighth_essays = x) (h2 : seventh_essays = (1/2 : ℚ) * x - 2) (h3 : eighth_essays + seventh_essays = 118) : 
  seventh_essays = 38 :=
sorry

end essay_count_problem_l231_231223


namespace Sara_snow_volume_l231_231881

theorem Sara_snow_volume :
  let length := 30
  let width := 3
  let first_half_length := length / 2
  let second_half_length := length / 2
  let depth1 := 0.5
  let depth2 := 1.0 / 3.0
  let volume1 := first_half_length * width * depth1
  let volume2 := second_half_length * width * depth2
  volume1 + volume2 = 37.5 :=
by
  sorry

end Sara_snow_volume_l231_231881


namespace prob_t_prob_vowel_l231_231229

def word := "mathematics"
def total_letters : ℕ := 11
def t_count : ℕ := 2
def vowel_count : ℕ := 4

-- Definition of being a letter "t"
def is_t (c : Char) : Prop := c = 't'

-- Definition of being a vowel
def is_vowel (c : Char) : Prop := c = 'a' ∨ c = 'e' ∨ c = 'i'

theorem prob_t : (t_count : ℚ) / total_letters = 2 / 11 :=
by
  sorry

theorem prob_vowel : (vowel_count : ℚ) / total_letters = 4 / 11 :=
by
  sorry

end prob_t_prob_vowel_l231_231229


namespace clock_rings_in_january_l231_231802

theorem clock_rings_in_january :
  ∀ (days_in_january hours_per_day ring_interval : ℕ)
  (first_ring_time : ℕ) (january_first_hour : ℕ), 
  days_in_january = 31 →
  hours_per_day = 24 →
  ring_interval = 7 →
  january_first_hour = 2 →
  first_ring_time = 30 →
  (days_in_january * hours_per_day) / ring_interval + 1 = 107 := by
  intros days_in_january hours_per_day ring_interval first_ring_time january_first_hour
  sorry

end clock_rings_in_january_l231_231802


namespace growth_rate_of_yield_l231_231014

-- Let x be the growth rate of the average yield per acre
variable (x : ℝ)

-- Initial conditions
def initial_acres := 10
def initial_yield := 20000
def final_yield := 60000

-- Relationship between the growth rates
def growth_relation := x * initial_acres * (1 + 2 * x) * (1 + x) = final_yield / initial_yield

theorem growth_rate_of_yield (h : growth_relation x) : x = 0.5 :=
  sorry

end growth_rate_of_yield_l231_231014


namespace range_of_m_l231_231217

noncomputable def function_even_and_monotonic (f : ℝ → ℝ) := 
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x > f y)

variable (f : ℝ → ℝ)
variable (m : ℝ)

theorem range_of_m (h₁ : function_even_and_monotonic f) 
  (h₂ : f m > f (1 - m)) : m < 1 / 2 :=
sorry

end range_of_m_l231_231217


namespace find_divisor_l231_231356

theorem find_divisor (d : ℕ) (h1 : 127 = d * 5 + 2) : d = 25 :=
sorry

end find_divisor_l231_231356


namespace number_of_whole_numbers_in_intervals_l231_231998

theorem number_of_whole_numbers_in_intervals : 
  let interval_start := (5 / 3 : ℝ)
  let interval_end := 2 * Real.pi
  ∃ n : ℕ, interval_start < ↑n ∧ ↑n < interval_end ∧ (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) ∧ 
  (∀ m : ℕ, interval_start < ↑m ∧ ↑m < interval_end → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6)) :=
sorry

end number_of_whole_numbers_in_intervals_l231_231998


namespace find_delta_l231_231540

theorem find_delta (p q Δ : ℕ) (h₁ : Δ + q = 73) (h₂ : 2 * (Δ + q) + p = 172) (h₃ : p = 26) : Δ = 12 :=
by
  sorry

end find_delta_l231_231540


namespace hyperbola_sufficient_but_not_necessary_l231_231330

theorem hyperbola_sufficient_but_not_necessary :
  (∀ (C : Type) (x y : ℝ), C = {p : ℝ × ℝ | ((p.1)^2 / 16) - ((p.2)^2 / 9) = 1} →
  (∀ x, y = 3 * (x / 4) ∨ y = -3 * (x / 4)) →
  ∃ (C' : Type) (x' y' : ℝ), C' = {p : ℝ × ℝ | ((p.1)^2 / 64) - ((p.2)^2 / 36) = 1} ∧
  (∀ x', y' = 3 * (x' / 4) ∨ y' = -3 * (x' / 4))) :=
sorry

end hyperbola_sufficient_but_not_necessary_l231_231330


namespace sum_of_fractions_eq_13_5_l231_231906

noncomputable def sumOfFractions : ℚ :=
  (1/10 + 2/10 + 3/10 + 4/10 + 5/10 + 6/10 + 7/10 + 8/10 + 9/10 + 90/10)

theorem sum_of_fractions_eq_13_5 :
  sumOfFractions = 13.5 := by
  sorry

end sum_of_fractions_eq_13_5_l231_231906


namespace jason_arms_tattoos_l231_231854

variable (x : ℕ)

def jason_tattoos (x : ℕ) : ℕ := 2 * x + 3 * 2

def adam_tattoos (x : ℕ) : ℕ := 3 + 2 * (jason_tattoos x)

theorem jason_arms_tattoos : adam_tattoos x = 23 → x = 2 := by
  intro h
  sorry

end jason_arms_tattoos_l231_231854


namespace Jace_post_break_time_correct_l231_231010

noncomputable def Jace_post_break_time (total_distance : ℝ) (speed : ℝ) (pre_break_time : ℝ) : ℝ :=
  (total_distance - (speed * pre_break_time)) / speed

theorem Jace_post_break_time_correct :
  Jace_post_break_time 780 60 4 = 9 :=
by
  sorry

end Jace_post_break_time_correct_l231_231010


namespace molecular_weight_of_7_moles_of_CaO_l231_231164

/-- The molecular weight of 7 moles of calcium oxide (CaO) -/
def Ca_atomic_weight : Float := 40.08
def O_atomic_weight : Float := 16.00
def CaO_molecular_weight : Float := Ca_atomic_weight + O_atomic_weight

theorem molecular_weight_of_7_moles_of_CaO : 
    7 * CaO_molecular_weight = 392.56 := by 
sorry

end molecular_weight_of_7_moles_of_CaO_l231_231164


namespace area_hexagon_STUVWX_l231_231025

noncomputable def area_of_hexagon (area_PQR : ℕ) (small_area : ℕ) : ℕ := 
  area_PQR - (3 * small_area)

theorem area_hexagon_STUVWX : 
  let area_PQR := 45
  let small_area := 1 
  ∃ area_hexagon, area_hexagon = 42 := 
by
  let area_PQR := 45
  let small_area := 1
  let area_hexagon := area_of_hexagon area_PQR small_area
  use area_hexagon
  sorry

end area_hexagon_STUVWX_l231_231025


namespace alcohol_percentage_l231_231018

theorem alcohol_percentage (P : ℝ) : 
  (0.10 * 300) + (P / 100 * 450) = 0.22 * 750 → P = 30 :=
by
  intros h
  sorry

end alcohol_percentage_l231_231018


namespace correct_factorization_l231_231714

theorem correct_factorization :
  (x^2 - 2 * x + 1 = (x - 1)^2) ∧ 
  (¬ (x^2 - 4 * y^2 = (x + y) * (x - 4 * y))) ∧ 
  (¬ ((x + 4) * (x - 4) = x^2 - 16)) ∧ 
  (¬ (x^2 - 8 * x + 9 = (x - 4)^2 - 7)) :=
by
  sorry

end correct_factorization_l231_231714


namespace max_area_of_right_triangle_with_hypotenuse_4_l231_231001

theorem max_area_of_right_triangle_with_hypotenuse_4 : 
  (∀ (a b : ℝ), a^2 + b^2 = 16 → (∃ S, S = 1/2 * a * b ∧ S ≤ 4)) ∧ 
  (∃ a b : ℝ, a^2 + b^2 = 16 ∧ a = b ∧ 1/2 * a * b = 4) :=
by
  sorry

end max_area_of_right_triangle_with_hypotenuse_4_l231_231001


namespace negation_proposition_l231_231139

theorem negation_proposition : ∀ (a : ℝ), (a > 3) → (a^2 ≥ 9) :=
by
  intros a ha
  sorry

end negation_proposition_l231_231139


namespace flower_bed_area_l231_231250

noncomputable def area_of_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  (1/2) * a * b

theorem flower_bed_area : 
  area_of_triangle 6 8 10 (by norm_num) = 24 := 
sorry

end flower_bed_area_l231_231250


namespace find_real_x_l231_231387

theorem find_real_x (x : ℝ) : 
  (2 ≤ 3 * x / (3 * x - 7)) ∧ (3 * x / (3 * x - 7) < 6) ↔ (7 / 3 < x ∧ x < 42 / 15) :=
by
  sorry

end find_real_x_l231_231387


namespace twin_primes_solution_l231_231158

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def are_twin_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ (p = q + 2 ∨ q = p + 2)

theorem twin_primes_solution (p q : ℕ) :
  are_twin_primes p q ∧ is_prime (p^2 - p * q + q^2) ↔ (p, q) = (5, 3) ∨ (p, q) = (3, 5) := by
  sorry

end twin_primes_solution_l231_231158


namespace maximum_rectangle_area_l231_231534

variable (x y : ℝ)

def area (x y : ℝ) : ℝ :=
  x * y

def similarity_condition (x y : ℝ) : Prop :=
  (11 - x) / (y - 6) = 2

theorem maximum_rectangle_area :
  ∃ (x y : ℝ), similarity_condition x y ∧ area x y = 66 :=  by
  sorry

end maximum_rectangle_area_l231_231534


namespace scooter_price_and_installment_l231_231558

variable {P : ℝ} -- price of the scooter
variable {m : ℝ} -- monthly installment

theorem scooter_price_and_installment (h1 : 0.2 * P = 240) (h2 : (0.8 * P) = 12 * m) : 
  P = 1200 ∧ m = 80 := by
  sorry

end scooter_price_and_installment_l231_231558


namespace max_difference_proof_l231_231183

-- Define the revenue function R(x)
def R (x : ℕ+) : ℝ := 3000 * (x : ℝ) - 20 * (x : ℝ) ^ 2

-- Define the cost function C(x)
def C (x : ℕ+) : ℝ := 500 * (x : ℝ) + 4000

-- Define the profit function P(x) as revenue minus cost
def P (x : ℕ+) : ℝ := R x - C x

-- Define the marginal function M
def M (f : ℕ+ → ℝ) (x : ℕ+) : ℝ := f (⟨x + 1, Nat.succ_pos x⟩) - f x

-- Define the marginal profit function MP(x)
def MP (x : ℕ+) : ℝ := M P x

-- Statement of the proof
theorem max_difference_proof : 
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → P x ≤ P x_max) → -- P achieves its maximum at some x_max within constraints
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → MP x ≤ MP x_max) → -- MP achieves its maximum at some x_max within constraints
  (P x_max - MP x_max = 71680) := 
sorry -- proof omitted

end max_difference_proof_l231_231183


namespace simplify_expression_l231_231227

variable {a b : ℝ}

theorem simplify_expression : (4 * a^3 * b - 2 * a * b) / (2 * a * b) = 2 * a^2 - 1 := by
  sorry

end simplify_expression_l231_231227


namespace ratio_matt_fem_4_1_l231_231383

-- Define Fem's current age
def FemCurrentAge : ℕ := 11

-- Define the condition about the sum of their ages in two years
def AgeSumInTwoYears (MattCurrentAge : ℕ) : Prop :=
  (FemCurrentAge + 2) + (MattCurrentAge + 2) = 59

-- Define the desired ratio as a property
def DesiredRatio (MattCurrentAge : ℕ) : Prop :=
  MattCurrentAge / FemCurrentAge = 4

-- Create the theorem statement
theorem ratio_matt_fem_4_1 (M : ℕ) (h : AgeSumInTwoYears M) : DesiredRatio M :=
  sorry

end ratio_matt_fem_4_1_l231_231383


namespace exponents_to_99_l231_231564

theorem exponents_to_99 :
  (1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99) :=
sorry

end exponents_to_99_l231_231564


namespace female_cows_percentage_l231_231729

theorem female_cows_percentage (TotalCows PregnantFemaleCows : Nat) (PregnantPercentage : ℚ)
    (h1 : TotalCows = 44)
    (h2 : PregnantFemaleCows = 11)
    (h3 : PregnantPercentage = 0.50) :
    (PregnantFemaleCows / PregnantPercentage / TotalCows) * 100 = 50 := 
sorry

end female_cows_percentage_l231_231729


namespace triangle_inequality_l231_231126
-- Import necessary libraries

-- Define the problem
theorem triangle_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (α β γ : ℝ) (h_alpha : α = 2 * Real.sqrt (b * c)) (h_beta : β = 2 * Real.sqrt (c * a)) (h_gamma : γ = 2 * Real.sqrt (a * b)) :
  (a / α) + (b / β) + (c / γ) ≥ (3 / 2) :=
by
  sorry

end triangle_inequality_l231_231126


namespace parallel_vectors_x_value_l231_231279

-- Defining the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Condition for vectors a and b to be parallel
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value : ∃ x, are_parallel a (b x) ∧ x = 6 := by
  sorry

end parallel_vectors_x_value_l231_231279


namespace ones_digit_of_34_34_times_17_17_is_6_l231_231256

def cyclical_pattern_4 (n : ℕ) : ℕ :=
if n % 2 = 0 then 6 else 4

theorem ones_digit_of_34_34_times_17_17_is_6
  (h1 : 34 % 10 = 4)
  (h2 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4)
  (h3 : 17 % 2 = 1)
  (h4 : (34 * 17^17) % 2 = 0)
  (h5 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4) :
  (34^(34 * 17^17)) % 10 = 6 := 
by  
  sorry

end ones_digit_of_34_34_times_17_17_is_6_l231_231256


namespace point_P_through_graph_l231_231009

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x - 1)

theorem point_P_through_graph (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : 
  f a 1 = 5 :=
by
  unfold f
  sorry

end point_P_through_graph_l231_231009


namespace quadrilateral_choices_l231_231727

theorem quadrilateral_choices :
  let available_rods : List ℕ := (List.range' 1 41).diff [5, 12, 20]
  let valid_rods := available_rods.filter (λ x => 4 ≤ x ∧ x ≤ 36)
  valid_rods.length = 30 := sorry

end quadrilateral_choices_l231_231727


namespace systematic_sampling_condition_l231_231688

theorem systematic_sampling_condition (population sample_size total_removed segments individuals_per_segment : ℕ) 
  (h_population : population = 1650)
  (h_sample_size : sample_size = 35)
  (h_total_removed : total_removed = 5)
  (h_segments : segments = sample_size)
  (h_individuals_per_segment : individuals_per_segment = (population - total_removed) / sample_size)
  (h_modulo : population % sample_size = total_removed)
  :
  total_removed = 5 ∧ segments = 35 ∧ individuals_per_segment = 47 := 
by
  sorry

end systematic_sampling_condition_l231_231688


namespace largest_six_digit_number_l231_231457

/-- The largest six-digit number \( A \) that is divisible by 19, 
  the number obtained by removing its last digit is divisible by 17, 
  and the number obtained by removing the last two digits in \( A \) is divisible by 13 
  is \( 998412 \). -/
theorem largest_six_digit_number (A : ℕ) (h1 : A % 19 = 0) 
  (h2 : (A / 10) % 17 = 0) 
  (h3 : (A / 100) % 13 = 0) : 
  A = 998412 :=
sorry

end largest_six_digit_number_l231_231457


namespace total_number_of_items_in_base10_l231_231755

theorem total_number_of_items_in_base10 : 
  let clay_tablets := (2 * 5^0 + 3 * 5^1 + 4 * 5^2 + 1 * 5^3)
  let bronze_sculptures := (1 * 5^0 + 4 * 5^1 + 0 * 5^2 + 2 * 5^3)
  let stone_carvings := (2 * 5^0 + 3 * 5^1 + 2 * 5^2)
  let total_items := clay_tablets + bronze_sculptures + stone_carvings
  total_items = 580 := by
  sorry

end total_number_of_items_in_base10_l231_231755


namespace max_x_inequality_k_l231_231944

theorem max_x_inequality_k (k : ℝ) (h : ∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) : k = 8 :=
sorry

end max_x_inequality_k_l231_231944


namespace intersected_squares_and_circles_l231_231657

def is_intersected_by_line (p q : ℕ) : Prop :=
  p = q

def total_intersections : ℕ := 504 * 2

theorem intersected_squares_and_circles :
  total_intersections = 1008 :=
by
  sorry

end intersected_squares_and_circles_l231_231657


namespace find_tangent_line_l231_231180

def is_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

def is_tangent_to_circle (a b c : ℝ) : Prop :=
  let d := abs c / (Real.sqrt (a^2 + b^2))
  d = 1

def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem find_tangent_line :
  ∀ (k b : ℝ),
    is_perpendicular k 1 →
    is_tangent_to_circle 1 1 b →
    ∃ (x y : ℝ), in_first_quadrant x y ∧ x + y - b = 0 →
    b = Real.sqrt 2 := sorry

end find_tangent_line_l231_231180


namespace quadratic_has_solution_l231_231100

theorem quadratic_has_solution (a b : ℝ) : ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 :=
  by sorry

end quadratic_has_solution_l231_231100


namespace speed_first_hour_l231_231997

variable (x : ℕ)

-- Definitions based on conditions
def total_distance (x : ℕ) : ℕ := x + 50
def average_speed (x : ℕ) : Prop := (total_distance x) / 2 = 70

-- Theorem statement
theorem speed_first_hour : ∃ x, average_speed x ∧ x = 90 := by
  sorry

end speed_first_hour_l231_231997


namespace andy_wrong_questions_l231_231365

theorem andy_wrong_questions (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 3) : a = 6 := by
  sorry

end andy_wrong_questions_l231_231365


namespace sequence_general_term_l231_231616

namespace SequenceSum

def Sn (n : ℕ) : ℕ :=
  2 * n^2 + n

def a₁ (n : ℕ) : ℕ :=
  if n = 1 then Sn n else (Sn n - Sn (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n > 0) : 
  a₁ n = 4 * n - 1 :=
sorry

end SequenceSum

end sequence_general_term_l231_231616


namespace positive_numbers_with_cube_root_lt_10_l231_231663

def cube_root_lt_10 (n : ℕ) : Prop :=
  (↑n : ℝ)^(1 / 3 : ℝ) < 10

theorem positive_numbers_with_cube_root_lt_10 : 
  ∃ (count : ℕ), (count = 999) ∧ ∀ n : ℕ, (1 ≤ n ∧ n ≤ 999) → cube_root_lt_10 n :=
by
  sorry

end positive_numbers_with_cube_root_lt_10_l231_231663


namespace relay_race_time_reduction_l231_231265

theorem relay_race_time_reduction
    (T T1 T2 T3 T4 T5 : ℝ)
    (h1 : T1 = 0.1 * T)
    (h2 : T2 = 0.2 * T)
    (h3 : T3 = 0.24 * T)
    (h4 : T4 = 0.3 * T)
    (h5 : T5 = 0.16 * T) :
    ((T1 + T2 + T3 + T4 + T5) - (T1 + T2 + T3 + T4 + T5 / 2)) / (T1 + T2 + T3 + T4 + T5) = 0.08 :=
by
  sorry

end relay_race_time_reduction_l231_231265


namespace find_positive_number_l231_231190

theorem find_positive_number (x : ℝ) (h : x > 0) (h1 : x + 17 = 60 * (1 / x)) : x = 3 :=
sorry

end find_positive_number_l231_231190


namespace log_expression_value_l231_231717

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_expression_value :
  log_base 3 32 * log_base 4 9 - log_base 2 (3/4) + log_base 2 6 = 8 := 
by 
  sorry

end log_expression_value_l231_231717


namespace tank_fraction_l231_231736

theorem tank_fraction (x : ℚ) : 
  let tank1_capacity := 7000
  let tank2_capacity := 5000
  let tank3_capacity := 3000
  let tank2_fraction := 4 / 5
  let tank3_fraction := 1 / 2
  let total_water := 10850
  tank1_capacity * x + tank2_capacity * tank2_fraction + tank3_capacity * tank3_fraction = total_water → 
  x = 107 / 140 := 
by {
  sorry
}

end tank_fraction_l231_231736


namespace solve_equation_l231_231267

theorem solve_equation (x : ℝ) (h : (x - 7) / 2 - (1 + x) / 3 = 1) : x = 29 :=
sorry

end solve_equation_l231_231267


namespace inequality_x_pow_n_ge_n_x_l231_231165

theorem inequality_x_pow_n_ge_n_x (x : ℝ) (n : ℕ) (h1 : x ≠ 0) (h2 : x > -1) (h3 : n > 0) : 
  (1 + x)^n ≥ n * x := by
  sorry

end inequality_x_pow_n_ge_n_x_l231_231165


namespace height_of_Linda_room_l231_231011

theorem height_of_Linda_room (w l: ℝ) (h a1 a2 a3 paint_area: ℝ) 
  (hw: w = 20) (hl: l = 20) 
  (d1_h: a1 = 3) (d1_w: a2 = 7) 
  (d2_h: a3 = 4) (d2_w: a4 = 6) 
  (d3_h: a5 = 5) (d3_w: a6 = 7) 
  (total_paint_area: paint_area = 560):
  h = 6 := 
by
  sorry

end height_of_Linda_room_l231_231011


namespace people_joined_l231_231889

theorem people_joined (total_left : ℕ) (total_remaining : ℕ) (Molly_and_parents : ℕ)
  (h1 : total_left = 40) (h2 : total_remaining = 63) (h3 : Molly_and_parents = 3) :
  ∃ n, n = 100 := 
by
  sorry

end people_joined_l231_231889


namespace find_a_l231_231828

-- Definitions of universal set U, set P, and complement of P in U
def U (a : ℤ) : Set ℤ := {2, 4, 3 - a^2}
def P (a : ℤ) : Set ℤ := {2, a^2 - a + 2}
def complement_U_P (a : ℤ) : Set ℤ := {-1}

-- The Lean statement asserting the conditions and the proof goal
theorem find_a (a : ℤ) (h_union : U a = P a ∪ complement_U_P a) : a = -1 :=
sorry

end find_a_l231_231828


namespace mason_car_nuts_l231_231053

def busy_squirrels_num := 2
def busy_squirrel_nuts_per_day := 30
def sleepy_squirrel_num := 1
def sleepy_squirrel_nuts_per_day := 20
def days := 40

theorem mason_car_nuts : 
  busy_squirrels_num * busy_squirrel_nuts_per_day * days + sleepy_squirrel_nuts_per_day * days = 3200 :=
  by
    sorry

end mason_car_nuts_l231_231053


namespace solve_equation_l231_231232

-- Given conditions and auxiliary definitions
def is_solution (x y z : ℕ) : Prop := 2 ^ x + 3 ^ y - 7 = Nat.factorial z

-- Primary theorem: the equivalent proof problem
theorem solve_equation (x y z : ℕ) :
  (is_solution x y 3 → (x = 2 ∧ y = 2)) ∧
  (∀ z, (z ≤ 3 → z ≠ 3) → ¬is_solution x y z) ∧
  (z ≥ 4 → ¬is_solution x y z) :=
  sorry

end solve_equation_l231_231232


namespace range_of_values_for_sqrt_l231_231470

theorem range_of_values_for_sqrt (x : ℝ) : (x + 3 ≥ 0) ↔ (x ≥ -3) :=
by
  sorry

end range_of_values_for_sqrt_l231_231470


namespace sum_of_permutations_is_divisible_by_37_l231_231844

theorem sum_of_permutations_is_divisible_by_37
  (A B C : ℕ)
  (h : 37 ∣ (100 * A + 10 * B + C)) :
  37 ∣ (100 * B + 10 * C + A + 100 * C + 10 * A + B) :=
by
  sorry

end sum_of_permutations_is_divisible_by_37_l231_231844


namespace one_real_root_multiple_coinciding_roots_three_distinct_real_roots_three_coinciding_roots_at_origin_l231_231871

-- Definitions from conditions
def cubic_eq (x p q : ℝ) := x^3 + p * x + q

-- Correct answers in mathematical proofs
theorem one_real_root (p q : ℝ) : 4 * p^3 + 27 * q^2 > 0 → ∃ x : ℝ, cubic_eq x p q = 0 := sorry

theorem multiple_coinciding_roots (p q : ℝ) : 4 * p^3 + 27 * q^2 = 0 ∧ (p ≠ 0 ∨ q ≠ 0) → ∃ x : ℝ, cubic_eq x p q = 0 := sorry

theorem three_distinct_real_roots (p q : ℝ) : 4 * p^3 + 27 * q^2 < 0 → ∃ x₁ x₂ x₃ : ℝ, 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ cubic_eq x₁ p q = 0 ∧ cubic_eq x₂ p q = 0 ∧ cubic_eq x₃ p q = 0 := sorry

theorem three_coinciding_roots_at_origin : ∃ x : ℝ, cubic_eq x 0 0 = 0 := sorry

end one_real_root_multiple_coinciding_roots_three_distinct_real_roots_three_coinciding_roots_at_origin_l231_231871


namespace edges_parallel_to_axes_l231_231700

theorem edges_parallel_to_axes (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ)
  (hx : x1 = 0 ∨ y1 = 0 ∨ z1 = 0)
  (hy : x2 = x1 + 1 ∨ y2 = y1 + 1 ∨ z2 = z1 + 1)
  (hz : x3 = x1 + 1 ∨ y3 = y1 + 1 ∨ z3 = z1 + 1)
  (hv : x4*y4*z4 = 2011) :
  (x2-x1 ∣ 2011) ∧ (y2-y1 ∣ 2011) ∧ (z2-z1 ∣ 2011) := 
sorry

end edges_parallel_to_axes_l231_231700


namespace arithmetic_sequence_value_l231_231991

theorem arithmetic_sequence_value 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : 4 * a 3 + a 11 - 3 * a 5 = 10) : 
  (1 / 5 * a 4 = 1) := 
by
  sorry

end arithmetic_sequence_value_l231_231991


namespace geometric_sequence_property_l231_231049

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geom: ∀ n, a (n + 1) = a n * r) 
  (h_pos: ∀ n, a n > 0)
  (h_root1: a 3 * a 15 = 8)
  (h_root2: a 3 + a 15 = 6) :
  a 1 * a 17 / a 9 = 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_property_l231_231049


namespace focus_of_parabola_y_eq_x_sq_l231_231044

theorem focus_of_parabola_y_eq_x_sq : ∃ (f : ℝ × ℝ), f = (0, 1/4) ∧ (∃ (p : ℝ), p = 1/2 ∧ ∀ x, y = x^2 → y = 2 * p * (0, y).snd) :=
by
  sorry

end focus_of_parabola_y_eq_x_sq_l231_231044


namespace g_odd_l231_231544

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_odd {x₁ x₂ : ℝ} 
  (h₁ : |f x₁ + f x₂| ≥ |g x₁ + g x₂|)
  (hf_odd : ∀ x, f x = -f (-x)) : ∀ x, g x = -g (-x) :=
by
  -- The proof would go here, but it's omitted for the purpose of this translation.
  sorry

end g_odd_l231_231544


namespace point_coordinates_l231_231321

def point_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0 

theorem point_coordinates (m : ℝ) 
  (h1 : point_in_second_quadrant (-m-1) (2*m+1))
  (h2 : |2*m + 1| = 5) : (-m-1, 2*m+1) = (-3, 5) :=
sorry

end point_coordinates_l231_231321


namespace max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l231_231316

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 5*x + 5) / Real.exp x

theorem max_value_of_f_at_0 :
  f 0 = 5 := by
  sorry

theorem min_value_of_f_on_neg_inf_to_0 :
  f (-3) = -Real.exp 3 := by
  sorry

theorem range_of_a_for_ineq :
  ∀ x : ℝ, x^2 + 5*x + 5 - a * Real.exp x ≥ 0 ↔ a ≤ -Real.exp 3 := by
  sorry

end max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l231_231316


namespace fish_population_estimate_l231_231412

theorem fish_population_estimate :
  ∃ N : ℕ, (60 * 60) / 2 = N ∧ (2 / 60 : ℚ) = (60 / N : ℚ) :=
by
  use 1800
  simp
  sorry

end fish_population_estimate_l231_231412


namespace range_of_k_l231_231326

noncomputable def operation (a b : ℝ) : ℝ := Real.sqrt (a * b) + a + b

theorem range_of_k (k : ℝ) (h : operation 1 (k^2) < 3) : -1 < k ∧ k < 1 :=
by
  sorry

end range_of_k_l231_231326


namespace luke_points_per_round_l231_231525

-- Definitions for conditions
def total_points : ℤ := 84
def rounds : ℤ := 2
def points_per_round (total_points rounds : ℤ) : ℤ := total_points / rounds

-- Statement of the problem
theorem luke_points_per_round : points_per_round total_points rounds = 42 := 
by 
  sorry

end luke_points_per_round_l231_231525


namespace student_walks_fifth_to_first_l231_231255

theorem student_walks_fifth_to_first :
  let floors := 4
  let staircases := 2
  (staircases ^ floors) = 16 := by
  sorry

end student_walks_fifth_to_first_l231_231255


namespace has_exactly_two_solutions_iff_l231_231626

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l231_231626


namespace Sophie_l231_231008

-- Define the prices of each item
def price_cupcake : ℕ := 2
def price_doughnut : ℕ := 1
def price_apple_pie : ℕ := 2
def price_cookie : ℚ := 0.60

-- Define the quantities of each item
def qty_cupcake : ℕ := 5
def qty_doughnut : ℕ := 6
def qty_apple_pie : ℕ := 4
def qty_cookie : ℕ := 15

-- Define the total cost function for each item
def cost_cupcake := qty_cupcake * price_cupcake
def cost_doughnut := qty_doughnut * price_doughnut
def cost_apple_pie := qty_apple_pie * price_apple_pie
def cost_cookie := qty_cookie * price_cookie

-- Define total expenditure
def total_expenditure := cost_cupcake + cost_doughnut + cost_apple_pie + cost_cookie

-- Assertion of total expenditure
theorem Sophie's_total_expenditure : total_expenditure = 33 := by
  -- skipping proof
  sorry

end Sophie_l231_231008


namespace Mrs_Fredricksons_chickens_l231_231702

theorem Mrs_Fredricksons_chickens (C : ℕ) (h1 : 1/4 * C + 1/4 * (3/4 * C) = 35) : C = 80 :=
by
  sorry

end Mrs_Fredricksons_chickens_l231_231702


namespace base_7_sum_of_product_l231_231477

-- Definitions of the numbers in base-10 for base-7 numbers
def base_7_to_base_10 (d1 d0 : ℕ) : ℕ := d1 * 7 + d0

def sum_digits_base_7 (n : ℕ) : ℕ := 
  let d2 := n / 343
  let r2 := n % 343
  let d1 := r2 / 49
  let r1 := r2 % 49
  let d0 := r1 / 7 + r1 % 7
  d2 + d1 + d0

def convert_10_to_7 (n : ℕ) : ℕ := 
  let d1 := n / 7
  let r1 := n % 7
  d1 * 10 + r1

theorem base_7_sum_of_product : 
  let n36  := base_7_to_base_10 3 6
  let n52  := base_7_to_base_10 5 2
  let nadd := base_7_to_base_10 2 0
  let prod := n36 * n52
  let suma := prod + nadd
  convert_10_to_7 (sum_digits_base_7 suma) = 23 :=
by
  sorry

end base_7_sum_of_product_l231_231477


namespace five_x_minus_two_l231_231445

theorem five_x_minus_two (x : ℚ) (h : 4 * x - 8 = 13 * x + 3) : 5 * (x - 2) = -145 / 9 := by
  sorry

end five_x_minus_two_l231_231445


namespace original_price_of_cycle_l231_231762

theorem original_price_of_cycle 
    (selling_price : ℝ) 
    (loss_percentage : ℝ) 
    (h1 : selling_price = 1120)
    (h2 : loss_percentage = 0.20) : 
    ∃ P : ℝ, P = 1400 :=
by
  sorry

end original_price_of_cycle_l231_231762


namespace circle_properties_l231_231512

noncomputable def circle_eq (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0
noncomputable def line_eq (x y : ℝ) := x + 2*y - 4 = 0
noncomputable def perpendicular (x1 y1 x2 y2 : ℝ) := 
  (x1 * x2 + y1 * y2 = 0)

theorem circle_properties (m : ℝ) (x1 y1 x2 y2 : ℝ) :
  (∀ x y, circle_eq x y m) →
  (∀ x, line_eq x (y1 + y2)) →
  perpendicular (4 - 2*y1) y1 (4 - 2*y2) y2 →
  m = 8 / 5 ∧ 
  (∀ x y, (x^2 + y^2 - (8 / 5) * x - (16 / 5) * y = 0) ↔ 
           (x - (4 - 2*(16/5))) * (x - (4 - 2*(16/5))) + (y - (16/5)) * (y - (16/5)) = 5 - (8/5)) :=
sorry

end circle_properties_l231_231512


namespace point_symmetric_y_axis_l231_231260

theorem point_symmetric_y_axis (a b : ℤ) (h₁ : a = -(-2)) (h₂ : b = 3) : a + b = 5 := by
  sorry

end point_symmetric_y_axis_l231_231260


namespace find_possible_values_of_a_l231_231514

theorem find_possible_values_of_a (a b c : ℝ) (h1 : a * b + a + b = c) (h2 : b * c + b + c = a) (h3 : c * a + c + a = b) :
  a = 0 ∨ a = -1 ∨ a = -2 :=
by
  sorry

end find_possible_values_of_a_l231_231514


namespace susie_rhode_island_reds_l231_231509

variable (R G B_R B_G : ℕ)

def susie_golden_comets := G = 6
def britney_rir := B_R = 2 * R
def britney_golden_comets := B_G = G / 2
def britney_more_chickens := B_R + B_G = R + G + 8

theorem susie_rhode_island_reds
  (h1 : susie_golden_comets G)
  (h2 : britney_rir R B_R)
  (h3 : britney_golden_comets G B_G)
  (h4 : britney_more_chickens R G B_R B_G) :
  R = 11 :=
by
  sorry

end susie_rhode_island_reds_l231_231509


namespace arithmetic_sequence_problem_l231_231040

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) (h_incr : ∀ n, a (n + 1) > a n) (h_prod : a 4 * a 5 = 13) : a 3 * a 6 = -275 := 
sorry

end arithmetic_sequence_problem_l231_231040


namespace find_white_balls_l231_231145

-- Define the number of red balls
def red_balls : ℕ := 4

-- Define the probability of drawing a red ball
def prob_red : ℚ := 1 / 4

-- Define the number of white balls
def white_balls : ℕ := 12

theorem find_white_balls (x : ℕ) (h1 : (red_balls : ℚ) / (red_balls + x) = prob_red) : x = white_balls :=
by
  -- Proof is omitted
  sorry

end find_white_balls_l231_231145


namespace count_reflectional_symmetry_l231_231934

def tetrominoes : List String := ["I", "O", "T", "S", "Z", "L", "J"]

def has_reflectional_symmetry (tetromino : String) : Bool :=
  match tetromino with
  | "I" => true
  | "O" => true
  | "T" => true
  | "S" => false
  | "Z" => false
  | "L" => false
  | "J" => false
  | _   => false

theorem count_reflectional_symmetry : 
  (tetrominoes.filter has_reflectional_symmetry).length = 3 := by
  sorry

end count_reflectional_symmetry_l231_231934


namespace lengths_available_total_cost_l231_231032

def available_lengths := [1, 2, 3, 4, 5, 6]
def pipe_prices := [10, 15, 20, 25, 30, 35]

-- Given conditions
def purchased_pipes := [2, 5]
def target_perimeter_is_even := True

-- Prove: 
theorem lengths_available (x : ℕ) (hx : x ∈ available_lengths) : 
  3 < x ∧ x < 7 → x = 4 ∨ x = 5 ∨ x = 6 := by
  sorry

-- Prove: 
theorem total_cost (p : ℕ) (h : target_perimeter_is_even) : 
  p = 75 := by
  sorry

end lengths_available_total_cost_l231_231032


namespace budget_allocation_l231_231757

theorem budget_allocation 
  (total_degrees : ℝ := 360)
  (total_budget : ℝ := 100)
  (degrees_basic_astrophysics : ℝ := 43.2)
  (percent_microphotonics : ℝ := 12)
  (percent_home_electronics : ℝ := 24)
  (percent_food_additives : ℝ := 15)
  (percent_industrial_lubricants : ℝ := 8) :
  ∃ percent_genetically_modified_microorganisms : ℝ,
  percent_genetically_modified_microorganisms = 29 :=
sorry

end budget_allocation_l231_231757


namespace no_a_where_A_eq_B_singleton_l231_231353

def f (a x : ℝ) := x^2 + 4 * x - 2 * a
def g (a x : ℝ) := x^2 - a * x + a + 3

theorem no_a_where_A_eq_B_singleton :
  ∀ a : ℝ,
    (∃ x₁ : ℝ, (f a x₁ ≤ 0 ∧ ∀ x₂, f a x₂ ≤ 0 → x₂ = x₁)) ∧
    (∃ y₁ : ℝ, (g a y₁ ≤ 0 ∧ ∀ y₂, g a y₂ ≤ 0 → y₂ = y₁)) →
    (¬ ∃ z : ℝ, (f a z ≤ 0) ∧ (g a z ≤ 0)) := 
by
  sorry

end no_a_where_A_eq_B_singleton_l231_231353


namespace expression_takes_many_different_values_l231_231007

theorem expression_takes_many_different_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) : 
  ∃ v : ℝ, ∀ x, x ≠ 3 → x ≠ -2 → v = (3*x^2 - 2*x + 3)/((x - 3)*(x + 2)) - (5*x - 6)/((x - 3)*(x + 2)) := 
sorry

end expression_takes_many_different_values_l231_231007


namespace max_value_of_b_over_a_squared_l231_231774

variables {a b x y : ℝ}

def triangle_is_right (a b x y : ℝ) : Prop :=
  (a - x)^2 + (b - y)^2 = a^2 + b^2

theorem max_value_of_b_over_a_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b)
    (h4 : ∃ x y, a^2 + y^2 = b^2 + x^2 
                 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2
                 ∧ 0 ≤ x ∧ x < a 
                 ∧ 0 ≤ y ∧ y < b 
                 ∧ triangle_is_right a b x y) 
    : (b / a)^2 = 4 / 3 :=
sorry

end max_value_of_b_over_a_squared_l231_231774


namespace arcsin_sqrt_3_div_2_is_pi_div_3_l231_231351

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_3_div_2_is_pi_div_3_l231_231351


namespace percentage_of_Indian_women_l231_231198

-- Definitions of conditions
def total_people := 700 + 500 + 800
def indian_men := (20 / 100) * 700
def indian_children := (10 / 100) * 800
def total_indian_people := (21 / 100) * total_people
def indian_women := total_indian_people - indian_men - indian_children

-- Statement of the theorem
theorem percentage_of_Indian_women : 
  (indian_women / 500) * 100 = 40 :=
by
  sorry

end percentage_of_Indian_women_l231_231198


namespace monotonicity_of_f_extremum_of_f_on_interval_l231_231147

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem monotonicity_of_f : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → 1 ≤ x₂ → f x₁ < f x₂ := by
  sorry

theorem extremum_of_f_on_interval : 
  f 1 = 3 / 2 ∧ f 4 = 9 / 5 := by
  sorry

end monotonicity_of_f_extremum_of_f_on_interval_l231_231147


namespace pipeline_equation_correct_l231_231778

variables (m x n : ℝ) -- Length of the pipeline, kilometers per day, efficiency increase percentage
variable (h : 0 < n) -- Efficiency increase percentage is positive

theorem pipeline_equation_correct :
  (m / x) - (m / ((1 + (n / 100)) * x)) = 8 :=
sorry -- Proof omitted

end pipeline_equation_correct_l231_231778


namespace slope_of_line_l231_231162

theorem slope_of_line (x y : ℝ) (h : 3 * y = 4 * x + 9) : 4 / 3 = 4 / 3 :=
by sorry

end slope_of_line_l231_231162


namespace complement_intersection_l231_231358

-- Define sets P and Q.
def P : Set ℝ := {x | x ≥ 2}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define the complement of P.
def complement_P : Set ℝ := {x | x < 2}

-- The theorem we need to prove.
theorem complement_intersection : complement_P ∩ Q = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end complement_intersection_l231_231358


namespace average_visitors_per_day_l231_231136

theorem average_visitors_per_day:
  (∃ (Sundays OtherDays: ℕ) (visitors_per_sunday visitors_per_other_day: ℕ),
    Sundays = 4 ∧
    OtherDays = 26 ∧
    visitors_per_sunday = 600 ∧
    visitors_per_other_day = 240 ∧
    (Sundays + OtherDays = 30) ∧
    (Sundays * visitors_per_sunday + OtherDays * visitors_per_other_day) / 30 = 288) :=
sorry

end average_visitors_per_day_l231_231136


namespace find_a5_l231_231685

noncomputable def arithmetic_sequence (n : ℕ) (a d : ℤ) : ℤ :=
a + n * d

theorem find_a5 (a d : ℤ) (a_2_a_4_sum : arithmetic_sequence 1 a d + arithmetic_sequence 3 a d = 16)
  (a1 : arithmetic_sequence 0 a d = 1) :
  arithmetic_sequence 4 a d = 15 :=
by
  sorry

end find_a5_l231_231685


namespace speed_first_half_proof_l231_231451

noncomputable def speed_first_half
  (total_time: ℕ) 
  (distance: ℕ) 
  (second_half_speed: ℕ) 
  (first_half_time: ℕ) :
  ℕ :=
  distance / first_half_time

theorem speed_first_half_proof
  (total_time: ℕ)
  (distance: ℕ)
  (second_half_speed: ℕ)
  (half_distance: ℕ)
  (second_half_time: ℕ)
  (first_half_time: ℕ) :
  total_time = 12 →
  distance = 560 →
  second_half_speed = 40 →
  half_distance = distance / 2 →
  second_half_time = half_distance / second_half_speed →
  first_half_time = total_time - second_half_time →
  speed_first_half total_time half_distance second_half_speed first_half_time = 56 :=
by
  sorry

end speed_first_half_proof_l231_231451


namespace maximize_area_l231_231654

noncomputable def optimal_fencing (L W : ℝ) : Prop :=
  (2 * L + W = 1200) ∧ (∀ L1 W1, 2 * L1 + W1 = 1200 → L * W ≥ L1 * W1)

theorem maximize_area : ∃ L W, optimal_fencing L W ∧ L + W = 900 := sorry

end maximize_area_l231_231654


namespace sufficient_but_not_necessary_l231_231122

theorem sufficient_but_not_necessary {a b : ℝ} (h : a > b ∧ b > 0) : 
  a^2 > b^2 ∧ (¬ (a^2 > b^2 → a > b ∧ b > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_l231_231122


namespace white_pairs_coincide_l231_231942

theorem white_pairs_coincide
  (red_triangles_half : ℕ)
  (blue_triangles_half : ℕ)
  (white_triangles_half : ℕ)
  (red_pairs : ℕ)
  (blue_pairs : ℕ)
  (red_white_pairs : ℕ)
  (red_triangles_total_half : red_triangles_half = 4)
  (blue_triangles_total_half : blue_triangles_half = 6)
  (white_triangles_total_half : white_triangles_half = 10)
  (red_pairs_total : red_pairs = 3)
  (blue_pairs_total : blue_pairs = 4)
  (red_white_pairs_total : red_white_pairs = 3) :
  ∃ w : ℕ, w = 5 :=
by
  sorry

end white_pairs_coincide_l231_231942


namespace number_of_dolls_l231_231305

theorem number_of_dolls (total_toys : ℕ) (fraction_action_figures : ℚ) 
  (remaining_fraction_action_figures : fraction_action_figures = 1 / 4) 
  (remaining_fraction_dolls : 1 - fraction_action_figures = 3 / 4) 
  (total_toys_eq : total_toys = 24) : 
  (total_toys - total_toys * fraction_action_figures) = 18 := 
by 
  sorry

end number_of_dolls_l231_231305


namespace solve_mod_equation_l231_231591

theorem solve_mod_equation (y b n : ℤ) (h1 : 15 * y + 4 ≡ 7 [ZMOD 18]) (h2 : y ≡ b [ZMOD n]) (h3 : 2 ≤ n) (h4 : b < n) : b + n = 11 :=
sorry

end solve_mod_equation_l231_231591


namespace sqrt_inequality_l231_231495

theorem sqrt_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) : 
  x^2 + y^2 + 1 ≤ Real.sqrt ((x^3 + y + 1) * (y^3 + x + 1)) :=
sorry

end sqrt_inequality_l231_231495


namespace min_value_of_z_l231_231972

noncomputable def min_z (x y : ℝ) : ℝ :=
  2 * x + (Real.sqrt 3) * y

theorem min_value_of_z :
  ∃ x y : ℝ, 3 * x^2 + 4 * y^2 = 12 ∧ min_z x y = -5 :=
sorry

end min_value_of_z_l231_231972


namespace square_area_l231_231475

theorem square_area (side_length : ℕ) (h : side_length = 16) : side_length * side_length = 256 := by
  sorry

end square_area_l231_231475


namespace polynomial_perfect_square_l231_231623

theorem polynomial_perfect_square (k : ℝ) 
  (h : ∃ a : ℝ, x^2 + 8*x + k = (x + a)^2) : 
  k = 16 :=
by
  sorry

end polynomial_perfect_square_l231_231623


namespace white_washing_cost_l231_231215

theorem white_washing_cost
    (length width height : ℝ)
    (door_width door_height window_width window_height : ℝ)
    (num_doors num_windows : ℝ)
    (paint_cost : ℝ)
    (extra_paint_fraction : ℝ)
    (perimeter := 2 * (length + width))
    (door_area := num_doors * (door_width * door_height))
    (window_area := num_windows * (window_width * window_height))
    (wall_area := perimeter * height)
    (paint_area := wall_area - door_area - window_area)
    (total_area := paint_area * (1 + extra_paint_fraction))
    : total_area * paint_cost = 6652.8 :=
by sorry

end white_washing_cost_l231_231215


namespace max_num_pieces_l231_231300

-- Definition of areas
def largeCake_area : ℕ := 21 * 21
def smallPiece_area : ℕ := 3 * 3

-- Problem Statement
theorem max_num_pieces : largeCake_area / smallPiece_area = 49 := by
  sorry

end max_num_pieces_l231_231300


namespace new_avg_weight_of_boxes_l231_231841

theorem new_avg_weight_of_boxes :
  ∀ (x y : ℕ), x + y = 30 → (10 * x + 20 * y) / 30 = 18 → (10 * x + 20 * (y - 18)) / 12 = 15 :=
by
  intro x y h1 h2
  sorry

end new_avg_weight_of_boxes_l231_231841


namespace problem_inequality_l231_231604

variable (a b : ℝ)

theorem problem_inequality (h_pos : 0 < a) (h_pos' : 0 < b) (h_sum : a + b = 1) :
  (1 / a^2 - a^3) * (1 / b^2 - b^3) ≥ (31 / 8)^2 := 
  sorry

end problem_inequality_l231_231604


namespace mark_profit_l231_231899

def initialPrice : ℝ := 100
def finalPrice : ℝ := 3 * initialPrice
def salesTax : ℝ := 0.05 * initialPrice
def totalInitialCost : ℝ := initialPrice + salesTax
def transactionFee : ℝ := 0.03 * finalPrice
def profitBeforeTax : ℝ := finalPrice - totalInitialCost
def capitalGainsTax : ℝ := 0.15 * profitBeforeTax
def totalProfit : ℝ := profitBeforeTax - transactionFee - capitalGainsTax

theorem mark_profit : totalProfit = 147.75 := sorry

end mark_profit_l231_231899


namespace cos2alpha_minus_sin2alpha_l231_231210

theorem cos2alpha_minus_sin2alpha (α : ℝ) (h1 : α ∈ Set.Icc (-π/2) 0) 
  (h2 : (Real.sin (3 * α)) / (Real.sin α) = 13 / 5) :
  Real.cos (2 * α) - Real.sin (2 * α) = (3 + Real.sqrt 91) / 10 :=
sorry

end cos2alpha_minus_sin2alpha_l231_231210


namespace total_mass_of_individuals_l231_231297

def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def initial_sink_depth : Float := 0.018
def density_of_water : Float := 1000.0
def mass_of_second_person : Float := 75.0

theorem total_mass_of_individuals :
  let V1 := boat_length * boat_breadth * initial_sink_depth
  let m1 := V1 * density_of_water
  let total_mass := m1 + mass_of_second_person
  total_mass = 183 :=
by
  sorry

end total_mass_of_individuals_l231_231297


namespace number_of_companies_l231_231547

theorem number_of_companies (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by
  sorry

end number_of_companies_l231_231547


namespace coordinates_of_P_with_respect_to_y_axis_l231_231006

-- Define the coordinates of point P
def P_x : ℝ := 5
def P_y : ℝ := -1

-- Define the point P
def P : Prod ℝ ℝ := (P_x, P_y)

-- State the theorem
theorem coordinates_of_P_with_respect_to_y_axis :
  (P.1, P.2) = (-P_x, P_y) :=
sorry

end coordinates_of_P_with_respect_to_y_axis_l231_231006


namespace total_expenditure_l231_231950

variable (num_coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_april : ℕ)

theorem total_expenditure (h1 : num_coffees_per_day = 2) (h2 : cost_per_coffee = 2) (h3 : days_in_april = 30) :
  num_coffees_per_day * cost_per_coffee * days_in_april = 120 := by
  sorry

end total_expenditure_l231_231950


namespace problem_l231_231281

variable {w z : ℝ}

theorem problem (hw : w = 8) (hz : z = 3) (h : ∀ z w, z * (w^(1/3)) = 6) : w = 1 :=
by
  sorry

end problem_l231_231281


namespace find_range_of_a_l231_231176

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x => a * (x - 2 * Real.exp 1) * Real.log x + 1

def range_of_a (a : ℝ) : Prop :=
  (a < 0 ∨ a > 1 / Real.exp 1)

theorem find_range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ range_of_a a := by
  sorry

end find_range_of_a_l231_231176


namespace find_y_l231_231587

theorem find_y (h1 : ∀ (a b : ℤ), a * b = (a - 1) * (b - 1)) (h2 : y * 12 = 110) : y = 11 :=
by
  sorry

end find_y_l231_231587


namespace present_age_of_younger_l231_231699

-- Definition based on conditions
variable (y e : ℕ)
variable (h1 : e = y + 20)
variable (h2 : e - 8 = 5 * (y - 8))

-- Statement to be proven
theorem present_age_of_younger (y e: ℕ) (h1: e = y + 20) (h2: e - 8 = 5 * (y - 8)) : y = 13 := 
by 
  sorry

end present_age_of_younger_l231_231699


namespace BretCatchesFrogs_l231_231397

-- Define the number of frogs caught by Alster, Quinn, and Bret.
def AlsterFrogs : Nat := 2
def QuinnFrogs (a : Nat) : Nat := 2 * a
def BretFrogs (q : Nat) : Nat := 3 * q

-- The main theorem to prove
theorem BretCatchesFrogs : BretFrogs (QuinnFrogs AlsterFrogs) = 12 :=
by
  sorry

end BretCatchesFrogs_l231_231397


namespace volume_of_cube_l231_231988

theorem volume_of_cube (a : ℕ) (h : (a^3 - a = a^3 - 5)) : a^3 = 125 :=
by {
  -- The necessary algebraic manipulation follows
  sorry
}

end volume_of_cube_l231_231988


namespace find_monthly_income_l231_231624

-- Given condition
def deposit : ℝ := 3400
def percentage : ℝ := 0.15

-- Goal: Prove Sheela's monthly income
theorem find_monthly_income : (deposit / percentage) = 22666.67 := by
  -- Skip the proof for now
  sorry

end find_monthly_income_l231_231624


namespace correct_equations_l231_231446

-- Defining the problem statement
theorem correct_equations (m n : ℕ) :
  (∀ (m n : ℕ), 40 * m + 10 = 43 * m + 1 ∧ 
   (n - 10) / 40 = (n - 1) / 43) :=
by
  sorry

end correct_equations_l231_231446


namespace min_value_inverse_sum_l231_231196

variable (m n : ℝ)
variable (hm : 0 < m)
variable (hn : 0 < n)
variable (b : ℝ) (hb : b = 2)
variable (hline : 3 * m + n = 1)

theorem min_value_inverse_sum : 
  (1 / m + 4 / n) = 7 + 4 * Real.sqrt 3 :=
  sorry

end min_value_inverse_sum_l231_231196


namespace min_value_expression_l231_231627

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x * y * z) ≥ 216 :=
by
  sorry

end min_value_expression_l231_231627


namespace backup_settings_required_l231_231990

-- Definitions for the given conditions
def weight_of_silverware_piece : ℕ := 4
def pieces_of_silverware_per_setting : ℕ := 3
def weight_of_plate : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def total_weight_ounces : ℕ := 5040

-- Statement to prove
theorem backup_settings_required :
  (total_weight_ounces - 
     (tables * settings_per_table) * 
       (pieces_of_silverware_per_setting * weight_of_silverware_piece + 
        plates_per_setting * weight_of_plate)) /
  (pieces_of_silverware_per_setting * weight_of_silverware_piece + 
   plates_per_setting * weight_of_plate) = 20 := 
by sorry

end backup_settings_required_l231_231990


namespace family_has_11_eggs_l231_231557

def initialEggs : ℕ := 10
def eggsUsed : ℕ := 5
def chickens : ℕ := 2
def eggsPerChicken : ℕ := 3

theorem family_has_11_eggs :
  (initialEggs - eggsUsed) + (chickens * eggsPerChicken) = 11 := by
  sorry

end family_has_11_eggs_l231_231557


namespace sufficient_not_necessary_condition_l231_231368

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h₁ : -1 ≤ x ∧ x ≤ 2) : a > 4 → x^2 - a ≤ 0 :=
by
  sorry

end sufficient_not_necessary_condition_l231_231368


namespace speedster_convertibles_l231_231071

theorem speedster_convertibles 
  (T : ℕ) 
  (h1 : T > 0)
  (h2 : 30 = (2/3 : ℚ) * T)
  (h3 : ∀ n, n = (1/3 : ℚ) * T → ∃ m, m = (4/5 : ℚ) * n) :
  ∃ m, m = 12 := 
sorry

end speedster_convertibles_l231_231071


namespace first_player_wins_l231_231513

theorem first_player_wins :
  ∀ {table : Type} {coin : Type} 
  (can_place : table → coin → Prop) -- function defining if a coin can be placed on the table
  (not_overlap : ∀ (t : table) (c1 c2 : coin), (can_place t c1 ∧ can_place t c2) → c1 ≠ c2) -- coins do not overlap
  (first_move_center : table → coin) -- first player places the coin at the center
  (mirror_move : table → coin → coin), -- function to place a coin symmetrically
  (∃ strategy : (table → Prop) → (coin → Prop),
    (∀ (t : table) (p : table → Prop), p t → strategy p (mirror_move t (first_move_center t))) ∧ 
    (∀ (t : table) (p : table → Prop), strategy p (first_move_center t) → p t)) := sorry

end first_player_wins_l231_231513


namespace ratio_of_seconds_l231_231964

theorem ratio_of_seconds (x : ℕ) :
  (12 : ℕ) / 8 = x / 240 → x = 360 :=
by
  sorry

end ratio_of_seconds_l231_231964


namespace alyssa_games_next_year_l231_231704

/-- Alyssa went to 11 games this year -/
def games_this_year : ℕ := 11

/-- Alyssa went to 13 games last year -/
def games_last_year : ℕ := 13

/-- Alyssa will go to a total of 39 games -/
def total_games : ℕ := 39

/-- Alyssa plans to go to 15 games next year -/
theorem alyssa_games_next_year : 
  games_this_year + games_last_year <= total_games ∧
  total_games - (games_this_year + games_last_year) = 15 := by {
  sorry
}

end alyssa_games_next_year_l231_231704


namespace number_of_grade12_students_selected_l231_231919

def total_students : ℕ := 1500
def grade10_students : ℕ := 550
def grade11_students : ℕ := 450
def total_sample_size : ℕ := 300
def grade12_students : ℕ := total_students - grade10_students - grade11_students

theorem number_of_grade12_students_selected :
    (total_sample_size * grade12_students / total_students) = 100 := by
  sorry

end number_of_grade12_students_selected_l231_231919


namespace find_x_l231_231677

theorem find_x (x : ℝ) (h : 0.65 * x = 0.2 * 617.50) : x = 190 :=
by
  sorry

end find_x_l231_231677


namespace decreasing_function_in_interval_l231_231810

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x + Real.pi / 4)

theorem decreasing_function_in_interval (ω : ℝ) (h_omega_pos : ω > 0) (h_period : Real.pi / 3 < 2 * Real.pi / (2 * ω) ∧ 2 * Real.pi / (2 * ω) < Real.pi / 2)
    (h_symmetry : 2 * ω * 3 * Real.pi / 4 + Real.pi / 4 = (4:ℤ) * Real.pi) :
    ∀ x : ℝ, Real.pi / 6 < x ∧ x < Real.pi / 4 → f ω x < f ω (x + Real.pi / 100) :=
by
    intro x h_interval
    have ω_value : ω = 5 / 2 := sorry
    exact sorry

end decreasing_function_in_interval_l231_231810


namespace number_of_older_females_l231_231303

theorem number_of_older_females (total_population : ℕ) (num_groups : ℕ) (one_group_population : ℕ) :
  total_population = 1000 → num_groups = 5 → total_population = num_groups * one_group_population →
  one_group_population = 200 :=
by
  intro h1 h2 h3
  sorry

end number_of_older_females_l231_231303


namespace rose_bushes_unwatered_l231_231612

theorem rose_bushes_unwatered (n V A : ℕ) (V_set A_set : Finset ℕ) (hV : V = 1003) (hA : A = 1003) (hTotal : n = 2006) (hIntersection : V_set.card = 3) :
  n - (V + A - V_set.card) = 3 :=
by
  sorry

end rose_bushes_unwatered_l231_231612


namespace sin_neg_three_pi_over_four_l231_231696

theorem sin_neg_three_pi_over_four : Real.sin (-3 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_three_pi_over_four_l231_231696


namespace min_value_quadratic_l231_231977

theorem min_value_quadratic (x : ℝ) : 
  ∃ m, m = 3 * x^2 - 18 * x + 2048 ∧ ∀ x, 3 * x^2 - 18 * x + 2048 ≥ 2021 :=
by sorry

end min_value_quadratic_l231_231977


namespace number_of_rows_l231_231425

-- Definitions of the conditions
def total_students : ℕ := 23
def students_in_restroom : ℕ := 2
def students_absent : ℕ := 3 * students_in_restroom - 1
def students_per_desk : ℕ := 6
def fraction_full (r : ℕ) := (2 * r) / 3

-- The statement we need to prove 
theorem number_of_rows : (total_students - students_in_restroom - students_absent) / (students_per_desk * 2 / 3) = 4 :=
by
  sorry

end number_of_rows_l231_231425


namespace percentage_reduction_in_price_l231_231252

variable (R P : ℝ) (R_eq : R = 30) (H : 600 / R - 600 / P = 4)

theorem percentage_reduction_in_price (R_eq : R = 30) (H : 600 / R - 600 / P = 4) :
  ((P - R) / P) * 100 = 20 := sorry

end percentage_reduction_in_price_l231_231252


namespace sum_of_numbers_l231_231381

theorem sum_of_numbers (x y z : ℝ) (h1 : x ≤ y) (h2 : y ≤ z) 
  (h3 : y = 5) (h4 : (x + y + z) / 3 = x + 10) (h5 : (x + y + z) / 3 = z - 15) : 
  x + y + z = 30 := 
by 
  sorry

end sum_of_numbers_l231_231381


namespace fgf_of_3_l231_231282

-- Definitions of the functions f and g
def f (x : ℤ) : ℤ := 4 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

-- The statement we need to prove
theorem fgf_of_3 : f (g (f 3)) = 332 := by
  sorry

end fgf_of_3_l231_231282


namespace polygon_sides_l231_231030

theorem polygon_sides (x : ℕ) 
  (h1 : 180 * (x - 2) = 3 * 360) 
  : x = 8 := 
by
  sorry

end polygon_sides_l231_231030


namespace weight_of_mixture_l231_231293

variable (A B : ℝ)
variable (ratio_A_B : A / B = 9 / 11)
variable (consumed_A : A = 26.1)

theorem weight_of_mixture (A B : ℝ) (ratio_A_B : A / B = 9 / 11) (consumed_A : A = 26.1) : 
  A + B = 58 :=
sorry

end weight_of_mixture_l231_231293


namespace find_Minchos_chocolate_l231_231271

variable (M : ℕ)  -- Define M as a natural number

-- Define the conditions as Lean hypotheses
def TaeminChocolate := 5 * M
def KibumChocolate := 3 * M
def TotalChocolate := TaeminChocolate M + KibumChocolate M

theorem find_Minchos_chocolate (h : TotalChocolate M = 160) : M = 20 :=
by
  sorry

end find_Minchos_chocolate_l231_231271


namespace probability_of_rolling_2_4_6_on_8_sided_die_l231_231319

theorem probability_of_rolling_2_4_6_on_8_sided_die : 
  ∀ (ω : Fin 8), 
  (1 / 8) * (ite (ω = 1 ∨ ω = 3 ∨ ω = 5) 1 0) = 3 / 8 := 
by 
  sorry

end probability_of_rolling_2_4_6_on_8_sided_die_l231_231319


namespace bedrooms_count_l231_231474

/-- Number of bedrooms calculation based on given conditions -/
theorem bedrooms_count (B : ℕ) (h1 : ∀ b, b = 20 * B)
  (h2 : ∀ lr, lr = 20 * B)
  (h3 : ∀ bath, bath = 2 * 20 * B)
  (h4 : ∀ out, out = 2 * (20 * B + 20 * B + 40 * B))
  (h5 : ∀ siblings, siblings = 3)
  (h6 : ∀ work_time, work_time = 4 * 60) : B = 3 :=
by
  -- proof will be provided here
  sorry

end bedrooms_count_l231_231474


namespace find_x_l231_231064

def diamond (a b : ℝ) : ℝ := 3 * a * b - a + b

theorem find_x : ∃ x : ℝ, diamond 3 x = 24 ∧ x = 2.7 :=
by
  sorry

end find_x_l231_231064


namespace range_of_y_is_correct_l231_231581

noncomputable def range_of_y (n : ℝ) : ℝ :=
  if n > 2 then 1 / n else 2 * n^2 + 1

theorem range_of_y_is_correct :
  (∀ n, 0 < range_of_y n ∧ range_of_y n < 1 / 2 ∧ n > 2) ∨ (∀ n, 1 ≤ range_of_y n ∧ n ≤ 2) :=
sorry

end range_of_y_is_correct_l231_231581


namespace inequality_1_inequality_2_inequality_3_inequality_4_l231_231753

-- Definition for the first problem
theorem inequality_1 (x : ℝ) : |2 * x - 1| < 15 ↔ (-7 < x ∧ x < 8) := by
  sorry
  
-- Definition for the second problem
theorem inequality_2 (x : ℝ) : x^2 + 6 * x - 16 < 0 ↔ (-8 < x ∧ x < 2) := by
  sorry

-- Definition for the third problem
theorem inequality_3 (x : ℝ) : |2 * x + 1| > 13 ↔ (x < -7 ∨ x > 6) := by
  sorry

-- Definition for the fourth problem
theorem inequality_4 (x : ℝ) : x^2 - 2 * x > 0 ↔ (x < 0 ∨ x > 2) := by
  sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l231_231753


namespace quadratic_has_two_distinct_real_roots_l231_231747

/-- The quadratic equation x^2 + 2x - 3 = 0 has two distinct real roots. -/
theorem quadratic_has_two_distinct_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ ^ 2 + 2 * x₁ - 3 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 3 = 0) := by
sorry

end quadratic_has_two_distinct_real_roots_l231_231747


namespace factorization_cd_c_l231_231002

theorem factorization_cd_c (C D : ℤ) (h : ∀ y : ℤ, 20*y^2 - 117*y + 72 = (C*y - 8) * (D*y - 9)) : C * D + C = 25 :=
sorry

end factorization_cd_c_l231_231002


namespace no_intersection_points_l231_231125

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := abs (3 * x + 6)
def g (x : ℝ) : ℝ := -abs (4 * x - 3)

-- The main theorem to prove the number of intersection points is zero
theorem no_intersection_points : ∀ x : ℝ, f x ≠ g x := by
  intro x
  sorry -- Proof goes here

end no_intersection_points_l231_231125


namespace opposite_of_neg_five_l231_231966

theorem opposite_of_neg_five : ∃ (y : ℤ), -5 + y = 0 ∧ y = 5 :=
by
  use 5
  simp

end opposite_of_neg_five_l231_231966


namespace find_missing_number_l231_231582

theorem find_missing_number (x : ℝ) (h : 0.25 / x = 2 / 6) : x = 0.75 :=
by
  sorry

end find_missing_number_l231_231582


namespace divisor_of_z_in_form_4n_minus_1_l231_231254

theorem divisor_of_z_in_form_4n_minus_1
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (z : ℕ) 
  (hz : z = 4 * x * y / (x + y)) 
  (hz_odd : z % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ ∃ d : ℕ, d ∣ z ∧ d = 4 * n - 1 :=
sorry

end divisor_of_z_in_form_4n_minus_1_l231_231254


namespace joe_average_test_score_l231_231033

theorem joe_average_test_score 
  (A B C : ℕ) 
  (Hsum : A + B + C = 135) 
  : (A + B + C + 25) / 4 = 40 :=
by
  sorry

end joe_average_test_score_l231_231033


namespace arithmetic_sequence_sum_l231_231166

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 2 + a 12 = 32) : a 3 + a 11 = 32 :=
sorry

end arithmetic_sequence_sum_l231_231166


namespace eq_one_solution_in_interval_l231_231542

theorem eq_one_solution_in_interval (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ (2 * a * x^2 - x - 1 = 0) ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 1 ∧ y ≠ x → (2 * a * y^2 - y - 1 ≠ 0))) → (1 < a) :=
by
  sorry

end eq_one_solution_in_interval_l231_231542


namespace original_weight_l231_231497

variable (W : ℝ) -- Let W be the original weight of the side of beef

-- Conditions
def condition1 : ℝ := 0.80 * W -- Weight after first stage
def condition2 : ℝ := 0.70 * condition1 W -- Weight after second stage
def condition3 : ℝ := 0.75 * condition2 W -- Weight after third stage

-- Final weight is given as 570 pounds
theorem original_weight (h : condition3 W = 570) : W = 1357.14 :=
by 
  sorry

end original_weight_l231_231497


namespace fencing_cost_l231_231086

noncomputable def diameter : ℝ := 14
noncomputable def cost_per_meter : ℝ := 2.50
noncomputable def pi := Real.pi

noncomputable def circumference (d : ℝ) : ℝ := pi * d

noncomputable def total_cost (c : ℝ) (r : ℝ) : ℝ := r * c

theorem fencing_cost : total_cost (circumference diameter) cost_per_meter = 109.95 := by
  sorry

end fencing_cost_l231_231086


namespace ab5_a2_c5_a2_inequality_l231_231214

theorem ab5_a2_c5_a2_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ 5 - a ^ 2 + 3) * (b ^ 5 - b ^ 2 + 3) * (c ^ 5 - c ^ 2 + 3) ≥ (a + b + c) ^ 3 := 
by
  sorry

end ab5_a2_c5_a2_inequality_l231_231214


namespace Shekar_science_marks_l231_231555

theorem Shekar_science_marks (S : ℕ) : 
  let math_marks := 76
  let social_studies_marks := 82
  let english_marks := 67
  let biology_marks := 75
  let average_marks := 73
  let num_subjects := 5
  ((math_marks + S + social_studies_marks + english_marks + biology_marks) / num_subjects = average_marks) → S = 65 :=
by
  sorry

end Shekar_science_marks_l231_231555


namespace surface_area_small_prism_l231_231012

-- Definitions and conditions
variables (a b c : ℝ)

def small_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * a * b + 2 * a * c + 2 * b * c

def large_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * (3 * b) * (3 * b) + 2 * (3 * b) * (4 * c) + 2 * (4 * c) * (3 * b)

-- Conditions
def conditions : Prop :=
  (3 * b = 2 * a) ∧ (a = 3 * c) ∧ (large_cuboid_surface_area a b c = 360)

-- Desired result
def result : Prop :=
  small_cuboid_surface_area a b c = 88

-- The theorem
theorem surface_area_small_prism (a b c : ℝ) (h : conditions a b c) : result a b c :=
by
  sorry

end surface_area_small_prism_l231_231012


namespace sequence_formulas_range_of_k_l231_231905

variable {a b : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {k : ℝ}

-- (1) Prove the general formulas for {a_n} and {b_n}
theorem sequence_formulas (h1 : ∀ n, a n + b n = 2 * n - 1)
  (h2 : ∀ n, S n = 2 * n^2 - n)
  (hS : ∀ n, a (n + 1) = S (n + 1) - S n)
  (hS1 : a 1 = S 1) :
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, b n = -2 * n + 2) :=
sorry

-- (2) Prove the range of k
theorem range_of_k (h3 : ∀ n, a n = k * 2^(n - 1))
  (h4 : ∀ n, b n = 2 * n - 1 - k * 2^(n - 1))
  (h5 : ∀ n, b (n + 1) < b n) :
  k > 2 :=
sorry

end sequence_formulas_range_of_k_l231_231905


namespace proportion_of_bike_riders_is_correct_l231_231335

-- Define the given conditions as constants
def total_students : ℕ := 92
def bus_riders : ℕ := 20
def walkers : ℕ := 27

-- Define the remaining students after bus riders and after walkers
def remaining_after_bus_riders : ℕ := total_students - bus_riders
def bike_riders : ℕ := remaining_after_bus_riders - walkers

-- Define the expected proportion
def expected_proportion : ℚ := 45 / 72

-- State the theorem to be proved
theorem proportion_of_bike_riders_is_correct :
  (↑bike_riders / ↑remaining_after_bus_riders : ℚ) = expected_proportion := 
by
  sorry

end proportion_of_bike_riders_is_correct_l231_231335


namespace angle_sum_eq_pi_div_2_l231_231206

open Real

theorem angle_sum_eq_pi_div_2 (θ1 θ2 : ℝ) (h1 : 0 < θ1 ∧ θ1 < π / 2) (h2 : 0 < θ2 ∧ θ2 < π / 2)
  (h : (sin θ1)^2020 / (cos θ2)^2018 + (cos θ1)^2020 / (sin θ2)^2018 = 1) :
  θ1 + θ2 = π / 2 :=
sorry

end angle_sum_eq_pi_div_2_l231_231206


namespace three_digit_powers_of_two_count_l231_231192

theorem three_digit_powers_of_two_count : 
  ∃ n_count : ℕ, (∀ n : ℕ, (100 ≤ 2^n ∧ 2^n < 1000) ↔ (n = 7 ∨ n = 8 ∨ n = 9)) ∧ n_count = 3 :=
by
  sorry

end three_digit_powers_of_two_count_l231_231192


namespace find_prime_p_l231_231163

theorem find_prime_p
  (p : ℕ)
  (h_prime_p : Nat.Prime p)
  (h : Nat.Prime (p^3 + p^2 + 11 * p + 2)) :
  p = 3 :=
sorry

end find_prime_p_l231_231163


namespace playground_length_l231_231496

theorem playground_length
  (L_g : ℝ) -- length of the garden
  (L_p : ℝ) -- length of the playground
  (width_garden : ℝ := 24) -- width of the garden
  (width_playground : ℝ := 12) -- width of the playground
  (perimeter_garden : ℝ := 64) -- perimeter of the garden
  (area_garden : ℝ := L_g * 24) -- area of the garden
  (area_playground : ℝ := L_p * 12) -- area of the playground
  (areas_equal : area_garden = area_playground) -- equal areas
  (perimeter_condition : 2 * (L_g + 24) = 64) -- perimeter condition
  : L_p = 16 := 
by
  sorry

end playground_length_l231_231496


namespace hyperbola_asymptote_slopes_l231_231771

theorem hyperbola_asymptote_slopes:
  (∀ (x y : ℝ), (x^2 / 144 - y^2 / 81 = 1) → (y = (3 / 4) * x ∨ y = -(3 / 4) * x)) :=
by
  sorry

end hyperbola_asymptote_slopes_l231_231771


namespace find_x_l231_231478

theorem find_x : 2^4 + 3 = 5^2 - 6 :=
by
  sorry

end find_x_l231_231478


namespace acute_angle_sine_diff_l231_231711

theorem acute_angle_sine_diff (α β : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : 0 < β ∧ β < π / 2)
  (h₂ : Real.sin α = (Real.sqrt 5) / 5) (h₃ : Real.sin (α - β) = -(Real.sqrt 10) / 10) : β = π / 4 :=
sorry

end acute_angle_sine_diff_l231_231711


namespace production_rate_is_constant_l231_231069

def drum_rate := 6 -- drums per day

def days_needed_to_produce (n : ℕ) : ℕ := n / drum_rate

theorem production_rate_is_constant (n : ℕ) : days_needed_to_produce n = n / drum_rate :=
by
  sorry

end production_rate_is_constant_l231_231069


namespace log_equation_solution_l231_231589

theorem log_equation_solution (x : ℝ) (hx_pos : 0 < x) : 
  (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ↔ x ≠ 1 :=
by
  sorry

end log_equation_solution_l231_231589


namespace number_of_paths_from_A_to_D_l231_231285

-- Definitions based on conditions
def paths_A_to_B : ℕ := 2
def paths_B_to_C : ℕ := 2
def paths_A_to_C : ℕ := 1
def paths_C_to_D : ℕ := 2
def paths_B_to_D : ℕ := 2

-- Theorem statement
theorem number_of_paths_from_A_to_D : 
  paths_A_to_B * paths_B_to_C * paths_C_to_D + 
  paths_A_to_C * paths_C_to_D + 
  paths_A_to_B * paths_B_to_D = 14 :=
by {
  -- proof steps will go here
  sorry
}

end number_of_paths_from_A_to_D_l231_231285


namespace ball_returns_velocity_required_initial_velocity_to_stop_l231_231880

-- Define the conditions.
def distance_A_to_wall : ℝ := 5
def distance_wall_to_B : ℝ := 2
def distance_AB : ℝ := 9
def initial_velocity_v0 : ℝ := 5
def acceleration_a : ℝ := -0.4

-- Hypothesize that the velocity when the ball returns to A is 3 m/s.
theorem ball_returns_velocity (t : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  initial_velocity_v0 * t + (1 / 2) * acceleration_a * t^2 = distance_AB + distance_A_to_wall →
  initial_velocity_v0 + acceleration_a * t = 3 := sorry

-- Hypothesize that to stop exactly at A, the initial speed should be 4 m/s.
theorem required_initial_velocity_to_stop (t' : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  (0.4 * t') * t' + (1 / 2) * acceleration_a * t'^2 = distance_AB + distance_A_to_wall →
  0.4 * t' = 4 := sorry

end ball_returns_velocity_required_initial_velocity_to_stop_l231_231880


namespace measure_of_theta_l231_231535

theorem measure_of_theta 
  (ACB FEG DCE DEC : ℝ)
  (h1 : ACB = 10)
  (h2 : FEG = 26)
  (h3 : DCE = 14)
  (h4 : DEC = 33) : θ = 11 :=
by
  sorry

end measure_of_theta_l231_231535


namespace complex_quadrant_l231_231707

open Complex

theorem complex_quadrant :
  let z := (1 - I) * (3 + I)
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l231_231707


namespace workers_together_complete_work_in_14_days_l231_231766

noncomputable def efficiency (Wq : ℝ) := 1.4 * Wq

def work_done_in_one_day_p (Wp : ℝ) := Wp = 1 / 24

noncomputable def work_done_in_one_day_q (Wq : ℝ) := Wq = (1 / 24) / 1.4

noncomputable def combined_work_per_day (Wp Wq : ℝ) := Wp + Wq

noncomputable def days_to_complete_work (W : ℝ) := 1 / W

theorem workers_together_complete_work_in_14_days (Wp Wq : ℝ) 
  (h1 : Wp = efficiency Wq)
  (h2 : work_done_in_one_day_p Wp)
  (h3 : work_done_in_one_day_q Wq) :
  days_to_complete_work (combined_work_per_day Wp Wq) = 14 := 
sorry

end workers_together_complete_work_in_14_days_l231_231766


namespace intersection_solution_l231_231563

-- Define lines
def line1 (x : ℝ) : ℝ := -x + 4
def line2 (x : ℝ) (m : ℝ) : ℝ := 2 * x + m

-- Define system of equations
def system1 (x y : ℝ) : Prop := x + y = 4
def system2 (x y m : ℝ) : Prop := 2 * x - y + m = 0

-- Proof statement
theorem intersection_solution (m : ℝ) (n : ℝ) :
  (system1 3 n) ∧ (system2 3 n m) ∧ (line1 3 = n) ∧ (line2 3 m = n) →
  (3, n) = (3, 1) :=
  by 
  -- The proof would go here
  sorry

end intersection_solution_l231_231563


namespace ball_center_distance_traveled_l231_231765

theorem ball_center_distance_traveled (d : ℝ) (r1 r2 r3 r4 : ℝ) (R1 R2 R3 R4 : ℝ) :
  d = 6 → 
  R1 = 120 → 
  R2 = 50 → 
  R3 = 90 → 
  R4 = 70 → 
  r1 = R1 - 3 → 
  r2 = R2 + 3 → 
  r3 = R3 - 3 → 
  r4 = R4 + 3 → 
  (1/2) * 2 * π * r1 + (1/2) * 2 * π * r2 + (1/2) * 2 * π * r3 + (1/2) * 2 * π * r4 = 330 * π :=
by
  sorry

end ball_center_distance_traveled_l231_231765


namespace min_scalar_product_l231_231796

open Real

variable {a b : ℝ → ℝ}

-- Definitions used as conditions in the problem
def condition (a b : ℝ → ℝ) : Prop :=
  |2 * a - b| ≤ 3

-- The goal to prove based on the conditions and the correct answer
theorem min_scalar_product (h : condition a b) : 
  (a x) * (b x) ≥ -9 / 8 :=
sorry

end min_scalar_product_l231_231796


namespace perimeter_of_rectangular_garden_l231_231946

theorem perimeter_of_rectangular_garden (L W : ℝ) (h : L + W = 28) : 2 * (L + W) = 56 :=
by sorry

end perimeter_of_rectangular_garden_l231_231946


namespace hyperbola_equation_l231_231238

-- Define the hyperbola with vertices and other conditions
def Hyperbola (a b : ℝ) (h : a > 0 ∧ b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1)

-- Given conditions and the proof goal
theorem hyperbola_equation
  (a b : ℝ) (h : a > 0 ∧ b > 0)
  (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (k_PA k_PB : ℝ)
  (PA_PB_condition : k_PA * k_PB = 3)
  (MN_min_value : |(M.1 - N.1) + (M.2 - N.2)| = 4) :
  Hyperbola a b h →
  (a = 2 ∧ b = 2 * Real.sqrt 3 ∧ (∀ (x y : ℝ), (x^2 / 4 - y^2 / 12 = 1)) ∨ 
   a = 2 / 3 ∧ b = 2 * Real.sqrt 3 / 3 ∧ (∀ (x y : ℝ), (9 * x^2 / 4 - 3 * y^2 / 4 = 1)))
:=
sorry

end hyperbola_equation_l231_231238


namespace pizza_slices_with_both_toppings_l231_231369

theorem pizza_slices_with_both_toppings (total_slices ham_slices pineapple_slices slices_with_both : ℕ)
  (h_total: total_slices = 15)
  (h_ham: ham_slices = 8)
  (h_pineapple: pineapple_slices = 12)
  (h_slices_with_both: slices_with_both + (ham_slices - slices_with_both) + (pineapple_slices - slices_with_both) = total_slices)
  : slices_with_both = 5 :=
by
  -- the proof would go here, but we use sorry to skip it
  sorry

end pizza_slices_with_both_toppings_l231_231369


namespace student_marks_l231_231235

variable (max_marks : ℕ) (pass_percent : ℕ) (fail_by : ℕ)

theorem student_marks
  (h_max : max_marks = 400)
  (h_pass : pass_percent = 35)
  (h_fail : fail_by = 40)
  : max_marks * pass_percent / 100 - fail_by = 100 :=
by
  sorry

end student_marks_l231_231235


namespace problem_statement_l231_231867

theorem problem_statement (x y z : ℤ) (h1 : x = z - 2) (h2 : y = x + 1) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := 
by
  sorry

end problem_statement_l231_231867


namespace solve_for_x_l231_231142

theorem solve_for_x (x : ℝ) (h : |2000 * x + 2000| = 20 * 2000) : x = 19 ∨ x = -21 := 
by
  sorry

end solve_for_x_l231_231142


namespace equation_solution_l231_231171

theorem equation_solution (x : ℝ) :
  (1 / x + 1 / (x + 2) - 1 / (x + 4) - 1 / (x + 6) + 1 / (x + 8) = 0) →
  (x = -4 - 2 * Real.sqrt 3) ∨ (x = 2 - 2 * Real.sqrt 3) := by
  sorry

end equation_solution_l231_231171


namespace tank_fraction_after_adding_water_l231_231522

noncomputable def fraction_of_tank_full 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  : ℚ :=
(initial_fraction * total_capacity + additional_water) / total_capacity

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  (h_initial : initial_fraction = 3 / 4) 
  (h_addition : additional_water = 4) 
  (h_capacity : total_capacity = 32) 
: fraction_of_tank_full initial_fraction additional_water total_capacity = 7 / 8 :=
by
  sorry

end tank_fraction_after_adding_water_l231_231522


namespace fresh_grapes_water_percentage_l231_231135

/--
Given:
- Fresh grapes contain a certain percentage (P%) of water by weight.
- Dried grapes contain 25% water by weight.
- The weight of dry grapes obtained from 200 kg of fresh grapes is 66.67 kg.

Prove:
- The percentage of water (P) in fresh grapes is 75%.
-/
theorem fresh_grapes_water_percentage
  (P : ℝ) (H1 : ∃ P, P / 100 * 200 = 0.75 * 66.67) :
  P = 75 :=
sorry

end fresh_grapes_water_percentage_l231_231135


namespace smallest_possible_other_integer_l231_231853

theorem smallest_possible_other_integer (x m n : ℕ) (h1 : x > 0) (h2 : m = 70) 
  (h3 : gcd m n = x + 7) (h4 : lcm m n = x * (x + 7)) : n = 20 :=
sorry

end smallest_possible_other_integer_l231_231853


namespace angle_of_squares_attached_l231_231770

-- Definition of the problem scenario:
-- Three squares attached as described, needing to prove x = 39 degrees.

open Real

theorem angle_of_squares_attached (x : ℝ) (h : 
  let angle1 := 30
  let angle2 := 126
  let angle3 := 75
  angle1 + angle2 + angle3 + x = 3 * 90) :
  x = 39 :=
by 
  -- This proof is omitted
  sorry

end angle_of_squares_attached_l231_231770


namespace integer_solutions_of_linear_diophantine_eq_l231_231130

theorem integer_solutions_of_linear_diophantine_eq 
  (a b c : ℤ)
  (h_gcd : Int.gcd a b = 1)
  (x₀ y₀ : ℤ)
  (h_particular_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, (a * x + b * y = c) → ∃ (k : ℤ), (x = x₀ + k * b) ∧ (y = y₀ - k * a) := 
by
  sorry

end integer_solutions_of_linear_diophantine_eq_l231_231130


namespace total_percent_decrease_is_19_l231_231306

noncomputable def original_value : ℝ := 100
noncomputable def first_year_decrease : ℝ := 0.10
noncomputable def second_year_decrease : ℝ := 0.10
noncomputable def value_after_first_year : ℝ := original_value * (1 - first_year_decrease)
noncomputable def value_after_second_year : ℝ := value_after_first_year * (1 - second_year_decrease)
noncomputable def total_decrease_in_dollars : ℝ := original_value - value_after_second_year
noncomputable def total_percent_decrease : ℝ := (total_decrease_in_dollars / original_value) * 100

theorem total_percent_decrease_is_19 :
  total_percent_decrease = 19 := by
  sorry

end total_percent_decrease_is_19_l231_231306


namespace constant_term_is_24_l231_231831

noncomputable def constant_term_of_binomial_expansion 
  (a : ℝ) (hx : π * a^2 = 4 * π) : ℝ :=
  if ha : a = 2 then 24 else 0

theorem constant_term_is_24
  (a : ℝ) (hx : π * a^2 = 4 * π) :
  constant_term_of_binomial_expansion a hx = 24 :=
by
  sorry

end constant_term_is_24_l231_231831


namespace alice_net_amount_spent_l231_231643

noncomputable def net_amount_spent : ℝ :=
  let price_per_pint := 4
  let sunday_pints := 4
  let sunday_cost := sunday_pints * price_per_pint

  let monday_discount := 0.1
  let monday_pints := 3 * sunday_pints
  let monday_price_per_pint := price_per_pint * (1 - monday_discount)
  let monday_cost := monday_pints * monday_price_per_pint

  let tuesday_discount := 0.2
  let tuesday_pints := monday_pints / 3
  let tuesday_price_per_pint := price_per_pint * (1 - tuesday_discount)
  let tuesday_cost := tuesday_pints * tuesday_price_per_pint

  let wednesday_returned_pints := tuesday_pints / 2
  let wednesday_refund := wednesday_returned_pints * tuesday_price_per_pint

  sunday_cost + monday_cost + tuesday_cost - wednesday_refund

theorem alice_net_amount_spent : net_amount_spent = 65.60 := by
  sorry

end alice_net_amount_spent_l231_231643


namespace find_y_l231_231234

theorem find_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hr : x % y = 9) (hxy : (x : ℝ) / y = 96.45) : y = 20 :=
by
  sorry

end find_y_l231_231234


namespace license_plate_increase_l231_231691

def old_license_plates : ℕ := 26 * (10^5)

def new_license_plates : ℕ := 26^2 * (10^4)

theorem license_plate_increase :
  (new_license_plates / old_license_plates : ℝ) = 2.6 := by
  sorry

end license_plate_increase_l231_231691


namespace total_time_equiv_l231_231026

-- Define the number of chairs
def chairs := 7

-- Define the number of tables
def tables := 3

-- Define the time spent on each piece of furniture in minutes
def time_per_piece := 4

-- Prove the total time taken to assemble all furniture
theorem total_time_equiv : chairs + tables = 10 ∧ 4 * 10 = 40 := by
  sorry

end total_time_equiv_l231_231026


namespace Barry_reach_l231_231486

noncomputable def Larry_full_height : ℝ := 5
noncomputable def Larry_shoulder_height : ℝ := Larry_full_height - 0.2 * Larry_full_height
noncomputable def combined_reach : ℝ := 9

theorem Barry_reach :
  combined_reach - Larry_shoulder_height = 5 := 
by
  -- Correct answer verification comparing combined reach minus Larry's shoulder height equals 5
  sorry

end Barry_reach_l231_231486


namespace solve_system_of_equations_l231_231858

theorem solve_system_of_equations (a b c x y z : ℝ):
  (x - a * y + a^2 * z = a^3) →
  (x - b * y + b^2 * z = b^3) →
  (x - c * y + c^2 * z = c^3) →
  x = a * b * c ∧ y = a * b + a * c + b * c ∧ z = a + b + c :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end solve_system_of_equations_l231_231858


namespace domain_width_p_l231_231921

variable (f : ℝ → ℝ)
variable (h_dom_f : ∀ x, -12 ≤ x ∧ x ≤ 12 → f x = f x)

noncomputable def p (x : ℝ) : ℝ := f (x / 3)

theorem domain_width_p : (width : ℝ) = 72 :=
by
  let domain_p : Set ℝ := {x | -36 ≤ x ∧ x ≤ 36}
  have : width = 72 := sorry
  exact this

end domain_width_p_l231_231921


namespace bus_system_carry_per_day_l231_231617

theorem bus_system_carry_per_day (total_people : ℕ) (weeks : ℕ) (days_in_week : ℕ) (people_per_day : ℕ) :
  total_people = 109200000 →
  weeks = 13 →
  days_in_week = 7 →
  people_per_day = total_people / (weeks * days_in_week) →
  people_per_day = 1200000 :=
by
  intros htotal hweeks hdays hcalc
  sorry

end bus_system_carry_per_day_l231_231617


namespace find_x_from_average_l231_231310

theorem find_x_from_average :
  let sum_series := 5151
  let n := 102
  let known_average := 50 * (x + 1)
  (sum_series + x) / n = known_average → 
  x = 51 / 5099 :=
by
  intros
  sorry

end find_x_from_average_l231_231310


namespace joanna_estimate_is_larger_l231_231842

theorem joanna_estimate_is_larger 
  (u v ε₁ ε₂ : ℝ) 
  (huv : u > v) 
  (hv0 : v > 0) 
  (hε₁ : ε₁ > 0) 
  (hε₂ : ε₂ > 0) : 
  (u + ε₁) - (v - ε₂) > u - v := 
sorry

end joanna_estimate_is_larger_l231_231842


namespace tan_alpha_sin_double_angle_l231_231719

theorem tan_alpha_sin_double_angle (α : ℝ) (h : Real.tan α = 3/4) : Real.sin (2 * α) = 24/25 :=
by
  sorry

end tan_alpha_sin_double_angle_l231_231719


namespace solve_for_b_l231_231927

theorem solve_for_b (x y b : ℝ) (h1: 4 * x + y = b) (h2: 3 * x + 4 * y = 3 * b) (hx: x = 3) : b = 39 :=
sorry

end solve_for_b_l231_231927


namespace greatest_prime_factor_of_n_l231_231427

noncomputable def n : ℕ := 4^17 - 2^29

theorem greatest_prime_factor_of_n :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ n → q ≤ p :=
sorry

end greatest_prime_factor_of_n_l231_231427


namespace point_in_third_quadrant_l231_231552

theorem point_in_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : -m^2 < 0 ∧ -n < 0 :=
by
  sorry

end point_in_third_quadrant_l231_231552


namespace product_closest_value_l231_231939

-- Define the constants used in the problem
def a : ℝ := 2.5
def b : ℝ := 53.6
def c : ℝ := 0.4

-- Define the expression and the expected correct answer
def expression : ℝ := a * (b - c)
def correct_answer : ℝ := 133

-- State the theorem that the expression evaluates to the correct answer
theorem product_closest_value : expression = correct_answer :=
by
  sorry

end product_closest_value_l231_231939


namespace hyperbola_center_l231_231896

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x ^ 2 + 54 * x - 16 * y ^ 2 - 128 * y - 200 = 0) : 
  (x = -3) ∧ (y = -4) := 
sorry

end hyperbola_center_l231_231896


namespace min_photographs_42_tourists_3_monuments_l231_231366

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l231_231366


namespace garden_area_increase_l231_231992

-- Definitions corresponding to the conditions
def length := 40
def width := 20
def original_perimeter := 2 * (length + width)

-- Definition of the correct answer calculation
def original_area := length * width
def side_length := original_perimeter / 4
def new_area := side_length * side_length
def area_increase := new_area - original_area

-- The statement to be proven
theorem garden_area_increase : area_increase = 100 :=
by sorry

end garden_area_increase_l231_231992


namespace find_f1_l231_231440

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = x * f (x)

theorem find_f1 (f : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : functional_equation f) : 
  f 1 = 0 :=
sorry

end find_f1_l231_231440


namespace max_correct_answers_l231_231089

theorem max_correct_answers :
  ∃ (c w b : ℕ), c + w + b = 25 ∧ 4 * c - 3 * w = 57 ∧ c = 18 :=
by {
  sorry
}

end max_correct_answers_l231_231089


namespace eleven_pow_2010_mod_19_l231_231645

theorem eleven_pow_2010_mod_19 : (11 ^ 2010) % 19 = 3 := sorry

end eleven_pow_2010_mod_19_l231_231645


namespace smallest_number_starts_with_four_and_decreases_four_times_l231_231439

theorem smallest_number_starts_with_four_and_decreases_four_times :
  ∃ (X : ℕ), ∃ (A n : ℕ), (X = 4 * 10^n + A ∧ X = 4 * (10 * A + 4)) ∧ X = 410256 := 
by
  sorry

end smallest_number_starts_with_four_and_decreases_four_times_l231_231439


namespace perimeter_large_star_l231_231274

theorem perimeter_large_star (n m : ℕ) (P : ℕ)
  (triangle_perimeter : ℕ) (quad_perimeter : ℕ) (small_star_perimeter : ℕ)
  (hn : n = 5) (hm : m = 5)
  (h_triangle_perimeter : triangle_perimeter = 7)
  (h_quad_perimeter : quad_perimeter = 18)
  (h_small_star_perimeter : small_star_perimeter = 3) :
  m * quad_perimeter + small_star_perimeter = n * triangle_perimeter + P → P = 58 :=
by 
  -- Placeholder proof
  sorry

end perimeter_large_star_l231_231274


namespace salary_C_more_than_A_ratio_salary_E_to_A_and_B_l231_231856

variable (x : ℝ)
variables (salary_A salary_B salary_C salary_D salary_E combined_salary_BCD : ℝ)

-- Conditions
def conditions : Prop :=
  salary_B = 2 * salary_A ∧
  salary_C = 3 * salary_A ∧
  salary_D = 4 * salary_A ∧
  salary_E = 5 * salary_A ∧
  combined_salary_BCD = 15000 ∧
  combined_salary_BCD = salary_B + salary_C + salary_D

-- Statements to prove
theorem salary_C_more_than_A
  (cond : conditions salary_A salary_B salary_C salary_D salary_E combined_salary_BCD) :
  (salary_C - salary_A) / salary_A * 100 = 200 := by
  sorry

theorem ratio_salary_E_to_A_and_B
  (cond : conditions salary_A salary_B salary_C salary_D salary_E combined_salary_BCD) :
  salary_E / (salary_A + salary_B) = 5 / 3 := by
  sorry

end salary_C_more_than_A_ratio_salary_E_to_A_and_B_l231_231856


namespace sum_of_roots_l231_231751

theorem sum_of_roots (a β : ℝ) 
  (h1 : a^2 - 2 * a = 1) 
  (h2 : β^2 - 2 * β - 1 = 0) 
  (hne : a ≠ β) 
  : a + β = 2 := 
sorry

end sum_of_roots_l231_231751


namespace waxberry_problem_l231_231216

noncomputable def batch_cannot_be_sold : ℚ := 1 - (8 / 9 * 9 / 10)

def probability_distribution (X : ℚ) : ℚ := 
  if X = -3200 then (1 / 5)^4 else
  if X = -2000 then 4 * (1 / 5)^3 * (4 / 5) else
  if X = -800 then 6 * (1 / 5)^2 * (4 / 5)^2 else
  if X = 400 then 4 * (1 / 5) * (4 / 5)^3 else
  if X = 1600 then (4 / 5)^4 else 0

noncomputable def expected_profit : ℚ :=
  -3200 * probability_distribution (-3200) +
  -2000 * probability_distribution (-2000) +
  -800 * probability_distribution (-800) +
  400 * probability_distribution (400) +
  1600 * probability_distribution (1600)

theorem waxberry_problem : 
  batch_cannot_be_sold = 1 / 5 ∧ 
  (probability_distribution (-3200) = 1 / 625 ∧ 
   probability_distribution (-2000) = 16 / 625 ∧ 
   probability_distribution (-800) = 96 / 625 ∧ 
   probability_distribution (400) = 256 / 625 ∧ 
   probability_distribution (1600) = 256 / 625) ∧ 
  expected_profit = 640 :=
by 
  sorry

end waxberry_problem_l231_231216


namespace domain_of_f_l231_231501

def domain_f (x : ℝ) : Prop :=
  2 * x - 3 ≥ 0 ∧ x ≠ 3

def domain_set : Set ℝ :=
  { x | (3 / 2) ≤ x ∧ x < 3 ∨ 3 < x }

theorem domain_of_f :
  { x : ℝ | domain_f x } = domain_set := by
  sorry

end domain_of_f_l231_231501


namespace quadratic_non_real_roots_iff_l231_231930

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l231_231930


namespace johns_total_distance_l231_231390

theorem johns_total_distance :
  let monday := 1700
  let tuesday := monday + 200
  let wednesday := 0.7 * tuesday
  let thursday := 2 * wednesday
  let friday := 3.5 * 1000
  let saturday := 0
  monday + tuesday + wednesday + thursday + friday + saturday = 10090 := 
by
  sorry

end johns_total_distance_l231_231390


namespace inequality_sqrt_sum_ge_one_l231_231313

variable (a b c : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variable (prod_abc : a * b * c = 1)

theorem inequality_sqrt_sum_ge_one :
  (Real.sqrt (a / (8 + a)) + Real.sqrt (b / (8 + b)) + Real.sqrt (c / (8 + c)) ≥ 1) :=
by
  sorry

end inequality_sqrt_sum_ge_one_l231_231313


namespace discount_of_bag_l231_231258

def discounted_price (marked_price discount_rate : ℕ) : ℕ :=
  marked_price - ((discount_rate * marked_price) / 100)

theorem discount_of_bag : discounted_price 200 40 = 120 :=
by
  unfold discounted_price
  norm_num

end discount_of_bag_l231_231258


namespace probability_non_obtuse_l231_231199

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l231_231199


namespace ak_divisibility_l231_231914

theorem ak_divisibility {a k m n : ℕ} (h : a ^ k % (m ^ n) = 0) : a ^ (k * m) % (m ^ (n + 1)) = 0 :=
sorry

end ak_divisibility_l231_231914


namespace geometric_segment_l231_231253

theorem geometric_segment (AB A'B' : ℝ) (P D A B P' D' A' B' : ℝ) (x y a : ℝ) :
  AB = 3 ∧ A'B' = 6 ∧ (∀ P, dist P D = x) ∧ (∀ P', dist P' D' = 2 * x) ∧ x = a → x + y = 3 * a :=
by
  sorry

end geometric_segment_l231_231253


namespace minimum_apples_l231_231758

theorem minimum_apples (x : ℕ) : 
  (x ≡ 10 [MOD 3]) ∧ (x ≡ 11 [MOD 4]) ∧ (x ≡ 12 [MOD 5]) → x = 67 :=
sorry

end minimum_apples_l231_231758


namespace max_n_m_sum_l231_231323

-- Definition of the function f
def f (x : ℝ) : ℝ := -x^2 + 4 * x

-- Statement of the problem
theorem max_n_m_sum {m n : ℝ} (h : n > m) (h_range : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4) : n + m = 7 :=
sorry

end max_n_m_sum_l231_231323


namespace gcd_of_three_numbers_l231_231061

theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd 279 372) 465 = 93 := 
by 
  sorry

end gcd_of_three_numbers_l231_231061


namespace cubic_polynomial_greater_than_zero_l231_231370

theorem cubic_polynomial_greater_than_zero (x : ℝ) : x^3 + 3*x^2 + x - 5 > 0 → x > 1 :=
sorry

end cubic_polynomial_greater_than_zero_l231_231370


namespace sum_mod_17_eq_0_l231_231825

theorem sum_mod_17_eq_0 :
  (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 0 :=
by
  sorry

end sum_mod_17_eq_0_l231_231825


namespace tangent_line_of_circle_l231_231380
-- Import the required libraries

-- Define the given condition of the circle in polar coordinates
def polar_circle (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

-- Define the property of the tangent line in polar coordinates
def tangent_line (rho theta : ℝ) : Prop :=
  rho * Real.cos theta = 4

-- State the theorem to be proven
theorem tangent_line_of_circle (rho theta : ℝ) (h : polar_circle rho theta) :
  tangent_line rho theta :=
sorry

end tangent_line_of_circle_l231_231380


namespace rectangle_perimeter_l231_231618

theorem rectangle_perimeter (a b c d e f g : ℕ)
  (h1 : a + b + c = d)
  (h2 : d + e = g)
  (h3 : b + c = f)
  (h4 : c + f = g)
  (h5 : Nat.gcd (a + b + g) (d + e) = 1)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (e_pos : 0 < e)
  (f_pos : 0 < f)
  (g_pos : 0 < g) :
  2 * (a + b + g + d + e) = 40 :=
sorry

end rectangle_perimeter_l231_231618


namespace min_value_of_2a7_a11_l231_231900

noncomputable def a (n : ℕ) : ℝ := sorry -- placeholder for the sequence terms

-- Conditions
axiom geometric_sequence (n m : ℕ) (r : ℝ) (h : ∀ k, a k > 0) : a n = a 0 * r^n
axiom geometric_mean_condition : a 4 * a 14 = 8

-- Theorem to Prove
theorem min_value_of_2a7_a11 : ∀ n : ℕ, (∀ k, a k > 0) → 2 * a 7 + a 11 ≥ 8 :=
by
  intros
  sorry

end min_value_of_2a7_a11_l231_231900


namespace min_value_of_reciprocals_l231_231887

theorem min_value_of_reciprocals (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 2) : 
  ∃ (x : ℝ), x = 2 * a + b ∧ ∃ (y : ℝ), y = 2 * b + c ∧ ∃ (z : ℝ), z = 2 * c + a ∧ (1 / x + 1 / y + 1 / z = 27 / 8) :=
sorry

end min_value_of_reciprocals_l231_231887


namespace trisha_total_distance_l231_231466

theorem trisha_total_distance :
  let distance1 := 0.11
  let distance2 := 0.11
  let distance3 := 0.67
  distance1 + distance2 + distance3 = 0.89 :=
by
  sorry

end trisha_total_distance_l231_231466


namespace quadratic_rewrite_l231_231941

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 4) (h2 : 2 * d * e = 20) (h3 : e^2 + f = -24) :
  d * e = 10 :=
sorry

end quadratic_rewrite_l231_231941


namespace sought_line_eq_l231_231750

-- Definitions used in the conditions
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def line_perpendicular (x y : ℝ) : Prop := x + y = 0
def center_of_circle : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem sought_line_eq (x y : ℝ) :
  (circle_eq x y ∧ line_perpendicular x y ∧ (x, y) = center_of_circle) →
  (x + y + 1 = 0) :=
by
  sorry

end sought_line_eq_l231_231750


namespace inequality_abc_l231_231874

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 :=
by
  sorry

end inequality_abc_l231_231874


namespace find_ordered_triple_l231_231583

theorem find_ordered_triple :
  ∃ (a b c : ℝ), a > 2 ∧ b > 2 ∧ c > 2 ∧
    (a + b + c = 30) ∧
    ( (a = 13) ∧ (b = 11) ∧ (c = 6) ) ∧
    ( ( ( (a + 3)^2 / (b + c - 3) ) + ( (b + 5)^2 / (c + a - 5) ) + ( (c + 7)^2 / (a + b - 7) ) = 45 ) ) :=
sorry

end find_ordered_triple_l231_231583


namespace find_F_l231_231068

theorem find_F (C : ℝ) (F : ℝ) (h₁ : C = 35) (h₂ : C = 4 / 7 * (F - 40)) : F = 101.25 := by
  sorry

end find_F_l231_231068


namespace monitor_height_l231_231690

theorem monitor_height (width circumference : ℕ) (h_width : width = 12) (h_circumference : circumference = 38) :
  2 * (width + 7) = circumference :=
by
  sorry

end monitor_height_l231_231690


namespace max_sum_of_multiplication_table_l231_231015

-- Define primes and their sums
def primes : List ℕ := [2, 3, 5, 7, 17, 19]

noncomputable def sum_primes := primes.sum -- 2 + 3 + 5 + 7 + 17 + 19 = 53

-- Define two groups of primes to maximize the product of their sums
def group1 : List ℕ := [2, 3, 17]
def group2 : List ℕ := [5, 7, 19]

noncomputable def sum_group1 := group1.sum -- 2 + 3 + 17 = 22
noncomputable def sum_group2 := group2.sum -- 5 + 7 + 19 = 31

-- Formulate the proof problem
theorem max_sum_of_multiplication_table : 
  ∃ a b c d e f : ℕ, 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
    (a ∈ primes ∧ b ∈ primes ∧ c ∈ primes ∧ d ∈ primes ∧ e ∈ primes ∧ f ∈ primes) ∧ 
    (a + b + c = sum_group1 ∨ a + b + c = sum_group2) ∧ 
    (d + e + f = sum_group1 ∨ d + e + f = sum_group2) ∧ 
    (a + b + c) ≠ (d + e + f) ∧ 
    ((a + b + c) * (d + e + f) = 682) := 
by
  use 2, 3, 17, 5, 7, 19
  sorry

end max_sum_of_multiplication_table_l231_231015


namespace quadratic_relationship_l231_231212

variable (y_1 y_2 y_3 : ℝ)

-- Conditions
def vertex := (-2, 1)
def opens_downwards := true
def intersects_x_axis_at_two_points := true
def passes_through_points := [(1, y_1), (-1, y_2), (-4, y_3)]

-- Proof statement
theorem quadratic_relationship : y_1 < y_3 ∧ y_3 < y_2 :=
  sorry

end quadratic_relationship_l231_231212


namespace proposition_false_l231_231797

theorem proposition_false (a b : ℝ) (h : a + b > 0) : ¬ (a > 0 ∧ b > 0) := 
by {
  sorry -- this is a placeholder for the proof
}

end proposition_false_l231_231797


namespace mary_initial_stickers_l231_231846

theorem mary_initial_stickers (stickers_remaining : ℕ) 
  (front_page_stickers : ℕ) (other_page_stickers : ℕ) 
  (num_other_pages : ℕ) 
  (h1 : front_page_stickers = 3)
  (h2 : other_page_stickers = 7 * num_other_pages)
  (h3 : num_other_pages = 6)
  (h4 : stickers_remaining = 44) :
  ∃ initial_stickers : ℕ, initial_stickers = front_page_stickers + other_page_stickers + stickers_remaining ∧ initial_stickers = 89 :=
by
  sorry

end mary_initial_stickers_l231_231846


namespace necessary_and_sufficient_condition_l231_231485

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2 ≥ 2 * a * b) ↔ (a/b + b/a ≥ 2) :=
sorry

end necessary_and_sufficient_condition_l231_231485


namespace veridux_male_associates_l231_231550

theorem veridux_male_associates (total_employees female_employees total_managers female_managers : ℕ)
  (h1 : total_employees = 250)
  (h2 : female_employees = 90)
  (h3 : total_managers = 40)
  (h4 : female_managers = 40) :
  total_employees - female_employees = 160 :=
by
  sorry

end veridux_male_associates_l231_231550


namespace product_nonzero_except_cases_l231_231110

theorem product_nonzero_except_cases (n : ℤ) (h : n ≠ 5 ∧ n ≠ 17 ∧ n ≠ 257) : 
  (n - 5) * (n - 17) * (n - 257) ≠ 0 :=
by
  sorry

end product_nonzero_except_cases_l231_231110


namespace num_real_solutions_system_l231_231453

theorem num_real_solutions_system :
  ∃! (num_solutions : ℕ), 
  num_solutions = 5 ∧
  ∃ x y z w : ℝ, 
    (x = z + w + x * z) ∧ 
    (y = w + x + y * w) ∧ 
    (z = x + y + z * x) ∧ 
    (w = y + z + w * z) :=
sorry

end num_real_solutions_system_l231_231453


namespace abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l231_231481

theorem abs_eq_ax_plus_1_one_negative_root_no_positive_roots (a : ℝ) :
  (∃ x : ℝ, |x| = a * x + 1 ∧ x < 0) ∧ (∀ x : ℝ, |x| = a * x + 1 → x ≤ 0) → a > -1 :=
by
  sorry

end abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l231_231481


namespace minyoung_yoojung_flowers_l231_231898

theorem minyoung_yoojung_flowers (m y : ℕ) 
(h1 : m = 4 * y) 
(h2 : m = 24) : 
m + y = 30 := 
by
  sorry

end minyoung_yoojung_flowers_l231_231898


namespace remainder_when_divided_by_x_plus_2_l231_231575

-- Define the polynomial q(x) = D*x^4 + E*x^2 + F*x + 8
variable (D E F : ℝ)
def q (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 8

-- Given condition: q(2) = 12
axiom h1 : q D E F 2 = 12

-- Prove that q(-2) = 4
theorem remainder_when_divided_by_x_plus_2 : q D E F (-2) = 4 := by
  sorry

end remainder_when_divided_by_x_plus_2_l231_231575


namespace vector_line_form_to_slope_intercept_l231_231999

variable (x y : ℝ)

theorem vector_line_form_to_slope_intercept :
  (∀ (x y : ℝ), ((-1) * (x - 3) + 2 * (y + 4) = 0) ↔ (y = (-1/2) * x - 11/2)) :=
by
  sorry

end vector_line_form_to_slope_intercept_l231_231999


namespace seventy_second_number_in_S_is_573_l231_231398

open Nat

def S : Set Nat := { k | k % 8 = 5 }

theorem seventy_second_number_in_S_is_573 : ∃ k ∈ (Finset.range 650), k = 8 * 71 + 5 :=
by
  sorry -- Proof goes here

end seventy_second_number_in_S_is_573_l231_231398


namespace yards_mowed_by_christian_l231_231218

-- Definitions based on the provided conditions
def initial_savings := 5 + 7
def sue_earnings := 6 * 2
def total_savings := initial_savings + sue_earnings
def additional_needed := 50 - total_savings
def short_amount := 6
def christian_earnings := additional_needed - short_amount
def charge_per_yard := 5

theorem yards_mowed_by_christian : 
  (christian_earnings / charge_per_yard) = 4 :=
by
  sorry

end yards_mowed_by_christian_l231_231218


namespace percentage_increase_l231_231607

theorem percentage_increase (A B x y : ℝ) (h1 : A / B = (5 * y^2) / (6 * x)) (h2 : 2 * x + 3 * y = 42) :  
  (B - A) / A * 100 = ((126 - 9 * y - 5 * y^2) / (5 * y^2)) * 100 :=
by
  sorry

end percentage_increase_l231_231607


namespace negation_of_rectangular_parallelepipeds_have_12_edges_l231_231275

-- Define a structure for Rectangular Parallelepiped and the property of having edges
structure RectangularParallelepiped where
  hasEdges : ℕ → Prop

-- Problem statement
theorem negation_of_rectangular_parallelepipeds_have_12_edges :
  (∀ rect_p : RectangularParallelepiped, rect_p.hasEdges 12) →
  ∃ rect_p : RectangularParallelepiped, ¬ rect_p.hasEdges 12 := 
by
  sorry

end negation_of_rectangular_parallelepipeds_have_12_edges_l231_231275


namespace complex_number_quadrant_l231_231075

theorem complex_number_quadrant :
  let z := (2 * Complex.I) / (1 - Complex.I)
  Complex.re z < 0 ∧ Complex.im z > 0 :=
by
  sorry

end complex_number_quadrant_l231_231075


namespace sum_of_integers_ending_in_2_between_100_and_500_l231_231712

theorem sum_of_integers_ending_in_2_between_100_and_500 :
  let s : List ℤ := List.range' 102 400 10
  let sum_of_s := s.sum
  sum_of_s = 11880 :=
by
  sorry

end sum_of_integers_ending_in_2_between_100_and_500_l231_231712


namespace smallest_perfect_square_divisible_by_2_and_5_l231_231239

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℕ, n = k ^ 2) ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ (∃ k : ℕ, m = k ^ 2) ∧ (m % 2 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
sorry

end smallest_perfect_square_divisible_by_2_and_5_l231_231239


namespace train_passes_jogger_in_time_l231_231600

def jogger_speed_kmh : ℝ := 8
def train_speed_kmh : ℝ := 60
def initial_distance_m : ℝ := 360
def train_length_m : ℝ := 200

noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m
noncomputable def passing_time_s : ℝ := total_distance_m / relative_speed_ms

theorem train_passes_jogger_in_time :
  passing_time_s = 38.75 := by
  sorry

end train_passes_jogger_in_time_l231_231600


namespace sum_and_product_formulas_l231_231767

/-- 
Given an arithmetic sequence {a_n} with the sum of the first n terms S_n = 2n^2, 
and in the sequence {b_n}, b_1 = 1 and b_{n+1} = 3b_n (n ∈ ℕ*),
prove that:
(Ⅰ) The general formula for sequences {a_n} is a_n = 4n - 2,
(Ⅱ) The general formula for sequences {b_n} is b_n = 3^{n-1},
(Ⅲ) Let c_n = a_n * b_n, prove that the sum of the first n terms of the sequence {c_n}, denoted as T_n, is T_n = (2n - 2) * 3^n + 2.
-/
theorem sum_and_product_formulas (S_n : ℕ → ℕ) (b : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (T_n : ℕ → ℕ) :
  (∀ n, S_n n = 2 * n^2) →
  (b 1 = 1) →
  (∀ n, b (n + 1) = 3 * (b n)) →
  (∀ n, a n = S_n n - S_n (n - 1)) →
  ∀ n, (T_n n = (2*n - 2) * 3^n + 2) := sorry

end sum_and_product_formulas_l231_231767


namespace solve_equations_l231_231518

theorem solve_equations :
  (∃ x : ℝ, (x + 2) ^ 3 + 1 = 0 ∧ x = -3) ∧
  (∃ x : ℝ, ((3 * x - 2) ^ 2 = 64 ∧ (x = 10/3 ∨ x = -2))) :=
by {
  -- Prove the existence of solutions for both problems
  sorry
}

end solve_equations_l231_231518


namespace factorial_quotient_l231_231312

/-- Prove that the quotient of the factorial of 4! divided by 4! simplifies to 23!. -/
theorem factorial_quotient : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := 
by
  sorry

end factorial_quotient_l231_231312


namespace geometric_sequence_common_ratio_l231_231620

theorem geometric_sequence_common_ratio (r : ℝ) (a : ℝ) (a3 : ℝ) :
  a = 3 → a3 = 27 → r = 3 ∨ r = -3 :=
by
  intros ha ha3
  sorry

end geometric_sequence_common_ratio_l231_231620


namespace sprinter_time_no_wind_l231_231603

theorem sprinter_time_no_wind :
  ∀ (x y : ℝ), (90 / (x + y) = 10) → (70 / (x - y) = 10) → x = 8 * y → 100 / x = 12.5 :=
by
  intros x y h1 h2 h3
  sorry

end sprinter_time_no_wind_l231_231603


namespace gridPolygon_side_longer_than_one_l231_231890

-- Define the structure of a grid polygon
structure GridPolygon where
  area : ℕ  -- Area of the grid polygon
  perimeter : ℕ  -- Perimeter of the grid polygon
  no_holes : Prop  -- Polyon does not contain holes

-- Definition of a grid polygon with specific properties
def specificGridPolygon : GridPolygon :=
  { area := 300, perimeter := 300, no_holes := true }

-- The theorem we want to prove that ensures at least one side is longer than 1
theorem gridPolygon_side_longer_than_one (P : GridPolygon) (h_area : P.area = 300) (h_perimeter : P.perimeter = 300) (h_no_holes : P.no_holes) : ∃ side_length : ℝ, side_length > 1 :=
  by
  sorry

end gridPolygon_side_longer_than_one_l231_231890


namespace solution_set_of_inequality_l231_231670

theorem solution_set_of_inequality :
  {x : ℝ | |x - 5| + |x + 3| >= 10} = {x : ℝ | x ≤ -4} ∪ {x : ℝ | x ≥ 6} :=
by
  sorry

end solution_set_of_inequality_l231_231670


namespace compare_mixed_decimal_l231_231760

def mixed_number_value : ℚ := -2 - 1 / 3  -- Representation of -2 1/3 as a rational number
def decimal_value : ℚ := -2.3             -- Representation of -2.3 as a rational number

theorem compare_mixed_decimal : mixed_number_value < decimal_value :=
sorry

end compare_mixed_decimal_l231_231760


namespace david_twice_as_old_in_Y_years_l231_231795

variable (R D Y : ℕ)

-- Conditions
def rosy_current_age := R = 8
def david_is_older := D = R + 12
def twice_as_old_in_Y_years := D + Y = 2 * (R + Y)

-- Proof statement
theorem david_twice_as_old_in_Y_years
  (h1 : rosy_current_age R)
  (h2 : david_is_older R D)
  (h3 : twice_as_old_in_Y_years R D Y) :
  Y = 4 := sorry

end david_twice_as_old_in_Y_years_l231_231795


namespace area_of_R3_l231_231138

theorem area_of_R3 (r1 r2 r3 : ℝ) (h1: r1^2 = 25) 
                   (h2: r2 = (2/3) * r1) (h3: r3 = (2/3) * r2) :
                   r3^2 = 400 / 81 := 
by
  sorry

end area_of_R3_l231_231138


namespace distance_after_second_sign_l231_231311

-- Define the known conditions
def total_distance_ridden : ℕ := 1000
def distance_to_first_sign : ℕ := 350
def distance_between_signs : ℕ := 375

-- The distance Matt rode after passing the second sign
theorem distance_after_second_sign :
  total_distance_ridden - (distance_to_first_sign + distance_between_signs) = 275 := by
  sorry

end distance_after_second_sign_l231_231311


namespace sum_of_fractions_l231_231721

theorem sum_of_fractions : 
  (7 / 8 + 3 / 4) = (13 / 8) :=
by
  sorry

end sum_of_fractions_l231_231721


namespace derivative_is_even_then_b_eq_zero_l231_231974

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- The statement that the derivative is an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Our main theorem
theorem derivative_is_even_then_b_eq_zero : is_even (f' a b c) → b = 0 :=
by
  intro h
  have h1 := h 1
  have h2 := h (-1)
  sorry

end derivative_is_even_then_b_eq_zero_l231_231974


namespace probability_of_adjacent_rs_is_two_fifth_l231_231764

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def countArrangementsWithAdjacentRs : ℕ :=
factorial 4

noncomputable def countTotalArrangements : ℕ :=
factorial 5 / factorial 2

noncomputable def probabilityOfAdjacentRs : ℚ :=
(countArrangementsWithAdjacentRs : ℚ) / (countTotalArrangements : ℚ)

theorem probability_of_adjacent_rs_is_two_fifth :
  probabilityOfAdjacentRs = 2 / 5 := by
  sorry

end probability_of_adjacent_rs_is_two_fifth_l231_231764


namespace sonika_initial_deposit_l231_231693

variable (P R : ℝ)

theorem sonika_initial_deposit :
  (P + (P * R * 3) / 100 = 9200) → (P + (P * (R + 2.5) * 3) / 100 = 9800) → P = 8000 :=
by
  intros h1 h2
  sorry

end sonika_initial_deposit_l231_231693


namespace greatest_value_sum_eq_24_l231_231159

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l231_231159


namespace sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l231_231399

-- Proof 1: 
theorem sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3 :
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 :=
by
  sorry

-- Proof 2:
theorem sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12 :
  1 / Real.sqrt 24 + abs (Real.sqrt 6 - 3) + (1 / 2)⁻¹ - 2016 ^ 0 = 4 - 11 * Real.sqrt 6 / 12 :=
by
  sorry

-- Proof 3:
theorem sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6 :
  (Real.sqrt 3 + Real.sqrt 2) ^ 2 - (Real.sqrt 3 - Real.sqrt 2) ^ 2 = 4 * Real.sqrt 6 :=
by
  sorry

end sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l231_231399


namespace bucket_fill_proof_l231_231639

variables (x y : ℕ)
def tank_capacity : ℕ := 4 * x

theorem bucket_fill_proof (hx: y = x + 4) (hy: 4 * x = 3 * y): tank_capacity x = 48 :=
by {
  -- Proof steps will be here, but are elided for now
  sorry 
}

end bucket_fill_proof_l231_231639


namespace problem1_solution_problem2_solution_l231_231094

noncomputable def problem1 : Real :=
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2)

noncomputable def problem2 : Real :=
  (2 * Real.sqrt 3 + Real.sqrt 6) * (2 * Real.sqrt 3 - Real.sqrt 6)

theorem problem1_solution : problem1 = 0 := by
  sorry

theorem problem2_solution : problem2 = 6 := by
  sorry

end problem1_solution_problem2_solution_l231_231094


namespace least_number_divisible_by_23_l231_231957

theorem least_number_divisible_by_23 (n d : ℕ) (h_n : n = 1053) (h_d : d = 23) : ∃ x : ℕ, (n + x) % d = 0 ∧ x = 5 := by
  sorry

end least_number_divisible_by_23_l231_231957


namespace lcm_factor_is_one_l231_231601

theorem lcm_factor_is_one
  (A B : ℕ)
  (hcf : A.gcd B = 42)
  (larger_A : A = 588)
  (other_factor : ∃ X, A.lcm B = 42 * X * 14) :
  ∃ X, X = 1 :=
  sorry

end lcm_factor_is_one_l231_231601


namespace fraction_product_simplification_l231_231959

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by
  sorry

end fraction_product_simplification_l231_231959


namespace diego_can_carry_home_l231_231606

theorem diego_can_carry_home (T W G O A : ℕ) (hT : T = 20) (hW : W = 1) (hG : G = 1) (hO : O = 1) : A = T - (W + G + O) → A = 17 := by
  sorry

end diego_can_carry_home_l231_231606


namespace band_song_arrangements_l231_231104

theorem band_song_arrangements (n : ℕ) (t : ℕ) (r : ℕ) 
  (h1 : n = 8) (h2 : t = 3) (h3 : r = 5) : 
  ∃ (ways : ℕ), ways = 14400 := by
  sorry

end band_song_arrangements_l231_231104


namespace car_speed_kmph_l231_231087

noncomputable def speed_of_car (d : ℝ) (t : ℝ) : ℝ :=
  (d / t) * 3.6

theorem car_speed_kmph : speed_of_car 10 0.9999200063994881 = 36000.29 := by
  sorry

end car_speed_kmph_l231_231087


namespace averagePrice_is_20_l231_231532

-- Define the conditions
def books1 : Nat := 32
def cost1 : Nat := 1500

def books2 : Nat := 60
def cost2 : Nat := 340

-- Define the total books and total cost
def totalBooks : Nat := books1 + books2
def totalCost : Nat := cost1 + cost2

-- Define the average price calculation
def averagePrice : Nat := totalCost / totalBooks

-- The statement to prove
theorem averagePrice_is_20 : averagePrice = 20 := by
  -- Sorry is used here as a placeholder for the actual proof.
  sorry

end averagePrice_is_20_l231_231532


namespace intersection_of_sets_l231_231220

theorem intersection_of_sets (M : Set ℤ) (N : Set ℤ) (H_M : M = {0, 1, 2, 3, 4}) (H_N : N = {-2, 0, 2}) :
  M ∩ N = {0, 2} :=
by
  rw [H_M, H_N]
  ext
  simp
  sorry  -- Proof to be filled in

end intersection_of_sets_l231_231220


namespace no_solution_for_t_and_s_l231_231634

theorem no_solution_for_t_and_s (m : ℝ) :
  (¬∃ t s : ℝ, (1 + 7 * t = -3 + 2 * s) ∧ (3 - 5 * t = 4 + m * s)) ↔ m = -10 / 7 :=
by
  sorry

end no_solution_for_t_and_s_l231_231634


namespace prove_sums_l231_231743

-- Given conditions
def condition1 (a b : ℤ) : Prop := ∀ x : ℝ, (x + a) * (x + b) = x^2 + 9 * x + 14
def condition2 (b c : ℤ) : Prop := ∀ x : ℝ, (x + b) * (x - c) = x^2 + 7 * x - 30

-- We need to prove that a + b + c = 15
theorem prove_sums (a b c : ℤ) (h1: condition1 a b) (h2: condition2 b c) : a + b + c = 15 := 
sorry

end prove_sums_l231_231743


namespace max_product_of_real_roots_quadratic_eq_l231_231614

theorem max_product_of_real_roots_quadratic_eq : ∀ (k : ℝ), (∃ x y : ℝ, 4 * x ^ 2 - 8 * x + k = 0 ∧ 4 * y ^ 2 - 8 * y + k = 0) 
    → k = 4 :=
sorry

end max_product_of_real_roots_quadratic_eq_l231_231614


namespace range_of_a_l231_231818

theorem range_of_a (a : ℝ) : (3 + 5 > 1 - 2 * a) ∧ (3 + (1 - 2 * a) > 5) ∧ (5 + (1 - 2 * a) > 3) → -7 / 2 < a ∧ a < -1 / 2 :=
by
  sorry

end range_of_a_l231_231818


namespace find_f_of_fraction_l231_231968

noncomputable def f (t : ℝ) : ℝ := sorry

theorem find_f_of_fraction (x : ℝ) (h : f ((1-x^2)/(1+x^2)) = x) :
  f ((2*x)/(1+x^2)) = (1 - x) / (1 + x) ∨ f ((2*x)/(1+x^2)) = (x - 1) / (1 + x) :=
sorry

end find_f_of_fraction_l231_231968


namespace equidistant_trajectory_l231_231836

theorem equidistant_trajectory (x y : ℝ) (h : abs x = abs y) : y^2 = x^2 :=
by
  sorry

end equidistant_trajectory_l231_231836


namespace number_of_paths_3x3_l231_231531

-- Definition of the problem conditions
def grid_moves (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- Lean statement for the proof problem
theorem number_of_paths_3x3 : grid_moves 3 3 = 20 := by
  sorry

end number_of_paths_3x3_l231_231531


namespace base4_base7_digit_difference_l231_231543

def num_digits_base (n b : ℕ) : ℕ :=
  if b > 1 then Nat.log b n + 1 else 0

theorem base4_base7_digit_difference :
  let n := 1573
  num_digits_base n 4 - num_digits_base n 7 = 2 := by
  sorry

end base4_base7_digit_difference_l231_231543


namespace lilies_per_centerpiece_correct_l231_231288

-- Definitions based on the conditions
def num_centerpieces : ℕ := 6
def roses_per_centerpiece : ℕ := 8
def cost_per_flower : ℕ := 15
def total_budget : ℕ := 2700

-- Definition of the number of orchids per centerpiece using given condition
def orchids_per_centerpiece : ℕ := 2 * roses_per_centerpiece

-- Definition of the total cost for roses and orchids before calculating lilies
def total_rose_cost : ℕ := num_centerpieces * roses_per_centerpiece * cost_per_flower
def total_orchid_cost : ℕ := num_centerpieces * orchids_per_centerpiece * cost_per_flower
def total_rose_and_orchid_cost : ℕ := total_rose_cost + total_orchid_cost

-- Definition for the remaining budget for lilies
def remaining_budget_for_lilies : ℕ := total_budget - total_rose_and_orchid_cost

-- Number of lilies in total and per centerpiece
def total_lilies : ℕ := remaining_budget_for_lilies / cost_per_flower
def lilies_per_centerpiece : ℕ := total_lilies / num_centerpieces

-- The proof statement we want to assert
theorem lilies_per_centerpiece_correct : lilies_per_centerpiece = 6 :=
by
  sorry

end lilies_per_centerpiece_correct_l231_231288


namespace geom_inequality_l231_231376

variables {Point : Type} [MetricSpace Point] {O A B C K L H M : Point}

/-- Conditions -/
def circumcenter_of_triangle (O A B C : Point) : Prop := 
 -- Definition that O is the circumcenter of triangle ABC
 sorry 

def midpoint_of_arc (K B C A : Point) : Prop := 
 -- Definition that K is the midpoint of the arc BC not containing A
 sorry

def lies_on_line (K L A : Point) : Prop := 
 -- Definition that K lies on line AL
 sorry

def similar_triangles (A H L K M : Point) : Prop := 
 -- Definition that triangles AHL and KML are similar
 sorry 

def segment_inequality (AL KL : ℝ) : Prop := 
 -- Definition that AL < KL
 sorry 

/-- Proof Problem -/
theorem geom_inequality (h1 : circumcenter_of_triangle O A B C) 
                       (h2: midpoint_of_arc K B C A)
                       (h3: lies_on_line K L A)
                       (h4: similar_triangles A H L K M)
                       (h5: segment_inequality (dist A L) (dist K L)) : 
  dist A K < dist B C := 
sorry

end geom_inequality_l231_231376


namespace lunch_cost_calc_l231_231264

-- Define the given conditions
def gasoline_cost : ℝ := 8
def gift_cost : ℝ := 5
def grandma_gift : ℝ := 10
def initial_money : ℝ := 50
def return_trip_money : ℝ := 36.35

-- Calculate the total expenses and determine the money spent on lunch
def total_gifts_cost : ℝ := 2 * gift_cost
def total_money_received : ℝ := initial_money + 2 * grandma_gift
def total_gas_gift_cost : ℝ := gasoline_cost + total_gifts_cost
def expected_remaining_money : ℝ := total_money_received - total_gas_gift_cost
def lunch_cost : ℝ := expected_remaining_money - return_trip_money

-- State theorem
theorem lunch_cost_calc : lunch_cost = 15.65 := by
  sorry

end lunch_cost_calc_l231_231264


namespace right_angle_triangle_l231_231609

theorem right_angle_triangle (a b c : ℝ) (h : (a + b) ^ 2 - c ^ 2 = 2 * a * b) : a ^ 2 + b ^ 2 = c ^ 2 := 
by
  sorry

end right_angle_triangle_l231_231609


namespace num_solutions_non_negative_reals_l231_231502

-- Define the system of equations as a function to express the cyclic nature
def system_of_equations (n : ℕ) (x : ℕ → ℝ) (k : ℕ) : Prop :=
  x (k + 1 % n) + (x (if k = 0 then n else k) ^ 2) = 4 * x (if k = 0 then n else k)

-- Define the main theorem stating the number of solutions
theorem num_solutions_non_negative_reals {n : ℕ} (hn : 0 < n) : 
  ∃ (s : Finset (ℕ → ℝ)), (∀ x ∈ s, ∀ k, 0 ≤ (x k) ∧ system_of_equations n x k) ∧ s.card = 2^n :=
sorry

end num_solutions_non_negative_reals_l231_231502


namespace number_halfway_l231_231350

theorem number_halfway (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/10) : (a + b) / 2 = 11 / 120 := by
  sorry

end number_halfway_l231_231350


namespace range_of_a_l231_231386

variable (a : ℝ)

def condition1 : Prop := a < 0
def condition2 : Prop := -a / 2 ≥ 1
def condition3 : Prop := -1 - a - 5 ≤ a

theorem range_of_a :
  condition1 a ∧ condition2 a ∧ condition3 a → -3 ≤ a ∧ a ≤ -2 :=
by
  sorry

end range_of_a_l231_231386


namespace time_taken_by_A_l231_231263

theorem time_taken_by_A (t : ℚ) (h1 : 3 * (t + 1 / 2) = 4 * t) : t = 3 / 2 ∧ (t + 1 / 2) = 2 := 
  by
  intros
  sorry

end time_taken_by_A_l231_231263


namespace car_distance_covered_l231_231143

def distance_covered_by_car (time : ℝ) (speed : ℝ) : ℝ :=
  speed * time

theorem car_distance_covered :
  distance_covered_by_car (3 + 1/5 : ℝ) 195 = 624 :=
by
  sorry

end car_distance_covered_l231_231143


namespace train_length_l231_231066

theorem train_length :
  ∀ (t : ℝ) (v_man : ℝ) (v_train : ℝ),
  t = 41.9966402687785 →
  v_man = 3 →
  v_train = 63 →
  (v_train - v_man) * (5 / 18) * t = 699.94400447975 :=
by
  intros t v_man v_train ht hv_man hv_train
  -- Use the given conditions as definitions
  rw [ht, hv_man, hv_train]
  sorry

end train_length_l231_231066


namespace plane_equation_l231_231231

theorem plane_equation (x y z : ℝ) (A B C D : ℤ) (h1 : A = 9) (h2 : B = -6) (h3 : C = 4) (h4 : D = -133) (A_pos : A > 0) (gcd_condition : Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) : 
  A * x + B * y + C * z + D = 0 :=
sorry

end plane_equation_l231_231231


namespace unique_ordered_triple_l231_231590

theorem unique_ordered_triple : 
  ∃! (x y z : ℝ), x + y = 4 ∧ xy - z^2 = 4 ∧ x = 2 ∧ y = 2 ∧ z = 0 :=
by
  sorry

end unique_ordered_triple_l231_231590


namespace geometric_sequence_a7_l231_231091

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) := a_1 * q^(n - 1)

theorem geometric_sequence_a7 
  (a1 q : ℝ)
  (a1_neq_zero : a1 ≠ 0)
  (a9_eq_256 : a_n a1 q 9 = 256)
  (a1_a3_eq_4 : a_n a1 q 1 * a_n a1 q 3 = 4) :
  a_n a1 q 7 = 64 := 
sorry

end geometric_sequence_a7_l231_231091


namespace quadratic_inequality_l231_231611

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, x^2 - a * x + a > 0) : 0 < a ∧ a < 4 :=
sorry

end quadratic_inequality_l231_231611


namespace count_true_statements_l231_231396

open Set

variable {M P : Set α}

theorem count_true_statements (h : ¬ ∀ x ∈ M, x ∈ P) (hne : Nonempty M) :
  (¬ ∃ x, x ∈ M ∧ x ∈ P ∨ ∀ x, x ∈ M → x ∈ P) ∧ (∃ x, x ∈ M ∧ x ∉ P) ∧ 
  ¬ (∃ x, x ∈ M ∧ x ∈ P) ∧ (¬ ∀ x, x ∈ M → x ∈ P) :=
sorry

end count_true_statements_l231_231396


namespace composite_has_at_least_three_divisors_l231_231692

def is_composite (n : ℕ) : Prop := ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_divisors (n : ℕ) (h : is_composite n) : ∃ a b c, a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c :=
sorry

end composite_has_at_least_three_divisors_l231_231692


namespace max_area_of_triangle_ABC_l231_231328

-- Definitions for the problem conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (5, 4)
def parabola (x : ℝ) : ℝ := x^2 - 3 * x
def C (r : ℝ) : ℝ × ℝ := (r, parabola r)

-- Function to compute the Shoelace Theorem area of ABC
def shoelace_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Proof statement
theorem max_area_of_triangle_ABC : ∃ (r : ℝ), -2 ≤ r ∧ r ≤ 5 ∧ shoelace_area A B (C r) = 39 := 
  sorry

end max_area_of_triangle_ABC_l231_231328


namespace find_common_ratio_l231_231826

-- Define the geometric sequence with the given conditions
variable (a_n : ℕ → ℝ)
variable (q : ℝ)

axiom a2_eq : a_n 2 = 1
axiom a4_eq : a_n 4 = 4
axiom q_pos : q > 0

-- Define the nature of the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The specific problem statement to prove
theorem find_common_ratio (h: is_geometric_sequence a_n q) : q = 2 :=
by
  sorry

end find_common_ratio_l231_231826


namespace greatest_possible_large_chips_l231_231669

theorem greatest_possible_large_chips : 
  ∃ s l p: ℕ, s + l = 60 ∧ s = l + 2 * p ∧ Prime p ∧ l = 28 :=
by
  sorry

end greatest_possible_large_chips_l231_231669


namespace num_special_matrices_l231_231984

open Matrix

theorem num_special_matrices :
  ∃ (M : Matrix (Fin 4) (Fin 4) ℕ), 
    (∀ i j, 1 ≤ M i j ∧ M i j ≤ 16) ∧ 
    (∀ i j, i < j → M i j < M i (j + 1)) ∧ 
    (∀ i j, i < j → M i j < M (i + 1) j) ∧ 
    (∀ i, i < 3 → M i i < M (i + 1) (i + 1)) ∧ 
    (∀ i, i < 3 → M i (3 - i) < M (i + 1) (2 - i)) ∧ 
    (∃ n, n = 144) :=
sorry

end num_special_matrices_l231_231984


namespace triangle_proof_l231_231203

noncomputable def triangle_math_proof (A B C : ℝ) (AA1 BB1 CC1 : ℝ) : Prop :=
  AA1 = 2 * Real.sin (B + A / 2) ∧
  BB1 = 2 * Real.sin (C + B / 2) ∧
  CC1 = 2 * Real.sin (A + C / 2) ∧
  (Real.sin A + Real.sin B + Real.sin C) ≠ 0 ∧
  ∀ x, x = (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) / (Real.sin A + Real.sin B + Real.sin C) → x = 2

theorem triangle_proof (A B C AA1 BB1 CC1 : ℝ) (h : triangle_math_proof A B C AA1 BB1 CC1) :
  (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) /
  (Real.sin A + Real.sin B + Real.sin C) = 2 := by
  sorry

end triangle_proof_l231_231203


namespace locus_of_point_is_circle_l231_231546

theorem locus_of_point_is_circle (x y : ℝ) 
  (h : 10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3 * x - 4 * y|) : 
  ∃ (c : ℝ) (r : ℝ), ∀ (x y : ℝ), (x - c)^2 + (y - c)^2 = r^2 := 
sorry

end locus_of_point_is_circle_l231_231546


namespace hundredth_term_sequence_l231_231063

def numerators (n : ℕ) : ℕ := 1 + (n - 1) * 2
def denominators (n : ℕ) : ℕ := 2 + (n - 1) * 3

theorem hundredth_term_sequence : numerators 100 / denominators 100 = 199 / 299 := by
  sorry

end hundredth_term_sequence_l231_231063


namespace team_B_score_third_game_l231_231224

theorem team_B_score_third_game (avg_points : ℝ) (additional_needed : ℝ) (total_target : ℝ) (P : ℝ) :
  avg_points = 61.5 → additional_needed = 330 → total_target = 500 →
  2 * avg_points + P + additional_needed = total_target → P = 47 :=
by
  intros avg_points_eq additional_needed_eq total_target_eq total_eq
  rw [avg_points_eq, additional_needed_eq, total_target_eq] at total_eq
  sorry

end team_B_score_third_game_l231_231224


namespace range_of_b_min_value_a_add_b_min_value_ab_l231_231742

theorem range_of_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : b > 1 := sorry

theorem min_value_a_add_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a + b ≥ 8 := sorry

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a * b ≥ 16 := sorry

end range_of_b_min_value_a_add_b_min_value_ab_l231_231742


namespace exists_sum_of_squares_form_l231_231377

theorem exists_sum_of_squares_form (n : ℕ) (h : n % 25 = 9) :
  ∃ (a b c : ℕ), n = (a * (a + 1)) / 2 + (b * (b + 1)) / 2 + (c * (c + 1)) / 2 := 
by 
  sorry

end exists_sum_of_squares_form_l231_231377


namespace quadratic_equation_unique_solution_l231_231722

theorem quadratic_equation_unique_solution (p : ℚ) :
  (∃ x : ℚ, 3 * x^2 - 7 * x + p = 0) ∧ 
  ∀ y : ℚ, 3 * y^2 -7 * y + p ≠ 0 → ∀ z : ℚ, 3 * z^2 - 7 * z + p = 0 → y = z ↔ 
  p = 49 / 12 :=
by
  sorry

end quadratic_equation_unique_solution_l231_231722


namespace not_set_of_difficult_problems_l231_231004

-- Define the context and entities
inductive Exercise
| ex (n : Nat) : Exercise  -- Example definition for exercises, assumed to be numbered

def is_difficult (ex : Exercise) : Prop := sorry  -- Placeholder for the subjective predicate

-- Define the main problem statement
theorem not_set_of_difficult_problems
  (Difficult : Exercise → Prop) -- Subjective predicate defining difficult problems
  (H_subj : ∀ (e : Exercise), (Difficult e ↔ is_difficult e)) :
  ¬(∃ (S : Set Exercise), ∀ e, e ∈ S ↔ Difficult e) :=
sorry

end not_set_of_difficult_problems_l231_231004


namespace right_triangle_hypotenuse_l231_231598

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 75) (h₂ : b = 100) : ∃ c, c = 125 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end right_triangle_hypotenuse_l231_231598


namespace walkway_area_correct_l231_231911

-- Define the dimensions of one flower bed
def flower_bed_width := 8
def flower_bed_height := 3

-- Define the number of flower beds and the width of the walkways
def num_flowers_horizontal := 3
def num_flowers_vertical := 4
def walkway_width := 2

-- Calculate the total dimension of the garden including both flower beds and walkways
def total_garden_width := (num_flowers_horizontal * flower_bed_width) + ((num_flowers_horizontal + 1) * walkway_width)
def total_garden_height := (num_flowers_vertical * flower_bed_height) + ((num_flowers_vertical + 1) * walkway_width)

-- Calculate the total area of the garden and the total area of the flower beds
def total_garden_area := total_garden_width * total_garden_height
def total_flower_bed_area := (flower_bed_width * flower_bed_height) * (num_flowers_horizontal * num_flowers_vertical)

-- Calculate the total area of the walkways in the garden
def total_walkway_area := total_garden_area - total_flower_bed_area

-- The statement to be proven:
theorem walkway_area_correct : total_walkway_area = 416 := by
  sorry

end walkway_area_correct_l231_231911


namespace trapezium_other_side_length_l231_231838

theorem trapezium_other_side_length 
  (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_distance : distance = 13)
  (h_area : area = 247)
  (h_area_formula : area = 0.5 * (side1 + side2) * distance) :
  side2 = 20 :=
by
  rw [h_side1, h_distance, h_area] at h_area_formula
  sorry

end trapezium_other_side_length_l231_231838


namespace prob_each_student_gets_each_snack_l231_231411

-- Define the total number of snacks and their types
def total_snacks := 16
def snack_types := 4

-- Define the conditions for the problem
def students := 4
def snacks_per_type := 4

-- Define the probability calculation.
-- We would typically use combinatorial functions here, but for simplicity, use predefined values from the solution.
def prob_student_1 := 64 / 455
def prob_student_2 := 9 / 55
def prob_student_3 := 8 / 35
def prob_student_4 := 1 -- Always 1 for the final student's remaining snacks

-- Calculate the total probability
def total_prob := prob_student_1 * prob_student_2 * prob_student_3 * prob_student_4

-- The statement to prove the desired probability outcome
theorem prob_each_student_gets_each_snack : total_prob = (64 / 1225) :=
by
  sorry

end prob_each_student_gets_each_snack_l231_231411


namespace vertex_y_coordinate_l231_231687

theorem vertex_y_coordinate (x : ℝ) : 
    let a := -6
    let b := 24
    let c := -7
    ∃ k : ℝ, k = 17 ∧ ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x - 2)^2 + k) := 
by 
  sorry

end vertex_y_coordinate_l231_231687


namespace largest_five_digit_palindromic_number_l231_231024

def is_five_digit_palindrome (n : ℕ) : Prop := n / 10000 = n % 10 ∧ (n / 1000) % 10 = (n / 10) % 10

def is_four_digit_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100) % 10 = (n / 10) % 10

theorem largest_five_digit_palindromic_number :
  ∃ (abcba deed : ℕ), is_five_digit_palindrome abcba ∧ 10000 ≤ abcba ∧ abcba < 100000 ∧ is_four_digit_palindrome deed ∧ 1000 ≤ deed ∧ deed < 10000 ∧ abcba = 45 * deed ∧ abcba = 59895 :=
by
  sorry

end largest_five_digit_palindromic_number_l231_231024


namespace max_value_of_function_l231_231756

noncomputable def function_y (x : ℝ) : ℝ := x + Real.sin x

theorem max_value_of_function : 
  ∀ (a b : ℝ), a = 0 → b = Real.pi → 
  (∀ x : ℝ, x ∈ Set.Icc a b → x + Real.sin x ≤ Real.pi) :=
by
  intros a b ha hb x hx
  sorry

end max_value_of_function_l231_231756


namespace number_of_persons_in_first_group_eq_39_l231_231112

theorem number_of_persons_in_first_group_eq_39 :
  ∀ (P : ℕ),
    (P * 12 * 5 = 15 * 26 * 6) →
    P = 39 :=
by
  intros P h
  have h1 : P = (15 * 26 * 6) / (12 * 5) := sorry
  simp at h1
  exact h1

end number_of_persons_in_first_group_eq_39_l231_231112


namespace coin_toss_fairness_l231_231975

-- Statement of the problem as a Lean theorem.
theorem coin_toss_fairness (P_Heads P_Tails : ℝ) (h1 : P_Heads = 0.5) (h2 : P_Tails = 0.5) : 
  P_Heads = P_Tails ∧ P_Heads = 0.5 := 
sorry

end coin_toss_fairness_l231_231975


namespace present_age_of_A_l231_231346

theorem present_age_of_A {x : ℕ} (h₁ : ∃ (x : ℕ), 5 * x = A ∧ 3 * x = B)
                         (h₂ : ∀ (A B : ℕ), (A + 6) / (B + 6) = 7 / 5) : A = 15 :=
by sorry

end present_age_of_A_l231_231346


namespace cycle_selling_price_l231_231876

theorem cycle_selling_price
  (cost_price : ℝ)
  (gain_percentage : ℝ)
  (profit : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 930)
  (h2 : gain_percentage = 30.107526881720432)
  (h3 : profit = (gain_percentage / 100) * cost_price)
  (h4 : selling_price = cost_price + profit)
  : selling_price = 1210 := 
sorry

end cycle_selling_price_l231_231876


namespace hyperbola_sufficient_asymptotes_l231_231517

open Real

def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptotes_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

theorem hyperbola_sufficient_asymptotes (a b x y : ℝ) :
  (hyperbola_eq a b x y) → (asymptotes_eq a b x y) :=
by
  sorry

end hyperbola_sufficient_asymptotes_l231_231517


namespace sum_of_remainders_l231_231709

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : 
  (n % 2) + (n % 3) + (n % 6) + (n % 9) = 10 := 
by 
  sorry

end sum_of_remainders_l231_231709


namespace solution_to_problem_l231_231464

theorem solution_to_problem (x y : ℕ) : 
  (x.gcd y + x.lcm y = x + y) ↔ 
  ∃ (d k : ℕ), (x = d ∧ y = d * k) ∨ (x = d * k ∧ y = d) :=
by sorry

end solution_to_problem_l231_231464


namespace john_drinks_2_cups_per_day_l231_231845

noncomputable def fluid_ounces_in_gallon : ℕ := 128

noncomputable def half_gallon_in_fluid_ounces : ℕ := 64

noncomputable def standard_cup_size : ℕ := 8

noncomputable def cups_in_half_gallon : ℕ :=
  half_gallon_in_fluid_ounces / standard_cup_size

noncomputable def days_to_consume_half_gallon : ℕ := 4

noncomputable def cups_per_day : ℕ :=
  cups_in_half_gallon / days_to_consume_half_gallon

theorem john_drinks_2_cups_per_day :
  cups_per_day = 2 :=
by
  -- The proof is left as an exercise, but the statement should be correct.
  sorry

end john_drinks_2_cups_per_day_l231_231845


namespace total_flowers_eaten_l231_231568

theorem total_flowers_eaten :
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  f1 + f2 + f3 + f4 + f5 + f6 + f7 = 16.5 :=
by
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  sorry

end total_flowers_eaten_l231_231568


namespace part1_solution_set_l231_231401

theorem part1_solution_set (a : ℝ) (x : ℝ) : a = -2 → (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0 ↔ x ≠ -1 :=
by sorry

end part1_solution_set_l231_231401


namespace max_area_triangle_l231_231855

open Real

theorem max_area_triangle (a b : ℝ) (C : ℝ) (h₁ : a + b = 4) (h₂ : C = π / 6) : 
  (1 : ℝ) ≥ (1 / 2 * a * b * sin (π / 6)) := 
by 
  sorry

end max_area_triangle_l231_231855


namespace candidate_1_fails_by_40_marks_l231_231437

-- Definitions based on the conditions
def total_marks (T : ℕ) := T
def passing_marks (pass : ℕ) := pass = 160
def candidate_1_failed_by (marks_failed_by : ℕ) := ∃ (T : ℕ), (0.4 : ℝ) * T = 0.4 * T ∧ (0.6 : ℝ) * T - 20 = 160

-- Theorem to prove the first candidate fails by 40 marks
theorem candidate_1_fails_by_40_marks (marks_failed_by : ℕ) : candidate_1_failed_by marks_failed_by → marks_failed_by = 40 :=
by
  sorry

end candidate_1_fails_by_40_marks_l231_231437


namespace remainder_x_squared_mod_25_l231_231660

theorem remainder_x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 4 [ZMOD 25] :=
sorry

end remainder_x_squared_mod_25_l231_231660


namespace total_amount_spent_l231_231725

theorem total_amount_spent (avg_price_goat : ℕ) (num_goats : ℕ) (avg_price_cow : ℕ) (num_cows : ℕ) (total_spent : ℕ) 
  (h1 : avg_price_goat = 70) (h2 : num_goats = 10) (h3 : avg_price_cow = 400) (h4 : num_cows = 2) :
  total_spent = 1500 :=
by
  have cost_goats := avg_price_goat * num_goats
  have cost_cows := avg_price_cow * num_cows
  have total := cost_goats + cost_cows
  sorry

end total_amount_spent_l231_231725


namespace meaningful_fraction_implies_neq_neg4_l231_231436

theorem meaningful_fraction_implies_neq_neg4 (x : ℝ) : (x + 4 ≠ 0) ↔ (x ≠ -4) := 
by
  sorry

end meaningful_fraction_implies_neq_neg4_l231_231436


namespace arrange_logs_in_order_l231_231352

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.8 / Real.log 1.2
noncomputable def c : ℝ := Real.sqrt 1.5

theorem arrange_logs_in_order : b < a ∧ a < c := by
  sorry

end arrange_logs_in_order_l231_231352


namespace students_in_5th_6th_grades_l231_231389

-- Definitions for problem conditions
def is_three_digit_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def six_two_digit_sum_eq_twice (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧
               a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
               (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) = 2 * n

-- The proof problem statement in Lean 4
theorem students_in_5th_6th_grades :
  ∃ n : ℕ, is_three_digit_number n ∧ six_two_digit_sum_eq_twice n ∧ n = 198 :=
by
  sorry

end students_in_5th_6th_grades_l231_231389


namespace rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l231_231695

variables (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0)

def S1 := (x + 5) * (y + 5)
def S2 := (x - 2) * (y - 2)
def perimeter := 2 * (x + y)

theorem rectangle_perimeter (h : S1 - S2 = 196) :
  perimeter = 50 :=
sorry

theorem difference_multiple_of_7 (h : S1 - S2 = 196) :
  ∃ k : ℕ, S1 - S2 = 7 * k :=
sorry

theorem area_seamless_combination (h : S1 - S2 = 196) :
  S1 - x * y = (x + 5) * (y + 5) - x * y ∧ x = y + 5 :=
sorry

end rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l231_231695


namespace nonempty_solution_set_iff_a_gt_2_l231_231320

theorem nonempty_solution_set_iff_a_gt_2 (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < a) ↔ a > 2 :=
sorry

end nonempty_solution_set_iff_a_gt_2_l231_231320


namespace george_total_socks_l231_231339

-- Define the initial number of socks George had
def initial_socks : ℝ := 28.0

-- Define the number of socks he bought
def bought_socks : ℝ := 36.0

-- Define the number of socks his Dad gave him
def given_socks : ℝ := 4.0

-- Define the number of total socks
def total_socks : ℝ := initial_socks + bought_socks + given_socks

-- State the theorem we want to prove
theorem george_total_socks : total_socks = 68.0 :=
by
  sorry

end george_total_socks_l231_231339


namespace price_of_table_l231_231042

-- Given the conditions:
def chair_table_eq1 (C T : ℝ) : Prop := 2 * C + T = 0.6 * (C + 2 * T)
def chair_table_eq2 (C T : ℝ) : Prop := C + T = 72

-- Prove that the price of one table is $63
theorem price_of_table (C T : ℝ) (h1 : chair_table_eq1 C T) (h2 : chair_table_eq2 C T) : T = 63 := by
  sorry

end price_of_table_l231_231042


namespace original_amount_charged_l231_231152

variables (P : ℝ) (interest_rate : ℝ) (total_owed : ℝ)

theorem original_amount_charged :
  interest_rate = 0.09 →
  total_owed = 38.15 →
  (P + P * interest_rate = total_owed) →
  P = 35 :=
by
  intros h_interest_rate h_total_owed h_equation
  sorry

end original_amount_charged_l231_231152


namespace walt_total_invested_l231_231665

-- Given Conditions
def invested_at_seven : ℝ := 5500
def total_interest : ℝ := 970
def interest_rate_seven : ℝ := 0.07
def interest_rate_nine : ℝ := 0.09

-- Define the total amount invested
noncomputable def total_invested : ℝ := 12000

-- Prove the total amount invested
theorem walt_total_invested :
  interest_rate_seven * invested_at_seven + interest_rate_nine * (total_invested - invested_at_seven) = total_interest :=
by
  -- The proof goes here
  sorry

end walt_total_invested_l231_231665


namespace circle_center_is_neg4_2_l231_231291

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + 8 * x + y^2 - 4 * y = 16

theorem circle_center_is_neg4_2 :
  ∃ (h k : ℝ), (h = -4 ∧ k = 2) ∧
  ∀ (x y : ℝ), circle_center x y ↔ (x + 4)^2 + (y - 2)^2 = 36 :=
by
  sorry

end circle_center_is_neg4_2_l231_231291


namespace min_xy_solution_l231_231420

theorem min_xy_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 2 * x + 8 * y) :
  (x = 16 ∧ y = 4) :=
by
  sorry

end min_xy_solution_l231_231420


namespace y1_increasing_on_0_1_l231_231022

noncomputable def y1 (x : ℝ) : ℝ := |x|
noncomputable def y2 (x : ℝ) : ℝ := 3 - x
noncomputable def y3 (x : ℝ) : ℝ := 1 / x
noncomputable def y4 (x : ℝ) : ℝ := -x^2 + 4

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem y1_increasing_on_0_1 :
  is_increasing_on y1 0 1 ∧
  ¬ is_increasing_on y2 0 1 ∧
  ¬ is_increasing_on y3 0 1 ∧
  ¬ is_increasing_on y4 0 1 :=
by
  sorry

end y1_increasing_on_0_1_l231_231022


namespace decagon_diagonal_intersection_probability_l231_231249

def probability_intersect_within_decagon : ℚ :=
  let total_vertices := 10
  let total_pairs_points := Nat.choose total_vertices 2
  let total_diagonals := total_pairs_points - total_vertices
  let ways_to_pick_2_diagonals := Nat.choose total_diagonals 2
  let combinations_4_vertices := Nat.choose total_vertices 4
  (combinations_4_vertices : ℚ) / (ways_to_pick_2_diagonals : ℚ)

theorem decagon_diagonal_intersection_probability :
  probability_intersect_within_decagon = 42 / 119 :=
sorry

end decagon_diagonal_intersection_probability_l231_231249


namespace coachClass_seats_count_l231_231859

-- Defining the conditions as given in a)
variables (F : ℕ) -- Number of first-class seats
variables (totalSeats : ℕ := 567) -- Total number of seats is given as 567
variables (businessClassSeats : ℕ := 3 * F) -- Business class seats defined in terms of F
variables (coachClassSeats : ℕ := 7 * F + 5) -- Coach class seats defined in terms of F
variables (firstClassSeats : ℕ := F) -- The variable itself

-- The statement to prove
theorem coachClass_seats_count : 
  F + businessClassSeats + coachClassSeats = totalSeats →
  coachClassSeats = 362 :=
by
  sorry -- The proof would go here

end coachClass_seats_count_l231_231859


namespace quotient_division_l231_231433

noncomputable def poly_division_quotient : Polynomial ℚ :=
  Polynomial.div (9 * Polynomial.X ^ 4 + 8 * Polynomial.X ^ 3 - 12 * Polynomial.X ^ 2 - 7 * Polynomial.X + 4) (3 * Polynomial.X ^ 2 + 2 * Polynomial.X + 5)

theorem quotient_division :
  poly_division_quotient = (3 * Polynomial.X ^ 2 - 2 * Polynomial.X + 2) :=
sorry

end quotient_division_l231_231433


namespace proof_problem_l231_231594

open Set

def Point : Type := ℝ × ℝ

structure Triangle :=
(A : Point)
(B : Point)
(C : Point)

def area_of_triangle (T : Triangle) : ℝ :=
   0.5 * abs ((T.B.1 - T.A.1) * (T.C.2 - T.A.2) - (T.C.1 - T.A.1) * (T.B.2 - T.A.2))

def area_of_grid (length width : ℝ) : ℝ :=
   length * width

def problem_statement : Prop :=
   let T : Triangle := {A := (1,3), B := (5,1), C := (4,4)} 
   let S1 := area_of_triangle T
   let S := area_of_grid 6 5
   (S1 / S) = 1 / 6

theorem proof_problem : problem_statement := 
by
  sorry


end proof_problem_l231_231594


namespace value_of_f_2019_l231_231483

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ)

-- Assumptions
axiom f_zero : f 0 = 2
axiom f_period : ∀ x : ℝ, f (x + 3) = -f x

-- The property to be proved
theorem value_of_f_2019 : f 2019 = -2 := sorry

end value_of_f_2019_l231_231483


namespace sqrt_x_minus_2_range_l231_231324

theorem sqrt_x_minus_2_range (x : ℝ) : (↑0 ≤ (x - 2)) ↔ (x ≥ 2) := sorry

end sqrt_x_minus_2_range_l231_231324


namespace minimum_value_of_sum_of_squares_l231_231602

theorem minimum_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x - y - 2 * z = 6) : 
  x^2 + y^2 + z^2 ≥ 4 :=
sorry

end minimum_value_of_sum_of_squares_l231_231602


namespace eight_div_repeating_three_l231_231058

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l231_231058


namespace rectangle_area_eq_six_l231_231494

-- Define the areas of the small squares
def smallSquareArea : ℝ := 1

-- Define the number of small squares
def numberOfSmallSquares : ℤ := 2

-- Define the area of the larger square
def largeSquareArea : ℝ := (2 ^ 2)

-- Define the area of rectangle ABCD
def areaRectangleABCD : ℝ :=
  (numberOfSmallSquares * smallSquareArea) + largeSquareArea

-- The theorem we want to prove
theorem rectangle_area_eq_six :
  areaRectangleABCD = 6 := by sorry

end rectangle_area_eq_six_l231_231494


namespace quintic_polynomial_p_l231_231813

theorem quintic_polynomial_p (p q : ℝ) (h : (∀ x : ℝ, x^p + 4*x^3 - q*x^2 - 2*x + 5 = (x^5 + 4*x^3 - q*x^2 - 2*x + 5))) : -p = -5 :=
by {
  sorry
}

end quintic_polynomial_p_l231_231813


namespace statement1_statement2_statement3_statement4_correctness_A_l231_231516

variables {a b : Line} {α β γ : Plane}

def perpendicular (a : Line) (α : Plane) : Prop := sorry
def parallel (a b : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- Statement ①: If a ⊥ α and b ⊥ α, then a ∥ b
theorem statement1 (h1 : perpendicular a α) (h2 : perpendicular b α) : parallel a b := sorry

-- Statement ②: If a ⊥ α, b ⊥ β, and a ∥ b, then α ∥ β
theorem statement2 (h1 : perpendicular a α) (h2 : perpendicular b β) (h3 : parallel a b) : parallel_planes α β := sorry

-- Statement ③: If γ ⊥ α and γ ⊥ β, then α ∥ β
theorem statement3 (h1 : perpendicular γ α) (h2 : perpendicular γ β) : parallel_planes α β := sorry

-- Statement ④: If a ⊥ α and α ⊥ β, then a ∥ β
theorem statement4 (h1 : perpendicular a α) (h2 : parallel_planes α β) : parallel a b := sorry

-- The correct choice is A: Statements ① and ② are correct
theorem correctness_A : statement1_correct ∧ statement2_correct := sorry

end statement1_statement2_statement3_statement4_correctness_A_l231_231516


namespace maximum_M_value_l231_231079

noncomputable def max_value_of_M : ℝ :=
  Real.sqrt 2 + 1 

theorem maximum_M_value {x y z : ℝ} (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x)) ≤ max_value_of_M :=
by
  sorry

end maximum_M_value_l231_231079


namespace quarterback_sacked_times_l231_231680

theorem quarterback_sacked_times
    (total_throws : ℕ)
    (no_pass_percentage : ℚ)
    (half_sacked : ℚ)
    (no_passes : ℕ)
    (sacks : ℕ) :
    total_throws = 80 →
    no_pass_percentage = 0.30 →
    half_sacked = 0.50 →
    no_passes = total_throws * no_pass_percentage →
    sacks = no_passes / 2 →
    sacks = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end quarterback_sacked_times_l231_231680


namespace gcd_324_243_135_l231_231530

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_324_243_135_l231_231530


namespace proof_A_intersection_C_U_B_l231_231801

open Set

-- Given sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Complement of B with respect to U
def C_U_B : Set ℕ := U \ B

-- Prove that the intersection of A and C_U_B is {2, 3}
theorem proof_A_intersection_C_U_B :
  A ∩ C_U_B = {2, 3} := by
  sorry

end proof_A_intersection_C_U_B_l231_231801


namespace num_ways_to_select_3_colors_from_9_l231_231780

def num_ways_select_colors (n k : ℕ) : ℕ := Nat.choose n k

theorem num_ways_to_select_3_colors_from_9 : num_ways_select_colors 9 3 = 84 := by
  sorry

end num_ways_to_select_3_colors_from_9_l231_231780


namespace cost_of_book_first_sold_at_loss_l231_231910

theorem cost_of_book_first_sold_at_loss (C1 C2 C3 : ℝ) (h1 : C1 + C2 + C3 = 810)
    (h2 : 0.88 * C1 = 1.18 * C2) (h3 : 0.88 * C1 = 1.27 * C3) : 
    C1 = 333.9 := 
by
  -- Conditions given
  have h4 : C2 = 0.88 * C1 / 1.18 := by sorry
  have h5 : C3 = 0.88 * C1 / 1.27 := by sorry

  -- Substituting back into the total cost equation
  have h6 : C1 + 0.88 * C1 / 1.18 + 0.88 * C1 / 1.27 = 810 := by sorry

  -- Simplifying and solving for C1
  have h7 : C1 = 333.9 := by sorry

  -- Conclusion
  exact h7

end cost_of_book_first_sold_at_loss_l231_231910


namespace weaving_increase_l231_231278

theorem weaving_increase (a₁ : ℕ) (S₃₀ : ℕ) (d : ℚ) (hₐ₁ : a₁ = 5) (hₛ₃₀ : S₃₀ = 390)
  (h_sum : S₃₀ = 30 * (a₁ + (a₁ + 29 * d)) / 2) : d = 16 / 29 :=
by {
  sorry
}

end weaving_increase_l231_231278


namespace inequality_sqrt_ab_l231_231226

theorem inequality_sqrt_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
sorry

end inequality_sqrt_ab_l231_231226


namespace energy_soda_packs_l231_231484

-- Definitions and conditions
variables (total_bottles : ℕ) (regular_soda : ℕ) (diet_soda : ℕ) (pack_size : ℕ)
variables (complete_packs : ℕ) (remaining_regular : ℕ) (remaining_diet : ℕ) (remaining_energy : ℕ)

-- Conditions given in the problem
axiom h_total_bottles : total_bottles = 200
axiom h_regular_soda : regular_soda = 55
axiom h_diet_soda : diet_soda = 40
axiom h_pack_size : pack_size = 3

-- Proving the correct answer
theorem energy_soda_packs :
  complete_packs = (total_bottles - (regular_soda + diet_soda)) / pack_size ∧
  remaining_regular = regular_soda ∧
  remaining_diet = diet_soda ∧
  remaining_energy = (total_bottles - (regular_soda + diet_soda)) % pack_size :=
by
  sorry

end energy_soda_packs_l231_231484


namespace expression_for_A_div_B_l231_231960

theorem expression_for_A_div_B (x A B : ℝ)
  (h1 : x^3 + 1/x^3 = A)
  (h2 : x - 1/x = B) :
  A / B = B^2 + 3 := 
sorry

end expression_for_A_div_B_l231_231960


namespace stream_speed_l231_231773

theorem stream_speed (x : ℝ) (hb : ∀ t, t = 48 / (20 + x) → t = 24 / (20 - x)) : x = 20 / 3 :=
by
  have t := hb (48 / (20 + x)) rfl
  sorry

end stream_speed_l231_231773


namespace price_of_small_bags_l231_231344

theorem price_of_small_bags (price_medium_bag : ℤ) (price_large_bag : ℤ) 
  (money_mark_has : ℤ) (balloons_in_small_bag : ℤ) 
  (balloons_in_medium_bag : ℤ) (balloons_in_large_bag : ℤ) 
  (total_balloons : ℤ) : 
  price_medium_bag = 6 → 
  price_large_bag = 12 → 
  money_mark_has = 24 → 
  balloons_in_small_bag = 50 → 
  balloons_in_medium_bag = 75 → 
  balloons_in_large_bag = 200 → 
  total_balloons = 400 → 
  (money_mark_has / (total_balloons / balloons_in_small_bag)) = 3 :=
by 
  sorry

end price_of_small_bags_l231_231344


namespace haley_trees_initially_grew_l231_231299

-- Given conditions
def num_trees_died : ℕ := 2
def num_trees_survived : ℕ := num_trees_died + 7

-- Prove the total number of trees initially grown
theorem haley_trees_initially_grew : num_trees_died + num_trees_survived = 11 :=
by
  -- here we would provide the proof eventually
  sorry

end haley_trees_initially_grew_l231_231299


namespace find_a_for_extraneous_roots_find_a_for_no_solution_l231_231875

-- Define the original fractional equation
def eq_fraction (x a: ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Proposition for extraneous roots
theorem find_a_for_extraneous_roots (a: ℝ) (extraneous_roots : ∃ x : ℝ, (x - a) / (x - 2) - 5 / x = 1 ∧ (x = 0 ∨ x = 2)): a = 2 := by 
sorry

-- Proposition for no solution
theorem find_a_for_no_solution (a: ℝ) (no_solution : ∀ x : ℝ, (x - a) / (x - 2) - 5 / x ≠ 1): a = -3 ∨ a = 2 := by 
sorry

end find_a_for_extraneous_roots_find_a_for_no_solution_l231_231875


namespace expected_potato_yield_l231_231888

-- Definitions based on the conditions
def steps_length : ℕ := 3
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def yield_rate : ℚ := 3 / 4

-- Calculate the dimensions in feet
def garden_length_feet : ℕ := garden_length_steps * steps_length
def garden_width_feet : ℕ := garden_width_steps * steps_length

-- Calculate the area in square feet
def garden_area_feet : ℕ := garden_length_feet * garden_width_feet

-- Calculate the expected yield in pounds
def expected_yield_pounds : ℚ := garden_area_feet * yield_rate

-- The theorem to prove the expected yield
theorem expected_potato_yield :
  expected_yield_pounds = 3037.5 := by
  sorry  -- Proof is omitted as per the instructions.

end expected_potato_yield_l231_231888


namespace expression_range_l231_231038

theorem expression_range (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha' : a ≤ 2)
    (hb : 0 ≤ b) (hb' : b ≤ 2)
    (hc : 0 ≤ c) (hc' : c ≤ 2)
    (hd : 0 ≤ d) (hd' : d ≤ 2) :
  4 + 2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) 
  ∧ Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := 
sorry

end expression_range_l231_231038


namespace solution_set_of_inequality_l231_231005

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 1) * (2 - x) > 0} = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l231_231005


namespace smallest_positive_angle_terminal_side_eq_l231_231995

theorem smallest_positive_angle_terminal_side_eq (n : ℤ) :
  (0 ≤ n % 360 ∧ n % 360 < 360) → (∃ k : ℤ, n = -2015 + k * 360 ) → n % 360 = 145 :=
by
  sorry

end smallest_positive_angle_terminal_side_eq_l231_231995


namespace final_price_l231_231697

def initial_price : ℝ := 200
def discount_morning : ℝ := 0.40
def increase_noon : ℝ := 0.25
def discount_afternoon : ℝ := 0.20

theorem final_price : 
  let price_after_morning := initial_price * (1 - discount_morning)
  let price_after_noon := price_after_morning * (1 + increase_noon)
  let final_price := price_after_noon * (1 - discount_afternoon)
  final_price = 120 := 
by
  sorry

end final_price_l231_231697


namespace power_of_binomials_l231_231907

theorem power_of_binomials :
  (1 + Real.sqrt 2) ^ 2023 * (1 - Real.sqrt 2) ^ 2023 = -1 :=
by
  -- This is a placeholder for the actual proof steps.
  -- We use 'sorry' to indicate that the proof is omitted here.
  sorry

end power_of_binomials_l231_231907


namespace pigeon_count_correct_l231_231940

def initial_pigeon_count : ℕ := 1
def new_pigeon_count : ℕ := 1
def total_pigeon_count : ℕ := 2

theorem pigeon_count_correct : initial_pigeon_count + new_pigeon_count = total_pigeon_count :=
by
  sorry

end pigeon_count_correct_l231_231940


namespace range_of_a_l231_231860

theorem range_of_a (a : ℝ) :
  (¬ ∀ x : ℝ, x^2 + 2*x + a > 0) ↔ a ≤ 1 :=
sorry

end range_of_a_l231_231860


namespace geometric_series_sum_l231_231973

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  S = 341 / 1024 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  show S = 341 / 1024
  sorry

end geometric_series_sum_l231_231973


namespace cars_in_north_america_correct_l231_231361

def total_cars_produced : ℕ := 6755
def cars_produced_in_europe : ℕ := 2871

def cars_produced_in_north_america : ℕ := total_cars_produced - cars_produced_in_europe

theorem cars_in_north_america_correct : cars_produced_in_north_america = 3884 :=
by sorry

end cars_in_north_america_correct_l231_231361


namespace volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l231_231017

noncomputable def volume_of_reservoir (drain_rate : ℝ) (time_to_drain : ℝ) : ℝ :=
  drain_rate * time_to_drain

theorem volume_of_reservoir_proof :
  volume_of_reservoir 8 6 = 48 :=
by
  sorry

noncomputable def relationship_Q_t (volume : ℝ) (t : ℝ) : ℝ :=
  volume / t

theorem relationship_Q_t_proof :
  ∀ (t : ℝ), relationship_Q_t 48 t = 48 / t :=
by
  intro t
  sorry

noncomputable def min_hourly_drainage (volume : ℝ) (time : ℝ) : ℝ :=
  volume / time

theorem min_hourly_drainage_proof :
  min_hourly_drainage 48 5 = 9.6 :=
by
  sorry

theorem min_time_to_drain_proof :
  ∀ (max_capacity : ℝ), relationship_Q_t 48 max_capacity = 12 → 48 / 12 = 4 :=
by
  intro max_capacity h
  sorry

end volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l231_231017


namespace exponent_multiplication_l231_231105

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l231_231105


namespace axisymmetric_and_centrally_symmetric_l231_231987

def Polygon := String

def EquilateralTriangle : Polygon := "EquilateralTriangle"
def Square : Polygon := "Square"
def RegularPentagon : Polygon := "RegularPentagon"
def RegularHexagon : Polygon := "RegularHexagon"

def is_axisymmetric (p : Polygon) : Prop := 
  p = EquilateralTriangle ∨ p = Square ∨ p = RegularPentagon ∨ p = RegularHexagon

def is_centrally_symmetric (p : Polygon) : Prop := 
  p = Square ∨ p = RegularHexagon

theorem axisymmetric_and_centrally_symmetric :
  {p : Polygon | is_axisymmetric p ∧ is_centrally_symmetric p} = {Square, RegularHexagon} :=
by
  sorry

end axisymmetric_and_centrally_symmetric_l231_231987


namespace cos_shifted_alpha_l231_231869

theorem cos_shifted_alpha (α : ℝ) (h1 : Real.tan α = -3/4) (h2 : α ∈ Set.Ioc (3*Real.pi/2) (2*Real.pi)) :
  Real.cos (Real.pi/2 + α) = 3/5 :=
sorry

end cos_shifted_alpha_l231_231869


namespace average_of_distinct_u_l231_231247

theorem average_of_distinct_u :
  let u_values := { u : ℕ | ∃ (r_1 r_2 : ℕ), r_1 + r_2 = 6 ∧ r_1 * r_2 = u }
  u_values = {5, 8, 9} ∧ (5 + 8 + 9) / 3 = 22 / 3 :=
by
  sorry

end average_of_distinct_u_l231_231247


namespace number_of_elements_in_M_l231_231480

def positive_nats : Set ℕ := {n | n > 0}
def M : Set ℕ := {m | ∃ n ∈ positive_nats, m = 2 * n - 1 ∧ m < 60}

theorem number_of_elements_in_M : ∃ s : Finset ℕ, (∀ x, x ∈ s ↔ x ∈ M) ∧ s.card = 30 := 
by
  sorry

end number_of_elements_in_M_l231_231480


namespace circle_radius_inscribed_l231_231286

noncomputable def a : ℝ := 6
noncomputable def b : ℝ := 12
noncomputable def c : ℝ := 18

noncomputable def r : ℝ :=
  let term1 := 1/a
  let term2 := 1/b
  let term3 := 1/c
  let sqrt_term := Real.sqrt ((1/(a * b)) + (1/(a * c)) + (1/(b * c)))
  1 / ((term1 + term2 + term3) + 2 * sqrt_term)

theorem circle_radius_inscribed :
  r = 36 / 17 := 
by
  sorry

end circle_radius_inscribed_l231_231286


namespace beyonce_total_songs_l231_231338

theorem beyonce_total_songs (s a b t : ℕ) (h_s : s = 5) (h_a : a = 2 * 15) (h_b : b = 20) (h_t : t = s + a + b) : t = 55 := by
  rw [h_s, h_a, h_b] at h_t
  exact h_t

end beyonce_total_songs_l231_231338


namespace harry_lost_sea_creatures_l231_231181

def initial_sea_stars := 34
def initial_seashells := 21
def initial_snails := 29
def initial_crabs := 17

def sea_stars_reproduced := 5
def seashells_reproduced := 3
def snails_reproduced := 4

def final_items := 105

def sea_stars_after_reproduction := initial_sea_stars + (sea_stars_reproduced * 2 - sea_stars_reproduced)
def seashells_after_reproduction := initial_seashells + (seashells_reproduced * 2 - seashells_reproduced)
def snails_after_reproduction := initial_snails + (snails_reproduced * 2 - snails_reproduced)
def crabs_after_reproduction := initial_crabs

def total_after_reproduction := sea_stars_after_reproduction + seashells_after_reproduction + snails_after_reproduction + crabs_after_reproduction

theorem harry_lost_sea_creatures : total_after_reproduction - final_items = 8 :=
by
  sorry

end harry_lost_sea_creatures_l231_231181


namespace find_b_l231_231994

theorem find_b (b x : ℝ) (h₁ : 5 * x + 3 = b * x - 22) (h₂ : x = 5) : b = 10 := 
by 
  sorry

end find_b_l231_231994


namespace percentage_increase_on_bought_price_l231_231096

-- Define the conditions as Lean definitions
def original_price (P : ℝ) : ℝ := P
def bought_price (P : ℝ) : ℝ := 0.90 * P
def selling_price (P : ℝ) : ℝ := 1.62000000000000014 * P

-- Lean statement to prove the required result
theorem percentage_increase_on_bought_price (P : ℝ) :
  (selling_price P - bought_price P) / bought_price P * 100 = 80.00000000000002 := by
  sorry

end percentage_increase_on_bought_price_l231_231096


namespace length_of_integer_eq_24_l231_231553

theorem length_of_integer_eq_24 (k : ℕ) (h1 : k > 1) (h2 : ∃ (p1 p2 p3 p4 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ k = p1 * p2 * p3 * p4) : k = 24 := by
  sorry

end length_of_integer_eq_24_l231_231553


namespace equal_split_l231_231406

theorem equal_split (A B C : ℝ) (h1 : A < B) (h2 : B < C) : 
  (B + C - 2 * A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end equal_split_l231_231406


namespace negate_proposition_l231_231182

theorem negate_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^2 + 2 > 6)) ↔ (∃ x : ℝ, x > 2 ∧ x^2 + 2 ≤ 6) :=
by sorry

end negate_proposition_l231_231182


namespace find_sets_l231_231036

open Set

noncomputable def U := ℝ
def A := {x : ℝ | Real.log x / Real.log 2 <= 2}
def B := {x : ℝ | x ≥ 1}

theorem find_sets (x : ℝ) :
  (A = {x : ℝ | -1 ≤ x ∧ x < 3}) ∧
  (B = {x : ℝ | -2 < x ∧ x ≤ 3}) ∧
  (compl A ∩ B = {x : ℝ | (-2 < x ∧ x < -1) ∨ x = 3}) :=
  sorry

end find_sets_l231_231036


namespace salem_size_comparison_l231_231379

theorem salem_size_comparison (S L : ℕ) (hL: L = 58940)
  (hSalem: S - 130000 = 2 * 377050) :
  (S / L = 15) :=
sorry

end salem_size_comparison_l231_231379


namespace b_c_value_l231_231207

theorem b_c_value (a b c d : ℕ) 
  (h₁ : a + b = 12) 
  (h₂ : c + d = 3) 
  (h₃ : a + d = 6) : 
  b + c = 9 :=
sorry

end b_c_value_l231_231207


namespace houses_before_boom_l231_231092

theorem houses_before_boom (current_houses built_during_boom houses_before : ℕ) 
  (h1 : current_houses = 2000)
  (h2 : built_during_boom = 574)
  (h3 : current_houses = houses_before + built_during_boom) : 
  houses_before = 1426 := 
by
  -- Proof omitted
  sorry

end houses_before_boom_l231_231092


namespace algebraic_expression_value_l231_231578

theorem algebraic_expression_value 
  (x y : ℝ) 
  (h : 2 * x + y = 1) : 
  (y + 1) ^ 2 - (y ^ 2 - 4 * x + 4) = -1 := 
by 
  sorry

end algebraic_expression_value_l231_231578


namespace polar_distance_to_axis_l231_231637

theorem polar_distance_to_axis (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = Real.pi / 6) : 
  ρ * Real.sin θ = 1 := 
by
  rw [hρ, hθ]
  -- The remaining proof steps would go here
  sorry

end polar_distance_to_axis_l231_231637


namespace Suresh_completes_job_in_15_hours_l231_231468

theorem Suresh_completes_job_in_15_hours :
  ∃ S : ℝ,
    (∀ (T_A Ashutosh_time Suresh_time : ℝ), Ashutosh_time = 15 ∧ Suresh_time = 9 
    → T_A = Ashutosh_time → 6 / T_A + Suresh_time / S = 1) ∧ S = 15 :=
by
  sorry

end Suresh_completes_job_in_15_hours_l231_231468


namespace sum_of_numbers_is_919_l231_231646

-- Problem Conditions
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def is_three_digit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999
def satisfies_equation (x y : ℕ) : Prop := 1000 * x + y = 11 * x * y

-- Main Statement
theorem sum_of_numbers_is_919 (x y : ℕ) 
  (h1 : is_two_digit x) 
  (h2 : is_three_digit y) 
  (h3 : satisfies_equation x y) : 
  x + y = 919 := 
sorry

end sum_of_numbers_is_919_l231_231646


namespace greater_number_is_25_l231_231686

theorem greater_number_is_25 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
sorry

end greater_number_is_25_l231_231686


namespace original_difference_of_weights_l231_231679

variable (F S T : ℝ)

theorem original_difference_of_weights :
  (F + S + T = 75) →
  (F - 2 = 0.7 * (S + 2)) →
  (S + 1 = 0.8 * (T + 1)) →
  T - F = 10.16 :=
by
  intro h1 h2 h3
  sorry

end original_difference_of_weights_l231_231679


namespace solve_for_x_l231_231746

-- Define the operation *
def op (a b : ℝ) : ℝ := 2 * a - b

-- The theorem statement
theorem solve_for_x :
  (∃ x : ℝ, op x (op 1 3) = 2) ∧ (∀ x, op x -1 = 2)
  → x = 1/2 := by
  sorry

end solve_for_x_l231_231746


namespace distance_between_locations_A_and_B_l231_231500

theorem distance_between_locations_A_and_B 
  (speed_A speed_B speed_C : ℝ)
  (distance_CD : ℝ)
  (distance_initial_A : ℝ)
  (distance_A_to_B : ℝ)
  (h1 : speed_A = 3 * speed_C)
  (h2 : speed_A = 1.5 * speed_B)
  (h3 : distance_CD = 12)
  (h4 : distance_initial_A = 50)
  (h5 : distance_A_to_B = 130)
  : distance_A_to_B = 130 :=
by
  sorry

end distance_between_locations_A_and_B_l231_231500


namespace least_clock_equivalent_l231_231341

theorem least_clock_equivalent (t : ℕ) (h : t > 5) : 
  (t^2 - t) % 24 = 0 → t = 9 :=
by
  sorry

end least_clock_equivalent_l231_231341


namespace solve_cubic_equation_l231_231127

theorem solve_cubic_equation (x y z : ℤ) (h : x^3 - 3*y^3 - 9*z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end solve_cubic_equation_l231_231127


namespace range_of_m_l231_231400

noncomputable def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
noncomputable def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, P x → Q x m ∧ P x) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end range_of_m_l231_231400


namespace average_weight_when_D_joins_is_53_l231_231791

noncomputable def new_average_weight (A B C D E : ℕ) : ℕ :=
  (73 + B + C + D) / 4

theorem average_weight_when_D_joins_is_53 :
  (A + B + C) / 3 = 50 →
  A = 73 →
  (B + C + D + E) / 4 = 51 →
  E = D + 3 →
  73 + B + C + D = 212 →
  new_average_weight A B C D E = 53 :=
by
  sorry

end average_weight_when_D_joins_is_53_l231_231791


namespace problem_conditions_equation_right_triangle_vertex_coordinates_l231_231662

theorem problem_conditions_equation : 
  ∃ (a b c : ℝ), a = -1 ∧ b = -2 ∧ c = 3 ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - (-(x + 1))^2 + 4) ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - x^2 - 2 * x + 3)
:= sorry

theorem right_triangle_vertex_coordinates :
  ∀ x y : ℝ, x = -1 ∧ 
  (y = -2 ∨ y = 4 ∨ y = (3 + (17:ℝ).sqrt) / 2 ∨ y = (3 - (17:ℝ).sqrt) / 2)
  ∧ 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (-3, 0)
  let C : ℝ × ℝ := (0, 3)
  let P : ℝ × ℝ := (x, y)
  let BC : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2
  let PB : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
  let PC : ℝ := (P.1 - C.1)^2 + (P.2 - C.2)^2
  (BC + PB = PC ∨ BC + PC = PB ∨ PB + PC = BC)
:= sorry

end problem_conditions_equation_right_triangle_vertex_coordinates_l231_231662


namespace intersection_of_sets_l231_231901

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}

def B : Set ℝ := Ico 0 4  -- Ico stands for interval [0, 4)

theorem intersection_of_sets : A ∩ B = Ico 2 4 :=
by 
  sorry

end intersection_of_sets_l231_231901


namespace least_three_digit_eleven_heavy_l231_231133

def isElevenHeavy (n : ℕ) : Prop :=
  n % 11 > 6

theorem least_three_digit_eleven_heavy : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ isElevenHeavy n ∧ ∀ m : ℕ, (m >= 100 ∧ m < 1000 ∧ isElevenHeavy m) → n ≤ m :=
sorry

end least_three_digit_eleven_heavy_l231_231133


namespace min_value_of_exponential_l231_231081

theorem min_value_of_exponential (x y : ℝ) (h : x + 2 * y = 1) : 
  2^x + 4^y ≥ 2 * Real.sqrt 2 ∧ 
  (∀ a, (2^x + 4^y = a) → a ≥ 2 * Real.sqrt 2) :=
by
  sorry

end min_value_of_exponential_l231_231081


namespace parabola_properties_l231_231082

theorem parabola_properties :
  ∀ x : ℝ, (x - 3)^2 + 5 = (x-3)^2 + 5 ∧ 
  (x - 3)^2 + 5 > 0 ∧ 
  (∃ h : ℝ, h = 3 ∧ ∀ x1 x2 : ℝ, (x1 - h)^2 <= (x2 - h)^2) ∧ 
  (∃ h k : ℝ, h = 3 ∧ k = 5) := 
by 
  sorry

end parabola_properties_l231_231082


namespace cookie_ratio_l231_231084

theorem cookie_ratio (cookies_monday cookies_tuesday cookies_wednesday final_cookies : ℕ)
  (h1 : cookies_monday = 32)
  (h2 : cookies_tuesday = cookies_monday / 2)
  (h3 : final_cookies = 92)
  (h4 : cookies_wednesday = final_cookies + 4 - cookies_monday - cookies_tuesday) :
  cookies_wednesday / cookies_tuesday = 3 :=
by
  sorry

end cookie_ratio_l231_231084


namespace second_divisor_l231_231337

theorem second_divisor (N k D m : ℤ) (h1 : N = 35 * k + 25) (h2 : N = D * m + 4) : D = 17 := by
  -- Follow conditions from problem
  sorry

end second_divisor_l231_231337


namespace problem_part_a_problem_part_b_l231_231488

def is_two_squared (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2 ∧ a ≠ 0 ∧ b ≠ 0

def is_three_squared (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = a^2 + b^2 + c^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def is_four_squared (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a^2 + b^2 + c^2 + d^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def satisfies_prime_conditions (e : ℕ) : Prop :=
  Nat.Prime (e - 2) ∧ Nat.Prime e ∧ Nat.Prime (e + 4)

def satisfies_square_sum_conditions (a b c d e : ℕ) : Prop :=
  a^2 + b^2 + c^2 + d^2 + e^2 = 2020 ∧ a < b ∧ b < c ∧ c < d ∧ d < e

theorem problem_part_a : is_two_squared 2020 ∧ is_three_squared 2020 ∧ is_four_squared 2020 := sorry

theorem problem_part_b : ∃ a b c d e : ℕ, satisfies_prime_conditions e ∧ satisfies_square_sum_conditions a b c d e :=
  sorry

end problem_part_a_problem_part_b_l231_231488


namespace circle_equation_l231_231039

theorem circle_equation
  (a b r : ℝ)
  (ha : (4 - a)^2 + (1 - b)^2 = r^2)
  (hb : (2 - a)^2 + (1 - b)^2 = r^2)
  (ht : (b - 1) / (a - 2) = -1) :
  (a = 3) ∧ (b = 0) ∧ (r = 2) :=
by {
  sorry
}

-- Given the above values for a, b, r
def circle_equation_verified : Prop :=
  (∀ (x y : ℝ), ((x - 3)^2 + y^2) = 4)

example : circle_equation_verified :=
by {
  sorry
}

end circle_equation_l231_231039


namespace distinct_real_roots_c_l231_231336

theorem distinct_real_roots_c (c : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + c = 0 ∧ x₂^2 - 4*x₂ + c = 0) ↔ c < 4 := by
  sorry

end distinct_real_roots_c_l231_231336


namespace missing_digit_is_4_l231_231694

theorem missing_digit_is_4 (x : ℕ) (hx : 7385 = 7380 + x + 5)
  (hdiv : (7 + 3 + 8 + x + 5) % 9 = 0) : x = 4 :=
by
  sorry

end missing_digit_is_4_l231_231694


namespace equation_has_unique_integer_solution_l231_231865

theorem equation_has_unique_integer_solution:
  ∀ m n : ℤ, (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n → m = 0 ∧ n = 0 := by
  intro m n
  -- The proof is omitted
  sorry

end equation_has_unique_integer_solution_l231_231865


namespace multiple_solutions_no_solution_2891_l231_231121

theorem multiple_solutions (n : ℤ) (x y : ℤ) (h1 : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (u v : ℤ), u ≠ x ∧ v ≠ y ∧ u^3 - 3 * u * v^2 + v^3 = n :=
  sorry

theorem no_solution_2891 (x y : ℤ) (h2 : x^3 - 3 * x * y^2 + y^3 = 2891) :
  false :=
  sorry

end multiple_solutions_no_solution_2891_l231_231121


namespace inequality_of_function_inequality_l231_231144

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + Real.sqrt (x^2 + 1))) + 2 * x + Real.sin x

theorem inequality_of_function_inequality (x1 x2 : ℝ) (h : f x1 + f x2 > 0) : x1 + x2 > 0 :=
sorry

end inequality_of_function_inequality_l231_231144


namespace ferris_wheel_seats_l231_231579

-- Define the total number of seats S as a variable
variables (S : ℕ)

-- Define the conditions
def seat_capacity : ℕ := 15

def broken_seats : ℕ := 10

def max_riders : ℕ := 120

-- The theorem statement
theorem ferris_wheel_seats :
  ((S - broken_seats) * seat_capacity = max_riders) → S = 18 :=
by
  sorry

end ferris_wheel_seats_l231_231579


namespace min_value_of_m_l231_231424

theorem min_value_of_m (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + b * c + c * a = -1) (h3 : a * b * c = -m) : 
    m = - (min (-a ^ 3 + a ^ 2 + a ) (- (1 / 27))) := 
sorry

end min_value_of_m_l231_231424


namespace plant_height_increase_l231_231953

theorem plant_height_increase (total_increase : ℕ) (century_in_years : ℕ) (decade_in_years : ℕ) (years_in_2_centuries : ℕ) (num_decades : ℕ) : 
  total_increase = 1800 →
  century_in_years = 100 →
  decade_in_years = 10 →
  years_in_2_centuries = 2 * century_in_years →
  num_decades = years_in_2_centuries / decade_in_years →
  total_increase / num_decades = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end plant_height_increase_l231_231953


namespace ellipse_solution_l231_231852

theorem ellipse_solution :
  (∃ (a b : ℝ), a = 4 * Real.sqrt 2 + Real.sqrt 17 ∧ b = Real.sqrt (32 + 16 * Real.sqrt 34) ∧ (∀ (x y : ℝ), (3 * 0 ≤ y ∧ y ≤ 8) → (3 * 0 ≤ x ∧ x ≤ 5) → (Real.sqrt ((x+3)^2 + y^2) + Real.sqrt ((x-3)^2 + y^2) = 2 * a) → 
   (Real.sqrt ((x-0)^2 + (y-8)^2) = b))) :=
sorry

end ellipse_solution_l231_231852


namespace number_of_players_sold_eq_2_l231_231763

def initial_balance : ℕ := 100
def selling_price_per_player : ℕ := 10
def buying_cost_per_player : ℕ := 15
def number_of_players_bought : ℕ := 4
def final_balance : ℕ := 60

theorem number_of_players_sold_eq_2 :
  ∃ x : ℕ, (initial_balance + selling_price_per_player * x - buying_cost_per_player * number_of_players_bought = final_balance) ∧ (x = 2) :=
by
  sorry

end number_of_players_sold_eq_2_l231_231763


namespace parallelogram_area_72_l231_231261

def parallelogram_area (base height : ℕ) : ℕ :=
  base * height

theorem parallelogram_area_72 :
  parallelogram_area 12 6 = 72 :=
by
  sorry

end parallelogram_area_72_l231_231261


namespace choose_integers_l231_231113

def smallest_prime_divisor (n : ℕ) : ℕ := sorry
def number_of_divisors (n : ℕ) : ℕ := sorry

theorem choose_integers :
  ∃ (a : ℕ → ℕ), (∀ i, i < 2022 → a i < a (i + 1)) ∧
  (∀ k, 1 ≤ k ∧ k ≤ 2022 →
    number_of_divisors (a (k + 1) - a k - 1) > 2023^k ∧
    smallest_prime_divisor (a (k + 1) - a k) > 2023^k
  ) :=
sorry

end choose_integers_l231_231113


namespace domain_function_1_domain_function_2_domain_function_3_l231_231739

-- Define the conditions and the required domain equivalence in Lean 4
-- Problem (1)
theorem domain_function_1 (x : ℝ): x + 2 ≠ 0 ∧ x + 5 ≥ 0 ↔ x ≥ -5 ∧ x ≠ -2 := 
sorry

-- Problem (2)
theorem domain_function_2 (x : ℝ): x^2 - 4 ≥ 0 ∧ 4 - x^2 ≥ 0 ∧ x^2 - 9 ≠ 0 ↔ (x = 2 ∨ x = -2) :=
sorry

-- Problem (3)
theorem domain_function_3 (x : ℝ): x - 5 ≥ 0 ∧ |x| ≠ 7 ↔ x ≥ 5 ∧ x ≠ 7 :=
sorry

end domain_function_1_domain_function_2_domain_function_3_l231_231739


namespace pirates_treasure_l231_231393

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l231_231393


namespace intersection_first_quadrant_l231_231807

theorem intersection_first_quadrant (a : ℝ) : 
  (∃ x y : ℝ, (ax + y = 4) ∧ (x - y = 2) ∧ (0 < x) ∧ (0 < y)) ↔ (-1 < a ∧ a < 2) :=
by
  sorry

end intersection_first_quadrant_l231_231807


namespace sum_geometric_series_l231_231418

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l231_231418


namespace trapezoid_shorter_base_l231_231269

theorem trapezoid_shorter_base (L : ℝ) (S : ℝ) (m : ℝ)
  (hL : L = 100)
  (hm : m = 4)
  (h : m = (L - S) / 2) :
  S = 92 :=
by {
  sorry -- Proof is not required
}

end trapezoid_shorter_base_l231_231269


namespace problem1_solution_l231_231806

theorem problem1_solution (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 16)
  (h2 : 5 * x - 6 * y = 33) : 
  x = 6 ∧ y = -1 / 2 := 
  by
  sorry

end problem1_solution_l231_231806


namespace seq_1000_eq_2098_l231_231754

-- Define the sequence a_n
def seq (n : ℕ) : ℤ := sorry

-- Initial conditions
axiom a1 : seq 1 = 100
axiom a2 : seq 2 = 101

-- Recurrence relation condition
axiom recurrence_relation : ∀ n : ℕ, 1 ≤ n → seq n + seq (n+1) + seq (n+2) = 2 * ↑n + 3

-- Main theorem to prove
theorem seq_1000_eq_2098 : seq 1000 = 2098 :=
by {
  sorry
}

end seq_1000_eq_2098_l231_231754


namespace total_skips_l231_231989

-- Definitions based on conditions
def fifth_throw := 8
def fourth_throw := fifth_throw - 1
def third_throw := fourth_throw + 3
def second_throw := third_throw / 2
def first_throw := second_throw - 2

-- Statement of the proof problem
theorem total_skips : first_throw + second_throw + third_throw + fourth_throw + fifth_throw = 33 := by
  sorry

end total_skips_l231_231989


namespace transformation_correct_l231_231465

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

-- Define the transformation functions
noncomputable def shift_right_by_pi_over_10 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - Real.pi / 10)
noncomputable def stretch_x_by_factor_of_2 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x / 2)

-- Define the transformed function
noncomputable def transformed_function : ℝ → ℝ :=
  stretch_x_by_factor_of_2 (shift_right_by_pi_over_10 original_function)

-- Define the expected resulting function
noncomputable def expected_function (x : ℝ) : ℝ := Real.sin (x / 2 - Real.pi / 10)

-- State the theorem
theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = expected_function x :=
by
  sorry

end transformation_correct_l231_231465


namespace range_of_a_for_quadratic_inequality_l231_231829

theorem range_of_a_for_quadratic_inequality (a : ℝ) :
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1 > 0) →
  (a ≤ -2 ∨ a ≥ 6) :=
by
  sorry

end range_of_a_for_quadratic_inequality_l231_231829


namespace cos_30_eq_sqrt3_div_2_l231_231197

theorem cos_30_eq_sqrt3_div_2 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l231_231197


namespace tan_pi_minus_alpha_l231_231638

theorem tan_pi_minus_alpha (α : ℝ) (h : Real.tan (Real.pi - α) = -2) : 
  (1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5 / 2) :=
by
  sorry

end tan_pi_minus_alpha_l231_231638


namespace find_principal_amount_l231_231455

theorem find_principal_amount
  (P R T SI : ℝ) 
  (rate_condition : R = 12)
  (time_condition : T = 20)
  (interest_condition : SI = 2100) :
  SI = (P * R * T) / 100 → P = 875 :=
by
  sorry

end find_principal_amount_l231_231455


namespace calculate_expression_l231_231309

theorem calculate_expression : (7^2 - 5^2)^3 = 13824 := by
  sorry

end calculate_expression_l231_231309


namespace max_viewing_area_l231_231894

theorem max_viewing_area (L W: ℝ) (h1: 2 * L + 2 * W = 420) (h2: L ≥ 100) (h3: W ≥ 60) : 
  (L = 105) ∧ (W = 105) ∧ (L * W = 11025) :=
by
  sorry

end max_viewing_area_l231_231894


namespace geraldo_drank_7_pints_l231_231172

-- Conditions
def total_gallons : ℝ := 20
def num_containers : ℕ := 80
def gallons_to_pints : ℝ := 8
def containers_drank : ℝ := 3.5

-- Problem statement
theorem geraldo_drank_7_pints :
  let total_pints : ℝ := total_gallons * gallons_to_pints
  let pints_per_container : ℝ := total_pints / num_containers
  let pints_drank : ℝ := containers_drank * pints_per_container
  pints_drank = 7 :=
by
  sorry

end geraldo_drank_7_pints_l231_231172


namespace probability_red_on_other_side_l231_231799

def num_black_black_cards := 4
def num_black_red_cards := 2
def num_red_red_cards := 2

def num_red_sides_total := 
  num_black_black_cards * 0 +
  num_black_red_cards * 1 +
  num_red_red_cards * 2

def num_red_sides_with_red_on_other_side := 
  num_red_red_cards * 2

theorem probability_red_on_other_side :
  (num_red_sides_with_red_on_other_side : ℚ) / num_red_sides_total = 2 / 3 := by
  sorry

end probability_red_on_other_side_l231_231799


namespace problem_proof_equality_cases_l231_231548

theorem problem_proof (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (x * y - 10) ^ 2 ≥ 64 := sorry

theorem equality_cases (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : 
  (x * y - 10) ^ 2 = 64 ↔ ((x,y) = (1, 2) ∨ (x,y) = (-3, -6)) := sorry

end problem_proof_equality_cases_l231_231548


namespace sum_of_intercepts_l231_231000

theorem sum_of_intercepts (x y : ℝ) 
  (h_eq : y - 3 = -3 * (x - 5)) 
  (hx_intercept : y = 0 ∧ x = 6) 
  (hy_intercept : x = 0 ∧ y = 18) : 
  6 + 18 = 24 :=
by
  sorry

end sum_of_intercepts_l231_231000


namespace crayon_ratio_l231_231114

theorem crayon_ratio :
  ∀ (Karen Beatrice Gilbert Judah : ℕ),
    Karen = 128 →
    Beatrice = Karen / 2 →
    Beatrice = Gilbert →
    Gilbert = 4 * Judah →
    Judah = 8 →
    Beatrice / Gilbert = 1 :=
by
  intros Karen Beatrice Gilbert Judah hKaren hBeatrice hEqual hGilbert hJudah
  sorry

end crayon_ratio_l231_231114


namespace inequality_proof_l231_231635

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  a ^ a * b ^ b * c ^ c ≥ 1 / (a * b * c) := 
sorry

end inequality_proof_l231_231635


namespace solve_linear_combination_l231_231812

theorem solve_linear_combination (x y z : ℤ) 
    (h1 : x + 2 * y - z = 8) 
    (h2 : 2 * x - y + z = 18) : 
    8 * x + y + z = 70 := 
by 
    sorry

end solve_linear_combination_l231_231812


namespace solve_system_eq_l231_231429

theorem solve_system_eq (x y : ℝ) :
  x^2 + y^2 + 6 * x * y = 68 ∧ 2 * x^2 + 2 * y^2 - 3 * x * y = 16 ↔
  (x = 4 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ (x = -4 ∧ y = -2) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end solve_system_eq_l231_231429


namespace average_salary_l231_231567

theorem average_salary (T_salary : ℕ) (R_salary : ℕ) (total_salary : ℕ) (T_count : ℕ) (R_count : ℕ) (total_count : ℕ) :
    T_salary = 12000 * T_count →
    R_salary = 6000 * R_count →
    total_salary = T_salary + R_salary →
    T_count = 6 →
    R_count = total_count - T_count →
    total_count = 18 →
    (total_salary / total_count) = 8000 :=
by
  intros
  sorry

end average_salary_l231_231567


namespace sum_of_interior_angles_eq_1440_l231_231674

theorem sum_of_interior_angles_eq_1440 (h : ∀ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ)) : 
    (∃ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ) ∧ (n - 2) * 180 = 1440) :=
by
  sorry

end sum_of_interior_angles_eq_1440_l231_231674


namespace thirty_one_star_thirty_two_l231_231482

def complex_op (x y : ℝ) : ℝ :=
sorry

axiom op_zero (x : ℝ) : complex_op x 0 = 1

axiom op_associative (x y z : ℝ) : complex_op (complex_op x y) z = z * (x * y) + z

theorem thirty_one_star_thirty_two : complex_op 31 32 = 993 :=
by
  sorry

end thirty_one_star_thirty_two_l231_231482


namespace largest_consecutive_odd_integer_sum_l231_231524

theorem largest_consecutive_odd_integer_sum
  (x : Real)
  (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = -378.5) :
  x + 8 = -79.7 + 8 :=
by
  sorry

end largest_consecutive_odd_integer_sum_l231_231524


namespace arrangement_count_is_43200_l231_231538

noncomputable def number_of_arrangements : Nat :=
  let number_of_boys := 6
  let number_of_girls := 3
  let boys_arrangements := Nat.factorial number_of_boys
  let spaces := number_of_boys - 1
  let girls_arrangements := Nat.factorial (spaces) / Nat.factorial (spaces - number_of_girls)
  boys_arrangements * girls_arrangements

theorem arrangement_count_is_43200 :
  number_of_arrangements = 43200 := by
  sorry

end arrangement_count_is_43200_l231_231538


namespace portion_left_l231_231782

theorem portion_left (john_portion emma_portion final_portion : ℝ) (H1 : john_portion = 0.6) (H2 : emma_portion = 0.5 * (1 - john_portion)) :
  final_portion = 1 - john_portion - emma_portion :=
by
  sorry

end portion_left_l231_231782


namespace longest_boat_length_l231_231857

-- Definitions of the conditions
def total_savings : ℤ := 20000
def cost_per_foot : ℤ := 1500
def license_registration : ℤ := 500
def docking_fees := 3 * license_registration

-- Calculate the reserved amount for license, registration, and docking fees
def reserved_amount := license_registration + docking_fees

-- Calculate the amount left for the boat
def amount_left := total_savings - reserved_amount

-- Calculate the maximum length of the boat Mitch can afford
def max_boat_length := amount_left / cost_per_foot

-- Theorem to prove the longest boat Mitch can buy
theorem longest_boat_length : max_boat_length = 12 :=
by
  sorry

end longest_boat_length_l231_231857


namespace probability_sector_F_l231_231019

theorem probability_sector_F (prob_D prob_E prob_F : ℚ)
    (hD : prob_D = 1/4) 
    (hE : prob_E = 1/3) 
    (hSum : prob_D + prob_E + prob_F = 1) :
    prob_F = 5/12 := by
  sorry

end probability_sector_F_l231_231019


namespace find_all_x_satisfying_condition_l231_231388

theorem find_all_x_satisfying_condition :
  ∃ (x : Fin 2016 → ℝ), 
  (∀ i : Fin 2016, x (i + 1) % 2016 = x 0) ∧
  (∀ i : Fin 2016, x i ^ 2 + x i - 1 = x ((i + 1) % 2016)) ∧
  (∀ i : Fin 2016, x i = 1 ∨ x i = -1) :=
sorry

end find_all_x_satisfying_condition_l231_231388


namespace find_perimeter_correct_l231_231718

noncomputable def find_perimeter (L W : ℝ) (x : ℝ) :=
  L * W = (L + 6) * (W - 2) ∧
  L * W = (L - 12) * (W + 6) ∧
  x = 2 * (L + W)

theorem find_perimeter_correct : ∀ (L W : ℝ), L * W = (L + 6) * (W - 2) → 
                                      L * W = (L - 12) * (W + 6) → 
                                      2 * (L + W) = 132 :=
sorry

end find_perimeter_correct_l231_231718


namespace find_f_2_l231_231187

variable (f : ℤ → ℤ)

-- Definitions of the conditions
def is_monic_quartic (f : ℤ → ℤ) : Prop :=
  ∃ a b c d, ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d

variable (hf_monic : is_monic_quartic f)
variable (hf_conditions : f (-2) = -4 ∧ f 1 = -1 ∧ f 3 = -9 ∧ f (-4) = -16)

-- The main statement to prove
theorem find_f_2 : f 2 = -28 := sorry

end find_f_2_l231_231187


namespace find_y_values_l231_231430

def A (y : ℝ) : ℝ := 1 - y - 2 * y^2

theorem find_y_values (y : ℝ) (h₁ : y ≤ 1) (h₂ : y ≠ 0) (h₃ : y ≠ -1) (h₄ : y ≠ 0.5) :
  y^2 * A y / (y * A y) ≤ 1 ↔
  y ∈ Set.Iio (-1) ∪ Set.Ioo (-1) (1/2) ∪ Set.Ioc (1/2) 1 :=
by
  -- proof is omitted
  sorry

end find_y_values_l231_231430


namespace a4_equals_9_l231_231978

variable {a : ℕ → ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a4_equals_9 (h_geom : geometric_sequence a)
  (h_roots : ∃ a2 a6 : ℝ, a2^2 - 34 * a2 + 81 = 0 ∧ a6^2 - 34 * a6 + 81 = 0 ∧ a 2 = a2 ∧ a 6 = a6) :
  a 4 = 9 :=
sorry

end a4_equals_9_l231_231978


namespace length_of_PQ_l231_231848

theorem length_of_PQ
  (k : ℝ) -- height of the trapezoid
  (PQ RU : ℝ) -- sides of trapezoid PQRU
  (A1 : ℝ := (PQ * k) / 2) -- area of triangle PQR
  (A2 : ℝ := (RU * k) / 2) -- area of triangle PUR
  (ratio_A1_A2 : A1 / A2 = 5 / 2) -- given ratio of areas
  (sum_PQ_RU : PQ + RU = 180) -- given sum of PQ and RU
  : PQ = 900 / 7 :=
by
  sorry

end length_of_PQ_l231_231848


namespace possible_values_of_k_l231_231816

noncomputable def has_roots (p q r s t k : ℂ) : Prop :=
  (p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0) ∧ 
  (p * k^4 + q * k^3 + r * k^2 + s * k + t = 0) ∧
  (q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)

theorem possible_values_of_k (p q r s t k : ℂ) (hk : has_roots p q r s t k) : 
  k^5 = 1 :=
  sorry

end possible_values_of_k_l231_231816


namespace points_on_line_l231_231840

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l231_231840


namespace jogger_ahead_of_train_l231_231621

noncomputable def distance_ahead_of_train (v_j v_t : ℕ) (L_t t : ℕ) : ℕ :=
  let relative_speed_kmh := v_t - v_j
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  let total_distance := relative_speed_ms * t
  total_distance - L_t

theorem jogger_ahead_of_train :
  distance_ahead_of_train 10 46 120 46 = 340 :=
by
  sorry

end jogger_ahead_of_train_l231_231621


namespace radius_of_circle_l231_231935

noncomputable def radius (α : ℝ) : ℝ :=
  5 / Real.sin (α / 2)

theorem radius_of_circle (c α : ℝ) (h_c : c = 10) :
  (radius α) = 5 / Real.sin (α / 2) := by
  sorry

end radius_of_circle_l231_231935


namespace chocolate_discount_l231_231985

theorem chocolate_discount :
    let original_cost : ℝ := 2
    let final_price : ℝ := 1.43
    let discount := original_cost - final_price
    discount = 0.57 := by
  sorry

end chocolate_discount_l231_231985


namespace numbers_are_odd_l231_231779

theorem numbers_are_odd (n : ℕ) (sum : ℕ) (h1 : n = 49) (h2 : sum = 2401) : 
      (∀ i < n, ∃ j, sum = j * 2 * i + 1) :=
by
  sorry

end numbers_are_odd_l231_231779


namespace sufficient_condition_not_necessary_condition_l231_231469

/--
\(a > 1\) is a sufficient but not necessary condition for \(\frac{1}{a} < 1\).
-/
theorem sufficient_condition (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by
  sorry

theorem not_necessary_condition (a : ℝ) (h : 1 / a < 1) : a > 1 ∨ a < 0 :=
by
  sorry

end sufficient_condition_not_necessary_condition_l231_231469


namespace average_greater_than_median_by_22_l231_231641

/-- Define the weights of the siblings -/
def hammie_weight : ℕ := 120
def triplet1_weight : ℕ := 4
def triplet2_weight : ℕ := 4
def triplet3_weight : ℕ := 7
def brother_weight : ℕ := 10

/-- Define the list of weights -/
def weights : List ℕ := [hammie_weight, triplet1_weight, triplet2_weight, triplet3_weight, brother_weight]

/-- Define the median and average weight -/
def median_weight : ℕ := 7
def average_weight : ℕ := 29

theorem average_greater_than_median_by_22 : average_weight - median_weight = 22 := by
  sorry

end average_greater_than_median_by_22_l231_231641


namespace cost_of_video_game_console_l231_231141

-- Define the problem conditions
def earnings_Mar_to_Aug : ℕ := 460
def hours_Mar_to_Aug : ℕ := 23
def earnings_per_hour : ℕ := earnings_Mar_to_Aug / hours_Mar_to_Aug
def hours_Sep_to_Feb : ℕ := 8
def cost_car_fix : ℕ := 340
def additional_hours_needed : ℕ := 16

-- Proof that the cost of the video game console is $600
theorem cost_of_video_game_console :
  let initial_earnings := earnings_Mar_to_Aug
  let earnings_from_Sep_to_Feb := hours_Sep_to_Feb * earnings_per_hour
  let total_earnings_before_expenses := initial_earnings + earnings_from_Sep_to_Feb
  let current_savings := total_earnings_before_expenses - cost_car_fix
  let earnings_after_additional_work := additional_hours_needed * earnings_per_hour
  let total_savings := current_savings + earnings_after_additional_work
  total_savings = 600 :=
by
  sorry

end cost_of_video_game_console_l231_231141


namespace find_unknown_rate_l231_231345

-- Define the known quantities
def num_blankets1 := 4
def price1 := 100

def num_blankets2 := 5
def price2 := 150

def num_blankets3 := 3
def price3 := 200

def num_blankets4 := 6
def price4 := 75

def num_blankets_unknown := 2

def avg_price := 150
def total_blankets := num_blankets1 + num_blankets2 + num_blankets3 + num_blankets4 + num_blankets_unknown -- 20 blankets in total

-- Hypotheses
def total_known_cost := num_blankets1 * price1 + num_blankets2 * price2 + num_blankets3 * price3 + num_blankets4 * price4
-- 2200 Rs.

def total_cost := total_blankets * avg_price -- 3000 Rs.

theorem find_unknown_rate :
  (total_cost - total_known_cost) / num_blankets_unknown = 400 :=
by sorry

end find_unknown_rate_l231_231345


namespace sum_of_odd_coefficients_in_binomial_expansion_l231_231140

theorem sum_of_odd_coefficients_in_binomial_expansion :
  let a_0 := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  (a_1 + a_3 + a_5 + a_7 + a_9) = 512 := by
  sorry

end sum_of_odd_coefficients_in_binomial_expansion_l231_231140


namespace isosceles_triangle_bisector_properties_l231_231332

theorem isosceles_triangle_bisector_properties:
  ∀ (T : Type) (triangle : T)
  (is_isosceles : Prop) (vertex_angle_bisector_bisects_base : Prop) (vertex_angle_bisector_perpendicular_to_base : Prop),
  is_isosceles 
  → (vertex_angle_bisector_bisects_base ∧ vertex_angle_bisector_perpendicular_to_base) :=
sorry

end isosceles_triangle_bisector_properties_l231_231332


namespace circle_line_distance_condition_l231_231925

theorem circle_line_distance_condition :
  ∀ (c : ℝ), 
    (∃ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 8 = 0 ∧ (x - y + c = 2 ∨ x - y + c = -2)) →
    -2*Real.sqrt 2 ≤ c ∧ c ≤ 2*Real.sqrt 2 := 
sorry

end circle_line_distance_condition_l231_231925


namespace gcd_of_three_numbers_l231_231407

theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd 324 243) 135 = 27 := 
by 
  sorry

end gcd_of_three_numbers_l231_231407


namespace april_roses_l231_231650

theorem april_roses (R : ℕ) (h1 : 7 * (R - 4) = 35) : R = 9 :=
sorry

end april_roses_l231_231650


namespace Margie_can_drive_200_miles_l231_231438

/--
  Margie's car can go 40 miles per gallon of gas, and the price of gas is $5 per gallon.
  Prove that Margie can drive 200 miles with $25 worth of gas.
-/
theorem Margie_can_drive_200_miles (miles_per_gallon price_per_gallon money_available : ℕ) 
  (h1 : miles_per_gallon = 40) (h2 : price_per_gallon = 5) (h3 : money_available = 25) : 
  (money_available / price_per_gallon) * miles_per_gallon = 200 :=
by 
  /- The proof goes here -/
  sorry

end Margie_can_drive_200_miles_l231_231438


namespace fraction_books_sold_l231_231270

theorem fraction_books_sold (B : ℕ) (F : ℚ) (h1 : 36 = B - F * B) (h2 : 252 = 3.50 * F * B) : F = 2 / 3 := by
  -- Proof omitted
  sorry

end fraction_books_sold_l231_231270


namespace find_speed_l231_231034

-- Definitions corresponding to conditions
def JacksSpeed (x : ℝ) : ℝ := x^2 - 7 * x - 12
def JillsDistance (x : ℝ) : ℝ := x^2 - 3 * x - 10
def JillsTime (x : ℝ) : ℝ := x + 2

-- Theorem statement
theorem find_speed (x : ℝ) (hx : x ≠ -2) (h_speed_eq : JacksSpeed x = (JillsDistance x) / (JillsTime x)) : JacksSpeed x = 2 :=
by
  sorry

end find_speed_l231_231034


namespace cryptarithm_base_solution_l231_231272

theorem cryptarithm_base_solution :
  ∃ (K I T : ℕ) (d : ℕ), 
    O = 0 ∧
    2 * T = I ∧
    T + 1 = K ∧
    K + I = d ∧ 
    d = 7 ∧ 
    K ≠ I ∧ K ≠ T ∧ K ≠ O ∧
    I ≠ T ∧ I ≠ O ∧
    T ≠ O :=
sorry

end cryptarithm_base_solution_l231_231272


namespace smallest_a_for_polynomial_roots_l231_231971

theorem smallest_a_for_polynomial_roots :
  ∃ (a b c : ℕ), 
         (∃ (r s t u : ℕ), r > 0 ∧ s > 0 ∧ t > 0 ∧ u > 0 ∧ r * s * t * u = 5160 ∧ a = r + s + t + u) 
    ∧  (∀ (r' s' t' u' : ℕ), r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧ r' * s' * t' * u' = 5160 ∧ r' + s' + t' + u' < a → false) 
    := sorry

end smallest_a_for_polynomial_roots_l231_231971


namespace total_weight_kg_l231_231099

def envelope_weight_grams : ℝ := 8.5
def num_envelopes : ℝ := 800

theorem total_weight_kg : (envelope_weight_grams * num_envelopes) / 1000 = 6.8 :=
by
  sorry

end total_weight_kg_l231_231099


namespace rectangle_area_l231_231134

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 :=
by
  sorry

end rectangle_area_l231_231134


namespace original_price_of_RAM_l231_231713

variables (P : ℝ)

-- Conditions extracted from the problem statement
def priceAfterFire (P : ℝ) : ℝ := 1.30 * P
def priceAfterDecrease (P : ℝ) : ℝ := 1.04 * P

-- The given current price
axiom current_price : priceAfterDecrease P = 52

-- Theorem to prove the original price P
theorem original_price_of_RAM : P = 50 :=
sorry

end original_price_of_RAM_l231_231713


namespace angle_AMC_is_70_l231_231167

theorem angle_AMC_is_70 (A B C M : Type) (angle_MBA angle_MAB angle_ACB : ℝ) (AC BC : ℝ) :
  AC = BC → 
  angle_MBA = 30 → 
  angle_MAB = 10 → 
  angle_ACB = 80 → 
  ∃ angle_AMC : ℝ, angle_AMC = 70 :=
by
  sorry

end angle_AMC_is_70_l231_231167


namespace bisection_method_third_interval_l231_231917

noncomputable def bisection_method_interval (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : (ℝ × ℝ) :=
  sorry  -- Definition of the interval using bisection method, but this is not necessary.

theorem bisection_method_third_interval (f : ℝ → ℝ) :
  (bisection_method_interval f (-2) 4 3) = (-1/2, 1) :=
sorry

end bisection_method_third_interval_l231_231917


namespace total_number_of_edges_in_hexahedron_is_12_l231_231132

-- Define a hexahedron
structure Hexahedron where
  face_count : Nat
  edges_per_face : Nat
  edge_sharing : Nat

-- Total edges calculation function
def total_edges (h : Hexahedron) : Nat := (h.face_count * h.edges_per_face) / h.edge_sharing

-- The specific hexahedron (cube) in question
def cube : Hexahedron := {
  face_count := 6,
  edges_per_face := 4,
  edge_sharing := 2
}

-- The theorem to prove the number of edges in a hexahedron
theorem total_number_of_edges_in_hexahedron_is_12 : total_edges cube = 12 := by
  sorry

end total_number_of_edges_in_hexahedron_is_12_l231_231132


namespace translate_graph_upwards_l231_231245

theorem translate_graph_upwards (x : ℝ) :
  (∀ x, (3*x - 1) + 3 = 3*x + 2) :=
by
  intro x
  sorry

end translate_graph_upwards_l231_231245


namespace at_least_one_basketball_selected_l231_231447

theorem at_least_one_basketball_selected (balls : Finset ℕ) (basketballs : Finset ℕ) (volleyballs : Finset ℕ) :
  basketballs.card = 6 → volleyballs.card = 2 → balls ⊆ (basketballs ∪ volleyballs) →
  balls.card = 3 → ∃ b ∈ balls, b ∈ basketballs :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end at_least_one_basketball_selected_l231_231447


namespace abc_relationship_l231_231343

noncomputable def a : ℝ := Real.log 5 - Real.log 3
noncomputable def b : ℝ := (2/5) * Real.exp (2/3)
noncomputable def c : ℝ := 2/3

theorem abc_relationship : b > c ∧ c > a :=
by
  sorry

end abc_relationship_l231_231343


namespace chocolate_bar_percentage_l231_231963

theorem chocolate_bar_percentage (milk_chocolate dark_chocolate almond_chocolate white_chocolate : ℕ)
  (h1 : milk_chocolate = 25) (h2 : dark_chocolate = 25)
  (h3 : almond_chocolate = 25) (h4 : white_chocolate = 25) :
  (milk_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (dark_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (almond_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (white_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 :=
by
  sorry

end chocolate_bar_percentage_l231_231963


namespace actual_travel_time_l231_231549

noncomputable def distance : ℕ := 360
noncomputable def scheduled_time : ℕ := 9
noncomputable def speed_increase : ℕ := 5

theorem actual_travel_time (d : ℕ) (t_sched : ℕ) (Δv : ℕ) : 
  (d = distance) ∧ (t_sched = scheduled_time) ∧ (Δv = speed_increase) → 
  t_sched + Δv = 8 :=
by
  sorry

end actual_travel_time_l231_231549


namespace rich_knight_l231_231728

-- Definitions for the problem
inductive Status
| knight  -- Always tells the truth
| knave   -- Always lies

def tells_truth (s : Status) : Prop := 
  s = Status.knight

def lies (s : Status) : Prop := 
  s = Status.knave

def not_poor (s : Status) : Prop := 
  s = Status.knight ∨ s = Status.knave -- Knights can either be poor or wealthy

def wealthy (s : Status) : Prop :=
  s = Status.knight

-- Statement to be proven
theorem rich_knight (s : Status) (h_truth : tells_truth s) (h_not_poor : not_poor s) : wealthy s :=
by
  sorry

end rich_knight_l231_231728


namespace solution_set_x_plus_3_f_x_plus_4_l231_231298

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom even_f_x_plus_1 : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom deriv_negative_f : ∀ x : ℝ, x > 1 → f' x < 0
axiom f_at_4_equals_zero : f 4 = 0

-- To prove
theorem solution_set_x_plus_3_f_x_plus_4 :
  {x : ℝ | (x + 3) * f (x + 4) < 0} = {x : ℝ | -6 < x ∧ x < -3} ∪ {x : ℝ | x > 0} := sorry

end solution_set_x_plus_3_f_x_plus_4_l231_231298


namespace oliver_bumper_cars_proof_l231_231097

def rides_of_bumper_cars (total_tickets : ℕ) (tickets_per_ride : ℕ) (rides_ferris_wheel : ℕ) : ℕ :=
  (total_tickets - rides_ferris_wheel * tickets_per_ride) / tickets_per_ride

def oliver_bumper_car_rides : Prop :=
  rides_of_bumper_cars 30 3 7 = 3

theorem oliver_bumper_cars_proof : oliver_bumper_car_rides :=
by
  sorry

end oliver_bumper_cars_proof_l231_231097


namespace ceiling_is_multiple_of_3_l231_231903

-- Given conditions:
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1
axiom exists_three_real_roots : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧
  polynomial x1 = 0 ∧ polynomial x2 = 0 ∧ polynomial x3 = 0

-- Goal:
theorem ceiling_is_multiple_of_3 (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3)
  (hx1 : polynomial x1 = 0) (hx2 : polynomial x2 = 0) (hx3 : polynomial x3 = 0):
  ∀ n : ℕ, n > 0 → ∃ k : ℤ, k * 3 = ⌈x3^n⌉ := by
  sorry

end ceiling_is_multiple_of_3_l231_231903


namespace tank_empty_time_correct_l231_231821

noncomputable def tank_time_to_empty (leak_empty_time : ℕ) (inlet_rate : ℕ) (tank_capacity : ℕ) : ℕ :=
(tank_capacity / (tank_capacity / leak_empty_time - inlet_rate * 60))

theorem tank_empty_time_correct :
  tank_time_to_empty 6 3 4320 = 8 := by
  sorry

end tank_empty_time_correct_l231_231821


namespace range_of_a_l231_231170

theorem range_of_a (x : ℝ) (a : ℝ) (h1 : 2 < x) (h2 : a ≤ x + 1 / (x - 2)) : a ≤ 4 := 
sorry

end range_of_a_l231_231170


namespace probability_of_four_odd_slips_l231_231219

-- Define the conditions
def number_of_slips : ℕ := 10
def odd_slips : ℕ := 5
def even_slips : ℕ := 5
def slips_drawn : ℕ := 4

-- Define the required probability calculation
def probability_four_odd_slips : ℚ := (5 / 10) * (4 / 9) * (3 / 8) * (2 / 7)

-- State the theorem we want to prove
theorem probability_of_four_odd_slips :
  probability_four_odd_slips = 1 / 42 :=
by
  sorry

end probability_of_four_odd_slips_l231_231219


namespace range_of_m_l231_231937

theorem range_of_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) (h4 : 4 / a + 1 / (b - 1) > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l231_231937


namespace correct_evaluation_l231_231385

noncomputable def evaluate_expression : ℚ :=
  - (2 : ℚ) ^ 3 + (6 / 5) * (2 / 5)

theorem correct_evaluation : evaluate_expression = -7 - 13 / 25 :=
by
  unfold evaluate_expression
  sorry

end correct_evaluation_l231_231385


namespace smallest_prime_less_than_square_l231_231384

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l231_231384


namespace smallest_four_digit_divisible_by_55_l231_231348

theorem smallest_four_digit_divisible_by_55 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 55 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 55 = 0 → n ≤ m := by
  sorry

end smallest_four_digit_divisible_by_55_l231_231348


namespace operation_eval_l231_231592

def my_operation (a b : ℤ) := a * (b + 2) + a * (b + 1)

theorem operation_eval : my_operation 3 (-1) = 3 := by
  sorry

end operation_eval_l231_231592


namespace enchilada_taco_cost_l231_231570

theorem enchilada_taco_cost (e t : ℝ) 
  (h1 : 3 * e + 4 * t = 3.50) 
  (h2 : 4 * e + 3 * t = 3.90) : 
  4 * e + 5 * t = 4.56 := 
sorry

end enchilada_taco_cost_l231_231570


namespace problem_statement_l231_231954

variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)
variable (a b : ℝ)

theorem problem_statement (h1 : ∀ x, HasDerivAt f (f' x) x)
                         (h2 : ∀ x, HasDerivAt g (g' x) x)
                         (h3 : ∀ x, f' x < g' x)
                         (h4 : a = Real.log 2 / Real.log 5)
                         (h5 : b = Real.log 3 / Real.log 8) :
                         f a + g b > g a + f b := 
     sorry

end problem_statement_l231_231954


namespace mary_initial_nickels_l231_231872

variable {x : ℕ}

theorem mary_initial_nickels (h : x + 5 = 12) : x = 7 := by
  sorry

end mary_initial_nickels_l231_231872


namespace sin_product_identity_l231_231155

theorem sin_product_identity :
  (Real.sin (12 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * (Real.sin (72 * Real.pi / 180)) = 1 / 16 := 
by 
  sorry

end sin_product_identity_l231_231155


namespace angle_between_vectors_l231_231236

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem angle_between_vectors
  (a b : ℝ × ℝ)
  (h₁ : vector_norm a ≠ 0)
  (h₂ : vector_norm b ≠ 0)
  (h₃ : vector_norm a = vector_norm b)
  (h₄ : vector_norm a = vector_norm (a.1 + 2 * b.1, a.2 + 2 * b.2)) :
  ∃ θ : ℝ, θ = 180 ∧ cos θ = -1 := 
sorry

end angle_between_vectors_l231_231236


namespace perimeter_C_l231_231661

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l231_231661


namespace roots_relationship_l231_231684

theorem roots_relationship (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0)
  (h_triple : β = 3 * α)
  (h_vieta1 : α + β = -b / a)
  (h_vieta2 : α * β = c / a) : 
  3 * b^2 = 16 * a * c :=
sorry

end roots_relationship_l231_231684


namespace MinkyungHeight_is_correct_l231_231976

noncomputable def HaeunHeight : ℝ := 1.56
noncomputable def NayeonHeight : ℝ := HaeunHeight - 0.14
noncomputable def MinkyungHeight : ℝ := NayeonHeight + 0.27

theorem MinkyungHeight_is_correct : MinkyungHeight = 1.69 :=
by
  sorry

end MinkyungHeight_is_correct_l231_231976


namespace tom_new_collection_l231_231403

theorem tom_new_collection (initial_stamps mike_gift : ℕ) (harry_gift : ℕ := 2 * mike_gift + 10) (sarah_gift : ℕ := 3 * mike_gift - 5) (total_gifts : ℕ := mike_gift + harry_gift + sarah_gift) (new_collection : ℕ := initial_stamps + total_gifts) 
  (h_initial_stamps : initial_stamps = 3000) (h_mike_gift : mike_gift = 17) :
  new_collection = 3107 := by
  sorry

end tom_new_collection_l231_231403


namespace intersection_is_correct_l231_231222

def setA : Set ℕ := {0, 1, 2}
def setB : Set ℕ := {1, 2, 3}

theorem intersection_is_correct : setA ∩ setB = {1, 2} := by
  sorry

end intersection_is_correct_l231_231222


namespace c_is_perfect_square_or_not_even_c_cannot_be_even_l231_231706

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem c_is_perfect_square_or_not_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_odd : c % 2 = 1) : is_perfect_square c :=
sorry

theorem c_cannot_be_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_even : c % 2 = 0) : false :=
sorry

end c_is_perfect_square_or_not_even_c_cannot_be_even_l231_231706


namespace num_ways_distribute_balls_l231_231918

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end num_ways_distribute_balls_l231_231918


namespace number_of_rows_l231_231577

-- Definitions of conditions
def tomatoes : ℕ := 3 * 5
def cucumbers : ℕ := 5 * 4
def potatoes : ℕ := 30
def additional_vegetables : ℕ := 85
def spaces_per_row : ℕ := 15

-- Total number of vegetables already planted
def planted_vegetables : ℕ := tomatoes + cucumbers + potatoes

-- Total capacity of the garden
def garden_capacity : ℕ := planted_vegetables + additional_vegetables

-- Number of rows in the garden
def rows_in_garden : ℕ := garden_capacity / spaces_per_row

theorem number_of_rows : rows_in_garden = 10 := by
  sorry

end number_of_rows_l231_231577


namespace arc_length_of_circle_l231_231529

theorem arc_length_of_circle (r : ℝ) (alpha : ℝ) (h_r : r = 10) (h_alpha : alpha = (2 * Real.pi) / 6) : 
  (alpha * r) = (10 * Real.pi) / 3 :=
by
  rw [h_r, h_alpha]
  sorry

end arc_length_of_circle_l231_231529


namespace total_students_in_class_l231_231419

theorem total_students_in_class :
  ∃ x, (10 * 90 + 15 * 80 + x * 60) / (10 + 15 + x) = 72 → 10 + 15 + x = 50 :=
by
  -- Providing an existence proof and required conditions
  use 25
  intro h
  sorry

end total_students_in_class_l231_231419


namespace car_value_proof_l231_231892

-- Let's define the variables and the conditions.
def car_sold_value : ℝ := 20000
def sticker_price_new_car : ℝ := 30000
def percent_sold : ℝ := 0.80
def percent_paid : ℝ := 0.90
def out_of_pocket : ℝ := 11000

theorem car_value_proof :
  (percent_paid * sticker_price_new_car - percent_sold * car_sold_value = out_of_pocket) →
  car_sold_value = 20000 := 
by
  intros h
  -- Introduction of any intermediate steps if necessary should just invoke the sorry to indicate the need for proof later
  exact sorry

end car_value_proof_l231_231892


namespace sweet_treats_distribution_l231_231325

-- Define the number of cookies, cupcakes, brownies, and students
def cookies : ℕ := 20
def cupcakes : ℕ := 25
def brownies : ℕ := 35
def students : ℕ := 20

-- Define the total number of sweet treats
def total_sweet_treats : ℕ := cookies + cupcakes + brownies

-- Define the number of sweet treats each student will receive
def sweet_treats_per_student : ℕ := total_sweet_treats / students

-- Prove that each student will receive 4 sweet treats
theorem sweet_treats_distribution : sweet_treats_per_student = 4 := 
by sorry

end sweet_treats_distribution_l231_231325


namespace smallest_special_integer_l231_231505

noncomputable def is_special (N : ℕ) : Prop :=
  N > 1 ∧ 
  (N % 8 = 1) ∧ 
  (2 * 8 ^ Nat.log N (8) / 2 > N / 8 ^ Nat.log N (8)) ∧ 
  (N % 9 = 1) ∧ 
  (2 * 9 ^ Nat.log N (9) / 2 > N / 9 ^ Nat.log N (9))

theorem smallest_special_integer : ∃ (N : ℕ), is_special N ∧ N = 793 :=
by 
  use 793
  sorry

end smallest_special_integer_l231_231505


namespace linear_function_unique_l231_231851

noncomputable def f (x : ℝ) : ℝ := sorry

theorem linear_function_unique
  (h1 : ∀ x : ℝ, f (f x) = 4 * x + 6)
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :
  ∀ x : ℝ, f x = 2 * x + 2 :=
sorry

end linear_function_unique_l231_231851


namespace sum_of_ages_of_henrys_brothers_l231_231449

theorem sum_of_ages_of_henrys_brothers (a b c : ℕ) : 
  a = 2 * b → 
  b = c ^ 2 →
  a ≠ b ∧ a ≠ c ∧ b ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  a + b + c = 14 :=
by
  intro h₁ h₂ h₃ h₄
  sorry

end sum_of_ages_of_henrys_brothers_l231_231449


namespace min_supreme_supervisors_l231_231527

-- Definitions
def num_employees : ℕ := 50000
def supervisors (e : ℕ) : ℕ := 7 - e

-- Theorem statement
theorem min_supreme_supervisors (k : ℕ) (num_employees_le_reached : ∀ n : ℕ, 50000 ≤ n) : 
  k ≥ 28 := 
sorry

end min_supreme_supervisors_l231_231527


namespace paula_remaining_money_l231_231673

-- Definitions based on the conditions
def initialMoney : ℕ := 1000
def shirtCost : ℕ := 45
def pantsCost : ℕ := 85
def jacketCost : ℕ := 120
def shoeCost : ℕ := 95
def jeansOriginalPrice : ℕ := 140
def jeansDiscount : ℕ := 30 / 100  -- 30%

-- Using definitions to compute the spending and remaining money
def totalShirtCost : ℕ := 6 * shirtCost
def totalPantsCost : ℕ := 2 * pantsCost
def totalShoeCost : ℕ := 3 * shoeCost
def jeansDiscountValue : ℕ := jeansDiscount * jeansOriginalPrice
def jeansDiscountedPrice : ℕ := jeansOriginalPrice - jeansDiscountValue
def totalSpent : ℕ := totalShirtCost + totalPantsCost + jacketCost + totalShoeCost
def remainingMoney : ℕ := initialMoney - totalSpent - jeansDiscountedPrice

-- Proof problem statement
theorem paula_remaining_money : remainingMoney = 57 := by
  sorry

end paula_remaining_money_l231_231673


namespace quadratic_roots_real_distinct_l231_231962

theorem quadratic_roots_real_distinct (k : ℝ) :
  (k > (1/2)) ∧ (k ≠ 1) ↔
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k-1) * x1^2 + 2 * x1 - 2 = 0) ∧ ((k-1) * x2^2 + 2 * x2 - 2 = 0)) :=
by
  sorry

end quadratic_roots_real_distinct_l231_231962


namespace gcd_4536_13440_216_l231_231708

def gcd_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd a b) c

theorem gcd_4536_13440_216 : gcd_of_three_numbers 4536 13440 216 = 216 :=
by
  sorry

end gcd_4536_13440_216_l231_231708


namespace parametric_to_ordinary_eq_l231_231916

variable (t : ℝ)

theorem parametric_to_ordinary_eq (h1 : x = Real.sqrt t + 1) (h2 : y = 2 * Real.sqrt t - 1) (h3 : t ≥ 0) :
    y = 2 * x - 3 ∧ x ≥ 1 := by
  sorry

end parametric_to_ordinary_eq_l231_231916


namespace solid_circles_2006_l231_231784

noncomputable def circlePattern : Nat → Nat
| n => (2 + n * (n + 3)) / 2

theorem solid_circles_2006 :
  ∃ n, circlePattern n < 2006 ∧ circlePattern (n + 1) > 2006 ∧ n = 61 :=
by
  sorry

end solid_circles_2006_l231_231784


namespace fraction_multiplication_exponent_l231_231605

theorem fraction_multiplication_exponent :
  ( (8 : ℚ) / 9 )^2 * ( (1 : ℚ) / 3 )^2 = (64 / 729 : ℚ) := 
by
  sorry

end fraction_multiplication_exponent_l231_231605


namespace multiplicative_inverse_l231_231710

def A : ℕ := 123456
def B : ℕ := 162738
def N : ℕ := 503339
def modulo : ℕ := 1000000

theorem multiplicative_inverse :
  (A * B * N) % modulo = 1 :=
by
  -- placeholder for proof
  sorry

end multiplicative_inverse_l231_231710


namespace range_of_f_l231_231882

def f (x : ℕ) : ℤ := 2 * x - 3

def domain := {x : ℕ | 1 ≤ x ∧ x ≤ 5}

def range (f : ℕ → ℤ) (s : Set ℕ) : Set ℤ :=
  {y : ℤ | ∃ x ∈ s, f x = y}

theorem range_of_f :
  range f domain = {-1, 1, 3, 5, 7} :=
by
  sorry

end range_of_f_l231_231882


namespace sum_of_ammeter_readings_l231_231904

def I1 := 4 
def I2 := 4
def I3 := 2 * I2
def I5 := I3 + I2
def I4 := (5 / 3) * I5

theorem sum_of_ammeter_readings : I1 + I2 + I3 + I4 + I5 = 48 := by
  sorry

end sum_of_ammeter_readings_l231_231904


namespace hyperbola_focus_distance_l231_231277
open Real

theorem hyperbola_focus_distance
  (a b : ℝ)
  (ha : a = 5)
  (hb : b = 3)
  (hyperbola_eq : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (∃ M : ℝ × ℝ, M = (x, y)))
  (M : ℝ × ℝ)
  (hM_on_hyperbola : ∃ x y : ℝ, M = (x, y) ∧ x^2 / a^2 - y^2 / b^2 = 1)
  (F1_pos : ℝ)
  (h_dist_F1 : dist M (F1_pos, 0) = 18) :
  (∃ (F2_dist : ℝ), (F2_dist = 8 ∨ F2_dist = 28) ∧ dist M (F2_dist, 0) = F2_dist) := 
sorry

end hyperbola_focus_distance_l231_231277


namespace find_number_l231_231908

variable (x : ℝ)

theorem find_number (h : 20 * (x / 5) = 40) : x = 10 := by
  sorry

end find_number_l231_231908


namespace percentage_above_wholesale_cost_l231_231920

def wholesale_cost : ℝ := 200
def paid_price : ℝ := 228
def discount_rate : ℝ := 0.05

theorem percentage_above_wholesale_cost :
  ∃ P : ℝ, P = 20 ∧ 
    paid_price = (1 - discount_rate) * (wholesale_cost + P/100 * wholesale_cost) :=
by
  sorry

end percentage_above_wholesale_cost_l231_231920


namespace bread_consumption_l231_231085

-- Definitions using conditions
def members := 4
def slices_snacks := 2
def slices_per_loaf := 12
def total_loaves := 5
def total_days := 3

-- The main theorem to prove
theorem bread_consumption :
  (3 * members * (B + slices_snacks) = total_loaves * slices_per_loaf) → B = 3 :=
by
  intro h
  sorry

end bread_consumption_l231_231085


namespace quadratic_form_and_sum_l231_231169

theorem quadratic_form_and_sum (x : ℝ) : 
  ∃ (a b c : ℝ), 
  (15 * x^2 + 75 * x + 375 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 298.75) := 
sorry

end quadratic_form_and_sum_l231_231169


namespace log_product_zero_l231_231273

theorem log_product_zero :
  (Real.log 3 / Real.log 2 + Real.log 27 / Real.log 2) *
  (Real.log 4 / Real.log 4 + Real.log (1 / 4) / Real.log 4) = 0 := by
  -- Place proof here
  sorry

end log_product_zero_l231_231273


namespace cos_D_zero_l231_231932

noncomputable def area_of_triangle (a b: ℝ) (sinD: ℝ) : ℝ := 1 / 2 * a * b * sinD

theorem cos_D_zero (DE DF : ℝ) (D : ℝ) (h1 : area_of_triangle DE DF (Real.sin D) = 98) (h2 : Real.sqrt (DE * DF) = 14) : Real.cos D = 0 :=
  by
  sorry

end cos_D_zero_l231_231932


namespace find_a_l231_231373

theorem find_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 4 * y = 0 → (3 * x + y + a = 0))) → a = 1 :=
sorry

end find_a_l231_231373


namespace geometric_sequence_a3_l231_231879

variable {a : ℕ → ℝ} (h1 : a 1 > 0) (h2 : a 2 * a 4 = 25)
def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (h_geom : geometric_sequence a) : 
  a 3 = 5 := 
by
  sorry

end geometric_sequence_a3_l231_231879


namespace fish_distribution_l231_231705

theorem fish_distribution 
  (fish_caught : ℕ)
  (eyes_per_fish : ℕ := 2)
  (total_eyes : ℕ := 24)
  (people : ℕ := 3)
  (eyes_eaten_by_dog : ℕ := 2)
  (eyes_eaten_by_oomyapeck : ℕ := 22)
  (oomyapeck_total_eyes : eyes_eaten_by_oomyapeck + eyes_eaten_by_dog = total_eyes)
  (fish_per_person := fish_caught / people)
  (fish_eyes_relation : total_eyes = eyes_per_fish * fish_caught) :
  fish_per_person = 4 := by
  sorry

end fish_distribution_l231_231705


namespace avg_cost_is_12_cents_l231_231378

noncomputable def avg_cost_per_pencil 
    (price_per_package : ℝ)
    (num_pencils : ℕ)
    (shipping_cost : ℝ)
    (discount_rate : ℝ) : ℝ :=
  let price_after_discount := price_per_package - (discount_rate * price_per_package)
  let total_cost := price_after_discount + shipping_cost
  let total_cost_cents := total_cost * 100
  total_cost_cents / num_pencils

theorem avg_cost_is_12_cents :
  avg_cost_per_pencil 29.70 300 8.50 0.10 = 12 := 
by {
  sorry
}

end avg_cost_is_12_cents_l231_231378


namespace trick_deck_cost_l231_231221

theorem trick_deck_cost (x : ℝ) (h1 : 6 * x + 2 * x = 64) : x = 8 :=
  sorry

end trick_deck_cost_l231_231221


namespace unique_solution_l231_231137

theorem unique_solution (m n : ℕ) (h1 : n^4 ∣ 2 * m^5 - 1) (h2 : m^4 ∣ 2 * n^5 + 1) : m = 1 ∧ n = 1 :=
by
  sorry

end unique_solution_l231_231137


namespace negation_ln_eq_x_minus_1_l231_231559

theorem negation_ln_eq_x_minus_1 :
  ¬(∃ x : ℝ, 0 < x ∧ Real.log x = x - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by 
  sorry

end negation_ln_eq_x_minus_1_l231_231559


namespace trays_from_first_table_is_23_l231_231675

-- Definitions of conditions
def trays_per_trip : ℕ := 7
def trips_made : ℕ := 4
def trays_from_second_table : ℕ := 5

-- Total trays carried
def total_trays_carried : ℕ := trays_per_trip * trips_made

-- Number of trays picked from first table
def trays_from_first_table : ℕ :=
  total_trays_carried - trays_from_second_table

-- Theorem stating that the number of trays picked up from the first table is 23
theorem trays_from_first_table_is_23 : trays_from_first_table = 23 := by
  sorry

end trays_from_first_table_is_23_l231_231675


namespace smallest_n_l231_231118

theorem smallest_n (n : ℕ) : 634 * n ≡ 1275 * n [MOD 30] ↔ n = 30 :=
by
  sorry

end smallest_n_l231_231118


namespace evaluate_expression_l231_231460

theorem evaluate_expression :
  71 * Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 72 + 70 * Real.sqrt 2 :=
by
  sorry

end evaluate_expression_l231_231460


namespace line_tangent_to_circle_l231_231035

theorem line_tangent_to_circle (x y : ℝ) :
  (3 * x - 4 * y + 25 = 0) ∧ (x^2 + y^2 = 25) → (x = -3 ∧ y = 4) :=
by sorry

end line_tangent_to_circle_l231_231035


namespace evaluate_expression_l231_231945

theorem evaluate_expression :
  4 * 11 + 5 * 12 + 13 * 4 + 4 * 10 = 196 :=
by
  sorry

end evaluate_expression_l231_231945


namespace Sperner_theorem_example_l231_231923

theorem Sperner_theorem_example :
  ∀ (S : Finset (Finset ℕ)), (S.card = 10) →
  (∀ (A B : Finset ℕ), A ∈ S → B ∈ S → A ⊆ B → A = B) → S.card = 252 :=
by sorry

end Sperner_theorem_example_l231_231923


namespace negation_of_one_odd_l231_231574

-- Given a, b, c are natural numbers
def exactly_one_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 1)

def not_exactly_one_odd (a b c : ℕ) : Prop :=
  ¬ exactly_one_odd a b c

def at_least_two_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1) ∨
  (a % 2 = 1 ∧ c % 2 = 1) ∨
  (b % 2 = 1 ∧ c % 2 = 1)

def all_even (a b c : ℕ) : Prop :=
  (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0)

theorem negation_of_one_odd (a b c : ℕ) : ¬ exactly_one_odd a b c ↔ all_even a b c ∨ at_least_two_odd a b c := by
  sorry

end negation_of_one_odd_l231_231574


namespace express_scientific_notation_l231_231554

theorem express_scientific_notation : (152300 : ℝ) = 1.523 * 10^5 := 
by
  sorry

end express_scientific_notation_l231_231554


namespace twenty_million_in_scientific_notation_l231_231698

/-- Prove that 20 million in scientific notation is 2 * 10^7 --/
theorem twenty_million_in_scientific_notation : 20000000 = 2 * 10^7 :=
by
  sorry

end twenty_million_in_scientific_notation_l231_231698


namespace contrapositive_example_l231_231463

theorem contrapositive_example (x : ℝ) :
  (¬ (x = 3 ∧ x = 4)) → (x^2 - 7 * x + 12 ≠ 0) →
  (x^2 - 7 * x + 12 = 0) → (x = 3 ∨ x = 4) :=
by
  intros h h1 h2
  sorry  -- proof is not required

end contrapositive_example_l231_231463


namespace hyperbola_focal_length_l231_231067

def is_hyperbola (x y a : ℝ) : Prop := (x^2) / (a^2) - (y^2) = 1
def is_perpendicular_asymptote (slope_asymptote slope_line : ℝ) : Prop := slope_asymptote * slope_line = -1

theorem hyperbola_focal_length {a : ℝ} (h1 : is_hyperbola x y a)
  (h2 : is_perpendicular_asymptote (1 / a) (-1)) : 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
sorry

end hyperbola_focal_length_l231_231067


namespace james_driving_speed_l231_231073

theorem james_driving_speed
  (distance : ℝ)
  (total_time : ℝ)
  (stop_time : ℝ)
  (driving_time : ℝ)
  (speed : ℝ)
  (h1 : distance = 360)
  (h2 : total_time = 7)
  (h3 : stop_time = 1)
  (h4 : driving_time = total_time - stop_time)
  (h5 : speed = distance / driving_time) :
  speed = 60 := by
  -- Here you would put the detailed proof.
  sorry

end james_driving_speed_l231_231073


namespace no_positive_integral_solution_l231_231649

theorem no_positive_integral_solution :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ p : ℕ, Prime p ∧ n^2 - 45 * n + 520 = p :=
by {
  -- Since we only need the statement, we'll introduce the necessary steps without the full proof
  sorry
}

end no_positive_integral_solution_l231_231649


namespace remainder_of_x_plus_3uy_l231_231562

-- Given conditions
variables (x y u v : ℕ)
variable (Hdiv : x = u * y + v)
variable (H0_le_v : 0 ≤ v)
variable (Hv_lt_y : v < y)

-- Statement to prove
theorem remainder_of_x_plus_3uy (x y u v : ℕ) (Hdiv : x = u * y + v) (H0_le_v : 0 ≤ v) (Hv_lt_y : v < y) :
  (x + 3 * u * y) % y = v :=
sorry

end remainder_of_x_plus_3uy_l231_231562


namespace line_equation_exists_l231_231580

noncomputable def P : ℝ × ℝ := (-2, 5)
noncomputable def m : ℝ := -3 / 4

theorem line_equation_exists (x y : ℝ) : 
  (y - 5 = -3 / 4 * (x + 2)) ↔ (3 * x + 4 * y - 14 = 0) := 
by 
  sorry

end line_equation_exists_l231_231580


namespace polynomial_has_real_root_l231_231160

open Real Polynomial

variable {c d : ℝ}
variable {P : Polynomial ℝ}

theorem polynomial_has_real_root (hP1 : ∀ n : ℕ, c * |(n : ℝ)|^3 ≤ |P.eval (n : ℝ)|)
                                (hP2 : ∀ n : ℕ, |P.eval (n : ℝ)| ≤ d * |(n : ℝ)|^3)
                                (hc : 0 < c) (hd : 0 < d) : 
                                ∃ x : ℝ, P.eval x = 0 :=
sorry

end polynomial_has_real_root_l231_231160


namespace setD_is_empty_l231_231052

-- Definitions of sets A, B, C, D
def setA : Set ℝ := {x | x + 3 = 3}
def setB : Set (ℝ × ℝ) := {(x, y) | y^2 ≠ -x^2}
def setC : Set ℝ := {x | x^2 ≤ 0}
def setD : Set ℝ := {x | x^2 - x + 1 = 0}

-- Theorem stating that set D is the empty set
theorem setD_is_empty : setD = ∅ := 
by 
  sorry

end setD_is_empty_l231_231052


namespace number_of_typists_needed_l231_231421

theorem number_of_typists_needed :
  (∃ t : ℕ, (20 * 40) / 20 * 60 * t = 180) ↔ t = 30 :=
by sorry

end number_of_typists_needed_l231_231421


namespace value_of_number_l231_231109

theorem value_of_number (number y : ℝ) 
  (h1 : (number + 5) * (y - 5) = 0) 
  (h2 : ∀ n m : ℝ, (n + 5) * (m - 5) = 0 → n^2 + m^2 ≥ 25) 
  (h3 : number^2 + y^2 = 25) : number = -5 :=
sorry

end value_of_number_l231_231109


namespace parabola_vertex_expression_l231_231028

theorem parabola_vertex_expression (h k : ℝ) :
  (h = 2 ∧ k = 3) →
  ∃ (a : ℝ), (a ≠ 0) ∧
    (∀ x y : ℝ, y = a * (x - h)^2 + k ↔ y = -(x - 2)^2 + 3) :=
by
  sorry

end parabola_vertex_expression_l231_231028


namespace distance_between_cities_l231_231115

noncomputable def distance_A_to_B : ℕ := 180
noncomputable def distance_B_to_A : ℕ := 150
noncomputable def total_distance : ℕ := distance_A_to_B + distance_B_to_A

theorem distance_between_cities : total_distance = 330 := by
  sorry

end distance_between_cities_l231_231115


namespace symmetry_condition_l231_231242

theorem symmetry_condition (p q r s t u : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (yx_eq : ∀ x y, y = (p * x ^ 2 + q * x + r) / (s * x ^ 2 + t * x + u) ↔ x = (p * y ^ 2 + q * y + r) / (s * y ^ 2 + t * y + u)) :
  p = s ∧ q = t ∧ r = u :=
sorry

end symmetry_condition_l231_231242


namespace line_intersects_y_axis_at_eight_l231_231340

theorem line_intersects_y_axis_at_eight :
  ∃ b : ℝ, ∃ f : ℝ → ℝ, (∀ x, f x = 2 * x + b) ∧ f 1 = 10 ∧ f (-9) = -10 ∧ f 0 = 8 :=
by
  -- Definitions and calculations leading to verify the theorem
  sorry

end line_intersects_y_axis_at_eight_l231_231340


namespace gcd_polynomial_l231_231636

theorem gcd_polynomial (b : ℤ) (h : b % 2 = 0 ∧ 1171 ∣ b) : 
  Int.gcd (3 * b^2 + 17 * b + 47) (b + 5) = 1 :=
sorry

end gcd_polynomial_l231_231636


namespace num_arrangements_l231_231827

-- Define the problem conditions
def athletes : Finset ℕ := {0, 1, 2, 3, 4, 5}
def A : ℕ := 0
def B : ℕ := 1

-- Define the constraint that athlete A cannot run the first leg and athlete B cannot run the fourth leg
def valid_arrangements (sequence : Fin 4 → ℕ) : Prop :=
  sequence 0 ≠ A ∧ sequence 3 ≠ B

-- Main theorem statement: There are 252 valid arrangements
theorem num_arrangements : (Fin 4 → ℕ) → ℕ :=
  sorry

end num_arrangements_l231_231827


namespace market_value_of_stock_l231_231292

variable (face_value : ℝ) (annual_dividend yield : ℝ)

-- Given conditions:
def stock_four_percent := annual_dividend = 0.04 * face_value
def stock_yield_five_percent := yield = 0.05

-- Problem statement:
theorem market_value_of_stock (face_value := 100) (annual_dividend := 4) (yield := 0.05) 
  (h1 : stock_four_percent face_value annual_dividend) 
  (h2 : stock_yield_five_percent yield) : 
  (4 / 0.05) * 100 = 80 :=
by
  sorry

end market_value_of_stock_l231_231292


namespace sum_arithmetic_series_eq_250500_l231_231653

theorem sum_arithmetic_series_eq_250500 :
  let a1 := 2
  let d := 2
  let an := 1000
  let n := 500
  (a1 + (n-1) * d = an) →
  ((n * (a1 + an)) / 2 = 250500) :=
by
  sorry

end sum_arithmetic_series_eq_250500_l231_231653


namespace unique_integer_solution_quad_eqns_l231_231177

def is_single_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

theorem unique_integer_solution_quad_eqns : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ is_single_digit_prime a ∧ is_single_digit_prime b ∧ is_single_digit_prime c ∧ 
                     ∃ x : ℤ, a * x^2 + b * x + c = 0) ∧ S.card = 7 :=
by
  sorry

end unique_integer_solution_quad_eqns_l231_231177


namespace proposition_contradiction_l231_231863

-- Define the proposition P for natural numbers.
def P (n : ℕ+) : Prop := sorry

theorem proposition_contradiction (h1 : ∀ k : ℕ+, P k → P (k + 1)) (h2 : ¬ P 5) : ¬ P 4 :=
by
  sorry

end proposition_contradiction_l231_231863


namespace time_to_fill_one_barrel_with_leak_l231_231076

-- Define the conditions
def normal_time_per_barrel := 3
def time_to_fill_12_barrels_no_leak := normal_time_per_barrel * 12
def additional_time_due_to_leak := 24
def time_to_fill_12_barrels_with_leak (t : ℕ) := 12 * t

-- Define the theorem
theorem time_to_fill_one_barrel_with_leak :
  ∃ t : ℕ, time_to_fill_12_barrels_with_leak t = time_to_fill_12_barrels_no_leak + additional_time_due_to_leak ∧ t = 5 :=
by {
  use 5, 
  sorry
}

end time_to_fill_one_barrel_with_leak_l231_231076


namespace total_sample_size_l231_231571

theorem total_sample_size
    (undergrad_count : ℕ) (masters_count : ℕ) (doctoral_count : ℕ)
    (total_students : ℕ) (sample_size_doctoral : ℕ) (proportion_sample : ℕ)
    (n : ℕ)
    (H1 : undergrad_count = 12000)
    (H2 : masters_count = 1000)
    (H3 : doctoral_count = 200)
    (H4 : total_students = undergrad_count + masters_count + doctoral_count)
    (H5 : sample_size_doctoral = 20)
    (H6 : proportion_sample = sample_size_doctoral / doctoral_count)
    (H7 : n = proportion_sample * total_students) :
  n = 1320 := 
sorry

end total_sample_size_l231_231571


namespace find_number_subtract_four_l231_231194

theorem find_number_subtract_four (x : ℤ) (h : 35 + 3 * x = 50) : x - 4 = 1 := by
  sorry

end find_number_subtract_four_l231_231194


namespace community_members_after_five_years_l231_231123

theorem community_members_after_five_years:
  ∀ (a : ℕ → ℕ),
  a 0 = 20 →
  (∀ k : ℕ, a (k + 1) = 4 * a k - 15) →
  a 5 = 15365 :=
by
  intros a h₀ h₁
  sorry

end community_members_after_five_years_l231_231123


namespace decimal_between_0_996_and_0_998_ne_0_997_l231_231537

theorem decimal_between_0_996_and_0_998_ne_0_997 :
  ∃ x : ℝ, 0.996 < x ∧ x < 0.998 ∧ x ≠ 0.997 :=
by
  sorry

end decimal_between_0_996_and_0_998_ne_0_997_l231_231537


namespace subtraction_of_decimals_l231_231560

theorem subtraction_of_decimals : 7.42 - 2.09 = 5.33 := 
by
  sorry

end subtraction_of_decimals_l231_231560


namespace total_pages_in_book_l231_231371

-- Conditions
def hours_reading := 5
def pages_read := 2323
def increase_per_hour := 10
def extra_pages_read := 90

-- Main statement to prove
theorem total_pages_in_book (T : ℕ) :
  (∃ P : ℕ, P + (P + increase_per_hour) + (P + 2 * increase_per_hour) + 
   (P + 3 * increase_per_hour) + (P + 4 * increase_per_hour) = pages_read) ∧
  (pages_read = T - pages_read + extra_pages_read) →
  T = 4556 :=
by { sorry }

end total_pages_in_book_l231_231371


namespace initial_puppies_count_l231_231847

-- Define the initial conditions
def initial_birds : Nat := 12
def initial_cats : Nat := 5
def initial_spiders : Nat := 15
def initial_total_animals : Nat := 25
def half_birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_lost : Nat := 7

-- Define the remaining animals
def remaining_birds : Nat := initial_birds - half_birds_sold
def remaining_cats : Nat := initial_cats
def remaining_spiders : Nat := initial_spiders - spiders_lost

-- Define the total number of remaining animals excluding puppies
def remaining_non_puppy_animals : Nat := remaining_birds + remaining_cats + remaining_spiders

-- Define the remaining puppies
def remaining_puppies : Nat := initial_total_animals - remaining_non_puppy_animals
def initial_puppies : Nat := remaining_puppies + puppies_adopted

-- State the theorem
theorem initial_puppies_count :
  ∀ puppies : Nat, initial_puppies = 9 :=
by
  sorry

end initial_puppies_count_l231_231847


namespace initial_mixture_amount_l231_231664

/-- A solution initially contains an unknown amount of a mixture consisting of 15% sodium chloride
(NaCl), 30% potassium chloride (KCl), 35% sugar, and 20% water. To this mixture, 50 grams of sodium chloride
and 80 grams of potassium chloride are added. If the new salt content of the solution (NaCl and KCl combined)
is 47.5%, how many grams of the mixture were present initially?

Given:
  * The initial mixture consists of 15% NaCl and 30% KCl.
  * 50 grams of NaCl and 80 grams of KCl are added.
  * The new mixture has 47.5% NaCl and KCl combined.
  
Prove that the initial amount of the mixture was 2730 grams. -/
theorem initial_mixture_amount
    (x : ℝ)
    (h_initial_mixture : 0.15 * x + 50 + 0.30 * x + 80 = 0.475 * (x + 130)) :
    x = 2730 := by
  sorry

end initial_mixture_amount_l231_231664


namespace additional_chicken_wings_l231_231884

theorem additional_chicken_wings (friends : ℕ) (wings_per_friend : ℕ) (initial_wings : ℕ) (H1 : friends = 9) (H2 : wings_per_friend = 3) (H3 : initial_wings = 2) : 
  friends * wings_per_friend - initial_wings = 25 := by
  sorry

end additional_chicken_wings_l231_231884


namespace specimen_exchange_l231_231302

theorem specimen_exchange (x : ℕ) (h : x * (x - 1) = 110) : x * (x - 1) = 110 := by
  exact h

end specimen_exchange_l231_231302


namespace remainder_mod_5_is_0_l231_231490

theorem remainder_mod_5_is_0 :
  (88144 * 88145 + 88146 + 88147 + 88148 + 88149 + 88150) % 5 = 0 := by
  sorry

end remainder_mod_5_is_0_l231_231490


namespace find_min_values_l231_231748

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 - 2 * x * y + 6 * y^2 - 14 * x - 6 * y + 72

theorem find_min_values :
  (∀x y : ℝ, f x y ≥ f (15 / 2) (1 / 2)) ∧ f (15 / 2) (1 / 2) = 22.5 :=
by
  sorry

end find_min_values_l231_231748


namespace cloud9_total_money_l231_231808

-- Definitions
def I : ℕ := 12000
def G : ℕ := 16000
def R : ℕ := 1600

-- Total money taken after cancellations
def T : ℕ := I + G - R

-- Theorem stating that T is 26400
theorem cloud9_total_money : T = 26400 := by
  unfold T I G R
  sorry

end cloud9_total_money_l231_231808


namespace train_passes_man_in_12_seconds_l231_231185

noncomputable def time_to_pass_man (train_length: ℝ) (train_speed_kmph: ℝ) (man_speed_kmph: ℝ) : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18)
  train_length / relative_speed_mps

theorem train_passes_man_in_12_seconds :
  time_to_pass_man 220 60 6 = 12 := by
 sorry

end train_passes_man_in_12_seconds_l231_231185


namespace train_speed_l231_231450

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : length = 55) 
    (h2 : time = 5.5) 
    (h3 : speed = (length / time) * (3600 / 1000)) : 
    speed = 36 :=
sorry

end train_speed_l231_231450


namespace total_calories_consumed_l231_231528

def caramel_cookies := 10
def caramel_calories := 18

def chocolate_chip_cookies := 8
def chocolate_chip_calories := 22

def peanut_butter_cookies := 7
def peanut_butter_calories := 24

def selected_caramel_cookies := 5
def selected_chocolate_chip_cookies := 3
def selected_peanut_butter_cookies := 2

theorem total_calories_consumed : 
  (selected_caramel_cookies * caramel_calories) + 
  (selected_chocolate_chip_cookies * chocolate_chip_calories) + 
  (selected_peanut_butter_cookies * peanut_butter_calories) = 204 := 
by
  sorry

end total_calories_consumed_l231_231528


namespace soda_price_increase_l231_231173

theorem soda_price_increase (P : ℝ) (h1 : 1.5 * P = 6) : P = 4 :=
by
  -- Proof will be provided here
  sorry

end soda_price_increase_l231_231173


namespace total_quarters_l231_231246

def Sara_initial_quarters : Nat := 21
def quarters_given_by_dad : Nat := 49

theorem total_quarters : Sara_initial_quarters + quarters_given_by_dad = 70 := 
by
  sorry

end total_quarters_l231_231246


namespace first_term_of_geometric_sequence_l231_231360

-- Define a geometric sequence
def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Initialize conditions
variable (a r : ℝ)

-- Provided that the 3rd term and the 6th term
def third_term : Prop := geometric_sequence a r 2 = 5
def sixth_term : Prop := geometric_sequence a r 5 = 40

-- The theorem to prove that a == 5/4 given the conditions
theorem first_term_of_geometric_sequence : third_term a r ∧ sixth_term a r → a = 5 / 4 :=
by 
  sorry

end first_term_of_geometric_sequence_l231_231360


namespace minimum_loadings_to_prove_first_ingot_weighs_1kg_l231_231980

theorem minimum_loadings_to_prove_first_ingot_weighs_1kg :
  ∀ (w : Fin 11 → ℕ), 
    (∀ i, w i = i + 1) →
    (∃ s₁ s₂ : Finset (Fin 11), 
       s₁.card ≤ 6 ∧ s₂.card ≤ 6 ∧ 
       s₁.sum w = 11 ∧ s₂.sum w = 11 ∧ 
       (∀ s : Finset (Fin 11), s.sum w = 11 → s ≠ s₁ ∧ s ≠ s₂) ∧
       (w 0 = 1)) := sorry -- Fill in the proof here

end minimum_loadings_to_prove_first_ingot_weighs_1kg_l231_231980


namespace angle_C_measure_l231_231798

-- We define angles and the specific conditions given in the problem.
def measure_angle_A : ℝ := 80
def external_angle_C : ℝ := 100

theorem angle_C_measure :
  ∃ (C : ℝ) (A B : ℝ), (A + B = measure_angle_A) ∧
                       (C + external_angle_C = 180) ∧
                       (external_angle_C = measure_angle_A) →
                       C = 100 :=
by {
  -- skipping proof
  sorry
}

end angle_C_measure_l231_231798


namespace applicants_less_4_years_no_degree_l231_231237

theorem applicants_less_4_years_no_degree
    (total_applicants : ℕ)
    (A : ℕ) 
    (B : ℕ)
    (C : ℕ)
    (D : ℕ)
    (h_total : total_applicants = 30)
    (h_A : A = 10)
    (h_B : B = 18)
    (h_C : C = 9)
    (h_D : total_applicants - (A - C + B - C + C) = D) :
  D = 11 :=
by
  sorry

end applicants_less_4_years_no_degree_l231_231237


namespace theater_ticket_sales_l231_231608

theorem theater_ticket_sales (x y : ℕ) (h1 : x + y = 175) (h2 : 6 * x + 2 * y = 750) : y = 75 :=
sorry

end theater_ticket_sales_l231_231608


namespace average_speed_l231_231735

theorem average_speed
    (distance1 distance2 : ℕ)
    (time1 time2 : ℕ)
    (h1 : distance1 = 100)
    (h2 : distance2 = 80)
    (h3 : time1 = 1)
    (h4 : time2 = 1) :
    (distance1 + distance2) / (time1 + time2) = 90 :=
by
  sorry

end average_speed_l231_231735


namespace total_amount_l231_231820

def g_weight : ℝ := 2.5
def g_price : ℝ := 2.79
def r_weight : ℝ := 1.8
def r_price : ℝ := 3.25
def c_weight : ℝ := 1.2
def c_price : ℝ := 4.90
def o_weight : ℝ := 0.9
def o_price : ℝ := 5.75

theorem total_amount :
  g_weight * g_price + r_weight * r_price + c_weight * c_price + o_weight * o_price = 23.88 := by
  sorry

end total_amount_l231_231820


namespace determine_x_l231_231448

theorem determine_x (x : ℚ) (h : ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) : x = 3 / 2 :=
sorry

end determine_x_l231_231448


namespace find_weights_l231_231472

theorem find_weights (x y z : ℕ) (h1 : x + y + z = 11) (h2 : 3 * x + 7 * y + 14 * z = 108) :
  x = 1 ∧ y = 5 ∧ z = 5 :=
by
  sorry

end find_weights_l231_231472


namespace complex_magnitude_problem_l231_231924

open Complex

theorem complex_magnitude_problem
  (z w : ℂ)
  (hz : abs z = 1)
  (hw : abs w = 2)
  (hzw : abs (z + w) = 3) :
  abs ((1 / z) + (1 / w)) = 3 / 2 :=
by {
  sorry
}

end complex_magnitude_problem_l231_231924


namespace cube_root_of_27_l231_231504

theorem cube_root_of_27 : ∃ x : ℝ, x ^ 3 = 27 ↔ ∃ y : ℝ, y = 3 := by
  sorry

end cube_root_of_27_l231_231504


namespace inequality_solution_l231_231062

theorem inequality_solution (a b : ℝ) :
  (∀ x : ℝ, (-1/2 < x ∧ x < 2) → (ax^2 + bx + 2 > 0)) →
  a + b = 1 :=
by
  sorry

end inequality_solution_l231_231062


namespace abs_pos_of_ne_zero_l231_231891

theorem abs_pos_of_ne_zero (a : ℤ) (h : a ≠ 0) : |a| > 0 := sorry

end abs_pos_of_ne_zero_l231_231891


namespace inequality_solution_l231_231304

theorem inequality_solution (x y : ℝ) : 
  (x^2 - 4 * x * y + 4 * x^2 < x^2) ↔ (x < y ∧ y < 3 * x ∧ x > 0) := 
sorry

end inequality_solution_l231_231304


namespace shifted_function_correct_l231_231054

variable (x : ℝ)

/-- The original function -/
def original_function : ℝ := 3 * x - 4

/-- The function after shifting up by 2 units -/
def shifted_function : ℝ := original_function x + 2

theorem shifted_function_correct :
  shifted_function x = 3 * x - 2 :=
by
  sorry

end shifted_function_correct_l231_231054


namespace number_of_paths_K_to_L_l231_231280

-- Definition of the problem structure
def K : Type := Unit
def A : Type := Unit
def R : Type := Unit
def L : Type := Unit

-- Defining the number of paths between each stage
def paths_from_K_to_A := 2
def paths_from_A_to_R := 4
def paths_from_R_to_L := 8

-- The main theorem stating the number of paths from K to L
theorem number_of_paths_K_to_L : paths_from_K_to_A * 2 * 2 = 8 := by 
  sorry

end number_of_paths_K_to_L_l231_231280


namespace car_speed_constant_l231_231243

theorem car_speed_constant (v : ℝ) (hv : v ≠ 0)
  (condition_1 : (1 / 36) * 3600 = 100) 
  (condition_2 : (1 / v) * 3600 = 120) :
  v = 30 := by
  sorry

end car_speed_constant_l231_231243


namespace average_of_numbers_l231_231870

theorem average_of_numbers (a b c d e : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) (h₄ : d = 11) (h₅ : e = 12) :
  (a + b + c + d + e) / 5 = 10 :=
by
  sorry

end average_of_numbers_l231_231870


namespace find_m_l231_231476

theorem find_m (m : ℝ) 
  (A : ℝ × ℝ := (-2, m))
  (B : ℝ × ℝ := (m, 4))
  (h_slope : ((B.snd - A.snd) / (B.fst - A.fst)) = -2) : 
  m = -8 :=
by 
  sorry

end find_m_l231_231476


namespace max_value_of_f_l231_231633

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem max_value_of_f : ∃ x ∈ Set.Icc (Real.pi / 2) Real.pi, ∀ y ∈ Set.Icc (Real.pi / 2) Real.pi, f y ≤ f x ∧ f x = Real.pi := 
by
  sorry

end max_value_of_f_l231_231633


namespace mark_jump_rope_hours_l231_231647

theorem mark_jump_rope_hours 
    (record : ℕ := 54000)
    (jump_per_second : ℕ := 3)
    (seconds_per_hour : ℕ := 3600)
    (total_jumps_to_break_record : ℕ := 54001)
    (jumps_per_hour : ℕ := jump_per_second * seconds_per_hour) 
    (hours_needed : ℕ := total_jumps_to_break_record / jumps_per_hour) 
    (round_up : ℕ := if total_jumps_to_break_record % jumps_per_hour = 0 then hours_needed else hours_needed + 1) :
    round_up = 5 :=
sorry

end mark_jump_rope_hours_l231_231647


namespace sqrt_meaningful_iff_l231_231204

theorem sqrt_meaningful_iff (x : ℝ) : (3 - x ≥ 0) ↔ (x ≤ 3) := by
  sorry

end sqrt_meaningful_iff_l231_231204


namespace action_figure_value_l231_231740

theorem action_figure_value (
    V1 V2 V3 V4 : ℝ
) : 5 * 15 = 75 ∧ 
    V1 - 5 + V2 - 5 + V3 - 5 + V4 - 5 + (20 - 5) = 55 ∧
    V1 + V2 + V3 + V4 + 20 = 80 → 
    ∀ i, i = 15 := by
    sorry

end action_figure_value_l231_231740


namespace males_watch_tvxy_l231_231510

-- Defining the conditions
def total_watch := 160
def females_watch := 75
def males_dont_watch := 83
def total_dont_watch := 120

-- Proving that the number of males who watch TVXY equals 85
theorem males_watch_tvxy : (total_watch - females_watch) = 85 :=
by sorry

end males_watch_tvxy_l231_231510


namespace translate_point_correct_l231_231473

-- Define initial point
def initial_point : ℝ × ℝ := (0, 1)

-- Define translation downward
def translate_down (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 - units)

-- Define translation to the left
def translate_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Define the expected resulting point
def expected_point : ℝ × ℝ := (-4, -1)

-- Lean statement to prove the equivalence
theorem translate_point_correct :
  (translate_left (translate_down initial_point 2) 4) = expected_point :=
by 
  -- Here, we would prove it step by step if required
  sorry

end translate_point_correct_l231_231473


namespace translate_line_up_l231_231333

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := 2 * x - 4

-- Define the new line equation after translating upwards by 5 units
def new_line (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement to prove the translation result
theorem translate_line_up (x : ℝ) : original_line x + 5 = new_line x :=
by
  -- This would normally be where the proof goes, but we'll insert a placeholder
  sorry

end translate_line_up_l231_231333


namespace initial_alcohol_solution_percentage_l231_231432

noncomputable def initial_percentage_of_alcohol (P : ℝ) :=
  let initial_volume := 6 -- initial volume of solution in liters
  let added_alcohol := 1.2 -- added volume of pure alcohol in liters
  let final_volume := initial_volume + added_alcohol -- final volume in liters
  let final_percentage := 0.5 -- final percentage of alcohol
  ∃ P, (initial_volume * (P / 100) + added_alcohol) / final_volume = final_percentage

theorem initial_alcohol_solution_percentage : initial_percentage_of_alcohol 40 :=
by 
  -- Prove that initial percentage P is 40
  have hs : initial_percentage_of_alcohol 40 := by sorry
  exact hs

end initial_alcohol_solution_percentage_l231_231432


namespace num_of_winnable_players_l231_231108

noncomputable def num_players := 2 ^ 2013

def can_win_if (x y : Nat) : Prop := x ≤ y + 3

def single_elimination_tournament (players : Nat) : Nat :=
  -- Function simulating the single elimination based on the specified can_win_if condition
  -- Assuming the given conditions and returning the number of winnable players directly
  6038

theorem num_of_winnable_players : single_elimination_tournament num_players = 6038 :=
  sorry

end num_of_winnable_players_l231_231108


namespace inequality_solution_range_l231_231938

theorem inequality_solution_range (a : ℝ) : (∃ x : ℝ, |x+2| + |x-3| < a) ↔ a > 5 :=
by
  sorry

end inequality_solution_range_l231_231938


namespace janet_has_five_dimes_l231_231833

theorem janet_has_five_dimes (n d q : ℕ) 
    (h1 : n + d + q = 10) 
    (h2 : d + q = 7) 
    (h3 : n + d = 8) : 
    d = 5 :=
by
  -- Proof omitted
  sorry

end janet_has_five_dimes_l231_231833


namespace fred_red_marbles_l231_231228

theorem fred_red_marbles (total_marbles : ℕ) (dark_blue_marbles : ℕ) (green_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = 63) 
  (h2 : dark_blue_marbles ≥ total_marbles / 3)
  (h3 : green_marbles = 4)
  (h4 : red_marbles = total_marbles - dark_blue_marbles - green_marbles) : 
  red_marbles = 38 := 
sorry

end fred_red_marbles_l231_231228


namespace no_zero_root_l231_231047

theorem no_zero_root (x : ℝ) :
  (¬ (∃ x : ℝ, (4 * x ^ 2 - 3 = 49) ∧ x = 0)) ∧
  (¬ (∃ x : ℝ, (x ^ 2 - x - 20 = 0) ∧ x = 0)) :=
by
  sorry

end no_zero_root_l231_231047


namespace isosceles_triangle_perimeter_l231_231866

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 11) (h3 : a = b ∨ b = b) :
  (5 + 11 + 11 = 27) := 
by {
  sorry
}

end isosceles_triangle_perimeter_l231_231866


namespace eval_expr_l231_231443

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l231_231443


namespace range_of_a_l231_231817

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l231_231817


namespace solve_system_eq_l231_231619

theorem solve_system_eq (x y z : ℝ) :
    (x^2 - y^2 + z = 64 / (x * y)) ∧
    (y^2 - z^2 + x = 64 / (y * z)) ∧
    (z^2 - x^2 + y = 64 / (x * z)) ↔ 
    (x = 4 ∧ y = 4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = -4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = 4 ∧ z = -4) ∨ 
    (x = 4 ∧ y = -4 ∧ z = -4) := by
  sorry

end solve_system_eq_l231_231619


namespace first_class_seat_count_l231_231290

theorem first_class_seat_count :
  let seats_first_class := 10
  let seats_business_class := 30
  let seats_economy_class := 50
  let people_economy_class := seats_economy_class / 2
  let people_business_and_first := people_economy_class
  let unoccupied_business := 8
  let people_business_class := seats_business_class - unoccupied_business
  people_business_and_first - people_business_class = 3 := by
  sorry

end first_class_seat_count_l231_231290


namespace slope_negative_l231_231148

theorem slope_negative (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → mx1 + 5 > mx2 + 5) → m < 0 :=
by
  sorry

end slope_negative_l231_231148


namespace solve_problem_1_solve_problem_2_l231_231088

/-
Problem 1:
Given the equation 2(x - 1)^2 = 18, prove that x = 4 or x = -2.
-/
theorem solve_problem_1 (x : ℝ) : 2 * (x - 1)^2 = 18 → (x = 4 ∨ x = -2) :=
by
  sorry

/-
Problem 2:
Given the equation x^2 - 4x - 3 = 0, prove that x = 2 + √7 or x = 2 - √7.
-/
theorem solve_problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 → (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_problem_1_solve_problem_2_l231_231088


namespace pastries_count_l231_231119

def C : ℕ := 19
def P : ℕ := C + 112

theorem pastries_count : P = 131 := by
  -- P = 19 + 112
  -- P = 131
  sorry

end pastries_count_l231_231119


namespace solve_for_constants_l231_231157

def f (x : ℤ) (a b c : ℤ) : ℤ :=
if x > 0 then 2 * a * x + 4
else if x = 0 then a + b
else 3 * b * x + 2 * c

theorem solve_for_constants :
  ∃ a b c : ℤ, 
    f 1 a b c = 6 ∧ 
    f 0 a b c = 7 ∧ 
    f (-1) a b c = -4 ∧ 
    a + b + c = 14 :=
by
  sorry

end solve_for_constants_l231_231157


namespace intersection_of_lines_l231_231652

theorem intersection_of_lines :
  ∃ (x y : ℝ), (8 * x + 5 * y = 40) ∧ (3 * x - 10 * y = 15) ∧ (x = 5) ∧ (y = 0) := 
by 
  sorry

end intersection_of_lines_l231_231652


namespace remainder_when_4_pow_2023_div_17_l231_231701

theorem remainder_when_4_pow_2023_div_17 :
  ∀ (x : ℕ), (x = 4) → x^2 ≡ 16 [MOD 17] → x^2023 ≡ 13 [MOD 17] := by
  intros x hx h
  sorry

end remainder_when_4_pow_2023_div_17_l231_231701


namespace base12_remainder_div_7_l231_231793

-- Define the base-12 number 2543 in decimal form
def n : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0

-- Theorem statement: the remainder when n is divided by 7 is 6
theorem base12_remainder_div_7 : n % 7 = 6 := by
  sorry

end base12_remainder_div_7_l231_231793


namespace sufficient_but_not_necessary_condition_circle_l231_231507

theorem sufficient_but_not_necessary_condition_circle {a : ℝ} (h : a = 1) :
  ∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y + a = 0 → (∀ a, a < 2 → (x - 1)^2 + (y + 1)^2 = 2 - a) :=
by
  sorry

end sufficient_but_not_necessary_condition_circle_l231_231507


namespace hyperbola_with_foci_on_y_axis_l231_231367

variable (m n : ℝ)

-- condition stating that mn < 0
def mn_neg : Prop := m * n < 0

-- the main theorem statement
theorem hyperbola_with_foci_on_y_axis (h : mn_neg m n) : 
  (∃ a : ℝ, a > 0 ∧ ∀ x y : ℝ, m * x^2 - m * y^2 = n ↔ y^2 - x^2 = a) :=
sorry

end hyperbola_with_foci_on_y_axis_l231_231367


namespace evaluate_sixth_iteration_of_g_at_2_l231_231422

def g (x : ℤ) : ℤ := x^2 - 4 * x + 1

theorem evaluate_sixth_iteration_of_g_at_2 :
  g (g (g (g (g (g 2))))) = 59162302643740737293922 := by
  sorry

end evaluate_sixth_iteration_of_g_at_2_l231_231422


namespace sheets_in_set_l231_231342

-- Definitions of the conditions
def John_sheets_left (S E : ℕ) : Prop := S - E = 80
def Mary_sheets_used (S E : ℕ) : Prop := S = 4 * E

-- Theorems to prove the number of sheets
theorem sheets_in_set (S E : ℕ) (hJohn : John_sheets_left S E) (hMary : Mary_sheets_used S E) : S = 320 :=
by { 
  sorry 
}

end sheets_in_set_l231_231342


namespace sum_of_integers_is_27_24_or_20_l231_231877

theorem sum_of_integers_is_27_24_or_20 
    (x y : ℕ) 
    (h1 : 0 < x) 
    (h2 : 0 < y) 
    (h3 : x * y + x + y = 119) 
    (h4 : Nat.gcd x y = 1) 
    (h5 : x < 25) 
    (h6 : y < 25) 
    : x + y = 27 ∨ x + y = 24 ∨ x + y = 20 := 
sorry

end sum_of_integers_is_27_24_or_20_l231_231877


namespace seashells_in_jar_at_end_of_month_l231_231093

noncomputable def seashells_in_week (initial: ℕ) (increment: ℕ) (week: ℕ) : ℕ :=
  initial + increment * week

theorem seashells_in_jar_at_end_of_month :
  seashells_in_week 50 20 0 +
  seashells_in_week 50 20 1 +
  seashells_in_week 50 20 2 +
  seashells_in_week 50 20 3 = 320 :=
sorry

end seashells_in_jar_at_end_of_month_l231_231093


namespace number_of_tables_cost_price_l231_231070

theorem number_of_tables_cost_price
  (C S : ℝ)
  (N : ℝ)
  (h1 : N * C = 20 * S)
  (h2 : S = 0.75 * C) :
  N = 15 := by
  -- insert proof here
  sorry

end number_of_tables_cost_price_l231_231070


namespace problem_statement_l231_231184

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem problem_statement : same_terminal_side (-510) 210 :=
by
  sorry

end problem_statement_l231_231184


namespace incorrect_transformation_when_c_zero_l231_231790

theorem incorrect_transformation_when_c_zero {a b c : ℝ} (h : a * c = b * c) (hc : c = 0) : a ≠ b :=
by
  sorry

end incorrect_transformation_when_c_zero_l231_231790


namespace jill_marathon_time_l231_231322

def jack_marathon_distance : ℝ := 42
def jack_marathon_time : ℝ := 6
def speed_ratio : ℝ := 0.7

theorem jill_marathon_time :
  ∃ t_jill : ℝ, (t_jill = jack_marathon_distance / (jack_marathon_distance / jack_marathon_time / speed_ratio)) ∧
  t_jill = 4.2 :=
by
  -- The proof goes here
  sorry

end jill_marathon_time_l231_231322


namespace general_term_sequence_l231_231749

/--
Given the sequence a : ℕ → ℝ such that a 0 = 1/2,
a 1 = 1/4,
a 2 = -1/8,
a 3 = 1/16,
and we observe that
a n = (-(1/2))^n,
prove that this formula holds for all n : ℕ.
-/
theorem general_term_sequence (a : ℕ → ℝ) :
  (∀ n, a n = (-(1/2))^n) :=
sorry

end general_term_sequence_l231_231749


namespace equation_linear_implies_k_equals_neg2_l231_231824

theorem equation_linear_implies_k_equals_neg2 (k : ℤ) (x : ℝ) :
  (k - 2) * x^(abs k - 1) = k + 1 → abs k - 1 = 1 ∧ k - 2 ≠ 0 → k = -2 :=
by
  sorry

end equation_linear_implies_k_equals_neg2_l231_231824


namespace constant_term_expansion_eq_sixty_l231_231970

theorem constant_term_expansion_eq_sixty (a : ℝ) (h : 15 * a = 60) : a = 4 :=
by
  sorry

end constant_term_expansion_eq_sixty_l231_231970


namespace depletion_rate_l231_231021

theorem depletion_rate (initial_value final_value : ℝ) (years: ℕ) (r : ℝ) 
  (h1 : initial_value = 2500)
  (h2 : final_value = 2256.25)
  (h3 : years = 2)
  (h4 : final_value = initial_value * (1 - r) ^ years) :
  r = 0.05 :=
by
  sorry

end depletion_rate_l231_231021


namespace men_in_room_l231_231610
noncomputable def numMenInRoom (x : ℕ) : ℕ := 4 * x + 2

theorem men_in_room (x : ℕ) (h_initial_ratio : true) (h_after_events : true) (h_double_women : 2 * (5 * x - 3) = 24) :
  numMenInRoom x = 14 :=
sorry

end men_in_room_l231_231610


namespace ellipse_properties_l231_231789

noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x * x) / (a * a) + (y * y) / (b * b) = 1

theorem ellipse_properties (a b c k : ℝ) (h_ab : a > b) (h_b : b > 1) (h_c : 2 * c = 2) 
  (h_area : (2 * Real.sqrt 3 / 3)^2 = 4 / 3) (h_slope : k ≠ 0)
  (h_PD : |(c - 4 * k^2 / (3 + 4 * k^2))^2 + (-3 * k / (3 + 4 * k^2))^2| = 3 * Real.sqrt 2 / 7) :
  (ellipse_equation 1 0 a b ∧
   (a = 2 ∧ b = Real.sqrt 3) ∧
   k = 1 ∨ k = -1) :=
by
  -- Prove the standard equation of the ellipse C and the value of k
  sorry

end ellipse_properties_l231_231789


namespace minimum_value_Q_l231_231720

theorem minimum_value_Q (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 47 := 
  sorry

end minimum_value_Q_l231_231720


namespace problem_1_problem_2_l231_231241

noncomputable def f (ω x : ℝ) : ℝ :=
  Real.sin (ω * x + (Real.pi / 4))

theorem problem_1 (ω : ℝ) (hω : ω > 0) : f ω 0 = Real.sqrt 2 / 2 :=
by
  unfold f
  simp [Real.sin_pi_div_four]

theorem problem_2 : 
  ∃ x : ℝ, 
  0 ≤ x ∧ x ≤ Real.pi / 2 ∧ 
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.pi / 2 → f 2 y ≤ f 2 x) ∧ 
  f 2 x = 1 :=
by
  sorry

end problem_1_problem_2_l231_231241


namespace part1_max_min_part2_triangle_inequality_l231_231410

noncomputable def f (x k : ℝ) : ℝ :=
  (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem part1_max_min (k : ℝ): 
  (∀ x : ℝ, k ≥ 1 → 1 ≤ f x k ∧ f x k ≤ (1/3) * (k + 2)) ∧ 
  (∀ x : ℝ, k < 1 → (1/3) * (k + 2) ≤ f x k ∧ f x k ≤ 1) := 
sorry

theorem part2_triangle_inequality (k : ℝ) : 
  -1/2 < k ∧ k < 4 ↔ (∀ a b c : ℝ, (f a k + f b k > f c k) ∧ (f b k + f c k > f a k) ∧ (f c k + f a k > f b k)) :=
sorry

end part1_max_min_part2_triangle_inequality_l231_231410


namespace carrie_harvests_9000_l231_231832

noncomputable def garden_area (length width : ℕ) := length * width
noncomputable def total_plants (plants_per_sqft sqft : ℕ) := plants_per_sqft * sqft
noncomputable def total_cucumbers (yield_plants plants : ℕ) := yield_plants * plants

theorem carrie_harvests_9000 :
  garden_area 10 12 = 120 →
  total_plants 5 120 = 600 →
  total_cucumbers 15 600 = 9000 :=
by sorry

end carrie_harvests_9000_l231_231832


namespace sodium_chloride_formed_l231_231046

section 

-- Definitions based on the conditions
def hydrochloric_acid_moles : ℕ := 2
def sodium_bicarbonate_moles : ℕ := 2

-- Balanced chemical equation represented as a function (1:1 reaction ratio)
def reaction (hcl_moles naHCO3_moles : ℕ) : ℕ := min hcl_moles naHCO3_moles

-- Theorem stating the reaction outcome
theorem sodium_chloride_formed : reaction hydrochloric_acid_moles sodium_bicarbonate_moles = 2 :=
by
  -- Proof is omitted
  sorry

end

end sodium_chloride_formed_l231_231046


namespace drum_oil_ratio_l231_231491

theorem drum_oil_ratio (C_X C_Y : ℝ) (h1 : (1 / 2) * C_X + (1 / 5) * C_Y = 0.45 * C_Y) : 
  C_Y / C_X = 2 :=
by
  -- Cannot provide the proof
  sorry

end drum_oil_ratio_l231_231491


namespace find_angle_2_l231_231683

theorem find_angle_2 (angle1 : ℝ) (angle2 : ℝ) 
  (h1 : angle1 = 60) 
  (h2 : angle1 + angle2 = 180) : 
  angle2 = 120 := 
by
  sorry

end find_angle_2_l231_231683


namespace correct_statements_l231_231503

theorem correct_statements (a b c : ℝ) (h : ∀ x, ax^2 + bx + c > 0 ↔ -2 < x ∧ x < 3) :
  ( ∃ (x : ℝ), c*x^2 + b*x + a < 0 ↔ -1/2 < x ∧ x < 1/3 ) ∧
  ( ∃ (b : ℝ), ∀ b, 12/(3*b + 4) + b = 8/3 ) ∧
  ( ∀ m, ¬ (m < -1 ∨ m > 2) ) ∧
  ( c = 2 → ∀ n1 n2, (3*a*n1^2 + 6*b*n1 = -3 ∧ 3*a*n2^2 + 6*b*n2 = 1) → n2 - n1 ∈ [2, 4] ) :=
sorry

end correct_statements_l231_231503


namespace inequality_solution_set_l231_231682

theorem inequality_solution_set
  (a b c m n : ℝ) (h : a ≠ 0) 
  (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ m < x ∧ x < n)
  (h2 : 0 < m)
  (h3 : ∀ x : ℝ, cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) :
  (cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) := 
sorry

end inequality_solution_set_l231_231682


namespace adam_earnings_l231_231956

def lawns_to_mow : ℕ := 12
def lawns_forgotten : ℕ := 8
def earnings_per_lawn : ℕ := 9

theorem adam_earnings : (lawns_to_mow - lawns_forgotten) * earnings_per_lawn = 36 := by
  sorry

end adam_earnings_l231_231956


namespace perpendicular_lines_l231_231915

theorem perpendicular_lines (a : ℝ) :
  (∃ l₁ l₂ : ℝ, 2 * l₁ + l₂ + 1 = 0 ∧ l₁ + a * l₂ + 3 = 0 ∧ 2 * l₁ + 1 * l₂ + 1 * a = 0) → a = -2 :=
by
  sorry

end perpendicular_lines_l231_231915


namespace total_ticket_income_l231_231059

-- All given conditions as definitions/assumptions
def total_seats : ℕ := 200
def children_tickets : ℕ := 60
def adult_ticket_price : ℝ := 3.00
def children_ticket_price : ℝ := 1.50
def adult_tickets : ℕ := total_seats - children_tickets

-- The claim we need to prove
theorem total_ticket_income :
  (adult_tickets * adult_ticket_price + children_tickets * children_ticket_price) = 510.00 :=
by
  -- Placeholder to complete proof later
  sorry

end total_ticket_income_l231_231059


namespace magnitude_of_angle_A_range_of_b_plus_c_l231_231724

--- Definitions for the conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition a / (sqrt 3 * cos A) = c / sin C
axiom condition1 : a / (Real.sqrt 3 * Real.cos A) = c / Real.sin C

-- Given a = 6
axiom condition2 : a = 6

-- Conditions for sides b and c being positive
axiom condition3 : b > 0
axiom condition4 : c > 0
-- Condition for triangle inequality
axiom condition5 : b + c > a

-- Part (I) Find the magnitude of angle A
theorem magnitude_of_angle_A : A = Real.pi / 3 :=
by
  sorry

-- Part (II) Determine the range of values for b + c given a = 6
theorem range_of_b_plus_c : 6 < b + c ∧ b + c ≤ 12 :=
by
  sorry

end magnitude_of_angle_A_range_of_b_plus_c_l231_231724


namespace flowers_are_55_percent_daisies_l231_231023

noncomputable def percent_daisies (F : ℝ) (yellow : ℝ) (white_daisies : ℝ) (yellow_daisies : ℝ) : ℝ :=
  (yellow_daisies + white_daisies) / F * 100

theorem flowers_are_55_percent_daisies (F : ℝ) (yellow_t : ℝ) (yellow_d : ℝ) (white : ℝ) (white_d : ℝ) :
    yellow_t = 0.5 * yellow →
    yellow_d = yellow - yellow_t →
    white_d = (2 / 3) * white →
    yellow = (7 / 10) * F →
    white = F - yellow →
    percent_daisies F yellow white_d yellow_d = 55 :=
by
  sorry

end flowers_are_55_percent_daisies_l231_231023


namespace cost_per_sqft_is_6_l231_231878

-- Define the dimensions of the room
def room_length : ℕ := 25
def room_width : ℕ := 15
def room_height : ℕ := 12

-- Define the dimensions of the door
def door_height : ℕ := 6
def door_width : ℕ := 3

-- Define the dimensions of the windows
def window_height : ℕ := 4
def window_width : ℕ := 3
def number_of_windows : ℕ := 3

-- Define the total cost of whitewashing
def total_cost : ℕ := 5436

-- Calculate areas
def area_one_pair_of_walls : ℕ :=
  (room_length * room_height) * 2

def area_other_pair_of_walls : ℕ :=
  (room_width * room_height) * 2

def total_wall_area : ℕ :=
  area_one_pair_of_walls + area_other_pair_of_walls

def door_area : ℕ :=
  door_height * door_width

def window_area : ℕ :=
  window_height * window_width

def total_window_area : ℕ :=
  window_area * number_of_windows

def area_to_be_whitewashed : ℕ :=
  total_wall_area - (door_area + total_window_area)

def cost_per_sqft : ℕ :=
  total_cost / area_to_be_whitewashed

-- The theorem statement proving the cost per square foot is 6
theorem cost_per_sqft_is_6 : cost_per_sqft = 6 := 
  by
  -- Proof goes here
  sorry

end cost_per_sqft_is_6_l231_231878


namespace complement_P_subset_PQ_intersection_PQ_eq_Q_l231_231658

open Set

variable {R : Type*} [OrderedCommRing R]

def P (x : R) : Prop := -2 ≤ x ∧ x ≤ 10
def Q (m x : R) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem complement_P : (compl (setOf P)) = {x | x < -2} ∪ {x | x > 10} :=
by {
  sorry
}

theorem subset_PQ (m : R) : (∀ x, P x → Q m x) ↔ m ≥ 9 :=
by {
  sorry
}

theorem intersection_PQ_eq_Q (m : R) : (∀ x, Q m x → P x) ↔ m ≤ 9 :=
by {
  sorry
}

end complement_P_subset_PQ_intersection_PQ_eq_Q_l231_231658


namespace one_room_cheaper_by_l231_231149

-- Define the initial prices of the apartments
variables (a b : ℝ)

-- Define the increase rates and the new prices
def new_price_one_room := 1.21 * a
def new_price_two_room := 1.11 * b
def new_total_price := 1.15 * (a + b)

-- The main theorem encapsulating the problem
theorem one_room_cheaper_by : a + b ≠ 0 → 1.21 * a + 1.11 * b = 1.15 * (a + b) → b / a = 1.5 :=
by
  intro h_non_zero h_prices
  -- we assume the main theorem is true to structure the goal state
  sorry

end one_room_cheaper_by_l231_231149


namespace polynomial_expression_l231_231794

theorem polynomial_expression :
  (2 * x^2 + 3 * x + 7) * (x + 1) - (x + 1) * (x^2 + 4 * x - 63) + (3 * x - 14) * (x + 1) * (x + 5) = 4 * x^3 + 4 * x^2 :=
by
  sorry

end polynomial_expression_l231_231794


namespace intersection_points_calculation_l231_231993

-- Define the quadratic function and related functions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def u (a b c x : ℝ) : ℝ := - f a b c (-x)
def v (a b c x : ℝ) : ℝ := f a b c (x + 1)

-- Define the number of intersection points
def m : ℝ := 1
def n : ℝ := 0

-- The proof goal
theorem intersection_points_calculation (a b c : ℝ) : 7 * m + 3 * n = 7 :=
by sorry

end intersection_points_calculation_l231_231993


namespace joe_time_to_store_l231_231128

theorem joe_time_to_store :
  ∀ (r_w : ℝ) (r_r : ℝ) (t_w t_r t_total : ℝ), 
   (r_r = 2 * r_w) → (t_w = 10) → (t_r = t_w / 2) → (t_total = t_w + t_r) → (t_total = 15) := 
by
  intros r_w r_r t_w t_r t_total hrw hrw_eq hr_tw hr_t_total
  sorry

end joe_time_to_store_l231_231128


namespace chord_length_of_concentric_circles_l231_231631

theorem chord_length_of_concentric_circles 
  (R r : ℝ) (h1 : R^2 - r^2 = 15) (h2 : ∀ s, s = 2 * R) :
  ∃ c : ℝ, c = 2 * Real.sqrt 15 ∧ ∀ x, x = c := 
by 
  sorry

end chord_length_of_concentric_circles_l231_231631


namespace cost_per_kg_paint_l231_231020

-- Define the basic parameters
variables {sqft_per_kg : ℝ} -- the area covered by 1 kg of paint
variables {total_cost : ℝ} -- the total cost to paint the cube
variables {side_length : ℝ} -- the side length of the cube
variables {num_faces : ℕ} -- the number of faces of the cube

-- Define the conditions given in the problem
def conditions (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) : Prop :=
  sqft_per_kg = 16 ∧
  total_cost = 876 ∧
  side_length = 8 ∧
  num_faces = 6

-- Define the statement to prove, which is the cost per kg of paint
theorem cost_per_kg_paint (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) :
  conditions sqft_per_kg total_cost side_length num_faces →
  ∃ cost_per_kg : ℝ, cost_per_kg = 36.5 :=
by
  sorry

end cost_per_kg_paint_l231_231020


namespace select_students_l231_231098

-- Definitions for the conditions
variables (A B C D E : Prop)

-- Conditions
def condition1 : Prop := A → B ∧ ¬E
def condition2 : Prop := (B ∨ E) → ¬D
def condition3 : Prop := C ∨ D

-- The main theorem
theorem select_students (hA : A) (h1 : condition1 A B E) (h2 : condition2 B E D) (h3 : condition3 C D) : B ∧ C :=
by 
  sorry

end select_students_l231_231098


namespace abs_sum_less_b_l231_231354

theorem abs_sum_less_b (x : ℝ) (b : ℝ) (h : |2 * x - 8| + |2 * x - 6| < b) (hb : b > 0) : b > 2 :=
by
  sorry

end abs_sum_less_b_l231_231354


namespace vector_intersecting_line_parameter_l231_231948

theorem vector_intersecting_line_parameter :
  ∃ (a b s : ℝ), a = 3 * s + 5 ∧ b = 2 * s + 4 ∧
                   (∃ r, (a, b) = (3 * r, 2 * r)) ∧
                   (a, b) = (6, 14 / 3) :=
by
  sorry

end vector_intersecting_line_parameter_l231_231948


namespace second_player_can_form_palindrome_l231_231912

def is_palindrome (s : List Char) : Prop :=
  s = s.reverse

theorem second_player_can_form_palindrome :
  ∀ (moves : List Char), moves.length = 1999 →
  ∃ (sequence : List Char), sequence.length = 1999 ∧ is_palindrome sequence :=
by
  sorry

end second_player_can_form_palindrome_l231_231912


namespace right_triangle_inradius_l231_231289

theorem right_triangle_inradius (a b c : ℕ) (h : a = 6) (h2 : b = 8) (h3 : c = 10) :
  ((a^2 + b^2 = c^2) ∧ (1/2 * ↑a * ↑b = 24) ∧ ((a + b + c) / 2 = 12) ∧ (24 = 12 * 2)) :=
by 
  sorry

end right_triangle_inradius_l231_231289


namespace find_n_satisfies_equation_l231_231886

-- Definition of the problem:
def satisfies_equation (n : ℝ) : Prop := 
  (2 / (n + 1)) + (3 / (n + 1)) + (n / (n + 1)) = 4

-- The statement of the proof problem:
theorem find_n_satisfies_equation : 
  ∃ n : ℝ, satisfies_equation n ∧ n = 1/3 :=
by
  sorry

end find_n_satisfies_equation_l231_231886


namespace find_R_l231_231979

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 → m < n → ¬ (m ∣ n)

theorem find_R :
  ∃ R : ℤ, R > 0 ∧ (∃ Q : ℤ, is_prime (R^3 + 4 * R^2 + (Q - 93) * R + 14 * Q + 10)) ∧ R = 5 :=
  sorry

end find_R_l231_231979


namespace valid_five_digit_integers_l231_231952

/-- How many five-digit positive integers can be formed by arranging the digits 1, 1, 2, 3, 4 so 
that the two 1s are not next to each other -/
def num_valid_arrangements : ℕ :=
  36

theorem valid_five_digit_integers :
  ∃ n : ℕ, n = num_valid_arrangements :=
by
  use 36
  sorry

end valid_five_digit_integers_l231_231952


namespace find_parallelepiped_dimensions_l231_231723

theorem find_parallelepiped_dimensions :
  ∃ (x y z : ℕ),
    (x * y * z = 2 * (x * y + y * z + z * x)) ∧
    (x = 6 ∧ y = 6 ∧ z = 6 ∨
     x = 5 ∧ y = 5 ∧ z = 10 ∨
     x = 4 ∧ y = 8 ∧ z = 8 ∨
     x = 3 ∧ y = 12 ∧ z = 12 ∨
     x = 3 ∧ y = 7 ∧ z = 42 ∨
     x = 3 ∧ y = 8 ∧ z = 24 ∨
     x = 3 ∧ y = 9 ∧ z = 18 ∨
     x = 3 ∧ y = 10 ∧ z = 15 ∨
     x = 4 ∧ y = 5 ∧ z = 20 ∨
     x = 4 ∧ y = 6 ∧ z = 12) :=
by
  sorry

end find_parallelepiped_dimensions_l231_231723


namespace sqrt_of_second_number_l231_231492

-- Given condition: the arithmetic square root of a natural number n is x
variable (x : ℕ)
def first_number := x ^ 2
def second_number := first_number + 1

-- The theorem statement we want to prove
theorem sqrt_of_second_number (x : ℕ) : Real.sqrt (x^2 + 1) = Real.sqrt (first_number x + 1) :=
by
  sorry

end sqrt_of_second_number_l231_231492


namespace largest_root_of_quadratic_l231_231060

theorem largest_root_of_quadratic :
  ∀ (x : ℝ), x^2 - 9*x - 22 = 0 → x ≤ 11 :=
by
  sorry

end largest_root_of_quadratic_l231_231060


namespace cyc_inequality_l231_231788

theorem cyc_inequality (x y z : ℝ) (hx : 0 < x ∧ x < 2) (hy : 0 < y ∧ y < 2) (hz : 0 < z ∧ z < 2) 
  (hxyz : x^2 + y^2 + z^2 = 3) : 
  3 / 2 < (1 + y^2) / (x + 2) + (1 + z^2) / (y + 2) + (1 + x^2) / (z + 2) ∧ 
  (1 + y^2) / (x + 2) + (1 + z^2) / (y + 2) + (1 + x^2) / (z + 2) < 3 := 
by
  sorry

end cyc_inequality_l231_231788


namespace find_subtracted_value_l231_231585

theorem find_subtracted_value (N : ℕ) (V : ℕ) (hN : N = 2976) (h : (N / 12) - V = 8) : V = 240 := by
  sorry

end find_subtracted_value_l231_231585


namespace rational_root_even_denominator_l231_231961

theorem rational_root_even_denominator
  (a b c : ℤ)
  (sum_ab_even : (a + b) % 2 = 0)
  (c_odd : c % 2 = 1) :
  ∀ (p q : ℤ), (q ≠ 0) → (IsRationalRoot : a * (p * p) + b * p * q + c * (q * q) = 0) →
    gcd p q = 1 → q % 2 = 0 :=
by
  sorry

end rational_root_even_denominator_l231_231961


namespace radius_area_tripled_l231_231715

theorem radius_area_tripled (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = (n * (Real.sqrt 3 - 1)) / 2 :=
by {
  sorry
}

end radius_area_tripled_l231_231715


namespace recurring_decimal_sum_l231_231257

theorem recurring_decimal_sum :
  let x := (4 / 33)
  let y := (34 / 99)
  x + y = (46 / 99) := by
  sorry

end recurring_decimal_sum_l231_231257


namespace range_of_a_l231_231295

variable (a x : ℝ)

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def M (a : ℝ) : Set ℝ := if a = 2 then {2} else {x | 2 ≤ x ∧ x ≤ a}

theorem range_of_a (a : ℝ) (p : x ∈ M a) (h : a ≥ 2) (hpq : Set.Subset (M a) A) : 2 ≤ a ∧ a ≤ 4 :=
  sorry

end range_of_a_l231_231295


namespace car_C_has_highest_average_speed_l231_231195

-- Define the distances traveled by each car
def distance_car_A_1st_hour := 140
def distance_car_A_2nd_hour := 130
def distance_car_A_3rd_hour := 120

def distance_car_B_1st_hour := 170
def distance_car_B_2nd_hour := 90
def distance_car_B_3rd_hour := 130

def distance_car_C_1st_hour := 120
def distance_car_C_2nd_hour := 140
def distance_car_C_3rd_hour := 150

-- Define the total distance and average speed calculations
def total_distance_car_A := distance_car_A_1st_hour + distance_car_A_2nd_hour + distance_car_A_3rd_hour
def total_distance_car_B := distance_car_B_1st_hour + distance_car_B_2nd_hour + distance_car_B_3rd_hour
def total_distance_car_C := distance_car_C_1st_hour + distance_car_C_2nd_hour + distance_car_C_3rd_hour

def total_time := 3

def average_speed_car_A := total_distance_car_A / total_time
def average_speed_car_B := total_distance_car_B / total_time
def average_speed_car_C := total_distance_car_C / total_time

-- Lean proof statement
theorem car_C_has_highest_average_speed :
  average_speed_car_C > average_speed_car_A ∧ average_speed_car_C > average_speed_car_B :=
by
  sorry

end car_C_has_highest_average_speed_l231_231195


namespace graph_of_equation_is_two_intersecting_lines_l231_231016

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x - 2 * y)^2 = x^2 + y^2 ↔ (y = 0 ∨ y = 4 / 3 * x) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l231_231016


namespace vendor_second_day_sale_l231_231213

theorem vendor_second_day_sale (n : ℕ) :
  let sold_first_day := (50 * n) / 100
  let remaining_after_first_sale := n - sold_first_day
  let thrown_away_first_day := (20 * remaining_after_first_sale) / 100
  let remaining_after_first_day := remaining_after_first_sale - thrown_away_first_day
  let total_thrown_away := (30 * n) / 100
  let thrown_away_second_day := total_thrown_away - thrown_away_first_day
  let sold_second_day := remaining_after_first_day - thrown_away_second_day
  let percent_sold_second_day := (sold_second_day * 100) / remaining_after_first_day
  percent_sold_second_day = 50 :=
sorry

end vendor_second_day_sale_l231_231213


namespace team_a_completion_rate_l231_231741

theorem team_a_completion_rate :
  ∃ x : ℝ, (9000 / x - 9000 / (1.5 * x) = 15) ∧ x = 200 :=
by {
  sorry
}

end team_a_completion_rate_l231_231741


namespace perpendicular_slope_l231_231363

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end perpendicular_slope_l231_231363


namespace seventh_degree_solution_l231_231545

theorem seventh_degree_solution (a b x : ℝ) :
  (x^7 - 7 * a * x^5 + 14 * a^2 * x^3 - 7 * a^3 * x = b) ↔
  ∃ α β : ℝ, α + β = x ∧ α * β = a ∧ α^7 + β^7 = b :=
by
  sorry

end seventh_degree_solution_l231_231545


namespace y1_greater_than_y2_l231_231175

-- Define the function and points
def parabola (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m

-- Define the points A and B on the parabola
def A_y1 (m : ℝ) : ℝ := parabola 0 m
def B_y2 (m : ℝ) : ℝ := parabola 1 m

-- Theorem statement
theorem y1_greater_than_y2 (m : ℝ) : A_y1 m > B_y2 m := 
  sorry

end y1_greater_than_y2_l231_231175


namespace range_of_a_product_greater_than_one_l231_231730

namespace ProofProblem

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + x^2 - a * x + 2

variables {x1 x2 a : ℝ}

-- Conditions
axiom f_has_two_distinct_zeros : f x1 a = 0 ∧ f x2 a = 0 ∧ x1 ≠ x2

-- Goal 1: Prove the range of a
theorem range_of_a : a ∈ Set.Ioi 3 := sorry  -- Formal expression for (3, +∞) in Lean

-- Goal 2: Prove x1 * x2 > 1 given that a is in the correct range
theorem product_greater_than_one (ha : a ∈ Set.Ioi 3) : x1 * x2 > 1 := sorry

end ProofProblem

end range_of_a_product_greater_than_one_l231_231730


namespace max_soap_boxes_l231_231949

theorem max_soap_boxes :
  ∀ (L_carton W_carton H_carton L_soap_box W_soap_box H_soap_box : ℕ)
   (V_carton V_soap_box : ℕ) 
   (h1 : L_carton = 25) 
   (h2 : W_carton = 42)
   (h3 : H_carton = 60) 
   (h4 : L_soap_box = 7)
   (h5 : W_soap_box = 6)
   (h6 : H_soap_box = 10)
   (h7 : V_carton = L_carton * W_carton * H_carton)
   (h8 : V_soap_box = L_soap_box * W_soap_box * H_soap_box),
   V_carton / V_soap_box = 150 :=
by
  intros
  sorry

end max_soap_boxes_l231_231949


namespace number_times_quarter_squared_eq_four_cubed_l231_231395

theorem number_times_quarter_squared_eq_four_cubed : 
  ∃ (number : ℕ), number * (1 / 4 : ℚ) ^ 2 = (4 : ℚ) ^ 3 ∧ number = 1024 :=
by 
  use 1024
  sorry

end number_times_quarter_squared_eq_four_cubed_l231_231395


namespace B_and_C_together_l231_231394

theorem B_and_C_together (A B C : ℕ) (h1 : A + B + C = 1000) (h2 : A + C = 700) (h3 : C = 300) :
  B + C = 600 :=
by
  sorry

end B_and_C_together_l231_231394


namespace angel_vowels_written_l231_231459

theorem angel_vowels_written (num_vowels : ℕ) (times_written : ℕ) (h1 : num_vowels = 5) (h2 : times_written = 4) : num_vowels * times_written = 20 := by
  sorry

end angel_vowels_written_l231_231459


namespace no_solution_for_b_a_divides_a_b_minus_1_l231_231613

theorem no_solution_for_b_a_divides_a_b_minus_1 :
  ¬ (∃ a b : ℕ, 1 ≤ a ∧ 1 ≤ b ∧ b^a ∣ a^b - 1) :=
by
  sorry

end no_solution_for_b_a_divides_a_b_minus_1_l231_231613


namespace problem_solution_l231_231417

open Set

variable {U : Set ℕ} (M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hC : U \ M = {1, 3})

theorem problem_solution : 2 ∈ M :=
by
  sorry

end problem_solution_l231_231417


namespace math_problem_correct_l231_231928

noncomputable def math_problem : Prop :=
  (1 / ((3 / (Real.sqrt 5 + 2)) - (1 / (Real.sqrt 4 + 1)))) = ((27 * Real.sqrt 5 + 57) / 40)

theorem math_problem_correct : math_problem := by
  sorry

end math_problem_correct_l231_231928


namespace division_by_3_l231_231003

theorem division_by_3 (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := 
sorry

end division_by_3_l231_231003


namespace range_of_m_l231_231051

noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then 3 + 3 * x
  else if x <= 3 then -1
  else x + 5

theorem range_of_m (m : ℝ) (x : ℝ) (hx : f x ≥ 1 / m - 4) :
  m < 0 ∨ m = 1 :=
sorry

end range_of_m_l231_231051


namespace minimize_costs_l231_231929

def total_books : ℕ := 150000
def handling_fee_per_order : ℕ := 30
def storage_fee_per_1000_copies : ℕ := 40
def evenly_distributed_books : Prop := true --Assuming books are evenly distributed by default

noncomputable def optimal_order_frequency : ℕ := 10
noncomputable def optimal_batch_size : ℕ := 15000

theorem minimize_costs 
  (handling_fee_per_order : ℕ) 
  (storage_fee_per_1000_copies : ℕ) 
  (total_books : ℕ) 
  (evenly_distributed_books : Prop)
  : optimal_order_frequency = 10 ∧ optimal_batch_size = 15000 := sorry

end minimize_costs_l231_231929


namespace sum_mod_6_l231_231174

theorem sum_mod_6 :
  (60123 + 60124 + 60125 + 60126 + 60127 + 60128 + 60129 + 60130) % 6 = 4 :=
by
  sorry

end sum_mod_6_l231_231174


namespace f_2019_equals_neg2_l231_231597

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 4) = f x)
variable (h_defined : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2)

theorem f_2019_equals_neg2 : f 2019 = -2 :=
by 
  sorry

end f_2019_equals_neg2_l231_231597


namespace mean_weight_participants_l231_231120

def weights_120s := [123, 125]
def weights_130s := [130, 132, 133, 135, 137, 138]
def weights_140s := [141, 145, 145, 149, 149]
def weights_150s := [150, 152, 153, 155, 158]
def weights_160s := [164, 167, 167, 169]

def total_weights := weights_120s ++ weights_130s ++ weights_140s ++ weights_150s ++ weights_160s

def total_sum : ℕ := total_weights.sum
def total_count : ℕ := total_weights.length

theorem mean_weight_participants : (total_sum : ℚ) / total_count = 3217 / 22 := by
  sorry -- Proof goes here, but we're skipping it

end mean_weight_participants_l231_231120


namespace smallest_n_inequality_l231_231150

-- Define the main statement based on the identified conditions and answer.
theorem smallest_n_inequality (x y z w : ℝ) : 
  (x^2 + y^2 + z^2 + w^2)^2 ≤ 4 * (x^4 + y^4 + z^4 + w^4) :=
sorry

end smallest_n_inequality_l231_231150


namespace volume_ratio_of_cubes_l231_231768

theorem volume_ratio_of_cubes (e1 e2 : ℕ) (h1 : e1 = 9) (h2 : e2 = 36) :
  (e1^3 : ℚ) / (e2^3 : ℚ) = 1 / 64 := by
  sorry

end volume_ratio_of_cubes_l231_231768


namespace range_of_a_l231_231043

theorem range_of_a (a : ℝ) (h : ¬ ∃ t : ℝ, t^2 - a * t - a < 0) : -4 ≤ a ∧ a ≤ 0 :=
by 
  sorry

end range_of_a_l231_231043


namespace proof_intersection_l231_231244

def setA : Set ℤ := {x | abs x ≤ 2}

def setB : Set ℝ := {x | x^2 - 2 * x - 8 ≥ 0}

def complementB : Set ℝ := {x | x^2 - 2 * x - 8 < 0}

def intersectionAComplementB : Set ℤ := {x | x ∈ setA ∧ (x : ℝ) ∈ complementB}

theorem proof_intersection : intersectionAComplementB = {-1, 0, 1, 2} := by
  sorry

end proof_intersection_l231_231244


namespace sin_alpha_value_l231_231103

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_value_l231_231103


namespace smallest_n_divisor_lcm_gcd_l231_231815

theorem smallest_n_divisor_lcm_gcd :
  ∀ n : ℕ, n > 0 ∧ (∀ a b : ℕ, 60 = a ∧ n = b → (Nat.lcm a b / Nat.gcd a b = 50)) → n = 750 :=
by
  sorry

end smallest_n_divisor_lcm_gcd_l231_231815


namespace bakery_storage_l231_231783

theorem bakery_storage (S F B : ℕ) 
  (h1 : S * 4 = F * 5) 
  (h2 : F = 10 * B) 
  (h3 : F * 1 = (B + 60) * 8) : S = 3000 :=
sorry

end bakery_storage_l231_231783


namespace P_Q_sum_equals_44_l231_231837

theorem P_Q_sum_equals_44 (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3))) :
  P + Q = 44 :=
sorry

end P_Q_sum_equals_44_l231_231837


namespace watch_cost_price_l231_231996

theorem watch_cost_price (cost_price : ℝ)
  (h1 : SP_loss = 0.90 * cost_price)
  (h2 : SP_gain = 1.08 * cost_price)
  (h3 : SP_gain - SP_loss = 540) :
  cost_price = 3000 := 
sorry

end watch_cost_price_l231_231996


namespace range_of_a_l231_231314

def f (x a : ℝ) := |x - 2| + |x + a|

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 3) → a ≤ -5 ∨ a ≥ 1 :=
  sorry

end range_of_a_l231_231314


namespace thin_film_radius_volume_l231_231205

theorem thin_film_radius_volume :
  ∀ (r : ℝ) (V : ℝ) (t : ℝ), 
    V = 216 → t = 0.1 → π * r^2 * t = V → r = Real.sqrt (2160 / π) :=
by
  sorry

end thin_film_radius_volume_l231_231205


namespace range_of_a_l231_231499

-- Function definition for op
def op (x y : ℝ) : ℝ := x * (2 - y)

-- Predicate that checks the inequality for all t
def inequality_holds_for_all_t (a : ℝ) : Prop :=
  ∀ t : ℝ, (op (t - a) (t + a)) < 1

-- Prove that the range of a is (0, 2)
theorem range_of_a : 
  ∀ a : ℝ, inequality_holds_for_all_t a ↔ 0 < a ∧ a < 2 := 
by
  sorry

end range_of_a_l231_231499


namespace rectangle_side_length_l231_231734

theorem rectangle_side_length (x : ℝ) (h1 : 0 < x) (h2 : 2 * (x + 6) = 40) : x = 14 :=
by
  sorry

end rectangle_side_length_l231_231734


namespace find_k_l231_231391

noncomputable def g (x : ℕ) : ℤ := 2 * x^2 - 8 * x + 8

theorem find_k :
  (g 2 = 0) ∧ 
  (90 < g 9) ∧ (g 9 < 100) ∧
  (120 < g 10) ∧ (g 10 < 130) ∧
  ∃ (k : ℤ), 7000 * k < g 150 ∧ g 150 < 7000 * (k + 1)
  → ∃ (k : ℤ), k = 6 :=
by
  sorry

end find_k_l231_231391


namespace find_x_minus_y_l231_231315

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : x - y = 3 / 2 :=
by
  sorry

end find_x_minus_y_l231_231315


namespace prob_less_than_9_l231_231508

def prob_10 : ℝ := 0.24
def prob_9 : ℝ := 0.28
def prob_8 : ℝ := 0.19

theorem prob_less_than_9 : prob_10 + prob_9 + prob_8 < 1 → 1 - prob_10 - prob_9 = 0.48 := 
by {
  sorry
}

end prob_less_than_9_l231_231508


namespace snowboard_price_after_discounts_l231_231102

theorem snowboard_price_after_discounts
  (original_price : ℝ) (friday_discount_rate : ℝ) (monday_discount_rate : ℝ) 
  (sales_tax_rate : ℝ) (price_after_all_adjustments : ℝ) :
  original_price = 200 →
  friday_discount_rate = 0.40 →
  monday_discount_rate = 0.20 →
  sales_tax_rate = 0.05 →
  price_after_all_adjustments = 100.80 :=
by
  intros
  sorry

end snowboard_price_after_discounts_l231_231102


namespace circle_diameter_l231_231732

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l231_231732


namespace emily_age_proof_l231_231362

theorem emily_age_proof (e m : ℕ) (h1 : e = m - 18) (h2 : e + m = 54) : e = 18 :=
by
  sorry

end emily_age_proof_l231_231362


namespace total_wheels_l231_231526

def num_wheels_in_garage : Nat :=
  let cars := 2 * 4
  let lawnmower := 4
  let bicycles := 3 * 2
  let tricycle := 3
  let unicycle := 1
  let skateboard := 4
  let wheelbarrow := 1
  let wagon := 4
  let dolly := 2
  let shopping_cart := 4
  let scooter := 2
  cars + lawnmower + bicycles + tricycle + unicycle + skateboard + wheelbarrow + wagon + dolly + shopping_cart + scooter

theorem total_wheels : num_wheels_in_garage = 39 := by
  sorry

end total_wheels_l231_231526


namespace power_eq_l231_231588

open Real

theorem power_eq {x : ℝ} (h : x^3 + 4 * x = 8) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end power_eq_l231_231588


namespace gcd_all_abc_plus_cba_l231_231080

noncomputable def gcd_of_abc_cba (a : ℕ) (b : ℕ := 2 * a) (c : ℕ := 3 * a) : ℕ :=
  let abc := 64 * a + 8 * b + c
  let cba := 64 * c + 8 * b + a
  Nat.gcd (abc + cba) 300

theorem gcd_all_abc_plus_cba (a : ℕ) : gcd_of_abc_cba a = 300 :=
  sorry

end gcd_all_abc_plus_cba_l231_231080


namespace exist_amusing_numbers_l231_231862

/-- Definitions for an amusing number -/
def is_amusing (x : ℕ) : Prop :=
  (x >= 1000) ∧ (x <= 9999) ∧
  ∃ y : ℕ, y ≠ x ∧
  ((∀ d ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10],
    (d ≠ 0 → (y % 1000) ∈ [(d-1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10, (d+1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10]) ∧
    (d ≠ 9 → (y % 1000) ∈ [(d-1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10, (d+1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10]))) ∧
  (y % x = 0)

/-- Prove the existence of four amusing four-digit numbers -/
theorem exist_amusing_numbers :
  ∃ x1 x2 x3 x4 : ℕ, is_amusing x1 ∧ is_amusing x2 ∧ is_amusing x3 ∧ is_amusing x4 ∧ 
                   x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 :=
by sorry

end exist_amusing_numbers_l231_231862


namespace smallest_number_of_ducks_l231_231372

theorem smallest_number_of_ducks (n_ducks n_cranes : ℕ) (h1 : n_ducks = n_cranes) : 
  ∃ n, n_ducks = n ∧ n_cranes = n ∧ n = Nat.lcm 13 17 := by
  use 221
  sorry

end smallest_number_of_ducks_l231_231372


namespace remainder_7325_mod_11_l231_231072

theorem remainder_7325_mod_11 : 7325 % 11 = 6 := sorry

end remainder_7325_mod_11_l231_231072


namespace solve_eqn_l231_231521

theorem solve_eqn {x : ℝ} : x^4 + (3 - x)^4 = 130 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end solve_eqn_l231_231521


namespace ellipse_semi_focal_distance_range_l231_231202

theorem ellipse_semi_focal_distance_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) (h_ellipse : a^2 = b^2 + c^2) :
  1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 := 
sorry

end ellipse_semi_focal_distance_range_l231_231202


namespace difference_in_circumferences_l231_231596

theorem difference_in_circumferences (r_inner r_outer : ℝ) (h1 : r_inner = 15) (h2 : r_outer = r_inner + 8) : 
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 16 * Real.pi :=
by
  rw [h1, h2]
  sorry

end difference_in_circumferences_l231_231596


namespace area_of_circle_l231_231296

def circleEquation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

theorem area_of_circle :
  (∃ (center : ℝ × ℝ) (radius : ℝ), radius = 4 ∧ ∀ (x y : ℝ), circleEquation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) →
  (16 * Real.pi) = 16 * Real.pi := 
by 
  intro h
  have := h
  sorry

end area_of_circle_l231_231296


namespace number_of_liars_l231_231958

/-- There are 25 people in line, each of whom either tells the truth or lies.
The person at the front of the line says: "Everyone behind me is lying."
Everyone else says: "The person directly in front of me is lying."
Prove that the number of liars among these 25 people is 13. -/
theorem number_of_liars : 
  ∀ (persons : Fin 25 → Prop), 
    (persons 0 → ∀ n > 0, ¬persons n) →
    (∀ n : Nat, (1 ≤ n → n < 25 → persons n ↔ ¬persons (n - 1))) →
    (∃ l, l = 13 ∧ ∀ n : Nat, (0 ≤ n → n < 25 → persons n ↔ (n % 2 = 0))) :=
by
  sorry

end number_of_liars_l231_231958


namespace part1_part2_l231_231651

-- Definitions for the sides and the target equations
def triangleSides (a b c : ℝ) (A B C : ℝ) : Prop :=
  b * Real.sin (C / 2) ^ 2 + c * Real.sin (B / 2) ^ 2 = a / 2

-- The first part of the problem
theorem part1 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  b + c = 2 * a :=
  sorry

-- The second part of the problem
theorem part2 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  A ≤ π / 3 :=
  sorry

end part1_part2_l231_231651


namespace fraction_zero_x_eq_2_l231_231931

theorem fraction_zero_x_eq_2 (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 :=
by sorry

end fraction_zero_x_eq_2_l231_231931


namespace find_x_y_sum_l231_231561

theorem find_x_y_sum :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (∃ (a b : ℕ), 360 * x = a^2 ∧ 360 * y = b^4) ∧ x + y = 2260 :=
by {
  sorry
}

end find_x_y_sum_l231_231561


namespace largest_number_of_positive_consecutive_integers_l231_231193

theorem largest_number_of_positive_consecutive_integers (n a : ℕ) (h1 : a > 0) (h2 : n > 0) (h3 : (n * (2 * a + n - 1)) / 2 = 45) : 
  n = 9 := 
sorry

end largest_number_of_positive_consecutive_integers_l231_231193


namespace parabola_constant_unique_l231_231041

theorem parabola_constant_unique (b c : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 20) → y = x^2 + b * x + c) →
  (∀ x y : ℝ, (x = -2 ∧ y = -4) → y = x^2 + b * x + c) →
  c = 4 :=
by
    sorry

end parabola_constant_unique_l231_231041


namespace select_two_integers_divisibility_l231_231926

open Polynomial

theorem select_two_integers_divisibility
  (F : Polynomial ℤ)
  (m : ℕ)
  (a : Fin m → ℤ)
  (H : ∀ n : ℤ, ∃ i : Fin m, a i ∣ F.eval n) :
  ∃ i j : Fin m, i ≠ j ∧ ∀ n : ℤ, ∃ k : Fin m, k = i ∨ k = j ∧ a k ∣ F.eval n :=
by
  sorry

end select_two_integers_divisibility_l231_231926


namespace gemma_amount_given_l231_231374

theorem gemma_amount_given
  (cost_per_pizza : ℕ)
  (number_of_pizzas : ℕ)
  (tip : ℕ)
  (change_back : ℕ)
  (h1 : cost_per_pizza = 10)
  (h2 : number_of_pizzas = 4)
  (h3 : tip = 5)
  (h4 : change_back = 5) :
  number_of_pizzas * cost_per_pizza + tip + change_back = 50 := sorry

end gemma_amount_given_l231_231374


namespace disjoint_polynomial_sets_l231_231090

theorem disjoint_polynomial_sets (A B : ℤ) : 
  ∃ C : ℤ, ∀ x1 x2 : ℤ, x1^2 + A * x1 + B ≠ 2 * x2^2 + 2 * x2 + C :=
by
  sorry

end disjoint_polynomial_sets_l231_231090


namespace fraction_sum_ratio_l231_231830

theorem fraction_sum_ratio
    (a b c : ℝ) (m n : ℝ)
    (h1 : a = (b + c) / m)
    (h2 : b = (c + a) / n) :
    (m * n ≠ 1 → (a + b) / c = (m + n + 2) / (m * n - 1)) ∧ 
    (m = -1 ∧ n = -1 → (a + b) / c = -1) :=
by
    sorry

end fraction_sum_ratio_l231_231830


namespace final_selling_price_l231_231153

def actual_price : ℝ := 9356.725146198829
def price_after_first_discount (P : ℝ) : ℝ := P * 0.80
def price_after_second_discount (P1 : ℝ) : ℝ := P1 * 0.90
def price_after_third_discount (P2 : ℝ) : ℝ := P2 * 0.95

theorem final_selling_price :
  (price_after_third_discount (price_after_second_discount (price_after_first_discount actual_price))) = 6400 :=
by 
  -- Here we would need to provide the proof, but it is skipped with sorry
  sorry

end final_selling_price_l231_231153


namespace min_num_stamps_is_17_l231_231893

-- Definitions based on problem conditions
def initial_num_stamps : ℕ := 2 + 5 + 3 + 1
def initial_cost : ℝ := 2 * 0.10 + 5 * 0.20 + 3 * 0.50 + 1 * 2
def remaining_cost : ℝ := 10 - initial_cost
def additional_stamps : ℕ := 2 + 2 + 1 + 1
def total_stamps : ℕ := initial_num_stamps + additional_stamps

-- Proof that the minimum number of stamps bought is 17
theorem min_num_stamps_is_17 : total_stamps = 17 := by
  sorry

end min_num_stamps_is_17_l231_231893


namespace minimum_value_of_a_l231_231057

def is_prime (n : ℕ) : Prop := sorry  -- Provide the definition of a prime number

def is_perfect_square (n : ℕ) : Prop := sorry  -- Provide the definition of a perfect square

theorem minimum_value_of_a 
  (a b : ℕ) 
  (h1 : is_prime (a - b)) 
  (h2 : is_perfect_square (a * b)) 
  (h3 : a ≥ 2012) : 
  a = 2025 := 
sorry

end minimum_value_of_a_l231_231057


namespace complement_union_l231_231599

open Set

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3}
noncomputable def C_UA : Set ℕ := U \ A

-- Statement to prove
theorem complement_union (U A B C_UA : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3, 5})
  (hB : B = {2, 3}) 
  (hCUA : C_UA = U \ A) : 
  (C_UA ∪ B) = {2, 3, 4} := 
sorry

end complement_union_l231_231599


namespace unique_parallelogram_l231_231230

theorem unique_parallelogram :
  ∃! (A B D C : ℤ × ℤ), 
  A = (0, 0) ∧ 
  (B.2 = B.1) ∧ 
  (D.2 = 2 * D.1) ∧ 
  (C.2 = 3 * C.1) ∧ 
  (A.1 = 0 ∧ A.2 = 0) ∧ 
  (B.1 > 0 ∧ B.2 > 0) ∧ 
  (D.1 > 0 ∧ D.2 > 0) ∧ 
  (C.1 > 0 ∧ C.2 > 0) ∧ 
  (B.1 - A.1, B.2 - A.2) + (D.1 - A.1, D.2 - A.2) = (C.1 - A.1, C.2 - A.2) ∧
  (abs ((B.1 * C.2 + C.1 * D.2 + D.1 * A.2 + A.1 * B.2) - (A.1 * C.2 + B.1 * D.2 + C.1 * B.2 + D.1 * A.2)) / 2) = 2000000 
  := by sorry

end unique_parallelogram_l231_231230


namespace rubber_bands_per_large_ball_l231_231506

open Nat

theorem rubber_bands_per_large_ball :
  let total_rubber_bands := 5000
  let small_bands := 50
  let small_balls := 22
  let large_balls := 13
  let used_bands := small_balls * small_bands
  let remaining_bands := total_rubber_bands - used_bands
  let large_bands := remaining_bands / large_balls
  large_bands = 300 :=
by
  sorry

end rubber_bands_per_large_ball_l231_231506


namespace total_flight_time_l231_231240

theorem total_flight_time
  (distance : ℕ)
  (speed_out : ℕ)
  (speed_return : ℕ)
  (time_out : ℕ)
  (time_return : ℕ)
  (total_time : ℕ)
  (h1 : distance = 1500)
  (h2 : speed_out = 300)
  (h3 : speed_return = 500)
  (h4 : time_out = distance / speed_out)
  (h5 : time_return = distance / speed_return)
  (h6 : total_time = time_out + time_return) :
  total_time = 8 := 
  by {
    sorry
  }

end total_flight_time_l231_231240


namespace will_earnings_l231_231416

-- Defining the conditions
def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

-- Calculating the earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def total_earnings := monday_earnings + tuesday_earnings

-- Stating the problem
theorem will_earnings : total_earnings = 80 := by
  -- sorry is used to skip the actual proof
  sorry

end will_earnings_l231_231416


namespace ratio_in_range_l231_231666

theorem ratio_in_range {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end ratio_in_range_l231_231666


namespace no_extrema_1_1_l231_231431

noncomputable def f (x : ℝ) : ℝ :=
  x^3 - 3 * x

theorem no_extrema_1_1 : ∀ x : ℝ, (x > -1) ∧ (x < 1) → ¬ (∃ c : ℝ, c ∈ Set.Ioo (-1) (1) ∧ (∀ y ∈ Set.Ioo (-1) (1), f y ≤ f c ∨ f c ≤ f y)) :=
by
  sorry

end no_extrema_1_1_l231_231431


namespace eldorado_license_plates_count_l231_231413

def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def valid_license_plates_count : Nat :=
  let num_vowels := 5
  let num_letters := 26
  let num_digits := 10
  num_vowels * num_letters * num_letters * num_digits * num_digits

theorem eldorado_license_plates_count : valid_license_plates_count = 338000 := by
  sorry

end eldorado_license_plates_count_l231_231413


namespace product_of_solutions_l231_231405

theorem product_of_solutions (α β : ℝ) (h : 2 * α^2 + 8 * α - 45 = 0 ∧ 2 * β^2 + 8 * β - 45 = 0 ∧ α ≠ β) :
  α * β = -22.5 :=
sorry

end product_of_solutions_l231_231405


namespace problem1_problem2_l231_231515

-- Proof problem (1)
theorem problem1 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 1 < x ∧ x < 2} ∧ m = 1 →
  (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := 
by 
  sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 2 * m - 1 < x ∧ x < m + 1} →
  (B ⊆ A ↔ (m ≥ 2 ∨ (-1 ≤ m ∧ m < 2))) := 
by 
  sorry

end problem1_problem2_l231_231515


namespace inequality_solution_l231_231745

theorem inequality_solution (x : ℝ) : (5 < x ∧ x ≤ 6) ↔ (x-3)/(x-5) ≥ 3 :=
by
  sorry

end inequality_solution_l231_231745


namespace contractor_job_completion_l231_231331

theorem contractor_job_completion 
  (total_days : ℕ := 100) 
  (initial_workers : ℕ := 10) 
  (days_worked_initial : ℕ := 20) 
  (fraction_completed_initial : ℚ := 1/4) 
  (fired_workers : ℕ := 2) 
  : ∀ (remaining_days : ℕ), remaining_days = 75 → (remaining_days + days_worked_initial = 95) :=
by
  sorry

end contractor_job_completion_l231_231331


namespace smallest_number_starting_with_five_l231_231317

theorem smallest_number_starting_with_five :
  ∃ n : ℕ, ∃ m : ℕ, m = (5 * m + 5) / 4 ∧ 5 * n + m = 512820 ∧ m < 10^6 := sorry

end smallest_number_starting_with_five_l231_231317


namespace CoveredAreaIs84_l231_231933

def AreaOfStrip (length width : ℕ) : ℕ :=
  length * width

def TotalAreaWithoutOverlaps (numStrips areaOfOneStrip : ℕ) : ℕ :=
  numStrips * areaOfOneStrip

def OverlapArea (intersectionArea : ℕ) (numIntersections : ℕ) : ℕ :=
  intersectionArea * numIntersections

def ActualCoveredArea (totalArea overlapArea : ℕ) : ℕ :=
  totalArea - overlapArea

theorem CoveredAreaIs84 :
  let length := 12
  let width := 2
  let numStrips := 6
  let intersectionArea := width * width
  let numIntersections := 15
  let areaOfOneStrip := AreaOfStrip length width
  let totalAreaWithoutOverlaps := TotalAreaWithoutOverlaps numStrips areaOfOneStrip
  let totalOverlapArea := OverlapArea intersectionArea numIntersections
  ActualCoveredArea totalAreaWithoutOverlaps totalOverlapArea = 84 :=
by
  sorry

end CoveredAreaIs84_l231_231933


namespace karens_class_fund_l231_231276

noncomputable def ratio_of_bills (T W : ℕ) : ℕ × ℕ := (T / Nat.gcd T W, W / Nat.gcd T W)

theorem karens_class_fund (T W : ℕ) (hW : W = 3) (hfund : 10 * T + 20 * W = 120) :
  ratio_of_bills T W = (2, 1) :=
by
  sorry

end karens_class_fund_l231_231276


namespace find_gain_percent_l231_231864

theorem find_gain_percent (CP SP : ℝ) (h1 : CP = 20) (h2 : SP = 25) : 100 * ((SP - CP) / CP) = 25 := by
  sorry

end find_gain_percent_l231_231864


namespace sum_two_primes_eq_91_prod_is_178_l231_231738

theorem sum_two_primes_eq_91_prod_is_178
  (p1 p2 : ℕ) 
  (hp1 : p1.Prime) 
  (hp2 : p2.Prime) 
  (h_sum : p1 + p2 = 91) :
  p1 * p2 = 178 := 
sorry

end sum_two_primes_eq_91_prod_is_178_l231_231738


namespace width_of_door_is_correct_l231_231625

theorem width_of_door_is_correct
  (L : ℝ) (W : ℝ) (H : ℝ := 12)
  (door_height : ℝ := 6) (window_height : ℝ := 4) (window_width : ℝ := 3)
  (cost_per_square_foot : ℝ := 10) (total_cost : ℝ := 9060) :
  (L = 25 ∧ W = 15) →
  2 * (L + W) * H - (door_height * width_door + 3 * (window_height * window_width)) * cost_per_square_foot = total_cost →
  width_door = 3 :=
by
  intros h1 h2
  sorry

end width_of_door_is_correct_l231_231625


namespace quadratic_eq_c_has_equal_roots_l231_231967

theorem quadratic_eq_c_has_equal_roots (c : ℝ) (h : ∃ x : ℝ, x^2 - 4 * x + c = 0 ∧
                      ∀ y : ℝ, x^2 - 4 * x + c = 0 → y = x) : c = 4 := sorry

end quadratic_eq_c_has_equal_roots_l231_231967


namespace karlanna_marble_problem_l231_231804

theorem karlanna_marble_problem : 
  ∃ (m_values : Finset ℕ), 
  (∀ m ∈ m_values, ∃ n : ℕ, m * n = 450 ∧ m > 1 ∧ n > 1) ∧ 
  m_values.card = 16 := 
by
  sorry

end karlanna_marble_problem_l231_231804


namespace expected_value_correct_l231_231027

-- Define the probability distribution of the user's score in the first round
noncomputable def first_round_prob (X : ℕ) : ℚ :=
  if X = 3 then 1 / 4
  else if X = 2 then 1 / 2
  else if X = 1 then 1 / 4
  else 0

-- Define the conditional probability of the user's score in the second round given the first round score
noncomputable def second_round_prob (X Y : ℕ) : ℚ :=
  if X = 3 then
    if Y = 2 then 1 / 5
    else if Y = 1 then 4 / 5
    else 0
  else
    if Y = 2 then 1 / 3
    else if Y = 1 then 2 / 3
    else 0

-- Define the total score probability
noncomputable def total_score_prob (X Y : ℕ) : ℚ :=
  first_round_prob X * second_round_prob X Y

-- Compute the expected value of the user's total score
noncomputable def expected_value : ℚ :=
  (5 * (total_score_prob 3 2) +
   4 * (total_score_prob 3 1 + total_score_prob 2 2) +
   3 * (total_score_prob 2 1 + total_score_prob 1 2) +
   2 * (total_score_prob 1 1))

-- The theorem to be proven
theorem expected_value_correct : expected_value = 3.3 := 
by sorry

end expected_value_correct_l231_231027


namespace pyramid_volume_l231_231883

noncomputable def volume_of_pyramid (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) : ℝ :=
  let base_area := base_length * base_width
  let diagonal_length := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal_length / 2)^2)
  (1 / 3) * base_area * height

theorem pyramid_volume : volume_of_pyramid 7 9 15 = 21 * Real.sqrt 192.5 := by
  sorry

end pyramid_volume_l231_231883


namespace four_cells_different_colors_l231_231744

theorem four_cells_different_colors
  (n : ℕ)
  (h_n : n ≥ 2)
  (coloring : Fin n → Fin n → Fin (2 * n)) :
  ∃ (r1 r2 c1 c2 : Fin n),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧
    (coloring r1 c1 ≠ coloring r1 c2) ∧
    (coloring r1 c1 ≠ coloring r2 c1) ∧
    (coloring r1 c2 ≠ coloring r2 c2) ∧
    (coloring r2 c1 ≠ coloring r2 c2) := 
sorry

end four_cells_different_colors_l231_231744


namespace point_P_in_first_quadrant_l231_231045

def pointInFirstQuadrant (x y : Int) : Prop := x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : pointInFirstQuadrant 2 3 :=
by
  sorry

end point_P_in_first_quadrant_l231_231045


namespace find_a_l231_231154

theorem find_a (a : ℝ) :
  (∀ x y, x + y = a → x^2 + y^2 = 4) →
  (∀ A B : ℝ × ℝ, (A.1 + A.2 = a ∧ B.1 + B.2 = a ∧ A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4) →
      ‖(A.1, A.2) + (B.1, B.2)‖ = ‖(A.1, A.2) - (B.1, B.2)‖) →
  a = 2 ∨ a = -2 :=
by
  intros line_circle_intersect vector_eq_magnitude
  sorry

end find_a_l231_231154


namespace total_amount_l231_231188

theorem total_amount (x_share : ℝ) (y_share : ℝ) (w_share : ℝ) (hx : x_share = 0.30) (hy : y_share = 0.20) (hw : w_share = 10) :
  (w_share * (1 + x_share + y_share)) = 15 := by
  sorry

end total_amount_l231_231188


namespace cost_of_fencing_l231_231200

-- Define the conditions
def width_garden : ℕ := 12
def length_playground : ℕ := 16
def width_playground : ℕ := 12
def price_per_meter : ℕ := 15
def area_playground : ℕ := length_playground * width_playground
def area_garden : ℕ := area_playground
def length_garden : ℕ := area_garden / width_garden
def perimeter_garden : ℕ := 2 * (length_garden + width_garden)
def cost_fencing : ℕ := perimeter_garden * price_per_meter

-- State the theorem
theorem cost_of_fencing : cost_fencing = 840 := by
  sorry

end cost_of_fencing_l231_231200


namespace prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l231_231248

variable {A : Set Int}

-- Assuming set A is closed under subtraction
axiom A_closed_under_subtraction : ∀ x y, x ∈ A → y ∈ A → x - y ∈ A
axiom A_contains_4 : 4 ∈ A
axiom A_contains_9 : 9 ∈ A

theorem prove_0_in_A : 0 ∈ A :=
sorry

theorem prove_13_in_A : 13 ∈ A :=
sorry

theorem prove_74_in_A : 74 ∈ A :=
sorry

theorem prove_A_is_Z : A = Set.univ :=
sorry

end prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l231_231248


namespace suraj_innings_l231_231375

theorem suraj_innings (n A : ℕ) (h1 : A + 6 = 16) (h2 : (n * A + 112) / (n + 1) = 16) : n = 16 :=
by
  sorry

end suraj_innings_l231_231375


namespace work_done_by_force_l231_231785

def force : ℝ × ℝ := (-1, -2)
def displacement : ℝ × ℝ := (3, 4)

def work_done (F S : ℝ × ℝ) : ℝ :=
  F.1 * S.1 + F.2 * S.2

theorem work_done_by_force :
  work_done force displacement = -11 := 
by
  sorry

end work_done_by_force_l231_231785


namespace area_of_tangency_triangle_l231_231644

noncomputable def area_of_triangle : ℝ :=
  let r1 := 2
  let r2 := 3
  let r3 := 4
  let s := (r1 + r2 + r3) / 2
  let A := Real.sqrt (s * (s - (r1 + r2)) * (s - (r2 + r3)) * (s - (r1 + r3)))
  let inradius := A / s
  let area_points_of_tangency := A * (inradius / r1) * (inradius / r2) * (inradius / r3)
  area_points_of_tangency

theorem area_of_tangency_triangle :
  area_of_triangle = (16 * Real.sqrt 6) / 3 :=
sorry

end area_of_tangency_triangle_l231_231644


namespace point_reflection_xOy_l231_231225

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflection_over_xOy (P : Point3D) : Point3D := 
  {x := P.x, y := P.y, z := -P.z}

theorem point_reflection_xOy :
  reflection_over_xOy {x := 1, y := 2, z := 3} = {x := 1, y := 2, z := -3} := by
  sorry

end point_reflection_xOy_l231_231225


namespace inverse_function_domain_l231_231655

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem inverse_function_domain :
  ∃ (g : ℝ → ℝ), (∀ x, 0 ≤ x → f (g x) = x) ∧ (∀ y, 0 ≤ y → g (f y) = y) ∧ (∀ x, 0 ≤ x ↔ 0 ≤ g x) :=
by
  sorry

end inverse_function_domain_l231_231655


namespace defective_probability_bayesian_probabilities_l231_231726

noncomputable def output_proportion_A : ℝ := 0.25
noncomputable def output_proportion_B : ℝ := 0.35
noncomputable def output_proportion_C : ℝ := 0.40

noncomputable def defect_rate_A : ℝ := 0.05
noncomputable def defect_rate_B : ℝ := 0.04
noncomputable def defect_rate_C : ℝ := 0.02

noncomputable def probability_defective : ℝ :=
  output_proportion_A * defect_rate_A +
  output_proportion_B * defect_rate_B +
  output_proportion_C * defect_rate_C 

theorem defective_probability :
  probability_defective = 0.0345 := 
  by sorry

noncomputable def P_A_given_defective : ℝ :=
  (output_proportion_A * defect_rate_A) / probability_defective

noncomputable def P_B_given_defective : ℝ :=
  (output_proportion_B * defect_rate_B) / probability_defective

noncomputable def P_C_given_defective : ℝ :=
  (output_proportion_C * defect_rate_C) / probability_defective

theorem bayesian_probabilities :
  P_A_given_defective = 25 / 69 ∧
  P_B_given_defective = 28 / 69 ∧
  P_C_given_defective = 16 / 69 :=
  by sorry

end defective_probability_bayesian_probabilities_l231_231726


namespace cost_of_siding_l231_231349

def area_of_wall (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def area_of_roof (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length * width)

def area_of_sheet (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def sheets_needed (total_area : ℕ) (sheet_area : ℕ) : ℕ :=
  (total_area + sheet_area - 1) / sheet_area  -- Cooling the ceiling with integer arithmetic

def total_cost (sheets : ℕ) (price_per_sheet : ℕ) : ℕ :=
  sheets * price_per_sheet

theorem cost_of_siding : 
  ∀ (length_wall width_wall length_roof width_roof length_sheet width_sheet price_per_sheet : ℕ),
  length_wall = 10 → width_wall = 7 →
  length_roof = 10 → width_roof = 6 →
  length_sheet = 10 → width_sheet = 14 →
  price_per_sheet = 50 →
  total_cost (sheets_needed (area_of_wall length_wall width_wall + area_of_roof length_roof width_roof) (area_of_sheet length_sheet width_sheet)) price_per_sheet = 100 :=
by
  intros
  simp [area_of_wall, area_of_roof, area_of_sheet, sheets_needed, total_cost]
  sorry

end cost_of_siding_l231_231349


namespace quadratic_root_value_l231_231850

theorem quadratic_root_value (a : ℝ) (h : a^2 + 2 * a - 3 = 0) : 2 * a^2 + 4 * a = 6 :=
by
  sorry

end quadratic_root_value_l231_231850


namespace solve_problem_l231_231981

def is_solution (a : ℕ) : Prop :=
  a % 3 = 1 ∧ ∃ k : ℕ, a = 5 * k

theorem solve_problem : ∃ a : ℕ, is_solution a ∧ ∀ b : ℕ, is_solution b → a ≤ b := 
  sorry

end solve_problem_l231_231981


namespace fish_tank_problem_l231_231266

def number_of_fish_in_first_tank
  (F : ℕ)          -- Let F represent the number of fish in the first tank
  (twoF : ℕ)       -- Let twoF represent twice the number of fish in the first tank
  (total : ℕ) :    -- Let total represent the total number of fish
  Prop :=
  (2 * F = twoF)  -- The other two tanks each have twice as many fish as the first
  ∧ (F + twoF + twoF = total)  -- The sum of the fish in all three tanks equals the total number of fish

theorem fish_tank_problem
  (F : ℕ)
  (H : number_of_fish_in_first_tank F (2 * F) 100) : F = 20 :=
by
  sorry

end fish_tank_problem_l231_231266


namespace find_f7_l231_231189

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 7

theorem find_f7 (a b : ℝ) (h : f (-7) a b = -17) : f (7) a b = 31 := 
by
  sorry

end find_f7_l231_231189


namespace arith_seq_a1_a2_a3_sum_l231_231982

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a1_a2_a3_sum (a : ℕ → ℤ) (h_seq : arithmetic_seq a)
  (h1 : a 1 = 2) (h_sum : a 1 + a 2 + a 3 = 18) :
  a 4 + a 5 + a 6 = 54 :=
sorry

end arith_seq_a1_a2_a3_sum_l231_231982


namespace statue_of_liberty_ratio_l231_231632

theorem statue_of_liberty_ratio :
  let H_statue := 305 -- height in feet
  let H_model := 10 -- height in inches
  H_statue / H_model = 30.5 := 
by
  let H_statue := 305
  let H_model := 10
  sorry

end statue_of_liberty_ratio_l231_231632


namespace conic_section_is_parabola_l231_231479

-- Define the equation |y-3| = sqrt((x+4)^2 + y^2)
def equation (x y : ℝ) : Prop := |y - 3| = Real.sqrt ((x + 4) ^ 2 + y ^ 2)

-- The main theorem stating the conic section type is a parabola
theorem conic_section_is_parabola : ∀ x y : ℝ, equation x y → false := sorry

end conic_section_is_parabola_l231_231479


namespace compare_neg_fractions_l231_231211

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (3 / 4 : ℝ) :=
sorry

end compare_neg_fractions_l231_231211


namespace jen_triple_flips_l231_231251

-- Definitions based on conditions
def tyler_double_flips : ℕ := 12
def flips_per_double_flip : ℕ := 2
def flips_by_tyler : ℕ := tyler_double_flips * flips_per_double_flip
def flips_ratio : ℕ := 2
def flips_per_triple_flip : ℕ := 3
def flips_by_jen : ℕ := flips_by_tyler * flips_ratio

-- Lean 4 statement
theorem jen_triple_flips : flips_by_jen / flips_per_triple_flip = 16 :=
by 
    -- Proof contents should go here. We only need the statement as per the instruction.
    sorry

end jen_triple_flips_l231_231251


namespace firefighters_time_to_extinguish_fire_l231_231572

theorem firefighters_time_to_extinguish_fire (gallons_per_minute_per_hose : ℕ) (total_gallons : ℕ) (number_of_firefighters : ℕ)
  (H1 : gallons_per_minute_per_hose = 20)
  (H2 : total_gallons = 4000)
  (H3 : number_of_firefighters = 5): 
  (total_gallons / (gallons_per_minute_per_hose * number_of_firefighters)) = 40 := 
by 
  sorry

end firefighters_time_to_extinguish_fire_l231_231572


namespace eval_expression_l231_231622

theorem eval_expression : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 :=
by
  sorry

end eval_expression_l231_231622


namespace smallest_three_digit_multiple_of_6_5_8_9_eq_360_l231_231584

theorem smallest_three_digit_multiple_of_6_5_8_9_eq_360 :
  ∃ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ (n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧ n = 360 := 
by
  sorry

end smallest_three_digit_multiple_of_6_5_8_9_eq_360_l231_231584


namespace gcd_gx_x_l231_231947

-- Condition: x is a multiple of 7263
def isMultipleOf7263 (x : ℕ) : Prop := ∃ k : ℕ, x = 7263 * k

-- Definition of g(x)
def g (x : ℕ) : ℕ := (3*x + 4) * (9*x + 5) * (17*x + 11) * (x + 17)

-- Statement to be proven
theorem gcd_gx_x (x : ℕ) (h : isMultipleOf7263 x) : Nat.gcd (g x) x = 1 := by
  sorry

end gcd_gx_x_l231_231947


namespace find_t_l231_231897

variables {m n : ℝ}
variables (t : ℝ)
variables (mv nv : ℝ)
variables (dot_m_m dot_m_n dot_n_n : ℝ)
variables (cos_theta : ℝ)

-- Define the basic assumptions
axiom non_zero_vectors : m ≠ 0 ∧ n ≠ 0
axiom magnitude_condition : mv = 2 * nv
axiom cos_condition : cos_theta = 1 / 3
axiom perpendicular_condition : dot_m_n = (mv * nv * cos_theta) ∧ (t * dot_m_n + dot_m_m = 0)

-- Utilize the conditions and prove the target
theorem find_t : t = -6 :=
sorry

end find_t_l231_231897


namespace negation_of_exists_l231_231805

theorem negation_of_exists {x : ℝ} :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ ∀ x : ℝ, x^2 - 2 > 0 :=
by
  sorry

end negation_of_exists_l231_231805


namespace determine_remainder_l231_231595

theorem determine_remainder (a b c : ℕ) (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (H1 : (a + 2 * b + 3 * c) % 7 = 1) 
  (H2 : (2 * a + 3 * b + c) % 7 = 2) 
  (H3 : (3 * a + b + 2 * c) % 7 = 1) : 
  (a * b * c) % 7 = 0 := 
sorry

end determine_remainder_l231_231595


namespace remy_gallons_used_l231_231435

def roman_usage (R : ℕ) : Prop := R + (3 * R + 1) = 33

def remy_usage (R : ℕ) (Remy : ℕ) : Prop := Remy = 3 * R + 1

theorem remy_gallons_used :
  ∃ R Remy : ℕ, roman_usage R ∧ remy_usage R Remy ∧ Remy = 25 :=
  by
    sorry

end remy_gallons_used_l231_231435


namespace marked_price_is_300_max_discount_is_50_l231_231357

-- Definition of the conditions given in the problem:
def loss_condition (x : ℝ) : Prop := 0.4 * x - 30 = 0.7 * x - 60
def profit_condition (x : ℝ) : Prop := 0.7 * x - 60 - (0.4 * x - 30) = 90

-- Statement for the first problem: Prove the marked price is 300 yuan.
theorem marked_price_is_300 : ∃ x : ℝ, loss_condition x ∧ profit_condition x ∧ x = 300 := by
  exists 300
  simp [loss_condition, profit_condition]
  sorry

noncomputable def max_discount (x : ℝ) : ℝ := 100 - (30 + 0.4 * x) / x * 100

def no_loss_max_discount (d : ℝ) : Prop := d = 50

-- Statement for the second problem: Prove the maximum discount is 50%.
theorem max_discount_is_50 (x : ℝ) (h_loss : loss_condition x) (h_profit : profit_condition x) : no_loss_max_discount (max_discount x) := by
  simp [max_discount, no_loss_max_discount]
  sorry

end marked_price_is_300_max_discount_is_50_l231_231357


namespace abs_diff_expr_l231_231441

theorem abs_diff_expr :
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  |a| - |b| = 4 :=
by
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  sorry

end abs_diff_expr_l231_231441


namespace negation_of_proposition_l231_231283

theorem negation_of_proposition (a b : ℝ) : 
  ¬(a + b = 1 → a^2 + b^2 ≥ 1/2) ↔ (a + b ≠ 1 → a^2 + b^2 < 1/2) :=
by sorry

end negation_of_proposition_l231_231283


namespace difference_between_numbers_l231_231359

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 27630) (h2 : a = 5 * b + 5) : a - b = 18421 :=
  sorry

end difference_between_numbers_l231_231359


namespace pupils_in_program_l231_231308

theorem pupils_in_program {total_people parents : ℕ} (h1 : total_people = 238) (h2 : parents = 61) :
  total_people - parents = 177 := by
  sorry

end pupils_in_program_l231_231308


namespace circumcircle_radius_l231_231792

theorem circumcircle_radius (b A S : ℝ) (h_b : b = 2) 
  (h_A : A = 120 * Real.pi / 180) (h_S : S = Real.sqrt 3) : 
  ∃ R, R = 2 := 
by
  sorry

end circumcircle_radius_l231_231792


namespace chinese_pig_problem_l231_231161

variable (x : ℕ)

theorem chinese_pig_problem :
  100 * x - 90 * x = 100 :=
sorry

end chinese_pig_problem_l231_231161


namespace number_of_valid_sets_l231_231392

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8,9,10}
def valid_set (A : Set ℕ) : Prop :=
  ∃ a1 a2 a3, A = {a1, a2, a3} ∧ a3 ∈ U ∧ a2 ∈ U ∧ a1 ∈ U ∧ a3 ≥ a2 + 1 ∧ a2 ≥ a1 + 4

theorem number_of_valid_sets : ∃ (n : ℕ), n = 56 ∧ ∃ S : Finset (Set ℕ), (∀ A ∈ S, valid_set A) ∧ S.card = n := by
  sorry

end number_of_valid_sets_l231_231392


namespace solution_l231_231151

def solve_for_x (x : ℝ) : Prop :=
  7 + 3.5 * x = 2.1 * x - 25

theorem solution (x : ℝ) (h : solve_for_x x) : x = -22.857 :=
by
  sorry

end solution_l231_231151


namespace solve_equation_l231_231902

theorem solve_equation (x : ℝ) : x * (x-3)^2 * (5+x) = 0 ↔ (x = 0 ∨ x = 3 ∨ x = -5) := 
by
  sorry

end solve_equation_l231_231902


namespace karl_total_income_correct_l231_231689

noncomputable def price_of_tshirt : ℝ := 5
noncomputable def price_of_pants : ℝ := 4
noncomputable def price_of_skirt : ℝ := 6
noncomputable def price_of_refurbished_tshirt : ℝ := price_of_tshirt / 2

noncomputable def discount_for_skirts (n : ℕ) : ℝ := (n / 2) * 2 * price_of_skirt * 0.10
noncomputable def discount_for_tshirts (n : ℕ) : ℝ := (n / 5) * 5 * price_of_tshirt * 0.20
noncomputable def discount_for_pants (n : ℕ) : ℝ := 0 -- accounted for in quantity

noncomputable def sales_tax (amount : ℝ) : ℝ := amount * 0.08

noncomputable def total_income : ℝ := 
  let tshirt_income := 8 * price_of_tshirt + 7 * price_of_refurbished_tshirt - discount_for_tshirts 15
  let pants_income := 6 * price_of_pants - discount_for_pants 6
  let skirts_income := 12 * price_of_skirt - discount_for_skirts 12
  let income_before_tax := tshirt_income + pants_income + skirts_income
  income_before_tax + sales_tax income_before_tax

theorem karl_total_income_correct : total_income = 141.80 :=
by
  sorry

end karl_total_income_correct_l231_231689


namespace inequality_generalization_l231_231642

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) :
  x + n^n / x^n ≥ n + 1 :=
sorry

end inequality_generalization_l231_231642


namespace olivia_pays_in_dollars_l231_231347

theorem olivia_pays_in_dollars (q_chips q_soda : ℕ) 
  (h_chips : q_chips = 4) (h_soda : q_soda = 12) : (q_chips + q_soda) / 4 = 4 := by
  sorry

end olivia_pays_in_dollars_l231_231347


namespace possible_measures_for_angle_A_l231_231678

-- Definition of angles A and B, and their relationship
def is_supplementary_angles (A B : ℕ) : Prop := A + B = 180

def is_multiple_of (A B : ℕ) : Prop := ∃ k : ℕ, k ≥ 1 ∧ A = k * B

-- Prove there are 17 possible measures for angle A.
theorem possible_measures_for_angle_A : 
  (∀ (A B : ℕ), (A > 0) ∧ (B > 0) ∧ is_multiple_of A B ∧ is_supplementary_angles A B → 
  A = B * 17) := 
sorry

end possible_measures_for_angle_A_l231_231678


namespace bankers_discount_is_correct_l231_231031

-- Define the given conditions
def TD := 45   -- True discount in Rs.
def FV := 270  -- Face value in Rs.

-- Calculate Present Value based on the given conditions
def PV := FV - TD

-- Define the formula for Banker's Discount
def BD := TD + (TD ^ 2 / PV)

-- Prove that the Banker's Discount is Rs. 54 given the conditions
theorem bankers_discount_is_correct : BD = 54 :=
by
  -- Steps to prove the theorem can be filled here
  -- Add "sorry" to skip the actual proof
  sorry

end bankers_discount_is_correct_l231_231031


namespace range_of_a_l231_231539

-- Define the conditions and what we want to prove
theorem range_of_a (a : ℝ) (x : ℝ) 
    (h1 : ∀ x, |x - 1| + |x + 1| ≥ 3 * a)
    (h2 : ∀ x, (2 * a - 1) ^ x ≤ 1 → (2 * a - 1) < 1 ∧ (2 * a - 1) > 0) :
    (1 / 2 < a ∧ a ≤ 2 / 3) :=
by
  sorry -- Here will be the proof

end range_of_a_l231_231539


namespace part_a_l231_231259

def f_X (X : Set (ℝ × ℝ)) (n : ℕ) : ℝ :=
  sorry  -- Placeholder for the largest possible area function

theorem part_a (X : Set (ℝ × ℝ)) (m n : ℕ) (h1 : m ≥ n) (h2 : n > 2) :
  f_X X m + f_X X n ≥ f_X X (m + 1) + f_X X (n - 1) :=
sorry

end part_a_l231_231259


namespace equation_satisfied_by_r_l231_231556

theorem equation_satisfied_by_r {x y z r : ℝ} (h1: x ≠ y) (h2: y ≠ z) (h3: z ≠ x) 
    (h4: x ≠ 0) (h5: y ≠ 0) (h6: z ≠ 0) 
    (h7: ∃ (r: ℝ), x * (y - z) = (y * (z - x)) / r ∧ y * (z - x) = (z * (y - x)) / r ∧ z * (y - x) = (x * (y - z)) * r) 
    : r^2 - r + 1 = 0 := 
sorry

end equation_satisfied_by_r_l231_231556


namespace probability_of_joining_between_1890_and_1969_l231_231786

theorem probability_of_joining_between_1890_and_1969 :
  let total_provinces_and_territories := 13
  let joined_1890_to_1929 := 3
  let joined_1930_to_1969 := 1
  let total_joined_between_1890_and_1969 := joined_1890_to_1929 + joined_1930_to_1969
  total_joined_between_1890_and_1969 / total_provinces_and_territories = 4 / 13 :=
by
  sorry

end probability_of_joining_between_1890_and_1969_l231_231786


namespace find_BF_pqsum_l231_231520

noncomputable def square_side_length : ℝ := 900
noncomputable def EF_length : ℝ := 400
noncomputable def m_angle_EOF : ℝ := 45
noncomputable def center_mid_to_side : ℝ := square_side_length / 2

theorem find_BF_pqsum :
  let G_mid : ℝ := center_mid_to_side
  let x : ℝ := G_mid - (2 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let y : ℝ := (1 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let BF := G_mid - y
  BF = 250 + 50 * Real.sqrt 7 ->
  250 + 50 + 7 = 307 := sorry

end find_BF_pqsum_l231_231520


namespace train_cross_time_l231_231586

theorem train_cross_time (length_of_train : ℕ) (speed_in_kmh : ℕ) (conversion_factor : ℕ) (speed_in_mps : ℕ) (time : ℕ) :
  length_of_train = 120 →
  speed_in_kmh = 72 →
  conversion_factor = 1000 / 3600 →
  speed_in_mps = speed_in_kmh * conversion_factor →
  time = length_of_train / speed_in_mps →
  time = 6 :=
by
  intros hlength hspeed hconversion hspeed_mps htime
  have : conversion_factor = 5 / 18 := sorry
  have : speed_in_mps = 20 := sorry
  exact sorry

end train_cross_time_l231_231586


namespace five_digit_palindromes_count_l231_231095

def num_five_digit_palindromes : ℕ :=
  let choices_for_A := 9
  let choices_for_B := 10
  let choices_for_C := 10
  choices_for_A * choices_for_B * choices_for_C

theorem five_digit_palindromes_count : num_five_digit_palindromes = 900 :=
by
  unfold num_five_digit_palindromes
  sorry

end five_digit_palindromes_count_l231_231095


namespace find_k_l231_231811

-- Definitions of conditions
variables (x y k : ℤ)

-- System of equations as given in the problem
def system_eq1 := x + 2 * y = 7 + k
def system_eq2 := 5 * x - y = k

-- Condition that solutions x and y are additive inverses
def y_is_add_inv := y = -x

-- The statement we need to prove
theorem find_k (hx : system_eq1 x y k) (hy : system_eq2 x y k) (hz : y_is_add_inv x y) : k = -6 :=
by
  sorry -- proof will go here

end find_k_l231_231811


namespace decomposition_of_cube_l231_231835

theorem decomposition_of_cube (m : ℕ) (h : m^2 - m + 1 = 73) : m = 9 :=
sorry

end decomposition_of_cube_l231_231835


namespace train_crosses_signal_pole_in_20_seconds_l231_231969

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 285
noncomputable def total_time_to_cross_platform : ℝ := 39

-- Define the speed of the train
noncomputable def train_speed : ℝ := (train_length + platform_length) / total_time_to_cross_platform

-- Define the expected time to cross the signal pole
noncomputable def time_to_cross_signal_pole : ℝ := train_length / train_speed

theorem train_crosses_signal_pole_in_20_seconds :
  time_to_cross_signal_pole = 20 := by
  sorry

end train_crosses_signal_pole_in_20_seconds_l231_231969


namespace valid_three_digit_numbers_count_l231_231533

noncomputable def count_valid_numbers : ℕ :=
  let valid_first_digits := [2, 4, 6, 8].length
  let valid_other_digits := [0, 2, 4, 6, 8].length
  let total_even_digit_3_digit_numbers := valid_first_digits * valid_other_digits * valid_other_digits
  let no_4_or_8_first_digits := [2, 6].length
  let no_4_or_8_other_digits := [0, 2, 6].length
  let numbers_without_4_or_8 := no_4_or_8_first_digits * no_4_or_8_other_digits * no_4_or_8_other_digits
  let numbers_with_4_or_8 := total_even_digit_3_digit_numbers - numbers_without_4_or_8
  let valid_even_sum_count := 50  -- Assumed from the manual checking
  valid_even_sum_count

theorem valid_three_digit_numbers_count :
  count_valid_numbers = 50 :=
by
  sorry

end valid_three_digit_numbers_count_l231_231533


namespace rowing_time_to_and_fro_l231_231814

noncomputable def rowing_time (distance rowing_speed current_speed : ℤ) : ℤ :=
  let speed_to_place := rowing_speed - current_speed
  let speed_back_place := rowing_speed + current_speed
  let time_to_place := distance / speed_to_place
  let time_back_place := distance / speed_back_place
  time_to_place + time_back_place

theorem rowing_time_to_and_fro (distance rowing_speed current_speed : ℤ) :
  distance = 72 → rowing_speed = 10 → current_speed = 2 → rowing_time distance rowing_speed current_speed = 15 := by
  intros h_dist h_row_speed h_curr_speed
  rw [h_dist, h_row_speed, h_curr_speed]
  sorry

end rowing_time_to_and_fro_l231_231814


namespace George_spending_l231_231648

theorem George_spending (B m s : ℝ) (h1 : m = 0.25 * (B - s)) (h2 : s = 0.05 * (B - m)) : 
  (m + s) / B = 1 := 
by
  sorry

end George_spending_l231_231648


namespace largest_number_among_selected_students_l231_231233

def total_students := 80

def smallest_numbers (x y : ℕ) : Prop :=
  x = 6 ∧ y = 14

noncomputable def selected_students (n : ℕ) : ℕ :=
  6 + (n - 1) * 8

theorem largest_number_among_selected_students :
  ∀ (x y : ℕ), smallest_numbers x y → (selected_students 10 = 78) :=
by
  intros x y h
  rw [smallest_numbers] at h
  have h1 : x = 6 := h.1
  have h2 : y = 14 := h.2
  exact rfl

#check largest_number_among_selected_students

end largest_number_among_selected_students_l231_231233


namespace Eugene_buys_four_t_shirts_l231_231329

noncomputable def t_shirt_price : ℝ := 20
noncomputable def pants_price : ℝ := 80
noncomputable def shoes_price : ℝ := 150
noncomputable def discount : ℝ := 0.10

noncomputable def discounted_t_shirt_price : ℝ := t_shirt_price - (t_shirt_price * discount)
noncomputable def discounted_pants_price : ℝ := pants_price - (pants_price * discount)
noncomputable def discounted_shoes_price : ℝ := shoes_price - (shoes_price * discount)

noncomputable def num_pants : ℝ := 3
noncomputable def num_shoes : ℝ := 2
noncomputable def total_paid : ℝ := 558

noncomputable def total_cost_of_pants_and_shoes : ℝ := (num_pants * discounted_pants_price) + (num_shoes * discounted_shoes_price)
noncomputable def remaining_cost_for_t_shirts : ℝ := total_paid - total_cost_of_pants_and_shoes

noncomputable def num_t_shirts : ℝ := remaining_cost_for_t_shirts / discounted_t_shirt_price

theorem Eugene_buys_four_t_shirts : num_t_shirts = 4 := by
  sorry

end Eugene_buys_four_t_shirts_l231_231329


namespace find_value_of_expression_l231_231498

noncomputable def p : ℝ := 3
noncomputable def q : ℝ := 7
noncomputable def r : ℝ := 5

def inequality_holds (f : ℝ → ℝ) : Prop :=
  ∀ x, (f x ≥ 0 ↔ (x ∈ Set.Icc 3 7 ∨ x > 5))

def given_condition : Prop := p < q

theorem find_value_of_expression (f : ℝ → ℝ)
  (h : inequality_holds f)
  (hc : given_condition) :
  p + 2*q + 3*r = 32 := 
sorry

end find_value_of_expression_l231_231498


namespace repeating_decimal_fraction_l231_231458

noncomputable def x : ℚ := 75 / 99  -- 0.\overline{75}
noncomputable def y : ℚ := 223 / 99  -- 2.\overline{25}

theorem repeating_decimal_fraction : (x / y) = 2475 / 7329 :=
by
  -- Further proof details can be added here
  sorry

end repeating_decimal_fraction_l231_231458


namespace total_family_members_l231_231452

variable (members_father_side : Nat) (percent_incr : Nat)
variable (members_mother_side := members_father_side + (members_father_side * percent_incr / 100))
variable (total_members := members_father_side + members_mother_side)

theorem total_family_members 
  (h1 : members_father_side = 10) 
  (h2 : percent_incr = 30) :
  total_members = 23 :=
by
  sorry

end total_family_members_l231_231452


namespace value_of_a_12_l231_231566

variable {a : ℕ → ℝ} (h1 : a 6 + a 10 = 20) (h2 : a 4 = 2)

theorem value_of_a_12 : a 12 = 18 :=
by
  sorry

end value_of_a_12_l231_231566


namespace machines_finish_together_in_2_hours_l231_231536

def machineA_time := 4
def machineB_time := 12
def machineC_time := 6

def machineA_rate := 1 / machineA_time
def machineB_rate := 1 / machineB_time
def machineC_rate := 1 / machineC_time

def combined_rate := machineA_rate + machineB_rate + machineC_rate
def total_time := 1 / combined_rate

-- We want to prove that the total_time for machines A, B, and C to finish the job together is 2 hours.
theorem machines_finish_together_in_2_hours : total_time = 2 := by
  sorry

end machines_finish_together_in_2_hours_l231_231536


namespace product_of_means_eq_pm20_l231_231672

theorem product_of_means_eq_pm20 :
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  a * b = 20 ∨ a * b = -20 :=
by
  -- Placeholders for the actual proof
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  sorry

end product_of_means_eq_pm20_l231_231672


namespace largest_number_le_1_1_from_set_l231_231146

def is_largest_le (n : ℚ) (l : List ℚ) (bound : ℚ) : Prop :=
  (n ∈ l ∧ n ≤ bound) ∧ ∀ m ∈ l, m ≤ bound → m ≤ n

theorem largest_number_le_1_1_from_set : 
  is_largest_le (9/10) [14/10, 9/10, 12/10, 5/10, 13/10] (11/10) :=
by 
  sorry

end largest_number_le_1_1_from_set_l231_231146


namespace egor_last_payment_l231_231922

theorem egor_last_payment (a b c d : ℕ) (h_sum : a + b + c + d = 28)
  (h1 : b ≥ 2 * a) (h2 : c ≥ 2 * b) (h3 : d ≥ 2 * c) : d = 18 := by
  sorry

end egor_last_payment_l231_231922


namespace value_of_7_star_3_l231_231201

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - a * b

theorem value_of_7_star_3 : star 7 3 = 16 :=
by
  -- Proof would go here
  sorry

end value_of_7_star_3_l231_231201


namespace ab_c_work_days_l231_231487

noncomputable def W_ab : ℝ := 1 / 15
noncomputable def W_c : ℝ := 1 / 30
noncomputable def W_abc : ℝ := W_ab + W_c

theorem ab_c_work_days :
  (1 / W_abc) = 10 :=
by
  sorry

end ab_c_work_days_l231_231487


namespace height_of_second_triangle_l231_231294

theorem height_of_second_triangle
  (base1 : ℝ) (height1 : ℝ) (base2 : ℝ) (height2 : ℝ)
  (h_base1 : base1 = 15)
  (h_height1 : height1 = 12)
  (h_base2 : base2 = 20)
  (h_area_relation : (base2 * height2) / 2 = 2 * (base1 * height1) / 2) :
  height2 = 18 :=
sorry

end height_of_second_triangle_l231_231294


namespace area_inside_arcs_outside_square_l231_231156

theorem area_inside_arcs_outside_square (r : ℝ) (θ : ℝ) (L : ℝ) (a b c d : ℝ) :
  r = 6 ∧ θ = 45 ∧ L = 12 ∧ a = 15 ∧ b = 0 ∧ c = 15 ∧ d = 144 →
  (a + b + c + d = 174) :=
by
  intros h
  sorry

end area_inside_arcs_outside_square_l231_231156


namespace circles_disjoint_l231_231307

theorem circles_disjoint (a : ℝ) : ((x - 1)^2 + (y - 1)^2 = 4) ∧ (x^2 + (y - a)^2 = 1) → (a < 1 - 2 * Real.sqrt 2 ∨ a > 1 + 2 * Real.sqrt 2) :=
by sorry

end circles_disjoint_l231_231307


namespace max_rational_sums_is_1250_l231_231318

/-- We define a structure to represent the problem's conditions. -/
structure GridConfiguration where
  grid_rows : Nat
  grid_cols : Nat
  total_numbers : Nat
  rational_count : Nat
  irrational_count : Nat
  (h_grid : grid_rows = 50)
  (h_grid_col : grid_cols = 50)
  (h_total_numbers : total_numbers = 100)
  (h_rational_count : rational_count = 50)
  (h_irrational_count : irrational_count = 50)

/-- We define a function to calculate the number of rational sums in the grid. -/
def max_rational_sums (config : GridConfiguration) : Nat :=
  let x := config.rational_count / 2 -- rational numbers to the left
  let ni := 2 * x * x - 100 * x + 2500
  let rational_sums := 2500 - ni
  rational_sums

/-- The theorem stating the maximum number of rational sums is 1250. -/
theorem max_rational_sums_is_1250 (config : GridConfiguration) : max_rational_sums config = 1250 :=
  sorry

end max_rational_sums_is_1250_l231_231318


namespace solution_for_factorial_equation_l231_231493

theorem solution_for_factorial_equation:
  { (n, k) : ℕ × ℕ | 0 < n ∧ 0 < k ∧ n! + n = n^k } = {(2,2), (3,2), (5,3)} :=
by
  sorry

end solution_for_factorial_equation_l231_231493


namespace greatest_number_divides_with_remainders_l231_231284

theorem greatest_number_divides_with_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end greatest_number_divides_with_remainders_l231_231284


namespace base_k_to_decimal_is_5_l231_231408

theorem base_k_to_decimal_is_5 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 42) : k = 5 := sorry

end base_k_to_decimal_is_5_l231_231408


namespace acceptable_outfits_l231_231909

-- Definitions based on the given conditions
def shirts : Nat := 8
def pants : Nat := 5
def hats : Nat := 7
def pant_colors : List String := ["red", "black", "blue", "gray", "green"]
def shirt_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]
def hat_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]

-- Axiom that ensures distinct colors for pants, shirts, and hats.
axiom distinct_colors : ∀ color ∈ pant_colors, color ∈ shirt_colors ∧ color ∈ hat_colors

-- Problem statement
theorem acceptable_outfits : 
  let total_outfits := shirts * pants * hats
  let monochrome_outfits := List.length pant_colors
  let acceptable_outfits := total_outfits - monochrome_outfits
  acceptable_outfits = 275 :=
by
  sorry

end acceptable_outfits_l231_231909


namespace trapezoid_problem_l231_231913

theorem trapezoid_problem (b h x : ℝ) 
  (h1 : x = (12500 / (x - 75)) - 75)
  (h_cond : (b + 75) / (b + 25) = 3 / 2)
  (b_solution : b = 75) :
  (⌊(x^2 / 100)⌋ : ℤ) = 181 :=
by
  -- The statement only requires us to assert the proof goal
  sorry

end trapezoid_problem_l231_231913


namespace group_D_forms_a_definite_set_l231_231676

theorem group_D_forms_a_definite_set : 
  ∃ (S : Set ℝ), S = { x : ℝ | x = 1 ∨ x = -1 } :=
by
  sorry

end group_D_forms_a_definite_set_l231_231676


namespace sculptures_not_on_display_eq_1200_l231_231471

-- Define the number of pieces of art in the gallery
def total_pieces_art := 2700

-- Define the number of pieces on display (1/3 of total pieces)
def pieces_on_display := total_pieces_art / 3

-- Define the number of pieces not on display
def pieces_not_on_display := total_pieces_art - pieces_on_display

-- Define the number of sculptures on display (1/6 of pieces on display)
def sculptures_on_display := pieces_on_display / 6

-- Define the number of paintings not on display (1/3 of pieces not on display)
def paintings_not_on_display := pieces_not_on_display / 3

-- Prove the number of sculptures not on display
theorem sculptures_not_on_display_eq_1200 :
  total_pieces_art = 2700 →
  pieces_on_display = total_pieces_art / 3 →
  pieces_not_on_display = total_pieces_art - pieces_on_display →
  sculptures_on_display = pieces_on_display / 6 →
  paintings_not_on_display = pieces_not_on_display / 3 →
  pieces_not_on_display - paintings_not_on_display = 1200 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end sculptures_not_on_display_eq_1200_l231_231471


namespace graveling_cost_is_correct_l231_231056

noncomputable def graveling_cost (lawn_length lawn_breadth road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_breadth
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_area := road1_area + road2_area - intersection_area
  total_area * cost_per_sqm

theorem graveling_cost_is_correct :
  graveling_cost 80 60 10 2 = 2600 := by
  sorry

end graveling_cost_is_correct_l231_231056


namespace sin_squared_alpha_eq_one_add_sin_squared_beta_l231_231168

variable {α θ β : ℝ}

theorem sin_squared_alpha_eq_one_add_sin_squared_beta
  (h1 : Real.sin α = Real.sin θ + Real.cos θ)
  (h2 : Real.sin β ^ 2 = 2 * Real.sin θ * Real.cos θ) :
  Real.sin α ^ 2 = 1 + Real.sin β ^ 2 := 
sorry

end sin_squared_alpha_eq_one_add_sin_squared_beta_l231_231168


namespace least_positive_multiple_of_24_gt_450_l231_231050

theorem least_positive_multiple_of_24_gt_450 : 
  ∃ n : ℕ, n > 450 ∧ (∃ k : ℕ, n = 24 * k) → n = 456 :=
by 
  sorry

end least_positive_multiple_of_24_gt_450_l231_231050


namespace range_of_a_l231_231444

noncomputable def f (x : ℝ) := Real.log (x + 1)
def A (x : ℝ) := (f (1 - 2 * x) > f x)
def B (a x : ℝ) := (a - 1 < x) ∧ (x < 2 * a^2)

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, A x ∧ B a x) ↔ (a < -1 / 2) ∨ (1 < a ∧ a < 4 / 3) :=
sorry

end range_of_a_l231_231444


namespace fibonacci_rabbits_l231_231681

theorem fibonacci_rabbits : 
  ∀ (F : ℕ → ℕ), 
    (F 0 = 1) ∧ 
    (F 1 = 1) ∧ 
    (∀ n, F (n + 2) = F n + F (n + 1)) → 
    F 12 = 233 := 
by 
  intro F h; sorry

end fibonacci_rabbits_l231_231681


namespace find_amount_with_r_l231_231769

variable (p q r s : ℝ) (total : ℝ := 9000)

-- Condition 1: Total amount is 9000 Rs
def total_amount_condition := p + q + r + s = total

-- Condition 2: r has three-quarters of the combined amount of p, q, and s
def r_amount_condition := r = (3/4) * (p + q + s)

-- The goal is to prove that r = 10800
theorem find_amount_with_r (h1 : total_amount_condition p q r s) (h2 : r_amount_condition p q r s) :
  r = 10800 :=
sorry

end find_amount_with_r_l231_231769


namespace initial_water_amount_l231_231761

theorem initial_water_amount 
  (W : ℝ) 
  (evap_rate : ℝ) 
  (days : ℕ) 
  (percentage_evaporated : ℝ) 
  (evap_rate_eq : evap_rate = 0.012) 
  (days_eq : days = 50) 
  (percentage_evaporated_eq : percentage_evaporated = 0.06) 
  (total_evaporated_eq : evap_rate * days = 0.6) 
  (percentage_condition : percentage_evaporated * W = evap_rate * days) 
  : W = 10 := 
  by sorry

end initial_water_amount_l231_231761


namespace sin_sum_cos_product_l231_231186

theorem sin_sum_cos_product (A B C : Real) (h : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2) :=
by
  sorry

end sin_sum_cos_product_l231_231186


namespace intersection_complement_l231_231013

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5}

-- Define set M
def M : Set ℕ := {0, 3, 5}

-- Define set N
def N : Set ℕ := {1, 4, 5}

-- Define the complement of N in U
def complement_U_N : Set ℕ := U \ N

-- The main theorem statement
theorem intersection_complement : M ∩ complement_U_N = {0, 3} :=
by
  -- The proof would go here
  sorry

end intersection_complement_l231_231013


namespace vector_dot_product_problem_l231_231107

theorem vector_dot_product_problem :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (-1, 3)
  let C : ℝ × ℝ := (2, 1)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  let dot_prod := AB.1 * (2 * AC.1 + BC.1) + AB.2 * (2 * AC.2 + BC.2)
  dot_prod = -14 :=
by
  sorry

end vector_dot_product_problem_l231_231107


namespace weight_of_fish_in_barrel_l231_231667

/-- 
Given a barrel with an initial weight of 54 kg when full of fish,
and a weight of 29 kg after removing half of the fish,
prove that the initial weight of the fish in the barrel was 50 kg.
-/
theorem weight_of_fish_in_barrel (B F : ℝ)
  (h1: B + F = 54)
  (h2: B + F / 2 = 29) : F = 50 := 
sorry

end weight_of_fish_in_barrel_l231_231667


namespace sector_central_angle_l231_231733

theorem sector_central_angle 
  (R : ℝ) (P : ℝ) (θ : ℝ) (π : ℝ) (L : ℝ)
  (h1 : P = 83) 
  (h2 : R = 14)
  (h3 : P = 2 * R + L)
  (h4 : L = θ * R)
  (degree_conversion : θ * (180 / π) = 225) : 
  θ * (180 / π) = 225 :=
by sorry

end sector_central_angle_l231_231733


namespace big_container_capacity_l231_231573

theorem big_container_capacity (C : ℝ)
    (h1 : 0.75 * C - 0.40 * C = 14) : C = 40 :=
  sorry

end big_container_capacity_l231_231573


namespace pirates_total_coins_l231_231402

theorem pirates_total_coins :
  ∀ (x : ℕ), (x * (x + 1)) / 2 = 5 * x → 6 * x = 54 :=
by
  intro x
  intro h
  -- proof omitted
  sorry

end pirates_total_coins_l231_231402


namespace smallest_possible_positive_value_l231_231776

theorem smallest_possible_positive_value (a b : ℤ) (h : a > b) :
  ∃ (x : ℚ), x = (a + b) / (a - b) + (a - b) / (a + b) ∧ x = 2 :=
sorry

end smallest_possible_positive_value_l231_231776


namespace white_tiles_in_square_l231_231759

theorem white_tiles_in_square (n S : ℕ) (hn : n * n = S) (black_tiles : ℕ) (hblack_tiles : black_tiles = 81) (diagonal_black_tiles : n = 9) :
  S - black_tiles = 72 :=
by
  sorry

end white_tiles_in_square_l231_231759


namespace problem_solution_l231_231117

theorem problem_solution (a b c : ℝ) (h1 : a^2 - b^2 = 5) (h2 : a * b = 2) (h3 : a^2 + b^2 + c^2 = 8) : 
  a^4 + b^4 + c^4 = 38 :=
sorry

end problem_solution_l231_231117


namespace number_of_subsets_of_set_l231_231873

theorem number_of_subsets_of_set {n : ℕ} (h : n = 2016) :
  (2^2016) = 2^2016 :=
by
  sorry

end number_of_subsets_of_set_l231_231873


namespace holes_in_compartment_l231_231703

theorem holes_in_compartment :
  ∀ (rect : Type) (holes : ℕ) (compartments : ℕ),
  compartments = 9 →
  holes = 20 →
  (∃ (compartment : rect ) (n : ℕ), n ≥ 3) :=
by
  intros rect holes compartments h_compartments h_holes
  sorry

end holes_in_compartment_l231_231703


namespace largest_number_of_gold_coins_l231_231404

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l231_231404


namespace question1_geometric_sequence_question2_minimum_term_l231_231301

theorem question1_geometric_sequence (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  q = 0 →
  (a 1 = 1 / 2) →
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * (r ^ n)) →
  (p = 0 ∨ p = 1) :=
by sorry

theorem question2_minimum_term (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  p = 1 →
  (a 1 = 1 / 2) →
  (a 4 = min (min (a 1) (a 2)) (a 3)) →
  3 ≤ q ∧ q ≤ 27 / 4 :=
by sorry

end question1_geometric_sequence_question2_minimum_term_l231_231301


namespace jogger_ahead_distance_l231_231986

/-- The jogger is running at a constant speed of 9 km/hr, the train at a speed of 45 km/hr,
    it is 210 meters long and passes the jogger in 41 seconds.
    Prove the jogger is 200 meters ahead of the train. -/
theorem jogger_ahead_distance 
  (v_j : ℝ) (v_t : ℝ) (L : ℝ) (t : ℝ) (d : ℝ) 
  (hv_j : v_j = 9) (hv_t : v_t = 45) (hL : L = 210) (ht : t = 41) :
  d = 200 :=
by {
  -- The conditions and the final proof step, 
  -- actual mathematical proofs steps are not necessary according to the problem statement.
  sorry
}

end jogger_ahead_distance_l231_231986


namespace Drew_age_is_12_l231_231523

def Sam_age_current : ℕ := 46
def Sam_age_in_five_years : ℕ := Sam_age_current + 5

def Drew_age_now (D : ℕ) : Prop :=
  Sam_age_in_five_years = 3 * (D + 5)

theorem Drew_age_is_12 (D : ℕ) (h : Drew_age_now D) : D = 12 :=
by
  sorry

end Drew_age_is_12_l231_231523


namespace find_a8_a12_l231_231078

noncomputable def geometric_sequence_value_8_12 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a 0 else a 0 * q^n

theorem find_a8_a12 (a : ℕ → ℝ) (q : ℝ) (terms_geometric : ∀ n, a n = a 0 * q^n)
  (h2_6 : a 2 + a 6 = 3) (h6_10 : a 6 + a 10 = 12) :
  a 8 + a 12 = 24 :=
by
  sorry

end find_a8_a12_l231_231078


namespace num_shirts_sold_l231_231434

theorem num_shirts_sold (p_jeans : ℕ) (c_shirt : ℕ) (total_earnings : ℕ) (h1 : p_jeans = 10) (h2 : c_shirt = 10) (h3 : total_earnings = 400) : ℕ :=
  let c_jeans := 2 * c_shirt
  let n_shirts := 20
  have h4 : p_jeans * c_jeans + n_shirts * c_shirt = total_earnings := by sorry
  n_shirts

end num_shirts_sold_l231_231434


namespace third_measurement_multiple_of_one_l231_231074

-- Define the lengths in meters
def length1_meter : ℕ := 6
def length2_meter : ℕ := 5

-- Convert lengths to centimeters
def length1_cm := length1_meter * 100
def length2_cm := length2_meter * 100

-- Define that the greatest common divisor (gcd) of lengths in cm is 100 cm
def gcd_length : ℕ := Nat.gcd length1_cm length2_cm

-- Given that the gcd is 100 cm
theorem third_measurement_multiple_of_one
  (h1 : gcd_length = 100) :
  ∃ n : ℕ, n = 1 :=
sorry

end third_measurement_multiple_of_one_l231_231074


namespace more_karabases_than_barabases_l231_231037

/-- In the fairy-tale land of Perra-Terra, each Karabas is acquainted with nine Barabases, 
    and each Barabas is acquainted with ten Karabases. We aim to prove that there are more Karabases than Barabases. -/
theorem more_karabases_than_barabases (K B : ℕ) (h1 : 9 * K = 10 * B) : K > B := 
by {
    -- Following the conditions and conclusion
    sorry
}

end more_karabases_than_barabases_l231_231037


namespace science_fair_unique_students_l231_231659

/-!
# Problem statement:
At Euclid Middle School, there are three clubs participating in the Science Fair: the Robotics Club, the Astronomy Club, and the Chemistry Club.
There are 15 students in the Robotics Club, 10 students in the Astronomy Club, and 12 students in the Chemistry Club.
Assuming 2 students are members of all three clubs, prove that the total number of unique students participating in the Science Fair is 33.
-/

theorem science_fair_unique_students (R A C : ℕ) (all_three : ℕ) (hR : R = 15) (hA : A = 10) (hC : C = 12) (h_all_three : all_three = 2) :
    R + A + C - 2 * all_three = 33 :=
by
  -- Proof goes here
  sorry

end science_fair_unique_students_l231_231659


namespace internet_plan_cost_effective_l231_231775

theorem internet_plan_cost_effective (d : ℕ) :
  (∀ (d : ℕ), d > 150 → 1500 + 10 * d < 20 * d) ↔ d = 151 :=
sorry

end internet_plan_cost_effective_l231_231775


namespace composite_number_l231_231668

theorem composite_number (n : ℕ) (h : n ≥ 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = (10 ^ n + 1) * (10 ^ (n + 1) - 1) / 9 :=
by sorry

end composite_number_l231_231668


namespace problem_thre_is_15_and_10_percent_l231_231077

theorem problem_thre_is_15_and_10_percent (x y : ℝ) 
  (h1 : 3 = 0.15 * x) 
  (h2 : 3 = 0.10 * y) : 
  x - y = -10 := 
by 
  sorry

end problem_thre_is_15_and_10_percent_l231_231077


namespace asha_borrowed_from_mother_l231_231656

def total_money (M : ℕ) : ℕ := 20 + 40 + 70 + 100 + M

def remaining_money_after_spending_3_4 (total : ℕ) : ℕ := total * 1 / 4

theorem asha_borrowed_from_mother : ∃ M : ℕ, total_money M = 260 ∧ remaining_money_after_spending_3_4 (total_money M) = 65 :=
by
  sorry

end asha_borrowed_from_mother_l231_231656


namespace problem_statement_l231_231456

-- Define the universal set U, and sets A and B
def U : Set ℕ := { n | 1 ≤ n ∧ n ≤ 10 }
def A : Set ℕ := {1, 2, 3, 5, 8}
def B : Set ℕ := {1, 3, 5, 7, 9}

-- Define the complement of set A with respect to U
def complement_U_A : Set ℕ := { n | n ∈ U ∧ n ∉ A }

-- Define the intersection of complement_U_A and B
def intersection_complement_U_A_B : Set ℕ := { n | n ∈ complement_U_A ∧ n ∈ B }

-- Prove the given statement
theorem problem_statement : intersection_complement_U_A_B = {7, 9} := by
  sorry

end problem_statement_l231_231456


namespace age_transition_l231_231208

theorem age_transition (initial_ages : List ℕ) : 
  initial_ages = [19, 34, 37, 42, 48] →
  (∃ x, 0 < x ∧ x < 10 ∧ 
  new_ages = List.map (fun age => age + x) initial_ages ∧ 
  new_ages = [25, 40, 43, 48, 54]) →
  x = 6 :=
by
  intros h_initial_ages h_exist_x
  sorry

end age_transition_l231_231208


namespace domain_of_f_l231_231861

noncomputable def f (x : ℝ) := Real.log (1 - x)

theorem domain_of_f : ∀ x, f x = Real.log (1 - x) → (1 - x > 0) →  x < 1 :=
by
  intro x h₁ h₂
  exact lt_of_sub_pos h₂

end domain_of_f_l231_231861


namespace remainder_of_2_pow_2018_plus_1_mod_2018_l231_231671

theorem remainder_of_2_pow_2018_plus_1_mod_2018 : (2 ^ 2018 + 1) % 2018 = 2 := by
  sorry

end remainder_of_2_pow_2018_plus_1_mod_2018_l231_231671


namespace max_consecutive_sum_l231_231983

theorem max_consecutive_sum (N a : ℕ) (h : N * (2 * a + N - 1) = 240) : N ≤ 15 :=
by
  -- proof goes here
  sorry

end max_consecutive_sum_l231_231983


namespace total_snakes_among_pet_owners_l231_231106

theorem total_snakes_among_pet_owners :
  let owns_only_snakes := 15
  let owns_cats_and_snakes := 7
  let owns_dogs_and_snakes := 10
  let owns_birds_and_snakes := 2
  let owns_snakes_and_hamsters := 3
  let owns_cats_dogs_and_snakes := 4
  let owns_cats_snakes_and_hamsters := 2
  let owns_all_categories := 1
  owns_only_snakes + owns_cats_and_snakes + owns_dogs_and_snakes + owns_birds_and_snakes + owns_snakes_and_hamsters + owns_cats_dogs_and_snakes + owns_cats_snakes_and_hamsters + owns_all_categories = 44 :=
by
  sorry

end total_snakes_among_pet_owners_l231_231106


namespace total_distance_biked_two_days_l231_231327

def distance_yesterday : ℕ := 12
def distance_today : ℕ := (2 * distance_yesterday) - 3
def total_distance_biked : ℕ := distance_yesterday + distance_today

theorem total_distance_biked_two_days : total_distance_biked = 33 :=
by {
  -- Given distance_yesterday = 12
  -- distance_today calculated as (2 * distance_yesterday) - 3 = 21
  -- total_distance_biked = distance_yesterday + distance_today = 33
  sorry
}

end total_distance_biked_two_days_l231_231327


namespace fido_reach_fraction_simplified_l231_231055

noncomputable def fidoReach (s r : ℝ) : ℝ :=
  let octagonArea := 2 * (1 + Real.sqrt 2) * s^2
  let circleArea := Real.pi * (s / Real.sqrt (2 + Real.sqrt 2))^2
  circleArea / octagonArea

theorem fido_reach_fraction_simplified (s : ℝ) :
  (∃ a b : ℕ, fidoReach s (s / Real.sqrt (2 + Real.sqrt 2)) = (Real.sqrt a / b) * Real.pi ∧ a * b = 16) :=
  sorry

end fido_reach_fraction_simplified_l231_231055


namespace condition_necessary_but_not_sufficient_l231_231334

-- Definitions based on given conditions
variables {a b c : ℝ}

-- The condition that needs to be qualified
def condition (a b c : ℝ) := a > 0 ∧ b^2 - 4 * a * c < 0

-- The statement to be verified
def statement (a b c : ℝ) := ∀ x : ℝ, a * x^2 + b * x + c > 0

-- Prove that the condition is a necessary but not sufficient condition for the statement
theorem condition_necessary_but_not_sufficient :
  condition a b c → (¬ (condition a b c ↔ statement a b c)) :=
by
  sorry

end condition_necessary_but_not_sufficient_l231_231334


namespace total_earnings_l231_231843

theorem total_earnings (x y : ℝ) (h : 20 * x * y = 18 * x * y + 150) : 
  18 * x * y + 20 * x * y + 20 * x * y = 4350 :=
by sorry

end total_earnings_l231_231843


namespace pentagon_side_length_l231_231839

-- Define the side length of the equilateral triangle
def side_length_triangle : ℚ := 20 / 9

-- Define the perimeter of the equilateral triangle
def perimeter_triangle : ℚ := 3 * side_length_triangle

-- Define the side length of the regular pentagon
def side_length_pentagon : ℚ := 4 / 3

-- Prove that the side length of the regular pentagon has the same perimeter as the equilateral triangle
theorem pentagon_side_length (s : ℚ) (h1 : s = side_length_pentagon) :
  5 * s = perimeter_triangle :=
by
  -- Provide the solution
  sorry

end pentagon_side_length_l231_231839


namespace solve_for_x_l231_231640

-- Assumptions and conditions of the problem
def a : ℚ := 4 / 7
def b : ℚ := 1 / 5
def c : ℚ := 12
def d : ℚ := 105

-- The statement of the problem
theorem solve_for_x (x : ℚ) (h : a * b * x = c) : x = d :=
by sorry

end solve_for_x_l231_231640


namespace calculate_expression_l231_231209

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l231_231209


namespace find_usual_time_l231_231849

noncomputable def journey_time (S T : ℝ) : Prop :=
  (6 / 5) = (T + (1 / 5)) / T

theorem find_usual_time (S T : ℝ) (h1 : ∀ S T, S / (5 / 6 * S) = (T + (12 / 60)) / T) : T = 1 :=
by
  -- Let the conditions defined by the user be:
  -- h1 : condition (e.g., the cab speed and time relationship)
  -- Given that the cab is \(\frac{5}{6}\) times its speed and is late by 12 minutes
  let h1 := journey_time S T
  sorry

end find_usual_time_l231_231849


namespace neg_p_necessary_not_sufficient_for_neg_p_or_q_l231_231822

variables (p q : Prop)

theorem neg_p_necessary_not_sufficient_for_neg_p_or_q :
  (¬ p → ¬ (p ∨ q)) ∧ (¬ (p ∨ q) → ¬ p) :=
by {
  sorry
}

end neg_p_necessary_not_sufficient_for_neg_p_or_q_l231_231822


namespace perpendicular_line_directional_vector_l231_231116

theorem perpendicular_line_directional_vector
  (l1 : ℝ → ℝ → Prop)
  (l2 : ℝ → ℝ → Prop)
  (perpendicular : ∀ x y, l1 x y ↔ l2 y (-x))
  (l2_eq : ∀ x y, l2 x y ↔ 2 * x + 5 * y = 1) :
  ∃ d1 d2, (d1, d2) = (5, -2) ∧ (d1 * 2 + d2 * 5 = 0) :=
by
  sorry

end perpendicular_line_directional_vector_l231_231116


namespace mult_469158_9999_l231_231124

theorem mult_469158_9999 : 469158 * 9999 = 4691176842 := 
by sorry

end mult_469158_9999_l231_231124


namespace cost_per_liter_l231_231628

/-
Given:
- Service cost per vehicle: $2.10
- Number of mini-vans: 3
- Number of trucks: 2
- Total cost: $299.1
- Mini-van's tank size: 65 liters
- Truck's tank is 120% bigger than a mini-van's tank
- All tanks are empty

Prove that the cost per liter of fuel is $0.60
-/

theorem cost_per_liter (service_cost_per_vehicle : ℝ) 
(number_of_minivans number_of_trucks : ℕ)
(total_cost : ℝ)
(minivan_tank_size : ℝ)
(truck_tank_multiplier : ℝ)
(fuel_cost : ℝ)
(total_fuel : ℝ) :
  service_cost_per_vehicle = 2.10 ∧
  number_of_minivans = 3 ∧
  number_of_trucks = 2 ∧
  total_cost = 299.1 ∧
  minivan_tank_size = 65 ∧
  truck_tank_multiplier = 1.2 ∧
  fuel_cost = (total_cost - (number_of_minivans + number_of_trucks) * service_cost_per_vehicle) ∧
  total_fuel = (number_of_minivans * minivan_tank_size + number_of_trucks * (minivan_tank_size * (1 + truck_tank_multiplier))) →
  (fuel_cost / total_fuel) = 0.60 :=
sorry

end cost_per_liter_l231_231628


namespace num_dinosaur_dolls_l231_231630

-- Define the number of dinosaur dolls
def dinosaur_dolls : Nat := 3

-- Define the theorem to prove the number of dinosaur dolls
theorem num_dinosaur_dolls : dinosaur_dolls = 3 := by
  -- Add sorry to skip the proof
  sorry

end num_dinosaur_dolls_l231_231630


namespace infinitely_many_composite_z_l231_231409

theorem infinitely_many_composite_z (m n : ℕ) (h_m : m > 1) : ¬ (Nat.Prime (n^4 + 4*m^4)) :=
by
  sorry

end infinitely_many_composite_z_l231_231409


namespace total_cost_proof_l231_231885

noncomputable def cost_proof : Prop :=
  let M := 158.4
  let R := 66
  let F := 22
  (10 * M = 24 * R) ∧ (6 * F = 2 * R) ∧ (F = 22) →
  (4 * M + 3 * R + 5 * F = 941.6)

theorem total_cost_proof : cost_proof :=
by
  sorry

end total_cost_proof_l231_231885


namespace tiling_ratio_l231_231519

theorem tiling_ratio (n a b : ℕ) (ha : a ≠ 0) (H : b = a * 2^(n/2)) :
  b / a = 2^(n/2) :=
  by
  sorry

end tiling_ratio_l231_231519


namespace arrange_scores_l231_231489

variable {K Q M S : ℝ}

theorem arrange_scores (h1 : Q > K) (h2 : M > S) (h3 : S < max Q (max M K)) : S < M ∧ M < Q := by
  sorry

end arrange_scores_l231_231489


namespace attraction_ticket_cost_l231_231955

theorem attraction_ticket_cost
  (cost_park_entry : ℕ)
  (cost_attraction_parent : ℕ)
  (total_paid : ℕ)
  (num_children : ℕ)
  (num_parents : ℕ)
  (num_grandmother : ℕ)
  (x : ℕ)
  (h_costs : cost_park_entry = 5)
  (h_attraction_parent : cost_attraction_parent = 4)
  (h_family : num_children = 4 ∧ num_parents = 2 ∧ num_grandmother = 1)
  (h_total_paid : total_paid = 55)
  (h_equation : (num_children + num_parents + num_grandmother) * cost_park_entry + (num_parents + num_grandmother) * cost_attraction_parent + num_children * x = total_paid) :
  x = 2 := by
  sorry

end attraction_ticket_cost_l231_231955


namespace cone_to_prism_volume_ratio_l231_231101

noncomputable def ratio_of_volumes (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) : ℝ :=
  let r := a / 2
  let V_cone := (1/3) * Real.pi * r^2 * h
  let V_prism := a * (2 * a) * h
  V_cone / V_prism

theorem cone_to_prism_volume_ratio (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) :
  ratio_of_volumes a h pos_a pos_h = Real.pi / 24 := by
  sorry

end cone_to_prism_volume_ratio_l231_231101


namespace son_l231_231355

theorem son's_age (S M : ℕ) (h₁ : M = S + 25) (h₂ : M + 2 = 2 * (S + 2)) : S = 23 := by
  sorry

end son_l231_231355


namespace simple_interest_principal_l231_231737

theorem simple_interest_principal
  (P_CI : ℝ)
  (r_CI t_CI : ℝ)
  (CI : ℝ)
  (P_SI : ℝ)
  (r_SI t_SI SI : ℝ)
  (h_compound_interest : (CI = P_CI * (1 + r_CI / 100)^t_CI - P_CI))
  (h_simple_interest : SI = (1 / 2) * CI)
  (h_SI_formula : SI = P_SI * r_SI * t_SI / 100) :
  P_SI = 1750 :=
by
  have P_CI := 4000
  have r_CI := 10
  have t_CI := 2
  have r_SI := 8
  have t_SI := 3
  have CI := 840
  have SI := 420
  sorry

end simple_interest_principal_l231_231737


namespace pages_left_to_read_l231_231065

def total_pages : ℕ := 17
def pages_read : ℕ := 11

theorem pages_left_to_read : total_pages - pages_read = 6 := by
  sorry

end pages_left_to_read_l231_231065


namespace sqrt_inequality_l231_231809

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a ^ 2 / b) + Real.sqrt (b ^ 2 / a) ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end sqrt_inequality_l231_231809


namespace circle_area_l231_231834

open Real

theorem circle_area (x y : ℝ) :
  (∃ r, (x + 2)^2 + (y - 3 / 2)^2 = r^2) →
  r = 7 / 2 →
  ∃ A, A = (π * (r)^2) ∧ A = (49/4) * π :=
by
  sorry

end circle_area_l231_231834


namespace intervals_of_monotonicity_interval_max_min_l231_231781

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x - 2

theorem intervals_of_monotonicity :
  (∀ (x : ℝ), x < -1 → deriv f x < 0) ∧ 
  (∀ (x : ℝ), -1 < x ∧ x < 3 → deriv f x > 0) ∧ 
  (∀ (x : ℝ), x > 3 → deriv f x < 0) := 
sorry

theorem interval_max_min :
  f 2 = 20 → f (-1) = -7 := 
sorry

end intervals_of_monotonicity_interval_max_min_l231_231781


namespace question_d_not_true_l231_231576

variable {a b c d : ℚ}

theorem question_d_not_true (h : a * b = c * d) : (a + 1) / (c + 1) ≠ (d + 1) / (b + 1) := 
sorry

end question_d_not_true_l231_231576


namespace cricket_bat_selling_price_l231_231262

theorem cricket_bat_selling_price
    (profit : ℝ)
    (profit_percentage : ℝ)
    (CP : ℝ)
    (SP : ℝ)
    (h_profit : profit = 255)
    (h_profit_percentage : profit_percentage = 42.857142857142854)
    (h_CP : CP = 255 * 100 / 42.857142857142854)
    (h_SP : SP = CP + profit) :
    SP = 850 :=
by
  skip -- This is where the proof would go
  sorry -- Placeholder for the required proof

end cricket_bat_selling_price_l231_231262


namespace regression_line_zero_corr_l231_231129

-- Definitions based on conditions
variables {X Y : Type}
variables [LinearOrder X] [LinearOrder Y]
variables {f : X → Y}  -- representing the regression line

-- Condition: Regression coefficient b = 0
def regression_coefficient_zero (b : ℝ) : Prop := b = 0

-- Definition of correlation coefficient; here symbolically represented since full derivation requires in-depth statistics definitions
def correlation_coefficient (r : ℝ) : ℝ := r

-- The mathematical goal to prove
theorem regression_line_zero_corr {b r : ℝ} 
  (hb : regression_coefficient_zero b) : correlation_coefficient r = 0 := 
by
  sorry

end regression_line_zero_corr_l231_231129


namespace maximum_profit_at_110_l231_231048

noncomputable def profit (x : ℕ) : ℝ := 
if x > 0 ∧ x < 100 then 
  -0.5 * (x : ℝ)^2 + 90 * (x : ℝ) - 600 
else if x ≥ 100 then 
  -2 * (x : ℝ) - 24200 / (x : ℝ) + 4100 
else 
  0 -- To ensure totality, although this won't match the problem's condition that x is always positive

theorem maximum_profit_at_110 :
  ∃ (y_max : ℝ), ∀ (x : ℕ), profit 110 = y_max ∧ (∀ x ≠ 0, profit 110 ≥ profit x) :=
sorry

end maximum_profit_at_110_l231_231048


namespace basketball_points_l231_231415

/-
In a basketball league, each game must have a winner and a loser. 
A team earns 2 points for a win and 1 point for a loss. 
A certain team expects to earn at least 48 points in all 32 games of 
the 2012-2013 season in order to have a chance to enter the playoffs. 
If this team wins x games in the upcoming matches, prove that
the relationship that x should satisfy to reach the goal is:
    2x + (32 - x) ≥ 48.
-/
theorem basketball_points (x : ℕ) (h : 0 ≤ x ∧ x ≤ 32) :
    2 * x + (32 - x) ≥ 48 :=
sorry

end basketball_points_l231_231415


namespace students_taking_chem_or_phys_not_both_l231_231426

def students_taking_both : ℕ := 12
def students_taking_chemistry : ℕ := 30
def students_taking_only_physics : ℕ := 18

theorem students_taking_chem_or_phys_not_both : 
  (students_taking_chemistry - students_taking_both) + students_taking_only_physics = 36 := 
by
  sorry

end students_taking_chem_or_phys_not_both_l231_231426


namespace true_proposition_is_D_l231_231179

open Real

theorem true_proposition_is_D :
  (∃ x_0 : ℝ, exp x_0 ≤ 0) = False ∧
  (∀ x : ℝ, 2 ^ x > x ^ 2) = False ∧
  (∀ a b : ℝ, a + b = 0 ↔ a / b = -1) = False ∧
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) = True :=
by
    sorry

end true_proposition_is_D_l231_231179


namespace initial_avg_mark_l231_231461

variable (A : ℝ) -- The initial average mark

-- Conditions
def num_students : ℕ := 33
def avg_excluded_students : ℝ := 40
def num_excluded_students : ℕ := 3
def avg_remaining_students : ℝ := 95

-- Equation derived from the problem conditions
def initial_avg :=
  A * num_students - avg_excluded_students * num_excluded_students = avg_remaining_students * (num_students - num_excluded_students)

theorem initial_avg_mark :
  initial_avg A →
  A = 90 :=
by
  intro h
  sorry

end initial_avg_mark_l231_231461


namespace carlos_marbles_l231_231823

theorem carlos_marbles :
  ∃ N : ℕ, 
    (N % 9 = 2) ∧ 
    (N % 10 = 2) ∧ 
    (N % 11 = 2) ∧ 
    (N > 1) ∧ 
    N = 992 :=
by {
  -- We need this for the example; you would remove it in a real proof.
  sorry
}

end carlos_marbles_l231_231823


namespace sum_of_fifth_powers_l231_231772

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l231_231772


namespace weight_difference_l231_231131

open Real

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h1 : (W_A + W_B + W_C) / 3 = 50)
  (h2 : W_A = 73)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 53)
  (h4 : (W_B + W_C + W_D + W_E) / 4 = 51) :
  W_E - W_D = 3 := 
sorry

end weight_difference_l231_231131


namespace alpha_beta_diff_l231_231731

theorem alpha_beta_diff 
  (α β : ℝ)
  (h1 : α + β = 17)
  (h2 : α * β = 70) : |α - β| = 3 :=
by
  sorry

end alpha_beta_diff_l231_231731


namespace triangle_tan_A_and_area_l231_231287

theorem triangle_tan_A_and_area {A B C a b c : ℝ} (hB : B = Real.pi / 3)
  (h1 : (Real.cos A - 3 * Real.cos C) * b = (3 * c - a) * Real.cos B)
  (hb : b = Real.sqrt 14) : 
  ∃ tan_A : ℝ, tan_A = Real.sqrt 3 / 5 ∧  -- First part: the value of tan A
  ∃ S : ℝ, S = (3 * Real.sqrt 3) / 2 :=  -- Second part: the area of triangle ABC
by
  sorry

end triangle_tan_A_and_area_l231_231287


namespace n_calculation_l231_231414

theorem n_calculation (n : ℕ) (hn : 0 < n)
  (h1 : Int.lcm 24 n = 72)
  (h2 : Int.lcm n 27 = 108) :
  n = 36 :=
sorry

end n_calculation_l231_231414


namespace no_positive_integer_solutions_l231_231777

def f (x : ℤ) : ℤ := x^2 + x

theorem no_positive_integer_solutions 
    (a b : ℤ) (ha : 0 < a) (hb : 0 < b) : 4 * f a ≠ f b := by
  sorry

end no_positive_integer_solutions_l231_231777


namespace probability_non_expired_bags_l231_231895

theorem probability_non_expired_bags :
  let total_bags := 5
  let expired_bags := 2
  let selected_bags := 2
  let total_combinations := Nat.choose total_bags selected_bags
  let non_expired_bags := total_bags - expired_bags
  let favorable_outcomes := Nat.choose non_expired_bags selected_bags
  (favorable_outcomes : ℚ) / (total_combinations : ℚ) = 3 / 10 := by
  sorry

end probability_non_expired_bags_l231_231895
