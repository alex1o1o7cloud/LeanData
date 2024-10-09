import Mathlib

namespace muffins_baked_by_James_correct_l1320_132077

noncomputable def muffins_baked_by_James (muffins_baked_by_Arthur : ℝ) (ratio : ℝ) : ℝ :=
  muffins_baked_by_Arthur / ratio

theorem muffins_baked_by_James_correct :
  muffins_baked_by_James 115.0 12.0 = 9.5833 :=
by
  -- Add the proof here
  sorry

end muffins_baked_by_James_correct_l1320_132077


namespace union_of_sets_l1320_132098

def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5, 7}
def union_result : Set ℕ := {1, 3, 5, 7}

theorem union_of_sets : A ∪ B = union_result := by
  sorry

end union_of_sets_l1320_132098


namespace total_children_count_l1320_132074

theorem total_children_count (boys girls : ℕ) (hb : boys = 40) (hg : girls = 77) : boys + girls = 117 := by
  sorry

end total_children_count_l1320_132074


namespace find_x2_plus_y2_l1320_132075

theorem find_x2_plus_y2 : ∀ (x y : ℝ),
  3 * x + 4 * y = 30 →
  x + 2 * y = 13 →
  x^2 + y^2 = 36.25 :=
by
  intros x y h1 h2
  sorry

end find_x2_plus_y2_l1320_132075


namespace tan_alpha_values_l1320_132073

theorem tan_alpha_values (α : ℝ) (h : Real.sin α + Real.cos α = 7 / 5) : 
  (Real.tan α = 4 / 3) ∨ (Real.tan α = 3 / 4) := 
  sorry

end tan_alpha_values_l1320_132073


namespace shortest_chord_through_M_l1320_132066

noncomputable def point_M : ℝ × ℝ := (1, 0)
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y = 0

theorem shortest_chord_through_M :
  (∀ x y : ℝ, circle_C x y → x + y - 1 = 0) :=
by
  sorry

end shortest_chord_through_M_l1320_132066


namespace coffee_shop_lattes_l1320_132093

theorem coffee_shop_lattes (x : ℕ) (number_of_teas number_of_lattes : ℕ)
  (h1 : number_of_teas = 6)
  (h2 : number_of_lattes = 32)
  (h3 : number_of_lattes = x * number_of_teas + 8) :
  x = 4 :=
by
  sorry

end coffee_shop_lattes_l1320_132093


namespace quadratic_function_vertex_upwards_exists_l1320_132027

theorem quadratic_function_vertex_upwards_exists :
  ∃ (a : ℝ), a > 0 ∧ ∃ (f : ℝ → ℝ), (∀ x, f x = a * (x - 1) * (x - 1) - 2) :=
by
  sorry

end quadratic_function_vertex_upwards_exists_l1320_132027


namespace range_x_inequality_l1320_132061

theorem range_x_inequality (a b x : ℝ) (ha : a ≠ 0) :
  (x ≥ 1/2) ∧ (x ≤ 5/2) →
  |a + b| + |a - b| ≥ |a| * (|x - 1| + |x - 2|) :=
by
  sorry

end range_x_inequality_l1320_132061


namespace inverse_variation_y_at_x_l1320_132084

variable (k x y : ℝ)

theorem inverse_variation_y_at_x :
  (∀ x y k, y = k / x → y = 6 → x = 3 → k = 18) → 
  k = 18 →
  x = 12 →
  y = 18 / 12 →
  y = 3 / 2 := by
  intros h1 h2 h3 h4
  sorry

end inverse_variation_y_at_x_l1320_132084


namespace initial_amount_of_money_l1320_132039

-- Definitions based on conditions in a)
variables (n : ℚ) -- Bert left the house with n dollars
def after_hardware_store := (3 / 4) * n
def after_dry_cleaners := after_hardware_store - 9
def after_grocery_store := (1 / 2) * after_dry_cleaners
def after_bookstall := (2 / 3) * after_grocery_store
def after_donation := (4 / 5) * after_bookstall

-- Theorem statement
theorem initial_amount_of_money : after_donation = 27 → n = 72 :=
by
  sorry

end initial_amount_of_money_l1320_132039


namespace find_base_l1320_132024

noncomputable def f (a x : ℝ) := 1 + (Real.log x) / (Real.log a)

theorem find_base (a : ℝ) (hinv_pass : (∀ y : ℝ, (∀ x : ℝ, f a x = y → x = 4 → y = 3))) : a = 2 :=
by
  sorry

end find_base_l1320_132024


namespace num_white_squares_in_24th_row_l1320_132092

-- Define the function that calculates the total number of squares in the nth row
def total_squares (n : ℕ) : ℕ := 1 + 2 * (n - 1)

-- Define the function that calculates the number of white squares in the nth row
def white_squares (n : ℕ) : ℕ := (total_squares n - 2) / 2

-- Problem statement for the Lean 4 theorem
theorem num_white_squares_in_24th_row : white_squares 24 = 23 :=
by {
  -- Lean proof generation will be placed here
  sorry
}

end num_white_squares_in_24th_row_l1320_132092


namespace seq_an_general_term_and_sum_l1320_132020

theorem seq_an_general_term_and_sum
  (a_n : ℕ → ℕ)
  (S : ℕ → ℕ)
  (T : ℕ → ℕ)
  (H1 : ∀ n, S n = 2 * a_n n - a_n 1)
  (H2 : ∃ d : ℕ, a_n 1 = d ∧ a_n 2 + 1 = a_n 1 + d ∧ a_n 3 = a_n 2 + d) :
  (∀ n, a_n n = 2^n) ∧ (∀ n, T n = n * 2^(n + 1) + 2 - 2^(n + 1)) := 
  by
  sorry

end seq_an_general_term_and_sum_l1320_132020


namespace inverse_function_problem_l1320_132051

theorem inverse_function_problem
  (f : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (h₁ : ∀ x, f (f_inv x) = x)
  (h₂ : ∀ x, f_inv (f x) = x)
  (a b : ℝ)
  (h₃ : f_inv (a - 1) + f_inv (b - 1) = 1) :
  f (a * b) = 3 :=
by
  sorry

end inverse_function_problem_l1320_132051


namespace snake_alligator_consumption_l1320_132062

theorem snake_alligator_consumption :
  (616 / 7) = 88 :=
by
  sorry

end snake_alligator_consumption_l1320_132062


namespace inequality_I_l1320_132018

theorem inequality_I (a b x y : ℝ) (hx : x < a) (hy : y < b) : x * y < a * b :=
sorry

end inequality_I_l1320_132018


namespace final_probability_l1320_132099

def total_cards := 52
def kings := 4
def aces := 4
def chosen_cards := 3

namespace probability

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def prob_three_kings : ℚ :=
  (4 / 52) * (3 / 51) * (2 / 50)

def prob_exactly_two_aces : ℚ :=
  (choose 4 2 * choose 48 1) / choose 52 3

def prob_exactly_three_aces : ℚ :=
  (choose 4 3) / choose 52 3

def prob_at_least_two_aces : ℚ :=
  prob_exactly_two_aces + prob_exactly_three_aces

def prob_three_kings_or_two_aces : ℚ :=
  prob_three_kings + prob_at_least_two_aces

theorem final_probability :
  prob_three_kings_or_two_aces = 6 / 425 :=
by
  sorry

end probability

end final_probability_l1320_132099


namespace exists_periodic_sequence_of_period_ge_two_l1320_132070

noncomputable def periodic_sequence (x : ℕ → ℝ) (p : ℕ) : Prop :=
  ∀ n, x (n + p) = x n

theorem exists_periodic_sequence_of_period_ge_two :
  ∀ (p : ℕ), p ≥ 2 →
  ∃ (x : ℕ → ℝ), periodic_sequence x p ∧ 
  ∀ n, x (n + 1) = x n - (1 / x n) :=
by {
  sorry
}

end exists_periodic_sequence_of_period_ge_two_l1320_132070


namespace collective_apples_l1320_132015

theorem collective_apples :
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8 := by
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  show (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8
  sorry

end collective_apples_l1320_132015


namespace fg_of_3_eq_29_l1320_132037

def f (x : ℝ) : ℝ := 2 * x - 3
def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem fg_of_3_eq_29 : f (g 3) = 29 := by
  sorry

end fg_of_3_eq_29_l1320_132037


namespace find_p_l1320_132059

noncomputable def f (p : ℝ) : ℝ := 2 * p - 20

theorem find_p : (f ∘ f ∘ f) p = 6 → p = 18.25 := by
  sorry

end find_p_l1320_132059


namespace interior_box_surface_area_l1320_132013

-- Given conditions
def original_length : ℕ := 40
def original_width : ℕ := 60
def corner_side : ℕ := 8

-- Calculate the initial area
def area_original : ℕ := original_length * original_width

-- Calculate the area of one corner
def area_corner : ℕ := corner_side * corner_side

-- Calculate the total area removed by four corners
def total_area_removed : ℕ := 4 * area_corner

-- Theorem to state the final area remaining
theorem interior_box_surface_area : 
  area_original - total_area_removed = 2144 :=
by
  -- Place the proof here
  sorry

end interior_box_surface_area_l1320_132013


namespace solve_inequality_l1320_132089

theorem solve_inequality {x : ℝ} : (x^2 - 9 * x + 18 ≤ 0) ↔ 3 ≤ x ∧ x ≤ 6 :=
by
sorry

end solve_inequality_l1320_132089


namespace sqrt_condition_l1320_132055

theorem sqrt_condition (x : ℝ) : (x - 3 ≥ 0) ↔ (x = 3) :=
by sorry

end sqrt_condition_l1320_132055


namespace nancy_clay_pots_l1320_132045

theorem nancy_clay_pots : 
  ∃ M : ℕ, (M + 2 * M + 14 = 50) ∧ M = 12 :=
sorry

end nancy_clay_pots_l1320_132045


namespace y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l1320_132022

def y : ℕ := 36 + 48 + 72 + 144 + 216 + 432 + 1296

theorem y_is_multiple_of_12 : y % 12 = 0 := by
  sorry

theorem y_is_multiple_of_3 : y % 3 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_4 : y % 4 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_6 : y % 6 = 0 := by
  have h := y_is_multiple_of_12
  sorry

end y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l1320_132022


namespace max_a_inequality_l1320_132046

theorem max_a_inequality (a : ℝ) :
  (∀ x : ℝ, x * a ≤ Real.exp (x - 1) + x^2 + 1) → a ≤ 3 := 
sorry

end max_a_inequality_l1320_132046


namespace most_reasonable_sample_l1320_132060

-- Define what it means to be a reasonable sample
def is_reasonable_sample (sample : String) : Prop :=
  sample = "D"

-- Define the conditions for each sample
def sample_A := "A"
def sample_B := "B"
def sample_C := "C"
def sample_D := "D"

-- Define the problem statement
theorem most_reasonable_sample :
  is_reasonable_sample sample_D :=
sorry

end most_reasonable_sample_l1320_132060


namespace hypotenuse_of_isosceles_right_triangle_l1320_132068

theorem hypotenuse_of_isosceles_right_triangle (a : ℝ) (hyp : a = 8) : 
  ∃ c : ℝ, c = a * Real.sqrt 2 :=
by
  use 8 * Real.sqrt 2
  sorry

end hypotenuse_of_isosceles_right_triangle_l1320_132068


namespace Chloe_wins_l1320_132083

theorem Chloe_wins (C M : ℕ) (h_ratio : 8 * M = 3 * C) (h_Max : M = 9) : C = 24 :=
by {
    sorry
}

end Chloe_wins_l1320_132083


namespace absolute_value_property_l1320_132028

theorem absolute_value_property (a b c : ℤ) (h : |a - b| + |c - a| = 1) : |a - c| + |c - b| + |b - a| = 2 :=
sorry

end absolute_value_property_l1320_132028


namespace runners_speeds_and_track_length_l1320_132019

/-- Given two runners α and β on a circular track starting at point P and running with uniform speeds,
when α reaches the halfway point Q, β is 16 meters behind α. At a later time, their positions are 
symmetric with respect to the diameter PQ. In 1 2/15 seconds, β reaches point Q, and 13 13/15 seconds later, 
α finishes the race. This theorem calculates the speeds of the runners and the distance of the lap. -/
theorem runners_speeds_and_track_length (x y : ℕ)
    (distance : ℝ)
    (runner_speed_alpha runner_speed_beta : ℝ) 
    (half_track_time_alpha half_track_time_beta : ℝ)
    (mirror_time_alpha mirror_time_beta : ℝ)
    (additional_time_beta : ℝ) :
    half_track_time_alpha = 16 ∧ 
    half_track_time_beta = (272/15) ∧ 
    mirror_time_alpha = (17/15) * (272/15 - 16/32) ∧ 
    mirror_time_beta = (17/15) ∧ 
    additional_time_beta = (13 + (13/15))  ∧ 
    runner_speed_beta = (15/2) ∧ 
    runner_speed_alpha = (85/10) ∧ 
    distance = 272 :=
  sorry

end runners_speeds_and_track_length_l1320_132019


namespace bananas_to_oranges_l1320_132007

theorem bananas_to_oranges :
  (3 / 4 : ℝ) * 16 = 12 →
  (2 / 3 : ℝ) * 9 = 6 :=
by
  intro h
  sorry

end bananas_to_oranges_l1320_132007


namespace heather_counts_209_l1320_132008

def alice_numbers (n : ℕ) : ℕ := 5 * n - 2
def general_skip_numbers (m : ℕ) : ℕ := 3 * m - 1
def heather_number := 209

theorem heather_counts_209 :
  (∀ n, alice_numbers n > 0 ∧ alice_numbers n ≤ 500 → ¬heather_number = alice_numbers n) ∧
  (∀ m, general_skip_numbers m > 0 ∧ general_skip_numbers m ≤ 500 → ¬heather_number = general_skip_numbers m) ∧
  (1 ≤ heather_number ∧ heather_number ≤ 500) :=
by
  sorry

end heather_counts_209_l1320_132008


namespace value_of_f_2018_l1320_132021

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 3) * f x = -1
axiom initial_condition : f (-1) = 2

theorem value_of_f_2018 : f 2018 = -1 / 2 :=
by
  sorry

end value_of_f_2018_l1320_132021


namespace faster_train_pass_time_l1320_132072

-- Defining the conditions
def length_of_train : ℕ := 45 -- length in meters
def speed_of_faster_train : ℕ := 45 -- speed in km/hr
def speed_of_slower_train : ℕ := 36 -- speed in km/hr

-- Define relative speed
def relative_speed := (speed_of_faster_train - speed_of_slower_train) * 5 / 18 -- converting km/hr to m/s

-- Total distance to pass (sum of lengths of both trains)
def total_passing_distance := (2 * length_of_train) -- 2 trains of 45 meters each

-- Calculate the time to pass the slower train
def time_to_pass := total_passing_distance / relative_speed

-- The theorem to prove
theorem faster_train_pass_time : time_to_pass = 36 := by
  -- This is where the proof would be placed
  sorry

end faster_train_pass_time_l1320_132072


namespace mean_of_second_set_l1320_132042

theorem mean_of_second_set (x : ℝ)
  (H1 : (28 + x + 70 + 88 + 104) / 5 = 67) :
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 :=
sorry

end mean_of_second_set_l1320_132042


namespace solve_for_x_l1320_132016

/-- Let f(x) = 2 - 1 / (2 - x)^3.
Proof that f(x) = 1 / (2 - x)^3 implies x = 1. -/
theorem solve_for_x (x : ℝ) (h : 2 - 1 / (2 - x)^3 = 1 / (2 - x)^3) : x = 1 :=
  sorry

end solve_for_x_l1320_132016


namespace minimum_value_of_y_l1320_132033

noncomputable def y (x : ℝ) : ℝ :=
  x^2 + 12 * x + 108 / x^4

theorem minimum_value_of_y : ∃ x > 0, y x = 49 :=
by
  sorry

end minimum_value_of_y_l1320_132033


namespace nora_third_tree_oranges_l1320_132044

theorem nora_third_tree_oranges (a b c total : ℕ)
  (h_a : a = 80)
  (h_b : b = 60)
  (h_total : total = 260)
  (h_sum : total = a + b + c) :
  c = 120 :=
by
  -- The proof should go here
  sorry

end nora_third_tree_oranges_l1320_132044


namespace yardwork_payment_l1320_132025

theorem yardwork_payment :
  let earnings := [15, 20, 25, 40]
  let total_earnings := List.sum earnings
  let equal_share := total_earnings / earnings.length
  let high_earner := 40
  high_earner - equal_share = 15 :=
by
  sorry

end yardwork_payment_l1320_132025


namespace range_of_a_l1320_132023

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l1320_132023


namespace ratio_of_areas_l1320_132080

theorem ratio_of_areas (side_length : ℝ) (h : side_length = 6) :
  let area_triangle := (side_length^2 * Real.sqrt 3) / 4
  let area_square := side_length^2
  (area_triangle / area_square) = Real.sqrt 3 / 4 :=
by
  sorry

end ratio_of_areas_l1320_132080


namespace probability_at_least_one_boy_one_girl_l1320_132076

def boys := 12
def girls := 18
def total_members := 30
def committee_size := 6

def total_ways := Nat.choose total_members committee_size
def all_boys_ways := Nat.choose boys committee_size
def all_girls_ways := Nat.choose girls committee_size
def all_boys_or_girls_ways := all_boys_ways + all_girls_ways
def complementary_probability := all_boys_or_girls_ways / total_ways
def desired_probability := 1 - complementary_probability

theorem probability_at_least_one_boy_one_girl :
  desired_probability = (574287 : ℚ) / 593775 :=
  sorry

end probability_at_least_one_boy_one_girl_l1320_132076


namespace four_digit_numbers_l1320_132065

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l1320_132065


namespace quadratic_function_two_distinct_roots_l1320_132091

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks the conditions for the quadratic to have two distinct real roots
theorem quadratic_function_two_distinct_roots (a : ℝ) : 
  (0 < a ∧ a < 2) → (discriminant a (-4) 2 > 0) :=
by
  sorry

end quadratic_function_two_distinct_roots_l1320_132091


namespace expected_value_is_correct_l1320_132001

def probability_of_rolling_one : ℚ := 1 / 4

def probability_of_other_numbers : ℚ := 3 / 4 / 5

def win_amount : ℚ := 8

def loss_amount : ℚ := -3

def expected_value : ℚ := (probability_of_rolling_one * win_amount) + 
                          (probability_of_other_numbers * 5 * loss_amount)

theorem expected_value_is_correct : expected_value = -0.25 :=
by 
  unfold expected_value probability_of_rolling_one probability_of_other_numbers win_amount loss_amount
  sorry

end expected_value_is_correct_l1320_132001


namespace correct_statement_l1320_132005

-- Conditions as definitions
def deductive_reasoning (p q r : Prop) : Prop :=
  (p → q) → (q → r) → (p → r)

def correctness_of_conclusion := true  -- Indicates statement is defined to be correct

def pattern_of_reasoning (p q r : Prop) : Prop :=
  deductive_reasoning p q r

-- Statement to prove
theorem correct_statement (p q r : Prop) :
  pattern_of_reasoning p q r = deductive_reasoning p q r :=
by sorry

end correct_statement_l1320_132005


namespace group_1991_l1320_132052

theorem group_1991 (n : ℕ) (h1 : 1 ≤ n) (h2 : 1991 = 2 * n ^ 2 - 1) : n = 32 := 
sorry

end group_1991_l1320_132052


namespace initial_bottle_caps_l1320_132082

theorem initial_bottle_caps (bought_caps total_caps initial_caps : ℕ) 
  (hb : bought_caps = 41) (ht : total_caps = 43):
  initial_caps = 2 :=
by
  have h : total_caps = initial_caps + bought_caps := sorry
  have ha : initial_caps = total_caps - bought_caps := sorry
  exact sorry

end initial_bottle_caps_l1320_132082


namespace wall_width_l1320_132067

theorem wall_width
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_length : ℝ) (wall_height : ℝ)
  (num_bricks : ℕ)
  (brick_volume : ℝ := brick_length * brick_width * brick_height)
  (total_volume : ℝ := num_bricks * brick_volume) :
  brick_length = 0.20 → brick_width = 0.10 → brick_height = 0.08 →
  wall_length = 10 → wall_height = 8 → num_bricks = 12250 →
  total_volume = wall_length * wall_height * (0.245 : ℝ) :=
by 
  sorry

end wall_width_l1320_132067


namespace average_of_six_starting_from_d_plus_one_l1320_132057

theorem average_of_six_starting_from_d_plus_one (c d : ℝ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (c + 6) = ((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 6 := 
by 
-- Proof omitted; end with sorry
sorry

end average_of_six_starting_from_d_plus_one_l1320_132057


namespace percent_absent_is_correct_l1320_132064

theorem percent_absent_is_correct (total_students boys girls absent_boys absent_girls : ℝ) 
(h1 : total_students = 100)
(h2 : boys = 50)
(h3 : girls = 50)
(h4 : absent_boys = boys * (1 / 5))
(h5 : absent_girls = girls * (1 / 4)):
  (absent_boys + absent_girls) / total_students * 100 = 22.5 :=
by 
  sorry

end percent_absent_is_correct_l1320_132064


namespace pure_alcohol_addition_l1320_132000

variables (P : ℝ) (V : ℝ := 14.285714285714286 ) (initial_volume : ℝ := 100) (final_percent_alcohol : ℝ := 0.30)

theorem pure_alcohol_addition :
  P / 100 * initial_volume + V = final_percent_alcohol * (initial_volume + V) :=
by
  sorry

end pure_alcohol_addition_l1320_132000


namespace valid_permutations_count_l1320_132086

def num_permutations (seq : List ℕ) : ℕ :=
  -- A dummy implementation, the real function would calculate the number of valid permutations.
  sorry

theorem valid_permutations_count : num_permutations [1, 2, 3, 4, 5, 6] = 32 :=
by
  sorry

end valid_permutations_count_l1320_132086


namespace value_after_addition_l1320_132078

theorem value_after_addition (x : ℕ) (h : x / 9 = 8) : x + 11 = 83 :=
by
  sorry

end value_after_addition_l1320_132078


namespace loss_percentage_is_26_l1320_132026

/--
Given the cost price of a radio is Rs. 1500 and the selling price is Rs. 1110, 
prove that the loss percentage is 26%
-/
theorem loss_percentage_is_26 (cost_price selling_price : ℝ)
  (h₀ : cost_price = 1500)
  (h₁ : selling_price = 1110) :
  ((cost_price - selling_price) / cost_price) * 100 = 26 := 
by 
  sorry

end loss_percentage_is_26_l1320_132026


namespace min_value_fraction_sum_l1320_132096

open Real

theorem min_value_fraction_sum (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 2) :
    ∃ m, m = (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ∧ m = 9/4 :=
by
  sorry

end min_value_fraction_sum_l1320_132096


namespace irrational_of_sqrt_3_l1320_132031

theorem irrational_of_sqrt_3 :
  ¬ (∃ (a b : ℤ), b ≠ 0 ∧ ↑a / ↑b = Real.sqrt 3) :=
sorry

end irrational_of_sqrt_3_l1320_132031


namespace machine_made_8_shirts_today_l1320_132095

-- Define the conditions
def shirts_per_minute : ℕ := 2
def minutes_worked_today : ℕ := 4

-- Define the expected number of shirts made today
def shirts_made_today : ℕ := shirts_per_minute * minutes_worked_today

-- The theorem stating that the shirts made today should be 8
theorem machine_made_8_shirts_today : shirts_made_today = 8 := by
  sorry

end machine_made_8_shirts_today_l1320_132095


namespace insects_legs_l1320_132056

theorem insects_legs (n : ℕ) (l : ℕ) (h₁ : n = 6) (h₂ : l = 6) : n * l = 36 :=
by sorry

end insects_legs_l1320_132056


namespace average_next_seven_consecutive_is_correct_l1320_132090

-- Define the sum of seven consecutive integers starting at x.
def sum_seven_consecutive_integers (x : ℕ) : ℕ := 7 * x + 21

-- Define the next sequence of seven integers starting from y + 1.
def average_next_seven_consecutive_integers (x : ℕ) : ℕ :=
  let y := sum_seven_consecutive_integers x
  let start := y + 1
  (start + (start + 1) + (start + 2) + (start + 3) + (start + 4) + (start + 5) + (start + 6)) / 7

-- Problem statement
theorem average_next_seven_consecutive_is_correct (x : ℕ) : 
  average_next_seven_consecutive_integers x = 7 * x + 25 :=
by
  sorry

end average_next_seven_consecutive_is_correct_l1320_132090


namespace solution_set_quadratic_ineq_all_real_l1320_132038

theorem solution_set_quadratic_ineq_all_real (a b c : ℝ) :
  (∀ x : ℝ, (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) :=
by
  sorry

end solution_set_quadratic_ineq_all_real_l1320_132038


namespace geometric_sequence_sum_l1320_132040

theorem geometric_sequence_sum (S : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n : ℕ, n > 0 → S n = 2^n + a) →
  (S 1 = 2 + a) →
  (∀ n ≥ 2, a_n n = S n - S (n - 1)) →
  (a_n 1 = 1) →
  a = -1 :=
by
  sorry

end geometric_sequence_sum_l1320_132040


namespace sum_of_series_eq_half_l1320_132085

theorem sum_of_series_eq_half :
  (∑' k : ℕ, 3^(2^k) / (9^(2^k) - 1)) = 1 / 2 :=
by
  sorry

end sum_of_series_eq_half_l1320_132085


namespace compute_roots_sum_l1320_132081

def roots_quadratic_eq_a_b (a b : ℂ) : Prop :=
  a^2 - 6 * a + 8 = 0 ∧ b^2 - 6 * b + 8 = 0

theorem compute_roots_sum (a b : ℂ) (ha : roots_quadratic_eq_a_b a b) :
  a^5 + a^3 * b^3 + b^5 = -568 := by
  sorry

end compute_roots_sum_l1320_132081


namespace min_value_of_f_l1320_132043

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_of_f : ∃ x : ℝ, f x = x^3 - 3 * x^2 + 1 ∧ (∀ y : ℝ, f y ≥ f 2) :=
by
  sorry

end min_value_of_f_l1320_132043


namespace taxi_fare_range_l1320_132094

theorem taxi_fare_range (x : ℝ) (h : 12.5 + 2.4 * (x - 3) = 19.7) : 5 < x ∧ x ≤ 6 :=
by
  -- Given conditions and the equation, we need to prove the inequalities.
  have fare_eq : 12.5 + 2.4 * (x - 3) = 19.7 := h
  sorry

end taxi_fare_range_l1320_132094


namespace isosceles_triangle_largest_angle_l1320_132058

theorem isosceles_triangle_largest_angle (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = 50) :
  A + B + C = 180 →
  C = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l1320_132058


namespace travel_time_difference_in_minutes_l1320_132097

/-
A bus travels at an average speed of 40 miles per hour.
We need to prove that the difference in travel time between a 360-mile trip and a 400-mile trip equals 60 minutes.
-/

theorem travel_time_difference_in_minutes 
  (speed : ℝ) (distance1 distance2 : ℝ) 
  (h1 : speed = 40) 
  (h2 : distance1 = 360) 
  (h3 : distance2 = 400) :
  (distance2 / speed - distance1 / speed) * 60 = 60 := by
  sorry

end travel_time_difference_in_minutes_l1320_132097


namespace radius_of_inscribed_circle_l1320_132034

variable (height : ℝ) (alpha : ℝ)

theorem radius_of_inscribed_circle (h : ℝ) (α : ℝ) : 
∃ r : ℝ, r = (h / 2) * (Real.tan (Real.pi / 4 - α / 4)) ^ 2 := 
sorry

end radius_of_inscribed_circle_l1320_132034


namespace jack_sugar_usage_l1320_132032

theorem jack_sugar_usage (initial_sugar bought_sugar final_sugar x : ℕ) 
  (h1 : initial_sugar = 65) 
  (h2 : bought_sugar = 50) 
  (h3 : final_sugar = 97) 
  (h4 : final_sugar = initial_sugar - x + bought_sugar) : 
  x = 18 := 
by 
  sorry

end jack_sugar_usage_l1320_132032


namespace box_dimension_triples_l1320_132050

theorem box_dimension_triples (N : ℕ) :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ (1 / a + 1 / b + 1 / c = 1 / 8) → ∃ k, k = N := sorry 

end box_dimension_triples_l1320_132050


namespace average_weight_of_boys_l1320_132047

theorem average_weight_of_boys (n1 n2 : ℕ) (w1 w2 : ℚ) 
  (weight_avg_22_boys : w1 = 50.25) 
  (weight_avg_8_boys : w2 = 45.15) 
  (count_22_boys : n1 = 22) 
  (count_8_boys : n2 = 8) 
  : ((n1 * w1 + n2 * w2) / (n1 + n2) : ℚ) = 48.89 :=
by
  sorry

end average_weight_of_boys_l1320_132047


namespace sum_of_cubes_identity_l1320_132011

theorem sum_of_cubes_identity (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end sum_of_cubes_identity_l1320_132011


namespace geometric_series_sum_eq_l1320_132048

theorem geometric_series_sum_eq :
  let a := (1/3 : ℚ)
  let r := (1/3 : ℚ)
  let n := 8
  let S := a * (1 - r^n) / (1 - r)
  S = 3280 / 6561 :=
by
  sorry

end geometric_series_sum_eq_l1320_132048


namespace remainder_when_3x_7y_5z_div_31517_l1320_132010

theorem remainder_when_3x_7y_5z_div_31517
  (x y z : ℕ)
  (hx : x % 23 = 9)
  (hy : y % 29 = 15)
  (hz : z % 37 = 12) :
  (3 * x + 7 * y - 5 * z) % 31517 = ((69 * (x / 23) + 203 * (y / 29) - 185 * (z / 37) + 72) % 31517) := 
sorry

end remainder_when_3x_7y_5z_div_31517_l1320_132010


namespace james_weight_gain_l1320_132009

def cheezits_calories (bags : ℕ) (oz_per_bag : ℕ) (cal_per_oz : ℕ) : ℕ :=
  bags * oz_per_bag * cal_per_oz

def chocolate_calories (bars : ℕ) (cal_per_bar : ℕ) : ℕ :=
  bars * cal_per_bar

def popcorn_calories (bags : ℕ) (cal_per_bag : ℕ) : ℕ :=
  bags * cal_per_bag

def run_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def swim_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def cycle_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def total_calories_consumed : ℕ :=
  cheezits_calories 3 2 150 + chocolate_calories 2 250 + popcorn_calories 1 500

def total_calories_burned : ℕ :=
  run_calories 40 12 + swim_calories 30 15 + cycle_calories 20 10

def excess_calories : ℕ :=
  total_calories_consumed - total_calories_burned

def weight_gain (excess_cal : ℕ) (cal_per_lb : ℕ) : ℚ :=
  excess_cal / cal_per_lb

theorem james_weight_gain :
  weight_gain excess_calories 3500 = 770 / 3500 :=
sorry

end james_weight_gain_l1320_132009


namespace range_of_a_l1320_132079

noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x - a

theorem range_of_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ (-9 < a ∧ a < 5/3) :=
by
  sorry

end range_of_a_l1320_132079


namespace boat_speed_in_still_water_l1320_132002

theorem boat_speed_in_still_water (D V_s t_down t_up : ℝ) (h_val : V_s = 3) (h_down : D = (15 + V_s) * t_down) (h_up : D = (15 - V_s) * t_up) : 15 = 15 :=
by
  have h1 : 15 = (D / 1 - V_s) := sorry
  have h2 : 15 = (D / 1.5 + V_s) := sorry
  sorry

end boat_speed_in_still_water_l1320_132002


namespace intersection_P_Q_l1320_132053

-- Define the sets P and Q
def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {x | -1 ≤ x ∧ x < 1}

-- The proof statement
theorem intersection_P_Q : P ∩ Q = {-1, 0} :=
by
  sorry

end intersection_P_Q_l1320_132053


namespace no_solutions_of_pairwise_distinct_l1320_132054

theorem no_solutions_of_pairwise_distinct 
  (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∀ x : ℝ, ¬(x^3 - a * x^2 + b^3 = 0 ∧ x^3 - b * x^2 + c^3 = 0 ∧ x^3 - c * x^2 + a^3 = 0) :=
by
  -- Proof to be completed
  sorry

end no_solutions_of_pairwise_distinct_l1320_132054


namespace calculate_area_l1320_132087

def leftmost_rectangle_area (height width : ℕ) : ℕ := height * width
def middle_rectangle_area (height width : ℕ) : ℕ := height * width
def rightmost_rectangle_area (height width : ℕ) : ℕ := height * width

theorem calculate_area : 
  let leftmost_segment_height := 7
  let bottom_width := 6
  let segment_above_3 := 3
  let segment_above_2 := 2
  let rightmost_width := 5
  leftmost_rectangle_area leftmost_segment_height bottom_width + 
  middle_rectangle_area segment_above_3 segment_above_3 + 
  rightmost_rectangle_area segment_above_2 rightmost_width = 
  61 := by
    sorry

end calculate_area_l1320_132087


namespace intersection_A_B_l1320_132036

def A := {x : ℝ | |x| < 1}
def B := {x : ℝ | -2 < x ∧ x < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end intersection_A_B_l1320_132036


namespace other_employee_number_l1320_132014

-- Define the conditions
variables (total_employees : ℕ) (sample_size : ℕ) (e1 e2 e3 : ℕ)

-- Define the systematic sampling interval
def sampling_interval (total : ℕ) (size : ℕ) : ℕ := total / size

-- The Lean statement for the proof problem
theorem other_employee_number
  (h1 : total_employees = 52)
  (h2 : sample_size = 4)
  (h3 : e1 = 6)
  (h4 : e2 = 32)
  (h5 : e3 = 45) :
  ∃ e4 : ℕ, e4 = 19 := 
sorry

end other_employee_number_l1320_132014


namespace green_dots_third_row_l1320_132035

noncomputable def row_difference (a b : Nat) : Nat := b - a

theorem green_dots_third_row (a1 a2 a4 a5 a3 d : Nat)
  (h_a1 : a1 = 3)
  (h_a2 : a2 = 6)
  (h_a4 : a4 = 12)
  (h_a5 : a5 = 15)
  (h_d : row_difference a2 a1 = d)
  (h_d_consistent : row_difference a2 a1 = row_difference a4 a3) :
  a3 = 9 :=
sorry

end green_dots_third_row_l1320_132035


namespace tangent_line_parallel_x_axis_l1320_132063

def f (x : ℝ) : ℝ := x^4 - 4 * x

theorem tangent_line_parallel_x_axis :
  ∃ (m n : ℝ), (n = f m) ∧ (deriv f m = 0) ∧ (m, n) = (1, -3) := by
  sorry

end tangent_line_parallel_x_axis_l1320_132063


namespace translate_function_right_by_2_l1320_132088

theorem translate_function_right_by_2 (x : ℝ) : 
  (∀ x, (x - 2) ^ 2 + (x - 2) = x ^ 2 - 3 * x + 2) := 
by 
  sorry

end translate_function_right_by_2_l1320_132088


namespace greatest_y_l1320_132030

theorem greatest_y (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : y ≤ 24 :=
sorry

end greatest_y_l1320_132030


namespace prove_k_in_terms_of_x_l1320_132012

variables {A B k x : ℝ}

-- given conditions
def positive_numbers (A B : ℝ) := A > 0 ∧ B > 0
def ratio_condition (A B k : ℝ) := A = k * B
def percentage_condition (A B x : ℝ) := A = B + (x / 100) * B

-- proof statement
theorem prove_k_in_terms_of_x (A B k x : ℝ) (h1 : positive_numbers A B) (h2 : ratio_condition A B k) (h3 : percentage_condition A B x) (h4 : k > 1) :
  k = 1 + x / 100 :=
sorry

end prove_k_in_terms_of_x_l1320_132012


namespace leila_total_cakes_l1320_132029

def cakes_monday : ℕ := 6
def cakes_friday : ℕ := 9
def cakes_saturday : ℕ := 3 * cakes_monday
def total_cakes : ℕ := cakes_monday + cakes_friday + cakes_saturday

theorem leila_total_cakes : total_cakes = 33 :=
by sorry

end leila_total_cakes_l1320_132029


namespace time_difference_l1320_132004

-- Definitions of speeds and distance
def distance : Nat := 12
def alice_speed : Nat := 7
def bob_speed : Nat := 9

-- Calculations of total times based on speeds and distance
def alice_time : Nat := alice_speed * distance
def bob_time : Nat := bob_speed * distance

-- Statement of the problem
theorem time_difference : bob_time - alice_time = 24 := by
  sorry

end time_difference_l1320_132004


namespace greatest_of_given_numbers_l1320_132003

-- Defining the given conditions
def a := 1000 + 0.01
def b := 1000 * 0.01
def c := 1000 / 0.01
def d := 0.01 / 1000
def e := 1000 - 0.01

-- Prove that c is the greatest
theorem greatest_of_given_numbers : c = max a (max b (max d e)) :=
by
  -- Placeholder for the proof
  sorry

end greatest_of_given_numbers_l1320_132003


namespace sin_75_l1320_132006

theorem sin_75 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end sin_75_l1320_132006


namespace obtuse_equilateral_triangle_impossible_l1320_132041

-- Define a scalene triangle 
def is_scalene_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ A + B + C = 180

-- Define acute triangles
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

-- Define right triangles
def is_right_triangle (A B C : ℝ) : Prop :=
  A = 90 ∨ B = 90 ∨ C = 90

-- Define isosceles triangles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

-- Define obtuse triangles
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

-- Define equilateral triangles
def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = a ∧ A = 60 ∧ B = 60 ∧ C = 60

theorem obtuse_equilateral_triangle_impossible :
  ¬ ∃ (a b c A B C : ℝ), is_equilateral_triangle a b c A B C ∧ is_obtuse_triangle A B C :=
by
  sorry

end obtuse_equilateral_triangle_impossible_l1320_132041


namespace polynomial_remainder_l1320_132071

theorem polynomial_remainder (P : ℝ → ℝ) (h1 : P 19 = 16) (h2 : P 15 = 8) : 
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 15) * (x - 19) * Q x + 2 * x - 22 :=
by
  sorry

end polynomial_remainder_l1320_132071


namespace no_three_in_range_l1320_132069

theorem no_three_in_range (c : ℝ) : c > 4 → ¬ (∃ x : ℝ, x^2 + 2 * x + c = 3) :=
by
  sorry

end no_three_in_range_l1320_132069


namespace number_of_articles_l1320_132049

-- Define the conditions
def gain := 1 / 9
def cp_one_article := 1  -- cost price of one article

-- Define the cost price for x articles
def cp (x : ℕ) := x * cp_one_article

-- Define the selling price for 45 articles
def sp (x : ℕ) := x / 45

-- Define the selling price equation considering gain
def sp_one_article := (cp_one_article * (1 + gain))

-- Main theorem to prove
theorem number_of_articles (x : ℕ) (h : sp x = sp_one_article) : x = 50 :=
by
  sorry

-- The theorem imports all necessary conditions and definitions and prepares the problem for proof.

end number_of_articles_l1320_132049


namespace opposite_of_negative_2023_l1320_132017

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l1320_132017
