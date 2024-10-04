import Mathlib

namespace calories_per_pound_of_body_fat_l297_297242

theorem calories_per_pound_of_body_fat (gained_weight : ℕ) (calories_burned_per_day : ℕ) 
  (days_to_lose_weight : ℕ) (calories_consumed_per_day : ℕ) : 
  gained_weight = 5 → 
  calories_burned_per_day = 2500 → 
  days_to_lose_weight = 35 → 
  calories_consumed_per_day = 2000 → 
  (calories_burned_per_day * days_to_lose_weight - calories_consumed_per_day * days_to_lose_weight) / gained_weight = 3500 :=
by 
  intros h1 h2 h3 h4
  sorry

end calories_per_pound_of_body_fat_l297_297242


namespace tommy_initial_balloons_l297_297915

theorem tommy_initial_balloons (initial_balloons balloons_added total_balloons : ℝ)
  (h1 : balloons_added = 34.5)
  (h2 : total_balloons = 60.75)
  (h3 : total_balloons = initial_balloons + balloons_added) :
  initial_balloons = 26.25 :=
by sorry

end tommy_initial_balloons_l297_297915


namespace range_of_lg_x_l297_297842

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_lg_x {f : ℝ → ℝ} (h_even : is_even f)
    (h_decreasing : is_decreasing_on_nonneg f)
    (h_condition : f (Real.log x) > f 1) :
    x ∈ Set.Ioo (1/10 : ℝ) (10 : ℝ) :=
  sorry

end range_of_lg_x_l297_297842


namespace right_triangle_hypotenuse_l297_297955

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l297_297955


namespace shirts_per_minute_l297_297605

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (shirts_per_min : ℕ) 
  (h : total_shirts = 12 ∧ total_minutes = 6) :
  shirts_per_min = 2 :=
sorry

end shirts_per_minute_l297_297605


namespace percent_asian_population_in_West_l297_297337

-- Define the populations in different regions
def population_NE := 2
def population_MW := 3
def population_South := 4
def population_West := 10

-- Define the total population
def total_population := population_NE + population_MW + population_South + population_West

-- Calculate the percentage of the population in the West
def percentage_in_West := (population_West * 100) / total_population

-- The proof statement
theorem percent_asian_population_in_West : percentage_in_West = 53 := by
  sorry -- proof to be completed

end percent_asian_population_in_West_l297_297337


namespace correct_statements_l297_297898

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem correct_statements : 
  (∀ x, f (-x) = -f (x)) ∧  -- Statement A
  (∀ x₁ x₂, x₁ + x₂ = Real.pi / 2 → g x₁ = g x₂)  -- Statement C
:= by
  sorry

end correct_statements_l297_297898


namespace four_digit_square_number_divisible_by_11_with_unit_1_l297_297153

theorem four_digit_square_number_divisible_by_11_with_unit_1 
  : ∃ y : ℕ, y >= 1000 ∧ y <= 9999 ∧ (∃ n : ℤ, y = n^2) ∧ y % 11 = 0 ∧ y % 10 = 1 ∧ y = 9801 := 
by {
  -- sorry statement to skip the proof.
  sorry 
}

end four_digit_square_number_divisible_by_11_with_unit_1_l297_297153


namespace sequence_properties_l297_297719

theorem sequence_properties
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 = 5)
  (h2 : ∀ n, n % 2 = 1 → a (n + 1) - a n = 1)
  (h3 : ∀ n, n % 2 = 0 → a (n + 1) - a n = 3) :
  (a 1 = 2) ∧ (a 2 = 3) ∧
  (∀ n, a (2 * n - 1) = 2 * (2 * n - 1)) ∧
  (∀ n, a (2 * n) = 2 * 2 * n - 1) :=
by
  sorry

end sequence_properties_l297_297719


namespace committee_count_8_choose_4_l297_297457

theorem committee_count_8_choose_4 : (Nat.choose 8 4) = 70 :=
  by
  -- proof skipped
  sorry

end committee_count_8_choose_4_l297_297457


namespace a5_is_3_l297_297837

section
variable {a : ℕ → ℝ} 
variable (h_pos : ∀ n, 0 < a n)
variable (h_a1 : a 1 = 1)
variable (h_a2 : a 2 = Real.sqrt 3)
variable (h_recursive : ∀ n ≥ 2, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2)

theorem a5_is_3 : a 5 = 3 :=
  by
  sorry
end

end a5_is_3_l297_297837


namespace cost_of_fencing_each_side_l297_297858

theorem cost_of_fencing_each_side (total_cost : ℕ) (x : ℕ) (h : total_cost = 276) (hx : 4 * x = total_cost) : x = 69 :=
by {
  sorry
}

end cost_of_fencing_each_side_l297_297858


namespace solve_for_z_l297_297265

theorem solve_for_z (z i : ℂ) (h1 : 1 - i*z + 3*i = -1 + i*z + 3*i) (h2 : i^2 = -1) : z = -i := 
  sorry

end solve_for_z_l297_297265


namespace find_integer_l297_297156

theorem find_integer (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 9) (h3 : -1234 ≡ n [MOD 9]) : n = 8 := 
sorry

end find_integer_l297_297156


namespace hypotenuse_right_triangle_l297_297986

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l297_297986


namespace baseball_cards_start_count_l297_297042

theorem baseball_cards_start_count (X : ℝ) 
  (h1 : ∃ (x : ℝ), x = (X + 1) / 2)
  (h2 : ∃ (x' : ℝ), x' = X - ((X + 1) / 2) - 1)
  (h3 : ∃ (y : ℝ), y = 3 * (X - ((X + 1) / 2) - 1))
  (h4 : ∃ (z : ℝ), z = 18) : 
  X = 15 :=
by
  sorry

end baseball_cards_start_count_l297_297042


namespace division_by_ab_plus_one_is_perfect_square_l297_297879

theorem division_by_ab_plus_one_is_perfect_square
    (a b : ℕ) (h : 0 < a ∧ 0 < b)
    (hab : (ab + 1) ∣ (a^2 + b^2)) :
    ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) := 
sorry

end division_by_ab_plus_one_is_perfect_square_l297_297879


namespace find_x_l297_297566

variable (x : ℝ)

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x : delta (phi x) = 23 → x = -1 / 6 := by
  intro h
  sorry

end find_x_l297_297566


namespace average_of_remaining_two_numbers_l297_297895

theorem average_of_remaining_two_numbers 
  (avg_6 : ℝ) (avg1_2 : ℝ) (avg2_2 : ℝ)
  (n1 n2 n3 : ℕ)
  (h_avg6 : n1 = 6 ∧ avg_6 = 4.60)
  (h_avg1_2 : n2 = 2 ∧ avg1_2 = 3.4)
  (h_avg2_2 : n3 = 2 ∧ avg2_2 = 3.8) :
  ∃ avg_rem2 : ℝ, avg_rem2 = 6.6 :=
by {
  sorry
}

end average_of_remaining_two_numbers_l297_297895


namespace ratio_pages_l297_297422

theorem ratio_pages (pages_Selena pages_Harry : ℕ) (h₁ : pages_Selena = 400) (h₂ : pages_Harry = 180) : 
  pages_Harry / pages_Selena = 9 / 20 := 
by
  -- proof goes here
  sorry

end ratio_pages_l297_297422


namespace tesseract_hyper_volume_l297_297366

theorem tesseract_hyper_volume
  (a b c d : ℝ)
  (h1 : a * b * c = 72)
  (h2 : b * c * d = 75)
  (h3 : c * d * a = 48)
  (h4 : d * a * b = 50) :
  a * b * c * d = 3600 :=
sorry

end tesseract_hyper_volume_l297_297366


namespace selling_price_ratio_l297_297942

theorem selling_price_ratio (CP SP1 SP2 : ℝ) (h1 : SP1 = CP + 0.5 * CP) (h2 : SP2 = CP + 3 * CP) :
  SP2 / SP1 = 8 / 3 :=
by
  sorry

end selling_price_ratio_l297_297942


namespace shaded_cubes_count_l297_297347

theorem shaded_cubes_count :
  let faces := 6
  let shaded_on_one_face := 5
  let corner_cubes := 8
  let center_cubes := 2 * 1 -- center cubes shared among opposite faces
  let total_shaded_cubes := corner_cubes + center_cubes
  faces = 6 → shaded_on_one_face = 5 → corner_cubes = 8 → center_cubes = 2 →
  total_shaded_cubes = 10 := 
by
  intros _ _ _ _ 
  sorry

end shaded_cubes_count_l297_297347


namespace slower_train_speed_l297_297918

theorem slower_train_speed (faster_speed : ℝ) (time_passed : ℝ) (train_length : ℝ) (slower_speed: ℝ) :
  faster_speed = 50 ∧ time_passed = 15 ∧ train_length = 75 →
  slower_speed = 32 :=
by
  intro h
  sorry

end slower_train_speed_l297_297918


namespace apples_not_sold_l297_297411

theorem apples_not_sold : 
  ∀ (boxes_per_week : ℕ) (apples_per_box : ℕ) (sold_fraction : ℚ), 
  boxes_per_week = 10 → apples_per_box = 300 → sold_fraction = 3 / 4 → 
  let total_apples := boxes_per_week * apples_per_box in
  let sold_apples := sold_fraction * total_apples in
  let not_sold_apples := total_apples - sold_apples in
  not_sold_apples = 750 :=
begin
  intros,
  sorry,
end

end apples_not_sold_l297_297411


namespace additional_songs_added_l297_297526

theorem additional_songs_added (original_songs : ℕ) (song_duration : ℕ) (total_duration : ℕ) :
  original_songs = 25 → song_duration = 3 → total_duration = 105 → 
  (total_duration - original_songs * song_duration) / song_duration = 10 :=
by
  intros h1 h2 h3
  sorry

end additional_songs_added_l297_297526


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297316

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d > 1 ∧ d < n

def consecutiveComposites (start n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < n → isComposite (start + i)

theorem exists_nine_consecutive_composites :
  ∃ start, start + 8 ≤ 500 ∧ consecutiveComposites start 9 :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ start, start + 10 ≤ 500 ∧ consecutiveComposites start 11 :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297316


namespace vasya_improved_example1_vasya_improved_example2_l297_297779

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l297_297779


namespace frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l297_297578

-- Definitions of conditions
def grasshopper_jump : ℕ := 19
def mouse_jump_frog (frog_jump : ℕ) : ℕ := frog_jump + 20
def mouse_jump_grasshopper : ℕ := grasshopper_jump + 30

-- The proof problem statement
theorem frog_jumps_10_inches_more_than_grasshopper (frog_jump : ℕ) :
  mouse_jump_frog frog_jump = mouse_jump_grasshopper → frog_jump = 29 :=
by
  sorry

-- The ultimate question in the problem
theorem frog_jumps_10_inches_farther_than_grasshopper : 
  (∃ (frog_jump : ℕ), frog_jump = 29) → (frog_jump - grasshopper_jump = 10) :=
by
  sorry

end frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l297_297578


namespace gymnast_scores_difference_l297_297809

theorem gymnast_scores_difference
  (s1 s2 s3 s4 s5 : ℝ)
  (h1 : (s2 + s3 + s4 + s5) / 4 = 9.46)
  (h2 : (s1 + s2 + s3 + s4) / 4 = 9.66)
  (h3 : (s2 + s3 + s4) / 3 = 9.58)
  : |s5 - s1| = 8.3 :=
sorry

end gymnast_scores_difference_l297_297809


namespace soccer_lineup_count_l297_297558

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem soccer_lineup_count : 
  let total_players := 18
  let goalies := 1
  let defenders := 6
  let forwards := 4
  18 * choose 17 6 * choose 11 4 = 73457760 :=
by
  sorry

end soccer_lineup_count_l297_297558


namespace hypotenuse_length_l297_297945

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l297_297945


namespace negation_proposition_l297_297754

theorem negation_proposition :
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proposition_l297_297754


namespace runners_meet_fractions_l297_297917

theorem runners_meet_fractions (l V₁ V₂ : ℝ)
  (h1 : l / V₂ - l / V₁ = 10)
  (h2 : 720 * V₁ - 720 * V₂ = l) :
  (1 / V₁ = 1 / 80 ∧ 1 / V₂ = 1 / 90) ∨ (1 / V₁ = 1 / 90 ∧ 1 / V₂ = 1 / 80) :=
sorry

end runners_meet_fractions_l297_297917


namespace max_distance_right_triangle_l297_297276

theorem max_distance_right_triangle (a b : ℝ) 
  (h1: ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (B.1 ^ 2 + B.2 ^ 2 = 1) ∧ 
    (a * A.1 + 2 * b * A.2 = 1) ∧ (a * B.1 + 2 * b * B.2 = 1) ∧ 
    ∃ (C : ℝ × ℝ), C = (0,0) ∧ (A.1 * B.1 + A.2 * B.2 = 0)): 
  ∃ (d : ℝ), d = (Real.sqrt (a^2 + b^2)) ∧ d ≤ Real.sqrt 2 :=
sorry

end max_distance_right_triangle_l297_297276


namespace instantaneous_rate_of_change_at_e_l297_297189

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem instantaneous_rate_of_change_at_e : deriv f e = 0 := by
  sorry

end instantaneous_rate_of_change_at_e_l297_297189


namespace right_triangle_hypotenuse_l297_297981

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l297_297981


namespace integrate_differential_eq_l297_297241

theorem integrate_differential_eq {x y C : ℝ} {y' : ℝ → ℝ → ℝ} (h : ∀ x y, (4 * y - 3 * x - 5) * y' x y + 7 * x - 3 * y + 2 = 0) : 
    ∃ C : ℝ, ∀ x y : ℝ, 2 * y^2 - 3 * x * y + (7/2) * x^2 + 2 * x - 5 * y = C :=
by
  sorry

end integrate_differential_eq_l297_297241


namespace probability_not_win_l297_297448

theorem probability_not_win (n : ℕ) (h : 1 - 1 / (n : ℝ) = 0.9375) : n = 16 :=
sorry

end probability_not_win_l297_297448


namespace original_savings_l297_297000

theorem original_savings (tv_cost : ℝ) (furniture_fraction : ℝ) (total_fraction : ℝ) (original_savings : ℝ) :
  tv_cost = 300 → furniture_fraction = 3 / 4 → total_fraction = 1 → 
  (total_fraction - furniture_fraction) * original_savings = tv_cost →
  original_savings = 1200 :=
by 
  intros htv hfurniture htotal hsavings_eq
  sorry

end original_savings_l297_297000


namespace boat_speed_in_still_water_l297_297402

theorem boat_speed_in_still_water (b s : ℕ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := by
  sorry

end boat_speed_in_still_water_l297_297402


namespace high_school_competition_arrangements_l297_297932

theorem high_school_competition_arrangements :
  let students := [1, 2, 3, 4, 5]
  let subjects := ["Mathematics", "Physics", "Chemistry"]
  ∃ (arrangements : Nat), arrangements = 180 :=
by
  sorry

end high_school_competition_arrangements_l297_297932


namespace find_a3_l297_297657

-- Definitions from conditions
def arithmetic_sum (a1 a3 : ℕ) := (3 / 2) * (a1 + a3)
def common_difference := 2
def S3 := 12

-- Theorem to prove that a3 = 6
theorem find_a3 (a1 a3 : ℕ) (h₁ : arithmetic_sum a1 a3 = S3) (h₂ : a3 = a1 + common_difference * 2) : a3 = 6 :=
by
  sorry

end find_a3_l297_297657


namespace first_interest_rate_is_correct_l297_297262

theorem first_interest_rate_is_correct :
  let A1 := 1500.0000000000007
  let A2 := 2500 - A1
  let yearly_income := 135
  (15.0 * (r / 100) + 6.0 * (A2 / 100) = yearly_income) -> r = 5.000000000000003 :=
sorry

end first_interest_rate_is_correct_l297_297262


namespace alex_total_earnings_l297_297661

def total_earnings (hours_w1 hours_w2 wage : ℕ) : ℕ :=
  (hours_w1 + hours_w2) * wage

theorem alex_total_earnings
  (hours_w1 hours_w2 wage : ℕ)
  (h1 : hours_w1 = 28)
  (h2 : hours_w2 = hours_w1 - 10)
  (h3 : wage * 10 = 80) :
  total_earnings hours_w1 hours_w2 wage = 368 :=
by
  sorry

end alex_total_earnings_l297_297661


namespace largest_even_among_consecutives_l297_297758

theorem largest_even_among_consecutives (x : ℤ) (h : (x + (x + 2) + (x + 4) = x + 18)) : x + 4 = 10 :=
by
  sorry

end largest_even_among_consecutives_l297_297758


namespace middle_term_in_arithmetic_sequence_l297_297036

theorem middle_term_in_arithmetic_sequence :
  let a := 3^2 in let c := 3^4 in
  ∃ z : ℤ, (2 * z = a + c) ∧ z = 45 := by
let a := 3^2
let c := 3^4
use (a + c) / 2
split
-- Prove that 2 * ((a + c) / 2) = a + c
sorry
-- Prove that (a + c) / 2 = 45
sorry

end middle_term_in_arithmetic_sequence_l297_297036


namespace unique_linear_eq_sol_l297_297420

theorem unique_linear_eq_sol (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a b c : ℤ), (∀ x y : ℕ, (a * x + b * y = c ↔ x = m ∧ y = n)) :=
by
  sorry

end unique_linear_eq_sol_l297_297420


namespace ten_row_triangle_total_l297_297465

theorem ten_row_triangle_total:
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  rods + connectors = 231 :=
by
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  sorry

end ten_row_triangle_total_l297_297465


namespace maggie_earnings_l297_297556

def subscriptions_to_parents := 4
def subscriptions_to_grandfather := 1
def subscriptions_to_next_door_neighbor := 2
def subscriptions_to_another_neighbor := 2 * subscriptions_to_next_door_neighbor
def subscription_rate := 5

theorem maggie_earnings : 
  (subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_next_door_neighbor + subscriptions_to_another_neighbor) * subscription_rate = 55 := 
by
  sorry

end maggie_earnings_l297_297556


namespace sum_of_consecutive_integers_l297_297691

theorem sum_of_consecutive_integers (n a : ℕ) (h₁ : 2 ≤ n) (h₂ : (n * (2 * a + n - 1)) = 36) :
    ∃! (a' n' : ℕ), 2 ≤ n' ∧ (n' * (2 * a' + n' - 1)) = 36 :=
  sorry

end sum_of_consecutive_integers_l297_297691


namespace coordinates_of_B_l297_297462

-- Define the initial conditions
def A : ℝ × ℝ := (-2, 1)
def jump_units : ℝ := 4

-- Define the function to compute the new coordinates after the jump
def new_coordinates (start : ℝ × ℝ) (jump : ℝ) : ℝ × ℝ :=
  let (x, y) := start
  (x + jump, y)

-- State the theorem to be proved
theorem coordinates_of_B
  (A : ℝ × ℝ) (jump_units : ℝ)
  (hA : A = (-2, 1))
  (h_jump : jump_units = 4) :
  new_coordinates A jump_units = (2, 1) := 
by
  -- Placeholder for the actual proof
  sorry

end coordinates_of_B_l297_297462


namespace find_m_from_decomposition_l297_297365

theorem find_m_from_decomposition (m : ℕ) (h : m > 0) : (m^2 - m + 1 = 73) → (m = 9) :=
by
  sorry

end find_m_from_decomposition_l297_297365


namespace num_even_perfect_square_factors_of_2_6_5_3_7_8_l297_297689

def num_even_perfect_square_factors (n : ℕ) : ℕ :=
  if n = 2^6 * 5^3 * 7^8 then
    let valid_a := [2, 4, 6]
    let valid_c := [0, 2]
    let valid_b := [0, 2, 4, 6, 8]
    valid_a.length * valid_c.length * valid_b.length
  else 0

theorem num_even_perfect_square_factors_of_2_6_5_3_7_8 :
  num_even_perfect_square_factors (2^6 * 5^3 * 7^8) = 30 :=
by
  sorry

end num_even_perfect_square_factors_of_2_6_5_3_7_8_l297_297689


namespace sum_first_seven_terms_is_28_l297_297748

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence 
def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_a4_a6_sum : a 2 + a 4 + a 6 = 12

-- Prove that the sum of the first seven terms is 28
theorem sum_first_seven_terms_is_28 (h : is_arithmetic_seq a d) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
sorry

end sum_first_seven_terms_is_28_l297_297748


namespace students_taking_art_l297_297935

def total_students : ℕ := 500
def students_taking_music : ℕ := 20
def students_taking_both : ℕ := 10
def students_taking_neither : ℕ := 470

theorem students_taking_art :
  ∃ (A : ℕ), A = 20 ∧ total_students = 
             (students_taking_music - students_taking_both) + (A - students_taking_both) + students_taking_both + students_taking_neither :=
by
  sorry

end students_taking_art_l297_297935


namespace find_f_neg_a_l297_297520

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + 3 * Real.sin x + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 1) : f (-a) = 3 := by
  sorry

end find_f_neg_a_l297_297520


namespace largest_number_systematic_sampling_l297_297288

theorem largest_number_systematic_sampling (n k a1 a2: ℕ) (h1: n = 60) (h2: a1 = 3) (h3: a2 = 9) (h4: k = a2 - a1):
  ∃ largest, largest = a1 + k * (n / k - 1) := by
  sorry

end largest_number_systematic_sampling_l297_297288


namespace probability_rain_sunday_monday_l297_297279

def P_rain_Sunday : ℝ := 0.30
def P_rain_Monday_given_Sunday : ℝ := 0.50

theorem probability_rain_sunday_monday (h1 : P_rain_Sunday = 0.30) 
                                       (h2 : P_rain_Monday_given_Sunday = 0.50) : 
  P_rain_Sunday * P_rain_Monday_given_Sunday = 0.15 :=
by sorry

end probability_rain_sunday_monday_l297_297279


namespace jess_double_cards_l297_297883

theorem jess_double_cards (rob_total_cards jess_doubles : ℕ) 
    (one_third_rob_cards_doubles : rob_total_cards / 3 = rob_total_cards / 3)
    (jess_times_rob_doubles : jess_doubles = 5 * (rob_total_cards / 3)) :
    rob_total_cards = 24 → jess_doubles = 40 :=
  by
  sorry

end jess_double_cards_l297_297883


namespace negation_of_proposition_l297_297277

open Classical

theorem negation_of_proposition : (¬ ∀ x : ℝ, 2 * x + 4 ≥ 0) ↔ (∃ x : ℝ, 2 * x + 4 < 0) :=
by
  sorry

end negation_of_proposition_l297_297277


namespace right_triangle_hypotenuse_l297_297957

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l297_297957


namespace new_acute_angle_l297_297019

/- Definitions -/
def initial_angle_A (ACB : ℝ) (angle_CAB : ℝ) := angle_CAB = 40
def rotation_degrees (rotation : ℝ) := rotation = 480

/- Theorem Statement -/
theorem new_acute_angle (ACB : ℝ) (angle_CAB : ℝ) (rotation : ℝ) :
  initial_angle_A angle_CAB ACB ∧ rotation_degrees rotation → angle_CAB = 80 := 
by
  intros h
  -- This is where you'd provide the proof steps, but we use 'sorry' to indicate the proof is skipped.
  sorry

end new_acute_angle_l297_297019


namespace tetrahedron_volume_le_one_l297_297808

open Real

noncomputable def volume_tetrahedron (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  let (x0, y0, z0) := A
  let (x1, y1, z1) := B
  let (x2, y2, z2) := C
  let (x3, y3, z3) := D
  abs ((x1 - x0) * ((y2 - y0) * (z3 - z0) - (y3 - y0) * (z2 - z0)) -
       (x2 - x0) * ((y1 - y0) * (z3 - z0) - (y3 - y0) * (z1 - z0)) +
       (x3 - x0) * ((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0))) / 6

theorem tetrahedron_volume_le_one (A B C D : ℝ × ℝ × ℝ)
  (h1 : dist A B ≤ 2) (h2 : dist A C ≤ 2) (h3 : dist A D ≤ 2)
  (h4 : dist B C ≤ 2) (h5 : dist B D ≤ 2) (h6 : dist C D ≤ 2) :
  volume_tetrahedron A B C D ≤ 1 := by
  sorry

end tetrahedron_volume_le_one_l297_297808


namespace sqrt_of_16_is_4_l297_297491

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l297_297491


namespace solve_inequality_l297_297583

theorem solve_inequality (x : ℝ) : (2 * x - 3) / (x + 2) ≤ 1 ↔ (-2 < x ∧ x ≤ 5) :=
  sorry

end solve_inequality_l297_297583


namespace vasya_problem_l297_297767

theorem vasya_problem : 
  ∃ (e : ℝ), 
  (e = 333 / 3 - 33 / 3 ∨ e = 33 * 3 + 3 / 3) ∧
  (count threes in expression e < 10) ∧
  (e = 100) :=
by sorry

end vasya_problem_l297_297767


namespace max_prime_product_l297_297565

theorem max_prime_product : 
  ∃ (x y z : ℕ), 
    Prime x ∧ Prime y ∧ Prime z ∧ 
    x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    x + y + z = 49 ∧ 
    x * y * z = 4199 := 
by
  sorry

end max_prime_product_l297_297565


namespace range_of_a_l297_297698

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + 2 * x - 3

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → f a x ≤ f a y) ↔ (- (1/4:ℝ) ≤ a ∧ a ≤ 0) := by
  sorry

end range_of_a_l297_297698


namespace hyperbola_center_l297_297825

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0) : 
  x = 2 ∧ y = 4 :=
sorry

end hyperbola_center_l297_297825


namespace tom_initial_balloons_l297_297914

noncomputable def initial_balloons (x : ℕ) : ℕ :=
  if h₁ : x % 2 = 1 ∧ (x / 3) + 10 = 45 then x else 0

theorem tom_initial_balloons : initial_balloons 105 = 105 :=
by {
  -- Given x is an odd number and the equation (x / 3) + 10 = 45 holds, prove x = 105.
  -- These conditions follow from the problem statement directly.
  -- Proof is skipped.
  sorry
}

end tom_initial_balloons_l297_297914


namespace radius_increase_rate_l297_297137

theorem radius_increase_rate (r : ℝ) (u : ℝ)
  (h : r = 20) (dS_dt : ℝ) (h_dS_dt : dS_dt = 10 * Real.pi) :
  u = 1 / 4 :=
by
  have S := Real.pi * r^2
  have dS_dt_eq : dS_dt = 2 * Real.pi * r * u := sorry
  rw [h_dS_dt, h] at dS_dt_eq
  exact sorry

end radius_increase_rate_l297_297137


namespace circle_x_intercept_of_given_diameter_l297_297429

theorem circle_x_intercept_of_given_diameter (A B : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (10, 8)) : ∃ x : ℝ, ((A.1 + B.1) / 2, (A.2 + B.2) / 2).1 - 6 = 0 :=
by
  -- Sorry to skip the proof
  sorry

end circle_x_intercept_of_given_diameter_l297_297429


namespace triangle_angle_and_side_ratio_l297_297240

theorem triangle_angle_and_side_ratio
  (A B C : Real)
  (a b c : Real)
  (h1 : a / Real.sin A = b / Real.sin B)
  (h2 : b / Real.sin B = c / Real.sin C)
  (h3 : (a + c) / b = (Real.sin A - Real.sin B) / (Real.sin A - Real.sin C)) :
  C = Real.pi / 3 ∧ (1 < (a + b) / c ∧ (a + b) / c < 2) :=
by
  sorry


end triangle_angle_and_side_ratio_l297_297240


namespace arithmetic_sequence_15th_term_l297_297271

theorem arithmetic_sequence_15th_term :
  let first_term := 3
  let second_term := 8
  let third_term := 13
  let common_difference := second_term - first_term
  (first_term + (15 - 1) * common_difference) = 73 :=
by
  sorry

end arithmetic_sequence_15th_term_l297_297271


namespace coin_flips_137_l297_297424

-- Definitions and conditions
def steph_transformation_heads (x : ℤ) : ℤ := 2 * x - 1
def steph_transformation_tails (x : ℤ) : ℤ := (x + 1) / 2
def jeff_transformation_heads (y : ℤ) : ℤ := y + 8
def jeff_transformation_tails (y : ℤ) : ℤ := y - 3

-- The problem statement
theorem coin_flips_137
  (a b : ℤ)
  (h₁ : a - b = 7)
  (h₂ : 8 * a - 3 * b = 381)
  (steph_initial jeff_initial : ℤ)
  (h₃ : steph_initial = 4)
  (h₄ : jeff_initial = 4) : a + b = 137 := 
by
  sorry

end coin_flips_137_l297_297424


namespace distance_AB_l297_297869

def C1_polar (ρ θ : Real) : Prop :=
  ρ = 2 * Real.cos θ

def C2_polar (ρ θ : Real) : Prop :=
  ρ^2 * (1 + (Real.sin θ)^2) = 2

def ray_polar (θ : Real) : Prop :=
  θ = Real.pi / 6

theorem distance_AB :
  let ρ1 := 2 * Real.cos (Real.pi / 6)
  let ρ2 := Real.sqrt 10 * 2 / 5
  |ρ1 - ρ2| = Real.sqrt 3 - (2 * Real.sqrt 10) / 5 :=
by
  sorry

end distance_AB_l297_297869


namespace hypotenuse_right_triangle_l297_297985

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l297_297985


namespace total_length_of_free_sides_l297_297303

theorem total_length_of_free_sides (L W : ℝ) 
  (h1 : L = 2 * W) 
  (h2 : L * W = 128) : 
  L + 2 * W = 32 := by 
sorry

end total_length_of_free_sides_l297_297303


namespace find_XY_XZ_l297_297438

open Set

variable (P Q R X Y Z : Type) [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited X] [Inhabited Y] [Inhabited Z]
variable (length : (P → P → Real) → (Q → Q → Real) → (R → R → Real) → (X → X → Real) → (Y → Y → Real) → (Z → Z → Real) )


-- Definitions based on the conditions
def similar_triangles (PQ QR PR XY XZ YZ : Real) : Prop :=
  QR / YZ = PQ / XY ∧ QR / YZ = PR / XZ

def PQ : Real := 8
def QR : Real := 16
def YZ : Real := 32

-- We need to prove (XY = 16 ∧ XZ = 32) given the conditions of similarity
theorem find_XY_XZ (XY XZ : Real) (h_sim : similar_triangles PQ QR PQ XY XZ YZ) : XY = 16 ∧ XZ = 32 :=
by
  sorry

end find_XY_XZ_l297_297438


namespace problem_conditions_l297_297509

noncomputable def f (a b x : ℝ) : ℝ := abs (x + a) + abs (2 * x - b)

theorem problem_conditions (ha : 0 < a) (hb : 0 < b) 
  (hmin : ∃ x : ℝ, f a b x = 1) : 
  2 * a + b = 2 ∧ 
  ∀ (t : ℝ), (∀ a b : ℝ, 
    (0 < a) → (0 < b) → (a + 2 * b ≥ t * a * b)) → 
  t ≤ 9 / 2 :=
by
  sorry

end problem_conditions_l297_297509


namespace area_of_triangle_HFG_l297_297547

noncomputable def calculate_area_of_triangle (A B C : (ℝ × ℝ)) :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_HFG :
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 4)
  let D := (0, 4)
  let E := (2, 2)
  let F := (1, 4)
  let G := (0, 2)
  let H := ((2 + 1 + 0) / 3, (2 + 4 + 2) / 3)
  calculate_area_of_triangle H F G = 2/3 :=
by
  sorry

end area_of_triangle_HFG_l297_297547


namespace polar_area_enclosed_l297_297427

theorem polar_area_enclosed :
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  area = 8 * Real.pi / 3 :=
by
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  show area = 8 * Real.pi / 3
  sorry

end polar_area_enclosed_l297_297427


namespace log_increasing_condition_log_increasing_not_necessary_l297_297451

theorem log_increasing_condition (a : ℝ) (h : a > 2) : a > 1 :=
by sorry

theorem log_increasing_not_necessary (a : ℝ) : ∃ b, (b > 1 ∧ ¬(b > 2)) :=
by sorry

end log_increasing_condition_log_increasing_not_necessary_l297_297451


namespace trigonometric_identity_l297_297225

noncomputable def tan_alpha : ℝ := 4

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = tan_alpha) :
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / Real.cos (-α) = 3 :=
by
  sorry

end trigonometric_identity_l297_297225


namespace kanul_total_amount_l297_297450

variable (T : ℝ)
variable (H1 : 3000 + 2000 + 0.10 * T = T)

theorem kanul_total_amount : T = 5555.56 := 
by 
  /- with the conditions given, 
     we can proceed to prove T = 5555.56 -/
  sorry

end kanul_total_amount_l297_297450


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l297_297768

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l297_297768


namespace faster_growth_f_g_l297_297039

noncomputable def f (x : ℝ) : ℝ := x^3
noncomputable def g (x : ℝ) : ℝ := x^2 * Real.log x

theorem faster_growth_f_g (x : ℝ) (h : x > 0) : 
  ∃ x₀ > 0, ∀ x > x₀, f x > g x := 
by
  sorry

end faster_growth_f_g_l297_297039


namespace original_number_l297_297919

theorem original_number (x : ℕ) : 
  (∃ y : ℕ, y = x + 28 ∧ (y % 5 = 0) ∧ (y % 6 = 0) ∧ (y % 4 = 0) ∧ (y % 3 = 0)) → x = 32 :=
by
  sorry

end original_number_l297_297919


namespace solid_with_square_views_is_cube_l297_297581

-- Define the conditions and the solid type
def is_square_face (view : Type) : Prop := 
  -- Definition to characterize a square view. This is general,
  -- as the detailed characterization of a 'square' in Lean would depend
  -- on more advanced geometry modules, assuming a simple predicate here.
  sorry

structure Solid := (front_view : Type) (top_view : Type) (left_view : Type)

-- Conditions indicating that all views are squares
def all_views_square (S : Solid) : Prop :=
  is_square_face S.front_view ∧ is_square_face S.top_view ∧ is_square_face S.left_view

-- The theorem we are aiming to prove
theorem solid_with_square_views_is_cube (S : Solid) (h : all_views_square S) : S = {front_view := ℝ, top_view := ℝ, left_view := ℝ} := sorry

end solid_with_square_views_is_cube_l297_297581


namespace triangle_weight_l297_297711

variables (S C T : ℕ)

def scale1 := (S + C = 8)
def scale2 := (S + 2 * C = 11)
def scale3 := (C + 2 * T = 15)

theorem triangle_weight (h1 : scale1 S C) (h2 : scale2 S C) (h3 : scale3 C T) : T = 6 :=
by 
  sorry

end triangle_weight_l297_297711


namespace average_income_PQ_l297_297894

/-
Conditions:
1. The average monthly income of Q and R is Rs. 5250.
2. The average monthly income of P and R is Rs. 6200.
3. The monthly income of P is Rs. 3000.
-/

def avg_income_QR := 5250
def avg_income_PR := 6200
def income_P := 3000

theorem average_income_PQ :
  ∃ (Q R : ℕ), ((Q + R) / 2 = avg_income_QR) ∧ ((income_P + R) / 2 = avg_income_PR) ∧ 
               (∀ (p q : ℕ), p = income_P → q = (Q + income_P) / 2 → q = 2050) :=
by
  sorry

end average_income_PQ_l297_297894


namespace inequality_a_b_c_d_l297_297531

theorem inequality_a_b_c_d 
  (a b c d : ℝ) 
  (h0 : 0 ≤ a) 
  (h1 : a ≤ b) 
  (h2 : b ≤ c) 
  (h3 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
by
  sorry

end inequality_a_b_c_d_l297_297531


namespace min_value_sum_reciprocal_l297_297729

open Real

theorem min_value_sum_reciprocal (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
    (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
    1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 3 / 4 :=
by
  sorry

end min_value_sum_reciprocal_l297_297729


namespace turtles_received_l297_297252

theorem turtles_received (martha_turtles : ℕ) (marion_turtles : ℕ) (h1 : martha_turtles = 40) 
    (h2 : marion_turtles = martha_turtles + 20) : martha_turtles + marion_turtles = 100 := 
by {
    sorry
}

end turtles_received_l297_297252


namespace worst_player_is_nephew_l297_297397

-- Define the family members
inductive Player
| father : Player
| sister : Player
| son : Player
| nephew : Player

open Player

-- Define a twin relationship
def is_twin (p1 p2 : Player) : Prop :=
  (p1 = son ∧ p2 = nephew) ∨ (p1 = nephew ∧ p2 = son)

-- Define that two players are of opposite sex
def opposite_sex (p1 p2 : Player) : Prop :=
  (p1 = sister ∧ (p2 = father ∨ p2 = son ∨ p2 = nephew)) ∨
  (p2 = sister ∧ (p1 = father ∨ p1 = son ∨ p1 = nephew))

-- Predicate for the worst player
structure WorstPlayer (p : Player) : Prop :=
  (twin_exists : ∃ twin : Player, is_twin p twin)
  (opposite_sex_best : ∀ twin best, is_twin p twin → best ≠ twin → opposite_sex twin best)

-- The goal is to show that the worst player is the nephew
theorem worst_player_is_nephew : WorstPlayer nephew := sorry

end worst_player_is_nephew_l297_297397


namespace student_test_score_l297_297464

variable (C I : ℕ)

theorem student_test_score  
  (h1 : C + I = 100)
  (h2 : C - 2 * I = 64) :
  C = 88 :=
by
  -- Proof steps should go here
  sorry

end student_test_score_l297_297464


namespace second_candidate_votes_l297_297867

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℝ) (first_candidate_votes: ℕ)
    (h1 : total_votes = 2400)
    (h2 : first_candidate_percentage = 0.80)
    (h3 : first_candidate_votes = total_votes * first_candidate_percentage) :
    total_votes - first_candidate_votes = 480 := by
    sorry

end second_candidate_votes_l297_297867


namespace volume_pyramid_EFGH_l297_297413

open Real

noncomputable def volume_pyramid (EF FG EH : ℝ) (θ : ℝ) (cos_theta : ℝ) : ℝ :=
  (1 / 3) * EF * FG * EH * cos θ

theorem volume_pyramid_EFGH (EF FG EH : ℝ) (θ : ℝ) (cos_theta : ℝ)
  (hEFGH_rect : true) -- if necessary, define properties of a rectangle
  (h_cos_theta : cos θ = 4 / 5)
  : volume_pyramid EF FG EH θ cos_theta = (128 / 3) := by
  sorry

end volume_pyramid_EFGH_l297_297413


namespace parallel_lines_l297_297700

theorem parallel_lines (a : ℝ) :
  ((3 * a + 2) * x + a * y + 6 = 0) ↔
  (a * x - y + 3 = 0) →
  a = -1 :=
by sorry

end parallel_lines_l297_297700


namespace odd_pos_4_digit_ints_div_5_no_digit_5_l297_297224

open Nat

def is_valid_digit (d : Nat) : Prop :=
  d ≠ 5

def valid_odd_4_digit_ints_count : Nat :=
  let a := 8  -- First digit possibilities: {1, 2, 3, 4, 6, 7, 8, 9}
  let bc := 9  -- Second and third digit possibilities: {0, 1, 2, 3, 4, 6, 7, 8, 9}
  let d := 4  -- Fourth digit possibilities: {1, 3, 7, 9}
  a * bc * bc * d

theorem odd_pos_4_digit_ints_div_5_no_digit_5 : valid_odd_4_digit_ints_count = 2592 := by
  sorry

end odd_pos_4_digit_ints_div_5_no_digit_5_l297_297224


namespace alice_jack_meet_l297_297398

theorem alice_jack_meet (n : ℕ) :
  (∃ n, (7 * n) % 18 = (18 - 14 * n) % 18) ∧ n = 6 :=
by
  sorry

end alice_jack_meet_l297_297398


namespace area_triangle_l297_297359

noncomputable def area_of_triangle_ABC (AB BC : ℝ) : ℝ := 
    (1 / 2) * AB * BC 

theorem area_triangle (AC : ℝ) (h1 : AC = 40)
    (h2 : ∃ B C : ℝ, B = (1/2) * AC ∧ C = B * Real.sqrt 3) :
    area_of_triangle_ABC ((1 / 2) * AC) (((1 / 2) * AC) * Real.sqrt 3) = 200 * Real.sqrt 3 := 
sorry

end area_triangle_l297_297359


namespace task_completion_time_l297_297857

theorem task_completion_time (A B : ℝ) : 
  (14 * A / 80 + 10 * B / 96) = (20 * (A + B)) →
  (1 / (14 * A / 80 + 10 * B / 96)) = 480 / (84 * A + 50 * B) :=
by
  intros h
  sorry

end task_completion_time_l297_297857


namespace find_angle_x_l297_297238

theorem find_angle_x (A B C : Type) (angle_ABC angle_CAB x : ℝ) 
  (h1 : angle_ABC = 40) 
  (h2 : angle_CAB = 120)
  (triangle_sum : x + angle_ABC + (180 - angle_CAB) = 180) : 
  x = 80 :=
by 
  -- actual proof goes here
  sorry

end find_angle_x_l297_297238


namespace find_f_2013_l297_297818

noncomputable def f : ℝ → ℝ := sorry
axiom functional_eq : ∀ (m n : ℝ), f (m + n^2) = f m + 2 * (f n)^2
axiom f_1_ne_0 : f 1 ≠ 0

theorem find_f_2013 : f 2013 = 4024 * (f 1)^2 + f 1 :=
sorry

end find_f_2013_l297_297818


namespace evaluate_expression_at_three_l297_297208

theorem evaluate_expression_at_three : 
  (3^2 + 3 * (3^6) = 2196) :=
by
  sorry -- This is where the proof would go

end evaluate_expression_at_three_l297_297208


namespace total_amount_contribution_l297_297005

theorem total_amount_contribution : 
  let r := 285
  let s := 35
  let a := 30
  let d := a / 2
  let c := 35
  r + s + a + d + c = 400 :=
by
  sorry

end total_amount_contribution_l297_297005


namespace budget_equality_year_l297_297861

theorem budget_equality_year :
  ∀ Q R V W : ℕ → ℝ,
  Q 0 = 540000 ∧ R 0 = 660000 ∧ V 0 = 780000 ∧ W 0 = 900000 ∧
  (∀ n, Q (n+1) = Q n + 40000 ∧ 
         R (n+1) = R n + 30000 ∧ 
         V (n+1) = V n - 10000 ∧ 
         W (n+1) = W n - 20000) →
  ∃ n : ℕ, 1990 + n = 1995 ∧ 
  Q n + R n = V n + W n := 
by 
  sorry

end budget_equality_year_l297_297861


namespace orange_juice_fraction_l297_297589

theorem orange_juice_fraction 
    (capacity1 capacity2 : ℕ)
    (orange_fraction1 orange_fraction2 : ℚ)
    (h_capacity1 : capacity1 = 800)
    (h_capacity2 : capacity2 = 700)
    (h_orange_fraction1 : orange_fraction1 = 1/4)
    (h_orange_fraction2 : orange_fraction2 = 1/3) :
    (capacity1 * orange_fraction1 + capacity2 * orange_fraction2) / (capacity1 + capacity2) = 433.33 / 1500 :=
by sorry

end orange_juice_fraction_l297_297589


namespace shipping_cost_per_unit_leq_one_point_six_seven_l297_297798

variable (S : ℝ)

-- Conditions as definitions in Lean 4
def production_cost_per_component : ℝ := 80
def fixed_monthly_cost : ℝ := 16500
def components_per_month : ℕ := 150
def selling_price_per_component : ℝ := 191.67

-- The theorem to prove the shipping cost per unit
theorem shipping_cost_per_unit_leq_one_point_six_seven :
  let production_cost := components_per_month * production_cost_per_component
  let shipping_cost := components_per_month * S
  let total_revenue := components_per_month * selling_price_per_component
  total_revenue ≥ production_cost + shipping_cost + fixed_monthly_cost → S ≤ 1.67 :=
by
  sorry

end shipping_cost_per_unit_leq_one_point_six_seven_l297_297798


namespace right_triangle_hypotenuse_length_l297_297960

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l297_297960


namespace ratio_long_side_brush_width_l297_297323

theorem ratio_long_side_brush_width 
  (l : ℝ) (w : ℝ) (d : ℝ) (total_area : ℝ) (painted_area : ℝ) (b : ℝ) 
  (h1 : l = 9)
  (h2 : w = 4)
  (h3 : total_area = l * w)
  (h4 : total_area / 3 = painted_area)
  (h5 : d = Real.sqrt (l^2 + w^2))
  (h6 : d * b = painted_area) :
  l / b = (3 * Real.sqrt 97) / 4 :=
by
  sorry

end ratio_long_side_brush_width_l297_297323


namespace macy_miles_left_l297_297126

theorem macy_miles_left (goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) 
  (H1 : goal = 24) 
  (H2 : daily_miles = 3) 
  (H3 : days_run = 6) 
  : goal - daily_miles * days_run = 6 := 
by 
  sorry

end macy_miles_left_l297_297126


namespace giant_kite_area_72_l297_297831

-- Definition of the vertices of the medium kite
def vertices_medium_kite : List (ℕ × ℕ) := [(1,6), (4,9), (7,6), (4,1)]

-- Given condition function to check if the giant kite is created by doubling the height and width
def double_coordinates (c : (ℕ × ℕ)) : (ℕ × ℕ) := (2 * c.1, 2 * c.2)

def vertices_giant_kite : List (ℕ × ℕ) := vertices_medium_kite.map double_coordinates

-- Function to calculate the area of the kite based on its vertices
def kite_area (vertices : List (ℕ × ℕ)) : ℕ := sorry -- The way to calculate the kite area can be complex

-- Theorem to prove the area of the giant kite
theorem giant_kite_area_72 :
  kite_area vertices_giant_kite = 72 := 
sorry

end giant_kite_area_72_l297_297831


namespace width_of_hall_l297_297543

variable (L W H : ℕ) -- Length, Width, Height of the hall
variable (expenditure cost : ℕ) -- Expenditure and cost per square meter

-- Given conditions
def hall_length : L = 20 := by sorry
def hall_height : H = 5 := by sorry
def total_expenditure : expenditure = 28500 := by sorry
def cost_per_sq_meter : cost = 30 := by sorry

-- Derived value
def total_area_to_cover (W : ℕ) : ℕ :=
  (2 * (L * W) + 2 * (L * H) + 2 * (W * H))

theorem width_of_hall (W : ℕ) (h: total_area_to_cover L W H * cost = expenditure) : W = 15 := by
  sorry

end width_of_hall_l297_297543


namespace negation_proposition_l297_297433

theorem negation_proposition :
  (¬ (∀ x : ℝ, x ≥ 0)) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end negation_proposition_l297_297433


namespace shirt_cost_l297_297086

def george_initial_money : ℕ := 100
def total_spent_on_clothes (initial_money remaining_money : ℕ) : ℕ := initial_money - remaining_money
def socks_cost : ℕ := 11
def remaining_money_after_purchase : ℕ := 65

theorem shirt_cost
  (initial_money : ℕ)
  (remaining_money : ℕ)
  (total_spent : ℕ)
  (socks_cost : ℕ)
  (remaining_money_after_purchase : ℕ) :
  initial_money = 100 →
  remaining_money = 65 →
  total_spent = initial_money - remaining_money →
  total_spent = 35 →
  socks_cost = 11 →
  remaining_money_after_purchase = remaining_money →
  (total_spent - socks_cost = 24) :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h4] at *
  exact sorry

end shirt_cost_l297_297086


namespace square_of_number_ending_in_5_l297_297734

theorem square_of_number_ending_in_5 (a : ℤ) :
  (10 * a + 5) * (10 * a + 5) = 100 * a * (a + 1) + 25 := by
  sorry

end square_of_number_ending_in_5_l297_297734


namespace find_a_l297_297430

variable (a : ℝ) -- Declare a as a real number.

-- Define the given conditions.
def condition1 (a : ℝ) : Prop := a^2 - 2 * a = 0
def condition2 (a : ℝ) : Prop := a ≠ 2

-- Define the theorem stating that if conditions are true, then a must be 0.
theorem find_a (h1 : condition1 a) (h2 : condition2 a) : a = 0 :=
sorry -- Proof is not provided, it needs to be constructed.

end find_a_l297_297430


namespace initial_deposit_l297_297244

theorem initial_deposit :
  ∀ (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ),
    r = 0.05 → n = 1 → t = 2 → P * (1 + r / n) ^ (n * t) = 6615 → P = 6000 :=
by
  intros P r n t h_r h_n h_t h_eq
  rw [h_r, h_n, h_t] at h_eq
  norm_num at h_eq
  sorry

end initial_deposit_l297_297244


namespace no_unfenced_area_l297_297253

noncomputable def area : ℝ := 5000
noncomputable def cost_per_foot : ℝ := 30
noncomputable def budget : ℝ := 120000

theorem no_unfenced_area (area : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  (budget / cost_per_foot) >= 4 * (Real.sqrt (area)) → 0 = 0 :=
by
  intro h
  sorry

end no_unfenced_area_l297_297253


namespace choir_meets_every_5_days_l297_297573

theorem choir_meets_every_5_days (n : ℕ) (h1 : n = 15) (h2 : ∃ k : ℕ, 15 = 3 * k) : ∃ x : ℕ, 15 = x * 3 ∧ x = 5 := 
by
  sorry

end choir_meets_every_5_days_l297_297573


namespace exponents_multiplication_exponents_power_exponents_distributive_l297_297999

variables (x y m : ℝ)

theorem exponents_multiplication (x : ℝ) : (x^5) * (x^2) = x^7 :=
by sorry

theorem exponents_power (m : ℝ) : (m^2)^4 = m^8 :=
by sorry

theorem exponents_distributive (x y : ℝ) : (-2 * x * y^2)^3 = -8 * x^3 * y^6 :=
by sorry

end exponents_multiplication_exponents_power_exponents_distributive_l297_297999


namespace arithmetic_sequence_terms_l297_297234

variable (n : ℕ)
variable (sumOdd sumEven : ℕ)
variable (terms : ℕ)

theorem arithmetic_sequence_terms
  (h1 : sumOdd = 120)
  (h2 : sumEven = 110)
  (h3 : terms = 2 * n + 1)
  (h4 : sumOdd + sumEven = 230) :
  terms = 23 := 
sorry

end arithmetic_sequence_terms_l297_297234


namespace rectangle_area_12_l297_297701

theorem rectangle_area_12
  (L W : ℝ)
  (h1 : L + W = 7)
  (h2 : L^2 + W^2 = 25) :
  L * W = 12 :=
by
  sorry

end rectangle_area_12_l297_297701


namespace probability_two_dice_sum_seven_l297_297148

theorem probability_two_dice_sum_seven (z : ℕ) (w : ℚ) (h : z = 2) : w = 1 / 6 :=
by sorry

end probability_two_dice_sum_seven_l297_297148


namespace find_m_l297_297056

noncomputable def g (n : ℤ) : ℤ :=
if n % 2 ≠ 0 then 2 * n + 3
else if n % 3 = 0 then n / 3
else n - 1

theorem find_m :
  ∃ m : ℤ, m % 2 ≠ 0 ∧ g (g (g m)) = 36 ∧ m = 54 :=
by
  sorry

end find_m_l297_297056


namespace arithmetic_sequence_z_value_l297_297035

theorem arithmetic_sequence_z_value :
  ∃ z : ℤ, (3 ^ 2 = 9 ∧ 3 ^ 4 = 81) ∧ z = (9 + 81) / 2 :=
by
  -- the proof goes here
  sorry

end arithmetic_sequence_z_value_l297_297035


namespace sum_infinite_partial_fraction_l297_297639

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l297_297639


namespace melissa_total_time_l297_297418

variable (b : ℝ) (h : ℝ) (n : ℕ)
variable (shoes : ℕ)

-- Definition of the time taken for buckles and heels
def time_for_buckles := n * b
def time_for_heels := n * h

-- The total time Melissa spends repairing
def total_time := time_for_buckles + time_for_heels

theorem melissa_total_time :
  total_time b h 2 = 30 :=
by
  sorry

end melissa_total_time_l297_297418


namespace infinite_bad_numbers_l297_297515

-- Define types for natural numbers
variables {a b : ℕ}

-- The theorem statement
theorem infinite_bad_numbers (a b : ℕ) : ∃ᶠ (n : ℕ) in at_top, n > 0 ∧ ¬ (n^b + 1 ∣ a^n + 1) :=
sorry

end infinite_bad_numbers_l297_297515


namespace find_n_values_l297_297080

-- Define a function that calculates the polynomial expression
def prime_expression (n : ℕ) : ℕ :=
  n^4 - 27 * n^2 + 121

-- State the problem as a theorem
theorem find_n_values (n : ℕ) (h : Nat.Prime (prime_expression n)) : n = 2 ∨ n = 5 :=
  sorry

end find_n_values_l297_297080


namespace sum_series_l297_297620

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l297_297620


namespace relationship_among_values_l297_297680

-- Define the properties of the function f
variables (f : ℝ → ℝ)

-- Assume necessary conditions
axiom domain_of_f : ∀ x : ℝ, f x ≠ 0 -- Domain of f is ℝ
axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_function : ∀ x y : ℝ, (0 ≤ x) → (x ≤ y) → (f x ≤ f y) -- f is increasing for x in [0, + ∞)

-- Define the main theorem based on the problem statement
theorem relationship_among_values : f π > f (-3) ∧ f (-3) > f (-2) :=
by
  sorry

end relationship_among_values_l297_297680


namespace variance_of_scores_l297_297805

def scores : List ℝ := [8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (List.sum xs) / (xs.length)

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (List.sum (List.map (λ x => (x - m) ^ 2) xs)) / (xs.length)

theorem variance_of_scores : variance scores = 40 / 9 :=
by
  sorry

end variance_of_scores_l297_297805


namespace joan_payment_l297_297243

theorem joan_payment (cat_toy_cost cage_cost change_received : ℝ) 
  (h1 : cat_toy_cost = 8.77) 
  (h2 : cage_cost = 10.97) 
  (h3 : change_received = 0.26) : 
  cat_toy_cost + cage_cost - change_received = 19.48 := 
by 
  sorry

end joan_payment_l297_297243


namespace initial_quantity_of_milk_l297_297063

-- Define initial condition for the quantity of milk in container A
noncomputable def container_A : ℝ := 1184

-- Define the quantities of milk in containers B and C
def container_B (A : ℝ) : ℝ := 0.375 * A
def container_C (A : ℝ) : ℝ := 0.625 * A

-- Define the final equal quantities of milk after transfer
def equal_quantity (A : ℝ) : ℝ := container_B A + 148

-- The proof statement that must be true
theorem initial_quantity_of_milk :
  ∀ (A : ℝ), container_B A + 148 = equal_quantity A → A = container_A :=
by
  intros A h
  rw [equal_quantity] at h
  sorry

end initial_quantity_of_milk_l297_297063


namespace fg_minus_gf_l297_297266

-- Definitions provided by the conditions
def f (x : ℝ) : ℝ := 4 * x + 8
def g (x : ℝ) : ℝ := 2 * x - 3

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -17 := 
  sorry

end fg_minus_gf_l297_297266


namespace paint_fence_together_time_l297_297406

-- Define the times taken by Jamshid and Taimour
def Taimour_time := 18 -- Taimour takes 18 hours to paint the fence
def Jamshid_time := Taimour_time / 2 -- Jamshid takes half the time Taimour takes

-- Define the work rates
def Taimour_rate := 1 / Taimour_time
def Jamshid_rate := 1 / Jamshid_time

-- Define the combined work rate
def combined_rate := Taimour_rate + Jamshid_rate

-- Define the total time taken when working together
def together_time := 1 / combined_rate

-- State the main theorem
theorem paint_fence_together_time : together_time = 6 := 
sorry

end paint_fence_together_time_l297_297406


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l297_297774

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l297_297774


namespace marble_arrangement_count_l297_297922
noncomputable def countValidMarbleArrangements : Nat := 
  let totalArrangements := 120
  let restrictedPairsCount := 24
  totalArrangements - restrictedPairsCount

theorem marble_arrangement_count :
  countValidMarbleArrangements = 96 :=
  by
    sorry

end marble_arrangement_count_l297_297922


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l297_297783

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l297_297783


namespace inequality_proof_l297_297250

variable (a b c : ℝ)
variable (h_pos : a > 0) (h_pos2 : b > 0) (h_pos3 : c > 0)
variable (h_sum : a + b + c = 1)

theorem inequality_proof :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) > 1 / 2 := by
  sorry

end inequality_proof_l297_297250


namespace find_cos_C_find_area_l297_297112

open Real

variables {a b c : ℝ}
variables {A B C : ℝ} -- Angles of the triangle

-- Initial conditions of the triangle
axiom triangle_sides_opposite_angles : a = opposite A ∧ b = opposite B ∧ c = opposite C

-- Specified conditions for the proof
axiom condition_1 : 3 * a + b = 2 * c
axiom condition_2 : b = 2
axiom condition_3 : (1 / sin A) + (1 / sin C) = (4 * sqrt 3) / 3
axiom condition_4 : (2 * c - a) * cos B = b * cos A
axiom condition_5 : a ^ 2 + c ^ 2 - b ^ 2 = (4 * sqrt 3 / 3) * S
axiom condition_6 : 2 * b * sin (A + π / 6) = a + c

-- Proving questions
theorem find_cos_C (h1 : 3 * a + b = 2 * c) : cos C = -1 / 7 := sorry

theorem find_area (h2 : b = 2) (h3 : (1 / sin A) + (1 / sin C) = (4 * sqrt 3) / 3) : S = sqrt 3 := sorry

end find_cos_C_find_area_l297_297112


namespace inequality_of_negatives_l297_297841

theorem inequality_of_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end inequality_of_negatives_l297_297841


namespace hypotenuse_length_l297_297970

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l297_297970


namespace find_number_of_friends_l297_297289

def dante_balloons : Prop :=
  ∃ F : ℕ, (F > 0 ∧ (250 / F) - 11 = 39) ∧ F = 5

theorem find_number_of_friends : dante_balloons :=
by
  sorry

end find_number_of_friends_l297_297289


namespace dot_product_is_six_l297_297508

def a : ℝ × ℝ := (-2, 4)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_is_six : (a.1 * b.1 + a.2 * b.2) = 6 := 
by 
  -- definition and proof logic follows
  sorry

end dot_product_is_six_l297_297508


namespace correct_statements_l297_297162

variable (P Q : Prop)

-- Define statements
def is_neg_false_if_orig_true := (P → ¬P) = False
def is_converse_not_nec_true_if_orig_true := (P → Q) → ¬(Q → P)
def is_neg_true_if_converse_true := (Q → P) → (¬P → ¬Q)
def is_neg_true_if_contrapositive_true := (¬Q → ¬P) → (¬P → False)

-- Main proposition
theorem correct_statements : 
  is_converse_not_nec_true_if_orig_true P Q ∧ 
  is_neg_true_if_converse_true P Q :=
by
  sorry

end correct_statements_l297_297162


namespace sqrt_sixteen_is_four_l297_297486

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l297_297486


namespace no_rational_roots_l297_297203

theorem no_rational_roots (x : ℚ) : ¬(3 * x^4 + 2 * x^3 - 8 * x^2 - x + 1 = 0) :=
by sorry

end no_rational_roots_l297_297203


namespace nonneg_reals_sum_to_one_implies_ineq_l297_297375

theorem nonneg_reals_sum_to_one_implies_ineq
  (x y z : ℝ)
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 :=
sorry

end nonneg_reals_sum_to_one_implies_ineq_l297_297375


namespace count_matching_placements_l297_297259

theorem count_matching_placements : 
  ∃ (placements : Finset (Fin 6 → Fin 6)), 
  (∀ f ∈ placements, (Finset.card (Finset.filter (λ x => f x = x) (Finset.univ : Finset (Fin 6))) = 3)) ∧ 
  Finset.card placements = 40 :=
by
  sorry

end count_matching_placements_l297_297259


namespace macy_running_goal_l297_297123

/-- Macy's weekly running goal is 24 miles. She runs 3 miles per day. Calculate the miles 
    she has left to run after 6 days to meet her goal. --/
theorem macy_running_goal (miles_per_week goal_per_week : ℕ) (miles_per_day: ℕ) (days_run: ℕ) 
  (h1 : miles_per_week = 24) (h2 : miles_per_day = 3) (h3 : days_run = 6) : 
  miles_per_week - miles_per_day * days_run = 6 := 
  by 
    rw [h1, h2, h3]
    exact Nat.sub_eq_of_eq_add (by norm_num)

end macy_running_goal_l297_297123


namespace corrected_mean_l297_297432

theorem corrected_mean :
  let original_mean := 45
  let num_observations := 100
  let observations_wrong := [32, 12, 25]
  let observations_correct := [67, 52, 85]
  let original_total_sum := original_mean * num_observations
  let incorrect_sum := observations_wrong.sum
  let correct_sum := observations_correct.sum
  let adjustment := correct_sum - incorrect_sum
  let corrected_total_sum := original_total_sum + adjustment
  let corrected_new_mean := corrected_total_sum / num_observations
  corrected_new_mean = 46.35 := 
by
  sorry

end corrected_mean_l297_297432


namespace mod_equiv_n_l297_297155

theorem mod_equiv_n (n : ℤ) : 0 ≤ n ∧ n < 9 ∧ -1234 % 9 = n := 
by
  sorry

end mod_equiv_n_l297_297155


namespace solve_equation_l297_297011

theorem solve_equation (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  (x^2 + x + 1 = 1 / (x^2 - x + 1)) ↔ x = 1 ∨ x = -1 :=
by sorry

end solve_equation_l297_297011


namespace calculate_expression_l297_297740

variable (x y : ℝ)

theorem calculate_expression (h1 : x + y = 5) (h2 : x * y = 3) : 
   x + (x^4 / y^3) + (y^4 / x^3) + y = 27665 / 27 :=
by
  sorry

end calculate_expression_l297_297740


namespace trajectory_equation_l297_297093

-- Define the fixed points F1 and F2
structure Point where
  x : ℝ
  y : ℝ

def F1 : Point := ⟨-2, 0⟩
def F2 : Point := ⟨2, 0⟩

-- Define the moving point M and the condition it must satisfy
def satisfies_condition (M : Point) : Prop :=
  (Real.sqrt ((M.x + 2)^2 + M.y^2) - Real.sqrt ((M.x - 2)^2 + M.y^2)) = 4

-- The trajectory of the point M must satisfy y = 0 and x >= 2
def on_trajectory (M : Point) : Prop :=
  M.y = 0 ∧ M.x ≥ 2

-- The final theorem to be proved
theorem trajectory_equation (M : Point) (h : satisfies_condition M) : on_trajectory M := by
  sorry

end trajectory_equation_l297_297093


namespace juniors_in_club_l297_297864

theorem juniors_in_club
  (j s x y : ℝ)
  (h1 : x = 0.4 * j)
  (h2 : y = 0.25 * s)
  (h3 : j + s = 36)
  (h4 : x = 2 * y) :
  j = 20 :=
by
  sorry

end juniors_in_club_l297_297864


namespace part1_part2_l297_297522

noncomputable def f (x a : ℝ) := x * Real.log (x + 1) + (1/2 - a) * x + 2 - a

noncomputable def g (x a : ℝ) := f x a + Real.log (x + 1) + 1/2 * x

theorem part1 (a : ℝ) (x : ℝ) (h : x > 0) : 
  (a ≤ 2 → ∀ x, g x a > 0) ∧ 
  (a > 2 → ∀ x, x < Real.exp (a - 2) - 1 → g x a < 0) ∧
  (a > 2 → ∀ x, x > Real.exp (a - 2) - 1 → g x a > 0) :=
sorry

theorem part2 (a : ℤ) : 
  (∃ x ≥ 0, f x a < 0) → a ≥ 3 :=
sorry

end part1_part2_l297_297522


namespace total_bill_l297_297468

def number_of_adults := 2
def number_of_children := 5
def meal_cost := 3

theorem total_bill : number_of_adults * meal_cost + number_of_children * meal_cost = 21 :=
by
  sorry

end total_bill_l297_297468


namespace y_intercept_of_line_l297_297025

-- Define the line equation
def line_eq (x y : ℚ) : Prop := x - 2 * y - 3 = 0

-- Define the y_intercept function that finds the y-value when x is 0
def y_intercept (L : ℚ → ℚ → Prop) : ℚ :=
  if h : ∃ y, L 0 y then classical.some h else 0

-- Define the theorem to prove the y-intercept equals -3 / 2
theorem y_intercept_of_line : y_intercept line_eq = -3/2 :=
by { sorry }

end y_intercept_of_line_l297_297025


namespace area_of_ABIJKFGD_eq_62_5_l297_297710

-- Definitions for the basic squares and setup
def ABCD := Square (pointA pointB pointC pointD : EuclideanSpace ℝ ℝ) :=
  side_length : ℝ, sideLength = 5, side_square := 25 -- area 25 square with side length 5
def EFGD := Square (pointE pointF pointG pointD : EuclideanSpace ℝ ℝ) :=
  side_length : ℝ, sideLength = 5, side_square := 25 -- area 25 square with side length 5

-- Condition for midpoint and intersection of squares
def IsMidpoint (P Q R : EuclideanSpace ℝ ℝ) := 
  dist P R = dist Q R / 2 

def EachLines (Square1 Square2 Square3 : EuclideanSpace ℝ ℝ) :=
  L_exists : Line (L: EuclideanSpace ℝ ℝ),
     L = Midpoint P Q ∧ L liesIn (Line pointE pointF) -- midpoint conditions and point L on EF
  
-- Definition and proof of the total area of polygon
theorem area_of_ABIJKFGD_eq_62_5 :
  ∀ (pointA pointB pointC pointD pointE pointF pointG pointL: EuclideanSpace ℝ ℝ), 
  side_square ABCD = 25 ∧ side_square EFGD = 25 ∧
  side_length sqIJKL = 5 ∧ 
  IsMidpoint pointH pointBC pointEF ∧ IsMidpoint pointD pointJK ∧
  LExists_line_midpoint_L pointE pointF pointL → 
  total_area_of_polygon_ABIJKFGD = 62.5 := 
begin
  -- proof
  sorry, -- proof omitted
end

end area_of_ABIJKFGD_eq_62_5_l297_297710


namespace steps_in_staircase_using_210_toothpicks_l297_297656

-- Define the conditions
def first_step : Nat := 3
def increment : Nat := 2
def total_toothpicks_5_steps : Nat := 55

-- Define required theorem
theorem steps_in_staircase_using_210_toothpicks : ∃ (n : ℕ), (n * (n + 2) = 210) ∧ n = 13 :=
by
  sorry

end steps_in_staircase_using_210_toothpicks_l297_297656


namespace right_triangle_hypotenuse_length_l297_297954

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l297_297954


namespace hypotenuse_length_l297_297971

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l297_297971


namespace find_x_l297_297216

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
sorry

end find_x_l297_297216


namespace all_points_lie_on_line_l297_297084

theorem all_points_lie_on_line:
  ∀ (s : ℝ), s ≠ 0 → ∀ (x y : ℝ),
  x = (2 * s + 3) / s → y = (2 * s - 3) / s → x + y = 4 :=
by
  intros s hs x y hx hy
  sorry

end all_points_lie_on_line_l297_297084


namespace problem_statement_l297_297521

noncomputable def f (a x : ℝ) : ℝ := Real.log x - (a * (x + 1) / (x - 1))

theorem problem_statement (a : ℝ) :
  (a > 0 → ∀ x ∈ Ioi 1, deriv (f a) x > 0) ∧
  ((deriv (f a) 2 = 2) → a = 3/4) ∧
  (a > 0 → (∃ x1 x2 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ x1 ≠ x2 ∧ x1 * x2 = 1)) :=
by
  sorry

end problem_statement_l297_297521


namespace compute_fraction_l297_297268

theorem compute_fraction (a b c : ℝ) (h1 : a + b = 20) (h2 : b + c = 22) (h3 : c + a = 2022) :
  (a - b) / (c - a) = 1000 :=
by
  sorry

end compute_fraction_l297_297268


namespace angle_measure_l297_297230

theorem angle_measure (x : ℝ) (h : 90 - x = 3 * (180 - x)) : x = 45 := by
  sorry

end angle_measure_l297_297230


namespace cost_price_of_apple_l297_297461

-- Define the given conditions SP = 20, and the relation between SP and CP.
variables (SP CP : ℝ)
axiom h1 : SP = 20
axiom h2 : SP = CP - (1/6) * CP

-- Statement to be proved.
theorem cost_price_of_apple : CP = 24 :=
by
  sorry

end cost_price_of_apple_l297_297461


namespace value_of_x_squared_minus_y_squared_l297_297537

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l297_297537


namespace teairra_shirts_l297_297426

theorem teairra_shirts (S : ℕ) (pants_total : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) (neither_plaid_nor_purple : ℕ)
  (pants_total_eq : pants_total = 24)
  (plaid_shirts_eq : plaid_shirts = 3)
  (purple_pants_eq : purple_pants = 5)
  (neither_plaid_nor_purple_eq : neither_plaid_nor_purple = 21) :
  (S - plaid_shirts + (pants_total - purple_pants) = neither_plaid_nor_purple) → S = 5 :=
by
  sorry

end teairra_shirts_l297_297426


namespace power_of_two_plus_one_is_power_of_integer_l297_297672

theorem power_of_two_plus_one_is_power_of_integer (n : ℕ) (hn : 0 < n) (a k : ℕ) (ha : 2^n + 1 = a^k) (hk : 1 < k) : n = 3 :=
by
  sorry

end power_of_two_plus_one_is_power_of_integer_l297_297672


namespace lightest_pumpkin_weight_l297_297585

theorem lightest_pumpkin_weight 
  (A B C : ℕ)
  (h₁ : A + B = 12)
  (h₂ : B + C = 15)
  (h₃ : A + C = 13) :
  A = 5 :=
by
  sorry

end lightest_pumpkin_weight_l297_297585


namespace point_P_in_third_quadrant_l297_297870

def point_in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

theorem point_P_in_third_quadrant :
  point_in_third_quadrant (-3) (-2) :=
by
  sorry -- Proof of the statement, as per the steps given.

end point_P_in_third_quadrant_l297_297870


namespace largest_three_digit_n_l297_297158

theorem largest_three_digit_n (n : ℕ) : 
  (70 * n ≡ 210 [MOD 350]) ∧ (n < 1000) → 
  n = 998 := by
  sorry

end largest_three_digit_n_l297_297158


namespace collin_savings_l297_297345

theorem collin_savings :
  let cans_home := 12
  let cans_grandparents := 3 * cans_home
  let cans_neighbor := 46
  let cans_dad := 250
  let total_cans := cans_home + cans_grandparents + cans_neighbor + cans_dad
  let money_per_can := 0.25
  let total_money := total_cans * money_per_can
  let savings := total_money / 2
  savings = 43 := 
  by 
  sorry

end collin_savings_l297_297345


namespace intersection_with_y_axis_l297_297579

theorem intersection_with_y_axis :
  ∀ (y : ℝ), (∃ x : ℝ, y = 2 * x + 2 ∧ x = 0) → y = 2 :=
by
  sorry

end intersection_with_y_axis_l297_297579


namespace sequence_divisibility_l297_297686

theorem sequence_divisibility (a b c : ℤ) (u v : ℕ → ℤ) (N : ℕ)
  (hu0 : u 0 = 1) (hu1 : u 1 = 1)
  (hu : ∀ n ≥ 2, u n = 2 * u (n - 1) - 3 * u (n - 2))
  (hv0 : v 0 = a) (hv1 : v 1 = b) (hv2 : v 2 = c)
  (hv : ∀ n ≥ 3, v n = v (n - 1) - 3 * v (n - 2) + 27 * v (n - 3))
  (hdiv : ∀ n ≥ N, u n ∣ v n) : 3 * a = 2 * b + c :=
by
  sorry

end sequence_divisibility_l297_297686


namespace lcm_and_sum_of_14_21_35_l297_297190

def lcm_of_numbers_and_sum (a b c : ℕ) : ℕ × ℕ :=
  (Nat.lcm (Nat.lcm a b) c, a + b + c)

theorem lcm_and_sum_of_14_21_35 :
  lcm_of_numbers_and_sum 14 21 35 = (210, 70) :=
  sorry

end lcm_and_sum_of_14_21_35_l297_297190


namespace california_more_license_plates_l297_297881

theorem california_more_license_plates :
  let CA_format := 26^4 * 10^2
  let NY_format := 26^3 * 10^3
  CA_format - NY_format = 28121600 := by
  let CA_format : Nat := 26^4 * 10^2
  let NY_format : Nat := 26^3 * 10^3
  have CA_plates : CA_format = 45697600 := by sorry
  have NY_plates : NY_format = 17576000 := by sorry
  calc
    CA_format - NY_format = 45697600 - 17576000 := by rw [CA_plates, NY_plates]
                    _ = 28121600 := by norm_num

end california_more_license_plates_l297_297881


namespace koschei_never_escapes_l297_297763

-- Define a structure for the initial setup
structure Setup where
  koschei_initial_room : Nat -- Initial room of Koschei
  guard_positions : List (Bool) -- Guards' positions, True for West, False for East

-- Example of the required setup:
def initial_setup : Setup :=
  { koschei_initial_room := 1, guard_positions := [true, false, true] }

-- Function to simulate the movement of guards
def move_guards (guards : List Bool) (room : Nat) : List Bool :=
  guards.map (λ g => not g)

-- Function to check if all guards are on the same wall
def all_guards_same_wall (guards : List Bool) : Bool :=
  List.all guards id ∨ List.all guards (λ g => ¬g)

-- Main statement: 
theorem koschei_never_escapes (setup : Setup) :
  ∀ room : Nat, ¬(all_guards_same_wall (move_guards setup.guard_positions room)) :=
  sorry

end koschei_never_escapes_l297_297763


namespace carrot_lettuce_ratio_l297_297114

theorem carrot_lettuce_ratio :
  let lettuce_cal := 50
  let dressing_cal := 210
  let crust_cal := 600
  let pepperoni_cal := crust_cal / 3
  let cheese_cal := 400
  let total_pizza_cal := crust_cal + pepperoni_cal + cheese_cal
  let carrot_cal := C
  let total_salad_cal := lettuce_cal + carrot_cal + dressing_cal
  let jackson_salad_cal := (1 / 4) * total_salad_cal
  let jackson_pizza_cal := (1 / 5) * total_pizza_cal
  jackson_salad_cal + jackson_pizza_cal = 330 →
  carrot_cal / lettuce_cal = 2 :=
by
  intro lettuce_cal dressing_cal crust_cal pepperoni_cal cheese_cal total_pizza_cal carrot_cal total_salad_cal jackson_salad_cal jackson_pizza_cal h
  sorry

end carrot_lettuce_ratio_l297_297114


namespace helicopter_rental_cost_l297_297292

noncomputable def rentCost (hours_per_day : ℕ) (days : ℕ) (cost_per_hour : ℕ) : ℕ :=
  hours_per_day * days * cost_per_hour

theorem helicopter_rental_cost :
  rentCost 2 3 75 = 450 := 
by
  sorry

end helicopter_rental_cost_l297_297292


namespace quadratic_inequality_solution_l297_297204

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, x^2 - (m - 4) * x - m + 7 > 0) ↔ m ∈ Set.Ioo (-2 : ℝ) 6 :=
by
  sorry

end quadratic_inequality_solution_l297_297204


namespace solution_interval_l297_297564

theorem solution_interval (x : ℝ) : 
  (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7) ∨ (7 < x) ↔ 
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0) := sorry

end solution_interval_l297_297564


namespace average_side_length_of_squares_l297_297428

theorem average_side_length_of_squares (a1 a2 a3 a4 : ℕ) 
(h1 : a1 = 36) (h2 : a2 = 64) (h3 : a3 = 100) (h4 : a4 = 144) :
(Real.sqrt a1 + Real.sqrt a2 + Real.sqrt a3 + Real.sqrt a4) / 4 = 9 := 
by
  sorry

end average_side_length_of_squares_l297_297428


namespace range_of_m_for_circle_l297_297105

theorem range_of_m_for_circle (m : ℝ) :
  (∃ x y, x^2 + y^2 - 4 * x - 2 * y + m = 0) → m < 5 :=
by
  sorry

end range_of_m_for_circle_l297_297105


namespace four_digit_numbers_starting_with_1_l297_297147

theorem four_digit_numbers_starting_with_1 
: ∃ n : ℕ, (n = 234) ∧ 
  (∀ (x y z : ℕ), 
    (x ≠ y → x ≠ z → y ≠ z → -- ensuring these constraints
    x ≠ 1 → y ≠ 1 → z = 1 → -- exactly two identical digits which include 1
    (x * 1000 + y * 100 + z * 10 + 1) / 1000 = 1 ∨ (x * 1000 + z * 100 + y * 10 + 1) / 1000 = 1) ∨ 
    (∃ (x y : ℕ),  
    (x ≠ y → x ≠ 1 → y = 1 → 
    (x * 110 + y * 10 + 1) + (x * 11 + y * 10 + 1) + (x * 100 + y * 10 + 1) + (x * 110 + 1) = n))) := sorry

end four_digit_numbers_starting_with_1_l297_297147


namespace emily_necklaces_for_friends_l297_297820

theorem emily_necklaces_for_friends (n b B : ℕ)
  (h1 : n = 26)
  (h2 : b = 2)
  (h3 : B = 52)
  (h4 : n * b = B) : 
  n = 26 :=
by
  sorry

end emily_necklaces_for_friends_l297_297820


namespace infinite_series_converges_l297_297645

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l297_297645


namespace infinite_series_converges_l297_297643

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l297_297643


namespace sum_of_ages_is_59_l297_297874

variable (juliet maggie ralph nicky lucy lily alex : ℕ)

def juliet_age := 10
def maggie_age := juliet_age - 3
def ralph_age := juliet_age + 2
def nicky_age := ralph_age / 2
def lucy_age := ralph_age + 1
def lily_age := ralph_age + 1
def alex_age := lucy_age - 5

theorem sum_of_ages_is_59 :
  maggie_age + ralph_age + nicky_age + lucy_age + lily_age + alex_age = 59 :=
by
  let maggie := 7
  let ralph := 12
  let nicky := 6
  let lucy := 13
  let lily := 13
  let alex := 8
  show maggie + ralph + nicky + lucy + lily + alex = 59
  sorry

end sum_of_ages_is_59_l297_297874


namespace carol_rectangle_length_l297_297193

theorem carol_rectangle_length :
  let j_length := 6
  let j_width := 30
  let c_width := 15
  let c_length := j_length * j_width / c_width
  c_length = 12 := by
  sorry

end carol_rectangle_length_l297_297193


namespace gift_cost_l297_297079

theorem gift_cost (C F : ℕ) (hF : F = 15) (h_eq : C / (F - 4) = C / F + 12) : C = 495 :=
by
  -- Using the conditions given, we need to show that C computes to 495.
  -- Details are skipped using sorry.
  sorry

end gift_cost_l297_297079


namespace calculate_area_of_triangle_PQR_l297_297872

structure Triangle (α β γ : Type) :=
(PQ : α)
(PR : α)
(angle_PQR : β)

noncomputable def triangle_area {α : Type} [LinearOrderedField α]
  {β : Type} [LinearOrderedSemiring β]
  [Algebra α β]
  (t : Triangle α β α)
  (angle_PQR_in_degrees : α) : β :=
  0.5 * t.PQ * t.PR * real.sin (π * angle_PQR_in_degrees / 180)

theorem calculate_area_of_triangle_PQR : ∀ (PQ PR : ℝ) (angle_PQR : ℝ),
  PQ = 30 → PR = 24 → angle_PQR = 60 → triangle_area {PQ := PQ, PR := PR, angle_PQR := angle_PQR} 60 = 180 * real.sqrt 3 :=
by
  intros PQ PR angle_PQR hPQ hPR hangle_PQR 
  rw [hPQ, hPR, hangle_PQR]
  unfold triangle_area
  rw [← div_mul_cancel ((30 : ℝ) * 24 * real.sin (π * (60 : ℝ) / 180)) (2 : ℝ)]
  rw [real.sin_pi_div_three]
  norm_num
  sorry

end calculate_area_of_triangle_PQR_l297_297872


namespace sam_grew_3_carrots_l297_297008

-- Let Sandy's carrots and the total number of carrots be defined
def sandy_carrots : ℕ := 6
def total_carrots : ℕ := 9

-- Define the number of carrots grown by Sam
def sam_carrots : ℕ := total_carrots - sandy_carrots

-- The theorem to prove
theorem sam_grew_3_carrots : sam_carrots = 3 := by
  sorry

end sam_grew_3_carrots_l297_297008


namespace fruit_salad_cost_3_l297_297151

def cost_per_fruit_salad (num_people sodas_per_person soda_cost sandwich_cost num_snacks snack_cost total_cost : ℕ) : ℕ :=
  let total_soda_cost := num_people * sodas_per_person * soda_cost
  let total_sandwich_cost := num_people * sandwich_cost
  let total_snack_cost := num_snacks * snack_cost
  let total_known_cost := total_soda_cost + total_sandwich_cost + total_snack_cost
  let total_fruit_salad_cost := total_cost - total_known_cost
  total_fruit_salad_cost / num_people

theorem fruit_salad_cost_3 :
  cost_per_fruit_salad 4 2 2 5 3 4 60 = 3 :=
by
  sorry

end fruit_salad_cost_3_l297_297151


namespace sqrt_fraction_arith_sqrt_16_l297_297757

-- Prove that the square root of 4/9 is ±2/3
theorem sqrt_fraction (a b : ℕ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) (h_a : a = 4) (h_b : b = 9) : 
    (Real.sqrt (a / (b : ℝ)) = abs (Real.sqrt a / Real.sqrt b)) :=
by
    rw [h_a, h_b]
    sorry

-- Prove that the arithmetic square root of √16 is 4.
theorem arith_sqrt_16 : Real.sqrt (Real.sqrt 16) = 4 :=
by
    sorry

end sqrt_fraction_arith_sqrt_16_l297_297757


namespace find_largest_number_l297_297910

noncomputable def largest_number (a b c : ℚ) : ℚ :=
  if a + b + c = 77 ∧ c - b = 9 ∧ b - a = 5 then c else 0

theorem find_largest_number (a b c : ℚ) 
  (h1 : a + b + c = 77) 
  (h2 : c - b = 9) 
  (h3 : b - a = 5) : 
  c = 100 / 3 := 
sorry

end find_largest_number_l297_297910


namespace find_f_2017_l297_297221

theorem find_f_2017 (f : ℕ → ℕ) (H1 : ∀ x y : ℕ, f (x * y + 1) = f x * f y - f y - x + 2) (H2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end find_f_2017_l297_297221


namespace least_weight_of_oranges_l297_297067

theorem least_weight_of_oranges :
  ∀ (a o : ℝ), (a ≥ 8 + 3 * o) → (a ≤ 4 * o) → (o ≥ 8) :=
by
  intros a o h1 h2
  sorry

end least_weight_of_oranges_l297_297067


namespace repayment_amount_formula_l297_297293

def loan_principal := 480000
def repayment_years := 20
def repayment_months := repayment_years * 12
def monthly_interest_rate := 0.004
def monthly_principal_repayment := loan_principal / repayment_months

def interest_for_nth_month (n : ℕ) : ℚ :=
  (loan_principal - (n - 1) * monthly_principal_repayment) * monthly_interest_rate

def repayment_amount_nth_month (n : ℕ) : ℚ :=
  monthly_principal_repayment + interest_for_nth_month n

theorem repayment_amount_formula (n : ℕ) (hn : 1 ≤ n ∧ n ≤ repayment_months) :
  repayment_amount_nth_month n = 3928 - 8 * n := by
sorry

end repayment_amount_formula_l297_297293


namespace geometric_series_first_term_l297_297606

theorem geometric_series_first_term (r a S : ℝ) (hr : r = 1 / 8) (hS : S = 60) (hS_formula : S = a / (1 - r)) : 
  a = 105 / 2 := by
  rw [hr, hS] at hS_formula
  sorry

end geometric_series_first_term_l297_297606


namespace sum_series_l297_297616

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l297_297616


namespace sum_series_l297_297619

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l297_297619


namespace compare_fractions_l297_297213

theorem compare_fractions (x y : ℝ) (n : ℕ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : 0 < n) :
  (x^n / (1 - x^2) + y^n / (1 - y^2)) ≥ ((x^n + y^n) / (1 - x * y)) :=
by sorry

end compare_fractions_l297_297213


namespace count_yellow_balls_l297_297934

theorem count_yellow_balls (total white green yellow red purple : ℕ) (prob : ℚ)
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 30)
  (h_red : red = 9)
  (h_purple : purple = 3)
  (h_prob : prob = 0.88) :
  yellow = 8 :=
by
  -- The proof will be here
  sorry

end count_yellow_balls_l297_297934


namespace value_of_x_squared_minus_y_squared_l297_297538

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l297_297538


namespace sqrt_sixteen_is_four_l297_297487

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l297_297487


namespace quadratic_min_value_correct_l297_297833

theorem quadratic_min_value_correct : ∀ x : ℝ, ∃ y_min : ℝ, y_min = 2 ∧ (∃ x0 : ℝ, y_min = (x0 - 1)^2 + 2) :=  
by {
  intro x,
  use 2,
  split,
  { reflexivity, },
  { use 1,
    reflexivity, }
}

end quadratic_min_value_correct_l297_297833


namespace product_of_three_numbers_is_correct_l297_297437

noncomputable def sum_three_numbers_product (x y z n : ℚ) : Prop :=
  x + y + z = 200 ∧
  8 * x = y - 12 ∧
  8 * x = z + 12 ∧
  (x * y * z = 502147200 / 4913)

theorem product_of_three_numbers_is_correct :
  ∃ (x y z n : ℚ), sum_three_numbers_product x y z n :=
by
  sorry

end product_of_three_numbers_is_correct_l297_297437


namespace percentage_of_boys_and_additional_boys_l297_297106

theorem percentage_of_boys_and_additional_boys (total_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ)
  (total_students_eq : total_students = 42) (ratio_condition : boys_ratio = 3 ∧ girls_ratio = 4) :
  let total_groups := total_students / (boys_ratio + girls_ratio)
  let total_boys := boys_ratio * total_groups
  (total_boys * 100 / total_students = 300 / 7) ∧ (21 - total_boys = 3) :=
by {
  sorry
}

end percentage_of_boys_and_additional_boys_l297_297106


namespace minimum_red_chips_l297_297053

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ (1 / 3) * w) (h2 : b ≤ (1 / 4) * r) (h3 : w + b ≥ 70) : r ≥ 72 := by
  sorry

end minimum_red_chips_l297_297053


namespace compute_roots_sum_l297_297117

def roots_quadratic_eq_a_b (a b : ℂ) : Prop :=
  a^2 - 6 * a + 8 = 0 ∧ b^2 - 6 * b + 8 = 0

theorem compute_roots_sum (a b : ℂ) (ha : roots_quadratic_eq_a_b a b) :
  a^5 + a^3 * b^3 + b^5 = -568 := by
  sorry

end compute_roots_sum_l297_297117


namespace div_pow_two_sub_one_l297_297877

theorem div_pow_two_sub_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  (3^k ∣ 2^n - 1) ↔ (∃ m : ℕ, n = 2 * 3^(k-1) * m) :=
by
  sorry

end div_pow_two_sub_one_l297_297877


namespace money_has_48_l297_297045

-- Definitions derived from conditions:
def money (p : ℝ) := 
  p = (1/3 * p) + 32

-- The main theorem statement
theorem money_has_48 (p : ℝ) : money p → p = 48 := by
  intro h
  -- Skipping the proof
  sorry

end money_has_48_l297_297045


namespace largest_c_range_3_l297_297666

theorem largest_c_range_3 (c : ℝ) : (∃ x : ℝ, x^2 - 7*x + c = 3) ↔ (c ≤ 61 / 4) :=
begin
  sorry
end

end largest_c_range_3_l297_297666


namespace tan_family_total_cost_l297_297568

-- Define the number of people in each age group and respective discounts
def num_children : ℕ := 2
def num_adults : ℕ := 2
def num_seniors : ℕ := 2

def price_adult_ticket : ℝ := 10
def discount_senior : ℝ := 0.30
def discount_child : ℝ := 0.20
def group_discount : ℝ := 0.10

-- Calculate the cost for each group with discounts applied
def price_senior_ticket := price_adult_ticket * (1 - discount_senior)
def price_child_ticket := price_adult_ticket * (1 - discount_child)

-- Calculate the total cost of tickets before group discount
def total_cost_before_group_discount :=
  (price_senior_ticket * num_seniors) +
  (price_child_ticket * num_children) +
  (price_adult_ticket * num_adults)

-- Check if the family qualifies for group discount and apply if necessary
def total_cost_after_group_discount :=
  if (num_children + num_adults + num_seniors > 5)
  then total_cost_before_group_discount * (1 - group_discount)
  else total_cost_before_group_discount

-- Main theorem statement
theorem tan_family_total_cost : total_cost_after_group_discount = 45 := by
  sorry

end tan_family_total_cost_l297_297568


namespace min_value_sum_reciprocal_l297_297728

open Real

theorem min_value_sum_reciprocal (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
    (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
    1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 3 / 4 :=
by
  sorry

end min_value_sum_reciprocal_l297_297728


namespace collin_savings_l297_297343

-- Define conditions
noncomputable def can_value : ℝ := 0.25
def cans_at_home : ℕ := 12
def cans_from_grandparents : ℕ := 3 * cans_at_home
def cans_from_neighbor : ℕ := 46
def cans_from_dad : ℕ := 250
def total_cans : ℕ := cans_at_home + cans_from_grandparents + cans_from_neighbor + cans_from_dad
def total_earnings : ℝ := can_value * total_cans
def amount_to_save : ℝ := total_earnings / 2

-- Theorem statement
theorem collin_savings : amount_to_save = 43 := 
by sorry

end collin_savings_l297_297343


namespace value_of_a_l297_297702

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 20 - 7 * a) : a = 20 / 11 := by
  sorry

end value_of_a_l297_297702


namespace tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l297_297848

-- Define the function f
noncomputable def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 3

-- First proof problem: Equation of the tangent line at (2, 7)
theorem tangent_line_at_2_is_12x_minus_y_minus_17_eq_0 :
  ∀ x y : ℝ, y = f x → (x = 2) → y = 7 → (∃ (m b : ℝ), (m = 12) ∧ (b = -17) ∧ (∀ x, 12 * x - y - 17 = 0)) :=
by
  sorry

-- Second proof problem: Range of m for three distinct real roots
theorem range_of_m_for_three_distinct_real_roots :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) → -3 < m ∧ m < -2 :=
by 
  sorry

end tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l297_297848


namespace circle_radius_of_diameter_l297_297168

theorem circle_radius_of_diameter (d : ℝ) (h : d = 22) : d / 2 = 11 :=
by
  sorry

end circle_radius_of_diameter_l297_297168


namespace exponent_equality_l297_297172

theorem exponent_equality (n : ℕ) : 
    5^n = 5 * (5^2)^2 * (5^3)^3 → n = 14 := by
    sorry

end exponent_equality_l297_297172


namespace find_second_number_l297_297321

theorem find_second_number (x : ℕ) :
  22030 = (555 + x) * 2 * (x - 555) + 30 → 
  x = 564 :=
by
  intro h
  sorry

end find_second_number_l297_297321


namespace tangent_line_y_intercept_l297_297142

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

theorem tangent_line_y_intercept (a : ℝ) (h : 3 * (1:ℝ)^2 - a = 1) :
  (∃ (m b : ℝ), ∀ (x : ℝ), m = 1 ∧ y = x - 2 → y = m * x + b) := 
 by
  sorry

end tangent_line_y_intercept_l297_297142


namespace A_finish_work_in_6_days_l297_297598

theorem A_finish_work_in_6_days :
  ∃ (x : ℕ), (1 / (12:ℚ) + 1 / (x:ℚ) = 1 / (4:ℚ)) → x = 6 :=
by
  sorry

end A_finish_work_in_6_days_l297_297598


namespace compare_negatives_l297_297493

theorem compare_negatives : -2 > -3 :=
by
  sorry

end compare_negatives_l297_297493


namespace sixth_root_24414062515625_l297_297202

theorem sixth_root_24414062515625 :
  (∃ (x : ℕ), x^6 = 24414062515625) → (sqrt 6 24414062515625 = 51) :=
by
  -- Applying the condition expressed as sum of binomials
  have h : 24414062515625 = ∑ k in finset.range 7, binom 6 k * (50 ^ (6 - k)),
  sorry
  
  -- Utilize this condition to find the sixth root
  sorry

end sixth_root_24414062515625_l297_297202


namespace ordering_of_powers_l297_297352

theorem ordering_of_powers : (3 ^ 17) < (8 ^ 9) ∧ (8 ^ 9) < (4 ^ 15) := 
by 
  -- We proved (3 ^ 17) < (8 ^ 9)
  have h1 : (3 ^ 17) < (8 ^ 9) := sorry
  
  -- We proved (8 ^ 9) < (4 ^ 15)
  have h2 : (8 ^ 9) < (4 ^ 15) := sorry

  -- Therefore, combining both
  exact ⟨h1, h2⟩

end ordering_of_powers_l297_297352


namespace arithmetic_problem_l297_297893

theorem arithmetic_problem : 
  let x := 512.52 
  let y := 256.26 
  let diff := x - y 
  let result := diff * 3 
  result = 768.78 := 
by 
  sorry

end arithmetic_problem_l297_297893


namespace find_xyz_sum_l297_297416

theorem find_xyz_sum (x y z : ℝ) (h1 : x^2 + x * y + y^2 = 108)
                               (h2 : y^2 + y * z + z^2 = 49)
                               (h3 : z^2 + z * x + x^2 = 157) :
  x * y + y * z + z * x = 84 :=
sorry

end find_xyz_sum_l297_297416


namespace ihsan_children_l297_297541

theorem ihsan_children :
  ∃ n : ℕ, (n + n^2 + n^3 + n^4 = 2800) ∧ (n = 7) :=
sorry

end ihsan_children_l297_297541


namespace g_18_equals_5832_l297_297120

noncomputable def g (n : ℕ) : ℕ := sorry

axiom cond1 : ∀ (n : ℕ), (0 < n) → g (n + 1) > g n
axiom cond2 : ∀ (m n : ℕ), (0 < m ∧ 0 < n) → g (m * n) = g m * g n
axiom cond3 : ∀ (m n : ℕ), (0 < m ∧ 0 < n ∧ m ≠ n ∧ m^2 = n^3) → (g m = n ∨ g n = m)

theorem g_18_equals_5832 : g 18 = 5832 :=
by sorry

end g_18_equals_5832_l297_297120


namespace sqrt_of_sixteen_l297_297479

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_sixteen_l297_297479


namespace cricket_run_target_l297_297718

theorem cricket_run_target
  (run_rate_1st_period : ℝ)
  (overs_1st_period : ℕ)
  (run_rate_2nd_period : ℝ)
  (overs_2nd_period : ℕ)
  (target_runs : ℝ)
  (h1 : run_rate_1st_period = 3.2)
  (h2 : overs_1st_period = 10)
  (h3 : run_rate_2nd_period = 5)
  (h4 : overs_2nd_period = 50) :
  target_runs = (run_rate_1st_period * overs_1st_period) + (run_rate_2nd_period * overs_2nd_period) :=
by
  sorry

end cricket_run_target_l297_297718


namespace min_value_f_on_interval_l297_297668

open Real

noncomputable def f (x : ℝ) : ℝ := tan x ^ 2 - 4 * tan x - 8 * cot x + 4 * cot x ^ 2 + 5

theorem min_value_f_on_interval :
  ∃ x ∈ Ioo (π / 2) π, ∀ y ∈ Ioo (π / 2) π, f y ≥ f x ∧ f x = 9 - 8 * sqrt 2 := 
by
  -- proof omitted
  sorry

end min_value_f_on_interval_l297_297668


namespace base8_subtraction_l297_297340

theorem base8_subtraction : (52 - 27 : ℕ) = 23 := by sorry

end base8_subtraction_l297_297340


namespace cell_phones_sold_l297_297350

theorem cell_phones_sold (init_samsung init_iphone final_samsung final_iphone defective_samsung defective_iphone : ℕ)
    (h1 : init_samsung = 14) 
    (h2 : init_iphone = 8) 
    (h3 : final_samsung = 10) 
    (h4 : final_iphone = 5) 
    (h5 : defective_samsung = 2) 
    (h6 : defective_iphone = 1) : 
    init_samsung - defective_samsung - final_samsung + 
    init_iphone - defective_iphone - final_iphone = 4 := 
by
  sorry

end cell_phones_sold_l297_297350


namespace find_A_minus_C_l297_297146

theorem find_A_minus_C (A B C : ℤ) 
  (h1 : A = B - 397)
  (h2 : A = 742)
  (h3 : B = C + 693) : 
  A - C = 296 :=
by
  sorry

end find_A_minus_C_l297_297146


namespace problem_1_problem_2_l297_297425

-- Definitions for the sets A and B
def A (x : ℝ) : Prop := -1 < x ∧ x < 2
def B (a : ℝ) (x : ℝ) : Prop := 2 * a - 1 < x ∧ x < 2 * a + 3

-- Problem 1: Range of values for a such that A ⊂ B
theorem problem_1 (a : ℝ) : (∀ x, A x → B a x) ↔ (-1/2 ≤ a ∧ a ≤ 0) := sorry

-- Problem 2: Range of values for a such that A ∩ B = ∅
theorem problem_2 (a : ℝ) : (∀ x, A x → ¬ B a x) ↔ (a ≤ -2 ∨ 3/2 ≤ a) := sorry

end problem_1_problem_2_l297_297425


namespace macy_miles_left_l297_297125

theorem macy_miles_left (goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) 
  (H1 : goal = 24) 
  (H2 : daily_miles = 3) 
  (H3 : days_run = 6) 
  : goal - daily_miles * days_run = 6 := 
by 
  sorry

end macy_miles_left_l297_297125


namespace cone_volume_l297_297679

theorem cone_volume (r l: ℝ) (h: ℝ) (hr : r = 1) (hl : l = 2) (hh : h = Real.sqrt (l^2 - r^2)) : 
  (1 / 3) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 :=
by 
  sorry

end cone_volume_l297_297679


namespace rosemary_leaves_count_l297_297814

-- Define the number of pots for each plant type
def basil_pots : ℕ := 3
def rosemary_pots : ℕ := 9
def thyme_pots : ℕ := 6

-- Define the number of leaves each plant type has
def basil_leaves : ℕ := 4
def thyme_leaves : ℕ := 30
def total_leaves : ℕ := 354

-- Prove that the number of leaves on each rosemary plant is 18
theorem rosemary_leaves_count (R : ℕ) (h : basil_pots * basil_leaves + rosemary_pots * R + thyme_pots * thyme_leaves = total_leaves) : R = 18 :=
by {
  -- Following steps are within the theorem's proof
  sorry
}

end rosemary_leaves_count_l297_297814


namespace right_triangle_hypotenuse_l297_297983

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l297_297983


namespace haley_marbles_l297_297862

theorem haley_marbles (boys : ℕ) (marbles_per_boy : ℕ) (h_boys : boys = 13) (h_marbles_per_boy : marbles_per_boy = 2) :
  boys * marbles_per_boy = 26 := 
by 
  sorry

end haley_marbles_l297_297862


namespace g_at_10_is_neg48_l297_297272

variable (g : ℝ → ℝ)

-- Given condition
axiom functional_eqn : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2

-- Mathematical proof statement
theorem g_at_10_is_neg48 : g 10 = -48 :=
  sorry

end g_at_10_is_neg48_l297_297272


namespace triangle_area_l297_297157

theorem triangle_area (a b c : ℝ) (ha : a = 6) (hb : b = 5) (hc : c = 5) (isosceles : a = 2 * b) :
  let s := (a + b + c) / 2
  let area := (s * (s - a) * (s - b) * (s - c)).sqrt
  area = 12 :=
by
  sorry

end triangle_area_l297_297157


namespace midpoint_of_segment_l297_297403

def z1 : ℂ := 2 + 4 * Complex.I  -- Define the first endpoint
def z2 : ℂ := -6 + 10 * Complex.I  -- Define the second endpoint

theorem midpoint_of_segment :
  (z1 + z2) / 2 = -2 + 7 * Complex.I := by
  sorry

end midpoint_of_segment_l297_297403


namespace polygon_vertices_product_at_least_2014_l297_297654

theorem polygon_vertices_product_at_least_2014 :
  ∀ (vertices : Fin 90 → ℕ), 
    (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 90) → 
    ∃ i, (vertices i) * (vertices ((i + 1) % 90)) ≥ 2014 :=
sorry

end polygon_vertices_product_at_least_2014_l297_297654


namespace factorize_expression_l297_297503

theorem factorize_expression (m n : ℝ) :
  2 * m^3 * n - 32 * m * n = 2 * m * n * (m + 4) * (m - 4) :=
by
  sorry

end factorize_expression_l297_297503


namespace integer_solution_l297_297662

theorem integer_solution (n : ℤ) (hneq : n ≠ -2) :
  ∃ (m : ℤ), (n^3 + 8) = m * (n^2 - 4) ↔ n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end integer_solution_l297_297662


namespace sum_infinite_partial_fraction_l297_297638

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l297_297638


namespace probability_same_number_l297_297996

theorem probability_same_number :
  let n_billy := 500 / 30
  let n_bobbi := 500 / 45
  let n_common := 500 / Nat.lcm 30 45
  (n_common : ℚ) / (n_billy * n_bobbi) = 5 / 176 :=
by
  sorry

end probability_same_number_l297_297996


namespace train_passes_jogger_in_46_seconds_l297_297786

-- Definitions directly from conditions
def jogger_speed_kmh : ℕ := 10
def train_speed_kmh : ℕ := 46
def initial_distance_m : ℕ := 340
def train_length_m : ℕ := 120

-- Additional computed definitions based on conditions
def relative_speed_ms : ℕ := (train_speed_kmh - jogger_speed_kmh) * 1000 / 3600
def total_distance_m : ℕ := initial_distance_m + train_length_m

-- Prove that the time it takes for the train to pass the jogger is 46 seconds
theorem train_passes_jogger_in_46_seconds : total_distance_m / relative_speed_ms = 46 := by
  sorry

end train_passes_jogger_in_46_seconds_l297_297786


namespace probability_picking_pair_l297_297584

theorem probability_picking_pair : 
  let left_shoes := {A1, A2, A3} in
  let right_shoes := {B1, B2, B3} in
  let pairs := [(A1, B1), (A1, B2), (A1, B3), (A2, B1), (A2, B2), (A2, B3), (A3, B1), (A3, B2), (A3, B3)] in
  let desired_pairs := [(A1, B1), (A2, B2), (A3, B3)] in
  (desired_pairs.length / pairs.length) = (1 / 3) := 
by
  sorry

end probability_picking_pair_l297_297584


namespace expected_score_of_basketball_player_l297_297453

theorem expected_score_of_basketball_player :
  let p_inside : ℝ := 0.7
  let p_outside : ℝ := 0.4
  let attempts_inside : ℕ := 10
  let attempts_outside : ℕ := 5
  let points_inside : ℕ := 2
  let points_outside : ℕ := 3
  let E_inside : ℝ := attempts_inside * p_inside * points_inside
  let E_outside : ℝ := attempts_outside * p_outside * points_outside
  E_inside + E_outside = 20 :=
by
  sorry

end expected_score_of_basketball_player_l297_297453


namespace necessary_but_not_sufficient_condition_l297_297713

def represents_ellipse (k : ℝ) (x y : ℝ) :=
    1 < k ∧ k < 5 ∧ k ≠ 3

theorem necessary_but_not_sufficient_condition (k : ℝ) (x y : ℝ):
    (1 < k ∧ k < 5) → (represents_ellipse k x y) :=
by
  sorry

end necessary_but_not_sufficient_condition_l297_297713


namespace find_c_l297_297852

open Real

def vector := (ℝ × ℝ)

def a : vector := (1, 2)
def b : vector := (2, -3)

def is_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_c (c : vector) : 
  (is_parallel (c.1 + a.1, c.2 + a.2) b) ∧ (is_perpendicular c (a.1 + b.1, a.2 + b.2)) → 
  c = (-7 / 9, -20 / 9) := 
by
  sorry

end find_c_l297_297852


namespace math_problem_l297_297856

noncomputable def proof_problem (n : ℝ) (A B : ℝ) : Prop :=
  A = n^2 ∧ B = n^2 + 1 ∧ (1 * n^4 + 2 * n^2 + 3 + 2 * (n^2 + 1) + 1 = 5 * (2 * n^2 + 1)) → 
  A + B = 7 + 4 * Real.sqrt 2

theorem math_problem (n : ℝ) (A B : ℝ) :
  proof_problem n A B :=
sorry

end math_problem_l297_297856


namespace cost_of_advanced_purchase_ticket_l297_297331

theorem cost_of_advanced_purchase_ticket
  (x : ℝ)
  (door_cost : ℝ := 14)
  (total_tickets : ℕ := 140)
  (total_money : ℝ := 1720)
  (advanced_tickets_sold : ℕ := 100)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold)
  (advanced_revenue : ℝ := advanced_tickets_sold * x)
  (door_revenue : ℝ := door_tickets_sold * door_cost)
  (total_revenue : ℝ := advanced_revenue + door_revenue) :
  total_revenue = total_money → x = 11.60 :=
by
  intro h
  sorry

end cost_of_advanced_purchase_ticket_l297_297331


namespace solve_for_x_over_z_l297_297846

variables (x y z : ℝ)

theorem solve_for_x_over_z
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y = 6 * z) :
  x / z = 5 :=
sorry

end solve_for_x_over_z_l297_297846


namespace range_of_a_if_q_sufficient_but_not_necessary_for_p_l297_297088

variable {x a : ℝ}

def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

theorem range_of_a_if_q_sufficient_but_not_necessary_for_p :
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a) → a ∈ Set.Ici 1 := 
sorry

end range_of_a_if_q_sufficient_but_not_necessary_for_p_l297_297088


namespace num_spacy_subsets_15_l297_297658

def spacy_subsets (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | 3     => 4
  | n + 1 => spacy_subsets n + if n ≥ 2 then spacy_subsets (n - 2) else 1

theorem num_spacy_subsets_15 : spacy_subsets 15 = 406 := by
  sorry

end num_spacy_subsets_15_l297_297658


namespace jill_third_month_days_l297_297550

theorem jill_third_month_days :
  ∀ (days : ℕ),
    (earnings_first_month : ℕ) = 10 * 30 →
    (earnings_second_month : ℕ) = 20 * 30 →
    (total_earnings : ℕ) = 1200 →
    (total_earnings_two_months : ℕ) = earnings_first_month + earnings_second_month →
    (earnings_third_month : ℕ) = total_earnings - total_earnings_two_months →
    earnings_third_month = 300 →
    days = earnings_third_month / 20 →
    days = 15 := 
sorry

end jill_third_month_days_l297_297550


namespace solve_fractional_equation_l297_297743

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 2) : 
  (4 * x ^ 2 + 3 * x + 2) / (x - 2) = 4 * x + 5 ↔ x = -2 := by 
  sorry

end solve_fractional_equation_l297_297743


namespace breadth_of_landscape_l297_297791

noncomputable def landscape_breadth (L : ℕ) (playground_area : ℕ) (total_area : ℕ) (B : ℕ) : Prop :=
  B = 6 * L ∧ playground_area = 4200 ∧ playground_area = (1 / 7) * total_area ∧ total_area = L * B

theorem breadth_of_landscape : ∃ (B : ℕ), ∀ (L : ℕ), landscape_breadth L 4200 29400 B → B = 420 :=
by
  intros
  sorry

end breadth_of_landscape_l297_297791


namespace totalPawnsLeft_l297_297423

def sophiaInitialPawns := 8
def chloeInitialPawns := 8
def sophiaLostPawns := 5
def chloeLostPawns := 1

theorem totalPawnsLeft : (sophiaInitialPawns - sophiaLostPawns) + (chloeInitialPawns - chloeLostPawns) = 10 := by
  sorry

end totalPawnsLeft_l297_297423


namespace solve_eq_l297_297891

theorem solve_eq {x y z : ℕ} :
  2^x + 3^y - 7 = z! ↔ (x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry -- Proof should be provided here

end solve_eq_l297_297891


namespace expression_value_l297_297280

theorem expression_value (a b c : ℚ) (h₁ : b = 8) (h₂ : c = 5) (h₃ : a * b * c = 2 * (a + b + c) + 14) : 
  (c - a) ^ 2 + b = 8513 / 361 := by 
  sorry

end expression_value_l297_297280


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297314

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d > 1 ∧ d < n

def consecutiveComposites (start n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < n → isComposite (start + i)

theorem exists_nine_consecutive_composites :
  ∃ start, start + 8 ≤ 500 ∧ consecutiveComposites start 9 :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ start, start + 10 ≤ 500 ∧ consecutiveComposites start 11 :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297314


namespace speed_in_still_water_l297_297787

-- Define the velocities (speeds)
def speed_downstream (V_w V_s : ℝ) : ℝ := V_w + V_s
def speed_upstream (V_w V_s : ℝ) : ℝ := V_w - V_s

-- Define the given conditions
def downstream_condition (V_w V_s : ℝ) : Prop := speed_downstream V_w V_s = 9
def upstream_condition (V_w V_s : ℝ) : Prop := speed_upstream V_w V_s = 1

-- The main theorem to prove
theorem speed_in_still_water (V_s V_w : ℝ) (h1 : downstream_condition V_w V_s) (h2 : upstream_condition V_w V_s) : V_w = 5 :=
  sorry

end speed_in_still_water_l297_297787


namespace triangle_perimeter_l297_297699

theorem triangle_perimeter (x : ℕ) 
  (h1 : x % 2 = 1) 
  (h2 : 7 - 2 < x)
  (h3 : x < 2 + 7) :
  2 + 7 + x = 16 := 
sorry

end triangle_perimeter_l297_297699


namespace div_seven_and_sum_factors_l297_297738

theorem div_seven_and_sum_factors (a b c : ℤ) (h : (a = 0 ∨ b = 0 ∨ c = 0) ∧ ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  ∃ k : ℤ, (a + b + c)^7 - a^7 - b^7 - c^7 = k * 7 * (a + b) * (b + c) * (c + a) :=
by
  sorry

end div_seven_and_sum_factors_l297_297738


namespace math_problem_l297_297326

noncomputable def problem_statement : Prop := (7^2 - 5^2)^4 = 331776

theorem math_problem : problem_statement := by
  sorry

end math_problem_l297_297326


namespace distance_from_reflected_point_l297_297737

theorem distance_from_reflected_point
  (P : ℝ × ℝ) (P' : ℝ × ℝ)
  (hP : P = (3, 2))
  (hP' : P' = (3, -2))
  : dist P P' = 4 := sorry

end distance_from_reflected_point_l297_297737


namespace baskets_containing_neither_l297_297255

-- Definitions representing the conditions
def total_baskets : ℕ := 15
def baskets_with_apples : ℕ := 10
def baskets_with_oranges : ℕ := 8
def baskets_with_both : ℕ := 5

-- Theorem statement to prove the number of baskets containing neither apples nor oranges
theorem baskets_containing_neither : total_baskets - (baskets_with_apples + baskets_with_oranges - baskets_with_both) = 2 :=
by
  sorry

end baskets_containing_neither_l297_297255


namespace largest_of_numbers_l297_297604

theorem largest_of_numbers (a b c d : ℝ) (hₐ : a = 0) (h_b : b = -1) (h_c : c = -2) (h_d : d = Real.sqrt 3) :
  d = Real.sqrt 3 ∧ d > a ∧ d > b ∧ d > c :=
by
  -- Using sorry to skip the proof
  sorry

end largest_of_numbers_l297_297604


namespace estimate_pi_simulation_l297_297236

theorem estimate_pi_simulation :
  let side := 2
  let radius := 1
  let total_seeds := 1000
  let seeds_in_circle := 778
  (π : ℝ) * radius^2 / side^2 = (seeds_in_circle : ℝ) / total_seeds → π = 3.112 :=
by
  intros
  sorry

end estimate_pi_simulation_l297_297236


namespace sufficient_but_not_necessary_condition_l297_297016

theorem sufficient_but_not_necessary_condition (x : ℝ) : (0 < x ∧ x < 5) → |x - 2| < 3 :=
by
  sorry

end sufficient_but_not_necessary_condition_l297_297016


namespace minimum_value_ineq_l297_297727

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ (3 / 4) := sorry

end minimum_value_ineq_l297_297727


namespace spheres_do_not_protrude_l297_297939

-- Define the basic parameters
variables (R r : ℝ) (h_cylinder : ℝ) (h_cone : ℝ)
-- Assume conditions
axiom cylinder_height_diameter : h_cylinder = 2 * R
axiom cone_dimensions : h_cone = h_cylinder ∧ h_cone = R

-- The given radius relationship
axiom radius_relation : R = 3 * r

-- Prove the spheres do not protrude from the container
theorem spheres_do_not_protrude (R r h_cylinder h_cone : ℝ)
  (cylinder_height_diameter : h_cylinder = 2 * R)
  (cone_dimensions : h_cone = h_cylinder ∧ h_cone = R)
  (radius_relation : R = 3 * r) : r ≤ R / 2 :=
sorry

end spheres_do_not_protrude_l297_297939


namespace value_of_x_squared_minus_y_squared_l297_297536

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l297_297536


namespace sum_infinite_partial_fraction_l297_297634

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l297_297634


namespace axis_of_symmetry_l297_297104

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = f (4 - x)) : ∀ y, f 2 = y ↔ f 2 = y := 
by
  sorry

end axis_of_symmetry_l297_297104


namespace jasmine_percentage_after_adding_l297_297304

def initial_solution_volume : ℕ := 80
def initial_jasmine_percentage : ℝ := 0.10
def additional_jasmine_volume : ℕ := 5
def additional_water_volume : ℕ := 15

theorem jasmine_percentage_after_adding :
  let initial_jasmine_volume := initial_jasmine_percentage * initial_solution_volume
  let total_jasmine_volume := initial_jasmine_volume + additional_jasmine_volume
  let total_solution_volume := initial_solution_volume + additional_jasmine_volume + additional_water_volume
  let final_jasmine_percentage := (total_jasmine_volume / total_solution_volume) * 100
  final_jasmine_percentage = 13 := by
  sorry

end jasmine_percentage_after_adding_l297_297304


namespace neg_prop1_true_neg_prop2_false_l297_297302

-- Proposition 1: The logarithm of a positive number is always positive
def prop1 : Prop := ∀ x : ℝ, x > 0 → Real.log x > 0

-- Negation of Proposition 1: There exists a positive number whose logarithm is not positive
def neg_prop1 : Prop := ∃ x : ℝ, x > 0 ∧ Real.log x ≤ 0

-- Proposition 2: For all x in the set of integers Z, the last digit of x^2 is not 3
def prop2 : Prop := ∀ x : ℤ, (x * x % 10 ≠ 3)

-- Negation of Proposition 2: There exists an x in the set of integers Z such that the last digit of x^2 is 3
def neg_prop2 : Prop := ∃ x : ℤ, (x * x % 10 = 3)

-- Proof that the negation of Proposition 1 is true
theorem neg_prop1_true : neg_prop1 := 
  by sorry

-- Proof that the negation of Proposition 2 is false
theorem neg_prop2_false : ¬ neg_prop2 := 
  by sorry

end neg_prop1_true_neg_prop2_false_l297_297302


namespace series_sum_l297_297614

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l297_297614


namespace tangent_line_y_intercept_l297_297143

-- First, we define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

-- Define the derivative of the function f
def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - a

-- Define the slope of the line x - y - 1 = 0
def line_slope : ℝ := 1

-- Given condition: the tangent line at x = 1 is parallel to the line, thus it has a slope of 1.
-- Solve for a such that f_prime 1 a = 1
def solve_a : ℝ := 3 - 1

-- Recalculate specific version of f with a = 2
def f_specific (x : ℝ) : ℝ := f x 2

-- Define the tangent line at x = 1 and determine its y-intercept
noncomputable def tangent_line_intercept (x₀ : ℝ) (m : ℝ) (y₀ : ℝ) : ℝ := y₀ - m * x₀

-- Prove the intercept is -2
theorem tangent_line_y_intercept : tangent_line_intercept 1 1 (f_specific 1) = -2 := 
by 
  have a_eq_2 : solve_a = 2 := (by linarith)
  rw [solve_a, a_eq_2]
  sorry -- Placeholder for the complete proof.

end tangent_line_y_intercept_l297_297143


namespace joey_average_speed_l297_297409

noncomputable def average_speed_of_round_trip (distance_out : ℝ) (time_out : ℝ) (speed_return : ℝ) : ℝ :=
  let distance_return := distance_out
  let total_distance := distance_out + distance_return
  let time_return := distance_return / speed_return
  let total_time := time_out + time_return
  total_distance / total_time

theorem joey_average_speed :
  average_speed_of_round_trip 2 1 6.000000000000002 = 3 := by
  sorry

end joey_average_speed_l297_297409


namespace dave_deleted_apps_l297_297817

-- Definitions based on problem conditions
def original_apps : Nat := 16
def remaining_apps : Nat := 5

-- Theorem statement for proving how many apps Dave deleted
theorem dave_deleted_apps : original_apps - remaining_apps = 11 :=
by
  sorry

end dave_deleted_apps_l297_297817


namespace purple_chips_selected_is_one_l297_297542

noncomputable def chips_selected (B G P R x : ℕ) : Prop :=
  (1^B) * (5^G) * (x^P) * (11^R) = 140800 ∧ 5 < x ∧ x < 11

theorem purple_chips_selected_is_one :
  ∃ B G P R x, chips_selected B G P R x ∧ P = 1 :=
by {
  sorry
}

end purple_chips_selected_is_one_l297_297542


namespace find_N_l297_297812

variable (N : ℚ)
variable (p : ℚ)

def ball_probability_same_color 
  (green1 : ℚ) (total1 : ℚ) 
  (green2 : ℚ) (blue2 : ℚ) 
  (p : ℚ) : Prop :=
  (green1/total1) * (green2 / (green2 + blue2)) + 
  ((total1 - green1) / total1) * (blue2 / (green2 + blue2)) = p

theorem find_N :
  p = 0.65 → 
  ball_probability_same_color 5 12 20 N p → 
  N = 280 / 311 := 
by
  sorry

end find_N_l297_297812


namespace spent_more_on_candy_bar_l297_297348

-- Definitions of conditions
def money_Dan_has : ℕ := 2
def candy_bar_cost : ℕ := 6
def chocolate_cost : ℕ := 3

-- Statement of the proof problem
theorem spent_more_on_candy_bar : candy_bar_cost - chocolate_cost = 3 := by
  sorry

end spent_more_on_candy_bar_l297_297348


namespace molecular_weight_is_correct_l297_297295

noncomputable def molecular_weight_of_compound : ℝ :=
  3 * 39.10 + 2 * 51.996 + 7 * 15.999 + 4 * 1.008 + 1 * 14.007

theorem molecular_weight_is_correct : molecular_weight_of_compound = 351.324 := 
by
  sorry

end molecular_weight_is_correct_l297_297295


namespace number_of_consecutive_sum_sets_eq_18_l297_297694

theorem number_of_consecutive_sum_sets_eq_18 :
  ∃! (S : ℕ → ℕ) (n a : ℕ), (n ≥ 2) ∧ (S n = (n * (2 * a + n - 1)) / 2) ∧ (S n = 18) :=
sorry

end number_of_consecutive_sum_sets_eq_18_l297_297694


namespace sqrt_of_sixteen_l297_297478

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_sixteen_l297_297478


namespace joan_travel_time_correct_l297_297407

noncomputable def joan_travel_time (distance rate : ℕ) (lunch_break bathroom_breaks : ℕ) : ℕ := 
  let driving_time := distance / rate
  let break_time := lunch_break + 2 * bathroom_breaks
  driving_time + break_time / 60

theorem joan_travel_time_correct : joan_travel_time 480 60 30 15 = 9 := by
  sorry

end joan_travel_time_correct_l297_297407


namespace unit_prices_max_toys_l297_297054

-- For question 1
theorem unit_prices (x y : ℕ)
  (h₁ : y = x + 25)
  (h₂ : 2*y + x = 200) : x = 50 ∧ y = 75 :=
by {
  sorry
}

-- For question 2
theorem max_toys (cost_a cost_b q_a q_b : ℕ)
  (h₁ : cost_a = 50)
  (h₂ : cost_b = 75)
  (h₃ : q_b = 2 * q_a)
  (h₄ : 50 * q_a + 75 * q_b ≤ 20000) : q_a ≤ 100 :=
by {
  sorry
}

end unit_prices_max_toys_l297_297054


namespace units_digit_5_pow_17_mul_4_l297_297785

theorem units_digit_5_pow_17_mul_4 : ((5 ^ 17) * 4) % 10 = 0 :=
by
  sorry

end units_digit_5_pow_17_mul_4_l297_297785


namespace suggested_bacon_students_l297_297742

-- Definitions based on the given conditions
def students_mashed_potatoes : ℕ := 330
def students_tomatoes : ℕ := 76
def difference_bacon_mashed_potatoes : ℕ := 61

-- Lean 4 statement to prove the correct answer
theorem suggested_bacon_students : ∃ (B : ℕ), students_mashed_potatoes = B + difference_bacon_mashed_potatoes ∧ B = 269 := 
by
  sorry

end suggested_bacon_students_l297_297742


namespace infinite_series_converges_l297_297644

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l297_297644


namespace number_of_even_three_digit_numbers_l297_297009

theorem number_of_even_three_digit_numbers : 
  ∃ (count : ℕ), 
  count = 12 ∧ 
  (∀ (d1 d2 : ℕ), (0 ≤ d1 ∧ d1 ≤ 4) ∧ (Even d1) ∧ (0 ≤ d2 ∧ d2 ≤ 4) ∧ (Even d2) ∧ d1 ≠ d2 →
   ∃ (d3 : ℕ), (d3 = 1 ∨ d3 = 3) ∧ 
   ∃ (units tens hundreds : ℕ), 
     (units ∈ [0, 2, 4]) ∧ 
     (tens ∈ [0, 2, 4]) ∧ 
     (hundreds ∈ [1, 3]) ∧ 
     (units ≠ tens) ∧ 
     (units ≠ hundreds) ∧ 
     (tens ≠ hundreds) ∧ 
     ((units + tens * 10 + hundreds * 100) % 2 = 0) ∧ 
     count = 12) :=
sorry

end number_of_even_three_digit_numbers_l297_297009


namespace zed_to_wyes_l297_297401

theorem zed_to_wyes (value_ex: ℝ) (value_wye: ℝ) (value_zed: ℝ)
  (h1: 2 * value_ex = 29 * value_wye)
  (h2: value_zed = 16 * value_ex) : value_zed = 232 * value_wye := by
  sorry

end zed_to_wyes_l297_297401


namespace hypotenuse_right_triangle_l297_297989

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l297_297989


namespace sum_of_fractions_l297_297473

-- Definitions (Conditions)
def frac1 : ℚ := 5 / 13
def frac2 : ℚ := 9 / 11

-- Theorem (Equivalent Proof Problem)
theorem sum_of_fractions : frac1 + frac2 = 172 / 143 := 
by
  -- Proof skipped
  sorry

end sum_of_fractions_l297_297473


namespace total_call_charges_l297_297609

-- Definitions based on conditions
def base_fee : ℝ := 39
def included_minutes : ℕ := 300
def excess_charge_per_minute : ℝ := 0.19

-- Given variables
variable (x : ℕ) -- excess minutes
variable (y : ℝ) -- total call charges

-- Theorem stating the relationship between y and x
theorem total_call_charges (h : x > 0) : y = 0.19 * x + 39 := 
by sorry

end total_call_charges_l297_297609


namespace exist_2022_good_numbers_with_good_sum_l297_297553

def is_good (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1)

theorem exist_2022_good_numbers_with_good_sum :
  ∃ (a : Fin 2022 → ℕ), (∀ i j : Fin 2022, i ≠ j → a i ≠ a j) ∧ (∀ i : Fin 2022, is_good (a i)) ∧ is_good (Finset.univ.sum a) :=
sorry

end exist_2022_good_numbers_with_good_sum_l297_297553


namespace count_two_digit_integers_l297_297390

def two_digit_integers_satisfying_condition : Nat :=
  let candidates := [(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]
  candidates.length

theorem count_two_digit_integers :
  two_digit_integers_satisfying_condition = 8 :=
by
  sorry

end count_two_digit_integers_l297_297390


namespace greatest_possible_selling_price_l297_297167

variable (products : ℕ)
variable (average_price : ℝ)
variable (min_price : ℝ)
variable (less_than_1000_products : ℕ)

theorem greatest_possible_selling_price
  (h1 : products = 20)
  (h2 : average_price = 1200)
  (h3 : min_price = 400)
  (h4 : less_than_1000_products = 10) :
  ∃ max_price, max_price = 11000 := 
by
  sorry

end greatest_possible_selling_price_l297_297167


namespace solve_equation_l297_297664

open Real

noncomputable def verify_solution (x : ℝ) : Prop :=
  1 / ((x - 3) * (x - 4)) +
  1 / ((x - 4) * (x - 5)) +
  1 / ((x - 5) * (x - 6)) = 1 / 8

theorem solve_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6) :
  verify_solution x ↔ (x = (9 + sqrt 57) / 2 ∨ x = (9 - sqrt 57) / 2) := 
by
  sorry

end solve_equation_l297_297664


namespace problem_proof_l297_297750

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem problem_proof :
  (∀ x, g (x + Real.pi) = g x) ∧ (∀ y, g (2 * (Real.pi / 12) - y) = g y) :=
by
  sorry

end problem_proof_l297_297750


namespace candy_store_spending_l297_297386

variable (weekly_allowance : ℝ) (arcade_fraction : ℝ) (toy_store_fraction : ℝ)

def remaining_after_arcade (weekly_allowance arcade_fraction : ℝ) : ℝ :=
  weekly_allowance * (1 - arcade_fraction)

def remaining_after_toy_store (remaining_allowance toy_store_fraction : ℝ) : ℝ :=
  remaining_allowance * (1 - toy_store_fraction)

theorem candy_store_spending
  (h1 : weekly_allowance = 3.30)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : toy_store_fraction = 1 / 3) :
  remaining_after_toy_store (remaining_after_arcade weekly_allowance arcade_fraction) toy_store_fraction = 0.88 := 
sorry

end candy_store_spending_l297_297386


namespace donut_ate_even_neighbors_l297_297258

def cube neighbors (n : ℕ) : ℕ := sorry

theorem donut_ate_even_neighbors : 
  (cube neighbors 5) = 63 := 
by
  sorry

end donut_ate_even_neighbors_l297_297258


namespace number_of_consecutive_sum_sets_eq_18_l297_297693

theorem number_of_consecutive_sum_sets_eq_18 :
  ∃! (S : ℕ → ℕ) (n a : ℕ), (n ≥ 2) ∧ (S n = (n * (2 * a + n - 1)) / 2) ∧ (S n = 18) :=
sorry

end number_of_consecutive_sum_sets_eq_18_l297_297693


namespace sum_infinite_series_eq_l297_297630

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l297_297630


namespace total_fruit_count_l297_297285

-- Define the number of oranges
def oranges : ℕ := 6

-- Define the number of apples based on the number of oranges
def apples : ℕ := oranges - 2

-- Define the number of bananas based on the number of apples
def bananas : ℕ := 3 * apples

-- Define the number of peaches based on the number of bananas
def peaches : ℕ := bananas / 2

-- Define the total number of fruits in the basket
def total_fruits : ℕ := oranges + apples + bananas + peaches

-- Prove that the total number of pieces of fruit in the basket is 28
theorem total_fruit_count : total_fruits = 28 := by
  sorry

end total_fruit_count_l297_297285


namespace symmetric_point_to_origin_l297_297572

theorem symmetric_point_to_origin (a b : ℝ) :
  (∃ (a b : ℝ), (a / 2) - 2 * (b / 2) + 2 = 0 ∧ (b / a) * (1 / 2) = -1) →
  (a = -4 / 5 ∧ b = 8 / 5) :=
sorry

end symmetric_point_to_origin_l297_297572


namespace sqrt_sixteen_equals_four_l297_297482

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l297_297482


namespace no_two_right_angles_in_triangle_l297_297132

theorem no_two_right_angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 90) (h3 : B = 90): false :=
by
  -- we assume A = 90 and B = 90,
  -- then A + B + C > 180, which contradicts h1,
  sorry
  
example : (3 = 3) := by sorry  -- Given the context of the multiple-choice problem.

end no_two_right_angles_in_triangle_l297_297132


namespace largest_c_3_in_range_l297_297665

theorem largest_c_3_in_range (c : ℝ) : 
  (∃ x : ℝ, x^2 - 7*x + c = 3) ↔ c ≤ 61 / 4 := 
by sorry

end largest_c_3_in_range_l297_297665


namespace convert_to_rectangular_form_l297_297198

noncomputable def θ : ℝ := 15 * Real.pi / 2

noncomputable def EulerFormula (θ : ℝ) : ℂ := Complex.exp (Complex.I * θ)

theorem convert_to_rectangular_form : EulerFormula θ = Complex.I := by
  sorry

end convert_to_rectangular_form_l297_297198


namespace largest_divisor_of_expression_l297_297367

theorem largest_divisor_of_expression 
  (x : ℤ) (h_odd : x % 2 = 1) :
  384 ∣ (8*x + 4) * (8*x + 8) * (4*x + 2) :=
sorry

end largest_divisor_of_expression_l297_297367


namespace prove_incorrect_statement_l297_297593

-- Definitions based on given conditions
def isIrrational (x : ℝ) : Prop := ¬ ∃ a b : ℚ, x = a / b ∧ b ≠ 0
def isSquareRoot (x y : ℝ) : Prop := x * x = y
def hasSquareRoot (x : ℝ) : Prop := ∃ y : ℝ, isSquareRoot y x

-- Options translated into Lean
def optionA : Prop := ∀ x : ℝ, isIrrational x → ¬ hasSquareRoot x
def optionB (x : ℝ) : Prop := 0 < x → ∃ y : ℝ, y * y = x ∧ (-y) * (-y) = x
def optionC : Prop := isSquareRoot 0 0
def optionD (a : ℝ) : Prop := ∀ x : ℝ, x = -a → (x ^ 3 = - (a ^ 3))

-- The incorrect statement according to the solution
def incorrectStatement : Prop := optionA

-- The theorem to be proven
theorem prove_incorrect_statement : incorrectStatement :=
by
  -- Replace with the actual proof, currently a placeholder using sorry
  sorry

end prove_incorrect_statement_l297_297593


namespace socks_combination_correct_l297_297563

noncomputable def socks_combination : ℕ :=
nat.choose 6 4

theorem socks_combination_correct : socks_combination = 15 :=
by
  sorry

end socks_combination_correct_l297_297563


namespace prob_simultaneous_sequences_l297_297154

-- Definitions for coin probabilities
def prob_heads_A : ℝ := 0.3
def prob_tails_A : ℝ := 0.7
def prob_heads_B : ℝ := 0.4
def prob_tails_B : ℝ := 0.6

-- Definitions for required sequences
def seq_TTH_A : ℝ := prob_tails_A * prob_tails_A * prob_heads_A
def seq_HTT_B : ℝ := prob_heads_B * prob_tails_B * prob_tails_B

-- Main assertion
theorem prob_simultaneous_sequences :
  seq_TTH_A * seq_HTT_B = 0.021168 :=
by
  sorry

end prob_simultaneous_sequences_l297_297154


namespace pooh_piglet_cake_sharing_l297_297445

theorem pooh_piglet_cake_sharing (a b : ℚ) (h1 : a + b = 1) (h2 : b + a/3 = 3*b) : 
  a = 6/7 ∧ b = 1/7 :=
by
  sorry

end pooh_piglet_cake_sharing_l297_297445


namespace shaded_area_fraction_l297_297140

-- Define the problem conditions
def total_squares : ℕ := 18
def half_squares : ℕ := 10
def whole_squares : ℕ := 3

-- Define the total shaded area given the conditions
def shaded_area := (half_squares * (1/2) + whole_squares)

-- Define the total area of the rectangle
def total_area := total_squares

-- Lean 4 theorem statement
theorem shaded_area_fraction :
  shaded_area / total_area = (4 : ℚ) / 9 :=
by sorry

end shaded_area_fraction_l297_297140


namespace hyperbola_equation_l297_297128

theorem hyperbola_equation 
  (vertex : ℝ × ℝ) 
  (asymptote_slope : ℝ) 
  (h_vertex : vertex = (2, 0))
  (h_asymptote : asymptote_slope = Real.sqrt 2) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / 8 = 1) := 
by
    sorry

end hyperbola_equation_l297_297128


namespace not_axisymmetric_sqrt_x_l297_297040

theorem not_axisymmetric_sqrt_x :
  ¬ (is_axisymmetric (fun x => sqrt x)) :=
by
  sorry

end not_axisymmetric_sqrt_x_l297_297040


namespace m_plus_n_in_right_triangle_l297_297405

noncomputable def triangle (A B C : Point) : Prop :=
  ∃ (BD : ℕ) (x : ℕ) (y : ℕ),
  ∃ (AB BC AC : ℕ),
  ∃ (m n : ℕ),
  B ≠ C ∧
  C ≠ A ∧
  B ≠ A ∧
  m.gcd n = 1 ∧
  BD = 17^3 ∧
  BC = 17^2 * x ∧
  AB = 17 * x^2 ∧
  AC = 17 * x * y ∧
  BC^2 + AC^2 = AB^2 ∧
  (2 * 17 * x) = 17^2 ∧
  ∃ cB, cB = (BC : ℚ) / (AB : ℚ) ∧
  cB = (m : ℚ) / (n : ℚ)

theorem m_plus_n_in_right_triangle :
  ∀ (A B C : Point),
  A ≠ B ∧
  B ≠ C ∧
  C ≠ A ∧
  triangle A B C →
  ∃ m n : ℕ, m.gcd n = 1 ∧ m + n = 162 :=
sorry

end m_plus_n_in_right_triangle_l297_297405


namespace tourists_count_l297_297916

theorem tourists_count :
  ∃ (n : ℕ), (1 / 2 * n + 1 / 3 * n + 1 / 4 * n = 39) :=
by
  use 36
  sorry

end tourists_count_l297_297916


namespace values_of_x_l297_297524

def P (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x

theorem values_of_x (x : ℝ) :
  P x = P (x + 1) ↔ (x = 1 ∨ x = 4 / 3) :=
by sorry

end values_of_x_l297_297524


namespace max_halls_l297_297796

theorem max_halls (n : ℕ) (hall : ℕ → ℕ) (H : ∀ n, hall n = hall (3 * n + 1) ∧ hall n = hall (n + 10)) :
  ∃ (m : ℕ), m = 3 :=
by
  sorry

end max_halls_l297_297796


namespace lines_parallel_distinct_l297_297376

theorem lines_parallel_distinct (a : ℝ) : 
  (∀ x y : ℝ, (2 * x - a * y + 1 = 0) → ((a - 1) * x - y + a = 0)) ↔ 
  a = 2 := 
sorry

end lines_parallel_distinct_l297_297376


namespace exponent_sum_equality_l297_297103

theorem exponent_sum_equality {a : ℕ} (h1 : 2^12 + 1 = 17 * a) (h2: a = 2^8 + 2^7 + 2^6 + 2^5 + 2^0) : 
  ∃ a1 a2 a3 a4 a5 : ℕ, 
    a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ 
    2^a1 + 2^a2 + 2^a3 + 2^a4 + 2^a5 = a ∧ 
    a1 = 0 ∧ a2 = 5 ∧ a3 = 6 ∧ a4 = 7 ∧ a5 = 8 ∧ 
    5 = 5 :=
by {
  sorry
}

end exponent_sum_equality_l297_297103


namespace right_triangle_hypotenuse_length_l297_297976

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l297_297976


namespace solution_set_of_inequality_l297_297434

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 3) * (1 - x) ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := 
sorry

end solution_set_of_inequality_l297_297434


namespace right_triangle_hypotenuse_l297_297982

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l297_297982


namespace simplify_expression_l297_297135

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (x⁻¹ - x + 2) = (1 - (x - 1)^2) / x := 
sorry

end simplify_expression_l297_297135


namespace project_presentation_period_length_l297_297055

theorem project_presentation_period_length
  (students : ℕ)
  (presentation_time_per_student : ℕ)
  (number_of_periods : ℕ)
  (total_students : students = 32)
  (time_per_student : presentation_time_per_student = 5)
  (periods_needed : number_of_periods = 4) :
  (32 * 5) / 4 = 40 := 
by {
  sorry
}

end project_presentation_period_length_l297_297055


namespace range_of_a_l297_297849

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → (f a x) * (f a (1 - x)) ≥ 1) ↔ (1 ≤ a) ∨ (a ≤ - (1/4)) := 
by
  sorry

end range_of_a_l297_297849


namespace necessary_but_not_sufficient_condition_l297_297712

def represents_ellipse (k : ℝ) (x y : ℝ) :=
    1 < k ∧ k < 5 ∧ k ≠ 3

theorem necessary_but_not_sufficient_condition (k : ℝ) (x y : ℝ):
    (1 < k ∧ k < 5) → (represents_ellipse k x y) :=
by
  sorry

end necessary_but_not_sufficient_condition_l297_297712


namespace time_difference_halfway_point_l297_297349

noncomputable def danny_to_steve_time : ℝ := 31
noncomputable def steve_to_danny_time : ℝ := 62
noncomputable def wind_factor_danny : ℝ := 1.1
noncomputable def wind_factor_steve : ℝ := 0.9

theorem time_difference_halfway_point : 
  let D := 1 in -- assume the distance D = 1 for simplicity
  let speed_danny := D / danny_to_steve_time in
  let speed_steve := D / steve_to_danny_time in
  let speed_danny_wind := wind_factor_danny * speed_danny in
  let speed_steve_wind := wind_factor_steve * speed_steve in
  let time_danny_half := (D / 2) / speed_danny_wind in
  let time_steve_half := (D / 2) / speed_steve_wind in
  time_steve_half - time_danny_half ≈ 20.35 := 
begin
  sorry
end

end time_difference_halfway_point_l297_297349


namespace f_value_l297_297097

noncomputable def f : ℝ → ℝ
| x => if x > 1 then 2^(x-1) else Real.tan (Real.pi * x / 3)

theorem f_value : f (1 / f 2) = Real.sqrt 3 / 3 := by
  sorry

end f_value_l297_297097


namespace right_triangle_hypotenuse_length_l297_297953

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l297_297953


namespace simplify_expression_l297_297695

theorem simplify_expression (a b : ℝ) (h : a + b < 0) : 
  |a + b - 1| - |3 - (a + b)| = -2 :=
by 
  sorry

end simplify_expression_l297_297695


namespace smallest_x_abs_eq_9_l297_297829

theorem smallest_x_abs_eq_9 : ∃ x : ℝ, |x - 4| = 9 ∧ ∀ y : ℝ, |y - 4| = 9 → x ≤ y :=
by
  -- Prove there exists an x such that |x - 4| = 9 and for all y satisfying |y - 4| = 9, x is the minimum.
  sorry

end smallest_x_abs_eq_9_l297_297829


namespace vasya_problem_l297_297765

theorem vasya_problem : 
  ∃ (e : ℝ), 
  (e = 333 / 3 - 33 / 3 ∨ e = 33 * 3 + 3 / 3) ∧
  (count threes in expression e < 10) ∧
  (e = 100) :=
by sorry

end vasya_problem_l297_297765


namespace sum_of_youngest_and_oldest_nephews_l297_297753

theorem sum_of_youngest_and_oldest_nephews 
    (n1 n2 n3 n4 n5 n6 : ℕ) 
    (mean_eq : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = 10) 
    (median_eq : (n3 + n4) / 2 = 12) : 
    n1 + n6 = 12 := 
by 
    sorry

end sum_of_youngest_and_oldest_nephews_l297_297753


namespace puzzle_solution_l297_297046

theorem puzzle_solution :
  (∀ n m k : ℕ, n + m + k = 111 → 9 * (n + m + k) / 3 = 9) ∧
  (∀ n m k : ℕ, n + m + k = 444 → 12 * (n + m + k) / 12 = 12) ∧
  (∀ n m k : ℕ, n + m + k = 777 → (7 * 3 ≠ 15 → (7 * 3 - 6 = 15)) ) →
  ∀ n m k : ℕ, n + m + k = 888 → 8 * (n + m + k / 3) - 6 = 18 :=
by
  intros h n m k h1
  sorry

end puzzle_solution_l297_297046


namespace carl_profit_l297_297192

-- Define the conditions
def price_per_watermelon : ℕ := 3
def watermelons_start : ℕ := 53
def watermelons_end : ℕ := 18

-- Define the number of watermelons sold
def watermelons_sold : ℕ := watermelons_start - watermelons_end

-- Define the profit
def profit : ℕ := watermelons_sold * price_per_watermelon

-- State the theorem about Carl's profit
theorem carl_profit : profit = 105 :=
by
  -- Proof can be filled in later
  sorry

end carl_profit_l297_297192


namespace rita_coffee_cost_l297_297421

noncomputable def costPerPound (initialAmount spentAmount pounds : ℝ) : ℝ :=
  spentAmount / pounds

theorem rita_coffee_cost :
  ∀ (initialAmount remainingAmount pounds : ℝ),
    initialAmount = 70 ∧ remainingAmount = 35.68 ∧ pounds = 4 →
    costPerPound initialAmount (initialAmount - remainingAmount) pounds = 8.58 :=
by
  intros initialAmount remainingAmount pounds h
  simp [costPerPound, h]
  sorry

end rita_coffee_cost_l297_297421


namespace cheaper_fuji_shimla_l297_297283

variable (S R F : ℝ)
variable (h : 1.05 * (S + R) = R + 0.90 * F + 250)

theorem cheaper_fuji_shimla : S - F = (-0.15 * S - 0.05 * R) / 0.90 + 250 / 0.90 :=
by
  sorry

end cheaper_fuji_shimla_l297_297283


namespace min_m_quad_eq_integral_solutions_l297_297296

theorem min_m_quad_eq_integral_solutions :
  (∃ m : ℕ, (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42) ∧ m > 0) →
  (∃ m : ℕ, m = 130 ∧ (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42)) :=
by
  sorry

end min_m_quad_eq_integral_solutions_l297_297296


namespace seating_arrangement_l297_297400

theorem seating_arrangement (M : ℕ) (h1 : 8 * M = 12 * M) : M = 3 :=
by
  sorry

end seating_arrangement_l297_297400


namespace particles_probability_computation_l297_297233

theorem particles_probability_computation : 
  let L0 := 32
  let R0 := 68
  let N := 100
  let a := 1
  let b := 2
  let P_all_on_left := (a:ℚ) / b
  100 * a + b = 102 := by
  sorry

end particles_probability_computation_l297_297233


namespace triangle_ratio_l297_297175

theorem triangle_ratio (a b c : ℕ) (r s : ℕ) (h1 : a = 9) (h2 : b = 15) (h3 : c = 18) (h4 : r + s = a) (h5 : r < s) : r * 2 = s :=
by
  sorry

end triangle_ratio_l297_297175


namespace Robert_salary_loss_l297_297562

theorem Robert_salary_loss (S : ℝ) (x : ℝ) (h : x ≠ 0) (h1 : (S - (x/100) * S + (x/100) * (S - (x/100) * S) = (96/100) * S)) : x = 20 :=
by sorry

end Robert_salary_loss_l297_297562


namespace find_x0_l297_297222

-- Define the given conditions
variable (p x_0 : ℝ) (P : ℝ × ℝ) (O : ℝ × ℝ)
variable (h_parabola : x_0^2 = 2 * p * 1)
variable (h_p_gt_zero : p > 0)
variable (h_point_P : P = (x_0, 1))
variable (h_origin : O = (0, 0))
variable (h_distance_condition : dist (x_0, 1) (0, 0) = dist (x_0, 1) (0, -p / 2))

-- The theorem we aim to prove
theorem find_x0 : x_0 = 2 * Real.sqrt 2 :=
  sorry

end find_x0_l297_297222


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l297_297771

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l297_297771


namespace equation_of_line_passing_through_point_and_intersects_circle_smallest_circle_through_P_and_C_l297_297676

open Real EuclideanGeometry

-- Problem 1
theorem equation_of_line_passing_through_point_and_intersects_circle (P : Point ℝ) 
    (C : Circle ℝ) (A B : Point ℝ) (k₁ k₂ : ℝ) : 
    (P = (2, 1)) →
    (C = Circle.mk (Equiv.prod (Equiv.refl ℝ) (Equiv.refl ℝ)) ⟨-1, 2⟩ 4) →
    ((A ≠ B) ∧ (A ∈ C.points) ∧ (B ∈ C.points) ∧ (angle A C B = π / 2)) →
    (¬ collinear [P, A, B]) →
    (∀ k, k = k₁ ∨ k = k₂) →
    (k₁ = 1 ∨ k₁ = -7) ∧ (k₂ = 1 ∨ k₂ = -7) →
    (equation_of_line P k₁ = "x - y - 1 = 0") ∨ 
    (equation_of_line P k₂ = "7x + y - 15 = 0") :=
begin
  intros hP hC hA hB hK hk,
  sorry
end

-- Problem 2
theorem smallest_circle_through_P_and_C (P C : Point ℝ) (r : ℝ) :
    P = (2, 1) →
    C = (-1, 2) →
    let center := (1/2, 3/2) in
    r = √5/2 →
    equation_of_circle center r = "(x - 1/2)^2 + (y - 3/2)^2 = 5/2" :=
begin
  intros hP hC center hr,
  sorry
end

end equation_of_line_passing_through_point_and_intersects_circle_smallest_circle_through_P_and_C_l297_297676


namespace part1_part2_l297_297094

noncomputable def A (a : ℝ) := { x : ℝ | x^2 - a * x + a^2 - 19 = 0 }
def B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
def C := { x : ℝ | x^2 + 2 * x - 8 = 0 }

-- Proof Problem 1: Prove that if A ∩ B ≠ ∅ and A ∩ C = ∅, then a = -2
theorem part1 (a : ℝ) (h1 : (A a ∩ B) ≠ ∅) (h2 : (A a ∩ C) = ∅) : a = -2 :=
sorry

-- Proof Problem 2: Prove that if A ∩ B = A ∩ C ≠ ∅, then a = -3
theorem part2 (a : ℝ) (h1 : (A a ∩ B = A a ∩ C) ∧ (A a ∩ B) ≠ ∅) : a = -3 :=
sorry

end part1_part2_l297_297094


namespace find_a6_l297_297840

variable (S : ℕ → ℝ) (a : ℕ → ℝ)
variable (h1 : ∀ n ≥ 2, S n = 2 * a n)
variable (h2 : S 5 = 8)

theorem find_a6 : a 6 = 8 :=
by
  sorry

end find_a6_l297_297840


namespace number_chosen_l297_297788

theorem number_chosen (x : ℤ) (h : x / 4 - 175 = 10) : x = 740 := by
  sorry

end number_chosen_l297_297788


namespace hypotenuse_length_l297_297947

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l297_297947


namespace smallest_n_to_make_183_divisible_by_11_l297_297228

theorem smallest_n_to_make_183_divisible_by_11 : ∃ n : ℕ, 183 + n % 11 = 0 ∧ n = 4 :=
by
  have h1 : 183 % 11 = 7 := 
    sorry
  let n := 11 - (183 % 11)
  have h2 : 183 + n % 11 = 0 :=
    sorry
  exact ⟨n, h2, sorry⟩

end smallest_n_to_make_183_divisible_by_11_l297_297228


namespace f_at_63_l297_297144

-- Define the function f: ℤ → ℤ with given properties
def f : ℤ → ℤ :=
  sorry -- Placeholder, as we are only stating the problem, not the solution

-- Conditions
axiom f_at_1 : f 1 = 6
axiom f_eq : ∀ x : ℤ, f (2 * x + 1) = 3 * f x

-- The goal is to prove f(63) = 1458
theorem f_at_63 : f 63 = 1458 :=
  sorry

end f_at_63_l297_297144


namespace commute_solution_l297_297882

noncomputable def commute_problem : Prop :=
  let t : ℝ := 1                -- 1 hour from 7:00 AM to 8:00 AM
  let late_minutes : ℝ := 5 / 60  -- 5 minutes = 5/60 hours
  let early_minutes : ℝ := 4 / 60 -- 4 minutes = 4/60 hours
  let speed1 : ℝ := 30          -- 30 mph
  let speed2 : ℝ := 70          -- 70 mph
  let d1 : ℝ := speed1 * (t + late_minutes)
  let d2 : ℝ := speed2 * (t - early_minutes)

  ∃ (speed : ℝ), d1 = d2 ∧ speed = d1 / t ∧ speed = 32.5

theorem commute_solution : commute_problem :=
by sorry

end commute_solution_l297_297882


namespace modular_inverse_example_l297_297160

open Int

theorem modular_inverse_example :
  ∃ b : ℤ, 0 ≤ b ∧ b < 120 ∧ (7 * b) % 120 = 1 ∧ b = 103 :=
by
  sorry

end modular_inverse_example_l297_297160


namespace determine_sunday_l297_297256

def Brother := Prop -- A type to represent a brother

variable (A B : Brother)
variable (T D : Brother) -- T representing Tweedledum, D representing Tweedledee

-- Conditions translated into Lean
variable (H1 : (A = T) → (B = D))
variable (H2 : (B = D) → (A = T))

-- Define the day of the week as a proposition
def is_sunday := Prop

-- We want to state that given H1 and H2, it is Sunday
theorem determine_sunday (H1 : (A = T) → (B = D)) (H2 : (B = D) → (A = T)) : is_sunday := sorry

end determine_sunday_l297_297256


namespace quadratic_roots_are_distinct_real_l297_297261

theorem quadratic_roots_are_distinct_real (a b c : ℝ) (h_eq : a = 1) (h_b : b = 4) (h_c : c = 0) :
  a * c = 0 ∧ b^2 - 4 * a * c > 0 :=
by
  rw [h_eq, h_b, h_c]
  split
  case left => 
    calc 
      1 * 0 = 0 : by norm_num
  case right =>
    calc 
      4^2 - 4 * 1 * 0 = 16 - 0 : by norm_num
      ... = 16      : by norm_num
      ... > 0       : by norm_num

end quadratic_roots_are_distinct_real_l297_297261


namespace find_x_l297_297517

variables {x y z d e f : ℝ}
variables (h1 : xy / (x + 2 * y) = d)
variables (h2 : xz / (2 * x + z) = e)
variables (h3 : yz / (y + 2 * z) = f)

theorem find_x :
  x = 3 * d * e * f / (d * e - 2 * d * f + e * f) :=
sorry

end find_x_l297_297517


namespace trigonometric_identity_l297_297475

theorem trigonometric_identity :
  let cos60 := (1 / 2)
  let sin30 := (1 / 2)
  let tan45 := (1 : ℝ)
  4 * cos60 + 8 * sin30 - 5 * tan45 = 1 :=
by
  let cos60 := (1 / 2 : ℝ)
  let sin30 := (1 / 2 : ℝ)
  let tan45 := (1 : ℝ)
  show 4 * cos60 + 8 * sin30 - 5 * tan45 = 1
  sorry

end trigonometric_identity_l297_297475


namespace ratio_increase_productivity_l297_297020

theorem ratio_increase_productivity (initial current: ℕ) 
  (h_initial: initial = 10) 
  (h_current: current = 25) : 
  (current - initial) / initial = 3 / 2 := 
by
  sorry

end ratio_increase_productivity_l297_297020


namespace largest_difference_l297_297878

def U : ℕ := 2 * 1002 ^ 1003
def V : ℕ := 1002 ^ 1003
def W : ℕ := 1001 * 1002 ^ 1002
def X : ℕ := 2 * 1002 ^ 1002
def Y : ℕ := 1002 ^ 1002
def Z : ℕ := 1002 ^ 1001

theorem largest_difference : (U - V) = 1002 ^ 1003 ∧ 
  (V - W) = 1002 ^ 1002 ∧ 
  (W - X) = 999 * 1002 ^ 1002 ∧ 
  (X - Y) = 1002 ^ 1002 ∧ 
  (Y - Z) = 1001 * 1002 ^ 1001 ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 999 * 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1001 * 1002 ^ 1001) :=
by {
  sorry
}

end largest_difference_l297_297878


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297309

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem exists_nine_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 8 ≤ 500) ∧ ∀ i ∈ (List.range 9), is_composite (a + i) :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 10 ≤ 500) ∧ ∀ i ∈ (List.range 11), is_composite (a + i) :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297309


namespace binomial_sum_of_coefficients_l297_297231

theorem binomial_sum_of_coefficients (n : ℕ) (h₀ : (1 - 2)^n = 8) :
  (1 - 2)^n = -1 :=
sorry

end binomial_sum_of_coefficients_l297_297231


namespace men_work_in_80_days_l297_297860

theorem men_work_in_80_days (x : ℕ) (work_eq_20men_56days : x * 80 = 20 * 56) : x = 14 :=
by 
  sorry

end men_work_in_80_days_l297_297860


namespace least_incorrect_option_is_A_l297_297399

def dozen_units : ℕ := 12
def chairs_needed : ℕ := 4

inductive CompletionOption
| dozen
| dozens
| dozen_of
| dozens_of

def correct_option (op : CompletionOption) : Prop :=
  match op with
  | CompletionOption.dozen => dozen_units >= chairs_needed
  | CompletionOption.dozens => False
  | CompletionOption.dozen_of => False
  | CompletionOption.dozens_of => False

theorem least_incorrect_option_is_A : correct_option CompletionOption.dozen :=
by {
  sorry
}

end least_incorrect_option_is_A_l297_297399


namespace intersection_of_S_and_T_l297_297850

def S : Set ℝ := {x | x^2 - x ≥ 0}
def T : Set ℝ := {x | 0 < x}

theorem intersection_of_S_and_T : S ∩ T = {x | 1 ≤ x} := by
  sorry

end intersection_of_S_and_T_l297_297850


namespace solve_for_s_l297_297264

theorem solve_for_s :
  let numerator := Real.sqrt (7^2 + 24^2)
  let denominator := Real.sqrt (64 + 36)
  let s := numerator / denominator
  s = 5 / 2 :=
by
  sorry

end solve_for_s_l297_297264


namespace simplify_neg_neg_l297_297921

theorem simplify_neg_neg (a b : ℝ) : -(-a - b) = a + b :=
sorry

end simplify_neg_neg_l297_297921


namespace grading_combinations_l297_297178

/-- There are 12 students in the class. -/
def num_students : ℕ := 12

/-- There are 4 possible grades (A, B, C, and D). -/
def num_grades : ℕ := 4

/-- The total number of ways to assign grades. -/
theorem grading_combinations : (num_grades ^ num_students) = 16777216 := 
by
  sorry

end grading_combinations_l297_297178


namespace no_integers_p_q_l297_297354

theorem no_integers_p_q :
  ¬ ∃ p q : ℤ, ∀ x : ℤ, 3 ∣ (x^2 + p * x + q) :=
by
  sorry

end no_integers_p_q_l297_297354


namespace multiply_469160_999999_l297_297471

theorem multiply_469160_999999 :
  469160 * 999999 = 469159530840 :=
by
  sorry

end multiply_469160_999999_l297_297471


namespace value_of_a_value_of_sin_A_plus_pi_over_4_l297_297549

section TriangleABC

variables {a b c A B : ℝ}
variables (h_b : b = 3) (h_c : c = 1) (h_A_eq_2B : A = 2 * B)

theorem value_of_a : a = 2 * Real.sqrt 3 :=
sorry

theorem value_of_sin_A_plus_pi_over_4 : Real.sin (A + π / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end TriangleABC

end value_of_a_value_of_sin_A_plus_pi_over_4_l297_297549


namespace smartphone_cost_decrease_l297_297570

theorem smartphone_cost_decrease :
  ∀ (cost2010 cost2020 : ℝ),
  cost2010 = 600 →
  cost2020 = 450 →
  ((cost2010 - cost2020) / cost2010) * 100 = 25 :=
by
  intros cost2010 cost2020 h1 h2
  sorry

end smartphone_cost_decrease_l297_297570


namespace limit_one_plus_inv_x_to_e_l297_297130

-- Lean 4 statement for the proof problem
theorem limit_one_plus_inv_x_to_e (hx_pos : ∀ x, 0 < x) :
  (∀ x : ℝ, x ≠ 0 → tendsto (λ x, (1 + 1 / x) ^ x) at_top (𝓝 real.exp 1)) :=
begin
  sorry
end

end limit_one_plus_inv_x_to_e_l297_297130


namespace sum_after_50_rounds_l297_297709

def initial_states : List ℤ := [1, 0, -1]

def operation (n : ℤ) : ℤ :=
  match n with
  | 1   => n * n * n
  | 0   => n * n
  | -1  => -n
  | _ => n  -- although not necessary for current problem, this covers other possible states

def process_calculator (state : ℤ) (times: ℕ) : ℤ :=
  if state = 1 then state
  else if state = 0 then state
  else if state = -1 then state * (-1) ^ times
  else state

theorem sum_after_50_rounds :
  let final_states := initial_states.map (fun s => process_calculator s 50)
  final_states.sum = 2 := by
  simp only [initial_states, process_calculator]
  simp
  sorry

end sum_after_50_rounds_l297_297709


namespace unique_selection_l297_297092

open Nat

/-- Given a finite set S of natural numbers, the first player selects a number s from S.
The second player says a number x (not necessarily in S), and then the first player says σ₀(xs).
Both players know all elements of S. Prove that the second player can say only one number x and 
understand which number the first player has selected. -/
theorem unique_selection (S : Finset ℕ) (s : ℕ) (hS : s ∈ S) :
  ∃ (x : ℕ), ∀ s' ∈ S, s' ≠ s → 
  ∃! d, d = (x * s) ∧ (∀ t ∈ S, t ≠ s → (σ₀ (x * s) = σ₀ (x * t) → s = t)) :=
by
  sorry

end unique_selection_l297_297092


namespace residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l297_297166

noncomputable def phi (n : ℕ) : ℕ := Nat.totient n

theorem residues_exponent (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d ∧ ∀ x ∈ S, x^d % p = 1 :=
by sorry

theorem residues_divides_p_minus_one (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d :=
by sorry
  
theorem primitive_roots_phi (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  ∃ (S : Finset ℕ), S.card = phi (p-1) ∧ ∀ g ∈ S, IsPrimitiveRoot g p :=
by sorry

end residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l297_297166


namespace player_A_always_wins_l297_297735

theorem player_A_always_wins (a b c : ℤ) :
  ∃ (x1 x2 x3 : ℤ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (x - x1) * (x - x2) * (x - x3) = x^3 + a*x^2 + b*x + c :=
sorry

end player_A_always_wins_l297_297735


namespace part_1_part_2_equality_case_l297_297835

variables {m n : ℝ}

-- Definition of positive real numbers and given condition m > n and n > 1
def conditions_1 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m > n ∧ n > 1

-- Prove that given conditions, m^2 + n > mn + m
theorem part_1 (m n : ℝ) (h : conditions_1 m n) : m^2 + n > m * n + m :=
  by sorry

-- Definition of the condition m + 2n = 1
def conditions_2 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m + 2 * n = 1

-- Prove that given conditions, (2/m) + (1/n) ≥ 8
theorem part_2 (m n : ℝ) (h : conditions_2 m n) : (2 / m) + (1 / n) ≥ 8 :=
  by sorry

-- Prove that the minimum value is obtained when m = 2n = 1/2
theorem equality_case (m n : ℝ) (h : conditions_2 m n) : 
  (2 / m) + (1 / n) = 8 ↔ m = 1/2 ∧ n = 1/4 :=
  by sorry

end part_1_part_2_equality_case_l297_297835


namespace right_triangle_hypotenuse_l297_297959

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l297_297959


namespace recurring_decimal_36_exceeds_decimal_35_l297_297815

-- Definition of recurring decimal 0.36...
def recurring_decimal_36 : ℚ := 36 / 99

-- Definition of 0.35 as fraction
def decimal_35 : ℚ := 7 / 20

-- Statement of the math proof problem
theorem recurring_decimal_36_exceeds_decimal_35 :
  recurring_decimal_36 - decimal_35 = 3 / 220 := by
  sorry

end recurring_decimal_36_exceeds_decimal_35_l297_297815


namespace a2009_equals_7_l297_297685

def sequence_element (n k : ℕ) : ℚ :=
  if k = 0 then 0 else (n - k + 1) / k

def cumulative_count (n : ℕ) : ℕ := n * (n + 1) / 2

theorem a2009_equals_7 : 
  let n := 63
  let m := 2009
  let subset_cumulative_count := cumulative_count n
  (2 * m = n * (n + 1) - 14 ∧
   m = subset_cumulative_count - 7 ∧ 
   sequence_element n 8 = 7) →
  sequence_element n (subset_cumulative_count - m + 1) = 7 :=
by
  -- proof steps to be filled here
  sorry

end a2009_equals_7_l297_297685


namespace difference_even_odd_sums_l297_297444

def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)
def sum_first_n_odd_numbers (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : sum_first_n_even_numbers 1001 - sum_first_n_odd_numbers 1001 = 1001 := by
  sorry

end difference_even_odd_sums_l297_297444


namespace inexperienced_sailors_count_l297_297061

theorem inexperienced_sailors_count
  (I E : ℕ)
  (h1 : I + E = 17)
  (h2 : ∀ (rate_inexperienced hourly_rate experienced_rate : ℕ), hourly_rate = 10 → experienced_rate = 12 → rate_inexperienced = 2400)
  (h3 : ∀ (total_income experienced_salary : ℕ), total_income = 34560 → experienced_salary = 2880)
  (h4 : ∀ (monthly_income : ℕ), monthly_income = 34560)
  : I = 5 := sorry

end inexperienced_sailors_count_l297_297061


namespace part1_part2_l297_297377

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) * Real.exp x

theorem part1 (a x : ℝ) (h : a > 0) : f a x + a / Real.exp 1 > 0 := by
  sorry

theorem part2 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f (-1/2) x1 = f (-1/2) x2) : x1 + x2 < 2 := by
  sorry

end part1_part2_l297_297377


namespace right_triangle_hypotenuse_length_l297_297950

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l297_297950


namespace maximize_f_l297_297993

open Nat

-- Define the combination function
def comb (n k : ℕ) : ℕ := choose n k

-- Define the probability function f(n)
def f (n : ℕ) : ℚ := 
  (comb n 2 * comb (100 - n) 8 : ℚ) / comb 100 10

-- Define the theorem to find the value of n that maximizes f(n)
theorem maximize_f : ∃ n : ℕ, 2 ≤ n ∧ n ≤ 92 ∧ (∀ m : ℕ, 2 ≤ m ∧ m ≤ 92 → f n ≥ f m) ∧ n = 20 :=
by
  sorry

end maximize_f_l297_297993


namespace birth_date_of_older_friend_l297_297588

/-- Lean 4 statement for the proof problem --/
theorem birth_date_of_older_friend
  (d m y : ℕ)
  (h1 : y ≥ 1900 ∧ y < 2000)
  (h2 : d + 7 < 32) -- Assuming the month has at most 31 days
  (h3 : ((d+7) * 10^4 + m * 10^2 + y % 100) = 6 * (d * 10^4 + m * 10^2 + y % 100))
  (h4 : m > 0 ∧ m < 13)  -- Months are between 1 and 12
  (h5 : (d * 10^4 + m * 10^2 + y % 100) < (d+7) * 10^4 + m * 10^2 + y % 100) -- d < d+7 so older means smaller number
  : d = 1 ∧ m = 4 ∧ y = 1900 :=
by
  sorry -- Proof omitted

end birth_date_of_older_friend_l297_297588


namespace proof_m_n_sum_l297_297762

-- Definitions based on conditions
def m : ℕ := 2
def n : ℕ := 49

-- Problem statement as a Lean theorem
theorem proof_m_n_sum : m + n = 51 :=
by
  -- This is where the detailed proof would go. Using sorry to skip the proof.
  sorry

end proof_m_n_sum_l297_297762


namespace pieces_of_wood_for_table_l297_297673

theorem pieces_of_wood_for_table :
  ∀ (T : ℕ), (24 * T + 48 * 8 = 672) → T = 12 :=
by
  intro T
  intro h
  sorry

end pieces_of_wood_for_table_l297_297673


namespace part_a_part_b_l297_297449

def N := 10^40

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_perfect_square (a : ℕ) : Prop := ∃ m : ℕ, m * m = a

def is_perfect_cube (a : ℕ) : Prop := ∃ m : ℕ, m * m * m = a

def is_perfect_power (a : ℕ) : Prop := ∃ (m n : ℕ), n > 1 ∧ a = m^n

def num_divisors_not_square_or_cube (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that are neither perfect squares nor perfect cubes

def num_divisors_not_in_form_m_n (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that cannot be represented in the form m^n where n > 1

theorem part_a : num_divisors_not_square_or_cube N = 1093 := by
  sorry

theorem part_b : num_divisors_not_in_form_m_n N = 981 := by
  sorry

end part_a_part_b_l297_297449


namespace arithmetic_sequence_z_value_l297_297034

theorem arithmetic_sequence_z_value :
  ∃ z : ℤ, (3 ^ 2 = 9 ∧ 3 ^ 4 = 81) ∧ z = (9 + 81) / 2 :=
by
  -- the proof goes here
  sorry

end arithmetic_sequence_z_value_l297_297034


namespace evaluate_expression_l297_297076

theorem evaluate_expression : 
  ( (7 : ℝ) ^ (1 / 4) / (7 : ℝ) ^ (1 / 7) ) = 7 ^ (3 / 28) := 
by {
  sorry
}

end evaluate_expression_l297_297076


namespace value_of_x_squared_minus_y_squared_l297_297535

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l297_297535


namespace sample_size_proportion_l297_297176

theorem sample_size_proportion (n : ℕ) (ratio_A B C : ℕ) (A_sample : ℕ) (ratio_A_val : ratio_A = 5) (ratio_B_val : ratio_B = 2) (ratio_C_val : ratio_C = 3) (A_sample_val : A_sample = 15) (total_ratio : ratio_A + ratio_B + ratio_C = 10) : 
  15 / n = 5 / 10 → n = 30 :=
sorry

end sample_size_proportion_l297_297176


namespace hypotenuse_length_l297_297967

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l297_297967


namespace lines_intersect_l297_297941

noncomputable def line1 (t : ℚ) : ℚ × ℚ :=
(1 + 2 * t, 2 - 3 * t)

noncomputable def line2 (u : ℚ) : ℚ × ℚ :=
(-1 + 3 * u, 4 + u)

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = (-5 / 11, 46 / 11) ∧ line2 u = (-5 / 11, 46 / 11) :=
sorry

end lines_intersect_l297_297941


namespace parabola_vertex_l297_297574

theorem parabola_vertex :
  ∃ a k : ℝ, (∀ x y : ℝ, y^2 - 4*y + 2*x + 7 = 0 ↔ y = k ∧ x = a - (1/2)*(y - k)^2) ∧ a = -3/2 ∧ k = 2 :=
by
  sorry

end parabola_vertex_l297_297574


namespace positive_value_of_X_l297_297248

def hash_relation (X Y : ℕ) : ℕ := X^2 + Y^2

theorem positive_value_of_X (X : ℕ) (h : hash_relation X 7 = 290) : X = 17 :=
by sorry

end positive_value_of_X_l297_297248


namespace find_distance_between_sides_l297_297209

-- Define the given conditions
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area_trapezium : ℝ := 247

-- Define the distance h between parallel sides
def distance_between_sides (h : ℝ) : Prop :=
  area_trapezium = (1 / 2) * (length_side1 + length_side2) * h

-- Define the theorem we want to prove
theorem find_distance_between_sides : ∃ h : ℝ, distance_between_sides h ∧ h = 13 := by
  sorry

end find_distance_between_sides_l297_297209


namespace sampling_methods_correct_l297_297599

def company_sales_outlets (A B C D : ℕ) : Prop :=
  A = 150 ∧ B = 120 ∧ C = 180 ∧ D = 150 ∧ A + B + C + D = 600

def investigation_samples (total_samples large_outlets region_C_sample : ℕ) : Prop :=
  total_samples = 100 ∧ large_outlets = 20 ∧ region_C_sample = 7

def appropriate_sampling_methods (investigation1_method investigation2_method : String) : Prop :=
  investigation1_method = "Stratified sampling" ∧ investigation2_method = "Simple random sampling"

theorem sampling_methods_correct :
  company_sales_outlets 150 120 180 150 →
  investigation_samples 100 20 7 →
  appropriate_sampling_methods "Stratified sampling" "Simple random sampling" :=
by
  intros h1 h2
  sorry

end sampling_methods_correct_l297_297599


namespace halfway_between_one_eighth_and_one_third_is_correct_l297_297361

-- Define the fractions
def one_eighth : ℚ := 1 / 8
def one_third : ℚ := 1 / 3

-- Define the correct answer
def correct_answer : ℚ := 11 / 48

-- State the theorem to prove the halfway number is correct_answer
theorem halfway_between_one_eighth_and_one_third_is_correct : 
  (one_eighth + one_third) / 2 = correct_answer :=
sorry

end halfway_between_one_eighth_and_one_third_is_correct_l297_297361


namespace arithmetic_sequence_middle_term_l297_297033

theorem arithmetic_sequence_middle_term :
  ∀ (a b : ℕ) (z : ℕ), a = 9 → b = 81 → z = (a + b) / 2 → z = 45 :=
by
  intros a b z h_a h_b h_z
  rw [h_a, h_b] at h_z
  exact h_z

end arithmetic_sequence_middle_term_l297_297033


namespace work_required_to_lift_satellite_l297_297205

noncomputable def satellite_lifting_work (m H R3 g : ℝ) : ℝ :=
  m * g * R3^2 * ((1 / R3) - (1 / (R3 + H)))

theorem work_required_to_lift_satellite :
  satellite_lifting_work (7.0 * 10^3) (200 * 10^3) (6380 * 10^3) 10 = 13574468085 :=
by sorry

end work_required_to_lift_satellite_l297_297205


namespace find_integer_m_l297_297384

theorem find_integer_m (m : ℤ) :
  (∃! x : ℤ, |2 * x - m| ≤ 1 ∧ x = 2) → m = 4 :=
by
  intro h
  sorry

end find_integer_m_l297_297384


namespace not_dividable_by_wobbly_l297_297992

-- Define a wobbly number
def is_wobbly_number (n : ℕ) : Prop :=
  n > 0 ∧ (∀ k : ℕ, k < (Nat.log 10 n) → 
    (n / (10^k) % 10 ≠ 0 → n / (10^(k+1)) % 10 = 0) ∧
    (n / (10^k) % 10 = 0 → n / (10^(k+1)) % 10 ≠ 0))

-- Define sets of multiples of 10 and 25
def multiples_of (m : ℕ) (k : ℕ): Prop :=
  ∃ q : ℕ, k = q * m

def is_multiple_of_10 (k : ℕ) : Prop := multiples_of 10 k
def is_multiple_of_25 (k : ℕ) : Prop := multiples_of 25 k

theorem not_dividable_by_wobbly (n : ℕ) : 
  ¬ ∃ w : ℕ, is_wobbly_number w ∧ n ∣ w ↔ is_multiple_of_10 n ∨ is_multiple_of_25 n :=
by
  sorry

end not_dividable_by_wobbly_l297_297992


namespace min_rectangles_needed_l297_297297

theorem min_rectangles_needed : ∀ (n : ℕ), n = 12 → (n * n) / (3 * 2) = 24 :=
by sorry

end min_rectangles_needed_l297_297297


namespace diagonals_in_nonagon_l297_297600

-- Define the properties of the polygon
def convex : Prop := true
def sides (n : ℕ) : Prop := n = 9
def right_angles (count : ℕ) : Prop := count = 2

-- Define the formula for the number of diagonals in a polygon with 'n' sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem definition
theorem diagonals_in_nonagon :
  convex →
  (sides 9) →
  (right_angles 2) →
  number_of_diagonals 9 = 27 :=
by
  sorry

end diagonals_in_nonagon_l297_297600


namespace vasya_problem_l297_297764

theorem vasya_problem : 
  ∃ (e : ℝ), 
  (e = 333 / 3 - 33 / 3 ∨ e = 33 * 3 + 3 / 3) ∧
  (count threes in expression e < 10) ∧
  (e = 100) :=
by sorry

end vasya_problem_l297_297764


namespace intersection_of_A_and_B_l297_297096

-- Definitions of sets A and B based on the conditions
def A : Set ℝ := {x | 0 < x}
def B : Set ℝ := {0, 1, 2}

-- Theorem statement to prove A ∩ B = {1, 2}
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := 
  sorry

end intersection_of_A_and_B_l297_297096


namespace right_triangle_hypotenuse_l297_297980

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l297_297980


namespace brooke_earns_144_dollars_l297_297339

-- Definitions based on the identified conditions
def price_of_milk_per_gallon : ℝ := 3
def production_cost_per_gallon_of_butter : ℝ := 0.5
def sticks_of_butter_per_gallon : ℝ := 2
def price_of_butter_per_stick : ℝ := 1.5
def number_of_cows : ℕ := 12
def milk_per_cow : ℝ := 4
def number_of_customers : ℕ := 6
def min_milk_per_customer : ℝ := 4
def max_milk_per_customer : ℝ := 8

-- Auxiliary calculations
def total_milk_produced : ℝ := number_of_cows * milk_per_cow
def min_total_customer_demand : ℝ := number_of_customers * min_milk_per_customer
def max_total_customer_demand : ℝ := number_of_customers * max_milk_per_customer

-- Problem statement
theorem brooke_earns_144_dollars :
  (0 <= total_milk_produced) ∧
  (min_total_customer_demand <= max_total_customer_demand) ∧
  (total_milk_produced = max_total_customer_demand) →
  (total_milk_produced * price_of_milk_per_gallon = 144) :=
by
  -- Sorry is added here since the proof is not required
  sorry

end brooke_earns_144_dollars_l297_297339


namespace april_roses_l297_297066

theorem april_roses (price_per_rose earnings roses_left : ℤ) 
  (h1 : price_per_rose = 4)
  (h2 : earnings = 36)
  (h3 : roses_left = 4) :
  4 + (earnings / price_per_rose) = 13 :=
by
  sorry

end april_roses_l297_297066


namespace mowing_lawn_each_week_l297_297723

-- Definitions based on the conditions
def riding_speed : ℝ := 2 -- acres per hour with riding mower
def push_speed : ℝ := 1 -- acre per hour with push mower
def total_hours : ℝ := 5 -- total hours

-- The problem we want to prove
theorem mowing_lawn_each_week (A : ℝ) :
  (3 / 4) * A / riding_speed + (1 / 4) * A / push_speed = total_hours → 
  A = 15 :=
by
  sorry

end mowing_lawn_each_week_l297_297723


namespace mix_solutions_l297_297887

variables (Vx : ℚ)

def alcohol_content_x (Vx : ℚ) : ℚ := 0.10 * Vx
def alcohol_content_y : ℚ := 0.30 * 450
def final_alcohol_content (Vx : ℚ) : ℚ := 0.22 * (Vx + 450)

theorem mix_solutions (Vx : ℚ) (h : 0.10 * Vx + 0.30 * 450 = 0.22 * (Vx + 450)) :
  Vx = 300 :=
sorry

end mix_solutions_l297_297887


namespace has_two_zeros_of_f_l297_297379

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.exp x - a

theorem has_two_zeros_of_f (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (-1 / Real.exp 2 < a ∧ a < 0) := by
sorry

end has_two_zeros_of_f_l297_297379


namespace macy_miles_left_to_run_l297_297122

-- Define the given conditions
def goal : ℕ := 24
def miles_per_day : ℕ := 3
def days : ℕ := 6

-- Define the statement to be proven
theorem macy_miles_left_to_run :
  goal - (miles_per_day * days) = 6 :=
by
  sorry

end macy_miles_left_to_run_l297_297122


namespace minimum_value_ineq_l297_297726

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ (3 / 4) := sorry

end minimum_value_ineq_l297_297726


namespace parabola_equation_l297_297523

theorem parabola_equation (p : ℝ) (hp : 0 < p) (F : ℝ × ℝ) (Q : ℝ × ℝ) (PQ QF : ℝ)
  (hPQ : PQ = 8 / p) (hQF : QF = 8 / p + p / 2) (hDist : QF = 5 / 4 * PQ) : 
  ∃ x, y^2 = 4 * x :=
by
  sorry

end parabola_equation_l297_297523


namespace calculation_of_product_l297_297050

theorem calculation_of_product : (0.09)^3 * 0.0007 = 0.0000005103 := 
by
  sorry

end calculation_of_product_l297_297050


namespace factorize_expression_l297_297078

theorem factorize_expression (x : ℝ) :
  9 * x^2 - 6 * x + 1 = (3 * x - 1)^2 := 
by sorry

end factorize_expression_l297_297078


namespace sub_three_five_l297_297998

theorem sub_three_five : 3 - 5 = -2 := 
by 
  sorry

end sub_three_five_l297_297998


namespace point_P_coordinates_l297_297845

noncomputable def P_coordinates (θ : ℝ) : ℝ × ℝ :=
(3 * Real.cos θ, 4 * Real.sin θ)

theorem point_P_coordinates : 
  ∀ θ, (0 ≤ θ ∧ θ ≤ Real.pi ∧ 1 = (4 / 3) * Real.tan θ) →
  P_coordinates θ = (12 / 5, 12 / 5) :=
by
  intro θ h
  sorry

end point_P_coordinates_l297_297845


namespace find_radius_l297_297368

-- Define the given conditions as variables
variables (l A r : ℝ)

-- Conditions from the problem
-- 1. The arc length of the sector is 2 cm
def arc_length_eq : Prop := l = 2

-- 2. The area of the sector is 2 cm²
def area_eq : Prop := A = 2

-- Formula for the area of the sector
def sector_area (l r : ℝ) : ℝ := 0.5 * l * r

-- Define the goal to prove the radius is 2 cm
theorem find_radius (h₁ : arc_length_eq l) (h₂ : area_eq A) : r = 2 :=
by {
  sorry -- proof omitted
}

end find_radius_l297_297368


namespace circle_radius_is_2_chord_length_is_2sqrt3_l297_297319

-- Define the given conditions
def inclination_angle_line_incl60 : Prop := ∃ m, m = Real.sqrt 3
def circle_eq : Prop := ∀ x y, x^2 + y^2 - 4 * y = 0

-- Prove: radius of the circle
theorem circle_radius_is_2 (h : circle_eq) : radius = 2 := sorry

-- Prove: length of the chord cut by the line
theorem chord_length_is_2sqrt3 
  (h1 : inclination_angle_line_incl60) 
  (h2 : circle_eq) : chord_length = 2 * Real.sqrt 3 := sorry

end circle_radius_is_2_chord_length_is_2sqrt3_l297_297319


namespace common_chord_is_linear_l297_297749

-- Defining the equations of two intersecting circles
noncomputable def circle1 : ℝ → ℝ → ℝ := sorry
noncomputable def circle2 : ℝ → ℝ → ℝ := sorry

-- Defining a method to eliminate quadratic terms
noncomputable def eliminate_quadratic_terms (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Defining the linear equation representing the common chord
noncomputable def common_chord (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Statement of the problem
theorem common_chord_is_linear (circle1 circle2 : ℝ → ℝ → ℝ) :
  common_chord circle1 circle2 = eliminate_quadratic_terms circle1 circle2 := sorry

end common_chord_is_linear_l297_297749


namespace systematic_sampling_sequence_l297_297511

theorem systematic_sampling_sequence :
  ∃ k : ℕ, ∃ b : ℕ, (∀ n : ℕ, n < 6 → (3 + n * k = b + n * 10)) ∧ (b = 3 ∨ b = 13 ∨ b = 23 ∨ b = 33 ∨ b = 43 ∨ b = 53) :=
sorry

end systematic_sampling_sequence_l297_297511


namespace bake_sale_cookies_l297_297607

theorem bake_sale_cookies (R O C : ℕ) (H1 : R = 42) (H2 : R = 6 * O) (H3 : R = 2 * C) : R + O + C = 70 := by
  sorry

end bake_sale_cookies_l297_297607


namespace total_muffins_l297_297334

-- Define initial conditions
def initial_muffins : ℕ := 35
def additional_muffins : ℕ := 48

-- Define the main theorem we want to prove
theorem total_muffins : initial_muffins + additional_muffins = 83 :=
by
  sorry

end total_muffins_l297_297334


namespace value_of_x_squared_minus_y_squared_l297_297532

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l297_297532


namespace sum_of_series_l297_297649

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l297_297649


namespace solution_set_of_inequality_l297_297281

theorem solution_set_of_inequality (x: ℝ) : 
  (1 / x ≤ 1) ↔ (x < 0 ∨ x ≥ 1) :=
sorry

end solution_set_of_inequality_l297_297281


namespace regression_and_income_l297_297586

-- Define the given data points
def months : List ℝ := [1, 2, 3, 4, 5]
def income : List ℝ := [0.3, 0.3, 0.5, 0.9, 1]

-- Define the means of x and y, and sums needed
def x_mean := (months.sum) / 5
def y_mean := (income.sum) / 5
def xy_sum := (List.zipWith (*) months income).sum
def x2_sum := (months.map (λ x => x * x)).sum

-- Define the regression coefficients
def b := (xy_sum - 5 * x_mean * y_mean) / (x2_sum - 5 * x_mean ^ 2)
def a := y_mean - b * x_mean

-- Define the regression line
def regression_line (t : ℝ) : ℝ := a + b * t

-- Define the prediction for September (month 9)
def income_september := regression_line 9

theorem regression_and_income : 
  regression_line = λ t, 0.2 * t 
  ∧ income_september <= 2 := by
sorry

end regression_and_income_l297_297586


namespace arithmetic_sequence_mod_l297_297495

theorem arithmetic_sequence_mod :
  let a := 2
  let d := 5
  let l := 137
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  n = 28 ∧ S = 1946 →
  S % 20 = 6 :=
by
  intros h
  sorry

end arithmetic_sequence_mod_l297_297495


namespace center_of_hyperbola_l297_297822

theorem center_of_hyperbola :
  ∃ (h k : ℝ), (h = 2 ∧ k = 4) ∧ (9 * (x - h)^2 - 16 * (y - k)^2 = 180) :=
  sorry

end center_of_hyperbola_l297_297822


namespace vasya_improved_example1_vasya_improved_example2_l297_297776

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l297_297776


namespace problem1_problem2_l297_297492

theorem problem1 : 2 * Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  -- Proof omitted
  sorry

theorem problem2 : (Real.sqrt 12 - Real.sqrt 24) / Real.sqrt 6 - 2 * Real.sqrt (1/2) = -2 :=
by
  -- Proof omitted
  sorry

end problem1_problem2_l297_297492


namespace salary_increase_percentage_l297_297324

theorem salary_increase_percentage (old_salary new_salary : ℕ) (h1 : old_salary = 10000) (h2 : new_salary = 10200) : 
    ((new_salary - old_salary) / old_salary : ℚ) * 100 = 2 := 
by 
  sorry

end salary_increase_percentage_l297_297324


namespace S_10_minus_S_7_l297_297282

-- Define the first term and common difference of the arithmetic sequence
variables (a₁ d : ℕ)

-- Define the arithmetic sequence based on the first term and common difference
def arithmetic_sequence (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions given in the problem
axiom a_5_eq : a₁ + 4 * d = 8
axiom S_3_eq : sum_arithmetic_sequence a₁ 3 = 6

-- The goal: prove that S_10 - S_7 = 48
theorem S_10_minus_S_7 : sum_arithmetic_sequence a₁ 10 - sum_arithmetic_sequence a₁ 7 = 48 :=
sorry

end S_10_minus_S_7_l297_297282


namespace infinite_series_converges_l297_297642

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l297_297642


namespace pairs_of_boys_girls_l297_297736

theorem pairs_of_boys_girls (a_g b_g a_b b_b : ℕ) 
  (h1 : a_b = 3 * a_g)
  (h2 : b_b = 4 * b_g) :
  ∃ c : ℕ, b_b = 7 * b_g :=
sorry

end pairs_of_boys_girls_l297_297736


namespace line_integral_along_path_L_l297_297049

noncomputable def vector_field (ρ φ z : ℝ) : ℝ³ := (4 * ρ * Real.sin φ, z * Real.exp ρ, ρ + φ)

def path_L (ρ : ℝ) : ℝ³ := (ρ, Real.pi / 4, 0)

theorem line_integral_along_path_L :
  ∫ (ρ : ℝ) in 0..1, (4 * ρ * Real.sin (Real.pi / 4)) = Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end line_integral_along_path_L_l297_297049


namespace middle_term_in_arithmetic_sequence_l297_297037

theorem middle_term_in_arithmetic_sequence :
  let a := 3^2 in let c := 3^4 in
  ∃ z : ℤ, (2 * z = a + c) ∧ z = 45 := by
let a := 3^2
let c := 3^4
use (a + c) / 2
split
-- Prove that 2 * ((a + c) / 2) = a + c
sorry
-- Prove that (a + c) / 2 = 45
sorry

end middle_term_in_arithmetic_sequence_l297_297037


namespace new_area_of_rectangle_l297_297138

theorem new_area_of_rectangle (L W : ℝ) (h : L * W = 600) :
  let new_length := 0.8 * L
  let new_width := 1.05 * W
  new_length * new_width = 504 :=
by 
  sorry

end new_area_of_rectangle_l297_297138


namespace find_number_l297_297793

theorem find_number (x : ℝ) (h : 0.2 * x = 0.3 * 120 + 80) : x = 580 :=
by
  sorry

end find_number_l297_297793


namespace sum_f_values_l297_297195

theorem sum_f_values (a b c d e f g : ℕ) 
  (h1: 100 * a * b = 100 * d)
  (h2: c * d * e = 100 * d)
  (h3: b * d * f = 100 * d)
  (h4: b * f = 100)
  (h5: 100 * d = 100) : 
  100 + 50 + 25 + 20 + 10 + 5 + 4 + 2 + 1 = 217 :=
by
  sorry

end sum_f_values_l297_297195


namespace david_has_15_shells_l297_297731

-- Definitions from the conditions
def mia_shells (david_shells : ℕ) : ℕ := 4 * david_shells
def ava_shells (david_shells : ℕ) : ℕ := mia_shells david_shells + 20
def alice_shells (david_shells : ℕ) : ℕ := (ava_shells david_shells) / 2

-- Total number of shells
def total_shells (david_shells : ℕ) : ℕ := david_shells + mia_shells david_shells + ava_shells david_shells + alice_shells david_shells

-- Proving the number of shells David has is 15 given the total number of shells is 195
theorem david_has_15_shells : total_shells 15 = 195 :=
by
  sorry

end david_has_15_shells_l297_297731


namespace minimum_x_plus_3y_l297_297227

theorem minimum_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 3 * x + y) : x + 3 * y ≥ 16 :=
sorry

end minimum_x_plus_3y_l297_297227


namespace smallest_w_correct_l297_297696

-- Define the conditions
def is_factor (a b : ℕ) : Prop := ∃ k, a = b * k

-- Given conditions
def cond1 (w : ℕ) : Prop := is_factor (2^6) (1152 * w)
def cond2 (w : ℕ) : Prop := is_factor (3^4) (1152 * w)
def cond3 (w : ℕ) : Prop := is_factor (5^3) (1152 * w)
def cond4 (w : ℕ) : Prop := is_factor (7^2) (1152 * w)
def cond5 (w : ℕ) : Prop := is_factor (11) (1152 * w)
def is_positive (w : ℕ) : Prop := w > 0

-- The smallest possible value of w given all conditions
def smallest_w : ℕ := 16275

-- Proof statement
theorem smallest_w_correct : 
  ∀ (w : ℕ), cond1 w ∧ cond2 w ∧ cond3 w ∧ cond4 w ∧ cond5 w ∧ is_positive w ↔ w = smallest_w := sorry

end smallest_w_correct_l297_297696


namespace part1_inequality_part2_range_l297_297683

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 1)

-- Part 1: Prove that f(x) ≥ f(0) for all x
theorem part1_inequality : ∀ x : ℝ, f x ≥ f 0 :=
sorry

-- Part 2: Prove that the range of a satisfying 2f(x) ≥ f(a+1) for all x is -4.5 ≤ a ≤ 1.5
theorem part2_range (a : ℝ) (h : ∀ x : ℝ, 2 * f x ≥ f (a + 1)) : -4.5 ≤ a ∧ a ≤ 1.5 :=
sorry

end part1_inequality_part2_range_l297_297683


namespace series_sum_l297_297610

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l297_297610


namespace unique_vector_a_l297_297370

-- Defining the vectors
def vector_a (x y : ℝ) : ℝ × ℝ := (x, y)
def vector_b (x y : ℝ) : ℝ × ℝ := (x^2, y^2)
def vector_c : ℝ × ℝ := (1, 1)
def vector_d : ℝ × ℝ := (2, 2)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The Lean statement to prove
theorem unique_vector_a (x y : ℝ) 
  (h1 : dot_product (vector_a x y) vector_c = 1)
  (h2 : dot_product (vector_b x y) vector_d = 1) : 
  vector_a x y = vector_a (1/2) (1/2) :=
by {
  sorry 
}

end unique_vector_a_l297_297370


namespace sufficient_condition_l297_297239

theorem sufficient_condition (a b : ℝ) (h : b > a ∧ a > 0) : (a + 2) / (b + 2) > a / b :=
by sorry

end sufficient_condition_l297_297239


namespace carpet_area_l297_297819

theorem carpet_area (length_ft : ℕ) (width_ft : ℕ) (ft_per_yd : ℕ) (A_y : ℕ) 
  (h_length : length_ft = 15) (h_width : width_ft = 12) (h_ft_per_yd : ft_per_yd = 9) :
  A_y = (length_ft * width_ft) / ft_per_yd := 
by sorry

#check carpet_area

end carpet_area_l297_297819


namespace sum_of_consecutive_integers_l297_297692

theorem sum_of_consecutive_integers (n a : ℕ) (h₁ : 2 ≤ n) (h₂ : (n * (2 * a + n - 1)) = 36) :
    ∃! (a' n' : ℕ), 2 ≤ n' ∧ (n' * (2 * a' + n' - 1)) = 36 :=
  sorry

end sum_of_consecutive_integers_l297_297692


namespace right_triangle_hypotenuse_l297_297958

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l297_297958


namespace athlete_distance_l297_297333

theorem athlete_distance (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) (d : ℝ)
  (h1 : t = 24)
  (h2 : v_kmh = 30.000000000000004)
  (h3 : v_ms = v_kmh * 1000 / 3600)
  (h4 : d = v_ms * t) :
  d = 200 := 
sorry

end athlete_distance_l297_297333


namespace largest_k_for_right_triangle_l297_297081

noncomputable def k : ℝ := (3 * Real.sqrt 2 - 4) / 2

theorem largest_k_for_right_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2) :
    a^3 + b^3 + c^3 ≥ k * (a + b + c)^3 :=
sorry

end largest_k_for_right_triangle_l297_297081


namespace sqrt_of_sixteen_l297_297476

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_sixteen_l297_297476


namespace sixth_root_of_large_number_l297_297200

theorem sixth_root_of_large_number : 
  ∃ (x : ℕ), x = 51 ∧ x ^ 6 = 24414062515625 :=
by
  sorry

end sixth_root_of_large_number_l297_297200


namespace increasing_power_function_l297_297141

theorem increasing_power_function (m : ℝ) (h_power : m^2 - 1 = 1)
    (h_increasing : ∀ x : ℝ, x > 0 → (m^2 - 1) * m * x^(m-1) > 0) : m = Real.sqrt 2 :=
by
  sorry

end increasing_power_function_l297_297141


namespace correct_statements_l297_297899

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem correct_statements : 
  (∀ x, f (-x) = -f (x)) ∧  -- Statement A
  (∀ x₁ x₂, x₁ + x₂ = Real.pi / 2 → g x₁ = g x₂)  -- Statement C
:= by
  sorry

end correct_statements_l297_297899


namespace robin_packages_gum_l297_297133

/-
Conditions:
1. Robin has 14 packages of candy.
2. There are 6 pieces in each candy package.
3. Robin has 7 additional pieces.
4. Each package of gum contains 6 pieces.

Proof Problem:
Prove that the number of packages of gum Robin has is 15.
-/
theorem robin_packages_gum (candies_packages : ℕ) (pieces_per_candy_package : ℕ)
                          (additional_pieces : ℕ) (pieces_per_gum_package : ℕ) :
  candies_packages = 14 →
  pieces_per_candy_package = 6 →
  additional_pieces = 7 →
  pieces_per_gum_package = 6 →
  (candies_packages * pieces_per_candy_package + additional_pieces) / pieces_per_gum_package = 15 :=
by intros h1 h2 h3 h4; sorry

end robin_packages_gum_l297_297133


namespace range_of_x_when_m_is_4_range_of_m_l297_297512

-- Define the conditions for p and q
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0
def neg_p (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 5
def neg_q (x m : ℝ) : Prop := x ≤ m ∨ x ≥ 3 * m

-- Define the conditions for the values of m
def cond_m_pos (m : ℝ) : Prop := m > 0
def cond_sufficient (m : ℝ) : Prop := cond_m_pos m ∧ m ≤ 2 ∧ 3 * m ≥ 5

-- Problem 1
theorem range_of_x_when_m_is_4 (x : ℝ) : p x ∧ q x 4 → 4 < x ∧ x < 5 :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : (∀ x : ℝ, neg_q x m → neg_p x) → 5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_x_when_m_is_4_range_of_m_l297_297512


namespace no_integer_solution_for_Px_eq_x_l297_297322

theorem no_integer_solution_for_Px_eq_x (P : ℤ → ℤ) (hP_int_coeff : ∀ n : ℤ, ∃ k : ℤ, P n = k * n + k) 
  (hP3 : P 3 = 4) (hP4 : P 4 = 3) :
  ¬ ∃ x : ℤ, P x = x := 
by 
  sorry

end no_integer_solution_for_Px_eq_x_l297_297322


namespace judgments_correct_l297_297678

variables {l m : Line} (a : Plane)

def is_perpendicular (l : Line) (a : Plane) : Prop := -- Definition of perpendicularity between a line and a plane
sorry

def is_parallel (l m : Line) : Prop := -- Definition of parallel lines
sorry

def is_contained_in (m : Line) (a : Plane) : Prop := -- Definition of a line contained in a plane
sorry

theorem judgments_correct 
  (hl : is_perpendicular l a)
  (hm : l ≠ m) :
  (∀ m, is_perpendicular m l → is_parallel m a) ∧ 
  (is_perpendicular m a → is_parallel m l) ∧
  (is_contained_in m a → is_perpendicular m l) ∧
  (is_parallel m l → is_perpendicular m a) :=
sorry

end judgments_correct_l297_297678


namespace sum_of_series_l297_297651

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l297_297651


namespace necessary_but_not_sufficient_condition_l297_297714

theorem necessary_but_not_sufficient_condition (k : ℝ) :
  (1 < k) ∧ (k < 5) → 
  (k - 1 > 0) ∧ (5 - k > 0) ∧ ((k ≠ 3) → (k < 5 ∧ 1 < k)) :=
by
  intro h
  have hk_gt_1 := h.1
  have hk_lt_5 := h.2
  refine ⟨hk_gt_1, hk_lt_5, λ hk_neq_3, ⟨hk_lt_5, hk_gt_1⟩⟩
  sorry

end necessary_but_not_sufficient_condition_l297_297714


namespace right_triangle_hypotenuse_l297_297956

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l297_297956


namespace dolls_given_to_girls_correct_l297_297733

-- Define the total number of toys given
def total_toys_given : ℕ := 403

-- Define the number of toy cars given to boys
def toy_cars_given_to_boys : ℕ := 134

-- Define the number of dolls given to girls
def dolls_given_to_girls : ℕ := total_toys_given - toy_cars_given_to_boys

-- State the theorem to prove the number of dolls given to girls
theorem dolls_given_to_girls_correct : dolls_given_to_girls = 269 := by
  sorry

end dolls_given_to_girls_correct_l297_297733


namespace hypotenuse_length_l297_297946

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l297_297946


namespace roger_trays_l297_297134

theorem roger_trays (trays_per_trip trips trays_first_table : ℕ) 
  (h1 : trays_per_trip = 4) 
  (h2 : trips = 3) 
  (h3 : trays_first_table = 10) : 
  trays_per_trip * trips - trays_first_table = 2 :=
by
  -- Step proofs are omitted
  sorry

end roger_trays_l297_297134


namespace sum_infinite_series_eq_l297_297629

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l297_297629


namespace joan_total_travel_time_l297_297408

-- Definitions based on the conditions in the problem statement
def distance : ℝ := 480 -- miles
def speed : ℝ := 60    -- mph
def lunch_break_time : ℝ := 30 / 60 -- 30 minutes converted to hours
def bathroom_break_time : ℝ := (2 * 15) / 60 -- 2 bathroom breaks of 15 minutes each, converted to hours

-- Theorem to prove the total travel time is 9 hours
theorem joan_total_travel_time : 
  (distance / speed) + lunch_break_time + bathroom_break_time = 9 := 
by
  -- Skipping the proof steps as per the instructions
  sorry

end joan_total_travel_time_l297_297408


namespace Eugene_buys_four_t_shirts_l297_297544

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

end Eugene_buys_four_t_shirts_l297_297544


namespace at_least_one_expression_is_leq_neg_two_l297_297555

variable (a b c : ℝ)

theorem at_least_one_expression_is_leq_neg_two 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1 / b ≤ -2) ∨ (b + 1 / c ≤ -2) ∨ (c + 1 / a ≤ -2) :=
sorry

end at_least_one_expression_is_leq_neg_two_l297_297555


namespace absolute_value_expression_l297_297474

theorem absolute_value_expression : 
  (abs ((-abs (-1 + 2))^2 - 1) = 0) :=
sorry

end absolute_value_expression_l297_297474


namespace P_shape_points_length_10_l297_297163

def P_shape_points (side_length : ℕ) : ℕ :=
  let points_per_side := side_length + 1
  let total_points := points_per_side * 3
  total_points - 2

theorem P_shape_points_length_10 :
  P_shape_points 10 = 31 := 
by 
  sorry

end P_shape_points_length_10_l297_297163


namespace present_age_of_B_l297_297306

theorem present_age_of_B 
    (a b : ℕ) 
    (h1 : a + 10 = 2 * (b - 10)) 
    (h2 : a = b + 12) : 
    b = 42 := by 
  sorry

end present_age_of_B_l297_297306


namespace jia_jia_clover_count_l297_297001

theorem jia_jia_clover_count : ∃ x : ℕ, 3 * x + 4 = 100 ∧ x = 32 := by
  sorry

end jia_jia_clover_count_l297_297001


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l297_297780

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l297_297780


namespace necessary_but_not_sufficient_condition_l297_297715

theorem necessary_but_not_sufficient_condition (k : ℝ) :
  (1 < k) ∧ (k < 5) → 
  (k - 1 > 0) ∧ (5 - k > 0) ∧ ((k ≠ 3) → (k < 5 ∧ 1 < k)) :=
by
  intro h
  have hk_gt_1 := h.1
  have hk_lt_5 := h.2
  refine ⟨hk_gt_1, hk_lt_5, λ hk_neq_3, ⟨hk_lt_5, hk_gt_1⟩⟩
  sorry

end necessary_but_not_sufficient_condition_l297_297715


namespace hypotenuse_length_l297_297949

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l297_297949


namespace intersection_A_B_l297_297217

-- Definition of sets A and B based on given conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2 * x - 3 }
def B : Set ℝ := {y | ∃ x : ℝ, x < 0 ∧ y = x + 1 / x }

-- Proving the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {y | -4 ≤ y ∧ y ≤ -2} := 
by
  sorry

end intersection_A_B_l297_297217


namespace valid_five_digit_integers_l297_297387

/-- How many five-digit positive integers can be formed by arranging the digits 1, 1, 2, 3, 4 so 
that the two 1s are not next to each other -/
def num_valid_arrangements : ℕ :=
  36

theorem valid_five_digit_integers :
  ∃ n : ℕ, n = num_valid_arrangements :=
by
  use 36
  sorry

end valid_five_digit_integers_l297_297387


namespace not_all_mages_are_wizards_l297_297108

variable (M S W : Type → Prop)

theorem not_all_mages_are_wizards
  (h1 : ∃ x, M x ∧ ¬ S x)
  (h2 : ∀ x, M x ∧ W x → S x) :
  ∃ x, M x ∧ ¬ W x :=
sorry

end not_all_mages_are_wizards_l297_297108


namespace am_gm_inequality_l297_297885

theorem am_gm_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - (Real.sqrt (a * b)) ∧ 
  (a + b) / 2 - (Real.sqrt (a * b)) < (a - b)^2 / (8 * b) := 
sorry

end am_gm_inequality_l297_297885


namespace distance_between_lines_l297_297139

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem distance_between_lines : 
  ∀ (x y : ℝ), line1 x y → line2 x y → (|1 - (-1)| / Real.sqrt (1^2 + (-1)^2)) = Real.sqrt 2 := 
by 
  sorry

end distance_between_lines_l297_297139


namespace prove_range_of_a_l297_297519

noncomputable def f (x a : ℝ) : ℝ := (x + a - 1) * Real.exp x

def problem_condition1 (x a : ℝ) : Prop := 
  f x a ≥ (x^2 / 2 + a * x)

def problem_condition2 (x : ℝ) : Prop := 
  x ∈ Set.Ici 0 -- equivalent to [0, +∞)

theorem prove_range_of_a (a : ℝ) :
  (∀ x : ℝ, problem_condition2 x → problem_condition1 x a) → a ∈ Set.Ici 1 :=
sorry

end prove_range_of_a_l297_297519


namespace rectangle_dimensions_l297_297179

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : l = 3 * w) 
  (h2 : 2 * (l + w) = 2 * l * w) : 
  w = 4 / 3 ∧ l = 4 := 
by
  sorry

end rectangle_dimensions_l297_297179


namespace sum_series_l297_297617

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l297_297617


namespace value_of_x_squared_minus_y_squared_l297_297534

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l297_297534


namespace num_winners_is_4_l297_297330

variables (A B C D : Prop)

-- Conditions
axiom h1 : A → B
axiom h2 : B → (C ∨ ¬ A)
axiom h3 : ¬ D → (A ∧ ¬ C)
axiom h4 : D → A

-- Assumptions
axiom hA : A
axiom hD : D

-- Statement to prove
theorem num_winners_is_4 : A ∧ B ∧ C ∧ D :=
by {
  sorry
}

end num_winners_is_4_l297_297330


namespace largest_sum_of_ABCD_l297_297207

theorem largest_sum_of_ABCD :
  ∃ (A B C D : ℕ), 10 ≤ A ∧ A < 100 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100 ∧ 10 ≤ D ∧ D < 100 ∧
  B = 3 * C ∧ D = 2 * B - C ∧ A = B + D ∧ A + B + C + D = 204 :=
by
  sorry

end largest_sum_of_ABCD_l297_297207


namespace intersection_point_exists_circle_equation_standard_form_l297_297369

noncomputable def line1 (x y : ℝ) : Prop := 2 * x + y = 0
noncomputable def line2 (x y : ℝ) : Prop := x + y = 2
noncomputable def line3 (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0

theorem intersection_point_exists :
  ∃ (C : ℝ × ℝ), (line1 C.1 C.2 ∧ line2 C.1 C.2) ∧ C = (-2, 4) :=
sorry

theorem circle_equation_standard_form :
  ∃ (center : ℝ × ℝ) (radius : ℝ), center = (-2, 4) ∧ radius = 3 ∧
  ∀ x y : ℝ, ((x + 2) ^ 2 + (y - 4) ^ 2 = 9) :=
sorry

end intersection_point_exists_circle_equation_standard_form_l297_297369


namespace tony_remaining_money_l297_297031

theorem tony_remaining_money :
  let initial_amount := 20
  let ticket_cost := 8
  let hotdog_cost := 3
  initial_amount - ticket_cost - hotdog_cost = 9 :=
by
  let initial_amount := 20
  let ticket_cost := 8
  let hotdog_cost := 3
  show initial_amount - ticket_cost - hotdog_cost = 9
  sorry

end tony_remaining_money_l297_297031


namespace pizza_eaten_after_six_trips_l297_297041

theorem pizza_eaten_after_six_trips
  (initial_fraction: ℚ)
  (next_fraction : ℚ -> ℚ)
  (S: ℚ)
  (H0: initial_fraction = 1 / 4)
  (H1: ∀ (n: ℕ), next_fraction n = 1 / 2 ^ (n + 2))
  (H2: S = initial_fraction + (next_fraction 1) + (next_fraction 2) + (next_fraction 3) + (next_fraction 4) + (next_fraction 5)):
  S = 125 / 128 :=
by
  sorry

end pizza_eaten_after_six_trips_l297_297041


namespace molecular_weight_of_3_moles_HBrO3_l297_297920

-- Definitions from the conditions
def mol_weight_H : ℝ := 1.01  -- atomic weight of H
def mol_weight_Br : ℝ := 79.90  -- atomic weight of Br
def mol_weight_O : ℝ := 16.00  -- atomic weight of O

-- Definition of molecular weight of HBrO3
def mol_weight_HBrO3 : ℝ := mol_weight_H + mol_weight_Br + 3 * mol_weight_O

-- The goal: The molecular weight of 3 moles of HBrO3 is 386.73 grams
theorem molecular_weight_of_3_moles_HBrO3 : 3 * mol_weight_HBrO3 = 386.73 :=
by
  -- We will insert the proof here later
  sorry

end molecular_weight_of_3_moles_HBrO3_l297_297920


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297310

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem exists_nine_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 8 ≤ 500) ∧ ∀ i ∈ (List.range 9), is_composite (a + i) :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 10 ≤ 500) ∧ ∀ i ∈ (List.range 11), is_composite (a + i) :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297310


namespace paul_walking_time_l297_297721

variable (P : ℕ)

def is_walking_time (P : ℕ) : Prop :=
  P + 7 * (P + 2) = 46

theorem paul_walking_time (h : is_walking_time P) : P = 4 :=
by sorry

end paul_walking_time_l297_297721


namespace tourists_count_l297_297458

theorem tourists_count (n k : ℤ) (h1 : 2 * k % n = 1) (h2 : 3 * k % n = 13) : n = 23 := 
by
-- Proof is omitted
sorry

end tourists_count_l297_297458


namespace largest_integer_solution_l297_297751

theorem largest_integer_solution (x : ℤ) (h : 3 - 2 * x > 0) : x ≤ 1 :=
by sorry

end largest_integer_solution_l297_297751


namespace probability_no_defective_pencils_l297_297107

-- Definitions based on conditions
def total_pencils : ℕ := 11
def defective_pencils : ℕ := 2
def selected_pencils : ℕ := 3

-- Helper function to compute combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The proof statement
theorem probability_no_defective_pencils :
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination (total_pencils - defective_pencils) selected_pencils
  total_ways ≠ 0 → 
  (non_defective_ways / total_ways : ℚ) = 28 / 55 := 
by
  sorry

end probability_no_defective_pencils_l297_297107


namespace book_cost_l297_297595

theorem book_cost (C_1 C_2 : ℝ)
  (h1 : C_1 + C_2 = 420)
  (h2 : C_1 * 0.85 = C_2 * 1.19) :
  C_1 = 245 :=
by
  -- We skip the proof here using sorry.
  sorry

end book_cost_l297_297595


namespace ab_plus_cd_eq_12_l297_297070

theorem ab_plus_cd_eq_12 (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -1) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 5) :
  a * b + c * d = 12 := by
  sorry

end ab_plus_cd_eq_12_l297_297070


namespace range_of_a_minus_abs_b_l297_297855

theorem range_of_a_minus_abs_b (a b : ℝ) (h1: 1 < a) (h2: a < 3) (h3: -4 < b) (h4: b < 2) : 
  -3 < a - |b| ∧ a - |b| < 3 :=
sorry

end range_of_a_minus_abs_b_l297_297855


namespace fraction_broke_off_l297_297057

variable (p p_1 p_2 : ℝ)
variable (k : ℝ)

-- Conditions
def initial_mass : Prop := p_1 + p_2 = p
def value_relation : Prop := p_1^2 + p_2^2 = 0.68 * p^2

-- Goal
theorem fraction_broke_off (h1 : initial_mass p p_1 p_2)
                           (h2 : value_relation p p_1 p_2) :
  (p_2 / p) = 1 / 5 :=
sorry

end fraction_broke_off_l297_297057


namespace divisibility_by_91_l297_297419

theorem divisibility_by_91 (n : ℕ) : ∃ k : ℤ, 9^(n + 2) + 10^(2 * n + 1) = 91 * k := by
  sorry

end divisibility_by_91_l297_297419


namespace parity_of_expression_l297_297391

theorem parity_of_expression {a b c : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : 0 < c) :
  ∃ k : ℕ, 3 ^ a + (b - 1) ^ 2 * c = 2 * k + 1 :=
by
  sorry

end parity_of_expression_l297_297391


namespace rectangle_area_l297_297577

/-- A figure is formed by a triangle and a rectangle, using 60 equal sticks.
Each side of the triangle uses 6 sticks, and each stick measures 5 cm in length.
Prove that the area of the rectangle is 2250 cm². -/
theorem rectangle_area (sticks_total : ℕ) (sticks_per_side_triangle : ℕ) (stick_length_cm : ℕ)
    (sticks_used_triangle : ℕ) (sticks_left_rectangle : ℕ) (sticks_per_width_rectangle : ℕ)
    (width_sticks_rectangle : ℕ) (length_sticks_rectangle : ℕ) (width_cm : ℕ) (length_cm : ℕ)
    (area_rectangle : ℕ) 
    (h_sticks_total : sticks_total = 60)
    (h_sticks_per_side_triangle : sticks_per_side_triangle = 6)
    (h_stick_length_cm : stick_length_cm = 5)
    (h_sticks_used_triangle  : sticks_used_triangle = sticks_per_side_triangle * 3)
    (h_sticks_left_rectangle : sticks_left_rectangle = sticks_total - sticks_used_triangle)
    (h_sticks_per_width_rectangle : sticks_per_width_rectangle = 6 * 2) 
    (h_width_sticks_rectangle : width_sticks_rectangle = 6)
    (h_length_sticks_rectangle : length_sticks_rectangle = (sticks_left_rectangle - sticks_per_width_rectangle) / 2)
    (h_width_cm : width_cm = width_sticks_rectangle * stick_length_cm)
    (h_length_cm : length_cm = length_sticks_rectangle * stick_length_cm)
    (h_area_rectangle : area_rectangle = width_cm * length_cm) :
    area_rectangle = 2250 := 
by sorry

end rectangle_area_l297_297577


namespace fraction_comparison_l297_297161

theorem fraction_comparison :
  (1998:ℝ) ^ 2000 / (2000:ℝ) ^ 1998 > (1997:ℝ) ^ 1999 / (1999:ℝ) ^ 1997 :=
by sorry

end fraction_comparison_l297_297161


namespace simplify_trig_expression_l297_297886

theorem simplify_trig_expression :
  (sin (15 * real.pi / 180) + sin (30 * real.pi / 180) + sin (45 * real.pi / 180) + 
   sin (60 * real.pi / 180) + sin (75 * real.pi / 180)) / 
  (cos (10 * real.pi / 180) * cos (20 * real.pi / 180) * cos (30 * real.pi / 180)) = 
  (√2 * (4 * (cos (22.5 * real.pi / 180)) * (cos (7.5 * real.pi / 180)) + 1)) / 
  (2 * (cos (10 * real.pi / 180)) * (cos (20 * real.pi / 180)) * (cos (30 * real.pi / 180))) :=
sorry

end simplify_trig_expression_l297_297886


namespace int_solutions_fraction_l297_297505

theorem int_solutions_fraction :
  ∀ n : ℤ, (∃ k : ℤ, (n - 2) / (n + 1) = k) ↔ n = 0 ∨ n = -2 ∨ n = 2 ∨ n = -4 :=
by
  intro n
  sorry

end int_solutions_fraction_l297_297505


namespace quadratic_minimum_value_l297_297832

theorem quadratic_minimum_value :
  ∀ (x : ℝ), (x - 1)^2 + 2 ≥ 2 :=
by
  sorry

end quadratic_minimum_value_l297_297832


namespace hyperbola_center_l297_297824

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0) : 
  x = 2 ∧ y = 4 :=
sorry

end hyperbola_center_l297_297824


namespace rhombus_area_is_160_l297_297048

-- Define the values of the diagonals
def d1 : ℝ := 16
def d2 : ℝ := 20

-- Define the formula for the area of the rhombus
noncomputable def area_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- State the theorem to be proved
theorem rhombus_area_is_160 :
  area_rhombus d1 d2 = 160 :=
by
  sorry

end rhombus_area_is_160_l297_297048


namespace uniqueSumEqualNumber_l297_297590

noncomputable def sumPreceding (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem uniqueSumEqualNumber :
  ∃! n : ℕ, sumPreceding n = n := by
  sorry

end uniqueSumEqualNumber_l297_297590


namespace haylee_has_36_guppies_l297_297853

variables (H J C N : ℝ)
variables (total_guppies : ℝ := 84)

def jose_has_half_of_haylee := J = H / 2
def charliz_has_third_of_jose := C = J / 3
def nicolai_has_four_times_charliz := N = 4 * C
def total_guppies_eq_84 := H + J + C + N = total_guppies

theorem haylee_has_36_guppies 
  (hJ : jose_has_half_of_haylee H J)
  (hC : charliz_has_third_of_jose J C)
  (hN : nicolai_has_four_times_charliz C N)
  (htotal : total_guppies_eq_84 H J C N) :
  H = 36 := 
  sorry

end haylee_has_36_guppies_l297_297853


namespace max_z_value_l297_297374

theorem max_z_value (x y z : ℝ) (h : x + y + z = 3) (h' : x * y + y * z + z * x = 2) : z ≤ 5 / 3 :=
  sorry


end max_z_value_l297_297374


namespace field_length_l297_297929

-- Definitions of the conditions
def pond_area : ℝ := 25  -- area of the square pond
def width_to_length_ratio (w l : ℝ) : Prop := l = 2 * w  -- length is double the width
def pond_to_field_ratio (pond_area field_area : ℝ) : Prop := pond_area = (1/8) * field_area  -- pond area is 1/8 of field area

-- Statement to prove
theorem field_length (w l : ℝ) (h1 : width_to_length_ratio w l) (h2 : pond_to_field_ratio pond_area (l * w)) : l = 20 :=
by sorry

end field_length_l297_297929


namespace definite_integral_solution_l297_297472

noncomputable def integral_problem : ℝ := 
  by 
    sorry

theorem definite_integral_solution :
  integral_problem = (1/6 : ℝ) + Real.log 2 - Real.log 3 := 
by
  sorry

end definite_integral_solution_l297_297472


namespace taco_variants_count_l297_297587

theorem taco_variants_count :
  let toppings := 8
  let meat_variants := 3
  let shell_variants := 2
  2 ^ toppings * meat_variants * shell_variants = 1536 := by
sorry

end taco_variants_count_l297_297587


namespace square_perimeter_l297_297269

-- Define the area of the square
def square_area := 720

-- Define the side length of the square
noncomputable def side_length := Real.sqrt square_area

-- Define the perimeter of the square
noncomputable def perimeter := 4 * side_length

-- Statement: Prove that the perimeter is 48 * sqrt(5)
theorem square_perimeter : perimeter = 48 * Real.sqrt 5 :=
by
  -- The proof is omitted as instructed
  sorry

end square_perimeter_l297_297269


namespace solution_to_problem_l297_297663

-- Definitions of conditions
def condition_1 (x : ℝ) : Prop := 2 * x - 6 ≠ 0
def condition_2 (x : ℝ) : Prop := 5 ≤ x / (2 * x - 6) ∧ x / (2 * x - 6) < 10

-- Definition of solution set
def solution_set (x : ℝ) : Prop := 3 < x ∧ x < 60 / 19

-- The theorem to be proven
theorem solution_to_problem (x : ℝ) (h1 : condition_1 x) : condition_2 x ↔ solution_set x :=
by sorry

end solution_to_problem_l297_297663


namespace inequality_solution_set_l297_297582

theorem inequality_solution_set (x : ℝ) : (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end inequality_solution_set_l297_297582


namespace even_function_a_equals_one_l297_297682

theorem even_function_a_equals_one 
  (a : ℝ) 
  (h : ∀ x : ℝ, 2^(-x) + a * 2^x = 2^x + a * 2^(-x)) : 
  a = 1 := 
by
  sorry

end even_function_a_equals_one_l297_297682


namespace cells_that_remain_open_l297_297318

/-- A cell q remains open after iterative toggling if and only if it is a perfect square. -/
theorem cells_that_remain_open (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k ^ 2 = n) ↔ 
  (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ m : ℕ, i = m ^ 2)) := 
sorry

end cells_that_remain_open_l297_297318


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l297_297775

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l297_297775


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297313

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297313


namespace mary_younger_than_albert_l297_297810

variable (A M B : ℕ)

noncomputable def albert_age := 4 * B
noncomputable def mary_age := A / 2
noncomputable def betty_age := 4

theorem mary_younger_than_albert (h1 : A = 2 * M) (h2 : A = 4 * 4) (h3 : 4 = 4) :
  A - M = 8 :=
sorry

end mary_younger_than_albert_l297_297810


namespace cost_of_childrens_ticket_l297_297291

theorem cost_of_childrens_ticket (C : ℝ) : (16 * C + (5 * 5.50) = 83.50) → C = 3.50 :=
by
  intro h
  sorry

end cost_of_childrens_ticket_l297_297291


namespace minimum_AB_l297_297868

noncomputable def shortest_AB (a : ℝ) : ℝ :=
  let x := (Real.sqrt 3) / 4 * a
  x

theorem minimum_AB (a : ℝ) : ∃ x, (x = (Real.sqrt 3) / 4 * a) ∧ ∀ y, (y = (Real.sqrt 3) / 4 * a) → shortest_AB a = x :=
by
  sorry

end minimum_AB_l297_297868


namespace figure_100_squares_l297_297865

-- Define the initial conditions as given in the problem
def squares_in_figure (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 11
  | 2 => 25
  | 3 => 45
  | _ => sorry

-- Define the quadratic formula assumed from the problem conditions
def quadratic_formula (n : ℕ) : ℕ :=
  3 * n^2 + 5 * n + 3

-- Theorem: For figure 100, the number of squares is 30503
theorem figure_100_squares :
  squares_in_figure 100 = quadratic_formula 100 :=
by
  sorry

end figure_100_squares_l297_297865


namespace total_time_spent_l297_297417

-- Define the total time for one shoe
def time_per_shoe (time_buckle: ℕ) (time_heel: ℕ) : ℕ :=
  time_buckle + time_heel

-- Conditions
def time_buckle : ℕ := 5
def time_heel : ℕ := 10
def number_of_shoes : ℕ := 2

-- The proof problem statement
theorem total_time_spent :
  (time_per_shoe time_buckle time_heel) * number_of_shoes = 30 :=
by
  sorry

end total_time_spent_l297_297417


namespace cricket_team_members_l297_297913

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℕ := 27)
  (wk_age : ℕ := captain_age + 1)
  (total_avg_age : ℕ := 23)
  (remaining_avg_age : ℕ := total_avg_age - 1)
  (total_age : ℕ := n * total_avg_age)
  (captain_and_wk_age : ℕ := captain_age + wk_age)
  (remaining_age : ℕ := (n - 2) * remaining_avg_age) : n = 11 := 
by
  sorry

end cricket_team_members_l297_297913


namespace karen_average_speed_correct_l297_297551

def karen_time_duration : ℚ := (22 : ℚ) / 3
def karen_distance : ℚ := 230

def karen_average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed_correct :
  karen_average_speed karen_distance karen_time_duration = (31 + 4/11 : ℚ) :=
by
  sorry

end karen_average_speed_correct_l297_297551


namespace right_triangle_hypotenuse_length_l297_297963

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l297_297963


namespace find_slope_l297_297362

theorem find_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : ∃ m : ℝ, m = -4/7 ∧ (∀ x, y = m * x + 4) := 
by
  sorry

end find_slope_l297_297362


namespace ball_hits_ground_at_time_l297_297017

theorem ball_hits_ground_at_time :
  ∃ t : ℚ, -9.8 * t^2 + 5.6 * t + 10 = 0 ∧ t = 131 / 98 :=
by
  sorry

end ball_hits_ground_at_time_l297_297017


namespace range_of_a_l297_297681

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → (x^2 + 2*x + a) / x > 0) ↔ a > -3 :=
by
  sorry

end range_of_a_l297_297681


namespace sum_of_first_four_terms_of_arithmetic_sequence_l297_297431

theorem sum_of_first_four_terms_of_arithmetic_sequence
  (a d : ℤ)
  (h1 : a + 4 * d = 10)  -- Condition for the fifth term
  (h2 : a + 5 * d = 14)  -- Condition for the sixth term
  (h3 : a + 6 * d = 18)  -- Condition for the seventh term
  : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 0 :=  -- Prove the sum of the first four terms is 0
by
  sorry

end sum_of_first_four_terms_of_arithmetic_sequence_l297_297431


namespace infinite_series_converges_l297_297641

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l297_297641


namespace total_triangles_correct_l297_297494

-- Define the rectangle and additional constructions
structure Rectangle :=
  (A B C D : Type)
  (midpoint_AB midpoint_BC midpoint_CD midpoint_DA : Type)
  (AC BD diagonals : Type)

-- Hypothesize the structure
variables (rect : Rectangle)

-- Define the number of triangles
def number_of_triangles (r : Rectangle) : Nat := 16

-- The theorem statement
theorem total_triangles_correct : number_of_triangles rect = 16 :=
by
  sorry

end total_triangles_correct_l297_297494


namespace quadratic_function_value_l297_297730

theorem quadratic_function_value (x1 x2 a b : ℝ) (h1 : a ≠ 0)
  (h2 : 2012 = a * x1^2 + b * x1 + 2009)
  (h3 : 2012 = a * x2^2 + b * x2 + 2009) :
  (a * (x1 + x2)^2 + b * (x1 + x2) + 2009) = 2009 :=
by
  sorry

end quadratic_function_value_l297_297730


namespace min_t_value_l297_297847

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem min_t_value : 
  ∀ (x y : ℝ), x ∈ Set.Icc (-3 : ℝ) (2 : ℝ) → y ∈ Set.Icc (-3 : ℝ) (2 : ℝ)
  → |f (x) - f (y)| ≤ 20 :=
by
  sorry

end min_t_value_l297_297847


namespace smallest_n_for_terminating_decimal_l297_297784

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m, m < n → (∃ k1 k2 : ℕ, (m + 150 = 2^k1 * 5^k2 ∧ m > 0) → false)) ∧ (∃ k1 k2 : ℕ, (n + 150 = 2^k1 * 5^k2) ∧ n > 0) :=
sorry

end smallest_n_for_terminating_decimal_l297_297784


namespace smallest_possible_area_square_l297_297670

theorem smallest_possible_area_square : 
  ∃ (c : ℝ), (∀ (x y : ℝ), ((y = 3 * x - 20) ∨ (y = x^2)) ∧ 
      (10 * (9 + 4 * c) = ((c + 20) / Real.sqrt 10) ^ 2) ∧ 
      (c = 80) ∧ 
      (10 * (9 + 4 * c) = 3290)) :=
by {
  use 80,
  sorry
}

end smallest_possible_area_square_l297_297670


namespace largest_three_digit_n_l297_297159

-- Define the conditions and the proof statement
theorem largest_three_digit_n (n : ℕ) : 
  (70 * n ≡ 210 [MOD 350]) ∧ (n ≤ 999) ∧ (n ≥ 100) → n = 998 :=
begin
  -- Sorry as a placeholder for the proof
  sorry,
end

end largest_three_digit_n_l297_297159


namespace simplify_division_l297_297010

theorem simplify_division :
  (27 * 10^9) / (9 * 10^5) = 30000 :=
  sorry

end simplify_division_l297_297010


namespace sin_double_angle_l297_297674

theorem sin_double_angle (x : ℝ) (h : Real.tan (π / 4 - x) = 2) : Real.sin (2 * x) = -3 / 5 :=
by
  sorry

end sin_double_angle_l297_297674


namespace hypotenuse_length_l297_297972

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l297_297972


namespace fruit_total_l297_297286

noncomputable def fruit_count_proof : Prop :=
  let oranges := 6
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches = 28

theorem fruit_total : fruit_count_proof :=
by {
  sorry
}

end fruit_total_l297_297286


namespace sqrt_sixteen_is_four_l297_297485

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l297_297485


namespace tangent_lines_through_point_l297_297206

theorem tangent_lines_through_point {x y : ℝ} (h_circle : (x-1)^2 + (y-1)^2 = 1)
  (h_point : ∀ (x y: ℝ), (x, y) = (2, 4)) :
  (x = 2 ∨ 4 * x - 3 * y + 4 = 0) :=
sorry

end tangent_lines_through_point_l297_297206


namespace max_cross_section_area_l297_297802

noncomputable def prism_cross_section_area : ℝ :=
  let z_axis_parallel := true
  let square_base := 8
  let plane := ∀ x y z, 3 * x - 5 * y + 2 * z = 20
  121.6

theorem max_cross_section_area :
  prism_cross_section_area = 121.6 :=
sorry

end max_cross_section_area_l297_297802


namespace extreme_point_properties_l297_297567

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - 2 * a * x)

theorem extreme_point_properties (a x₁ x₂ : ℝ) (h₁ : 0 < a) (h₂ : a < 1 / 4) 
  (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) (h₅ : x₁ < x₂) :
  f x₁ a < 0 ∧ f x₂ a > (-1 / 2) := 
sorry

end extreme_point_properties_l297_297567


namespace only_zero_function_satisfies_conditions_l297_297443

def is_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n > m → f n ≥ f m

def satisfies_functional_equation (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, f (n * m) = f n + f m

theorem only_zero_function_satisfies_conditions :
  ∀ f : ℕ → ℕ, 
  (is_increasing f) ∧ (satisfies_functional_equation f) → (∀ n : ℕ, f n = 0) :=
by
  sorry

end only_zero_function_satisfies_conditions_l297_297443


namespace number_of_special_m_gons_correct_l297_297851

noncomputable def number_of_special_m_gons (m n : ℕ) (h1 : 4 < m) (h2 : m < n) : ℕ :=
  (2 * n + 1) * (Nat.choose n (m - 1) + Nat.choose (n + 1) (m - 1))

theorem number_of_special_m_gons_correct (m n : ℕ) (h1 : 4 < m) (h2 : m < n) :
  number_of_special_m_gons m n h1 h2 =
  (2 * n + 1) * (Nat.choose n (m - 1) + Nat.choose (n + 1) (m - 1)) :=
by
  unfold number_of_special_m_gons
  sorry

end number_of_special_m_gons_correct_l297_297851


namespace revenue_times_l297_297789

noncomputable def revenue_ratio (D : ℝ) : ℝ :=
  let revenue_Nov := (2 / 5) * D
  let revenue_Jan := (1 / 3) * revenue_Nov
  let average := (revenue_Nov + revenue_Jan) / 2
  D / average

theorem revenue_times (D : ℝ) (hD : D ≠ 0) : revenue_ratio D = 3.75 :=
by
  -- skipped proof
  sorry

end revenue_times_l297_297789


namespace holds_for_even_positive_l297_297394

variable {n : ℕ}
variable (p : ℕ → Prop)

-- Conditions
axiom base_case : p 2
axiom inductive_step : ∀ k, p k → p (k + 2)

-- Theorem to prove
theorem holds_for_even_positive (n : ℕ) (h : n > 0) (h_even : n % 2 = 0) : p n :=
sorry

end holds_for_even_positive_l297_297394


namespace probability_rachel_robert_in_picture_l297_297004

theorem probability_rachel_robert_in_picture :
  let lap_rachel := 120 -- Rachel's lap time in seconds
  let lap_robert := 100 -- Robert's lap time in seconds
  let duration := 900 -- 15 minutes in seconds
  let picture_duration := 60 -- Picture duration in seconds
  let one_third_rachel := lap_rachel / 3 -- One third of Rachel's lap time
  let one_third_robert := lap_robert / 3 -- One third of Robert's lap time
  let rachel_in_window_start := 20 -- Rachel in the window from 20 to 100s
  let rachel_in_window_end := 100
  let robert_in_window_start := 0 -- Robert in the window from 0 to 66.66s
  let robert_in_window_end := 66.66
  let overlap_start := max rachel_in_window_start robert_in_window_start -- The start of overlap
  let overlap_end := min rachel_in_window_end robert_in_window_end -- The end of overlap
  let overlap_duration := overlap_end - overlap_start -- Duration of the overlap
  let probability := overlap_duration / picture_duration -- Probability of both in the picture
  probability = 46.66 / 60 := sorry

end probability_rachel_robert_in_picture_l297_297004


namespace number_of_monomials_l297_297392

-- Define the degree of a monomial
def degree (x_deg y_deg z_deg : ℕ) : ℕ := x_deg + y_deg + z_deg

-- Define a condition for the coefficient of the monomial
def monomial_coefficient (coeff : ℤ) : Prop := coeff = -3

-- Define a condition for the presence of the variables x, y, z
def contains_vars (x_deg y_deg z_deg : ℕ) : Prop := x_deg ≥ 1 ∧ y_deg ≥ 1 ∧ z_deg ≥ 1

-- Define the proof for the number of such monomials
theorem number_of_monomials :
  ∃ (x_deg y_deg z_deg : ℕ), contains_vars x_deg y_deg z_deg ∧ monomial_coefficient (-3) ∧ degree x_deg y_deg z_deg = 5 ∧ (6 = 6) :=
by
  sorry

end number_of_monomials_l297_297392


namespace reducible_fraction_least_n_l297_297082

theorem reducible_fraction_least_n : ∃ n : ℕ, (0 < n) ∧ (n-15 > 0) ∧ (gcd (n-15) (3*n+4) > 1) ∧
  (∀ m : ℕ, (0 < m) ∧ (m-15 > 0) ∧ (gcd (m-15) (3*m+4) > 1) → n ≤ m) :=
by
  sorry

end reducible_fraction_least_n_l297_297082


namespace odd_function_h_l297_297529

noncomputable def f (x h k : ℝ) : ℝ := Real.log (abs ((1 / (x + 1)) + k)) + h

theorem odd_function_h (k : ℝ) (h : ℝ) (H : ∀ x : ℝ, x ≠ -1 → f x h k = -f (-x) h k) :
  h = Real.log 2 :=
sorry

end odd_function_h_l297_297529


namespace macy_running_goal_l297_297124

/-- Macy's weekly running goal is 24 miles. She runs 3 miles per day. Calculate the miles 
    she has left to run after 6 days to meet her goal. --/
theorem macy_running_goal (miles_per_week goal_per_week : ℕ) (miles_per_day: ℕ) (days_run: ℕ) 
  (h1 : miles_per_week = 24) (h2 : miles_per_day = 3) (h3 : days_run = 6) : 
  miles_per_week - miles_per_day * days_run = 6 := 
  by 
    rw [h1, h2, h3]
    exact Nat.sub_eq_of_eq_add (by norm_num)

end macy_running_goal_l297_297124


namespace carbonate_ions_in_Al2_CO3_3_l297_297507

theorem carbonate_ions_in_Al2_CO3_3 (total_weight : ℕ) (formula : String) 
  (molecular_weight : ℕ) (ions_in_formula : String) : 
  formula = "Al2(CO3)3" → molecular_weight = 234 → ions_in_formula = "CO3" → total_weight = 3 := 
by
  intros formula_eq weight_eq ions_eq
  sorry

end carbonate_ions_in_Al2_CO3_3_l297_297507


namespace price_equation_l297_297936

variable (x : ℝ)

def first_discount (x : ℝ) : ℝ := x - 5

def second_discount (price_after_first_discount : ℝ) : ℝ := 0.8 * price_after_first_discount

theorem price_equation
  (hx : second_discount (first_discount x) = 60) :
  0.8 * (x - 5) = 60 := by
  sorry

end price_equation_l297_297936


namespace even_perfect_square_factors_count_l297_297690

theorem even_perfect_square_factors_count :
  let x := 2^6 * 5^3 * 7^8,
      factors_count := (({i | 0 ≤ i ∧ i ≤ 6 ∧ even i ∧ 2 ≤ i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 3 ∧ even i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 8 ∧ even i}.card)) in
  factors_count = 30 :=
by
  let x := 2^6 * 5^3 * 7^8,
      factors_count := (({i | 0 ≤ i ∧ i ≤ 6 ∧ even i ∧ 2 ≤ i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 3 ∧ even i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 8 ∧ even i}.card))
  have h_factors_count : factors_count = 30 := sorry
  exact h_factors_count

end even_perfect_square_factors_count_l297_297690


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l297_297770

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l297_297770


namespace deputy_more_enemies_than_friends_l297_297237

theorem deputy_more_enemies_than_friends (deputies : Type) 
  (friendship hostility indifference : deputies → deputies → Prop)
  (h_symm_friend : ∀ (a b : deputies), friendship a b → friendship b a)
  (h_symm_hostile : ∀ (a b : deputies), hostility a b → hostility b a)
  (h_symm_indiff : ∀ (a b : deputies), indifference a b → indifference b a)
  (h_enemy_exists : ∀ (d : deputies), ∃ (e : deputies), hostility d e)
  (h_principle : ∀ (a b c : deputies), hostility a b → friendship b c → hostility a c) :
  ∃ (d : deputies), ∃ (f e : ℕ), f < e :=
sorry

end deputy_more_enemies_than_friends_l297_297237


namespace triangle_ABC_is_right_triangle_l297_297873

-- Define the triangle and the given conditions
variable (a b c : ℝ)
variable (h1 : a + c = 2*b)
variable (h2 : c - a = 1/2*b)

-- State the problem
theorem triangle_ABC_is_right_triangle : c^2 = a^2 + b^2 :=
by
  sorry

end triangle_ABC_is_right_triangle_l297_297873


namespace hotel_accommodation_l297_297459

theorem hotel_accommodation :
  ∃ (arrangements : ℕ), arrangements = 27 :=
by
  -- problem statement
  let triple_room := 1
  let double_room := 1
  let single_room := 1
  let adults := 3
  let children := 2
  
  -- use the given conditions and properties of combinations to calculate arrangements
  sorry

end hotel_accommodation_l297_297459


namespace hypotenuse_length_l297_297974

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l297_297974


namespace trajectory_midpoint_l297_297697

theorem trajectory_midpoint {x y : ℝ} (hx : 2 * y + 1 = 2 * (2 * x)^2 + 1) :
  y = 4 * x^2 := 
by sorry

end trajectory_midpoint_l297_297697


namespace compare_abc_l297_297099

open Real

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Assuming the conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom derivative : ∀ x : ℝ, f' x = deriv f x
axiom monotonicity_condition : ∀ x > 0, x * f' x < f x

-- Definitions of a, b, and c
noncomputable def a := 2 * f (1 / 2)
noncomputable def b := - (1 / 2) * f (-2)
noncomputable def c := - (1 / log 2) * f (log (1 / 2))

theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l297_297099


namespace mutually_exclusive_necessary_for_complementary_l297_297525

variables {Ω : Type} -- Define the sample space type
variables (A1 A2 : Ω → Prop) -- Define the events as predicates over the sample space

-- Define mutually exclusive events
def mutually_exclusive (A1 A2 : Ω → Prop) : Prop :=
∀ ω, A1 ω → ¬ A2 ω

-- Define complementary events
def complementary (A1 A2 : Ω → Prop) : Prop :=
∀ ω, (A1 ω ↔ ¬ A2 ω)

-- The proof problem: Statement 1 is a necessary but not sufficient condition for Statement 2
theorem mutually_exclusive_necessary_for_complementary (A1 A2 : Ω → Prop) :
  (mutually_exclusive A1 A2) → (complementary A1 A2) → (mutually_exclusive A1 A2) ∧ ¬ (complementary A1 A2 → mutually_exclusive A1 A2) :=
sorry

end mutually_exclusive_necessary_for_complementary_l297_297525


namespace total_amount_paid_after_discount_l297_297060

-- Define the given conditions
def marked_price_per_article : ℝ := 10
def discount_percentage : ℝ := 0.60
def number_of_articles : ℕ := 2

-- Proving the total amount paid
theorem total_amount_paid_after_discount : 
  (marked_price_per_article * number_of_articles) * (1 - discount_percentage) = 8 := by
  sorry

end total_amount_paid_after_discount_l297_297060


namespace problem_l297_297596

theorem problem (k : ℕ) (h1 : 30^k ∣ 929260) : 3^k - k^3 = 2 :=
sorry

end problem_l297_297596


namespace oppose_estimation_l297_297759

-- Define the conditions
def survey_total : ℕ := 50
def favorable_attitude : ℕ := 15
def total_population : ℕ := 9600

-- Calculate the proportion opposed
def proportion_opposed : ℚ := (survey_total - favorable_attitude) / survey_total

-- Define the statement to be proved
theorem oppose_estimation : 
  proportion_opposed * total_population = 6720 := by
  sorry

end oppose_estimation_l297_297759


namespace find_x_l297_297215

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
sorry

end find_x_l297_297215


namespace percent_of_g_is_a_l297_297047

theorem percent_of_g_is_a (a b c d e f g : ℤ) (h1 : (a + b + c + d + e + f + g) / 7 = 9)
: (a / g) * 100 = 50 := 
sorry

end percent_of_g_is_a_l297_297047


namespace absolute_value_half_l297_297530

theorem absolute_value_half (a : ℝ) (h : |a| = 1/2) : a = 1/2 ∨ a = -1/2 :=
sorry

end absolute_value_half_l297_297530


namespace necessary_but_not_sufficient_l297_297717

variable (k : ℝ)

def is_ellipse : Prop := 
  (k > 1) ∧ (k < 5) ∧ (k ≠ 3)

theorem necessary_but_not_sufficient :
  (1 < k) ∧ (k < 5) → is_ellipse k :=
by sorry

end necessary_but_not_sufficient_l297_297717


namespace sqrt_sixteen_equals_four_l297_297481

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l297_297481


namespace work_rate_proof_l297_297923

theorem work_rate_proof (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : C = 1 / 60) : 
  1 / (A + B + C) = 12 :=
by
  sorry

end work_rate_proof_l297_297923


namespace total_weight_tommy_ordered_l297_297030

theorem total_weight_tommy_ordered :
  let apples := 3
  let oranges := 1
  let grapes := 3
  let strawberries := 3
  apples + oranges + grapes + strawberries = 10 := by
  sorry

end total_weight_tommy_ordered_l297_297030


namespace sum_infinite_partial_fraction_l297_297637

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l297_297637


namespace greatest_integer_exceeds_100x_l297_297415

noncomputable def x : ℝ :=
  (∑ n in Finset.range 36, Real.cos (n+1) * (Real.pi / 180)) /
  (∑ n in Finset.range 36, Real.sin (n+1) * (Real.pi / 180))

theorem greatest_integer_exceeds_100x : 
  ⌊100 * x⌋ = 273 :=
sorry

end greatest_integer_exceeds_100x_l297_297415


namespace consecutive_vertices_product_l297_297652

theorem consecutive_vertices_product (n : ℕ) (hn : n = 90) :
  ∃ (i : ℕ), 1 ≤ i ∧ i ≤ n ∧ ((i * (i % n + 1)) ≥ 2014) := 
sorry

end consecutive_vertices_product_l297_297652


namespace grassy_width_excluding_path_l297_297944

theorem grassy_width_excluding_path
  (l : ℝ) (w : ℝ) (p : ℝ)
  (h1: l = 110) (h2: w = 65) (h3: p = 2.5) :
  w - 2 * p = 60 :=
by
  sorry

end grassy_width_excluding_path_l297_297944


namespace value_of_expression_l297_297038

theorem value_of_expression : 48^2 - 2 * 48 * 3 + 3^2 = 2025 :=
by
  sorry

end value_of_expression_l297_297038


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297315

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d > 1 ∧ d < n

def consecutiveComposites (start n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < n → isComposite (start + i)

theorem exists_nine_consecutive_composites :
  ∃ start, start + 8 ≤ 500 ∧ consecutiveComposites start 9 :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ start, start + 10 ≤ 500 ∧ consecutiveComposites start 11 :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297315


namespace system_solution_find_a_l297_297171

theorem system_solution (x y : ℝ) (a : ℝ) :
  (|16 + 6 * x - x ^ 2 - y ^ 2| + |6 * x| = 16 + 12 * x - x ^ 2 - y ^ 2)
  ∧ ((a + 15) * y + 15 * x - a = 0) →
  ( (x - 3) ^ 2 + y ^ 2 ≤ 25 ∧ x ≥ 0 ) :=
sorry

theorem find_a (a : ℝ) :
  ∃ (x y : ℝ), 
  ((a + 15) * y + 15 * x - a = 0 ∧ x ≥ 0 ∧ (x - 3) ^ 2 + y ^ 2 ≤ 25) ↔ 
  (a = -20 ∨ a = -12) :=
sorry

end system_solution_find_a_l297_297171


namespace phoebe_age_l297_297232

theorem phoebe_age (P : ℕ) (h₁ : ∀ P, 60 = 4 * (P + 5)) (h₂: 55 + 5 = 60) : P = 10 := 
by
  have h₃ : 60 = 4 * (P + 5) := h₁ P
  sorry

end phoebe_age_l297_297232


namespace find_a2_plus_b2_l297_297371

theorem find_a2_plus_b2
  (a b : ℝ)
  (h1 : a^3 - 3 * a * b^2 = 39)
  (h2 : b^3 - 3 * a^2 * b = 26) :
  a^2 + b^2 = 13 :=
sorry

end find_a2_plus_b2_l297_297371


namespace eight_base_subtraction_l297_297341

theorem eight_base_subtraction : ∀ (a b : ℕ), a = 52 → b = 27 → (a - b = 25 : Zmod 8) := by
  intros a b ha hb
  rw [ha, hb]
  norm_num
  sorry

end eight_base_subtraction_l297_297341


namespace circle_sector_cones_sum_radii_l297_297937

theorem circle_sector_cones_sum_radii :
  let r := 5
  let a₁ := 1
  let a₂ := 2
  let a₃ := 3
  let total_area := π * r * r
  let θ₁ := (a₁ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₂ := (a₂ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₃ := (a₃ / (a₁ + a₂ + a₃)) * 2 * π
  let r₁ := (a₁ / (a₁ + a₂ + a₃)) * r
  let r₂ := (a₂ / (a₁ + a₂ + a₃)) * r
  let r₃ := (a₃ / (a₁ + a₂ + a₃)) * r
  r₁ + r₂ + r₃ = 5 :=
by {
  sorry
}

end circle_sector_cones_sum_radii_l297_297937


namespace fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l297_297671

open Complex

def inFourthQuadrant (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) > 0 ∧ (m^2 + 3*m - 28) < 0

def onNegativeHalfXAxis (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) < 0 ∧ (m^2 + 3*m - 28) = 0

def inUpperHalfPlaneIncludingRealAxis (m : ℝ) : Prop :=
  (m^2 + 3*m - 28) ≥ 0

theorem fourth_quadrant_for_m (m : ℝ) :
  (-7 < m ∧ m < 3) ↔ inFourthQuadrant m := 
sorry

theorem negative_half_x_axis_for_m (m : ℝ) :
  (m = 4) ↔ onNegativeHalfXAxis m :=
sorry

theorem upper_half_plane_for_m (m : ℝ) :
  (m ≤ -7 ∨ m ≥ 4) ↔ inUpperHalfPlaneIncludingRealAxis m :=
sorry

end fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l297_297671


namespace integral_solutions_l297_297499

theorem integral_solutions (a b c : ℤ) (h : a^2 + b^2 + c^2 = a^2 * b^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end integral_solutions_l297_297499


namespace find_last_number_l297_297928

theorem find_last_number (A B C D : ℝ) (h1 : A + B + C = 18) (h2 : B + C + D = 9) (h3 : A + D = 13) : D = 2 :=
by
sorry

end find_last_number_l297_297928


namespace total_votes_cast_l297_297235

variable (total_votes : ℕ)
variable (emily_votes : ℕ)
variable (emily_share : ℚ := 4 / 15)
variable (dexter_share : ℚ := 1 / 3)

theorem total_votes_cast :
  emily_votes = 48 → 
  emily_share * total_votes = emily_votes → 
  total_votes = 180 := by
  intro h_emily_votes
  intro h_emily_share
  sorry

end total_votes_cast_l297_297235


namespace max_k_l297_297382

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k (k : ℤ) : (∀ x : ℝ, 1 < x → f x - k * x + k > 0) → k ≤ 3 :=
by
  sorry

end max_k_l297_297382


namespace fruit_total_l297_297287

noncomputable def fruit_count_proof : Prop :=
  let oranges := 6
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches = 28

theorem fruit_total : fruit_count_proof :=
by {
  sorry
}

end fruit_total_l297_297287


namespace turtles_received_l297_297251

theorem turtles_received (martha_turtles : ℕ) (marion_turtles : ℕ) (h1 : martha_turtles = 40) 
    (h2 : marion_turtles = martha_turtles + 20) : martha_turtles + marion_turtles = 100 := 
by {
    sorry
}

end turtles_received_l297_297251


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297312

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297312


namespace sum_infinite_partial_fraction_l297_297635

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l297_297635


namespace number_of_elements_in_S_l297_297116

def S : Set ℕ := { n : ℕ | ∃ k : ℕ, n > 1 ∧ (10^10 - 1) % n = 0 }

theorem number_of_elements_in_S (h1 : Nat.Prime 9091) :
  ∃ T : Finset ℕ, T.card = 127 ∧ ∀ n, n ∈ T ↔ n ∈ S :=
sorry

end number_of_elements_in_S_l297_297116


namespace bertha_no_daughters_count_l297_297608

open Nat

-- Definitions for the conditions
def daughters : ℕ := 8
def total_women : ℕ := 42
def granddaughters : ℕ := total_women - daughters
def daughters_who_have_daughters := granddaughters / 6
def daughters_without_daughters := daughters - daughters_who_have_daughters
def total_without_daughters := granddaughters + daughters_without_daughters

-- The theorem to prove
theorem bertha_no_daughters_count : total_without_daughters = 37 := by
  sorry

end bertha_no_daughters_count_l297_297608


namespace remaining_numbers_l297_297015

-- Define the problem statement in Lean 4
theorem remaining_numbers (S S5 S3 : ℝ) (A3 : ℝ) 
  (h1 : S / 8 = 20) 
  (h2 : S5 / 5 = 12) 
  (h3 : S3 = S - S5) 
  (h4 : A3 = 100 / 3) : 
  S3 / A3 = 3 :=
sorry

end remaining_numbers_l297_297015


namespace grade_assignment_ways_l297_297460

-- Define the number of students and the number of grade choices
def students : ℕ := 12
def grade_choices : ℕ := 4

-- Define the number of ways to assign grades
def num_ways_to_assign_grades : ℕ := grade_choices ^ students

-- Prove that the number of ways to assign grades is 16777216
theorem grade_assignment_ways :
  num_ways_to_assign_grades = 16777216 :=
by
  -- Calculation validation omitted (proof step)
  sorry

end grade_assignment_ways_l297_297460


namespace largest_of_three_numbers_l297_297307

noncomputable def hcf := 23
noncomputable def factors := [11, 12, 13]

/-- The largest of the three numbers, given the H.C.F is 23 and the other factors of their L.C.M are 11, 12, and 13, is 39468. -/
theorem largest_of_three_numbers : hcf * factors.prod = 39468 := by
  sorry

end largest_of_three_numbers_l297_297307


namespace rest_area_milepost_l297_297575

theorem rest_area_milepost (milepost_first : ℕ) (milepost_seventh : ℕ) (h_first : milepost_first = 20) (h_seventh : milepost_seventh = 140) : 
  ∃ milepost_rest : ℕ, milepost_rest = (milepost_first + milepost_seventh) / 2 ∧ milepost_rest = 80 :=
by
  sorry

end rest_area_milepost_l297_297575


namespace dividend_calculation_l297_297170

theorem dividend_calculation
  (divisor : Int)
  (quotient : Int)
  (remainder : Int)
  (dividend : Int)
  (h_divisor : divisor = 800)
  (h_quotient : quotient = 594)
  (h_remainder : remainder = -968)
  (h_dividend : dividend = (divisor * quotient) + remainder) :
  dividend = 474232 := by
  sorry

end dividend_calculation_l297_297170


namespace rectangle_area_l297_297023

theorem rectangle_area (w l : ℕ) (h_sum : w + l = 14) (h_w : w = 6) : w * l = 48 := by
  sorry

end rectangle_area_l297_297023


namespace find_abc_l297_297938

noncomputable def x (t : ℝ) := 3 * Real.cos t - 2 * Real.sin t
noncomputable def y (t : ℝ) := 3 * Real.sin t

theorem find_abc :
  ∃ a b c : ℝ, 
  (a = 1/9) ∧ 
  (b = 4/27) ∧ 
  (c = 5/27) ∧ 
  (∀ t : ℝ, a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1) :=
by
  sorry

end find_abc_l297_297938


namespace rational_function_nonnegative_l297_297821

noncomputable def rational_function (x : ℝ) : ℝ :=
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3)

theorem rational_function_nonnegative :
  ∀ x, 0 ≤ x ∧ x < 3 → 0 ≤ rational_function x :=
sorry

end rational_function_nonnegative_l297_297821


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l297_297781

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l297_297781


namespace sixth_root_of_large_number_l297_297199

theorem sixth_root_of_large_number : 
  ∃ (x : ℕ), x = 51 ∧ x ^ 6 = 24414062515625 :=
by
  sorry

end sixth_root_of_large_number_l297_297199


namespace inv_f_zero_l297_297554

noncomputable def f (a b x : Real) : Real := 1 / (2 * a * x + 3 * b)

theorem inv_f_zero (a b : Real) (ha : a ≠ 0) (hb : b ≠ 0) : f a b (1 / (3 * b)) = 0 :=
by 
  sorry

end inv_f_zero_l297_297554


namespace find_a_plus_b_l297_297875

variables {x y z a b : ℝ}

noncomputable def log_base (a b : ℝ) := log b / log a

def satisfies_conditions (x y z : ℝ) : Prop :=
  log_base 2 (x + y) = z ∧ log_base 2 (x^2 + y^2) = z + 2

theorem find_a_plus_b (h : ∀ (x y z : ℝ), satisfies_conditions x y z → x^3 + y^3 = a * 2^(3 * z) + b * 2^(2 * z)) :
  a + b = 6.5 :=
sorry

end find_a_plus_b_l297_297875


namespace simplify_fraction_l297_297263

theorem simplify_fraction (k : ℝ) : 
  (∃ a b : ℝ, (6 * k^2 + 18) / 6 = a * k^2 + b ∧ a = 1 ∧ b = 3 ∧ (a / b) = 1/3) := by
  sorry

end simplify_fraction_l297_297263


namespace ratio_volumes_l297_297828

noncomputable def V_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def V_cone (r : ℝ) : ℝ := (1 / 3) * Real.pi * r^3

theorem ratio_volumes (r : ℝ) (hr : r > 0) : 
  (V_cone r) / (V_sphere r) = 1 / 4 :=
by
  sorry

end ratio_volumes_l297_297828


namespace son_age_l297_297943

theorem son_age (S M : ℕ) (h1 : M = S + 30) (h2 : M + 2 = 2 * (S + 2)) : S = 28 := 
by
  -- The proof can be filled in here.
  sorry

end son_age_l297_297943


namespace center_of_hyperbola_l297_297823

theorem center_of_hyperbola :
  ∃ (h k : ℝ), (h = 2 ∧ k = 4) ∧ (9 * (x - h)^2 - 16 * (y - k)^2 = 180) :=
  sorry

end center_of_hyperbola_l297_297823


namespace cube_volume_given_surface_area_l297_297024

/-- Surface area of a cube given the side length. -/
def surface_area (side_length : ℝ) := 6 * side_length^2

/-- Volume of a cube given the side length. -/
def volume (side_length : ℝ) := side_length^3

theorem cube_volume_given_surface_area :
  ∃ side_length : ℝ, surface_area side_length = 24 ∧ volume side_length = 8 :=
by
  sorry

end cube_volume_given_surface_area_l297_297024


namespace no_roots_impl_a_neg_l297_297395

theorem no_roots_impl_a_neg {a : ℝ} : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x - 1/x + a ≠ 0) → a < 0 :=
sorry

end no_roots_impl_a_neg_l297_297395


namespace number_mul_five_l297_297043

theorem number_mul_five (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 :=
by
  sorry

end number_mul_five_l297_297043


namespace hypotenuse_length_l297_297965

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l297_297965


namespace zhang_hua_repayment_l297_297294

noncomputable def principal_amount : ℕ := 480000
noncomputable def repayment_period : ℕ := 240
noncomputable def monthly_interest_rate : ℝ := 0.004
noncomputable def principal_payment : ℝ := principal_amount / repayment_period -- 2000, but keeping general form

noncomputable def interest (month : ℕ) : ℝ :=
  (principal_amount - (month - 1) * principal_payment) * monthly_interest_rate

noncomputable def monthly_repayment (month : ℕ) : ℝ :=
  principal_payment + interest month

theorem zhang_hua_repayment (n : ℕ) (h : 1 ≤ n ∧ n ≤ repayment_period) :
  monthly_repayment n = 3928 - 8 * n := 
by
  -- proof would be placed here
  sorry

end zhang_hua_repayment_l297_297294


namespace example_problem_l297_297470

theorem example_problem : 2 + 3 * 4 - 5 + 6 / 3 = 11 := by
  sorry

end example_problem_l297_297470


namespace min_n_probability_l297_297136

-- Define the number of members in teams
def num_members (n : ℕ) : ℕ := n

-- Define the total number of handshakes
def total_handshakes (n : ℕ) : ℕ := n * n

-- Define the number of ways to choose 2 handshakes from total handshakes
def choose_two_handshakes (n : ℕ) : ℕ := (total_handshakes n).choose 2

-- Define the number of ways to choose event A (involves exactly 3 different members)
def event_a_count (n : ℕ) : ℕ := 2 * n.choose 1 * (n - 1).choose 1

-- Define the probability of event A
def probability_event_a (n : ℕ) : ℚ := (event_a_count n : ℚ) / (choose_two_handshakes n : ℚ)

-- The minimum value of n such that the probability of event A is less than 1/10
theorem min_n_probability :
  ∃ n : ℕ, (probability_event_a n < (1 : ℚ) / 10) ∧ n ≥ 20 :=
by {
  sorry
}

end min_n_probability_l297_297136


namespace total_price_of_books_l297_297442

theorem total_price_of_books (total_books: ℕ) (math_books: ℕ) (math_book_cost: ℕ) (history_book_cost: ℕ) (price: ℕ) 
  (h1 : total_books = 90) 
  (h2 : math_books = 54) 
  (h3 : math_book_cost = 4) 
  (h4 : history_book_cost = 5)
  (h5 : price = 396) :
  let history_books := total_books - math_books
  let math_books_price := math_books * math_book_cost
  let history_books_price := history_books * history_book_cost
  let total_price := math_books_price + history_books_price
  total_price = price := 
  by
    sorry

end total_price_of_books_l297_297442


namespace bus_seating_capacity_l297_297706

-- Conditions
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seat_capacity : ℕ := 3
def back_seat_capacity : ℕ := 9
def total_seats : ℕ := left_side_seats + right_side_seats

-- Proof problem statement
theorem bus_seating_capacity :
  (total_seats * seat_capacity) + back_seat_capacity = 90 := by
  sorry

end bus_seating_capacity_l297_297706


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l297_297773

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l297_297773


namespace sum_infinite_series_eq_l297_297632

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l297_297632


namespace pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l297_297807

section pencil_case_problem

variables (x m : ℕ)

-- Part 1: The cost prices of each $A$ type and $B$ type pencil cases.
def cost_price_A (x : ℕ) : Prop := 
  (800 : ℝ) / x = (1000 : ℝ) / (x + 2)

-- Part 2.1: Maximum quantity of $B$ type pencil cases.
def max_quantity_B (m : ℕ) : Prop := 
  3 * m - 50 + m ≤ 910

-- Part 2.2: Number of different scenarios for purchasing the pencil cases.
def profit_condition (m : ℕ) : Prop := 
  4 * (3 * m - 50) + 5 * m > 3795

theorem pencil_case_solution_part1 (hA : cost_price_A x) : 
  x = 8 := 
sorry

theorem pencil_case_solution_part2_1 (hB : max_quantity_B m) : 
  m ≤ 240 := 
sorry

theorem pencil_case_solution_part2_2 (hB : max_quantity_B m) (hp : profit_condition m) : 
  236 ≤ m ∧ m ≤ 240 := 
sorry

end pencil_case_problem

end pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l297_297807


namespace kara_forgot_medication_times_l297_297245

theorem kara_forgot_medication_times :
  let ounces_per_medication := 4
  let medication_times_per_day := 3
  let days_per_week := 7
  let total_weeks := 2
  let total_water_intaken := 160
  let expected_total_water := (ounces_per_medication * medication_times_per_day * days_per_week * total_weeks)
  let water_difference := expected_total_water - total_water_intaken
  let forget_times := water_difference / ounces_per_medication
  forget_times = 2 := by sorry

end kara_forgot_medication_times_l297_297245


namespace binomial_variance_l297_297089

variable {n : ℕ}
variable {p : ℚ}

theorem binomial_variance (h₁ : n = 4) (h₂ : p = 1/2) : 
  variance (binomial n p) = 1 := by
  sorry

end binomial_variance_l297_297089


namespace add_base_3_l297_297185

def base3_addition : Prop :=
  2 + (1 * 3^2 + 2 * 3^1 + 0 * 3^0) + 
  (2 * 3^2 + 0 * 3^1 + 1 * 3^0) + 
  (1 * 3^3 + 2 * 3^1 + 0 * 3^0) = 
  (1 * 3^3) + (1 * 3^2) + (0 * 3^1) + (2 * 3^0)

theorem add_base_3 : base3_addition :=
by 
  -- We will skip the proof as per instructions
  sorry

end add_base_3_l297_297185


namespace louisa_average_speed_l297_297790

-- Problem statement
theorem louisa_average_speed :
  ∃ v : ℝ, (250 / v * v = 250 ∧ 350 / v * v = 350) ∧ ((350 / v) = (250 / v) + 3) ∧ v = 100 / 3 := by
  sorry

end louisa_average_speed_l297_297790


namespace total_first_half_points_l297_297707

-- Define the sequences for Tigers and Lions
variables (a ar b d : ℕ)
-- Defining conditions
def tied_first_quarter : Prop := a = b
def geometric_tigers : Prop := ∃ r : ℕ, ar = a * r ∧ ar^2 = a * r^2 ∧ ar^3 = a * r^3
def arithmetic_lions : Prop := b+d = b + d ∧ b+2*d = b + 2*d ∧ b+3*d = b + 3*d
def tigers_win_by_four : Prop := (a + ar + ar^2 + ar^3) = (b + (b + d) + (b + 2*d) + (b + 3*d)) + 4
def score_limit : Prop := (a + ar + ar^2 + ar^3) ≤ 120 ∧ (b + (b + d) + (b + 2*d) + (b + 3*d)) ≤ 120

-- Goal: The total number of points scored by the two teams in the first half is 23
theorem total_first_half_points : tied_first_quarter a b ∧ geometric_tigers a ar ∧ arithmetic_lions b d ∧ tigers_win_by_four a ar b d ∧ score_limit a ar b d → 
(a + ar) + (b + d) = 23 := 
by {
  sorry
}

end total_first_half_points_l297_297707


namespace figure_area_l297_297548

-- Given conditions
def right_angles (α β γ δ: ℕ): Prop :=
  α = 90 ∧ β = 90 ∧ γ = 90 ∧ δ = 90

def segment_lengths (a b c d e f g: ℕ): Prop :=
  a = 15 ∧ b = 8 ∧ c = 7 ∧ d = 3 ∧ e = 4 ∧ f = 2 ∧ g = 5

-- Define the problem
theorem figure_area :
  ∀ (α β γ δ a b c d e f g: ℕ),
    right_angles α β γ δ →
    segment_lengths a b c d e f g →
    a * b - (g * 1 + (d * f)) = 109 :=
by
  sorry

end figure_area_l297_297548


namespace two_d_minus_c_zero_l297_297270

theorem two_d_minus_c_zero :
  ∃ (c d : ℕ), (∀ x : ℕ, x^2 - 18 * x + 72 = (x - c) * (x - d)) ∧ c > d ∧ (2 * d - c = 0) := 
sorry

end two_d_minus_c_zero_l297_297270


namespace travel_time_reduction_impossible_proof_l297_297065

noncomputable def travel_time_reduction_impossible : Prop :=
  ∀ (x : ℝ), x > 60 → ¬ (1 / x * 60 = 1 - 1)

theorem travel_time_reduction_impossible_proof : travel_time_reduction_impossible :=
sorry

end travel_time_reduction_impossible_proof_l297_297065


namespace rectangular_plot_area_l297_297149

/-- The ratio between the length and the breadth of a rectangular plot is 7 : 5.
    If the perimeter of the plot is 288 meters, then the area of the plot is 5040 square meters.
-/
theorem rectangular_plot_area
    (L B : ℝ)
    (h1 : L / B = 7 / 5)
    (h2 : 2 * (L + B) = 288) :
    L * B = 5040 :=
by
  sorry

end rectangular_plot_area_l297_297149


namespace find_t_l297_297101

variable (a b c : ℝ × ℝ)
variable (t : ℝ)

-- Definitions based on given conditions
def vec_a : ℝ × ℝ := (3, 1)
def vec_b : ℝ × ℝ := (1, 3)
def vec_c (t : ℝ) : ℝ × ℝ := (t, 2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Condition that (vec_a - vec_c) is perpendicular to vec_b
def perpendicular_condition (t : ℝ) : Prop :=
  dot_product (vec_a - vec_c t) vec_b = 0

-- Proof statement
theorem find_t : ∃ t : ℝ, perpendicular_condition t ∧ t = 0 := 
by
  sorry

end find_t_l297_297101


namespace clothes_washer_final_price_l297_297797

theorem clothes_washer_final_price
  (P : ℝ) (d1 d2 d3 : ℝ)
  (hP : P = 500)
  (hd1 : d1 = 0.10)
  (hd2 : d2 = 0.20)
  (hd3 : d3 = 0.05) :
  (P * (1 - d1) * (1 - d2) * (1 - d3)) / P = 0.684 :=
by
  sorry

end clothes_washer_final_price_l297_297797


namespace reflection_points_reflection_line_l297_297273

-- Definitions of given points and line equation
def original_point : ℝ × ℝ := (2, 3)
def reflected_point : ℝ × ℝ := (8, 7)

-- Definitions of line parameters for y = mx + b
variable {m b : ℝ}

-- Statement of the reflection condition
theorem reflection_points_reflection_line : m + b = 9.5 := by
  -- sorry to skip the actual proof
  sorry

end reflection_points_reflection_line_l297_297273


namespace necessary_but_not_sufficient_l297_297716

variable (k : ℝ)

def is_ellipse : Prop := 
  (k > 1) ∧ (k < 5) ∧ (k ≠ 3)

theorem necessary_but_not_sufficient :
  (1 < k) ∧ (k < 5) → is_ellipse k :=
by sorry

end necessary_but_not_sufficient_l297_297716


namespace rainfall_difference_l297_297761

-- Define the conditions
def day1_rainfall := 26
def day2_rainfall := 34
def average_rainfall := 140
def less_rainfall := 58

-- Calculate the total rainfall this year in the first three days
def total_rainfall_this_year := average_rainfall - less_rainfall

-- Calculate the total rainfall in the first two days
def total_first_two_days := day1_rainfall + day2_rainfall

-- Calculate the rainfall on the third day
def day3_rainfall := total_rainfall_this_year - total_first_two_days

-- The proof problem
theorem rainfall_difference : day2_rainfall - day3_rainfall = 12 := 
by
  sorry

end rainfall_difference_l297_297761


namespace digit_difference_l297_297169

theorem digit_difference (X Y : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (hY : 0 ≤ Y ∧ Y ≤ 9) (h : 10 * X + Y - (10 * Y + X) = 81) : X - Y = 9 :=
sorry

end digit_difference_l297_297169


namespace eccentricity_of_ellipse_l297_297514

theorem eccentricity_of_ellipse :
  ∀ (A B : ℝ × ℝ) (has_axes_intersection : A.2 = 0 ∧ B.2 = 0) 
    (product_of_slopes : ∀ (P : ℝ × ℝ), P ≠ A ∧ P ≠ B → (P.2 / (P.1 - A.1)) * (P.2 / (P.1 + B.1)) = -1/2),
  ∃ (e : ℝ), e = 1 / Real.sqrt 2 :=
by
  sorry

end eccentricity_of_ellipse_l297_297514


namespace right_triangle_hypotenuse_length_l297_297977

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l297_297977


namespace contradiction_prop_l297_297516

theorem contradiction_prop (p : Prop) : 
  (∃ x : ℝ, x < -1 ∧ x^2 - x + 1 < 0) → (∀ x : ℝ, x < -1 → x^2 - x + 1 ≥ 0) :=
sorry

end contradiction_prop_l297_297516


namespace find_x_l297_297214

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
sorry

end find_x_l297_297214


namespace cargo_loaded_in_bahamas_l297_297804

def initial : ℕ := 5973
def final : ℕ := 14696
def loaded : ℕ := final - initial

theorem cargo_loaded_in_bahamas : loaded = 8723 := by
  sorry

end cargo_loaded_in_bahamas_l297_297804


namespace max_profit_achieved_when_x_is_1_l297_297455

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2
noncomputable def fixed_costs : ℝ := 40
noncomputable def material_cost (x : ℕ) : ℝ := 5 * x
noncomputable def profit (x : ℕ) : ℝ := revenue x - (fixed_costs + material_cost x)
noncomputable def marginal_profit (x : ℕ) : ℝ := profit (x + 1) - profit x

theorem max_profit_achieved_when_x_is_1 :
  marginal_profit 1 = 24.40 :=
by
  -- Skip the proof
  sorry

end max_profit_achieved_when_x_is_1_l297_297455


namespace lara_total_space_larger_by_1500_square_feet_l297_297246

theorem lara_total_space_larger_by_1500_square_feet :
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  total_area - area_square = 1500 :=
by
  -- Definitions
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  
  -- Calculation
  have h_area_rect : area_rect = 1500 := by
    norm_num [area_rect, length_rect, width_rect]

  have h_area_square : area_square = 2500 := by
    norm_num [area_square, side_square]

  have h_total_area : total_area = 4000 := by
    norm_num [total_area, h_area_rect, h_area_square]

  -- Final comparison
  have h_difference : total_area - area_square = 1500 := by
    norm_num [total_area, area_square, h_area_square]

  exact h_difference

end lara_total_space_larger_by_1500_square_feet_l297_297246


namespace total_weight_peppers_l297_297102

def weight_green_peppers : ℝ := 0.3333333333333333
def weight_red_peppers : ℝ := 0.3333333333333333

theorem total_weight_peppers : weight_green_peppers + weight_red_peppers = 0.6666666666666666 := 
by sorry

end total_weight_peppers_l297_297102


namespace last_digit_of_N_l297_297991

def sum_of_first_n_natural_numbers (N : ℕ) : ℕ :=
  N * (N + 1) / 2

theorem last_digit_of_N (N : ℕ) (h : sum_of_first_n_natural_numbers N = 3080) :
  N % 10 = 8 :=
by {
  sorry
}

end last_digit_of_N_l297_297991


namespace olympic_triathlon_total_distance_l297_297866

theorem olympic_triathlon_total_distance (x : ℝ) (L S : ℝ)
  (hL : L = 4 * x)
  (hS : S = (3 / 80) * x)
  (h_diff : L - S = 8.5) :
  x + L + S = 51.5 := by
  sorry

end olympic_triathlon_total_distance_l297_297866


namespace circle_tangent_radius_l297_297317

theorem circle_tangent_radius (k : ℝ) (r : ℝ) (hk : k > 4) 
  (h_tangent1 : dist (0, k) (x, x) = r)
  (h_tangent2 : dist (0, k) (x, -x) = r) 
  (h_tangent3 : dist (0, k) (x, 4) = r) : 
  r = 4 * Real.sqrt 2 := 
sorry

end circle_tangent_radius_l297_297317


namespace line_slope_l297_297363

theorem line_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : -4 / 7 :=
by
  sorry

end line_slope_l297_297363


namespace largest_two_numbers_l297_297467

def a : Real := 2^(1/2)
def b : Real := 3^(1/3)
def c : Real := 8^(1/8)
def d : Real := 9^(1/9)

theorem largest_two_numbers : 
  (max (max (max a b) c) d = b) ∧ 
  (max (max a c) d = a) := 
sorry

end largest_two_numbers_l297_297467


namespace polynomial_identity_l297_297560

theorem polynomial_identity 
  (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  a^2 * ((x - b) * (x - c) / ((a - b) * (a - c))) +
  b^2 * ((x - c) * (x - a) / ((b - c) * (b - a))) +
  c^2 * ((x - a) * (x - b) / ((c - a) * (c - b))) = x^2 :=
by
  sorry

end polynomial_identity_l297_297560


namespace series_sum_l297_297615

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l297_297615


namespace arithmetic_expression_value_l297_297732

theorem arithmetic_expression_value :
  68 + (105 / 15) + (26 * 19) - 250 - (390 / 6) = 254 :=
by
  sorry

end arithmetic_expression_value_l297_297732


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l297_297772

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l297_297772


namespace vertices_of_parabolas_is_parabola_l297_297118

theorem vertices_of_parabolas_is_parabola 
  (a c k : ℝ) (ha : 0 < a) (hc : 0 < c) (hk : 0 < k) :
  ∃ (f : ℝ → ℝ), (∀ t : ℝ, f t = (-k^2 / (4 * a)) * t^2 + c) ∧ 
  ∀ (pt : ℝ × ℝ), (∃ t : ℝ, pt = (-(k * t) / (2 * a), f t)) → 
  ∃ a' b' c', (∀ t : ℝ, pt.2 = a' * pt.1^2 + b' * pt.1 + c') ∧ (a < 0) :=
by sorry

end vertices_of_parabolas_is_parabola_l297_297118


namespace rational_square_of_one_minus_product_l297_297859

theorem rational_square_of_one_minus_product (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : 
  ∃ (q : ℚ), 1 - x * y = q^2 := 
by 
  sorry

end rational_square_of_one_minus_product_l297_297859


namespace part_i_l297_297305

theorem part_i (n : ℤ) : (∃ k : ℤ, n = 225 * k + 99) ↔ (n % 9 = 0 ∧ (n + 1) % 25 = 0) :=
by 
  sorry

end part_i_l297_297305


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l297_297782

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l297_297782


namespace interesting_quadruples_count_l297_297659

/-- Definition of interesting ordered quadruples (a, b, c, d) where 1 ≤ a < b < c < d ≤ 15 and a + b > c + d --/
def is_interesting_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + b > c + d

/-- The number of interesting ordered quadruples (a, b, c, d) is 455 --/
theorem interesting_quadruples_count : 
  (∃ (s : Finset (ℕ × ℕ × ℕ × ℕ)), 
    s.card = 455 ∧ ∀ (a b c d : ℕ), 
    ((a, b, c, d) ∈ s ↔ is_interesting_quadruple a b c d)) :=
sorry

end interesting_quadruples_count_l297_297659


namespace proof_2d_minus_r_l297_297226

theorem proof_2d_minus_r (d r: ℕ) (h1 : 1059 % d = r)
  (h2 : 1482 % d = r) (h3 : 2340 % d = r) (hd : d > 1) : 2 * d - r = 6 := 
by 
  sorry

end proof_2d_minus_r_l297_297226


namespace negative_number_is_d_l297_297332

def a : Int := -(-2)
def b : Int := abs (-2)
def c : Int := (-2) ^ 2
def d : Int := (-2) ^ 3

theorem negative_number_is_d : d < 0 :=
  by
  sorry

end negative_number_is_d_l297_297332


namespace A_iff_B_l297_297677

-- Define Proposition A: ab > b^2
def PropA (a b : ℝ) : Prop := a * b > b ^ 2

-- Define Proposition B: 1/b < 1/a < 0
def PropB (a b : ℝ) : Prop := 1 / b < 1 / a ∧ 1 / a < 0

theorem A_iff_B (a b : ℝ) : (PropA a b) ↔ (PropB a b) := sorry

end A_iff_B_l297_297677


namespace coprime_gcd_l297_297003

theorem coprime_gcd (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (2 * a + b) (a * (a + b)) = 1 := 
sorry

end coprime_gcd_l297_297003


namespace age_double_condition_l297_297800

theorem age_double_condition (S M X : ℕ) (h1 : S = 44) (h2 : M = S + 46) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end age_double_condition_l297_297800


namespace value_of_a2_l297_297229

variable {R : Type*} [Ring R] (x a_0 a_1 a_2 a_3 : R)

theorem value_of_a2 
  (h : ∀ x : R, x^3 = a_0 + a_1 * (x - 2) + a_2 * (x - 2)^2 + a_3 * (x - 2)^3) :
  a_2 = 6 :=
sorry

end value_of_a2_l297_297229


namespace cube_surface_area_including_inside_l297_297603

theorem cube_surface_area_including_inside 
  (original_edge_length : ℝ) 
  (hole_side_length : ℝ) 
  (original_cube_surface_area : ℝ)
  (removed_hole_area : ℝ)
  (newly_exposed_internal_area : ℝ) 
  (total_surface_area : ℝ) 
  (h1 : original_edge_length = 3)
  (h2 : hole_side_length = 1)
  (h3 : original_cube_surface_area = 6 * (original_edge_length * original_edge_length))
  (h4 : removed_hole_area = 6 * (hole_side_length * hole_side_length))
  (h5 : newly_exposed_internal_area = 6 * 4 * (hole_side_length * hole_side_length))
  (h6 : total_surface_area = original_cube_surface_area - removed_hole_area + newly_exposed_internal_area) : 
  total_surface_area = 72 :=
by
  sorry

end cube_surface_area_including_inside_l297_297603


namespace sum_series_l297_297621

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l297_297621


namespace sum_of_series_l297_297646

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l297_297646


namespace seq_max_value_l297_297383

theorem seq_max_value {a_n : ℕ → ℝ} (h : ∀ n, a_n n = (↑n + 2) * (3 / 4) ^ n) : 
  ∃ n, a_n n = max (a_n 1) (a_n 2) → (n = 1 ∨ n = 2) :=
by 
  sorry

end seq_max_value_l297_297383


namespace total_number_of_valid_guesses_l297_297456

noncomputable def valid_guesses (digits : Multiset ℕ) (prizes : list ℕ) : ℕ :=
  (Multiset.card digits).choose 3 * 12

theorem total_number_of_valid_guesses :
  valid_guesses {2, 2, 2, 2, 4, 4, 4} [D, E, F] = 420 :=
by {
  sorry
}

end total_number_of_valid_guesses_l297_297456


namespace geometric_sequence_product_l297_297404

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (n : ℕ) (h : a 4 = 4) :
  a 2 * a 6 = 16 := by
  -- Definition of geomtric sequence
  -- a_n = a_0 * r^n
  -- Using the fact that the product of corresponding terms equidistant from two ends is constant
  sorry

end geometric_sequence_product_l297_297404


namespace simplify_expression_l297_297014

theorem simplify_expression (x : ℝ) : 
  ((3 * x - 6) - 5 * x) / 3 = - (2 / 3) * x - 2 :=
by sorry

end simplify_expression_l297_297014


namespace sarah_total_distance_walked_l297_297073

noncomputable def total_distance : ℝ :=
  let rest_time : ℝ := 1 / 3
  let total_time : ℝ := 3.5
  let time_spent_walking : ℝ := total_time - rest_time -- time spent walking
  let uphill_speed : ℝ := 3 -- in mph
  let downhill_speed : ℝ := 4 -- in mph
  let d := time_spent_walking * (uphill_speed * downhill_speed) / (uphill_speed + downhill_speed) -- half distance D
  2 * d

theorem sarah_total_distance_walked :
  total_distance = 10.858 := sorry

end sarah_total_distance_walked_l297_297073


namespace cube_root_of_x_sqrt_x_eq_x_half_l297_297336

variable (x : ℝ) (h : 0 < x)

theorem cube_root_of_x_sqrt_x_eq_x_half : (x * Real.sqrt x) ^ (1/3) = x ^ (1/2) := by
  sorry

end cube_root_of_x_sqrt_x_eq_x_half_l297_297336


namespace pumps_work_hours_l297_297933

theorem pumps_work_hours (d : ℕ) (h_d_pos : d > 0) : 6 * (8 / d) * d = 48 :=
by
  -- The proof is omitted
  sorry

end pumps_work_hours_l297_297933


namespace right_triangle_hypotenuse_length_l297_297961

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l297_297961


namespace second_number_in_pair_l297_297152

theorem second_number_in_pair (n m : ℕ) (h1 : (n, m) = (57, 58)) (h2 : ∃ (n m : ℕ), n < 1500 ∧ m < 1500 ∧ (n + m) % 5 = 0) : m = 58 :=
by {
  sorry
}

end second_number_in_pair_l297_297152


namespace combined_age_71_in_6_years_l297_297704

-- Given conditions
variable (combinedAgeIn15Years : ℕ) (h_condition : combinedAgeIn15Years = 107)

-- Define the question
def combinedAgeIn6Years : ℕ := combinedAgeIn15Years - 4 * (15 - 6)

-- State the theorem to prove the question == answer given conditions
theorem combined_age_71_in_6_years (h_condition : combinedAgeIn15Years = 107) : combinedAgeIn6Years combinedAgeIn15Years = 71 := 
by 
  sorry

end combined_age_71_in_6_years_l297_297704


namespace min_value_of_f_on_interval_l297_297667

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan x ^ 2 - 4 * Real.tan x - 8 * (1 / Real.tan x) + 4 * (1 / Real.tan x) ^ 2 + 5

theorem min_value_of_f_on_interval :
  is_min (f x) (9 - 8 * Real.sqrt 2) (Ioo (Real.pi / 2) Real.pi) :=
sorry

end min_value_of_f_on_interval_l297_297667


namespace max_value_PXQ_l297_297299

theorem max_value_PXQ :
  ∃ (X P Q : ℕ), (XX = 10 * X + X) ∧ (10 * X + X) * X = 100 * P + 10 * X + Q ∧ 
  (X = 1 ∨ X = 5 ∨ X = 6) ∧ 
  (100 * P + 10 * X + Q) = 396 :=
sorry

end max_value_PXQ_l297_297299


namespace proof_problem_l297_297098

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem proof_problem (x1 x2 : ℝ) (h₁ : x1 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₂ : x2 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₃ : f x1 + f x2 > 0) : 
  x1 + x2 > 0 :=
sorry

end proof_problem_l297_297098


namespace probability_product_multiple_of_4_or_both_even_l297_297357

theorem probability_product_multiple_of_4_or_both_even :
  let prob_multiple_of_4 := (1 / 4) + (1 / 5) - ((1 / 4) * (1 / 5)),
      prob_both_even := (1 / 2) * (1 / 2),
      prob_both_conditions := (2 / 12) * (2 / 10),
      total_prob := prob_multiple_of_4 + prob_both_even - prob_both_conditions
  in total_prob = (37 / 60) :=
by
  sorry

end probability_product_multiple_of_4_or_both_even_l297_297357


namespace find_gamma_k_l297_297756

noncomputable def alpha (n d : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def beta (n r : ℕ) : ℕ := r^(n - 1)
noncomputable def gamma (n d r : ℕ) : ℕ := alpha n d + beta n r

theorem find_gamma_k (k d r : ℕ) (hk1 : gamma (k-1) d r = 200) (hk2 : gamma (k+1) d r = 2000) :
    gamma k d r = 387 :=
sorry

end find_gamma_k_l297_297756


namespace correct_operation_B_l297_297300

theorem correct_operation_B (a b : ℝ) : - (a - b) = -a + b := 
by sorry

end correct_operation_B_l297_297300


namespace complement_of_M_in_U_l297_297880

-- Definition of the universal set U
def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }

-- Definition of the set M
def M : Set ℝ := { 1 }

-- The statement to prove
theorem complement_of_M_in_U : (U \ M) = {x | 1 < x ∧ x ≤ 5} :=
by
  sorry

end complement_of_M_in_U_l297_297880


namespace conor_work_times_per_week_l297_297346

-- Definitions for the conditions
def vegetables_per_day (eggplants carrots potatoes : ℕ) : ℕ :=
  eggplants + carrots + potatoes

def total_vegetables_per_week (days vegetables_per_day : ℕ) : ℕ :=
  days * vegetables_per_day

-- Theorem statement to be proven
theorem conor_work_times_per_week :
  let eggplants := 12
  let carrots := 9
  let potatoes := 8
  let weekly_total := 116
  vegetables_per_day eggplants carrots potatoes = 29 →
  total_vegetables_per_week 4 29 = 116 →
  4 = weekly_total / 29 :=
by
  intros _ _ h1 h2
  sorry

end conor_work_times_per_week_l297_297346


namespace three_digit_numbers_divisible_by_5_l297_297389

theorem three_digit_numbers_divisible_by_5 : 
  let first_term := 100
  let last_term := 995
  let common_difference := 5 
  (last_term - first_term) / common_difference + 1 = 180 :=
by
  sorry

end three_digit_numbers_divisible_by_5_l297_297389


namespace value_of_x_squared_minus_y_squared_l297_297539

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l297_297539


namespace minimum_seats_occupied_l297_297026

theorem minimum_seats_occupied (total_seats : ℕ) (h : total_seats = 180) : 
  ∃ occupied_seats : ℕ, occupied_seats = 45 ∧ 
  ∀ additional_person,
    (∀ i : ℕ, i < total_seats → 
     (occupied_seats ≤ i → i < occupied_seats + 1 ∨ i > occupied_seats + 1)) →
    additional_person = occupied_seats + 1  :=
by
  sorry

end minimum_seats_occupied_l297_297026


namespace find_m_n_l297_297044

theorem find_m_n (m n : ℕ) (h : 26019 * m - 649 * n = 118) : m = 2 ∧ n = 80 :=
by 
  sorry

end find_m_n_l297_297044


namespace max_matching_pairs_l297_297557

theorem max_matching_pairs 
  (total_pairs : ℕ := 23) 
  (total_colors : ℕ := 6) 
  (total_sizes : ℕ := 3) 
  (lost_shoes : ℕ := 9)
  (shoes_per_pair : ℕ := 2) 
  (total_shoes := total_pairs * shoes_per_pair) 
  (remaining_shoes := total_shoes - lost_shoes) :
  ∃ max_pairs : ℕ, max_pairs = total_pairs - lost_shoes / shoes_per_pair :=
sorry

end max_matching_pairs_l297_297557


namespace solution_set_of_inequality_l297_297220

variable {a b x : ℝ}

theorem solution_set_of_inequality (h : ∃ y, y = 3*(-5) + a ∧ y = -2*(-5) + b) :
  (3*x + a < -2*x + b) ↔ (x < -5) :=
by sorry

end solution_set_of_inequality_l297_297220


namespace Brian_age_in_eight_years_l297_297194

-- Definitions based on conditions
variable {Christian Brian : ℕ}
variable (h1 : Christian = 2 * Brian)
variable (h2 : Christian + 8 = 72)

-- Target statement to prove Brian's age in eight years
theorem Brian_age_in_eight_years : (Brian + 8) = 40 :=
by 
  sorry

end Brian_age_in_eight_years_l297_297194


namespace diving_assessment_l297_297501

theorem diving_assessment (total_athletes : ℕ) (selected_athletes : ℕ) (not_meeting_standard : ℕ) 
  (first_level_sample : ℕ) (first_level_total : ℕ) (athletes : Set ℕ) :
  total_athletes = 56 → 
  selected_athletes = 8 → 
  not_meeting_standard = 2 → 
  first_level_sample = 3 → 
  (∀ (A B C D E : ℕ), athletes = {A, B, C, D, E} → first_level_total = 5 → 
  (∃ proportion_standard number_first_level probability_E, 
    proportion_standard = (8 - 2) / 8 ∧  -- first part: proportion of athletes who met the standard
    number_first_level = 56 * (3 / 8) ∧ -- second part: number of first-level athletes
    probability_E = 4 / 10))           -- third part: probability of athlete E being chosen
:= sorry

end diving_assessment_l297_297501


namespace negation_equivalence_l297_297580

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ - 2 < 0) ↔ (∀ x₀ : ℝ, x₀^2 + x₀ - 2 ≥ 0) :=
by sorry

end negation_equivalence_l297_297580


namespace sets_equal_l297_297687

theorem sets_equal (M N : Set ℝ) (hM : M = { x | x^2 = 1 }) (hN : N = { a | ∀ x ∈ M, a * x = 1 }) : M = N :=
sorry

end sets_equal_l297_297687


namespace sequence_count_l297_297755

theorem sequence_count (a : ℕ → ℤ) (h₁ : a 1 = 0) (h₂ : a 11 = 4) 
  (h₃ : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → |a (k + 1) - a k| = 1) : 
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end sequence_count_l297_297755


namespace carrie_bought_tshirts_l297_297069

variable (cost_per_tshirt : ℝ) (total_spent : ℝ)

theorem carrie_bought_tshirts (h1 : cost_per_tshirt = 9.95) (h2 : total_spent = 248) :
  ⌊total_spent / cost_per_tshirt⌋ = 24 :=
by
  sorry

end carrie_bought_tshirts_l297_297069


namespace minimum_wins_l297_297446

theorem minimum_wins (x y : ℕ) (h_score : 3 * x + y = 10) (h_games : x + y ≤ 7) (h_bounds : 0 < x ∧ x < 4) : x = 2 :=
by
  sorry

end minimum_wins_l297_297446


namespace sixth_root_24414062515625_l297_297201

theorem sixth_root_24414062515625 :
  (∃ (x : ℕ), x^6 = 24414062515625) → (sqrt 6 24414062515625 = 51) :=
by
  -- Applying the condition expressed as sum of binomials
  have h : 24414062515625 = ∑ k in finset.range 7, binom 6 k * (50 ^ (6 - k)),
  sorry
  
  -- Utilize this condition to find the sixth root
  sorry

end sixth_root_24414062515625_l297_297201


namespace max_prime_difference_l297_297552

theorem max_prime_difference (a b c d : ℕ) 
  (p1 : Prime a) (p2 : Prime b) (p3 : Prime c) (p4 : Prime d)
  (p5 : Prime (a + b + c + 18 + d)) (p6 : Prime (a + b + c + 18 - d))
  (p7 : Prime (b + c)) (p8 : Prime (c + d))
  (h1 : a + b + c = 2010) (h2 : a ≠ 3) (h3 : b ≠ 3) (h4 : c ≠ 3) (h5 : d ≠ 3) (h6 : d ≤ 50)
  (distinct_primes : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ (a + b + c + 18 + d)
                    ∧ a ≠ (a + b + c + 18 - d) ∧ a ≠ (b + c) ∧ a ≠ (c + d)
                    ∧ b ≠ c ∧ b ≠ d ∧ b ≠ (a + b + c + 18 + d)
                    ∧ b ≠ (a + b + c + 18 - d) ∧ b ≠ (b + c) ∧ b ≠ (c + d)
                    ∧ c ≠ d ∧ c ≠ (a + b + c + 18 + d)
                    ∧ c ≠ (a + b + c + 18 - d) ∧ c ≠ (b + c) ∧ c ≠ (c + d)
                    ∧ d ≠ (a + b + c + 18 + d) ∧ d ≠ (a + b + c + 18 - d)
                    ∧ d ≠ (b + c) ∧ d ≠ (c + d)
                    ∧ (a + b + c + 18 + d) ≠ (a + b + c + 18 - d)
                    ∧ (a + b + c + 18 + d) ≠ (b + c) ∧ (a + b + c + 18 + d) ≠ (c + d)
                    ∧ (a + b + c + 18 - d) ≠ (b + c) ∧ (a + b + c + 18 - d) ≠ (c + d)
                    ∧ (b + c) ≠ (c + d)) :
  ∃ max_diff : ℕ, max_diff = 2067 := sorry

end max_prime_difference_l297_297552


namespace golden_section_PB_l297_297219

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem golden_section_PB {A B P : ℝ} (h1 : P = (1 - 1/(golden_ratio)) * A + (1/(golden_ratio)) * B)
  (h2 : AB = 2)
  (h3 : A ≠ B) : PB = 3 - Real.sqrt 5 :=
by
  sorry

end golden_section_PB_l297_297219


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297311

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297311


namespace amount_per_friend_l297_297447

-- Definitions based on conditions
def cost_of_erasers : ℝ := 5 * 200
def cost_of_pencils : ℝ := 7 * 800
def total_cost : ℝ := cost_of_erasers + cost_of_pencils
def number_of_friends : ℝ := 4

-- The proof statement
theorem amount_per_friend : (total_cost / number_of_friends) = 1650 := by
  sorry

end amount_per_friend_l297_297447


namespace zora_is_shorter_by_eight_l297_297722

noncomputable def zora_height (z : ℕ) (b : ℕ) (i : ℕ) (zara : ℕ) (average_height : ℕ) : Prop :=
  i = z + 4 ∧
  zara = b ∧
  average_height = 61 ∧
  (z + i + zara + b) / 4 = average_height

theorem zora_is_shorter_by_eight (Z B : ℕ)
  (h1 : zora_height Z B (Z + 4) 64 61) : (B - Z) = 8 :=
by
  sorry

end zora_is_shorter_by_eight_l297_297722


namespace problem_statement_l297_297897

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * (x - 1))

theorem problem_statement :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x1 x2 : ℝ, x1 + x2 = π / 2 → g x1 = g x2) :=
by 
  sorry

end problem_statement_l297_297897


namespace steve_popsicle_sticks_l297_297884

theorem steve_popsicle_sticks (S Sid Sam : ℕ) (h1 : Sid = 2 * S) (h2 : Sam = 3 * Sid) (h3 : S + Sid + Sam = 108) : S = 12 :=
by
  sorry

end steve_popsicle_sticks_l297_297884


namespace roots_product_l297_297373

theorem roots_product {a b : ℝ} (h1 : a^2 - a - 2 = 0) (h2 : b^2 - b - 2 = 0) 
(roots : a ≠ b ∧ ∀ x, x^2 - x - 2 = 0 ↔ (x = a ∨ x = b)) : (a - 1) * (b - 1) = -2 := by
  -- proof
  sorry

end roots_product_l297_297373


namespace hypotenuse_length_l297_297948

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l297_297948


namespace unique_digit_10D4_count_unique_digit_10D4_l297_297083

theorem unique_digit_10D4 (D : ℕ) (hD : D < 10) : 
  (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0 ↔ D = 4 :=
by
  sorry

theorem count_unique_digit_10D4 :
  ∃! D, (D < 10 ∧ (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0) :=
by
  use 4
  simp [unique_digit_10D4]
  sorry

end unique_digit_10D4_count_unique_digit_10D4_l297_297083


namespace max_value_is_5_l297_297414

noncomputable def max_value (θ φ : ℝ) : ℝ :=
  3 * Real.sin θ * Real.cos φ + 2 * Real.sin φ ^ 2

theorem max_value_is_5 (θ φ : ℝ) (h1 : 0 ≤ θ) (h2 : θ ≤ Real.pi / 2) (h3 : 0 ≤ φ) (h4 : φ ≤ Real.pi / 2) :
  max_value θ φ ≤ 5 :=
sorry

end max_value_is_5_l297_297414


namespace correct_sunset_time_l297_297545

-- Definitions corresponding to the conditions
def length_of_daylight : ℕ × ℕ := (10, 30) -- (hours, minutes)
def sunrise_time : ℕ × ℕ := (6, 50) -- (hours, minutes)

-- The reaching goal is to prove the sunset time
def sunset_time (sunrise : ℕ × ℕ) (daylight : ℕ × ℕ) : ℕ × ℕ :=
  let (sh, sm) := sunrise
  let (dh, dm) := daylight
  let total_minutes := sm + dm
  let extra_hour := total_minutes / 60
  let final_minutes := total_minutes % 60
  (sh + dh + extra_hour, final_minutes)

-- The theorem to prove
theorem correct_sunset_time :
  sunset_time sunrise_time length_of_daylight = (17, 20) := sorry

end correct_sunset_time_l297_297545


namespace highlighter_total_l297_297927

theorem highlighter_total 
  (pink_highlighters : ℕ)
  (yellow_highlighters : ℕ)
  (blue_highlighters : ℕ)
  (h_pink : pink_highlighters = 4)
  (h_yellow : yellow_highlighters = 2)
  (h_blue : blue_highlighters = 5) :
  pink_highlighters + yellow_highlighters + blue_highlighters = 11 :=
by
  sorry

end highlighter_total_l297_297927


namespace arithmetic_sequence_middle_term_l297_297032

theorem arithmetic_sequence_middle_term :
  ∀ (a b : ℕ) (z : ℕ), a = 9 → b = 81 → z = (a + b) / 2 → z = 45 :=
by
  intros a b z h_a h_b h_z
  rw [h_a, h_b] at h_z
  exact h_z

end arithmetic_sequence_middle_term_l297_297032


namespace count_lines_in_2008_cube_l297_297396

def num_lines_through_centers_of_unit_cubes (n : ℕ) : ℕ :=
  n * n * 3 + n * 2 * 3 + 4

theorem count_lines_in_2008_cube :
  num_lines_through_centers_of_unit_cubes 2008 = 12115300 :=
by
  -- The actual proof would go here
  sorry

end count_lines_in_2008_cube_l297_297396


namespace percentage_both_questions_correct_l297_297926

-- Definitions for the conditions in the problem
def percentage_first_question_correct := 85
def percentage_second_question_correct := 65
def percentage_neither_question_correct := 5
def percentage_one_or_more_questions_correct := 100 - percentage_neither_question_correct

-- Theorem stating that 55 percent answered both questions correctly
theorem percentage_both_questions_correct :
  percentage_first_question_correct + percentage_second_question_correct - percentage_one_or_more_questions_correct = 55 :=
by
  sorry

end percentage_both_questions_correct_l297_297926


namespace file_organization_ratio_l297_297355

variable (X : ℕ) -- The number of files organized in the morning
variable (total_files morning_files afternoon_files missing_files : ℕ)

-- Conditions
def condition1 : total_files = 60 := by sorry
def condition2 : afternoon_files = 15 := by sorry
def condition3 : missing_files = 15 := by sorry
def condition4 : morning_files = X := by sorry
def condition5 : morning_files + afternoon_files + missing_files = total_files := by sorry

-- Question
def ratio_morning_to_total : Prop :=
  let organized_files := total_files - afternoon_files - missing_files
  (organized_files / total_files : ℚ) = 1 / 2

-- Proof statement
theorem file_organization_ratio : 
  ∀ (X total_files morning_files afternoon_files missing_files : ℕ), 
    total_files = 60 → 
    afternoon_files = 15 → 
    missing_files = 15 → 
    morning_files = X → 
    morning_files + afternoon_files + missing_files = total_files → 
    (X / 60 : ℚ) = 1 / 2 := by 
  sorry

end file_organization_ratio_l297_297355


namespace determine_x_l297_297498

theorem determine_x (x : ℕ) 
  (hx1 : x % 6 = 0) 
  (hx2 : x^2 > 196) 
  (hx3 : x < 30) : 
  x = 18 ∨ x = 24 := 
sorry

end determine_x_l297_297498


namespace initial_red_marbles_l297_297705

theorem initial_red_marbles
    (r g : ℕ)
    (h1 : 3 * r = 5 * g)
    (h2 : 2 * (r - 15) = g + 18) :
    r = 34 := by
  sorry

end initial_red_marbles_l297_297705


namespace parking_space_unpainted_side_l297_297602

theorem parking_space_unpainted_side 
  (L W : ℝ) 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 125) : 
  L = 8.90 := 
by 
  sorry

end parking_space_unpainted_side_l297_297602


namespace sqrt_function_of_x_l297_297301

theorem sqrt_function_of_x (x : ℝ) (h : x > 0) : ∃! y : ℝ, y = Real.sqrt x :=
by
  sorry

end sqrt_function_of_x_l297_297301


namespace average_age_in_club_l297_297708

theorem average_age_in_club :
  let women_avg_age := 32
  let men_avg_age := 38
  let children_avg_age := 10
  let women_count := 12
  let men_count := 18
  let children_count := 10
  let total_ages := (women_avg_age * women_count) + (men_avg_age * men_count) + (children_avg_age * children_count)
  let total_people := women_count + men_count + children_count
  let overall_avg_age := (total_ages : ℝ) / (total_people : ℝ)
  overall_avg_age = 29.2 := by
  sorry

end average_age_in_club_l297_297708


namespace extremum_of_function_l297_297506

theorem extremum_of_function (k : ℝ) (h₀ : k ≠ 1) :
  (k > 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≤ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) ∧
  (k < 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≥ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) :=
by
  sorry

end extremum_of_function_l297_297506


namespace gasoline_price_increase_l297_297900

theorem gasoline_price_increase :
  ∀ (p_low p_high : ℝ), p_low = 14 → p_high = 23 → 
  ((p_high - p_low) / p_low) * 100 = 64.29 :=
by
  intro p_low p_high h_low h_high
  rw [h_low, h_high]
  sorry

end gasoline_price_increase_l297_297900


namespace sum_fraction_series_l297_297627

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l297_297627


namespace solve_eq1_solve_eq2_l297_297744

-- Define the theorem for the first equation
theorem solve_eq1 (x : ℝ) (h : 2 * x - 7 = 5 * x - 1) : x = -2 :=
sorry

-- Define the theorem for the second equation
theorem solve_eq2 (x : ℝ) (h : (x - 2) / 2 - (x - 1) / 6 = 1) : x = 11 / 2 :=
sorry

end solve_eq1_solve_eq2_l297_297744


namespace parallel_lines_a_perpendicular_lines_a_l297_297385

-- Definitions of the lines
def l1 (a x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) := x + (a - 1) * y + a^2 - 1 = 0

-- Statement for parallel lines problem
theorem parallel_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = -1) :=
by
  sorry

-- Statement for perpendicular lines problem
theorem perpendicular_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y → (-a / 2) * (1 / (a - 1)) = -1) → (a = 2 / 3) :=
by
  sorry

end parallel_lines_a_perpendicular_lines_a_l297_297385


namespace hypotenuse_right_triangle_l297_297988

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l297_297988


namespace sum_of_nine_consecutive_quotients_multiple_of_9_l297_297212

def a (i : ℕ) : ℕ := (10^(2 * i) - 1) / 9
def q (i : ℕ) : ℕ := a i / 11
def s (i : ℕ) : ℕ := q i + q (i + 1) + q (i + 2) + q (i + 3) + q (i + 4) + q (i + 5) + q (i + 6) + q (i + 7) + q (i + 8)

theorem sum_of_nine_consecutive_quotients_multiple_of_9 (i n : ℕ) (h : n > 8) 
  (h2 : i ≤ n - 8) : s i % 9 = 0 :=
sorry

end sum_of_nine_consecutive_quotients_multiple_of_9_l297_297212


namespace solve_equation_l297_297889

-- Given conditions and auxiliary definitions
def is_solution (x y z : ℕ) : Prop := 2 ^ x + 3 ^ y - 7 = Nat.factorial z

-- Primary theorem: the equivalent proof problem
theorem solve_equation (x y z : ℕ) :
  (is_solution x y 3 → (x = 2 ∧ y = 2)) ∧
  (∀ z, (z ≤ 3 → z ≠ 3) → ¬is_solution x y z) ∧
  (z ≥ 4 → ¬is_solution x y z) :=
  sorry

end solve_equation_l297_297889


namespace determine_k_l297_297058

-- Definitions of the vectors a and b.
variables (a b : ℝ)

-- Noncomputable definition of the scalar k.
noncomputable def k_value : ℝ :=
  (2 : ℚ) / 7

-- Definition of line through vectors a and b as a parametric equation.
def line_through (a b : ℝ) (t : ℝ) : ℝ :=
  a + t * (b - a)

-- Hypothesis: The vector k * a + (5/7) * b is on the line passing through a and b.
def vector_on_line (a b : ℝ) (k : ℝ) : Prop :=
  ∃ t : ℝ, k * a + (5/7) * b = line_through a b t

-- Proof that k must be 2/7 for the vector to be on the line.
theorem determine_k (a b : ℝ) : vector_on_line a b k_value :=
by sorry

end determine_k_l297_297058


namespace train_speed_is_correct_l297_297328

-- Define the conditions
def length_of_train : ℕ := 140 -- length in meters
def time_to_cross_pole : ℕ := 7 -- time in seconds

-- Define the expected speed in km/h
def expected_speed_in_kmh : ℕ := 72 -- speed in km/h

-- Prove that the speed of the train in km/h is 72
theorem train_speed_is_correct :
  (length_of_train / time_to_cross_pole) * 36 / 10 = expected_speed_in_kmh :=
by
  sorry

end train_speed_is_correct_l297_297328


namespace max_H2O_produced_l297_297351

theorem max_H2O_produced :
  ∀ (NaOH H2SO4 H2O : ℝ)
  (n_NaOH : NaOH = 1.5)
  (n_H2SO4 : H2SO4 = 1)
  (balanced_reaction : 2 * NaOH + H2SO4 = 2 * H2O + 1 * (NaOH + H2SO4)),
  H2O = 1.5 :=
by
  intros NaOH H2SO4 H2O n_NaOH n_H2SO4 balanced_reaction
  sorry

end max_H2O_produced_l297_297351


namespace length_of_field_l297_297275

variable (w : ℕ) (l : ℕ)

def length_field_is_double_width (w l : ℕ) : Prop :=
  l = 2 * w

def pond_area_equals_one_eighth_field_area (w l : ℕ) : Prop :=
  36 = 1 / 8 * (l * w)

theorem length_of_field (w l : ℕ) (h1 : length_field_is_double_width w l) (h2 : pond_area_equals_one_eighth_field_area w l) : l = 24 := 
by
  sorry

end length_of_field_l297_297275


namespace hypotenuse_length_l297_297966

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l297_297966


namespace probability_even_integer_division_l297_297747

theorem probability_even_integer_division (r k : ℤ)
  (hr : -5 < r ∧ r < 7)
  (hk : 2 ≤ k ∧ k ≤ 9) :
  let R_even := {r : ℤ | -5 < r ∧ r < 7 ∧ even r}
      K := {k : ℤ | 2 ≤ k ∧ k ≤ 9}
      valid_pairs := { (r, k) : ℤ × ℤ | r ∈ R_even ∧ k ∈ K ∧ k ∣ r }
      total_pairs := R_even.prod K
  in (valid_pairs.card : ℚ) / (total_pairs.card : ℚ) = 17 / 48 :=
by
  sorry

end probability_even_integer_division_l297_297747


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l297_297769

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l297_297769


namespace increasing_exponential_is_necessary_condition_l297_297452

variable {a : ℝ}

theorem increasing_exponential_is_necessary_condition (h : ∀ x y : ℝ, x < y → a ^ x < a ^ y) :
    (a > 1) ∧ (¬ (a > 2 → a > 1)) :=
by
  sorry

end increasing_exponential_is_necessary_condition_l297_297452


namespace infinite_series_converges_l297_297640

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l297_297640


namespace necessary_but_not_sufficient_l297_297930

theorem necessary_but_not_sufficient (a b x y : ℤ) (ha : 0 < a) (hb : 0 < b) (h1 : x - y > a + b) (h2 : x * y > a * b) : 
  (x > a ∧ y > b) := sorry

end necessary_but_not_sufficient_l297_297930


namespace solution_set_of_inequality_l297_297908

theorem solution_set_of_inequality :
  {x : ℝ | -1 < x ∧ x < 2} = {x : ℝ | (x - 2) / (x + 1) < 0} :=
sorry

end solution_set_of_inequality_l297_297908


namespace sqrt_of_16_is_4_l297_297490

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l297_297490


namespace find_x_coordinate_l297_297463

theorem find_x_coordinate (m b x y : ℝ) (h1: m = 4) (h2: b = 100) (h3: y = 300) (line_eq: y = m * x + b) : x = 50 :=
by {
  sorry
}

end find_x_coordinate_l297_297463


namespace number_is_160_l297_297320

theorem number_is_160 (x : ℝ) (h : x / 5 + 4 = x / 4 - 4) : x = 160 :=
by
  sorry

end number_is_160_l297_297320


namespace hyperbola_eccentricity_l297_297540

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > b) (hb : b > 0)
  (h_ellipse : (a^2 - b^2) / a^2 = 3 / 4) :
  (a^2 + b^2) / a^2 = 5 / 4 :=
by
  -- We start with the given conditions and need to show the result
  sorry  -- Proof omitted

end hyperbola_eccentricity_l297_297540


namespace necessary_but_not_sufficient_l297_297903

-- Define the quadratic equation
def quadratic_eq (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + a

-- State the necessary but not sufficient condition proof statement
theorem necessary_but_not_sufficient (a : ℝ) :
  (∃ x y : ℝ, quadratic_eq a x = 0 ∧ quadratic_eq a y = 0 ∧ x > 0 ∧ y < 0) → a < 1 :=
sorry

end necessary_but_not_sufficient_l297_297903


namespace sandwich_cost_l297_297298

theorem sandwich_cost (total_cost soda_cost sandwich_count soda_count : ℝ) :
  total_cost = 8.38 → soda_cost = 0.87 → sandwich_count = 2 → soda_count = 4 → 
  (∀ S, sandwich_count * S + soda_count * soda_cost = total_cost → S = 2.45) :=
by
  intros h_total h_soda h_sandwich_count h_soda_count S h_eqn
  sorry

end sandwich_cost_l297_297298


namespace hypotenuse_length_l297_297969

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l297_297969


namespace total_fruit_count_l297_297284

-- Define the number of oranges
def oranges : ℕ := 6

-- Define the number of apples based on the number of oranges
def apples : ℕ := oranges - 2

-- Define the number of bananas based on the number of apples
def bananas : ℕ := 3 * apples

-- Define the number of peaches based on the number of bananas
def peaches : ℕ := bananas / 2

-- Define the total number of fruits in the basket
def total_fruits : ℕ := oranges + apples + bananas + peaches

-- Prove that the total number of pieces of fruit in the basket is 28
theorem total_fruit_count : total_fruits = 28 := by
  sorry

end total_fruit_count_l297_297284


namespace solve_equation_l297_297890

-- Given conditions and auxiliary definitions
def is_solution (x y z : ℕ) : Prop := 2 ^ x + 3 ^ y - 7 = Nat.factorial z

-- Primary theorem: the equivalent proof problem
theorem solve_equation (x y z : ℕ) :
  (is_solution x y 3 → (x = 2 ∧ y = 2)) ∧
  (∀ z, (z ≤ 3 → z ≠ 3) → ¬is_solution x y z) ∧
  (z ≥ 4 → ¬is_solution x y z) :=
  sorry

end solve_equation_l297_297890


namespace condition_on_a_and_b_l297_297854

theorem condition_on_a_and_b (a b p q : ℝ) 
    (h1 : (∀ x : ℝ, (x + a) * (x + b) = x^2 + p * x + q))
    (h2 : p > 0)
    (h3 : q < 0) :
    (a < 0 ∧ b > 0 ∧ b > -a) ∨ (a > 0 ∧ b < 0 ∧ a > -b) :=
by
  sorry

end condition_on_a_and_b_l297_297854


namespace probability_even_equals_prime_l297_297356

noncomputable def even_numbers : Finset ℕ := {2, 4, 6}
noncomputable def prime_numbers : Finset ℕ := {2, 3, 5}

-- Define a function that calculates the probability of rolling a 6-sided die
def die_prob (s : Finset ℕ) : ℚ :=
  (∥s∥.toRat / 6)

-- Define the probability of getting 4 even numbers and 4 prime numbers in 8 rolls
def even_prime_probability: ℚ :=
let die_rolls := 8
let half_rolls := die_rolls / 2
let combination := nat.choose die_rolls half_rolls
let even_prob := die_prob even_numbers
let prime_prob := die_prob prime_numbers in
combination * (even_prob ^ half_rolls) * (prime_prob ^ half_rolls)

theorem probability_even_equals_prime :
  even_prime_probability = 35 / 128 :=
by
  sorry

end probability_even_equals_prime_l297_297356


namespace compositeShapeSum_is_42_l297_297601

-- Define the pentagonal prism's properties
structure PentagonalPrism where
  faces : ℕ := 7
  edges : ℕ := 15
  vertices : ℕ := 10

-- Define the pyramid addition effect
structure PyramidAddition where
  additional_faces : ℕ := 5
  additional_edges : ℕ := 5
  additional_vertices : ℕ := 1
  covered_faces : ℕ := 1

-- Definition of composite shape properties
def compositeShapeSum (prism : PentagonalPrism) (pyramid : PyramidAddition) : ℕ :=
  (prism.faces - pyramid.covered_faces + pyramid.additional_faces) +
  (prism.edges + pyramid.additional_edges) +
  (prism.vertices + pyramid.additional_vertices)

-- The theorem to be proved: that the total sum is 42
theorem compositeShapeSum_is_42 : compositeShapeSum ⟨7, 15, 10⟩ ⟨5, 5, 1, 1⟩ = 42 := by
  sorry

end compositeShapeSum_is_42_l297_297601


namespace appropriate_sampling_method_l297_297760

/--
Given there are 40 products in total, consisting of 10 first-class products,
25 second-class products, and 5 defective products, if we need to select
8 products for quality analysis, then the appropriate sampling method is
the stratified sampling method.
-/
theorem appropriate_sampling_method
  (total_products : ℕ)
  (first_class_products : ℕ)
  (second_class_products : ℕ)
  (defective_products : ℕ)
  (selected_products : ℕ)
  (stratified_sampling : ℕ → ℕ → ℕ → ℕ → Prop) :
  total_products = 40 →
  first_class_products = 10 →
  second_class_products = 25 →
  defective_products = 5 →
  selected_products = 8 →
  stratified_sampling total_products first_class_products second_class_products defective_products →
  stratified_sampling total_products first_class_products second_class_products defective_products :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end appropriate_sampling_method_l297_297760


namespace probability_non_edge_unit_square_l297_297597

theorem probability_non_edge_unit_square : 
  let total_squares := 100
  let perimeter_squares := 36
  let non_perimeter_squares := total_squares - perimeter_squares
  let probability := (non_perimeter_squares : ℚ) / total_squares
  probability = 16 / 25 :=
by
  sorry

end probability_non_edge_unit_square_l297_297597


namespace sum_infinite_series_eq_l297_297631

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l297_297631


namespace exponential_growth_equation_l297_297335

-- Define the initial and final greening areas and the years in consideration.
def initial_area : ℝ := 1000
def final_area : ℝ := 1440
def years : ℝ := 2

-- Define the average annual growth rate.
variable (x : ℝ)

-- State the theorem about the exponential growth equation.
theorem exponential_growth_equation :
  initial_area * (1 + x) ^ years = final_area :=
sorry

end exponential_growth_equation_l297_297335


namespace rectangle_dimensions_l297_297904

theorem rectangle_dimensions (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w * l) = 2 * (2 * w + w)) :
  w = 6 ∧ l = 12 := 
by sorry

end rectangle_dimensions_l297_297904


namespace problem_f_sum_zero_l297_297119

variable (f : ℝ → ℝ)

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def symmetrical (f : ℝ → ℝ) : Prop := ∀ x, f (1 - x) = f x

-- Prove the required sum is zero given the conditions.
theorem problem_f_sum_zero (hf_odd : odd f) (hf_symm : symmetrical f) : 
  f 1 + f 2 + f 3 + f 4 + f 5 = 0 := by
  sorry

end problem_f_sum_zero_l297_297119


namespace necessary_but_not_sufficient_l297_297792

-- Definitions from conditions
def abs_gt_2 (x : ℝ) : Prop := |x| > 2
def x_lt_neg_2 (x : ℝ) : Prop := x < -2

-- Statement to prove
theorem necessary_but_not_sufficient : 
  ∀ x : ℝ, (abs_gt_2 x → x_lt_neg_2 x) ∧ (¬(x_lt_neg_2 x → abs_gt_2 x)) := 
by 
  sorry

end necessary_but_not_sufficient_l297_297792


namespace score_of_29_impossible_l297_297062

theorem score_of_29_impossible :
  ¬ ∃ (c u w : ℕ), c + u + w = 10 ∧ 3 * c + u = 29 :=
by {
  sorry
}

end score_of_29_impossible_l297_297062


namespace solve_for_x_l297_297703

theorem solve_for_x {x : ℤ} (h : 3 * x + 7 = -2) : x = -3 := 
by
  sorry

end solve_for_x_l297_297703


namespace series_sum_l297_297613

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l297_297613


namespace count_multiples_5_or_7_but_not_35_l297_297388

def count_multiples (n d : ℕ) : ℕ :=
  n / d

def inclusion_exclusion (a b c : ℕ) : ℕ :=
  a + b - c

theorem count_multiples_5_or_7_but_not_35 : 
  count_multiples 3000 5 + count_multiples 3000 7 - count_multiples 3000 35 = 943 :=
by
  sorry

end count_multiples_5_or_7_but_not_35_l297_297388


namespace right_triangle_hypotenuse_length_l297_297975

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l297_297975


namespace n_pow4_sub_n_pow2_divisible_by_12_l297_297559

theorem n_pow4_sub_n_pow2_divisible_by_12 (n : ℤ) (h : n > 1) : 12 ∣ (n^4 - n^2) :=
by sorry

end n_pow4_sub_n_pow2_divisible_by_12_l297_297559


namespace solution_set_inequality_l297_297435

theorem solution_set_inequality {x : ℝ} : 
  ((x - 1)^2 < 1) ↔ (0 < x ∧ x < 2) := by
  sorry

end solution_set_inequality_l297_297435


namespace percentage_blue_and_red_l297_297794

theorem percentage_blue_and_red (F : ℕ) (h_even: F % 2 = 0)
  (h1: ∃ C, 50 / 100 * C = F / 2)
  (h2: ∃ C, 60 / 100 * C = F / 2)
  (h3: ∃ C, 40 / 100 * C = F / 2) :
  ∃ C, (50 / 100 * C + 60 / 100 * C - 100 / 100 * C) = 10 / 100 * C :=
sorry

end percentage_blue_and_red_l297_297794


namespace abc_not_all_positive_l297_297834

theorem abc_not_all_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ac > 0) (h3 : abc > 0) : 
  ¬(a > 0 ∧ b > 0 ∧ c > 0) ↔ (a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0) := 
by 
sorry

end abc_not_all_positive_l297_297834


namespace right_triangle_hypotenuse_length_l297_297978

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l297_297978


namespace triangle_area_correct_l297_297724

def vector_a : ℝ × ℝ := (4, -3)
def vector_b : ℝ × ℝ := (-6, 5)
def vector_c : ℝ × ℝ := (2 * -6, 2 * 5)

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * |a.1 * c.2 - a.2 * c.1|

theorem triangle_area_correct :
  area_of_triangle (4, -3) (0, 0) (-12, 10) = 2 := by
  sorry

end triangle_area_correct_l297_297724


namespace right_triangle_hypotenuse_length_l297_297979

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l297_297979


namespace find_k_l297_297752

variable (x y k : ℝ)

-- Definition: the line equations and the intersection condition
def line1_eq (x y k : ℝ) : Prop := 3 * x - 2 * y = k
def line2_eq (x y : ℝ) : Prop := x - 0.5 * y = 10
def intersect_at_x (x : ℝ) : Prop := x = -6

-- The theorem we need to prove
theorem find_k (h1 : line1_eq x y k)
               (h2 : line2_eq x y)
               (h3 : intersect_at_x x) :
               k = 46 :=
sorry

end find_k_l297_297752


namespace angle_ABC_is_83_degrees_l297_297739

theorem angle_ABC_is_83_degrees (A B C D K : Type)
  (angle_BAC : Real) (angle_CAD : Real) (angle_ACD : Real)
  (AB AC AD : Real) (angle_ABC : Real) :
  angle_BAC = 60 ∧ angle_CAD = 60 ∧ angle_ACD = 23 ∧ AB + AD = AC → 
  angle_ABC = 83 :=
by
  sorry

end angle_ABC_is_83_degrees_l297_297739


namespace determine_p_l297_297836

def is_tangent (circle_eq : ℝ → ℝ → Prop) (parabola_eq : ℝ → ℝ → Prop) (p : ℝ) : Prop :=
  ∃ x y : ℝ, parabola_eq x y ∧ circle_eq x y ∧ x = -p / 2 

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

theorem determine_p (p : ℝ) (hpos : p > 0) :
  (is_tangent circle_eq (parabola_eq p) p) ↔ p = 2 := 
sorry

end determine_p_l297_297836


namespace donut_cubes_eaten_l297_297257

def cube_dimensions := 5

def total_cubes_in_cube : ℕ := cube_dimensions ^ 3

def even_neighbors (faces_sharing_cubes : ℕ) : Prop :=
  faces_sharing_cubes % 2 = 0

/-- A corner cube in a 5x5x5 cube has 3 neighbors. --/
def corner_cube_neighbors := 3

/-- An edge cube in a 5x5x5 cube (excluding corners) has 4 neighbors. --/
def edge_cube_neighbors := 4

/-- A face center cube in a 5x5x5 cube has 5 neighbors. --/
def face_center_cube_neighbors := 5

/-- An inner cube in a 5x5x5 cube has 6 neighbors. --/
def inner_cube_neighbors := 6

/-- Count of edge cubes that share 4 neighbors in a 5x5x5 cube. --/
def edge_cubes_count := 12 * (cube_dimensions - 2)

def inner_cubes_count := (cube_dimensions - 2) ^ 3

theorem donut_cubes_eaten :
  (edge_cubes_count + inner_cubes_count) = 63 := by
  sorry

end donut_cubes_eaten_l297_297257


namespace fish_weight_l297_297145

variable (Γ T : ℝ)
variable (X : ℝ := 1)  -- The tail's weight is given to be 1 kg

theorem fish_weight : 
  (Γ = X + T / 2) → 
  (T = Γ + X) →
  (Γ + T + X = 8) :=
by
  intros h1 h2
  sorry

end fish_weight_l297_297145


namespace response_activity_solutions_l297_297183

theorem response_activity_solutions (x y z : ℕ) :
  5 * x + 4 * y + 3 * z = 15 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1) :=
by
  sorry

end response_activity_solutions_l297_297183


namespace sum_infinite_partial_fraction_l297_297636

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l297_297636


namespace ratio_of_down_payment_l297_297006

theorem ratio_of_down_payment (C D : ℕ) (daily_min : ℕ) (days : ℕ) (balance : ℕ) (total_cost : ℕ) 
  (h1 : total_cost = 120)
  (h2 : daily_min = 6)
  (h3 : days = 10)
  (h4 : balance = daily_min * days) 
  (h5 : D + balance = total_cost) : 
  D / total_cost = 1 / 2 := 
  by
  sorry

end ratio_of_down_payment_l297_297006


namespace rate_per_sq_meter_l297_297274

theorem rate_per_sq_meter (length width : ℝ) (total_cost : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : total_cost = 16500) : 
  total_cost / (length * width) = 800 :=
by
  sorry

end rate_per_sq_meter_l297_297274


namespace add_coefficients_l297_297068

theorem add_coefficients (a : ℕ) : 2 * a + a = 3 * a :=
by 
  sorry

end add_coefficients_l297_297068


namespace exists_strictly_increasing_sequence_l297_297660

theorem exists_strictly_increasing_sequence 
  (N : ℕ) : 
  (∃ (t : ℕ), t^2 ≤ N ∧ N < t^2 + t) →
  (∃ (s : ℕ → ℕ), (∀ n : ℕ, s n < s (n + 1)) ∧ 
   (∃ k : ℕ, ∀ n : ℕ, s (n + 1) - s n = k) ∧
   (∀ n : ℕ, s (s n) - s (s (n - 1)) ≤ N 
      ∧ N < s (1 + s n) - s (s (n - 1)))) :=
by
  sorry

end exists_strictly_increasing_sequence_l297_297660


namespace ben_daily_spending_l297_297469

variable (S : ℕ)

def daily_savings (S : ℕ) : ℕ := 50 - S

def total_savings (S : ℕ) : ℕ := 7 * daily_savings S

def final_amount (S : ℕ) : ℕ := 2 * total_savings S + 10

theorem ben_daily_spending :
  final_amount 15 = 500 :=
by
  unfold final_amount
  unfold total_savings
  unfold daily_savings
  sorry

end ben_daily_spending_l297_297469


namespace macy_miles_left_to_run_l297_297121

-- Define the given conditions
def goal : ℕ := 24
def miles_per_day : ℕ := 3
def days : ℕ := 6

-- Define the statement to be proven
theorem macy_miles_left_to_run :
  goal - (miles_per_day * days) = 6 :=
by
  sorry

end macy_miles_left_to_run_l297_297121


namespace betty_sugar_l297_297338

theorem betty_sugar (f s : ℝ) (hf1 : f ≥ 8 + (3 / 4) * s) (hf2 : f ≤ 3 * s) : s ≥ 4 := 
sorry

end betty_sugar_l297_297338


namespace sqrt_of_16_is_4_l297_297488

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l297_297488


namespace percentage_saved_is_10_l297_297186

-- Given conditions
def rent_expenses : ℕ := 5000
def milk_expenses : ℕ := 1500
def groceries_expenses : ℕ := 4500
def education_expenses : ℕ := 2500
def petrol_expenses : ℕ := 2000
def misc_expenses : ℕ := 3940
def savings : ℕ := 2160

-- Define the total expenses
def total_expenses : ℕ := rent_expenses + milk_expenses + groceries_expenses + education_expenses + petrol_expenses + misc_expenses

-- Define the total monthly salary
def total_monthly_salary : ℕ := total_expenses + savings

-- Define the percentage of savings
def percentage_saved : ℕ := (savings * 100) / total_monthly_salary

-- Prove that the percentage saved is 10%
theorem percentage_saved_is_10 :
  percentage_saved = 10 :=
sorry

end percentage_saved_is_10_l297_297186


namespace arithmetic_sequence_sum_l297_297090

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 3 = 4)
  (h3 : ∀ n, a (n + 1) - a n = a 2 - a 1) :
  a 4 + a 5 = 17 :=
  sorry

end arithmetic_sequence_sum_l297_297090


namespace sum_infinite_series_eq_l297_297633

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l297_297633


namespace sixteen_a_four_plus_one_div_a_four_l297_297021

theorem sixteen_a_four_plus_one_div_a_four (a : ℝ) (h : 2 * a - 1 / a = 3) :
  16 * a^4 + (1 / a^4) = 161 :=
sorry

end sixteen_a_four_plus_one_div_a_four_l297_297021


namespace right_triangle_hypotenuse_l297_297984

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l297_297984


namespace parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l297_297111

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
(2 + Real.sqrt 2 * Real.cos θ, 
 2 + Real.sqrt 2 * Real.sin θ)

theorem parametric_eq_of_curve_C (θ : ℝ) : 
    ∃ x y, 
    (x, y) = curve_C θ ∧ 
    (x - 2)^2 + (y - 2)^2 = 2 := by sorry

theorem max_x_plus_y_on_curve_C :
    ∃ x y θ, 
    (x, y) = curve_C θ ∧ 
    (∀ p : ℝ × ℝ, (p.1, p.2) = curve_C θ → 
    p.1 + p.2 ≤ 6) ∧
    x + y = 6 ∧
    x = 3 ∧ 
    y = 3 := by sorry

end parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l297_297111


namespace joe_lists_count_l297_297052

theorem joe_lists_count : ∃ (n : ℕ), n = 15 * 14 := sorry

end joe_lists_count_l297_297052


namespace min_value_exists_max_value_exists_l297_297500

noncomputable def y (x : ℝ) : ℝ := 3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem min_value_exists :
  (∃ k : ℤ, y (π / 6 + 2 * k * π) = -2) ∧ (∃ k : ℤ, y (5 * π / 6 + 2 * k * π) = -2) :=
by 
  sorry

theorem max_value_exists :
  ∃ k : ℤ, y (-π / 2 + 2 * k * π) = 7 :=
by 
  sorry

end min_value_exists_max_value_exists_l297_297500


namespace annual_population_growth_l297_297569

noncomputable def annual_percentage_increase := 
  let P0 := 15000
  let P2 := 18150  
  exists (r : ℝ), (P0 * (1 + r)^2 = P2) ∧ (r = 0.1)

theorem annual_population_growth : annual_percentage_increase :=
by
  -- Placeholder proof
  sorry

end annual_population_growth_l297_297569


namespace no_five_coin_combination_for_70_cents_l297_297669

/-- Define the values of each coin type -/
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25

/-- Prove that it is not possible to achieve a total value of 70 cents with exactly five coins -/
theorem no_five_coin_combination_for_70_cents :
  ¬ ∃ a b c d e : ℕ, a + b + c + d + e = 5 ∧ a * penny + b * nickel + c * dime + d * quarter + e * quarter = 70 :=
sorry

end no_five_coin_combination_for_70_cents_l297_297669


namespace solution_x_chemical_b_l297_297888

theorem solution_x_chemical_b (percentage_x_a percentage_y_a percentage_y_b : ℝ) :
  percentage_x_a = 0.3 →
  percentage_y_a = 0.4 →
  percentage_y_b = 0.6 →
  (0.8 * percentage_x_a + 0.2 * percentage_y_a = 0.32) →
  (100 * (1 - percentage_x_a) = 70) :=
by {
  sorry
}

end solution_x_chemical_b_l297_297888


namespace p_6_is_126_l297_297247

noncomputable def p (x : ℝ) : ℝ := sorry

axiom h1 : p 1 = 1
axiom h2 : p 2 = 2
axiom h3 : p 3 = 3
axiom h4 : p 4 = 4
axiom h5 : p 5 = 5

theorem p_6_is_126 : p 6 = 126 := sorry

end p_6_is_126_l297_297247


namespace air_conditioner_consumption_l297_297994

theorem air_conditioner_consumption :
  ∀ (total_consumption_8_hours : ℝ)
    (hours_8 : ℝ)
    (hours_per_day : ℝ)
    (days : ℝ),
    total_consumption_8_hours / hours_8 * hours_per_day * days = 27 :=
by
  intros total_consumption_8_hours hours_8 hours_per_day days
  sorry

end air_conditioner_consumption_l297_297994


namespace sum_fraction_series_l297_297625

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l297_297625


namespace find_y_given_conditions_l297_297746

theorem find_y_given_conditions (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 5 * t + 9) (h3 : x = 0) : y = 33 / 2 := by
  sorry

end find_y_given_conditions_l297_297746


namespace range_of_a_for_two_zeros_l297_297381

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∀ x : ℝ, (x + 1) * Real.exp x - a = 0 → -- There's no need to delete this part, see below note 
                                              -- The question of "exactly" is virtually ensured by other parts of the Lean theories
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                (x₁ + 1) * Real.exp x₁ - a = 0 ∧
                (x₂ + 1) * Real.exp x₂ - a = 0) → 
  (-1 / Real.exp 2 < a ∧ a < 0) :=
sorry

end range_of_a_for_two_zeros_l297_297381


namespace pattern_C_not_foldable_without_overlap_l297_297838

-- Define the four patterns, denoted as PatternA, PatternB, PatternC, and PatternD.
inductive Pattern
| A : Pattern
| B : Pattern
| C : Pattern
| D : Pattern

-- Define a predicate for a pattern being foldable into a cube without overlap.
def foldable_into_cube (p : Pattern) : Prop := sorry

theorem pattern_C_not_foldable_without_overlap : ¬ foldable_into_cube Pattern.C := sorry

end pattern_C_not_foldable_without_overlap_l297_297838


namespace container_capacity_l297_297164

theorem container_capacity (C : ℝ) 
  (h1 : (0.30 * C : ℝ) + 27 = 0.75 * C) : C = 60 :=
sorry

end container_capacity_l297_297164


namespace infinite_series_sum_l297_297358

theorem infinite_series_sum :
  ∑' (k : ℕ), (k + 1) / 4^(k + 1) = 4 / 9 :=
sorry

end infinite_series_sum_l297_297358


namespace sum_of_series_l297_297650

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l297_297650


namespace train_travel_time_l297_297940

def travel_time (departure arrival : Nat) : Nat :=
  arrival - departure

theorem train_travel_time : travel_time 425 479 = 54 := by
  sorry

end train_travel_time_l297_297940


namespace solve_floor_equation_l297_297876

theorem solve_floor_equation (x : ℚ) 
  (h : ⌊(5 + 6 * x) / 8⌋ = (15 * x - 7) / 5) : 
  x = 7 / 15 ∨ x = 4 / 5 := 
sorry

end solve_floor_equation_l297_297876


namespace Bianca_pictures_distribution_l297_297995

theorem Bianca_pictures_distribution 
(pictures_total : ℕ) 
(pictures_in_one_album : ℕ) 
(albums_remaining : ℕ) 
(h1 : pictures_total = 33)
(h2 : pictures_in_one_album = 27)
(h3 : albums_remaining = 3)
: (pictures_total - pictures_in_one_album) / albums_remaining = 2 := 
by 
  sorry

end Bianca_pictures_distribution_l297_297995


namespace sequence_length_div_by_four_l297_297906

theorem sequence_length_div_by_four (a : ℕ) (h0 : a = 11664) (H : ∀ n, a = (4 ^ n) * b → b ≠ 0 ∧ n ≤ 3) : 
  ∃ n, n + 1 = 4 :=
by
  sorry

end sequence_length_div_by_four_l297_297906


namespace order_fractions_l297_297002

theorem order_fractions : (16/13 : ℚ) < 21/17 ∧ 21/17 < 20/15 :=
by {
  -- use cross-multiplication:
  -- 16*17 < 21*13 -> 272 < 273 -> true
  -- 16*15 < 20*13 -> 240 < 260 -> true
  -- 21*15 < 20*17 -> 315 < 340 -> true
  sorry
}

end order_fractions_l297_297002


namespace magazine_ad_extra_cost_l297_297902

/--
The cost of purchasing a laptop through a magazine advertisement includes four monthly 
payments of $60.99 each and a one-time shipping and handling fee of $19.99. The in-store 
price of the laptop is $259.99. Prove that purchasing the laptop through the magazine 
advertisement results in an extra cost of 396 cents.
-/
theorem magazine_ad_extra_cost : 
  let in_store_price := 259.99
  let monthly_payment := 60.99
  let num_payments := 4
  let shipping_handling := 19.99
  let total_magazine_cost := (num_payments * monthly_payment) + shipping_handling
  (total_magazine_cost - in_store_price) * 100 = 396 := 
by
  sorry

end magazine_ad_extra_cost_l297_297902


namespace sequence_property_l297_297441

theorem sequence_property (x : ℝ) (a : ℕ → ℝ) (h : ∀ n, a n = 1 + x ^ (n + 1) + x ^ (n + 2)) (h_given : (a 2) ^ 2 = (a 1) * (a 3)) :
  ∀ n ≥ 3, (a n) ^ 2 = (a (n - 1)) * (a (n + 1)) :=
by
  intros n hn
  sorry

end sequence_property_l297_297441


namespace john_initial_bench_weight_l297_297115

variable (B : ℕ)

theorem john_initial_bench_weight (B : ℕ) (HNewTotal : 1490 = 490 + B + 600) : B = 400 :=
by
  sorry

end john_initial_bench_weight_l297_297115


namespace consecutive_vertices_product_l297_297653

theorem consecutive_vertices_product (n : ℕ) (hn : n = 90) :
  ∃ (i : ℕ), 1 ≤ i ∧ i ≤ n ∧ ((i * (i % n + 1)) ≥ 2014) := 
sorry

end consecutive_vertices_product_l297_297653


namespace sum_series_l297_297618

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l297_297618


namespace intersection_A_B_l297_297688

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_A_B : A ∩ B = {2} := by
  -- Proof to be filled
  sorry

end intersection_A_B_l297_297688


namespace rope_for_second_post_l297_297816

theorem rope_for_second_post 
(r1 r2 r3 r4 : ℕ) 
(h_total : r1 + r2 + r3 + r4 = 70)
(h_r1 : r1 = 24)
(h_r3 : r3 = 14)
(h_r4 : r4 = 12) 
: r2 = 20 := 
by 
  sorry

end rope_for_second_post_l297_297816


namespace sequence_recurrence_l297_297364

noncomputable def a (n : ℕ) : ℤ := Int.floor ((1 + Real.sqrt 2) ^ n)

theorem sequence_recurrence (k : ℕ) (h : 2 ≤ k) : 
  ∀ n : ℕ, 
  (a 2 * k = 2 * a (2 * k - 1) + a (2 * k - 2)) ∧
  (a (2 * k + 1) = 2 * a (2 * k) + a (2 * k - 1) + 2) :=
sorry

end sequence_recurrence_l297_297364


namespace binary_mul_correct_l297_297827

def bin_to_nat (l : List ℕ) : ℕ :=
  l.foldl (λ n b => 2 * n + b) 0

def p : List ℕ := [1,0,1,1,0,1]
def q : List ℕ := [1,1,0,1]
def r : List ℕ := [1,0,0,0,1,0,0,0,1,1]

theorem binary_mul_correct :
  bin_to_nat p * bin_to_nat q = bin_to_nat r := by
  sorry

end binary_mul_correct_l297_297827


namespace sum_of_series_l297_297647

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l297_297647


namespace hypotenuse_length_l297_297973

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l297_297973


namespace area_of_inscribed_rectangle_l297_297561

theorem area_of_inscribed_rectangle 
    (DA : ℝ) 
    (GD HD : ℝ) 
    (rectangle_inscribed : ∀ (A B C D G H : Type), true) 
    (radius : ℝ) 
    (GH : ℝ):
    DA = 20 ∧ GD = 5 ∧ HD = 5 ∧ GH = GD + DA + HD ∧ radius = GH / 2 → 
    200 * Real.sqrt 2 = DA * (Real.sqrt (radius^2 - (GD^2))) :=
by
  sorry

end area_of_inscribed_rectangle_l297_297561


namespace grains_in_gray_parts_l297_297502

theorem grains_in_gray_parts (total1 total2 shared : ℕ) (h1 : total1 = 87) (h2 : total2 = 110) (h_shared : shared = 68) :
  (total1 - shared) + (total2 - shared) = 61 :=
by sorry

end grains_in_gray_parts_l297_297502


namespace math_problem_l297_297839

theorem math_problem (x y : ℝ) (h : (x + 2 * y) ^ 3 + x ^ 3 + 2 * x + 2 * y = 0) : x + y - 1 = -1 := 
sorry

end math_problem_l297_297839


namespace discs_contain_equal_minutes_l297_297059

theorem discs_contain_equal_minutes (total_time discs_capacity : ℕ) 
  (h1 : total_time = 520) (h2 : discs_capacity = 65) :
  ∃ discs_needed : ℕ, discs_needed = total_time / discs_capacity ∧ 
  ∀ (k : ℕ), k = total_time / discs_needed → k = 65 :=
by
  sorry

end discs_contain_equal_minutes_l297_297059


namespace evaluate_expression_l297_297077

theorem evaluate_expression : 
  (1 - (2 / 5)) / (1 - (1 / 4)) = (4 / 5) := 
by 
  sorry

end evaluate_expression_l297_297077


namespace sum_fraction_series_l297_297626

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l297_297626


namespace milk_needed_for_cookies_l297_297028

-- Define the given conditions
def liters_to_cups (liters : ℕ) : ℕ := liters * 4

def milk_per_cookies (cups cookies : ℕ) : ℚ := cups / cookies

-- Define the problem statement
theorem milk_needed_for_cookies (h1 : milk_per_cookies 20 30 = milk_per_cookies x 12) : x = 8 :=
sorry

end milk_needed_for_cookies_l297_297028


namespace no_ordered_pairs_no_real_solutions_l297_297210

noncomputable theory
open polynomial

-- Define the conditions for the discriminant being positive meaning no real solutions
def no_real_solutions (b c : ℕ) : Prop :=
  -27 * (c : ℤ)^2 - 4 * (b : ℤ)^3 > 0 ∧ -27 * (b : ℤ)^2 - 4 * (c : ℤ)^3 > 0

-- The main theorem stating there are no such positive integer pairs (b, c)
theorem no_ordered_pairs_no_real_solutions :
  ¬ ∃ b c : ℕ, b > 0 ∧ c > 0 ∧ no_real_solutions b c :=
sorry

end no_ordered_pairs_no_real_solutions_l297_297210


namespace right_triangle_hypotenuse_length_l297_297952

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l297_297952


namespace condition_for_a_pow_zero_eq_one_l297_297571

theorem condition_for_a_pow_zero_eq_one (a : Real) : a ≠ 0 ↔ a^0 = 1 :=
by
  sorry

end condition_for_a_pow_zero_eq_one_l297_297571


namespace melany_fence_l297_297254

-- Definitions
def L (total_budget cost_per_foot : ℝ) : ℝ := total_budget / cost_per_foot
noncomputable def length_not_fenced (perimeter length_bought : ℝ) : ℝ := perimeter - length_bought

-- Constants
def total_budget : ℝ := 120000
def cost_per_foot : ℝ := 30
def perimeter : ℝ := 5000

-- Proof problem in Lean 4 statement
theorem melany_fence : length_not_fenced perimeter (L total_budget cost_per_foot) = 1000 := by
  sorry

end melany_fence_l297_297254


namespace sqrt_sixteen_equals_four_l297_297480

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l297_297480


namespace zoe_correct_percentage_l297_297863

variable (t : ℝ) -- total number of problems

-- Conditions
variable (chloe_solved_fraction : ℝ := 0.60)
variable (zoe_solved_fraction : ℝ := 0.40)
variable (chloe_correct_percentage_alone : ℝ := 0.75)
variable (chloe_correct_percentage_total : ℝ := 0.85)
variable (zoe_correct_percentage_alone : ℝ := 0.95)

theorem zoe_correct_percentage (h1 : chloe_solved_fraction = 0.60)
                               (h2 : zoe_solved_fraction = 0.40)
                               (h3 : chloe_correct_percentage_alone = 0.75)
                               (h4 : chloe_correct_percentage_total = 0.85)
                               (h5 : zoe_correct_percentage_alone = 0.95) :
  (zoe_correct_percentage_alone * zoe_solved_fraction * 100 + (chloe_correct_percentage_total - chloe_correct_percentage_alone * chloe_solved_fraction) * 100 = 78) :=
sorry

end zoe_correct_percentage_l297_297863


namespace geometric_sequence_S4_l297_297013

noncomputable section

def geometric_series_sum (a1 q : ℚ) (n : ℕ) : ℚ := 
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_S4 (a1 : ℚ) (q : ℚ)
  (h1 : a1 * q^3 = 2 * a1)
  (h2 : 5 / 2 = a1 * (q^3 + 2 * q^6)) :
  geometric_series_sum a1 q 4 = 30 := by
  sorry

end geometric_sequence_S4_l297_297013


namespace puppies_left_l297_297811

namespace AlyssaPuppies

def initPuppies : ℕ := 12
def givenAway : ℕ := 7
def remainingPuppies : ℕ := 5

theorem puppies_left (initPuppies givenAway remainingPuppies : ℕ) : 
  initPuppies - givenAway = remainingPuppies :=
by
  sorry

end AlyssaPuppies

end puppies_left_l297_297811


namespace series_sum_l297_297612

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l297_297612


namespace problem_statement_l297_297896

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * (x - 1))

theorem problem_statement :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x1 x2 : ℝ, x1 + x2 = π / 2 → g x1 = g x2) :=
by 
  sorry

end problem_statement_l297_297896


namespace range_of_y_l297_297528

theorem range_of_y (y : ℝ) (h₁ : y < 0) (h₂ : ⌈y⌉ * ⌊y⌋ = 110) : -11 < y ∧ y < -10 := 
sorry

end range_of_y_l297_297528


namespace total_profit_Q2_is_correct_l297_297912

-- Conditions as definitions
def profit_Q1_A := 1500
def profit_Q1_B := 2000
def profit_Q1_C := 1000

def profit_Q2_A := 2500
def profit_Q2_B := 3000
def profit_Q2_C := 1500

def profit_Q3_A := 3000
def profit_Q3_B := 2500
def profit_Q3_C := 3500

def profit_Q4_A := 2000
def profit_Q4_B := 3000
def profit_Q4_C := 2000

-- The total profit calculation for the second quarter
def total_profit_Q2 := profit_Q2_A + profit_Q2_B + profit_Q2_C

-- Proof statement
theorem total_profit_Q2_is_correct : total_profit_Q2 = 7000 := by
  sorry

end total_profit_Q2_is_correct_l297_297912


namespace perimeter_of_square_with_area_36_l297_297925

theorem perimeter_of_square_with_area_36 : 
  ∀ (A : ℝ), A = 36 → (∃ P : ℝ, P = 24 ∧ (∃ s : ℝ, s^2 = A ∧ P = 4 * s)) :=
by
  sorry

end perimeter_of_square_with_area_36_l297_297925


namespace incorrect_table_value_l297_297029

theorem incorrect_table_value (a b c : ℕ) (values : List ℕ) (correct : values = [2051, 2197, 2401, 2601, 2809, 3025, 3249, 3481]) : 
  (2401 ∉ [2051, 2197, 2399, 2601, 2809, 3025, 3249, 3481]) :=
sorry

end incorrect_table_value_l297_297029


namespace sum_of_series_l297_297648

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l297_297648


namespace geometric_series_sum_squares_l297_297196

theorem geometric_series_sum_squares (a r : ℝ) (hr : -1 < r) (hr2 : r < 1) :
  (∑' n : ℕ, a^2 * r^(3 * n)) = a^2 / (1 - r^3) :=
by
  -- Note: Proof goes here
  sorry

end geometric_series_sum_squares_l297_297196


namespace shane_chewed_pieces_l297_297075

theorem shane_chewed_pieces :
  ∀ (Elyse Rick Shane: ℕ),
  Elyse = 100 →
  Rick = Elyse / 2 →
  Shane = Rick / 2 →
  Shane_left = 14 →
  (Shane - Shane_left) = 11 :=
by
  intros Elyse Rick Shane Elyse_def Rick_def Shane_def Shane_left_def
  sorry

end shane_chewed_pieces_l297_297075


namespace vasya_improved_example1_vasya_improved_example2_l297_297778

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l297_297778


namespace fruits_in_good_condition_l297_297924

def percentage_good_fruits (num_oranges num_bananas pct_rotten_oranges pct_rotten_bananas : ℕ) : ℚ :=
  let total_fruits := num_oranges + num_bananas
  let rotten_oranges := (pct_rotten_oranges * num_oranges) / 100
  let rotten_bananas := (pct_rotten_bananas * num_bananas) / 100
  let good_fruits := total_fruits - (rotten_oranges + rotten_bananas)
  (good_fruits * 100) / total_fruits

theorem fruits_in_good_condition :
  percentage_good_fruits 600 400 15 8 = 87.8 := sorry

end fruits_in_good_condition_l297_297924


namespace actual_value_wrongly_copied_l297_297018

theorem actual_value_wrongly_copied (mean_initial : ℝ) (n : ℕ) (wrong_value : ℝ) (mean_correct : ℝ) :
  mean_initial = 140 → n = 30 → wrong_value = 135 → mean_correct = 140.33333333333334 →
  ∃ actual_value : ℝ, actual_value = 145 :=
by
  intros
  sorry

end actual_value_wrongly_copied_l297_297018


namespace range_of_a_l297_297393

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - a| ≥ 5) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by
  sorry

end range_of_a_l297_297393


namespace sally_has_more_cards_l297_297007

def SallyInitial : ℕ := 27
def DanTotal : ℕ := 41
def SallyBought : ℕ := 20
def SallyTotal := SallyInitial + SallyBought

theorem sally_has_more_cards : SallyTotal - DanTotal = 6 := by
  sorry

end sally_has_more_cards_l297_297007


namespace find_c_l297_297085

def conditions (c d : ℝ) : Prop :=
  -- The polynomial 6x^3 + 7cx^2 + 3dx + 2c = 0 has three distinct positive roots
  ∃ u v w : ℝ, 0 < u ∧ 0 < v ∧ 0 < w ∧ u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
  (6 * u^3 + 7 * c * u^2 + 3 * d * u + 2 * c = 0) ∧
  (6 * v^3 + 7 * c * v^2 + 3 * d * v + 2 * c = 0) ∧
  (6 * w^3 + 7 * c * w^2 + 3 * d * w + 2 * c = 0) ∧
  -- Sum of the base-2 logarithms of the roots is 6
  Real.log (u * v * w) / Real.log 2 = 6

theorem find_c (c d : ℝ) (h : conditions c d) : c = -192 :=
sorry

end find_c_l297_297085


namespace hypotenuse_right_triangle_l297_297987

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l297_297987


namespace series_sum_l297_297611

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l297_297611


namespace value_of_x_squared_minus_y_squared_l297_297533

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l297_297533


namespace sum_fraction_series_l297_297622

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l297_297622


namespace roll_probability_l297_297187

noncomputable def probability_allison_rolls_greater : ℚ :=
  let p_brian := 5 / 6  -- Probability of Brian rolling 5 or lower
  let p_noah := 1       -- Probability of Noah rolling 5 or lower (since all faces roll 5 or lower)
  p_brian * p_noah

theorem roll_probability :
  probability_allison_rolls_greater = 5 / 6 := by
  sorry

end roll_probability_l297_297187


namespace quadratic_inequality_l297_297127

theorem quadratic_inequality (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) : ∀ x : ℝ, c * x^2 - b * x + a > c * x - b := 
by
  sorry

end quadratic_inequality_l297_297127


namespace johns_total_weekly_gas_consumption_l297_297410

-- Definitions of conditions
def highway_mpg : ℝ := 30
def city_mpg : ℝ := 25
def work_miles_each_way : ℝ := 20
def work_days_per_week : ℝ := 5
def highway_miles_each_way : ℝ := 15
def city_miles_each_way : ℝ := 5
def leisure_highway_miles_per_week : ℝ := 30
def leisure_city_miles_per_week : ℝ := 10
def idling_gas_consumption_per_week : ℝ := 0.3

-- Proof problem
theorem johns_total_weekly_gas_consumption :
  let work_commute_miles_per_week := work_miles_each_way * 2 * work_days_per_week
  let highway_miles_work := highway_miles_each_way * 2 * work_days_per_week
  let city_miles_work := city_miles_each_way * 2 * work_days_per_week
  let total_highway_miles := highway_miles_work + leisure_highway_miles_per_week
  let total_city_miles := city_miles_work + leisure_city_miles_per_week
  let highway_gas_consumption := total_highway_miles / highway_mpg
  let city_gas_consumption := total_city_miles / city_mpg
  (highway_gas_consumption + city_gas_consumption + idling_gas_consumption_per_week) = 8.7 := by
  sorry

end johns_total_weekly_gas_consumption_l297_297410


namespace angle_bisector_slope_l297_297064

-- Definitions of the conditions
def line1_slope := 2
def line2_slope := 4

-- The proof statement: Prove that the slope of the angle bisector is -12/7
theorem angle_bisector_slope : (line1_slope + line2_slope + Real.sqrt (line1_slope^2 + line2_slope^2 + 2 * line1_slope * line2_slope)) / 
                               (1 - line1_slope * line2_slope) = -12/7 :=
by
  sorry

end angle_bisector_slope_l297_297064


namespace opposite_of_seven_l297_297278

theorem opposite_of_seven : ∃ x : ℤ, 7 + x = 0 ∧ x = -7 :=
by
  sorry

end opposite_of_seven_l297_297278


namespace determine_b_l297_297071

noncomputable def has_exactly_one_real_solution (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - b*x^3 - 3*b*x + b^2 - 2 = 0 ∧ ∀ y : ℝ, y ≠ x → y^4 - b*y^3 - 3*b*y + b^2 - 2 ≠ 0

theorem determine_b (b : ℝ) :
  has_exactly_one_real_solution b → b < 7 / 4 :=
by
  sorry

end determine_b_l297_297071


namespace solve_linear_system_l297_297436

theorem solve_linear_system :
  ∃ x y z : ℝ, 
    (2 * x + y + z = -1) ∧ 
    (3 * y - z = -1) ∧ 
    (3 * x + 2 * y + 3 * z = -5) ∧ 
    (x = 1) ∧ 
    (y = -1) ∧ 
    (z = -2) :=
by
  sorry

end solve_linear_system_l297_297436


namespace arithmetic_sequence_a5_l297_297546

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : a 5 = 5 :=
sorry

end arithmetic_sequence_a5_l297_297546


namespace find_f_2_l297_297510

-- Condition: f(x + 1) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Statement to prove
theorem find_f_2 : f 2 = -1 := by
  sorry

end find_f_2_l297_297510


namespace nth_term_arithmetic_seq_l297_297091

variable (a_n : Nat → Int)
variable (S : Nat → Int)
variable (a_1 : Int)

-- Conditions
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
  ∃ d : Int, ∀ n : Nat, a_n (n + 1) = a_n n + d

def first_term (a_1 : Int) : Prop :=
  a_1 = 1

def sum_first_three_terms (S : Nat → Int) : Prop :=
  S 3 = 9

theorem nth_term_arithmetic_seq :
  (is_arithmetic_sequence a_n) →
  (first_term 1) →
  (sum_first_three_terms S) →
  ∀ n : Nat, a_n n = 2 * n - 1 :=
  sorry

end nth_term_arithmetic_seq_l297_297091


namespace find_n_l297_297372

theorem find_n (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ) (h : (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7
                      = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 29 - 7) : 7 = 7 :=
by
  sorry

end find_n_l297_297372


namespace liam_annual_income_l297_297909

theorem liam_annual_income (q : ℝ) (I : ℝ) (T : ℝ) 
  (h1 : T = (q + 0.5) * 0.01 * I) 
  (h2 : I > 50000) 
  (h3 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000)) : 
  I = 56000 :=
by
  sorry

end liam_annual_income_l297_297909


namespace right_triangle_hypotenuse_length_l297_297951

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l297_297951


namespace solve_eq_l297_297892

theorem solve_eq {x y z : ℕ} :
  2^x + 3^y - 7 = z! ↔ (x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry -- Proof should be provided here

end solve_eq_l297_297892


namespace find_side_lengths_and_angles_of_triangle_by_tangency_points_l297_297113

noncomputable def side_lengths_and_angles (a1 b1 c1 : ℝ) (α β γ a b c : ℝ) : Prop :=
  ∃ r s,
    s = (a1 + b1 + c1) / 2 ∧ 
    r = (a1 * b1 * c1) / (4 * real.sqrt (s * (s - a1) * (s - b1) * (s - c1))) ∧
    α ≈ 2 * real.arccos (a1 / (2 * r)) ∧
    β ≈ 2 * real.arccos (b1 / (2 * r)) ∧
    γ ≈ 2 * real.arccos (c1 / (2 * r)) ∧
    a ≈ r * (real.cot (real.arccos (b1 / (2 * r))) + real.cot (real.arccos (c1 / (2 * r)))) ∧
    b ≈ r * (real.cot (real.arccos (a1 / (2 * r))) + real.cot (real.arccos (c1 / (2 * r)))) ∧
    c ≈ r * (real.cot (real.arccos (a1 / (2 * r))) + real.cot (real.arccos (b1 / (2 * r))))

theorem find_side_lengths_and_angles_of_triangle_by_tangency_points :
  side_lengths_and_angles 25 29 36 
    92.7944 (deg) 73.7392 (deg) 13.4658 (deg)
    177.7 170.8 41.4 :=
  sorry

end find_side_lengths_and_angles_of_triangle_by_tangency_points_l297_297113


namespace calculate_revolutions_l297_297997

def wheel_diameter : ℝ := 8
def distance_traveled_miles : ℝ := 0.5
def feet_per_mile : ℝ := 5280
def distance_traveled_feet : ℝ := distance_traveled_miles * feet_per_mile

theorem calculate_revolutions :
  let radius : ℝ := wheel_diameter / 2
  let circumference : ℝ := 2 * Real.pi * radius
  let revolutions : ℝ := distance_traveled_feet / circumference
  revolutions = 330 / Real.pi := by
  sorry

end calculate_revolutions_l297_297997


namespace path_length_l297_297801

theorem path_length (scale_ratio : ℕ) (map_path_length : ℝ) 
  (h1 : scale_ratio = 500)
  (h2 : map_path_length = 3.5) : 
  (map_path_length * scale_ratio = 1750) :=
sorry

end path_length_l297_297801


namespace fifteenth_permutation_is_6318_l297_297440

-- Define the set of digits
def digits : Finset ℕ := {1, 3, 6, 8} 

-- Define the factorial function
def fact (n : ℕ) : ℕ := nat.factorial n

-- Function to generate permutations
def permutations (s : Finset ℕ) : Finset (List ℕ) :=
  s.val.permutations.filter (λ l, l.length = s.card)

-- The specific permutation we are interested in
-- We assume the permutations are sorted lexicographically
def nth_permutation (s : Finset ℕ) (n : ℕ) : List ℕ :=
  list.quicksort (≤) (s.val.permutations) !! (n - 1)

theorem fifteenth_permutation_is_6318 : nth_permutation digits 15 = [6, 3, 1, 8] := by
  sorry

end fifteenth_permutation_is_6318_l297_297440


namespace vasya_problem_l297_297766

theorem vasya_problem : 
  ∃ (e : ℝ), 
  (e = 333 / 3 - 33 / 3 ∨ e = 33 * 3 + 3 / 3) ∧
  (count threes in expression e < 10) ∧
  (e = 100) :=
by sorry

end vasya_problem_l297_297766


namespace imaginary_part_of_complex_z_l297_297901

noncomputable def complex_z : ℂ := (1 + Complex.I) / (1 - Complex.I) + (1 - Complex.I) ^ 2

theorem imaginary_part_of_complex_z : complex_z.im = -1 := by
  sorry

end imaginary_part_of_complex_z_l297_297901


namespace minimum_shirts_for_savings_l297_297184

theorem minimum_shirts_for_savings (x : ℕ) : 75 + 8 * x < 16 * x ↔ 10 ≤ x :=
by
  sorry

end minimum_shirts_for_savings_l297_297184


namespace f_g_3_value_l297_297223

def f (x : ℝ) := x^3 + 1
def g (x : ℝ) := 3 * x + 2

theorem f_g_3_value : f (g 3) = 1332 := by
  sorry

end f_g_3_value_l297_297223


namespace difference_of_squares_l297_297591

theorem difference_of_squares 
  (x y : ℝ) 
  (optionA := (-x + y) * (x + y))
  (optionB := (-x + y) * (x - y))
  (optionC := (x + 2) * (2 + x))
  (optionD := (2 * x + 3) * (3 * x - 2)) :
  optionA = -(x + y)^2 ∨ optionA = (x + y) * (y - x) :=
sorry

end difference_of_squares_l297_297591


namespace Collin_savings_l297_297344

-- Definitions used in Lean 4 statement based on conditions.
def cans_from_home : ℕ := 12
def cans_from_grandparents : ℕ := 3 * cans_from_home
def cans_from_neighbor : ℕ := 46
def cans_from_dad : ℕ := 250
def price_per_can : ℝ := 0.25

-- Calculations based on the problem
def total_cans : ℕ := cans_from_home + cans_from_grandparents + cans_from_neighbor + cans_from_dad
def total_money : ℝ := price_per_can * total_cans
def savings : ℝ := total_money / 2

-- Statement to prove
theorem Collin_savings : savings = 43 := by
  sorry

end Collin_savings_l297_297344


namespace hypotenuse_length_l297_297968

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l297_297968


namespace polynomial_not_factorable_l297_297131

theorem polynomial_not_factorable :
  ¬ ∃ (A B : Polynomial ℤ), A.degree < 5 ∧ B.degree < 5 ∧ A * B = (Polynomial.C 1 * Polynomial.X ^ 5 - Polynomial.C 3 * Polynomial.X ^ 4 + Polynomial.C 6 * Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X ^ 2 + Polynomial.C 9 * Polynomial.X - Polynomial.C 6) :=
by
  sorry

end polynomial_not_factorable_l297_297131


namespace sqrt_sixteen_is_four_l297_297484

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l297_297484


namespace length_ratio_is_correct_width_ratio_is_correct_l297_297180

-- Definitions based on the conditions
def room_length : ℕ := 25
def room_width : ℕ := 15

-- Calculated perimeter
def room_perimeter : ℕ := 2 * (room_length + room_width)

-- Ratios to be proven
def length_to_perimeter_ratio : ℚ := room_length / room_perimeter
def width_to_perimeter_ratio : ℚ := room_width / room_perimeter

-- Stating the theorems to be proved
theorem length_ratio_is_correct : length_to_perimeter_ratio = 5 / 16 :=
by sorry

theorem width_ratio_is_correct : width_to_perimeter_ratio = 3 / 16 :=
by sorry

end length_ratio_is_correct_width_ratio_is_correct_l297_297180


namespace linear_function_quadrants_l297_297799

theorem linear_function_quadrants : 
  ∀ (x y : ℝ), y = -5 * x + 3 
  → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by 
  intro x y h
  sorry

end linear_function_quadrants_l297_297799


namespace eq_holds_for_n_l297_297072

theorem eq_holds_for_n (n : ℕ) (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a + b + c + d = n * Real.sqrt (a * b * c * d) ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

end eq_holds_for_n_l297_297072


namespace sqrt_sixteen_equals_four_l297_297483

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l297_297483


namespace minimum_value_of_a_l297_297684

variable (a x y : ℝ)

-- Condition
def condition (x y : ℝ) (a : ℝ) : Prop := 
  (x + y) * ((1/x) + (a/y)) ≥ 9

-- Main statement
theorem minimum_value_of_a : (∀ x > 0, ∀ y > 0, condition x y a) → a ≥ 4 :=
sorry

end minimum_value_of_a_l297_297684


namespace six_diggers_five_hours_l297_297290

theorem six_diggers_five_hours (holes_per_hour_per_digger : ℝ) 
  (h1 : 3 * holes_per_hour_per_digger * 3 = 3) :
  6 * (holes_per_hour_per_digger) * 5 = 10 :=
by
  -- The proof will go here, but we only need to state the theorem
  sorry

end six_diggers_five_hours_l297_297290


namespace adults_attended_l297_297466

def adult_ticket_cost : ℕ := 25
def children_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400

theorem adults_attended (A C: ℕ) (h1 : adult_ticket_cost * A + children_ticket_cost * C = total_receipts)
                       (h2 : A + C = total_attendance) : A = 120 :=
by
  sorry

end adults_attended_l297_297466


namespace profit_equation_l297_297181

noncomputable def price_and_profit (x : ℝ) : ℝ :=
  (1 + 0.5) * x * 0.8 - x

theorem profit_equation : ∀ x : ℝ, price_and_profit x = 8 → ((1 + 0.5) * x * 0.8 - x = 8) :=
 by intros x h
    exact h

end profit_equation_l297_297181


namespace car_storm_distance_30_l297_297795

noncomputable def car_position (t : ℝ) : ℝ × ℝ :=
  (0, 3/4 * t)

noncomputable def storm_center (t : ℝ) : ℝ × ℝ :=
  (150 - (3/4 / Real.sqrt 2) * t, -(3/4 / Real.sqrt 2) * t)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem car_storm_distance_30 :
  ∃ (t : ℝ), distance (car_position t) (storm_center t) = 30 :=
sorry

end car_storm_distance_30_l297_297795


namespace result_of_operation_given_y_l297_297905

def operation (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem result_of_operation_given_y :
  ∀ (y : ℤ), y = 11 → operation y 10 = 90 :=
by
  intros y hy
  rw [hy]
  show operation 11 10 = 90
  sorry

end result_of_operation_given_y_l297_297905


namespace initial_weight_l297_297325

theorem initial_weight (W : ℝ) (h₁ : W > 0): 
  W * 0.85 * 0.75 * 0.90 = 450 := 
by 
  sorry

end initial_weight_l297_297325


namespace intersection_M_N_l297_297100

def M := {y : ℝ | y <= 4}
def N := {x : ℝ | x > 0}

theorem intersection_M_N : {x : ℝ | x > 0} ∩ {y : ℝ | y <= 4} = {z : ℝ | 0 < z ∧ z <= 4} :=
by
  sorry

end intersection_M_N_l297_297100


namespace number_of_possible_values_of_a_l297_297012

theorem number_of_possible_values_of_a :
  ∃ (a_values : Finset ℕ), 
    (∀ a ∈ a_values, 3 ∣ a ∧ a ∣ 18 ∧ a ∣ 27 ∧ 0 < a) ∧
    a_values.card = 2 :=
by
  sorry

end number_of_possible_values_of_a_l297_297012


namespace fourth_person_height_l297_297027

theorem fourth_person_height (h : ℝ)
  (h2 : h + 2 = h₂)
  (h3 : h + 4 = h₃)
  (h4 : h + 10 = h₄)
  (average_height : (h + h₂ + h₃ + h₄) / 4 = 77) :
  h₄ = 83 :=
by
  sorry

end fourth_person_height_l297_297027


namespace probability_three_dice_sum_to_fourth_l297_297211

-- Define the probability problem conditions
def total_outcomes : ℕ := 8^4
def favorable_outcomes : ℕ := 1120

-- Final probability for the problem
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Lean statement for the proof problem
theorem probability_three_dice_sum_to_fourth :
  probability favorable_outcomes total_outcomes = 35 / 128 :=
by sorry

end probability_three_dice_sum_to_fourth_l297_297211


namespace simplify_expression_l297_297725

variables {a b : ℝ}

-- Define the conditions
def condition (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a^4 + b^4 = a + b)

-- Define the target goal
def goal (a b : ℝ) : Prop := 
  (a / b + b / a - 1 / (a * b^2)) = (-a - b) / (a * b^2)

-- Statement of the theorem
theorem simplify_expression (h : condition a b) : goal a b :=
by 
  sorry

end simplify_expression_l297_297725


namespace equilibrium_stability_l297_297720

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x - 2)

theorem equilibrium_stability (x : ℝ) :
  (x = 0 → HasDerivAt f (-1) 0 ∧ (-1 < 0)) ∧
  (x = Real.log 2 → HasDerivAt f (2 * Real.log 2) (Real.log 2) ∧ (2 * Real.log 2 > 0)) :=
by
  sorry

end equilibrium_stability_l297_297720


namespace tan_beta_solution_l297_297675

theorem tan_beta_solution
  (α β : ℝ)
  (h₁ : Real.tan α = 2)
  (h₂ : Real.tan (α + β) = -1) :
  Real.tan β = 3 := 
sorry

end tan_beta_solution_l297_297675


namespace problem_ineq_l297_297518

theorem problem_ineq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq : a ≠ b) :
  (a^2 * b + a + b^2) * (a * b^2 + a^2 + b) > 9 * a^2 * b^2 := 
by 
  sorry

end problem_ineq_l297_297518


namespace largest_divisor_of_n_cube_minus_n_minus_six_l297_297826

theorem largest_divisor_of_n_cube_minus_n_minus_six (n : ℤ) : 6 ∣ (n^3 - n - 6) :=
by sorry

end largest_divisor_of_n_cube_minus_n_minus_six_l297_297826


namespace sin_2alpha_over_cos_alpha_sin_beta_value_l297_297218

variable (α β : ℝ)

-- Given conditions
axiom alpha_pos : 0 < α
axiom alpha_lt_pi_div_2 : α < Real.pi / 2
axiom beta_pos : 0 < β
axiom beta_lt_pi_div_2 : β < Real.pi / 2
axiom cos_alpha_eq : Real.cos α = 3 / 5
axiom cos_beta_plus_alpha_eq : Real.cos (β + α) = 5 / 13

-- The results to prove
theorem sin_2alpha_over_cos_alpha : (Real.sin (2 * α) / (Real.cos α ^ 2 + Real.cos (2 * α)) = 12) :=
sorry

theorem sin_beta_value : (Real.sin β = 16 / 65) :=
sorry


end sin_2alpha_over_cos_alpha_sin_beta_value_l297_297218


namespace max_next_person_weight_l297_297741

def avg_weight_adult := 150
def avg_weight_child := 70
def max_weight_elevator := 1500
def num_adults := 7
def num_children := 5

def total_weight_adults := num_adults * avg_weight_adult
def total_weight_children := num_children * avg_weight_child
def current_weight := total_weight_adults + total_weight_children

theorem max_next_person_weight : 
  max_weight_elevator - current_weight = 100 := 
by 
  sorry

end max_next_person_weight_l297_297741


namespace principal_amount_borrowed_l297_297165

theorem principal_amount_borrowed
  (R : ℝ) (T : ℝ) (SI : ℝ) (P : ℝ) 
  (hR : R = 12) 
  (hT : T = 20) 
  (hSI : SI = 2100) 
  (hFormula : SI = (P * R * T) / 100) : 
  P = 875 := 
by 
  -- Assuming the initial steps 
  sorry

end principal_amount_borrowed_l297_297165


namespace range_of_a_for_two_zeros_l297_297380

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∀ x : ℝ, (x + 1) * Real.exp x - a = 0 → -- There's no need to delete this part, see below note 
                                              -- The question of "exactly" is virtually ensured by other parts of the Lean theories
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                (x₁ + 1) * Real.exp x₁ - a = 0 ∧
                (x₂ + 1) * Real.exp x₂ - a = 0) → 
  (-1 / Real.exp 2 < a ∧ a < 0) :=
sorry

end range_of_a_for_two_zeros_l297_297380


namespace sum_fraction_series_l297_297623

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l297_297623


namespace sqrt_of_sixteen_l297_297477

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_sixteen_l297_297477


namespace right_triangle_hypotenuse_length_l297_297962

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l297_297962


namespace minimum_value_of_expression_l297_297095

theorem minimum_value_of_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) :
  ∃ (x : ℝ), x = (1 / (a - 1) + 9 / (b - 1)) ∧ x = 6 :=
by
  sorry

end minimum_value_of_expression_l297_297095


namespace strawberry_unit_prices_l297_297594

theorem strawberry_unit_prices (x y : ℝ) (h1 : x = 1.5 * y) (h2 : 2 * x - 2 * y = 10) : x = 15 ∧ y = 10 :=
by
  sorry

end strawberry_unit_prices_l297_297594


namespace sqrt_of_16_is_4_l297_297489

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l297_297489


namespace solve_system_of_equations_l297_297745

theorem solve_system_of_equations :
  ∃ x y : ℚ, (4 * x - 6 * y = -3) ∧ (8 * x + 3 * y = 6) ∧ (x + y = 1.25) :=
by
  use 9/20, 8/10  -- providing the solutions for x and y
  split
  { -- proving the first equation holds
    norm_num
    exact rfl },
  split
  { -- proving the second equation holds
    norm_num
    exact rfl },
  { -- proving the sum of the solutions is correct
    norm_num
    exact rfl }

end solve_system_of_equations_l297_297745


namespace max_value_of_y_no_min_value_l297_297576

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem max_value_of_y_no_min_value :
  (∃ x, -2 < x ∧ x < 2 ∧ function_y x = 5) ∧
  (∀ y, ∃ x, -2 < x ∧ x < 2 ∧ function_y x >= y) :=
by
  sorry

end max_value_of_y_no_min_value_l297_297576


namespace g_at_10_l297_297497

noncomputable def g (n : ℕ) : ℝ := sorry

axiom g_definition : g 2 = 4
axiom g_recursive : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (3 * g (2 * m) + g (2 * n)) / 4

theorem g_at_10 : g 10 = 64 := sorry

end g_at_10_l297_297497


namespace find_a_plus_b_plus_c_l297_297504

-- Definitions of conditions
def is_vertex (a b c : ℝ) (vertex_x vertex_y : ℝ) := 
  ∀ x : ℝ, vertex_y = (a * (vertex_x ^ 2)) + (b * vertex_x) + c

def contains_point (a b c : ℝ) (x y : ℝ) := 
  y = (a * (x ^ 2)) + (b * x) + c

theorem find_a_plus_b_plus_c
  (a b c : ℝ)
  (h_vertex : is_vertex a b c 3 4)
  (h_symmetry : ∃ h : ℝ, ∀ x : ℝ, a * (x - h) ^ 2 = a * (h - x) ^ 2)
  (h_contains : contains_point a b c 1 0)
  : a + b + c = 0 := 
sorry

end find_a_plus_b_plus_c_l297_297504


namespace calculate_fraction_product_l297_297188

noncomputable def b8 := 2 * (8^2) + 6 * (8^1) + 2 * (8^0) -- 262_8 in base 10
noncomputable def b4 := 1 * (4^1) + 3 * (4^0) -- 13_4 in base 10
noncomputable def b7 := 1 * (7^2) + 4 * (7^1) + 4 * (7^0) -- 144_7 in base 10
noncomputable def b5 := 2 * (5^1) + 4 * (5^0) -- 24_5 in base 10

theorem calculate_fraction_product : 
  ((b8 : ℕ) / (b4 : ℕ)) * ((b7 : ℕ) / (b5 : ℕ)) = 147 :=
by
  sorry

end calculate_fraction_product_l297_297188


namespace train_length_correct_l297_297327

open Real

-- Define the conditions
def bridge_length : ℝ := 150
def time_to_cross_bridge : ℝ := 7.5
def time_to_cross_lamp_post : ℝ := 2.5

-- Define the length of the train
def train_length : ℝ := 75

theorem train_length_correct :
  ∃ L : ℝ, (L / time_to_cross_lamp_post = (L + bridge_length) / time_to_cross_bridge) ∧ L = train_length :=
by
  sorry

end train_length_correct_l297_297327


namespace odd_function_f_x_pos_l297_297267

variable (f : ℝ → ℝ)

theorem odd_function_f_x_pos {x : ℝ} (h1 : ∀ x < 0, f x = x^2 + x)
  (h2 : ∀ x, f x = -f (-x)) (hx : 0 < x) :
  f x = -x^2 + x := by
  sorry

end odd_function_f_x_pos_l297_297267


namespace cone_dimensions_l297_297454

noncomputable def cone_height (r_sector : ℝ) (r_cone_base : ℝ) : ℝ :=
  Real.sqrt (r_sector^2 - r_cone_base^2)

noncomputable def cone_volume (radius : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * radius^2 * height

theorem cone_dimensions 
  (r_circle : ℝ) (num_sectors : ℕ) (r_cone_base : ℝ) :
  r_circle = 12 → num_sectors = 4 → r_cone_base = 3 → 
  cone_height r_circle r_cone_base = 3 * Real.sqrt 15 ∧ 
  cone_volume r_cone_base (cone_height r_circle r_cone_base) = 9 * Real.pi * Real.sqrt 15 :=
by
  intros
  sorry

end cone_dimensions_l297_297454


namespace total_games_in_season_l297_297182

theorem total_games_in_season (teams: ℕ) (division_teams: ℕ) (intra_division_games: ℕ) (inter_division_games: ℕ) (total_games: ℕ) : 
  teams = 18 → division_teams = 9 → intra_division_games = 3 → inter_division_games = 2 → total_games = 378 :=
by
  sorry

end total_games_in_season_l297_297182


namespace bricks_in_wall_is_720_l297_297439

/-- 
Two bricklayers have varying speeds: one could build a wall in 12 hours and 
the other in 15 hours if working alone. Their efficiency decreases by 12 bricks
per hour when they work together. The contractor placed them together on this 
project and the wall was completed in 6 hours.
Prove that the number of bricks in the wall is 720.
-/
def number_of_bricks_in_wall (y : ℕ) : Prop :=
  let rate1 := y / 12
  let rate2 := y / 15
  let combined_rate := rate1 + rate2 - 12
  6 * combined_rate = y

theorem bricks_in_wall_is_720 : ∃ y : ℕ, number_of_bricks_in_wall y ∧ y = 720 :=
  by sorry

end bricks_in_wall_is_720_l297_297439


namespace boxes_to_eliminate_l297_297109

noncomputable def total_boxes : ℕ := 26
noncomputable def high_value_boxes : ℕ := 6
noncomputable def threshold_probability : ℚ := 1 / 2

-- Define the condition for having the minimum number of boxes
def min_boxes_needed_for_probability (total high_value : ℕ) (prob : ℚ) : ℕ :=
  total - high_value - ((total - high_value) / 2)

theorem boxes_to_eliminate :
  min_boxes_needed_for_probability total_boxes high_value_boxes threshold_probability = 15 :=
by
  sorry

end boxes_to_eliminate_l297_297109


namespace geometric_sequence_second_term_l297_297150

theorem geometric_sequence_second_term (b : ℝ) (hb : b > 0) 
  (h1 : ∃ r : ℝ, 210 * r = b) 
  (h2 : ∃ r : ℝ, b * r = 135 / 56) : 
  b = 22.5 := 
sorry

end geometric_sequence_second_term_l297_297150


namespace white_tshirts_per_pack_l297_297496

-- Define the given conditions
def packs_white := 5
def packs_blue := 3
def t_shirts_per_blue_pack := 9
def total_t_shirts := 57

-- Define the total number of blue t-shirts
def total_blue_t_shirts := packs_blue * t_shirts_per_blue_pack

-- Define the variable W for the number of white t-shirts per pack
variable (W : ℕ)

-- Define the total number of white t-shirts
def total_white_t_shirts := packs_white * W

-- State the theorem to prove
theorem white_tshirts_per_pack :
    total_white_t_shirts + total_blue_t_shirts = total_t_shirts → W = 6 :=
by
  sorry

end white_tshirts_per_pack_l297_297496


namespace right_triangle_hypotenuse_length_l297_297964

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l297_297964


namespace speed_of_bus_l297_297174

def distance : ℝ := 500.04
def time : ℝ := 20.0
def conversion_factor : ℝ := 3.6

theorem speed_of_bus :
  (distance / time) * conversion_factor = 90.0072 := 
sorry

end speed_of_bus_l297_297174


namespace log_b_1024_number_of_positive_integers_b_l297_297527

theorem log_b_1024 (b : ℕ) : (∃ n : ℕ, b^n = 1024) ↔ b ∈ {2, 4, 32, 1024} :=
by sorry

theorem number_of_positive_integers_b : (∃ b : ℕ, ∃ n : ℕ, b^n = 1024 ∧ n > 0) ↔ 4 :=
by {
  have h := log_b_1024,
  sorry
}

end log_b_1024_number_of_positive_integers_b_l297_297527


namespace trig_expression_value_l297_297843

theorem trig_expression_value (θ : ℝ) (h1 : ∀ x : ℝ, 3 * Real.sin x + 4 * Real.cos x ≤ 5)
(h2 : 3 * Real.sin θ + 4 * Real.cos θ = 5)
(h3 : Real.sin θ = 3 / 5)
(h4 : Real.cos θ = 4 / 5) :
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / (Real.cos (2 * θ)) = 15 / 7 := 
sorry

end trig_expression_value_l297_297843


namespace sin_double_angle_l297_297087

theorem sin_double_angle (α : ℝ) (h_tan : Real.tan α < 0) (h_sin : Real.sin α = - (Real.sqrt 3) / 3) :
  Real.sin (2 * α) = - (2 * Real.sqrt 2) / 3 := 
by
  sorry

end sin_double_angle_l297_297087


namespace general_solution_of_diff_eq_l297_297360

theorem general_solution_of_diff_eq {C1 C2 : ℝ} (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = C1 * Real.exp (-x) + C2 * Real.exp (-2 * x) + x^2 - 5 * x - 2) →
  (∀ x, (deriv (deriv y)) x + 3 * (deriv y) x + 2 * y x = 2 * x^2 - 4 * x - 17) :=
by
  intro hy
  sorry

end general_solution_of_diff_eq_l297_297360


namespace find_x_l297_297074

variables (t x : ℕ)

theorem find_x (h1 : 0 < t) (h2 : t = 4) (h3 : ((9 / 10 : ℚ) * (t * x : ℚ)) - 6 = 48) : x = 15 :=
by
  sorry

end find_x_l297_297074


namespace area_excluding_hole_correct_l297_297177

def large_rectangle_area (x: ℝ) : ℝ :=
  4 * (x + 7) * (x + 5)

def hole_area (x: ℝ) : ℝ :=
  9 * (2 * x - 3) * (x - 2)

def area_excluding_hole (x: ℝ) : ℝ :=
  large_rectangle_area x - hole_area x

theorem area_excluding_hole_correct (x: ℝ) :
  area_excluding_hole x = -14 * x^2 + 111 * x + 86 :=
by
  -- The proof is omitted
  sorry

end area_excluding_hole_correct_l297_297177


namespace part_I_part_II_l297_297844

-- Part (I) 
theorem part_I (a b : ℝ) : (∀ x : ℝ, x^2 - 5 * a * x + b > 0 ↔ (x > 4 ∨ x < 1)) → 
(a = 1 ∧ b = 4) :=
by { sorry }

-- Part (II) 
theorem part_II (x y : ℝ) (a b : ℝ) (h : x + y = 2 ∧ a = 1 ∧ b = 4) : 
x > 0 → y > 0 → 
(∃ t : ℝ, t = a / x + b / y ∧ t ≥ 9 / 2) :=
by { sorry }

end part_I_part_II_l297_297844


namespace determine_b_l297_297353

variable (a b c : ℝ)

theorem determine_b
  (h1 : -a / 3 = -c)
  (h2 : 1 + a + b + c = -c)
  (h3 : c = 5) :
  b = -26 :=
by
  sorry

end determine_b_l297_297353


namespace probability_of_cycles_l297_297249

noncomputable def Probability_of_Cycles (n : ℕ) (f : ℕ → ℕ) (a : ℕ) :=
  let cycle := {b // ∃ c, b ≥ 1 ∧ c ≥ 1 ∧ (f^[b] 1 = a) ∧ (f^[c] a = 1)}
  classical.some (set.exists_of_finite_of_ne_empty (set.finite_of_finite_cycles (finset.univ (fin n))))

theorem probability_of_cycles (n : ℕ) (a : ℕ) (h_a : a ∈ finset.univ (fin n)) :
  ∃ (f : fin n → fin n), Probability_of_Cycles n f a = 1 / n :=
sorry

end probability_of_cycles_l297_297249


namespace calculate_expression_l297_297191

theorem calculate_expression : abs (-2) - Real.sqrt 4 + 3^2 = 9 := by
  sorry

end calculate_expression_l297_297191


namespace digit_after_decimal_is_4_l297_297110

noncomputable def sum_fractions : ℚ := (2 / 9) + (3 / 11)

theorem digit_after_decimal_is_4 :
  (sum_fractions - sum_fractions.floor) * 10 = 4 :=
by
  sorry

end digit_after_decimal_is_4_l297_297110


namespace distinct_real_roots_of_quadratic_l297_297260

/-
Given a quadratic equation x^2 + 4x = 0,
prove that the equation has two distinct real roots.
-/

theorem distinct_real_roots_of_quadratic : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 0 → (b^2 - 4 * a * c) > 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ (r₁^2 + 4 * r₁ = 0) ∧ (r₂^2 + 4 * r₂ = 0) := 
by
  intros a b c ha hb hc hΔ
  sorry -- Proof to be provided later

end distinct_real_roots_of_quadratic_l297_297260


namespace diff_of_squares_l297_297592

theorem diff_of_squares (x y : ℝ) :
  (∃ a b : ℝ, (a - b) * (a + b) = (-x + y) * (x + y)) ∧ 
  ¬ (∃ a b : ℝ, (a - b) * (a + b) = (-x + y) * (x - y)) ∧ 
  ¬ (∃ a b : ℝ, (a - b) * (a + b) = (x + 2) * (2 + x)) ∧ 
  ¬ (∃ a b : ℝ, (a - b) * (a + b) = (2x + 3) * (3x - 2)) := 
by 
  sorry

end diff_of_squares_l297_297592


namespace sum_infinite_series_eq_l297_297628

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l297_297628


namespace non_powers_of_a_meet_condition_l297_297513

-- Definitions used directly from the conditions detailed in the problem:
def Sa (a x : ℕ) : ℕ := sorry -- S_{a}(x): sum of the digits of x in base a
def Fa (a x : ℕ) : ℕ := sorry -- F_{a}(x): number of digits of x in base a
def fa (a x : ℕ) : ℕ := sorry -- f_{a}(x): position of the first non-zero digit from the right in base a

theorem non_powers_of_a_meet_condition (a M : ℕ) (h₁: a > 1) (h₂ : M ≥ 2020) :
  ∀ n : ℕ, (n > 0) → (∀ k : ℕ, (k > 0) → (Sa a (k * n) = Sa a n ∧ Fa a (k * n) - fa a (k * n) > M)) ↔ (∃ α : ℕ, n = a ^ α) :=
sorry

end non_powers_of_a_meet_condition_l297_297513


namespace bamboo_tube_rice_capacity_l297_297931

theorem bamboo_tube_rice_capacity :
  ∃ (a d : ℝ), 3 * a + 3 * d * (1 + 2) = 4.5 ∧ 
               4 * (a + 5 * d) + 4 * d * (6 + 7 + 8) = 3.8 ∧ 
               (a + 3 * d) + (a + 4 * d) = 2.5 :=
by
  sorry

end bamboo_tube_rice_capacity_l297_297931


namespace sum_fraction_series_l297_297624

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l297_297624


namespace apples_not_sold_correct_l297_297412

-- Define the constants and conditions
def boxes_ordered_per_week : ℕ := 10
def apples_per_box : ℕ := 300
def fraction_sold : ℚ := 3 / 4

-- Define the total number of apples ordered in a week
def total_apples_ordered : ℕ := boxes_ordered_per_week * apples_per_box

-- Define the total number of apples sold in a week
def apples_sold : ℚ := fraction_sold * total_apples_ordered

-- Define the total number of apples not sold in a week
def apples_not_sold : ℚ := total_apples_ordered - apples_sold

-- Lean statement to prove the total number of apples not sold is 750
theorem apples_not_sold_correct :
  apples_not_sold = 750 := 
sorry

end apples_not_sold_correct_l297_297412


namespace sector_area_l297_297803

theorem sector_area (r : ℝ) (alpha : ℝ) (h : r = 2) (h2 : alpha = π / 3) : 
  1/2 * alpha * r^2 = (2 * π) / 3 := by
  sorry

end sector_area_l297_297803


namespace fraction_value_l297_297342

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 5) = 15 / 16 := sorry

end fraction_value_l297_297342


namespace inscribed_square_ratio_l297_297990

-- Define the problem context:
variables {x y : ℝ}

-- Conditions on the triangles and squares:
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0

def inscribed_square_first_triangle (a b c x : ℝ) : Prop :=
  is_right_triangle a b c ∧ a = 5 ∧ b = 12 ∧ c = 13 ∧
  x = 60 / 17

def inscribed_square_second_triangle (d e f y : ℝ) : Prop :=
  is_right_triangle d e f ∧ d = 6 ∧ e = 8 ∧ f = 10 ∧
  y = 25 / 8

-- Lean theorem to be proven with given conditions:
theorem inscribed_square_ratio :
  inscribed_square_first_triangle 5 12 13 x →
  inscribed_square_second_triangle 6 8 10 y →
  x / y = 96 / 85 := by
  sorry

end inscribed_square_ratio_l297_297990


namespace oranges_in_bin_l297_297806

theorem oranges_in_bin (initial_oranges thrown_out new_oranges : ℕ) (h1 : initial_oranges = 34) (h2 : thrown_out = 20) (h3 : new_oranges = 13) :
  (initial_oranges - thrown_out + new_oranges = 27) :=
by
  sorry

end oranges_in_bin_l297_297806


namespace problem1_l297_297051

theorem problem1 (α β : ℝ) 
  (tan_sum : Real.tan (α + β) = 2 / 5) 
  (tan_diff : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
sorry

end problem1_l297_297051


namespace slope_angle_of_tangent_line_expx_at_0_l297_297907

theorem slope_angle_of_tangent_line_expx_at_0 :
  let f := fun x : ℝ => Real.exp x 
  let f' := fun x : ℝ => Real.exp x
  ∀ x : ℝ, f' x = Real.exp x → 
  (∃ α : ℝ, Real.tan α = 1) →
  α = Real.pi / 4 :=
by
  intros f f' h_deriv h_slope
  sorry

end slope_angle_of_tangent_line_expx_at_0_l297_297907


namespace no_prime_satisfies_condition_l297_297830

theorem no_prime_satisfies_condition (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, (Real.sqrt (p + n) + Real.sqrt n) = k :=
by
  sorry

end no_prime_satisfies_condition_l297_297830


namespace max_marked_cells_no_shared_vertices_l297_297911

theorem max_marked_cells_no_shared_vertices (N : ℕ) (cube_side : ℕ) (total_cells : ℕ) (total_vertices : ℕ) :
  cube_side = 3 →
  total_cells = cube_side ^ 3 →
  total_vertices = 8 + 12 * 2 + 6 * 4 →
  ∀ (max_cells : ℕ), (4 * max_cells ≤ total_vertices) → (max_cells ≤ 14) :=
by
  sorry

end max_marked_cells_no_shared_vertices_l297_297911


namespace has_two_zeros_of_f_l297_297378

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.exp x - a

theorem has_two_zeros_of_f (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (-1 / Real.exp 2 < a ∧ a < 0) := by
sorry

end has_two_zeros_of_f_l297_297378


namespace percentage_increased_is_correct_l297_297173

-- Define the initial and final numbers
def initial_number : Nat := 150
def final_number : Nat := 210

-- Define the function to compute the percentage increase
def percentage_increase (initial final : Nat) : Float :=
  ((final - initial).toFloat / initial.toFloat) * 100.0

-- The theorem we need to prove
theorem percentage_increased_is_correct :
  percentage_increase initial_number final_number = 40 := 
by
  simp [percentage_increase, initial_number, final_number]
  sorry

end percentage_increased_is_correct_l297_297173


namespace polygon_vertices_product_at_least_2014_l297_297655

theorem polygon_vertices_product_at_least_2014 :
  ∀ (vertices : Fin 90 → ℕ), 
    (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 90) → 
    ∃ i, (vertices i) * (vertices ((i + 1) % 90)) ≥ 2014 :=
sorry

end polygon_vertices_product_at_least_2014_l297_297655


namespace find_a4_l297_297871

variable (a_1 d : ℝ)

def a_n (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

axiom condition1 : (a_n a_1 d 2 + a_n a_1 d 6) / 2 = 5 * Real.sqrt 3
axiom condition2 : (a_n a_1 d 3 + a_n a_1 d 7) / 2 = 7 * Real.sqrt 3

theorem find_a4 : a_n a_1 d 4 = 5 * Real.sqrt 3 :=
by
  -- Proof should go here, but we insert "sorry" to mark it as incomplete.
  sorry

end find_a4_l297_297871


namespace vasya_improved_example1_vasya_improved_example2_l297_297777

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l297_297777


namespace urn_contains_three_red_three_blue_after_five_operations_l297_297813

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5

noncomputable def calculate_probability (initial_red: ℕ) (initial_blue: ℕ) (operations: ℕ) : ℚ :=
  sorry

theorem urn_contains_three_red_three_blue_after_five_operations :
  calculate_probability initial_red_balls initial_blue_balls total_operations = 8 / 105 :=
by sorry

end urn_contains_three_red_three_blue_after_five_operations_l297_297813


namespace triangle_area_is_six_l297_297197

-- Conditions
def line_equation (Q : ℝ) : Prop :=
  ∀ (x y : ℝ), 12 * x - 4 * y + (Q - 305) = 0

def area_of_triangle (Q R : ℝ) : Prop :=
  R = (305 - Q) ^ 2 / 96

-- Question: Given a line equation forming a specific triangle, prove the area R equals 6.
theorem triangle_area_is_six (Q : ℝ) (h1 : Q = 281 ∨ Q = 329) :
  ∃ R : ℝ, line_equation Q → area_of_triangle Q R → R = 6 :=
by {
  sorry -- Proof to be provided
}

end triangle_area_is_six_l297_297197


namespace proof_by_contradiction_example_l297_297129

theorem proof_by_contradiction_example (a b c : ℝ) (h : a < 3 ∧ b < 3 ∧ c < 3) : a < 1 ∨ b < 1 ∨ c < 1 := 
by
  have h1 : a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 := sorry
  sorry

end proof_by_contradiction_example_l297_297129


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297308

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem exists_nine_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 8 ≤ 500) ∧ ∀ i ∈ (List.range 9), is_composite (a + i) :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 10 ≤ 500) ∧ ∀ i ∈ (List.range 11), is_composite (a + i) :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l297_297308


namespace towel_price_40_l297_297329

/-- Let x be the price of each towel bought second by the woman. 
    Given that she bought 3 towels at Rs. 100 each, 5 towels at x Rs. each, 
    and 2 towels at Rs. 550 each, and the average price of the towels was Rs. 160,
    we need to prove that x equals 40. -/
theorem towel_price_40 
    (x : ℝ)
    (h_avg_price : (300 + 5 * x + 1100) / 10 = 160) : 
    x = 40 :=
sorry

end towel_price_40_l297_297329


namespace arithmetic_sequence_problem_l297_297022

variable {a b : ℕ → ℕ}
variable (S T : ℕ → ℕ)

-- Conditions
def condition (n : ℕ) : Prop :=
  S n / T n = (2 * n + 1) / (3 * n + 2)

-- Conjecture to prove
theorem arithmetic_sequence_problem (h : ∀ n, condition S T n) :
  (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
by
  sorry

end arithmetic_sequence_problem_l297_297022
