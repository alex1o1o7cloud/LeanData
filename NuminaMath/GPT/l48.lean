import Mathlib

namespace jana_height_l48_48429

theorem jana_height (jess_height : ℕ) (kelly_height : ℕ) (jana_height : ℕ) 
  (h1 : kelly_height = jess_height - 3) 
  (h2 : jana_height = kelly_height + 5) 
  (h3 : jess_height = 72) : 
  jana_height = 74 := 
by
  sorry

end jana_height_l48_48429


namespace arithmetic_proof_l48_48230

def arithmetic_expression := 3889 + 12.952 - 47.95000000000027
def expected_result := 3854.002

theorem arithmetic_proof : arithmetic_expression = expected_result := by
  -- The proof goes here
  sorry

end arithmetic_proof_l48_48230


namespace circle_eq1_circle_eq2_l48_48407

-- Problem 1: Circle with center M(-5, 3) and passing through point A(-8, -1)
theorem circle_eq1 : ∀ (x y : ℝ), (x + 5) ^ 2 + (y - 3) ^ 2 = 25 :=
by
  sorry

-- Problem 2: Circle passing through three points A(-2, 4), B(-1, 3), C(2, 6)
theorem circle_eq2 : ∀ (x y : ℝ), x ^ 2 + (y - 5) ^ 2 = 5 :=
by
  sorry

end circle_eq1_circle_eq2_l48_48407


namespace buildings_collapsed_l48_48382

theorem buildings_collapsed (B : ℕ) (h₁ : 2 * B = X) (h₂ : 4 * B = Y) (h₃ : 8 * B = Z) (h₄ : B + 2 * B + 4 * B + 8 * B = 60) : B = 4 :=
by
  sorry

end buildings_collapsed_l48_48382


namespace family_ages_l48_48589

theorem family_ages :
  ∃ (x j b m F M : ℕ), 
    (b = j - x) ∧
    (m = j - 2 * x) ∧
    (j * b = F) ∧
    (b * m = M) ∧
    (j + b + m + F + M = 90) ∧
    (F = M + x ∨ F = M - x) ∧
    (j = 6) ∧ 
    (b = 6) ∧ 
    (m = 6) ∧ 
    (F = 36) ∧ 
    (M = 36) :=
sorry

end family_ages_l48_48589


namespace dvd_cd_ratio_l48_48559

theorem dvd_cd_ratio (total_sales : ℕ) (dvd_sales : ℕ) (cd_sales : ℕ) (h1 : total_sales = 273) (h2 : dvd_sales = 168) (h3 : cd_sales = total_sales - dvd_sales) : (dvd_sales / Nat.gcd dvd_sales cd_sales) = 8 ∧ (cd_sales / Nat.gcd dvd_sales cd_sales) = 5 :=
by
  sorry

end dvd_cd_ratio_l48_48559


namespace solve_for_x_l48_48951

theorem solve_for_x 
    (x : ℝ) 
    (h : (4 * x - 2) / (5 * x - 5) = 3 / 4) 
    : x = -7 :=
sorry

end solve_for_x_l48_48951


namespace kit_time_to_ticket_window_l48_48931

theorem kit_time_to_ticket_window 
  (rate : ℝ)
  (remaining_distance : ℝ)
  (yard_to_feet_conv : ℝ)
  (new_rate : rate = 90 / 30)
  (remaining_distance_in_feet : remaining_distance = 100 * yard_to_feet_conv)
  (yard_to_feet_conv_val : yard_to_feet_conv = 3) :
  (remaining_distance / rate = 100) := 
by 
  simp [new_rate, remaining_distance_in_feet, yard_to_feet_conv_val]
  sorry

end kit_time_to_ticket_window_l48_48931


namespace cos_difference_identity_l48_48131

theorem cos_difference_identity (α : ℝ)
  (h : Real.sin (α + π / 6) + Real.cos α = - (Real.sqrt 3) / 3) :
  Real.cos (π / 6 - α) = -1 / 3 := 
sorry

end cos_difference_identity_l48_48131


namespace sin_150_eq_half_l48_48716

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48716


namespace part1_correct_part2_correct_l48_48232

noncomputable def part1 : ℝ :=
  let total_ways := Nat.choose 16 3
  let ways_A0 := Nat.choose 12 3
  let ways_A1 := Nat.choose 4 1 * Nat.choose 12 2
  (ways_A0 + ways_A1) / total_ways

theorem part1_correct : part1 = 121 / 140 :=
sorry

noncomputable def binomial_distribution (n : ℕ) (p : ℝ) : ℕ → ℝ
| k => Nat.choose n k * (p^k) * ((1-p)^(n-k))

theorem part2_correct :
  ∀ k ∈ ({0, 1, 2, 3} : Finset ℕ), binomial_distribution 3 (1/4) k =
  match k with
  | 0 => 27 / 64
  | 1 => 27 / 64
  | 2 => 9 / 64
  | 3 => 1 / 64
  | _ => 0 :=
sorry


end part1_correct_part2_correct_l48_48232


namespace determine_b_l48_48396

theorem determine_b (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x) ∧ (∀ x y : ℝ, y - 2 = (b + 9) * x) → 
  b = -6 :=
by
  sorry

end determine_b_l48_48396


namespace correct_answer_is_C_l48_48105

def exactly_hits_n_times (n k : ℕ) : Prop :=
  n = k

def hits_no_more_than (n k : ℕ) : Prop :=
  n ≤ k

def hits_at_least (n k : ℕ) : Prop :=
  n ≥ k

def is_mutually_exclusive (P Q : Prop) : Prop :=
  ¬ (P ∧ Q)

def is_non_opposing (P Q : Prop) : Prop :=
  ¬ P ∧ ¬ Q

def events_are_mutually_exclusive_and_non_opposing (n : ℕ) : Prop :=
  let event1 := exactly_hits_n_times 5 3
  let event2 := exactly_hits_n_times 5 4
  is_mutually_exclusive event1 event2 ∧ is_non_opposing event1 event2

theorem correct_answer_is_C : events_are_mutually_exclusive_and_non_opposing 5 :=
by
  sorry

end correct_answer_is_C_l48_48105


namespace mean_weight_is_70_357_l48_48203

def weights_50 : List ℕ := [57]
def weights_60 : List ℕ := [60, 64, 64, 66, 69]
def weights_70 : List ℕ := [71, 73, 73, 75, 77, 78, 79, 79]

def weights := weights_50 ++ weights_60 ++ weights_70

def total_weight : ℕ := List.sum weights
def total_players : ℕ := List.length weights
def mean_weight : ℚ := (total_weight : ℚ) / total_players

theorem mean_weight_is_70_357 :
  mean_weight = 70.357 := 
sorry

end mean_weight_is_70_357_l48_48203


namespace number_of_dogs_l48_48196

-- Define variables for the number of cats (C) and dogs (D)
variables (C D : ℕ)

-- Define the conditions from the problem statement
def condition1 : Prop := C = D - 6
def condition2 : Prop := C * 3 = D * 2

-- State the theorem that D should be 18 given the conditions
theorem number_of_dogs (h1 : condition1 C D) (h2 : condition2 C D) : D = 18 :=
  sorry

end number_of_dogs_l48_48196


namespace minimum_at_2_l48_48069

open Real

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem minimum_at_2 : 
  (∀ x : ℝ, ((x ≠ 2) → (f(2) ≤ f(x)))) :=
by
  sorry

end minimum_at_2_l48_48069


namespace complex_division_l48_48456

def i : ℂ := Complex.I

theorem complex_division :
  (i^3 / (1 + i)) = -1/2 - 1/2 * i := 
by sorry

end complex_division_l48_48456


namespace sin_150_eq_half_l48_48714

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48714


namespace positive_solution_y_l48_48012

theorem positive_solution_y (x y z : ℝ) 
  (h1 : x * y = 8 - 3 * x - 2 * y) 
  (h2 : y * z = 15 - 5 * y - 3 * z) 
  (h3 : x * z = 40 - 5 * x - 4 * z) : 
  y = 4 := 
sorry

end positive_solution_y_l48_48012


namespace village_population_equal_in_years_l48_48830

theorem village_population_equal_in_years :
  ∀ (n : ℕ), (70000 - 1200 * n = 42000 + 800 * n) ↔ n = 14 :=
by {
  sorry
}

end village_population_equal_in_years_l48_48830


namespace father_age_l48_48876

theorem father_age (F D : ℕ) (h1 : F = 4 * D) (h2 : (F + 5) + (D + 5) = 50) : F = 32 :=
by
  sorry

end father_age_l48_48876


namespace find_x_l48_48455

def set_of_numbers := [1, 2, 4, 5, 6, 9, 9, 10]

theorem find_x {x : ℝ} (h : (set_of_numbers.sum + x) / 9 = 7) : x = 17 :=
by
  sorry

end find_x_l48_48455


namespace train_ride_length_l48_48861

theorem train_ride_length :
  let reading_time := 2
  let eating_time := 1
  let watching_time := 3
  let napping_time := 3
  reading_time + eating_time + watching_time + napping_time = 9 := 
by
  sorry

end train_ride_length_l48_48861


namespace golden_triangle_ratio_l48_48969

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_ratio :
  let t := golden_ratio in
  (1 - 2 * Real.sin (27 * Real.pi / 180) ^ 2) / (2 * t * Real.sqrt (4 - t ^ 2)) = 1 / 4 := 
by
  let t := golden_ratio
  sorry

end golden_triangle_ratio_l48_48969


namespace max_f_value_l48_48732

noncomputable def S (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℝ :=
  n / (n + 32) / (n + 2)

theorem max_f_value : ∀ n : ℕ, f n ≤ (1 / 50) :=
sorry

end max_f_value_l48_48732


namespace ratio_of_shares_l48_48990

theorem ratio_of_shares (A B C : ℝ) (x : ℝ):
  A = 240 → 
  A + B + C = 600 →
  A = x * (B + C) →
  B = (2/3) * (A + C) →
  A / (B + C) = 2 / 3 :=
by
  intros hA hTotal hFraction hB
  sorry

end ratio_of_shares_l48_48990


namespace Hugo_win_probability_l48_48293

noncomputable def probability_Hugo_wins_with_6 :
  ℕ × ℕ × ℕ × ℕ × ℕ → ℝ := sorry

theorem Hugo_win_probability (players_rolls : Fin 5 → Fin 6) (Hugo_win : Bool) :
  Hugo_win = true → probability_Hugo_wins_with_6 (players_rolls 0, players_rolls 1, players_rolls 2, players_rolls 3, players_rolls 4) = 4375 / 7776 :=
by
  sorry

end Hugo_win_probability_l48_48293


namespace ball_bounce_height_l48_48636

theorem ball_bounce_height :
  ∃ b : ℕ, ∀ n < b, (320 * (3 / 4 : ℝ) ^ n) ≥ 40 ∧ (320 * (3 / 4 : ℝ) ^ b) < 40 :=
begin
  sorry
end

end ball_bounce_height_l48_48636


namespace average_speed_is_correct_l48_48641

namespace CyclistTrip

-- Define the trip parameters
def distance_north := 10 -- kilometers
def speed_north := 15 -- kilometers per hour
def rest_time := 10 / 60 -- hours
def distance_south := 10 -- kilometers
def speed_south := 20 -- kilometers per hour

-- The total trip distance
def total_distance := distance_north + distance_south -- kilometers

-- Calculate the time for each segment
def time_north := distance_north / speed_north -- hours
def time_south := distance_south / speed_south -- hours

-- Total time for the trip
def total_time := time_north + rest_time + time_south -- hours

-- Calculate the average speed
def average_speed := total_distance / total_time -- kilometers per hour

theorem average_speed_is_correct : average_speed = 15 := by
  sorry

end CyclistTrip

end average_speed_is_correct_l48_48641


namespace total_cost_of_fruit_l48_48626

theorem total_cost_of_fruit (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 58) 
  (h2 : 3 * x + 2 * y = 72) : 
  3 * x + 3 * y = 78 := 
by
  sorry

end total_cost_of_fruit_l48_48626


namespace fixed_point_of_line_l48_48340

theorem fixed_point_of_line (k : ℝ) : ∃ (p : ℝ × ℝ), p = (-3, 4) ∧ ∀ (x y : ℝ), (y - 4 = -k * (x + 3)) → (-3, 4) = (x, y) :=
by
  sorry

end fixed_point_of_line_l48_48340


namespace mom_has_enough_money_l48_48576

def original_price : ℝ := 268
def discount_rate : ℝ := 0.2
def money_brought : ℝ := 230
def discounted_price := original_price * (1 - discount_rate)

theorem mom_has_enough_money : money_brought ≥ discounted_price := by
  sorry

end mom_has_enough_money_l48_48576


namespace xy_inequality_l48_48787

theorem xy_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 2) : 
  x^2 * y^2 * (x^2 + y^2) ≤ 2 := 
sorry

end xy_inequality_l48_48787


namespace shortest_chord_intercept_l48_48146

theorem shortest_chord_intercept (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 3 → x + m * y - m - 1 = 0 → m = 1) :=
sorry

end shortest_chord_intercept_l48_48146


namespace origin_eq_smallest_abs_value_rat_l48_48464

theorem origin_eq_smallest_abs_value_rat :
  (0 : ℚ) = (0 : ℚ) :=
by 
  sorry

end origin_eq_smallest_abs_value_rat_l48_48464


namespace probability_first_spade_last_ace_l48_48206

-- Define the problem parameters
def standard_deck : ℕ := 52
def spades_count : ℕ := 13
def aces_count : ℕ := 4
def ace_of_spades : ℕ := 1

-- Probability of drawing a spade but not an ace as the first card
def prob_spade_not_ace_first : ℚ := 12 / 52

-- Probability of drawing any of the four aces among the two remaining cards
def prob_ace_among_two_remaining : ℚ := 4 / 50

-- Probability of drawing the ace of spades as the first card
def prob_ace_of_spades_first : ℚ := 1 / 52

-- Probability of drawing one of three remaining aces among two remaining cards
def prob_three_aces_among_two_remaining : ℚ := 3 / 50

-- Combined probability according to the cases
def final_probability : ℚ := (prob_spade_not_ace_first * prob_ace_among_two_remaining) + (prob_ace_of_spades_first * prob_three_aces_among_two_remaining)

-- The theorem stating that the computed probability matches the expected result
theorem probability_first_spade_last_ace : final_probability = 51 / 2600 := 
  by
    -- inserting proof steps here would solve the theorem
    sorry

end probability_first_spade_last_ace_l48_48206


namespace remainder_calculation_l48_48625

theorem remainder_calculation 
  (x : ℤ) (y : ℝ)
  (hx : 0 < x)
  (hy : y = 70.00000000000398)
  (hx_div_y : (x : ℝ) / y = 86.1) :
  x % y = 7 :=
by
  sorry

end remainder_calculation_l48_48625


namespace probability_of_at_least_one_die_shows_2_is_correct_l48_48472

-- Definitions for the conditions
def total_outcomes : ℕ := 64
def neither_die_shows_2_outcomes : ℕ := 49
def favorability (total : ℕ) (exclusion : ℕ) : ℕ := total - exclusion
def favorable_outcomes : ℕ := favorability total_outcomes neither_die_shows_2_outcomes
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Mathematically equivalent proof problem statement
theorem probability_of_at_least_one_die_shows_2_is_correct : 
  probability favorable_outcomes total_outcomes = 15 / 64 :=
sorry

end probability_of_at_least_one_die_shows_2_is_correct_l48_48472


namespace water_pump_calculation_l48_48418

-- Define the given initial conditions
variables (f h j g k l m : ℕ)

-- Provide the correctly calculated answer
theorem water_pump_calculation (hf : f > 0) (hg : g > 0) (hk : k > 0) (hm : m > 0) : 
  (k * l * m * j * h) / (10000 * f * g) = (k * (j * h / (f * g)) * l * m) / 10000 := 
sorry

end water_pump_calculation_l48_48418


namespace added_number_after_doubling_l48_48847

theorem added_number_after_doubling (x y : ℤ) (h1 : x = 4) (h2 : 3 * (2 * x + y) = 51) : y = 9 :=
by
  -- proof goes here
  sorry

end added_number_after_doubling_l48_48847


namespace arithmetic_sequence_a2015_l48_48765

theorem arithmetic_sequence_a2015 :
  (∀ n : ℕ, n > 0 → (∃ a_n a_n1 : ℝ,
    a_n1 = a_n + 2 ∧ a_n + a_n1 = 4 * n - 58))
  → (∃ a_2015 : ℝ, a_2015 = 4000) :=
by
  intro h
  sorry

end arithmetic_sequence_a2015_l48_48765


namespace musical_chairs_l48_48354

def is_prime_power (m : ℕ) : Prop :=
  ∃ (p k : ℕ), Nat.Prime p ∧ k > 0 ∧ m = p ^ k

theorem musical_chairs (n m : ℕ) (h1 : 1 < m) (h2 : m ≤ n) (h3 : ¬ is_prime_power m) :
  ∃ f : Fin n → Fin n, (∀ x, f x ≠ x) ∧ (∀ x, (f^[m]) x = x) :=
sorry

end musical_chairs_l48_48354


namespace MNPQ_is_rectangle_l48_48296

variable {Point : Type}
variable {A B C D M N P Q : Point}

def is_parallelogram (A B C D : Point) : Prop := sorry
def altitude (X Y : Point) : Prop := sorry
def rectangle (M N P Q : Point) : Prop := sorry

theorem MNPQ_is_rectangle 
  (h_parallelogram : is_parallelogram A B C D)
  (h_alt1 : altitude B M)
  (h_alt2 : altitude B N)
  (h_alt3 : altitude D P)
  (h_alt4 : altitude D Q) :
  rectangle M N P Q :=
sorry

end MNPQ_is_rectangle_l48_48296


namespace ratio_of_games_played_to_losses_l48_48843

-- Conditions
def games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := games_played - games_won

-- Prove the ratio of games played to games lost is 2:1
theorem ratio_of_games_played_to_losses
  (h_played : games_played = 10)
  (h_won : games_won = 5) :
  (games_played / Nat.gcd games_played games_lost : ℕ) /
  (games_lost / Nat.gcd games_played games_lost : ℕ) = 2 / 1 :=
by
  sorry

end ratio_of_games_played_to_losses_l48_48843


namespace brad_money_l48_48287

noncomputable def money_problem : Prop :=
  ∃ (B J D : ℝ), 
    J = 2 * B ∧
    J = (3/4) * D ∧
    B + J + D = 68 ∧
    B = 12

theorem brad_money : money_problem :=
by {
  -- Insert proof steps here if necessary
  sorry
}

end brad_money_l48_48287


namespace solve_equation_l48_48058

theorem solve_equation (x : ℝ) : (x - 1) * (x + 3) = 5 ↔ x = 2 ∨ x = -4 := by
  sorry

end solve_equation_l48_48058


namespace probability_leftmost_blue_off_rightmost_red_on_l48_48792

noncomputable def calculate_probability : ℚ :=
  let total_arrangements := Nat.choose 8 4
  let total_on_choices := Nat.choose 8 4
  let favorable_arrangements := Nat.choose 6 3 * Nat.choose 7 3
  favorable_arrangements / (total_arrangements * total_on_choices)

theorem probability_leftmost_blue_off_rightmost_red_on :
  calculate_probability = 1 / 7 := 
by
  sorry

end probability_leftmost_blue_off_rightmost_red_on_l48_48792


namespace probability_of_same_color_l48_48613

noncomputable theory
open ProbabilityTheory

-- Define the setup of the problem
def bagA : Finset (String × ℕ) := {("white", 1), ("red", 2), ("black", 3)}
def bagB : Finset (String × ℕ) := {("white", 2), ("red", 3), ("black", 1)}

-- Define the event of drawing a ball of each color
def event (c : String) : Finset (String × String) :=
  ({c} ×ˢ Finset.image (λ (b : String × ℕ), b.1) bagA) ∩ ({c} ×ˢ Finset.image (λ (b : String × ℕ), b.1) bagB)

-- Define the possible outcomes
def possible_outcomes : Finset (String × String) :=
  Finset.product (Finset.image (λ (b : String × ℕ), b.1) bagA) (Finset.image (λ (b : String × ℕ), b.1) bagB)

-- Define the event of same color outcome
def same_color_event : Finset (String × String) :=
  event "white" ∪ event "red" ∪ event "black"

-- Calculate the probability of the same color event
def prob_same_color : ℚ :=
  (same_color_event.card : ℚ) / (possible_outcomes.card : ℚ)

-- Proof statement
theorem probability_of_same_color : prob_same_color = 11 / 36 := by
  sorry

end probability_of_same_color_l48_48613


namespace area_of_right_triangle_l48_48781

theorem area_of_right_triangle
  (X Y Z: ℝ × ℝ)
  (right_angle_at_Z: Z = (0, 0))
  (hypotenuse_length: (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 2500)
  (median_through_X: Yrs X = X.2 - X.1 + 5 = 0)
  (median_through_Y: Yrs Y = Y.2 - 3 * Y.1 + 6 = 0)
: area_triangle XYZ = 3750 / 17 := by
  sorry

end area_of_right_triangle_l48_48781


namespace sum_of_coordinates_l48_48822

theorem sum_of_coordinates :
  let in_distance_from_line (p : (ℝ × ℝ)) (d : ℝ) (line_y : ℝ) : Prop := abs (p.2 - line_y) = d
  let in_distance_from_point (p1 p2 : (ℝ × ℝ)) (d : ℝ) : Prop := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = d^2
  ∃ (P1 P2 P3 P4 : ℝ × ℝ),
  in_distance_from_line P1 4 13 ∧ in_distance_from_point P1 (7, 13) 10 ∧
  in_distance_from_line P2 4 13 ∧ in_distance_from_point P2 (7, 13) 10 ∧
  in_distance_from_line P3 4 13 ∧ in_distance_from_point P3 (7, 13) 10 ∧
  in_distance_from_line P4 4 13 ∧ in_distance_from_point P4 (7, 13) 10 ∧
  (P1.1 + P2.1 + P3.1 + P4.1) + (P1.2 + P2.2 + P3.2 + P4.2) = 80 :=
sorry

end sum_of_coordinates_l48_48822


namespace students_at_start_of_year_l48_48652

-- Define the initial number of students as a variable S
variables (S : ℕ)

-- Define the conditions
def condition_1 := S - 18 + 14 = 29

-- State the theorem to be proved
theorem students_at_start_of_year (h : condition_1 S) : S = 33 :=
sorry

end students_at_start_of_year_l48_48652


namespace part_a_possible_final_number_l48_48488

theorem part_a_possible_final_number :
  ∃ (n : ℕ), n = 97 ∧ 
  (∃ f : {x // x ≠ 0} → ℕ → ℕ, 
    f ⟨1, by decide⟩ 0 = 1 ∧ 
    f ⟨2, by decide⟩ 1 = 2 ∧ 
    f ⟨4, by decide⟩ 2 = 4 ∧ 
    f ⟨8, by decide⟩ 3 = 8 ∧ 
    f ⟨16, by decide⟩ 4 = 16 ∧ 
    f ⟨32, by decide⟩ 5 = 32 ∧ 
    f ⟨64, by decide⟩ 6 = 64 ∧ 
    f ⟨128, by decide⟩ 7 = 128 ∧ 
    ∀ i j : {x // x ≠ 0}, f i j = (f i j - f i j)) := sorry

end part_a_possible_final_number_l48_48488


namespace hyperbola_center_l48_48121

theorem hyperbola_center (x y : ℝ) :
  ( ∃ (h k : ℝ), ∀ (x y : ℝ), (4 * x - 8)^2 / 9^2 - (5 * y - 15)^2 / 7^2 = 1 → (h, k) = (2, 3) ) :=
by
  existsi 2
  existsi 3
  intros x y h
  sorry

end hyperbola_center_l48_48121


namespace symmetric_circle_eq_a_l48_48603

theorem symmetric_circle_eq_a :
  ∀ (a : ℝ), (∀ x y : ℝ, (x^2 + y^2 - a * x + 2 * y + 1 = 0) ↔ (∃ x y : ℝ, (x - y = 1) ∧ ( x^2 + y^2 = 1))) → a = 2 :=
by
  sorry

end symmetric_circle_eq_a_l48_48603


namespace innovation_contribution_l48_48485

variable (material : String)
variable (contribution : String → Prop)
variable (A B C D : Prop)

-- Conditions
axiom condA : contribution material → A
axiom condB : contribution material → ¬B
axiom condC : contribution material → ¬C
axiom condD : contribution material → ¬D

-- The problem statement
theorem innovation_contribution :
  contribution material → A :=
by
  -- dummy proof as placeholder
  sorry

end innovation_contribution_l48_48485


namespace abs_five_minus_e_l48_48124

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_e : |5 - e| = 2.282 := 
by 
    -- Proof is omitted 
    sorry

end abs_five_minus_e_l48_48124


namespace union_M_N_l48_48907

def M : Set ℕ := {1, 2}
def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end union_M_N_l48_48907


namespace temperature_at_noon_l48_48153

-- Definitions of the given conditions.
def morning_temperature : ℝ := 4
def temperature_drop : ℝ := 10

-- The theorem statement that needs to be proven.
theorem temperature_at_noon : morning_temperature - temperature_drop = -6 :=
by
  -- The proof can be filled in by solving the stated theorem.
  sorry

end temperature_at_noon_l48_48153


namespace geometric_sequence_sum_correct_l48_48266

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 2 then 2^(n + 1) - 2
else 64 * (1 - (1 / 2)^n)

theorem geometric_sequence_sum_correct (a1 q : ℝ) (n : ℕ) 
  (h1 : q > 0) 
  (h2 : a1 + a1 * q^4 = 34) 
  (h3 : a1^2 * q^4 = 64) :
  geometric_sequence_sum a1 q n = 
  if q = 2 then 2^(n + 1) - 2 else 64 * (1 - (1 / 2)^n) :=
sorry

end geometric_sequence_sum_correct_l48_48266


namespace sin_150_eq_one_half_l48_48688

theorem sin_150_eq_one_half :
  let θ := 150
  let θ_ref := 30
  let θ_complement := 180 - θ_ref
  θ = θ_complement →
  (∀ θ, θ_ref = 30 * 60.toReal)
    ∧ (∀ θ, (cos θ_ref, sin θ_ref) = (Real.sqrt 3 / 2, 1 / 2)) →
  sin θ = 1 / 2 :=
by
  intros θ θ_ref θ_complement h1 h2
  have h3 : (cos θ_complement, sin θ_complement) = (-(cos θ_ref), sin θ_ref) := by
    sorry
  rw h3
  sorry

end sin_150_eq_one_half_l48_48688


namespace cooper_savings_l48_48722

theorem cooper_savings :
  let daily_savings := 34
  let days_in_year := 365
  daily_savings * days_in_year = 12410 :=
by
  sorry

end cooper_savings_l48_48722


namespace A_min_votes_for_victory_l48_48952

theorem A_min_votes_for_victory:
  ∀ (initial_votes_A initial_votes_B initial_votes_C total_votes remaining_votes min_votes_A: ℕ),
  initial_votes_A = 350 →
  initial_votes_B = 370 →
  initial_votes_C = 280 →
  total_votes = 1500 →
  remaining_votes = 500 →
  min_votes_A = 261 →
  initial_votes_A + min_votes_A > initial_votes_B + (remaining_votes - min_votes_A) :=
by
  intros _ _ _ _ _ _
  sorry

end A_min_votes_for_victory_l48_48952


namespace increasing_interval_l48_48195

def my_function (x : ℝ) : ℝ := -(x - 3) * |x|

theorem increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → my_function x ≤ my_function y :=
by
  sorry

end increasing_interval_l48_48195


namespace calculate_otimes_l48_48655

def otimes (a b : ℚ) : ℚ := (a + b) / (a - b)

theorem calculate_otimes :
  otimes (otimes 8 6) 12 = -19 / 5 := by
  sorry

end calculate_otimes_l48_48655


namespace total_groups_correct_l48_48183

-- Definitions from conditions
def eggs := 57
def egg_group_size := 7

def bananas := 120
def banana_group_size := 10

def marbles := 248
def marble_group_size := 8

-- Calculate the number of groups for each type of object
def egg_groups := eggs / egg_group_size
def banana_groups := bananas / banana_group_size
def marble_groups := marbles / marble_group_size

-- Total number of groups
def total_groups := egg_groups + banana_groups + marble_groups

-- Proof statement
theorem total_groups_correct : total_groups = 51 := by
  sorry

end total_groups_correct_l48_48183


namespace largest_integer_x_cubed_lt_three_x_squared_l48_48977

theorem largest_integer_x_cubed_lt_three_x_squared : 
  ∃ x : ℤ, x^3 < 3 * x^2 ∧ (∀ y : ℤ, y^3 < 3 * y^2 → y ≤ x) :=
  sorry

end largest_integer_x_cubed_lt_three_x_squared_l48_48977


namespace pear_juice_processed_l48_48063

theorem pear_juice_processed
  (total_pears : ℝ)
  (export_percentage : ℝ)
  (juice_percentage_of_remainder : ℝ) :
  total_pears = 8.5 →
  export_percentage = 0.30 →
  juice_percentage_of_remainder = 0.60 →
  ((total_pears * (1 - export_percentage)) * juice_percentage_of_remainder) = 3.6 :=
by
  intros
  sorry

end pear_juice_processed_l48_48063


namespace opposite_of_negative_fraction_l48_48813

theorem opposite_of_negative_fraction : -(- (1/2023 : ℚ)) = 1/2023 := 
sorry

end opposite_of_negative_fraction_l48_48813


namespace intersection_point_proof_l48_48389

def intersect_point : Prop := 
  ∃ x y : ℚ, (5 * x - 6 * y = 3) ∧ (8 * x + 2 * y = 22) ∧ x = 69 / 29 ∧ y = 43 / 29

theorem intersection_point_proof : intersect_point :=
  sorry

end intersection_point_proof_l48_48389


namespace total_profit_for_the_month_l48_48997

theorem total_profit_for_the_month (mean_profit_month : ℕ) (num_days_month : ℕ)
(mean_profit_first15 : ℕ) (num_days_first15 : ℕ) 
(mean_profit_last15 : ℕ) (num_days_last15 : ℕ) 
(h1 : mean_profit_month = 350) (h2 : num_days_month = 30) 
(h3 : mean_profit_first15 = 285) (h4 : num_days_first15 = 15) 
(h5 : mean_profit_last15 = 415) (h6 : num_days_last15 = 15) : 
(mean_profit_first15 * num_days_first15 + mean_profit_last15 * num_days_last15) = 10500 := by
  sorry

end total_profit_for_the_month_l48_48997


namespace each_friend_should_contribute_equally_l48_48726

-- Define the total expenses and number of friends
def total_expenses : ℝ := 35 + 9 + 9 + 6 + 2
def number_of_friends : ℕ := 5

-- Define the expected contribution per friend
def expected_contribution : ℝ := 12.20

-- Theorem statement
theorem each_friend_should_contribute_equally :
  total_expenses / number_of_friends = expected_contribution :=
by
  sorry

end each_friend_should_contribute_equally_l48_48726


namespace find_raspberries_l48_48165

def total_berries (R : ℕ) : ℕ := 30 + 20 + R

def fresh_berries (R : ℕ) : ℕ := 2 * total_berries R / 3

def fresh_berries_to_keep (R : ℕ) : ℕ := fresh_berries R / 2

def fresh_berries_to_sell (R : ℕ) : ℕ := fresh_berries R - fresh_berries_to_keep R

theorem find_raspberries (R : ℕ) : fresh_berries_to_sell R = 20 → R = 10 := 
by 
sorry

-- To ensure the problem is complete and solvable, we also need assumptions on the domain:
example : ∃ R : ℕ, fresh_berries_to_sell R = 20 := 
by 
  use 10 
  sorry

end find_raspberries_l48_48165


namespace quadratic_roots_l48_48454

theorem quadratic_roots : ∀ (x : ℝ), x^2 + 5 * x - 4 = 0 ↔ x = (-5 + Real.sqrt 41) / 2 ∨ x = (-5 - Real.sqrt 41) / 2 := 
by
  sorry

end quadratic_roots_l48_48454


namespace proposition_C_l48_48523

-- Given conditions
variables {a b : ℝ}

-- Proposition C is the correct one
theorem proposition_C (h : a^3 > b^3) : a > b := by
  sorry

end proposition_C_l48_48523


namespace problem_statement_l48_48140

theorem problem_statement (a : ℝ) :
  (∀ x : ℝ, (1/2 < x ∧ x < 2 → ax^2 + 5 * x - 2 > 0)) →
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) → ax^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end problem_statement_l48_48140


namespace flight_distance_each_way_l48_48849

variables (D : ℝ) (T_out T_return total_time : ℝ)

-- Defining conditions
def condition1 : Prop := T_out = D / 300
def condition2 : Prop := T_return = D / 500
def condition3 : Prop := total_time = 8

-- Given conditions
axiom h1 : condition1 D T_out
axiom h2 : condition2 D T_return
axiom h3 : condition3 total_time

-- The proof problem statement
theorem flight_distance_each_way : T_out + T_return = total_time → D = 1500 :=
by
  sorry

end flight_distance_each_way_l48_48849


namespace sum_of_mixed_numbers_is_between_18_and_19_l48_48076

theorem sum_of_mixed_numbers_is_between_18_and_19 :
  let a := 2 + 3 / 8;
  let b := 4 + 1 / 3;
  let c := 5 + 2 / 21;
  let d := 6 + 1 / 11;
  18 < a + b + c + d ∧ a + b + c + d < 19 :=
by
  sorry

end sum_of_mixed_numbers_is_between_18_and_19_l48_48076


namespace scientific_notation_of_00000065_l48_48481

theorem scientific_notation_of_00000065:
  (6.5 * 10^(-7)) = 0.00000065 :=
by
  -- Proof goes here
  sorry

end scientific_notation_of_00000065_l48_48481


namespace no_bijective_function_l48_48036

open Set

def is_bijective {α β : Type*} (f : α → β) : Prop :=
  Function.Bijective f

def are_collinear {P : Type*} (A B C : P) : Prop :=
  sorry -- placeholder for the collinearity predicate on points

def are_parallel_or_concurrent {L : Type*} (l₁ l₂ l₃ : L) : Prop :=
  sorry -- placeholder for the condition that lines are parallel or concurrent

theorem no_bijective_function (P : Type*) (D : Type*) :
  ¬ ∃ (f : P → D), is_bijective f ∧
    ∀ A B C : P, are_collinear A B C → are_parallel_or_concurrent (f A) (f B) (f C) :=
by
  sorry

end no_bijective_function_l48_48036


namespace max_value_of_f_l48_48805

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_of_f : ∃ M : ℝ, M = 1 / 3 ∧ ∀ x : ℝ, f x ≤ M :=
by
  sorry

end max_value_of_f_l48_48805


namespace sufficient_but_not_necessary_condition_l48_48610

theorem sufficient_but_not_necessary_condition : ∀ (y : ℝ), (y = 2 → y^2 = 4) ∧ (y^2 = 4 → (y = 2 ∨ y = -2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l48_48610


namespace tips_fraction_of_income_l48_48362

theorem tips_fraction_of_income
  (S T : ℝ)
  (h1 : T = (2 / 4) * S) :
  T / (S + T) = 1 / 3 :=
by
  -- Proof goes here
  sorry

end tips_fraction_of_income_l48_48362


namespace sin_150_eq_half_l48_48662

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by sorry

end sin_150_eq_half_l48_48662


namespace age_difference_l48_48629

variable (A B C : ℕ)

-- Conditions
def ages_total_condition (a b c : ℕ) : Prop :=
  a + b = b + c + 11

-- Proof problem statement
theorem age_difference (a b c : ℕ) (h : ages_total_condition a b c) : a - c = 11 :=
by
  sorry

end age_difference_l48_48629


namespace factorial_fraction_is_integer_l48_48039

theorem factorial_fraction_is_integer (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ∃ k : ℤ, (a.factorial * b.factorial * (a + b).factorial) / (2 * a).factorial * (2 * b).factorial = k := 
sorry

end factorial_fraction_is_integer_l48_48039


namespace golden_triangle_expression_l48_48971

noncomputable def t : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_expression :
  t = (Real.sqrt 5 - 1) / 2 →
  (1 - 2 * (Real.sin (27 * Real.pi / 180))^2) / (2 * t * Real.sqrt (4 - t^2)) = 1 / 4 :=
by
  intro h_t
  have h1 : t = (Real.sqrt 5 - 1) / 2 := h_t
  sorry

end golden_triangle_expression_l48_48971


namespace lewis_earnings_during_harvest_l48_48936

-- Define the conditions
def regular_earnings_per_week : ℕ := 28
def overtime_earnings_per_week : ℕ := 939
def number_of_weeks : ℕ := 1091

-- Define the total earnings per week
def total_earnings_per_week := regular_earnings_per_week + overtime_earnings_per_week

-- Define the total earnings during the harvest season
def total_earnings_during_harvest := total_earnings_per_week * number_of_weeks

-- Theorem statement
theorem lewis_earnings_during_harvest : total_earnings_during_harvest = 1055497 := by
  sorry

end lewis_earnings_during_harvest_l48_48936


namespace sin_150_eq_half_l48_48679

noncomputable def calculate_sin_150 : ℝ := Real.sin (150 * Real.pi / 180)

theorem sin_150_eq_half : calculate_sin_150 = 1 / 2 :=
by
  -- We would include the detailed steps if not skipping the proof as per the instruction.
  sorry

end sin_150_eq_half_l48_48679


namespace sum_two_and_four_l48_48359

theorem sum_two_and_four : 2 + 4 = 6 := by
  sorry

end sum_two_and_four_l48_48359


namespace product_ABC_sol_l48_48584

theorem product_ABC_sol (A B C : ℚ) : 
  (∀ x : ℚ, x^2 - 20 = A * (x + 2) * (x - 3) + B * (x - 2) * (x - 3) + C * (x - 2) * (x + 2)) → 
  A * B * C = 2816 / 35 := 
by 
  intro h
  sorry

end product_ABC_sol_l48_48584


namespace smallest_n_for_sqrt_18n_integer_l48_48134

theorem smallest_n_for_sqrt_18n_integer :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (∃ k : ℕ, k^2 = 18 * m) → n <= m) ∧ (∃ k : ℕ, k^2 = 18 * n) :=
sorry

end smallest_n_for_sqrt_18n_integer_l48_48134


namespace T_perimeter_is_20_l48_48607

-- Define the perimeter of a rectangle given its length and width
def perimeter_rectangle (length width : ℝ) : ℝ :=
  2 * length + 2 * width

-- Given conditions
def rect1_length : ℝ := 1
def rect1_width : ℝ := 4
def rect2_length : ℝ := 2
def rect2_width : ℝ := 5
def overlap_height : ℝ := 1

-- Calculate the perimeter of each rectangle
def perimeter_rect1 : ℝ := perimeter_rectangle rect1_length rect1_width
def perimeter_rect2 : ℝ := perimeter_rectangle rect2_length rect2_width

-- Calculate the overlap adjustment
def overlap_adjustment : ℝ := 2 * overlap_height

-- The total perimeter of the T shape
def perimeter_T : ℝ := perimeter_rect1 + perimeter_rect2 - overlap_adjustment

-- The proof statement that we need to show
theorem T_perimeter_is_20 : perimeter_T = 20 := by
  sorry

end T_perimeter_is_20_l48_48607


namespace max_area_equilateral_in_rectangle_l48_48075

-- Define the dimensions of the rectangle
def length_efgh : ℕ := 15
def width_efgh : ℕ := 8

-- The maximum possible area of an equilateral triangle inscribed in the rectangle
theorem max_area_equilateral_in_rectangle : 
  ∃ (s : ℝ), 
  s = ((16 * Real.sqrt 3) / 3) ∧ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ length_efgh → 
    (∃ (area : ℝ), area = (Real.sqrt 3 / 4 * s^2) ∧
      area = 64 * Real.sqrt 3)) :=
by sorry

end max_area_equilateral_in_rectangle_l48_48075


namespace cube_root_of_neg_eight_l48_48955

theorem cube_root_of_neg_eight : ∃ x : ℝ, x ^ 3 = -8 ∧ x = -2 := by 
  sorry

end cube_root_of_neg_eight_l48_48955


namespace percent_neither_condition_l48_48852

namespace TeachersSurvey

variables (Total HighBloodPressure HeartTrouble Both: ℕ)

theorem percent_neither_condition :
  Total = 150 → HighBloodPressure = 90 → HeartTrouble = 50 → Both = 30 →
  (HighBloodPressure + HeartTrouble - Both) = 110 →
  ((Total - (HighBloodPressure + HeartTrouble - Both)) * 100 / Total) = 2667 / 100 :=
by
  intros hTotal hBP hHT hBoth hUnion
  sorry

end TeachersSurvey

end percent_neither_condition_l48_48852


namespace sin_150_eq_half_l48_48696

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by 
  sorry

end sin_150_eq_half_l48_48696


namespace final_result_always_4_l48_48351

-- The function that performs the operations described in the problem
def transform (x : Nat) : Nat :=
  let step1 := 2 * x
  let step2 := step1 + 3
  let step3 := step2 * 5
  let step4 := step3 + 7
  let last_digit := step4 % 10
  let step6 := last_digit + 18
  step6 / 5

-- The theorem statement claiming that for any single-digit number x, the result of transform x is always 4
theorem final_result_always_4 (x : Nat) (h : x < 10) : transform x = 4 := by
  sorry

end final_result_always_4_l48_48351


namespace polynomial_degrees_l48_48422

-- Define the degree requirement for the polynomial.
def polynomial_deg_condition (m n : ℕ) : Prop :=
  2 + m = 5 ∧ n - 2 = 0 ∧ 2 + 2 = 5

theorem polynomial_degrees (m n : ℕ) (h : polynomial_deg_condition m n) : m - n = 1 :=
by
  have h1 : 2 + m = 5 := h.1
  have h2 : n - 2 = 0 := h.2.1
  have h3 := h.2.2
  have : m = 3 := by linarith
  have : n = 2 := by linarith
  linarith

end polynomial_degrees_l48_48422


namespace perpendicular_vectors_l48_48001

def vec := ℝ × ℝ

def dot_product (a b : vec) : ℝ :=
  a.1 * b.1 + a.2 * b.2

variables (m : ℝ)
def a : vec := (1, 2)
def b : vec := (m, 1)

theorem perpendicular_vectors (h : dot_product a (b m) = 0) : m = -2 :=
sorry

end perpendicular_vectors_l48_48001


namespace Vasya_not_11_more_than_Kolya_l48_48304

def is_L_shaped (n : ℕ) : Prop :=
  n % 2 = 1

def total_cells : ℕ :=
  14400

theorem Vasya_not_11_more_than_Kolya (k v : ℕ) :
  (is_L_shaped k) → (is_L_shaped v) → (k + v = total_cells) → (k % 2 = 0) → (v % 2 = 0) → (v - k ≠ 11) := 
by
  sorry

end Vasya_not_11_more_than_Kolya_l48_48304


namespace find_q_zero_l48_48570

-- Assuming the polynomials p, q, and r are defined, and their relevant conditions are satisfied.

def constant_term (f : ℕ → ℝ) : ℝ := f 0

theorem find_q_zero (p q r : ℕ → ℝ)
  (h : p * q = r)
  (h_p_const : constant_term p = 5)
  (h_r_const : constant_term r = -10) :
  q 0 = -2 :=
sorry

end find_q_zero_l48_48570


namespace sin_150_eq_half_l48_48676

theorem sin_150_eq_half :
  sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48676


namespace proper_subsets_count_l48_48197

theorem proper_subsets_count (s : Finset ℤ) (h : s = {-1, 0, 1}) : s.powerset.card - 1 = 7 :=
by
  have hs : s.card = 3 := by rw [h, Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem, Finset.card_singleton]; dec_trivial
  rw [←Finset.card_powerset, Finset.card_eq_to_nat, Finset.card_powerset, hs, pow_succ, pow_succ, one_add_one_eq_two] at hs
  simp only [nat.cast_ite, nat.cast_one, nat.cast_bit0, nat.cast_add, nat.cast_mul, nat.cast_pow, nat.cast_bit1]
  sorry

end proper_subsets_count_l48_48197


namespace parry_position_probability_l48_48292

theorem parry_position_probability :
    let total_members := 20
    let positions := ["President", "Vice President", "Secretary", "Treasurer"]
    let remaining_for_secretary := 18
    let remaining_for_treasurer := 17
    let prob_parry_secretary := (1 : ℚ) / remaining_for_secretary
    let prob_parry_treasurer_given_not_secretary := (1 : ℚ) / remaining_for_treasurer
    let overall_probability := prob_parry_secretary + prob_parry_treasurer_given_not_secretary * (remaining_for_treasurer / remaining_for_secretary)
    overall_probability = (1 : ℚ) / 9 := 
by
  sorry

end parry_position_probability_l48_48292


namespace prism_width_calculation_l48_48492

theorem prism_width_calculation 
  (l h d : ℝ) 
  (h_l : l = 4) 
  (h_h : h = 10) 
  (h_d : d = 14) :
  ∃ w : ℝ, w = 4 * Real.sqrt 5 ∧ (l^2 + w^2 + h^2 = d^2) := 
by
  use 4 * Real.sqrt 5
  sorry

end prism_width_calculation_l48_48492


namespace sin_150_eq_half_l48_48715

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48715


namespace quadratic_solution_l48_48016

theorem quadratic_solution (x : ℝ) (h1 : x^2 - 6 * x + 8 = 0) (h2 : x ≠ 0) :
  x = 2 ∨ x = 4 :=
sorry

end quadratic_solution_l48_48016


namespace solve_equation_l48_48888

noncomputable def smallest_solution : ℝ :=
(15 - Real.sqrt 549) / 6

theorem solve_equation :
  ∃ x : ℝ, 
    (3 * x / (x - 3) + (3 * x^2 - 27) / x = 18) ∧
    x = smallest_solution :=
by
  sorry

end solve_equation_l48_48888


namespace sum_of_midpoint_coordinates_l48_48622

theorem sum_of_midpoint_coordinates :
  let (x1, y1) := (8, 16)
  let (x2, y2) := (2, -8)
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 :=
by
  sorry

end sum_of_midpoint_coordinates_l48_48622


namespace distance_rowed_downstream_l48_48097

def speed_of_boat_still_water : ℝ := 70 -- km/h
def distance_upstream : ℝ := 240 -- km
def time_upstream : ℝ := 6 -- hours
def time_downstream : ℝ := 5 -- hours

theorem distance_rowed_downstream :
  let V_b := speed_of_boat_still_water
  let V_upstream := distance_upstream / time_upstream
  let V_s := V_b - V_upstream
  let V_downstream := V_b + V_s
  V_downstream * time_downstream = 500 :=
by
  sorry

end distance_rowed_downstream_l48_48097


namespace log_identity_l48_48251

theorem log_identity : (Real.log 2)^3 + 3 * (Real.log 2) * (Real.log 5) + (Real.log 5)^3 = 1 :=
by
  sorry

end log_identity_l48_48251


namespace election_debate_conditions_l48_48169

theorem election_debate_conditions (n : ℕ) (h_n : n ≥ 3) :
  ¬ ∃ (p : ℕ), n = 2 * (2 ^ p - 2) + 1 :=
sorry

end election_debate_conditions_l48_48169


namespace acute_triangle_tangent_sum_geq_3_sqrt_3_l48_48839

theorem acute_triangle_tangent_sum_geq_3_sqrt_3 {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) (h_sum : α + β + γ = π)
  (acute_α : α < π / 2) (acute_β : β < π / 2) (acute_γ : γ < π / 2) :
  Real.tan α + Real.tan β + Real.tan γ >= 3 * Real.sqrt 3 :=
sorry

end acute_triangle_tangent_sum_geq_3_sqrt_3_l48_48839


namespace gabby_money_needed_l48_48524

def make_up_price : ℝ := 65
def skin_care_price_eur : ℝ := 40
def hair_tool_price_gbp : ℝ := 50
def initial_savings_usd : ℝ := 35
def initial_savings_eur : ℝ := 10
def mom_money_usd : ℝ := 20
def dad_money_gbp : ℝ := 20
def chores_money_eur : ℝ := 15
def exchange_rate_usd_to_eur : ℝ := 0.85
def exchange_rate_usd_to_gbp : ℝ := 0.75

def skincare_price_usd : ℝ := skin_care_price_eur / exchange_rate_usd_to_eur
def hair_tool_price_usd : ℝ := hair_tool_price_gbp / exchange_rate_usd_to_gbp
def savings_eur_to_usd : ℝ := initial_savings_eur / exchange_rate_usd_to_eur
def chores_money_usd : ℝ := chores_money_eur / exchange_rate_usd_to_eur
def dad_money_usd : ℝ := dad_money_gbp / exchange_rate_usd_to_gbp

def total_cost_usd : ℝ := make_up_price + skincare_price_usd + hair_tool_price_usd
def total_savings_usd : ℝ := initial_savings_usd + mom_money_usd + savings_eur_to_usd + chores_money_usd + dad_money_usd
def additional_money_needed : ℝ := total_cost_usd - total_savings_usd

theorem gabby_money_needed : additional_money_needed = 67.65 := by
  sorry

end gabby_money_needed_l48_48524


namespace perimeter_of_triangle_l48_48504

theorem perimeter_of_triangle (x y : ℝ) (h : 0 < x) (h1 : 0 < y) (h2 : x < y) :
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  (2 * leg_length + hypotenuse = (y - x) * (1 + 1 / Real.sqrt 2)) :=
by
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  sorry

end perimeter_of_triangle_l48_48504


namespace problem_I4_1_l48_48022

theorem problem_I4_1 (a : ℝ) : ((∃ y : ℝ, x + 2 * y + 3 = 0) ∧ (∃ y : ℝ, 4 * x - a * y + 5 = 0) ∧ 
  (∃ m1 m2 : ℝ, m1 = -(1 / 2) ∧ m2 = 4 / a ∧ m1 * m2 = -1)) → a = 2 :=
sorry

end problem_I4_1_l48_48022


namespace zero_nim_sum_move_nonzero_nim_sum_move_winning_strategy_move_for_345_l48_48341

variable (n : ℕ)
variable {m : ℕ}
variable {piles : List ℕ}

-- Given the condition for nim-sum
def nim_sum (piles : List ℕ) : ℕ :=
  List.foldr xor 0 piles

-- Part (a)
theorem zero_nim_sum_move (piles : List ℕ) (h : nim_sum piles = 0)
: ∃ piles', nim_sum piles' ≠ 0 :=
sorry

-- Part (b)
theorem nonzero_nim_sum_move (piles : List ℕ) (h : nim_sum piles ≠ 0)
: ∃ piles', nim_sum piles' = 0 :=
sorry

-- Part (c)
theorem winning_strategy (piles : List ℕ)
: (nim_sum piles = 0 ∧ ∀ piles', nim_sum piles' ≠ 0) ∨ (nim_sum piles ≠ 0 ∧ ∃ piles', nim_sum piles' = 0) :=
sorry

-- Part (d)
def move_for_piles (piles : List ℕ) : List ℕ :=
if piles = [3, 4, 5] then [1, 4, 5] else piles

theorem move_for_345 (piles : List ℕ) (h : piles = [3, 4, 5])
: move_for_piles piles = [1, 4, 5] :=
by {simp [move_for_piles, h]}

end zero_nim_sum_move_nonzero_nim_sum_move_winning_strategy_move_for_345_l48_48341


namespace jana_height_l48_48430

theorem jana_height (Jess_height : ℕ) (h1 : Jess_height = 72) 
  (Kelly_height : ℕ) (h2 : Kelly_height = Jess_height - 3) 
  (Jana_height : ℕ) (h3 : Jana_height = Kelly_height + 5) : 
  Jana_height = 74 := by
  subst h1
  subst h2
  subst h3
  sorry

end jana_height_l48_48430


namespace minimum_of_a_plus_b_l48_48897

theorem minimum_of_a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 4/b = 1) : a + b ≥ 9 :=
by sorry

end minimum_of_a_plus_b_l48_48897


namespace find_x_l48_48164

theorem find_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 152) : x = 16 := 
by
  sorry

end find_x_l48_48164


namespace distribution_ways_l48_48353

theorem distribution_ways :
  ∃ (n : ℕ) (erasers pencils notebooks pens : ℕ),
  pencils = 4 ∧ notebooks = 2 ∧ pens = 3 ∧ 
  n = 6 := sorry

end distribution_ways_l48_48353


namespace sin_150_eq_half_l48_48678

noncomputable def calculate_sin_150 : ℝ := Real.sin (150 * Real.pi / 180)

theorem sin_150_eq_half : calculate_sin_150 = 1 / 2 :=
by
  -- We would include the detailed steps if not skipping the proof as per the instruction.
  sorry

end sin_150_eq_half_l48_48678


namespace inequality_solution_l48_48367

theorem inequality_solution (x : ℝ) : 
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 :=
by
  sorry

end inequality_solution_l48_48367


namespace inscribed_circle_radius_l48_48833

noncomputable def radius_inscribed_circle (DE DF EF : ℝ) : ℝ := 
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius :
  radius_inscribed_circle 8 5 9 = 6 * Real.sqrt 11 / 11 :=
by
  sorry

end inscribed_circle_radius_l48_48833


namespace range_of_a_no_real_roots_l48_48011

theorem range_of_a_no_real_roots (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 + ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_no_real_roots_l48_48011


namespace divisible_by_six_l48_48552

theorem divisible_by_six (n : ℕ) (hn : n > 0) (h : 72 ∣ n^2) : 6 ∣ n :=
sorry

end divisible_by_six_l48_48552


namespace single_elimination_games_l48_48261

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ games : ℕ, games = n - 1 :=
by
  use 22
  sorry

end single_elimination_games_l48_48261


namespace negation_p_l48_48446

def nonneg_reals := { x : ℝ // 0 ≤ x }

def p := ∀ x : nonneg_reals, Real.exp x.1 ≥ 1

theorem negation_p :
  ¬ p ↔ ∃ x : nonneg_reals, Real.exp x.1 < 1 :=
by
  sorry

end negation_p_l48_48446


namespace rockets_win_30_l48_48151

-- Given conditions
def hawks_won (h : ℕ) (w : ℕ) : Prop := h > w
def rockets_won (r : ℕ) (k : ℕ) (l : ℕ) : Prop := r > k ∧ r < l
def knicks_at_least (k : ℕ) : Prop := k ≥ 15
def clippers_won (c : ℕ) (l : ℕ) : Prop := c < l

-- Possible number of games won
def possible_games : List ℕ := [15, 20, 25, 30, 35, 40]

-- Prove Rockets won 30 games
theorem rockets_win_30 (h w r k l c : ℕ) 
  (h_w: hawks_won h w)
  (r_kl : rockets_won r k l)
  (k_15: knicks_at_least k)
  (c_l : clippers_won c l)
  (h_mem : h ∈ possible_games)
  (w_mem : w ∈ possible_games)
  (r_mem : r ∈ possible_games)
  (k_mem : k ∈ possible_games)
  (l_mem : l ∈ possible_games)
  (c_mem : c ∈ possible_games) :
  r = 30 :=
sorry

end rockets_win_30_l48_48151


namespace largest_divisor_of_expression_l48_48912

theorem largest_divisor_of_expression (x : ℤ) (h_even : x % 2 = 0) :
  ∃ k, (∀ x, x % 2 = 0 → k ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) ∧ 
       (∀ m, (∀ x, x % 2 = 0 → m ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) → m ≤ k) ∧ 
       k = 32 :=
sorry

end largest_divisor_of_expression_l48_48912


namespace sin_150_eq_half_l48_48666

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48666


namespace sin_150_eq_half_l48_48699

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by 
  sorry

end sin_150_eq_half_l48_48699


namespace product_of_fraction_l48_48212

-- Define the repeating decimal as given in the problem
def repeating_decimal : Rat := 0.018 -- represents 0.\overline{018}

-- Define the given fraction obtained by simplifying
def simplified_fraction : Rat := 2 / 111

-- The goal is to prove that the product of the numerator and denominator of 
-- the simplified fraction of the repeating decimal is 222
theorem product_of_fraction (y : Rat) (hy : y = 0.018) (fraction_eq : y = 18 / 999) : 
  (2:ℕ) * (111:ℕ) = 222 :=
by
  sorry

end product_of_fraction_l48_48212


namespace accommodate_students_l48_48378

-- Define the parameters
def number_of_classrooms := 15
def one_third_classrooms := number_of_classrooms / 3
def desks_per_classroom_30 := 30
def desks_per_classroom_25 := 25

-- Define the number of classrooms for each type
def classrooms_with_30_desks := one_third_classrooms
def classrooms_with_25_desks := number_of_classrooms - classrooms_with_30_desks

-- Calculate total number of students that can be accommodated
def total_students : ℕ := 
  (classrooms_with_30_desks * desks_per_classroom_30) +
  (classrooms_with_25_desks * desks_per_classroom_25)

-- Prove that total number of students that the school can accommodate is 400
theorem accommodate_students : total_students = 400 := sorry

end accommodate_students_l48_48378


namespace geometric_probability_l48_48735

noncomputable def circle_C : set (ℝ × ℝ) := {p | (p.1 - 1) ^ 2 + p.2 ^ 2 ≤ 1}

theorem geometric_probability :
  let P : measure_theory.probability_space (ℝ × ℝ) := sorry
  in ∫⁻ p in circle_C, if p.1 < 1 then (1 : ℝ) else 0 = (1 / 2 : ℝ) :=
sorry

end geometric_probability_l48_48735


namespace average_age_of_women_l48_48591

theorem average_age_of_women (A : ℝ) (W1 W2 : ℝ)
  (cond1 : 10 * (A + 6) - 10 * A = 60)
  (cond2 : W1 + W2 = 60 + 40) :
  (W1 + W2) / 2 = 50 := 
by
  sorry

end average_age_of_women_l48_48591


namespace min_total_rope_cut_l48_48854

theorem min_total_rope_cut (len1 len2 len3 p1 p2 p3 p4: ℕ) (hl1 : len1 = 52) (hl2 : len2 = 37)
  (hl3 : len3 = 25) (hp1 : p1 = 7) (hp2 : p2 = 3) (hp3 : p3 = 1) 
  (hp4 : ∃ x y z : ℕ, x * p1 + y * p2 + z * p3 = len1 + len2 - len3 ∧ x + y + z ≤ 25) :
  p4 = 82 := 
sorry

end min_total_rope_cut_l48_48854


namespace polynomial_at_most_one_integer_root_l48_48328

theorem polynomial_at_most_one_integer_root (n : ℤ) :
  ∀ x1 x2 : ℤ, (x1 ≠ x2) → 
  (x1 ^ 4 - 1993 * x1 ^ 3 + (1993 + n) * x1 ^ 2 - 11 * x1 + n = 0) → 
  (x2 ^ 4 - 1993 * x2 ^ 3 + (1993 + n) * x2 ^ 2 - 11 * x2 + n = 0) → 
  false :=
by
  sorry

end polynomial_at_most_one_integer_root_l48_48328


namespace max_cos_y_cos_x_l48_48198

noncomputable def max_cos_sum : ℝ :=
  1 + (Real.sqrt (2 + Real.sqrt 2)) / 2

theorem max_cos_y_cos_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.sin x + Real.cos (3 * x) = 0)
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) + Real.cos (2 * x)) :
  ∃ (x y : ℝ), Real.cos y + Real.cos x = max_cos_sum :=
sorry

end max_cos_y_cos_x_l48_48198


namespace math_problem_l48_48006

noncomputable def answer := 21

theorem math_problem 
  (a b c d x : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |x| = 3) : 
  2 * x^2 - (a * b - c - d) + |a * b + 3| = answer := 
sorry

end math_problem_l48_48006


namespace scientific_notation_7nm_l48_48190

theorem scientific_notation_7nm :
  ∀ (x : ℝ), x = 0.000000007 → x = 7 * 10^(-9) :=
begin
  intros x hx,
  sorry
end

end scientific_notation_7nm_l48_48190


namespace probability_heads_fair_coin_l48_48470

-- Define the events and the fair nature of the coin
variable {Ω : Type} [ProbabilitySpace Ω]
variable (coin_toss : Ω → ℙ)

def fair_coin : Prop :=
  (coin_toss ℙ.heads = 1/2) ∧ (coin_toss ℙ.tails = 1/2)

-- Theorem stating the probability of heads in a fair coin toss
theorem probability_heads_fair_coin (h : fair_coin coin_toss) : coin_toss ℙ.heads = 1/2 :=
sorry

end probability_heads_fair_coin_l48_48470


namespace sum_first_five_terms_l48_48133

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 1, ∀ n, a (n + 1) = a n * q

theorem sum_first_five_terms (h₁ : is_geometric_sequence a) 
  (h₂ : a 1 > 0) 
  (h₃ : a 1 * a 7 = 64) 
  (h₄ : a 3 + a 5 = 20) : 
  a 1 * (1 - (2 : ℝ) ^ 5) / (1 - 2) = 31 := 
by
  sorry

end sum_first_five_terms_l48_48133


namespace infinite_B_l48_48423

open Set Function

variable (A B : Type) 

theorem infinite_B (hA_inf : Infinite A) (f : A → B) : Infinite B :=
by
  sorry

end infinite_B_l48_48423


namespace blake_spending_on_oranges_l48_48860

theorem blake_spending_on_oranges (spending_on_oranges spending_on_apples spending_on_mangoes initial_amount change_amount: ℝ)
  (h1 : spending_on_apples = 50)
  (h2 : spending_on_mangoes = 60)
  (h3 : initial_amount = 300)
  (h4 : change_amount = 150)
  (h5 : initial_amount - change_amount = spending_on_oranges + spending_on_apples + spending_on_mangoes) :
  spending_on_oranges = 40 := by
  sorry

end blake_spending_on_oranges_l48_48860


namespace root_eq_neg_l48_48558

theorem root_eq_neg {a : ℝ} (h : 3 * a - 9 < 0) : (a - 4) * (a - 5) > 0 :=
by
  sorry

end root_eq_neg_l48_48558


namespace polynomial_constant_l48_48405

theorem polynomial_constant (P : ℝ → ℝ → ℝ) (h : ∀ x y : ℝ, P (x + y) (y - x) = P x y) : 
  ∃ c : ℝ, ∀ x y : ℝ, P x y = c := 
sorry

end polynomial_constant_l48_48405


namespace find_p_l48_48644

def parabola_def (p : ℝ) : Prop := p > 0 ∧ ∀ (m : ℝ), (2 - (-p/2) = 4)

theorem find_p (p : ℝ) (m : ℝ) (h₁ : parabola_def p) (h₂ : (m ^ 2) = 2 * p * 2) 
(h₃ : (m ^ 2) = 2 * p * 2 → dist (2, m) (p / 2, 0) = 4) :
p = 4 :=
by
  sorry

end find_p_l48_48644


namespace book_pages_l48_48118

theorem book_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) : 
  pages_per_day = 8 → days = 12 → total_pages = pages_per_day * days → total_pages = 96 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end book_pages_l48_48118


namespace normal_probability_l48_48757

noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ :=
∫ t in -∞..x, (1 / (σ * sqrt(2 * π))) * exp (-(t - μ)^2 / (2 * σ^2))

theorem normal_probability (a : ℝ) :
  let x := normal_cdf 2 2 in
  x a = 0.2 →
  x (4 - a) = 0.8 :=
by
  sorry

end normal_probability_l48_48757


namespace quadratic_root_is_imaginary_unit_l48_48017

theorem quadratic_root_is_imaginary_unit (p q : ℝ)
  (h_eq : ∀ z : ℂ, z^2 + p * z + q = 0 → (z = 1 + complex.I ∨ z = 1 - complex.I))
  : q = 2 :=
sorry

end quadratic_root_is_imaginary_unit_l48_48017


namespace average_age_increase_l48_48614

theorem average_age_increase 
  (n : Nat) 
  (a : ℕ) 
  (b : ℕ) 
  (total_students : Nat)
  (avg_age_9 : ℕ) 
  (tenth_age : ℕ) 
  (original_total_age : Nat)
  (new_total_age : Nat)
  (new_avg_age : ℕ)
  (age_increase : ℕ) 
  (h1 : n = 9) 
  (h2 : avg_age_9 = 8) 
  (h3 : tenth_age = 28)
  (h4 : total_students = 10)
  (h5 : original_total_age = n * avg_age_9) 
  (h6 : new_total_age = original_total_age + tenth_age)
  (h7 : new_avg_age = new_total_age / total_students)
  (h8 : age_increase = new_avg_age - avg_age_9) :
  age_increase = 2 := 
by 
  sorry

end average_age_increase_l48_48614


namespace smallest_number_of_coins_l48_48053

theorem smallest_number_of_coins : ∃ (n : ℕ), 
  n ≡ 2 [MOD 5] ∧ 
  n ≡ 1 [MOD 4] ∧ 
  n ≡ 0 [MOD 3] ∧ 
  n = 57 := 
by
  sorry

end smallest_number_of_coins_l48_48053


namespace white_truck_chance_l48_48659

-- Definitions from conditions
def trucks : ℕ := 50
def cars : ℕ := 40
def vans : ℕ := 30

def red_trucks : ℕ := 50 / 2
def black_trucks : ℕ := (20 * 50) / 100

-- The remaining percentage (30%) of trucks is assumed to be white.
def white_trucks : ℕ := (30 * 50) / 100

def total_vehicles : ℕ := trucks + cars + vans

-- Given
def percentage_white_truck : ℕ := (white_trucks * 100) / total_vehicles

-- Theorem that proves the problem statement
theorem white_truck_chance : percentage_white_truck = 13 := 
by
  -- Proof will be written here (currently stubbed)
  sorry

end white_truck_chance_l48_48659


namespace find_x_value_l48_48748

theorem find_x_value (PQ_is_straight_line : True) 
  (angles_on_line : List ℕ) (h : angles_on_line = [x, x, x, x, x])
  (sum_of_angles : angles_on_line.sum = 180) :
  x = 36 :=
by
  sorry

end find_x_value_l48_48748


namespace lattice_points_count_l48_48549

theorem lattice_points_count : ∃ (S : Finset (ℤ × ℤ)), 
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 45}.toFinset = S ∧ S.card = 6 := 
sorry

end lattice_points_count_l48_48549


namespace find_n_l48_48569

   theorem find_n (n : ℕ) : 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (n = 34 ∨ n = 37) :=
   by
     intros
     sorry
   
end find_n_l48_48569


namespace least_n_probability_lt_1_over_10_l48_48838

theorem least_n_probability_lt_1_over_10 : 
  ∃ (n : ℕ), (1 / 2 : ℝ) ^ n < 1 / 10 ∧ ∀ m < n, ¬ ((1 / 2 : ℝ) ^ m < 1 / 10) :=
by
  sorry

end least_n_probability_lt_1_over_10_l48_48838


namespace train_speed_ratio_l48_48473

theorem train_speed_ratio 
  (v_A v_B : ℝ)
  (h1 : v_A = 2 * v_B)
  (h2 : 27 = L_A / v_A)
  (h3 : 17 = L_B / v_B)
  (h4 : 22 = (L_A + L_B) / (v_A + v_B))
  (h5 : v_A + v_B ≤ 60) :
  v_A / v_B = 2 := by
  sorry

-- Conditions given must be defined properly
variables (L_A L_B : ℝ)

end train_speed_ratio_l48_48473


namespace fraction_of_number_l48_48837

theorem fraction_of_number (N : ℕ) (hN : N = 180) : 
  (6 + (1 / 2) * (1 / 3) * (1 / 5) * N) = (1 / 25) * N := 
by
  sorry

end fraction_of_number_l48_48837


namespace smallest_third_term_GP_l48_48104

theorem smallest_third_term_GP : 
  ∃ d : ℝ, 
    (11 + d) ^ 2 = 9 * (29 + 2 * d) ∧
    min (29 + 2 * 10) (29 + 2 * -14) = 1 :=
by
  sorry

end smallest_third_term_GP_l48_48104


namespace lines_perpendicular_l48_48150

theorem lines_perpendicular 
  (x y : ℝ)
  (first_angle : ℝ)
  (second_angle : ℝ)
  (h1 : first_angle = 50 + x - y)
  (h2 : second_angle = first_angle - (10 + 2 * x - 2 * y)) :
  first_angle + second_angle = 90 :=
by 
  sorry

end lines_perpendicular_l48_48150


namespace average_length_remaining_strings_l48_48363

theorem average_length_remaining_strings 
  (T1 : ℕ := 6) (avg_length1 : ℕ := 80) 
  (T2 : ℕ := 2) (avg_length2 : ℕ := 70) :
  (6 * avg_length1 - 2 * avg_length2) / 4 = 85 := 
by
  sorry

end average_length_remaining_strings_l48_48363


namespace pups_more_than_adults_l48_48926

-- Define the counts of dogs
def H := 5  -- number of huskies
def P := 2  -- number of pitbulls
def G := 4  -- number of golden retrievers

-- Define the number of pups each type of dog had
def pups_per_husky_and_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky_and_pitbull + additional_pups_per_golden_retriever

-- Calculate the total number of pups
def total_pups := H * pups_per_husky_and_pitbull + P * pups_per_husky_and_pitbull + G * pups_per_golden_retriever

-- Calculate the total number of adult dogs
def total_adult_dogs := H + P + G

-- Prove that the number of pups is 30 more than the number of adult dogs
theorem pups_more_than_adults : total_pups - total_adult_dogs = 30 :=
by
  -- fill in the proof later
  sorry

end pups_more_than_adults_l48_48926


namespace log_expression_l48_48657

theorem log_expression :
  (Real.log 2)^2 + Real.log 2 * Real.log 5 + Real.log 5 = 1 := by
  sorry

end log_expression_l48_48657


namespace value_of_c_l48_48163

theorem value_of_c :
  ∃ (a b c : ℕ), 
  30 = 2 * (10 + a) ∧ 
  b = 2 * (a + 30) ∧ 
  c = 2 * (b + 30) ∧ 
  c = 200 := 
sorry

end value_of_c_l48_48163


namespace circle_symmetric_to_line_l48_48138

theorem circle_symmetric_to_line (m : ℝ) :
  (∃ (x y : ℝ), (x^2 + y^2 - m * x + 3 * y + 3 = 0) ∧ (m * x + y - m = 0))
  → m = 3 :=
by
  sorry

end circle_symmetric_to_line_l48_48138


namespace sin_150_eq_half_l48_48673

theorem sin_150_eq_half :
  sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48673


namespace tetrahedrons_with_equal_face_areas_have_unequal_volumes_l48_48031

theorem tetrahedrons_with_equal_face_areas_have_unequal_volumes :
  ∃ (T1 T2 : EuclideanGeometry.tetrahedron), 
  (∀ f1 ∈ T1.faces, ∃ f2 ∈ T2.faces, f1.area = f2.area) ∧ T1.volume ≠ T2.volume := by
sorry

end tetrahedrons_with_equal_face_areas_have_unequal_volumes_l48_48031


namespace mean_home_runs_l48_48604

theorem mean_home_runs :
  let n_5 := 3
  let n_8 := 5
  let n_9 := 3
  let n_11 := 1
  let total_home_runs := 5 * n_5 + 8 * n_8 + 9 * n_9 + 11 * n_11
  let total_players := n_5 + n_8 + n_9 + n_11
  let mean := total_home_runs / total_players
  mean = 7.75 :=
by
  sorry

end mean_home_runs_l48_48604


namespace lines_parallel_value_of_a_l48_48018

theorem lines_parallel_value_of_a (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 → (∃ m₁ : ℝ, y = -a / 3 * x + m₁))
  → (∀ x y : ℝ, 2 * x + (a + 1) * y + 1 = 0 → (∃ m₂ : ℝ, y = -2 / (a + 1) * x + m₂))
  → a = -3 :=
by 
  intros h1 h2
  have h3 : ∃ m₁ : ℝ, ∃ m₂ : ℝ, m₁ = m₂ :=
    by 
      obtain ⟨m₁, hm₁⟩ := h1 0 0 (by simp)
      obtain ⟨m₂, hm₂⟩ := h2 0 0 (by simp)
      use [m₁, m₂]
      sorry
  sorry

end lines_parallel_value_of_a_l48_48018


namespace sin_150_equals_half_l48_48694

noncomputable def sin_150_eq_half : Prop := 
  sin (150 * real.pi / 180) = 1 / 2

theorem sin_150_equals_half : sin_150_eq_half :=
  by sorry

end sin_150_equals_half_l48_48694


namespace problem_l48_48547

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem problem (surj_f : ∀ y, ∃ x, f x = y) 
                (inj_g : ∀ x1 x2, g x1 = g x2 → x1 = x2)
                (f_ge_g : ∀ n, f n ≥ g n) :
  ∀ n, f n = g n := 
by 
  sorry

end problem_l48_48547


namespace calculate_expression_l48_48867

theorem calculate_expression : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  calc
    10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2
    _ = 10 - 9 + 56 + 6 - 20 + 3 - 2 : by rw [mul_comm 8 7, mul_comm 5 4] -- Perform multiplications
    _ = 1 + 56 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 10 - 9
    _ = 57 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 1 + 56
    _ = 63 - 20 + 3 - 2 : by norm_num  -- Simplify 57 + 6
    _ = 43 + 3 - 2 : by norm_num -- Simplify 63 - 20
    _ = 46 - 2 : by norm_num -- Simplify 43 + 3
    _ = 44 : by norm_num -- Simplify 46 - 2

end calculate_expression_l48_48867


namespace inequality_holds_l48_48778

theorem inequality_holds (x : ℝ) (n : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 1) (h3 : n > 0) : 
  (1 + x) ^ n ≥ (1 - x) ^ n + 2 * n * x * (1 - x ^ 2) ^ ((n - 1) / 2) :=
sorry

end inequality_holds_l48_48778


namespace distinct_solutions_abs_eq_l48_48257

theorem distinct_solutions_abs_eq (x : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |x1 - |3 * x1 + 2|| = 4 ∧ |x2 - |3 * x2 + 2|| = 4 ∧
    (∀ x3 : ℝ, |x3 - |3 * x3 + 2|| = 4 → (x3 = x1 ∨ x3 = x2))) :=
sorry

end distinct_solutions_abs_eq_l48_48257


namespace rope_length_l48_48103

-- Definitions and assumptions directly derived from conditions
variable (total_length : ℕ)
variable (part_length : ℕ)
variable (sub_part_length : ℕ)

-- Conditions
def condition1 : Prop := total_length / 4 = part_length
def condition2 : Prop := (part_length / 2) * 2 = part_length
def condition3 : Prop := part_length / 2 = sub_part_length
def condition4 : Prop := sub_part_length = 25

-- Proof problem statement
theorem rope_length (h1 : condition1 total_length part_length)
                    (h2 : condition2 part_length)
                    (h3 : condition3 part_length sub_part_length)
                    (h4 : condition4 sub_part_length) :
                    total_length = 100 := 
sorry

end rope_length_l48_48103


namespace sum_geometric_series_nine_l48_48530

noncomputable def geometric_series_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  S n = a 0 * (1 - a 1 ^ n) / (1 - a 1)

theorem sum_geometric_series_nine
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (S_3 : S 3 = 12)
  (S_6 : S 6 = 60) :
  S 9 = 252 := by
  sorry

end sum_geometric_series_nine_l48_48530


namespace randy_total_trees_l48_48790

def mango_trees : ℕ := 60
def coconut_trees : ℕ := mango_trees / 2 - 5
def total_trees (mangos coconuts : ℕ) : ℕ := mangos + coconuts

theorem randy_total_trees : total_trees mango_trees coconut_trees = 85 :=
by
  sorry

end randy_total_trees_l48_48790


namespace binom_divisibility_l48_48727

open Nat

theorem binom_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  (binom n p) - (Nat.floor (n / p : ℚ)) ≡ 0 [MOD p] :=
sorry

end binom_divisibility_l48_48727


namespace partition_2004_ways_l48_48746

theorem partition_2004_ways : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2004 → 
  ∃! (q r : ℕ), 2004 = q * n + r ∧ 0 ≤ r ∧ r < n :=
by
  sorry

end partition_2004_ways_l48_48746


namespace point_in_fourth_quadrant_l48_48156

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) : a > 0 ∧ b < 0 :=
by 
  have hb : b < 0 := sorry
  exact ⟨h1, hb⟩

end point_in_fourth_quadrant_l48_48156


namespace women_in_club_l48_48370

theorem women_in_club (total_members : ℕ) (men : ℕ) (total_members_eq : total_members = 52) (men_eq : men = 37) :
  ∃ women : ℕ, women = 15 :=
by
  sorry

end women_in_club_l48_48370


namespace jasmine_weight_l48_48633

theorem jasmine_weight :
  (∀ (chips_weight cookie_weight: ℕ),
    chips_weight = 20 ∧
    cookie_weight = 9 ∧
    ∃ (num_bags num_tins: ℕ),
      num_bags = 6 ∧
      num_tins = 4 * num_bags ∧
      let total_weight_oz := num_bags * chips_weight + num_tins * cookie_weight in
      total_weight_oz / 16 = 21) :=
begin
  intros,
  use [6], -- number of bags of chips
  use [4 * 6], -- number of tins of cookies
  split; norm_num,
  simp,
  sorry
end

end jasmine_weight_l48_48633


namespace Luke_spent_money_l48_48046

theorem Luke_spent_money : ∀ (initial_money additional_money current_money x : ℕ),
  initial_money = 48 →
  additional_money = 21 →
  current_money = 58 →
  (initial_money + additional_money - current_money) = x →
  x = 11 :=
by
  intros initial_money additional_money current_money x h1 h2 h3 h4
  sorry

end Luke_spent_money_l48_48046


namespace simplify_expression_l48_48502

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
  (x-2) ^ 2 - x * (x-1) + (x^3 - 4 * x^2) / x^2 = -2 * x := 
by 
  sorry

end simplify_expression_l48_48502


namespace inequality_condition_l48_48729

theorem inequality_condition (a x : ℝ) : 
  x^3 + 13 * a^2 * x > 5 * a * x^2 + 9 * a^3 ↔ x > a := 
by
  sorry

end inequality_condition_l48_48729


namespace cole_average_speed_l48_48254

noncomputable def cole_average_speed_to_work : ℝ :=
  let time_to_work := 1.2
  let return_trip_speed := 105
  let total_round_trip_time := 2
  let time_to_return := total_round_trip_time - time_to_work
  let distance_to_work := return_trip_speed * time_to_return
  distance_to_work / time_to_work

theorem cole_average_speed : cole_average_speed_to_work = 70 := by
  sorry

end cole_average_speed_l48_48254


namespace usual_walk_time_l48_48976

theorem usual_walk_time (S T : ℝ)
  (h : S / (2/3 * S) = (T + 15) / T) : T = 30 :=
by
  sorry

end usual_walk_time_l48_48976


namespace lines_intersection_points_l48_48256

theorem lines_intersection_points :
  let line1 (x y : ℝ) := 2 * y - 3 * x = 4
  let line2 (x y : ℝ) := 3 * x + y = 5
  let line3 (x y : ℝ) := 6 * x - 4 * y = 8
  ∃ p1 p2 : (ℝ × ℝ),
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
    (p1 = (2, 5)) ∧ (p2 = (14/9, 1/3)) :=
by
  sorry

end lines_intersection_points_l48_48256


namespace present_age_of_son_l48_48101

variable (S M : ℕ)

-- Conditions
def condition1 := M = S + 28
def condition2 := M + 2 = 2 * (S + 2)

-- Theorem to be proven
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 26 := by
  sorry

end present_age_of_son_l48_48101


namespace range_of_a_l48_48531

theorem range_of_a (a : ℝ) (h₁ : ∀ x : ℝ, x > 0 → x + 4 / x ≥ a) (h₂ : ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) :
  a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l48_48531


namespace area_above_the_line_l48_48355

-- Definitions of the circle and the line equations
def circle_eqn (x y : ℝ) := (x - 5)^2 + (y - 3)^2 = 1
def line_eqn (x y : ℝ) := y = x - 5

-- The main statement to prove
theorem area_above_the_line : 
  ∃ (A : ℝ), A = (3 / 4) * Real.pi ∧ 
  ∀ (x y : ℝ), 
    circle_eqn x y ∧ y > x - 5 → 
    A > 0 := 
sorry

end area_above_the_line_l48_48355


namespace curves_intersect_on_x_axis_l48_48769

theorem curves_intersect_on_x_axis (t θ a : ℝ) (h : a > 0) :
  (∃ t, (t + 1, 1 - 2 * t).snd = 0) →
  (∃ θ, (a * Real.cos θ, 3 * Real.cos θ).snd = 0) →
  (t + 1 = a * Real.cos θ) →
  a = 3 / 2 :=
by
  intro h1 h2 h3
  sorry

end curves_intersect_on_x_axis_l48_48769


namespace randy_total_trees_l48_48789

theorem randy_total_trees (mango_trees : ℕ) (coconut_trees : ℕ) 
  (h1 : mango_trees = 60) 
  (h2 : coconut_trees = (mango_trees / 2) - 5) : 
  mango_trees + coconut_trees = 85 :=
by
  sorry

end randy_total_trees_l48_48789


namespace pow_mod_eq_l48_48588

theorem pow_mod_eq :
  (13 ^ 7) % 11 = 7 :=
by
  sorry

end pow_mod_eq_l48_48588


namespace choose_bar_length_l48_48447

theorem choose_bar_length (x : ℝ) (h1 : 1 < x) (h2 : x < 4) : x = 3 :=
by
  sorry

end choose_bar_length_l48_48447


namespace perpendicular_lines_condition_l48_48162

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, x + (m + 1) * y = 2 - m → m * x + 2 * y = -8) ↔ m = -2 / 3 :=
by sorry

end perpendicular_lines_condition_l48_48162


namespace sin_150_eq_half_l48_48698

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by 
  sorry

end sin_150_eq_half_l48_48698


namespace circle_representation_l48_48288

theorem circle_representation (a : ℝ): 
  (∃ (x y : ℝ), (x^2 + y^2 + 2*x + a = 0) ∧ (∃ D E F, D = 2 ∧ E = 0 ∧ F = -a ∧ (D^2 + E^2 - 4*F > 0))) ↔ (a > -1) :=
by 
  sorry

end circle_representation_l48_48288


namespace monic_cubic_polynomial_has_root_l48_48879

noncomputable def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem monic_cubic_polynomial_has_root :
  Q (Real.cbrt 3 + 2) = 0 :=
sorry

end monic_cubic_polynomial_has_root_l48_48879


namespace find_c_k_l48_48820

-- Definitions of the arithmetic and geometric sequences
def a (n d : ℕ) := 1 + (n - 1) * d
def b (n r : ℕ) := r ^ (n - 1)
def c (n d r : ℕ) := a n d + b n r

-- Conditions for the specific problem
theorem find_c_k (k d r : ℕ) (h1 : 1 + (k - 2) * d + r ^ (k - 2) = 150) (h2 : 1 + k * d + r ^ k = 1500) : c k d r = 314 :=
by
  sorry

end find_c_k_l48_48820


namespace find_larger_number_l48_48193

theorem find_larger_number (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := 
sorry

end find_larger_number_l48_48193


namespace positive_integer_as_sum_of_distinct_factors_l48_48946

-- Defining that all elements of a list are factors of a given number
def AllFactorsOf (factors : List ℕ) (n : ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Defining that the sum of elements in the list equals a given number
def SumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

-- Theorem statement
theorem positive_integer_as_sum_of_distinct_factors (n m : ℕ) (hn : 0 < n) (hm : 1 ≤ m ∧ m ≤ n!) :
  ∃ factors : List ℕ, factors.length ≤ n ∧ AllFactorsOf factors n! ∧ SumList factors = m := 
sorry

end positive_integer_as_sum_of_distinct_factors_l48_48946


namespace molly_age_l48_48055

theorem molly_age (S M : ℕ) (h1 : S / M = 4 / 3) (h2 : S + 6 = 34) : M = 21 :=
by
  sorry

end molly_age_l48_48055


namespace length_of_bridge_correct_l48_48853

open Real

noncomputable def length_of_bridge (length_of_train : ℝ) (time_to_cross : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed := speed_kmph * (1000 / 3600)
  let total_distance := speed * time_to_cross
  total_distance - length_of_train

theorem length_of_bridge_correct :
  length_of_bridge 200 34.997200223982084 36 = 149.97200223982084 := by
  sorry

end length_of_bridge_correct_l48_48853


namespace regular_polygon_sides_l48_48872

theorem regular_polygon_sides (h : ∀ n : ℕ, 140 * n = 180 * (n - 2)) : n = 9 :=
sorry

end regular_polygon_sides_l48_48872


namespace commission_rate_l48_48114

theorem commission_rate (old_salary new_base_salary sale_amount : ℝ) (required_sales : ℕ) (condition: (old_salary = 75000) ∧ (new_base_salary = 45000) ∧ (sale_amount = 750) ∧ (required_sales = 267)) :
  ∃ commission_rate : ℝ, abs (commission_rate - 0.14981) < 0.0001 :=
by
  sorry

end commission_rate_l48_48114


namespace solve_digits_l48_48981

theorem solve_digits : ∃ A B C : ℕ, (A = 1 ∧ B = 0 ∧ (C = 9 ∨ C = 1)) ∧ 
  (∃ (X : ℕ), X ≥ 2 ∧ (C = X - 1 ∨ C = 1)) ∧ 
  (A * 1000 + B * 100 + B * 10 + C) * (C * 100 + C * 10 + A) = C * 100000 + C * 10000 + C * 1000 + C * 100 + A * 10 + C :=
by sorry

end solve_digits_l48_48981


namespace Sam_and_Tina_distance_l48_48782

theorem Sam_and_Tina_distance (marguerite_distance : ℕ) (marguerite_time : ℕ)
  (sam_time : ℕ) (tina_time : ℕ) (sam_distance : ℕ) (tina_distance : ℕ)
  (h1 : marguerite_distance = 150) (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) (h4 : tina_time = 2)
  (h5 : sam_distance = (marguerite_distance / marguerite_time) * sam_time)
  (h6 : tina_distance = (marguerite_distance / marguerite_time) * tina_time) :
  sam_distance = 200 ∧ tina_distance = 100 :=
by
  sorry

end Sam_and_Tina_distance_l48_48782


namespace count_arithmetic_sequence_terms_l48_48507

theorem count_arithmetic_sequence_terms : 
  ∃ n : ℕ, 
  (∀ k : ℕ, k ≥ 1 → 6 + (k - 1) * 4 = 202 → n = k) ∧ n = 50 :=
by
  sorry

end count_arithmetic_sequence_terms_l48_48507


namespace sin_150_eq_one_half_l48_48685

theorem sin_150_eq_one_half :
  let θ := 150
  let θ_ref := 30
  let θ_complement := 180 - θ_ref
  θ = θ_complement →
  (∀ θ, θ_ref = 30 * 60.toReal)
    ∧ (∀ θ, (cos θ_ref, sin θ_ref) = (Real.sqrt 3 / 2, 1 / 2)) →
  sin θ = 1 / 2 :=
by
  intros θ θ_ref θ_complement h1 h2
  have h3 : (cos θ_complement, sin θ_complement) = (-(cos θ_ref), sin θ_ref) := by
    sorry
  rw h3
  sorry

end sin_150_eq_one_half_l48_48685


namespace brenda_age_l48_48497

theorem brenda_age
  (A B J : ℕ)
  (h1 : A = 4 * B)
  (h2 : J = B + 9)
  (h3 : A = J)
  : B = 3 :=
by 
  sorry

end brenda_age_l48_48497


namespace slices_with_both_pepperoni_and_mushrooms_l48_48841

theorem slices_with_both_pepperoni_and_mushrooms (n : ℕ)
  (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (all_have_topping : ∀ (s : ℕ), s < total_slices → s < pepperoni_slices ∨ s < mushroom_slices ∨ s < (total_slices - pepperoni_slices - mushroom_slices) )
  (total_condition : total_slices = 16)
  (pepperoni_condition : pepperoni_slices = 8)
  (mushroom_condition : mushroom_slices = 12) :
  (8 - n) + (12 - n) + n = 16 → n = 4 :=
sorry

end slices_with_both_pepperoni_and_mushrooms_l48_48841


namespace total_area_correct_l48_48766

-- Define the conditions from the problem
def side_length_small : ℕ := 2
def side_length_medium : ℕ := 4
def side_length_large : ℕ := 8

-- Define the areas of individual squares
def area_small : ℕ := side_length_small * side_length_small
def area_medium : ℕ := side_length_medium * side_length_medium
def area_large : ℕ := side_length_large * side_length_large

-- Define the additional areas as suggested by vague steps in the solution
def area_term1 : ℕ := 4 * 4 / 2 * 2
def area_term2 : ℕ := 2 * 2 / 2
def area_term3 : ℕ := (8 + 2) * 2 / 2 * 2

-- Define the total area as the sum of all calculated parts
def total_area : ℕ := area_large + (area_medium * 3) + area_small + area_term1 + area_term2 + area_term3

-- The theorem to prove total area is 150 square centimeters
theorem total_area_correct : total_area = 150 :=
by
  -- Proof goes here (steps from the solution)...
  sorry

end total_area_correct_l48_48766


namespace divisor_correct_l48_48229

/--
Given that \(10^{23} - 7\) divided by \(d\) leaves a remainder 3, 
prove that \(d\) is equal to \(10^{23} - 10\).
-/
theorem divisor_correct :
  ∃ d : ℤ, (10^23 - 7) % d = 3 ∧ d = 10^23 - 10 :=
by
  sorry

end divisor_correct_l48_48229


namespace quadratic_solutions_l48_48751

theorem quadratic_solutions (x : ℝ) (b : ℝ) (h_symmetry : -b / (2 * 1) = 2) :
  (x ^ 2 + b * x - 5 = 2 * x - 13) ↔ (x = 2 ∨ x = 4) :=
by {
  -- Given -b / 2 = 2, we can solve for b
  have h_b : b = -4,
  -- sorry skips the calculation steps needed for the solution
  sorry,
  -- Substituting b = -4 into the equation x^2 - 4x - 5 = 2x - 13 and simplifying
  have h_eq : x^2 - 6 * x + 8 = 0,
  -- sorry again skips the detailed algebra steps
  sorry,
  -- Factoring the simplified equation and solving for x
  rw [h_eq],
  -- sorry to conclude the equivalence
  sorry,
}

end quadratic_solutions_l48_48751


namespace corridor_length_correct_l48_48024

/-- Scale representation in the blueprint: 1 cm represents 10 meters. --/
def scale_cm_to_m (cm: ℝ): ℝ := cm * 10

/-- Length of the corridor in the blueprint. --/
def blueprint_length_cm: ℝ := 9.5

/-- Real-life length of the corridor. --/
def real_life_length: ℝ := 95

/-- Proof that the real-life length of the corridor is correctly calculated. --/
theorem corridor_length_correct :
  scale_cm_to_m blueprint_length_cm = real_life_length :=
by
  sorry

end corridor_length_correct_l48_48024


namespace vasya_kolya_difference_impossible_l48_48306

theorem vasya_kolya_difference_impossible : 
  ∀ k v : ℕ, (∃ q₁ q₂ : ℕ, 14400 = q₁ * 2 + q₂ * 2 + 1 + 1) → ¬ ∃ k, ∃ v, (v - k = 11 ∧ 14400 = k * q₁ + v * q₂) :=
by sorry

end vasya_kolya_difference_impossible_l48_48306


namespace jana_height_l48_48431

theorem jana_height (Jess_height : ℕ) (h1 : Jess_height = 72) 
  (Kelly_height : ℕ) (h2 : Kelly_height = Jess_height - 3) 
  (Jana_height : ℕ) (h3 : Jana_height = Kelly_height + 5) : 
  Jana_height = 74 := by
  subst h1
  subst h2
  subst h3
  sorry

end jana_height_l48_48431


namespace golden_triangle_expression_l48_48970

noncomputable def t : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_expression :
  t = (Real.sqrt 5 - 1) / 2 →
  (1 - 2 * (Real.sin (27 * Real.pi / 180))^2) / (2 * t * Real.sqrt (4 - t^2)) = 1 / 4 :=
by
  intro h_t
  have h1 : t = (Real.sqrt 5 - 1) / 2 := h_t
  sorry

end golden_triangle_expression_l48_48970


namespace product_xyz_l48_48747

/-- Prove that if x + 1/y = 2 and y + 1/z = 3, then xyz = 1/11. -/
theorem product_xyz {x y z : ℝ} (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 3) : x * y * z = 1 / 11 :=
sorry

end product_xyz_l48_48747


namespace work_completion_time_l48_48092

theorem work_completion_time 
(w : ℝ)  -- total amount of work
(A B : ℝ)  -- work rate of a and b per day
(h1 : A + B = w / 30)  -- combined work rate
(h2 : 20 * (A + B) + 20 * A = w) : 
  (1 / A = 60) :=
sorry

end work_completion_time_l48_48092


namespace exists_h_not_divisible_l48_48400

theorem exists_h_not_divisible : ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ % ⌊h * 1969^(n-1)⌋ = 0) :=
by
  sorry

end exists_h_not_divisible_l48_48400


namespace joao_speed_l48_48452

theorem joao_speed (d : ℝ) (v1 : ℝ) (t1 t2 : ℝ) (h1 : v1 = 10) (h2 : t1 = 6 / 60) (h3 : t2 = 8 / 60) : 
  d = v1 * t1 → d = 10 * (6 / 60) → (d / t2) = 7.5 := 
by
  sorry

end joao_speed_l48_48452


namespace geometric_product_seven_terms_l48_48741

theorem geometric_product_seven_terms (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 1) 
  (h2 : a 6 + a 4 = 2 * (a 3 + a 1)) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) :
  (a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7) = 128 := 
by 
  -- Steps involving algebraic manipulation and properties of geometric sequences should be here
  sorry

end geometric_product_seven_terms_l48_48741


namespace determine_b_l48_48394

theorem determine_b (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x) ∧ (∀ x y : ℝ, y - 2 = (b + 9) * x) → 
  b = -6 :=
by
  sorry

end determine_b_l48_48394


namespace opposite_of_three_minus_one_l48_48465

theorem opposite_of_three_minus_one : -(3 - 1) = -2 := 
by
  sorry

end opposite_of_three_minus_one_l48_48465


namespace find_b_when_a_is_1600_l48_48325

variable (a b : ℝ)

def inversely_vary (a b : ℝ) : Prop := a * b = 400

theorem find_b_when_a_is_1600 
  (h1 : inversely_vary 800 0.5)
  (h2 : inversely_vary a b)
  (h3 : a = 1600) :
  b = 0.25 := by
  sorry

end find_b_when_a_is_1600_l48_48325


namespace fraction_division_correct_l48_48832

theorem fraction_division_correct :
  (5/6 : ℚ) / (7/9) / (11/13) = 195/154 := 
by {
  sorry
}

end fraction_division_correct_l48_48832


namespace triangle_obtuse_l48_48426

-- We need to set up the definitions for angles and their relationships in triangles.

variable {A B C : ℝ} -- representing the angles of the triangle in radians

structure Triangle (A B C : ℝ) : Prop where
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  sum_to_pi : A + B + C = Real.pi -- representing the sum of angles in a triangle

-- Definition to state the condition in the problem
def triangle_condition (A B C : ℝ) : Prop :=
  Triangle A B C ∧ (Real.cos A * Real.cos B - Real.sin A * Real.sin B > 0)

-- Theorem to prove the triangle is obtuse under the given condition
theorem triangle_obtuse {A B C : ℝ} (h : triangle_condition A B C) : ∃ C', C' = C ∧ C' > Real.pi / 2 :=
sorry

end triangle_obtuse_l48_48426


namespace line_through_A1_slope_neg4_over_3_line_through_A2_l48_48366

-- (1) The line passing through point (1, 3) with a slope -4/3
theorem line_through_A1_slope_neg4_over_3 : 
    ∃ (a b c : ℝ), a * 1 + b * 3 + c = 0 ∧ ∃ m : ℝ, m = -4 / 3 ∧ a * m + b = 0 ∧ b ≠ 0 ∧ c = -13 := by
sorry

-- (2) The line passing through point (-5, 2) with x-intercept twice the y-intercept
theorem line_through_A2 : 
    ∃ (a b c : ℝ), (a * -5 + b * 2 + c = 0) ∧ ((∃ m : ℝ, m = 2 ∧ a * m + b = 0 ∧ b = -a) ∨ ((b = -2 / 5 * a) ∧ (a * 2 + b = 0))) := by
sorry

end line_through_A1_slope_neg4_over_3_line_through_A2_l48_48366


namespace parallel_lines_slope_eq_l48_48399

theorem parallel_lines_slope_eq (b : ℝ) :
    (∀ x y : ℝ, 3 * y - 3 * b = 9 * x → ∀ x' y' : ℝ, y' - 2 = (b + 9) * x' → 3 = b + 9) →
    b = -6 := 
by 
  intros h
  have h1 : 3 = b + 9 := sorry -- proof omitted
  rw h1
  norm_num

end parallel_lines_slope_eq_l48_48399


namespace sock_pairs_proof_l48_48823

noncomputable def numPairsOfSocks : ℕ :=
  let n : ℕ := sorry
  n

theorem sock_pairs_proof : numPairsOfSocks = 6 := by
  sorry

end sock_pairs_proof_l48_48823


namespace num_expr_div_by_10_l48_48580

theorem num_expr_div_by_10 : (11^11 + 12^12 + 13^13) % 10 = 0 := by
  sorry

end num_expr_div_by_10_l48_48580


namespace mike_spent_on_new_tires_l48_48895

-- Define the given amounts
def amount_spent_on_speakers : ℝ := 118.54
def total_amount_spent_on_car_parts : ℝ := 224.87

-- Define the amount spent on new tires
def amount_spent_on_new_tires : ℝ := total_amount_spent_on_car_parts - amount_spent_on_speakers

-- The theorem we want to prove
theorem mike_spent_on_new_tires : amount_spent_on_new_tires = 106.33 :=
by
  -- the proof would go here
  sorry

end mike_spent_on_new_tires_l48_48895


namespace circle_equation_l48_48538

theorem circle_equation (a b r : ℝ) 
    (h₁ : b = -4 * a)
    (h₂ : abs (a + b - 1) / Real.sqrt 2 = r)
    (h₃ : (b + 2) / (a - 3) * (-1) = -1)
    (h₄ : a = 1)
    (h₅ : b = -4)
    (h₆ : r = 2 * Real.sqrt 2) :
    ∀ x y: ℝ, (x - 1) ^ 2 + (y + 4) ^ 2 = 8 := 
by
  intros
  sorry

end circle_equation_l48_48538


namespace find_a_l48_48136

theorem find_a (a : ℝ) (h : ∀ x y : ℝ, ax + y - 4 = 0 → x + (a + 3/2) * y + 2 = 0 → True) : a = 1/2 :=
sorry

end find_a_l48_48136


namespace union_A_B_complement_A_l48_48545

-- Definition of Universe U
def U : Set ℝ := Set.univ

-- Definition of set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Definition of set B
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- Theorem 1: Proving the union A ∪ B
theorem union_A_B : A ∪ B = {x | -2 < x ∧ x ≤ 3} := 
sorry

-- Theorem 2: Proving the complement of A with respect to U
theorem complement_A : (U \ A) = {x | x < -1 ∨ x > 3} := 
sorry

end union_A_B_complement_A_l48_48545


namespace minimal_abs_diff_l48_48284

theorem minimal_abs_diff (a b : ℤ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a * b - 3 * a + 7 * b = 222) : |a - b| = 54 :=
by
  sorry

end minimal_abs_diff_l48_48284


namespace find_x_l48_48520

theorem find_x (x : ℝ) (h1: x > 0) (h2 : 1 / 2 * x * (3 * x) = 72) : x = 4 * Real.sqrt 3 :=
sorry

end find_x_l48_48520


namespace denominator_of_simplified_fraction_l48_48403

theorem denominator_of_simplified_fraction : 
  ∀ (num denom : ℕ),
  num = 201920192019 → denom = 191719171917 →
  (201920192019 = 2019 * 100010001) →
  (191719171917 = 1917 * 100010001) →
  (2019 = 3 * 673) →
  (1917 = 3 * 639) →
  (639 = 3^2 * 71) →
  prime 673 →
  (673 % 3 ≠ 0) →
  (673 % 71 ≠ 0) →
  let simplified_denominator := Nat.gcd num denom in
  simplified_denominator = 639 :=
by
  intros num denom hnum hdenom hnum_fact hdenom_fact h2019_fact h1917_fact h639_fact prime_673 h673_mod3 h673_mod71 simplified_denominator
  sorry

end denominator_of_simplified_fraction_l48_48403


namespace sin_150_eq_half_l48_48711

theorem sin_150_eq_half :
  (sin (150 : ℝ)) = 1 / 2 :=
by
  -- Conditions in form of definitions
  let θ : ℝ := 30
  have h1 : 150 = 180 - θ := by ring 
  have h2 : sin (180 - θ) = sin θ := by exact sin_sub_π θ
  have h3 : sin θ = 1 / 2 := by exact real.sin_of_real (by norm_num)
  sorry  -- Proof omitted

end sin_150_eq_half_l48_48711


namespace sin_150_eq_one_half_l48_48687

theorem sin_150_eq_one_half :
  let θ := 150
  let θ_ref := 30
  let θ_complement := 180 - θ_ref
  θ = θ_complement →
  (∀ θ, θ_ref = 30 * 60.toReal)
    ∧ (∀ θ, (cos θ_ref, sin θ_ref) = (Real.sqrt 3 / 2, 1 / 2)) →
  sin θ = 1 / 2 :=
by
  intros θ θ_ref θ_complement h1 h2
  have h3 : (cos θ_complement, sin θ_complement) = (-(cos θ_ref), sin θ_ref) := by
    sorry
  rw h3
  sorry

end sin_150_eq_one_half_l48_48687


namespace Jasmine_total_weight_in_pounds_l48_48635

-- Definitions for the conditions provided
def weight_chips_ounces : ℕ := 20
def weight_cookies_ounces : ℕ := 9
def bags_chips : ℕ := 6
def tins_cookies : ℕ := 4 * bags_chips
def total_weight_ounces : ℕ := (weight_chips_ounces * bags_chips) + (weight_cookies_ounces * tins_cookies)
def total_weight_pounds : ℕ := total_weight_ounces / 16

-- The proof problem statement
theorem Jasmine_total_weight_in_pounds : total_weight_pounds = 21 := 
by
  sorry

end Jasmine_total_weight_in_pounds_l48_48635


namespace minimum_value_x_l48_48542

theorem minimum_value_x (a b x : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
    (H : 4 * a + b * (1 - a) = 0) 
    (Hinequality : ∀ (a b : ℝ), a > 0 → b > 0 → 
        (4 * a + b * (1 - a) = 0 → 
        (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))) : 
    x >= 1 := 
sorry

end minimum_value_x_l48_48542


namespace probability_abs_diff_gt_one_l48_48054

-- Define and prove the probability problem
theorem probability_abs_diff_gt_one :
  let coin_flip_probability := 1/2
  let choose_number_probability := 1/2
  let interval := set.Icc 0 2 -- Closed interval [0, 2]
  let x_choice := if (coin_flip_probability = 1/2) then 2 else 0
  let y_choice := if (coin_flip_probability = 1/2) then 2 else 0
  let x := if (coin_flip_probability = 1/2) then if (choose_number_probability = 1/2) then 2 else 0 else (set.Icc 0 2)
  let y := if (coin_flip_probability = 1/2) then if (choose_number_probability = 1/2) then 2 else 0 else (set.Icc 0 2)
  in (|x - y| > 1) =
      (9/16) :=
  sorry

end probability_abs_diff_gt_one_l48_48054


namespace mul_99_101_equals_9999_l48_48117

theorem mul_99_101_equals_9999 : 99 * 101 = 9999 := by
  sorry

end mul_99_101_equals_9999_l48_48117


namespace opposite_of_x_is_positive_l48_48816

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive_l48_48816


namespace sum_of_center_coordinates_l48_48466

theorem sum_of_center_coordinates 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : (x1, y1) = (4, 3)) 
  (h2 : (x2, y2) = (-6, 5)) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 = 3 := by
  sorry

end sum_of_center_coordinates_l48_48466


namespace min_value_expression_l48_48085

open Real

theorem min_value_expression : ∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 :=
by
  intro x y
  sorry

end min_value_expression_l48_48085


namespace find_modulus_difference_l48_48780

theorem find_modulus_difference (z1 z2 : ℂ) 
  (h1 : complex.abs z1 = 2) 
  (h2 : complex.abs z2 = 2) 
  (h3 : z1 + z2 = complex.mk (real.sqrt 3) 1) : 
  complex.abs (z1 - z2) = 2 * real.sqrt 3 :=
sorry

end find_modulus_difference_l48_48780


namespace sin_150_eq_half_l48_48675

theorem sin_150_eq_half :
  sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48675


namespace one_quarter_between_l48_48194

def one_quarter_way (a b : ℚ) : ℚ :=
  a + 1 / 4 * (b - a)

theorem one_quarter_between :
  one_quarter_way (1 / 7) (1 / 4) = 23 / 112 :=
by
  sorry

end one_quarter_between_l48_48194


namespace solve_for_n_l48_48796

theorem solve_for_n (n : ℝ) : 0.03 * n + 0.05 * (30 + n) + 2 = 8.5 → n = 62.5 :=
by
  intros h
  sorry

end solve_for_n_l48_48796


namespace nine_fifths_sum_l48_48587

open Real

theorem nine_fifths_sum (a b: ℝ) (ha: a > 0) (hb: b > 0)
    (h1: a * (sqrt a) + b * (sqrt b) = 183) 
    (h2: a * (sqrt b) + b * (sqrt a) = 182) : 
    9 / 5 * (a + b) = 657 := 
by 
    sorry

end nine_fifths_sum_l48_48587


namespace sin_150_equals_half_l48_48693

noncomputable def sin_150_eq_half : Prop := 
  sin (150 * real.pi / 180) = 1 / 2

theorem sin_150_equals_half : sin_150_eq_half :=
  by sorry

end sin_150_equals_half_l48_48693


namespace log13_x_equals_log13_43_l48_48421

theorem log13_x_equals_log13_43 (x : ℤ): 
  (log 13 x = log 13 43) -> log 7 (x + 6) = 2 := by
  sorry

end log13_x_equals_log13_43_l48_48421


namespace yellow_balls_count_l48_48638

theorem yellow_balls_count (purple blue total_needed : ℕ) 
  (h_purple : purple = 7) 
  (h_blue : blue = 5) 
  (h_total : total_needed = 19) : 
  ∃ (yellow : ℕ), yellow = 6 :=
by
  sorry

end yellow_balls_count_l48_48638


namespace polynomial_evaluation_l48_48401

noncomputable def x : ℝ :=
  (3 + 3 * Real.sqrt 5) / 2

theorem polynomial_evaluation :
  (x^2 - 3 * x - 9 = 0) → (x^3 - 3 * x^2 - 9 * x + 7 = 7) :=
by
  intros h
  sorry

end polynomial_evaluation_l48_48401


namespace hyperbola_eccentricity_correct_l48_48737

noncomputable def hyperbola_eccentricity : ℝ := 2

variables {a b : ℝ}
variables (ha_pos : 0 < a) (hb_pos : 0 < b)
variables (h_hyperbola : ∃ x y, x^2/a^2 - y^2/b^2 = 1)
variables (h_circle_chord_len : ∃ d, d = 2 ∧ ∃ x y, ((x - 2)^2 + y^2 = 4) ∧ (x * b/a = -y))

theorem hyperbola_eccentricity_correct :
  ∀ (a b : ℝ), 0 < a → 0 < b → (∃ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  ∧ (∃ d, d = 2 ∧ ∃ x y, (x - 2)^2 + y^2 = 4 ∧ (x * b / a = -y)) →
  (eccentricity = 2) :=
by
  intro a b ha_pos hb_pos h_conditions
  have e := hyperbola_eccentricity
  sorry


end hyperbola_eccentricity_correct_l48_48737


namespace distance_between_parallel_sides_l48_48406

-- Define the givens
def length_side_a : ℝ := 24  -- length of one parallel side
def length_side_b : ℝ := 14  -- length of the other parallel side
def area_trapezium : ℝ := 342  -- area of the trapezium

-- We need to prove that the distance between parallel sides (h) is 18 cm
theorem distance_between_parallel_sides (h : ℝ)
  (H1 :  area_trapezium = (1/2) * (length_side_a + length_side_b) * h) :
  h = 18 :=
by sorry

end distance_between_parallel_sides_l48_48406


namespace car_Y_average_speed_l48_48385

theorem car_Y_average_speed 
  (car_X_speed : ℝ)
  (car_X_time_before_Y : ℝ)
  (car_X_distance_when_Y_starts : ℝ)
  (car_X_total_distance : ℝ)
  (car_X_travel_time : ℝ)
  (car_Y_distance : ℝ)
  (car_Y_travel_time : ℝ)
  (h_car_X_speed : car_X_speed = 35)
  (h_car_X_time_before_Y : car_X_time_before_Y = 72 / 60)
  (h_car_X_distance_when_Y_starts : car_X_distance_when_Y_starts = car_X_speed * car_X_time_before_Y)
  (h_car_X_total_distance : car_X_total_distance = car_X_distance_when_Y_starts + car_X_distance_when_Y_starts)
  (h_car_X_travel_time : car_X_travel_time = car_X_total_distance / car_X_speed)
  (h_car_Y_distance : car_Y_distance = 490)
  (h_car_Y_travel_time : car_Y_travel_time = car_X_travel_time) :
  (car_Y_distance / car_Y_travel_time) = 32.24 := 
sorry

end car_Y_average_speed_l48_48385


namespace sum_of_ages_l48_48937

-- Given conditions and definitions
variables (M J : ℝ)

def condition1 : Prop := M = J + 8
def condition2 : Prop := M + 6 = 3 * (J - 3)

-- Proof goal
theorem sum_of_ages (h1 : condition1 M J) (h2 : condition2 M J) : M + J = 31 := 
by sorry

end sum_of_ages_l48_48937


namespace sin_150_eq_half_l48_48663

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by sorry

end sin_150_eq_half_l48_48663


namespace probability_of_winning_l48_48630

def roll_is_seven (d1 d2 : ℕ) : Prop :=
  d1 + d2 = 7

theorem probability_of_winning (d1 d2 : ℕ) (h : roll_is_seven d1 d2) :
  (1/6 : ℚ) = 1/6 :=
by
  sorry

end probability_of_winning_l48_48630


namespace walnuts_count_l48_48294

def nuts_problem (p a c w : ℕ) : Prop :=
  p + a + c + w = 150 ∧
  a = p / 2 ∧
  c = 4 * a ∧
  w = 3 * c

theorem walnuts_count (p a c w : ℕ) (h : nuts_problem p a c w) : w = 96 :=
by sorry

end walnuts_count_l48_48294


namespace inequality_solution_l48_48506

noncomputable def operation (a b : ℝ) : ℝ := (a + 3 * b) - a * b

theorem inequality_solution (x : ℝ) : operation 5 x < 13 → x > -4 := by
  sorry

end inequality_solution_l48_48506


namespace combined_resistance_l48_48627

theorem combined_resistance (x y r : ℝ) (hx : x = 5) (hy : y = 7) (h_parallel : 1 / r = 1 / x + 1 / y) : 
  r = 35 / 12 := 
by 
  sorry

end combined_resistance_l48_48627


namespace minimum_AP_BP_l48_48170

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 3)
def parabola (P : ℝ × ℝ) : Prop := P.2 * P.2 = 8 * P.1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

theorem minimum_AP_BP : 
  ∀ (P : ℝ × ℝ), parabola P → distance A P + distance B P ≥ 3 * Real.sqrt 10 :=
by 
  intros P hP
  sorry

end minimum_AP_BP_l48_48170


namespace sin_150_eq_half_l48_48661

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by sorry

end sin_150_eq_half_l48_48661


namespace g_at_5_l48_48438

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation :
  ∀ (x : ℝ), g x + 3 * g (2 - x) = 4 * x ^ 2 - 5 * x + 1

theorem g_at_5 : g 5 = -5 / 4 :=
by
  let h := functional_equation
  sorry

end g_at_5_l48_48438


namespace find_number_l48_48413

theorem find_number (x : ℝ) (h : 45 - 3 * x = 12) : x = 11 :=
sorry

end find_number_l48_48413


namespace matching_shoes_probability_is_one_ninth_l48_48093

def total_shoes : ℕ := 10
def pairs_of_shoes : ℕ := 5
def total_combinations : ℕ := (total_shoes * (total_shoes - 1)) / 2
def matching_combinations : ℕ := pairs_of_shoes

def matching_shoes_probability : ℚ := matching_combinations / total_combinations

theorem matching_shoes_probability_is_one_ninth :
  matching_shoes_probability = 1 / 9 :=
by
  sorry

end matching_shoes_probability_is_one_ninth_l48_48093


namespace coefficient_x_neg_4_expansion_l48_48922

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the function to calculate the coefficient of the term containing x^(-4)
def coeff_term_x_neg_4 : ℕ :=
  let k := 10
  binom 12 k

theorem coefficient_x_neg_4_expansion :
  coeff_term_x_neg_4 = 66 := by
  -- Calculation here would show that binom 12 10 is indeed 66
  sorry

end coefficient_x_neg_4_expansion_l48_48922


namespace abs_difference_of_numbers_l48_48349

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 104) : |x - y| = 4 * Real.sqrt 10 :=
by
  sorry

end abs_difference_of_numbers_l48_48349


namespace distinct_real_roots_sum_l48_48551

theorem distinct_real_roots_sum (p r_1 r_2 : ℝ) (h_eq : ∀ x, x^2 + p * x + 18 = 0)
  (h_distinct : r_1 ≠ r_2) (h_root1 : x^2 + p * x + 18 = 0)
  (h_root2 : x^2 + p * x + 18 = 0) : |r_1 + r_2| > 6 :=
sorry

end distinct_real_roots_sum_l48_48551


namespace math_problem_l48_48869

theorem math_problem : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end math_problem_l48_48869


namespace find_number_l48_48448

theorem find_number (x : ℚ) (h : 1 + 1 / x = 5 / 2) : x = 2 / 3 :=
by
  sorry

end find_number_l48_48448


namespace triangle_height_in_terms_of_s_l48_48645

theorem triangle_height_in_terms_of_s (s h : ℝ)
  (rectangle_area : 2 * s * s = 2 * s^2)
  (base_of_triangle : base = s)
  (areas_equal : (1 / 2) * s * h = 2 * s^2) :
  h = 4 * s :=
by
  sorry

end triangle_height_in_terms_of_s_l48_48645


namespace discounted_price_is_correct_l48_48842

def marked_price : ℕ := 125
def discount_rate : ℚ := 4 / 100

def calculate_discounted_price (marked_price : ℕ) (discount_rate : ℚ) : ℚ :=
  marked_price - (discount_rate * marked_price)

theorem discounted_price_is_correct :
  calculate_discounted_price marked_price discount_rate = 120 := by
  sorry

end discounted_price_is_correct_l48_48842


namespace floor_plus_r_eq_10_3_implies_r_eq_5_3_l48_48883

noncomputable def floor (x : ℝ) : ℤ := sorry -- Assuming the function exists

theorem floor_plus_r_eq_10_3_implies_r_eq_5_3 (r : ℝ) 
  (h : floor r + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_r_eq_10_3_implies_r_eq_5_3_l48_48883


namespace sequence_a4_eq_neg3_l48_48278

theorem sequence_a4_eq_neg3 (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) : a 4 = -3 :=
by
  sorry

end sequence_a4_eq_neg3_l48_48278


namespace probability_at_least_one_defective_probability_at_most_one_defective_l48_48030

noncomputable def machine_a_defect_rate : ℝ := 0.05
noncomputable def machine_b_defect_rate : ℝ := 0.1

/-- 
Prove the probability that there is at least one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_least_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - (1 - pA) * (1 - pB)) = 0.145 :=
  sorry

/-- 
Prove the probability that there is at most one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_most_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - pA * pB) = 0.995 :=
  sorry

end probability_at_least_one_defective_probability_at_most_one_defective_l48_48030


namespace passengers_on_board_l48_48144

/-- 
Given the fractions of passengers from different continents and remaining 42 passengers,
show that the total number of passengers P is 240.
-/
theorem passengers_on_board :
  ∃ P : ℕ,
    (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) ∧ P = 240 :=
by
  let P := 240
  have h : (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) := sorry
  exact ⟨P, h, rfl⟩

end passengers_on_board_l48_48144


namespace accommodate_students_l48_48377

-- Define the parameters
def number_of_classrooms := 15
def one_third_classrooms := number_of_classrooms / 3
def desks_per_classroom_30 := 30
def desks_per_classroom_25 := 25

-- Define the number of classrooms for each type
def classrooms_with_30_desks := one_third_classrooms
def classrooms_with_25_desks := number_of_classrooms - classrooms_with_30_desks

-- Calculate total number of students that can be accommodated
def total_students : ℕ := 
  (classrooms_with_30_desks * desks_per_classroom_30) +
  (classrooms_with_25_desks * desks_per_classroom_25)

-- Prove that total number of students that the school can accommodate is 400
theorem accommodate_students : total_students = 400 := sorry

end accommodate_students_l48_48377


namespace sin_150_eq_half_l48_48719

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48719


namespace cell_chain_length_l48_48474

theorem cell_chain_length (d n : ℕ) (h₁ : d = 5 * 10^2) (h₂ : n = 2 * 10^3) : d * n = 10^6 :=
by
  sorry

end cell_chain_length_l48_48474


namespace tan_alpha_eq_neg_one_l48_48903

theorem tan_alpha_eq_neg_one (α : ℝ) (h1 : |Real.sin α| = |Real.cos α|)
    (h2 : π / 2 < α ∧ α < π) : Real.tan α = -1 :=
sorry

end tan_alpha_eq_neg_one_l48_48903


namespace greatest_a_for_integer_solutions_l48_48598

theorem greatest_a_for_integer_solutions :
  ∃ a : ℕ, 
    (∀ x : ℤ, x^2 + a * x = -21 → ∃ y : ℤ, y * (y + a) = -21) ∧ 
    ∀ b : ℕ, (∀ x : ℤ, x^2 + b * x = -21 → ∃ y : ℤ, y * (y + b) = -21) → b ≤ a :=
begin
  -- Proof goes here
  sorry
end

end greatest_a_for_integer_solutions_l48_48598


namespace problem1_problem2_problem3_l48_48736

noncomputable def f : ℝ → ℝ := sorry -- Define your function here satisfying the conditions

theorem problem1 (h1 : ∀ x : ℝ, f (2 - x) = f x)
                 (h2 : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x)) :
  f (-1) = 1 - Real.log 3 := sorry

theorem problem2 (h1 : ∀ x : ℝ, f (2 - x) = f x)
                 (h2 : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x)) :
  ∀ x : ℝ, f (2 - 2 * x) < f (x + 3) ↔ x ∈ Set.Ico (-1/3) 3 := sorry

theorem problem3 (h1 : ∀ x : ℝ, f (2 - x) = f x)
                 (h2 : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x))
                 (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ f x = Real.log (a / x + 2 * a)) ↔ a > 2/3 := sorry

end problem1_problem2_problem3_l48_48736


namespace smallest_integer_x_l48_48621

theorem smallest_integer_x (x : ℤ) : 
  ( ∀ x : ℤ, ( 2 * (x : ℚ) / 5 + 3 / 4 > 7 / 5 → 2 ≤ x )) :=
by
  intro x
  sorry

end smallest_integer_x_l48_48621


namespace contrapositive_l48_48593

theorem contrapositive (x y : ℝ) : (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by
  intro h
  sorry

end contrapositive_l48_48593


namespace geometric_sequence_sum_l48_48155

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 0 = 3)
(h_sum : a 0 + a 1 + a 2 = 21) (hq : ∀ n, a (n + 1) = a n * q) : a 2 + a 3 + a 4 = 84 := by
  sorry

end geometric_sequence_sum_l48_48155


namespace not_all_elements_distinct_l48_48850

open Rational

-- Define the sequence as a function from ℕ to non-negative rational numbers
def sequence (a : ℕ → ℚ) := ∀ m n : ℕ, a m + a n = a (m * n)

-- Define the proof goal: not all elements of the sequence are distinct
theorem not_all_elements_distinct (a : ℕ → ℚ) (h_seq : sequence a) : ∃ m n : ℕ, m ≠ n ∧ a m = a n :=
sorry

end not_all_elements_distinct_l48_48850


namespace distance_ratio_l48_48503

theorem distance_ratio (D90 D180 : ℝ) 
  (h1 : D90 + D180 = 3600) 
  (h2 : D90 / 90 + D180 / 180 = 30) : 
  D90 / D180 = 1 := 
by 
  sorry

end distance_ratio_l48_48503


namespace D_72_eq_81_l48_48436

-- Definition of the function for the number of decompositions
def D (n : Nat) : Nat :=
  -- D(n) would ideally be implemented here as per the given conditions
  sorry

-- Prime factorization of 72
def prime_factorization_72 : List Nat :=
  [2, 2, 2, 3, 3]

-- Statement to prove
theorem D_72_eq_81 : D 72 = 81 :=
by
  -- Placeholder for actual proof
  sorry

end D_72_eq_81_l48_48436


namespace compare_a_b_c_l48_48896

noncomputable def a : ℝ := (1 / 3)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 2)
noncomputable def c : ℝ := Real.logb (1 / 3) (1 / 4)

theorem compare_a_b_c : b < a ∧ a < c := by
  sorry

end compare_a_b_c_l48_48896


namespace number_sum_20_eq_30_l48_48611

theorem number_sum_20_eq_30 : ∃ x : ℤ, 20 + x = 30 → x = 10 :=
by {
  sorry
}

end number_sum_20_eq_30_l48_48611


namespace rope_segments_after_folds_l48_48648

theorem rope_segments_after_folds (n : ℕ) : 
  (if n = 1 then 3 else 
   if n = 2 then 5 else 
   if n = 3 then 9 else 2^n + 1) = 2^n + 1 :=
by sorry

end rope_segments_after_folds_l48_48648


namespace price_of_wheat_flour_l48_48828

theorem price_of_wheat_flour
  (initial_amount : ℕ)
  (price_rice : ℕ)
  (num_rice : ℕ)
  (price_soda : ℕ)
  (num_soda : ℕ)
  (num_wheat_flour : ℕ)
  (remaining_balance : ℕ)
  (total_spent : ℕ)
  (amount_spent_on_rice_and_soda : ℕ)
  (amount_spent_on_wheat_flour : ℕ)
  (price_per_packet_wheat_flour : ℕ) 
  (h_initial_amount : initial_amount = 500)
  (h_price_rice : price_rice = 20)
  (h_num_rice : num_rice = 2)
  (h_price_soda : price_soda = 150)
  (h_num_soda : num_soda = 1)
  (h_num_wheat_flour : num_wheat_flour = 3)
  (h_remaining_balance : remaining_balance = 235)
  (h_total_spent : total_spent = initial_amount - remaining_balance)
  (h_amount_spent_on_rice_and_soda : amount_spent_on_rice_and_soda = price_rice * num_rice + price_soda * num_soda)
  (h_amount_spent_on_wheat_flour : amount_spent_on_wheat_flour = total_spent - amount_spent_on_rice_and_soda)
  (h_price_per_packet_wheat_flour : price_per_packet_wheat_flour = amount_spent_on_wheat_flour / num_wheat_flour) :
  price_per_packet_wheat_flour = 25 :=
by 
  sorry

end price_of_wheat_flour_l48_48828


namespace correct_polynomial_and_result_l48_48478

theorem correct_polynomial_and_result :
  ∃ p q r : Polynomial ℝ,
    q = X^2 - 3 * X + 5 ∧
    p + q = 5 * X^2 - 2 * X + 4 ∧
    p = 4 * X^2 + X - 1 ∧
    r = p - q ∧
    r = 3 * X^2 + 4 * X - 6 :=
by {
  sorry
}

end correct_polynomial_and_result_l48_48478


namespace pedro_squares_correct_l48_48785

def squares_jesus : ℕ := 60
def squares_linden : ℕ := 75
def squares_pedro (s_jesus s_linden : ℕ) : ℕ := (s_jesus + s_linden) + 65

theorem pedro_squares_correct :
  squares_pedro squares_jesus squares_linden = 200 :=
by
  sorry

end pedro_squares_correct_l48_48785


namespace teachers_with_neither_percentage_l48_48851

def total_teachers : Nat := 150
def teachers_with_high_bp : Nat := 90
def teachers_with_heart_trouble : Nat := 50
def teachers_with_both : Nat := 30

theorem teachers_with_neither_percentage :
  (total_teachers - ((teachers_with_high_bp - teachers_with_both) + (teachers_with_heart_trouble - teachers_with_both) + teachers_with_both)) * 100 / total_teachers = 26.67 :=
by
  sorry

end teachers_with_neither_percentage_l48_48851


namespace base_9_units_digit_of_sum_l48_48889

def base_n_units_digit (n : ℕ) (a : ℕ) : ℕ :=
a % n

theorem base_9_units_digit_of_sum : base_n_units_digit 9 (45 + 76) = 2 :=
by
  sorry

end base_9_units_digit_of_sum_l48_48889


namespace sin_150_eq_half_l48_48672

theorem sin_150_eq_half :
  sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48672


namespace ram_marks_l48_48182

theorem ram_marks (total_marks : ℕ) (percentage : ℕ) (h_total : total_marks = 500) (h_percentage : percentage = 90) : 
  (percentage * total_marks / 100) = 450 := by
  sorry

end ram_marks_l48_48182


namespace number_of_whole_numbers_without_1_or_2_l48_48282

/-- There are 439 whole numbers between 1 and 500 that do not contain the digit 1 or 2. -/
theorem number_of_whole_numbers_without_1_or_2 : 
  ∃ n : ℕ, n = 439 ∧ ∀ m, 1 ≤ m ∧ m ≤ 500 → ∀ d ∈ (m.digits 10), d ≠ 1 ∧ d ≠ 2 :=
sorry

end number_of_whole_numbers_without_1_or_2_l48_48282


namespace cost_per_meal_is_8_l48_48650

-- Define the conditions
def number_of_adults := 2
def number_of_children := 5
def total_bill := 56
def total_people := number_of_adults + number_of_children

-- Define the cost per meal
def cost_per_meal := total_bill / total_people

-- State the theorem we want to prove
theorem cost_per_meal_is_8 : cost_per_meal = 8 := 
by
  -- The proof would go here, but we'll use sorry to skip it
  sorry

end cost_per_meal_is_8_l48_48650


namespace boxes_in_attic_l48_48327

theorem boxes_in_attic (B : ℕ)
  (h1 : 6 ≤ B)
  (h2 : ∀ T : ℕ, T = (B - 6) / 2 ∧ T = 10)
  (h3 : ∀ O : ℕ, O = 180 + 2 * T ∧ O = 20 * T) :
  B = 26 :=
by
  sorry

end boxes_in_attic_l48_48327


namespace expression_evaluation_l48_48356

theorem expression_evaluation :
  (0.15)^3 - (0.06)^3 / (0.15)^2 + 0.009 + (0.06)^2 = 0.006375 :=
by
  sorry

end expression_evaluation_l48_48356


namespace total_canoes_built_l48_48383

-- Given conditions as definitions
def a1 : ℕ := 10
def r : ℕ := 3

-- Define the geometric series sum for first four terms
noncomputable def sum_of_geometric_series (a1 r : ℕ) (n : ℕ) : ℕ :=
  a1 * ((r^n - 1) / (r - 1))

-- Prove that the total number of canoes built by the end of April is 400
theorem total_canoes_built (a1 r : ℕ) (n : ℕ) : sum_of_geometric_series a1 r n = 400 :=
  sorry

end total_canoes_built_l48_48383


namespace vans_hold_people_per_van_l48_48175

theorem vans_hold_people_per_van (students adults vans total_people people_per_van : ℤ) 
    (h1: students = 12) 
    (h2: adults = 3) 
    (h3: vans = 3) 
    (h4: total_people = students + adults) 
    (h5: people_per_van = total_people / vans) :
    people_per_van = 5 := 
by
    -- Steps will go here
    sorry

end vans_hold_people_per_van_l48_48175


namespace negation_proposition_real_l48_48808

theorem negation_proposition_real :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_proposition_real_l48_48808


namespace shifted_parabola_expression_l48_48590

theorem shifted_parabola_expression (x : ℝ) :
  let y_original := x^2
  let y_shifted_right := (x - 1)^2
  let y_shifted_up := y_shifted_right + 2
  y_shifted_up = (x - 1)^2 + 2 :=
by
  sorry

end shifted_parabola_expression_l48_48590


namespace find_train_speed_l48_48094

-- Define the given conditions
def train_length : ℕ := 2500  -- length of the train in meters
def time_to_cross_pole : ℕ := 100  -- time to cross the pole in seconds

-- Define the expected speed
def expected_speed : ℕ := 25  -- expected speed in meters per second

-- The theorem we need to prove
theorem find_train_speed : 
  (train_length / time_to_cross_pole) = expected_speed := 
by 
  sorry

end find_train_speed_l48_48094


namespace negation_of_prop_l48_48810

open Classical

theorem negation_of_prop (h : ∀ x : ℝ, x^2 + x + 1 > 0) : ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
sorry

end negation_of_prop_l48_48810


namespace curve_equation_l48_48139

theorem curve_equation
  (a b : ℝ)
  (h1 : a * 0 ^ 2 + b * (5 / 3) ^ 2 = 2)
  (h2 : a * 1 ^ 2 + b * 1 ^ 2 = 2) :
  (16 / 25) * x^2 + (9 / 25) * y^2 = 1 := 
by {
  sorry
}

end curve_equation_l48_48139


namespace find_g_3_l48_48440

theorem find_g_3 (p q r : ℝ) (g : ℝ → ℝ) (h1 : g x = p * x^7 + q * x^3 + r * x + 7) (h2 : g (-3) = -11) (h3 : ∀ x, g (x) + g (-x) = 14) : g 3 = 25 :=
by 
  sorry

end find_g_3_l48_48440


namespace trivia_team_members_l48_48107

theorem trivia_team_members (x : ℕ) (h : 3 * (x - 6) = 27) : x = 15 := 
by
  sorry

end trivia_team_members_l48_48107


namespace neg_square_positive_l48_48811

theorem neg_square_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := sorry

end neg_square_positive_l48_48811


namespace yancheng_marathon_half_marathon_estimated_probability_l48_48953

noncomputable def estimated_probability
  (surveyed_participants_frequencies : List (ℕ × Real)) : Real :=
by
  -- Define the surveyed participants and their corresponding frequencies
  -- In this example, [(20, 0.35), (50, 0.40), (100, 0.39), (200, 0.415), (500, 0.418), (2000, 0.411)]
  sorry

theorem yancheng_marathon_half_marathon_estimated_probability :
  let surveyed_participants_frequencies := [
    (20, 0.350),
    (50, 0.400),
    (100, 0.390),
    (200, 0.415),
    (500, 0.418),
    (2000, 0.411)
  ]
  estimated_probability surveyed_participants_frequencies = 0.40 :=
by
  sorry

end yancheng_marathon_half_marathon_estimated_probability_l48_48953


namespace digit_B_identification_l48_48301

theorem digit_B_identification (B : ℕ) 
  (hB_range : 0 ≤ B ∧ B < 10) 
  (h_units_digit : (5 * B % 10) = 5) 
  (h_product : (10 * B + 5) * (90 + B) = 9045) : 
  B = 9 :=
sorry

end digit_B_identification_l48_48301


namespace range_of_a_l48_48899

theorem range_of_a (a : ℝ) (x : ℝ) : (x^2 + 2*x > 3) → (x > a) → (¬ (x^2 + 2*x > 3) → ¬ (x > a)) → a ≥ 1 :=
by
  intros hp hq hr
  sorry

end range_of_a_l48_48899


namespace total_items_to_buy_l48_48827

theorem total_items_to_buy (total_money : ℝ) (cost_sandwich : ℝ) (cost_drink : ℝ) (num_items : ℕ) :
  total_money = 30 → cost_sandwich = 4.5 → cost_drink = 1 → num_items = 9 :=
by
  sorry

end total_items_to_buy_l48_48827


namespace inequality_solution_l48_48238

theorem inequality_solution (x : ℝ) : x > 0 ∧ (x^(1/3) < 3 - x) ↔ x < 3 :=
by 
  sorry

end inequality_solution_l48_48238


namespace at_least_one_not_less_than_l48_48901

variables {A B C D a b c : ℝ}

theorem at_least_one_not_less_than :
  (a = A * C) →
  (b = A * D + B * C) →
  (c = B * D) →
  (a + b + c = (A + B) * (C + D)) →
  a ≥ (4 * (A + B) * (C + D) / 9) ∨ b ≥ (4 * (A + B) * (C + D) / 9) ∨ c ≥ (4 * (A + B) * (C + D) / 9) :=
by
  intro h1 h2 h3 h4
  sorry

end at_least_one_not_less_than_l48_48901


namespace determine_b_when_lines_parallel_l48_48393

theorem determine_b_when_lines_parallel (b : ℝ) : 
  (∀ x y, 3 * y - 3 * b = 9 * x ↔ y - 2 = (b + 9) * x) → b = -6 :=
by
  sorry

end determine_b_when_lines_parallel_l48_48393


namespace probability_arithmetic_sequence_l48_48906

open Finset

def combinations {α : Type} [DecidableEq α] (s : Finset α) (k : ℕ) : Finset (Finset α) :=
  (Fintype.piFinset (λ _ : Finₓ k, s)).filter (λ t, t.card = k)

theorem probability_arithmetic_sequence :
  let s := {1, 2, 3, 4, 5, 6}
  let n := (combinations s 3).card
  let favorable := { {1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}, {1, 3, 5}, {2, 4, 6} }
  (n = 20) → (favorable.card = 6) → ((favorable.card : ℚ) / n = 3 / 10) := 
by
  intros s n favorable h1 h2
  sorry

end probability_arithmetic_sequence_l48_48906


namespace product_of_fractional_parts_eq_222_l48_48214

theorem product_of_fractional_parts_eq_222 : 
  let x := 18 / 999 in let y := x.num / x.denom in y.num * y.denom = 222 :=
by 
  sorry

end product_of_fractional_parts_eq_222_l48_48214


namespace determine_b_when_lines_parallel_l48_48391

theorem determine_b_when_lines_parallel (b : ℝ) : 
  (∀ x y, 3 * y - 3 * b = 9 * x ↔ y - 2 = (b + 9) * x) → b = -6 :=
by
  sorry

end determine_b_when_lines_parallel_l48_48391


namespace opposite_of_negative_fraction_l48_48814

theorem opposite_of_negative_fraction : -(- (1/2023 : ℚ)) = 1/2023 := 
sorry

end opposite_of_negative_fraction_l48_48814


namespace min_value_x_l48_48544

theorem min_value_x (a b x : ℝ) (ha : 0 < a) (hb : 0 < b)
(hcond : 4 * a + b * (1 - a) = 0)
(hineq : ∀ a b, 0 < a → 0 < b → 4 * a + b * (1 - a) = 0 → (1 / (a ^ 2) + 16 / (b ^ 2) ≥ 1 + x / 2 - x ^ 2)) :
  x = 1 :=
sorry

end min_value_x_l48_48544


namespace range_of_a_l48_48555

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x > x ^ 2 + a) → a < -8 :=
by sorry

end range_of_a_l48_48555


namespace Fedya_third_l48_48581

/-- Definitions for order of children's arrival -/
inductive Child
| Roman | Fedya | Liza | Katya | Andrew

open Child

def arrival_order (order : Child → ℕ) : Prop :=
  order Liza > order Roman ∧
  order Katya < order Liza ∧
  order Fedya = order Katya + 1 ∧
  order Katya ≠ 1

/-- Theorem stating that Fedya is third based on the given conditions -/
theorem Fedya_third (order : Child → ℕ) (H : arrival_order order) : order Fedya = 3 :=
sorry

end Fedya_third_l48_48581


namespace speed_for_remaining_distance_l48_48177

theorem speed_for_remaining_distance
  (t_total : ℝ) (v1 : ℝ) (d_total : ℝ)
  (t_total_def : t_total = 1.4)
  (v1_def : v1 = 4)
  (d_total_def : d_total = 5.999999999999999) :
  ∃ v2 : ℝ, v2 = 5 := 
by
  sorry

end speed_for_remaining_distance_l48_48177


namespace min_dot_product_PA_PB_l48_48740

noncomputable def point_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
noncomputable def point_on_ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem min_dot_product_PA_PB (A B P : ℝ × ℝ)
  (hA : point_on_circle A.1 A.2)
  (hB : point_on_circle B.1 B.2)
  (hAB : A ≠ B ∧ (B.1 = -A.1) ∧ (B.2 = -A.2))
  (hP : point_on_ellipse P.1 P.2) :
  ∃ PA PB : ℝ × ℝ, 
    PA = (P.1 - A.1, P.2 - A.2) ∧ PB = (P.1 - B.1, P.2 - B.2) ∧
    (PA.1 * PB.1 + PA.2 * PB.2) = 2 :=
by sorry

end min_dot_product_PA_PB_l48_48740


namespace integer_solutions_eq_l48_48126

theorem integer_solutions_eq (x y z : ℤ) :
  (x + y + z) ^ 5 = 80 * x * y * z * (x ^ 2 + y ^ 2 + z ^ 2) ↔
  ∃ a : ℤ, (x = a ∧ y = -a ∧ z = 0) ∨ (x = -a ∧ y = a ∧ z = 0) ∨ (x = a ∧ y = 0 ∧ z = -a) ∨ (x = -a ∧ y = 0 ∧ z = a) ∨ (x = 0 ∧ y = a ∧ z = -a) ∨ (x = 0 ∧ y = -a ∧ z = a) :=
by sorry

end integer_solutions_eq_l48_48126


namespace original_plan_months_l48_48640

theorem original_plan_months (x : ℝ) (h : 1 / (x - 6) = 1.4 * (1 / x)) : x = 21 :=
by
  sorry

end original_plan_months_l48_48640


namespace boys_play_theater_with_Ocho_l48_48052

variables (Ocho_friends : ℕ) (half_girls : Ocho_friends / 2 = 4)

theorem boys_play_theater_with_Ocho : (Ocho_friends / 2) = 4 := by
  -- Ocho_friends is the total number of Ocho's friends
  -- half_girls is given as a condition that half of Ocho's friends are girls
  -- thus, we directly use this to conclude that the number of boys is 4
  sorry

end boys_play_theater_with_Ocho_l48_48052


namespace necessary_but_not_sufficient_condition_geometric_sequence_l48_48338

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * a (n - 1) / a n 

theorem necessary_but_not_sufficient_condition_geometric_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2) :
  (is_geometric_sequence a → (∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2)) ∧ (∃ b : ℕ → ℝ, (b n = 0 ∨ b n = b (n - 1) ∨ b n = b (n + 1)) ∧ ¬ is_geometric_sequence b) := 
sorry

end necessary_but_not_sufficient_condition_geometric_sequence_l48_48338


namespace red_ball_return_probability_l48_48772

/-- Let A_total be the total number of balls in box A, initially consisting of 1 red and 5 white. 
    Let B_total be the total number of white balls in box B initially.
    We randomly select 3 balls from box A to box B, and after mixing, select 3 balls from box B to box A.
    This theorem proves that the probability of the red ball being transferred from A to B and back to A is 0.25. -/
theorem red_ball_return_probability (A_total : ℕ) (B_total : ℕ) 
  (A_total = 6) (B_total = 3) : 
  let combinations (n k : ℕ) := Nat.choose n k in
  let total_ways_to_choose_3_from_6 := combinations A_total 3 in
  let successful_ways :=
    (combinations (A_total - 1) 2) * (combinations (A_total - 1) 2) in
  let probability := successful_ways / (total_ways_to_choose_3 * total_ways_to_choose_3) in
  probability = 0.25 :=
by
  sorry

end red_ball_return_probability_l48_48772


namespace quadratic_roots_l48_48996

theorem quadratic_roots (A B C : ℝ) (r s p : ℝ) (h1 : 2 * A * r^2 + 3 * B * r + 4 * C = 0)
  (h2 : 2 * A * s^2 + 3 * B * s + 4 * C = 0) (h3 : r + s = -3 * B / (2 * A)) (h4 : r * s = 2 * C / A) :
  p = (16 * A * C - 9 * B^2) / (4 * A^2) :=
by
  sorry

end quadratic_roots_l48_48996


namespace complex_division_l48_48535

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (2 / (1 + i)) = (1 - i) :=
by
  sorry

end complex_division_l48_48535


namespace average_class_score_l48_48152

theorem average_class_score (total_students assigned_day_students make_up_date_students : ℕ)
  (assigned_day_percentage make_up_date_percentage assigned_day_avg_score make_up_date_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_percentage = 0.70)
  (h3 : make_up_date_percentage = 0.30)
  (h4 : assigned_day_students = 70)
  (h5 : make_up_date_students = 30)
  (h6 : assigned_day_avg_score = 55)
  (h7 : make_up_date_avg_score = 95) :
  (assigned_day_avg_score * assigned_day_students + make_up_date_avg_score * make_up_date_students) / total_students = 67 :=
by
  sorry

end average_class_score_l48_48152


namespace time_per_potato_l48_48994

-- Definitions from the conditions
def total_potatoes : ℕ := 12
def cooked_potatoes : ℕ := 6
def remaining_potatoes : ℕ := total_potatoes - cooked_potatoes
def total_time : ℕ := 36
def remaining_time_per_potato : ℕ := total_time / remaining_potatoes

-- Theorem to be proved
theorem time_per_potato : remaining_time_per_potato = 6 := by
  sorry

end time_per_potato_l48_48994


namespace part1_part2_l48_48905

-- Define the first part of the problem
theorem part1 (a b : ℝ) :
  (∀ x : ℝ, |x^2 + a * x + b| ≤ 2 * |x - 4| * |x + 2|) → (a = -2 ∧ b = -8) :=
sorry

-- Define the second part of the problem
theorem part2 (a b m : ℝ) :
  (∀ x : ℝ, x > 1 → x^2 + a * x + b ≥ (m + 2) * x - m - 15) → m ≤ 2 :=
sorry

end part1_part2_l48_48905


namespace polynomial_value_l48_48143

-- Define the conditions as Lean definitions
def condition (x : ℝ) : Prop := x^2 + 2 * x + 1 = 4

-- State the theorem to be proved
theorem polynomial_value (x : ℝ) (h : condition x) : 2 * x^2 + 4 * x + 5 = 11 :=
by
  -- Proof goes here
  sorry

end polynomial_value_l48_48143


namespace joan_has_6_balloons_l48_48929

theorem joan_has_6_balloons (initial_balloons : ℕ) (lost_balloons : ℕ) (h1 : initial_balloons = 8) (h2 : lost_balloons = 2) : initial_balloons - lost_balloons = 6 :=
sorry

end joan_has_6_balloons_l48_48929


namespace equal_piles_l48_48065

theorem equal_piles (initial_rocks final_piles : ℕ) (moves : ℕ) (total_rocks : ℕ) (rocks_per_pile : ℕ) :
  initial_rocks = 36 →
  final_piles = 7 →
  moves = final_piles - 1 →
  total_rocks = initial_rocks + moves →
  rocks_per_pile = total_rocks / final_piles →
  rocks_per_pile = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end equal_piles_l48_48065


namespace Vasya_not_11_more_than_Kolya_l48_48303

def is_L_shaped (n : ℕ) : Prop :=
  n % 2 = 1

def total_cells : ℕ :=
  14400

theorem Vasya_not_11_more_than_Kolya (k v : ℕ) :
  (is_L_shaped k) → (is_L_shaped v) → (k + v = total_cells) → (k % 2 = 0) → (v % 2 = 0) → (v - k ≠ 11) := 
by
  sorry

end Vasya_not_11_more_than_Kolya_l48_48303


namespace parallel_lines_slope_eq_l48_48397

theorem parallel_lines_slope_eq (b : ℝ) :
    (∀ x y : ℝ, 3 * y - 3 * b = 9 * x → ∀ x' y' : ℝ, y' - 2 = (b + 9) * x' → 3 = b + 9) →
    b = -6 := 
by 
  intros h
  have h1 : 3 = b + 9 := sorry -- proof omitted
  rw h1
  norm_num

end parallel_lines_slope_eq_l48_48397


namespace problem_statement_l48_48171

def f (x : ℤ) : ℤ := 3*x + 4
def g (x : ℤ) : ℤ := 4*x - 3

theorem problem_statement : (f (g (f 2))) / (g (f (g 2))) = 115 / 73 :=
by
  sorry

end problem_statement_l48_48171


namespace rohan_house_rent_percentage_l48_48451

variable (salary savings food entertainment conveyance : ℕ)
variable (spend_on_house : ℚ)

-- Given conditions
axiom h1 : salary = 5000
axiom h2 : savings = 1000
axiom h3 : food = 40
axiom h4 : entertainment = 10
axiom h5 : conveyance = 10

-- Define savings percentage
def savings_percentage (salary savings : ℕ) : ℚ := (savings : ℚ) / salary * 100

-- Define percentage equation
def total_percentage (food entertainment conveyance spend_on_house savings_percentage : ℚ) : ℚ :=
  food + spend_on_house + entertainment + conveyance + savings_percentage

-- Prove that house rent percentage is 20%
theorem rohan_house_rent_percentage : 
  food = 40 → entertainment = 10 → conveyance = 10 → salary = 5000 → savings = 1000 → 
  total_percentage 40 10 10 spend_on_house (savings_percentage 5000 1000) = 100 →
  spend_on_house = 20 := by
  intros
  sorry

end rohan_house_rent_percentage_l48_48451


namespace problem_I_problem_II_l48_48988

-- Question I
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 :=
by
  sorry

-- Question II
theorem problem_II (a : ℝ) : (∀ x : ℝ, |x - a| + |2 * x - 1| ≥ 2) ↔ (a ≤ -3/2 ∨ a ≥ 5/2) :=
by
  sorry

end problem_I_problem_II_l48_48988


namespace maria_travel_fraction_before_first_stop_l48_48259

theorem maria_travel_fraction_before_first_stop (D : ℕ) (x : ℚ) :
  D = 480 ∧ 
  (1 - 4 * x) * 480 / 4 + x * 480 = 300 →
  x = 1 / 2 :=
by
  intros hD hx
  sorry

end maria_travel_fraction_before_first_stop_l48_48259


namespace right_triangle_one_leg_div_by_3_l48_48179

theorem right_triangle_one_leg_div_by_3 {a b c : ℕ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (h : a^2 + b^2 = c^2) : 3 ∣ a ∨ 3 ∣ b := 
by 
  sorry

end right_triangle_one_leg_div_by_3_l48_48179


namespace ambulance_ride_cost_l48_48776

noncomputable def hospital_bill
  (total_bill : ℝ)
  (medication_percentage : ℝ)
  (overnight_percentage : ℝ)
  (food_cost : ℝ) : ℝ :=
  let medication_cost := medication_percentage * total_bill in
  let remaining_after_medication := total_bill - medication_cost in
  let overnight_cost := overnight_percentage * remaining_after_medication in
  remaining_after_medication - overnight_cost - food_cost

theorem ambulance_ride_cost
  (total_bill : ℝ)
  (medication_percentage : ℝ)
  (overnight_percentage : ℝ)
  (food_cost : ℝ)
  (ambulance_cost : ℝ)
  (h : total_bill = 5000)
  (h_medication : medication_percentage = 0.50)
  (h_overnight : overnight_percentage = 0.25)
  (h_food : food_cost = 175)
  (h_ambulance : ambulance_cost = 1700) :
  hospital_bill total_bill medication_percentage overnight_percentage food_cost = ambulance_cost := by
  sorry

end ambulance_ride_cost_l48_48776


namespace simplify_expression_l48_48862

variable (x : ℝ)

theorem simplify_expression : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3 - 3 * x^3) 
  = (-x^3 - x^2 + 23 * x - 3) :=
by
  sorry

end simplify_expression_l48_48862


namespace real_return_l48_48942

theorem real_return (n i r: ℝ) (h₁ : n = 0.21) (h₂ : i = 0.10) : 
  (1 + r) = (1 + n) / (1 + i) → r = 0.10 :=
by
  intro h₃
  sorry

end real_return_l48_48942


namespace ratio_P_S_l48_48958

theorem ratio_P_S (S N P : ℝ) 
  (hN : N = S / 4) 
  (hP : P = N / 4) : 
  P / S = 1 / 16 := 
by 
  sorry

end ratio_P_S_l48_48958


namespace urn_gold_coins_percentage_l48_48110

noncomputable def percentage_gold_coins_in_urn
  (total_objects : ℕ)
  (beads_percentage : ℝ)
  (rings_percentage : ℝ)
  (coins_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  : ℝ := 
  let gold_coins_percentage := 100 - silver_coins_percentage
  let coins_total_percentage := total_objects * coins_percentage / 100
  coins_total_percentage * gold_coins_percentage / 100

theorem urn_gold_coins_percentage 
  (total_objects : ℕ)
  (beads_percentage rings_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  (h1 : beads_percentage = 15)
  (h2 : rings_percentage = 15)
  (h3 : beads_percentage + rings_percentage = 30)
  (h4 : coins_percentage = 100 - 30)
  (h5 : silver_coins_percentage = 35)
  : percentage_gold_coins_in_urn total_objects beads_percentage rings_percentage (100 - 30) 35 = 45.5 :=
sorry

end urn_gold_coins_percentage_l48_48110


namespace other_type_jelly_amount_l48_48793

-- Combined total amount of jelly
def total_jelly := 6310

-- Amount of one type of jelly
def type_one_jelly := 4518

-- Amount of the other type of jelly
def type_other_jelly := total_jelly - type_one_jelly

theorem other_type_jelly_amount :
  type_other_jelly = 1792 :=
by
  sorry

end other_type_jelly_amount_l48_48793


namespace min_value_fraction_l48_48536

noncomputable section

open Real

theorem min_value_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 4) : 
  ∃ t : ℝ, (∀ x' y' : ℝ, (x' > 0 ∧ y' > 0 ∧ x' + 2 * y' = 4) → (2 / x' + 1 / y') ≥ t) ∧ t = 2 :=
by
  sorry

end min_value_fraction_l48_48536


namespace sum_sequence_l48_48770

theorem sum_sequence (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = -2/3)
  (h2 : ∀ n, n ≥ 2 → S n = -1 / (S (n - 1) + 2)) :
  ∀ n, S n = -(n + 1) / (n + 2) := 
by 
  sorry

end sum_sequence_l48_48770


namespace initial_students_count_l48_48233

variable (initial_students : ℕ)
variable (number_of_new_boys : ℕ := 5)
variable (initial_percentage_girls : ℝ := 0.40)
variable (new_percentage_girls : ℝ := 0.32)

theorem initial_students_count (h : initial_percentage_girls * initial_students = new_percentage_girls * (initial_students + number_of_new_boys)) : 
  initial_students = 20 := 
by 
  sorry

end initial_students_count_l48_48233


namespace hexagon_ratio_l48_48371

theorem hexagon_ratio (A B : ℝ) (h₁ : A = 8) (h₂ : B = 2)
                      (A_above : ℝ) (h₃ : A_above = (3 + B))
                      (H : 3 + B = 1 / 2 * (A + B)) 
                      (XQ QY : ℝ) (h₄ : XQ + QY = 4)
                      (h₅ : 3 + B = 4 + B / 2) :
  XQ / QY = 2 := 
by
  sorry

end hexagon_ratio_l48_48371


namespace geometric_sequence_arithmetic_Sn_l48_48933

theorem geometric_sequence_arithmetic_Sn (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (n : ℕ) :
  (∀ n, a n = a1 * q ^ (n - 1)) →
  (∀ n, S n = a1 * (1 - q ^ n) / (1 - q)) →
  (∀ n, S (n + 1) - S n = S n - S (n - 1)) →
  q = 1 :=
by
  sorry

end geometric_sequence_arithmetic_Sn_l48_48933


namespace find_k_value_l48_48891

theorem find_k_value
  (k : ℤ)
  (h : 3 * 2^2001 - 3 * 2^2000 - 2^1999 + 2^1998 = k * 2^1998) : k = 11 :=
by
  sorry

end find_k_value_l48_48891


namespace kelcie_books_multiple_l48_48939

theorem kelcie_books_multiple (x : ℕ) :
  let megan_books := 32
  let kelcie_books := megan_books / 4
  let greg_books := x * kelcie_books + 9
  let total_books := megan_books + kelcie_books + greg_books
  total_books = 65 → x = 2 :=
by
  intros megan_books kelcie_books greg_books total_books h
  sorry

end kelcie_books_multiple_l48_48939


namespace symmetric_points_sum_l48_48921

variable {p q : ℤ}

theorem symmetric_points_sum (h1 : p = -6) (h2 : q = 2) : p + q = -4 := by
  sorry

end symmetric_points_sum_l48_48921


namespace jars_of_peanut_butter_l48_48783

theorem jars_of_peanut_butter (x : Nat) : 
  (16 * x + 28 * x + 40 * x + 52 * x = 2032) → 
  (4 * x = 60) :=
by
  intro h
  sorry

end jars_of_peanut_butter_l48_48783


namespace simplify_expression_1_simplify_expression_2_l48_48453

-- Define the algebraic simplification problem for the first expression
theorem simplify_expression_1 (x y : ℝ) : 5 * x - 3 * (2 * x - 3 * y) + x = 9 * y :=
by
  sorry

-- Define the algebraic simplification problem for the second expression
theorem simplify_expression_2 (a : ℝ) : 3 * a^2 + 5 - 2 * a^2 - 2 * a + 3 * a - 8 = a^2 + a - 3 :=
by
  sorry

end simplify_expression_1_simplify_expression_2_l48_48453


namespace polynomial_roots_l48_48209

variable (AT TB m n : ℝ)

-- conditions
axiom sum_condition : AT + TB = m
axiom product_condition : AT * TB = n^2

-- statement
theorem polynomial_roots :
  Polynomial.X^2 - m * Polynomial.X + n^2 =
  Polynomial.of_roots [AT, TB] := by
  sorry

end polynomial_roots_l48_48209


namespace average_runs_l48_48023

theorem average_runs (games : ℕ) (runs1 matches1 runs2 matches2 runs3 matches3 : ℕ)
  (h1 : runs1 = 1) 
  (h2 : matches1 = 1) 
  (h3 : runs2 = 4) 
  (h4 : matches2 = 2)
  (h5 : runs3 = 5) 
  (h6 : matches3 = 3) 
  (h_games : games = matches1 + matches2 + matches3) :
  (runs1 * matches1 + runs2 * matches2 + runs3 * matches3) / games = 4 :=
by
  sorry

end average_runs_l48_48023


namespace description_of_T_l48_48437

def T (x y : ℝ) : Prop :=
  (5 = x+3 ∧ y-6 ≤ 5) ∨
  (5 = y-6 ∧ x+3 ≤ 5) ∨
  ((x+3 = y-6) ∧ 5 ≤ x+3)

theorem description_of_T :
  ∀ (x y : ℝ), T x y ↔ (x = 2 ∧ y ≤ 11) ∨ (y = 11 ∧ x ≤ 2) ∨ (y = x + 9 ∧ x ≥ 2) :=
sorry

end description_of_T_l48_48437


namespace no_triangle_satisfies_condition_l48_48123

theorem no_triangle_satisfies_condition (x y z : ℝ) (h_tri : x + y > z ∧ x + z > y ∧ y + z > x) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
by
  sorry

end no_triangle_satisfies_condition_l48_48123


namespace simplify_and_evaluate_l48_48329

theorem simplify_and_evaluate (x : ℝ) (h : x = 2 + Real.sqrt 2) :
  (1 - 3 / (x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = Real.sqrt 2 / 2 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l48_48329


namespace negation_of_every_student_is_punctual_l48_48463

variable (Student : Type) (student punctual : Student → Prop)

theorem negation_of_every_student_is_punctual :
  ¬ (∀ x, student x → punctual x) ↔ ∃ x, student x ∧ ¬ punctual x := by
sorry

end negation_of_every_student_is_punctual_l48_48463


namespace a2b2_div_ab1_is_square_l48_48267

theorem a2b2_div_ab1_is_square (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, (a^2 + b^2) / (ab + 1) = k^2 :=
sorry

end a2b2_div_ab1_is_square_l48_48267


namespace find_value_of_10n_l48_48749

theorem find_value_of_10n (n : ℝ) (h : 2 * n = 14) : 10 * n = 70 :=
sorry

end find_value_of_10n_l48_48749


namespace sin_150_eq_half_l48_48665

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by sorry

end sin_150_eq_half_l48_48665


namespace total_education_duration_l48_48798

-- Definitions from the conditions
def high_school_duration : ℕ := 4 - 1
def tertiary_education_duration : ℕ := 3 * high_school_duration

-- The theorem statement
theorem total_education_duration : high_school_duration + tertiary_education_duration = 12 :=
by
  sorry

end total_education_duration_l48_48798


namespace randy_total_trees_l48_48788

theorem randy_total_trees (mango_trees : ℕ) (coconut_trees : ℕ) 
  (h1 : mango_trees = 60) 
  (h2 : coconut_trees = (mango_trees / 2) - 5) : 
  mango_trees + coconut_trees = 85 :=
by
  sorry

end randy_total_trees_l48_48788


namespace repeating_decimal_product_of_num_and_den_l48_48216

theorem repeating_decimal_product_of_num_and_den (x : ℚ) (h : x = 18 / 999) (h_simplified : x.num * x.den = 222) : x.num * x.den = 222 :=
by {
  sorry
}

end repeating_decimal_product_of_num_and_den_l48_48216


namespace geom_seq_sum_l48_48561

theorem geom_seq_sum (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 + a 2 = 16) 
  (h2 : a 3 + a 4 = 24) 
  (h_geom : ∀ n, a (n+1) = r * a n):
  a 7 + a 8 = 54 :=
sorry

end geom_seq_sum_l48_48561


namespace evaluate_expression_l48_48874

/-- Given conditions: -/
def a : ℕ := 3998
def b : ℕ := 3999

theorem evaluate_expression :
  b^3 - 2 * a * b^2 - 2 * a^2 * b + (b - 2)^3 = 95806315 :=
  sorry

end evaluate_expression_l48_48874


namespace complex_power_sum_eq_five_l48_48571

noncomputable def w : ℂ := sorry

theorem complex_power_sum_eq_five (h : w^3 + w^2 + 1 = 0) : 
  w^100 + w^101 + w^102 + w^103 + w^104 = 5 :=
sorry

end complex_power_sum_eq_five_l48_48571


namespace parabola_intersection_l48_48210

noncomputable def parabola_intersection_probability : ℚ :=
  let choices := {-3, -2, -1, 0, 1, 2}
  let probs := {p : set (ℤ × ℤ × ℤ × ℤ) | 
    ∃ a b c d, 
      a ∈ choices ∧ b ∈ choices ∧ c ∈ choices ∧ d ∈ choices ∧ 
        (a ≠ c ∨ d = b)} 
  (probs.card : ℚ) / (choices.card ^ 4 : ℚ)

theorem parabola_intersection : 
  parabola_intersection_probability = 31/36 :=
sorry

end parabola_intersection_l48_48210


namespace balance_scale_comparison_l48_48620

theorem balance_scale_comparison :
  (4 / 3) * Real.pi * (8 : ℝ)^3 > (4 / 3) * Real.pi * (3 : ℝ)^3 + (4 / 3) * Real.pi * (5 : ℝ)^3 :=
by
  sorry

end balance_scale_comparison_l48_48620


namespace sum_cubed_identity_l48_48332

theorem sum_cubed_identity
  (p q r : ℝ)
  (h1 : p + q + r = 5)
  (h2 : pq + pr + qr = 7)
  (h3 : pqr = -10) :
  p^3 + q^3 + r^3 = -10 := 
by
  sorry

end sum_cubed_identity_l48_48332


namespace sum_of_xyz_l48_48537

theorem sum_of_xyz (x y z : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : z > 0)
  (h4 : x^2 + y^2 + x * y = 3)
  (h5 : y^2 + z^2 + y * z = 4)
  (h6 : z^2 + x^2 + z * x = 7) :
  x + y + z = Real.sqrt 13 :=
by sorry -- Proof omitted, but the statement formulation is complete and checks the equality under given conditions.

end sum_of_xyz_l48_48537


namespace b_investment_months_after_a_l48_48642

-- Definitions based on the conditions
def a_investment : ℕ := 100
def b_investment : ℕ := 200
def total_yearly_investment_period : ℕ := 12
def total_profit : ℕ := 100
def a_share_of_profit : ℕ := 50
def x (x_val : ℕ) : Prop := x_val = 6

-- Main theorem to prove
theorem b_investment_months_after_a (x_val : ℕ) 
  (h1 : a_investment = 100)
  (h2 : b_investment = 200)
  (h3 : total_yearly_investment_period = 12)
  (h4 : total_profit = 100)
  (h5 : a_share_of_profit = 50) :
  (100 * total_yearly_investment_period) = 200 * (total_yearly_investment_period - x_val) → 
  x x_val := 
by
  sorry

end b_investment_months_after_a_l48_48642


namespace Q_root_l48_48878

def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem Q_root : Q (3^(1 / 3 : ℝ) + 2) = 0 := sorry

end Q_root_l48_48878


namespace square_perimeter_ratio_l48_48064

theorem square_perimeter_ratio (a₁ a₂ s₁ s₂ : ℝ) 
  (h₁ : a₁ / a₂ = 16 / 25)
  (h₂ : a₁ = s₁^2)
  (h₃ : a₂ = s₂^2) :
  (4 : ℝ) / 5 = s₁ / s₂ :=
by sorry

end square_perimeter_ratio_l48_48064


namespace r_iterated_six_times_l48_48444

def r (θ : ℚ) : ℚ := 1 / (1 - 2 * θ)

theorem r_iterated_six_times (θ : ℚ) : r (r (r (r (r (r θ))))) = θ :=
by sorry

example : r (r (r (r (r (r 10))))) = 10 :=
by rw [r_iterated_six_times 10]

end r_iterated_six_times_l48_48444


namespace rectangle_area_is_correct_l48_48374

-- Define the conditions
def length : ℕ := 135
def breadth (l : ℕ) : ℕ := l / 3

-- Define the area of the rectangle
def area (l b : ℕ) : ℕ := l * b

-- The statement to prove
theorem rectangle_area_is_correct : area length (breadth length) = 6075 := by
  -- Proof goes here, this is just the statement
  sorry

end rectangle_area_is_correct_l48_48374


namespace point_in_second_quadrant_coordinates_l48_48553

variable (x y : ℝ)
variable (P : ℝ × ℝ)
variable (h1 : P.1 = x)
variable (h2 : P.2 = y)

def isInSecondQuadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distanceToXAxis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distanceToYAxis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem point_in_second_quadrant_coordinates (h1 : isInSecondQuadrant P)
    (h2 : distanceToXAxis P = 2)
    (h3 : distanceToYAxis P = 1) :
    P = (-1, 2) :=
by 
  sorry

end point_in_second_quadrant_coordinates_l48_48553


namespace grapes_purchased_l48_48244

-- Define the given conditions
def price_per_kg_grapes : ℕ := 68
def kg_mangoes : ℕ := 9
def price_per_kg_mangoes : ℕ := 48
def total_paid : ℕ := 908

-- Define the proof problem
theorem grapes_purchased : ∃ (G : ℕ), (price_per_kg_grapes * G + price_per_kg_mangoes * kg_mangoes = total_paid) ∧ (G = 7) :=
by {
  use 7,
  sorry
}

end grapes_purchased_l48_48244


namespace number_of_16_member_event_committees_l48_48157

theorem number_of_16_member_event_committees : 
  (∃ (teams : Fin 5 → Finset (Fin 8)), 
    ∀ t, (Finset.card (teams t) = 8) ∧
    finset.card 
    { committee : Finset (Fin 40) // 
      (∀ (t : Fin 5), 3 ≤ committee.card ∧ (∀ x, x ∈ committee → ((x.val.div 8) = t) → (teams t).card - (if t = t then 4 else 0))) 
    } = 3443073600) :=
by sorry

end number_of_16_member_event_committees_l48_48157


namespace base6_sum_l48_48262

-- Define each of the numbers in base 6
def base6_555 : ℕ := 5 * 6^2 + 5 * 6^1 + 5 * 6^0
def base6_55 : ℕ := 5 * 6^1 + 5 * 6^0
def base6_5 : ℕ := 5 * 6^0
def base6_1103 : ℕ := 1 * 6^3 + 1 * 6^2 + 0 * 6^1 + 3 * 6^0 

-- The problem statement is to prove the sum equals the expected result in base 6
theorem base6_sum : base6_555 + base6_55 + base6_5 = base6_1103 :=
by
  sorry

end base6_sum_l48_48262


namespace value_of_expression_l48_48731

theorem value_of_expression (x y : ℝ) (h1 : x + y = 3) (h2 : x^2 + y^2 - x * y = 4) : 
  x^4 + y^4 + x^3 * y + x * y^3 = 36 :=
by
  sorry

end value_of_expression_l48_48731


namespace pair_B_equal_l48_48381

theorem pair_B_equal : (∀ x : ℝ, 4 * x^4 = |x|) :=
by sorry

end pair_B_equal_l48_48381


namespace find_common_difference_l48_48515

theorem find_common_difference 
  (a : ℕ → ℝ)
  (a1 : a 1 = 5)
  (a25 : a 25 = 173)
  (h : ∀ n : ℕ, a (n+1) = a 1 + n * (a 2 - a 1)) : 
  a 2 - a 1 = 7 :=
by 
  sorry

end find_common_difference_l48_48515


namespace common_difference_of_arithmetic_seq_l48_48416

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, (m - n = 1) → (a (m + 1) - a m) = (a (n + 1) - a n)

/-- The common difference of an arithmetic sequence given certain conditions. -/
theorem common_difference_of_arithmetic_seq (a: ℕ → ℤ) (d : ℤ):
    a 1 + a 2 = 4 → 
    a 3 + a 4 = 16 →
    arithmetic_sequence a →
    (a 2 - a 1) = d → d = 3 :=
by
  intros h1 h2 h3 h4
  -- Proof to be filled in here
  sorry

end common_difference_of_arithmetic_seq_l48_48416


namespace gcd_markers_l48_48047

variable (n1 n2 n3 : ℕ)

-- Let the markers Mary, Luis, and Ali bought be represented by n1, n2, and n3
def MaryMarkers : ℕ := 36
def LuisMarkers : ℕ := 45
def AliMarkers : ℕ := 75

theorem gcd_markers : Nat.gcd (Nat.gcd MaryMarkers LuisMarkers) AliMarkers = 3 := by
  sorry

end gcd_markers_l48_48047


namespace minimum_value_expression_l48_48087

theorem minimum_value_expression (x y : ℝ) : ∃ (x y : ℝ), x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = 0 :=
by
  use 4, -3
  split
  · rfl
  split
  · rfl
  calc
    4^2 + (-3)^2 - 8 * 4 + 6 * (-3) + 25
      = 16 + 9 - 32 - 18 + 25 : by norm_num
  ... = 0 : by norm_num
  done

end minimum_value_expression_l48_48087


namespace Kim_sales_on_Friday_l48_48433

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

end Kim_sales_on_Friday_l48_48433


namespace min_value_x_l48_48543

theorem min_value_x (a b x : ℝ) (ha : 0 < a) (hb : 0 < b)
(hcond : 4 * a + b * (1 - a) = 0)
(hineq : ∀ a b, 0 < a → 0 < b → 4 * a + b * (1 - a) = 0 → (1 / (a ^ 2) + 16 / (b ^ 2) ≥ 1 + x / 2 - x ^ 2)) :
  x = 1 :=
sorry

end min_value_x_l48_48543


namespace dorothy_score_l48_48061

theorem dorothy_score (T I D : ℝ) 
  (hT : T = 2 * I)
  (hI : I = (3 / 5) * D)
  (hSum : T + I + D = 252) : 
  D = 90 := 
by {
  sorry
}

end dorothy_score_l48_48061


namespace find_number_l48_48231

theorem find_number (x : ℝ) (h : 0.40 * x = 130 + 190) : x = 800 :=
by {
  -- The proof will go here
  sorry
}

end find_number_l48_48231


namespace multiple_of_1897_l48_48522

theorem multiple_of_1897 (n : ℕ) : ∃ k : ℤ, 2903^n - 803^n - 464^n + 261^n = k * 1897 := by
  sorry

end multiple_of_1897_l48_48522


namespace mean_correct_and_no_seven_l48_48250

-- Define the set of numbers.
def numbers : List ℕ := 
  [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

-- Define the arithmetic mean of the numbers in the set.
def arithmetic_mean (l : List ℕ) : ℕ := (l.sum / l.length)

-- Specify the mean value
def mean_value : ℕ := 109629012

-- State the theorem that the mean value is correct and does not contain the digit 7.
theorem mean_correct_and_no_seven : arithmetic_mean numbers = mean_value ∧ ¬ 7 ∈ (mean_value.digits 10) :=
  sorry

end mean_correct_and_no_seven_l48_48250


namespace sin_150_eq_half_l48_48680

noncomputable def calculate_sin_150 : ℝ := Real.sin (150 * Real.pi / 180)

theorem sin_150_eq_half : calculate_sin_150 = 1 / 2 :=
by
  -- We would include the detailed steps if not skipping the proof as per the instruction.
  sorry

end sin_150_eq_half_l48_48680


namespace bc_fraction_of_ad_l48_48945

theorem bc_fraction_of_ad
  {A B D C : Type}
  (length_AB length_BD length_AC length_CD length_AD length_BC : ℝ)
  (h1 : length_AB = 3 * length_BD)
  (h2 : length_AC = 4 * length_CD)
  (h3 : length_AD = length_AB + length_BD + length_CD)
  (h4 : length_BC = length_AC - length_AB) :
  length_BC / length_AD = 5 / 6 :=
by sorry

end bc_fraction_of_ad_l48_48945


namespace solve_system_l48_48283

theorem solve_system (s t : ℚ) (h1 : 7 * s + 6 * t = 156) (h2 : s = t / 2 + 3) : s = 192 / 19 :=
sorry

end solve_system_l48_48283


namespace remainder_when_sum_divided_by_15_l48_48360

theorem remainder_when_sum_divided_by_15 (a b c : ℕ) 
  (h1 : a % 15 = 11) 
  (h2 : b % 15 = 12) 
  (h3 : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
  sorry

end remainder_when_sum_divided_by_15_l48_48360


namespace max_area_rectangle_l48_48199

/-- Given a rectangle with a perimeter of 40, the rectangle with the maximum area is a square
with sides of length 10. The maximum area is thus 100. -/
theorem max_area_rectangle (a b : ℝ) (h : a + b = 20) : a * b ≤ 100 :=
by
  sorry

end max_area_rectangle_l48_48199


namespace initial_marbles_l48_48510

theorem initial_marbles (M : ℕ) (h1 : M + 9 = 104) : M = 95 := by
  sorry

end initial_marbles_l48_48510


namespace sin_150_eq_half_l48_48710

theorem sin_150_eq_half :
  (sin (150 : ℝ)) = 1 / 2 :=
by
  -- Conditions in form of definitions
  let θ : ℝ := 30
  have h1 : 150 = 180 - θ := by ring 
  have h2 : sin (180 - θ) = sin θ := by exact sin_sub_π θ
  have h3 : sin θ = 1 / 2 := by exact real.sin_of_real (by norm_num)
  sorry  -- Proof omitted

end sin_150_eq_half_l48_48710


namespace jellybeans_left_l48_48468

/-- 
There are 100 jellybeans in a glass jar. Mrs. Copper’s kindergarten class normally has 24 kids, 
but 2 children called in sick and stayed home that day. The remaining children 
who attended school eat 3 jellybeans each. How many jellybeans are still left in the jar?
 -/
theorem jellybeans_left (j_0 k s b : ℕ) (h_j0 : j_0 = 100) (h_k : k = 24) (h_s : s = 2) (h_b : b = 3) :
  j_0 - (k - s) * b = 34 :=
by
  rw [h_j0, h_k, h_s, h_b]
  norm_num
  sorry

end jellybeans_left_l48_48468


namespace pauline_bought_2_pounds_of_meat_l48_48579

theorem pauline_bought_2_pounds_of_meat :
  ∀ (cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent : ℝ) 
    (num_bell_peppers : ℕ),
  cost_taco_shells = 5 →
  cost_bell_pepper = 1.5 →
  cost_meat_per_pound = 3 →
  total_spent = 17 →
  num_bell_peppers = 4 →
  (total_spent - (cost_taco_shells + (num_bell_peppers * cost_bell_pepper))) / cost_meat_per_pound = 2 :=
by
  intros cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent num_bell_peppers 
         h1 h2 h3 h4 h5
  sorry

end pauline_bought_2_pounds_of_meat_l48_48579


namespace Mr_Brown_selling_price_l48_48049

theorem Mr_Brown_selling_price 
  (initial_price : ℕ := 100000)
  (profit_percentage : ℚ := 10)
  (loss_percentage : ℚ := 10) :
  let profit := initial_price * (profit_percentage / 100),
      price_to_Brown := initial_price + profit,
      loss := price_to_Brown * (loss_percentage / 100),
      selling_price := price_to_Brown - loss
  in selling_price = 99000 := by
    sorry

end Mr_Brown_selling_price_l48_48049


namespace most_likely_outcome_l48_48795

-- Define the probabilities for each outcome
def P_all_boys := (1/2)^6
def P_all_girls := (1/2)^6
def P_3_girls_3_boys := (Nat.choose 6 3) * (1/2)^6
def P_4_one_2_other := 2 * (Nat.choose 6 2) * (1/2)^6

-- Terms with values of each probability
lemma outcome_A : P_all_boys = 1 / 64 := by sorry
lemma outcome_B : P_all_girls = 1 / 64 := by sorry
lemma outcome_C : P_3_girls_3_boys = 20 / 64 := by sorry
lemma outcome_D : P_4_one_2_other = 30 / 64 := by sorry

-- Prove the main statement
theorem most_likely_outcome :
  P_4_one_2_other > P_all_boys ∧ P_4_one_2_other > P_all_girls ∧ P_4_one_2_other > P_3_girls_3_boys :=
by
  rw [outcome_A, outcome_B, outcome_C, outcome_D]
  sorry

end most_likely_outcome_l48_48795


namespace trigonometric_expression_l48_48532

theorem trigonometric_expression
  (α : ℝ)
  (h1 : Real.sin α = 3 / 5)
  (h2 : α ∈ Set.Ioo (π / 2) π) :
  (Real.cos (2 * α) / (Real.sqrt 2 * Real.sin (α + π / 4))) = -7 / 5 := 
sorry

end trigonometric_expression_l48_48532


namespace five_coins_not_155_l48_48892

def coin_values : List ℕ := [5, 25, 50]

def can_sum_to (n : ℕ) (count : ℕ) : Prop :=
  ∃ (a b c : ℕ), a + b + c = count ∧ a * 5 + b * 25 + c * 50 = n

theorem five_coins_not_155 : ¬ can_sum_to 155 5 :=
  sorry

end five_coins_not_155_l48_48892


namespace infinitely_many_n_l48_48582

theorem infinitely_many_n (p : ℕ) (hp : p.Prime) (hp2 : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ n * 2^n + 1 :=
sorry

end infinitely_many_n_l48_48582


namespace seventh_term_l48_48080

def nth_term (n : ℕ) (a : ℝ) : ℝ :=
  (-2) ^ n * a ^ (2 * n - 1)

theorem seventh_term (a : ℝ) : nth_term 7 a = -128 * a ^ 13 :=
by sorry

end seventh_term_l48_48080


namespace volume_of_pyramid_l48_48409

-- Definitions based on conditions
def regular_quadrilateral_pyramid (h r : ℝ) := 
  ∃ a : ℝ, ∃ S : ℝ, ∃ V : ℝ,
  a = 2 * h * ((h^2 - r^2) / r^2).sqrt ∧
  S = (2 * h * ((h^2 - r^2) / r^2).sqrt)^2 ∧
  V = (4 * h^5 - 4 * h^3 * r^2) / (3 * r^2)

-- Lean 4 theorem statement
theorem volume_of_pyramid (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  ∃ V : ℝ, V = (4 * h^5 - 4 * h^3 * r^2) / (3 * r^2) :=
sorry

end volume_of_pyramid_l48_48409


namespace polar_to_line_distance_l48_48300

theorem polar_to_line_distance : 
  let point_polar := (2, Real.pi / 3)
  let line_polar := (2, 0)  -- Corresponding (rho, theta) for the given line
  let point_rect := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
  let line_rect := 2  -- x = 2 in rectangular coordinates
  let distance := abs (line_rect - point_rect.1)
  distance = 1 := by
{
  sorry
}

end polar_to_line_distance_l48_48300


namespace negation_of_positive_x2_plus_2_l48_48806

theorem negation_of_positive_x2_plus_2 (h : ∀ x : ℝ, x^2 + 2 > 0) : ¬ (∀ x : ℝ, x^2 + 2 > 0) = False := 
by
  sorry

end negation_of_positive_x2_plus_2_l48_48806


namespace expression_evaluation_l48_48864

theorem expression_evaluation :
  10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 :=
by
  sorry

end expression_evaluation_l48_48864


namespace part1_part2_l48_48742

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

theorem part2 (a x : ℝ) (h : a ≠ -3) :
  (f x a > 4 * a - (a + 3) * x) ↔ 
  ((a > -3 ∧ (x < -3 ∨ x > a)) ∨ (a < -3 ∧ (x < a ∨ x > -3))) :=
by
  sorry

end part1_part2_l48_48742


namespace avg_width_is_3_5_l48_48119

def book_widths : List ℚ := [4, (3/4), 1.25, 3, 2, 7, 5.5]

noncomputable def average (l : List ℚ) : ℚ :=
  l.sum / l.length

theorem avg_width_is_3_5 : average book_widths = 23.5 / 7 :=
by
  sorry

end avg_width_is_3_5_l48_48119


namespace initial_green_hard_hats_l48_48026

noncomputable def initial_pink_hard_hats : ℕ := 26
noncomputable def initial_yellow_hard_hats : ℕ := 24
noncomputable def carl_taken_pink_hard_hats : ℕ := 4
noncomputable def john_taken_pink_hard_hats : ℕ := 6
noncomputable def john_taken_green_hard_hats (G : ℕ) : ℕ := 2 * john_taken_pink_hard_hats
noncomputable def remaining_pink_hard_hats : ℕ := initial_pink_hard_hats - carl_taken_pink_hard_hats - john_taken_pink_hard_hats
noncomputable def total_remaining_hard_hats (G : ℕ) : ℕ := remaining_pink_hard_hats + (G - john_taken_green_hard_hats G) + initial_yellow_hard_hats

theorem initial_green_hard_hats (G : ℕ) :
  total_remaining_hard_hats G = 43 ↔ G = 15 := by
  sorry

end initial_green_hard_hats_l48_48026


namespace mario_age_is_4_l48_48612

-- Define the conditions
def sum_of_ages (mario maria : ℕ) : Prop := mario + maria = 7
def mario_older_by_one (mario maria : ℕ) : Prop := mario = maria + 1

-- State the theorem to prove Mario's age is 4 given the conditions
theorem mario_age_is_4 (mario maria : ℕ) (h1 : sum_of_ages mario maria) (h2 : mario_older_by_one mario maria) : mario = 4 :=
sorry -- Proof to be completed later

end mario_age_is_4_l48_48612


namespace alex_silver_tokens_l48_48855

theorem alex_silver_tokens :
  ∃ x y : ℕ, 
    (100 - 3 * x + y ≤ 2) ∧ 
    (50 + 2 * x - 4 * y ≤ 3) ∧
    (x + y = 74) :=
by
  sorry

end alex_silver_tokens_l48_48855


namespace mod_pow_eq_l48_48331

theorem mod_pow_eq (m : ℕ) (h1 : 13^4 % 11 = m) (h2 : 0 ≤ m ∧ m < 11) : m = 5 := by
  sorry

end mod_pow_eq_l48_48331


namespace find_angle_D_l48_48923

theorem find_angle_D (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : B = C + 40) : D = 70 := by
  sorry

end find_angle_D_l48_48923


namespace sin_150_eq_half_l48_48704

theorem sin_150_eq_half :
  Float.sin (150 * Float.pi / 180) = 1 / 2 := sorry

end sin_150_eq_half_l48_48704


namespace number_of_students_l48_48802

theorem number_of_students
    (average_marks : ℕ)
    (wrong_mark : ℕ)
    (correct_mark : ℕ)
    (correct_average_marks : ℕ)
    (h1 : average_marks = 100)
    (h2 : wrong_mark = 50)
    (h3 : correct_mark = 10)
    (h4 : correct_average_marks = 96)
  : ∃ n : ℕ, (100 * n - 40) / n = 96 ∧ n = 10 :=
by
  sorry

end number_of_students_l48_48802


namespace triangle_area_x_l48_48519

theorem triangle_area_x (x : ℝ) (h_pos : x > 0) (h_area : 1 / 2 * x * (3 * x) = 72) : x = 4 * real.sqrt 3 :=
sorry

end triangle_area_x_l48_48519


namespace games_did_not_work_l48_48873

theorem games_did_not_work 
  (games_from_friend : ℕ) 
  (games_from_garage_sale : ℕ) 
  (good_games : ℕ) 
  (total_games : ℕ := games_from_friend + games_from_garage_sale) 
  (did_not_work : ℕ := total_games - good_games) :
  games_from_friend = 41 ∧ 
  games_from_garage_sale = 14 ∧ 
  good_games = 24 → 
  did_not_work = 31 := 
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end games_did_not_work_l48_48873


namespace totalArrangements_l48_48800

-- Define the number of chickens, dogs, and cats
def numChickens : ℕ := 6
def numDogs : ℕ := 4
def numCats : ℕ := 5

-- Define the factorial function for easier readability
noncomputable def fact (n : ℕ) : ℕ := Nat.factorial n

-- Count the number of ways to place the animals in the specified order
noncomputable def countArrangements : ℕ :=
  2 * (fact numChickens) * (fact numDogs) * (fact numCats)

-- State the main theorem
theorem totalArrangements : countArrangements = 4_147_200 := by
  sorry

end totalArrangements_l48_48800


namespace set_contains_all_rationals_l48_48035

variable (S : Set ℚ)
variable (h1 : (0 : ℚ) ∈ S)
variable (h2 : ∀ x ∈ S, x + 1 ∈ S ∧ x - 1 ∈ S)
variable (h3 : ∀ x ∈ S, x ≠ 0 → x ≠ 1 → 1 / (x * (x - 1)) ∈ S)

theorem set_contains_all_rationals : ∀ q : ℚ, q ∈ S :=
by
  sorry

end set_contains_all_rationals_l48_48035


namespace illegal_simplification_works_for_specific_values_l48_48106

-- Definitions for the variables
def a : ℕ := 43
def b : ℕ := 17
def c : ℕ := 26

-- Define the sum of cubes
def sum_of_cubes (x y : ℕ) : ℕ := x ^ 3 + y ^ 3

-- Define the illegal simplification fraction
def illegal_simplification_fraction_correct (a b c : ℕ) : Prop :=
  (a^3 + b^3) / (a^3 + c^3) = (a + b) / (a + c)

-- The theorem to prove
theorem illegal_simplification_works_for_specific_values :
  illegal_simplification_fraction_correct a b c :=
by
  -- Proof will reside here
  sorry

end illegal_simplification_works_for_specific_values_l48_48106


namespace fraction_of_sum_l48_48639

theorem fraction_of_sum (l : List ℝ) (n : ℝ) (h_len : l.length = 21) (h_mem : n ∈ l)
  (h_n_avg : n = 4 * (l.erase n).sum / 20) :
  n / l.sum = 1 / 6 := by
  sorry

end fraction_of_sum_l48_48639


namespace sum_of_integers_l48_48201

theorem sum_of_integers :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a < 30 ∧ b < 30 ∧ (a * b + a + b = 167) ∧ Nat.gcd a b = 1 ∧ (a + b = 24) :=
by {
  sorry
}

end sum_of_integers_l48_48201


namespace camel_height_in_feet_correct_l48_48918

def hare_height_in_inches : ℕ := 14
def multiplication_factor : ℕ := 24
def inches_to_feet_ratio : ℕ := 12

theorem camel_height_in_feet_correct :
  (hare_height_in_inches * multiplication_factor) / inches_to_feet_ratio = 28 := by
  sorry

end camel_height_in_feet_correct_l48_48918


namespace find_percentage_l48_48784

variable (dollars_1 dollars_2 dollars_total interest_total percentage_unknown : ℝ)
variable (investment_1 investment_rest interest_2 : ℝ)
variable (P : ℝ)

-- Assuming given conditions
axiom H1 : dollars_total = 12000
axiom H2 : dollars_1 = 5500
axiom H3 : interest_total = 970
axiom H4 : investment_rest = dollars_total - dollars_1
axiom H5 : interest_2 = investment_rest * 0.09
axiom H6 : interest_total = dollars_1 * P + interest_2

-- Prove that P = 0.07
theorem find_percentage : P = 0.07 :=
by
  -- Placeholder for the proof that needs to be filled in
  sorry

end find_percentage_l48_48784


namespace probability_of_shaded_triangle_l48_48027

def triangle (name: String) := name

def triangles := ["AEC", "AEB", "BED", "BEC", "BDC", "ABD"]
def shaded_triangles := ["BEC", "BDC", "ABD"]

theorem probability_of_shaded_triangle :
  (shaded_triangles.length : ℚ) / (triangles.length : ℚ) = 1 / 2 := 
by
  sorry

end probability_of_shaded_triangle_l48_48027


namespace solve_quadratic_inequality_l48_48330

theorem solve_quadratic_inequality (a : ℝ) (x : ℝ) :
  (x^2 - a * x + a - 1 ≤ 0) ↔
  (a < 2 ∧ a - 1 ≤ x ∧ x ≤ 1) ∨
  (a = 2 ∧ x = 1) ∨
  (a > 2 ∧ 1 ≤ x ∧ x ≤ a - 1) := 
by
  sorry

end solve_quadratic_inequality_l48_48330


namespace rectangle_side_multiple_of_6_l48_48375

theorem rectangle_side_multiple_of_6 (a b : ℕ) (h : ∃ n : ℕ, a * b = n * 6) : a % 6 = 0 ∨ b % 6 = 0 :=
sorry

end rectangle_side_multiple_of_6_l48_48375


namespace ocean_depth_at_base_of_cone_l48_48100

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

noncomputable def submerged_height_fraction (total_height volume_fraction : ℝ) : ℝ :=
  total_height * (volume_fraction)^(1/3)

theorem ocean_depth_at_base_of_cone (total_height radius : ℝ) 
  (above_water_volume_fraction : ℝ) : ℝ :=
  let above_water_height := submerged_height_fraction total_height above_water_volume_fraction
  total_height - above_water_height

example : ocean_depth_at_base_of_cone 10000 2000 (3 / 5) = 1566 := by
  sorry

end ocean_depth_at_base_of_cone_l48_48100


namespace cafeteria_pies_l48_48803

theorem cafeteria_pies (total_apples initial_apples_per_pie held_out_apples : ℕ) (h : total_apples = 150) (g : held_out_apples = 24) (p : initial_apples_per_pie = 15) :
  ((total_apples - held_out_apples) / initial_apples_per_pie) = 8 :=
by
  -- problem-specific proof steps would go here
  sorry

end cafeteria_pies_l48_48803


namespace dealer_cannot_prevent_l48_48237

theorem dealer_cannot_prevent (m n : ℕ) (h : m < 3 * n ∧ n < 3 * m) :
  ∃ (a b : ℕ), (a = 3 * b ∨ b = 3 * a) ∨ (a = 0 ∧ b = 0):=
sorry

end dealer_cannot_prevent_l48_48237


namespace speed_of_stream_l48_48482

-- Definitions based on the conditions provided
def speed_still_water : ℝ := 15
def upstream_time_ratio := 2

-- Proof statement
theorem speed_of_stream (v : ℝ) 
  (h1 : ∀ d t_up t_down, (15 - v) * t_up = d ∧ (15 + v) * t_down = d ∧ t_up = upstream_time_ratio * t_down) : 
  v = 5 :=
sorry

end speed_of_stream_l48_48482


namespace inequality_and_equality_l48_48180

theorem inequality_and_equality (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * b * c + 4 * a * c ∧ (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * b * c + 4 * a * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end inequality_and_equality_l48_48180


namespace intersection_of_sets_l48_48279

theorem intersection_of_sets {A B : Set Nat} (hA : A = {1, 3, 9}) (hB : B = {1, 5, 9}) :
  A ∩ B = {1, 9} :=
sorry

end intersection_of_sets_l48_48279


namespace trig_expression_value_l48_48415

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 1/2) :
  (1 + 2 * Real.sin (π - α) * Real.cos (-2 * π - α)) / 
  (Real.sin (-α) ^ 2 - Real.sin (5 * π / 2 - α) ^ 2) = -3 :=
by
  sorry

end trig_expression_value_l48_48415


namespace eval_nested_fractions_l48_48509

theorem eval_nested_fractions : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end eval_nested_fractions_l48_48509


namespace unique_solution_l48_48513

variables {x y z : ℝ}

def equation1 (x y z : ℝ) : Prop :=
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) = x*y*z

def equation2 (x y z : ℝ) : Prop :=
  (x^4 + x^2*y^2 + y^4) * (y^4 + y^2*z^2 + z^4) * (z^4 + z^2*x^2 + x^4) = x^3*y^3*z^3

theorem unique_solution :
  equation1 x y z ∧ equation2 x y z → x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
by
  sorry

end unique_solution_l48_48513


namespace sale_in_second_month_l48_48236

theorem sale_in_second_month
  (sale1 sale3 sale4 sale5 sale6 : ℕ)
  (average_sale : ℕ)
  (total_months : ℕ)
  (h_sale1 : sale1 = 5420)
  (h_sale3 : sale3 = 6200)
  (h_sale4 : sale4 = 6350)
  (h_sale5 : sale5 = 6500)
  (h_sale6 : sale6 = 6470)
  (h_average_sale : average_sale = 6100)
  (h_total_months : total_months = 6) :
  ∃ sale2 : ℕ, sale2 = 5660 := 
by
  sorry

end sale_in_second_month_l48_48236


namespace vikki_take_home_pay_l48_48975

-- Define the conditions
def hours_worked : ℕ := 42
def pay_rate : ℝ := 10
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5

-- Define the gross earnings function
def gross_earnings (hours_worked : ℕ) (pay_rate : ℝ) : ℝ := hours_worked * pay_rate

-- Define the deductions functions
def tax_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def insurance_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def total_deductions (tax : ℝ) (insurance : ℝ) (dues : ℝ) : ℝ := tax + insurance + dues

-- Define the take-home pay function
def take_home_pay (gross : ℝ) (deductions : ℝ) : ℝ := gross - deductions

theorem vikki_take_home_pay :
  take_home_pay (gross_earnings hours_worked pay_rate)
    (total_deductions (tax_deduction (gross_earnings hours_worked pay_rate) tax_rate)
                      (insurance_deduction (gross_earnings hours_worked pay_rate) insurance_rate)
                      union_dues) = 310 :=
by
  sorry

end vikki_take_home_pay_l48_48975


namespace age_problem_l48_48585

-- Definitions from conditions
variables (p q : ℕ) -- ages of p and q as natural numbers
variables (Y : ℕ) -- number of years ago p was half the age of q

-- Main statement
theorem age_problem :
  (p + q = 28) ∧ (p / q = 3 / 4) ∧ (p - Y = (q - Y) / 2) → Y = 8 :=
by
  sorry

end age_problem_l48_48585


namespace brenda_age_problem_l48_48496

variable (A B J : Nat)

theorem brenda_age_problem
  (h1 : A = 4 * B) 
  (h2 : J = B + 9) 
  (h3 : A = J) : 
  B = 3 := 
by 
  sorry

end brenda_age_problem_l48_48496


namespace find_t_l48_48744

variables {t : ℝ}

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (t : ℝ) : ℝ × ℝ := (-2, t)

def are_parallel (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

theorem find_t (h : are_parallel vector_a (vector_b t)) : t = -4 :=
by sorry

end find_t_l48_48744


namespace arithmetic_sequence_condition_l48_48137

theorem arithmetic_sequence_condition {a : ℕ → ℤ} 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (m p q : ℕ) (hpq_pos : 0 < p) (hq_pos : 0 < q) (hm_pos : 0 < m) : 
  (p + q = 2 * m) → (a p + a q = 2 * a m) ∧ ¬((a p + a q = 2 * a m) → (p + q = 2 * m)) :=
by 
  sorry

end arithmetic_sequence_condition_l48_48137


namespace bacon_calories_percentage_l48_48115

-- Mathematical statement based on the problem
theorem bacon_calories_percentage :
  ∀ (total_sandwich_calories : ℕ) (number_of_bacon_strips : ℕ) (calories_per_strip : ℕ),
    total_sandwich_calories = 1250 →
    number_of_bacon_strips = 2 →
    calories_per_strip = 125 →
    (number_of_bacon_strips * calories_per_strip) * 100 / total_sandwich_calories = 20 :=
by
  intros total_sandwich_calories number_of_bacon_strips calories_per_strip h1 h2 h3 
  sorry

end bacon_calories_percentage_l48_48115


namespace sin_150_eq_half_l48_48664

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by sorry

end sin_150_eq_half_l48_48664


namespace find_all_x_satisfying_condition_l48_48514

theorem find_all_x_satisfying_condition :
  ∃ (x : Fin 2016 → ℝ), 
  (∀ i : Fin 2016, x (i + 1) % 2016 = x 0) ∧
  (∀ i : Fin 2016, x i ^ 2 + x i - 1 = x ((i + 1) % 2016)) ∧
  (∀ i : Fin 2016, x i = 1 ∨ x i = -1) :=
sorry

end find_all_x_satisfying_condition_l48_48514


namespace quadratic_discriminant_single_solution_l48_48412

theorem quadratic_discriminant_single_solution :
  ∃ (n : ℝ), (∀ x : ℝ, 9 * x^2 + n * x + 36 = 0 → x = (-n) / (2 * 9)) → n = 36 :=
by
  sorry

end quadratic_discriminant_single_solution_l48_48412


namespace problem_l48_48911

theorem problem (h : ℤ) : (∃ x : ℤ, x = -2 ∧ x^3 + h * x - 12 = 0) → h = -10 := by
  sorry

end problem_l48_48911


namespace number_of_values_l48_48073

/-- Given:
  - The mean of some values was 190.
  - One value 165 was wrongly copied as 130 for the computation of the mean.
  - The correct mean is 191.4.
  Prove: the total number of values is 25. --/
theorem number_of_values (n : ℕ) (h₁ : (190 : ℝ) = ((190 * n) - (165 - 130)) / n) (h₂ : (191.4 : ℝ) = ((190 * n + 35) / n)) : n = 25 :=
sorry

end number_of_values_l48_48073


namespace geom_seq_sum_l48_48560

theorem geom_seq_sum (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 + a 2 = 16) 
  (h2 : a 3 + a 4 = 24) 
  (h_geom : ∀ n, a (n+1) = r * a n):
  a 7 + a 8 = 54 :=
sorry

end geom_seq_sum_l48_48560


namespace race_distance_100_l48_48298

noncomputable def race_distance (a b c d : ℝ) :=
  (d / a = (d - 20) / b) ∧
  (d / b = (d - 10) / c) ∧
  (d / a = (d - 28) / c) 

theorem race_distance_100 (a b c d : ℝ) (h1 : d / a = (d - 20) / b) (h2 : d / b = (d - 10) / c) (h3 : d / a = (d - 28) / c) : 
  d = 100 :=
  sorry

end race_distance_100_l48_48298


namespace find_b10_l48_48606

def sequence_b (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = b (n + 1) + b n

theorem find_b10 (b : ℕ → ℕ) (h0 : ∀ n, b n > 0) (h1 : b 9 = 544) (h2 : sequence_b b) : b 10 = 883 :=
by
  -- We could provide steps of the proof here, but we use 'sorry' to omit the proof content
  sorry

end find_b10_l48_48606


namespace carol_used_tissue_paper_l48_48386

theorem carol_used_tissue_paper (initial_pieces : ℕ) (remaining_pieces : ℕ) (usage: ℕ)
  (h1 : initial_pieces = 97)
  (h2 : remaining_pieces = 93)
  (h3: usage = initial_pieces - remaining_pieces) : 
  usage = 4 :=
by
  -- We only need to set up the problem; proof can be provided later.
  sorry

end carol_used_tissue_paper_l48_48386


namespace golden_triangle_ratio_l48_48968

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_ratio :
  let t := golden_ratio in
  (1 - 2 * Real.sin (27 * Real.pi / 180) ^ 2) / (2 * t * Real.sqrt (4 - t ^ 2)) = 1 / 4 := 
by
  let t := golden_ratio
  sorry

end golden_triangle_ratio_l48_48968


namespace sin_150_eq_half_l48_48674

theorem sin_150_eq_half :
  sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48674


namespace jasmine_carries_21_pounds_l48_48634

variable (weightChips : ℕ) (weightCookies : ℕ) (numBags : ℕ) (multiple : ℕ)

def totalWeightInPounds (weightChips weightCookies numBags multiple : ℕ) : ℕ :=
  let totalWeightInOunces := (weightChips * numBags) + (weightCookies * (numBags * multiple))
  totalWeightInOunces / 16

theorem jasmine_carries_21_pounds :
  weightChips = 20 → weightCookies = 9 → numBags = 6 → multiple = 4 → totalWeightInPounds weightChips weightCookies numBags multiple = 21 :=
by
  intros h1 h2 h3 h4
  simp [totalWeightInPounds, h1, h2, h3, h4]
  sorry

end jasmine_carries_21_pounds_l48_48634


namespace dilution_problem_l48_48326

/-- Samantha needs to add 7.2 ounces of water to achieve a 25% alcohol concentration
given that she starts with 12 ounces of solution containing 40% alcohol. -/
theorem dilution_problem (x : ℝ) : (12 + x) * 0.25 = 4.8 ↔ x = 7.2 :=
by sorry

end dilution_problem_l48_48326


namespace min_xy_min_a_b_l48_48632

-- Problem 1 Lean Statement
theorem min_xy {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 1 / (4 * y) = 1) : xy ≥ 2 := sorry

-- Problem 2 Lean Statement
theorem min_a_b {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : ab = a + 2 * b + 4) : a + b ≥ 3 + 2 * Real.sqrt 6 := sorry

end min_xy_min_a_b_l48_48632


namespace repeating_decimal_product_of_num_and_den_l48_48217

theorem repeating_decimal_product_of_num_and_den (x : ℚ) (h : x = 18 / 999) (h_simplified : x.num * x.den = 222) : x.num * x.den = 222 :=
by {
  sorry
}

end repeating_decimal_product_of_num_and_den_l48_48217


namespace count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l48_48095

-- Statement for Question 1
theorem count_numbers_with_digit_7 :
  ∃ n, n = 19 ∧ (∀ k, (k < 100 → (k / 10 = 7 ∨ k % 10 = 7) ↔ k ≠ 77)) :=
sorry

-- Statement for Question 2
theorem count_numbers_divisible_by_3_or_5 :
  ∃ n, n = 47 ∧ (∀ k, (k < 100 → (k % 3 = 0 ∨ k % 5 = 0)) ↔ (k % 15 = 0)) :=
sorry

end count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l48_48095


namespace inverse_of_matrix_A_l48_48724

open Matrix

variable {α : Type*} [Field α] [DecidableEq α]

def matrix_A : Matrix (Fin 2) (Fin 2) α :=
  ![![4, -3], ![5, -2]]

noncomputable def matrix_A_inv : Matrix (Fin 2) (Fin 2) α :=
  ![![-(2 / 7 : α), 3/7], ![-5/7, 4/7]]

theorem inverse_of_matrix_A :
  matrix_A.det ≠ 0 →
  matrix_A⁻¹ = matrix_A_inv :=
by
  intros h_det
  sorry

end inverse_of_matrix_A_l48_48724


namespace complement_of_A_inter_B_eq_l48_48435

noncomputable def A : Set ℝ := {x | abs (x - 1) ≤ 1}
noncomputable def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -Real.sqrt 2 ≤ x ∧ x < 1}
noncomputable def A_inter_B : Set ℝ := {x | x ∈ A ∧ x ∈ B}
noncomputable def complement_A_inter_B : Set ℝ := {x | x ∉ A_inter_B}

theorem complement_of_A_inter_B_eq :
  complement_A_inter_B = {x : ℝ | x ≠ 0} :=
  sorry

end complement_of_A_inter_B_eq_l48_48435


namespace eagles_score_l48_48291

variables (F E : ℕ)

theorem eagles_score (h1 : F + E = 56) (h2 : F = E + 8) : E = 24 := 
sorry

end eagles_score_l48_48291


namespace sum_of_g_35_l48_48040

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 - 3
noncomputable def g (y : ℝ) : ℝ := y^2 + y + 1

theorem sum_of_g_35 : g 35 = 21 := 
by
  sorry

end sum_of_g_35_l48_48040


namespace central_angle_unchanged_l48_48963

theorem central_angle_unchanged (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0) :
  (s / r) = (2 * s / (2 * r)) :=
by
  sorry

end central_angle_unchanged_l48_48963


namespace bananas_on_first_day_l48_48224

theorem bananas_on_first_day (total_bananas : ℕ) (days : ℕ) (increment : ℕ) (bananas_first_day : ℕ) :
  (total_bananas = 100) ∧ (days = 5) ∧ (increment = 6) ∧ ((bananas_first_day + (bananas_first_day + increment) + 
  (bananas_first_day + 2*increment) + (bananas_first_day + 3*increment) + (bananas_first_day + 4*increment)) = total_bananas) → 
  bananas_first_day = 8 :=
by
  sorry

end bananas_on_first_day_l48_48224


namespace last_student_remaining_l48_48365

/-- Definition of the function f, given n as a natural number -/
def f (n : ℕ) : ℕ :=
  let binary_rep := (n.bits)
  binary_rep.enum.sum (λ ⟨i, b⟩, 2 ^ i * ((-1)^(b + 1)))

/-- Main theorem stating that f computes the number of the last student remaining -/
theorem last_student_remaining (n : ℕ) :
  ∃ k (a : fin k → bool),
  n = bitvec.to_nat k (λ i, a i)
  ∧ f n = (∑ i in finset.range k, 2 ^ i * (-1)^(bitvec.to_nat k (λ i, a i) + 1))
:= sorry

end last_student_remaining_l48_48365


namespace sea_lions_at_zoo_l48_48651

def ratio_sea_lions_to_penguins (S P : ℕ) : Prop := P = 11 * S / 4
def ratio_sea_lions_to_flamingos (S F : ℕ) : Prop := F = 7 * S / 4
def penguins_more_sea_lions (S P : ℕ) : Prop := P = S + 84
def flamingos_more_penguins (P F : ℕ) : Prop := F = P + 42

theorem sea_lions_at_zoo (S P F : ℕ)
  (h1 : ratio_sea_lions_to_penguins S P)
  (h2 : ratio_sea_lions_to_flamingos S F)
  (h3 : penguins_more_sea_lions S P)
  (h4 : flamingos_more_penguins P F) :
  S = 42 :=
sorry

end sea_lions_at_zoo_l48_48651


namespace exponentiation_rule_l48_48241

theorem exponentiation_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end exponentiation_rule_l48_48241


namespace find_value_of_A_l48_48980

-- Define the conditions
variable (A : ℕ)
variable (divisor : ℕ := 9)
variable (quotient : ℕ := 2)
variable (remainder : ℕ := 6)

-- The main statement of the proof problem
theorem find_value_of_A (h : A = quotient * divisor + remainder) : A = 24 :=
by
  -- Proof would go here
  sorry

end find_value_of_A_l48_48980


namespace brenda_age_l48_48498

theorem brenda_age
  (A B J : ℕ)
  (h1 : A = 4 * B)
  (h2 : J = B + 9)
  (h3 : A = J)
  : B = 3 :=
by 
  sorry

end brenda_age_l48_48498


namespace complex_quadrant_l48_48871

theorem complex_quadrant (z : ℂ) (h : z = (↑(1/2) : ℂ) + (↑(1/2) : ℂ) * I ) : 
  0 < z.re ∧ 0 < z.im :=
by {
sorry -- Proof goes here
}

end complex_quadrant_l48_48871


namespace product_numerator_denominator_l48_48218

def recurring_decimal_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  n / d

theorem product_numerator_denominator (n : ℕ) (d : ℕ) (x : Rat)
  (hx : recurring_decimal_to_fraction 18 999 = x)
  (hn : n = 2)
  (hd : d = 111) :
  n * d = 222 := by
  have h_frac : x = 0.018 -- This follows from the definition and will be used in the proof
  sorry

end product_numerator_denominator_l48_48218


namespace floor_plus_self_eq_l48_48882

theorem floor_plus_self_eq (r : ℝ) (h : ⌊r⌋ + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_self_eq_l48_48882


namespace real_rate_of_return_is_10_percent_l48_48943

-- Given definitions based on conditions
def nominal_rate := 0.21
def inflation_rate := 0.10

-- Statement to prove
theorem real_rate_of_return_is_10_percent (r : ℝ) :
  1 + r = (1 + nominal_rate) / (1 + inflation_rate) → r = 0.10 := 
by
  sorry

end real_rate_of_return_is_10_percent_l48_48943


namespace find_temp_friday_l48_48483

-- Definitions for conditions
variables (M T W Th F : ℝ)

-- Condition 1: Average temperature for Monday to Thursday is 48 degrees
def avg_temp_mon_thu : Prop := (M + T + W + Th) / 4 = 48

-- Condition 2: Average temperature for Tuesday to Friday is 46 degrees
def avg_temp_tue_fri : Prop := (T + W + Th + F) / 4 = 46

-- Condition 3: Temperature on Monday is 39 degrees
def temp_monday : Prop := M = 39

-- Theorem: Temperature on Friday is 31 degrees
theorem find_temp_friday (h1 : avg_temp_mon_thu M T W Th)
                         (h2 : avg_temp_tue_fri T W Th F)
                         (h3 : temp_monday M) :
  F = 31 :=
sorry

end find_temp_friday_l48_48483


namespace fraction_not_covered_correct_l48_48102

def area_floor : ℕ := 64
def width_rug : ℕ := 2
def length_rug : ℕ := 7
def area_rug := width_rug * length_rug
def area_not_covered := area_floor - area_rug
def fraction_not_covered := (area_not_covered : ℚ) / area_floor

theorem fraction_not_covered_correct :
  fraction_not_covered = 25 / 32 :=
by
  -- Proof goes here
  sorry

end fraction_not_covered_correct_l48_48102


namespace one_third_way_l48_48624

theorem one_third_way (x₁ x₂ : ℚ) (w₁ w₂ : ℕ) (h₁ : x₁ = 1/4) (h₂ : x₂ = 3/4) (h₃ : w₁ = 2) (h₄ : w₂ = 1) : 
  (w₁ * x₁ + w₂ * x₂) / (w₁ + w₂) = 5 / 12 :=
by 
  rw [h₁, h₂, h₃, h₄]
  -- Simplification of the weighted average to get 5/12
  sorry

end one_third_way_l48_48624


namespace central_angle_remains_unchanged_l48_48961

theorem central_angle_remains_unchanged
  (r l : ℝ)
  (h_r : r > 0)
  (h_l : l > 0) :
  (l / r) = (2 * l) / (2 * r) :=
by
  sorry

end central_angle_remains_unchanged_l48_48961


namespace vector_parallel_y_value_l48_48526

theorem vector_parallel_y_value (y : ℝ) 
  (a : ℝ × ℝ := (3, 2)) 
  (b : ℝ × ℝ := (6, y)) 
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  y = 4 :=
by sorry

end vector_parallel_y_value_l48_48526


namespace Tate_education_years_l48_48797

theorem Tate_education_years :
  (let normal_highschool_years := 4 in
   let highschool_years := normal_highschool_years - 1 in
   let college_years := 3 * highschool_years in
   highschool_years + college_years = 12) :=
begin
  let normal_highschool_years := 4,
  let highschool_years := normal_highschool_years - 1,
  let college_years := 3 * highschool_years,
  have h : highschool_years + college_years = 12,
  sorry
end

end Tate_education_years_l48_48797


namespace friends_same_group_probability_l48_48334

open ProbabilityTheory

variable {Ω : Type*} [Fintype Ω]

/-- 600 students at King Middle School are divided into three groups of equal size for lunch.
  Each group has lunch at a different time. A computer randomly assigns each student to one
  of three lunch groups. -/
theorem friends_same_group_probability :
  let students := 600 
  let groups := 3
  let group_size := students / groups
  let Al Bob Carol : Ω
  let choice : Ω → Fin groups := λ s, (random_choice s) in
  P (event (choice Al = choice Bob ∧ choice Bob = choice Carol)) = 1 / 9 :=
by
  sorry

end friends_same_group_probability_l48_48334


namespace find_digit_B_l48_48346

def six_digit_number (B : ℕ) : ℕ := 303200 + B

def is_prime_six_digit (B : ℕ) : Prop := Prime (six_digit_number B)

theorem find_digit_B :
  ∃ B : ℕ, (B ≤ 9) ∧ (is_prime_six_digit B) ∧ (B = 9) :=
sorry

end find_digit_B_l48_48346


namespace fractional_part_sum_le_l48_48316

open Real Int
noncomputable theory

theorem fractional_part_sum_le (n : ℕ) :
  (∑ i in Finset.range (n^2 + 1) \ {0}, (sqrt i - ⌊sqrt i⌋)) ≤ (n^2 - 1) / 2 :=
begin
  sorry -- proof is left as an exercise
end

end fractional_part_sum_le_l48_48316


namespace not_all_squares_congruent_l48_48983

-- Define what it means to be a square
structure Square :=
  (side : ℝ)
  (angle : ℝ)
  (is_square : side > 0 ∧ angle = 90)

-- Define congruency of squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side = s2.side ∧ s1.angle = s2.angle

-- The main statement to prove 
theorem not_all_squares_congruent : ∃ s1 s2 : Square, ¬ congruent s1 s2 :=
by
  sorry

end not_all_squares_congruent_l48_48983


namespace lattice_points_count_l48_48548

theorem lattice_points_count : ∃ (S : Finset (ℤ × ℤ)), 
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 45}.toFinset = S ∧ S.card = 6 := 
sorry

end lattice_points_count_l48_48548


namespace shortTreesPlanted_l48_48618

-- Definitions based on conditions
def currentShortTrees : ℕ := 31
def tallTrees : ℕ := 32
def futureShortTrees : ℕ := 95

-- The proposition to be proved
theorem shortTreesPlanted :
  futureShortTrees - currentShortTrees = 64 :=
by
  sorry

end shortTreesPlanted_l48_48618


namespace opposite_of_negative_fraction_l48_48812

theorem opposite_of_negative_fraction : -(- (1/2023 : ℚ)) = 1/2023 := 
sorry

end opposite_of_negative_fraction_l48_48812


namespace find_k_hyperbola_l48_48021

-- Define the given conditions
variables (k : ℝ)
def condition1 : Prop := k < 0
def condition2 : Prop := 2 * k^2 + k - 2 = -1

-- State the proof goal
theorem find_k_hyperbola (h1 : condition1 k) (h2 : condition2 k) : k = -1 :=
by
  sorry

end find_k_hyperbola_l48_48021


namespace ratio_of_volumes_l48_48320

variables (A B : ℚ)

theorem ratio_of_volumes 
  (h1 : (3/8) * A = (5/8) * B) :
  A / B = 5 / 3 :=
sorry

end ratio_of_volumes_l48_48320


namespace sum_max_min_values_l48_48565

noncomputable def y (x : ℝ) : ℝ := 2 * x^2 + 32 / x

theorem sum_max_min_values :
  y 1 = 34 ∧ y 2 = 24 ∧ y 4 = 40 → ((y 4 + y 2) = 64) :=
by
  sorry

end sum_max_min_values_l48_48565


namespace george_speed_to_school_l48_48000

theorem george_speed_to_school :
  ∀ (D S_1 S_2 D_1 S_x : ℝ),
  D = 1.5 ∧ S_1 = 3 ∧ S_2 = 2 ∧ D_1 = 0.75 →
  S_x = (D - D_1) / ((D / S_1) - (D_1 / S_2)) →
  S_x = 6 :=
by
  intros D S_1 S_2 D_1 S_x h1 h2
  rw [h1.1, h1.2.1, h1.2.2.1, h1.2.2.2] at *
  sorry

end george_speed_to_school_l48_48000


namespace jennifer_money_left_over_l48_48432

theorem jennifer_money_left_over :
  let original_amount := 120
  let sandwich_cost := original_amount / 5
  let museum_ticket_cost := original_amount / 6
  let book_cost := original_amount / 2
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  let money_left := original_amount - total_spent
  money_left = 16 :=
by
  let original_amount := 120
  let sandwich_cost := original_amount / 5
  let museum_ticket_cost := original_amount / 6
  let book_cost := original_amount / 2
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  let money_left := original_amount - total_spent
  exact sorry

end jennifer_money_left_over_l48_48432


namespace f_f_2_eq_2_l48_48743

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 2 * Real.exp (x - 1)
else Real.log (x ^ 2 - 1) / Real.log 3

theorem f_f_2_eq_2 : f (f 2) = 2 :=
by
  sorry

end f_f_2_eq_2_l48_48743


namespace water_remaining_45_days_l48_48379

-- Define the initial conditions and the evaporation rate
def initial_volume : ℕ := 400
def evaporation_rate : ℕ := 1
def days : ℕ := 45

-- Define a function to compute the remaining water volume
def remaining_volume (initial_volume : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_volume - (evaporation_rate * days)

-- Theorem stating that the water remaining after 45 days is 355 gallons
theorem water_remaining_45_days : remaining_volume 400 1 45 = 355 :=
by
  -- proof goes here
  sorry

end water_remaining_45_days_l48_48379


namespace inequality_solution_l48_48059

theorem inequality_solution (x : ℝ) : 
  (x < 2 ∨ x = 3) ↔ (x - 3) / ((x - 2) * (x - 3)) ≤ 0 := 
by {
  sorry
}

end inequality_solution_l48_48059


namespace max_value_of_expression_l48_48005

theorem max_value_of_expression (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + b = 1) : 
  2 * Real.sqrt (a * b) - 4 * a ^ 2 - b ^ 2 ≤ (Real.sqrt 2 - 1) / 2 :=
sorry

end max_value_of_expression_l48_48005


namespace steven_amanda_hike_difference_l48_48658

variable (Camila_hikes : ℕ)
variable (Camila_weeks : ℕ)
variable (hikes_per_week : ℕ)

def Amanda_hikes (Camila_hikes : ℕ) : ℕ := 8 * Camila_hikes

def Steven_hikes (Camila_hikes : ℕ)(Camila_weeks : ℕ)(hikes_per_week : ℕ) : ℕ :=
  Camila_hikes + Camila_weeks * hikes_per_week

theorem steven_amanda_hike_difference
  (hCamila : Camila_hikes = 7)
  (hWeeks : Camila_weeks = 16)
  (hHikesPerWeek : hikes_per_week = 4) :
  Steven_hikes Camila_hikes Camila_weeks hikes_per_week - Amanda_hikes Camila_hikes = 15 := by
  sorry

end steven_amanda_hike_difference_l48_48658


namespace sisters_work_together_days_l48_48925

-- Definitions based on conditions
def task_completion_rate_older_sister : ℚ := 1/10
def task_completion_rate_younger_sister : ℚ := 1/20
def work_done_by_older_sister_alone : ℚ := 4 * task_completion_rate_older_sister
def remaining_task_after_older_sister : ℚ := 1 - work_done_by_older_sister_alone
def combined_work_rate : ℚ := task_completion_rate_older_sister + task_completion_rate_younger_sister

-- Statement of the proof problem
theorem sisters_work_together_days : 
  (combined_work_rate * x = remaining_task_after_older_sister) → 
  (x = 4) :=
by
  sorry

end sisters_work_together_days_l48_48925


namespace compositeQuotientCorrect_l48_48122

namespace CompositeNumbersProof

def firstFiveCompositesProduct : ℕ :=
  21 * 22 * 24 * 25 * 26

def subsequentFiveCompositesProduct : ℕ :=
  27 * 28 * 30 * 32 * 33

def compositeQuotient : ℚ :=
  firstFiveCompositesProduct / subsequentFiveCompositesProduct

theorem compositeQuotientCorrect : compositeQuotient = 1 / 1964 := by sorry

end CompositeNumbersProof

end compositeQuotientCorrect_l48_48122


namespace range_of_a_l48_48754

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
sorry

end range_of_a_l48_48754


namespace negation_proposition_real_l48_48807

theorem negation_proposition_real :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_proposition_real_l48_48807


namespace solution_set_inequality_l48_48725

noncomputable def f (x : ℝ) : ℝ := x * (1 - 3 * x)

theorem solution_set_inequality : {x : ℝ | f x > 0} = { x | (0 < x) ∧ (x < 1/3) } := by
  sorry

end solution_set_inequality_l48_48725


namespace floor_plus_r_eq_10_3_implies_r_eq_5_3_l48_48884

noncomputable def floor (x : ℝ) : ℤ := sorry -- Assuming the function exists

theorem floor_plus_r_eq_10_3_implies_r_eq_5_3 (r : ℝ) 
  (h : floor r + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_r_eq_10_3_implies_r_eq_5_3_l48_48884


namespace expression_evaluation_l48_48865

theorem expression_evaluation :
  10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 :=
by
  sorry

end expression_evaluation_l48_48865


namespace largest_value_l48_48458

def value (word : List Char) : Nat :=
  word.foldr (fun c acc =>
    acc + match c with
      | 'A' => 1
      | 'B' => 2
      | 'C' => 3
      | 'D' => 4
      | 'E' => 5
      | _ => 0
    ) 0

theorem largest_value :
  value ['B', 'E', 'E'] > value ['D', 'A', 'D'] ∧
  value ['B', 'E', 'E'] > value ['B', 'A', 'D'] ∧
  value ['B', 'E', 'E'] > value ['C', 'A', 'B'] ∧
  value ['B', 'E', 'E'] > value ['B', 'E', 'D'] :=
by sorry

end largest_value_l48_48458


namespace parallel_lines_slope_eq_l48_48398

theorem parallel_lines_slope_eq (b : ℝ) :
    (∀ x y : ℝ, 3 * y - 3 * b = 9 * x → ∀ x' y' : ℝ, y' - 2 = (b + 9) * x' → 3 = b + 9) →
    b = -6 := 
by 
  intros h
  have h1 : 3 = b + 9 := sorry -- proof omitted
  rw h1
  norm_num

end parallel_lines_slope_eq_l48_48398


namespace sin_150_eq_half_l48_48682

noncomputable def calculate_sin_150 : ℝ := Real.sin (150 * Real.pi / 180)

theorem sin_150_eq_half : calculate_sin_150 = 1 / 2 :=
by
  -- We would include the detailed steps if not skipping the proof as per the instruction.
  sorry

end sin_150_eq_half_l48_48682


namespace ambulance_ride_cost_l48_48777

-- Define the conditions as per the given problem.
def totalBill : ℝ := 5000
def medicationPercentage : ℝ := 0.5
def overnightStayPercentage : ℝ := 0.25
def foodCost : ℝ := 175

-- Define the question to be proved.
theorem ambulance_ride_cost :
  let medicationCost := totalBill * medicationPercentage
  let remainingAfterMedication := totalBill - medicationCost
  let overnightStayCost := remainingAfterMedication * overnightStayPercentage
  let remainingAfterOvernight := remainingAfterMedication - overnightStayCost
  let remainingAfterFood := remainingAfterOvernight - foodCost
  remainingAfterFood = 1700 :=
by
  -- Proof can be completed here
  sorry

end ambulance_ride_cost_l48_48777


namespace sin_150_eq_half_l48_48709

theorem sin_150_eq_half :
  (sin (150 : ℝ)) = 1 / 2 :=
by
  -- Conditions in form of definitions
  let θ : ℝ := 30
  have h1 : 150 = 180 - θ := by ring 
  have h2 : sin (180 - θ) = sin θ := by exact sin_sub_π θ
  have h3 : sin θ = 1 / 2 := by exact real.sin_of_real (by norm_num)
  sorry  -- Proof omitted

end sin_150_eq_half_l48_48709


namespace farmer_purchase_l48_48490

theorem farmer_purchase : ∃ r c : ℕ, 30 * r + 45 * c = 1125 ∧ r > 0 ∧ c > 0 ∧ r = 3 ∧ c = 23 := 
by 
  sorry

end farmer_purchase_l48_48490


namespace range_of_x_l48_48128

theorem range_of_x :
  (∀ t : ℝ, |t - 3| + |2 * t + 1| ≥ |2 * x - 1| + |x + 2|) →
  (-1/2 ≤ x ∧ x ≤ 5/6) :=
by
  intro h 
  sorry

end range_of_x_l48_48128


namespace sin_150_eq_half_l48_48702

theorem sin_150_eq_half :
  Float.sin (150 * Float.pi / 180) = 1 / 2 := sorry

end sin_150_eq_half_l48_48702


namespace average_people_per_hour_l48_48028

theorem average_people_per_hour 
    (total_people : ℕ) 
    (days : ℕ) 
    (hours_per_day : ℕ) 
    (h1 : total_people = 3000) 
    (h2 : days = 5) 
    (h3 : hours_per_day = 24) :
    total_people / (days * hours_per_day) = 25 :=
begin
    -- The proof would go here, but it is omitted as per the instructions.
    sorry
end

end average_people_per_hour_l48_48028


namespace boat_speed_proof_l48_48844

noncomputable def speed_in_still_water : ℝ := sorry -- Defined but proof skipped

def stream_speed : ℝ := 4
def distance_downstream : ℝ := 32
def distance_upstream : ℝ := 16

theorem boat_speed_proof (v : ℝ) :
  (distance_downstream / (v + stream_speed) = distance_upstream / (v - stream_speed)) →
  v = 12 :=
by
  sorry

end boat_speed_proof_l48_48844


namespace min_final_exam_score_l48_48242

theorem min_final_exam_score (q1 q2 q3 q4 final_exam : ℤ)
    (H1 : q1 = 90) (H2 : q2 = 85) (H3 : q3 = 77) (H4 : q4 = 96) :
    (1/2) * (q1 + q2 + q3 + q4) / 4 + (1/2) * final_exam ≥ 90 ↔ final_exam ≥ 93 :=
by
    sorry

end min_final_exam_score_l48_48242


namespace sin_150_eq_half_l48_48700

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by 
  sorry

end sin_150_eq_half_l48_48700


namespace face_opposite_to_A_is_D_l48_48255

-- Definitions of faces
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Given conditions
def C_is_on_top : Face := C
def B_is_to_the_right_of_C : Face := B
def forms_cube (f1 f2 : Face) : Prop := -- Some property indicating that the faces are part of a folded cube
sorry

-- The theorem statement to prove that the face opposite to face A is D
theorem face_opposite_to_A_is_D (h1 : C_is_on_top = C) (h2 : B_is_to_the_right_of_C = B) (h3 : forms_cube A D)
    : ∃ f : Face, f = D := sorry

end face_opposite_to_A_is_D_l48_48255


namespace inheritance_amount_l48_48434

theorem inheritance_amount (x : ℝ) (h1 : x * 0.25 + (x * 0.75) * 0.15 + 2500 = 16500) : x = 38621 := 
by
  sorry

end inheritance_amount_l48_48434


namespace range_of_a_l48_48148

noncomputable def f (a x : ℝ) := (a - Real.sin x) / Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (π / 6 < x) → (x < π / 3) → (f a x) ≤ (f a (x + ε))) → 2 ≤ a :=
by
  sorry

end range_of_a_l48_48148


namespace find_y_l48_48158

-- Definition of the modified magic square
variable (a b c d e y : ℕ)

-- Conditions from the modified magic square problem
axiom h1 : y + 5 + c = 120 + a + c
axiom h2 : y + (y - 115) + e = 120 + b + e
axiom h3 : y + 25 + 120 = 5 + (y - 115) + (2*y - 235)

-- The statement to prove
theorem find_y : y = 245 :=
by
  sorry

end find_y_l48_48158


namespace find_ctg_half_l48_48512

noncomputable def ctg (x : ℝ) := 1 / (Real.tan x)

theorem find_ctg_half
  (x : ℝ)
  (h : Real.sin x - Real.cos x = (1 + 2 * Real.sqrt 2) / 3) :
  ctg (x / 2) = Real.sqrt 2 / 2 ∨ ctg (x / 2) = 3 - 2 * Real.sqrt 2 :=
by
  sorry

end find_ctg_half_l48_48512


namespace largest_whole_number_value_l48_48978

theorem largest_whole_number_value (n : ℕ) : 
  (1 : ℚ) / 5 + (n : ℚ) / 8 < 9 / 5 → n ≤ 12 := 
sorry

end largest_whole_number_value_l48_48978


namespace find_minimum_value_of_f_l48_48313

def f (x : ℝ) : ℝ := (x ^ 2 + 4 * x + 5) * (x ^ 2 + 4 * x + 2) + 2 * x ^ 2 + 8 * x + 1

theorem find_minimum_value_of_f : ∃ x : ℝ, f x = -9 :=
by
  sorry

end find_minimum_value_of_f_l48_48313


namespace number_of_ways_to_choose_one_top_and_one_bottom_l48_48857

theorem number_of_ways_to_choose_one_top_and_one_bottom :
  let number_of_hoodies := 5
  let number_of_sweatshirts := 4
  let number_of_jeans := 3
  let number_of_slacks := 5
  let total_tops := number_of_hoodies + number_of_sweatshirts
  let total_bottoms := number_of_jeans + number_of_slacks
  total_tops * total_bottoms = 72 := 
by
  sorry

end number_of_ways_to_choose_one_top_and_one_bottom_l48_48857


namespace circle_center_distance_l48_48898

theorem circle_center_distance (R : ℝ) : 
  ∃ (d : ℝ), 
  (∀ (θ : ℝ), θ = 30 → 
  ∀ (r : ℝ), r = 2.5 →
  ∀ (center_on_other_side : ℝ), center_on_other_side = R + R →
  d = 5) :=
by 
  use 5
  intros θ θ_eq r r_eq center_on_other_side center_eq
  sorry

end circle_center_distance_l48_48898


namespace pyramid_coloring_ways_l48_48521

theorem pyramid_coloring_ways (colors : Fin 5) 
  (coloring_condition : ∀ (a b : Fin 5), a ≠ b) :
  ∃ (ways: Nat), ways = 420 :=
by
  -- Given:
  -- 1. There are 5 available colors
  -- 2. Each vertex of the pyramid is colored differently from the vertices connected by an edge
  -- Prove:
  -- There are 420 ways to color the pyramid's vertices
  sorry

end pyramid_coloring_ways_l48_48521


namespace smallest_value_of_x_l48_48358

theorem smallest_value_of_x (x : ℝ) (h : 4 * x^2 - 20 * x + 24 = 0) : x = 2 :=
    sorry

end smallest_value_of_x_l48_48358


namespace exists_student_not_wet_l48_48443

theorem exists_student_not_wet (n : ℕ) (students : Fin (2 * n + 1) → ℝ) (distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → students i ≠ students j) : 
  ∃ i : Fin (2 * n + 1), ∀ j : Fin (2 * n + 1), (j ≠ i → students j ≠ students i) :=
  sorry

end exists_student_not_wet_l48_48443


namespace hyperbola_eccentricity_l48_48020

theorem hyperbola_eccentricity (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 1) (h_eccentricity : ∀ e : ℝ, e = 2) :
    k = -1 / 3 := 
sorry

end hyperbola_eccentricity_l48_48020


namespace determine_numbers_l48_48045

theorem determine_numbers (A B n : ℤ) (h1 : 0 ≤ n ∧ n ≤ 9) (h2 : A = 10 * B + n) (h3 : A + B = 2022) : 
  A = 1839 ∧ B = 183 :=
by
  -- proof will be filled in here
  sorry

end determine_numbers_l48_48045


namespace gray_region_area_l48_48083

theorem gray_region_area (d_small r_large r_small π : ℝ) (h1 : d_small = 6)
    (h2 : r_large = 3 * r_small) (h3 : r_small = d_small / 2) :
    (π * r_large ^ 2 - π * r_small ^ 2) = 72 * π := 
by
  -- The proof will be filled here
  sorry

end gray_region_area_l48_48083


namespace inequality_holds_l48_48533

noncomputable def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := Real.exp x + x - 2
def g (x : ℝ) : ℝ := Real.log x + x - 2

def root_a (a : ℝ) : Prop := f a = 0
def root_b (b : ℝ) : Prop := g b = 0

theorem inequality_holds (a b : ℝ) (H_a : root_a a) (H_b : root_b b) 
  (H_a_interval : 0 < a ∧ a < 1) (H_b_interval : 1 < b ∧ b < 2) 
  (H_f_increasing : ∀ (x y : ℝ), 0 < x → x < y → f x < f y) :
  f(a) < f(1) ∧ f(1) < f(b) :=
by 
  sorry

end inequality_holds_l48_48533


namespace probability_of_yellow_face_l48_48333

def total_faces : ℕ := 12
def red_faces : ℕ := 5
def yellow_faces : ℕ := 4
def blue_faces : ℕ := 2
def green_faces : ℕ := 1

theorem probability_of_yellow_face : (yellow_faces : ℚ) / (total_faces : ℚ) = 1 / 3 := by
  sorry

end probability_of_yellow_face_l48_48333


namespace average_bc_l48_48592

variables (A B C : ℝ)

-- Conditions
def average_abc := (A + B + C) / 3 = 45
def average_ab := (A + B) / 2 = 40
def weight_b := B = 31

-- Proof statement
theorem average_bc (A B C : ℝ) (h_avg_abc : average_abc A B C) (h_avg_ab : average_ab A B) (h_b : weight_b B) :
  (B + C) / 2 = 43 :=
sorry

end average_bc_l48_48592


namespace correct_transformation_l48_48525

-- Given conditions
variables {a b : ℝ}
variable (h : 3 * a = 4 * b)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)

-- Statement of the problem
theorem correct_transformation : (a / 4) = (b / 3) :=
sorry

end correct_transformation_l48_48525


namespace common_intersection_implies_cd_l48_48071

theorem common_intersection_implies_cd (a b c d : ℝ) (h : a ≠ b) (x y : ℝ) 
  (H1 : y = a * x + a) (H2 : y = b * x + b) (H3 : y = c * x + d) : c = d := by
  sorry

end common_intersection_implies_cd_l48_48071


namespace rectangle_area_l48_48753

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 := 
by 
  sorry

end rectangle_area_l48_48753


namespace salary_increase_after_five_years_l48_48321

theorem salary_increase_after_five_years :
  ∀ (S : ℝ), (S * (1.15)^5 - S) / S * 100 = 101.14 := by
sorry

end salary_increase_after_five_years_l48_48321


namespace folding_hexagon_quadrilateral_folding_hexagon_pentagon_l48_48127

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

end folding_hexagon_quadrilateral_folding_hexagon_pentagon_l48_48127


namespace money_problem_l48_48972

variable {c d : ℝ}

theorem money_problem (h1 : 3 * c - 2 * d < 30) (h2 : 4 * c + d = 60) : 
  c < 150 / 11 ∧ d > 60 / 11 := 
by 
  sorry

end money_problem_l48_48972


namespace find_third_number_l48_48539

-- Given conditions
variable (A B C : ℕ)
variable (LCM HCF : ℕ)
variable (h1 : A = 36)
variable (h2 : B = 44)
variable (h3 : LCM = 792)
variable (h4 : HCF = 12)
variable (h5 : A * B * C = LCM * HCF)

-- Desired proof
theorem find_third_number : C = 6 :=
by
  sorry

end find_third_number_l48_48539


namespace number_of_boxes_l48_48098

-- Definitions based on conditions
def pieces_per_box := 500
def total_pieces := 3000

-- Theorem statement, we need to prove that the number of boxes is 6
theorem number_of_boxes : total_pieces / pieces_per_box = 6 :=
by {
  sorry
}

end number_of_boxes_l48_48098


namespace find_m_value_l48_48461

-- Definitions of the given lines
def l1 (x y : ℝ) (m : ℝ) : Prop := x + m * y + 6 = 0
def l2 (x y : ℝ) (m : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Parallel lines condition
def parallel (m : ℝ) : Prop :=
  ∀ x y : ℝ, l1 x y m = l2 x y m

-- Proof that the value of m for the lines to be parallel is indeed -1
theorem find_m_value : parallel (-1) :=
by
  sorry

end find_m_value_l48_48461


namespace cos_value_l48_48739

theorem cos_value (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (4 * π / 3 - α) = -3 / 5 := 
by 
  sorry

end cos_value_l48_48739


namespace Tino_has_correct_jellybeans_total_jellybeans_l48_48207

-- Define the individuals and their amounts of jellybeans
def Arnold_jellybeans := 5
def Lee_jellybeans := 2 * Arnold_jellybeans
def Tino_jellybeans := Lee_jellybeans + 24
def Joshua_jellybeans := 3 * Arnold_jellybeans

-- Verify Tino's jellybean count
theorem Tino_has_correct_jellybeans : Tino_jellybeans = 34 :=
by
  -- Unfold definitions and perform calculations
  sorry

-- Verify the total jellybean count
theorem total_jellybeans : (Arnold_jellybeans + Lee_jellybeans + Tino_jellybeans + Joshua_jellybeans) = 64 :=
by
  -- Unfold definitions and perform calculations
  sorry

end Tino_has_correct_jellybeans_total_jellybeans_l48_48207


namespace Janet_sold_six_action_figures_l48_48927

variable {x : ℕ}

theorem Janet_sold_six_action_figures
  (h₁ : 10 - x + 4 + 2 * (10 - x + 4) = 24) :
  x = 6 :=
by
  sorry

end Janet_sold_six_action_figures_l48_48927


namespace coin_flip_probability_l48_48987

-- Define a discrete probability space for a fair coin
def coin_flip := {0, 1}  -- 0 for tails, 1 for heads

-- Probability that a fair coin lands heads
def prob_heads : ℚ := 1 / 2

-- Probability that a fair coin lands tails
def prob_tails : ℚ := 1 / 2

-- Define the event of interest: heads on first flip and tails on last four flips
def event_of_interest (flips : List ℕ) : Prop :=
  flips.length = 5 ∧
  flips.head = 1 ∧
  flips.tail = [0, 0, 0, 0]

theorem coin_flip_probability :
  ∀ flips : List ℕ,
  (∀ flip ∈ flips, flip ∈ coin_flip) →
  event_of_interest flips →
  (prob_heads * prob_tails ^ 4) = (1 / 32) :=
by
  sorry

end coin_flip_probability_l48_48987


namespace probability_at_least_six_points_distribution_and_expectation_l48_48099

section part_one

/-- Prove that the probability that Team A will score at least 6 points after 
answering one question each from Traffic Safety and Fire Safety is 5/6, given 
the correct rates. -/
theorem probability_at_least_six_points (p_traffic : ℝ) (p_fire : ℝ) :
  p_traffic = 2/3 → p_fire = 1/2 → 
  let P := p_traffic * p_fire + p_traffic * (1 - p_fire) + (1 - p_traffic) * p_fire in
  P = 5/6 :=
by
  intro h1 h2
  let p := p_traffic * p_fire + p_traffic * (1 - p_fire) + (1 - p_traffic) * p_fire
  sorry

end part_one

section part_two

/-- Prove that the score distribution for Team A after answering 3 distinct 
category questions (Traffic Safety, Fire Safety, Water Safety) is Y ∈ {3, 7, 11, 15} 
with probabilities specified and that the expected score is 9, given the correct rates. -/
theorem distribution_and_expectation (r_traffic r_fire r_water : ℝ) :
  r_traffic = 2/3 → r_fire = 1/2 → r_water = 1/3 → 
  let P := [((3, 1/9)), ((7, 7/18)), ((11, 7/18)), ((15, 1/9))]
  let E_Y := 3 * (1 / 9) + 7 * (7 / 18) + 11 * (7 / 18) + 15 * (1 / 9)
  P.map Prod.snd = [1/9, 7/18, 7/18, 1/9] ∧ E_Y = 9 :=
by
  intro h1 h2 h3
  sorry

end part_two

end probability_at_least_six_points_distribution_and_expectation_l48_48099


namespace anya_age_l48_48245

theorem anya_age (n : ℕ) (h : 110 ≤ (n * (n + 1)) / 2 ∧ (n * (n + 1)) / 2 ≤ 130) : n = 15 :=
sorry

end anya_age_l48_48245


namespace sequence_bounded_l48_48072

theorem sequence_bounded (a : ℕ → ℝ) :
  a 0 = 2 →
  (∀ n, a (n+1) = (2 * a n + 1) / (a n + 2)) →
  ∀ n, 1 < a n ∧ a n < 1 + 1 / 3^n :=
by
  intro h₀ h₁
  sorry

end sequence_bounded_l48_48072


namespace bowling_ball_weight_l48_48188

noncomputable def weight_of_one_bowling_ball : ℕ := 20

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = weight_of_one_bowling_ball := by
  sorry

end bowling_ball_weight_l48_48188


namespace gravel_cost_l48_48013

def cost_per_cubic_foot := 8
def cubic_yards := 3
def cubic_feet_per_cubic_yard := 27

theorem gravel_cost :
  (cubic_yards * cubic_feet_per_cubic_yard) * cost_per_cubic_foot = 648 :=
by sorry

end gravel_cost_l48_48013


namespace greatest_a_for_integer_solutions_l48_48597

theorem greatest_a_for_integer_solutions :
  ∃ a : ℕ, 
    (∀ x : ℤ, x^2 + a * x = -21 → ∃ y : ℤ, y * (y + a) = -21) ∧ 
    ∀ b : ℕ, (∀ x : ℤ, x^2 + b * x = -21 → ∃ y : ℤ, y * (y + b) = -21) → b ≤ a :=
begin
  -- Proof goes here
  sorry
end

end greatest_a_for_integer_solutions_l48_48597


namespace cos_2theta_value_l48_48265

open Real

theorem cos_2theta_value (θ : ℝ) 
  (h: sin (2 * θ) - 4 * sin (θ + π / 3) * sin (θ - π / 6) = sqrt 3 / 3) : 
  cos (2 * θ) = 1 / 3 :=
  sorry

end cos_2theta_value_l48_48265


namespace does_not_pass_through_third_quadrant_l48_48460

theorem does_not_pass_through_third_quadrant :
  ¬ ∃ (x y : ℝ), 2 * x + 3 * y = 5 ∧ x < 0 ∧ y < 0 :=
by
  -- Proof goes here
  sorry

end does_not_pass_through_third_quadrant_l48_48460


namespace number_of_teams_l48_48763

theorem number_of_teams (n : ℕ) (G : ℕ) (h1 : G = 28) (h2 : G = n * (n - 1) / 2) : n = 8 := 
  by
  -- Proof skipped
  sorry

end number_of_teams_l48_48763


namespace product_bases_l48_48656

def base2_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 2 + (d.toNat - '0'.toNat)) 0

def base3_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 3 + (d.toNat - '0'.toNat)) 0

def base4_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 4 + (d.toNat - '0'.toNat)) 0

theorem product_bases :
  base2_to_nat "1101" * base3_to_nat "202" * base4_to_nat "22" = 2600 :=
by
  sorry

end product_bases_l48_48656


namespace evaluate_g_at_5_l48_48070

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem evaluate_g_at_5 : g 5 = 15 :=
by
    -- proof steps here
    sorry

end evaluate_g_at_5_l48_48070


namespace find_x_l48_48186

def custom_op (a b : ℝ) : ℝ :=
  a^2 - 3 * b

theorem find_x (x : ℝ) : 
  (custom_op (custom_op 7 x) 3 = 18) ↔ (x = 17.71 ∨ x = 14.96) := 
by
  sorry

end find_x_l48_48186


namespace simplify_eval_expression_l48_48185

theorem simplify_eval_expression (a b : ℤ) (h₁ : a = 2) (h₂ : b = -1) : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 2 * a * b) / (-2 * b) = -7 := 
by 
  sorry

end simplify_eval_expression_l48_48185


namespace johnny_money_left_l48_48930

def total_saved (september october november : ℕ) : ℕ := september + october + november

def money_left (total amount_spent : ℕ) : ℕ := total - amount_spent

theorem johnny_money_left 
    (saved_september : ℕ)
    (saved_october : ℕ)
    (saved_november : ℕ)
    (spent_video_game : ℕ)
    (h1 : saved_september = 30)
    (h2 : saved_october = 49)
    (h3 : saved_november = 46)
    (h4 : spent_video_game = 58) :
    money_left (total_saved saved_september saved_october saved_november) spent_video_game = 67 := 
by sorry

end johnny_money_left_l48_48930


namespace volume_s_l48_48308

def condition1 (x y : ℝ) : Prop := |9 - x| + y ≤ 12
def condition2 (x y : ℝ) : Prop := 3 * y - x ≥ 18
def S (x y : ℝ) : Prop := condition1 x y ∧ condition2 x y

def is_volume_correct (m n : ℕ) (p : ℕ) :=
  (m + n + p = 153) ∧ (m = 135) ∧ (n = 8) ∧ (p = 10)

theorem volume_s (m n p : ℕ) :
  (∀ x y : ℝ, S x y) → is_volume_correct m n p :=
by 
  sorry

end volume_s_l48_48308


namespace range_of_a_function_greater_than_exp_neg_x_l48_48009

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a / x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ f x a = 0) → (0 < a ∧ a ≤ 1 / Real.exp 1) :=
sorry

theorem function_greater_than_exp_neg_x (a : ℝ) (h : a ≥ 2 / Real.exp 1) (x : ℝ) (hx : 0 < x) : f x a > Real.exp (-x) :=
sorry

end range_of_a_function_greater_than_exp_neg_x_l48_48009


namespace max_area_rectangle_l48_48200

/-- Given a rectangle with a perimeter of 40, the rectangle with the maximum area is a square
with sides of length 10. The maximum area is thus 100. -/
theorem max_area_rectangle (a b : ℝ) (h : a + b = 20) : a * b ≤ 100 :=
by
  sorry

end max_area_rectangle_l48_48200


namespace half_n_lt_m_lt_two_n_l48_48038

theorem half_n_lt_m_lt_two_n (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n)
  (h : ∃ x : ℤ, (x + m) * (x + n) = x + m + n) :
  1 / 2 * n < m ∧ m < 2 * n :=
sorry

end half_n_lt_m_lt_two_n_l48_48038


namespace numbers_not_equal_l48_48032

theorem numbers_not_equal
  (a b c S : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a + b^2 + c^2 = S)
  (h2 : b + a^2 + c^2 = S)
  (h3 : c + a^2 + b^2 = S) :
  ¬ (a = b ∧ b = c) :=
by sorry

end numbers_not_equal_l48_48032


namespace opposite_of_x_is_positive_l48_48815

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive_l48_48815


namespace greatest_possible_value_of_a_l48_48601

theorem greatest_possible_value_of_a 
  (x a : ℤ)
  (h : x^2 + a * x = -21)
  (ha_pos : 0 < a)
  (hx_int : x ∈ [-21, -7, -3, -1].toFinset): 
  a ≤ 22 := sorry

end greatest_possible_value_of_a_l48_48601


namespace least_value_q_minus_p_l48_48924

def p : ℝ := 2
def q : ℝ := 5

theorem least_value_q_minus_p (y : ℝ) (h : p < y ∧ y < q) : q - p = 3 :=
by
  sorry

end least_value_q_minus_p_l48_48924


namespace log_eq_l48_48420

theorem log_eq (x : ℝ) (h : log 7 (x + 6) = 2) : log 13 x = log 13 43 :=
by
  sorry

end log_eq_l48_48420


namespace part1_part2_l48_48269

noncomputable def vector_parallel {a c : ℝ × ℝ} (h : ∃ k : ℝ, c = (k * a.1, k * a.2)) : Prop :=
  ∃ k : ℝ, c = ((k * a.1), (k * a.2))

theorem part1
  (a c : ℝ × ℝ)
  (h_a : a = (1, -2))
  (h_mag : real.sqrt (c.1 ^ 2 + c.2 ^ 2) = 2 * real.sqrt 5)
  (h_parallel : vector_parallel h_c h_a) :
  c = (2, -4) ∨ c = (-2, 4) :=
sorry

theorem part2
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h_a : a = (1, -2))
  (h_b_mag : real.sqrt(b.1 ^ 2 + b.2 ^ 2) = 1)
  (h_perp : (a.1 + b.1) * (a.1 - 2 * b.1) + (a.2 + b.2) * (a.2 - 2 * b.2) = 0) :
  real.cos (vector_angle a b) = 3 * real.sqrt 5 / 5 :=
sorry

end part1_part2_l48_48269


namespace solve_equation_l48_48057

noncomputable def equation (x : ℝ) : Prop := x * (x - 2) + x - 2 = 0

theorem solve_equation : ∀ x, equation x ↔ (x = 2 ∨ x = -1) :=
by sorry

end solve_equation_l48_48057


namespace third_price_reduction_l48_48081

theorem third_price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h1 : (original_price * (1 - x)^2 = final_price))
  (h2 : final_price = 100)
  (h3 : original_price = 100 / (1 - 0.19)) :
  (original_price * (1 - x)^3 = 90) :=
by
  sorry

end third_price_reduction_l48_48081


namespace factorize_x_pow_m_minus_x_pow_m_minus_2_l48_48723

theorem factorize_x_pow_m_minus_x_pow_m_minus_2 (x : ℝ) (m : ℕ) (h : m > 1) : 
  x ^ m - x ^ (m - 2) = (x ^ (m - 2)) * (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_pow_m_minus_x_pow_m_minus_2_l48_48723


namespace exists_cycle_length_not_divisible_by_3_l48_48154

variable (V : Type) [Fintype V] [DecidableEq V]

structure Graph (V : Type) :=
  (adj : V → V → Prop)
  (symm : ∀ {x y}, adj x y → adj y x)
  (irrefl : ∀ x, ¬ adj x x)

variable (G : Graph V)

-- Define 3-regular (cubic) graph property
def is_3_regular : Prop := 
  ∀ v : V, Fintype.card {w : V // G.adj v w} = 3

theorem exists_cycle_length_not_divisible_by_3 
  (hG : ∀ v : V, Fintype.card {w : V // G.adj v w} = 3) :
  ∃ (c : List V), List.Length c ≠ 0 ∧ ∀ e ∈ c.zip (c.tail), G.adj e.fst e.snd ∧ List.Length c % 3 ≠ 0 :=
sorry

end exists_cycle_length_not_divisible_by_3_l48_48154


namespace gray_area_correct_l48_48084

-- Define the conditions of the problem
def diameter_small : ℝ := 6
def radius_small : ℝ := diameter_small / 2
def radius_large : ℝ := 3 * radius_small

-- Define the areas based on the conditions
def area_small : ℝ := Real.pi * radius_small^2
def area_large : ℝ := Real.pi * radius_large^2
def gray_area : ℝ := area_large - area_small

-- Write the theorem that proves the required area of the gray region
theorem gray_area_correct : gray_area = 72 * Real.pi :=
by
  sorry

end gray_area_correct_l48_48084


namespace min_value_of_sum_l48_48268

theorem min_value_of_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x * y + 2 * x + y = 4) : x + y ≥ 2 * Real.sqrt 6 - 3 :=
sorry

end min_value_of_sum_l48_48268


namespace lacy_percentage_correct_l48_48940

variable (x : ℕ)

-- Definitions from the conditions
def total_problems := 8 * x
def missed_problems := 2 * x
def answered_problems := total_problems - missed_problems
def bonus_problems := x
def bonus_points := 2 * bonus_problems
def regular_points := answered_problems - bonus_problems
def total_points_scored := bonus_points + regular_points
def total_available_points := 8 * x + 2 * x

theorem lacy_percentage_correct :
  total_points_scored / total_available_points * 100 = 90 := by
  -- Proof steps would go here, but are not required per instructions.
  sorry

end lacy_percentage_correct_l48_48940


namespace resistance_of_second_resistor_l48_48159

theorem resistance_of_second_resistor 
  (R1 R_total R2 : ℝ) 
  (hR1: R1 = 9) 
  (hR_total: R_total = 4.235294117647059) 
  (hFormula: 1/R_total = 1/R1 + 1/R2) : 
  R2 = 8 :=
by
  sorry

end resistance_of_second_resistor_l48_48159


namespace friends_meeting_probability_l48_48973

noncomputable def n_value (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2) : ℝ :=
  d - e * Real.sqrt f

theorem friends_meeting_probability (n : ℝ) (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2)
  (H : n = n_value d e f h1 h2 h3) : d + e + f = 92 :=
  by
  sorry

end friends_meeting_probability_l48_48973


namespace greatest_possible_difference_l48_48913

theorem greatest_possible_difference (x y : ℝ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) :
  ∃ n : ℤ, n = 9 ∧ ∀ x' y' : ℤ, (6 < x' ∧ x' < 10 ∧ 10 < y' ∧ y' < 17) → (y' - x' ≤ n) :=
by {
  -- here goes the actual proof
  sorry
}

end greatest_possible_difference_l48_48913


namespace goose_price_remains_affordable_l48_48760

theorem goose_price_remains_affordable :
  ∀ (h v : ℝ),
  h + v = 1 →
  h + (v / 2) = 1 →
  h * 1.2 ≤ 1 :=
by
  intros h v h_eq v_eq
  /- Proof will go here -/
  sorry

end goose_price_remains_affordable_l48_48760


namespace sin_150_eq_half_l48_48681

noncomputable def calculate_sin_150 : ℝ := Real.sin (150 * Real.pi / 180)

theorem sin_150_eq_half : calculate_sin_150 = 1 / 2 :=
by
  -- We would include the detailed steps if not skipping the proof as per the instruction.
  sorry

end sin_150_eq_half_l48_48681


namespace number_of_third_year_students_to_sample_l48_48372

theorem number_of_third_year_students_to_sample
    (total_students : ℕ)
    (first_year_students : ℕ)
    (second_year_students : ℕ)
    (third_year_students : ℕ)
    (total_to_sample : ℕ)
    (h_total : total_students = 1200)
    (h_first : first_year_students = 480)
    (h_second : second_year_students = 420)
    (h_third : third_year_students = 300)
    (h_sample : total_to_sample = 100) :
    third_year_students * total_to_sample / total_students = 25 :=
by
  sorry

end number_of_third_year_students_to_sample_l48_48372


namespace cost_of_soap_for_year_l48_48318

theorem cost_of_soap_for_year
  (months_per_bar cost_per_bar : ℕ)
  (months_in_year : ℕ)
  (h1 : months_per_bar = 2)
  (h2 : cost_per_bar = 8)
  (h3 : months_in_year = 12) :
  (months_in_year / months_per_bar) * cost_per_bar = 48 := by
  sorry

end cost_of_soap_for_year_l48_48318


namespace find_alpha_l48_48342

def point (α : ℝ) : Prop := 3^α = Real.sqrt 3

theorem find_alpha (α : ℝ) (h : point α) : α = 1/2 := 
by 
  sorry

end find_alpha_l48_48342


namespace number_before_star_is_five_l48_48556

theorem number_before_star_is_five (n : ℕ) (h1 : n % 72 = 0) (h2 : n % 10 = 0) (h3 : ∃ k, n = 400 + 10 * k) : (n / 10) % 10 = 5 :=
sorry

end number_before_star_is_five_l48_48556


namespace product_of_fraction_l48_48213

-- Define the repeating decimal as given in the problem
def repeating_decimal : Rat := 0.018 -- represents 0.\overline{018}

-- Define the given fraction obtained by simplifying
def simplified_fraction : Rat := 2 / 111

-- The goal is to prove that the product of the numerator and denominator of 
-- the simplified fraction of the repeating decimal is 222
theorem product_of_fraction (y : Rat) (hy : y = 0.018) (fraction_eq : y = 18 / 999) : 
  (2:ℕ) * (111:ℕ) = 222 :=
by
  sorry

end product_of_fraction_l48_48213


namespace solve_for_x_l48_48758

theorem solve_for_x (x : ℝ) (h : 2 - 1 / (1 - x) = 1 / (1 - x)) : x = 0 :=
sorry

end solve_for_x_l48_48758


namespace camel_height_in_feet_l48_48916

theorem camel_height_in_feet (h_ht_14 : ℕ) (ratio : ℕ) (inch_to_ft : ℕ) : ℕ :=
  let hare_height := 14
  let camel_height_in_inches := hare_height * 24
  let camel_height_in_feet := camel_height_in_inches / 12
  camel_height_in_feet
#print camel_height_in_feet

example : camel_height_in_feet 14 24 12 = 28 := by sorry

end camel_height_in_feet_l48_48916


namespace smallest_n_for_divisibility_problem_l48_48834

theorem smallest_n_for_divisibility_problem :
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n * (n + 1) ≠ 0 ∧
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ ¬ (n * (n + 1)) % k = 0) ∧
  ∀ m : ℕ, m > 0 ∧ m < n → (∀ k : ℕ, 1 ≤ k ∧ k ≤ m → (m * (m + 1)) % k ≠ 0)) → n = 4 := sorry

end smallest_n_for_divisibility_problem_l48_48834


namespace password_count_correct_l48_48993

-- Defining variables
def n_letters := 26
def n_digits := 10

-- The number of permutations for selecting 2 different letters
def perm_letters := n_letters * (n_letters - 1)
-- The number of permutations for selecting 2 different numbers
def perm_digits := n_digits * (n_digits - 1)

-- The total number of possible passwords
def total_permutations := perm_letters * perm_digits

-- The theorem we need to prove
theorem password_count_correct :
  total_permutations = (n_letters * (n_letters - 1)) * (n_digits * (n_digits - 1)) :=
by
  -- The proof goes here
  sorry

end password_count_correct_l48_48993


namespace sin_150_eq_one_half_l48_48689

theorem sin_150_eq_one_half :
  let θ := 150
  let θ_ref := 30
  let θ_complement := 180 - θ_ref
  θ = θ_complement →
  (∀ θ, θ_ref = 30 * 60.toReal)
    ∧ (∀ θ, (cos θ_ref, sin θ_ref) = (Real.sqrt 3 / 2, 1 / 2)) →
  sin θ = 1 / 2 :=
by
  intros θ θ_ref θ_complement h1 h2
  have h3 : (cos θ_complement, sin θ_complement) = (-(cos θ_ref), sin θ_ref) := by
    sorry
  rw h3
  sorry

end sin_150_eq_one_half_l48_48689


namespace product_of_fractional_parts_eq_222_l48_48215

theorem product_of_fractional_parts_eq_222 : 
  let x := 18 / 999 in let y := x.num / x.denom in y.num * y.denom = 222 :=
by 
  sorry

end product_of_fractional_parts_eq_222_l48_48215


namespace total_bananas_in_collection_l48_48336

-- Definitions based on the conditions
def group_size : ℕ := 18
def number_of_groups : ℕ := 10

-- The proof problem statement
theorem total_bananas_in_collection : group_size * number_of_groups = 180 := by
  sorry

end total_bananas_in_collection_l48_48336


namespace quadratic_solutions_l48_48752

-- Definition of the conditions given in the problem
def quadratic_axis_symmetry (b : ℝ) : Prop :=
  -b / 2 = 2

def equation_solutions (x b : ℝ) : Prop :=
  x^2 + b*x - 5 = 2*x - 13

-- The math proof problem statement in Lean 4
theorem quadratic_solutions (b : ℝ) (x1 x2 : ℝ) :
  quadratic_axis_symmetry b →
  equation_solutions x1 b →
  equation_solutions x2 b →
  (x1 = 2 ∧ x2 = 4) ∨ (x1 = 4 ∧ x2 = 2) :=
by
  sorry

end quadratic_solutions_l48_48752


namespace geometric_sequence_sum_l48_48562

-- Define the problem conditions and the result
theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ), a 1 + a 2 = 16 ∧ a 3 + a 4 = 24 → a 7 + a 8 = 54 :=
by
  -- Preliminary steps and definitions to prove the theorem
  sorry

end geometric_sequence_sum_l48_48562


namespace probability_correct_l48_48350

noncomputable def probability_of_two_or_more_co_presidents : ℚ :=
  let p_club (n : ℕ) := 
    (Nat.choose 3 2 * Nat.choose (n - 3) 2 + Nat.choose 3 3 * Nat.choose (n - 3) 1) /
    Nat.choose n 4
  let p_10 := p_club 10
  let p_12 := p_club 12
  let p_15 := p_club 15
  (1 / 3) * (p_10 + p_12 + p_15)

theorem probability_correct : probability_of_two_or_more_co_presidents = 2 / 9 := by
  sorry

end probability_correct_l48_48350


namespace camel_height_in_feet_l48_48917

theorem camel_height_in_feet (h_ht_14 : ℕ) (ratio : ℕ) (inch_to_ft : ℕ) : ℕ :=
  let hare_height := 14
  let camel_height_in_inches := hare_height * 24
  let camel_height_in_feet := camel_height_in_inches / 12
  camel_height_in_feet
#print camel_height_in_feet

example : camel_height_in_feet 14 24 12 = 28 := by sorry

end camel_height_in_feet_l48_48917


namespace jack_christina_speed_l48_48033

noncomputable def speed_of_jack_christina (d_jack_christina : ℝ) (v_lindy : ℝ) (d_lindy : ℝ) (relative_speed_factor : ℝ := 2) : ℝ :=
d_lindy * relative_speed_factor / d_jack_christina

theorem jack_christina_speed :
  speed_of_jack_christina 240 10 400 = 3 := by
  sorry

end jack_christina_speed_l48_48033


namespace domain_of_f_l48_48516

noncomputable def f (x : ℝ) : ℝ := (5 * x - 2) / Real.sqrt (x^2 - 3 * x - 4)

theorem domain_of_f :
  {x : ℝ | ∃ (f_x : ℝ), f x = f_x} = {x : ℝ | (x < -1) ∨ (x > 4)} :=
by
  sorry

end domain_of_f_l48_48516


namespace find_relationship_l48_48733

variables (x y : ℝ)

def AB : ℝ × ℝ := (6, 1)
def BC : ℝ × ℝ := (x, y)
def CD : ℝ × ℝ := (-2, -3)
def DA : ℝ × ℝ := (4 - x, -2 - y)
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_relationship (h_parallel : parallel (x, y) (4 - x, -2 - y)) : x + 2 * y = 0 :=
sorry

end find_relationship_l48_48733


namespace no_solutions_Y_l48_48567

theorem no_solutions_Y (Y : ℕ) : 2 * Y + Y + 3 * Y = 14 ↔ false :=
by 
  sorry

end no_solutions_Y_l48_48567


namespace vasya_kolya_difference_impossible_l48_48305

theorem vasya_kolya_difference_impossible : 
  ∀ k v : ℕ, (∃ q₁ q₂ : ℕ, 14400 = q₁ * 2 + q₂ * 2 + 1 + 1) → ¬ ∃ k, ∃ v, (v - k = 11 ∧ 14400 = k * q₁ + v * q₂) :=
by sorry

end vasya_kolya_difference_impossible_l48_48305


namespace equilateral_triangle_sum_perimeters_l48_48649

theorem equilateral_triangle_sum_perimeters (s : ℝ) (h : ∑' n, 3 * s / 2 ^ n = 360) : 
  s = 60 := 
by 
  sorry

end equilateral_triangle_sum_perimeters_l48_48649


namespace A_wins_one_prob_A_wins_at_least_2_of_3_prob_l48_48944

-- Define the probability of A and B guessing correctly
def prob_A_correct : ℚ := 5/6
def prob_B_correct : ℚ := 3/5

-- Definition of the independent events for A and B
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- The probability of A winning in one activity
def prob_A_wins_one : ℚ := prob_A_correct * prob_B_incorrect

-- Proof (statement) that A's probability of winning one activity is 1/3
theorem A_wins_one_prob :
  prob_A_wins_one = 1/3 :=
sorry

-- Binomial coefficient for choosing 2 wins out of 3 activities
def binom_coeff_n_2 (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Probability of A winning exactly 2 out of 3 activities
def prob_A_wins_exactly_2_of_3 : ℚ :=
  binom_coeff_n_2 3 2 * prob_A_wins_one^2 * (1 - prob_A_wins_one)

-- Probability of A winning all 3 activities
def prob_A_wins_all_3 : ℚ :=
  prob_A_wins_one^3

-- The probability of A winning at least 2 out of 3 activities
def prob_A_wins_at_least_2_of_3 : ℚ :=
  prob_A_wins_exactly_2_of_3 + prob_A_wins_all_3

-- Proof (statement) that A's probability of winning at least 2 out of 3 activities is 7/27
theorem A_wins_at_least_2_of_3_prob :
  prob_A_wins_at_least_2_of_3 = 7/27 :=
sorry

end A_wins_one_prob_A_wins_at_least_2_of_3_prob_l48_48944


namespace sin_150_eq_half_l48_48707

theorem sin_150_eq_half :
  Float.sin (150 * Float.pi / 180) = 1 / 2 := sorry

end sin_150_eq_half_l48_48707


namespace shortest_side_of_triangle_with_medians_l48_48821

noncomputable def side_lengths_of_triangle_with_medians (a b c m_a m_b m_c : ℝ) : Prop :=
  m_a = 3 ∧ m_b = 4 ∧ m_c = 5 →
  a^2 = 2*b^2 + 2*c^2 - 36 ∧
  b^2 = 2*a^2 + 2*c^2 - 64 ∧
  c^2 = 2*a^2 + 2*b^2 - 100

theorem shortest_side_of_triangle_with_medians :
  ∀ (a b c : ℝ), side_lengths_of_triangle_with_medians a b c 3 4 5 → 
  min a (min b c) = c :=
sorry

end shortest_side_of_triangle_with_medians_l48_48821


namespace initial_overs_l48_48761

variable (x : ℝ)

/-- 
Proof that the number of initial overs x is 10, given the conditions:
1. The run rate in the initial x overs was 3.2 runs per over.
2. The run rate in the remaining 50 overs was 5 runs per over.
3. The total target is 282 runs.
4. The runs scored in the remaining 50 overs should be 250 runs.
-/
theorem initial_overs (hx : 3.2 * x + 250 = 282) : x = 10 :=
sorry

end initial_overs_l48_48761


namespace sin_150_eq_half_l48_48708

theorem sin_150_eq_half :
  (sin (150 : ℝ)) = 1 / 2 :=
by
  -- Conditions in form of definitions
  let θ : ℝ := 30
  have h1 : 150 = 180 - θ := by ring 
  have h2 : sin (180 - θ) = sin θ := by exact sin_sub_π θ
  have h3 : sin θ = 1 / 2 := by exact real.sin_of_real (by norm_num)
  sorry  -- Proof omitted

end sin_150_eq_half_l48_48708


namespace charles_finishes_in_11_days_l48_48252

theorem charles_finishes_in_11_days : 
  ∀ (total_pages : ℕ) (pages_mon : ℕ) (pages_tue : ℕ) (pages_wed : ℕ) (pages_thu : ℕ) 
    (does_not_read_on_weekend : Prop),
  total_pages = 96 →
  pages_mon = 7 →
  pages_tue = 12 →
  pages_wed = 10 →
  pages_thu = 6 →
  does_not_read_on_weekend →
  ∃ days_to_finish : ℕ, days_to_finish = 11 :=
by
  intros
  sorry

end charles_finishes_in_11_days_l48_48252


namespace three_digit_multiples_of_6_and_9_l48_48909

theorem three_digit_multiples_of_6_and_9 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ k, n = k * 18)}.finite.card = 50 :=
by
  sorry

end three_digit_multiples_of_6_and_9_l48_48909


namespace games_needed_in_single_elimination_l48_48998

theorem games_needed_in_single_elimination (teams : ℕ) (h : teams = 23) : 
  ∃ games : ℕ, games = teams - 1 ∧ games = 22 :=
by
  existsi (teams - 1)
  sorry

end games_needed_in_single_elimination_l48_48998


namespace tate_total_education_years_l48_48799

theorem tate_total_education_years (normal_duration_hs : ℕ)
  (hs_years_less_than_normal : ℕ) 
  (mult_factor_bs_phd : ℕ) :
  normal_duration_hs = 4 → hs_years_less_than_normal = 1 → mult_factor_bs_phd = 3 →
  let hs_years := normal_duration_hs - hs_years_less_than_normal in
  let college_years := mult_factor_bs_phd * hs_years in
  hs_years + college_years = 12 :=
by
  intro h_normal h_less h_factor
  let hs_years := 4 - 1
  let college_years := 3 * hs_years
  show hs_years + college_years = 12
  sorry

end tate_total_education_years_l48_48799


namespace tan_210_eq_neg_sqrt3_over_3_l48_48402

noncomputable def angle_210 : ℝ := 210 * (Real.pi / 180)
noncomputable def angle_30 : ℝ := 30 * (Real.pi / 180)

theorem tan_210_eq_neg_sqrt3_over_3 : Real.tan angle_210 = -Real.sqrt 3 / 3 :=
by
  sorry -- Proof omitted

end tan_210_eq_neg_sqrt3_over_3_l48_48402


namespace inequality_does_not_hold_l48_48910

theorem inequality_does_not_hold {x y : ℝ} (h : x > y) : ¬ (-2 * x > -2 * y) ∧ (2023 * x > 2023 * y) ∧ (x - 1 > y - 1) ∧ (-x / 3 < -y / 3) :=
by {
  sorry
}

end inequality_does_not_hold_l48_48910


namespace sin_150_eq_half_l48_48683

noncomputable def calculate_sin_150 : ℝ := Real.sin (150 * Real.pi / 180)

theorem sin_150_eq_half : calculate_sin_150 = 1 / 2 :=
by
  -- We would include the detailed steps if not skipping the proof as per the instruction.
  sorry

end sin_150_eq_half_l48_48683


namespace arithmetic_example_l48_48249

theorem arithmetic_example : 2546 + 240 / 60 - 346 = 2204 := by
  sorry

end arithmetic_example_l48_48249


namespace range_of_independent_variable_x_l48_48202

noncomputable def range_of_x (x : ℝ) : Prop :=
  x > -2

theorem range_of_independent_variable_x (x : ℝ) :
  ∀ x, (x + 2 > 0) → range_of_x x :=
by
  intro x h
  unfold range_of_x
  linarith

end range_of_independent_variable_x_l48_48202


namespace shaded_area_concentric_circles_l48_48299

theorem shaded_area_concentric_circles (R : ℝ) (r : ℝ) (hR : π * R^2 = 100 * π) (hr : r = R / 2) :
  (1 / 2) * π * R^2 + (1 / 2) * π * r^2 = 62.5 * π :=
by
  -- Given conditions
  have R10 : R = 10 := sorry  -- Derived from hR
  have r5 : r = 5 := sorry    -- Derived from hr and R10
  -- Proof steps likely skipped
  sorry

end shaded_area_concentric_circles_l48_48299


namespace polynomial_horner_method_l48_48475

-- Define the polynomial f
def f (x : ℕ) :=
  7 * x ^ 7 + 6 * x ^ 6 + 5 * x ^ 5 + 4 * x ^ 4 + 3 * x ^ 3 + 2 * x ^ 2 + x

-- Define x as given in the condition
def x : ℕ := 3

-- State that f(x) = 262 when x = 3
theorem polynomial_horner_method : f x = 262 :=
  by
  sorry

end polynomial_horner_method_l48_48475


namespace latte_cost_l48_48574

theorem latte_cost (L : ℝ) 
  (latte_days : ℝ := 5)
  (iced_coffee_cost : ℝ := 2)
  (iced_coffee_days : ℝ := 3)
  (weeks_in_year : ℝ := 52)
  (spending_reduction : ℝ := 0.25)
  (savings : ℝ := 338) 
  (current_annual_spending : ℝ := 4 * savings)
  (weekly_spending : ℝ := latte_days * L + iced_coffee_days * iced_coffee_cost)
  (annual_spending_eq : weeks_in_year * weekly_spending = current_annual_spending) :
  L = 4 := 
sorry

end latte_cost_l48_48574


namespace sarah_earnings_l48_48056

-- Conditions
def monday_hours : ℚ := 1 + 3 / 4
def wednesday_hours : ℚ := 65 / 60
def thursday_hours : ℚ := 2 + 45 / 60
def friday_hours : ℚ := 45 / 60
def saturday_hours : ℚ := 2

def weekday_rate : ℚ := 4
def weekend_rate : ℚ := 6

-- Definition for total earnings
def total_weekday_earnings : ℚ :=
  (monday_hours + wednesday_hours + thursday_hours + friday_hours) * weekday_rate

def total_weekend_earnings : ℚ :=
  saturday_hours * weekend_rate

def total_earnings : ℚ :=
  total_weekday_earnings + total_weekend_earnings

-- Statement to prove
theorem sarah_earnings : total_earnings = 37.3332 := by
  sorry

end sarah_earnings_l48_48056


namespace range_of_m_l48_48528

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h₁ : x1 < x2) (h₂ : y1 < y2)
  (A_on_line : y1 = (2 * m - 1) * x1 + 1)
  (B_on_line : y2 = (2 * m - 1) * x2 + 1) :
  m > 0.5 :=
sorry

end range_of_m_l48_48528


namespace central_angle_remains_unchanged_l48_48962

theorem central_angle_remains_unchanged
  (r l : ℝ)
  (h_r : r > 0)
  (h_l : l > 0) :
  (l / r) = (2 * l) / (2 * r) :=
by
  sorry

end central_angle_remains_unchanged_l48_48962


namespace integral_problem_l48_48786

open Real

noncomputable def integrand (x : ℝ) : ℝ := (x ^ 4 * (1 - x) ^ 4) / (1 + x ^ 2)

theorem integral_problem :
  ∫ x in 0..1, integrand x = 22 / 7 - π :=
by
  sorry

end integral_problem_l48_48786


namespace smallest_d_l48_48960

noncomputable def abc_identity_conditions (a b c d e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  ∀ x : ℝ, (x + a) * (x + b) * (x + c) = x^3 + 3 * d * x^2 + 3 * x + e^3

theorem smallest_d (a b c d e : ℝ) (h : abc_identity_conditions a b c d e) : d = 1 := 
sorry

end smallest_d_l48_48960


namespace cylinder_increase_l48_48208

theorem cylinder_increase (x : ℝ) (r h : ℝ) (π : ℝ) 
  (h₁ : r = 5) (h₂ : h = 10) 
  (h₃ : π > 0) 
  (h_equal_volumes : π * (r + x) ^ 2 * h = π * r ^ 2 * (h + x)) :
  x = 5 / 2 :=
by
  -- Proof is omitted
  sorry

end cylinder_increase_l48_48208


namespace daphne_necklaces_l48_48178

/--
Given:
1. Total cost of necklaces and earrings is $240,000.
2. Necklaces are equal in price.
3. Earrings were three times as expensive as any one necklace.
4. Cost of a single necklace is $40,000.

Prove:
Princess Daphne bought 3 necklaces.
-/
theorem daphne_necklaces (total_cost : ℤ) (price_necklace : ℤ) (price_earrings : ℤ) (n : ℤ)
  (h1 : total_cost = 240000)
  (h2 : price_necklace = 40000)
  (h3 : price_earrings = 3 * price_necklace)
  (h4 : total_cost = n * price_necklace + price_earrings) : n = 3 :=
by
  sorry

end daphne_necklaces_l48_48178


namespace polynomial_perfect_square_l48_48344

theorem polynomial_perfect_square (m : ℤ) : (∃ a : ℤ, a^2 = 25 ∧ x^2 + m*x + 25 = (x + a)^2) ↔ (m = 10 ∨ m = -10) :=
by sorry

end polynomial_perfect_square_l48_48344


namespace aston_found_pages_l48_48501

-- Given conditions
def pages_per_comic := 25
def initial_untorn_comics := 5
def total_comics_now := 11

-- The number of pages Aston found on the floor
theorem aston_found_pages :
  (total_comics_now - initial_untorn_comics) * pages_per_comic = 150 := 
by
  sorry

end aston_found_pages_l48_48501


namespace area_acpq_eq_sum_areas_aekl_cdmn_l48_48762

variables (A B C D E P Q M N K L : Point)

def is_acute_angled_triangle (A B C : Point) : Prop := sorry
def is_altitude (A B C D : Point) : Prop := sorry
def is_square (A P Q C : Point) : Prop := sorry
def is_rectangle (A E K L : Point) : Prop := sorry
def is_rectangle' (C D M N : Point) : Prop := sorry
def length (P Q : Point) : Real := sorry
def area (P Q R S : Point) : Real := sorry

-- Conditions
axiom abc_acute : is_acute_angled_triangle A B C
axiom ad_altitude : is_altitude A B C D
axiom ce_altitude : is_altitude C A B E
axiom acpq_square : is_square A P Q C
axiom aekl_rectangle : is_rectangle A E K L
axiom cdmn_rectangle : is_rectangle' C D M N
axiom al_eq_ab : length A L = length A B
axiom cn_eq_cb : length C N = length C B

-- Question proof statement
theorem area_acpq_eq_sum_areas_aekl_cdmn :
  area A C P Q = area A E K L + area C D M N :=
sorry

end area_acpq_eq_sum_areas_aekl_cdmn_l48_48762


namespace determine_z_l48_48534

theorem determine_z (i z : ℂ) (hi : i^2 = -1) (h : i * z = 2 * z + 1) : 
  z = - (2/5 : ℂ) - (1/5 : ℂ) * i := by
  sorry

end determine_z_l48_48534


namespace algebraic_expression_value_l48_48275

theorem algebraic_expression_value (a b : ℝ) (h : ∃ x : ℝ, x = 2 ∧ 3 * (a - x) = 2 * (b * x - 4)) :
  9 * a^2 - 24 * a * b + 16 * b^2 + 25 = 29 :=
by sorry

end algebraic_expression_value_l48_48275


namespace sin_150_equals_half_l48_48692

noncomputable def sin_150_eq_half : Prop := 
  sin (150 * real.pi / 180) = 1 / 2

theorem sin_150_equals_half : sin_150_eq_half :=
  by sorry

end sin_150_equals_half_l48_48692


namespace sum_of_roots_l48_48956

-- Define the quadratic equation whose roots are the excluded domain values C and D
def quadratic_eq (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

-- Define C and D as the roots of the quadratic equation
def is_root (x : ℝ) : Prop := quadratic_eq x

-- Define C and D as the specific roots of the given quadratic equation
axiom C : ℝ
axiom D : ℝ

-- Assert that C and D are the roots of the quadratic equation
axiom hC : is_root C
axiom hD : is_root D

-- Statement to prove
theorem sum_of_roots : C + D = 3 :=
by sorry

end sum_of_roots_l48_48956


namespace length_of_each_piece_l48_48145

theorem length_of_each_piece :
  ∀ (ribbon_length remaining_length pieces : ℕ),
  ribbon_length = 51 →
  remaining_length = 36 →
  pieces = 100 →
  (ribbon_length - remaining_length) / pieces * 100 = 15 :=
by
  intros ribbon_length remaining_length pieces h1 h2 h3
  sorry

end length_of_each_piece_l48_48145


namespace number_of_girls_attending_winter_festival_l48_48317

variables (g b : ℝ)
variables (totalStudents attendFestival: ℝ)

theorem number_of_girls_attending_winter_festival
  (H1 : g + b = 1500)
  (H2 : (3/5) * g + (2/5) * b = 800) :
  (3/5 * g) = 600 :=
sorry

end number_of_girls_attending_winter_festival_l48_48317


namespace maximize_log_power_l48_48932

theorem maximize_log_power (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (hab : a * b = 100) :
  ∃ x : ℝ, (a ^ (Real.logb 10 b)^2 = 10^x) ∧ x = 32 / 27 :=
by
  sorry

end maximize_log_power_l48_48932


namespace seashells_after_giving_cannot_determine_starfish_l48_48653

-- Define the given conditions
def initial_seashells : Nat := 66
def seashells_given : Nat := 52
def seashells_left : Nat := 14

-- The main theorem to prove
theorem seashells_after_giving (initial : Nat) (given : Nat) (left : Nat) :
  initial = 66 -> given = 52 -> left = 14 -> initial - given = left :=
by 
  intros 
  sorry

-- The starfish count question
def starfish (count: Option Nat) : Prop :=
  count = none

-- Prove that we cannot determine the number of starfish Benny found
theorem cannot_determine_starfish (count: Option Nat) :
  count = none :=
by 
  intros 
  sorry

end seashells_after_giving_cannot_determine_starfish_l48_48653


namespace concurrent_lines_l48_48003

theorem concurrent_lines
  (c : circle)
  (A B O C E D H Z S I K L M P U R X T Q : point)
  (tangent_c_C : tangent C)
  (C_on_c : C ∈ c)
  (E_intersection : ∃ E, tangent_c_C ∩ line_through A B = {E})
  (D_dir_perp : is_perp D (line_through C D) (line_through A B))
  (CD_eq_CH_CZ : ∃ H Z, C D = C H ∧ C H = C Z ∧ H ∈ c ∧ Z ∈ c)
  (intersections_HZ : ∃ S I K, line_through H Z ∩ line_through C O = {S} ∧ line_through H Z ∩ line_through C D = {I} ∧ line_through H Z ∩ line_through A B = {K})
  (parallel_line_I_AB : ∃ L M, line_through I parallel line_through A B ∩ line_through C O = {L} ∧ line_through I parallel line_through A B ∩ line_through C K = {M})
  (circumcircle_LMD : circle_of_triangle L M D∩line_through A B = {P} ∧ circle_of_triangle L M D∩line_through C K = {U})
  (tangents_k : ∃ (e1 e2 e3 : line), tangent L e1 ∧ tangent M e2 ∧ tangent P e3)
  (RXT_intersections : ∃ R X T, intersection e1 e2 = {R} ∧ intersection e2 e3 = {X} ∧ intersection e1 e3 = {T})
  (Q_center : Q = center_of_circle (circle_of_triangle L M D))
  :
  concurrent (line_through R D) (line_through T U) (line_through X S) ∧ on_line (point_of_concurrency (line_through R D) (line_through T U) (line_through X S)) (line_through I Q) := by
  sorry

end concurrent_lines_l48_48003


namespace sin_150_eq_half_l48_48705

theorem sin_150_eq_half :
  Float.sin (150 * Float.pi / 180) = 1 / 2 := sorry

end sin_150_eq_half_l48_48705


namespace num_5_letter_words_with_at_least_one_A_l48_48745

theorem num_5_letter_words_with_at_least_one_A :
  let total := 6 ^ 5
  let without_A := 5 ^ 5
  total - without_A = 4651 := by
sorry

end num_5_letter_words_with_at_least_one_A_l48_48745


namespace gcd_entries_tends_to_infinity_l48_48518

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 2], ![4, 3]]

def I2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

def d (n : ℕ) : ℤ :=
  let M := A ^ n - I2 in
  Int.gcd (Int.gcd M[0, 0] M[0, 1]) (Int.gcd M[1, 0] M[1, 1])

theorem gcd_entries_tends_to_infinity : ∀ ε > 0, ∃ n ≥ 1, d(n) > ε :=
by
  sorry

end gcd_entries_tends_to_infinity_l48_48518


namespace g_possible_values_l48_48043

noncomputable def g (x y z : ℝ) : ℝ :=
  (x + y) / x + (y + z) / y + (z + x) / z

theorem g_possible_values (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 ≤ g x y z :=
by
  sorry

end g_possible_values_l48_48043


namespace total_price_of_bananas_and_oranges_l48_48596

variable (price_orange price_pear price_banana : ℝ)

axiom total_cost_orange_pear : price_orange + price_pear = 120
axiom cost_pear : price_pear = 90
axiom diff_orange_pear_banana : price_orange - price_pear = price_banana

theorem total_price_of_bananas_and_oranges :
  let num_bananas := 200
  let num_oranges := 2 * num_bananas
  let cost_bananas := num_bananas * price_banana
  let cost_oranges := num_oranges * price_orange
  cost_bananas + cost_oranges = 24000 :=
by
  sorry

end total_price_of_bananas_and_oranges_l48_48596


namespace smallest_n_for_divisibility_l48_48357

theorem smallest_n_for_divisibility (n : ℕ) (h1 : 24 ∣ n^2) (h2 : 1080 ∣ n^3) : n = 120 :=
sorry

end smallest_n_for_divisibility_l48_48357


namespace calculate_expression_l48_48863

theorem calculate_expression : (3.15 * 2.5) - 1.75 = 6.125 := 
by
  -- The proof is omitted, indicated by sorry
  sorry

end calculate_expression_l48_48863


namespace day_care_center_toddlers_l48_48467

theorem day_care_center_toddlers (I T : ℕ) (h_ratio1 : 7 * I = 3 * T) (h_ratio2 : 7 * (I + 12) = 5 * T) :
  T = 42 :=
by
  sorry

end day_care_center_toddlers_l48_48467


namespace total_points_l48_48577

noncomputable def Noa_score : ℕ := 30
noncomputable def Phillip_score : ℕ := 2 * Noa_score
noncomputable def Lucy_score : ℕ := (3 / 2) * Phillip_score

theorem total_points : 
  Noa_score + Phillip_score + Lucy_score = 180 := 
by
  sorry

end total_points_l48_48577


namespace regular_octahedron_has_4_pairs_l48_48295

noncomputable def regular_octahedron_parallel_edges : ℕ :=
  4

theorem regular_octahedron_has_4_pairs
  (h : true) : regular_octahedron_parallel_edges = 4 :=
by
  sorry

end regular_octahedron_has_4_pairs_l48_48295


namespace ant_prob_7min_l48_48500

open ProbabilityTheory

noncomputable theory

def ant_prob_path (A B C : ℕ × ℕ) (time : ℕ) : probability_space :=
{
  prob_path : A → C → B → time,
  A = (0, 0),
  B = (0, 1),
  C = (1, 0),
  time = 7,
}

theorem ant_prob_7min (A B C : ℕ × ℕ) (time : ℕ) : 
  (ant_prob_path A B C time).prob_path = 1 / 4 :=
by {
  sorry
}

end ant_prob_7min_l48_48500


namespace normal_distribution_problem_l48_48904

noncomputable def normal_probability_condition (X : ℝ → ℝ) (σ : ℝ) : Prop :=
  (∀ x, X(x) ∼ Normal 0 (σ^2)) ∧ ℙ(X ∈ Set.Icc (-2 : ℝ) 0) = 0.4

theorem normal_distribution_problem (X : ℝ → ℝ) (σ : ℝ) (h : normal_probability_condition X σ) :
  ℙ(X > 2) = 0.1 :=
sorry

end normal_distribution_problem_l48_48904


namespace games_per_season_l48_48160

-- Define the problem parameters
def total_goals : ℕ := 1244
def louie_last_match_goals : ℕ := 4
def louie_previous_goals : ℕ := 40
def louie_season_total_goals := louie_last_match_goals + louie_previous_goals
def brother_goals_per_game := 2 * louie_last_match_goals
def seasons : ℕ := 3

-- Prove the number of games in each season
theorem games_per_season : ∃ G : ℕ, louie_season_total_goals + (seasons * brother_goals_per_game * G) = total_goals ∧ G = 50 := 
by {
  sorry
}

end games_per_season_l48_48160


namespace incorrect_gcd_statement_l48_48109

theorem incorrect_gcd_statement :
  ¬(gcd 85 357 = 34) ∧ (gcd 16 12 = 4) ∧ (gcd 78 36 = 6) ∧ (gcd 105 315 = 105) :=
by
  sorry

end incorrect_gcd_statement_l48_48109


namespace product_evaluation_l48_48875

theorem product_evaluation : 
  (7 - 5) * (7 - 4) * (7 - 3) * (7 - 2) * (7- 1) * 7 = 5040 := 
by 
  sorry

end product_evaluation_l48_48875


namespace range_of_cars_l48_48077

def fuel_vehicle_cost_per_km (x : ℕ) : ℚ := (40 * 9) / x
def new_energy_vehicle_cost_per_km (x : ℕ) : ℚ := (60 * 0.6) / x

theorem range_of_cars : ∃ x : ℕ, fuel_vehicle_cost_per_km x = new_energy_vehicle_cost_per_km x + 0.54 ∧ x = 600 := 
by {
  sorry
}

end range_of_cars_l48_48077


namespace action_figure_prices_l48_48307

noncomputable def prices (x y z w : ℝ) : Prop :=
  12 * x + 8 * y + 5 * z + 10 * w = 220 ∧
  x / 4 = y / 3 ∧
  x / 4 = z / 2 ∧
  x / 4 = w / 1

theorem action_figure_prices :
  ∃ x y z w : ℝ, prices x y z w ∧
    x = 220 / 23 ∧
    y = (3 / 4) * (220 / 23) ∧
    z = (1 / 2) * (220 / 23) ∧
    w = (1 / 4) * (220 / 23) :=
  sorry

end action_figure_prices_l48_48307


namespace find_m_and_other_root_l48_48902

theorem find_m_and_other_root (m : ℝ) (r : ℝ) :
    (∃ x : ℝ, x^2 + m*x - 2 = 0) ∧ (x = -1) → (m = -1 ∧ r = 2) :=
by
  sorry

end find_m_and_other_root_l48_48902


namespace simplify_expression1_simplify_expression2_l48_48950

variable {a b x y : ℝ}

theorem simplify_expression1 : 3 * a - 5 * b - 2 * a + b = a - 4 * b :=
by sorry

theorem simplify_expression2 : 4 * x^2 + 5 * x * y - 2 * (2 * x^2 - x * y) = 7 * x * y :=
by sorry

end simplify_expression1_simplify_expression2_l48_48950


namespace arithmetic_mean_six_expressions_l48_48019

theorem arithmetic_mean_six_expressions (x : ℝ) :
  (x + 10 + 17 + 2 * x + 15 + 2 * x + 6 + 3 * x - 5) / 6 = 30 →
  x = 137 / 8 :=
by
  sorry

end arithmetic_mean_six_expressions_l48_48019


namespace daniel_age_is_13_l48_48113

-- Define Aunt Emily's age
def aunt_emily_age : ℕ := 48

-- Define Brianna's age as a third of Aunt Emily's age
def brianna_age : ℕ := aunt_emily_age / 3

-- Define that Daniel's age is 3 years less than Brianna's age
def daniel_age : ℕ := brianna_age - 3

-- Theorem to prove Daniel's age is 13 given the conditions
theorem daniel_age_is_13 :
  brianna_age = aunt_emily_age / 3 →
  daniel_age = brianna_age - 3 →
  daniel_age = 13 :=
  sorry

end daniel_age_is_13_l48_48113


namespace paul_packed_total_toys_l48_48176

def toys_in_box : ℕ := 8
def number_of_boxes : ℕ := 4
def total_toys_packed (toys_in_box number_of_boxes : ℕ) : ℕ := toys_in_box * number_of_boxes

theorem paul_packed_total_toys :
  total_toys_packed toys_in_box number_of_boxes = 32 :=
by
  sorry

end paul_packed_total_toys_l48_48176


namespace ellipse_graph_equivalence_l48_48870

theorem ellipse_graph_equivalence :
  ∀ x y : ℝ, x^2 + 4 * y^2 - 6 * x + 8 * y + 9 = 0 ↔ (x - 3)^2 / 4 + (y + 1)^2 / 1 = 1 := by
  sorry

end ellipse_graph_equivalence_l48_48870


namespace distribute_4_balls_into_4_boxes_l48_48281

noncomputable def distribute_indistinguishable_balls (n k : ℕ) : ℕ :=
  binomial (n + k - 1) (k - 1)

theorem distribute_4_balls_into_4_boxes :
  distribute_indistinguishable_balls 4 4 = 35 :=
by
  -- We use a binomial coefficient to calculate the number of ways
  -- to distribute 4 indistinguishable balls into 4 distinguishable boxes
  sorry

end distribute_4_balls_into_4_boxes_l48_48281


namespace find_n_l48_48411

def sum_for (x : ℕ) : ℕ :=
  if x > 1 then (List.range (2*x)).sum else 0

theorem find_n (n : ℕ) (h : n * (sum_for 4) = 360) : n = 10 :=
by
  sorry

end find_n_l48_48411


namespace gas_pressure_inversely_proportional_l48_48111

variable {T : Type} [Nonempty T]

theorem gas_pressure_inversely_proportional
  (P : T → ℝ) (V : T → ℝ)
  (h_inv : ∀ t, P t * V t = 24) -- Given that pressure * volume = k where k = 24
  (t₀ t₁ : T)
  (hV₀ : V t₀ = 3) (hP₀ : P t₀ = 8) -- Initial condition: volume = 3 liters, pressure = 8 kPa
  (hV₁ : V t₁ = 6) -- New condition: volume = 6 liters
  : P t₁ = 4 := -- We need to prove that the new pressure is 4 kPa
by 
  sorry

end gas_pressure_inversely_proportional_l48_48111


namespace alice_score_record_l48_48425

def total_points : ℝ := 72
def average_points_others : ℝ := 4.7
def others_count : ℕ := 7

def total_points_others : ℝ := others_count * average_points_others
def alice_points : ℝ := total_points - total_points_others

theorem alice_score_record : alice_points = 39.1 :=
by {
  -- Proof should be inserted here
  sorry
}

end alice_score_record_l48_48425


namespace sin_150_eq_half_l48_48677

theorem sin_150_eq_half :
  sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48677


namespace least_number_to_subtract_l48_48477

theorem least_number_to_subtract (n d : ℕ) (n_val : n = 13602) (d_val : d = 87) : 
  ∃ r, (n - r) % d = 0 ∧ r = 30 := by
  sorry

end least_number_to_subtract_l48_48477


namespace ratio_green_students_l48_48048

/-- 
Miss Molly surveyed her class of 30 students about their favorite color.
Some portion of the class answered green, one-third of the girls answered pink,
and the rest of the class answered yellow. There are 18 girls in the class, 
and 9 students like yellow best. Prove that the ratio of students who answered green 
to the total number of students is 1:2.
-/
theorem ratio_green_students (total_students girls pink_ratio yellow_best green_students : ℕ)
    (h_total : total_students = 30)
    (h_girls : girls = 18)
    (h_pink_ratio : pink_ratio = girls / 3)
    (h_yellow : yellow_best = 9)
    (h_green : green_students = total_students - (pink_ratio + yellow_best)) :
  green_students / total_students = 1 / 2 :=
  sorry

end ratio_green_students_l48_48048


namespace weight_problem_l48_48848

theorem weight_problem (w1 w2 w3 : ℝ) (h1 : w1 + w2 + w3 = 100)
  (h2 : w1 + 2 * w2 + w3 = 101) (h3 : w1 + w2 + 2 * w3 = 102) : 
  w1 ≥ 90 ∨ w2 ≥ 90 ∨ w3 ≥ 90 :=
by
  sorry

end weight_problem_l48_48848


namespace soda_difference_l48_48491

-- Define the number of regular soda bottles
def R : ℕ := 79

-- Define the number of diet soda bottles
def D : ℕ := 53

-- The theorem that states the number of regular soda bottles minus the number of diet soda bottles is 26
theorem soda_difference : R - D = 26 := 
by
  sorry

end soda_difference_l48_48491


namespace mr_brown_selling_price_l48_48050

noncomputable def initial_price : ℝ := 100000
noncomputable def profit_percentage : ℝ := 0.10
noncomputable def loss_percentage : ℝ := 0.10

def selling_price_mr_brown (initial_price profit_percentage : ℝ) : ℝ :=
  initial_price * (1 + profit_percentage)

def selling_price_to_friend (selling_price_mr_brown loss_percentage : ℝ) : ℝ :=
  selling_price_mr_brown * (1 - loss_percentage)

theorem mr_brown_selling_price :
  selling_price_to_friend (selling_price_mr_brown initial_price profit_percentage) loss_percentage = 99000 :=
by
  sorry

end mr_brown_selling_price_l48_48050


namespace distinct_solutions_eq_four_l48_48258

theorem distinct_solutions_eq_four : ∃! (x : ℝ), abs (x - abs (3 * x + 2)) = 4 :=
by sorry

end distinct_solutions_eq_four_l48_48258


namespace mayo_bottle_count_l48_48628

-- Define the given ratio and the number of ketchup bottles
def ratio_ketchup : ℕ := 3
def ratio_mustard : ℕ := 3
def ratio_mayo : ℕ := 2
def num_ketchup_bottles : ℕ := 6

-- Define the proof problem: The number of mayo bottles
theorem mayo_bottle_count :
  (num_ketchup_bottles / ratio_ketchup) * ratio_mayo = 4 :=
by sorry

end mayo_bottle_count_l48_48628


namespace fourth_grade_students_l48_48112

theorem fourth_grade_students:
  (initial_students = 35) →
  (first_semester_left = 6) →
  (first_semester_joined = 4) →
  (first_semester_transfers = 2) →
  (second_semester_left = 3) →
  (second_semester_joined = 7) →
  (second_semester_transfers = 2) →
  final_students = initial_students - first_semester_left + first_semester_joined - second_semester_left + second_semester_joined :=
  sorry

end fourth_grade_students_l48_48112


namespace average_people_per_hour_l48_48029

theorem average_people_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) (total_hours : ℕ) (average_per_hour : ℕ) :
  total_people = 3000 ∧ days = 5 ∧ hours_per_day = 24 ∧ total_hours = days * hours_per_day ∧ average_per_hour = total_people / total_hours → 
  average_per_hour = 25 :=
by
  sorry

end average_people_per_hour_l48_48029


namespace pencils_and_pens_cost_l48_48339

theorem pencils_and_pens_cost (p q : ℝ)
  (h1 : 8 * p + 3 * q = 5.60)
  (h2 : 2 * p + 5 * q = 4.25) :
  3 * p + 4 * q = 9.68 :=
sorry

end pencils_and_pens_cost_l48_48339


namespace nth_term_150_l48_48149

-- Conditions
def a : ℕ := 2
def d : ℕ := 5
def arithmetic_sequence (n : ℕ) : ℕ := a + (n - 1) * d

-- Question and corresponding answer proof
theorem nth_term_150 : arithmetic_sequence 150 = 747 := by
  sorry

end nth_term_150_l48_48149


namespace solve_for_t_l48_48008

variable (A P0 r t : ℝ)

theorem solve_for_t (h : A = P0 * Real.exp (r * t)) : t = (Real.log (A / P0)) / r :=
  by
  sorry

end solve_for_t_l48_48008


namespace frac_val_of_x_y_l48_48572

theorem frac_val_of_x_y (x y : ℝ) (h: (4 : ℝ) < (2 * x - 3 * y) / (2 * x + 3 * y) ∧ (2 * x - 3 * y) / (2 * x + 3 * y) < 8) (ht: ∃ t : ℤ, x = t * y) : x / y = -2 := 
by
  sorry

end frac_val_of_x_y_l48_48572


namespace original_commercial_length_l48_48974

theorem original_commercial_length (x : ℝ) (h : 0.70 * x = 21) : x = 30 := sorry

end original_commercial_length_l48_48974


namespace fixed_point_for_all_k_l48_48893

theorem fixed_point_for_all_k (k : ℝ) : (5, 225) ∈ { p : ℝ × ℝ | ∃ k : ℝ, p.snd = 9 * p.fst^2 + k * p.fst - 5 * k } :=
by
  sorry

end fixed_point_for_all_k_l48_48893


namespace find_x_l48_48289

theorem find_x (x : ℝ) :
  let P1 := (2, 10)
  let P2 := (6, 2)
  
  -- Slope of the line joining (2, 10) and (6, 2)
  let slope12 := (P2.2 - P1.2) / (P2.1 - P1.1)
  
  -- Slope of the line joining (2, 10) and (x, -3)
  let P3 := (x, -3)
  let slope13 := (P3.2 - P1.2) / (P3.1 - P1.1)
  
  -- Condition that both slopes are equal
  slope12 = slope13
  
  -- To Prove: x must be 8.5
  → x = 8.5 :=
sorry

end find_x_l48_48289


namespace solve_for_x_l48_48835

theorem solve_for_x :
  ∃ (x : ℝ), x ≠ 0 ∧ (5 * x)^10 = (10 * x)^5 ∧ x = 2 / 5 :=
by
  sorry

end solve_for_x_l48_48835


namespace sarah_rye_flour_l48_48948

-- Definitions
variables (b c p t r : ℕ)

-- Conditions
def condition1 : Prop := b = 10
def condition2 : Prop := c = 3
def condition3 : Prop := p = 2
def condition4 : Prop := t = 20

-- Proposition to prove
theorem sarah_rye_flour : condition1 b → condition2 c → condition3 p → condition4 t → r = t - (b + c + p) → r = 5 :=
by
  intros h1 h2 h3 h4 hr
  rw [h1, h2, h3, h4] at hr
  exact hr

end sarah_rye_flour_l48_48948


namespace find_positive_integer_l48_48088

variable (z : ℕ)

theorem find_positive_integer
  (h1 : (4 * z)^2 - z = 2345)
  (h2 : 0 < z) :
  z = 7 :=
sorry

end find_positive_integer_l48_48088


namespace bugs_eat_same_flowers_l48_48174

theorem bugs_eat_same_flowers (num_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) 
  (h1 : num_bugs = 3) (h2 : total_flowers = 6) (h3 : flowers_per_bug = total_flowers / num_bugs) : 
  flowers_per_bug = 2 :=
by
  sorry

end bugs_eat_same_flowers_l48_48174


namespace rebus_system_solution_l48_48184

theorem rebus_system_solution :
  ∃ (M A H P h : ℕ), 
  (M > 0) ∧ (P > 0) ∧ 
  (M ≠ A) ∧ (M ≠ H) ∧ (M ≠ P) ∧ (M ≠ h) ∧
  (A ≠ H) ∧ (A ≠ P) ∧ (A ≠ h) ∧ 
  (H ≠ P) ∧ (H ≠ h) ∧ (P ≠ h) ∧
  ((M * 10 + A) * (M * 10 + A) = M * 100 + H * 10 + P) ∧ 
  ((A * 10 + M) * (A * 10 + M) = P * 100 + h * 10 + M) ∧ 
  (((M = 1) ∧ (A = 3) ∧ (H = 6) ∧ (P = 9) ∧ (h = 6)) ∨
   ((M = 3) ∧ (A = 1) ∧ (H = 9) ∧ (P = 6) ∧ (h = 9))) :=
by
  sorry

end rebus_system_solution_l48_48184


namespace probability_excellent_probability_good_or_better_l48_48368

noncomputable def total_selections : ℕ := 10
noncomputable def total_excellent_selections : ℕ := 1
noncomputable def total_good_or_better_selections : ℕ := 7
noncomputable def P_excellent : ℚ := 1 / 10
noncomputable def P_good_or_better : ℚ := 7 / 10

theorem probability_excellent (total_selections total_excellent_selections : ℕ) :
  (total_excellent_selections : ℚ) / total_selections = 1 / 10 := by
  sorry

theorem probability_good_or_better (total_selections total_good_or_better_selections : ℕ) :
  (total_good_or_better_selections : ℚ) / total_selections = 7 / 10 := by
  sorry

end probability_excellent_probability_good_or_better_l48_48368


namespace smallest_x_for_gx_eq_g1458_l48_48388

noncomputable def g : ℝ → ℝ := sorry -- You can define the function later.

theorem smallest_x_for_gx_eq_g1458 :
  (∀ x : ℝ, x > 0 → g (3 * x) = 4 * g x) ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → g x = 2 - 2 * |x - 2|)
  → ∃ x : ℝ, x ≥ 0 ∧ g x = g 1458 ∧ ∀ y : ℝ, y ≥ 0 ∧ g y = g 1458 → x ≤ y ∧ x = 162 := 
by
  sorry

end smallest_x_for_gx_eq_g1458_l48_48388


namespace cricket_team_members_l48_48995

-- Define variables and conditions
variable (n : ℕ) -- let n be the number of team members
variable (T : ℕ) -- let T be the total age of the team
variable (average_team_age : ℕ := 24) -- given average age of the team
variable (wicket_keeper_age : ℕ := average_team_age + 3) -- wicket keeper is 3 years older
variable (remaining_players_average_age : ℕ := average_team_age - 1) -- remaining players' average age

-- Given condition which relates to the total age
axiom total_age_condition : T = average_team_age * n

-- Given condition for the total age of remaining players
axiom remaining_players_total_age : T - 24 - 27 = remaining_players_average_age * (n - 2)

-- Prove the number of members in the cricket team
theorem cricket_team_members : n = 5 :=
by
  sorry

end cricket_team_members_l48_48995


namespace cube_sum_eq_2702_l48_48441

noncomputable def x : ℝ := (2 + Real.sqrt 3) / (2 - Real.sqrt 3)
noncomputable def y : ℝ := (2 - Real.sqrt 3) / (2 + Real.sqrt 3)

theorem cube_sum_eq_2702 : x^3 + y^3 = 2702 :=
by
  sorry

end cube_sum_eq_2702_l48_48441


namespace find_num_students_B_l48_48616

-- Given conditions as definitions
def num_students_A : ℕ := 24
def avg_weight_A : ℚ := 40
def avg_weight_B : ℚ := 35
def avg_weight_class : ℚ := 38

-- The total weight for sections A and B
def total_weight_A : ℚ := num_students_A * avg_weight_A
def total_weight_B (x: ℕ) : ℚ := x * avg_weight_B

-- The number of students in section B
noncomputable def num_students_B : ℕ := 16

-- The proof problem: Prove that number of students in section B is 16
theorem find_num_students_B (x: ℕ) (h: (total_weight_A + total_weight_B x) / (num_students_A + x) = avg_weight_class) : 
  x = 16 :=
by
  sorry

end find_num_students_B_l48_48616


namespace sin_150_equals_half_l48_48691

noncomputable def sin_150_eq_half : Prop := 
  sin (150 * real.pi / 180) = 1 / 2

theorem sin_150_equals_half : sin_150_eq_half :=
  by sorry

end sin_150_equals_half_l48_48691


namespace parabola_line_non_intersect_l48_48172

theorem parabola_line_non_intersect (r s : ℝ) (Q : ℝ × ℝ) (P : ℝ → ℝ)
  (hP : ∀ x, P x = x^2)
  (hQ : Q = (10, 6))
  (h_cond : ∀ m : ℝ, ¬∃ x : ℝ, (Q.snd - 6 = m * (Q.fst - 10)) ∧ (P x = x^2) ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end parabola_line_non_intersect_l48_48172


namespace paint_snake_l48_48469

theorem paint_snake (num_cubes : ℕ) (paint_per_cube : ℕ) (end_paint : ℕ) (total_paint : ℕ) 
  (h_cubes : num_cubes = 2016)
  (h_paint_per_cube : paint_per_cube = 60)
  (h_end_paint : end_paint = 20)
  (h_total_paint : total_paint = 121000) :
  total_paint = (num_cubes * paint_per_cube) + 2 * end_paint :=
by
  rw [h_cubes, h_paint_per_cube, h_end_paint]
  sorry

end paint_snake_l48_48469


namespace part_a_part_b_l48_48941

-- Define the problem as described
noncomputable def can_transform_to_square (figure : Type) (parts : ℕ) (all_triangles : Bool) : Bool :=
sorry  -- This is a placeholder for the actual implementation

-- The figure satisfies the condition to cut into four parts and rearrange into a square
theorem part_a (figure : Type) : can_transform_to_square figure 4 false = true :=
sorry

-- The figure satisfies the condition to cut into five triangular parts and rearrange into a square
theorem part_b (figure : Type) : can_transform_to_square figure 5 true = true :=
sorry

end part_a_part_b_l48_48941


namespace sum_interior_angles_polygon_l48_48595

theorem sum_interior_angles_polygon (n : ℕ) (h : 180 * (n - 2) = 1440) :
  180 * ((n + 3) - 2) = 1980 := by
  sorry

end sum_interior_angles_polygon_l48_48595


namespace tank_full_volume_l48_48235

theorem tank_full_volume (x : ℝ) (h1 : 5 / 6 * x > 0) (h2 : 5 / 6 * x - 15 = 1 / 3 * x) : x = 30 :=
by
  -- The proof is omitted as per the requirement.
  sorry

end tank_full_volume_l48_48235


namespace hyperbola_focal_distance_solution_l48_48147

-- Definitions corresponding to the problem conditions
def hyperbola_equation (x y m : ℝ) :=
  x^2 / m - y^2 / 6 = 1

def focal_distance (c : ℝ) := 2 * c

-- Theorem statement to prove m = 3 based on given conditions
theorem hyperbola_focal_distance_solution (m : ℝ) (h_eq : ∀ x y : ℝ, hyperbola_equation x y m) (h_focal : focal_distance 3 = 6) :
  m = 3 :=
by {
  -- sorry is used here as a placeholder for the actual proof steps
  sorry
}

end hyperbola_focal_distance_solution_l48_48147


namespace largest_number_of_HCF_LCM_l48_48062

theorem largest_number_of_HCF_LCM (HCF : ℕ) (k1 k2 : ℕ) (n1 n2 : ℕ) 
  (hHCF : HCF = 50)
  (hk1 : k1 = 11) 
  (hk2 : k2 = 12) 
  (hn1 : n1 = HCF * k1) 
  (hn2 : n2 = HCF * k2) :
  max n1 n2 = 600 := by
  sorry

end largest_number_of_HCF_LCM_l48_48062


namespace root_probability_l48_48935

-- Definition of the binomial distribution and related probability functions
def binomial_prob (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

-- Condition
def X_binomial_distribution : Prop := (X : ℕ) → binomial_prob 5 (1/2) X ≠ 0

-- Theorem proving the root probability
theorem root_probability {X : ℕ} (hX : X_binomial_distribution) : 
  (∑ k in finset.range 5, binomial_prob 5 (1/2) k) = 31/32 :=
begin
  -- We leave the proof to a more detailed development {specific steps required}
  sorry
end

end root_probability_l48_48935


namespace As_annual_income_l48_48959

theorem As_annual_income :
  let Cm := 14000
  let Bm := Cm + 0.12 * Cm
  let Am := (5 / 2) * Bm
  Am * 12 = 470400 := by
  sorry

end As_annual_income_l48_48959


namespace sin_150_eq_half_l48_48697

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by 
  sorry

end sin_150_eq_half_l48_48697


namespace dozen_pen_cost_l48_48594

-- Definitions based on the conditions
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x
def total_cost (x : ℝ) (y : ℝ) : ℝ := 3 * cost_of_pen x + y * cost_of_pencil x

open Classical
noncomputable def cost_dozen_pens (x : ℝ) : ℝ := 12 * cost_of_pen x

theorem dozen_pen_cost (x y : ℝ) (h : total_cost x y = 150) : cost_dozen_pens x = 60 * x :=
by
  sorry

end dozen_pen_cost_l48_48594


namespace second_parentheses_expression_eq_zero_l48_48759

def custom_op (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem second_parentheses_expression_eq_zero :
  custom_op (Real.sqrt 6) (Real.sqrt 6) = 0 := by
  sorry

end second_parentheses_expression_eq_zero_l48_48759


namespace price_of_basketball_l48_48116

-- Problem definitions based on conditions
def price_of_soccer_ball (x : ℝ) : Prop :=
  let price_of_basketball := 2 * x
  x + price_of_basketball = 186

theorem price_of_basketball (x : ℝ) (h : price_of_soccer_ball x) : 2 * x = 124 :=
by
  sorry

end price_of_basketball_l48_48116


namespace brenda_age_problem_l48_48495

variable (A B J : Nat)

theorem brenda_age_problem
  (h1 : A = 4 * B) 
  (h2 : J = B + 9) 
  (h3 : A = J) : 
  B = 3 := 
by 
  sorry

end brenda_age_problem_l48_48495


namespace sin_150_eq_half_l48_48706

theorem sin_150_eq_half :
  Float.sin (150 * Float.pi / 180) = 1 / 2 := sorry

end sin_150_eq_half_l48_48706


namespace general_formula_arithmetic_sum_of_geometric_terms_l48_48529

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 2 = 2 ∧ a 5 = 8

noncomputable def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℤ) : Prop :=
  b 1 = 1 ∧ b 2 + b 3 = a 4

noncomputable def sum_of_terms (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, T n = (2:ℝ)^n - 1

theorem general_formula_arithmetic (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∀ n, a n = 2 * n - 2 :=
sorry

theorem sum_of_geometric_terms (a : ℕ → ℤ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h : arithmetic_sequence a) (h2 : geometric_sequence b a) :
  sum_of_terms T b :=
sorry

end general_formula_arithmetic_sum_of_geometric_terms_l48_48529


namespace quadratic_roots_eq_l48_48042

theorem quadratic_roots_eq (a : ℝ) (b : ℝ) :
  (∀ x, (2 * x^2 - 3 * x - 8 = 0) → 
         ((x + 3)^2 + a * (x + 3) + b = 0)) → 
  b = 9.5 :=
by
  sorry

end quadratic_roots_eq_l48_48042


namespace find_x_l48_48315

-- Define the vectors and collinearity condition
def vector_a : ℝ × ℝ := (3, 6)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 8)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (b.1 = k * a.1) ∧ (b.2 = k * a.2)

-- Define the proof problem
theorem find_x (x : ℝ) (h : collinear vector_a (vector_b x)) : x = 4 :=
  sorry

end find_x_l48_48315


namespace jellybean_problem_l48_48989

theorem jellybean_problem:
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  black + green + orange = 27 →
  green - orange = 1 :=
by
  intros black green orange h_black h_green h_total
  sorry

end jellybean_problem_l48_48989


namespace find_t_l48_48877

theorem find_t : ∃ t, ∀ (x y : ℝ), (x, y) = (0, 1) ∨ (x, y) = (-6, -3) → (t, 7) ∈ {p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ ((0, 1) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) ∧ ((-6, -3) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) } → t = 9 :=
by
  sorry

end find_t_l48_48877


namespace find_d_l48_48890

theorem find_d (x y d : ℕ) (h_midpoint : (1 + 5)/2 = 3 ∧ (3 + 11)/2 = 7) 
  : x + y = d ↔ d = 10 := 
sorry

end find_d_l48_48890


namespace smaller_than_negative_one_l48_48243

theorem smaller_than_negative_one :
  ∃ x ∈ ({0, -1/2, 1, -2} : Set ℝ), x < -1 ∧ x = -2 :=
by
  -- the proof part is skipped
  sorry

end smaller_than_negative_one_l48_48243


namespace product_of_squares_l48_48608

theorem product_of_squares (a_1 a_2 a_3 b_1 b_2 b_3 : ℕ) (N : ℕ) (h1 : (a_1 * b_1)^2 = N) (h2 : (a_2 * b_2)^2 = N) (h3 : (a_3 * b_3)^2 = N) 
: (a_1^2 * b_1^2) = 36 ∨  (a_2^2 * b_2^2) = 36 ∨ (a_3^2 * b_3^2) = 36:= 
sorry

end product_of_squares_l48_48608


namespace commutative_not_associative_l48_48263

variable (k : ℝ) (h_k : 0 < k)

noncomputable def star (x y : ℝ) : ℝ := (x * y + k) / (x + y + k)

theorem commutative (x y : ℝ) (h_x : 0 < x) (h_y : 0 < y) :
  star k x y = star k y x :=
by sorry

theorem not_associative (x y z : ℝ) (h_x : 0 < x) (h_y : 0 < y) (h_z : 0 < z) :
  ¬(star k (star k x y) z = star k x (star k y z)) :=
by sorry

end commutative_not_associative_l48_48263


namespace original_price_of_boots_l48_48253

theorem original_price_of_boots (P : ℝ) (h : P * 0.80 = 72) : P = 90 :=
by 
  sorry

end original_price_of_boots_l48_48253


namespace new_price_after_increase_l48_48756

def original_price (y : ℝ) : Prop := 2 * y = 540

theorem new_price_after_increase (y : ℝ) (h : original_price y) : 1.3 * y = 351 :=
by sorry

end new_price_after_increase_l48_48756


namespace dubblefud_red_balls_zero_l48_48767

theorem dubblefud_red_balls_zero
  (R B G : ℕ)
  (H1 : 2^R * 4^B * 5^G = 16000)
  (H2 : B = G) : R = 0 :=
sorry

end dubblefud_red_balls_zero_l48_48767


namespace last_number_remaining_l48_48166

theorem last_number_remaining :
  (∃ f : ℕ → ℕ, ∃ n : ℕ, (∀ k < n, f (2 * k) = 2 * k + 2 ∧
                         ∀ k < n, f (2 * k + 1) = 2 * k + 1 + 2^(k+1)) ∧ 
                         n = 200 ∧ f (2 * n) = 128) :=
sorry

end last_number_remaining_l48_48166


namespace discount_difference_l48_48387

theorem discount_difference (bill_amt : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) :
  bill_amt = 12000 → d1 = 0.42 → d2 = 0.35 → d3 = 0.05 →
  (bill_amt * (1 - d2) * (1 - d3) - bill_amt * (1 - d1) = 450) :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end discount_difference_l48_48387


namespace min_value_at_2_l48_48068

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_at_2 : ∃ x : ℝ, f x = 2 :=
sorry

end min_value_at_2_l48_48068


namespace sum_of_elements_in_T_l48_48309

noncomputable def digit_sum : ℕ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 504
noncomputable def repeating_sum : ℕ := digit_sum * 1111
noncomputable def sum_T : ℚ := repeating_sum / 9999

theorem sum_of_elements_in_T : sum_T = 2523 := by
  sorry

end sum_of_elements_in_T_l48_48309


namespace unique_solution_a_eq_sqrt3_l48_48755

theorem unique_solution_a_eq_sqrt3 (a : ℝ) :
  (∃! x : ℝ, x^2 - a * |x| + a^2 - 3 = 0) ↔ a = -Real.sqrt 3 := by
  sorry

end unique_solution_a_eq_sqrt3_l48_48755


namespace find_b_l48_48880

theorem find_b (b : ℕ) (h1 : 40 < b) (h2 : b < 120) 
    (h3 : b % 4 = 3) (h4 : b % 5 = 3) (h5 : b % 6 = 3) : 
    b = 63 := by
  sorry

end find_b_l48_48880


namespace sin_150_eq_half_l48_48667

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48667


namespace trig_identity_proof_l48_48132

theorem trig_identity_proof (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) :
  Real.sin (2 * α - π / 6) + Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end trig_identity_proof_l48_48132


namespace units_digit_p_plus_2_l48_48272

theorem units_digit_p_plus_2 {p : ℕ} 
  (h1 : p % 2 = 0) 
  (h2 : p % 10 ≠ 0) 
  (h3 : (p^3 % 10) = (p^2 % 10)) : 
  (p + 2) % 10 = 8 :=
sorry

end units_digit_p_plus_2_l48_48272


namespace twelve_months_game_probability_l48_48079

/-- The card game "Twelve Months" involves turning over cards according to a set of rules.
Given the rules, we are asked to find the probability that all 12 columns of cards can be fully turned over. -/
def twelve_months_probability : ℚ :=
  1 / 12

theorem twelve_months_game_probability :
  twelve_months_probability = 1 / 12 :=
by
  -- The conditions and their representations are predefined.
  sorry

end twelve_months_game_probability_l48_48079


namespace total_cost_sandwiches_and_sodas_l48_48631

theorem total_cost_sandwiches_and_sodas :
  let price_sandwich : Real := 2.49
  let price_soda : Real := 1.87
  let quantity_sandwich : ℕ := 2
  let quantity_soda : ℕ := 4
  (quantity_sandwich * price_sandwich + quantity_soda * price_soda) = 12.46 := 
by
  sorry

end total_cost_sandwiches_and_sodas_l48_48631


namespace digit_D_value_l48_48343

/- The main conditions are:
1. A, B, C, D are digits (0 through 9)
2. Addition equation: AB + CA = D0
3. Subtraction equation: AB - CA = 00
-/

theorem digit_D_value (A B C D : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (hD : D < 10)
  (add_eq : 10 * A + B + 10 * C + A = 10 * D + 0)
  (sub_eq : 10 * A + B - (10 * C + A) = 0) :
  D = 1 :=
sorry

end digit_D_value_l48_48343


namespace new_class_average_l48_48914

theorem new_class_average (total_students : ℕ) (students_group1 : ℕ) (avg1 : ℝ) (students_group2 : ℕ) (avg2 : ℝ) : 
  total_students = 40 → students_group1 = 28 → avg1 = 68 → students_group2 = 12 → avg2 = 77 → 
  ((students_group1 * avg1 + students_group2 * avg2) / total_students) = 70.7 :=
by
  sorry

end new_class_average_l48_48914


namespace speedster_convertibles_count_l48_48991

-- Definitions of conditions
def total_inventory (T : ℕ) : Prop := (T / 3) = 60
def number_of_speedsters (T S : ℕ) : Prop := S = (2 / 3) * T
def number_of_convertibles (S C : ℕ) : Prop := C = (4 / 5) * S

-- Primary statement to prove
theorem speedster_convertibles_count (T S C : ℕ) (h1 : total_inventory T) (h2 : number_of_speedsters T S) (h3 : number_of_convertibles S C) : C = 96 :=
by
  -- Conditions and given values are defined
  sorry

end speedster_convertibles_count_l48_48991


namespace marie_erasers_l48_48319

theorem marie_erasers (initial_erasers : ℕ) (lost_erasers : ℕ) (final_erasers : ℕ) 
  (h1 : initial_erasers = 95) (h2 : lost_erasers = 42) : final_erasers = 53 :=
by
  sorry

end marie_erasers_l48_48319


namespace value_of_c_l48_48187

variables (a b c : ℝ)

theorem value_of_c :
  a + b = 3 ∧
  a * c + b = 18 ∧
  b * c + a = 6 →
  c = 7 :=
by
  intro h
  sorry

end value_of_c_l48_48187


namespace triangular_region_area_l48_48240

theorem triangular_region_area : 
  ∀ (x y : ℝ),  (3 * x + 4 * y = 12) →
  (0 ≤ x ∧ 0 ≤ y) →
  ∃ (A : ℝ), A = 6 := 
by 
  sorry

end triangular_region_area_l48_48240


namespace max_square_area_in_rhombus_l48_48826

noncomputable def side_length_triangle := 10
noncomputable def height_triangle := Real.sqrt (side_length_triangle^2 - (side_length_triangle / 2)^2)
noncomputable def diag_long := 2 * height_triangle
noncomputable def diag_short := side_length_triangle
noncomputable def side_square := diag_short / Real.sqrt 2
noncomputable def area_square := side_square^2

theorem max_square_area_in_rhombus :
  area_square = 50 := by sorry

end max_square_area_in_rhombus_l48_48826


namespace shadow_building_length_l48_48845

-- Definitions based on conditions
def height_flagpole : ℚ := 18
def shadow_flagpole : ℚ := 45
def height_building : ℚ := 24

-- Question to be proved
theorem shadow_building_length : 
  ∃ (shadow_building : ℚ), 
    (height_flagpole / shadow_flagpole = height_building / shadow_building) ∧ 
    shadow_building = 60 :=
by
  sorry

end shadow_building_length_l48_48845


namespace exists_infinitely_many_n_odd_floor_l48_48484

def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem exists_infinitely_many_n_odd_floor (α : ℝ) : 
  ∃ᶠ n in at_top, odd ⌊n^2 * α⌋ := sorry

end exists_infinitely_many_n_odd_floor_l48_48484


namespace jake_bitcoins_l48_48427

theorem jake_bitcoins (initial : ℕ) (donation1 : ℕ) (fraction : ℕ) (multiplier : ℕ) (donation2 : ℕ) :
  initial = 80 →
  donation1 = 20 →
  fraction = 2 →
  multiplier = 3 →
  donation2 = 10 →
  (initial - donation1) / fraction * multiplier - donation2 = 80 :=
by
  sorry

end jake_bitcoins_l48_48427


namespace ratio_of_p_to_r_l48_48818

theorem ratio_of_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : r / s = 4 / 3) 
  (h3 : s / q = 1 / 8) : 
  p / r = 15 / 2 := 
by 
  sorry

end ratio_of_p_to_r_l48_48818


namespace max_n_inequality_l48_48728

open Finset

theorem max_n_inequality (n : ℕ) (hn : n = 99) :
  ∀ A : Finset ℕ, (A ⊆ range (n + 1) ∧ 10 ≤ A.card) → 
  ∃ a b ∈ A, a ≠ b ∧ |a - b| ≤ 10 := 
by
  intro A hA
  sorry

end max_n_inequality_l48_48728


namespace car_speed_first_hour_l48_48348

theorem car_speed_first_hour
  (x : ℕ)
  (speed_second_hour : ℕ := 80)
  (average_speed : ℕ := 90)
  (total_time : ℕ := 2)
  (h : average_speed = (x + speed_second_hour) / total_time) :
  x = 100 :=
by
  sorry

end car_speed_first_hour_l48_48348


namespace simplify_fraction_l48_48750

variable (x y : ℝ)

theorem simplify_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + 1/y ≠ 0) (h2 : y + 1/x ≠ 0) : 
  (x + 1/y) / (y + 1/x) = x / y :=
sorry

end simplify_fraction_l48_48750


namespace inequality1_solution_inequality2_solution_l48_48583

open Real

-- First problem: proving the solution set for x + |2x + 3| >= 2
theorem inequality1_solution (x : ℝ) : x + abs (2 * x + 3) >= 2 ↔ (x <= -5 ∨ x >= -1/3) := 
sorry

-- Second problem: proving the solution set for |x - 1| - |x - 5| < 2
theorem inequality2_solution (x : ℝ) : abs (x - 1) - abs (x - 5) < 2 ↔ x < 4 :=
sorry

end inequality1_solution_inequality2_solution_l48_48583


namespace option_b_correct_l48_48982

theorem option_b_correct (a : ℝ) : (-a)^3 / (-a)^2 = -a :=
by sorry

end option_b_correct_l48_48982


namespace common_root_values_max_n_and_a_range_l48_48276

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a+1) * x - 4 * (a+5)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x + 5

-- Part 1
theorem common_root_values (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ g a x = 0) → a = -9/16 ∨ a = -6 ∨ a = -4 ∨ a = 0 :=
sorry

-- Part 2
theorem max_n_and_a_range (a : ℝ) (m n : ℕ) (x0 : ℝ) :
  (m < n ∧ (m : ℝ) < x0 ∧ x0 < (n : ℝ) ∧ f a x0 < 0 ∧ g a x0 < 0) →
  n = 4 ∧ -1 ≤ a ∧ a ≤ -2/9 :=
sorry

end common_root_values_max_n_and_a_range_l48_48276


namespace potion_combinations_l48_48646

-- Definitions of conditions
def roots : Nat := 3
def minerals : Nat := 5
def incompatible_combinations : Nat := 2

-- Statement of the problem
theorem potion_combinations : (roots * minerals) - incompatible_combinations = 13 := by
  sorry

end potion_combinations_l48_48646


namespace smallest_value_among_options_l48_48410

theorem smallest_value_among_options (x : ℕ) (h : x = 9) :
    min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min ((x+3)/8) ((x-3)/8)))) = (3/4) :=
by
  sorry

end smallest_value_among_options_l48_48410


namespace surface_area_is_correct_volume_is_approximately_correct_l48_48239

noncomputable def surface_area_of_CXYZ (height : ℝ) (side_length : ℝ) : ℝ :=
  let area_CZX_CZY := 48
  let area_CXY := 9 * Real.sqrt 3
  let area_XYZ := 9 * Real.sqrt 15
  2 * area_CZX_CZY + area_CXY + area_XYZ

theorem surface_area_is_correct (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  surface_area_of_CXYZ height side_length = 96 + 9 * Real.sqrt 3 + 9 * Real.sqrt 15 :=
by
  sorry

noncomputable def volume_of_CXYZ (height : ℝ ) (side_length : ℝ) : ℝ :=
  -- Placeholder for the volume calculation approximation method.
  486

theorem volume_is_approximately_correct
  (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  volume_of_CXYZ height side_length = 486 :=
by
  sorry

end surface_area_is_correct_volume_is_approximately_correct_l48_48239


namespace arithmetic_sequence_term_l48_48161

theorem arithmetic_sequence_term (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 3 = 4) : a 10 = 18 :=
by
  sorry

end arithmetic_sequence_term_l48_48161


namespace min_value_of_f_l48_48311

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 4*x + 5) * (x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = -9 :=
  sorry

end min_value_of_f_l48_48311


namespace simplify_expression_l48_48089

theorem simplify_expression (x : ℝ) : (3 * x + 2) - 2 * (2 * x - 1) = 3 * x + 2 - 4 * x + 2 := 
by sorry

end simplify_expression_l48_48089


namespace rectangle_sides_l48_48887

theorem rectangle_sides (x y : ℕ) :
  (2 * x + 2 * y = x * y) →
  x > 0 →
  y > 0 →
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) ∨ (x = 4 ∧ y = 4) :=
by
  sorry

end rectangle_sides_l48_48887


namespace jill_total_phone_time_l48_48774

def phone_time : ℕ → ℕ
| 0 => 5
| (n + 1) => 2 * phone_time n

theorem jill_total_phone_time (n : ℕ) (h : n = 4) : 
  phone_time 0 + phone_time 1 + phone_time 2 + phone_time 3 + phone_time 4 = 155 :=
by
  cases h
  sorry

end jill_total_phone_time_l48_48774


namespace zhou_catches_shuttle_probability_l48_48361

-- Condition 1: Shuttle arrival time and duration
def shuttle_arrival_start : ℕ := 420 -- 7:00 AM in minutes since midnight
def shuttle_duration : ℕ := 15

-- Condition 2: Zhou's random arrival time window
def zhou_arrival_start : ℕ := 410 -- 6:50 AM in minutes since midnight
def zhou_arrival_end : ℕ := 465 -- 7:45 AM in minutes since midnight

-- Total time available for Zhou to arrive (55 minutes) 
def total_time : ℕ := zhou_arrival_end - zhou_arrival_start

-- Time window when Zhou needs to arrive to catch the shuttle (15 minutes)
def successful_time : ℕ := shuttle_arrival_start + shuttle_duration - shuttle_arrival_start

-- Calculate the probability that Zhou catches the shuttle
theorem zhou_catches_shuttle_probability : 
  (successful_time : ℚ) / total_time = 3 / 11 := 
by 
  -- We don't need the actual proof steps, just the statement
  sorry

end zhou_catches_shuttle_probability_l48_48361


namespace product_mnp_l48_48066

theorem product_mnp (a x y b : ℝ) (m n p : ℕ):
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x = 2 * a ^ 5 * (b ^ 5 - 2)) ∧
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x + 6 * a ^ 5 = (a ^ m * x - 2 * a ^ n) * (a ^ p * y - 3 * a ^ 3)) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  intros h
  sorry

end product_mnp_l48_48066


namespace difference_between_greatest_and_smallest_S_l48_48205

-- Conditions
def num_students := 47
def rows := 6
def columns := 8

-- The definition of position value calculation
def position_value (i j m n : ℕ) := i - m + (j - n)

-- The definition of S
def S (initial_empty final_empty : (ℕ × ℕ)) : ℤ :=
  let (i_empty, j_empty) := initial_empty
  let (i'_empty, j'_empty) := final_empty
  (i'_empty + j'_empty) - (i_empty + j_empty)

-- Main statement
theorem difference_between_greatest_and_smallest_S :
  let max_S := S (1, 1) (6, 8)
  let min_S := S (6, 8) (1, 1)
  max_S - min_S = 24 :=
sorry

end difference_between_greatest_and_smallest_S_l48_48205


namespace find_certain_number_l48_48825

-- Definitions of conditions from the problem
def greatest_number : ℕ := 10
def divided_1442_by_greatest_number_leaves_remainder := (1442 % greatest_number = 12)
def certain_number_mod_greatest_number (x : ℕ) := (x % greatest_number = 6)

-- Theorem statement
theorem find_certain_number (x : ℕ) (h1 : greatest_number = 10)
  (h2 : 1442 % greatest_number = 12)
  (h3 : certain_number_mod_greatest_number x) : x = 1446 :=
sorry

end find_certain_number_l48_48825


namespace chocolate_bar_min_breaks_l48_48091

theorem chocolate_bar_min_breaks (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∃ k, k = m * n - 1 := by
  sorry

end chocolate_bar_min_breaks_l48_48091


namespace coefficient_c_nonzero_l48_48605

-- We are going to define the given polynomial and its conditions
def P (x : ℝ) (a b c d e : ℝ) : ℝ :=
  x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

-- Given conditions
def five_x_intercepts (P : ℝ → ℝ) (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0 ∧ P x4 = 0 ∧ P x5 = 0

def double_root_at_zero (P : ℝ → ℝ) : Prop :=
  P 0 = 0 ∧ deriv P 0 = 0

-- Equivalent proof problem
theorem coefficient_c_nonzero (a b c d e : ℝ)
  (h1 : P 0 a b c d e = 0)
  (h2 : deriv (P · a b c d e) 0 = 0)
  (h3 : ∀ x, P x a b c d e = x^2 * (x - 1) * (x - 2) * (x - 3))
  (h4 : ∀ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) : 
  c ≠ 0 := 
sorry

end coefficient_c_nonzero_l48_48605


namespace area_fraction_above_line_l48_48505

-- Define the points of the rectangle
def A := (2,0)
def B := (7,0)
def C := (7,4)
def D := (2,4)

-- Define the points used for the line
def P := (2,1)
def Q := (7,3)

-- The area of the rectangle
def rect_area := (7 - 2) * 4

-- The fraction of the area of the rectangle above the line
theorem area_fraction_above_line : 
  ∀ A B C D P Q, 
    A = (2,0) → B = (7,0) → C = (7,4) → D = (2,4) →
    P = (2,1) → Q = (7,3) →
    (rect_area = 20) → 1 - ((1/2) * 5 * 2 / 20) = 3 / 4 :=
by
  intros A B C D P Q
  intros hA hB hC hD hP hQ h_area
  sorry

end area_fraction_above_line_l48_48505


namespace sin_150_eq_one_half_l48_48686

theorem sin_150_eq_one_half :
  let θ := 150
  let θ_ref := 30
  let θ_complement := 180 - θ_ref
  θ = θ_complement →
  (∀ θ, θ_ref = 30 * 60.toReal)
    ∧ (∀ θ, (cos θ_ref, sin θ_ref) = (Real.sqrt 3 / 2, 1 / 2)) →
  sin θ = 1 / 2 :=
by
  intros θ θ_ref θ_complement h1 h2
  have h3 : (cos θ_complement, sin θ_complement) = (-(cos θ_ref), sin θ_ref) := by
    sorry
  rw h3
  sorry

end sin_150_eq_one_half_l48_48686


namespace ratio_pea_patch_to_radish_patch_l48_48643

-- Definitions
def sixth_of_pea_patch : ℝ := 5
def whole_radish_patch : ℝ := 15

-- Theorem to prove
theorem ratio_pea_patch_to_radish_patch :
  (6 * sixth_of_pea_patch) / whole_radish_patch = 2 :=
by 
  -- skip the actual proof since it's not required
  sorry

end ratio_pea_patch_to_radish_patch_l48_48643


namespace value_of_expression_l48_48220

theorem value_of_expression (x : ℝ) (h : x = 5) : (x^2 + x - 12) / (x - 4) = 18 :=
by 
  sorry

end value_of_expression_l48_48220


namespace sin_150_eq_half_l48_48660

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by sorry

end sin_150_eq_half_l48_48660


namespace tan_theta_sub_pi_over_4_l48_48270

open Real

theorem tan_theta_sub_pi_over_4 (θ : ℝ) (h1 : -π / 2 < θ ∧ θ < 0) 
  (h2 : sin (θ + π / 4) = 3 / 5) : tan (θ - π / 4) = -4 / 3 :=
by
  sorry

end tan_theta_sub_pi_over_4_l48_48270


namespace find_integer_a_l48_48260

theorem find_integer_a (x d e a : ℤ) :
  ((x - a)*(x - 8) - 3 = (x + d)*(x + e)) → (a = 6) :=
by
  sorry

end find_integer_a_l48_48260


namespace certain_number_z_l48_48286

theorem certain_number_z (x y z : ℝ) (h1 : 0.5 * x = y + z) (h2 : x - 2 * y = 40) : z = 20 :=
by 
  sorry

end certain_number_z_l48_48286


namespace inequality_abc_l48_48900

theorem inequality_abc (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a * b * c = 8) :
  (a^2 / Real.sqrt ((1 + a^3) * (1 + b^3))) + (b^2 / Real.sqrt ((1 + b^3) * (1 + c^3))) +
  (c^2 / Real.sqrt ((1 + c^3) * (1 + a^3))) ≥ 4 / 3 :=
sorry

end inequality_abc_l48_48900


namespace A_share_in_profit_l48_48494

-- Given conditions:
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12600

-- The statement we need to prove:
theorem A_share_in_profit :
  (3 / 10) * total_profit = 3780 := by
  sorry

end A_share_in_profit_l48_48494


namespace choir_members_max_l48_48804

theorem choir_members_max (m y n : ℕ) (h_square : m = y^2 + 11) (h_rect : m = n * (n + 5)) : 
  m = 300 := 
sorry

end choir_members_max_l48_48804


namespace ball_bounce_height_l48_48637

theorem ball_bounce_height :
  ∃ b : ℕ, ∀ n < b, (320 * (3 / 4 : ℝ) ^ n) ≥ 40 ∧ (320 * (3 / 4 : ℝ) ^ b) < 40 :=
begin
  sorry
end

end ball_bounce_height_l48_48637


namespace average_salary_of_all_workers_is_correct_l48_48335

noncomputable def average_salary_all_workers (n_total n_tech : ℕ) (avg_salary_tech avg_salary_others : ℝ) : ℝ :=
  let n_others := n_total - n_tech
  let total_salary_tech := n_tech * avg_salary_tech
  let total_salary_others := n_others * avg_salary_others
  let total_salary := total_salary_tech + total_salary_others
  total_salary / n_total

theorem average_salary_of_all_workers_is_correct :
  average_salary_all_workers 21 7 12000 6000 = 8000 :=
by
  unfold average_salary_all_workers
  sorry

end average_salary_of_all_workers_is_correct_l48_48335


namespace total_bike_price_l48_48938

theorem total_bike_price 
  (marion_bike_cost : ℝ := 356)
  (stephanie_bike_base_cost : ℝ := 2 * marion_bike_cost)
  (stephanie_discount_rate : ℝ := 0.10)
  (patrick_bike_base_cost : ℝ := 3 * marion_bike_cost)
  (patrick_discount_rate : ℝ := 0.75)
  (stephanie_bike_cost : ℝ := stephanie_bike_base_cost * (1 - stephanie_discount_rate))
  (patrick_bike_cost : ℝ := patrick_bike_base_cost * patrick_discount_rate):
  marion_bike_cost + stephanie_bike_cost + patrick_bike_cost = 1797.80 := 
by 
  sorry

end total_bike_price_l48_48938


namespace spencer_total_distance_l48_48168

-- Define the individual segments of Spencer's travel
def walk1 : ℝ := 1.2
def bike1 : ℝ := 1.8
def bus1 : ℝ := 3
def walk2 : ℝ := 0.4
def walk3 : ℝ := 0.6
def bike2 : ℝ := 2
def walk4 : ℝ := 1.5

-- Define the conversion factors
def bike_to_walk_conversion : ℝ := 0.5
def bus_to_walk_conversion : ℝ := 0.8

-- Calculate the total walking distance
def total_walking_distance : ℝ := walk1 + walk2 + walk3 + walk4

-- Calculate the total biking distance as walking equivalent
def total_biking_distance_as_walking : ℝ := (bike1 + bike2) * bike_to_walk_conversion

-- Calculate the total bus distance as walking equivalent
def total_bus_distance_as_walking : ℝ := bus1 * bus_to_walk_conversion

-- Define the total walking equivalent distance
def total_distance : ℝ := total_walking_distance + total_biking_distance_as_walking + total_bus_distance_as_walking

-- Theorem stating the total distance covered is 8 miles
theorem spencer_total_distance : total_distance = 8 := by
  unfold total_distance
  unfold total_walking_distance
  unfold total_biking_distance_as_walking
  unfold total_bus_distance_as_walking
  norm_num
  sorry

end spencer_total_distance_l48_48168


namespace sum_of_reciprocals_of_distances_l48_48609

theorem sum_of_reciprocals_of_distances :
  let e := (sqrt 5) / 3 in
  let directrix := (9:ℝ) / (sqrt 5) in
  let points := (fin 24).toList.map (λ i => (3 * cos (i * π / 12), 2 * sin (i * π / 12))) in
  let distances := points.map (λ p => abs (fst p - directrix)) in
  let reciprocals := distances.map (λ d => 1 / d) in
  (reciprocals.sum = 6 * sqrt 5) :=
sorry

end sum_of_reciprocals_of_distances_l48_48609


namespace percentage_class_takes_lunch_l48_48225

theorem percentage_class_takes_lunch (total_students boys girls : ℕ)
  (h_total: total_students = 100)
  (h_ratio: boys = 6 * total_students / (6 + 4))
  (h_girls: girls = 4 * total_students / (6 + 4))
  (boys_lunch_ratio : ℝ)
  (girls_lunch_ratio : ℝ)
  (h_boys_lunch_ratio : boys_lunch_ratio = 0.60)
  (h_girls_lunch_ratio : girls_lunch_ratio = 0.40):
  ((boys_lunch_ratio * boys + girls_lunch_ratio * girls) / total_students) * 100 = 52 :=
by
  sorry

end percentage_class_takes_lunch_l48_48225


namespace total_floors_l48_48508

theorem total_floors (P Q R S T X F : ℕ) (h1 : 1 < X) (h2 : X < 50) :
  F = 1 + P - Q + R - S + T + X :=
sorry

end total_floors_l48_48508


namespace ratio_friday_to_monday_l48_48302

-- Definitions from conditions
def rabbits : ℕ := 16
def monday_toys : ℕ := 6
def wednesday_toys : ℕ := 2 * monday_toys
def saturday_toys : ℕ := wednesday_toys / 2
def total_toys : ℕ := 3 * rabbits

-- Definition to represent the number of toys bought on Friday
def friday_toys : ℕ := total_toys - (monday_toys + wednesday_toys + saturday_toys)

-- Theorem to prove the ratio is 4:1
theorem ratio_friday_to_monday : friday_toys / monday_toys = 4 := by
  -- Placeholder for the proof
  sorry

end ratio_friday_to_monday_l48_48302


namespace students_in_class_l48_48967

variable (G B : ℕ)

def total_plants (G B : ℕ) : ℕ := 3 * G + B / 3

theorem students_in_class (h1 : total_plants G B = 24) (h2 : B / 3 = 6) : G + B = 24 :=
by
  sorry

end students_in_class_l48_48967


namespace password_count_l48_48992

noncomputable def permutations : ℕ → ℕ → ℕ
| n, r := (n! / (n-r)!)

theorem password_count:
    let n_eng := 26 in
    let r_eng := 2 in
    let n_num := 10 in
    let r_num := 2 in
    permutations n_eng r_eng * permutations n_num r_num = (26 * 25) * (10 * 9) :=
by
  sorry

end password_count_l48_48992


namespace total_people_can_ride_l48_48858

theorem total_people_can_ride (num_people_per_teacup : Nat) (num_teacups : Nat) (h1 : num_people_per_teacup = 9) (h2 : num_teacups = 7) : num_people_per_teacup * num_teacups = 63 := by
  sorry

end total_people_can_ride_l48_48858


namespace four_digit_positive_integers_count_l48_48280

def first_two_digit_choices : Finset ℕ := {2, 3, 6}
def last_two_digit_choices : Finset ℕ := {3, 7, 9}

theorem four_digit_positive_integers_count :
  (first_two_digit_choices.card * first_two_digit_choices.card) *
  (last_two_digit_choices.card * (last_two_digit_choices.card - 1)) = 54 := by
sorry

end four_digit_positive_integers_count_l48_48280


namespace evaluate_polynomial_at_3_using_horners_method_l48_48384

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem evaluate_polynomial_at_3_using_horners_method : f 3 = 1641 := by
 sorry

end evaluate_polynomial_at_3_using_horners_method_l48_48384


namespace sin_150_equals_half_l48_48695

noncomputable def sin_150_eq_half : Prop := 
  sin (150 * real.pi / 180) = 1 / 2

theorem sin_150_equals_half : sin_150_eq_half :=
  by sorry

end sin_150_equals_half_l48_48695


namespace sin_150_eq_half_l48_48712

theorem sin_150_eq_half :
  (sin (150 : ℝ)) = 1 / 2 :=
by
  -- Conditions in form of definitions
  let θ : ℝ := 30
  have h1 : 150 = 180 - θ := by ring 
  have h2 : sin (180 - θ) = sin θ := by exact sin_sub_π θ
  have h3 : sin θ = 1 / 2 := by exact real.sin_of_real (by norm_num)
  sorry  -- Proof omitted

end sin_150_eq_half_l48_48712


namespace min_value_of_f_l48_48310

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 4*x + 5) * (x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = -9 :=
  sorry

end min_value_of_f_l48_48310


namespace segment_length_greater_than_inradius_sqrt_two_l48_48819

variables {a b c : ℝ} -- sides of the triangle
variables {P Q : ℝ} -- points on sides of the triangle
variables {S_ABC S_PCQ : ℝ} -- areas of the triangles
variables {s : ℝ} -- semi-perimeter of the triangle
variables {r : ℝ} -- radius of the inscribed circle
variables {ℓ : ℝ} -- length of segment dividing the triangle's area

-- Given conditions in the form of assumptions
variables (h1 : S_PCQ = S_ABC / 2)
variables (h2 : PQ = ℓ)
variables (h3 : r = S_ABC / s)

-- The statement of the theorem
theorem segment_length_greater_than_inradius_sqrt_two
  (h1 : S_PCQ = S_ABC / 2) 
  (h2 : PQ = ℓ) 
  (h3 : r = S_ABC / s)
  (h4 : s = (a + b + c) / 2) 
  (h5 : S_ABC = Real.sqrt (s * (s - a) * (s - b) * (s - c))) 
  (h6 : ℓ^2 = a^2 + b^2 - (a^2 + b^2 - c^2) / 2) :
  ℓ > r * Real.sqrt 2 :=
sorry

end segment_length_greater_than_inradius_sqrt_two_l48_48819


namespace greatest_possible_value_of_a_l48_48599

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ (x : ℤ), x * (x + a) = -21 → x^2 + a * x + 21 = 0) ∧
  (∀ (a' : ℕ), (∀ (x : ℤ), x * (x + a') = -21 → x^2 + a' * x + 21 = 0) → a' ≤ a) ∧
  a = 22 :=
sorry

end greatest_possible_value_of_a_l48_48599


namespace square_difference_l48_48004

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 36) 
  (h₂ : x * y = 8) : 
  (x - y)^2 = 4 :=
by
  sorry

end square_difference_l48_48004


namespace jana_height_l48_48428

theorem jana_height (jess_height : ℕ) (kelly_height : ℕ) (jana_height : ℕ) 
  (h1 : kelly_height = jess_height - 3) 
  (h2 : jana_height = kelly_height + 5) 
  (h3 : jess_height = 72) : 
  jana_height = 74 := 
by
  sorry

end jana_height_l48_48428


namespace sin_150_eq_half_l48_48713

theorem sin_150_eq_half :
  (sin (150 : ℝ)) = 1 / 2 :=
by
  -- Conditions in form of definitions
  let θ : ℝ := 30
  have h1 : 150 = 180 - θ := by ring 
  have h2 : sin (180 - θ) = sin θ := by exact sin_sub_π θ
  have h3 : sin θ = 1 / 2 := by exact real.sin_of_real (by norm_num)
  sorry  -- Proof omitted

end sin_150_eq_half_l48_48713


namespace midpoint_of_AB_l48_48010

theorem midpoint_of_AB (xA xB : ℝ) (p : ℝ) (h_parabola : ∀ y, y^2 = 4 * xA → y^2 = 4 * xB)
  (h_focus : (2 : ℝ) = p)
  (h_length_AB : (abs (xB - xA)) = 5) :
  (xA + xB) / 2 = 3 / 2 :=
sorry

end midpoint_of_AB_l48_48010


namespace correct_calculation_c_l48_48479

theorem correct_calculation_c (a : ℝ) :
  (a^4 / a = a^3) :=
by
  rw [←div_eq_mul_inv, pow_sub, pow_one]
  sorry

end correct_calculation_c_l48_48479


namespace exists_not_perfect_square_l48_48949

theorem exists_not_perfect_square (a b c : ℤ) : ∃ (n : ℕ), n > 0 ∧ ¬ ∃ k : ℕ, n^3 + a * n^2 + b * n + c = k^2 :=
by
  sorry

end exists_not_perfect_square_l48_48949


namespace factorize_expression_l48_48125

theorem factorize_expression (a x y : ℤ) : a * x - a * y = a * (x - y) :=
  sorry

end factorize_expression_l48_48125


namespace final_remaining_is_correct_total_sum_is_correct_l48_48324

open Nat

-- Definition of the initial sequence of numbers
def initial_sequence (n : ℕ) : List ℕ := List.range' 1 n

-- Definition of the sum of the first n natural numbers
def sum_of_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition of final remaining number
def final_remaining_number (n : ℕ) : ℕ :=
  sum_of_natural_numbers n

-- Definition of the number of operations
def n_operations (n : ℕ) : ℕ := (n + 15) / 16

-- Definition of the total sum of all numbers
noncomputable def total_sum_of_all_numbers (n : ℕ) : ℕ :=
  let initial_sum := sum_of_natural_numbers n
  in initial_sum + 100k {op_sum | op_sum = initial_sum * (\all 16th operation)} 

-- Theorem for the final remaining number
theorem final_remaining_is_correct : final_remaining_number 2011 = 2023066 :=
by
  exact (by norm_num : 2011 * 1007 = 2023066)

-- Theorem for the total sum of all numbers
theorem total_sum_is_correct : total_sum_of_all_numbers 2011 = 7822326 :=
sorry

end final_remaining_is_correct_total_sum_is_correct_l48_48324


namespace two_talents_students_l48_48493

-- Definitions and conditions
def total_students : ℕ := 120
def cannot_sing : ℕ := 50
def cannot_dance : ℕ := 75
def cannot_act : ℕ := 35

-- Definitions based on conditions
def can_sing : ℕ := total_students - cannot_sing
def can_dance : ℕ := total_students - cannot_dance
def can_act : ℕ := total_students - cannot_act

-- The main theorem statement
theorem two_talents_students : can_sing + can_dance + can_act - total_students = 80 :=
by
  -- substituting actual numbers to prove directly
  have h_can_sing : can_sing = 70 := rfl
  have h_can_dance : can_dance = 45 := rfl
  have h_can_act : can_act = 85 := rfl
  sorry

end two_talents_students_l48_48493


namespace sin_150_eq_half_l48_48717

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48717


namespace sin_150_eq_half_l48_48703

theorem sin_150_eq_half :
  Float.sin (150 * Float.pi / 180) = 1 / 2 := sorry

end sin_150_eq_half_l48_48703


namespace DiagonalsOfShapesBisectEachOther_l48_48090

structure Shape where
  bisect_diagonals : Prop

def is_parallelogram (s : Shape) : Prop := s.bisect_diagonals
def is_rectangle (s : Shape) : Prop := s.bisect_diagonals
def is_rhombus (s : Shape) : Prop := s.bisect_diagonals
def is_square (s : Shape) : Prop := s.bisect_diagonals

theorem DiagonalsOfShapesBisectEachOther (s : Shape) :
  is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s → s.bisect_diagonals := by
  sorry

end DiagonalsOfShapesBisectEachOther_l48_48090


namespace star_three_and_four_l48_48120

def star (a b : ℝ) : ℝ := 4 * a + 5 * b - 2 * a * b

theorem star_three_and_four : star 3 4 = 8 :=
by
  sorry

end star_three_and_four_l48_48120


namespace max_single_player_salary_is_426000_l48_48564

noncomputable def max_single_player_salary (total_salary_cap : ℤ) (min_salary : ℤ) (num_players : ℤ) : ℤ :=
  total_salary_cap - (num_players - 1) * min_salary

theorem max_single_player_salary_is_426000 :
  ∃ y, max_single_player_salary 800000 17000 23 = y ∧ y = 426000 :=
by
  sorry

end max_single_player_salary_is_426000_l48_48564


namespace find_a_l48_48554

noncomputable def f (a x : ℝ) := 3*x^3 - 9*x + a
noncomputable def f' (x : ℝ) : ℝ := 9*x^2 - 9

theorem find_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :
  a = 6 ∨ a = -6 :=
by sorry

end find_a_l48_48554


namespace calculate_treatment_received_l48_48373

variable (drip_rate : ℕ) (duration_hours : ℕ) (drops_convert : ℕ) (ml_convert : ℕ)

theorem calculate_treatment_received (h1 : drip_rate = 20) (h2 : duration_hours = 2) 
    (h3 : drops_convert = 100) (h4 : ml_convert = 5) : 
    (drip_rate * (duration_hours * 60) * ml_convert) / drops_convert = 120 := 
by
  sorry

end calculate_treatment_received_l48_48373


namespace olympic_volunteers_selection_l48_48264

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem olympic_volunteers_selection :
  (choose 4 3 * choose 3 1) + (choose 4 2 * choose 3 2) + (choose 4 1 * choose 3 3) = 34 := 
by
  sorry

end olympic_volunteers_selection_l48_48264


namespace probability_same_group_l48_48617

open ProbabilityTheory

theorem probability_same_group (p : ProbMassFunction (Fin 3)) :
  (∀ a b : Fin 3, p a = p b) → 
  P (λ (x : Fin 3 × Fin 3), x.fst = x.snd) = 1 / 3 :=
by
  sorry

end probability_same_group_l48_48617


namespace fraction_of_orange_juice_correct_l48_48211

-- Define the capacities of the pitchers
def capacity := 800

-- Define the fractions of orange juice and apple juice in the first pitcher
def orangeJuiceFraction1 := 1 / 4
def appleJuiceFraction1 := 1 / 8

-- Define the fractions of orange juice and apple juice in the second pitcher
def orangeJuiceFraction2 := 1 / 5
def appleJuiceFraction2 := 1 / 10

-- Define the total volumes of the contents in each pitcher
def totalVolume := 2 * capacity -- total volume in the large container after pouring

-- Define the orange juice volumes in each pitcher
def orangeJuiceVolume1 := orangeJuiceFraction1 * capacity
def orangeJuiceVolume2 := orangeJuiceFraction2 * capacity

-- Calculate the total volume of orange juice in the large container
def totalOrangeJuiceVolume := orangeJuiceVolume1 + orangeJuiceVolume2

-- Define the fraction of orange juice in the large container
def orangeJuiceFraction := totalOrangeJuiceVolume / totalVolume

theorem fraction_of_orange_juice_correct :
  orangeJuiceFraction = 9 / 40 :=
by
  sorry

end fraction_of_orange_juice_correct_l48_48211


namespace greatest_possible_sum_l48_48390

theorem greatest_possible_sum (x y : ℤ) (h : x^2 + y^2 = 100) : x + y ≤ 14 :=
sorry

end greatest_possible_sum_l48_48390


namespace kamal_average_marks_l48_48568

theorem kamal_average_marks :
  (76 / 120) * 0.2 + 
  (60 / 110) * 0.25 + 
  (82 / 100) * 0.15 + 
  (67 / 90) * 0.2 + 
  (85 / 100) * 0.15 + 
  (78 / 95) * 0.05 = 0.70345 :=
by 
  sorry

end kamal_average_marks_l48_48568


namespace mode_and_median_of_seedlings_l48_48189

theorem mode_and_median_of_seedlings :
  let heights := [25, 26, 27, 26, 27, 28, 29, 26, 29] in
  (mode heights = [26]) ∧ (median heights = some 27) :=
by {
  let heights := [25, 26, 27, 26, 27, 28, 29, 26, 29];
  sorry
}

end mode_and_median_of_seedlings_l48_48189


namespace find_f_neg5_l48_48527

theorem find_f_neg5 (a b : ℝ) (Sin : ℝ → ℝ) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x + b * (Sin x) ^ 3 + 1)
  (h_f5 : f 5 = 7) :
  f (-5) = -5 := 
by
  sorry

end find_f_neg5_l48_48527


namespace total_husk_is_30_bags_l48_48915

-- Define the total number of cows and the number of days.
def numCows : ℕ := 30
def numDays : ℕ := 30

-- Define the rate of consumption: one cow eats one bag in 30 days.
def consumptionRate (cows : ℕ) (days : ℕ) : ℕ := cows / days

-- Define the total amount of husk consumed in 30 days by 30 cows.
def totalHusk (cows : ℕ) (days : ℕ) (rate : ℕ) : ℕ := cows * rate

-- State the problem in a theorem.
theorem total_husk_is_30_bags : totalHusk numCows numDays 1 = 30 := by
  sorry

end total_husk_is_30_bags_l48_48915


namespace probability_of_two_approvals_l48_48025

theorem probability_of_two_approvals (P_A : ℝ) : P_A = 0.6 → 
  ∑ (k in Finset.range(5)), 
    if k = 2 then (((nat.choose 4 k) : ℝ) * (P_A^k) * (1 - P_A)^(4-k)) 
    else 0 = 0.3456 :=
by
  intro h
  rw h
  sorry

end probability_of_two_approvals_l48_48025


namespace find_g_at_4_l48_48957

def g (x : ℝ) : ℝ := sorry

theorem find_g_at_4 (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 5.5 :=
by
  sorry

end find_g_at_4_l48_48957


namespace perimeter_of_monster_is_correct_l48_48920

/-
  The problem is to prove that the perimeter of a shaded sector of a circle
  with radius 2 cm and a central angle of 120 degrees (where the mouth is a chord)
  is equal to (8 * π / 3 + 2 * sqrt 3) cm.
-/

noncomputable def perimeter_of_monster (r : ℝ) (theta_deg : ℝ) : ℝ :=
  let theta_rad := theta_deg * Real.pi / 180
  let chord_length := 2 * r * Real.sin (theta_rad / 2)
  let arc_length := (2 * (2 * Real.pi) * (240 / 360))
  arc_length + chord_length

theorem perimeter_of_monster_is_correct : perimeter_of_monster 2 120 = (8 * Real.pi / 3 + 2 * Real.sqrt 3) :=
by
  sorry

end perimeter_of_monster_is_correct_l48_48920


namespace find_angle_BAC_l48_48771

-- Definitions and Hypotheses
variables (A B C P : Type) (AP PC AB AC : Real) (angle_BPC : Real)

-- Hypotheses
-- AP = PC
-- AB = AC
-- angle BPC = 120 
axiom AP_eq_PC : AP = PC
axiom AB_eq_AC : AB = AC
axiom angle_BPC_eq_120 : angle_BPC = 120

-- Theorem
theorem find_angle_BAC (AP_eq_PC : AP = PC) (AB_eq_AC : AB = AC) (angle_BPC_eq_120 : angle_BPC = 120) : angle_BAC = 60 :=
sorry

end find_angle_BAC_l48_48771


namespace total_fish_catch_l48_48586

noncomputable def Johnny_fishes : ℕ := 8
noncomputable def Sony_fishes : ℕ := 4 * Johnny_fishes
noncomputable def total_fishes : ℕ := Sony_fishes + Johnny_fishes

theorem total_fish_catch : total_fishes = 40 := by
  sorry

end total_fish_catch_l48_48586


namespace find_x_if_delta_phi_eq_3_l48_48414

variable (x : ℚ)

def delta (x : ℚ) := 4 * x + 9
def phi (x : ℚ) := 9 * x + 6

theorem find_x_if_delta_phi_eq_3 : 
  delta (phi x) = 3 → x = -5 / 6 := by 
  sorry

end find_x_if_delta_phi_eq_3_l48_48414


namespace total_distance_covered_l48_48167

-- Define the basic conditions
def num_marathons : Nat := 15
def miles_per_marathon : Nat := 26
def yards_per_marathon : Nat := 385
def yards_per_mile : Nat := 1760

-- Define the total miles and total yards covered
def total_miles : Nat := num_marathons * miles_per_marathon
def total_yards : Nat := num_marathons * yards_per_marathon

-- Convert excess yards into miles and calculate the remaining yards
def extra_miles : Nat := total_yards / yards_per_mile
def remaining_yards : Nat := total_yards % yards_per_mile

-- Compute the final total distance
def total_distance_miles : Nat := total_miles + extra_miles
def total_distance_yards : Nat := remaining_yards

-- The theorem that needs to be proven
theorem total_distance_covered :
  total_distance_miles = 393 ∧ total_distance_yards = 495 :=
by
  sorry

end total_distance_covered_l48_48167


namespace equation_one_solution_equation_two_solution_l48_48227

theorem equation_one_solution (x : ℝ) : 4 * (x - 1)^2 - 9 = 0 ↔ (x = 5 / 2) ∨ (x = - 1 / 2) := 
by sorry

theorem equation_two_solution (x : ℝ) : x^2 - 6 * x - 7 = 0 ↔ (x = 7) ∨ (x = - 1) :=
by sorry

end equation_one_solution_equation_two_solution_l48_48227


namespace sin_150_eq_half_l48_48701

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by 
  sorry

end sin_150_eq_half_l48_48701


namespace greatest_possible_value_of_a_l48_48600

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ (x : ℤ), x * (x + a) = -21 → x^2 + a * x + 21 = 0) ∧
  (∀ (a' : ℕ), (∀ (x : ℤ), x * (x + a') = -21 → x^2 + a' * x + 21 = 0) → a' ≤ a) ∧
  a = 22 :=
sorry

end greatest_possible_value_of_a_l48_48600


namespace find_minimum_value_of_f_l48_48312

def f (x : ℝ) : ℝ := (x ^ 2 + 4 * x + 5) * (x ^ 2 + 4 * x + 2) + 2 * x ^ 2 + 8 * x + 1

theorem find_minimum_value_of_f : ∃ x : ℝ, f x = -9 :=
by
  sorry

end find_minimum_value_of_f_l48_48312


namespace no_such_function_exists_l48_48885

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ x y z : ℤ, f (x * y) + f (x * z) - f x * f (y * z) ≤ -1

theorem no_such_function_exists : (∃ f : ℤ → ℤ, satisfies_condition f) = false :=
by
  sorry

end no_such_function_exists_l48_48885


namespace sin_150_equals_half_l48_48690

noncomputable def sin_150_eq_half : Prop := 
  sin (150 * real.pi / 180) = 1 / 2

theorem sin_150_equals_half : sin_150_eq_half :=
  by sorry

end sin_150_equals_half_l48_48690


namespace problem_product_of_areas_eq_3600x6_l48_48489

theorem problem_product_of_areas_eq_3600x6 
  (x : ℝ) 
  (bottom_area : ℝ) 
  (side_area : ℝ) 
  (front_area : ℝ)
  (bottom_area_eq : bottom_area = 12 * x ^ 2)
  (side_area_eq : side_area = 15 * x ^ 2)
  (front_area_eq : front_area = 20 * x ^ 2)
  (dimensions_proportional : ∃ a b c : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x 
                            ∧ bottom_area = a * b ∧ side_area = a * c ∧ front_area = b * c)
  : bottom_area * side_area * front_area = 3600 * x ^ 6 :=
by 
  -- Proof omitted
  sorry

end problem_product_of_areas_eq_3600x6_l48_48489


namespace average_age_increase_l48_48615

noncomputable def average_increase (ages : List ℕ) := 
  let total_ages := ages.sum 
  let n := ages.length
  total_ages / n

theorem average_age_increase :
  ∀ (s1 : Finset ℕ) (s2 : Finset ℕ) (t : ℕ),
  s1.card = 9 → s2.card = 1 →
  t ∈ s2 →
  s1.sum (λ x, x) / 9 = 8 →
  28 = t →
  average_increase (s1.val ++ s2.val) - average_increase (s1.val) = 2 := by
  sorry

end average_age_increase_l48_48615


namespace container_volumes_l48_48831

variable (a : ℕ)

theorem container_volumes (h₁ : a = 18) :
  a^3 = 5832 ∧ (a - 4)^3 = 2744 ∧ (a - 6)^3 = 1728 :=
by {
  sorry
}

end container_volumes_l48_48831


namespace positive_root_gt_1008_l48_48856

noncomputable def P (x : ℝ) : ℝ := sorry
-- where P is a non-constant polynomial with integer coefficients bounded by 2015 in absolute value
-- Assume it has been properly defined according to the conditions in the problem statement

theorem positive_root_gt_1008 (x : ℝ) (hx : 0 < x) (hroot : P x = 0) : x > 1008 := 
sorry

end positive_root_gt_1008_l48_48856


namespace Koschei_no_equal_coins_l48_48824

theorem Koschei_no_equal_coins (a : Fin 6 → ℕ)
  (initial_condition : a 0 = 1 ∧ a 1 = 0 ∧ a 2 = 0 ∧ a 3 = 0 ∧ a 4 = 0 ∧ a 5 = 0) :
  ¬ ( ∃ k : ℕ, ( ( ∀ i : Fin 6, a i = k ) ) ) :=
by
  sorry

end Koschei_no_equal_coins_l48_48824


namespace complex_magnitude_difference_proof_l48_48779

noncomputable def complex_magnitude_difference (z1 z2 : ℂ) : ℂ := 
  |z1 - z2|

theorem complex_magnitude_difference_proof
  (z1 z2 : ℂ)
  (h1 : |z1| = 2)
  (h2 : |z2| = 2)
  (h3 : z1 + z2 = √3 + 1 * complex.I) :
  complex_magnitude_difference z1 z2 = 2 * √3 := by
  sorry

end complex_magnitude_difference_proof_l48_48779


namespace train_rate_first_hour_l48_48647

-- Define the conditions
def rateAtFirstHour (r : ℕ) : Prop :=
  (11 / 2) * (r + (r + 100)) = 660

-- Prove the rate is 10 mph
theorem train_rate_first_hour (r : ℕ) : rateAtFirstHour r → r = 10 :=
by 
  sorry

end train_rate_first_hour_l48_48647


namespace max_ratio_1099_l48_48129

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_ratio_1099 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → (sum_of_digits n : ℚ) / n ≤ (sum_of_digits 1099 : ℚ) / 1099 :=
by
  intros n hn
  sorry

end max_ratio_1099_l48_48129


namespace find_range_of_a_l48_48445

noncomputable def range_of_a (a : ℝ) (n : ℕ) : Prop :=
  1 + 1 / (n : ℝ) ≤ a ∧ a < 1 + 1 / ((n - 1) : ℝ)

theorem find_range_of_a (a : ℝ) (n : ℕ) (h1 : 1 < a) (h2 : 2 ≤ n) :
  (∃ x : ℕ, ∀ x₀ < x, (⌊a * (x₀ : ℝ)⌋ : ℝ) = x₀) ↔ range_of_a a n := by
  sorry

end find_range_of_a_l48_48445


namespace math_problem_l48_48868

theorem math_problem : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end math_problem_l48_48868


namespace integer_k_values_l48_48274

theorem integer_k_values (a b k : ℝ) (m : ℝ) (ha : a > 0) (hb : b > 0) (hba_int : ∃ n : ℤ, n ≠ 0 ∧ b = (n : ℝ) * a) 
  (hA : a = a * k + m) (hB : 8 * b = b * k + m) : k = 9 ∨ k = 15 := 
by
  sorry

end integer_k_values_l48_48274


namespace camel_height_in_feet_correct_l48_48919

def hare_height_in_inches : ℕ := 14
def multiplication_factor : ℕ := 24
def inches_to_feet_ratio : ℕ := 12

theorem camel_height_in_feet_correct :
  (hare_height_in_inches * multiplication_factor) / inches_to_feet_ratio = 28 := by
  sorry

end camel_height_in_feet_correct_l48_48919


namespace probability_no_one_receives_own_letter_four_l48_48965

def num_derangements : ℕ → ℕ
| 0     := 1
| 1     := 0
| (n+2) := (n+1) * (num_derangements (n+1) + num_derangements n)

def factorial (n : ℕ) : ℕ := nat.factorial n

def probability_no_one_receives_own_letter (n : ℕ) : ℝ :=
  num_derangements n / factorial n

theorem probability_no_one_receives_own_letter_four :
  probability_no_one_receives_own_letter 4 = 3 / 8 :=
by
  -- We already know that num_derangements 4 = 9 and factorial 4 = 24
  suffices h : num_derangements 4 = 9 ∧ factorial 4 = 24, by
    simp [probability_no_one_receives_own_letter, h.left, h.right],
  split,
  -- these values can be computed, but are written here for clarity:
  exact rfl,  -- From computation: num_derangements 4 = 9
  exact rfl   -- From computation: factorial 4 = 24

end probability_no_one_receives_own_letter_four_l48_48965


namespace part1_part2_l48_48271

theorem part1 (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 :=
sorry

theorem part2 (a b : ℝ) (n : ℝ) :
  n = 6 → (a > 0 ∧ b > 0 ∧ (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = 1) → (4 * a + 7 * b) ≥ 9 :=
sorry

end part1_part2_l48_48271


namespace sin_150_eq_half_l48_48670

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48670


namespace probability_of_forming_phrase_l48_48078

theorem probability_of_forming_phrase :
  let cards := ["中", "国", "梦"]
  let n := 6
  let m := 1
  ∃ (p : ℚ), p = (m / n : ℚ) ∧ p = 1 / 6 :=
by
  sorry

end probability_of_forming_phrase_l48_48078


namespace sin_150_eq_half_l48_48668

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48668


namespace jill_total_time_l48_48773

def time_spent_on_day (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2 * time_spent_on_day (n - 1)

def total_time_over_week : ℕ :=
  (List.range 5).map (λ n => time_spent_on_day (n + 1)).sum

theorem jill_total_time :
  total_time_over_week = 155 :=
by
  sorry

end jill_total_time_l48_48773


namespace opposite_of_x_is_positive_l48_48817

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive_l48_48817


namespace spoiled_milk_percentage_l48_48034

theorem spoiled_milk_percentage (p_egg p_flour p_all_good : ℝ) (h_egg : p_egg = 0.40) (h_flour : p_flour = 0.75) (h_all_good : p_all_good = 0.24) : 
  (1 - (p_all_good / (p_egg * p_flour))) = 0.20 :=
by
  sorry

end spoiled_milk_percentage_l48_48034


namespace classroom_students_l48_48966

theorem classroom_students (n : ℕ) (h1 : 20 < n ∧ n < 30) 
  (h2 : ∃ n_y : ℕ, n = 3 * n_y + 1) 
  (h3 : ∃ n_y' : ℕ, n = (4 * (n - 1)) / 3 + 1) :
  n = 25 := 
by sorry

end classroom_students_l48_48966


namespace determine_b_l48_48395

theorem determine_b (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x) ∧ (∀ x y : ℝ, y - 2 = (b + 9) * x) → 
  b = -6 :=
by
  sorry

end determine_b_l48_48395


namespace product_numerator_denominator_l48_48219

def recurring_decimal_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  n / d

theorem product_numerator_denominator (n : ℕ) (d : ℕ) (x : Rat)
  (hx : recurring_decimal_to_fraction 18 999 = x)
  (hn : n = 2)
  (hd : d = 111) :
  n * d = 222 := by
  have h_frac : x = 0.018 -- This follows from the definition and will be used in the proof
  sorry

end product_numerator_denominator_l48_48219


namespace sin_150_eq_half_l48_48671

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48671


namespace Tony_fills_pool_in_90_minutes_l48_48775

def minutes (r : ℚ) : ℚ := 1 / r

theorem Tony_fills_pool_in_90_minutes (J S T : ℚ) 
  (hJ : J = 1 / 30)       -- Jim's rate in pools per minute
  (hS : S = 1 / 45)       -- Sue's rate in pools per minute
  (h_combined : J + S + T = 1 / 15) -- Combined rate of all three

  : minutes T = 90 :=     -- Tony can fill the pool alone in 90 minutes
by sorry

end Tony_fills_pool_in_90_minutes_l48_48775


namespace first_group_number_l48_48082

variable (x : ℕ)

def number_of_first_group :=
  x = 6

theorem first_group_number (H1 : ∀ k : ℕ, k = 8 * 15 + x)
                          (H2 : k = 126) : 
                          number_of_first_group x :=
by
  sorry

end first_group_number_l48_48082


namespace max_regions_with_five_lines_l48_48768

def max_regions (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * (n + 1) / 2 + 1

theorem max_regions_with_five_lines (n : ℕ) (h : n = 5) : max_regions n = 16 :=
by {
  rw [h, max_regions];
  norm_num;
  done
}

end max_regions_with_five_lines_l48_48768


namespace sum_a_m_eq_2_pow_n_b_n_l48_48934

noncomputable def a_n (x : ℝ) (n : ℕ) : ℝ := (Finset.range (n + 1)).sum (λ k => x ^ k)

noncomputable def b_n (x : ℝ) (n : ℕ) : ℝ := 
  (Finset.range (n + 1)).sum (λ k => ((x + 1) / 2) ^ k)

theorem sum_a_m_eq_2_pow_n_b_n 
  (x : ℝ) (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ m => a_n x m * Nat.choose (n + 1) (m + 1)) = 2 ^ n * b_n x n :=
by
  sorry

end sum_a_m_eq_2_pow_n_b_n_l48_48934


namespace geometric_sequence_problem_l48_48417

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h5 : a 5 * a 6 = 3)
  (h9 : a 9 * a 10 = 9) :
  a 7 * a 8 = 3 * Real.sqrt 3 :=
by
  sorry

end geometric_sequence_problem_l48_48417


namespace problem_statement_l48_48573

def f (x : ℝ) : ℝ := x^6 + x^2 + 7 * x

theorem problem_statement : f 3 - f (-3) = 42 := by
  sorry

end problem_statement_l48_48573


namespace last_three_digits_W_555_2_l48_48364

noncomputable def W : ℕ → ℕ → ℕ
| n, 0 => n ^ n
| n, (k + 1) => W (W n k) k

theorem last_three_digits_W_555_2 : (W 555 2) % 1000 = 375 := 
by
  sorry

end last_three_digits_W_555_2_l48_48364


namespace number_of_green_hats_l48_48226

theorem number_of_green_hats (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) : 
  G = 40 := by
  sorry

end number_of_green_hats_l48_48226


namespace find_a_b_l48_48408

theorem find_a_b (a b : ℝ) (h₁ : a^2 = 64 * b) (h₂ : a^2 = 4 * b) : a = 0 ∧ b = 0 :=
by
  sorry

end find_a_b_l48_48408


namespace total_volume_stacked_dice_l48_48623

def die_volume (width length height : ℕ) : ℕ := 
  width * length * height

def total_dice (horizontal vertical layers : ℕ) : ℕ := 
  horizontal * vertical * layers

theorem total_volume_stacked_dice :
  let width := 1
  let length := 1
  let height := 1
  let horizontal := 7
  let vertical := 5
  let layers := 3
  let single_die_volume := die_volume width length height
  let num_dice := total_dice horizontal vertical layers
  single_die_volume * num_dice = 105 :=
by
  sorry  -- proof to be provided

end total_volume_stacked_dice_l48_48623


namespace green_balloons_count_l48_48248

-- Define the conditions
def total_balloons : Nat := 50
def red_balloons : Nat := 12
def blue_balloons : Nat := 7

-- Define the proof problem
theorem green_balloons_count : 
  let green_balloons := total_balloons - (red_balloons + blue_balloons)
  green_balloons = 31 :=
by
  sorry

end green_balloons_count_l48_48248


namespace problem_solution_l48_48499

def equal_group_B : Prop :=
  (-2)^3 = -(2^3)

theorem problem_solution : equal_group_B := by
  sorry

end problem_solution_l48_48499


namespace mopping_time_is_30_l48_48142

def vacuuming_time := 45
def dusting_time := 60
def brushing_time_per_cat := 5
def number_of_cats := 3
def total_free_time := 180
def free_time_left := 30

def total_cleaning_time := total_free_time - free_time_left
def brushing_time := brushing_time_per_cat * number_of_cats
def time_other_tasks := vacuuming_time + dusting_time + brushing_time

theorem mopping_time_is_30 : total_cleaning_time - time_other_tasks = 30 := by
  -- Calculation proof would go here
  sorry

end mopping_time_is_30_l48_48142


namespace possible_AC_values_l48_48546

-- Given points A, B, and C on a straight line 
-- with AB = 1 and BC = 3, prove that AC can be 2 or 4.

theorem possible_AC_values (A B C : ℝ) (hAB : abs (B - A) = 1) (hBC : abs (C - B) = 3) : 
  abs (C - A) = 2 ∨ abs (C - A) = 4 :=
sorry

end possible_AC_values_l48_48546


namespace parallelogram_area_l48_48986

theorem parallelogram_area (base height : ℝ) (h_base : base = 14) (h_height : height = 24) :
  base * height = 336 :=
by 
  rw [h_base, h_height]
  sorry

end parallelogram_area_l48_48986


namespace geometric_series_sum_l48_48979

theorem geometric_series_sum :
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  S = 117775277204 / 30517578125 := by
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  have : S = 117775277204 / 30517578125 := sorry
  exact this

end geometric_series_sum_l48_48979


namespace largest_a_for_integer_solution_l48_48517

theorem largest_a_for_integer_solution :
  ∃ a : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a * x + 3 * y = 1) ∧ (∀ a' : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a' * x + 3 * y = 1) → a' ≤ a) ∧ a = 1 :=
sorry

end largest_a_for_integer_solution_l48_48517


namespace randy_total_trees_l48_48791

def mango_trees : ℕ := 60
def coconut_trees : ℕ := mango_trees / 2 - 5
def total_trees (mangos coconuts : ℕ) : ℕ := mangos + coconuts

theorem randy_total_trees : total_trees mango_trees coconut_trees = 85 :=
by
  sorry

end randy_total_trees_l48_48791


namespace solve_proof_problem_l48_48014

variables (a b c d : ℝ)

noncomputable def proof_problem : Prop :=
  a = 3 * b ∧ b = 3 * c ∧ c = 5 * d → (a * c) / (b * d) = 15

theorem solve_proof_problem : proof_problem a b c d :=
by
  sorry

end solve_proof_problem_l48_48014


namespace trigonometric_expression_equals_one_l48_48720

theorem trigonometric_expression_equals_one :
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2

  (1 - 1 / cos30) * (1 + 1 / sin60) *
  (1 - 1 / sin30) * (1 + 1 / cos60) = 1 :=
by
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2
  sorry

end trigonometric_expression_equals_one_l48_48720


namespace volume_of_normal_block_is_3_l48_48369

variable (w d l : ℝ)
def V_normal : ℝ := w * d * l
def V_large : ℝ := (2 * w) * (2 * d) * (3 * l)

theorem volume_of_normal_block_is_3 (h : V_large w d l = 36) : V_normal w d l = 3 :=
by sorry

end volume_of_normal_block_is_3_l48_48369


namespace range_of_a_l48_48947

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → x + (4 / x) - 1 - a^2 + 2 * a > 0) : -1 < a ∧ a < 3 :=
sorry

end range_of_a_l48_48947


namespace solution_set_sgn_inequality_l48_48908

noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x < 0 then -1 else 0

theorem solution_set_sgn_inequality :
  {x : ℝ | (x + 1) * sgn x > 2} = {x : ℝ | x < -3} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_sgn_inequality_l48_48908


namespace sin_150_eq_half_l48_48669

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48669


namespace negation_of_exists_gt_1_l48_48486

theorem negation_of_exists_gt_1 :
  (∀ x : ℝ, x ≤ 1) ↔ ¬ (∃ x : ℝ, x > 1) :=
sorry

end negation_of_exists_gt_1_l48_48486


namespace geometric_sequence_sum_l48_48563

-- Define the problem conditions and the result
theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ), a 1 + a 2 = 16 ∧ a 3 + a 4 = 24 → a 7 + a 8 = 54 :=
by
  -- Preliminary steps and definitions to prove the theorem
  sorry

end geometric_sequence_sum_l48_48563


namespace sin_150_eq_one_half_l48_48684

theorem sin_150_eq_one_half :
  let θ := 150
  let θ_ref := 30
  let θ_complement := 180 - θ_ref
  θ = θ_complement →
  (∀ θ, θ_ref = 30 * 60.toReal)
    ∧ (∀ θ, (cos θ_ref, sin θ_ref) = (Real.sqrt 3 / 2, 1 / 2)) →
  sin θ = 1 / 2 :=
by
  intros θ θ_ref θ_complement h1 h2
  have h3 : (cos θ_complement, sin θ_complement) = (-(cos θ_ref), sin θ_ref) := by
    sorry
  rw h3
  sorry

end sin_150_eq_one_half_l48_48684


namespace percentage_women_red_and_men_dark_l48_48449

-- Define the conditions as variables
variables (w_fair_hair w_dark_hair w_red_hair m_fair_hair m_dark_hair m_red_hair : ℝ)

-- Define the percentage of women with red hair and men with dark hair
def women_red_men_dark (w_red_hair m_dark_hair : ℝ) : ℝ := w_red_hair + m_dark_hair

-- Define the main theorem to be proven
theorem percentage_women_red_and_men_dark 
  (hw_fair_hair : w_fair_hair = 30)
  (hw_dark_hair : w_dark_hair = 28)
  (hw_red_hair : w_red_hair = 12)
  (hm_fair_hair : m_fair_hair = 20)
  (hm_dark_hair : m_dark_hair = 35)
  (hm_red_hair : m_red_hair = 5) :
  women_red_men_dark w_red_hair m_dark_hair = 47 := 
sorry

end percentage_women_red_and_men_dark_l48_48449


namespace range_of_function_l48_48886

theorem range_of_function :
  ∀ x, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -3 ≤ 2 * Real.sin x - 1 ∧ 2 * Real.sin x - 1 ≤ 1 :=
by
  intros x h
  sorry

end range_of_function_l48_48886


namespace Vasya_Capital_Decreased_l48_48450

theorem Vasya_Capital_Decreased (C : ℝ) (Du Dd : ℕ) 
  (h1 : 1000 * Du - 2000 * Dd = 0)
  (h2 : Du = 2 * Dd) :
  C * ((1.1:ℝ) ^ Du) * ((0.8:ℝ) ^ Dd) < C :=
by
  -- Assuming non-zero initial capital
  have hC : C ≠ 0 := sorry
  -- Substitution of Du = 2 * Dd
  rw [h2] at h1 
  -- From h1 => 1000 * 2 * Dd - 2000 * Dd = 0 => true always
  have hfalse : true := by sorry
  -- Substitution of h2 in the Vasya capital formula
  let cf := C * ((1.1:ℝ) ^ (2 * Dd)) * ((0.8:ℝ) ^ Dd)
  -- Further simplification
  have h₀ : C * ((1.1 : ℝ) ^ 2) ^ Dd * (0.8 : ℝ) ^ Dd = cf := by sorry
  -- Calculation of the effective multiplier
  have h₁ : (1.1 : ℝ) ^ 2 = 1.21 := by sorry
  have h₂ : 1.21 * (0.8 : ℝ) = 0.968 := by sorry
  -- Conclusion from the effective multiplier being < 1
  exact sorry

end Vasya_Capital_Decreased_l48_48450


namespace geom_prog_terms_exist_l48_48204

theorem geom_prog_terms_exist (b3 b6 : ℝ) (h1 : b3 = -1) (h2 : b6 = 27 / 8) :
  ∃ (b1 q : ℝ), b1 = -4 / 9 ∧ q = -3 / 2 :=
by
  sorry

end geom_prog_terms_exist_l48_48204


namespace ratio_lt_one_l48_48476

def product_sequence (k j : ℕ) := List.prod (List.range' k j)

theorem ratio_lt_one :
  let a := product_sequence 2020 4
  let b := product_sequence 2120 4
  a / b < 1 :=
by
  sorry

end ratio_lt_one_l48_48476


namespace find_a_plus_b_l48_48734

def f (x a b : ℝ) := x^3 + a * x^2 + b * x + a^2

def extremum_at_one (a b : ℝ) : Prop :=
  f 1 a b = 10 ∧ (3 * 1^2 + 2 * a * 1 + b = 0)

theorem find_a_plus_b (a b : ℝ) (h : extremum_at_one a b) : a + b = -7 :=
by
  sorry

end find_a_plus_b_l48_48734


namespace solve_quadratic_equation_l48_48247

theorem solve_quadratic_equation (x : ℝ) :
    2 * x * (x - 5) = 3 * (5 - x) ↔ (x = 5 ∨ x = -3/2) :=
by
  sorry

end solve_quadratic_equation_l48_48247


namespace largest_common_value_less_than_1000_l48_48954

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a = 999 ∧ (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 8 * m) ∧ a < 1000 :=
by
  sorry

end largest_common_value_less_than_1000_l48_48954


namespace h_2023_eq_4052_l48_48234

theorem h_2023_eq_4052 (h : ℕ → ℕ) (h1 : h 1 = 2) (h2 : h 2 = 2) 
    (h3 : ∀ n ≥ 3, h n = h (n-1) - h (n-2) + 2 * n) : h 2023 = 4052 := 
by
  -- Use conditions as given
  sorry

end h_2023_eq_4052_l48_48234


namespace fraction_calls_by_team_B_l48_48223

-- Define the conditions
variables (A B C : ℝ)
axiom ratio_agents : A = (5 / 8) * B
axiom ratio_calls : ∀ (c : ℝ), c = (6 / 5) * C

-- Prove the fraction of the total calls processed by team B
theorem fraction_calls_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : ∀ (c : ℝ), c = (6 / 5) * C) :
  (B * C) / ((5 / 8) * B * (6 / 5) * C + B * C) = 4 / 7 :=
by {
  -- proof is omitted, so we use sorry
  sorry
}

end fraction_calls_by_team_B_l48_48223


namespace range_of_a_l48_48836

theorem range_of_a (x : ℝ) (h : 1 < x) : ∀ a, (∀ x, 1 < x → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
sorry

end range_of_a_l48_48836


namespace value_of_x_l48_48290

theorem value_of_x (x : ℝ) (h1 : |x| - 1 = 0) (h2 : x - 1 ≠ 0) : x = -1 := 
sorry

end value_of_x_l48_48290


namespace prob_X_greater_than_2_l48_48130

open ProbabilityTheory MeasureTheory

noncomputable def normalDist : Measure ℝ := measure_space.measure (continuous_probability_space.gaussian 0 σ^2)

theorem prob_X_greater_than_2 (σ : ℝ) (h1 : 0 < σ)
  (h2 : ∫ s in Ico (-2 : ℝ) 0, normalDist s = 0.4) : ∫ s in Ioi (2 : ℝ), normalDist s = 0.1 :=
sorry

end prob_X_greater_than_2_l48_48130


namespace fraction_meaningful_condition_l48_48067

theorem fraction_meaningful_condition (x : ℝ) : (4 / (x + 2) ≠ 0) ↔ (x ≠ -2) := 
by 
  sorry

end fraction_meaningful_condition_l48_48067


namespace equal_side_length_is_4_or_10_l48_48337

-- Define the conditions
def isosceles_triangle (base_length equal_side_length : ℝ) :=
  base_length = 7 ∧
  (equal_side_length > base_length ∧ equal_side_length - base_length = 3) ∨
  (equal_side_length < base_length ∧ base_length - equal_side_length = 3)

-- Lean 4 statement to prove
theorem equal_side_length_is_4_or_10 (base_length equal_side_length : ℝ) 
  (h : isosceles_triangle base_length equal_side_length) : 
  equal_side_length = 4 ∨ equal_side_length = 10 :=
by 
  sorry

end equal_side_length_is_4_or_10_l48_48337


namespace number_of_possible_values_of_a_l48_48060

theorem number_of_possible_values_of_a :
  ∃ a_values : Finset ℕ, 
    (∀ a ∈ a_values, 5 ∣ a) ∧ 
    (∀ a ∈ a_values, a ∣ 30) ∧ 
    (∀ a ∈ a_values, 0 < a) ∧ 
    a_values.card = 4 :=
by
  sorry

end number_of_possible_values_of_a_l48_48060


namespace scientific_notation_correct_l48_48191

noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
  let a := x * 10^9
  (a, -9)

theorem scientific_notation_correct :
  scientific_notation 0.000000007 = (7, -9) :=
by
  sorry

end scientific_notation_correct_l48_48191


namespace find_n_l48_48557

-- Define the operation ø
def op (x w : ℕ) : ℕ := (2 ^ x) / (2 ^ w)

-- Prove that n operating with 2 and then 1 equals 8 implies n = 3
theorem find_n (n : ℕ) (H : op (op n 2) 1 = 8) : n = 3 :=
by
  -- Proof will be provided later
  sorry

end find_n_l48_48557


namespace original_population_multiple_of_5_l48_48074

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem original_population_multiple_of_5 (x y z : ℕ) 
  (H1 : is_perfect_square (x * x)) 
  (H2 : x * x + 200 = y * y) 
  (H3 : y * y + 180 = z * z) : 
  ∃ k : ℕ, x * x = 5 * k := 
sorry

end original_population_multiple_of_5_l48_48074


namespace sales_tax_difference_l48_48457

theorem sales_tax_difference : 
  let price : Float := 50
  let tax1 : Float := 0.0725
  let tax2 : Float := 0.07
  let sales_tax1 := price * tax1
  let sales_tax2 := price * tax2
  sales_tax1 - sales_tax2 = 0.125 := 
by
  sorry

end sales_tax_difference_l48_48457


namespace correct_calculation_l48_48480

theorem correct_calculation (a : ℝ) : a^4 / a = a^3 :=
by {
  sorry
}

end correct_calculation_l48_48480


namespace find_d_minus_r_l48_48015

theorem find_d_minus_r :
  ∃ d r : ℕ, d > 1 ∧ (1059 % d = r) ∧ (1417 % d = r) ∧ (2312 % d = r) ∧ (d - r = 15) :=
sorry

end find_d_minus_r_l48_48015


namespace determine_b_when_lines_parallel_l48_48392

theorem determine_b_when_lines_parallel (b : ℝ) : 
  (∀ x y, 3 * y - 3 * b = 9 * x ↔ y - 2 = (b + 9) * x) → b = -6 :=
by
  sorry

end determine_b_when_lines_parallel_l48_48392


namespace prove_sequences_and_sum_l48_48738

theorem prove_sequences_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (a 1 = 5) →
  (a 2 = 2) →
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) →
  (∀ n, ∃ r1, (a (n + 1) - 2 * a n) = (a 2 - 2 * a 1) * r1 ^ n) ∧
  (∀ n, ∃ r2, (a (n + 1) - (1 / 2) * a n) = (a 2 - (1 / 2) * a 1) * r2 ^ n) ∧
  (∀ n, S n = (4 * n) / 3 + (4 ^ n) / 36 - 1 / 36) :=
by
  sorry

end prove_sequences_and_sum_l48_48738


namespace max_equal_product_l48_48511

theorem max_equal_product (a b c d e f : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 20) (h4 : d = 30) (h5 : e = 40) (h6 : f = 60) :
  ∃ S, (a * b * c * d * e * f) * 450 = S^3 ∧ S = 18000 := 
by
  sorry

end max_equal_product_l48_48511


namespace minimum_value_x_l48_48541

theorem minimum_value_x (a b x : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
    (H : 4 * a + b * (1 - a) = 0) 
    (Hinequality : ∀ (a b : ℝ), a > 0 → b > 0 → 
        (4 * a + b * (1 - a) = 0 → 
        (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))) : 
    x >= 1 := 
sorry

end minimum_value_x_l48_48541


namespace domain_transformation_l48_48135

-- Definitions of conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

def domain_g (x : ℝ) : Prop := 1 < x ∧ x ≤ 3

-- Theorem stating the proof problem
theorem domain_transformation : 
  (∀ x, domain_f x → 0 ≤ x+1 ∧ x+1 ≤ 4) →
  (∀ x, (0 ≤ x+1 ∧ x+1 ≤ 4) → (x-1 > 0) → domain_g x) :=
by
  intros h1 x hx
  sorry

end domain_transformation_l48_48135


namespace billy_weight_l48_48859

variable (B Bd C D : ℝ)

theorem billy_weight
  (h1 : B = Bd + 9)
  (h2 : Bd = C + 5)
  (h3 : C = D - 8)
  (h4 : C = 145)
  (h5 : D = 2 * Bd) :
  B = 85.5 :=
by
  sorry

end billy_weight_l48_48859


namespace sin_150_eq_half_l48_48718

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l48_48718


namespace range_of_m_l48_48002

theorem range_of_m (x m : ℝ) : (|x - 3| ≤ 2) → ((x - m + 1) * (x - m - 1) ≤ 0) → 
  (¬(|x - 3| ≤ 2) → ¬((x - m + 1) * (x - m - 1) ≤ 0)) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro h1 h2 h3
  sorry

end range_of_m_l48_48002


namespace total_spent_l48_48894

variable (B D : ℝ)

-- Conditions
def condition1 : Prop := D = 0.90 * B
def condition2 : Prop := B = D + 15

-- Question
theorem total_spent : condition1 B D ∧ condition2 B D → B + D = 285 := 
by
  intros h
  sorry

end total_spent_l48_48894


namespace certain_number_is_50_l48_48096

theorem certain_number_is_50 (x : ℝ) (h : 4 = 0.08 * x) : x = 50 :=
by {
    sorry
}

end certain_number_is_50_l48_48096


namespace mass_of_man_l48_48222

theorem mass_of_man (L B : ℝ) (h : ℝ) (ρ : ℝ) (V : ℝ) : L = 8 ∧ B = 3 ∧ h = 0.01 ∧ ρ = 1 ∧ V = L * 100 * B * 100 * h → V / 1000 = 240 :=
by
  sorry

end mass_of_man_l48_48222


namespace father_age_38_l48_48846

variable (F S : ℕ)
variable (h1 : S = 14)
variable (h2 : F - 10 = 7 * (S - 10))

theorem father_age_38 : F = 38 :=
by
  sorry

end father_age_38_l48_48846


namespace minimum_value_m_sq_plus_n_sq_l48_48442

theorem minimum_value_m_sq_plus_n_sq :
  ∃ (m n : ℝ), (m ≠ 0) ∧ (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 4 ∧ (m * x^2 + (2 * n + 1) * x - m - 2) = 0) ∧
  (m^2 + n^2) = 0.01 :=
by
  sorry

end minimum_value_m_sq_plus_n_sq_l48_48442


namespace player_b_wins_l48_48352

theorem player_b_wins : 
  ∃ B_strategy : (ℕ → ℕ → Prop), (∀ A_turn : ℕ → Prop, 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (A_turn i ↔ ¬ A_turn (i + 1))) → 
  ((B_strategy 1 2019) ∨ ∃ k : ℕ, 1 ≤ k ∧ k ≤ 2019 ∧ B_strategy k (k + 1) ∧ ¬ A_turn k)) :=
sorry

end player_b_wins_l48_48352


namespace sum_of_squares_pentagon_greater_icosagon_l48_48376

noncomputable def compare_sum_of_squares (R : ℝ) : Prop :=
  let a_5 := 2 * R * Real.sin (Real.pi / 5)
  let a_20 := 2 * R * Real.sin (Real.pi / 20)
  4 * a_20^2 < a_5^2

theorem sum_of_squares_pentagon_greater_icosagon (R : ℝ) : 
  compare_sum_of_squares R :=
  sorry

end sum_of_squares_pentagon_greater_icosagon_l48_48376


namespace tablet_battery_life_l48_48575

noncomputable def battery_life_remaining
  (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) : ℝ :=
  let no_use_consumption_rate := 1 / no_use_life
  let use_consumption_rate := 1 / use_life
  let no_use_time := total_on_time - use_time
  let total_battery_used := no_use_time * no_use_consumption_rate + use_time * use_consumption_rate
  let remaining_battery := 1 - total_battery_used
  remaining_battery / no_use_consumption_rate

theorem tablet_battery_life (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) :
  battery_life_remaining no_use_life use_life total_on_time use_time = 6 :=
by
  -- The proof will go here, we use sorry for now to skip the proof step.
  sorry

end tablet_battery_life_l48_48575


namespace find_x0_l48_48540

def f (x : ℝ) := x * abs x

theorem find_x0 (x0 : ℝ) (h : f x0 = 4) : x0 = 2 :=
by
  sorry

end find_x0_l48_48540


namespace correct_operation_result_l48_48578

-- Define the conditions
def original_number : ℤ := 231
def incorrect_result : ℤ := 13

-- Define the two incorrect operations and the intended corrections
def reverse_subtract : ℤ := incorrect_result + 20
def reverse_division : ℤ := reverse_subtract * 7

-- Define the intended operations
def intended_multiplication : ℤ := original_number * 7
def intended_addition : ℤ := intended_multiplication + 20

-- The theorem we need to prove
theorem correct_operation_result :
  original_number = reverse_division →
  intended_addition > 1100 :=
by
  intros h
  sorry

end correct_operation_result_l48_48578


namespace angle_C_in_parallelogram_l48_48297

theorem angle_C_in_parallelogram (ABCD : Type)
  (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : angle_A = angle_C)
  (h2 : angle_B = angle_D)
  (h3 : angle_A + angle_B = 180)
  (h4 : angle_A / angle_B = 3) :
  angle_C = 135 :=
  sorry

end angle_C_in_parallelogram_l48_48297


namespace increase_in_average_weight_l48_48192

variable {A X : ℝ}

-- Given initial conditions
axiom average_initial_weight_8 : X = (8 * A - 62 + 90) / 8 - A

-- The goal to prove
theorem increase_in_average_weight : X = 3.5 :=
by
  sorry

end increase_in_average_weight_l48_48192


namespace Y_minus_X_eq_92_l48_48037

def arithmetic_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def X : ℕ := arithmetic_sum 10 2 46
def Y : ℕ := arithmetic_sum 12 2 46

theorem Y_minus_X_eq_92 : Y - X = 92 := by
  sorry

end Y_minus_X_eq_92_l48_48037


namespace xiao_ming_equation_l48_48985

-- Defining the parameters of the problem
def distance : ℝ := 2000
def regular_time (x : ℝ) := x
def increased_speed := 5
def time_saved := 2

-- Problem statement to be proven in Lean 4:
theorem xiao_ming_equation (x : ℝ) (h₁ : x > 2) : 
  (distance / (x - time_saved)) - (distance / regular_time x) = increased_speed :=
by
  sorry

end xiao_ming_equation_l48_48985


namespace round_trip_ticket_percentage_l48_48323

theorem round_trip_ticket_percentage (p : ℕ → Prop) : 
  (∀ n, p n → n = 375) → (∀ n, p n → n = 375) :=
by
  sorry

end round_trip_ticket_percentage_l48_48323


namespace integer_1000_column_l48_48380

def column_sequence (n : ℕ) : String :=
  let sequence := ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"]
  sequence.get! (n % 10)

theorem integer_1000_column : column_sequence 999 = "C" :=
by
  sorry

end integer_1000_column_l48_48380


namespace triangle_fraction_correct_l48_48721

def point : Type := ℤ × ℤ

def area_triangle (A B C : point) : ℚ :=
  (1 / 2 : ℚ) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℚ))

def area_grid (length width : ℚ) : ℚ :=
  length * width

noncomputable def fraction_covered (A B C : point) (grid_length grid_width : ℚ) : ℚ :=
  area_triangle A B C / area_grid grid_length grid_width

theorem triangle_fraction_correct :
  fraction_covered (-2, 3) (2, -2) (3, 5) 8 6 = 11 / 32 :=
by
  sorry

end triangle_fraction_correct_l48_48721


namespace problem1_problem2_problem3_l48_48273

-- Define the function f
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b - 2^x) / (2^(x + 1) + a)

-- Problem 1
theorem problem1 (h_odd : ∀ x, f x a b = -f (-x) a b) : a = 2 ∧ b = 1 :=
sorry

-- Problem 2
theorem problem2 : (∀ x, f x 2 1 = -f (-x) 2 1) → ∀ x y, x < y → f x 2 1 > f y 2 1 :=
sorry

-- Problem 3
theorem problem3 (h_pos : ∀ x ≥ 1, f (k * 3^x) 2 1 + f (3^x - 9^x + 2) 2 1 > 0) : k < 4 / 3 :=
sorry

end problem1_problem2_problem3_l48_48273


namespace sum_of_acutes_tan_eq_pi_over_4_l48_48314

theorem sum_of_acutes_tan_eq_pi_over_4 {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h : (1 + Real.tan α) * (1 + Real.tan β) = 2) : α + β = π / 4 :=
sorry

end sum_of_acutes_tan_eq_pi_over_4_l48_48314


namespace possible_values_of_a2b_b2c_c2a_l48_48173

theorem possible_values_of_a2b_b2c_c2a (a b c : ℝ) (h : a + b + c = 1) : ∀ x : ℝ, ∃ a b c : ℝ, a + b + c = 1 ∧ a^2 * b + b^2 * c + c^2 * a = x :=
by
  sorry

end possible_values_of_a2b_b2c_c2a_l48_48173


namespace students_answered_both_correctly_l48_48322

theorem students_answered_both_correctly (x y z w total : ℕ) (h1 : x = 22) (h2 : y = 20) 
  (h3 : z = 3) (h4 : total = 25) (h5 : x + y - w - z = total) : w = 17 :=
by
  sorry

end students_answered_both_correctly_l48_48322


namespace find_line_equation_l48_48007

theorem find_line_equation 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1)
  (P : ℝ × ℝ) (P_coord : P = (1, 3/2))
  (line_l : ∀ x : ℝ, ℝ)
  (line_eq : ∀ x : ℝ, y = k * x + b) 
  (intersects : ∀ A B : ℝ × ℝ, A ≠ P ∧ B ≠ P)
  (perpendicular : ∀ A B : ℝ × ℝ, (A.1 - 1) * (B.1 - 1) + (A.2 - 3 / 2) * (B.2 - 3 / 2) = 0)
  (bisected_by_y_axis : ∀ A B : ℝ × ℝ, A.1 + B.1 = 0) :
  ∃ k : ℝ, k = 3 / 2 ∨ k = -3 / 2 :=
sorry

end find_line_equation_l48_48007


namespace calculate_expression_l48_48866

theorem calculate_expression : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  calc
    10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2
    _ = 10 - 9 + 56 + 6 - 20 + 3 - 2 : by rw [mul_comm 8 7, mul_comm 5 4] -- Perform multiplications
    _ = 1 + 56 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 10 - 9
    _ = 57 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 1 + 56
    _ = 63 - 20 + 3 - 2 : by norm_num  -- Simplify 57 + 6
    _ = 43 + 3 - 2 : by norm_num -- Simplify 63 - 20
    _ = 46 - 2 : by norm_num -- Simplify 43 + 3
    _ = 44 : by norm_num -- Simplify 46 - 2

end calculate_expression_l48_48866


namespace central_angle_unchanged_l48_48964

theorem central_angle_unchanged (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0) :
  (s / r) = (2 * s / (2 * r)) :=
by
  sorry

end central_angle_unchanged_l48_48964


namespace total_number_of_balls_l48_48619

def number_of_yellow_balls : Nat := 6
def probability_yellow_ball : Rat := 1 / 9

theorem total_number_of_balls (N : Nat) (h1 : number_of_yellow_balls = 6) (h2 : probability_yellow_ball = 1 / 9) :
    6 / N = 1 / 9 → N = 54 := 
by
  sorry

end total_number_of_balls_l48_48619


namespace necessary_and_sufficient_condition_l48_48764

variables {a : ℕ → ℝ}
-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define the monotonically increasing condition
def is_monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

-- Define the specific statement
theorem necessary_and_sufficient_condition (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 < a 3 ↔ is_monotonically_increasing a) :=
by sorry

end necessary_and_sufficient_condition_l48_48764


namespace frosting_problem_l48_48654

-- Define the conditions
def cagney_rate := 1/15  -- Cagney's rate in cupcakes per second
def lacey_rate := 1/45   -- Lacey's rate in cupcakes per second
def total_time := 600  -- Total time in seconds (10 minutes)

-- Function to calculate the combined rate
def combined_rate (r1 r2 : ℝ) : ℝ := r1 + r2

-- Hypothesis combining the conditions
def hypothesis : Prop :=
  combined_rate cagney_rate lacey_rate = 1/11.25

-- Statement to prove: together they can frost 53 cupcakes within 10 minutes 
theorem frosting_problem : ∀ (total_time: ℝ) (hyp : hypothesis),
  total_time / (cagney_rate + lacey_rate) = 53 :=
by
  intro total_time hyp
  sorry

end frosting_problem_l48_48654


namespace min_value_fraction_108_l48_48044

noncomputable def min_value_fraction (x y z w : ℝ) : ℝ :=
(x + y) / (x * y * z * w)

theorem min_value_fraction_108 (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) (h_sum : x + y + z + w = 1) :
  min_value_fraction x y z w = 108 :=
sorry

end min_value_fraction_108_l48_48044


namespace pebble_sequence_10_l48_48246

-- A definition for the sequence based on the given conditions and pattern.
def pebble_sequence : ℕ → ℕ
| 0 => 1
| 1 => 5
| 2 => 12
| 3 => 22
| (n + 4) => pebble_sequence (n + 3) + (3 * (n + 1) + 1)

-- Theorem that states the value at the 10th position in the sequence.
theorem pebble_sequence_10 : pebble_sequence 9 = 145 :=
sorry

end pebble_sequence_10_l48_48246


namespace modified_counting_game_53rd_term_l48_48928

theorem modified_counting_game_53rd_term :
  let a : ℕ := 1
  let d : ℕ := 2
  a + (53 - 1) * d = 105 :=
by 
  sorry

end modified_counting_game_53rd_term_l48_48928


namespace quadratic_equal_roots_k_value_l48_48141

theorem quadratic_equal_roots_k_value (k : ℝ) :
  (∃ x : ℝ, x^2 + k - 3 = 0 ∧ (0^2 - 4 * 1 * (k - 3) = 0)) → k = 3 := 
by
  sorry

end quadratic_equal_roots_k_value_l48_48141


namespace max_f_geq_l48_48459

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x) + Real.sin (3 * x)

theorem max_f_geq (x : ℝ) : ∃ x, f x ≥ (3 + Real.sqrt 3) / 2 := sorry

end max_f_geq_l48_48459


namespace max_trees_cut_l48_48840

theorem max_trees_cut (n : ℕ) (h : n = 2001) :
  (∃ m : ℕ, m = n * n ∧ ∀ (x y : ℕ), x < n ∧ y < n → (x % 2 = 0 ∧ y % 2 = 0 → m = 1001001)) := sorry

end max_trees_cut_l48_48840


namespace negation_of_prop_l48_48809

open Classical

theorem negation_of_prop (h : ∀ x : ℝ, x^2 + x + 1 > 0) : ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
sorry

end negation_of_prop_l48_48809


namespace tracey_initial_candies_l48_48471

theorem tracey_initial_candies (x : ℕ) :
  (x % 4 = 0) ∧ (104 ≤ x) ∧ (x ≤ 112) ∧
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ 6 ∧ (x / 2 - 40 - k = 10)) →
  (x = 108 ∨ x = 112) :=
by
  sorry

end tracey_initial_candies_l48_48471


namespace enlarged_sticker_height_l48_48829

theorem enlarged_sticker_height (original_width original_height new_width : ℕ) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 12) : (new_width / original_width) * original_height = 8 := 
by 
  -- Prove the height of the enlarged sticker is 8 inches
  sorry

end enlarged_sticker_height_l48_48829


namespace find_n_l48_48487

theorem find_n (n : ℕ) (h : ∀ x : ℝ, (n : ℝ) < x ∧ x < (n + 1 : ℝ) → 3 * x - 5 = 0) :
  n = 1 :=
sorry

end find_n_l48_48487


namespace price_of_sugar_and_salt_l48_48345

theorem price_of_sugar_and_salt:
  (∀ (sugar_price salt_price : ℝ), 2 * sugar_price + 5 * salt_price = 5.50 ∧ sugar_price = 1.50 →
  3 * sugar_price + salt_price = 5) := 
by 
  sorry

end price_of_sugar_and_salt_l48_48345


namespace irreducible_fraction_denominator_l48_48404

theorem irreducible_fraction_denominator :
  let num := 201920192019
  let denom := 191719171917
  let gcd_num_denom := Int.gcd num denom
  let irreducible_denom := denom / gcd_num_denom
  irreducible_denom = 639 :=
by
  sorry

end irreducible_fraction_denominator_l48_48404


namespace range_of_a_l48_48277

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 0 → a * 4^x - 2^x + 2 > 0) → a > -1 :=
by sorry

end range_of_a_l48_48277


namespace simplify_polynomial_l48_48794

noncomputable def f (r : ℝ) : ℝ := 2 * r^3 + r^2 + 4 * r - 3
noncomputable def g (r : ℝ) : ℝ := r^3 + r^2 + 6 * r - 8

theorem simplify_polynomial (r : ℝ) : f r - g r = r^3 - 2 * r + 5 := by
  sorry

end simplify_polynomial_l48_48794


namespace cos_sum_identity_l48_48424

-- Definitions based on given conditions
def geometric_sequence (a b c : ℝ) : Prop :=
  (a / b = b / c)

def triangle_sine_relations (A B C R : ℝ) : Prop :=
  ∃ a b c, a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C

-- The proof problem
theorem cos_sum_identity (A B C R : ℝ) (a b c : ℝ)
    (geo_seq : geometric_sequence a b c)
    (tri_rel : triangle_sine_relations A B C R)
    (ha : a = 2 * R * sin A) (hb : b = 2 * R * sin B) (hc : c = 2 * R * sin C) :
  cos (2 * B) + cos B + cos (A - C) = cos B + cos (A - C) :=
by sorry

end cos_sum_identity_l48_48424


namespace simplify_expression_l48_48228

theorem simplify_expression (x y : ℝ) (hxy : x ≠ y) : 
  ((x - y) ^ 3 / (x - y) ^ 2) * (y - x) = -(x - y) ^ 2 := 
by
  sorry

end simplify_expression_l48_48228


namespace min_value_of_expression_l48_48086

open Real

noncomputable def min_expression_value : ℝ :=
  let expr := λ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y + 25
  0

theorem min_value_of_expression : ∃ x y : ℝ, x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = min_expression_value :=
by {
  use [4, -3],
  split,
  { refl },
  split,
  { refl },
  sorry
}

end min_value_of_expression_l48_48086


namespace floor_plus_self_eq_l48_48881

theorem floor_plus_self_eq (r : ℝ) (h : ⌊r⌋ + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_self_eq_l48_48881


namespace guacamole_serving_and_cost_l48_48730

theorem guacamole_serving_and_cost 
  (initial_avocados : ℕ) 
  (additional_avocados : ℕ) 
  (avocados_per_serving : ℕ) 
  (x : ℝ) 
  (h_initial : initial_avocados = 5) 
  (h_additional : additional_avocados = 4) 
  (h_serving : avocados_per_serving = 3) :
  (initial_avocados + additional_avocados) / avocados_per_serving = 3 
  ∧ additional_avocados * x = 4 * x := by
  sorry

end guacamole_serving_and_cost_l48_48730


namespace fraction_of_tips_l48_48999

variable (S T : ℝ) -- assuming S is salary and T is tips
variable (h : T / (S + T) = 0.7142857142857143)

/-- 
If the fraction of the waiter's income from tips is 0.7142857142857143,
then the fraction of his salary that were his tips is 2.5.
-/
theorem fraction_of_tips (h : T / (S + T) = 0.7142857142857143) : T / S = 2.5 :=
sorry

end fraction_of_tips_l48_48999


namespace sum_of_terms_in_arithmetic_sequence_eq_l48_48566

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_terms_in_arithmetic_sequence_eq :
  arithmetic_sequence a →
  (a 2 + a 3 + a 10 + a 11 = 36) →
  (a 3 + a 10 = 18) :=
by
  intros h_seq h_sum
  -- Proof placeholder
  sorry

end sum_of_terms_in_arithmetic_sequence_eq_l48_48566


namespace prime_factors_count_900_l48_48550

theorem prime_factors_count_900 : 
  ∃ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x ∧ x ∣ 900) ∧ S.card = 3 :=
by 
  sorry

end prime_factors_count_900_l48_48550


namespace largest_reciprocal_l48_48221

theorem largest_reciprocal :
  let a := (2 : ℚ) / 7
  let b := (3 : ℚ) / 8
  let c := (1 : ℚ)
  let d := (4 : ℚ)
  let e := (2000 : ℚ)
  1 / a > 1 / b ∧ 1 / a > 1 / c ∧ 1 / a > 1 / d ∧ 1 / a > 1 / e := 
by
  sorry

end largest_reciprocal_l48_48221


namespace solve_ab_cd_l48_48285

theorem solve_ab_cd (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -2) 
  (h3 : a + c + d = 5) 
  (h4 : b + c + d = 4) 
  : a * b + c * d = 26 / 9 := 
by {
  sorry
}

end solve_ab_cd_l48_48285


namespace part_i_part_ii_l48_48041

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + x - a
noncomputable def g (x a : ℝ) : ℝ := Real.sqrt (f x a)

theorem part_i (a : ℝ) :
  (∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f x a ≥ 0) ↔ (a ≤ 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

theorem part_ii (a : ℝ) :
  (∃ x0 y0 : ℝ, (x0, y0) ∈ (Set.Icc (-1) 1) ∧ y0 = Real.cos (2 * x0) ∧ g (g y0 a) a = y0) ↔ (1 ≤ a ∧ a ≤ Real.exp 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

end part_i_part_ii_l48_48041


namespace all_numbers_positive_l48_48051

noncomputable def condition (a : Fin 9 → ℝ) : Prop :=
  ∀ (S : Finset (Fin 9)), S.card = 4 → S.sum (a : Fin 9 → ℝ) < (Finset.univ \ S).sum (a : Fin 9 → ℝ)

theorem all_numbers_positive (a : Fin 9 → ℝ) (h : condition a) : ∀ i, 0 < a i :=
by
  sorry

end all_numbers_positive_l48_48051


namespace lunch_cost_calc_l48_48181

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

end lunch_cost_calc_l48_48181


namespace monotonically_increasing_interval_l48_48462

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

theorem monotonically_increasing_interval : 
  ∀ x ∈ Set.Icc (-Real.pi) 0, 
  x ∈ Set.Icc (-Real.pi/6) 0 ↔ deriv f x = 0 := sorry

end monotonically_increasing_interval_l48_48462


namespace exists_unique_t_exists_m_pos_l48_48419

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem exists_unique_t (m : ℝ) (h : m = 1) : 
  ∃! (t : ℝ), t ∈ Set.Ioc (1 / 2) 1 ∧ deriv (f 1) t = 0 := sorry

theorem exists_m_pos : ∃ (m : ℝ), 0 < m ∧ m < 1 ∧ ∀ (x : ℝ), 0 < x → f m x > 0 := sorry

end exists_unique_t_exists_m_pos_l48_48419


namespace sequence_formula_l48_48984

-- Define the properties of the sequence
axiom seq_prop_1 (a : ℕ → ℝ) (m n : ℕ) (h : m > n) : a (m - n) = a m - a n

axiom seq_increasing (a : ℕ → ℝ) : ∀ n m : ℕ, n < m → a n < a m

-- Formulate the theorem to prove the general sequence formula
theorem sequence_formula (a : ℕ → ℝ) (h1 : ∀ m n : ℕ, m > n → a (m - n) = a m - a n)
    (h2 : ∀ n m : ℕ, n < m → a n < a m) :
    ∃ k > 0, ∀ n, a n = k * n :=
sorry

end sequence_formula_l48_48984


namespace solve_quadratic_eq_l48_48347

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2024 * x ↔ x = 0 ∨ x = 2024 :=
by sorry

end solve_quadratic_eq_l48_48347


namespace abel_arrival_earlier_l48_48108

variable (distance : ℕ) (speed_abel : ℕ) (speed_alice : ℕ) (start_delay_alice : ℕ)

theorem abel_arrival_earlier (h_dist : distance = 1000) 
                             (h_speed_abel : speed_abel = 50) 
                             (h_speed_alice : speed_alice = 40) 
                             (h_start_delay : start_delay_alice = 1) : 
                             (start_delay_alice + distance / speed_alice) * 60 - (distance / speed_abel) * 60 = 360 :=
by
  sorry

end abel_arrival_earlier_l48_48108


namespace average_age_6_members_birth_correct_l48_48801

/-- The average age of 7 members of a family is 29 years. -/
def average_age_7_members := 29

/-- The present age of the youngest member is 5 years. -/
def age_youngest_member := 5

/-- Total age of 7 members of the family -/
def total_age_7_members := 7 * average_age_7_members

/-- Total age of 6 members at present -/
def total_age_6_members_present := total_age_7_members - age_youngest_member

/-- Total age of 6 members at time of birth of youngest member -/
def total_age_6_members_birth := total_age_6_members_present - (6 * age_youngest_member)

/-- Average age of 6 members at time of birth of youngest member -/
def average_age_6_members_birth := total_age_6_members_birth / 6

/-- Prove the average age of 6 members at the time of birth of the youngest member -/
theorem average_age_6_members_birth_correct :
  average_age_6_members_birth = 28 :=
by
  sorry

end average_age_6_members_birth_correct_l48_48801


namespace greatest_possible_value_of_a_l48_48602

theorem greatest_possible_value_of_a 
  (x a : ℤ)
  (h : x^2 + a * x = -21)
  (ha_pos : 0 < a)
  (hx_int : x ∈ [-21, -7, -3, -1].toFinset): 
  a ≤ 22 := sorry

end greatest_possible_value_of_a_l48_48602


namespace inv_g_of_43_div_16_l48_48439

noncomputable def g (x : ℚ) : ℚ := (x^3 - 5) / 4

theorem inv_g_of_43_div_16 : g (3 * (↑7)^(1/3) / 2) = 43 / 16 :=
by 
  sorry

end inv_g_of_43_div_16_l48_48439
