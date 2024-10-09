import Mathlib

namespace no_product_equal_remainder_l810_81057

theorem no_product_equal_remainder (n : ℤ) : 
  ¬ (n = (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 1) = n * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 2) = n * (n + 1) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 3) = n * (n + 1) * (n + 2) * (n + 4) * (n + 5) ∨
     (n + 4) = n * (n + 1) * (n + 2) * (n + 3) * (n + 5) ∨
     (n + 5) = n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end no_product_equal_remainder_l810_81057


namespace birds_count_is_30_l810_81012

def total_animals : ℕ := 77
def number_of_kittens : ℕ := 32
def number_of_hamsters : ℕ := 15

def number_of_birds : ℕ := total_animals - number_of_kittens - number_of_hamsters

theorem birds_count_is_30 : number_of_birds = 30 := by
  sorry

end birds_count_is_30_l810_81012


namespace kids_go_to_camp_l810_81034

-- Define the total number of kids in Lawrence County
def total_kids : ℕ := 1059955

-- Define the number of kids who stay home
def stay_home : ℕ := 495718

-- Define the expected number of kids who go to camp
def expected_go_to_camp : ℕ := 564237

-- The theorem to prove the number of kids who go to camp
theorem kids_go_to_camp :
  total_kids - stay_home = expected_go_to_camp :=
by
  -- Proof is omitted
  sorry

end kids_go_to_camp_l810_81034


namespace find_b_l810_81069

theorem find_b (a b c : ℕ) (h1 : a + b + c = 99) (h2 : a + 6 = b - 6) (h3 : b - 6 = 5 * c) : b = 51 :=
sorry

end find_b_l810_81069


namespace one_cow_one_bag_in_39_days_l810_81024

-- Definitions
def cows : ℕ := 52
def husks : ℕ := 104
def days : ℕ := 78

-- Problem: Given that 52 cows eat 104 bags of husk in 78 days,
-- Prove that one cow will eat one bag of husk in 39 days.
theorem one_cow_one_bag_in_39_days (cows_cons : cows = 52) (husks_cons : husks = 104) (days_cons : days = 78) :
  ∃ d : ℕ, d = 39 :=
by
  -- Placeholder for the proof.
  sorry

end one_cow_one_bag_in_39_days_l810_81024


namespace evaluate_seventy_two_square_minus_twenty_four_square_l810_81002

theorem evaluate_seventy_two_square_minus_twenty_four_square :
  72 ^ 2 - 24 ^ 2 = 4608 := 
by {
  sorry
}

end evaluate_seventy_two_square_minus_twenty_four_square_l810_81002


namespace three_term_arithmetic_seq_l810_81008

noncomputable def arithmetic_sequence_squares (x y z : ℤ) : Prop :=
  x^2 + z^2 = 2 * y^2

theorem three_term_arithmetic_seq (x y z : ℤ) :
  (∃ a b : ℤ, a = (x + z) / 2 ∧ b = (x - z) / 2 ∧ x^2 + z^2 = 2 * y^2) ↔
  arithmetic_sequence_squares x y z :=
by
  sorry

end three_term_arithmetic_seq_l810_81008


namespace tom_total_spent_correct_l810_81036

-- Definitions for discount calculations
def original_price_skateboard : ℝ := 9.46
def discount_rate_skateboard : ℝ := 0.10
def discounted_price_skateboard : ℝ := original_price_skateboard * (1 - discount_rate_skateboard)

def original_price_marbles : ℝ := 9.56
def discount_rate_marbles : ℝ := 0.10
def discounted_price_marbles : ℝ := original_price_marbles * (1 - discount_rate_marbles)

def price_shorts : ℝ := 14.50

def original_price_action_figures : ℝ := 12.60
def discount_rate_action_figures : ℝ := 0.20
def discounted_price_action_figures : ℝ := original_price_action_figures * (1 - discount_rate_action_figures)

-- Total for all discounted items
def total_discounted_items : ℝ := 
  discounted_price_skateboard + discounted_price_marbles + price_shorts + discounted_price_action_figures

-- Currency conversion for video game
def price_video_game_eur : ℝ := 20.50
def exchange_rate_eur_to_usd : ℝ := 1.12
def price_video_game_usd : ℝ := price_video_game_eur * exchange_rate_eur_to_usd

-- Total amount spent including the video game
def total_spent : ℝ := total_discounted_items + price_video_game_usd

-- Lean proof statement
theorem tom_total_spent_correct :
  total_spent = 64.658 :=
by {
  -- This is a placeholder "by sorry" which means the proof is missing.
  sorry
}

end tom_total_spent_correct_l810_81036


namespace john_drove_total_distance_l810_81009

-- Define different rates and times for John's trip
def rate1 := 45 -- mph
def rate2 := 55 -- mph
def time1 := 2 -- hours
def time2 := 3 -- hours

-- Define the distances for each segment of the trip
def distance1 := rate1 * time1
def distance2 := rate2 * time2

-- Define the total distance
def total_distance := distance1 + distance2

-- The theorem to prove that John drove 255 miles in total
theorem john_drove_total_distance : total_distance = 255 :=
by
  sorry

end john_drove_total_distance_l810_81009


namespace students_without_favorite_subject_l810_81073

theorem students_without_favorite_subject (total_students : ℕ) (like_math : ℕ) (like_english : ℕ) (like_science : ℕ) :
  total_students = 30 →
  like_math = total_students * 1 / 5 →
  like_english = total_students * 1 / 3 →
  like_science = (total_students - (like_math + like_english)) * 1 / 7 →
  total_students - (like_math + like_english + like_science) = 12 :=
by
  intro h_total h_math h_english h_science
  sorry

end students_without_favorite_subject_l810_81073


namespace children_to_add_l810_81026

def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def desired_children := 30

theorem children_to_add : (desired_children - children) = 10 := by
  sorry

end children_to_add_l810_81026


namespace total_blue_balloons_l810_81028

theorem total_blue_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h_joan : joan_balloons = 40) (h_melanie : melanie_balloons = 41) : joan_balloons + melanie_balloons = 81 := by
  sorry

end total_blue_balloons_l810_81028


namespace sampling_is_systematic_l810_81060

-- Conditions
def production_line (units_per_day : ℕ) : Prop := units_per_day = 128

def sampling_inspection (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  samples_per_day = 8 ∧ inspection_time = 30 ∧ inspection_days = 7

-- Question
def sampling_method (method : String) (units_per_day : ℕ) (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  production_line units_per_day ∧ sampling_inspection samples_per_day inspection_time inspection_days → method = "systematic sampling"

-- Theorem stating the question == answer given conditions
theorem sampling_is_systematic : sampling_method "systematic sampling" 128 8 30 7 :=
by
  sorry

end sampling_is_systematic_l810_81060


namespace f_odd_and_increasing_l810_81049

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) := sorry

end f_odd_and_increasing_l810_81049


namespace bob_questions_three_hours_l810_81090

theorem bob_questions_three_hours : 
  let first_hour := 13
  let second_hour := first_hour * 2
  let third_hour := second_hour * 2
  first_hour + second_hour + third_hour = 91 :=
by
  sorry

end bob_questions_three_hours_l810_81090


namespace no_non_trivial_power_ending_222_l810_81096

theorem no_non_trivial_power_ending_222 (x y : ℕ) (hx : x > 1) (hy : y > 1) : ¬ (∃ n : ℕ, n % 1000 = 222 ∧ n = x^y) :=
by
  sorry

end no_non_trivial_power_ending_222_l810_81096


namespace zero_clever_numbers_l810_81064

def isZeroClever (n : Nat) : Prop :=
  ∃ a b c : Nat, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
  n = 1000 * a + 10 * b + c ∧
  n = 9 * (100 * a + 10 * b + c)

theorem zero_clever_numbers :
  ∀ n : Nat, isZeroClever n → n = 2025 ∨ n = 4050 ∨ n = 6075 :=
by
  -- Proof to be provided
  sorry

end zero_clever_numbers_l810_81064


namespace side_length_S2_l810_81037

def square_side_length 
  (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : Prop :=
  s = 650

theorem side_length_S2 (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : square_side_length w h R1 R2 S1 S2 S3 r s combined_rectangle cond1 cond2 cond3 cond4 cond5 cond6 cond7 cond8 :=
sorry

end side_length_S2_l810_81037


namespace grain_spilled_l810_81044

def original_grain : ℕ := 50870
def remaining_grain : ℕ := 918

theorem grain_spilled : (original_grain - remaining_grain) = 49952 :=
by
  -- Proof goes here
  sorry

end grain_spilled_l810_81044


namespace cannot_form_set_l810_81061

/-- Define the set of non-negative real numbers not exceeding 20 --/
def setA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 20}

/-- Define the set of solutions of the equation x^2 - 9 = 0 within the real numbers --/
def setB : Set ℝ := {x | x^2 - 9 = 0}

/-- Define the set of all students taller than 170 cm enrolled in a certain school in the year 2013 --/
def setC : Type := sorry

/-- Define the (pseudo) set of all approximate values of sqrt(3) --/
def pseudoSetD : Set ℝ := {x | x = Real.sqrt 3}

/-- Main theorem stating that setD cannot form a mathematically valid set --/
theorem cannot_form_set (x : ℝ) : x ∈ pseudoSetD → False := sorry

end cannot_form_set_l810_81061


namespace carl_first_to_roll_six_l810_81078

-- Definitions based on problem conditions
def prob_six := 1 / 6
def prob_not_six := 5 / 6

-- Define geometric series sum formula for the given context
theorem carl_first_to_roll_six :
  ∑' n : ℕ, (prob_not_six^(3*n+1) * prob_six) = 25 / 91 :=
by
  sorry

end carl_first_to_roll_six_l810_81078


namespace rhombus_diagonals_sum_squares_l810_81032

-- Definition of the rhombus side length condition
def is_rhombus_side_length (side_length : ℝ) : Prop :=
  side_length = 2

-- Lean 4 statement for the proof problem
theorem rhombus_diagonals_sum_squares (side_length : ℝ) (d1 d2 : ℝ) 
  (h : is_rhombus_side_length side_length) :
  side_length = 2 → (d1^2 + d2^2 = 16) :=
by
  sorry

end rhombus_diagonals_sum_squares_l810_81032


namespace solve_for_x_l810_81003

theorem solve_for_x (x : ℝ) (h : 1 - 2 * (1 / (1 + x)) = 1 / (1 + x)) : x = 2 := 
  sorry

end solve_for_x_l810_81003


namespace alyosha_cube_problem_l810_81010

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l810_81010


namespace tangent_product_le_one_third_l810_81083

theorem tangent_product_le_one_third (α β : ℝ) (h : α + β = π / 3) (hα : 0 < α) (hβ : 0 < β) : 
  Real.tan α * Real.tan β ≤ 1 / 3 :=
sorry

end tangent_product_le_one_third_l810_81083


namespace solid_color_marble_percentage_l810_81066

theorem solid_color_marble_percentage (solid striped dotted swirl red blue green yellow purple : ℝ)
  (h_solid: solid = 0.7) (h_striped: striped = 0.1) (h_dotted: dotted = 0.1) (h_swirl: swirl = 0.1)
  (h_red: red = 0.25) (h_blue: blue = 0.25) (h_green: green = 0.2) (h_yellow: yellow = 0.15) (h_purple: purple = 0.15) :
  solid * (red + blue + green) * 100 = 49 :=
by
  sorry

end solid_color_marble_percentage_l810_81066


namespace fuel_consumption_l810_81025

-- Define the initial conditions based on the problem
variable (s Q : ℝ)

-- Distance and fuel data points
def data_points : List (ℝ × ℝ) := [(0, 50), (100, 42), (200, 34), (300, 26), (400, 18)]

-- Define the function Q and required conditions
theorem fuel_consumption :
  (∀ p ∈ data_points, ∃ k b, Q = k * s + b ∧
    ((p.1 = 0 → b = 50) ∧
     (p.1 = 100 → Q = 42 → k = -0.08))) :=
by
  sorry

end fuel_consumption_l810_81025


namespace money_sum_l810_81039

theorem money_sum (A B C : ℕ) (h1 : A + C = 300) (h2 : B + C = 600) (h3 : C = 200) : A + B + C = 700 :=
by
  sorry

end money_sum_l810_81039


namespace largest_possible_distance_between_spheres_l810_81023

noncomputable def largest_distance_between_spheres : ℝ :=
  110 + Real.sqrt 1818

theorem largest_possible_distance_between_spheres :
  let center1 := (3, -5, 7)
  let radius1 := 15
  let center2 := (-10, 20, -25)
  let radius2 := 95
  ∀ A B : ℝ × ℝ × ℝ,
    (dist A center1 = radius1) →
    (dist B center2 = radius2) →
    dist A B ≤ largest_distance_between_spheres :=
  sorry

end largest_possible_distance_between_spheres_l810_81023


namespace solve_recurrence_relation_l810_81019

def recurrence_relation (a : ℕ → ℤ) : Prop :=
  ∀ n ≥ 3, a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) + 24 * n - 6

def initial_conditions (a : ℕ → ℤ) : Prop :=
  a 0 = -4 ∧ a 1 = -2 ∧ a 2 = 2

def explicit_solution (n : ℕ) : ℤ :=
  -4 + 17 * n - 21 * n^2 + 5 * n^3 + n^4

theorem solve_recurrence_relation :
  ∀ (a : ℕ → ℤ),
    recurrence_relation a →
    initial_conditions a →
    ∀ n, a n = explicit_solution n := by
  intros a h_recur h_init n
  sorry

end solve_recurrence_relation_l810_81019


namespace find_x_l810_81035

theorem find_x (x : ℤ) (h_pos : x > 0) 
  (n := x^2 + 2 * x + 17) 
  (d := 2 * x + 5)
  (h_div : n = d * x + 7) : x = 2 := 
sorry

end find_x_l810_81035


namespace juan_original_number_l810_81052

theorem juan_original_number (x : ℝ) (h : (3 * (x + 3) - 4) / 2 = 10) : x = 5 :=
by
  sorry

end juan_original_number_l810_81052


namespace potato_bag_weight_l810_81081

theorem potato_bag_weight :
  ∃ w : ℝ, w = 16 / (w / 4) ∧ w = 16 := 
by
  sorry

end potato_bag_weight_l810_81081


namespace cole_drive_time_l810_81095

noncomputable def T_work (D : ℝ) : ℝ := D / 75
noncomputable def T_home (D : ℝ) : ℝ := D / 105

theorem cole_drive_time (v1 v2 T : ℝ) (D : ℝ) 
  (h_v1 : v1 = 75) (h_v2 : v2 = 105) (h_T : T = 4)
  (h_round_trip : T_work D + T_home D = T) : 
  T_work D = 140 / 60 :=
sorry

end cole_drive_time_l810_81095


namespace correct_number_of_three_digit_numbers_l810_81054

def count_valid_three_digit_numbers : Nat :=
  let hundreds := [1, 2, 3, 4, 6, 7, 9].length
  let tens_units := [0, 1, 2, 3, 4, 6, 7, 9].length
  hundreds * tens_units * tens_units

theorem correct_number_of_three_digit_numbers :
  count_valid_three_digit_numbers = 448 :=
by
  unfold count_valid_three_digit_numbers
  sorry

end correct_number_of_three_digit_numbers_l810_81054


namespace total_volume_of_cubes_l810_81004

theorem total_volume_of_cubes :
  let sarah_side_length := 3
  let sarah_num_cubes := 8
  let tom_side_length := 4
  let tom_num_cubes := 4
  let sarah_volume := sarah_num_cubes * sarah_side_length^3
  let tom_volume := tom_num_cubes * tom_side_length^3
  sarah_volume + tom_volume = 472 := by
  -- Definitions coming from conditions
  let sarah_side_length := 3
  let sarah_num_cubes := 8
  let tom_side_length := 4
  let tom_num_cubes := 4
  let sarah_volume := sarah_num_cubes * sarah_side_length^3
  let tom_volume := tom_num_cubes * tom_side_length^3
  -- Total volume of all cubes
  have h : sarah_volume + tom_volume = 472 := by sorry

  exact h

end total_volume_of_cubes_l810_81004


namespace part_a_l810_81062

theorem part_a (n : ℤ) (m : ℤ) (h : m = n + 2) : 
  n * m + 1 = (n + 1) ^ 2 := by
  sorry

end part_a_l810_81062


namespace sum_of_ages_l810_81033

variable (P_years Q_years : ℝ) (D_years : ℝ)

-- conditions
def condition_1 : Prop := Q_years = 37.5
def condition_2 : Prop := P_years = 3 * (Q_years - D_years)
def condition_3 : Prop := P_years - Q_years = D_years

-- statement to prove
theorem sum_of_ages (h1 : condition_1 Q_years) (h2 : condition_2 P_years Q_years D_years) (h3 : condition_3 P_years Q_years D_years) :
  P_years + Q_years = 93.75 :=
by sorry

end sum_of_ages_l810_81033


namespace garment_industry_initial_men_l810_81022

theorem garment_industry_initial_men (M : ℕ) :
  (M * 8 * 10 = 6 * 20 * 8) → M = 12 :=
by
  sorry

end garment_industry_initial_men_l810_81022


namespace product_of_fractions_l810_81065

theorem product_of_fractions (a b c d e f : ℚ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) 
  (h_d : d = 2) (h_e : e = 3) (h_f : f = 4) :
  (a / b) * (d / e) * (c / f) = 1 / 4 :=
by
  sorry

end product_of_fractions_l810_81065


namespace polar_distance_l810_81058

/-
Problem:
In the polar coordinate system, it is known that A(2, π / 6), B(4, 5π / 6). Then, the distance between points A and B is 2√7.

Conditions:
- Point A in polar coordinates: A(2, π / 6)
- Point B in polar coordinates: B(4, 5π / 6)
-/

/-- The distance between two points in the polar coordinate system A(2, π / 6) and B(4, 5π / 6) is 2√7. -/
theorem polar_distance :
  let A_ρ := 2
  let A_θ := π / 6
  let B_ρ := 4
  let B_θ := 5 * π / 6
  let A_x := A_ρ * Real.cos A_θ
  let A_y := A_ρ * Real.sin A_θ
  let B_x := B_ρ * Real.cos B_θ
  let B_y := B_ρ * Real.sin B_θ
  let distance := Real.sqrt ((B_x - A_x)^2 + (B_y - A_y)^2)
  distance = 2 * Real.sqrt 7 := by
  sorry

end polar_distance_l810_81058


namespace problem_statement_l810_81075

def f : ℝ → ℝ :=
  sorry

lemma even_function (x : ℝ) : f (-x) = f x :=
  sorry

lemma periodicity (x : ℝ) (hx : 0 ≤ x) : f (x + 2) = -f x :=
  sorry

lemma value_in_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 2) : f x = Real.log (x + 1) :=
  sorry

theorem problem_statement : f (-2001) + f 2012 = 1 :=
  sorry

end problem_statement_l810_81075


namespace egg_price_l810_81051

theorem egg_price (num_eggs capital_remaining : ℕ) (total_cost price_per_egg : ℝ)
  (h1 : num_eggs = 30)
  (h2 : capital_remaining = 5)
  (h3 : total_cost = 5)
  (h4 : num_eggs - capital_remaining = 25)
  (h5 : 25 * price_per_egg = total_cost) :
  price_per_egg = 0.20 := sorry

end egg_price_l810_81051


namespace carlos_marbles_l810_81085

theorem carlos_marbles:
  ∃ M, M > 1 ∧ 
       M % 5 = 1 ∧ 
       M % 7 = 1 ∧ 
       M % 11 = 1 ∧ 
       M % 4 = 2 ∧ 
       M = 386 := by
  sorry

end carlos_marbles_l810_81085


namespace chess_tournament_l810_81047

theorem chess_tournament :
  ∀ (n : ℕ), (∃ (players : ℕ) (total_games : ℕ),
  players = 8 ∧ total_games = 56 ∧ total_games = (players * (players - 1) * n) / 2) →
  n = 2 :=
by
  intros n h
  rcases h with ⟨players, total_games, h_players, h_total_games, h_eq⟩
  have := h_eq
  sorry

end chess_tournament_l810_81047


namespace inconsistent_conditions_l810_81030

-- Definitions based on the given conditions
def B : Nat := 59
def C : Nat := 27
def D : Nat := 31
def A := B * C + D

theorem inconsistent_conditions (A_is_factor : ∃ k : Nat, 4701 = k * A) : false := by
  sorry

end inconsistent_conditions_l810_81030


namespace probability_square_not_touching_vertex_l810_81043

theorem probability_square_not_touching_vertex :
  let total_squares := 64
  let squares_touching_vertices := 16
  let squares_not_touching_vertices := total_squares - squares_touching_vertices
  let probability := (squares_not_touching_vertices : ℚ) / total_squares
  probability = 3 / 4 :=
by
  sorry

end probability_square_not_touching_vertex_l810_81043


namespace slipper_cost_l810_81029

def original_price : ℝ := 50.00
def discount_rate : ℝ := 0.10
def embroidery_rate_per_shoe : ℝ := 5.50
def number_of_shoes : ℕ := 2
def shipping_cost : ℝ := 10.00

theorem slipper_cost :
  (original_price - original_price * discount_rate) + 
  (embroidery_rate_per_shoe * number_of_shoes) + 
  shipping_cost = 66.00 :=
by sorry

end slipper_cost_l810_81029


namespace sqrt_49_times_sqrt_25_l810_81087

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l810_81087


namespace log_expression_correct_l810_81056

-- The problem involves logarithms and exponentials
theorem log_expression_correct : 
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 50) + (Real.log 25) + Real.exp (Real.log 3) = 5 := 
  by 
    sorry

end log_expression_correct_l810_81056


namespace infinite_geometric_series_sum_l810_81014

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 2
  let r := (1 : ℝ) / 2
  (a + a * r + a * r^2 + a * r^3 + ∑' n : ℕ, a * r^n) = 1 :=
by
  sorry

end infinite_geometric_series_sum_l810_81014


namespace number_of_cows_l810_81098

theorem number_of_cows (n : ℝ) (h1 : n / 2 + n / 4 + n / 5 + 7 = n) : n = 140 := 
sorry

end number_of_cows_l810_81098


namespace cos_150_eq_neg_sqrt3_over_2_l810_81074

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l810_81074


namespace ValleyFalcons_all_items_l810_81031

noncomputable def num_fans_receiving_all_items (capacity : ℕ) (tshirt_interval : ℕ) 
  (cap_interval : ℕ) (wristband_interval : ℕ) : ℕ :=
  (capacity / Nat.lcm (Nat.lcm tshirt_interval cap_interval) wristband_interval)

theorem ValleyFalcons_all_items:
  num_fans_receiving_all_items 3000 50 25 60 = 10 :=
by
  -- This is where the mathematical proof would go
  sorry

end ValleyFalcons_all_items_l810_81031


namespace proportion_solution_l810_81082

theorem proportion_solution (x : ℝ) (h : x / 6 = 4 / 0.39999999999999997) : x = 60 := sorry

end proportion_solution_l810_81082


namespace ceil_minus_val_eq_one_minus_frac_l810_81099

variable (x : ℝ)

theorem ceil_minus_val_eq_one_minus_frac (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ f : ℝ, 0 ≤ f ∧ f < 1 ∧ ⌈x⌉ - x = 1 - f := 
sorry

end ceil_minus_val_eq_one_minus_frac_l810_81099


namespace proof_problem1_proof_problem2_proof_problem3_l810_81007

-- Definition of the three mathematical problems
def problem1 : Prop := 8 / (-2) - (-4) * (-3) = -16

def problem2 : Prop := -2^3 + (-3) * ((-2)^3 + 5) = 1

def problem3 (x : ℝ) : Prop := (2 * x^2)^3 * x^2 - x^10 / x^2 = 7 * x^8

-- Statements of the proofs required
theorem proof_problem1 : problem1 :=
by sorry

theorem proof_problem2 : problem2 :=
by sorry

theorem proof_problem3 (x : ℝ) : problem3 x :=
by sorry

end proof_problem1_proof_problem2_proof_problem3_l810_81007


namespace arithmetic_sequence_sum_l810_81097

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    (∀ n, a (n + 1) = a n + d) →
    (a 1 + a 4 + a 7 = 45) →
    (a 2 + a_5 + a_8 = 39) →
    (a 3 + a_6 + a_9 = 33) :=
by 
  intros a d h_arith_seq h_cond1 h_cond2
  sorry

end arithmetic_sequence_sum_l810_81097


namespace find_k_l810_81048

theorem find_k (x y k : ℝ) (h1 : x = 1) (h2 : y = 4) (h3 : k * x + y = 3) : k = -1 :=
by
  sorry

end find_k_l810_81048


namespace range_of_x_l810_81084

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a ∧ a ≤ 3) (h : a * x^2 + (a - 2) * x - 2 > 0) :
  x < -1 ∨ x > 2 / 3 :=
sorry

end range_of_x_l810_81084


namespace sum_largest_smallest_5_6_7_l810_81068

/--
Given the digits 5, 6, and 7, if we form all possible three-digit numbers using each digit exactly once, 
then the sum of the largest and smallest of these numbers is 1332.
-/
theorem sum_largest_smallest_5_6_7 : 
  let d1 := 5
  let d2 := 6
  let d3 := 7
  let smallest := 100 * d1 + 10 * d2 + d3
  let largest := 100 * d3 + 10 * d2 + d1
  smallest + largest = 1332 := 
by
  sorry

end sum_largest_smallest_5_6_7_l810_81068


namespace billion_to_scientific_notation_l810_81072

theorem billion_to_scientific_notation : 
  (98.36 * 10^9) = 9.836 * 10^10 := 
by
  sorry

end billion_to_scientific_notation_l810_81072


namespace bricks_in_chimney_l810_81086

-- Define the conditions
def brenda_rate (h : ℕ) : ℚ := h / 8
def brandon_rate (h : ℕ) : ℚ := h / 12
def combined_rate (h : ℕ) : ℚ := (brenda_rate h + brandon_rate h) - 15
def total_bricks_in_6_hours (h : ℕ) : ℚ := 6 * combined_rate h

-- The proof statement
theorem bricks_in_chimney : ∃ h : ℕ, total_bricks_in_6_hours h = h ∧ h = 360 :=
by
  -- Proof goes here
  sorry

end bricks_in_chimney_l810_81086


namespace largest_integral_solution_l810_81088

theorem largest_integral_solution (x : ℤ) : (1 / 4 : ℝ) < (x / 7 : ℝ) ∧ (x / 7 : ℝ) < (3 / 5 : ℝ) → x = 4 :=
by {
  sorry
}

end largest_integral_solution_l810_81088


namespace determinant_of_A_l810_81070

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![ -5, 8],
    ![ 3, -4]]

theorem determinant_of_A : A.det = -4 := by
  sorry

end determinant_of_A_l810_81070


namespace average_weight_of_a_and_b_is_40_l810_81055

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := (A + B + C) / 3 = 42
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 40

-- Theorem statement
theorem average_weight_of_a_and_b_is_40 (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : 
    (A + B) / 2 = 40 := by
  sorry

end average_weight_of_a_and_b_is_40_l810_81055


namespace expression_evaluation_l810_81093

theorem expression_evaluation :
  100 + (120 / 15) + (18 * 20) - 250 - (360 / 12) = 188 := by
  sorry

end expression_evaluation_l810_81093


namespace triangle_value_l810_81018

-- Define the operation \(\triangle\)
def triangle (m n p q : ℕ) : ℕ := (m * m) * p * q / n

-- Define the problem statement
theorem triangle_value : triangle 5 6 9 4 = 150 := by
  sorry

end triangle_value_l810_81018


namespace arithmetic_mean_equality_l810_81000

variable (x y a b : ℝ)

theorem arithmetic_mean_equality (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / 2 * ((x + a) / y + (y - b) / x)) = (x^2 + a * x + y^2 - b * y) / (2 * x * y) :=
  sorry

end arithmetic_mean_equality_l810_81000


namespace intersection_complement_l810_81067

open Set

noncomputable def U := ℝ
noncomputable def A := {x : ℝ | x^2 + 2 * x < 3}
noncomputable def B := {x : ℝ | x - 2 ≤ 0 ∧ x ≠ 0}

theorem intersection_complement :
  A ∩ -B = {x : ℝ | -3 < x ∧ x ≤ 0} :=
sorry

end intersection_complement_l810_81067


namespace intersecting_sets_a_eq_1_l810_81079

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := { x | a * x^2 - 1 = 0 }
def N : Set ℝ := { -1/2, 1/2, 1 }

-- Define the intersection condition
def sets_intersect (M N : Set ℝ) : Prop :=
  ∃ x, x ∈ M ∧ x ∈ N

-- Statement of the problem
theorem intersecting_sets_a_eq_1 (a : ℝ) (h_intersect : sets_intersect (M a) N) : a = 1 :=
  sorry

end intersecting_sets_a_eq_1_l810_81079


namespace margin_in_terms_of_ratio_l810_81077

variable (S m : ℝ)

theorem margin_in_terms_of_ratio (h1 : M = (1/m) * S) (h2 : C = S - M) : M = (1/m) * S :=
sorry

end margin_in_terms_of_ratio_l810_81077


namespace train_length_l810_81071

theorem train_length :
  (∃ (L : ℝ), (L / 30 = (L + 2500) / 120) ∧ L = 75000 / 90) :=
sorry

end train_length_l810_81071


namespace sticker_arrangement_l810_81045

theorem sticker_arrangement : 
  ∀ (n : ℕ), n = 35 → 
  (∀ k : ℕ, k = 8 → 
    ∃ m : ℕ, m = 5 ∧ (n + m) % k = 0) := 
by sorry

end sticker_arrangement_l810_81045


namespace algebra_expression_value_l810_81094

variable (x : ℝ)

theorem algebra_expression_value (h : x^2 - 3 * x - 12 = 0) : 3 * x^2 - 9 * x + 5 = 41 := 
sorry

end algebra_expression_value_l810_81094


namespace range_of_m_for_function_l810_81016

noncomputable def isFunctionDefinedForAllReal (f : ℝ → ℝ) := ∀ x : ℝ, true

theorem range_of_m_for_function :
  (∀ x : ℝ, x^2 - 2 * m * x + m + 2 > 0) ↔ (-1 < m ∧ m < 2) :=
sorry

end range_of_m_for_function_l810_81016


namespace calculate_expression_l810_81059

theorem calculate_expression : 5 * 12 + 6 * 11 - 2 * 15 + 7 * 9 = 159 := by
  sorry

end calculate_expression_l810_81059


namespace class_size_is_10_l810_81080

theorem class_size_is_10 
  (num_92 : ℕ) (num_80 : ℕ) (last_score : ℕ) (target_avg : ℕ) (total_score : ℕ) 
  (h_num_92 : num_92 = 5) (h_num_80 : num_80 = 4) (h_last_score : last_score = 70) 
  (h_target_avg : target_avg = 85) (h_total_score : total_score = 85 * (num_92 + num_80 + 1)) 
  : (num_92 * 92 + num_80 * 80 + last_score = total_score) → 
    (num_92 + num_80 + 1 = 10) :=
by {
  sorry
}

end class_size_is_10_l810_81080


namespace min_distance_convex_lens_l810_81063

theorem min_distance_convex_lens (t k f : ℝ) (hf : f > 0) (ht : t ≥ f)
    (h_lens: 1 / t + 1 / k = 1 / f) :
  t = 2 * f → t + k = 4 * f :=
by
  sorry

end min_distance_convex_lens_l810_81063


namespace find_the_number_l810_81046

theorem find_the_number (x : ℕ) : (220040 = (x + 445) * (2 * (x - 445)) + 40) → x = 555 :=
by
  intro h
  sorry

end find_the_number_l810_81046


namespace total_gold_cost_l810_81092

-- Given conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℕ := 15
def anna_grams : ℕ := 50
def anna_cost_per_gram : ℕ := 20

-- Theorem statement to prove
theorem total_gold_cost :
  (gary_grams * gary_cost_per_gram + anna_grams * anna_cost_per_gram) = 1450 := 
by
  sorry

end total_gold_cost_l810_81092


namespace part1_factorization_part2_factorization_l810_81015

-- Part 1
theorem part1_factorization (x : ℝ) :
  (x - 1) * (6 * x + 5) = 6 * x^2 - x - 5 :=
by {
  sorry
}

-- Part 2
theorem part2_factorization (x : ℝ) :
  (x - 1) * (x + 3) * (x - 2) = x^3 - 7 * x + 6 :=
by {
  sorry
}

end part1_factorization_part2_factorization_l810_81015


namespace time_ratio_xiao_ming_schools_l810_81011

theorem time_ratio_xiao_ming_schools
  (AB BC CD : ℝ) 
  (flat_speed uphill_speed downhill_speed : ℝ)
  (h1 : AB + BC + CD = 1) 
  (h2 : AB / BC = 1 / 2)
  (h3 : BC / CD = 2 / 1)
  (h4 : flat_speed / uphill_speed = 3 / 2)
  (h5 : uphill_speed / downhill_speed = 2 / 4) :
  (AB / flat_speed + BC / uphill_speed + CD / downhill_speed) / 
  (AB / flat_speed + BC / downhill_speed + CD / uphill_speed) = 19 / 16 :=
by
  sorry

end time_ratio_xiao_ming_schools_l810_81011


namespace solve_for_x0_l810_81006

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 2 then x^2 + 2 else 2 * x

theorem solve_for_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 ∨ x0 = - Real.sqrt 6 :=
  by
  sorry

end solve_for_x0_l810_81006


namespace age_ratio_l810_81038

variable (Cindy Jan Marcia Greg: ℕ)

theorem age_ratio 
  (h1 : Cindy = 5)
  (h2 : Jan = Cindy + 2)
  (h3: Greg = 16)
  (h4 : Greg = Marcia + 2)
  (h5 : ∃ k : ℕ, Marcia = k * Jan) 
  : Marcia / Jan = 2 := 
    sorry

end age_ratio_l810_81038


namespace gwen_total_books_l810_81089

def mystery_shelves : Nat := 6
def mystery_books_per_shelf : Nat := 7

def picture_shelves : Nat := 4
def picture_books_per_shelf : Nat := 5

def biography_shelves : Nat := 3
def biography_books_per_shelf : Nat := 3

def scifi_shelves : Nat := 2
def scifi_books_per_shelf : Nat := 9

theorem gwen_total_books :
    (mystery_books_per_shelf * mystery_shelves) +
    (picture_books_per_shelf * picture_shelves) +
    (biography_books_per_shelf * biography_shelves) +
    (scifi_books_per_shelf * scifi_shelves) = 89 := 
by 
    sorry

end gwen_total_books_l810_81089


namespace clock_hands_overlap_l810_81021

theorem clock_hands_overlap (t : ℝ) :
  (∀ (h_angle m_angle : ℝ), h_angle = 30 + 0.5 * t ∧ m_angle = 6 * t ∧ h_angle = m_angle ∧ h_angle = 45) → t = 8 :=
by
  intro h
  sorry

end clock_hands_overlap_l810_81021


namespace quadratic_root_four_times_another_l810_81053

theorem quadratic_root_four_times_another (a : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + a * x + 2 * a = 0 ∧ x2 = 4 * x1) → a = 25 / 2 :=
by
  sorry

end quadratic_root_four_times_another_l810_81053


namespace roots_reciprocal_sum_eq_three_halves_l810_81041

theorem roots_reciprocal_sum_eq_three_halves
  {a b : ℝ}
  (h1 : a^2 - 6 * a + 4 = 0)
  (h2 : b^2 - 6 * b + 4 = 0)
  (h_roots : a ≠ b) :
  1/a + 1/b = 3/2 := by
  sorry

end roots_reciprocal_sum_eq_three_halves_l810_81041


namespace product_equality_l810_81076

theorem product_equality : (2.05 * 4.1 = 20.5 * 0.41) :=
by
  sorry

end product_equality_l810_81076


namespace josh_marbles_l810_81020

theorem josh_marbles (initial_marbles lost_marbles remaining_marbles : ℤ) 
  (h1 : initial_marbles = 19) 
  (h2 : lost_marbles = 11) 
  (h3 : remaining_marbles = initial_marbles - lost_marbles) : 
  remaining_marbles = 8 := 
by
  sorry

end josh_marbles_l810_81020


namespace arithmetic_prog_leq_l810_81017

def t3 (s : List ℤ) : ℕ := 
  sorry -- Placeholder for function calculating number of 3-term arithmetic progressions

theorem arithmetic_prog_leq (a : List ℤ) (k : ℕ) (h_sorted : a = List.range k)
  : t3 a ≤ t3 (List.range k) :=
sorry -- Proof here

end arithmetic_prog_leq_l810_81017


namespace distance_to_fourth_buoy_l810_81027

theorem distance_to_fourth_buoy
  (buoy_interval_distance : ℕ)
  (total_distance_to_third_buoy : ℕ)
  (h : total_distance_to_third_buoy = buoy_interval_distance * 3) :
  (buoy_interval_distance * 4 = 96) :=
by
  sorry

end distance_to_fourth_buoy_l810_81027


namespace polygon_sides_l810_81013

theorem polygon_sides (a : ℝ) (n : ℕ) (h1 : a = 140) (h2 : 180 * (n-2) = n * a) : n = 9 := 
by sorry

end polygon_sides_l810_81013


namespace tom_reads_pages_l810_81091

-- Definition of conditions
def initial_speed : ℕ := 12   -- pages per hour
def speed_factor : ℕ := 3
def time_period : ℕ := 2     -- hours

-- Calculated speeds
def increased_speed (initial_speed speed_factor : ℕ) : ℕ := initial_speed * speed_factor
def total_pages (increased_speed time_period : ℕ) : ℕ := increased_speed * time_period

-- Theorem statement
theorem tom_reads_pages :
  total_pages (increased_speed initial_speed speed_factor) time_period = 72 :=
by
  -- Omitting proof as only theorem statement is required
  sorry

end tom_reads_pages_l810_81091


namespace water_needed_in_pints_l810_81040

-- Define the input data
def parts_water : ℕ := 5
def parts_lemon : ℕ := 2
def pints_per_gallon : ℕ := 8
def total_gallons : ℕ := 3

-- Define the total parts of the mixture
def total_parts : ℕ := parts_water + parts_lemon

-- Define the total pints of lemonade
def total_pints : ℕ := total_gallons * pints_per_gallon

-- Define the pints per part of the mixture
def pints_per_part : ℚ := total_pints / total_parts

-- Define the total pints of water needed
def pints_water : ℚ := parts_water * pints_per_part

-- The theorem stating what we need to prove
theorem water_needed_in_pints : pints_water = 17 + 1 / 7 := by
  sorry

end water_needed_in_pints_l810_81040


namespace jerry_won_47_tickets_l810_81001

open Nat

-- Define the initial number of tickets
def initial_tickets : Nat := 4

-- Define the number of tickets spent on the beanie
def tickets_spent_on_beanie : Nat := 2

-- Define the current total number of tickets Jerry has
def current_tickets : Nat := 49

-- Define the number of tickets Jerry won later
def tickets_won_later : Nat := current_tickets - (initial_tickets - tickets_spent_on_beanie)

-- The theorem to prove
theorem jerry_won_47_tickets :
  tickets_won_later = 47 :=
by sorry

end jerry_won_47_tickets_l810_81001


namespace triangle_angles_l810_81042

-- Defining a structure for a triangle with angles
structure Triangle :=
(angleA angleB angleC : ℝ)

-- Define the condition for the triangle mentioned in the problem
def triangle_condition (t : Triangle) : Prop :=
  ∃ (α : ℝ), α = 22.5 ∧ t.angleA = 90 ∧ t.angleB = α ∧ t.angleC = 67.5

theorem triangle_angles :
  ∃ (t : Triangle), triangle_condition t :=
by
  -- The proof outline
  -- We need to construct a triangle with the given angle conditions
  -- angleA = 90°, angleB = 22.5°, angleC = 67.5°
  sorry

end triangle_angles_l810_81042


namespace inverse_h_l810_81050

-- Define the functions f, g, and h as given in the conditions
def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := 3 * x + 2
def h (x : ℝ) := f (g x)

-- State the problem of proving the inverse of h
theorem inverse_h : ∀ x, h⁻¹ (x : ℝ) = (x - 5) / 12 :=
sorry

end inverse_h_l810_81050


namespace statement_B_statement_D_l810_81005

variable {a b c d : ℝ}

theorem statement_B (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : (c / a) > (c / b) := 
by sorry

theorem statement_D (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : (a * c) < (b * d) := 
by sorry

end statement_B_statement_D_l810_81005
