import Mathlib

namespace painting_problem_equation_l671_67154

def dougPaintingRate := 1 / 3
def davePaintingRate := 1 / 4
def combinedPaintingRate := dougPaintingRate + davePaintingRate
def timeRequiredToComplete (t : ℝ) : Prop := 
  (t - 1) * combinedPaintingRate = 2 / 3

theorem painting_problem_equation : ∃ t : ℝ, timeRequiredToComplete t :=
sorry

end painting_problem_equation_l671_67154


namespace barbara_typing_time_l671_67152

theorem barbara_typing_time:
  let original_speed := 212
  let speed_decrease := 40
  let document_length := 3440
  let new_speed := original_speed - speed_decrease
  (new_speed > 0) → 
  (document_length / new_speed = 20) :=
by
  intros
  sorry

end barbara_typing_time_l671_67152


namespace smallest_sum_of_cubes_two_ways_l671_67178

theorem smallest_sum_of_cubes_two_ways :
  ∃ (n : ℕ) (a b c d e f : ℕ),
  n = a^3 + b^3 + c^3 ∧ n = d^3 + e^3 + f^3 ∧
  (a, b, c) ≠ (d, e, f) ∧
  (d, e, f) ≠ (a, b, c) ∧ n = 251 :=
by
  sorry

end smallest_sum_of_cubes_two_ways_l671_67178


namespace single_elimination_games_l671_67118

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = n - 1 :=
by
  have h1 : n = 512 := h
  use 511
  sorry

end single_elimination_games_l671_67118


namespace units_digit_base_9_l671_67160

theorem units_digit_base_9 (a b : ℕ) (h1 : a = 3 * 9 + 5) (h2 : b = 4 * 9 + 7) : 
  ((a + b) % 9) = 3 := by
  sorry

end units_digit_base_9_l671_67160


namespace average_earnings_per_minute_l671_67181

theorem average_earnings_per_minute (race_duration : ℕ) (lap_distance : ℕ) (certificate_rate : ℝ) (laps_run : ℕ) :
  race_duration = 12 → 
  lap_distance = 100 → 
  certificate_rate = 3.5 → 
  laps_run = 24 → 
  ((laps_run * lap_distance / 100) * certificate_rate) / race_duration = 7 :=
by
  intros hrace_duration hlap_distance hcertificate_rate hlaps_run
  rw [hrace_duration, hlap_distance, hcertificate_rate, hlaps_run]
  sorry

end average_earnings_per_minute_l671_67181


namespace mean_of_sets_l671_67193

theorem mean_of_sets (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) : 
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
by
  sorry

end mean_of_sets_l671_67193


namespace necessary_but_not_sufficient_condition_l671_67189

-- Define the set A
def A := {x : ℝ | -1 < x ∧ x < 2}

-- Define the necessary but not sufficient condition
def necessary_condition (a : ℝ) : Prop := a ≥ 1

-- Define the proposition that needs to be proved
def proposition (a : ℝ) : Prop := ∀ x ∈ A, x^2 - a < 0

-- The proof statement
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  necessary_condition a → ∃ x ∈ A, proposition a :=
sorry

end necessary_but_not_sufficient_condition_l671_67189


namespace arithmetic_sequence_third_term_l671_67128

theorem arithmetic_sequence_third_term (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) :
  (S 5 = 10) ∧ (S n = n * (a 1 + a n) / 2) ∧ (a 5 = a 1 + 4 * d) ∧ 
  (∀ n, a n = a 1 + (n-1) * d) → (a 3 = 2) :=
by
  intro h
  sorry

end arithmetic_sequence_third_term_l671_67128


namespace positive_integers_sum_reciprocal_l671_67120

theorem positive_integers_sum_reciprocal (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_sum : a + b + c = 2010) (h_recip : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c = 1/58) :
  (a = 1740 ∧ b = 180 ∧ c = 90) ∨ 
  (a = 1740 ∧ b = 90 ∧ c = 180) ∨ 
  (a = 180 ∧ b = 90 ∧ c = 1740) ∨ 
  (a = 180 ∧ b = 1740 ∧ c = 90) ∨ 
  (a = 90 ∧ b = 1740 ∧ c = 180) ∨ 
  (a = 90 ∧ b = 180 ∧ c = 1740) := 
sorry

end positive_integers_sum_reciprocal_l671_67120


namespace champagne_bottles_needed_l671_67159

-- Define the initial conditions of the problem
def num_guests : ℕ := 120
def glasses_per_guest : ℕ := 2
def servings_per_bottle : ℕ := 6

-- The statement we need to prove
theorem champagne_bottles_needed : 
  (num_guests * glasses_per_guest) / servings_per_bottle = 40 := 
by
  sorry

end champagne_bottles_needed_l671_67159


namespace find_function_l671_67191

theorem find_function (α : ℝ) (hα : 0 < α) (f : ℕ+ → ℝ) 
  (h : ∀ k m : ℕ+, α * m ≤ k → k ≤ (α + 1) * m → f (k + m) = f k + f m) :
  ∃ D : ℝ, ∀ n : ℕ+, f n = n * D :=
sorry

end find_function_l671_67191


namespace math_test_score_l671_67130

theorem math_test_score (K E M : ℕ) 
  (h₁ : (K + E) / 2 = 92) 
  (h₂ : (K + E + M) / 3 = 94) : 
  M = 98 := 
by 
  sorry

end math_test_score_l671_67130


namespace ratio_of_wealth_l671_67190

theorem ratio_of_wealth (P W : ℝ) (hP : P > 0) (hW : W > 0) : 
  let wX := (0.40 * W) / (0.20 * P)
  let wY := (0.30 * W) / (0.10 * P)
  (wX / wY) = 2 / 3 := 
by
  sorry

end ratio_of_wealth_l671_67190


namespace greatest_prime_factor_f24_is_11_value_of_f12_l671_67104

def is_even (n : ℕ) : Prop := n % 2 = 0

def f (n : ℕ) : ℕ := (List.range' 2 ((n + 1) / 2)).map (λ x => 2 * x) |> List.prod

theorem greatest_prime_factor_f24_is_11 : 
  ¬ ∃ p, Prime p ∧ p ∣ f 24 ∧ p > 11 := 
  sorry

theorem value_of_f12 : f 12 = 46080 := 
  sorry

end greatest_prime_factor_f24_is_11_value_of_f12_l671_67104


namespace transformation_composition_l671_67185

-- Define the transformations f and g
def f (m n : ℝ) : ℝ × ℝ := (m, -n)
def g (m n : ℝ) : ℝ × ℝ := (-m, -n)

-- The proof statement that we need to prove
theorem transformation_composition : g (f (-3) 2).1 (f (-3) 2).2 = (3, 2) :=
by sorry

end transformation_composition_l671_67185


namespace find_a6_l671_67123

-- Define the arithmetic sequence properties
variables (a : ℕ → ℤ) (d : ℤ)

-- Define the initial conditions
axiom h1 : a 4 = 1
axiom h2 : a 7 = 16
axiom h_arith_seq : ∀ n, a (n + 1) - a n = d

-- Statement to prove
theorem find_a6 : a 6 = 11 :=
by
  sorry

end find_a6_l671_67123


namespace radius_of_circle_l671_67172

def circle_eq_def (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

theorem radius_of_circle {x y r : ℝ} (h : circle_eq_def x y) : r = 3 := 
by
  -- Proof skipped
  sorry

end radius_of_circle_l671_67172


namespace find_expression_for_x_l671_67197

variable (x : ℝ) (hx : x^3 + (1 / x^3) = -52)

theorem find_expression_for_x : x + (1 / x) = -4 :=
by sorry

end find_expression_for_x_l671_67197


namespace left_handed_rock_music_lovers_l671_67121

theorem left_handed_rock_music_lovers (total_club_members left_handed_members rock_music_lovers right_handed_dislike_rock: ℕ)
  (h1 : total_club_members = 25)
  (h2 : left_handed_members = 10)
  (h3 : rock_music_lovers = 18)
  (h4 : right_handed_dislike_rock = 3)
  (h5 : total_club_members = left_handed_members + (total_club_members - left_handed_members))
  : (∃ x : ℕ, x = 6 ∧ x + (left_handed_members - x) + (rock_music_lovers - x) + right_handed_dislike_rock = total_club_members) :=
sorry

end left_handed_rock_music_lovers_l671_67121


namespace sum_arithmetic_sequence_satisfies_conditions_l671_67192

theorem sum_arithmetic_sequence_satisfies_conditions :
  ∀ (a : ℕ → ℤ) (d : ℤ),
  (a 1 = 1) ∧ (d ≠ 0) ∧ ((a 3)^2 = (a 2) * (a 6)) →
  (6 * a 1 + (6 * 5 / 2) * d = -24) :=
by
  sorry

end sum_arithmetic_sequence_satisfies_conditions_l671_67192


namespace min_value_expr_l671_67119

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 24 :=
sorry

end min_value_expr_l671_67119


namespace cos_sum_of_angles_l671_67126

theorem cos_sum_of_angles (α β : Real) (h1 : Real.sin α = 4/5) (h2 : (π/2) < α ∧ α < π) 
(h3 : Real.cos β = -5/13) (h4 : 0 < β ∧ β < π/2) : 
  Real.cos (α + β) = -33/65 := 
by
  sorry

end cos_sum_of_angles_l671_67126


namespace smallest_cubes_to_fill_box_l671_67148

theorem smallest_cubes_to_fill_box
  (L W D : ℕ)
  (hL : L = 30)
  (hW : W = 48)
  (hD : D = 12) :
  ∃ (n : ℕ), n = (L * W * D) / ((Nat.gcd (Nat.gcd L W) D) ^ 3) ∧ n = 80 := 
by
  sorry

end smallest_cubes_to_fill_box_l671_67148


namespace sum_of_numbers_l671_67106

def a : ℝ := 217
def b : ℝ := 2.017
def c : ℝ := 0.217
def d : ℝ := 2.0017

theorem sum_of_numbers :
  a + b + c + d = 221.2357 :=
by
  sorry

end sum_of_numbers_l671_67106


namespace sum_squares_seven_consecutive_not_perfect_square_l671_67184

theorem sum_squares_seven_consecutive_not_perfect_square : 
  ∀ (n : ℤ), ¬ ∃ k : ℤ, k * k = (n-3)^2 + (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2 :=
by
  sorry

end sum_squares_seven_consecutive_not_perfect_square_l671_67184


namespace solve_inequality_l671_67163

theorem solve_inequality (x : Real) : 
  (abs ((3 * x + 2) / (x - 2)) > 3) ↔ (x ∈ Set.Ioo (2 / 3) 2) := by
  sorry

end solve_inequality_l671_67163


namespace neg_sqrt_comparison_l671_67182

theorem neg_sqrt_comparison : -Real.sqrt 7 > -Real.sqrt 11 := by
  sorry

end neg_sqrt_comparison_l671_67182


namespace surface_area_circumscribed_sphere_l671_67140

theorem surface_area_circumscribed_sphere (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
    4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2) / 2)^2) = 50 * Real.pi :=
by
  rw [ha, hb, hc]
  -- prove the equality step-by-step
  sorry

end surface_area_circumscribed_sphere_l671_67140


namespace expected_number_of_digits_l671_67195

-- Define a noncomputable expected_digits function for an icosahedral die
noncomputable def expected_digits : ℝ :=
  let p1 := 9 / 20
  let p2 := 11 / 20
  (p1 * 1) + (p2 * 2)

theorem expected_number_of_digits :
  expected_digits = 1.55 :=
by
  -- The proof will be filled in here
  sorry

end expected_number_of_digits_l671_67195


namespace ninth_term_geometric_sequence_l671_67114

noncomputable def geometric_seq (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem ninth_term_geometric_sequence (a r : ℝ) (h_positive : ∀ n, 0 < geometric_seq a r n)
  (h_fifth_term : geometric_seq a r 5 = 32)
  (h_eleventh_term : geometric_seq a r 11 = 2) :
  geometric_seq a r 9 = 2 :=
by
{
  sorry
}

end ninth_term_geometric_sequence_l671_67114


namespace calculate_square_difference_l671_67161

theorem calculate_square_difference : 2023^2 - 2022^2 = 4045 := by
  sorry

end calculate_square_difference_l671_67161


namespace interest_rate_per_annum_l671_67175

theorem interest_rate_per_annum
  (P : ℕ := 450) 
  (t : ℕ := 8) 
  (I : ℕ := P - 306) 
  (simple_interest : ℕ := P * r * t / 100) :
  r = 4 :=
by
  sorry

end interest_rate_per_annum_l671_67175


namespace hexagonal_tiles_in_box_l671_67164

theorem hexagonal_tiles_in_box :
  ∃ a b c : ℕ, a + b + c = 35 ∧ 3 * a + 4 * b + 6 * c = 128 ∧ c = 6 :=
by
  sorry

end hexagonal_tiles_in_box_l671_67164


namespace find_circle_diameter_l671_67129

noncomputable def circle_diameter (AB CD : ℝ) (h_AB : AB = 16) (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) : ℝ :=
  2 * 10

theorem find_circle_diameter (AB CD : ℝ)
  (h_AB : AB = 16)
  (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) :
  circle_diameter AB CD h_AB h_CD h_perp = 20 := 
  by sorry

end find_circle_diameter_l671_67129


namespace triangle_height_l671_67198

theorem triangle_height (A : ℝ) (b : ℝ) (h : ℝ) 
  (hA : A = 615) 
  (hb : b = 123)
  (h_area : A = 0.5 * b * h) : 
  h = 10 :=
by 
  -- Placeholder for the proof
  sorry

end triangle_height_l671_67198


namespace alma_carrots_leftover_l671_67116

/-- Alma has 47 baby carrots and wishes to distribute them equally among 4 goats.
    We need to prove that the number of leftover carrots after such distribution is 3. -/
theorem alma_carrots_leftover (total_carrots : ℕ) (goats : ℕ) (leftover : ℕ) 
  (h1 : total_carrots = 47) (h2 : goats = 4) (h3 : leftover = total_carrots % goats) : 
  leftover = 3 :=
by
  sorry

end alma_carrots_leftover_l671_67116


namespace words_per_minute_after_break_l671_67131

variable (w : ℕ)

theorem words_per_minute_after_break (h : 10 * 5 - (w * 5) = 10) : w = 8 := by
  sorry

end words_per_minute_after_break_l671_67131


namespace find_a10_l671_67149

variable {q : ℝ}
variable {a : ℕ → ℝ}

-- Sequence conditions
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom positive_ratio : 0 < q
axiom condition_1 : a 2 = 1
axiom condition_2 : a 4 * a 8 = 2 * (a 5) ^ 2

theorem find_a10 : a 10 = 16 := by
  sorry

end find_a10_l671_67149


namespace molecular_weight_BaSO4_l671_67124

-- Definitions for atomic weights of elements.
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_S : ℝ := 32.07
def atomic_weight_O : ℝ := 16.00

-- Defining the number of atoms in BaSO4
def num_Ba : ℕ := 1
def num_S : ℕ := 1
def num_O : ℕ := 4

-- Statement to be proved
theorem molecular_weight_BaSO4 :
  (num_Ba * atomic_weight_Ba + num_S * atomic_weight_S + num_O * atomic_weight_O) = 233.40 := 
by
  sorry

end molecular_weight_BaSO4_l671_67124


namespace boris_stopped_saving_in_may_2020_l671_67157

theorem boris_stopped_saving_in_may_2020 :
  ∀ (B V : ℕ) (start_date_B start_date_V stop_date : ℕ), 
    (∀ t, start_date_B + t ≤ stop_date → B = 200 * t) →
    (∀ t, start_date_V + t ≤ stop_date → V = 300 * t) → 
    V = 6 * B →
    stop_date = 17 → 
    B / 200 = 4 → 
    stop_date - B/200 = 2020 * 12 + 5 :=
by
  sorry

end boris_stopped_saving_in_may_2020_l671_67157


namespace inequality_abc_l671_67115

theorem inequality_abc 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a * b * c ≥ (b + c - a) * (a + c - b) * (a + b - c) := 
by 
  sorry

end inequality_abc_l671_67115


namespace cost_of_1000_gums_in_dollars_l671_67107

theorem cost_of_1000_gums_in_dollars :
  let cost_per_piece_in_cents := 1
  let pieces := 1000
  let cents_per_dollar := 100
  ∃ cost_in_dollars : ℝ, cost_in_dollars = (cost_per_piece_in_cents * pieces) / cents_per_dollar :=
sorry

end cost_of_1000_gums_in_dollars_l671_67107


namespace solve_quadratic1_solve_quadratic2_l671_67102

open Real

-- Equation 1
theorem solve_quadratic1 (x : ℝ) : x^2 - 6 * x + 8 = 0 → x = 2 ∨ x = 4 := 
by sorry

-- Equation 2
theorem solve_quadratic2 (x : ℝ) : x^2 - 8 * x + 1 = 0 → x = 4 + sqrt 15 ∨ x = 4 - sqrt 15 := 
by sorry

end solve_quadratic1_solve_quadratic2_l671_67102


namespace value_of_y_when_x_is_zero_l671_67150

noncomputable def quadratic_y (h x : ℝ) : ℝ := -(x + h)^2

theorem value_of_y_when_x_is_zero :
  ∀ (h : ℝ), (∀ x, x < -3 → quadratic_y h x < quadratic_y h (-3)) →
            (∀ x, x > -3 → quadratic_y h x < quadratic_y h (-3)) →
            quadratic_y h 0 = -9 :=
by
  sorry

end value_of_y_when_x_is_zero_l671_67150


namespace Q_lies_in_third_quadrant_l671_67111

theorem Q_lies_in_third_quadrant (b : ℝ) (P_in_fourth_quadrant : 2 > 0 ∧ b < 0) :
    b < 0 ∧ -2 < 0 ↔
    (b < 0 ∧ -2 < 0) :=
by
  sorry

end Q_lies_in_third_quadrant_l671_67111


namespace neg_number_among_set_l671_67133

theorem neg_number_among_set :
  ∃ n ∈ ({5, 1, -2, 0} : Set ℤ), n < 0 ∧ n = -2 :=
by
  sorry

end neg_number_among_set_l671_67133


namespace find_first_term_of_arithmetic_sequence_l671_67151

theorem find_first_term_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a3 : a 3 = 2)
  (h_d : d = -1/2) : a 1 = 3 :=
sorry

end find_first_term_of_arithmetic_sequence_l671_67151


namespace birds_not_herons_are_geese_l671_67142

-- Define the given conditions
def percentage_geese : ℝ := 0.35
def percentage_swans : ℝ := 0.20
def percentage_herons : ℝ := 0.15
def percentage_ducks : ℝ := 0.30

-- Definition without herons
def percentage_non_herons : ℝ := 1 - percentage_herons

-- Theorem to prove
theorem birds_not_herons_are_geese :
  (percentage_geese / percentage_non_herons) * 100 = 41 :=
by
  sorry

end birds_not_herons_are_geese_l671_67142


namespace initial_donuts_30_l671_67187

variable (x y : ℝ)
variable (p : ℝ := 0.30)

theorem initial_donuts_30 (h1 : y = 9) (h2 : y = p * x) : x = 30 := by
  sorry

end initial_donuts_30_l671_67187


namespace part1_part1_eq_part2_tangent_part3_center_range_l671_67183

-- Define the conditions
def A : ℝ × ℝ := (0, 3)
def line_l (x : ℝ) : ℝ := 2 * x - 4
def circle_center_condition (x : ℝ) : ℝ := -x + 5
def radius : ℝ := 1

-- Part (1)
theorem part1 (x y : ℝ) (hx : y = line_l x) (hy : y = circle_center_condition x) :
  (x = 3 ∧ y = 2) :=
sorry

theorem part1_eq :
  ∃ C : ℝ × ℝ, C = (3, 2) ∧ ∀ (x y : ℝ), (x - 3) ^ 2 + (y - 2) ^ 2 = 1 :=
sorry

-- Part (2)
theorem part2_tangent (x y : ℝ) (hx : y = 3) (hy : 3 * x + 4 * y - 12 = 0) :
  ∀ (a b : ℝ), a = 0 ∧ b = -3 / 4 :=
sorry

-- Part (3)
theorem part3_center_range (a : ℝ) (M : ℝ × ℝ) :
  (|2 * a - 4 - 3 / 2| ≤ 1) ->
  (9 / 4 ≤ a ∧ a ≤ 13 / 4) :=
sorry

end part1_part1_eq_part2_tangent_part3_center_range_l671_67183


namespace wood_burned_in_afternoon_l671_67146

theorem wood_burned_in_afternoon 
  (burned_morning : ℕ) 
  (start_bundles : ℕ) 
  (end_bundles : ℕ) 
  (burned_afternoon : ℕ) 
  (h1 : burned_morning = 4) 
  (h2 : start_bundles = 10) 
  (h3 : end_bundles = 3) 
  (h4 : burned_morning + burned_afternoon = start_bundles - end_bundles) :
  burned_afternoon = 3 := 
sorry

end wood_burned_in_afternoon_l671_67146


namespace nancy_initial_files_correct_l671_67138

-- Definitions based on the problem conditions
def initial_files (deleted_files : ℕ) (folder_count : ℕ) (files_per_folder : ℕ) : ℕ :=
  (folder_count * files_per_folder) + deleted_files

-- The proof statement
theorem nancy_initial_files_correct :
  initial_files 31 7 7 = 80 :=
by
  sorry

end nancy_initial_files_correct_l671_67138


namespace sector_central_angle_l671_67125

theorem sector_central_angle (r l θ : ℝ) (h_perimeter : 2 * r + l = 8) (h_area : (1 / 2) * l * r = 4) : θ = 2 :=
by
  sorry

end sector_central_angle_l671_67125


namespace num_employees_excluding_manager_l671_67113

/-- 
If the average monthly salary of employees is Rs. 1500, 
and adding a manager with salary Rs. 14100 increases 
the average salary by Rs. 600, prove that the number 
of employees (excluding the manager) is 20.
-/
theorem num_employees_excluding_manager 
  (avg_salary : ℕ) 
  (manager_salary : ℕ) 
  (new_avg_increase : ℕ) : 
  (∃ n : ℕ, 
    avg_salary = 1500 ∧ 
    manager_salary = 14100 ∧ 
    new_avg_increase = 600 ∧ 
    n = 20) := 
sorry

end num_employees_excluding_manager_l671_67113


namespace david_average_speed_l671_67179

theorem david_average_speed (d t : ℚ) (h1 : d = 49 / 3) (h2 : t = 7 / 3) :
  (d / t) = 7 :=
by
  rw [h1, h2]
  norm_num

end david_average_speed_l671_67179


namespace minimum_volume_sum_l671_67177

section pyramid_volume

variables {R : Type*} [OrderedRing R]
variables {V : Type*} [AddCommGroup V] [Module R V]

-- Define the volumes of the pyramids
variables (V_SABR1 V_SR2P2R3Q2 V_SCDR4 : R)
variables (V_SR1P1R2Q1 V_SR3P3R4Q3 : R)

-- Given condition
axiom volume_condition : V_SR1P1R2Q1 + V_SR3P3R4Q3 = 78

-- The theorem to be proved
theorem minimum_volume_sum : 
  V_SABR1^2 + V_SR2P2R3Q2^2 + V_SCDR4^2 ≥ 2028 :=
sorry

end pyramid_volume

end minimum_volume_sum_l671_67177


namespace roots_of_abs_exp_eq_b_l671_67170

theorem roots_of_abs_exp_eq_b (b : ℝ) (h : 0 < b ∧ b < 1) : 
  ∃! (x1 x2 : ℝ), x1 ≠ x2 ∧ abs (2^x1 - 1) = b ∧ abs (2^x2 - 1) = b :=
sorry

end roots_of_abs_exp_eq_b_l671_67170


namespace Doug_lost_marbles_l671_67153

theorem Doug_lost_marbles (D E L : ℕ) 
    (h1 : E = D + 22) 
    (h2 : E = D - L + 30) 
    : L = 8 := by
  sorry

end Doug_lost_marbles_l671_67153


namespace triangle_area_squared_l671_67166

theorem triangle_area_squared
  (R : ℝ)
  (A : ℝ)
  (AC_minus_AB : ℝ)
  (area : ℝ)
  (hx : R = 4)
  (hy : A = 60)
  (hz : AC_minus_AB = 4)
  (area_eq : area = 8 * Real.sqrt 3) :
  area^2 = 192 :=
by
  -- We include the conditions 
  have hR := hx
  have hA := hy
  have hAC_AB := hz
  have harea := area_eq
  -- We will use these to construct the required proof 
  sorry

end triangle_area_squared_l671_67166


namespace range_of_k_l671_67134

theorem range_of_k
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : a^2 + c^2 = 16)
  (h2 : b^2 + c^2 = 25) : 
  9 < a^2 + b^2 ∧ a^2 + b^2 < 41 :=
by
  sorry

end range_of_k_l671_67134


namespace sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l671_67169

theorem sum_of_last_three_digits_9_pow_15_plus_15_pow_15 :
  (9 ^ 15 + 15 ^ 15) % 1000 = 24 :=
by
  sorry

end sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l671_67169


namespace number_of_freshmen_to_sample_l671_67136

-- Define parameters
def total_students : ℕ := 900
def sample_size : ℕ := 45
def freshmen_count : ℕ := 400
def sophomores_count : ℕ := 300
def juniors_count : ℕ := 200

-- Define the stratified sampling calculation
def stratified_sampling_calculation (group_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_size

-- Theorem stating that the number of freshmen to be sampled is 20
theorem number_of_freshmen_to_sample : stratified_sampling_calculation freshmen_count total_students sample_size = 20 := by
  sorry

end number_of_freshmen_to_sample_l671_67136


namespace problem_statement_l671_67110

theorem problem_statement :
  ∃ (a b c : ℕ), gcd a (gcd b c) = 1 ∧
  (∃ x y : ℝ, 2 * y = 8 * x - 7) ∧
  a ^ 2 + b ^ 2 + (c:ℤ) ^ 2 = 117 :=
sorry

end problem_statement_l671_67110


namespace inequality_proof_l671_67144

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) +
    (b / Real.sqrt (b^2 + 8 * a * c)) +
    (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_proof_l671_67144


namespace fundraising_exceeded_goal_l671_67122

theorem fundraising_exceeded_goal (ken mary scott : ℕ) (goal: ℕ) 
  (h_ken : ken = 600)
  (h_mary_ken : mary = 5 * ken)
  (h_mary_scott : mary = 3 * scott)
  (h_goal : goal = 4000) :
  (ken + mary + scott) - goal = 600 := 
  sorry

end fundraising_exceeded_goal_l671_67122


namespace no_solution_fermat_like_l671_67180

theorem no_solution_fermat_like (x y z k : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) 
  (hxk : x < k) (hyk : y < k) (hxk_eq : x ^ k + y ^ k = z ^ k) : false :=
sorry

end no_solution_fermat_like_l671_67180


namespace min_value_expression_l671_67101

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 = 4 :=
sorry

end min_value_expression_l671_67101


namespace geometric_sequence_common_ratio_l671_67194

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 3 = 1/2)
  (h3 : a 1 * (1 + q) = 3) :
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l671_67194


namespace carly_practice_time_l671_67109

-- conditions
def practice_time_butterfly_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def practice_time_backstroke_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def total_weekly_practice (butterfly_hours : ℕ) (backstroke_hours : ℕ) : ℕ :=
  butterfly_hours + backstroke_hours

def monthly_practice (weekly_hours : ℕ) (weeks_per_month : ℕ) : ℕ :=
  weekly_hours * weeks_per_month

-- Proof Problem Statement
theorem carly_practice_time :
  practice_time_butterfly_weekly 3 4 + practice_time_backstroke_weekly 2 6 * 4 = 96 :=
by
  sorry

end carly_practice_time_l671_67109


namespace print_time_nearest_whole_l671_67199

theorem print_time_nearest_whole 
  (pages_per_minute : ℕ) (total_pages : ℕ) (expected_time : ℕ)
  (h1 : pages_per_minute = 25) (h2 : total_pages = 575) : 
  expected_time = 23 :=
by
  sorry

end print_time_nearest_whole_l671_67199


namespace find_x_coordinate_l671_67147

-- Define the center and radius of the circle
structure Circle where
  center : (ℝ × ℝ)
  radius : ℝ

-- Define the points on the circle
def lies_on_circle (C : Circle) (P : ℝ × ℝ) : Prop :=
  let (x_c, y_c) := C.center
  let (x_p, y_p) := P
  (x_p - x_c)^2 + (y_p - y_c)^2 = C.radius^2

-- Lean 4 statement
theorem find_x_coordinate :
  ∀ (C : Circle), C.radius = 2 → lies_on_circle C (2, 0) ∧ lies_on_circle C (-2, 0) → 2 = 2 := by
  intro C h_radius ⟨h_lies_on_2_0, h_lies_on__2_0⟩
  sorry

end find_x_coordinate_l671_67147


namespace contractor_pays_male_worker_rs_35_l671_67139

theorem contractor_pays_male_worker_rs_35
  (num_male_workers : ℕ)
  (num_female_workers : ℕ)
  (num_child_workers : ℕ)
  (female_worker_wage : ℕ)
  (child_worker_wage : ℕ)
  (average_wage_per_day : ℕ)
  (total_workers : ℕ := num_male_workers + num_female_workers + num_child_workers)
  (total_wage : ℕ := average_wage_per_day * total_workers)
  (total_female_wage : ℕ := num_female_workers * female_worker_wage)
  (total_child_wage : ℕ := num_child_workers * child_worker_wage)
  (total_male_wage : ℕ := total_wage - total_female_wage - total_child_wage) :
  num_male_workers = 20 →
  num_female_workers = 15 →
  num_child_workers = 5 →
  female_worker_wage = 20 →
  child_worker_wage = 8 →
  average_wage_per_day = 26 →
  total_male_wage / num_male_workers = 35 :=
by
  intros h20 h15 h5 h20w h8w h26
  sorry

end contractor_pays_male_worker_rs_35_l671_67139


namespace lcm_23_46_827_l671_67173

theorem lcm_23_46_827 :
  (23 * 46 * 827) / gcd (23 * 2) 827 = 38042 := by
  sorry

end lcm_23_46_827_l671_67173


namespace factory_production_eq_l671_67168

theorem factory_production_eq (x : ℝ) (h1 : x > 50) : 450 / (x - 50) - 400 / x = 1 := 
by 
  sorry

end factory_production_eq_l671_67168


namespace range_of_a_l671_67105

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
by
  sorry

end range_of_a_l671_67105


namespace value_of_y_l671_67117

theorem value_of_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 :=
by
  sorry

end value_of_y_l671_67117


namespace find_second_expression_l671_67186

theorem find_second_expression (a : ℕ) (x : ℕ) (h1 : (2 * a + 16 + x) / 2 = 69) (h2 : a = 26) : x = 70 := 
by
  sorry

end find_second_expression_l671_67186


namespace train_length_l671_67188

/-- 
  Given:
  - jogger_speed is the jogger's speed in km/hr (9 km/hr)
  - train_speed is the train's speed in km/hr (45 km/hr)
  - jogger_ahead is the jogger's initial lead in meters (240 m)
  - passing_time is the time in seconds for the train to pass the jogger (36 s)
  
  Prove that the length of the train is 120 meters.
-/
theorem train_length
  (jogger_speed : ℕ) -- in km/hr
  (train_speed : ℕ) -- in km/hr
  (jogger_ahead : ℕ) -- in meters
  (passing_time : ℕ) -- in seconds
  (h_jogger_speed : jogger_speed = 9)
  (h_train_speed : train_speed = 45)
  (h_jogger_ahead : jogger_ahead = 240)
  (h_passing_time : passing_time = 36)
  : ∃ length_of_train : ℕ, length_of_train = 120 :=
by
  sorry

end train_length_l671_67188


namespace largest_integer_among_four_l671_67174

theorem largest_integer_among_four 
  (p q r s : ℤ)
  (h1 : p + q + r = 210)
  (h2 : p + q + s = 230)
  (h3 : p + r + s = 250)
  (h4 : q + r + s = 270) :
  max (max p q) (max r s) = 110 :=
by
  sorry

end largest_integer_among_four_l671_67174


namespace calculate_expression_l671_67176

variable (x y : ℚ)

theorem calculate_expression (h₁ : x = 4 / 6) (h₂ : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  -- proof steps here
  sorry

end calculate_expression_l671_67176


namespace range_f_x_le_neg_five_l671_67145

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x then 2^x - 3 else
if h : x < 0 then 3 - 2^(-x) else 0

theorem range_f_x_le_neg_five :
  ∀ x : ℝ, f x ≤ -5 ↔ x ≤ -3 :=
by sorry

end range_f_x_le_neg_five_l671_67145


namespace total_number_of_members_l671_67137

-- Define the basic setup
def committees := Fin 5
def members := {m : Finset committees // m.card = 2}

-- State the theorem
theorem total_number_of_members :
  (∃ s : Finset members, s.card = 10) :=
sorry

end total_number_of_members_l671_67137


namespace base_number_is_4_l671_67162

theorem base_number_is_4 (some_number : ℕ) (h : 16^8 = some_number^16) : some_number = 4 :=
sorry

end base_number_is_4_l671_67162


namespace part1_inequality_l671_67158

theorem part1_inequality (a b x y : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) 
    (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_a_ge_x : a ≥ x) : 
    (a - x) ^ 2 + (b - y) ^ 2 ≤ (a + b - x) ^ 2 + y ^ 2 := 
by 
  sorry

end part1_inequality_l671_67158


namespace clock_angle_7_15_l671_67155

noncomputable def hour_angle_at (hour : ℕ) (minutes : ℕ) : ℝ :=
  hour * 30 + (minutes * 0.5)

noncomputable def minute_angle_at (minutes : ℕ) : ℝ :=
  minutes * 6

noncomputable def small_angle (angle1 angle2 : ℝ) : ℝ :=
  let diff := abs (angle1 - angle2)
  if diff <= 180 then diff else 360 - diff

theorem clock_angle_7_15 : small_angle (hour_angle_at 7 15) (minute_angle_at 15) = 127.5 :=
by
  sorry

end clock_angle_7_15_l671_67155


namespace a_is_perfect_square_l671_67103

variable (a b : ℕ)
variable (h1 : 0 < a) 
variable (h2 : 0 < b)
variable (h3 : b % 2 = 1)
variable (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b)

theorem a_is_perfect_square (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b % 2 = 1) 
  (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b) : ∃ n : ℕ, a = n ^ 2 :=
sorry

end a_is_perfect_square_l671_67103


namespace largest_number_of_right_angles_in_convex_octagon_l671_67196

theorem largest_number_of_right_angles_in_convex_octagon : 
  ∀ (angles : Fin 8 → ℝ), 
  (∀ i, 0 < angles i ∧ angles i < 180) → 
  (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5 + angles 6 + angles 7 = 1080) → 
  ∃ k, k ≤ 6 ∧ (∀ i < 8, if angles i = 90 then k = 6 else true) := 
by 
  sorry

end largest_number_of_right_angles_in_convex_octagon_l671_67196


namespace perp_a_beta_l671_67112

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry
noncomputable def Incident (l : line) (p : plane) : Prop := sorry
noncomputable def Perpendicular (l1 l2 : line) : Prop := sorry
noncomputable def Parallel (l1 l2 : line) : Prop := sorry

variables {α β : plane} {a AB : line}

-- Conditions extracted from the problem
axiom condition1 : Perpendicular α β
axiom condition2 : Incident AB β ∧ Incident AB α
axiom condition3 : Parallel a α
axiom condition4 : Perpendicular a AB

-- The statement that needs to be proved
theorem perp_a_beta : Perpendicular a β :=
  sorry

end perp_a_beta_l671_67112


namespace circumradius_of_consecutive_triangle_l671_67135

theorem circumradius_of_consecutive_triangle
  (a b c : ℕ)
  (h : a = b - 1)
  (h1 : c = b + 1)
  (r : ℝ)
  (h2 : r = 4)
  (h3 : a + b > c)
  (h4 : a + c > b)
  (h5 : b + c > a)
  : ∃ R : ℝ, R = 65 / 8 :=
by {
  sorry
}

end circumradius_of_consecutive_triangle_l671_67135


namespace xy_difference_l671_67108

theorem xy_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by {
    sorry
}

end xy_difference_l671_67108


namespace area_enclosed_by_S_l671_67141

open Complex

def five_presentable (v : ℂ) : Prop := abs v = 5

def S : Set ℂ := {u | ∃ v : ℂ, five_presentable v ∧ u = v - (1 / v)}

theorem area_enclosed_by_S : 
  ∃ (area : ℝ), area = 624 / 25 * Real.pi :=
by
  sorry

end area_enclosed_by_S_l671_67141


namespace line_equation_through_P_and_intercepts_l671_67171

-- Define the conditions
structure Point (α : Type*) := 
  (x : α) 
  (y : α)

-- Given point P
def P : Point ℝ := ⟨5, 6⟩

-- Equation of a line passing through (x₀, y₀) and 
-- having the intercepts condition: the x-intercept is twice the y-intercept

theorem line_equation_through_P_and_intercepts :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (a * 5 + b * 6 + c = 0) ∧ 
   ((-c / a = 2 * (-c / b)) ∧ (c ≠ 0)) ∧
   (a = 1 ∧ b = 2 ∧ c = -17) ∨
   (a = 6 ∧ b = -5 ∧ c = 0)) :=
sorry

end line_equation_through_P_and_intercepts_l671_67171


namespace proof_true_proposition_l671_67165

open Classical

def P : Prop := ∀ x : ℝ, x^2 ≥ 0
def Q : Prop := ∃ x : ℚ, x^2 ≠ 3
def true_proposition (p q : Prop) := p ∨ ¬q

theorem proof_true_proposition : P ∧ ¬Q → true_proposition P Q :=
by
  intro h
  sorry

end proof_true_proposition_l671_67165


namespace suitable_survey_l671_67100

inductive Survey
| FavoriteTVPrograms : Survey
| PrintingErrors : Survey
| BatteryServiceLife : Survey
| InternetUsage : Survey

def is_suitable_for_census (s : Survey) : Prop :=
  match s with
  | Survey.PrintingErrors => True
  | _ => False

theorem suitable_survey : is_suitable_for_census Survey.PrintingErrors = True :=
by
  sorry

end suitable_survey_l671_67100


namespace find_four_digit_number_l671_67167

-- Definitions of the digit variables a, b, c, d, and their constraints.
def four_digit_expressions_meet_condition (abcd abc ab : ℕ) (a : ℕ) :=
  ∃ (b c d : ℕ), abcd = (1000 * a + 100 * b + 10 * c + d)
  ∧ abc = (100 * a + 10 * b + c)
  ∧ ab = (10 * a + b)
  ∧ abcd - abc - ab - a = 1787

-- Main statement to be proven.
theorem find_four_digit_number
: ∀ a b c d : ℕ, 
  four_digit_expressions_meet_condition (1000 * a + 100 * b + 10 * c + d) (100 * a + 10 * b + c) (10 * a + b) a
  → (a = 2 ∧ b = 0 ∧ ((c = 0 ∧ d = 9) ∨ (c = 1 ∧ d = 0))) :=
sorry

end find_four_digit_number_l671_67167


namespace fraction_equality_l671_67156

theorem fraction_equality (a b c : ℝ) (hc : c ≠ 0) (h : a / c = b / c) : a = b := 
by
  sorry

end fraction_equality_l671_67156


namespace geom_sequence_sum_correct_l671_67132

noncomputable def geom_sequence_sum (a₁ a₄ : ℕ) (S₅ : ℕ) :=
  ∃ q : ℕ, a₁ = 1 ∧ a₄ = a₁ * q ^ 3 ∧ S₅ = (a₁ * (1 - q ^ 5)) / (1 - q)

theorem geom_sequence_sum_correct : geom_sequence_sum 1 8 31 :=
by {
  sorry
}

end geom_sequence_sum_correct_l671_67132


namespace target_runs_is_282_l671_67127

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_segment : ℝ := 10
def run_rate_remaining_20_overs : ℝ := 12.5
def overs_second_segment : ℝ := 20

-- Define the calculation of runs in the first 10 overs
def runs_first_segment : ℝ := run_rate_first_10_overs * overs_first_segment

-- Define the calculation of runs in the remaining 20 overs
def runs_second_segment : ℝ := run_rate_remaining_20_overs * overs_second_segment

-- Define the target runs
def target_runs : ℝ := runs_first_segment + runs_second_segment

-- State the theorem
theorem target_runs_is_282 : target_runs = 282 :=
by
  -- This is where the proof would go, but it is omitted.
  sorry

end target_runs_is_282_l671_67127


namespace division_value_of_712_5_by_12_5_is_57_l671_67143

theorem division_value_of_712_5_by_12_5_is_57 : 712.5 / 12.5 = 57 :=
  by
    sorry

end division_value_of_712_5_by_12_5_is_57_l671_67143
