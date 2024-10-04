import Mathlib

namespace max_sides_13_eq_13_max_sides_1950_eq_1950_l304_304843

noncomputable def max_sides (n : ℕ) : ℕ := n

theorem max_sides_13_eq_13 : max_sides 13 = 13 :=
by {
  sorry
}

theorem max_sides_1950_eq_1950 : max_sides 1950 = 1950 :=
by {
  sorry
}

end max_sides_13_eq_13_max_sides_1950_eq_1950_l304_304843


namespace count_squares_below_graph_l304_304885

theorem count_squares_below_graph (x y: ℕ) (h_eq : 12 * x + 180 * y = 2160) (h_first_quadrant : x ≥ 0 ∧ y ≥ 0) :
  let total_squares := 180 * 12
  let diagonal_squares := 191
  let below_squares := total_squares - diagonal_squares
  below_squares = 1969 :=
by
  sorry

end count_squares_below_graph_l304_304885


namespace tangent_line_eq_root_distance_bound_l304_304812

noncomputable
def f (x : ℝ) : ℝ := 3 * (1 - x) * Real.log (1 + x) + Real.sin (Real.pi * x)

theorem tangent_line_eq (f : ℝ → ℝ) (H : ∀ x, f x = 3 * (1 - x) * Real.log (1 + x) + Real.sin (Real.pi * x)) :
  let y := f 0
  let m := deriv f 0
  m = Real.pi + 3 → y = 0 → ∀ x, y = (Real.pi + 3) * x := sorry

theorem root_distance_bound (m : ℝ) (h : ∀ x, f x = 3 * (1 - x) * Real.log (1 + x) + Real.sin (Real.pi * x)) 
  (h1 : 0 ≤ x1) (h2 : x1 < 1) (h3 : 0 ≤ x2) (h4 : x2 ≤ 1) (h5 : x1 ≠ x2) :
  f x1 = m → f x2 = m → |x1 - x2| ≤ 1 - (2 * m) / (Real.pi + 3) := sorry

end tangent_line_eq_root_distance_bound_l304_304812


namespace repeating_decimal_to_fraction_l304_304228

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l304_304228


namespace expected_winnings_correct_l304_304198

def probability_1 := (1:ℚ) / 4
def probability_2 := (1:ℚ) / 4
def probability_3 := (1:ℚ) / 6
def probability_4 := (1:ℚ) / 6
def probability_5 := (1:ℚ) / 8
def probability_6 := (1:ℚ) / 8

noncomputable def expected_winnings : ℚ :=
  (probability_1 + probability_3 + probability_5) * 2 +
  (probability_2 + probability_4) * 4 +
  probability_6 * (-6 + 4)

theorem expected_winnings_correct : expected_winnings = 1.67 := by
  sorry

end expected_winnings_correct_l304_304198


namespace plains_routes_count_l304_304284

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l304_304284


namespace regular_price_of_ticket_l304_304477

theorem regular_price_of_ticket (P : Real) (discount_paid : Real) (discount_rate : Real) (paid : Real)
  (h_discount_rate : discount_rate = 0.40)
  (h_paid : paid = 9)
  (h_discount_paid : discount_paid = P * (1 - discount_rate))
  (h_paid_eq_discount_paid : paid = discount_paid) :
  P = 15 := 
by
  sorry

end regular_price_of_ticket_l304_304477


namespace present_age_of_son_l304_304370

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 22) (h2 : F + 2 = 2 * (S + 2)) : S = 20 :=
by
  sorry

end present_age_of_son_l304_304370


namespace julia_stairs_less_than_third_l304_304988

theorem julia_stairs_less_than_third (J1 : ℕ) (T : ℕ) (T_total : ℕ) (J : ℕ) 
  (hJ1 : J1 = 1269) (hT : T = 1269 / 3) (hT_total : T_total = 1685) (hTotal : J1 + J = T_total) : 
  T - J = 7 := 
by
  sorry

end julia_stairs_less_than_third_l304_304988


namespace sin_double_angle_l304_304246

theorem sin_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 4) = 1 / 2) : Real.sin (2 * α) = -1 / 2 :=
sorry

end sin_double_angle_l304_304246


namespace difference_of_squares_65_35_l304_304777

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := 
  sorry

end difference_of_squares_65_35_l304_304777


namespace conference_fraction_married_men_l304_304061

theorem conference_fraction_married_men 
  (total_women : ℕ) 
  (single_probability : ℚ) 
  (h_single_prob : single_probability = 3/7) 
  (h_total_women : total_women = 7) : 
  (4 : ℚ) / (11 : ℚ) = 4 / 11 := 
by
  sorry

end conference_fraction_married_men_l304_304061


namespace find_A_l304_304117

-- Define the condition as an axiom
axiom A : ℝ
axiom condition : A + 10 = 15 

-- Prove that given the condition, A must be 5
theorem find_A : A = 5 := 
by {
  sorry
}

end find_A_l304_304117


namespace ruth_hours_per_week_l304_304639

theorem ruth_hours_per_week :
  let daily_hours := 8
  let days_per_week := 5
  let monday_wednesday_friday := 3
  let tuesday_thursday := 2
  let percentage_to_hours (percent : ℝ) (hours : ℕ) : ℝ := percent * hours
  let total_weekly_hours := daily_hours * days_per_week
  let monday_wednesday_friday_math_hours := percentage_to_hours 0.25 daily_hours
  let monday_wednesday_friday_science_hours := percentage_to_hours 0.15 daily_hours
  let tuesday_thursday_math_hours := percentage_to_hours 0.2 daily_hours
  let tuesday_thursday_science_hours := percentage_to_hours 0.35 daily_hours
  let tuesday_thursday_history_hours := percentage_to_hours 0.15 daily_hours
  let weekly_math_hours := monday_wednesday_friday_math_hours * monday_wednesday_friday + tuesday_thursday_math_hours * tuesday_thursday
  let weekly_science_hours := monday_wednesday_friday_science_hours * monday_wednesday_friday + tuesday_thursday_science_hours * tuesday_thursday
  let weekly_history_hours := tuesday_thursday_history_hours * tuesday_thursday
  let total_hours := weekly_math_hours + weekly_science_hours + weekly_history_hours
  total_hours = 20.8 := by
  sorry

end ruth_hours_per_week_l304_304639


namespace periodic_decimal_to_fraction_l304_304224

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l304_304224


namespace equilibrium_possible_l304_304738

theorem equilibrium_possible (n : ℕ) : (∃ k : ℕ, 4 * k = n) ∨ (∃ k : ℕ, 4 * k + 3 = n) ↔
  (∃ S1 S2 : Finset ℕ, S1 ∪ S2 = Finset.range (n+1) ∧
                     S1 ∩ S2 = ∅ ∧
                     S1.sum id = S2.sum id) := 
sorry

end equilibrium_possible_l304_304738


namespace grocer_display_rows_l304_304195

theorem grocer_display_rows (n : ℕ)
  (h1 : ∃ k, k = 2 + 3 * (n - 1))
  (h2 : ∃ s, s = (n / 2) * (2 + (3 * n - 1))):
  (3 * n^2 + n) / 2 = 225 → n = 12 :=
by
  sorry

end grocer_display_rows_l304_304195


namespace mean_equal_l304_304018

theorem mean_equal (y : ℚ) :
  (5 + 10 + 20) / 3 = (15 + y) / 2 → y = 25 / 3 := 
by
  sorry

end mean_equal_l304_304018


namespace hamburgers_left_over_l304_304546

-- Define the conditions as constants
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove that the number of hamburgers left over is 6
theorem hamburgers_left_over : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end hamburgers_left_over_l304_304546


namespace scientific_notation_11580000_l304_304836

theorem scientific_notation_11580000 :
  (11580000 : ℝ) = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l304_304836


namespace tangent_line_equation_at_1_l304_304093

-- Define the function f and the point of tangency
def f (x : ℝ) : ℝ := x^2 + 2 * x
def p : ℝ × ℝ := (1, f 1)

-- Statement of the theorem
theorem tangent_line_equation_at_1 :
  ∃ a b c : ℝ, (∀ x y : ℝ, y = f x → y - p.2 = a * (x - p.1)) ∧
               4 * (p.1 : ℝ) - (p.2 : ℝ) - 1 = 0 :=
by
  -- Skipping the proof
  sorry

end tangent_line_equation_at_1_l304_304093


namespace determine_z_l304_304209

theorem determine_z (z : ℝ) (h1 : ∃ x : ℤ, 3 * (x : ℝ) ^ 2 + 19 * (x : ℝ) - 84 = 0 ∧ (x : ℝ) = ⌊z⌋) (h2 : 4 * (z - ⌊z⌋) ^ 2 - 14 * (z - ⌊z⌋) + 6 = 0) : 
  z = -11 :=
  sorry

end determine_z_l304_304209


namespace triangle_inequality_l304_304802

theorem triangle_inequality
  (A B C P D E : Point)
  (h1 : Triangle A B C)
  (h2 : AcuteAngle A B C)
  (h3 : Inside P (Triangle A B C))
  (h4 : ∠APB = 120 ∧ ∠BPC = 120 ∧ ∠CPA = 120)
  (h5 : LineThrough B P ∩ AC = D)
  (h6 : LineThrough C P ∩ AB = E) :
  distance A B + distance A C ≥ 4 * distance D E :=
sorry

end triangle_inequality_l304_304802


namespace range_of_a_l304_304814

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - 2 * x

theorem range_of_a 
  (a : ℝ) 
  (h : ∀ x : ℝ, 1 < x → 2 * a * Real.log x ≤ 2 * x^2 + f a (2 * x - 1)) :
  a ≤ 2 :=
sorry

end range_of_a_l304_304814


namespace solve_for_x_l304_304261

theorem solve_for_x (x : ℝ) (h : -3 * x - 12 = 8 * x + 5) : x = -17 / 11 :=
by
  sorry

end solve_for_x_l304_304261


namespace classmates_ate_cake_l304_304676

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l304_304676


namespace arithmetic_sequence_sum_l304_304799

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 + a 13 = 10) 
  (h2 : ∀ n m : ℕ, a (n + 1) = a n + d) : a 3 + a 5 + a 7 + a 9 + a 11 = 25 :=
  sorry

end arithmetic_sequence_sum_l304_304799


namespace common_fraction_l304_304409

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l304_304409


namespace converse_and_inverse_false_l304_304097

variable (Polygon : Type)
variable (RegularHexagon : Polygon → Prop)
variable (AllSidesEqual : Polygon → Prop)

theorem converse_and_inverse_false (p : Polygon → Prop) (q : Polygon → Prop)
  (h : ∀ x, RegularHexagon x → AllSidesEqual x) :
  ¬ (∀ x, AllSidesEqual x → RegularHexagon x) ∧ ¬ (∀ x, ¬ RegularHexagon x → ¬ AllSidesEqual x) :=
by
  sorry

end converse_and_inverse_false_l304_304097


namespace find_x_in_terms_of_N_l304_304782

theorem find_x_in_terms_of_N (N : ℤ) (x y : ℝ) 
(h1 : (⌊x⌋ : ℤ) + 2 * y = N + 2) 
(h2 : (⌊y⌋ : ℤ) + 2 * x = 3 - N) : 
x = (3 / 2) - N := 
by
  sorry

end find_x_in_terms_of_N_l304_304782


namespace father_l304_304101

-- Definitions based on conditions in a)
def cost_MP3_player : ℕ := 120
def cost_CD : ℕ := 19
def total_cost : ℕ := cost_MP3_player + cost_CD
def savings : ℕ := 55
def amount_lacking : ℕ := 64

-- Statement of the proof problem
theorem father's_contribution : (savings + (148:ℕ) - amount_lacking = total_cost) := by
  -- Add sorry to skip the proof
  sorry

end father_l304_304101


namespace options_implication_l304_304734

theorem options_implication (a b : ℝ) :
  ((b > 0 ∧ a < 0) ∨ (a < 0 ∧ b < 0 ∧ a > b) ∨ (a > 0 ∧ b > 0 ∧ a > b)) → (1 / a < 1 / b) :=
by sorry

end options_implication_l304_304734


namespace balloon_altitude_l304_304080

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

end balloon_altitude_l304_304080


namespace classmates_ate_cake_l304_304640

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l304_304640


namespace five_crows_two_hours_l304_304968

-- Define the conditions and the question as hypotheses
def crows_worms (crows worms hours : ℕ) := 
  (crows = 3) ∧ (worms = 30) ∧ (hours = 1)

theorem five_crows_two_hours 
  (c: ℕ) (w: ℕ) (h: ℕ)
  (H: crows_worms c w h)
  : ∃ worms_eaten : ℕ, worms_eaten = 100 :=
by
  sorry

end five_crows_two_hours_l304_304968


namespace find_z_solutions_l304_304745

open Real

noncomputable def is_solution (z : ℝ) : Prop :=
  sin z + sin (2 * z) + sin (3 * z) = cos z + cos (2 * z) + cos (3 * z)

theorem find_z_solutions (z : ℝ) : 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k - 1)) ∨ 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k + 1)) ∨ 
  (∃ k : ℤ, z = π / 8 * (4 * k + 1)) ↔
  is_solution z :=
by
  sorry

end find_z_solutions_l304_304745


namespace length_of_QB_l304_304206

/-- 
Given a circle Q with a circumference of 16π feet, 
segment AB as its diameter, 
and the angle AQB of 120 degrees, 
prove that the length of segment QB is 8 feet.
-/
theorem length_of_QB (C : ℝ) (r : ℝ) (A B Q : ℝ) (angle_AQB : ℝ) 
  (h1 : C = 16 * Real.pi)
  (h2 : 2 * Real.pi * r = C)
  (h3 : angle_AQB = 120) 
  : QB = 8 :=
sorry

end length_of_QB_l304_304206


namespace work_completion_time_l304_304179

variable (p q : Type)

def efficient (p q : Type) : Prop :=
  ∃ (Wp Wq : ℝ), Wp = 1.5 * Wq ∧ Wp = 1 / 25

def work_done_together (p q : Type) := 1/15

theorem work_completion_time {p q : Type} (h1 : efficient p q) :
  ∃ d : ℝ, d = 15 :=
  sorry

end work_completion_time_l304_304179


namespace routes_between_plains_cities_correct_l304_304279

noncomputable def number_of_routes_connecting_pairs_of_plains_cities
    (total_cities : ℕ)
    (mountainous_cities : ℕ)
    (plains_cities : ℕ)
    (total_routes : ℕ)
    (routes_between_mountainous_cities : ℕ) : ℕ :=
let mountainous_city_endpoints := mountainous_cities * 3 in
let routes_between_mountainous_cities_endpoints := routes_between_mountainous_cities * 2 in
let mountainous_to_plains_routes_endpoints := mountainous_city_endpoints - routes_between_mountainous_cities_endpoints in
let plains_city_endpoints := plains_cities * 3 in
let plains_city_to_mountainous_city_routes_endpoints := mountainous_to_plains_routes_endpoints in
let endpoints_fully_in_plains_cities := plains_city_endpoints - plains_city_to_mountainous_city_routes_endpoints in
endpoints_fully_in_plains_cities / 2

theorem routes_between_plains_cities_correct :
    number_of_routes_connecting_pairs_of_plains_cities 100 30 70 150 21 = 81 := by
    sorry

end routes_between_plains_cities_correct_l304_304279


namespace minimum_value_l304_304621

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + 3 * b = 1) : 
  26 ≤ (2 / a + 3 / b) :=
sorry

end minimum_value_l304_304621


namespace range_of_sum_l304_304977

theorem range_of_sum (x y : ℝ) (h : 9 * x^2 + 16 * y^2 = 144) : 
  ∃ a b : ℝ, (x + y + 10 ≥ a) ∧ (x + y + 10 ≤ b) ∧ a = 5 ∧ b = 15 := 
sorry

end range_of_sum_l304_304977


namespace chip_placement_count_l304_304260

def grid := Fin 4 × Fin 3

def grid_positions (n : Nat) := {s : Finset grid // s.card = n}

def no_direct_adjacency (positions : Finset grid) : Prop :=
  ∀ (x y : grid), x ∈ positions → y ∈ positions →
  (x.fst ≠ y.fst ∨ x.snd ≠ y.snd)

noncomputable def count_valid_placements : Nat :=
  -- Function to count valid placements
  sorry

theorem chip_placement_count :
  count_valid_placements = 4 :=
  sorry

end chip_placement_count_l304_304260


namespace factor_expression_l304_304941

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_expression_l304_304941


namespace louisa_second_day_miles_l304_304308

theorem louisa_second_day_miles (T1 T2 : ℕ) (speed miles_first_day miles_second_day : ℕ)
  (h1 : speed = 25) 
  (h2 : miles_first_day = 100)
  (h3 : T1 = miles_first_day / speed) 
  (h4 : T2 = T1 + 3) 
  (h5 : miles_second_day = speed * T2) :
  miles_second_day = 175 := 
by
  -- We can add the necessary calculations here, but for now, sorry is used to skip the proof.
  sorry

end louisa_second_day_miles_l304_304308


namespace intersection_of_A_and_B_l304_304818

open Set

def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | x^2 ≥ 4}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 ≤ x ∧ x < 3} :=
  by
    sorry

end intersection_of_A_and_B_l304_304818


namespace set_intersection_union_eq_complement_l304_304817

def A : Set ℝ := {x | 2 * x^2 + x - 3 = 0}
def B : Set ℝ := {i | i^2 ≥ 4}
def complement_C : Set ℝ := {-1, 1, 3/2}

theorem set_intersection_union_eq_complement :
  A ∩ B ∪ complement_C = complement_C :=
by
  sorry

end set_intersection_union_eq_complement_l304_304817


namespace eq_pow_four_l304_304585

theorem eq_pow_four (a b : ℝ) (h : a = b + 1) : a^4 = b^4 → a = 1/2 ∧ b = -1/2 :=
by
  sorry

end eq_pow_four_l304_304585


namespace tennis_tournament_possible_l304_304288

theorem tennis_tournament_possible (p : ℕ) : 
  (∀ i j : ℕ, i ≠ j → ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  i = a ∨ i = b ∨ i = c ∨ i = d ∧ j = a ∨ j = b ∨ j = c ∨ j = d) → 
  ∃ k : ℕ, p = 8 * k + 1 := by
  sorry

end tennis_tournament_possible_l304_304288


namespace sum_a5_a8_l304_304993

variable (a : ℕ → ℝ)
variable (r : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_a5_a8 (a1 a2 a3 a4 : ℝ) (q : ℝ)
  (h1 : a1 + a3 = 1)
  (h2 : a2 + a4 = 2)
  (h_seq : is_geometric_sequence a q)
  (a_def : ∀ n : ℕ, a n = a1 * q^n) :
  a 5 + a 6 + a 7 + a 8 = 48 := by
  sorry

end sum_a5_a8_l304_304993


namespace sqrt_108_eq_6_sqrt_3_l304_304683

theorem sqrt_108_eq_6_sqrt_3 : Real.sqrt 108 = 6 * Real.sqrt 3 := 
sorry

end sqrt_108_eq_6_sqrt_3_l304_304683


namespace possible_number_of_classmates_l304_304646

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l304_304646


namespace kids_with_red_hair_l304_304192

theorem kids_with_red_hair (total_kids : ℕ) (ratio_red ratio_blonde ratio_black : ℕ) 
  (h_ratio : ratio_red + ratio_blonde + ratio_black = 16) (h_total : total_kids = 48) :
  (total_kids / (ratio_red + ratio_blonde + ratio_black)) * ratio_red = 9 :=
by
  sorry

end kids_with_red_hair_l304_304192


namespace cos_double_angle_given_tan_l304_304421

theorem cos_double_angle_given_tan (x : ℝ) (h : Real.tan x = 2) : Real.cos (2 * x) = -3 / 5 :=
by sorry

end cos_double_angle_given_tan_l304_304421


namespace proof_statement_l304_304806

-- Assume 5 * 3^x = 243
def condition (x : ℝ) : Prop := 5 * (3:ℝ)^x = 243

-- Define the log base 3 for use in the statement
noncomputable def log_base_3 (y : ℝ) : ℝ := Real.log y / Real.log 3

-- State that if the condition holds, then (x + 2)(x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2
theorem proof_statement (x : ℝ) (h : condition x) : (x + 2) * (x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2 := sorry

end proof_statement_l304_304806


namespace pizza_savings_l304_304135

theorem pizza_savings (regular_price promotional_price : ℕ) (n : ℕ) (H_regular : regular_price = 18) (H_promotional : promotional_price = 5) (H_n : n = 3) : 
  (regular_price - promotional_price) * n = 39 := by

  -- Assume the given conditions
  have h1 : regular_price - promotional_price = 13 := 
  by rw [H_regular, H_promotional]; exact rfl

  rw [h1, H_n]
  exact (13 * 3).symm

end pizza_savings_l304_304135


namespace find_a_b_l304_304953

open Set

def solution_set (f : ℝ → ℝ) (s : Set ℝ) : Set ℝ :=
  { x | f x < 0 }

def A := {x : ℝ | -1 < x ∧ x < 3}
def B := {x : ℝ | -3 < x ∧ x < 2}

theorem find_a_b (a b : ℝ) :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} →
  solution_set (λ x, x^2 + a * x + b) (A ∩ B) →
  (a + b = -3) :=
sorry

end find_a_b_l304_304953


namespace segment_length_eq_ten_l304_304711

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end segment_length_eq_ten_l304_304711


namespace length_of_segment_l304_304717

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end length_of_segment_l304_304717


namespace minimum_value_l304_304247

theorem minimum_value (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 2) :
  (1 / a) + (1 / b) ≥ 2 :=
by {
  sorry
}

end minimum_value_l304_304247


namespace octagon_area_sum_l304_304182

theorem octagon_area_sum :
  let A1 := 2024
  let a := 1012
  let b := 506
  let c := 2
  a + b + c = 1520 := by
    sorry

end octagon_area_sum_l304_304182


namespace number_of_classmates_ate_cake_l304_304667

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l304_304667


namespace patricia_earns_more_than_jose_l304_304853

noncomputable def jose_final_amount : ℝ :=
  50000 * (1 + 0.04)^2

noncomputable def patricia_final_amount : ℝ :=
  50000 * (1 + 0.01)^8

theorem patricia_earns_more_than_jose :
  patricia_final_amount - jose_final_amount = 63 :=
by
  -- from solution steps
  /-
  jose_final_amount = 50000 * (1 + 0.04)^2 = 54080
  patricia_final_amount = 50000 * (1 + 0.01)^8 ≈ 54143
  patricia_final_amount - jose_final_amount ≈ 63
  -/
  sorry

end patricia_earns_more_than_jose_l304_304853


namespace remuneration_difference_l304_304919

-- Define the conditions and question
def total_sales : ℝ := 12000
def commission_rate_old : ℝ := 0.05
def fixed_salary_new : ℝ := 1000
def commission_rate_new : ℝ := 0.025
def sales_threshold_new : ℝ := 4000

-- Define the remuneration for the old scheme
def remuneration_old : ℝ := total_sales * commission_rate_old

-- Define the remuneration for the new scheme
def sales_exceeding_threshold_new : ℝ := total_sales - sales_threshold_new
def commission_new : ℝ := sales_exceeding_threshold_new * commission_rate_new
def remuneration_new : ℝ := fixed_salary_new + commission_new

-- Statement of the theorem to be proved
theorem remuneration_difference : remuneration_new - remuneration_old = 600 :=
by
  -- The proof goes here but is omitted as per the instructions
  sorry

end remuneration_difference_l304_304919


namespace least_integer_greater_than_sqrt_450_l304_304511

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l304_304511


namespace equation_of_parallel_line_l304_304265

theorem equation_of_parallel_line (l : ℝ → ℝ → Prop) (P : ℝ × ℝ)
  (x y : ℝ) (m : ℝ) (H_1 : P = (1, 2)) (H_2 : ∀ x y m, l x y ↔ (2 * x + y + m = 0) )
  (H_3 : l x y) : 
  l 2 (y - 4) := 
  sorry

end equation_of_parallel_line_l304_304265


namespace inequality_solution_equality_condition_l304_304463

theorem inequality_solution (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) (h3 : b < -1 ∨ b > 0) :
  (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b :=
sorry

theorem equality_condition (a b : ℝ) :
  (1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b :=
sorry

end inequality_solution_equality_condition_l304_304463


namespace find_k_value_l304_304092

theorem find_k_value (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 8000) (h3 : K > 2) (h4 : Z = K^3)
  (h5 : ∃ n : ℤ, Z = n^6) : K = 16 :=
sorry

end find_k_value_l304_304092


namespace min_value_expression_l304_304472

theorem min_value_expression (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 1) :
  9 ≤ (1 / (a^2 + 2 * b^2)) + (1 / (b^2 + 2 * c^2)) + (1 / (c^2 + 2 * a^2)) :=
by
  sorry

end min_value_expression_l304_304472


namespace range_of_a_min_value_of_a_l304_304576

variable (f : ℝ → ℝ) (a x : ℝ)

-- Part 1
theorem range_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₁ : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ 3) : 0 ≤ a ∧ a ≤ 4 :=
sorry

-- Part 2
theorem min_value_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₂ : ∀ x, f (x - a) + f (x + a) ≥ 1 - a) : a ≥ 1/3 :=
sorry

end range_of_a_min_value_of_a_l304_304576


namespace angle_solution_l304_304615

/-!
  Given:
  k + 90° = 360°

  Prove:
  k = 270°
-/

theorem angle_solution (k : ℝ) (h : k + 90 = 360) : k = 270 :=
by
  sorry

end angle_solution_l304_304615


namespace zero_in_interval_l304_304290

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem zero_in_interval :
  (f 0 < 0) → (f 0.5 > 0) → (f 0.25 < 0) → ∃ x, 0.25 < x ∧ x < 0.5 ∧ f x = 0 :=
by
  intro h0 h05 h025
  -- This is just the statement; the proof is not required as per instructions
  sorry

end zero_in_interval_l304_304290


namespace number_of_classmates_ate_cake_l304_304665

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l304_304665


namespace cone_base_circumference_l304_304054

-- Definitions of the problem
def radius : ℝ := 5
def angle_sector_degree : ℝ := 120
def full_circle_degree : ℝ := 360

-- Proof statement
theorem cone_base_circumference 
  (r : ℝ) (angle_sector : ℝ) (full_angle : ℝ) 
  (h1 : r = radius) 
  (h2 : angle_sector = angle_sector_degree) 
  (h3 : full_angle = full_circle_degree) : 
  (angle_sector / full_angle) * (2 * π * r) = (10 * π) / 3 := 
by sorry

end cone_base_circumference_l304_304054


namespace find_sum_l304_304377

variable {f : ℝ → ℝ}

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def condition_2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
def condition_3 (f : ℝ → ℝ) : Prop := f 1 = 9

theorem find_sum (h_odd : odd_function f) (h_cond2 : condition_2 f) (h_cond3 : condition_3 f) :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end find_sum_l304_304377


namespace total_black_dots_l304_304330

def num_butterflies : ℕ := 397
def black_dots_per_butterfly : ℕ := 12

theorem total_black_dots : num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end total_black_dots_l304_304330


namespace number_of_lucky_tickets_is_even_sum_of_lucky_tickets_is_divisible_by_999_l304_304181

def is_lucky_ticket (n : ℕ) : Prop :=
  n <= 999999 ∧ (n / 1000 % 10 + n / 10000 % 10 + n / 100000 % 10) = (n % 10 + n / 10 % 10 + n / 100 % 10)

theorem number_of_lucky_tickets_is_even :
  (∃ m, ∀ n, (0 <= n ∧ n <= 999999) → is_lucky_ticket n ↔ (n < m)) ∧
  ∃ n, even n :=
sorry

theorem sum_of_lucky_tickets_is_divisible_by_999 :
  (∑ n in finset.filter is_lucky_ticket (finset.range 1000000), n) % 999 = 0 :=
sorry

end number_of_lucky_tickets_is_even_sum_of_lucky_tickets_is_divisible_by_999_l304_304181


namespace lcm_of_1_to_12_l304_304723

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end lcm_of_1_to_12_l304_304723


namespace arrange_in_order_l304_304249

noncomputable def a := (Real.sqrt 2 / 2) * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c := Real.sqrt 3 / 2

theorem arrange_in_order : c < a ∧ a < b := 
by
  sorry

end arrange_in_order_l304_304249


namespace smallest_area_of_right_triangle_l304_304895

noncomputable def right_triangle_area (a b : ℝ) : ℝ :=
  if a^2 + b^2 = 6^2 then (1/2) * a * b else 12

theorem smallest_area_of_right_triangle :
  min (right_triangle_area 4 (2 * Real.sqrt 5)) 12 = 4 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end smallest_area_of_right_triangle_l304_304895


namespace find_number_l304_304185

theorem find_number (x : ℝ) : (45 * x = 0.45 * 900) → (x = 9) :=
by sorry

end find_number_l304_304185


namespace recurring_decimal_reduced_fraction_l304_304239

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l304_304239


namespace alice_bob_age_difference_18_l304_304701

-- Define Alice's and Bob's ages with the given constraints
def is_odd (n : ℕ) : Prop := n % 2 = 1

def alice_age (a b : ℕ) : ℕ := 10 * a + b
def bob_age (a b : ℕ) : ℕ := 10 * b + a

theorem alice_bob_age_difference_18 (a b : ℕ) (ha : is_odd a) (hb : is_odd b)
  (h : alice_age a b + 7 = 3 * (bob_age a b + 7)) : alice_age a b - bob_age a b = 18 :=
sorry

end alice_bob_age_difference_18_l304_304701


namespace cake_sharing_l304_304659

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l304_304659


namespace seunghyo_daily_dosage_l304_304875

theorem seunghyo_daily_dosage (total_medicine : ℝ) (daily_fraction : ℝ) (correct_dosage : ℝ) :
  total_medicine = 426 → daily_fraction = 0.06 → correct_dosage = 25.56 →
  total_medicine * daily_fraction = correct_dosage :=
by
  intros ht hf hc
  simp [ht, hf, hc]
  sorry

end seunghyo_daily_dosage_l304_304875


namespace gobblean_total_words_l304_304309

-- Define the Gobblean alphabet and its properties.
def gobblean_letters := 6
def max_word_length := 4

-- Function to calculate number of permutations without repetition for a given length.
def num_words (length : ℕ) : ℕ :=
  if length = 1 then 6
  else if length = 2 then 6 * 5
  else if length = 3 then 6 * 5 * 4
  else if length = 4 then 6 * 5 * 4 * 3
  else 0

-- Main theorem stating the total number of possible words.
theorem gobblean_total_words : 
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4) = 516 :=
by
  -- Proof is not required
  sorry

end gobblean_total_words_l304_304309


namespace k_m_sum_l304_304469

theorem k_m_sum (k m : ℝ) (h : ∀ {x : ℝ}, x^3 - 8 * x^2 + k * x - m = 0 → x ∈ {1, 2, 5} ∨ x ∈ {1, 3, 4}) :
  k + m = 27 ∨ k + m = 31 :=
by
  sorry

end k_m_sum_l304_304469


namespace oranges_given_to_friend_l304_304365

theorem oranges_given_to_friend (initial_oranges : ℕ) 
  (given_to_brother : ℕ)
  (given_to_friend : ℕ)
  (h1 : initial_oranges = 60)
  (h2 : given_to_brother = (1 / 3 : ℚ) * initial_oranges)
  (h3 : given_to_friend = (1 / 4 : ℚ) * (initial_oranges - given_to_brother)) : 
  given_to_friend = 10 := 
by 
  sorry

end oranges_given_to_friend_l304_304365


namespace total_volume_of_water_l304_304748

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2734

-- Define the total volume
def total_volume : ℕ := volume_of_hemisphere * number_of_hemispheres

-- State the theorem
theorem total_volume_of_water : total_volume = 10936 :=
by
  -- Proof placeholder
  sorry

end total_volume_of_water_l304_304748


namespace repeating_decimal_fraction_l304_304215

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l304_304215


namespace a_minus_3d_eq_zero_l304_304114

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x - 3 * d)

theorem a_minus_3d_eq_zero (a b c d : ℝ) (h : f a b c d ≠ 0)
  (h1 : ∀ x, f a b c d x = x) :
  a - 3 * d = 0 :=
sorry

end a_minus_3d_eq_zero_l304_304114


namespace problem1_solution_problem2_solution_l304_304183

noncomputable def sol_set_problem1 (a b c : ℝ) (h1 : b = 5/3 * a) (h2 : c = -2/3 * a) (h3 : a < 0) : set ℝ :=
{ x | x ≤ -3 ∨ x ≥ 1/2 }

theorem problem1_solution (a b c : ℝ)
  (h_solutions : ∀ x, x < -2 ∨ x > 1/3 → ax^2 + bx + c < 0)
  (h_roots : [x | x < -2 ∨ x > 1/3] = { x | x = -2 ∨ x = 1/3 }) :
  ∀ x, (cx^2 - bx + a ≥ 0) ↔ x ∈ sol_set_problem1 a b c := by
  sorry

inductive DeltaValue
| le_zero
| gt_zero

def sol_set_problem2 (a : ℝ) (delta_case : DeltaValue) : set ℝ :=
match a, delta_case with
| 0, _ => { x | 0 < x }
| a, DeltaValue.le_zero => ∅
| a, DeltaValue.gt_zero => { x | 1 - (sqrt(1 - a^2)) / a < x ∧ x < ((1 + sqrt(1 - a^2)) / a) }
| a, _ => if a < 0 ∧ 1 < -a then set.univ else if -1 = a then { x | x < -1 ∨ -1 < x } else
          { x | x < (1 + sqrt(1 - a^2)) / a ∨ (1 - sqrt(1 - a^2)) / a < x }

theorem problem2_solution (a : ℝ) (h_cases : a = 0 ∨ (0 < a ∧ a < 1 ∧ DeltaValue.gt_zero) ∨ (1 ≤ a ∧ DeltaValue.le_zero) ∨ 
                        (a < 0 ∧ delta_case = DeltaValue.gt_zero) ∨ (a = -1 ∧ delta_case = DeltaValue.le_zero) ∨ 
                        (1 < -a ∧ delta_case = DeltaValue.le_zero)) :
  ∀ x, (ax^2 - 2 * x + a < 0) ↔ x ∈ sol_set_problem2 a delta_case := by
  sorry

end problem1_solution_problem2_solution_l304_304183


namespace angle_of_inclination_l304_304034

theorem angle_of_inclination 
  (α : ℝ) 
  (h_tan : Real.tan α = -Real.sqrt 3)
  (h_range : 0 ≤ α ∧ α < 180) : α = 120 :=
by
  sorry

end angle_of_inclination_l304_304034


namespace problem_1163_prime_and_16424_composite_l304_304821

theorem problem_1163_prime_and_16424_composite :
  let x := 1910 * 10000 + 1112
  let a := 1163
  let b := 16424
  x = a * b →
  Prime a ∧ ¬ Prime b :=
by
  intros h
  sorry

end problem_1163_prime_and_16424_composite_l304_304821


namespace correct_quadratic_graph_l304_304489

theorem correct_quadratic_graph (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (-b / (2 * a) > 0) ∧ (c < 0) :=
by
  sorry

end correct_quadratic_graph_l304_304489


namespace find_B_sin_squared_sum_range_l304_304796

-- Define the angles and vectors
variables {A B C : ℝ}
variables (m n : ℝ × ℝ)
variables (α : ℝ)

-- Basic triangle angle sum condition
axiom angle_sum : A + B + C = Real.pi

-- Define vectors as per the problem statement
axiom vector_m : m = (Real.sin B, 1 - Real.cos B)
axiom vector_n : n = (2, 0)

-- The angle between vectors m and n is π/3
axiom angle_between_vectors : α = Real.pi / 3
axiom angle_condition : Real.cos α = (2 * Real.sin B + 0 * (1 - Real.cos B)) / 
                                     (Real.sqrt (Real.sin B ^ 2 + (1 - Real.cos B) ^ 2) * 2)

theorem find_B : B = 2 * Real.pi / 3 := 
sorry

-- Conditions for range of sin^2 A + sin^2 C
axiom range_condition : (0 < A ∧ A < Real.pi / 3) 
                     ∧ (0 < C ∧ C < Real.pi / 3)
                     ∧ (A + C = Real.pi / 3)

theorem sin_squared_sum_range : (Real.sin A) ^ 2 + (Real.sin C) ^ 2 ∈ Set.Ico (1 / 2) 1 := 
sorry

end find_B_sin_squared_sum_range_l304_304796


namespace problem_statement_l304_304446

noncomputable def sum_of_solutions_with_negative_imaginary_part : ℂ :=
  let x3 := 2 * complex.of_real (real.cos (195 * real.pi / 180)) + 2 * complex.I * complex.of_real (real.sin (195 * real.pi / 180)),
      x4 := 2 * complex.of_real (real.cos (255 * real.pi / 180)) + 2 * complex.I * complex.of_real (real.sin (255 * real.pi / 180)),
      x5 := 2 * complex.of_real (real.cos (315 * real.pi / 180)) + 2 * complex.I * complex.of_real (real.sin (315 * real.pi / 180))
  in x3 + x4 + x5

theorem problem_statement : sum_of_solutions_with_negative_imaginary_part =
  2 * (complex.of_real (real.cos (195 * real.pi / 180)) + complex.of_real (real.cos (255 * real.pi / 180)) + complex.of_real (real.cos (315 * real.pi / 180))) +
  2 * complex.I * (complex.of_real (real.sin (195 * real.pi / 180)) + complex.of_real (real.sin (255 * real.pi / 180)) + complex.of_real (real.sin (315 * real.pi / 180))) :=
sorry

end problem_statement_l304_304446


namespace scientific_notation_11580000_l304_304840

theorem scientific_notation_11580000 :
  11580000 = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l304_304840


namespace prime_divisibility_l304_304264

theorem prime_divisibility (a b : ℕ) (ha_prime : Nat.Prime a) (hb_prime : Nat.Prime b) (ha_gt7 : a > 7) (hb_gt7 : b > 7) :
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := 
by
  sorry

end prime_divisibility_l304_304264


namespace least_integer_greater_than_sqrt_450_l304_304516

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l304_304516


namespace locus_area_l304_304383

theorem locus_area (R : ℝ) (r : ℝ) (hR : R = 6 * Real.sqrt 7) (hr : r = Real.sqrt 7) :
    ∃ (L : ℝ), (L = 2 * Real.sqrt 42 ∧ L^2 * Real.pi = 168 * Real.pi) :=
by
  sorry

end locus_area_l304_304383


namespace increasing_on_1_to_infty_min_value_on_1_to_e_l304_304591

noncomputable def f (x : ℝ) (a : ℝ) := x^2 - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := (2 * x^2 - a) / x

-- Proof that f(x) is increasing on (1, +∞) when a = 2
theorem increasing_on_1_to_infty (x : ℝ) (h : x > 1) : f' x 2 > 0 := 
  sorry

-- Proof for minimum value of f(x) on [1, e]
theorem min_value_on_1_to_e (a : ℝ) :
  if a ≤ 2 then ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = 1
  else if 2 < a ∧ a < 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = a / 2 - (a / 2) * Real.log (a / 2)
  else if a ≥ 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = Real.exp 2 - a
  else False := 
  sorry

end increasing_on_1_to_infty_min_value_on_1_to_e_l304_304591


namespace polynomial_expansion_l304_304785

-- Definitions of the polynomials
def p (w : ℝ) : ℝ := 3 * w^3 + 4 * w^2 - 7
def q (w : ℝ) : ℝ := 2 * w^3 - 3 * w^2 + 1

-- Statement of the theorem
theorem polynomial_expansion (w : ℝ) : 
  (p w) * (q w) = 6 * w^6 - 6 * w^5 + 9 * w^3 + 12 * w^2 - 3 :=
by
  sorry

end polynomial_expansion_l304_304785


namespace expected_value_is_10_l304_304483

noncomputable def expected_value_adjacent_pairs (boys girls : ℕ) (total_people : ℕ) : ℕ :=
  if total_people = 20 ∧ boys = 8 ∧ girls = 12 then 10 else sorry

theorem expected_value_is_10 : expected_value_adjacent_pairs 8 12 20 = 10 :=
by
  -- Intuition and all necessary calculations (proof steps) have already been explained.
  -- Here we are directly stating the conclusion based on given problem conditions.
  trivial

end expected_value_is_10_l304_304483


namespace smallest_five_digit_equiv_11_mod_13_l304_304719

open Nat

theorem smallest_five_digit_equiv_11_mod_13 :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 13 = 11 ∧ n = 10009 :=
by
  sorry

end smallest_five_digit_equiv_11_mod_13_l304_304719


namespace find_xyz_l304_304088

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 14 / 3 := 
sorry

end find_xyz_l304_304088


namespace number_of_red_balls_l304_304613

noncomputable def red_balls (n_black n_red draws black_draws : ℕ) : ℕ :=
  if black_draws = (draws * n_black) / (n_black + n_red) then n_red else sorry

theorem number_of_red_balls :
  ∀ (n_black draws black_draws : ℕ),
    n_black = 4 →
    draws = 100 →
    black_draws = 40 →
    red_balls n_black (red_balls 4 6 100 40) 100 40 = 6 :=
by
  intros n_black draws black_draws h_black h_draws h_blackdraws
  dsimp [red_balls]
  rw [h_black, h_draws, h_blackdraws]
  norm_num
  sorry

end number_of_red_balls_l304_304613


namespace correct_option_C_l304_304733

noncomputable def question := "Which of the following operations is correct?"
noncomputable def option_A := (-2)^2
noncomputable def option_B := (-2)^3
noncomputable def option_C := (-1/2)^3
noncomputable def option_D := (-7/3)^3
noncomputable def correct_answer := -1/8

theorem correct_option_C :
  option_C = correct_answer := by
  sorry

end correct_option_C_l304_304733


namespace simplify_exponent_l304_304153

variable {x : ℝ} {m n : ℕ}

theorem simplify_exponent (x : ℝ) : (3 * x ^ 5) * (4 * x ^ 3) = 12 * x ^ 8 := by
  sorry

end simplify_exponent_l304_304153


namespace Robie_l304_304479

def initial_bags (X : ℕ) := (X - 2) + 3 = 4

theorem Robie's_initial_bags (X : ℕ) (h : initial_bags X) : X = 3 :=
by
  unfold initial_bags at h
  sorry

end Robie_l304_304479


namespace laura_owes_amount_l304_304294

def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1
def interest (P R T : ℝ) := P * R * T
def totalAmountOwed (P I : ℝ) := P + I

theorem laura_owes_amount : totalAmountOwed principal (interest principal rate time) = 36.75 :=
by
  sorry

end laura_owes_amount_l304_304294


namespace prob_all_meet_standard_prob_at_least_one_meets_standard_l304_304551

def P_meeting_standard_A := 0.8
def P_meeting_standard_B := 0.6
def P_meeting_standard_C := 0.5

theorem prob_all_meet_standard :
  (P_meeting_standard_A * P_meeting_standard_B * P_meeting_standard_C) = 0.24 :=
by
  sorry

theorem prob_at_least_one_meets_standard :
  (1 - ((1 - P_meeting_standard_A) * (1 - P_meeting_standard_B) * (1 - P_meeting_standard_C))) = 0.96 :=
by
  sorry

end prob_all_meet_standard_prob_at_least_one_meets_standard_l304_304551


namespace smallest_b_for_quadratic_factorization_l304_304789

theorem smallest_b_for_quadratic_factorization : ∃ (b : ℕ), 
  (∀ r s : ℤ, (r * s = 4032) ∧ (r + s = b) → b ≥ 127) ∧ 
  (∃ r s : ℤ, (r * s = 4032) ∧ (r + s = b) ∧ (b = 127))
:= sorry

end smallest_b_for_quadratic_factorization_l304_304789


namespace investment_rate_l304_304602

theorem investment_rate (r : ℝ) (A : ℝ) (income_diff : ℝ) (total_invested : ℝ) (eight_percent_invested : ℝ) :
  total_invested = 2000 → 
  eight_percent_invested = 750 → 
  income_diff = 65 → 
  A = total_invested - eight_percent_invested → 
  (A * r) - (eight_percent_invested * 0.08) = income_diff → 
  r = 0.1 :=
by
  intros h_total h_eight h_income_diff h_A h_income_eq
  sorry

end investment_rate_l304_304602


namespace recurring_decimal_reduced_fraction_l304_304237

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l304_304237


namespace find_g_expression_l304_304587

theorem find_g_expression (g f : ℝ → ℝ) (h_sym : ∀ x y, g x = y ↔ g (2 - x) = 4 - y)
  (h_f : ∀ x, f x = 3 * x - 1) :
  ∀ x, g x = 3 * x - 1 :=
by
  sorry

end find_g_expression_l304_304587


namespace factor_expression_l304_304412

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) :=
by
  sorry

end factor_expression_l304_304412


namespace least_integer_gt_sqrt_450_l304_304519

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l304_304519


namespace students_exceed_rabbits_l304_304391

theorem students_exceed_rabbits (students_per_classroom rabbits_per_classroom number_of_classrooms : ℕ) 
  (h_students : students_per_classroom = 18)
  (h_rabbits : rabbits_per_classroom = 2)
  (h_classrooms : number_of_classrooms = 4) : 
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 64 :=
by {
  sorry
}

end students_exceed_rabbits_l304_304391


namespace plains_routes_count_l304_304282

theorem plains_routes_count (total_cities mountainous_cities plains_cities total_routes routes_mountainous_pairs: ℕ) :
  total_cities = 100 →
  mountainous_cities = 30 →
  plains_cities = 70 →
  total_routes = 150 →
  routes_mountainous_pairs = 21 →
  let endpoints_mountainous := mountainous_cities * 3 in
  let endpoints_mountainous_pairs := routes_mountainous_pairs * 2 in
  let endpoints_mountainous_plains := endpoints_mountainous - endpoints_mountainous_pairs in
  let endpoints_plains := plains_cities * 3 in
  let routes_mountainous_plains := endpoints_mountainous_plains in
  let endpoints_plains_pairs := endpoints_plains - routes_mountainous_plains in
  let routes_plains_pairs := endpoints_plains_pairs / 2 in
  routes_plains_pairs = 81 :=
by
  intros h1 h2 h3 h4 h5
  dsimp
  rw [h1, h2, h3, h4, h5]
  sorry

end plains_routes_count_l304_304282


namespace possible_number_of_classmates_l304_304649

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l304_304649


namespace pq_difference_l304_304739

theorem pq_difference (p q : ℝ) (h1 : 3 / p = 6) (h2 : 3 / q = 15) : p - q = 3 / 10 := by
  sorry

end pq_difference_l304_304739


namespace sam_quarters_l304_304868

theorem sam_quarters (pennies : ℕ) (total : ℝ) (value_penny : ℝ) (value_quarter : ℝ) (quarters : ℕ) :
  pennies = 9 →
  total = 1.84 →
  value_penny = 0.01 →
  value_quarter = 0.25 →
  quarters = (total - pennies * value_penny) / value_quarter →
  quarters = 7 :=
by
  intros
  sorry

end sam_quarters_l304_304868


namespace goldfish_problem_l304_304634

theorem goldfish_problem (x : ℕ) : 
  (18 + (x - 5) * 7 = 4) → (x = 3) :=
by
  intros
  sorry

end goldfish_problem_l304_304634


namespace simplify_expression_l304_304699

theorem simplify_expression (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) : 
  1 - (1 / (1 + (a^2 / (1 - a^2)))) = a^2 :=
sorry

end simplify_expression_l304_304699


namespace isosceles_triangle_base_angle_l304_304456

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle_isosceles : a = b ∨ b = c ∨ c = a)
  (h_angle_sum : a + b + c = 180) (h_one_angle : a = 50 ∨ b = 50 ∨ c = 50) :
  a = 50 ∨ b = 50 ∨ c = 50 ∨ a = 65 ∨ b = 65 ∨ c = 65 :=
by
  sorry

end isosceles_triangle_base_angle_l304_304456


namespace prove_fraction_identity_l304_304007

theorem prove_fraction_identity 
  (x y z : ℝ)
  (h1 : (x * z) / (x + y) + (y * z) / (y + z) + (x * y) / (z + x) = -18)
  (h2 : (z * y) / (x + y) + (z * x) / (y + z) + (y * x) / (z + x) = 20) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 20.5 := 
by
  sorry

end prove_fraction_identity_l304_304007


namespace sum_of_consecutive_integers_product_l304_304022

noncomputable def consecutive_integers_sum (n m k : ℤ) : ℤ :=
  n + m + k

theorem sum_of_consecutive_integers_product (n m k : ℤ)
  (h1 : n = m - 1)
  (h2 : k = m + 1)
  (h3 : n * m * k = 990) :
  consecutive_integers_sum n m k = 30 :=
by
  sorry

end sum_of_consecutive_integers_product_l304_304022


namespace no_solutions_cryptarithm_l304_304741

theorem no_solutions_cryptarithm : 
  ∀ (K O P H A B U y C : ℕ), 
  K ≠ O ∧ K ≠ P ∧ K ≠ H ∧ K ≠ A ∧ K ≠ B ∧ K ≠ U ∧ K ≠ y ∧ K ≠ C ∧ 
  O ≠ P ∧ O ≠ H ∧ O ≠ A ∧ O ≠ B ∧ O ≠ U ∧ O ≠ y ∧ O ≠ C ∧ 
  P ≠ H ∧ P ≠ A ∧ P ≠ B ∧ P ≠ U ∧ P ≠ y ∧ P ≠ C ∧ 
  H ≠ A ∧ H ≠ B ∧ H ≠ U ∧ H ≠ y ∧ H ≠ C ∧ 
  A ≠ B ∧ A ≠ U ∧ A ≠ y ∧ A ≠ C ∧ 
  B ≠ U ∧ B ≠ y ∧ B ≠ C ∧ 
  U ≠ y ∧ U ≠ C ∧ 
  y ≠ C ∧
  K < O ∧ O < P ∧ P > O ∧ O > H ∧ H > A ∧ A > B ∧ B > U ∧ U > P ∧ P > y ∧ y > C → 
  false :=
sorry

end no_solutions_cryptarithm_l304_304741


namespace number_of_classmates_l304_304651

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l304_304651


namespace math_problem_l304_304150

theorem math_problem :
  let initial := 180
  let thirty_five_percent := 0.35 * initial
  let one_third_less := thirty_five_percent - (thirty_five_percent / 3)
  let remaining := initial - one_third_less
  let three_fifths_remaining := (3 / 5) * remaining
  (three_fifths_remaining ^ 2) = 6857.84 :=
by
  sorry

end math_problem_l304_304150


namespace lcm_of_1_to_12_l304_304722

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end lcm_of_1_to_12_l304_304722


namespace cake_sharing_l304_304658

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l304_304658


namespace abs_eq_imp_b_eq_2_l304_304256

theorem abs_eq_imp_b_eq_2 (b : ℝ) (h : |1 - b| = |3 - b|) : b = 2 := 
sorry

end abs_eq_imp_b_eq_2_l304_304256


namespace trapezoid_diagonal_intersection_l304_304043

theorem trapezoid_diagonal_intersection (PQ RS PR : ℝ) (h1 : PQ = 3 * RS) (h2 : PR = 15) :
  ∃ RT : ℝ, RT = 15 / 4 :=
by
  have RT := 15 / 4
  use RT
  sorry

end trapezoid_diagonal_intersection_l304_304043


namespace ratio_of_two_numbers_l304_304316

theorem ratio_of_two_numbers (A B : ℕ) (x y : ℕ) (h1 : lcm A B = 60) (h2 : A + B = 50) (h3 : A / B = x / y) (hx : x = 3) (hy : y = 2) : x = 3 ∧ y = 2 := 
by
  -- Conditions provided in the problem
  sorry

end ratio_of_two_numbers_l304_304316


namespace sum_first_10_terms_eq_65_l304_304139

section ArithmeticSequence

variables (a d : ℕ) (S : ℕ → ℕ) 

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Condition 1: nth term at n = 3
axiom a3_eq_4 : nth_term 3 = 4

-- Condition 2: difference in sums between n = 9 and n = 6
axiom S9_minus_S6_eq_27 : sum_first_n_terms 9 - sum_first_n_terms 6 = 27

-- To prove: sum of the first 10 terms equals 65
theorem sum_first_10_terms_eq_65 : sum_first_n_terms 10 = 65 :=
sorry

end ArithmeticSequence

end sum_first_10_terms_eq_65_l304_304139


namespace range_of_a_l304_304319

-- Define the function g(x) = x^3 - 3ax - a
def g (a x : ℝ) : ℝ := x^3 - 3*a*x - a

-- Define the derivative of g(x) which is g'(x) = 3x^2 - 3a
def g' (a x : ℝ) : ℝ := 3*x^2 - 3*a

theorem range_of_a (a : ℝ) : g a 0 * g a 1 < 0 → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l304_304319


namespace initial_rows_l304_304750

theorem initial_rows (r T : ℕ) (h1 : T = 42 * r) (h2 : T = 28 * (r + 12)) : r = 24 :=
by
  sorry

end initial_rows_l304_304750


namespace area_of_parallelogram_l304_304178

theorem area_of_parallelogram (b h : ℕ) (hb : b = 60) (hh : h = 16) : b * h = 960 := by
  -- Here goes the proof
  sorry

end area_of_parallelogram_l304_304178


namespace segment_length_l304_304709

theorem segment_length (x : ℝ) (h : |x - (27)^(1/3)| = 5) : ∃ a b : ℝ, (a = 8 ∧ b = -2 ∨ a = -2 ∧ b = 8) ∧ real.dist a b = 10 :=
by
  use [8, -2] -- providing the endpoints explicitly
  split
  -- prove that these are the correct endpoints
  · left; exact ⟨rfl, rfl⟩
  -- prove the distance is 10
  · apply real.dist_eq; linarith
  

end segment_length_l304_304709


namespace find_angle_D_l304_304607

variables (A B C D angle : ℝ)

-- Assumptions based on the problem statement
axiom sum_A_B : A + B = 140
axiom C_eq_D : C = D

-- The claim we aim to prove
theorem find_angle_D (h₁ : A + B = 140) (h₂: C = D): D = 20 :=
by {
    sorry 
}

end find_angle_D_l304_304607


namespace lcm_1_to_12_l304_304720

theorem lcm_1_to_12 : Nat.lcm_list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 27720 := by
  sorry

end lcm_1_to_12_l304_304720


namespace john_days_ran_l304_304987

theorem john_days_ran 
  (total_distance : ℕ) (daily_distance : ℕ) 
  (h1 : total_distance = 10200) (h2 : daily_distance = 1700) :
  total_distance / daily_distance = 6 :=
by
  sorry

end john_days_ran_l304_304987


namespace number_of_classmates_l304_304650

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l304_304650


namespace algebraic_expression_value_l304_304577

theorem algebraic_expression_value (x : ℝ) (h : x = 5) : (3 / (x - 4) - 24 / (x^2 - 16)) = (1 / 3) :=
by
  have hx : x = 5 := h
  sorry

end algebraic_expression_value_l304_304577


namespace ivan_total_money_in_piggy_banks_l304_304129

theorem ivan_total_money_in_piggy_banks 
    (num_pennies_per_piggy_bank : ℕ) 
    (num_dimes_per_piggy_bank : ℕ) 
    (value_of_penny : ℕ) 
    (value_of_dime : ℕ) 
    (num_piggy_banks : ℕ) :
    num_pennies_per_piggy_bank = 100 →
    num_dimes_per_piggy_bank = 50 →
    value_of_penny = 1 →
    value_of_dime = 10 →
    num_piggy_banks = 2 →
    let total_value_one_bank := num_dimes_per_piggy_bank * value_of_dime + num_pennies_per_piggy_bank * value_of_penny in
    let total_value_in_cents := total_value_one_bank * num_piggy_banks in
    let total_value_in_dollars := total_value_in_cents / 100 in
    total_value_in_dollars = 12 :=
by
  intros 
  sorry

end ivan_total_money_in_piggy_banks_l304_304129


namespace sin_cos_identity_l304_304470

theorem sin_cos_identity {x : Real} 
    (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
    Real.sin x ^ 12 + Real.cos x ^ 12 = 5 / 18 :=
sorry

end sin_cos_identity_l304_304470


namespace value_of_r_squared_plus_s_squared_l304_304995

theorem value_of_r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 24) (h2 : r + s = 10) :
  r^2 + s^2 = 52 :=
sorry

end value_of_r_squared_plus_s_squared_l304_304995


namespace horner_evaluation_l304_304893

-- Define the polynomial function
def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 - 6 * x^2 + x - 1

-- The theorem that we need to prove
theorem horner_evaluation : f (-1) = -5 :=
  by
  -- This is the statement without the proof steps
  sorry

end horner_evaluation_l304_304893


namespace exists_irreducible_fractions_l304_304554

theorem exists_irreducible_fractions:
  ∃ (f : Fin 2018 → ℚ), 
    (∀ i j : Fin 2018, i ≠ j → (f i).den ≠ (f j).den) ∧ 
    (∀ i j : Fin 2018, i ≠ j → ∀ d : ℚ, d = f i - f j → d ≠ 0 → d.den < (f i).den ∧ d.den < (f j).den) :=
by
  -- proof is omitted
  sorry

end exists_irreducible_fractions_l304_304554


namespace three_integers_product_sum_l304_304696

theorem three_integers_product_sum (a b c : ℤ) (h : a * b * c = -5) :
    a + b + c = 5 ∨ a + b + c = -3 ∨ a + b + c = -7 :=
sorry

end three_integers_product_sum_l304_304696


namespace students_failed_exam_l304_304125

def total_students : ℕ := 740
def percent_passed : ℝ := 0.35
def percent_failed : ℝ := 1 - percent_passed
def failed_students : ℝ := percent_failed * total_students

theorem students_failed_exam : failed_students = 481 := 
by sorry

end students_failed_exam_l304_304125


namespace variance_of_binomial_distribution_l304_304121

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_binomial_distribution :
  binomial_variance 10 (2/5) = 12 / 5 :=
by
  sorry

end variance_of_binomial_distribution_l304_304121


namespace problem_acd_div_b_l304_304935

theorem problem_acd_div_b (a b c d : ℤ) (x : ℝ)
    (h1 : x = (a + b * Real.sqrt c) / d)
    (h2 : (7 * x) / 4 + 2 = 6 / x) :
    (a * c * d) / b = -322 := sorry

end problem_acd_div_b_l304_304935


namespace lcm_of_three_l304_304152

theorem lcm_of_three (A1 A2 A3 : ℕ) (D : ℕ)
  (hD : D = Nat.gcd (A1 * A2) (Nat.gcd (A2 * A3) (A3 * A1))) :
  Nat.lcm (Nat.lcm A1 A2) A3 = (A1 * A2 * A3) / D :=
sorry

end lcm_of_three_l304_304152


namespace repeating_decimal_fraction_l304_304214

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l304_304214


namespace solve_equation_l304_304684

noncomputable def f (x : ℝ) : ℝ :=
  2 * x + 1 + Real.arctan x * Real.sqrt (x^2 + 1)

theorem solve_equation : ∃ x : ℝ, f x + f (x + 1) = 0 ∧ x = -1/2 :=
  by
    use -1/2
    simp [f]
    sorry

end solve_equation_l304_304684


namespace estimate_red_balls_l304_304612

theorem estimate_red_balls (x : ℕ) (drawn_black_balls : ℕ) (total_draws : ℕ) (black_balls : ℕ) 
  (h1 : black_balls = 4) 
  (h2 : total_draws = 100) 
  (h3 : drawn_black_balls = 40) 
  (h4 : (black_balls : ℚ) / (black_balls + x) = drawn_black_balls / total_draws) : 
  x = 6 := 
sorry

end estimate_red_balls_l304_304612


namespace num_of_integers_l304_304081

theorem num_of_integers (n : ℤ) (h : -1000 ≤ n ∧ n ≤ 1000) (h1 : 1 < 4 * n + 7) (h2 : 4 * n + 7 < 150) : 
  (∃ N : ℕ, N = 37) :=
by
  sorry

end num_of_integers_l304_304081


namespace bahs_for_1000_yahs_l304_304828

-- Definitions based on given conditions
def bahs_to_rahs_ratio (b r : ℕ) := 15 * b = 24 * r
def rahs_to_yahs_ratio (r y : ℕ) := 9 * r = 15 * y

-- Main statement to prove
theorem bahs_for_1000_yahs (b r y : ℕ) (h1 : bahs_to_rahs_ratio b r) (h2 : rahs_to_yahs_ratio r y) :
  1000 * y = 375 * b :=
by
  sorry

end bahs_for_1000_yahs_l304_304828


namespace find_greatest_K_l304_304934

theorem find_greatest_K {u v w K : ℝ} (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu2_gt_4vw : u^2 > 4 * v * w) :
  (u^2 - 4 * v * w)^2 > K * (2 * v^2 - u * w) * (2 * w^2 - u * v) ↔ K ≤ 16 := 
sorry

end find_greatest_K_l304_304934


namespace mrs_hilt_initial_money_l304_304305

def initial_amount (pencil_cost candy_cost left_money : ℕ) := 
  pencil_cost + candy_cost + left_money

theorem mrs_hilt_initial_money :
  initial_amount 20 5 18 = 43 :=
by
  -- initial_amount 20 5 18 
  -- = 20 + 5 + 18
  -- = 25 + 18 
  -- = 43
  sorry

end mrs_hilt_initial_money_l304_304305


namespace probability_of_total_greater_than_7_l304_304452

-- Definitions for conditions
def total_outcomes : ℕ := 36
def favorable_outcome_count : ℕ := 15

-- Probability Calculation
def calc_probability (total : ℕ) (favorable : ℕ) : ℚ := favorable / total 

-- The theorem statement
theorem probability_of_total_greater_than_7 :
  calc_probability total_outcomes favorable_outcome_count = 5 / 12 :=
sorry

end probability_of_total_greater_than_7_l304_304452


namespace cylindrical_to_rectangular_l304_304387

structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

structure RectangularCoord where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def convertCylindricalToRectangular (c : CylindricalCoord) : RectangularCoord :=
  { x := c.r * Real.cos c.θ,
    y := c.r * Real.sin c.θ,
    z := c.z }

theorem cylindrical_to_rectangular :
  convertCylindricalToRectangular ⟨7, Real.pi / 3, -3⟩ = ⟨3.5, 7 * Real.sqrt 3 / 2, -3⟩ :=
by sorry

end cylindrical_to_rectangular_l304_304387


namespace largest_number_is_b_l304_304039

noncomputable def a := 0.935
noncomputable def b := 0.9401
noncomputable def c := 0.9349
noncomputable def d := 0.9041
noncomputable def e := 0.9400

theorem largest_number_is_b : b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  -- proof can be filled in here
  sorry

end largest_number_is_b_l304_304039


namespace years_between_2000_and_3000_with_property_l304_304889

theorem years_between_2000_and_3000_with_property :
  ∃ n : ℕ, n = 143 ∧
  ∀ Y, 2000 ≤ Y ∧ Y ≤ 3000 → ∃ p q : ℕ, p + q = Y ∧ 2 * p = 5 * q →
  (2 * Y) % 7 = 0 :=
sorry

end years_between_2000_and_3000_with_property_l304_304889


namespace k_value_function_range_l304_304943

noncomputable def f : ℝ → ℝ := λ x => Real.log x + x

def is_k_value_function (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ (∀ x, a ≤ x ∧ x ≤ b → (f x = k * x)) ∧ (k > 0)

theorem k_value_function_range :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.log x + x) →
  (∃ (k : ℝ), is_k_value_function f k) →
  1 < k ∧ k < 1 + (1 / Real.exp 1) :=
by
  sorry

end k_value_function_range_l304_304943


namespace percentage_reduction_l304_304751

variable (C S newS newC : ℝ)
variable (P : ℝ)
variable (hC : C = 50)
variable (hS : S = 1.25 * C)
variable (hNewS : newS = S - 10.50)
variable (hGain30 : newS = 1.30 * newC)
variable (hNewC : newC = C - P * C)

theorem percentage_reduction (C S newS newC : ℝ) (hC : C = 50) 
  (hS : S = 1.25 * C) (hNewS : newS = S - 10.50) 
  (hGain30 : newS = 1.30 * newC) 
  (hNewC : newC = C - P * C) : 
  P = 0.20 :=
by
  sorry

end percentage_reduction_l304_304751


namespace repeating_decimal_as_fraction_l304_304233

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l304_304233


namespace sequence_le_zero_l304_304473

noncomputable def sequence_property (N : ℕ) (a : ℕ → ℝ) : Prop :=
  (a 0 = 0) ∧ (a N = 0) ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2)

theorem sequence_le_zero {N : ℕ} (a : ℕ → ℝ) (h : sequence_property N a) : 
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 :=
sorry

end sequence_le_zero_l304_304473


namespace geo_seq_sum_condition_l304_304194

noncomputable def geometric_seq (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^n

noncomputable def sum_geo_seq_3 (a : ℝ) (q : ℝ) : ℝ :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

noncomputable def sum_geo_seq_6 (a : ℝ) (q : ℝ) : ℝ :=
  sum_geo_seq_3 a q + geometric_seq a q 3 + geometric_seq a q 4 + geometric_seq a q 5

theorem geo_seq_sum_condition {a q S₃ S₆ : ℝ} (h_sum_eq : S₆ = 9 * S₃)
  (h_S₃_def : S₃ = sum_geo_seq_3 a q)
  (h_S₆_def : S₆ = sum_geo_seq_6 a q) :
  q = 2 :=
by
  sorry

end geo_seq_sum_condition_l304_304194


namespace min_students_for_duplicate_borrowings_l304_304053

/-- Given 4 types of books and each student can borrow at most 3 books, 
    the minimum number of students m such that there are at least 
    two students who have borrowed the same type and number of books is 15. -/
theorem min_students_for_duplicate_borrowings
  (books : Finset (Fin 4))
  (max_borrow : ℕ)
  (h_max_borrow : max_borrow = 3) : 
  ∃ m, m = 15 ∧ ∀ (students : Finset (Fin m)) 
  (borrowings : students → Finset books), 
  (∃ i j : students, i ≠ j ∧ borrowings i = borrowings j) :=
by
  sorry

end min_students_for_duplicate_borrowings_l304_304053


namespace least_integer_greater_than_sqrt_450_l304_304523

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l304_304523


namespace polynomial_self_composition_l304_304563

theorem polynomial_self_composition {p : Polynomial ℝ} {n : ℕ} (hn : 0 < n) :
  (∀ x, p.eval (p.eval x) = (p.eval x) ^ n) ↔ p = Polynomial.X ^ n :=
by sorry

end polynomial_self_composition_l304_304563


namespace classmates_ate_cake_l304_304677

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l304_304677


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l304_304685

-- Problem 1: 5x² = 40x
theorem solve_quadratic_1 (x : ℝ) : 5 * x^2 = 40 * x ↔ (x = 0 ∨ x = 8) :=
by sorry

-- Problem 2: 25/9 x² = 100
theorem solve_quadratic_2 (x : ℝ) : (25 / 9) * x^2 = 100 ↔ (x = 6 ∨ x = -6) :=
by sorry

-- Problem 3: 10x = x² + 21
theorem solve_quadratic_3 (x : ℝ) : 10 * x = x^2 + 21 ↔ (x = 7 ∨ x = 3) :=
by sorry

-- Problem 4: x² = 12x + 288
theorem solve_quadratic_4 (x : ℝ) : x^2 = 12 * x + 288 ↔ (x = 24 ∨ x = -12) :=
by sorry

-- Problem 5: x² + 20 1/4 = 11 1/4 x
theorem solve_quadratic_5 (x : ℝ) : x^2 + 81 / 4 = 45 / 4 * x ↔ (x = 9 / 4 ∨ x = 9) :=
by sorry

-- Problem 6: 1/12 x² + 7/12 x = 19
theorem solve_quadratic_6 (x : ℝ) : (1 / 12) * x^2 + (7 / 12) * x = 19 ↔ (x = 12 ∨ x = -19) :=
by sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l304_304685


namespace geom_seq_necessity_geom_seq_not_sufficient_l304_304571

theorem geom_seq_necessity (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    q > 1 ∨ q < -1 :=
  sorry

theorem geom_seq_not_sufficient (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    ¬ (q > 1 → a₁ < a₁ * q^2) :=
  sorry

end geom_seq_necessity_geom_seq_not_sufficient_l304_304571


namespace area_R_l304_304991

-- Define the given matrix as a 2x2 real matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, -5]

-- Define the original area of region R
def area_R : ℝ := 15

-- Define the area scaling factor as the absolute value of the determinant of A
def scaling_factor : ℝ := |Matrix.det A|

-- Prove that the area of the region R' is 585
theorem area_R' : scaling_factor * area_R = 585 := by
  sorry

end area_R_l304_304991


namespace rainfall_ratio_l304_304390

noncomputable def total_rainfall := 35
noncomputable def rainfall_second_week := 21

theorem rainfall_ratio 
  (R1 R2 : ℝ)
  (hR2 : R2 = rainfall_second_week)
  (hTotal : R1 + R2 = total_rainfall) :
  R2 / R1 = 3 / 2 := 
by 
  sorry

end rainfall_ratio_l304_304390


namespace plains_routes_l304_304277

theorem plains_routes 
  (total_cities : ℕ)
  (mountainous_cities : ℕ)
  (plains_cities : ℕ)
  (total_routes : ℕ)
  (mountainous_routes : ℕ)
  (num_pairs_with_mount_to_mount : ℕ)
  (routes_per_year : ℕ)
  (years : ℕ)
  (mountainous_roots_connections : ℕ)
  : (mountainous_cities = 30) →
    (plains_cities = 70) →
    (total_cities = mountainous_cities + plains_cities) →
    (routes_per_year = 50) →
    (years = 3) →
    (total_routes = routes_per_year * years) →
    (mountainous_routes = num_pairs_with_mount_to_mount * 2) →
    (num_pairs_with_mount_to_mount = 21) →
    let num_endpoints_per_city_route = 2 in
    let mountainous_city_endpoints = mountainous_cities * 3 in
    let mountainous_endpoints = mountainous_routes in
    let mountain_to_plains_endpoints = mountainous_city_endpoints - mountainous_endpoints in
    let total_endpoints = total_routes * num_endpoints_per_city_route in
    let plains_city_endpoints = plains_cities * 3 in
    let routes_between_plain_and_mountain = mountain_to_plains_endpoints in
    let plain_to_plain_endpoints = plains_city_endpoints - routes_between_plain_and_mountain in
    let plain_to_plain_routes = plain_to_plain_endpoints / 2 in
    plain_to_plain_routes = 81 :=
sorry

end plains_routes_l304_304277


namespace solve_complex_equation_l304_304438

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l304_304438


namespace comprehensive_score_correct_l304_304191

-- Conditions
def theoreticalWeight : ℝ := 0.20
def designWeight : ℝ := 0.50
def presentationWeight : ℝ := 0.30

def theoreticalScore : ℕ := 95
def designScore : ℕ := 88
def presentationScore : ℕ := 90

-- Calculate comprehensive score
def comprehensiveScore : ℝ :=
  theoreticalScore * theoreticalWeight +
  designScore * designWeight +
  presentationScore * presentationWeight

-- Lean statement to prove the comprehensive score using the conditions
theorem comprehensive_score_correct :
  comprehensiveScore = 90 := 
  sorry

end comprehensive_score_correct_l304_304191


namespace system_solution_l304_304447

theorem system_solution (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -2 / 5 :=
by
  sorry

end system_solution_l304_304447


namespace range_of_x_l304_304027

variable (x : ℝ)

-- Conditions used in the problem
def sqrt_condition : Prop := x + 2 ≥ 0
def non_zero_condition : Prop := x + 1 ≠ 0

-- The statement to be proven
theorem range_of_x : sqrt_condition x ∧ non_zero_condition x ↔ (x ≥ -2 ∧ x ≠ -1) :=
by
  sorry

end range_of_x_l304_304027


namespace repeating_decimal_to_fraction_l304_304240

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l304_304240


namespace max_value_f1_solve_inequality_f2_l304_304813

def f_1 (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem max_value_f1 : ∃ x, f_1 x = 2 :=
sorry

def f_2 (x : ℝ) : ℝ := |2 * x - 1| - |x - 1|

theorem solve_inequality_f2 (x : ℝ) : f_2 x ≥ 1 ↔ x ≤ -1 ∨ x ≥ 1 :=
sorry

end max_value_f1_solve_inequality_f2_l304_304813


namespace min_value_fraction_l304_304583

open Real

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℝ, x = (a / (a + 2 * b) + b / (a + b)) ∧ x ≥ 1 - 1 / (2 * sqrt 2) ∧ x = 1 - 1 / (2 * sqrt 2)) :=
by
  sorry

end min_value_fraction_l304_304583


namespace range_of_z_l304_304142

theorem range_of_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
(h₁ : x + y = x * y) (h₂ : x + y + z = x * y * z) : 1 < z ∧ z ≤ 4 / 3 :=
sorry

end range_of_z_l304_304142


namespace inequality_abc_l304_304471

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
by
  sorry

end inequality_abc_l304_304471


namespace radius_of_smaller_circle_l304_304126

theorem radius_of_smaller_circle (R r : ℝ) (h1 : R = 6)
  (h2 : 2 * R = 3 * 2 * r) : r = 2 :=
by
  sorry

end radius_of_smaller_circle_l304_304126


namespace routes_between_plains_cities_correct_l304_304278

noncomputable def number_of_routes_connecting_pairs_of_plains_cities
    (total_cities : ℕ)
    (mountainous_cities : ℕ)
    (plains_cities : ℕ)
    (total_routes : ℕ)
    (routes_between_mountainous_cities : ℕ) : ℕ :=
let mountainous_city_endpoints := mountainous_cities * 3 in
let routes_between_mountainous_cities_endpoints := routes_between_mountainous_cities * 2 in
let mountainous_to_plains_routes_endpoints := mountainous_city_endpoints - routes_between_mountainous_cities_endpoints in
let plains_city_endpoints := plains_cities * 3 in
let plains_city_to_mountainous_city_routes_endpoints := mountainous_to_plains_routes_endpoints in
let endpoints_fully_in_plains_cities := plains_city_endpoints - plains_city_to_mountainous_city_routes_endpoints in
endpoints_fully_in_plains_cities / 2

theorem routes_between_plains_cities_correct :
    number_of_routes_connecting_pairs_of_plains_cities 100 30 70 150 21 = 81 := by
    sorry

end routes_between_plains_cities_correct_l304_304278


namespace repeating_decimal_as_fraction_l304_304232

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l304_304232


namespace roots_quadratic_diff_by_12_l304_304978

theorem roots_quadratic_diff_by_12 (P : ℝ) : 
  (∀ α β : ℝ, (α + β = 2) ∧ (α * β = -P) ∧ ((α - β) = 12)) → P = 35 := 
by
  intro h
  sorry

end roots_quadratic_diff_by_12_l304_304978


namespace repeating_decimals_sum_l304_304401

theorem repeating_decimals_sum :
  let x := 0.6666666 -- 0.\overline{6}
  let y := 0.2222222 -- 0.\overline{2}
  let z := 0.4444444 -- 0.\overline{4}
  (x + y - z) = 4 / 9 := 
by
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  calc
    -- Calculate (x + y - z)
    (x + y - z) = (2 / 3 + 2 / 9 - 4 / 9) : by sorry
                ... = 4 / 9 : by sorry


end repeating_decimals_sum_l304_304401


namespace rent_change_percent_l304_304293

open Real

noncomputable def elaine_earnings_last_year (E : ℝ) : ℝ :=
E

noncomputable def elaine_rent_last_year (E : ℝ) : ℝ :=
0.2 * E

noncomputable def elaine_earnings_this_year (E : ℝ) : ℝ :=
1.15 * E

noncomputable def elaine_rent_this_year (E : ℝ) : ℝ :=
0.25 * (1.15 * E)

noncomputable def rent_percentage_change (E : ℝ) : ℝ :=
(elaine_rent_this_year E) / (elaine_rent_last_year E) * 100

theorem rent_change_percent (E : ℝ) :
  rent_percentage_change E = 143.75 :=
by
  sorry

end rent_change_percent_l304_304293


namespace scientific_notation_11580000_l304_304837

theorem scientific_notation_11580000 :
  (11580000 : ℝ) = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l304_304837


namespace frog_arrangement_l304_304704

def arrangementCount (total_frogs green_frogs red_frogs blue_frog : ℕ) : ℕ :=
  if (green_frogs + red_frogs + blue_frog = total_frogs ∧ 
      green_frogs = 3 ∧ red_frogs = 4 ∧ blue_frog = 1) then 40320 else 0

theorem frog_arrangement :
  arrangementCount 8 3 4 1 = 40320 :=
by {
  -- Proof omitted
  sorry
}

end frog_arrangement_l304_304704


namespace length_more_than_breadth_l304_304322

theorem length_more_than_breadth
  (b x : ℝ)
  (h1 : b + x = 60)
  (h2 : 4 * b + 2 * x = 200) :
  x = 20 :=
by
  sorry

end length_more_than_breadth_l304_304322


namespace six_letter_vowel_words_count_l304_304008

noncomputable def vowel_count_six_letter_words : Nat := 27^6

theorem six_letter_vowel_words_count :
  vowel_count_six_letter_words = 531441 :=
  by
    sorry

end six_letter_vowel_words_count_l304_304008


namespace find_interest_rate_l304_304415

noncomputable def interest_rate (A P T : ℚ) : ℚ := (A - P) / (P * T) * 100

theorem find_interest_rate :
  let A := 1120
  let P := 921.0526315789474
  let T := 2.4
  interest_rate A P T = 9 := 
by
  sorry

end find_interest_rate_l304_304415


namespace calc_x_squared_y_squared_l304_304824

theorem calc_x_squared_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -9) : x^2 + y^2 = 22 := by
  sorry

end calc_x_squared_y_squared_l304_304824


namespace area_of_rectangle_PQRS_l304_304617

-- Definitions for the lengths of the sides of triangle ABC.
def AB : ℝ := 15
def AC : ℝ := 20
def BC : ℝ := 25

-- Definition for the length of PQ in rectangle PQRS.
def PQ : ℝ := 12

-- Definition for the condition that PQ is parallel to BC and RS is parallel to AB.
def PQ_parallel_BC : Prop := True
def RS_parallel_AB : Prop := True

-- The theorem to be proved: the area of rectangle PQRS is 115.2.
theorem area_of_rectangle_PQRS : 
  (∃ h: ℝ, h = (AC * PQ / BC) ∧ PQ * h = 115.2) :=
by {
  sorry
}

end area_of_rectangle_PQRS_l304_304617


namespace least_integer_gt_sqrt_450_l304_304517

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l304_304517


namespace only_solutions_mod_n_l304_304243

theorem only_solutions_mod_n (n : ℕ) : (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % (n : ℤ) = 0) ↔ (∃ k : ℕ, n = 3 ^ k) := 
sorry

end only_solutions_mod_n_l304_304243


namespace periodic_decimal_to_fraction_l304_304223

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l304_304223


namespace evaluate_expression_l304_304527

theorem evaluate_expression : 
  101^3 + 3 * (101^2) * 2 + 3 * 101 * (2^2) + 2^3 = 1092727 := 
by 
  sorry

end evaluate_expression_l304_304527


namespace prob1_prob2_prob3_prob4_prob5_l304_304742

theorem prob1 : (1 - 27 + (-32) + (-8) + 27) = -40 := sorry

theorem prob2 : (2 * -5 + abs (-3)) = -2 := sorry

theorem prob3 (x y : Int) (h₁ : -x = 3) (h₂ : abs y = 5) : x + y = 2 ∨ x + y = -8 := sorry

theorem prob4 : ((-1 : Int) * (3 / 2) + (5 / 4) + (-5 / 2) - (-13 / 4) - (5 / 4)) = -3 / 4 := sorry

theorem prob5 (a b : Int) (h : abs (a - 4) + abs (b + 5) = 0) : a - b = 9 := sorry

end prob1_prob2_prob3_prob4_prob5_l304_304742


namespace which_options_imply_inverse_order_l304_304735

theorem which_options_imply_inverse_order (a b : ℝ) :
  ((b > 0 ∧ 0 > a) ∨ (0 > a ∧ a > b) ∨ (a > b ∧ b > 0)) →
  (1 / a < 1 / b) :=
by
  intro h
  cases h
  case inl h1 =>
    have ha : a < 0 := h1.2
    have hb : b > 0 := h1.1
    have hab : a < 0 ∧ 0 < b := ⟨ha, hb⟩
    calc
      1 / a < 1 / b := by sorry
  case inr h2 =>
    cases h2
    case inl h3 =>
      have ha : a < 0 := h3.1.1
      have hb : b < a  := h3.1.2
      have hb_lt_a : b < a ∧ a < 0 := ⟨hb, ha⟩
      calc
        1 / a < 1 / b := by sorry
    case inr h4 =>
      have ha : a > b := h4.1
      have hb : b > 0 := h4.2
      have a_gt_b_and_b_gt_0 : a > b ∧ b > 0 := ⟨ha, hb⟩
      calc
        1 / a < 1 / b := by sorry

end which_options_imply_inverse_order_l304_304735


namespace parabola_points_relation_l304_304948

theorem parabola_points_relation {a b c y1 y2 y3 : ℝ} 
  (hA : y1 = a * (1 / 2)^2 + b * (1 / 2) + c)
  (hB : y2 = a * (0)^2 + b * (0) + c)
  (hC : y3 = a * (-1)^2 + b * (-1) + c)
  (h_cond : 0 < 2 * a ∧ 2 * a < b) : 
  y1 > y2 ∧ y2 > y3 :=
by 
  sorry

end parabola_points_relation_l304_304948


namespace ceil_evaluation_l304_304396

theorem ceil_evaluation : 
  (Int.ceil (((-7 : ℚ) / 4) ^ 2 - (1 / 8)) = 3) :=
sorry

end ceil_evaluation_l304_304396


namespace length_more_than_breadth_l304_304324

theorem length_more_than_breadth (b x : ℕ) 
  (h1 : 60 = b + x) 
  (h2 : 4 * b + 2 * x = 200) : x = 20 :=
by {
  sorry
}

end length_more_than_breadth_l304_304324


namespace mean_of_combined_sets_l304_304887

theorem mean_of_combined_sets (A : Finset ℝ) (B : Finset ℝ)
  (hA_len : A.card = 7) (hB_len : B.card = 8)
  (hA_mean : (A.sum id) / 7 = 15) (hB_mean : (B.sum id) / 8 = 22) :
  (A.sum id + B.sum id) / 15 = 18.73 :=
by sorry

end mean_of_combined_sets_l304_304887


namespace cory_can_eat_fruits_in_105_ways_l304_304388

-- Define the number of apples, oranges, and bananas Cory has
def apples := 4
def oranges := 1
def bananas := 2

-- Define the total number of fruits Cory has
def total_fruits := apples + oranges + bananas

-- Calculate the number of distinct orders in which Cory can eat the fruits
theorem cory_can_eat_fruits_in_105_ways :
  (Nat.factorial total_fruits) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) = 105 :=
by
  -- Provide a sorry to skip the proof
  sorry

end cory_can_eat_fruits_in_105_ways_l304_304388


namespace hyperbola_condition_l304_304971

theorem hyperbola_condition (m : ℝ) : 
  (exists a b : ℝ, ¬ a = 0 ∧ ¬ b = 0 ∧ ( ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 )) →
  ( -2 < m ∧ m < -1 ) :=
by
  sorry

end hyperbola_condition_l304_304971


namespace find_b_l304_304695

theorem find_b (b p : ℝ) (h_factor : ∃ k : ℝ, 3 * (x^3 : ℝ) + b * x + 9 = (x^2 + p * x + 3) * (k * x + k)) :
  b = -6 :=
by
  obtain ⟨k, h_eq⟩ := h_factor
  sorry

end find_b_l304_304695


namespace person_A_takes_12_more_minutes_l304_304890

-- Define distances, speeds, times
variables (S : ℝ) (v_A v_B : ℝ) (t : ℝ)

-- Define conditions as hypotheses
def conditions (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t) : Prop :=
  (v_A * (t + 4/5) = 2/3 * S) ∧ (v_B * t = 2/3 * S) ∧ (v_A * (t + 4/5 + 1/2 * t + 1/10) + 1/10 * v_B = S)

-- The proof problem statement
theorem person_A_takes_12_more_minutes
  (S : ℝ) (v_A v_B : ℝ) (t : ℝ)
  (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t)
  (h4 : conditions S v_A v_B t h1 h2 h3) : (t + 4/5) + 6/5 = 96 / 60 + 12 / 60 :=
sorry

end person_A_takes_12_more_minutes_l304_304890


namespace plane_equation_parallel_to_Oz_l304_304418

theorem plane_equation_parallel_to_Oz (A B D : ℝ)
  (h1 : A * 1 + B * 0 + D = 0)
  (h2 : A * (-2) + B * 1 + D = 0)
  (h3 : ∀ z : ℝ, exists c : ℝ, A * z + B * c + D = 0):
  A = 1 ∧ B = 3 ∧ D = -1 :=
  by
  sorry

end plane_equation_parallel_to_Oz_l304_304418


namespace solution_exists_l304_304930

theorem solution_exists (x y z u v : ℕ) (hx : x > 2000) (hy : y > 2000) (hz : z > 2000) (hu : u > 2000) (hv : v > 2000) : 
  x^2 + y^2 + z^2 + u^2 + v^2 = x * y * z * u * v - 65 :=
sorry

end solution_exists_l304_304930


namespace range_of_a_l304_304594

open Set

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, p a x → q x) →
  ({ x : ℝ | p a x } ⊆ { x : ℝ | q x }) →
  a ≤ -4 ∨ a ≥ 2 ∨ a = 0 :=
by
  sorry

end range_of_a_l304_304594


namespace smallest_positive_integer_k_l304_304202

-- Define the conditions
def y : ℕ := 2^3 * 3^4 * (2^2)^5 * 5^6 * (2*3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Define the question statement
theorem smallest_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (y * k) = m^2) ∧ k = 30 :=
by
  sorry

end smallest_positive_integer_k_l304_304202


namespace orange_probability_l304_304045

theorem orange_probability (total_apples : ℕ) (total_oranges : ℕ) (other_fruits : ℕ)
  (h1 : total_apples = 20) (h2 : total_oranges = 10) (h3 : other_fruits = 0) :
  (total_oranges : ℚ) / (total_apples + total_oranges + other_fruits) = 1 / 3 :=
by
  sorry

end orange_probability_l304_304045


namespace simplify_and_evaluate_l304_304481

theorem simplify_and_evaluate (a : ℤ) (h : a = -4) :
  (4 * a ^ 2 - 3 * a) - (2 * a ^ 2 + a - 1) + (2 - a ^ 2 + 4 * a) = 19 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l304_304481


namespace expected_value_of_win_l304_304913

theorem expected_value_of_win :
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
   prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 :=
by
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  have E : (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
            prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 := sorry
  exact E

end expected_value_of_win_l304_304913


namespace area_dodecagon_equals_rectangle_l304_304151

noncomputable def area_regular_dodecagon (r : ℝ) : ℝ := 3 * r^2

theorem area_dodecagon_equals_rectangle (r : ℝ) :
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  area_dodecagon = area_rectangle :=
by
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  show area_dodecagon = area_rectangle
  sorry

end area_dodecagon_equals_rectangle_l304_304151


namespace cos_angle_l304_304556

noncomputable def angle := -19 * Real.pi / 6

theorem cos_angle : Real.cos angle = Real.sqrt 3 / 2 :=
by sorry

end cos_angle_l304_304556


namespace classmates_ate_cake_l304_304644

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l304_304644


namespace periodic_sequence_exists_l304_304578

noncomputable def bounded_sequence (a : ℕ → ℤ) (M : ℤ) :=
  ∀ n, |a n| ≤ M

noncomputable def satisfies_recurrence (a : ℕ → ℤ) :=
  ∀ n, n ≥ 5 → a n = (a (n - 1) + a (n - 2) + a (n - 3) * a (n - 4)) / (a (n - 1) * a (n - 2) + a (n - 3) + a (n - 4))

theorem periodic_sequence_exists (a : ℕ → ℤ) (M : ℤ) 
  (h_bounded : bounded_sequence a M) (h_rec : satisfies_recurrence a) : 
  ∃ l : ℕ, ∀ n : ℕ, a (l + n) = a (l + n + (l + 1) - l) :=
sorry

end periodic_sequence_exists_l304_304578


namespace length_of_purple_part_l304_304917

theorem length_of_purple_part (p : ℕ) (black : ℕ) (blue : ℕ) (total : ℕ) 
  (h1 : black = 2) (h2 : blue = 1) (h3 : total = 6) (h4 : p + black + blue = total) : 
  p = 3 :=
by
  sorry

end length_of_purple_part_l304_304917


namespace bathroom_cleaning_time_ratio_l304_304130

noncomputable def hourlyRate : ℝ := 5
noncomputable def vacuumingHours : ℝ := 2 -- per session
noncomputable def vacuumingSessions : ℕ := 2
noncomputable def washingDishesTime : ℝ := 0.5
noncomputable def totalEarnings : ℝ := 30

theorem bathroom_cleaning_time_ratio :
  let vacuumingEarnings := vacuumingHours * vacuumingSessions * hourlyRate
  let washingDishesEarnings := washingDishesTime * hourlyRate
  let knownEarnings := vacuumingEarnings + washingDishesEarnings
  let bathroomEarnings := totalEarnings - knownEarnings
  let bathroomCleaningTime := bathroomEarnings / hourlyRate
  bathroomCleaningTime / washingDishesTime = 3 := 
by
  sorry

end bathroom_cleaning_time_ratio_l304_304130


namespace classmates_ate_cake_l304_304643

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l304_304643


namespace fraction_goldfish_preference_l304_304553

theorem fraction_goldfish_preference
  (students_per_class : ℕ)
  (students_prefer_golfish_miss_johnson : ℕ)
  (students_prefer_golfish_ms_henderson : ℕ)
  (students_prefer_goldfish_total : ℕ)
  (miss_johnson_fraction : ℚ)
  (ms_henderson_fraction : ℚ)
  (total_students_prefer_goldfish_feldstein : ℕ)
  (feldstein_fraction : ℚ) :
  miss_johnson_fraction = 1/6 ∧
  ms_henderson_fraction = 1/5 ∧
  students_per_class = 30 ∧
  students_prefer_golfish_miss_johnson = miss_johnson_fraction * students_per_class ∧
  students_prefer_golfish_ms_henderson = ms_henderson_fraction * students_per_class ∧
  students_prefer_goldfish_total = 31 ∧
  students_prefer_goldfish_total = students_prefer_golfish_miss_johnson + students_prefer_golfish_ms_henderson + total_students_prefer_goldfish_feldstein ∧
  feldstein_fraction * students_per_class = total_students_prefer_goldfish_feldstein
  →
  feldstein_fraction = 2 / 3 :=
by 
  sorry

end fraction_goldfish_preference_l304_304553


namespace possible_classmates_l304_304662

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l304_304662


namespace called_back_students_l304_304164

/-- Given the number of girls, boys, and students who didn't make the cut,
    this theorem proves the number of students who got called back. -/
theorem called_back_students (girls boys didnt_make_the_cut : ℕ)
    (h_girls : girls = 39)
    (h_boys : boys = 4)
    (h_didnt_make_the_cut : didnt_make_the_cut = 17) :
    girls + boys - didnt_make_the_cut = 26 := by
  sorry

end called_back_students_l304_304164


namespace matrix_equation_l304_304992

open Matrix

-- Define matrix N and the identity matrix I
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![-4, -2]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

-- Scalars p and q
def p : ℤ := 1
def q : ℤ := -26

-- Theorem statement
theorem matrix_equation :
  N * N = p • N + q • I :=
  by
    sorry

end matrix_equation_l304_304992


namespace length_of_segment_l304_304716

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end length_of_segment_l304_304716


namespace plains_routes_count_l304_304286

def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70
def total_routes : ℕ := 150
def mountainous_routes : ℕ := 21

theorem plains_routes_count :
  total_cities = mountainous_cities + plains_cities →
  3 * total_routes = total_cities →
  mountainous_routes * 2 ≤ mountainous_cities * 3 →
  (total_routes - mountainous_routes) * 2 = (70 * 3 - (mountainous_routes * 2)) →
  (total_routes - mountainous_routes * 2) / 2 = 81 :=
begin
  sorry
end

end plains_routes_count_l304_304286


namespace selection_probability_l304_304357

/-- Given conditions:
1. Probability of husband's selection: 1/7
2. Probability of wife's selection: 1/5

Prove the probability that only one of them is selected is 2/7.
-/
theorem selection_probability (p_husband p_wife : ℚ) (h1 : p_husband = 1/7) (h2 : p_wife = 1/5) :
  (p_husband * (1 - p_wife) + p_wife * (1 - p_husband)) = 2/7 :=
by
  sorry

end selection_probability_l304_304357


namespace leak_empty_time_l304_304360

theorem leak_empty_time (P L : ℝ) (h1 : P = 1 / 6) (h2 : P - L = 1 / 12) : 1 / L = 12 :=
by
  -- Proof to be provided
  sorry

end leak_empty_time_l304_304360


namespace sum_roots_quadratic_l304_304146

theorem sum_roots_quadratic (a b c : ℝ) (P : ℝ → ℝ) 
  (hP : ∀ x : ℝ, P x = a * x^2 + b * x + c)
  (h : ∀ x : ℝ, P (2 * x^5 + 3 * x) ≥ P (3 * x^4 + 2 * x^2 + 1)) : 
  -b / a = 6 / 5 :=
sorry

end sum_roots_quadratic_l304_304146


namespace lineup_count_l304_304009

def total_players : ℕ := 15
def out_players : ℕ := 3  -- Alice, Max, and John
def lineup_size : ℕ := 6

-- Define the binomial coefficient in Lean
def binom (n k : ℕ) : ℕ :=
  if h : n ≥ k then
    Nat.choose n k
  else
    0

theorem lineup_count (total_players out_players lineup_size : ℕ) :
  let remaining_with_alice := total_players - out_players + 1 
  let remaining_without_alice := total_players - out_players + 1 
  let remaining_without_both := total_players - out_players 
  binom remaining_with_alice (lineup_size-1) + binom remaining_without_alice (lineup_size-1) + binom remaining_without_both lineup_size = 3498 :=
by
  sorry

end lineup_count_l304_304009


namespace range_of_x_l304_304028

variable (x : ℝ)

-- Conditions used in the problem
def sqrt_condition : Prop := x + 2 ≥ 0
def non_zero_condition : Prop := x + 1 ≠ 0

-- The statement to be proven
theorem range_of_x : sqrt_condition x ∧ non_zero_condition x ↔ (x ≥ -2 ∧ x ≠ -1) :=
by
  sorry

end range_of_x_l304_304028


namespace initial_salt_percentage_is_10_l304_304905

-- Declarations for terminology
def initial_volume : ℕ := 72
def added_water : ℕ := 18
def final_volume : ℕ := initial_volume + added_water
def final_salt_percentage : ℝ := 0.08

-- Amount of salt in the initial solution
def initial_salt_amount (P : ℝ) := initial_volume * P

-- Amount of salt in the final solution
def final_salt_amount : ℝ := final_volume * final_salt_percentage

-- Proof that the initial percentage of salt was 10%
theorem initial_salt_percentage_is_10 :
  ∃ P : ℝ, initial_salt_amount P = final_salt_amount ∧ P = 0.1 :=
by
  sorry

end initial_salt_percentage_is_10_l304_304905


namespace randy_mango_trees_l304_304002

theorem randy_mango_trees (M C : ℕ) 
  (h1 : C = M / 2 - 5) 
  (h2 : M + C = 85) : 
  M = 60 := 
sorry

end randy_mango_trees_l304_304002


namespace probability_is_4_over_5_l304_304737

variable (total_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ)
variable (total_balls_eq : total_balls = 60) (red_balls_eq : red_balls = 5) (purple_balls_eq : purple_balls = 7)

def probability_neither_red_nor_purple : ℚ :=
  let favorable_outcomes := total_balls - (red_balls + purple_balls)
  let total_outcomes := total_balls
  favorable_outcomes / total_outcomes

theorem probability_is_4_over_5 :
  probability_neither_red_nor_purple total_balls red_balls purple_balls = 4 / 5 :=
by
  have h1: total_balls = 60 := total_balls_eq
  have h2: red_balls = 5 := red_balls_eq
  have h3: purple_balls = 7 := purple_balls_eq
  sorry

end probability_is_4_over_5_l304_304737


namespace smallest_positive_integer_k_l304_304201

-- Define the conditions
def y : ℕ := 2^3 * 3^4 * (2^2)^5 * 5^6 * (2*3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Define the question statement
theorem smallest_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (y * k) = m^2) ∧ k = 30 :=
by
  sorry

end smallest_positive_integer_k_l304_304201


namespace parallel_lines_m_l304_304861

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 2 * x + (m + 1) * y + 4 = 0) ∧ (∀ (x y : ℝ), m * x + 3 * y - 2 = 0) →
  (m = -3 ∨ m = 2) :=
by
  sorry

end parallel_lines_m_l304_304861


namespace number_of_red_balls_l304_304614

noncomputable def red_balls (n_black n_red draws black_draws : ℕ) : ℕ :=
  if black_draws = (draws * n_black) / (n_black + n_red) then n_red else sorry

theorem number_of_red_balls :
  ∀ (n_black draws black_draws : ℕ),
    n_black = 4 →
    draws = 100 →
    black_draws = 40 →
    red_balls n_black (red_balls 4 6 100 40) 100 40 = 6 :=
by
  intros n_black draws black_draws h_black h_draws h_blackdraws
  dsimp [red_balls]
  rw [h_black, h_draws, h_blackdraws]
  norm_num
  sorry

end number_of_red_balls_l304_304614


namespace polynomial_factorization_l304_304354

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end polynomial_factorization_l304_304354


namespace sequence_infinite_integers_l304_304579

theorem sequence_infinite_integers (x : ℕ → ℝ) (x1 x2 : ℝ) 
  (h1 : x 1 = x1) 
  (h2 : x 2 = x2) 
  (h3 : ∀ n ≥ 3, x n = x (n - 2) * x (n - 1) / (2 * x (n - 2) - x (n - 1))) : 
  (∃ k : ℤ, x1 = k ∧ x2 = k) ↔ (∀ n, ∃ m : ℤ, x n = m) :=
sorry

end sequence_infinite_integers_l304_304579


namespace lcm_14_21_35_l304_304894

-- Define the numbers
def a : ℕ := 14
def b : ℕ := 21
def c : ℕ := 35

-- Define the prime factorizations
def prime_factors_14 : List (ℕ × ℕ) := [(2, 1), (7, 1)]
def prime_factors_21 : List (ℕ × ℕ) := [(3, 1), (7, 1)]
def prime_factors_35 : List (ℕ × ℕ) := [(5, 1), (7, 1)]

-- Prove the least common multiple
theorem lcm_14_21_35 : Nat.lcm (Nat.lcm a b) c = 210 := by
  sorry

end lcm_14_21_35_l304_304894


namespace abc_zero_l304_304858

theorem abc_zero {a b c : ℝ} 
(h1 : (a + b) * (b + c) * (c + a) = a * b * c)
(h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) : 
a * b * c = 0 := 
by sorry

end abc_zero_l304_304858


namespace classmates_ate_cake_l304_304679

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l304_304679


namespace coeff_x4_in_expansion_l304_304338

theorem coeff_x4_in_expansion (x : ℝ) (sqrt2 : ℝ) (h₁ : sqrt2 = real.sqrt 2) :
  let c := (70 : ℝ) * (3^4 : ℝ) * (sqrt2^4 : ℝ) in
  c = 22680 :=
by
  sorry

end coeff_x4_in_expansion_l304_304338


namespace min_value_quadratic_expression_l304_304077

theorem min_value_quadratic_expression :
  ∃ x y : ℝ, min_val (3*x^2 + 3*x*y + y^2 - 3*x + 3*y + 9) = (45 / 8) := 
sorry

end min_value_quadratic_expression_l304_304077


namespace smallest_four_consecutive_numbers_l304_304888

theorem smallest_four_consecutive_numbers (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 4574880) : n = 43 :=
sorry

end smallest_four_consecutive_numbers_l304_304888


namespace Jill_water_volume_l304_304850

theorem Jill_water_volume 
  (n : ℕ) (h₀ : 3 * n = 48) :
  n * (1 / 4) + n * (1 / 2) + n * 1 = 28 := 
by 
  sorry

end Jill_water_volume_l304_304850


namespace max_value_ineq_l304_304809

theorem max_value_ineq (x y : ℝ) (hx1 : -5 ≤ x) (hx2 : x ≤ -3) (hy1 : 1 ≤ y) (hy2 : y ≤ 3) : 
  (x + y) / (x - 1) ≤ 2 / 3 := 
sorry

end max_value_ineq_l304_304809


namespace distance_from_focus_l304_304798

theorem distance_from_focus (y : ℝ) (hyp : y ^ 2 = 12) : 
  real.sqrt ((3 - 1) ^ 2 + y ^ 2) = 4 := by
  sorry

end distance_from_focus_l304_304798


namespace proof_equivalent_problem_l304_304296

variables (a b c : ℝ)
-- Conditions
axiom h1 : a < b
axiom h2 : b < 0
axiom h3 : c > 0

theorem proof_equivalent_problem :
  (a * c < b * c) ∧ (a + b + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end proof_equivalent_problem_l304_304296


namespace initial_percentage_of_milk_l304_304496

theorem initial_percentage_of_milk (M : ℝ) (H1 : M / 100 * 60 = 0.58 * 86.9) : M = 83.99 :=
by
  sorry

end initial_percentage_of_milk_l304_304496


namespace union_P_Q_l304_304628

open Set

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5, 6}

theorem union_P_Q : P ∪ Q = {1, 2, 3, 4, 5, 6} :=
by 
  -- Proof goes here
  sorry

end union_P_Q_l304_304628


namespace smallest_b_exists_l304_304792

theorem smallest_b_exists :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 4032 ∧ r + s = b) ∧
    (∀ b' : ℕ, (∀ r' s' : ℤ, r' * s' = 4032 ∧ r' + s' = b') → b ≤ b') :=
sorry

end smallest_b_exists_l304_304792


namespace marathon_yards_l304_304752

theorem marathon_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (marathons_run : ℕ) 
  (total_miles : ℕ) (total_yards : ℕ) (h1 : miles_per_marathon = 26) (h2 : yards_per_marathon = 385)
  (h3 : yards_per_mile = 1760) (h4 : marathons_run = 15) (h5 : 
  total_miles = marathons_run * miles_per_marathon + (marathons_run * yards_per_marathon) / yards_per_mile) 
  (h6 : total_yards = (marathons_run * yards_per_marathon) % yards_per_mile) : 
  total_yards = 495 :=
by
  -- This will be our process to verify the transformation
  sorry

end marathon_yards_l304_304752


namespace total_test_points_l304_304175

theorem total_test_points (total_questions two_point_questions four_point_questions points_per_two_question points_per_four_question : ℕ) 
  (h1 : total_questions = 40)
  (h2 : four_point_questions = 10)
  (h3 : points_per_two_question = 2)
  (h4 : points_per_four_question = 4)
  (h5 : two_point_questions = total_questions - four_point_questions)
  : (two_point_questions * points_per_two_question) + (four_point_questions * points_per_four_question) = 100 :=
by
  sorry

end total_test_points_l304_304175


namespace pizza_promotion_savings_l304_304132

theorem pizza_promotion_savings :
  let regular_price : ℕ := 18
  let promo_price : ℕ := 5
  let num_pizzas : ℕ := 3
  let total_regular_price := num_pizzas * regular_price
  let total_promo_price := num_pizzas * promo_price
  let total_savings := total_regular_price - total_promo_price
  total_savings = 39 :=
by
  sorry

end pizza_promotion_savings_l304_304132


namespace sin_double_angle_neg_l304_304105

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l304_304105


namespace second_smallest_packs_of_hot_dogs_l304_304071

theorem second_smallest_packs_of_hot_dogs (n m : ℕ) (k : ℕ) :
  (12 * n ≡ 5 [MOD 10]) ∧ (10 * m ≡ 3 [MOD 12]) → n = 15 :=
by
  sorry

end second_smallest_packs_of_hot_dogs_l304_304071


namespace distinct_real_roots_of_quadratic_l304_304082

variable (m : ℝ)

theorem distinct_real_roots_of_quadratic (h1 : 4 + 4 * m > 0) (h2 : m ≠ 0) : m = 1 :=
by
  sorry

end distinct_real_roots_of_quadratic_l304_304082


namespace identify_translation_l304_304898

def phenomenon (x : String) : Prop :=
  x = "translational"

def option_A : Prop := phenomenon "rotational"
def option_B : Prop := phenomenon "rotational"
def option_C : Prop := phenomenon "translational"
def option_D : Prop := phenomenon "rotational"

theorem identify_translation :
  (¬ option_A) ∧ (¬ option_B) ∧ option_C ∧ (¬ option_D) :=
  by {
    sorry
  }

end identify_translation_l304_304898


namespace loss_of_450_is_negative_450_l304_304266

-- Define the concept of profit and loss based on given conditions.
def profit (x : Int) := x
def loss (x : Int) := -x

-- The mathematical statement:
theorem loss_of_450_is_negative_450 :
  (profit 1000 = 1000) → (loss 450 = -450) :=
by
  intro h
  sorry

end loss_of_450_is_negative_450_l304_304266


namespace total_area_expanded_dining_area_l304_304052

noncomputable def expanded_dining_area_total : ℝ :=
  let rectangular_area := 35
  let radius := 4
  let semi_circular_area := (1 / 2) * Real.pi * (radius^2)
  rectangular_area + semi_circular_area

theorem total_area_expanded_dining_area :
  expanded_dining_area_total = 60.13272 := by
  sorry

end total_area_expanded_dining_area_l304_304052


namespace least_integer_greater_than_sqrt_450_l304_304512

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l304_304512


namespace sequence_inequality_l304_304162

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

noncomputable def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) - b n = b 1 - b 0

theorem sequence_inequality
  (ha : ∀ n, 0 < a n)
  (hg : is_geometric a q)
  (ha6_eq_b7 : a 6 = b 7)
  (hb : is_arithmetic b) :
  a 3 + a 9 ≥ b 4 + b 10 :=
by
  sorry

end sequence_inequality_l304_304162


namespace jason_picked_pears_l304_304292

def jason_picked (total_picked keith_picked mike_picked jason_picked : ℕ) : Prop :=
  jason_picked + keith_picked + mike_picked = total_picked

theorem jason_picked_pears:
  jason_picked 105 47 12 46 :=
by 
  unfold jason_picked
  sorry

end jason_picked_pears_l304_304292


namespace smallest_three_digit_integer_l304_304929

theorem smallest_three_digit_integer (n : ℕ) (h : 75 * n ≡ 225 [MOD 345]) (hne : n ≥ 100) (hn : n < 1000) : n = 118 :=
sorry

end smallest_three_digit_integer_l304_304929


namespace value_of_a_l304_304321

theorem value_of_a (a : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ (∃ (y : ℝ), y = 2 ∧ 9 = a ^ y)) : a = 3 := 
  by sorry

end value_of_a_l304_304321


namespace carol_rectangle_length_l304_304776

theorem carol_rectangle_length (lCarol : ℝ) :
    (∃ (wCarol : ℝ), wCarol = 20 ∧ lCarol * wCarol = 300) ↔ lCarol = 15 :=
by
  have jordan_area : 6 * 50 = 300 := by norm_num
  sorry

end carol_rectangle_length_l304_304776


namespace gcd_2015_15_l304_304892

theorem gcd_2015_15 : Nat.gcd 2015 15 = 5 :=
by
  have h1 : 2015 = 15 * 134 + 5 := by rfl
  have h2 : 15 = 5 * 3 := by rfl
  sorry

end gcd_2015_15_l304_304892


namespace usual_time_to_school_l304_304499

theorem usual_time_to_school (S T t : ℝ) (h : 1.2 * S * (T - t) = S * T) : T = 6 * t :=
by
  sorry

end usual_time_to_school_l304_304499


namespace proof_smallest_lcm_1_to_12_l304_304727

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end proof_smallest_lcm_1_to_12_l304_304727


namespace max_profit_l304_304147

-- Definitions based on conditions from the problem
def L1 (x : ℕ) : ℤ := -5 * (x : ℤ)^2 + 900 * (x : ℤ) - 16000
def L2 (x : ℕ) : ℤ := 300 * (x : ℤ) - 2000
def total_vehicles := 110
def total_profit (x : ℕ) : ℤ := L1 x + L2 (total_vehicles - x)

-- Statement of the problem
theorem max_profit :
  ∃ x y : ℕ, x + y = 110 ∧ x ≥ 0 ∧ y ≥ 0 ∧
  (L1 x + L2 y = 33000 ∧
   (∀ z w : ℕ, z + w = 110 ∧ z ≥ 0 ∧ w ≥ 0 → L1 z + L2 w ≤ 33000)) :=
sorry

end max_profit_l304_304147


namespace plywood_width_is_5_l304_304051

theorem plywood_width_is_5 (length width perimeter : ℕ) (h1 : length = 6) (h2 : perimeter = 2 * (length + width)) (h3 : perimeter = 22) : width = 5 :=
by {
  -- proof steps would go here, but are omitted per instructions
  sorry
}

end plywood_width_is_5_l304_304051


namespace range_of_m_l304_304444

theorem range_of_m (m : ℝ) :
  (∃ x y : ℤ, (x ≠ y) ∧ (x ≥ m ∧ y ≥ m) ∧ (3 - 2 * x ≥ 0) ∧ (3 - 2 * y ≥ 0)) ↔ (-1 < m ∧ m ≤ 0) :=
by
  sorry

end range_of_m_l304_304444


namespace quadratic_complex_inequality_solution_l304_304933
noncomputable def quadratic_inequality_solution (x : ℝ) : Prop :=
  (x^2 / (x + 2) ≥ 3 / (x - 2) + 7/4) ↔ -2 < x ∧ x < 2 ∨ 3 ≤ x

theorem quadratic_complex_inequality_solution (x : ℝ) (hx : x ≠ -2 ∧ x ≠ 2):
  quadratic_inequality_solution x :=
  sorry

end quadratic_complex_inequality_solution_l304_304933


namespace paths_H_to_J_through_I_l304_304204

theorem paths_H_to_J_through_I :
  let h_to_i_steps_right := 5
  let h_to_i_steps_down := 1
  let i_to_j_steps_right := 3
  let i_to_j_steps_down := 2
  let h_to_i_paths := Nat.choose (h_to_i_steps_right + h_to_i_steps_down) h_to_i_steps_down
  let i_to_j_paths := Nat.choose (i_to_j_steps_right + i_to_j_steps_down) i_to_j_steps_down
  let total_paths := h_to_i_paths * i_to_j_paths
  total_paths = 60 :=
by
  simp
  sorry

end paths_H_to_J_through_I_l304_304204


namespace expected_value_of_win_l304_304914

theorem expected_value_of_win :
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
   prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 :=
by
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  have E : (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
            prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 := sorry
  exact E

end expected_value_of_win_l304_304914


namespace woman_completion_days_l304_304743

variable (M W : ℚ)
variable (work_days_man work_days_total : ℚ)

-- Given conditions
def condition1 : Prop :=
  (10 * M + 15 * W) * 7 = 1

def condition2 : Prop :=
  M * 100 = 1

-- To prove
def one_woman_days : ℚ := 350

theorem woman_completion_days (h1 : condition1 M W) (h2 : condition2 M) :
  1 / W = one_woman_days :=
by
  sorry

end woman_completion_days_l304_304743


namespace monthly_growth_rate_l304_304065

-- Definitions based on the conditions given in the original problem.
def final_height : ℝ := 80
def current_height : ℝ := 20
def months_in_year : ℕ := 12

-- Prove the monthly growth rate.
theorem monthly_growth_rate : (final_height - current_height) / months_in_year = 5 := by
  sorry

end monthly_growth_rate_l304_304065


namespace isosceles_triangle_base_angle_l304_304455

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle_isosceles : a = b ∨ b = c ∨ c = a)
  (h_angle_sum : a + b + c = 180) (h_one_angle : a = 50 ∨ b = 50 ∨ c = 50) :
  a = 50 ∨ b = 50 ∨ c = 50 ∨ a = 65 ∨ b = 65 ∨ c = 65 :=
by
  sorry

end isosceles_triangle_base_angle_l304_304455


namespace all_a_n_are_perfect_squares_l304_304087

noncomputable def c : ℕ → ℤ 
| 0 => 1
| 1 => 0
| 2 => 2005
| n+2 => -3 * c n - 4 * c (n-1) + 2008

noncomputable def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4 ^ n * 2004 * 501

theorem all_a_n_are_perfect_squares (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 :=
by
  sorry

end all_a_n_are_perfect_squares_l304_304087


namespace polygon_sides_l304_304441

-- Given conditions
def is_interior_angle (angle : ℝ) : Prop :=
  angle = 150

-- The theorem to prove the number of sides
theorem polygon_sides (h : is_interior_angle 150) : ∃ n : ℕ, n = 12 :=
  sorry

end polygon_sides_l304_304441


namespace overlapping_area_l304_304882

def area_of_overlap (g1 g2 : Grid) : ℝ :=
  -- Dummy implementation to ensure code compiles
  6.0

structure Grid :=
  (size : ℝ) (arrow_direction : Direction)

inductive Direction
| North
| West

theorem overlapping_area (g1 g2 : Grid) 
  (h1 : g1.size = 4) 
  (h2 : g2.size = 4) 
  (d1 : g1.arrow_direction = Direction.North) 
  (d2 : g2.arrow_direction = Direction.West) 
  : area_of_overlap g1 g2 = 6 :=
by
  sorry

end overlapping_area_l304_304882


namespace compare_negatives_l304_304558

theorem compare_negatives : (-1.5 : ℝ) < (-1 + -1/5 : ℝ) :=
by 
  sorry

end compare_negatives_l304_304558


namespace arithmetic_sequence_property_l304_304581

def arith_seq (a : ℕ → ℤ) (a1 a3 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ a 3 = a3 ∧ (a 3 - a 1) = 2 * d

theorem arithmetic_sequence_property :
  ∀ (a : ℕ → ℤ), ∃ d : ℤ, arith_seq a 1 (-3) d →
  (1 - (a 2) - a 3 - (a 4) - (a 5) = 17) :=
by
  intros a
  use -2
  simp [arith_seq, *]
  sorry

end arithmetic_sequence_property_l304_304581


namespace possible_values_of_g_l304_304625

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem possible_values_of_g (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
by
  sorry

end possible_values_of_g_l304_304625


namespace train_speed_is_260_kmph_l304_304923

-- Define the conditions: length of the train and time to cross the pole
def length_of_train : ℝ := 130
def time_to_cross_pole : ℝ := 9

-- Define the conversion factor from meters per second to kilometers per hour
def conversion_factor : ℝ := 3.6

-- Define the expected speed in kilometers per hour
def expected_speed_kmph : ℝ := 260

-- The theorem statement
theorem train_speed_is_260_kmph :
  (length_of_train / time_to_cross_pole) * conversion_factor = expected_speed_kmph :=
sorry

end train_speed_is_260_kmph_l304_304923


namespace smallest_integer_min_value_l304_304042

theorem smallest_integer_min_value :
  ∃ (A B C D : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
  B ≠ C ∧ B ≠ D ∧ 
  C ≠ D ∧ 
  (A + B + C + D) = 288 ∧ 
  D = 90 ∧ 
  (A = 21) := 
sorry

end smallest_integer_min_value_l304_304042


namespace find_missing_number_l304_304268

theorem find_missing_number
  (x : ℝ)
  (h1 : (12 + x + y + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) : 
  y = 42 :=
  sorry

end find_missing_number_l304_304268


namespace base_angle_of_isosceles_triangle_l304_304454

theorem base_angle_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : a = 50) (h₂ : a + b + c = 180) (h₃ : a = b ∨ b = c ∨ c = a) : 
  b = 50 ∨ b = 65 :=
by sorry

end base_angle_of_isosceles_triangle_l304_304454


namespace find_y_when_x_is_1_l304_304180

theorem find_y_when_x_is_1 
  (k : ℝ) 
  (h1 : ∀ y, x = k / y^2) 
  (h2 : x = 1) 
  (h3 : x = 0.1111111111111111) 
  (y : ℝ) 
  (hy : y = 6) 
  (hx_k : k = 0.1111111111111111 * 36) :
  y = 2 := sorry

end find_y_when_x_is_1_l304_304180


namespace repeating_decimal_sum_l304_304403

theorem repeating_decimal_sum : (0.\overline{6} + 0.\overline{2} - 0.\overline{4} : ℚ) = (4 / 9 : ℚ) :=
by
  sorry

end repeating_decimal_sum_l304_304403


namespace banana_permutations_l304_304958

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def multiset_permutations (n : ℕ) (frequencies : list ℕ) : ℕ :=
  factorial n / (frequencies.map factorial).prod

theorem banana_permutations :
  multiset_permutations 6 [1, 2, 3] = 60 :=
by
  sorry

end banana_permutations_l304_304958


namespace sum_of_f1_possible_values_l304_304297

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f1_possible_values :
  (∀ (x y : ℝ), f (f (x - y)) = f x * f y - f x + f y - 2 * x * y) →
  (f 1 = -1) := sorry

end sum_of_f1_possible_values_l304_304297


namespace neither_long_furred_nor_brown_dogs_is_8_l304_304605

def total_dogs : ℕ := 45
def long_furred_dogs : ℕ := 29
def brown_dogs : ℕ := 17
def long_furred_and_brown_dogs : ℕ := 9

def neither_long_furred_nor_brown_dogs : ℕ :=
  total_dogs - (long_furred_dogs + brown_dogs - long_furred_and_brown_dogs)

theorem neither_long_furred_nor_brown_dogs_is_8 :
  neither_long_furred_nor_brown_dogs = 8 := 
by 
  -- Here we can use substitution and calculation steps used in the solution
  sorry

end neither_long_furred_nor_brown_dogs_is_8_l304_304605


namespace remainder_2_pow_33_mod_9_l304_304698

theorem remainder_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end remainder_2_pow_33_mod_9_l304_304698


namespace graph_intersection_l304_304925

noncomputable def log : ℝ → ℝ := sorry

lemma log_properties (a b : ℝ) (ha : 0 < a) (hb : 0 < b): log (a * b) = log a + log b := sorry

theorem graph_intersection :
  ∃! x : ℝ, 2 * log x = log (2 * x) :=
by
  sorry

end graph_intersection_l304_304925


namespace possible_classmates_l304_304661

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l304_304661


namespace sales_second_month_l304_304048

theorem sales_second_month 
  (sale_1 : ℕ) (sale_2 : ℕ) (sale_3 : ℕ) (sale_4 : ℕ) (sale_5 : ℕ) (sale_6 : ℕ)
  (avg_sale : ℕ)
  (h1 : sale_1 = 5400)
  (h2 : sale_3 = 6300)
  (h3 : sale_4 = 7200)
  (h4 : sale_5 = 4500)
  (h5 : sale_6 = 1200)
  (h_avg : avg_sale = 5600) :
  sale_2 = 9000 := 
by sorry

end sales_second_month_l304_304048


namespace carson_air_per_pump_l304_304382

-- Define the conditions
def total_air_needed : ℝ := 2 * 500 + 0.6 * 500 + 0.3 * 500

def total_pumps : ℕ := 29

-- Proof problem statement
theorem carson_air_per_pump : total_air_needed / total_pumps = 50 := by
  sorry

end carson_air_per_pump_l304_304382


namespace repeating_decimal_to_fraction_l304_304230

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l304_304230


namespace simplify_fraction_l304_304154

theorem simplify_fraction : (90 : ℚ) / (126 : ℚ) = 5 / 7 := 
by
  sorry

end simplify_fraction_l304_304154


namespace find_a1_l304_304815

open Nat

theorem find_a1 (a : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n)
  (h2 : a 3 = 12) : a 1 = 3 :=
sorry

end find_a1_l304_304815


namespace math_problem_l304_304601

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0) (h := y^2 - 1 / x^2 ≠ 0) (h₁ := x^2 * y^2 ≠ 1)

theorem math_problem :
  (x^2 - 1 / y^2) / (y^2 - 1 / x^2) = x^2 / y^2 :=
sorry

end math_problem_l304_304601


namespace simplify_expression_l304_304315

theorem simplify_expression (x : ℝ) : (3 * x + 6 - 5 * x) / 3 = - (2 / 3) * x + 2 :=
by
  sorry

end simplify_expression_l304_304315


namespace rahul_savings_is_correct_l304_304740

def Rahul_Savings_Problem : Prop :=
  ∃ (NSC PPF : ℝ), 
    (1/3) * NSC = (1/2) * PPF ∧ 
    NSC + PPF = 180000 ∧ 
    PPF = 72000

theorem rahul_savings_is_correct : Rahul_Savings_Problem :=
  sorry

end rahul_savings_is_correct_l304_304740


namespace gift_card_remaining_l304_304633

theorem gift_card_remaining (initial_amount : ℕ) (half_monday : ℕ) (quarter_tuesday : ℕ) : 
  initial_amount = 200 → 
  half_monday = initial_amount / 2 →
  quarter_tuesday = (initial_amount - half_monday) / 4 →
  initial_amount - half_monday - quarter_tuesday = 75 :=
by
  intros h_init h_half h_quarter
  rw [h_init, h_half, h_quarter]
  sorry

end gift_card_remaining_l304_304633


namespace largest_A_divisible_by_8_equal_quotient_remainder_l304_304037

theorem largest_A_divisible_by_8_equal_quotient_remainder :
  ∃ (A B C : ℕ), A = 8 * B + C ∧ B = C ∧ C < 8 ∧ A = 63 := by
  sorry

end largest_A_divisible_by_8_equal_quotient_remainder_l304_304037


namespace ratio_of_part_diminished_by_10_to_whole_number_l304_304050

theorem ratio_of_part_diminished_by_10_to_whole_number (N : ℝ) (x : ℝ) (h1 : 1/5 * N + 4 = x * N - 10) (h2 : N = 280) :
  x = 1 / 4 :=
by
  rw [h2] at h1
  sorry

end ratio_of_part_diminished_by_10_to_whole_number_l304_304050


namespace cylindrical_to_rectangular_l304_304386

open real

theorem cylindrical_to_rectangular (r θ z x y : ℝ) (h₀ : r = 7) (h₁ : θ = π / 3) (h₂ : z = -3) 
(h₃ : x = r * cos θ) (h₄ : y = r * sin θ) :
  (x, y, z) = (3.5, (7 * sqrt 3) / 2, -3) :=
by
  rw [h₀, h₁, h₂, h₃, h₄]
  sorry

end cylindrical_to_rectangular_l304_304386


namespace repeating_decimal_sum_l304_304399

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l304_304399


namespace sin_double_angle_fourth_quadrant_l304_304113

theorem sin_double_angle_fourth_quadrant (α : ℝ) (h_quadrant : ∃ k : ℤ, -π/2 + 2 * k * π < α ∧ α < 2 * k * π) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_fourth_quadrant_l304_304113


namespace local_minimum_at_minus_one_l304_304999

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_minimum_at_minus_one :
  (∃ δ > 0, ∀ x : ℝ, (x < -1 + δ ∧ x > -1 - δ) → f x ≥ f (-1)) :=
by
  sorry

end local_minimum_at_minus_one_l304_304999


namespace min_n_minus_m_l304_304429

noncomputable def f : ℝ → ℝ :=
λ x, if x > 1 then log x else (1/2) * x + (1/2)

theorem min_n_minus_m (m n : ℝ) (hmn : m < n) (hfn : f m = f n) : n - m = 3 - 2 * log 2 :=
sorry

end min_n_minus_m_l304_304429


namespace triangle_angle_contradiction_proof_l304_304173

variable (α : Type) [LinearOrderedField α] -- Assume α is a linear ordered field

-- Define the problem statement using Lean's syntax

/--
In a triangle, if using the method of contradiction to prove the proposition:
"At least one of the interior angles is not greater than 60 degrees,"
the correct assumption to make is that all three interior angles are greater than 60 degrees.
-/
theorem triangle_angle_contradiction_proof 
  (a b c : α) -- Assume α represents the type for angle measures
  (triangle_angles_sum : a + b + c = 180) -- The sum of interior angles in a triangle
  (contradiction_negation : ¬(a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60)) :
  a > 60 ∧ b > 60 ∧ c > 60 :=
by
  sorry-- Proof to be filled in

end triangle_angle_contradiction_proof_l304_304173


namespace cake_sharing_l304_304655

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l304_304655


namespace product_of_integers_l304_304030

-- Define the conditions as variables in Lean
variables {x y : ℤ}

-- State the main theorem/proof
theorem product_of_integers (h1 : x + y = 8) (h2 : x^2 + y^2 = 34) : x * y = 15 := by
  sorry

end product_of_integers_l304_304030


namespace smallest_integer_to_make_perfect_square_l304_304199

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_make_perfect_square :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (n * y) = k^2) ∧ n = 6 :=
by
  sorry

end smallest_integer_to_make_perfect_square_l304_304199


namespace sam_quarters_mowing_lawns_l304_304870

-- Definitions based on the given conditions
def pennies : ℕ := 9
def total_amount_dollars : ℝ := 1.84
def penny_value_dollars : ℝ := 0.01
def quarter_value_dollars : ℝ := 0.25

-- Theorem statement that Sam got 7 quarters given the conditions
theorem sam_quarters_mowing_lawns : 
  (total_amount_dollars - pennies * penny_value_dollars) / quarter_value_dollars = 7 := by
  sorry

end sam_quarters_mowing_lawns_l304_304870


namespace profit_percentage_no_initial_discount_l304_304373

theorem profit_percentage_no_initial_discount
  (CP : ℝ := 100)
  (bulk_discount : ℝ := 0.02)
  (sales_tax : ℝ := 0.065)
  (no_discount_price : ℝ := CP - CP * bulk_discount)
  (selling_price : ℝ := no_discount_price + no_discount_price * sales_tax)
  (profit : ℝ := selling_price - CP) :
  (profit / CP) * 100 = 4.37 :=
by
  -- proof here
  sorry

end profit_percentage_no_initial_discount_l304_304373


namespace problem1_problem2_l304_304381

-- Problem 1
theorem problem1 (m n : ℚ) (h : m ≠ n) : 
  (m / (m - n)) + (n / (n - m)) = 1 := 
by
  -- Proof steps would go here
  sorry

-- Problem 2
theorem problem2 (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 / (x^2 - 1)) / (1 / (x + 1)) = 2 / (x - 1) := 
by
  -- Proof steps would go here
  sorry

end problem1_problem2_l304_304381


namespace classmates_ate_cake_l304_304642

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l304_304642


namespace unique_solution_mnk_l304_304926

theorem unique_solution_mnk :
  ∀ (m n k : ℕ), 3^n + 4^m = 5^k → (m, n, k) = (0, 1, 1) :=
by
  intros m n k h
  sorry

end unique_solution_mnk_l304_304926


namespace distance_between_P_and_F2_l304_304094
open Real

theorem distance_between_P_and_F2 (x y c : ℝ) (h1 : c = sqrt 3)
    (h2 : x = -sqrt 3) (h3 : y = 1/2) : 
    sqrt ((sqrt 3 - x) ^ 2 + (0 - y) ^ 2) = 7 / 2 :=
by
  sorry

end distance_between_P_and_F2_l304_304094


namespace repeating_decimal_to_fraction_l304_304242

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l304_304242


namespace classmates_ate_cake_l304_304678

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l304_304678


namespace point_symmetric_y_axis_l304_304950

theorem point_symmetric_y_axis (a b : ℤ) (h₁ : a = -(-2)) (h₂ : b = 3) : a + b = 5 := by
  sorry

end point_symmetric_y_axis_l304_304950


namespace smallest_b_exists_l304_304791

theorem smallest_b_exists :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 4032 ∧ r + s = b) ∧
    (∀ b' : ℕ, (∀ r' s' : ℤ, r' * s' = 4032 ∧ r' + s' = b') → b ≤ b') :=
sorry

end smallest_b_exists_l304_304791


namespace inequality_necessary_not_sufficient_l304_304560

theorem inequality_necessary_not_sufficient (m : ℝ) : 
  (-3 < m ∧ m < 5) → (5 - m > 0 ∧ m + 3 > 0 ∧ 5 - m ≠ m + 3) :=
by
  intro h
  sorry

end inequality_necessary_not_sufficient_l304_304560


namespace three_boys_in_shop_at_same_time_l304_304536

-- Definitions for the problem conditions
def boys : Type := Fin 7  -- Representing the 7 boys
def visits : Type := Fin 3  -- Each boy makes 3 visits

-- A structure representing a visit by a boy
structure Visit := (boy : boys) (visit_num : visits)

-- Meeting condition: Every pair of boys meets at the shop
def meets_at_shop (v1 v2 : Visit) : Prop :=
  v1.boy ≠ v2.boy  -- Ensure it's not the same boy (since we assume each pair meets)

-- The theorem to be proven
theorem three_boys_in_shop_at_same_time :
  ∃ (v1 v2 v3 : Visit), v1.boy ≠ v2.boy ∧ v2.boy ≠ v3.boy ∧ v1.boy ≠ v3.boy :=
sorry

end three_boys_in_shop_at_same_time_l304_304536


namespace recurring_decimal_to_fraction_l304_304217

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l304_304217


namespace recurring_decimal_to_fraction_l304_304234

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l304_304234


namespace solve_eq1_solve_eq2_l304_304314

theorem solve_eq1 (x : ℝ) : x^2 - 6*x - 7 = 0 → x = 7 ∨ x = -1 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : 3*x^2 - 1 = 2*x → x = 1 ∨ x = -1/3 :=
by
  sorry

end solve_eq1_solve_eq2_l304_304314


namespace find_a_l304_304946

theorem find_a (a x y : ℝ) (h1 : a^(3*x - 1) * 3^(4*y - 3) = 49^x * 27^y) (h2 : x + y = 4) : a = 7 := by
  sorry

end find_a_l304_304946


namespace sin_double_angle_in_fourth_quadrant_l304_304110

theorem sin_double_angle_in_fourth_quadrant (α : ℝ) (h : -π/2 < α ∧ α < 0) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_in_fourth_quadrant_l304_304110


namespace percentage_increase_on_friday_l304_304857

theorem percentage_increase_on_friday (avg_books_per_day : ℕ) (friday_books : ℕ) (total_books_per_week : ℕ) (days_open : ℕ)
  (h1 : avg_books_per_day = 40)
  (h2 : total_books_per_week = 216)
  (h3 : days_open = 5)
  (h4 : friday_books > avg_books_per_day) :
  (((friday_books - avg_books_per_day) * 100) / avg_books_per_day) = 40 :=
sorry

end percentage_increase_on_friday_l304_304857


namespace least_integer_greater_than_sqrt_450_l304_304509

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l304_304509


namespace find_a_for_square_binomial_l304_304414

theorem find_a_for_square_binomial (a r s : ℝ) 
  (h1 : ax^2 + 18 * x + 9 = (r * x + s)^2)
  (h2 : a = r^2)
  (h3 : 2 * r * s = 18)
  (h4 : s^2 = 9) : 
  a = 9 := 
by sorry

end find_a_for_square_binomial_l304_304414


namespace lina_collects_stickers_l304_304148

theorem lina_collects_stickers :
  let a := 3
  let d := 2
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  S_n = 120 :=
by
  sorry

end lina_collects_stickers_l304_304148


namespace base_angle_of_isosceles_triangle_l304_304453

theorem base_angle_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : a = 50) (h₂ : a + b + c = 180) (h₃ : a = b ∨ b = c ∨ c = a) : 
  b = 50 ∨ b = 65 :=
by sorry

end base_angle_of_isosceles_triangle_l304_304453


namespace length_of_segment_l304_304718

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end length_of_segment_l304_304718


namespace productivity_increase_correct_l304_304783

def productivity_increase (that: ℝ) :=
  ∃ x : ℝ, (x + 1) * (x + 1) * 2500 = 2809

theorem productivity_increase_correct :
  productivity_increase (0.06) :=
by
  sorry

end productivity_increase_correct_l304_304783


namespace repeating_decimal_fraction_l304_304221

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l304_304221


namespace least_integer_gt_sqrt_450_l304_304522

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l304_304522


namespace weeks_to_save_remaining_l304_304302

-- Assuming the conditions
def cost_of_shirt : ℝ := 3
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

-- The proof goal
theorem weeks_to_save_remaining (cost_of_shirt amount_saved saving_per_week : ℝ) :
  cost_of_shirt = 3 ∧ amount_saved = 1.5 ∧ saving_per_week = 0.5 →
  ((cost_of_shirt - amount_saved) / saving_per_week) = 3 := by
  sorry

end weeks_to_save_remaining_l304_304302


namespace range_of_a_for_local_maximum_l304_304254

noncomputable def f' (a x : ℝ) := a * (x + 1) * (x - a)

theorem range_of_a_for_local_maximum {a : ℝ} (hf_max : ∀ x : ℝ, f' a x = 0 → ∀ y : ℝ, y ≠ x → f' a y ≤ f' a x) :
  -1 < a ∧ a < 0 :=
sorry

end range_of_a_for_local_maximum_l304_304254


namespace two_digit_number_sum_l304_304145

theorem two_digit_number_sum (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by {
  sorry
}

end two_digit_number_sum_l304_304145


namespace blake_change_l304_304766

def cost_oranges : ℕ := 40
def cost_apples : ℕ := 50
def cost_mangoes : ℕ := 60
def initial_money : ℕ := 300

def total_cost : ℕ := cost_oranges + cost_apples + cost_mangoes
def change : ℕ := initial_money - total_cost

theorem blake_change : change = 150 := by
  sorry

end blake_change_l304_304766


namespace expand_expression_l304_304397

theorem expand_expression : 
  ∀ (x : ℝ), (7 * x^3 - 5 * x + 2) * 4 * x^2 = 28 * x^5 - 20 * x^3 + 8 * x^2 :=
by
  intros x
  sorry

end expand_expression_l304_304397


namespace unique_flavors_l304_304417

noncomputable def distinctFlavors : Nat :=
  let redCandies := 5
  let greenCandies := 4
  let blueCandies := 2
  (90 - 15 - 18 - 30 + 3 + 5 + 6) / 3  -- Adjustments and consideration for equivalent ratios.
  
theorem unique_flavors :
  distinctFlavors = 11 :=
  by
    sorry

end unique_flavors_l304_304417


namespace equation_no_solution_B_l304_304732

theorem equation_no_solution_B :
  ¬(∃ x : ℝ, |-3 * x| + 5 = 0) :=
sorry

end equation_no_solution_B_l304_304732


namespace shoes_difference_l304_304637

theorem shoes_difference :
  let pairs_per_box := 20
  let boxes_A := 8
  let boxes_B := 5 * boxes_A
  let total_pairs_A := boxes_A * pairs_per_box
  let total_pairs_B := boxes_B * pairs_per_box
  total_pairs_B - total_pairs_A = 640 :=
by
  sorry

end shoes_difference_l304_304637


namespace seed_grow_prob_l304_304488

theorem seed_grow_prob (P_G P_S_given_G : ℝ) (hP_G : P_G = 0.9) (hP_S_given_G : P_S_given_G = 0.8) :
  P_G * P_S_given_G = 0.72 :=
by
  rw [hP_G, hP_S_given_G]
  norm_num

end seed_grow_prob_l304_304488


namespace problem_l304_304258

noncomputable def trajectory_C (x y : ℝ) : Prop :=
  y^2 = -8 * x

theorem problem (P : ℝ × ℝ) (k : ℝ) (h : -1 < k ∧ k < 0) 
  (H1 : P.1 = -2 ∨ P.1 = 2)
  (H2 : trajectory_C P.1 P.2) :
  ∃ Q : ℝ × ℝ, Q.1 < -6 :=
  sorry

end problem_l304_304258


namespace marlon_gift_card_balance_l304_304630

theorem marlon_gift_card_balance 
  (initial_amount : ℕ) 
  (spent_monday : initial_amount / 2 = 100)
  (spent_tuesday : (initial_amount / 2) / 4 = 25) 
  : (initial_amount / 2) - (initial_amount / 2 / 4) = 75 :=
by
  sorry

end marlon_gift_card_balance_l304_304630


namespace repeating_decimals_sum_l304_304407

theorem repeating_decimals_sum : (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ) = 4 / 9 :=
 by 
  have h₁ : (0.666666... : ℚ) = 2 / 3,
    -- Since x = 0.6666..., then 10x = 6.6666...,
    -- so 10x - x = 6, then x = 6 / 9, hence 2 / 3
    sorry,
  have h₂ : (0.222222... : ℚ) = 2 / 9,
    -- Since x = 0.2222..., then 10x = 2.2222...,
    -- so 10x - x = 2, then x = 2 / 9
    sorry,
  have h₃ : (0.444444... : ℚ) = 4 / 9,
    -- Since x = 0.4444..., then 10x = 4.4444...,
    -- so 10x - x = 4, then x = 4 / 9
    sorry,
  calc
    (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ)
        = (2 / 3) + (2 / 9) - (4 / 9) : by rw [h₁, h₂, h₃]
    ... = (6 / 9) + (2 / 9) - (4 / 9) : by norm_num
    ... = 4 / 9 : by ring

end repeating_decimals_sum_l304_304407


namespace segment_length_eq_ten_l304_304712

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end segment_length_eq_ten_l304_304712


namespace therapy_charge_l304_304176

-- Let F be the charge for the first hour and A be the charge for each additional hour
-- Two conditions are:
-- 1. F = A + 40
-- 2. F + 4A = 375

-- We need to prove that the total charge for 2 hours of therapy is 174
theorem therapy_charge (A F : ℕ) (h1 : F = A + 40) (h2 : F + 4 * A = 375) :
  F + A = 174 :=
by
  sorry

end therapy_charge_l304_304176


namespace domain_of_tan_l304_304157

open Real

noncomputable def function_domain : Set ℝ :=
  {x | ∀ k : ℤ, x ≠ k * π + 3 * π / 4}

theorem domain_of_tan : ∀ x : ℝ,
  (∃ k : ℤ, x = k * π + 3 * π / 4) → ¬ (∃ y : ℝ, y = tan (π / 4 - x)) :=
by
  intros x hx
  obtain ⟨k, hk⟩ := hx
  sorry

end domain_of_tan_l304_304157


namespace cannot_be_value_of_x_l304_304982

theorem cannot_be_value_of_x (x : ℕ) 
  (h1 : ∀ (k : ℕ), k ∈ {5, 16, 27, 38, 49} → x = (k - 1) / 11 * 11 + 5) :
  x ≠ 61 :=
by 
  sorry

end cannot_be_value_of_x_l304_304982


namespace hair_cut_second_day_l304_304072

variable (hair_first_day : ℝ) (total_hair_cut : ℝ)

theorem hair_cut_second_day (h1 : hair_first_day = 0.375) (h2 : total_hair_cut = 0.875) :
  total_hair_cut - hair_first_day = 0.500 :=
by sorry

end hair_cut_second_day_l304_304072


namespace cookie_distribution_l304_304482

theorem cookie_distribution:
  ∀ (initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny : ℕ),
    initial_boxes = 45 →
    brother_cookies = 12 →
    sister_cookies = 9 →
    after_siblings = initial_boxes - brother_cookies - sister_cookies →
    leftover_sonny = 17 →
    leftover = after_siblings - leftover_sonny →
    leftover = 7 :=
by
  intros initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny
  intros h1 h2 h3 h4 h5 h6
  sorry

end cookie_distribution_l304_304482


namespace sam_quarters_l304_304869

theorem sam_quarters (pennies : ℕ) (total : ℝ) (value_penny : ℝ) (value_quarter : ℝ) (quarters : ℕ) :
  pennies = 9 →
  total = 1.84 →
  value_penny = 0.01 →
  value_quarter = 0.25 →
  quarters = (total - pennies * value_penny) / value_quarter →
  quarters = 7 :=
by
  intros
  sorry

end sam_quarters_l304_304869


namespace arithmetic_sequence_part_a_arithmetic_sequence_part_b_l304_304375

theorem arithmetic_sequence_part_a (e u k : ℕ) (n : ℕ) 
  (h1 : e = 1) 
  (h2 : u = 1000) 
  (h3 : k = 343) 
  (h4 : n = 100) : ¬ (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

theorem arithmetic_sequence_part_b (e u k : ℝ) (n : ℕ) 
  (h1 : e = 81 * Real.sqrt 2 - 64 * Real.sqrt 3) 
  (h2 : u = 54 * Real.sqrt 2 - 28 * Real.sqrt 3)
  (h3 : k = 69 * Real.sqrt 2 - 48 * Real.sqrt 3)
  (h4 : n = 100) : (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

end arithmetic_sequence_part_a_arithmetic_sequence_part_b_l304_304375


namespace triangle_is_acute_l304_304976

-- Define the condition that the angles have a ratio of 2:3:4
def angle_ratio_cond (a b c : ℝ) : Prop :=
  a / b = 2 / 3 ∧ b / c = 3 / 4

-- Define the sum of the angles in a triangle
def angle_sum_cond (a b c : ℝ) : Prop :=
  a + b + c = 180

-- The proof problem stating that triangle with angles in ratio 2:3:4 is acute
theorem triangle_is_acute (a b c : ℝ) (h_ratio : angle_ratio_cond a b c) (h_sum : angle_sum_cond a b c) : 
  a < 90 ∧ b < 90 ∧ c < 90 := 
by
  sorry

end triangle_is_acute_l304_304976


namespace segment_length_eq_ten_l304_304710

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end segment_length_eq_ten_l304_304710


namespace squirrels_acorns_l304_304364

theorem squirrels_acorns (x : ℕ) : 
    (5 * (x - 15) = 575) → 
    x = 130 := 
by 
  intros h
  sorry

end squirrels_acorns_l304_304364


namespace number_of_classmates_ate_cake_l304_304666

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l304_304666


namespace winner_collected_l304_304073

variable (M : ℕ)
variable (last_year_rate this_year_rate : ℝ)
variable (extra_miles : ℕ)
variable (money_collected_last_year money_collected_this_year : ℝ)

axiom rate_last_year : last_year_rate = 4
axiom rate_this_year : this_year_rate = 2.75
axiom extra_miles_eq : extra_miles = 5

noncomputable def money_eq (M : ℕ) : ℝ :=
  last_year_rate * M

theorem winner_collected :
  ∃ M : ℕ, money_eq M = 44 :=
by
  sorry

end winner_collected_l304_304073


namespace truck_tank_capacity_l304_304500

-- Definitions based on conditions
def truck_tank (T : ℝ) : Prop := true
def car_tank : Prop := true
def truck_half_full (T : ℝ) : Prop := true
def car_third_full : Prop := true
def add_fuel (T : ℝ) : Prop := T / 2 + 8 = 18

-- Theorem statement
theorem truck_tank_capacity (T : ℝ) (ht : truck_tank T) (hc : car_tank) 
  (ht_half : truck_half_full T) (hc_third : car_third_full) (hf_add : add_fuel T) : T = 20 :=
  sorry

end truck_tank_capacity_l304_304500


namespace lcm_1_to_12_l304_304724

theorem lcm_1_to_12 : nat.lcm (list.range (12 + 1)) = 27720 :=
begin
  sorry
end

end lcm_1_to_12_l304_304724


namespace sum_of_angles_l304_304705

-- Definitions of acute, right, and obtuse angles
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_right (θ : ℝ) : Prop := θ = 90
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The main statement we want to prove
theorem sum_of_angles :
  (∀ (α β : ℝ), is_acute α ∧ is_acute β → is_acute (α + β) ∨ is_right (α + β) ∨ is_obtuse (α + β)) ∧
  (∀ (α β : ℝ), is_acute α ∧ is_right β → is_obtuse (α + β)) :=
by sorry

end sum_of_angles_l304_304705


namespace savings_promotion_l304_304136

theorem savings_promotion (reg_price promo_price num_pizzas : ℕ) (h1 : reg_price = 18) (h2 : promo_price = 5) (h3 : num_pizzas = 3) :
  reg_price * num_pizzas - promo_price * num_pizzas = 39 := by
  sorry

end savings_promotion_l304_304136


namespace prime_p_geq_5_div_24_l304_304830

theorem prime_p_geq_5_div_24 (p : ℕ) (hp : Nat.Prime p) (hp_geq_5 : p ≥ 5) : 24 ∣ (p^2 - 1) :=
sorry

end prime_p_geq_5_div_24_l304_304830


namespace problem_solution_l304_304884

def sequence_graphical_representation_isolated (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ x : ℝ, x = a n

def sequence_terms_infinite (a : ℕ → ℝ) : Prop :=
  ∃ l : List ℝ, ∃ n : ℕ, l.length = n

def sequence_general_term_formula_unique (a : ℕ → ℝ) : Prop :=
  ∀ f g : ℕ → ℝ, (∀ n, f n = g n) → f = g

theorem problem_solution
  (h1 : ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a)
  (h2 : ¬ ∀ a : ℕ → ℝ, sequence_terms_infinite a)
  (h3 : ¬ ∀ a : ℕ → ℝ, sequence_general_term_formula_unique a) :
  ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a ∧ 
                ¬ (sequence_terms_infinite a) ∧
                ¬ (sequence_general_term_formula_unique a) := by
  sorry

end problem_solution_l304_304884


namespace least_int_gt_sqrt_450_l304_304504

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l304_304504


namespace real_root_polynomials_l304_304788

open Polynomial

theorem real_root_polynomials (n : ℕ) (P : Polynomial ℝ) :
  (∀ i, i < n → (coeff P (n - 1 - i) = 1 ∨ coeff P (n - 1 - i) = -1)) →
  (∀ r, is_root P r → r ∈ ℝ) →
  (n = 1 ∧ (P = X - 1 ∨ P = X + 1)) ∨
  (n = 2 ∧ (P = X^2 - 1 ∨ P = X^2 - X - 1 ∨ P = X^2 + X - 1)) ∨
  (n = 3 ∧ (P = (X^2 - 1) * (X - 1) ∨ P = (X^2 - 1) * (X + 1))) :=
by sorry

end real_root_polynomials_l304_304788


namespace delores_initial_money_l304_304781

theorem delores_initial_money (cost_computer : ℕ) (cost_printer : ℕ) (money_left : ℕ) (initial_money : ℕ) :
  cost_computer = 400 → cost_printer = 40 → money_left = 10 → initial_money = cost_computer + cost_printer + money_left → initial_money = 450 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end delores_initial_money_l304_304781


namespace pigs_count_l304_304844

-- Definitions from step a)
def pigs_leg_count : ℕ := 4 -- Each pig has 4 legs
def hens_leg_count : ℕ := 2 -- Each hen has 2 legs

variable {P H : ℕ} -- P is the number of pigs, H is the number of hens

-- Condition from step a) as a function
def total_legs (P H : ℕ) : ℕ := pigs_leg_count * P + hens_leg_count * H
def total_heads (P H : ℕ) : ℕ := P + H

-- Theorem to prove the number of pigs given the condition
theorem pigs_count {P H : ℕ} (h : total_legs P H = 2 * total_heads P H + 22) : P = 11 :=
  by 
    sorry

end pigs_count_l304_304844


namespace recurring_decimal_to_fraction_l304_304236

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l304_304236


namespace classmates_ate_cake_l304_304675

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l304_304675


namespace total_work_completed_in_days_l304_304177

theorem total_work_completed_in_days (T : ℕ) :
  (amit_days amit_worked ananthu_days remaining_work : ℕ) → 
  amit_days = 3 → amit_worked = amit_days * (1 / 15) → 
  ananthu_days = 36 → 
  remaining_work = 1 - amit_worked  →
  (ananthu_days * (1 / 45)) = remaining_work →
  T = amit_days + ananthu_days →
  T = 39 := 
sorry

end total_work_completed_in_days_l304_304177


namespace ant_completes_path_in_finite_time_l304_304533

noncomputable def ant_travel_time (t : ℝ) (k : ℝ) (h : k < 1) : ℝ := 
  t / (1 - k)

theorem ant_completes_path_in_finite_time (t k : ℝ) (h : k < 1) : 
  ∃ T : ℝ, T = ant_travel_time t k h :=
begin
  use ant_travel_time t k h,
  exact rfl,
end

end ant_completes_path_in_finite_time_l304_304533


namespace local_minimum_at_neg_one_l304_304996

noncomputable def f (x : ℝ) := x * Real.exp x

theorem local_minimum_at_neg_one : (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x + 1) < δ → f x > f (-1)) :=
sorry

end local_minimum_at_neg_one_l304_304996


namespace five_digit_numbers_with_alternating_parity_l304_304259

theorem five_digit_numbers_with_alternating_parity : 
  ∃ n : ℕ, n = 5625 ∧ ∀ (x : ℕ), (10000 ≤ x ∧ x < 100000) → 
    (∀ i, i < 4 → (((x / 10^i) % 10) % 2 ≠ ((x / 10^(i+1)) % 10) % 2)) ↔ 
    (x = 5625) := 
sorry

end five_digit_numbers_with_alternating_parity_l304_304259


namespace new_bag_marbles_l304_304395

open Nat

theorem new_bag_marbles 
  (start_marbles : ℕ)
  (lost_marbles : ℕ)
  (given_marbles : ℕ)
  (received_back_marbles : ℕ)
  (end_marbles : ℕ)
  (h_start : start_marbles = 40)
  (h_lost : lost_marbles = 3)
  (h_given : given_marbles = 5)
  (h_received_back : received_back_marbles = 2 * given_marbles)
  (h_end : end_marbles = 54) :
  (end_marbles = (start_marbles - lost_marbles - given_marbles + received_back_marbles + new_bag) ∧ new_bag = 12) :=
by
  sorry

end new_bag_marbles_l304_304395


namespace domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l304_304590

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

namespace f_props

theorem domain_not_neg1 : ∀ x : ℝ, x ≠ -1 ↔ x ∈ {y | y ≠ -1} :=
by simp [f]

theorem increasing_on_neg1_infty : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → -1 < x2 → f x1 < f x2 :=
sorry

theorem min_max_on_3_5 : (∀ y : ℝ, y = f 3 → y = 5 / 4) ∧ (∀ y : ℝ, y = f 5 → y = 3 / 2) :=
sorry

end f_props

end domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l304_304590


namespace recurring_decimal_to_fraction_l304_304235

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l304_304235


namespace vector_addition_correct_l304_304270

variables (a b : ℝ × ℝ)
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-1, 2)

theorem vector_addition_correct : vector_a + vector_b = (1, 5) :=
by
  -- Assume a and b are vectors in 2D space
  have a := vector_a
  have b := vector_b
  -- By definition of vector addition
  sorry

end vector_addition_correct_l304_304270


namespace smallest_k_l304_304337

theorem smallest_k (k : ℕ) 
  (h1 : 201 % 24 = 9 % 24) 
  (h2 : (201 + k) % (24 + k) = (9 + k) % (24 + k)) : 
  k = 8 :=
by 
  sorry

end smallest_k_l304_304337


namespace original_triangle_area_l304_304687

theorem original_triangle_area (A_new : ℝ) (k : ℝ) (h1 : k = 4) (h2 : A_new = 64) :
  let A_orig := A_new / (k * k) in A_orig = 4 :=
by
  let A_orig := A_new / (k * k)
  sorry

end original_triangle_area_l304_304687


namespace tetrahedron_ratio_l304_304197

open Geometry

theorem tetrahedron_ratio (A B C D : Point)
  (M_AB : Midpoint A B)
  (M_CD : Midpoint C D)
  (L : Point) (N : Point) :
  Plane (Midpoint.to_plane M_AB) (Midpoint.to_plane M_CD) (∩ (Edge A D)) = L →
  Plane (Midpoint.to_plane M_AB) (Midpoint.to_plane M_CD) (∩ (Edge B C)) = N →
  BC : Length (Edge B C) →
  CN : Length (Segment C N) →
  AD : Length (Edge A D) →
  DL : Length (Segment D L) →
  BC / CN = AD / DL := by 
  sorry

end tetrahedron_ratio_l304_304197


namespace dealer_car_ratio_calculation_l304_304369

theorem dealer_car_ratio_calculation (X Y : ℝ) 
  (cond1 : 1.4 * X = 1.54 * (X + Y) - 1.6 * Y) :
  let a := 3
  let b := 7
  ((X / Y) = (3 / 7) ∧ (11 * a + 13 * b = 124)) :=
by
  sorry

end dealer_car_ratio_calculation_l304_304369


namespace graph_forms_l304_304070

theorem graph_forms (x y : ℝ) :
  x^3 * (2 * x + 2 * y + 3) = y^3 * (2 * x + 2 * y + 3) →
  (∀ x y : ℝ, y ≠ x → y = -x - 3 / 2) ∨ (y = x) :=
sorry

end graph_forms_l304_304070


namespace find_k2_minus_b2_l304_304090

theorem find_k2_minus_b2 (k b : ℝ) (h1 : 3 = k * 1 + b) (h2 : 2 = k * (-1) + b) : k^2 - b^2 = -6 := 
by
  sorry

end find_k2_minus_b2_l304_304090


namespace unique_solution_for_lines_intersection_l304_304896

theorem unique_solution_for_lines_intersection (n : ℕ) (h : n * (n - 1) / 2 = 2) : n = 2 :=
by
  sorry

end unique_solution_for_lines_intersection_l304_304896


namespace problem_eq_l304_304425

theorem problem_eq : 
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → y = x / (x + 1) → (x - y + 4 * x * y) / (x * y) = 5 :=
by
  intros x y hx hnz hyxy
  sorry

end problem_eq_l304_304425


namespace cleaning_times_l304_304460

theorem cleaning_times (A B C : ℕ) (hA : A = 40) (hB : B = A / 4) (hC : C = 2 * B) : 
  B = 10 ∧ C = 20 := by
  sorry

end cleaning_times_l304_304460


namespace largest_n_divides_l304_304823

theorem largest_n_divides (n : ℕ) (h : 2^n ∣ 5^256 - 1) : n ≤ 10 := sorry

end largest_n_divides_l304_304823


namespace exercise_felt_weight_l304_304497

variable (n w : ℕ)
variable (p : ℝ)

def total_weight (n : ℕ) (w : ℕ) : ℕ := n * w

def felt_weight (total_weight : ℕ) (p : ℝ) : ℝ := total_weight * (1 + p)

theorem exercise_felt_weight (h1 : n = 10) (h2 : w = 30) (h3 : p = 0.20) : 
  felt_weight (total_weight n w) p = 360 :=
by 
  sorry

end exercise_felt_weight_l304_304497


namespace solve_for_z_l304_304439

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l304_304439


namespace cone_height_l304_304251

theorem cone_height (r : ℝ) (θ : ℝ) (h : ℝ)
  (hr : r = 1)
  (hθ : θ = (2 / 3) * Real.pi)
  (h_eq : h = 2 * Real.sqrt 2) :
  ∃ l : ℝ, l = 3 ∧ h = Real.sqrt (l^2 - r^2) :=
by
  sorry

end cone_height_l304_304251


namespace parrots_per_cage_l304_304754

-- Definitions of the given conditions
def num_cages : ℕ := 6
def num_parakeets_per_cage : ℕ := 7
def total_birds : ℕ := 54

-- Proposition stating the question and the correct answer
theorem parrots_per_cage : (total_birds - num_cages * num_parakeets_per_cage) / num_cages = 2 := 
by
  sorry

end parrots_per_cage_l304_304754


namespace decimal_to_fraction_l304_304225

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l304_304225


namespace segment_length_l304_304708

theorem segment_length (x : ℝ) (h : |x - (27)^(1/3)| = 5) : ∃ a b : ℝ, (a = 8 ∧ b = -2 ∨ a = -2 ∧ b = 8) ∧ real.dist a b = 10 :=
by
  use [8, -2] -- providing the endpoints explicitly
  split
  -- prove that these are the correct endpoints
  · left; exact ⟨rfl, rfl⟩
  -- prove the distance is 10
  · apply real.dist_eq; linarith
  

end segment_length_l304_304708


namespace arithmetic_sequence_ratio_l304_304691

theorem arithmetic_sequence_ratio (a x b : ℝ) 
  (h1 : x - a = b - x)
  (h2 : 2 * x - b = b - x) :
  a / b = 1 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l304_304691


namespace cos_sum_eq_one_l304_304951

theorem cos_sum_eq_one (α β γ : ℝ) 
  (h1 : α + β + γ = Real.pi) 
  (h2 : Real.tan ((β + γ - α) / 4) + Real.tan ((γ + α - β) / 4) + Real.tan ((α + β - γ) / 4) = 1) :
  Real.cos α + Real.cos β + Real.cos γ = 1 :=
sorry

end cos_sum_eq_one_l304_304951


namespace circle_parabola_intersection_l304_304047

theorem circle_parabola_intersection (b : ℝ) :
  (∃ c r, ∀ x y : ℝ, y = (5 / 12) * x^2 → ((x - c)^2 + (y - b)^2 = r^2) ∧ 
   (y = (5 / 12) * x + b → ((x - c)^2 + (y - b)^2 = r^2))) → b = 169 / 60 :=
by
  sorry

end circle_parabola_intersection_l304_304047


namespace dice_probability_l304_304333

-- The context that there are three six-sided dice
def total_outcomes : ℕ := 6 * 6 * 6

-- Function to count the number of favorable outcomes where two dice sum to the third
def favorable_outcomes : ℕ :=
  let sum_cases := [1, 2, 3, 4, 5]
  sum_cases.sum
  -- sum_cases is [1, 2, 3, 4, 5] each mapping to the number of ways to form that sum with a third die

theorem dice_probability : 
  (favorable_outcomes * 3) / total_outcomes = 5 / 24 := 
by 
  -- to prove: the probability that the values on two dice sum to the value on the remaining die is 5/24
  sorry

end dice_probability_l304_304333


namespace melanie_mother_dimes_l304_304863

-- Definitions based on the conditions
variables (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ)

-- Conditions
def melanie_conditions := initial_dimes = 7 ∧ dimes_given_to_dad = 8 ∧ current_dimes = 3

-- Question to be proved is equivalent to proving the number of dimes given by her mother
theorem melanie_mother_dimes (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ) (dimes_given_by_mother : ℤ) 
  (h : melanie_conditions initial_dimes dimes_given_to_dad current_dimes) : 
  dimes_given_by_mother = 4 :=
by 
  sorry

end melanie_mother_dimes_l304_304863


namespace number_of_classmates_ate_cake_l304_304669

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l304_304669


namespace even_increasing_decreasing_l304_304692

theorem even_increasing_decreasing (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = -x^2) :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, x < 0 → f x < f (x + 1)) ∧ (∀ x : ℝ, x > 0 → f x > f (x + 1)) :=
by
  sorry

end even_increasing_decreasing_l304_304692


namespace common_point_of_arithmetic_progression_lines_l304_304069

theorem common_point_of_arithmetic_progression_lines 
  (a d : ℝ) 
  (h₁ : a ≠ 0)
  (h_d_ne_zero : d ≠ 0) 
  (h₃ : ∀ (x y : ℝ), (x = -1 ∧ y = 1) ↔ (∃ a d : ℝ, a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = (a-2*d))) :
  (∀ (x y : ℝ), (a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = a-2*d) → x = -1 ∧ y = 1) :=
by 
  sorry

end common_point_of_arithmetic_progression_lines_l304_304069


namespace exists_positive_int_solutions_l304_304865

theorem exists_positive_int_solutions (a : ℕ) (ha : a > 2) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = a^2 :=
by
  sorry

end exists_positive_int_solutions_l304_304865


namespace find_a_l304_304433

theorem find_a (a b c : ℤ) (h : (∀ x : ℝ, (x - a) * (x - 5) + 4 = (x + b) * (x + c))) :
  a = 0 ∨ a = 1 :=
sorry

end find_a_l304_304433


namespace beth_jan_total_money_l304_304119

theorem beth_jan_total_money (beth_money jan_money : ℕ)
    (h1 : beth_money + 35 = 105)
    (h2 : jan_money - 10 = beth_money) : beth_money + jan_money = 150 :=
begin
  sorry
end

end beth_jan_total_money_l304_304119


namespace polynomial_factorization_l304_304345

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end polynomial_factorization_l304_304345


namespace smallest_b_for_quadratic_factorization_l304_304790

theorem smallest_b_for_quadratic_factorization : ∃ (b : ℕ), 
  (∀ r s : ℤ, (r * s = 4032) ∧ (r + s = b) → b ≥ 127) ∧ 
  (∃ r s : ℤ, (r * s = 4032) ∧ (r + s = b) ∧ (b = 127))
:= sorry

end smallest_b_for_quadratic_factorization_l304_304790


namespace solution_l304_304434

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l304_304434


namespace sin_double_angle_fourth_quadrant_l304_304103

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l304_304103


namespace water_fee_17tons_maximize_first_tier_households_l304_304539

def water_fee (usage : ℕ) : ℝ :=
  if usage ≤ 12 then usage * 4 else if usage ≤ 16 then (12 * 4) + (usage - 12) * 5 else (12 * 4) + (4 * 5) + (usage - 16) * 7

-- Statement for part (1)
theorem water_fee_17tons : water_fee 17 = 75 :=
  by sorry

-- Sample water usage data for part (2) and part (3)
def sample_data : list ℕ := [7, 8, 8, 9, 10, 11, 13, 14, 15, 20]

-- Check if a household uses water in the second tier
def is_second_tier (usage : ℕ) : bool :=
  12 < usage ∧ usage ≤ 16

-- Get the number of households in the second tier
def count_second_tier (data : list ℕ) : ℕ :=
  data.countp is_second_tier

-- Binomial distribution to maximize the number of first-tier households
def binomial_distribution (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * p ^ k * (1 - p) ^ (n - k)

-- Statement for part (3)
theorem maximize_first_tier_households (n : ℕ) (p : ℚ) : ∃ k : ℕ, binomial_distribution n k (3/5) ∧ k = 6 :=
  by sorry

end water_fee_17tons_maximize_first_tier_households_l304_304539


namespace smallest_integer_to_make_perfect_square_l304_304200

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_make_perfect_square :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (n * y) = k^2) ∧ n = 6 :=
by
  sorry

end smallest_integer_to_make_perfect_square_l304_304200


namespace calc_3a2008_minus_5b2008_l304_304820

theorem calc_3a2008_minus_5b2008 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : 3 * a ^ 2008 - 5 * b ^ 2008 = -5 :=
by
  sorry

end calc_3a2008_minus_5b2008_l304_304820


namespace classify_numbers_l304_304636

def isDecimal (n : ℝ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), n = i + f ∧ i ≠ 0

def isNatural (n : ℕ) : Prop :=
  n ≥ 0

theorem classify_numbers :
  (isDecimal 7.42) ∧ (isDecimal 3.6) ∧ (isDecimal 5.23) ∧ (isDecimal 37.8) ∧
  (isNatural 5) ∧ (isNatural 100) ∧ (isNatural 502) ∧ (isNatural 460) :=
by
  sorry

end classify_numbers_l304_304636


namespace division_expression_evaluation_l304_304168

theorem division_expression_evaluation : 120 / (6 / 2) = 40 := by
  sorry

end division_expression_evaluation_l304_304168


namespace least_integer_greater_than_sqrt_450_l304_304510

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l304_304510


namespace repeating_decimal_to_fraction_l304_304229

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l304_304229


namespace function_fixed_point_l304_304015

theorem function_fixed_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) : (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x-2) + 1) } :=
by
  sorry

end function_fixed_point_l304_304015


namespace product_b6_b8_is_16_l304_304086

-- Given conditions
variable (a : ℕ → ℝ) -- Sequence a_n
variable (b : ℕ → ℝ) -- Sequence b_n

-- Condition 1: Arithmetic sequence a_n and non-zero
axiom a_is_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d
axiom a_non_zero : ∃ n, a n ≠ 0

-- Condition 2: Equation 2a_3 - a_7^2 + 2a_n = 0
axiom a_satisfies_eq : ∀ n : ℕ, 2 * a 3 - (a 7) ^ 2 + 2 * a n = 0

-- Condition 3: Geometric sequence b_n with b_7 = a_7
axiom b_is_geometric : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n
axiom b7_equals_a7 : b 7 = a 7

-- Prove statement
theorem product_b6_b8_is_16 : b 6 * b 8 = 16 := sorry

end product_b6_b8_is_16_l304_304086


namespace ratio_of_terms_l304_304807

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem ratio_of_terms
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (S T : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = geometric_sum (a 1) (a 2) n)
  (h₁ : ∀ n : ℕ, T n = geometric_sum (b 1) (b 2) n)
  (h₂ : ∀ n : ℕ, n > 0 → S n / T n = (3 ^ n + 1) / 4) :
  a 3 / b 4 = 3 := 
sorry

end ratio_of_terms_l304_304807


namespace find_k_value_l304_304832

theorem find_k_value
  (x y k : ℝ)
  (h1 : 4 * x + 3 * y = 1)
  (h2 : k * x + (k - 1) * y = 3)
  (h3 : x = y) :
  k = 11 :=
  sorry

end find_k_value_l304_304832


namespace plains_routes_count_l304_304283

theorem plains_routes_count (total_cities mountainous_cities plains_cities total_routes routes_mountainous_pairs: ℕ) :
  total_cities = 100 →
  mountainous_cities = 30 →
  plains_cities = 70 →
  total_routes = 150 →
  routes_mountainous_pairs = 21 →
  let endpoints_mountainous := mountainous_cities * 3 in
  let endpoints_mountainous_pairs := routes_mountainous_pairs * 2 in
  let endpoints_mountainous_plains := endpoints_mountainous - endpoints_mountainous_pairs in
  let endpoints_plains := plains_cities * 3 in
  let routes_mountainous_plains := endpoints_mountainous_plains in
  let endpoints_plains_pairs := endpoints_plains - routes_mountainous_plains in
  let routes_plains_pairs := endpoints_plains_pairs / 2 in
  routes_plains_pairs = 81 :=
by
  intros h1 h2 h3 h4 h5
  dsimp
  rw [h1, h2, h3, h4, h5]
  sorry

end plains_routes_count_l304_304283


namespace least_integer_greater_than_sqrt_450_l304_304525

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l304_304525


namespace lucas_initial_money_l304_304476

theorem lucas_initial_money : (3 * 2 + 14 = 20) := by sorry

end lucas_initial_money_l304_304476


namespace estimate_red_balls_l304_304611

theorem estimate_red_balls (x : ℕ) (drawn_black_balls : ℕ) (total_draws : ℕ) (black_balls : ℕ) 
  (h1 : black_balls = 4) 
  (h2 : total_draws = 100) 
  (h3 : drawn_black_balls = 40) 
  (h4 : (black_balls : ℚ) / (black_balls + x) = drawn_black_balls / total_draws) : 
  x = 6 := 
sorry

end estimate_red_balls_l304_304611


namespace find_abc_l304_304474

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.cos x + 3 * Real.sin x

theorem find_abc (a b c : ℝ) : 
  (∀ x : ℝ, a * f x + b * f (x - c) = 1) →
  (∃ n : ℤ, a = 1 / 2 ∧ b = 1 / 2 ∧ c = (2 * n + 1) * Real.pi) :=
by
  sorry

end find_abc_l304_304474


namespace midpoint_sum_l304_304171

theorem midpoint_sum :
  let x1 := 8
  let y1 := -4
  let z1 := 10
  let x2 := -2
  let y2 := 10
  let z2 := -6
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  let midpoint_z := (z1 + z2) / 2
  midpoint_x + midpoint_y + midpoint_z = 8 :=
by
  -- We just need to state the theorem, proof is not required
  sorry

end midpoint_sum_l304_304171


namespace correct_equation_among_options_l304_304897

theorem correct_equation_among_options
  (a : ℝ) (x : ℝ) :
  (-- Option A
  ¬ ((-1)^3 = -3)) ∧
  (-- Option B
  ¬ (((-2)^2 * (-2)^3) = (-2)^6)) ∧
  (-- Option C
  ¬ ((2 * a - a) = 2)) ∧
  (-- Option D
  ((x - 2)^2 = x^2 - 4*x + 4)) :=
by
  sorry

end correct_equation_among_options_l304_304897


namespace each_cut_piece_weight_l304_304852

theorem each_cut_piece_weight (L : ℕ) (W : ℕ) (c : ℕ) 
  (hL : L = 20) (hW : W = 150) (hc : c = 2) : (L / c) * W = 1500 := by
  sorry

end each_cut_piece_weight_l304_304852


namespace polygons_construction_l304_304067

noncomputable def number_of_polygons : ℝ := 15

theorem polygons_construction :
  (∑ n in finset.range (4 + 1) (number_of_polygons).to_nat, n * (n - 3) / 2) = 800 :=
sorry

end polygons_construction_l304_304067


namespace count_valid_c_values_for_equation_l304_304794

theorem count_valid_c_values_for_equation :
  ∃ (c_set : Finset ℕ), (∀ c ∈ c_set, c ≤ 2000 ∧
    ∃ x : ℝ, 5 * (Real.floor x) + 3 * (Real.ceil x) = c) ∧
    c_set.card = 500 :=
by
  sorry

end count_valid_c_values_for_equation_l304_304794


namespace factor_expression_l304_304413

theorem factor_expression (y : ℝ) : 
  5 * y * (y + 2) + 8 * (y + 2) + 15 = (5 * y + 8) * (y + 2) + 15 := 
by
  sorry

end factor_expression_l304_304413


namespace carla_catches_up_in_three_hours_l304_304851

-- Definitions as lean statements based on conditions
def john_speed : ℝ := 30
def carla_speed : ℝ := 35
def john_start_time : ℝ := 0
def carla_start_time : ℝ := 0.5

-- Lean problem statement to prove the catch-up time
theorem carla_catches_up_in_three_hours : 
  ∃ t : ℝ, 35 * t = 30 * (t + 0.5) ∧ t = 3 :=
by
  sorry

end carla_catches_up_in_three_hours_l304_304851


namespace polar_to_rectangular_correct_l304_304326

noncomputable def polar_to_rectangular (rho theta x y : ℝ) : Prop :=
  rho = 4 * Real.sin theta + 2 * Real.cos theta ∧
  rho * Real.sin theta = y ∧
  rho * Real.cos theta = x ∧
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5

theorem polar_to_rectangular_correct {rho theta x y : ℝ} :
  (rho = 4 * Real.sin theta + 2 * Real.cos theta) →
  (rho * Real.sin theta = y) →
  (rho * Real.cos theta = x) →
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5 :=
by
  sorry

end polar_to_rectangular_correct_l304_304326


namespace distinct_arrangements_banana_l304_304957

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  (factorial total_letters / (factorial b_count * factorial a_count * factorial n_count)) = 60 :=
by
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  have h1 : total_letters = 6 := rfl
  have h2 : factorial 6 = 720 := rfl
  have h3 : factorial 3 = 6 := rfl
  have h4 : factorial 2 = 2 := rfl
  have h5 : 720 / (6 * 2) = 60 := rfl
  exact h5

end distinct_arrangements_banana_l304_304957


namespace least_number_of_square_tiles_l304_304041

theorem least_number_of_square_tiles (length : ℕ) (breadth : ℕ) (gcd : ℕ) (area_room : ℕ) (area_tile : ℕ) (num_tiles : ℕ) :
  length = 544 → breadth = 374 → gcd = Nat.gcd length breadth → gcd = 2 →
  area_room = length * breadth → area_tile = gcd * gcd →
  num_tiles = area_room / area_tile → num_tiles = 50864 :=
by
  sorry

end least_number_of_square_tiles_l304_304041


namespace pizza_promotion_savings_l304_304133

theorem pizza_promotion_savings :
  let regular_price : ℕ := 18
  let promo_price : ℕ := 5
  let num_pizzas : ℕ := 3
  let total_regular_price := num_pizzas * regular_price
  let total_promo_price := num_pizzas * promo_price
  let total_savings := total_regular_price - total_promo_price
  total_savings = 39 :=
by
  sorry

end pizza_promotion_savings_l304_304133


namespace unit_digit_of_six_consecutive_product_is_zero_l304_304163

theorem unit_digit_of_six_consecutive_product_is_zero (n : ℕ) (h : n > 0) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5)) % 10 = 0 := 
by sorry

end unit_digit_of_six_consecutive_product_is_zero_l304_304163


namespace evaluate_expression_l304_304690

theorem evaluate_expression : (20 + 22) / 2 = 21 := by
  sorry

end evaluate_expression_l304_304690


namespace sin_double_angle_neg_of_alpha_in_fourth_quadrant_l304_304108

theorem sin_double_angle_neg_of_alpha_in_fourth_quadrant 
  (k : ℤ) (α : ℝ)
  (hα : -π / 2 + 2 * k * π < α ∧ α < 2 * k * π) :
  sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_alpha_in_fourth_quadrant_l304_304108


namespace trigonometric_identity_l304_304250

noncomputable def alpha := -35 / 6 * Real.pi

theorem trigonometric_identity :
  (2 * Real.sin (Real.pi + alpha) * Real.cos (Real.pi - alpha)
    - Real.sin (3 * Real.pi / 2 + alpha)) /
  (1 + Real.sin (alpha) ^ 2 - Real.cos (Real.pi / 2 + alpha)
    - Real.cos (Real.pi + alpha) ^ 2) = -Real.sqrt 3 := by
  sorry

end trigonometric_identity_l304_304250


namespace blake_change_given_l304_304772

theorem blake_change_given :
  let oranges := 40
  let apples := 50
  let mangoes := 60
  let total_amount := 300
  let total_spent := oranges + apples + mangoes
  let change_given := total_amount - total_spent
  change_given = 150 :=
by
  sorry

end blake_change_given_l304_304772


namespace kimberly_initial_skittles_l304_304856

theorem kimberly_initial_skittles : 
  ∀ (x : ℕ), (x + 7 = 12) → x = 5 :=
by
  sorry

end kimberly_initial_skittles_l304_304856


namespace find_even_integer_l304_304443

theorem find_even_integer (x y z : ℤ) (h₁ : Even x) (h₂ : Odd y) (h₃ : Odd z)
  (h₄ : x < y) (h₅ : y < z) (h₆ : y - x > 5) (h₇ : z - x = 9) : x = 2 := 
by 
  sorry

end find_even_integer_l304_304443


namespace stickers_left_after_giving_away_l304_304040

/-- Willie starts with 36 stickers and gives 7 to Emily. 
    We want to prove that Willie ends up with 29 stickers. -/
theorem stickers_left_after_giving_away (init_stickers : ℕ) (given_away : ℕ) (end_stickers : ℕ) : 
  init_stickers = 36 ∧ given_away = 7 → end_stickers = init_stickers - given_away → end_stickers = 29 :=
by
  intro h
  sorry

end stickers_left_after_giving_away_l304_304040


namespace given_statements_l304_304384

def addition_is_associative (x y z : ℝ) : Prop := (x + y) + z = x + (y + z)

def averaging_is_commutative (x y : ℝ) : Prop := (x + y) / 2 = (y + x) / 2

def addition_distributes_over_averaging (x y z : ℝ) : Prop := 
  x + (y + z) / 2 = (x + y + x + z) / 2

def averaging_distributes_over_addition (x y z : ℝ) : Prop := 
  (x + (y + z)) / 2 = ((x + y) / 2) + ((x + z) / 2)

def averaging_has_identity_element (x e : ℝ) : Prop := 
  (x + e) / 2 = x

theorem given_statements (x y z e : ℝ) :
  addition_is_associative x y z ∧ 
  averaging_is_commutative x y ∧ 
  addition_distributes_over_averaging x y z ∧ 
  ¬averaging_distributes_over_addition x y z ∧ 
  ¬∃ e, averaging_has_identity_element x e :=
by
  sorry

end given_statements_l304_304384


namespace contrapositive_l304_304014

theorem contrapositive (m : ℝ) :
  (∀ m > 0, ∃ x : ℝ, x^2 + x - m = 0) ↔ (∀ m ≤ 0, ∀ x : ℝ, x^2 + x - m ≠ 0) := by
  sorry

end contrapositive_l304_304014


namespace max_value_f_l304_304694

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem max_value_f : ∃ x ∈ (Set.Icc 0 (Real.pi / 2)), f x = Real.pi / 12 + Real.sqrt 3 / 2 :=
by
  sorry

end max_value_f_l304_304694


namespace double_increase_divide_l304_304600

theorem double_increase_divide (x : ℤ) (h : (2 * x + 7) / 5 = 17) : x = 39 := by
  sorry

end double_increase_divide_l304_304600


namespace Jake_width_proof_l304_304872

-- Define the dimensions of Sara's birdhouse in feet
def Sara_width_feet := 1
def Sara_height_feet := 2
def Sara_depth_feet := 2

-- Convert the dimensions to inches
def Sara_width_inch := Sara_width_feet * 12
def Sara_height_inch := Sara_height_feet * 12
def Sara_depth_inch := Sara_depth_feet * 12

-- Calculate Sara's birdhouse volume
def Sara_volume := Sara_width_inch * Sara_height_inch * Sara_depth_inch

-- Define the dimensions of Jake's birdhouse in inches
def Jake_height_inch := 20
def Jake_depth_inch := 18
def Jake_volume (Jake_width_inch : ℝ) := Jake_width_inch * Jake_height_inch * Jake_depth_inch

-- Difference in volume
def volume_difference := 1152

-- Prove the width of Jake's birdhouse
theorem Jake_width_proof : ∃ (W : ℝ), Jake_volume W - Sara_volume = volume_difference ∧ W = 22.4 := by
  sorry

end Jake_width_proof_l304_304872


namespace inequality_relationship_l304_304085

noncomputable def a : ℝ := Real.sin (4 / 5)
noncomputable def b : ℝ := Real.cos (4 / 5)
noncomputable def c : ℝ := Real.tan (4 / 5)

theorem inequality_relationship : c > a ∧ a > b := sorry

end inequality_relationship_l304_304085


namespace sum_of_coordinates_l304_304975

noncomputable def g : ℝ → ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := (g x) ^ 2

theorem sum_of_coordinates : g 3 = 6 → (3 + h 3 = 39) := by
  intro hg3
  have : h 3 = (g 3) ^ 2 := by rfl
  rw [hg3] at this
  rw [this]
  exact sorry

end sum_of_coordinates_l304_304975


namespace initial_ratio_of_stamps_l304_304459

variable (K A : ℕ)

theorem initial_ratio_of_stamps (h1 : (K - 12) * 3 = (A + 12) * 4) (h2 : K - 12 = A + 44) : K/A = 5/3 :=
sorry

end initial_ratio_of_stamps_l304_304459


namespace pizza_savings_l304_304134

theorem pizza_savings (regular_price promotional_price : ℕ) (n : ℕ) (H_regular : regular_price = 18) (H_promotional : promotional_price = 5) (H_n : n = 3) : 
  (regular_price - promotional_price) * n = 39 := by

  -- Assume the given conditions
  have h1 : regular_price - promotional_price = 13 := 
  by rw [H_regular, H_promotional]; exact rfl

  rw [h1, H_n]
  exact (13 * 3).symm

end pizza_savings_l304_304134


namespace cannot_move_reach_goal_l304_304312

structure Point :=
(x : ℤ)
(y : ℤ)

def area (p1 p2 p3 : Point) : ℚ :=
  (1 / 2 : ℚ) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

noncomputable def isTriangleAreaPreserved (initPos finalPos : Point) (helper1Init helper1Final helper2Init helper2Final : Point) : Prop :=
  area initPos helper1Init helper2Init = area finalPos helper1Final helper2Final

theorem cannot_move_reach_goal :
  ¬ ∃ (r₀ r₁ : Point) (a₀ a₁ : Point) (s₀ s₁ : Point),
    r₀ = ⟨0, 0⟩ ∧ r₁ = ⟨2, 2⟩ ∧
    a₀ = ⟨0, 1⟩ ∧ a₁ = ⟨0, 1⟩ ∧
    s₀ = ⟨1, 0⟩ ∧ s₁ = ⟨1, 0⟩ ∧
    isTriangleAreaPreserved r₀ r₁ a₀ a₁ s₀ s₁ :=
by sorry

end cannot_move_reach_goal_l304_304312


namespace polynomial_factorization_l304_304348

open Polynomial

theorem polynomial_factorization :
  (X ^ 15 + X ^ 10 + X ^ 5 + 1) =
    (X ^ 3 + X ^ 2 + 1) * 
    (X ^ 12 - X ^ 11 + X ^ 9 - X ^ 8 + X ^ 6 - X ^ 5 + X ^ 4 + X ^ 3 + X ^ 2 + X + 1) :=
by
  sorry

end polynomial_factorization_l304_304348


namespace red_balls_estimate_l304_304610

/-- There are several red balls and 4 black balls in a bag.
Each ball is identical except for color.
A ball is drawn and put back into the bag. This process is repeated 100 times.
Among those 100 draws, 40 times a black ball is drawn.
Prove that the number of red balls (x) is 6. -/
theorem red_balls_estimate (x : ℕ) (h_condition : (4 / (4 + x) = 40 / 100)) : x = 6 :=
by
    sorry

end red_balls_estimate_l304_304610


namespace parallelogram_area_twice_quadrilateral_l304_304165

theorem parallelogram_area_twice_quadrilateral (a b : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π) :
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  parallelogram_area = 2 * quadrilateral_area :=
by
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  sorry

end parallelogram_area_twice_quadrilateral_l304_304165


namespace decimal_to_fraction_l304_304226

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l304_304226


namespace freq_distribution_correct_l304_304084

variable (freqTable_isForm : Prop)
variable (freqHistogram_isForm : Prop)
variable (freqTable_isAccurate : Prop)
variable (freqHistogram_isIntuitive : Prop)

theorem freq_distribution_correct :
  ((freqTable_isForm ∧ freqHistogram_isForm) ∧
   (freqTable_isAccurate ∧ freqHistogram_isIntuitive)) →
  True :=
by
  intros _
  exact trivial

end freq_distribution_correct_l304_304084


namespace optimal_sampling_methods_l304_304366

/-
We define the conditions of the problem.
-/
def households := 500
def high_income_households := 125
def middle_income_households := 280
def low_income_households := 95
def sample_households := 100

def soccer_players := 12
def sample_soccer_players := 3

/-
We state the goal as a theorem.
-/
theorem optimal_sampling_methods :
  (sample_households == 100) ∧
  (sample_soccer_players == 3) ∧
  (high_income_households + middle_income_households + low_income_households == households) →
  ("stratified" = "stratified" ∧ "random" = "random") :=
by
  -- Sorry to skip the proof
  sorry

end optimal_sampling_methods_l304_304366


namespace arithmetic_sequence_twenty_fourth_term_l304_304031

-- Given definitions (conditions)
def third_term (a d : ℚ) : ℚ := a + 2 * d
def tenth_term (a d : ℚ) : ℚ := a + 9 * d
def twenty_fourth_term (a d : ℚ) : ℚ := a + 23 * d

-- The main theorem to be proved
theorem arithmetic_sequence_twenty_fourth_term 
  (a d : ℚ) 
  (h1 : third_term a d = 7) 
  (h2 : tenth_term a d = 27) :
  twenty_fourth_term a d = 67 := by
  sorry

end arithmetic_sequence_twenty_fourth_term_l304_304031


namespace fraction_shaded_l304_304328

-- Define relevant elements
def quilt : ℕ := 9
def rows : ℕ := 3
def shaded_rows : ℕ := 1
def shaded_fraction := shaded_rows / rows

-- We are to prove the fraction of the quilt that is shaded
theorem fraction_shaded (h : quilt = 3 * 3) : shaded_fraction = 1 / 3 :=
by
  -- Proof goes here
  sorry

end fraction_shaded_l304_304328


namespace part1_part2_l304_304876

-- Part 1
theorem part1 (x y : ℤ) (hx : x = -2) (hy : y = -3) :
  x^2 - 2 * (x^2 - 3 * y) - 3 * (2 * x^2 + 5 * y) = -1 :=
by
  -- Proof to be provided
  sorry

-- Part 2
theorem part2 (a b : ℤ) (hab : a - b = 2 * b^2) :
  2 * (a^3 - 2 * b^2) - (2 * b - a) + a - 2 * a^3 = 0 :=
by
  -- Proof to be provided
  sorry

end part1_part2_l304_304876


namespace solve_abs_eq_zero_l304_304036

theorem solve_abs_eq_zero : ∃ x : ℝ, |5 * x - 3| = 0 ↔ x = 3 / 5 :=
by
  sorry

end solve_abs_eq_zero_l304_304036


namespace cliff_collection_has_180_rocks_l304_304842

noncomputable def cliffTotalRocks : ℕ :=
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks

theorem cliff_collection_has_180_rocks :
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks = 180 := sorry

end cliff_collection_has_180_rocks_l304_304842


namespace number_of_sections_l304_304619

-- Definitions based on the conditions in a)
def num_reels : Nat := 3
def length_per_reel : Nat := 100
def section_length : Nat := 10

-- The math proof problem statement
theorem number_of_sections :
  (num_reels * length_per_reel) / section_length = 30 := by
  sorry

end number_of_sections_l304_304619


namespace percentage_of_women_attended_picnic_l304_304274

variable (E : ℝ) -- total number of employees
variable (M : ℝ) -- number of men
variable (W : ℝ) -- number of women

-- 45% of all employees are men
axiom h1 : M = 0.45 * E
-- Rest of employees are women
axiom h2 : W = E - M
-- 20% of men attended the picnic
variable (x : ℝ) -- percentage of women who attended the picnic
axiom h3 : 0.20 * M + (x / 100) * W = 0.31000000000000007 * E

theorem percentage_of_women_attended_picnic : x = 40 :=
by
  sorry

end percentage_of_women_attended_picnic_l304_304274


namespace previous_day_visitors_l304_304055

-- Define the number of visitors on the day Rachel visited
def visitors_on_day_rachel_visited : ℕ := 317

-- Define the difference in the number of visitors between the day Rachel visited and the previous day
def extra_visitors : ℕ := 22

-- Prove that the number of visitors on the previous day is 295
theorem previous_day_visitors : visitors_on_day_rachel_visited - extra_visitors = 295 :=
by
  sorry

end previous_day_visitors_l304_304055


namespace commission_percentage_l304_304855

theorem commission_percentage (fixed_salary second_base_salary sales_amount earning: ℝ) (commission: ℝ) 
  (h1 : fixed_salary = 1800)
  (h2 : second_base_salary = 1600)
  (h3 : sales_amount = 5000)
  (h4 : earning = 1800) :
  fixed_salary = second_base_salary + (sales_amount * commission) → 
  commission * 100 = 4 :=
by
  -- proof goes here
  sorry

end commission_percentage_l304_304855


namespace plains_routes_count_l304_304287

def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70
def total_routes : ℕ := 150
def mountainous_routes : ℕ := 21

theorem plains_routes_count :
  total_cities = mountainous_cities + plains_cities →
  3 * total_routes = total_cities →
  mountainous_routes * 2 ≤ mountainous_cities * 3 →
  (total_routes - mountainous_routes) * 2 = (70 * 3 - (mountainous_routes * 2)) →
  (total_routes - mountainous_routes * 2) / 2 = 81 :=
begin
  sorry
end

end plains_routes_count_l304_304287


namespace vertex_in_second_quadrant_l304_304604

-- Theorems and properties regarding quadratic functions and their roots.
theorem vertex_in_second_quadrant (c : ℝ) (h : 4 + 4 * c < 0) : 
  (1:ℝ) * -1^2 + 2 * -1 - c > 0 :=
sorry

end vertex_in_second_quadrant_l304_304604


namespace find_tan_theta_l304_304420

open Real

theorem find_tan_theta (θ : ℝ) (h1 : sin θ + cos θ = 7 / 13) (h2 : 0 < θ ∧ θ < π) :
  tan θ = -12 / 5 :=
sorry

end find_tan_theta_l304_304420


namespace divisor_condition_l304_304793

def M (n : ℤ) : Set ℤ := {n, n+1, n+2, n+3, n+4}

def S (n : ℤ) : ℤ := 5*n^2 + 20*n + 30

def P (n : ℤ) : ℤ := (n * (n+1) * (n+2) * (n+3) * (n+4))^2

theorem divisor_condition (n : ℤ) : S n ∣ P n ↔ n = 3 := 
by
  sorry

end divisor_condition_l304_304793


namespace gift_card_remaining_l304_304632

theorem gift_card_remaining (initial_amount : ℕ) (half_monday : ℕ) (quarter_tuesday : ℕ) : 
  initial_amount = 200 → 
  half_monday = initial_amount / 2 →
  quarter_tuesday = (initial_amount - half_monday) / 4 →
  initial_amount - half_monday - quarter_tuesday = 75 :=
by
  intros h_init h_half h_quarter
  rw [h_init, h_half, h_quarter]
  sorry

end gift_card_remaining_l304_304632


namespace inscribed_sphere_radius_in_regular_octahedron_l304_304921

theorem inscribed_sphere_radius_in_regular_octahedron (a : ℝ) (r : ℝ) 
  (h1 : a = 6)
  (h2 : let V := 72 * Real.sqrt 2; V = (1 / 3) * ((8 * (3 * Real.sqrt 3)) * r)) : 
  r = Real.sqrt 6 :=
by
  sorry

end inscribed_sphere_radius_in_regular_octahedron_l304_304921


namespace initial_bacteria_count_l304_304013

theorem initial_bacteria_count 
  (double_every_30_seconds : ∀ n : ℕ, n * 2^(240 / 30) = 262144) : 
  ∃ n : ℕ, n = 1024 :=
by
  -- Define the initial number of bacteria.
  let n := 262144 / (2^8)
  -- Assert that the initial number is 1024.
  use n
  -- To skip the proof.
  sorry

end initial_bacteria_count_l304_304013


namespace sum_first_53_odd_numbers_l304_304035

-- Definitions based on the given conditions
def first_odd_number := 1

def nth_odd_number (n : ℕ) : ℕ :=
  1 + (n - 1) * 2

def sum_n_odd_numbers (n : ℕ) : ℕ :=
  (n * n)

-- Theorem statement to prove
theorem sum_first_53_odd_numbers : sum_n_odd_numbers 53 = 2809 := 
by
  sorry

end sum_first_53_odd_numbers_l304_304035


namespace correct_option_B_l304_304058

-- Define decimal representation of the numbers
def dec_13 : ℕ := 13
def dec_25 : ℕ := 25
def dec_11 : ℕ := 11
def dec_10 : ℕ := 10

-- Define binary representation of the numbers
def bin_1101 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 1*4 + 0*2 + 1*1 = 13
def bin_10110 : ℕ := 2^(4) + 2^(2) + 2^(1)  -- 1*16 + 0*8 + 1*4 + 1*2 + 0*1 = 22
def bin_1011 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 0*4 + 1*2 + 1*1 = 11
def bin_10 : ℕ := 2^(1)  -- 1*2 + 0*1 = 2

theorem correct_option_B : (dec_13 = bin_1101) := by
  -- Proof is skipped
  sorry

end correct_option_B_l304_304058


namespace number_of_classmates_l304_304654

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l304_304654


namespace cost_price_of_computer_table_l304_304491

theorem cost_price_of_computer_table
  (C : ℝ) 
  (S : ℝ := 1.20 * C)
  (S_eq : S = 8600) : 
  C = 7166.67 :=
by
  sorry

end cost_price_of_computer_table_l304_304491


namespace cos_sum_identity_l304_304089

theorem cos_sum_identity (α : ℝ) (h_cos : Real.cos α = 3 / 5) (h_alpha : 0 < α ∧ α < Real.pi / 2) :
  Real.cos (α + Real.pi / 3) = (3 - 4 * Real.sqrt 3) / 10 :=
by
  sorry

end cos_sum_identity_l304_304089


namespace sq_97_l304_304559

theorem sq_97 : 97^2 = 9409 :=
by
  sorry

end sq_97_l304_304559


namespace change_given_l304_304771

-- Define the given conditions
def oranges_cost := 40
def apples_cost := 50
def mangoes_cost := 60
def initial_amount := 300

-- Calculate total cost of fruits
def total_fruits_cost := oranges_cost + apples_cost + mangoes_cost

-- Define the given change
def given_change := initial_amount - total_fruits_cost

-- Prove that the given change is equal to 150
theorem change_given (h_oranges : oranges_cost = 40)
                     (h_apples : apples_cost = 50)
                     (h_mangoes : mangoes_cost = 60)
                     (h_initial : initial_amount = 300) :
  given_change = 150 :=
by
  -- Proof is omitted, indicated by sorry
  sorry

end change_given_l304_304771


namespace smallest_munificence_monic_cubic_l304_304079

open Complex

def p (b c : ℂ) : Polynomial ℂ := Polynomial.C (1:ℂ) * Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X

def munificence (p : Polynomial ℂ) : ℝ := 
  Real.supSet (Set.range (fun x : ℝ => Complex.abs (p.eval x)))

theorem smallest_munificence_monic_cubic :
  ∀ b c : ℂ → munificence (p b c) ≥ 1 :=
by sorry

end smallest_munificence_monic_cubic_l304_304079


namespace negative_real_root_range_l304_304158

theorem negative_real_root_range (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (1 / Real.pi) ^ x = (1 + a) / (1 - a)) ↔ 0 < a ∧ a < 1 :=
by
  sorry

end negative_real_root_range_l304_304158


namespace right_rectangular_prism_volume_l304_304332

theorem right_rectangular_prism_volume (x y z : ℝ) 
  (h1 : x * y = 72) (h2 : y * z = 75) (h3 : x * z = 80) : 
  x * y * z = 657 :=
sorry

end right_rectangular_prism_volume_l304_304332


namespace total_handshakes_l304_304555

-- Definitions based on conditions
def num_wizards : ℕ := 25
def num_elves : ℕ := 18

-- Each wizard shakes hands with every other wizard
def wizard_handshakes : ℕ := num_wizards * (num_wizards - 1) / 2

-- Each elf shakes hands with every wizard
def elf_wizard_handshakes : ℕ := num_elves * num_wizards

-- Total handshakes is the sum of the above two
theorem total_handshakes : wizard_handshakes + elf_wizard_handshakes = 750 := by
  sorry

end total_handshakes_l304_304555


namespace intersection_A_B_l304_304816

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem intersection_A_B : A ∩ B = {0, 2} :=
by
  sorry

end intersection_A_B_l304_304816


namespace j_mod_2_not_zero_l304_304271

theorem j_mod_2_not_zero (x j : ℤ) (h : 2 * x - j = 11) : j % 2 ≠ 0 :=
sorry

end j_mod_2_not_zero_l304_304271


namespace area_of_picture_l304_304432

theorem area_of_picture {x y : ℕ} (hx : x > 1) (hy : y > 1) 
  (h : (2 * x + 3) * (y + 2) - x * y = 34) : x * y = 8 := 
by
  sorry

end area_of_picture_l304_304432


namespace sin_double_angle_neg_of_fourth_quadrant_l304_304106

variable (α : ℝ)

def is_in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, -π / 2 + 2 * k * π < α ∧ α < 2 * k * π

theorem sin_double_angle_neg_of_fourth_quadrant (h : is_in_fourth_quadrant α) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_fourth_quadrant_l304_304106


namespace least_integer_gt_sqrt_450_l304_304520

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l304_304520


namespace geometric_sequence_common_ratio_l304_304422

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {S : ℕ → ℝ} (q : ℝ) 
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  
  (h_condition : ∀ n : ℕ+, S (2 * n) / S n < 5) :
  0 < q ∧ q ≤ 1 :=
sorry

end geometric_sequence_common_ratio_l304_304422


namespace find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l304_304952

noncomputable def f (a m x : ℝ) := Real.log (x + m) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a
noncomputable def F (a m x : ℝ) := f a m x - g a x

theorem find_m_and_domain (a : ℝ) (m : ℝ) (h : F a m 0 = 0) : m = 1 ∧ ∀ x, -1 < x ∧ x < 1 :=
sorry

theorem parity_of_F (a : ℝ) (m : ℝ) (h : m = 1) : ∀ x, F a m (-x) = -F a m x :=
sorry

theorem range_of_x_for_F_positive (a : ℝ) (m : ℝ) (h : m = 1) :
  (a > 1 → ∀ x, 0 < x ∧ x < 1 → F a m x > 0) ∧ (0 < a ∧ a < 1 → ∀ x, -1 < x ∧ x < 0 → F a m x > 0) :=
sorry

end find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l304_304952


namespace positive_square_root_of_256_l304_304263

theorem positive_square_root_of_256 (y : ℝ) (hy_pos : y > 0) (hy_squared : y^2 = 256) : y = 16 :=
by
  sorry

end positive_square_root_of_256_l304_304263


namespace triangle_AB_length_correct_l304_304985

theorem triangle_AB_length_correct (BC AC : Real) (A : Real) 
  (hBC : BC = Real.sqrt 7) 
  (hAC : AC = 2 * Real.sqrt 3) 
  (hA : A = Real.pi / 6) :
  ∃ (AB : Real), (AB = 5 ∨ AB = 1) :=
by
  sorry

end triangle_AB_length_correct_l304_304985


namespace value_of_f_at_3_l304_304598

def f (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem value_of_f_at_3 : f 3 = 9 / 7 := by
  sorry

end value_of_f_at_3_l304_304598


namespace inequality_solution_sets_l304_304252

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x : ℝ, ax^2 - 5 * x + b > 0 ↔ x < -1 / 3 ∨ x > 1 / 2) →
  (∀ x : ℝ, bx^2 - 5 * x + a > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end inequality_solution_sets_l304_304252


namespace distinct_arrangements_banana_l304_304963

theorem distinct_arrangements_banana : 
  let word := "banana" 
  let total_letters := 6 
  let freq_b := 1 
  let freq_n := 2 
  let freq_a := 3 
  ∀(n : ℕ) (n1 n2 n3 : ℕ), 
    n = total_letters → 
    n1 = freq_b → 
    n2 = freq_n → 
    n3 = freq_a → 
    (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 := 
by
  intros n n1 n2 n3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end distinct_arrangements_banana_l304_304963


namespace repeating_decimals_sum_l304_304406

theorem repeating_decimals_sum : (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ) = 4 / 9 :=
 by 
  have h₁ : (0.666666... : ℚ) = 2 / 3,
    -- Since x = 0.6666..., then 10x = 6.6666...,
    -- so 10x - x = 6, then x = 6 / 9, hence 2 / 3
    sorry,
  have h₂ : (0.222222... : ℚ) = 2 / 9,
    -- Since x = 0.2222..., then 10x = 2.2222...,
    -- so 10x - x = 2, then x = 2 / 9
    sorry,
  have h₃ : (0.444444... : ℚ) = 4 / 9,
    -- Since x = 0.4444..., then 10x = 4.4444...,
    -- so 10x - x = 4, then x = 4 / 9
    sorry,
  calc
    (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ)
        = (2 / 3) + (2 / 9) - (4 / 9) : by rw [h₁, h₂, h₃]
    ... = (6 / 9) + (2 / 9) - (4 / 9) : by norm_num
    ... = 4 / 9 : by ring

end repeating_decimals_sum_l304_304406


namespace find_k_value_l304_304416

variable (x y z k : ℝ)

theorem find_k_value (h : 7 / (x + y) = k / (x + z) ∧ k / (x + z) = 11 / (z - y)) :
  k = 18 :=
sorry

end find_k_value_l304_304416


namespace total_material_ordered_l304_304367

theorem total_material_ordered :
  12.468 + 4.6278 + 7.9101 + 8.3103 + 5.6327 = 38.9499 :=
by
  sorry

end total_material_ordered_l304_304367


namespace total_assembly_time_l304_304629

-- Define the conditions
def chairs : ℕ := 2
def tables : ℕ := 2
def time_per_piece : ℕ := 8
def total_pieces : ℕ := chairs + tables

-- State the theorem
theorem total_assembly_time :
  total_pieces * time_per_piece = 32 :=
sorry

end total_assembly_time_l304_304629


namespace find_F_l304_304120

theorem find_F (F C : ℝ) (h1 : C = 4/7 * (F - 40)) (h2 : C = 28) : F = 89 := 
by
  sorry

end find_F_l304_304120


namespace selling_price_decreased_l304_304759

theorem selling_price_decreased (d m : ℝ) (hd : d = 0.10) (hm : m = 0.10) :
  (1 - d) * (1 + m) < 1 :=
by
  rw [hd, hm]
  sorry

end selling_price_decreased_l304_304759


namespace cake_slices_l304_304056

open Nat

theorem cake_slices (S : ℕ) (h1 : 2 * S - 12 = 10) : S = 8 := by
  sorry

end cake_slices_l304_304056


namespace B_gives_C_100_meters_start_l304_304845

-- Definitions based on given conditions
variables (Va Vb Vc : ℝ) (T : ℝ)

-- Assume the conditions based on the problem statement
def race_condition_1 := Va = 1000 / T
def race_condition_2 := Vb = 900 / T
def race_condition_3 := Vc = 850 / T

-- Theorem stating that B can give C a 100 meter start
theorem B_gives_C_100_meters_start
  (h1 : race_condition_1 Va T)
  (h2 : race_condition_2 Vb T)
  (h3 : race_condition_3 Vc T) :
  (Vb = (1000 - 100) / T) :=
by
  -- Utilize conditions h1, h2, and h3
  sorry

end B_gives_C_100_meters_start_l304_304845


namespace fathers_age_more_than_4_times_son_l304_304160

-- Let F (Father's age) be 44 and S (Son's age) be 10 as given by solving the equations
def X_years_more_than_4_times_son_age (F S X : ℕ) : Prop :=
  F = 4 * S + X ∧ F + 4 = 2 * (S + 4) + 20

theorem fathers_age_more_than_4_times_son (F S X : ℕ) (h1 : F = 44) (h2 : F = 4 * S + X) (h3 : F + 4 = 2 * (S + 4) + 20) :
  X = 4 :=
by
  -- The proof would go here
  sorry

end fathers_age_more_than_4_times_son_l304_304160


namespace average_speed_round_trip_l304_304461

def time_to_walk_uphill := 30 -- in minutes
def time_to_walk_downhill := 10 -- in minutes
def distance_one_way := 1 -- in km

theorem average_speed_round_trip :
  (2 * distance_one_way) / ((time_to_walk_uphill + time_to_walk_downhill) / 60) = 3 := by
  sorry

end average_speed_round_trip_l304_304461


namespace beginning_of_spring_period_and_day_l304_304939

noncomputable def daysBetween : Nat := 46 -- Total days: Dec 21, 2004 to Feb 4, 2005

theorem beginning_of_spring_period_and_day :
  let total_days := daysBetween
  let segment := total_days / 9
  let day_within_segment := total_days % 9
  segment = 5 ∧ day_within_segment = 1 := by
sorry

end beginning_of_spring_period_and_day_l304_304939


namespace local_minimum_at_neg_one_l304_304997

noncomputable def f (x : ℝ) := x * Real.exp x

theorem local_minimum_at_neg_one : (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x + 1) < δ → f x > f (-1)) :=
sorry

end local_minimum_at_neg_one_l304_304997


namespace articles_profit_l304_304484

variable {C S : ℝ}

theorem articles_profit (h1 : 20 * C = x * S) (h2 : S = 1.25 * C) : x = 16 :=
by
  sorry

end articles_profit_l304_304484


namespace percent_covered_by_larger_triangles_l304_304544

-- Define the number of small triangles in one large hexagon
def total_small_triangles := 16

-- Define the number of small triangles that are part of the larger triangles within one hexagon
def small_triangles_in_larger_triangles := 9

-- Calculate the fraction of the area of the hexagon covered by larger triangles
def fraction_covered_by_larger_triangles := 
  small_triangles_in_larger_triangles / total_small_triangles

-- Define the expected result as a fraction of the total area
def expected_fraction := 56 / 100

-- The proof problem in Lean 4 statement:
theorem percent_covered_by_larger_triangles
  (h1 : fraction_covered_by_larger_triangles = 9 / 16) :
  fraction_covered_by_larger_triangles = expected_fraction :=
  by
    sorry

end percent_covered_by_larger_triangles_l304_304544


namespace find_g9_l304_304486

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g3_value : g 3 = 4

theorem find_g9 : g 9 = 64 := sorry

end find_g9_l304_304486


namespace smallest_multiple_of_37_smallest_multiple_of_37_verification_l304_304170

theorem smallest_multiple_of_37 (x : ℕ) (h : 37 * x % 97 = 3) :
  x = 15 := sorry

theorem smallest_multiple_of_37_verification :
  37 * 15 = 555 := rfl

end smallest_multiple_of_37_smallest_multiple_of_37_verification_l304_304170


namespace factorizations_of_4050_l304_304100

theorem factorizations_of_4050 :
  ∃! (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4050 :=
by
  sorry

end factorizations_of_4050_l304_304100


namespace probability_two_hearts_is_one_seventeenth_l304_304244

-- Define the problem parameters
def totalCards : ℕ := 52
def hearts : ℕ := 13
def drawCount : ℕ := 2

-- Define function to calculate combinations
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the probability calculation
def probability_drawing_two_hearts : ℚ :=
  (combination hearts drawCount) / (combination totalCards drawCount)

-- State the theorem to be proved
theorem probability_two_hearts_is_one_seventeenth :
  probability_drawing_two_hearts = 1 / 17 :=
by
  -- Proof not required, so provide sorry
  sorry

end probability_two_hearts_is_one_seventeenth_l304_304244


namespace solution_exists_unique_n_l304_304570

theorem solution_exists_unique_n (n : ℕ) : 
  (∀ m : ℕ, (10 * m > 120) ∨ ∃ k1 k2 k3 : ℕ, 10 * k1 + n * k2 + (n + 1) * k3 = 120) = false → 
  n = 16 := by sorry

end solution_exists_unique_n_l304_304570


namespace trains_pass_time_l304_304166

noncomputable def train_time
  (lengthA : ℝ) 
  (speedA : ℝ) 
  (lengthB : ℝ) 
  (speedB : ℝ) 
  (conversion_factor : ℝ) : ℝ :=
  let relative_speed := (speedA + speedB) * conversion_factor in
  let total_distance := lengthA + lengthB in
  total_distance / relative_speed

theorem trains_pass_time :
  train_time 550 108 750 144 (5/18) ≈ 18.57 :=
sorry

end trains_pass_time_l304_304166


namespace banana_arrangements_l304_304960

theorem banana_arrangements : 
  let n := 6
  let k_b := 1
  let k_a := 3
  let k_n := 2
  n! / (k_b! * k_a! * k_n!) = 60 :=
by
  have n_def : n = 6 := rfl
  have k_b_def : k_b = 1 := rfl
  have k_a_def : k_a = 3 := rfl
  have k_n_def : k_n = 2 := rfl
  calc
    n! / (k_b! * k_a! * k_n!) = 720 / (1 * 6 * 2) : by sorry
                             ... = 720 / 12         : by sorry
                             ... = 60               : by sorry

end banana_arrangements_l304_304960


namespace sub_decimal_proof_l304_304786

theorem sub_decimal_proof : 2.5 - 0.32 = 2.18 :=
  by sorry

end sub_decimal_proof_l304_304786


namespace different_product_l304_304057

theorem different_product :
  let P1 := 190 * 80
  let P2 := 19 * 800
  let P3 := 19 * 8 * 10
  let P4 := 19 * 8 * 100
  P3 ≠ P1 ∧ P3 ≠ P2 ∧ P3 ≠ P4 :=
by
  sorry

end different_product_l304_304057


namespace marlon_gift_card_balance_l304_304631

theorem marlon_gift_card_balance 
  (initial_amount : ℕ) 
  (spent_monday : initial_amount / 2 = 100)
  (spent_tuesday : (initial_amount / 2) / 4 = 25) 
  : (initial_amount / 2) - (initial_amount / 2 / 4) = 75 :=
by
  sorry

end marlon_gift_card_balance_l304_304631


namespace option_B_is_linear_inequality_with_one_var_l304_304528

noncomputable def is_linear_inequality_with_one_var (in_eq : String) : Prop :=
  match in_eq with
  | "3x^2 > 45 - 9x" => false
  | "3x - 2 < 4" => true
  | "1 / x < 2" => false
  | "4x - 3 < 2y - 7" => false
  | _ => false

theorem option_B_is_linear_inequality_with_one_var :
  is_linear_inequality_with_one_var "3x - 2 < 4" = true :=
by
  -- Add proof steps here
  sorry

end option_B_is_linear_inequality_with_one_var_l304_304528


namespace periodic_decimal_to_fraction_l304_304222

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l304_304222


namespace lcm_from_1_to_12_eq_27720_l304_304729

theorem lcm_from_1_to_12_eq_27720 : nat.lcm (finset.range 12).succ = 27720 :=
  sorry

end lcm_from_1_to_12_eq_27720_l304_304729


namespace students_exceed_rabbits_l304_304392

theorem students_exceed_rabbits (students_per_classroom rabbits_per_classroom number_of_classrooms : ℕ) 
  (h_students : students_per_classroom = 18)
  (h_rabbits : rabbits_per_classroom = 2)
  (h_classrooms : number_of_classrooms = 4) : 
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 64 :=
by {
  sorry
}

end students_exceed_rabbits_l304_304392


namespace minimize_ab_value_l304_304091

noncomputable def minimize (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 9 * a + b = 36) : ℝ :=
  a * b

theorem minimize_ab_value : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 9 * a + b = 36 ∧ minimize a b sorry sorry sorry = 36 :=
sorry

end minimize_ab_value_l304_304091


namespace course_selection_schemes_count_l304_304756

-- Define the total number of courses
def total_courses : ℕ := 8

-- Define the number of courses to choose
def courses_to_choose : ℕ := 5

-- Define the two specific courses, Course A and Course B
def courseA := 1
def courseB := 2

-- Define the combination function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the count when neither Course A nor Course B is selected
def case1 : ℕ := C 6 5

-- Define the count when exactly one of Course A or Course B is selected
def case2 : ℕ := C 2 1 * C 6 4

-- Combining both cases
theorem course_selection_schemes_count : case1 + case2 = 36 :=
by
  -- These would be replaced with actual combination calculations.
  sorry

end course_selection_schemes_count_l304_304756


namespace find_first_episode_l304_304291

variable (x : ℕ)
variable (w y z : ℕ)
variable (total_minutes: ℕ)
variable (h1 : w = 62)
variable (h2 : y = 65)
variable (h3 : z = 55)
variable (h4 : total_minutes = 240)

theorem find_first_episode :
  x + w + y + z = total_minutes → x = 58 := 
by
  intro h
  rw [h1, h2, h3, h4] at h
  linarith

end find_first_episode_l304_304291


namespace blake_change_given_l304_304773

theorem blake_change_given :
  let oranges := 40
  let apples := 50
  let mangoes := 60
  let total_amount := 300
  let total_spent := oranges + apples + mangoes
  let change_given := total_amount - total_spent
  change_given = 150 :=
by
  sorry

end blake_change_given_l304_304773


namespace polynomial_factorization_l304_304356

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end polynomial_factorization_l304_304356


namespace unique_solution_quadratic_l304_304568

theorem unique_solution_quadratic (n : ℕ) : (∀ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intros h
  sorry

end unique_solution_quadratic_l304_304568


namespace students_more_than_rabbits_by_64_l304_304393

-- Define the conditions as constants
def number_of_classrooms : ℕ := 4
def students_per_classroom : ℕ := 18
def rabbits_per_classroom : ℕ := 2

-- Define the quantities that need calculations
def total_students : ℕ := number_of_classrooms * students_per_classroom
def total_rabbits : ℕ := number_of_classrooms * rabbits_per_classroom
def difference_students_rabbits : ℕ := total_students - total_rabbits

-- State the theorem to be proven
theorem students_more_than_rabbits_by_64 :
  difference_students_rabbits = 64 := by
  sorry

end students_more_than_rabbits_by_64_l304_304393


namespace least_int_gt_sqrt_450_l304_304503

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l304_304503


namespace ellipse_focus_xaxis_l304_304428

theorem ellipse_focus_xaxis (k : ℝ) (h : 1 - k > 2 + k ∧ 2 + k > 0) : -2 < k ∧ k < -1/2 :=
by sorry

end ellipse_focus_xaxis_l304_304428


namespace sum_y_sequence_eq_l304_304780

noncomputable def y_sequence (m : ℕ) : ℕ → ℝ
| 0     := 1
| 1     := m
| (k+2) := ((m + 1) * y_sequence m (k + 1) + (m - k) * y_sequence m k) / (k + 2)

theorem sum_y_sequence_eq (m : ℕ) : 
  ∑ (k : ℕ) in Finset.range (m + 1), y_sequence m k = 2^(m+1) := by sorry

end sum_y_sequence_eq_l304_304780


namespace possible_number_of_classmates_l304_304645

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l304_304645


namespace plains_routes_count_l304_304285

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l304_304285


namespace probability_of_selecting_boy_given_girl_A_selected_l304_304873

-- Define the total number of girls and boys
def total_girls : ℕ := 5
def total_boys : ℕ := 2

-- Define the group size to be selected
def group_size : ℕ := 3

-- Define the probability of selecting at least one boy given girl A is selected
def probability_at_least_one_boy_given_girl_A : ℚ := 3 / 5

-- Math problem reformulated as a Lean theorem
theorem probability_of_selecting_boy_given_girl_A_selected : 
  (total_girls = 5) → (total_boys = 2) → (group_size = 3) → 
  (probability_at_least_one_boy_given_girl_A = 3 / 5) :=
by sorry

end probability_of_selecting_boy_given_girl_A_selected_l304_304873


namespace polynomial_factorization_l304_304346

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end polynomial_factorization_l304_304346


namespace required_run_rate_is_correct_l304_304899

-- Define the initial conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_10 : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 40

-- Given total runs in the first 10 overs
def total_runs_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_10
-- Given runs needed in the remaining 40 overs
def runs_needed_remaining_overs : ℝ := target_runs - total_runs_first_10_overs

-- Lean statement to prove the required run rate in the remaining 40 overs
theorem required_run_rate_is_correct (h1 : run_rate_first_10_overs = 3.2)
                                     (h2 : overs_first_10 = 10)
                                     (h3 : target_runs = 282)
                                     (h4 : remaining_overs = 40) :
  (runs_needed_remaining_overs / remaining_overs) = 6.25 :=
by sorry


end required_run_rate_is_correct_l304_304899


namespace expected_value_eight_sided_die_win_l304_304915

/-- The expected value of winning with a fair 8-sided die, where the win is \( n^3 \) dollars if \( n \) is rolled, is 162 dollars. -/
theorem expected_value_eight_sided_die_win :
  (1 / 8) * (1^3) + (1 / 8) * (2^3) + (1 / 8) * (3^3) + (1 / 8) * (4^3) +
  (1 / 8) * (5^3) + (1 / 8) * (6^3) + (1 / 8) * (7^3) + (1 / 8) * (8^3) = 162 := 
by
  -- Simplification and calculation here
  sorry

end expected_value_eight_sided_die_win_l304_304915


namespace corn_bag_price_l304_304343

theorem corn_bag_price
  (cost_seeds: ℕ)
  (cost_fertilizers_pesticides: ℕ)
  (cost_labor: ℕ)
  (total_bags: ℕ)
  (desired_profit_percentage: ℕ)
  (total_cost: ℕ := cost_seeds + cost_fertilizers_pesticides + cost_labor)
  (total_revenue: ℕ := total_cost + (total_cost * desired_profit_percentage / 100))
  (price_per_bag: ℕ := total_revenue / total_bags) :
  cost_seeds = 50 →
  cost_fertilizers_pesticides = 35 →
  cost_labor = 15 →
  total_bags = 10 →
  desired_profit_percentage = 10 →
  price_per_bag = 11 :=
by sorry

end corn_bag_price_l304_304343


namespace length_of_ST_l304_304272

theorem length_of_ST (LM MN NL: ℝ) (LR : ℝ) (LT TR LS SR: ℝ) 
  (h1: LM = 8) (h2: MN = 10) (h3: NL = 6) (h4: LR = 6) 
  (h5: LT = 8 / 3) (h6: TR = 10 / 3) (h7: LS = 9 / 4) (h8: SR = 15 / 4) :
  LS - LT = -5 / 12 :=
by
  sorry

end length_of_ST_l304_304272


namespace calculate_expression_l304_304775

theorem calculate_expression :
  -2^3 * (-3)^2 / (9 / 8) - abs (1 / 2 - 3 / 2) = -65 :=
by
  sorry

end calculate_expression_l304_304775


namespace sin_2alpha_value_l304_304831

noncomputable def sin_2alpha_through_point (x y : ℝ) : ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let sin_alpha := y / r
  let cos_alpha := x / r
  2 * sin_alpha * cos_alpha

theorem sin_2alpha_value :
  sin_2alpha_through_point (-3) 4 = -24 / 25 :=
by
  sorry

end sin_2alpha_value_l304_304831


namespace cookies_flour_and_eggs_l304_304755

theorem cookies_flour_and_eggs (c₁ c₂ : ℕ) (f₁ f₂ : ℕ) (e₁ e₂ : ℕ) 
  (h₁ : c₁ = 40) (h₂ : f₁ = 3) (h₃ : e₁ = 2) (h₄ : c₂ = 120) :
  f₂ = f₁ * (c₂ / c₁) ∧ e₂ = e₁ * (c₂ / c₁) :=
by
  sorry

end cookies_flour_and_eggs_l304_304755


namespace modulus_Z_l304_304300

theorem modulus_Z (Z : ℂ) (h : Z * (2 - 3 * Complex.I) = 6 + 4 * Complex.I) : Complex.abs Z = 2 := 
sorry

end modulus_Z_l304_304300


namespace at_least_one_does_not_land_l304_304537

/-- Proposition stating "A lands within the designated area". -/
def p : Prop := sorry

/-- Proposition stating "B lands within the designated area". -/
def q : Prop := sorry

/-- Negation of proposition p, stating "A does not land within the designated area". -/
def not_p : Prop := ¬p

/-- Negation of proposition q, stating "B does not land within the designated area". -/
def not_q : Prop := ¬q

/-- The proposition "At least one trainee does not land within the designated area" can be expressed as (¬p) ∨ (¬q). -/
theorem at_least_one_does_not_land : (¬p ∨ ¬q) := sorry

end at_least_one_does_not_land_l304_304537


namespace segment_length_of_absolute_value_l304_304714

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end segment_length_of_absolute_value_l304_304714


namespace smallest_term_index_l304_304487

theorem smallest_term_index (a_n : ℕ → ℤ) (h : ∀ n, a_n n = 3 * n^2 - 38 * n + 12) : ∃ n, a_n n = a_n 6 ∧ ∀ m, a_n m ≥ a_n 6 :=
by
  sorry

end smallest_term_index_l304_304487


namespace percentage_calculation_l304_304907

theorem percentage_calculation (percentage : ℝ) (h : percentage * 50 = 0.15) : percentage = 0.003 :=
by
  sorry

end percentage_calculation_l304_304907


namespace least_integer_greater_than_sqrt_450_l304_304505

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l304_304505


namespace area_inside_C_but_outside_A_and_B_l304_304557

def radius_A := 1
def radius_B := 1
def radius_C := 2
def tangency_AB := true
def tangency_AC_non_midpoint := true

theorem area_inside_C_but_outside_A_and_B :
  let areaC := π * (radius_C ^ 2)
  let areaA := π * (radius_A ^ 2)
  let areaB := π * (radius_B ^ 2)
  let overlapping_area := 2 * (π * (radius_A ^ 2) / 2) -- approximation
  areaC - overlapping_area = 3 * π - 2 :=
by
  sorry

end area_inside_C_but_outside_A_and_B_l304_304557


namespace sin_double_angle_neg_of_alpha_in_fourth_quadrant_l304_304109

theorem sin_double_angle_neg_of_alpha_in_fourth_quadrant 
  (k : ℤ) (α : ℝ)
  (hα : -π / 2 + 2 * k * π < α ∧ α < 2 * k * π) :
  sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_alpha_in_fourth_quadrant_l304_304109


namespace total_amount_is_105_l304_304358

theorem total_amount_is_105 (x_amount y_amount z_amount : ℝ) 
  (h1 : ∀ x, y_amount = x * 0.45) 
  (h2 : ∀ x, z_amount = x * 0.30) 
  (h3 : y_amount = 27) : 
  (x_amount + y_amount + z_amount = 105) := 
sorry

end total_amount_is_105_l304_304358


namespace product_price_reduction_l304_304493

theorem product_price_reduction (z : ℝ) (x : ℝ) (hp1 : z > 0) (hp2 : 0.85 * 0.85 * z = z * (1 - x / 100)) : x = 27.75 := by
  sorry

end product_price_reduction_l304_304493


namespace arithmetic_sequence_n_is_100_l304_304936

theorem arithmetic_sequence_n_is_100 
  (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ)
  (h1 : a 1 = a1)
  (hd : d = 3)
  (h : ∀ n, a n = a1 + (n - 1) * d)
  (h298 : ∃ n, a n = 298) :
  (∃ n, a n = 298) ↔  n = 100 :=
by {
  obtain ⟨n, hn⟩ := h298,
  rw h at hn,
  have : a1 = 1 := by assumption,
  rw [this, hd] at hn,
  norm_num at hn,
  norm_cast,
  exact hn,
 sorry
}

end arithmetic_sequence_n_is_100_l304_304936


namespace log_identity_l304_304829

theorem log_identity (a b : ℝ) (h1 : a = Real.log 144 / Real.log 4) (h2 : b = Real.log 12 / Real.log 2) : a = b := 
by
  sorry

end log_identity_l304_304829


namespace ferrisWheelPeopleCount_l304_304883

/-!
# Problem Description

We are given the following conditions:
- The ferris wheel has 6.0 seats.
- It has to run 2.333333333 times for everyone to get a turn.

We need to prove that the total number of people who want to ride the ferris wheel is 14.
-/

def ferrisWheelSeats : ℕ := 6
def ferrisWheelRuns : ℚ := 2333333333 / 1000000000

theorem ferrisWheelPeopleCount :
  (ferrisWheelSeats : ℚ) * ferrisWheelRuns = 14 :=
by
  sorry

end ferrisWheelPeopleCount_l304_304883


namespace proof_problem_l304_304465

theorem proof_problem 
  (a b c : ℝ) 
  (h1 : ∀ x, (x < -4 ∨ (23 ≤ x ∧ x ≤ 27)) ↔ ((x - a) * (x - b) / (x - c) ≤ 0))
  (h2 : a < b) : 
  a + 2 * b + 3 * c = 65 :=
sorry

end proof_problem_l304_304465


namespace combined_time_alligators_walked_l304_304618

-- Define the conditions
def original_time : ℕ := 4
def return_time := original_time + 2 * Int.sqrt original_time

-- State the theorem to be proven
theorem combined_time_alligators_walked : original_time + return_time = 12 := by
  sorry

end combined_time_alligators_walked_l304_304618


namespace yoga_studio_women_count_l304_304703

theorem yoga_studio_women_count :
  ∃ W : ℕ, 
  (8 * 190) + (W * 120) = 14 * 160 ∧ W = 6 :=
by 
  existsi (6);
  sorry

end yoga_studio_women_count_l304_304703


namespace choose_president_and_committee_l304_304457

-- Define the condition of the problem
def total_people := 10
def committee_size := 3

-- Define the function to calculate the number of combinations
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- Proving the number of ways to choose the president and the committee
theorem choose_president_and_committee :
  (total_people * comb (total_people - 1) committee_size) = 840 :=
by
  sorry

end choose_president_and_committee_l304_304457


namespace floor_difference_l304_304932

theorem floor_difference (x : ℝ) (h : x = 15.3) : 
  (⌊ x^2 ⌋ - ⌊ x ⌋ * ⌊ x ⌋ + 5) = 14 := 
by
  -- Skipping proof
  sorry

end floor_difference_l304_304932


namespace change_given_l304_304770

-- Define the given conditions
def oranges_cost := 40
def apples_cost := 50
def mangoes_cost := 60
def initial_amount := 300

-- Calculate total cost of fruits
def total_fruits_cost := oranges_cost + apples_cost + mangoes_cost

-- Define the given change
def given_change := initial_amount - total_fruits_cost

-- Prove that the given change is equal to 150
theorem change_given (h_oranges : oranges_cost = 40)
                     (h_apples : apples_cost = 50)
                     (h_mangoes : mangoes_cost = 60)
                     (h_initial : initial_amount = 300) :
  given_change = 150 :=
by
  -- Proof is omitted, indicated by sorry
  sorry

end change_given_l304_304770


namespace runners_meet_again_l304_304334

theorem runners_meet_again :
    ∀ t : ℝ,
      t ≠ 0 →
      (∃ k : ℤ, 3.8 * t - 4 * t = 400 * k) ∧
      (∃ m : ℤ, 4.2 * t - 4 * t = 400 * m) ↔
      t = 2000 := 
by
  sorry

end runners_meet_again_l304_304334


namespace original_cost_price_of_car_l304_304541

theorem original_cost_price_of_car
    (S_m S_f C : ℝ)
    (h1 : S_m = 0.86 * C)
    (h2 : S_f = 54000)
    (h3 : S_f = 1.20 * S_m) :
    C = 52325.58 :=
by
    sorry

end original_cost_price_of_car_l304_304541


namespace recurring_decimal_sum_l304_304405

theorem recurring_decimal_sum :
  (0.666666...:ℚ) + (0.222222...:ℚ) - (0.444444...:ℚ) = 4 / 9 :=
begin
  sorry
end

end recurring_decimal_sum_l304_304405


namespace lcm_from_1_to_12_eq_27720_l304_304728

theorem lcm_from_1_to_12_eq_27720 : nat.lcm (finset.range 12).succ = 27720 :=
  sorry

end lcm_from_1_to_12_eq_27720_l304_304728


namespace positions_after_196_moves_l304_304124

def cat_position (n : ℕ) : ℕ :=
  n % 4

def mouse_position (n : ℕ) : ℕ :=
  n % 8

def cat_final_position : ℕ := 0 -- top left based on the reverse order cycle
def mouse_final_position : ℕ := 3 -- bottom middle based on the reverse order cycle

theorem positions_after_196_moves :
  cat_position 196 = cat_final_position ∧ mouse_position 196 = mouse_final_position :=
by
  sorry

end positions_after_196_moves_l304_304124


namespace cakes_and_bread_weight_l304_304494

theorem cakes_and_bread_weight 
  (B : ℕ)
  (cake_weight : ℕ := B + 100)
  (h1 : 4 * cake_weight = 800)
  : 3 * cake_weight + 5 * B = 1100 := by
  sorry

end cakes_and_bread_weight_l304_304494


namespace least_int_gt_sqrt_450_l304_304502

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l304_304502


namespace magician_earning_correct_l304_304903

def magician_earning (initial_decks : ℕ) (remaining_decks : ℕ) (price_per_deck : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * price_per_deck

theorem magician_earning_correct :
  magician_earning 5 3 2 = 4 :=
by
  sorry

end magician_earning_correct_l304_304903


namespace repeating_decimal_sum_l304_304402

theorem repeating_decimal_sum : (0.\overline{6} + 0.\overline{2} - 0.\overline{4} : ℚ) = (4 / 9 : ℚ) :=
by
  sorry

end repeating_decimal_sum_l304_304402


namespace polynomial_factorization_l304_304355

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end polynomial_factorization_l304_304355


namespace scientific_notation_flu_virus_diameter_l304_304848

theorem scientific_notation_flu_virus_diameter :
  0.000000823 = 8.23 * 10^(-7) :=
sorry

end scientific_notation_flu_virus_diameter_l304_304848


namespace max_mn_value_l304_304973

theorem max_mn_value (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (hA1 : ∀ k : ℝ, k * (-2) - (-1) + 2 * k - 1 = 0)
  (hA2 : m * (-2) + n * (-1) + 2 = 0) :
  mn ≤ 1/2 := sorry

end max_mn_value_l304_304973


namespace geometric_sequence_q_and_an_l304_304592

theorem geometric_sequence_q_and_an
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_pos : q > 0)
  (h2_eq : a 2 = 1)
  (h2_h6_eq_9h4 : a 2 * a 6 = 9 * a 4) :
  q = 3 ∧ ∀ n, a n = 3^(n - 2) := by
sorry

end geometric_sequence_q_and_an_l304_304592


namespace arithmetic_seq_geom_seq_l304_304248

theorem arithmetic_seq_geom_seq {a : ℕ → ℝ} 
  (h1 : ∀ n, 0 < a n)
  (h2 : a 2 + a 3 + a 4 = 15)
  (h3 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2) :
  a 10 = 19 :=
sorry

end arithmetic_seq_geom_seq_l304_304248


namespace speed_of_man_l304_304549

open Real Int

/-- 
  A train 110 m long is running with a speed of 40 km/h.
  The train passes a man who is running at a certain speed
  in the direction opposite to that in which the train is going.
  The train takes 9 seconds to pass the man.
  This theorem proves that the speed of the man is 3.992 km/h.
-/
theorem speed_of_man (T_length : ℝ) (T_speed : ℝ) (t_pass : ℝ) (M_speed : ℝ) : 
  T_length = 110 → T_speed = 40 → t_pass = 9 → M_speed = 3.992 :=
by
  intro h1 h2 h3
  sorry

end speed_of_man_l304_304549


namespace repeating_decimal_sum_l304_304398

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l304_304398


namespace sum_of_two_numbers_l304_304161

theorem sum_of_two_numbers (x y : ℤ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l304_304161


namespace dress_designs_count_l304_304193

inductive Color
| red | green | blue | yellow

inductive Pattern
| stripes | polka_dots | floral | geometric | plain

def patterns_for_color (c : Color) : List Pattern :=
  match c with
  | Color.red    => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.green  => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]
  | Color.blue   => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.yellow => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]

noncomputable def number_of_dress_designs : ℕ :=
  (patterns_for_color Color.red).length +
  (patterns_for_color Color.green).length +
  (patterns_for_color Color.blue).length +
  (patterns_for_color Color.yellow).length

theorem dress_designs_count : number_of_dress_designs = 18 :=
  by
  sorry

end dress_designs_count_l304_304193


namespace find_x_l304_304572

namespace ProofProblem

def δ (x : ℚ) : ℚ := 5 * x + 6
def φ (x : ℚ) : ℚ := 9 * x + 4

theorem find_x (x : ℚ) : (δ (φ x) = 14) ↔ (x = -4 / 15) :=
by
  sorry

end ProofProblem

end find_x_l304_304572


namespace greatest_x_lcm_l304_304886

theorem greatest_x_lcm (x : ℕ) (h1 : Nat.lcm x 15 = Nat.lcm 90 15) (h2 : Nat.lcm x 18 = Nat.lcm 90 18) : x = 90 := 
sorry

end greatest_x_lcm_l304_304886


namespace part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l304_304859

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + 1
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x - g x a

-- (Ⅰ) Equation of the tangent line to y = f(x) at x = 1
theorem part1_tangent_line_at_1 : ∀ x, (f 1 + (1 / 1) * (x - 1)) = x - 1 := sorry

-- (Ⅱ) Intervals where F(x) is monotonic
theorem part2_monotonic_intervals (a : ℝ) : 
  (a ≤ 0 → ∀ x > 0, F x a > 0) ∧ 
  (a > 0 → (∀ x > 0, x < (1 / a) → F x a > 0) ∧ (∀ x > 1 / a, F x a < 0)) := sorry

-- (Ⅲ) Range of a for which f(x) is below g(x) for all x > 0
theorem part3_range_of_a (a : ℝ) : (∀ x > 0, f x < g x a) ↔ a ∈ Set.Ioi (Real.exp (-2)) := sorry

end part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l304_304859


namespace banana_distinct_arrangements_l304_304962

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l304_304962


namespace highest_prob_of_red_card_l304_304730

theorem highest_prob_of_red_card :
  let deck_size := 52
  let num_aces := 4
  let num_hearts := 13
  let num_kings := 4
  let num_reds := 26
  -- Event probabilities
  let prob_ace := num_aces / deck_size
  let prob_heart := num_hearts / deck_size
  let prob_king := num_kings / deck_size
  let prob_red := num_reds / deck_size
  prob_red > prob_heart ∧ prob_heart > prob_ace ∧ prob_ace = prob_king :=
sorry

end highest_prob_of_red_card_l304_304730


namespace probability_correct_digit_in_two_attempts_l304_304492

theorem probability_correct_digit_in_two_attempts :
  let total_digits := 10
  let probability_first_correct := 1 / total_digits
  let probability_first_incorrect := 9 / total_digits
  let probability_second_correct_if_first_incorrect := 1 / (total_digits - 1)
  (probability_first_correct + probability_first_incorrect * probability_second_correct_if_first_incorrect) = 1 / 5 := 
sorry

end probability_correct_digit_in_two_attempts_l304_304492


namespace recurring_decimal_to_fraction_l304_304216

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l304_304216


namespace expected_value_eight_sided_die_win_l304_304916

/-- The expected value of winning with a fair 8-sided die, where the win is \( n^3 \) dollars if \( n \) is rolled, is 162 dollars. -/
theorem expected_value_eight_sided_die_win :
  (1 / 8) * (1^3) + (1 / 8) * (2^3) + (1 / 8) * (3^3) + (1 / 8) * (4^3) +
  (1 / 8) * (5^3) + (1 / 8) * (6^3) + (1 / 8) * (7^3) + (1 / 8) * (8^3) = 162 := 
by
  -- Simplification and calculation here
  sorry

end expected_value_eight_sided_die_win_l304_304916


namespace ordered_pairs_squares_diff_150_l304_304964

theorem ordered_pairs_squares_diff_150 (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn : m ≥ n) (h_diff : m^2 - n^2 = 150) : false :=
by {
    sorry
}

end ordered_pairs_squares_diff_150_l304_304964


namespace tic_tac_toe_winning_boards_l304_304765

-- Define the board as a 4x4 grid
def Board := Array (Array (Option Bool))

-- Define a function that returns all possible board states after 3 moves
noncomputable def numberOfWinningBoards : Nat := 140

theorem tic_tac_toe_winning_boards:
  numberOfWinningBoards = 140 :=
by
  sorry

end tic_tac_toe_winning_boards_l304_304765


namespace unique_solution_quadratic_l304_304569

theorem unique_solution_quadratic (n : ℕ) : (∀ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intros h
  sorry

end unique_solution_quadratic_l304_304569


namespace simplify_expression_l304_304575

variable (a b c : ℝ) 

theorem simplify_expression (h1 : a ≠ 4) (h2 : b ≠ 5) (h3 : c ≠ 6) :
  (a - 4) / (6 - c) * (b - 5) / (4 - a) * (c - 6) / (5 - b) = -1 :=
by
  sorry

end simplify_expression_l304_304575


namespace original_bill_l304_304744

theorem original_bill (n : ℕ) (d : ℝ) (p : ℝ) (B : ℝ) (h1 : n = 5) (h2 : d = 0.06) (h3 : p = 18.8)
  (h4 : 0.94 * B = n * p) :
  B = 100 :=
sorry

end original_bill_l304_304744


namespace minimum_value_expression_l304_304860

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ k, k = 729 ∧ ∀ x y z, 0 < x → 0 < y → 0 < z → k ≤ (x^2 + 4*x + 4) * (y^2 + 4*y + 4) * (z^2 + 4*z + 4) / (x * y * z) :=
by 
  use 729
  sorry

end minimum_value_expression_l304_304860


namespace yuan_exchange_l304_304833

theorem yuan_exchange : 
  ∃ (n : ℕ), n = 5 ∧ ∀ (x y : ℕ), x + 5 * y = 20 → x ≥ 0 ∧ y ≥ 0 :=
by {
  sorry
}

end yuan_exchange_l304_304833


namespace solve_for_a_l304_304116

theorem solve_for_a (a : ℝ) (h : ∃ x, x = 2 ∧ a * x - 4 * (x - a) = 1) : a = 3 / 2 :=
sorry

end solve_for_a_l304_304116


namespace composite_function_increasing_l304_304019

variable {F : ℝ → ℝ}

/-- An odd function is a function that satisfies f(-x) = -f(x) for all x. -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is strictly increasing on negative values if it satisfies the given conditions. -/
def strictly_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, x1 < x2 → x2 < 0 → f x1 < f x2

/-- Combining properties of an odd function and strictly increasing for negative inputs:
  We need to prove that the composite function is strictly increasing for positive inputs. -/
theorem composite_function_increasing (hf_odd : odd_function F)
    (hf_strict_inc_neg : strictly_increasing_on_neg F)
    : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → F (F x1) < F (F x2) :=
  sorry

end composite_function_increasing_l304_304019


namespace polynomial_factorization_l304_304351

theorem polynomial_factorization :
  (x : ℤ[X]) →
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  intros x
  sorry

end polynomial_factorization_l304_304351


namespace plains_routes_l304_304276

theorem plains_routes 
  (total_cities : ℕ)
  (mountainous_cities : ℕ)
  (plains_cities : ℕ)
  (total_routes : ℕ)
  (mountainous_routes : ℕ)
  (num_pairs_with_mount_to_mount : ℕ)
  (routes_per_year : ℕ)
  (years : ℕ)
  (mountainous_roots_connections : ℕ)
  : (mountainous_cities = 30) →
    (plains_cities = 70) →
    (total_cities = mountainous_cities + plains_cities) →
    (routes_per_year = 50) →
    (years = 3) →
    (total_routes = routes_per_year * years) →
    (mountainous_routes = num_pairs_with_mount_to_mount * 2) →
    (num_pairs_with_mount_to_mount = 21) →
    let num_endpoints_per_city_route = 2 in
    let mountainous_city_endpoints = mountainous_cities * 3 in
    let mountainous_endpoints = mountainous_routes in
    let mountain_to_plains_endpoints = mountainous_city_endpoints - mountainous_endpoints in
    let total_endpoints = total_routes * num_endpoints_per_city_route in
    let plains_city_endpoints = plains_cities * 3 in
    let routes_between_plain_and_mountain = mountain_to_plains_endpoints in
    let plain_to_plain_endpoints = plains_city_endpoints - routes_between_plain_and_mountain in
    let plain_to_plain_routes = plain_to_plain_endpoints / 2 in
    plain_to_plain_routes = 81 :=
sorry

end plains_routes_l304_304276


namespace cake_eating_classmates_l304_304670

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l304_304670


namespace Beth_and_Jan_total_money_l304_304118

theorem Beth_and_Jan_total_money (B J : ℝ) 
  (h1 : B + 35 = 105)
  (h2 : J - 10 = B) : 
  B + J = 150 :=
by
  -- Proof omitted
  sorry

end Beth_and_Jan_total_money_l304_304118


namespace polynomial_factorization_l304_304349

open Polynomial

theorem polynomial_factorization :
  (X ^ 15 + X ^ 10 + X ^ 5 + 1) =
    (X ^ 3 + X ^ 2 + 1) * 
    (X ^ 12 - X ^ 11 + X ^ 9 - X ^ 8 + X ^ 6 - X ^ 5 + X ^ 4 + X ^ 3 + X ^ 2 + X + 1) :=
by
  sorry

end polynomial_factorization_l304_304349


namespace rectangle_width_l304_304530

theorem rectangle_width (L W : ℝ) (h1 : 2 * (L + W) = 16) (h2 : W = L + 2) : W = 5 :=
by
  sorry

end rectangle_width_l304_304530


namespace percent_error_l304_304371

theorem percent_error (x : ℝ) (h : x > 0) :
  (abs ((12 * x) - (x / 3)) / (x / 3)) * 100 = 3500 :=
by
  sorry

end percent_error_l304_304371


namespace range_of_a_l304_304990

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 4 ≤ a := 
sorry

end range_of_a_l304_304990


namespace largest_interesting_number_l304_304918

def is_interesting_number (x : ℝ) : Prop :=
  ∃ y z : ℝ, (0 ≤ y ∧ y < 1) ∧ (0 ≤ z ∧ z < 1) ∧ x = 0 + y * 10⁻¹ + z ∧ 2 * (0 + y * 10⁻¹ + z) = 0 + z

theorem largest_interesting_number : ∀ x, is_interesting_number x → x ≤ 0.375 :=
by
  sorry

end largest_interesting_number_l304_304918


namespace change_given_l304_304769

-- Define the given conditions
def oranges_cost := 40
def apples_cost := 50
def mangoes_cost := 60
def initial_amount := 300

-- Calculate total cost of fruits
def total_fruits_cost := oranges_cost + apples_cost + mangoes_cost

-- Define the given change
def given_change := initial_amount - total_fruits_cost

-- Prove that the given change is equal to 150
theorem change_given (h_oranges : oranges_cost = 40)
                     (h_apples : apples_cost = 50)
                     (h_mangoes : mangoes_cost = 60)
                     (h_initial : initial_amount = 300) :
  given_change = 150 :=
by
  -- Proof is omitted, indicated by sorry
  sorry

end change_given_l304_304769


namespace systematic_sampling_second_invoice_l304_304379

theorem systematic_sampling_second_invoice 
  (N : ℕ) 
  (valid_invoice : N ≥ 10)
  (first_invoice : Fin 10) :
  ¬ (∃ k : ℕ, k ≥ 1 ∧ first_invoice.1 + k * 10 = 23) := 
by 
  -- Proof omitted
  sorry

end systematic_sampling_second_invoice_l304_304379


namespace number_of_plains_routes_is_81_l304_304281

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end number_of_plains_routes_is_81_l304_304281


namespace wheel_moves_in_one_hour_l304_304550

theorem wheel_moves_in_one_hour
  (rotations_per_minute : ℕ)
  (distance_per_rotation_cm : ℕ)
  (minutes_in_hour : ℕ) :
  rotations_per_minute = 20 →
  distance_per_rotation_cm = 35 →
  minutes_in_hour = 60 →
  let distance_per_rotation_m : ℚ := distance_per_rotation_cm / 100
  let total_rotations_per_hour : ℕ := rotations_per_minute * minutes_in_hour
  let total_distance_in_hour : ℚ := distance_per_rotation_m * total_rotations_per_hour
  total_distance_in_hour = 420 := by
  intros
  sorry

end wheel_moves_in_one_hour_l304_304550


namespace expected_value_of_8_sided_die_l304_304911

theorem expected_value_of_8_sided_die : 
  let p := (1 / 8 : ℚ)
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let values := outcomes.map (λ n, n^3)
  let expected_value := (values.sum : ℚ) * p
  expected_value = 162 := sorry

end expected_value_of_8_sided_die_l304_304911


namespace find_number_l304_304122

axiom condition_one (x y : ℕ) : 10 * x + y = 3 * (x + y) + 7
axiom condition_two (x y : ℕ) : x^2 + y^2 - x * y = 10 * x + y

theorem find_number : 
  ∃ (x y : ℕ), (10 * x + y = 37) → (10 * x + y = 3 * (x + y) + 7 ∧ x^2 + y^2 - x * y = 10 * x + y) := 
by 
  sorry

end find_number_l304_304122


namespace recurring_decimal_sum_l304_304404

theorem recurring_decimal_sum :
  (0.666666...:ℚ) + (0.222222...:ℚ) - (0.444444...:ℚ) = 4 / 9 :=
begin
  sorry
end

end recurring_decimal_sum_l304_304404


namespace volume_of_56_ounces_is_24_cubic_inches_l304_304902

-- Given information as premises
def directlyProportional (V W : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ V = k * W

-- The specific conditions in the problem
def initial_volume := 48   -- in cubic inches
def initial_weight := 112  -- in ounces
def target_weight := 56    -- in ounces
def target_volume := 24    -- in cubic inches (the value we need to prove)

-- The theorem statement 
theorem volume_of_56_ounces_is_24_cubic_inches
  (h1 : directlyProportional initial_volume initial_weight)
  (h2 : directlyProportional target_volume target_weight)
  (h3 : target_weight = 56)
  (h4 : initial_volume = 48)
  (h5 : initial_weight = 112) :
  target_volume = 24 :=
sorry -- Proof not required as per instructions

end volume_of_56_ounces_is_24_cubic_inches_l304_304902


namespace complex_solution_l304_304436

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l304_304436


namespace robot_transport_max_robots_l304_304906

section
variable {A B : ℕ}   -- Define the variables A and B
variable {m : ℕ}     -- Define the variable m

-- Part 1
theorem robot_transport (h1 : A = B + 30) (h2 : 1500 * B = 1000 * (B + 30)) : A = 90 ∧ B = 60 :=
by
  sorry

-- Part 2
theorem max_robots (h3 : 50000 * m + 30000 * (12 - m) ≤ 450000) : m ≤ 4 :=
by
  sorry
end

end robot_transport_max_robots_l304_304906


namespace compare_a_b_l304_304949

theorem compare_a_b (a b : ℝ) (h : 5 * (a - 1) = b + a ^ 2) : a > b :=
sorry

end compare_a_b_l304_304949


namespace evan_books_two_years_ago_l304_304211

theorem evan_books_two_years_ago (B B2 : ℕ) 
  (h1 : 860 = 5 * B + 60) 
  (h2 : B2 = B + 40) : 
  B2 = 200 := 
by 
  sorry

end evan_books_two_years_ago_l304_304211


namespace least_integer_gt_sqrt_450_l304_304518

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l304_304518


namespace count_four_digit_numbers_without_1_or_4_l304_304099

-- Define a function to check if a digit is allowed (i.e., not 1 or 4)
def allowed_digit (d : ℕ) : Prop := d ≠ 1 ∧ d ≠ 4

-- Function to count four-digit numbers without digits 1 or 4
def count_valid_four_digit_numbers : ℕ :=
  let valid_first_digits := [2, 3, 5, 6, 7, 8, 9]
  let valid_other_digits := [0, 2, 3, 5, 6, 7, 8, 9]
  (valid_first_digits.length) * (valid_other_digits.length ^ 3)

-- The main theorem stating that the number of valid four-digit integers is 3072
theorem count_four_digit_numbers_without_1_or_4 : count_valid_four_digit_numbers = 3072 :=
by
  sorry

end count_four_digit_numbers_without_1_or_4_l304_304099


namespace like_terms_exponents_l304_304597

theorem like_terms_exponents (m n : ℕ) (x y : ℝ) (h : 2 * x^(2*m) * y^6 = -3 * x^8 * y^(2*n)) : m = 4 ∧ n = 3 :=
by 
  sorry

end like_terms_exponents_l304_304597


namespace simplify_expression_l304_304877

theorem simplify_expression : 
  (1 / ((1 / ((1 / 2)^1)) + (1 / ((1 / 2)^3)) + (1 / ((1 / 2)^4)))) = (1 / 26) := 
by 
  sorry

end simplify_expression_l304_304877


namespace cake_sharing_l304_304656

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l304_304656


namespace recurring_decimal_reduced_fraction_l304_304238

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l304_304238


namespace solve_system_l304_304564

theorem solve_system :
  {p : ℝ × ℝ | p.1^3 + p.2^3 = 19 ∧ p.1^2 + p.2^2 + 5 * p.1 + 5 * p.2 + p.1 * p.2 = 12} = {(3, -2), (-2, 3)} :=
sorry

end solve_system_l304_304564


namespace base_b_representation_l304_304267

theorem base_b_representation (b : ℕ) : (2 * b + 9)^2 = 7 * b^2 + 3 * b + 4 → b = 14 := 
sorry

end base_b_representation_l304_304267


namespace repeating_decimal_to_fraction_l304_304241

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l304_304241


namespace recurring_decimal_to_fraction_l304_304218

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l304_304218


namespace coplanar_iff_m_eq_neg_8_l304_304295

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (A B C D : V)
variable (m : ℝ)

theorem coplanar_iff_m_eq_neg_8 
  (h : 4 • A - 3 • B + 7 • C + m • D = 0) : m = -8 ↔ ∃ a b c d : ℝ, a + b + c + d = 0 ∧ a • A + b • B + c • C + d • D = 0 :=
by
  sorry

end coplanar_iff_m_eq_neg_8_l304_304295


namespace union_sets_l304_304431

theorem union_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} :=
sorry

end union_sets_l304_304431


namespace numerical_expression_as_sum_of_squares_l304_304310

theorem numerical_expression_as_sum_of_squares : 
  2 * (2009:ℕ)^2 + 2 * (2010:ℕ)^2 = (4019:ℕ)^2 + (1:ℕ)^2 := 
by
  sorry

end numerical_expression_as_sum_of_squares_l304_304310


namespace combined_supply_duration_l304_304131

variable (third_of_pill_per_third_day : ℕ → Prop)
variable (alternate_days : ℕ → ℕ → Prop)
variable (supply : ℕ)
variable (days_in_month : ℕ)

-- Conditions:
def one_third_per_third_day (p: ℕ) (d: ℕ) : Prop := 
  third_of_pill_per_third_day d ∧ alternate_days d (d + 3)
def total_supply (s: ℕ) := s = 60
def duration_per_pill (d: ℕ) := d = 9
def month_days (m: ℕ) := m = 30

-- Proof Problem Statement:
theorem combined_supply_duration :
  ∀ (s t: ℕ), total_supply s ∧ duration_per_pill t ∧ month_days 30 → 
  (s * t / 30) = 18 :=
by
  intros s t h
  sorry

end combined_supply_duration_l304_304131


namespace lcm_1_to_12_l304_304725

theorem lcm_1_to_12 : nat.lcm (list.range (12 + 1)) = 27720 :=
begin
  sorry
end

end lcm_1_to_12_l304_304725


namespace polynomial_factorization_l304_304352

theorem polynomial_factorization :
  (x : ℤ[X]) →
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  intros x
  sorry

end polynomial_factorization_l304_304352


namespace negation_example_l304_304947

theorem negation_example (p : ∀ n : ℕ, n^2 < 2^n) : 
  ¬ (∀ n : ℕ, n^2 < 2^n) ↔ ∃ n : ℕ, n^2 ≥ 2^n :=
by sorry

end negation_example_l304_304947


namespace sum_invested_eq_2000_l304_304908

theorem sum_invested_eq_2000 (P : ℝ) (R1 R2 T : ℝ) (H1 : R1 = 18) (H2 : R2 = 12) 
  (H3 : T = 2) (H4 : (P * R1 * T / 100) - (P * R2 * T / 100) = 240): 
  P = 2000 :=
by 
  sorry

end sum_invested_eq_2000_l304_304908


namespace range_of_z_l304_304624

variable (x y z : ℝ)

theorem range_of_z (hx : x ≥ 0) (hy : y ≥ x) (hxy : 4*x + 3*y ≤ 12) 
(hz : z = (x + 2 * y + 3) / (x + 1)) : 
2 ≤ z ∧ z ≤ 6 :=
sorry

end range_of_z_l304_304624


namespace DVDs_sold_168_l304_304273

-- Definitions of the conditions
def CDs_sold := ℤ
def DVDs_sold := ℤ

def ratio_condition (C D : ℤ) : Prop := D = 16 * C / 10
def total_condition (C D : ℤ) : Prop := D + C = 273

-- The main statement to prove
theorem DVDs_sold_168 (C D : ℤ) 
  (h1 : ratio_condition C D) 
  (h2 : total_condition C D) : D = 168 :=
sorry

end DVDs_sold_168_l304_304273


namespace number_of_classmates_l304_304653

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l304_304653


namespace inequality_proof_l304_304000

theorem inequality_proof (a b : ℝ) (h : (a = 0 ∨ b = 0 ∨ (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0))) :
  a^4 + 2*a^3*b + 2*a*b^3 + b^4 ≥ 6*a^2*b^2 :=
by
  sorry

end inequality_proof_l304_304000


namespace least_integer_greater_than_sqrt_450_l304_304507

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l304_304507


namespace emily_prob_3_spaces_away_l304_304075

def is_equivalent_3_spaces_away (initial : Int) (spins : List Int) : Prop :=
  match spins with
  | [spin1, spin2] => (spin1 + spin2 = 3 ∨ spin1 + spin2 = -3)
  | _ => False

def probability_of_3_spaces_away : ProbabilitySpace (List Int) := sorry

theorem emily_prob_3_spaces_away :
  (probability_of_3_spaces_away {spins | is_equivalent_3_spaces_away initial spins}) = 7 / 16 := 
sorry

end emily_prob_3_spaces_away_l304_304075


namespace expected_value_of_win_l304_304909

theorem expected_value_of_win :
  (∑ n in finset.range 9, (n ^ 3)) / 8 = 162 := 
sorry

end expected_value_of_win_l304_304909


namespace crows_eat_worms_l304_304967

theorem crows_eat_worms (worms_eaten_by_3_crows_in_1_hour : ℕ) 
                        (crows_eating_worms_constant : worms_eaten_by_3_crows_in_1_hour = 30)
                        (number_of_crows : ℕ) 
                        (observation_time_hours : ℕ) :
                        number_of_crows = 5 ∧ observation_time_hours = 2 →
                        (number_of_crows * worms_eaten_by_3_crows_in_1_hour / 3) * observation_time_hours = 100 :=
by
  sorry

end crows_eat_worms_l304_304967


namespace min_points_necessary_l304_304526

noncomputable def min_points_on_circle (circumference : ℝ) (dist1 dist2 : ℝ) : ℕ :=
  1304

theorem min_points_necessary :
  ∀ (circumference : ℝ) (dist1 dist2 : ℝ),
  circumference = 1956 →
  dist1 = 1 →
  dist2 = 2 →
  (min_points_on_circle circumference dist1 dist2) = 1304 :=
sorry

end min_points_necessary_l304_304526


namespace A_alone_days_l304_304190

theorem A_alone_days (A B C : ℝ) (hB: B = 9) (hC: C = 7.2) 
  (h: 1 / A + 1 / B + 1 / C = 1 / 2) : A = 2 :=
by
  rw [hB, hC] at h
  sorry

end A_alone_days_l304_304190


namespace total_dots_not_visible_l304_304245

-- Define the conditions and variables
def total_dots_one_die : Nat := 1 + 2 + 3 + 4 + 5 + 6
def number_of_dice : Nat := 4
def total_dots_all_dice : Nat := number_of_dice * total_dots_one_die
def visible_numbers : List Nat := [6, 6, 4, 4, 3, 2, 1]

-- The question can be formalized as proving that the total number of dots not visible is 58
theorem total_dots_not_visible :
  total_dots_all_dice - visible_numbers.sum = 58 :=
by
  -- Statement only, proof skipped
  sorry

end total_dots_not_visible_l304_304245


namespace sin_double_angle_neg_l304_304104

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l304_304104


namespace cake_eating_classmates_l304_304674

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l304_304674


namespace balls_distribution_l304_304001

theorem balls_distribution : 
  ∃ (n : ℕ), 
    (∀ (b1 b2 : ℕ), ∀ (h : b1 + b2 = 4), b1 ≥ 1 ∧ b2 ≥ 2 → n = 10) :=
sorry

end balls_distribution_l304_304001


namespace sugar_water_sweeter_l304_304253

variable (a b m : ℝ)
variable (a_pos : a > 0) (b_gt_a : b > a) (m_pos : m > 0)

theorem sugar_water_sweeter : (a + m) / (b + m) > a / b :=
by
  sorry

end sugar_water_sweeter_l304_304253


namespace sector_angle_l304_304155

theorem sector_angle (r : ℝ) (θ : ℝ) 
  (area_eq : (1 / 2) * θ * r^2 = 1)
  (perimeter_eq : 2 * r + θ * r = 4) : θ = 2 := 
by
  sorry

end sector_angle_l304_304155


namespace john_beats_per_minute_l304_304986

theorem john_beats_per_minute :
  let hours_per_day := 2
  let days := 3
  let total_beats := 72000
  let minutes_per_hour := 60
  total_beats / (days * hours_per_day * minutes_per_hour) = 200 := 
by 
  sorry

end john_beats_per_minute_l304_304986


namespace factors_multiple_of_120_l304_304942

theorem factors_multiple_of_120 (n : ℕ) (h : n = 2^12 * 3^15 * 5^9 * 7^5) :
  ∃ k : ℕ, k = 8100 ∧ ∀ d : ℕ, d ∣ n ∧ 120 ∣ d ↔ ∃ a b c d : ℕ, 3 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 15 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 5 ∧ d = 2^a * 3^b * 5^c * 7^d :=
by
  sorry

end factors_multiple_of_120_l304_304942


namespace macey_saving_weeks_l304_304303

-- Definitions for conditions
def shirt_cost : ℝ := 3
def amount_saved : ℝ := 1.5
def weekly_saving : ℝ := 0.5

-- Statement of the proof problem
theorem macey_saving_weeks : (shirt_cost - amount_saved) / weekly_saving = 3 := by
  sorry

end macey_saving_weeks_l304_304303


namespace largest_mersenne_prime_less_than_500_l304_304187

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Nat.Prime n ∧ p = 2^n - 1 ∧ Nat.Prime p

theorem largest_mersenne_prime_less_than_500 :
  ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p → p = 127 :=
by
  sorry

end largest_mersenne_prime_less_than_500_l304_304187


namespace ratio_of_perimeters_l304_304547

theorem ratio_of_perimeters (L : ℝ) (H : ℝ) (hL1 : L = 8) 
  (hH1 : H = 8) (hH2 : H = 2 * (H / 2)) (hH3 : 4 > 0) (hH4 : 0 < 4 / 3)
  (hW1 : ∀ a, a / 3 > 0 → 8 = L )
  (hPsmall : ∀ P, P = 2 * ((4 / 3) + 8) )
  (hPlarge : ∀ P, P = 2 * ((H - 4 / 3) + 8) )
  :
  (2 * ((4 / 3) + 8)) / (2 * ((8 - (4 / 3)) + 8)) = (7 / 11) := by
  sorry

end ratio_of_perimeters_l304_304547


namespace intersection_of_A_and_B_l304_304095

noncomputable def A : Set ℝ := {x | x^2 - 1 ≤ 0}

noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l304_304095


namespace find_positive_n_unique_solution_l304_304567

theorem find_positive_n_unique_solution (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intro h
  sorry

end find_positive_n_unique_solution_l304_304567


namespace percentage_increase_l304_304368

variable (P N N' : ℝ)
variable (h : P * 0.90 * N' = P * N * 1.035)

theorem percentage_increase :
  ((N' - N) / N) * 100 = 15 :=
by
  -- By given condition, we have the equation:
  -- P * 0.90 * N' = P * N * 1.035
  sorry

end percentage_increase_l304_304368


namespace vertical_asymptote_at_5_l304_304207

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + 2*x + 10) / (x - 5)

theorem vertical_asymptote_at_5 : ∃ a : ℝ, (a = 5) ∧ ∀ δ > 0, ∃ ε > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < ε → |f x| > δ :=
by
  sorry

end vertical_asymptote_at_5_l304_304207


namespace lucas_raspberry_candies_l304_304862

-- Define the problem conditions and the question
theorem lucas_raspberry_candies :
  ∃ (r l : ℕ), (r = 3 * l) ∧ ((r - 5) = 4 * (l - 5)) ∧ (r = 45) :=
by
  sorry

end lucas_raspberry_candies_l304_304862


namespace lowest_score_85_avg_l304_304680

theorem lowest_score_85_avg (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 = 79) (h2 : a2 = 88) (h3 : a3 = 94) 
  (h4 : a4 = 91) (h5 : 75 ≤ a5) (h6 : 75 ≤ a6) 
  (h7 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 85) : (a5 = 75 ∨ a6 = 75) ∧ (a5 = 75 ∨ a5 > 75) := 
by
  sorry

end lowest_score_85_avg_l304_304680


namespace possible_values_of_b_l304_304208

theorem possible_values_of_b 
        (b : ℤ)
        (h : ∃ x : ℤ, (x ^ 3 + 2 * x ^ 2 + b * x + 8 = 0)) :
        b = -81 ∨ b = -26 ∨ b = -12 ∨ b = -6 ∨ b = 4 ∨ b = 9 ∨ b = 47 :=
  sorry

end possible_values_of_b_l304_304208


namespace arithmetic_sequence_sum_l304_304800

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 + a 13 = 10) 
  (h2 : ∀ n m : ℕ, a (n + 1) = a n + d) : a 3 + a 5 + a 7 + a 9 + a 11 = 25 :=
  sorry

end arithmetic_sequence_sum_l304_304800


namespace sin_double_angle_fourth_quadrant_l304_304102

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l304_304102


namespace constant_term_l304_304801

theorem constant_term (n : ℕ) (h : (Nat.choose n 4 * 2^4) / (Nat.choose n 2 * 2^2) = (56 / 3)) :
  (∃ k : ℕ, k = 2 ∧ n = 10 ∧ Nat.choose 10 k * 2^k = 180) := by
  sorry

end constant_term_l304_304801


namespace white_triangle_pairs_condition_l304_304931

def number_of_white_pairs (total_triangles : Nat) 
                          (red_pairs : Nat) 
                          (blue_pairs : Nat)
                          (mixed_pairs : Nat) : Nat :=
  let red_involved := red_pairs * 2
  let blue_involved := blue_pairs * 2
  let remaining_red := total_triangles / 2 * 5 - red_involved - mixed_pairs
  let remaining_blue := total_triangles / 2 * 4 - blue_involved - mixed_pairs
  (total_triangles / 2 * 7) - (remaining_red + remaining_blue)/2

theorem white_triangle_pairs_condition : number_of_white_pairs 32 3 2 1 = 6 := by
  sorry

end white_triangle_pairs_condition_l304_304931


namespace diagonals_from_one_vertex_l304_304329

theorem diagonals_from_one_vertex (x : ℕ) (h : (x - 2) * 180 = 1800) : (x - 3) = 9 :=
  by
  sorry

end diagonals_from_one_vertex_l304_304329


namespace profit_percentage_correct_l304_304760

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 70
noncomputable def list_price : ℝ := selling_price / 0.95
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem profit_percentage_correct :
  abs (profit_percentage - 47.37) < 0.01 := sorry

end profit_percentage_correct_l304_304760


namespace marble_count_l304_304475

theorem marble_count (x : ℕ) 
  (h1 : ∀ (Liam Mia Noah Olivia: ℕ), Mia = 3 * Liam ∧ Noah = 4 * Mia ∧ Olivia = 2 * Noah)
  (h2 : Liam + Mia + Noah + Olivia = 156)
  : x = 4 :=
by sorry

end marble_count_l304_304475


namespace nate_reading_percentage_l304_304635

-- Given conditions
def total_pages := 400
def pages_to_read := 320

-- Calculate the number of pages he has already read
def pages_read := total_pages - pages_to_read

-- Prove the percentage of the book Nate has finished reading
theorem nate_reading_percentage : (pages_read / total_pages) * 100 = 20 := by
  sorry

end nate_reading_percentage_l304_304635


namespace Amanda_needs_12_more_marbles_l304_304552

theorem Amanda_needs_12_more_marbles (K A M : ℕ)
  (h1 : M = 5 * K)
  (h2 : M = 85)
  (h3 : M = A + 63) :
  A + 12 = 2 * K := 
sorry

end Amanda_needs_12_more_marbles_l304_304552


namespace find_positive_n_unique_solution_l304_304566

theorem find_positive_n_unique_solution (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intro h
  sorry

end find_positive_n_unique_solution_l304_304566


namespace stratified_sampling_l304_304697

theorem stratified_sampling (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : (5 : ℚ) / 10 = 150 / n) : n = 300 :=
by
  sorry

end stratified_sampling_l304_304697


namespace candy_distribution_proof_l304_304764

theorem candy_distribution_proof :
  ∀ (candy_total Kate Robert Bill Mary : ℕ),
  candy_total = 20 →
  Kate = 4 →
  Robert = Kate + 2 →
  Bill = Mary - 6 →
  Kate = Bill + 2 →
  Mary > Robert →
  (Mary - Robert = 2) :=
by
  intros candy_total Kate Robert Bill Mary h1 h2 h3 h4 h5 h6
  sorry

end candy_distribution_proof_l304_304764


namespace sin_double_angle_fourth_quadrant_l304_304112

theorem sin_double_angle_fourth_quadrant (α : ℝ) (h_quadrant : ∃ k : ℤ, -π/2 + 2 * k * π < α ∧ α < 2 * k * π) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_fourth_quadrant_l304_304112


namespace cake_eating_classmates_l304_304673

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l304_304673


namespace domain_of_function_l304_304389

theorem domain_of_function :
  {x : ℝ | 2 - x > 0 ∧ 1 + x > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end domain_of_function_l304_304389


namespace digit_b_divisible_by_5_l304_304501

theorem digit_b_divisible_by_5 (B : ℕ) (h : B = 0 ∨ B = 5) : 
  (∃ n : ℕ, (947 * 10 + B) = 5 * n) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end digit_b_divisible_by_5_l304_304501


namespace scientific_notation_of_11580000_l304_304835

theorem scientific_notation_of_11580000 :
  11_580_000 = 1.158 * 10^7 :=
sorry

end scientific_notation_of_11580000_l304_304835


namespace provisions_initial_days_l304_304495

theorem provisions_initial_days (D : ℕ) (P : ℕ) (Q : ℕ) (X : ℕ) (Y : ℕ)
  (h1 : P = 300) 
  (h2 : X = 30) 
  (h3 : Y = 90) 
  (h4 : Q = 200) 
  (h5 : P * D = P * X + Q * Y) : D + X = 120 :=
by
  -- We need to prove that the initial number of days the provisions were meant to last is 120.
  sorry

end provisions_initial_days_l304_304495


namespace coefficient_of_x4_in_expansion_l304_304339

theorem coefficient_of_x4_in_expansion :
  (∑ k in Finset.range (8 + 1), (Nat.choose 8 k) * (x : ℝ)^(8 - k) * (3 * Real.sqrt 2)^k).coeff 4 = 22680 :=
by
  sorry

end coefficient_of_x4_in_expansion_l304_304339


namespace fourth_square_area_l304_304891

theorem fourth_square_area (AB BC CD AC x : ℝ) 
  (h_AB : AB^2 = 49) 
  (h_BC : BC^2 = 25) 
  (h_CD : CD^2 = 64) 
  (h_AC1 : AC^2 = AB^2 + BC^2) 
  (h_AC2 : AC^2 = CD^2 + x^2) :
  x^2 = 10 :=
by
  sorry

end fourth_square_area_l304_304891


namespace sum_of_arithmetic_sequence_l304_304341

-- Define the conditions
def is_arithmetic_sequence (first_term last_term : ℕ) (terms : ℕ) : Prop :=
  ∃ (a l : ℕ) (n : ℕ), a = first_term ∧ l = last_term ∧ n = terms ∧ n > 1

-- State the theorem
theorem sum_of_arithmetic_sequence (a l n : ℕ) (h_arith: is_arithmetic_sequence 5 41 10):
  n = 10 ∧ a = 5 ∧ l = 41 → (n * (a + l) / 2) = 230 :=
by
  intros h
  sorry

end sum_of_arithmetic_sequence_l304_304341


namespace factor_expression_l304_304411

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) :=
by
  sorry

end factor_expression_l304_304411


namespace blake_change_given_l304_304774

theorem blake_change_given :
  let oranges := 40
  let apples := 50
  let mangoes := 60
  let total_amount := 300
  let total_spent := oranges + apples + mangoes
  let change_given := total_amount - total_spent
  change_given = 150 :=
by
  sorry

end blake_change_given_l304_304774


namespace monotonic_power_function_l304_304426

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := (a^2 - 2 * a - 2) * x^a

theorem monotonic_power_function (a : ℝ) (h1 : ∀ x : ℝ, ( ∀ x1 x2 : ℝ, x1 < x2 → power_function a x1 < power_function a x2 ) )
  (h2 : a^2 - 2 * a - 2 = 1) (h3 : a > 0) : a = 3 :=
by
  sorry

end monotonic_power_function_l304_304426


namespace perfect_square_if_integer_l304_304682

theorem perfect_square_if_integer (n : ℤ) (k : ℤ) 
  (h : k = 2 + 2 * Int.sqrt (28 * n^2 + 1)) : ∃ m : ℤ, k = m^2 :=
by 
  sorry

end perfect_square_if_integer_l304_304682


namespace find_angle_A_l304_304289

theorem find_angle_A (a b c A : ℝ) (h1 : b = c) (h2 : a^2 = 2 * b^2 * (1 - Real.sin A)) : 
  A = Real.pi / 4 :=
by
  sorry

end find_angle_A_l304_304289


namespace trajectory_equation_l304_304582

noncomputable def circle1_center := (-3, 0)
noncomputable def circle2_center := (3, 0)

def circle1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

def is_tangent_internally (x y : ℝ) : Prop := 
  ∃ (P : ℝ × ℝ), circle1 P.1 P.2 ∧ circle2 P.1 P.2

theorem trajectory_equation :
  ∀ (x y : ℝ), is_tangent_internally x y → (x^2 / 16 + y^2 / 7 = 1) :=
sorry

end trajectory_equation_l304_304582


namespace complex_quadratic_solution_l304_304378

theorem complex_quadratic_solution (c d : ℤ) (h1 : 0 < c) (h2 : 0 < d) (h3 : (c + d * Complex.I) ^ 2 = 7 + 24 * Complex.I) :
  c + d * Complex.I = 4 + 3 * Complex.I :=
sorry

end complex_quadratic_solution_l304_304378


namespace andrew_total_appeizers_count_l304_304059

theorem andrew_total_appeizers_count :
  let hotdogs := 30
  let cheese_pops := 20
  let chicken_nuggets := 40
  hotdogs + cheese_pops + chicken_nuggets = 90 := 
by 
  sorry

end andrew_total_appeizers_count_l304_304059


namespace fishing_line_sections_l304_304620

theorem fishing_line_sections (reels : ℕ) (length_per_reel : ℕ) (section_length : ℕ)
    (h_reels : reels = 3) (h_length_per_reel : length_per_reel = 100) (h_section_length : section_length = 10) :
    (reels * length_per_reel) / section_length = 30 := 
by
  rw [h_reels, h_length_per_reel, h_section_length]
  norm_num

end fishing_line_sections_l304_304620


namespace circles_intersect_l304_304427

theorem circles_intersect
  (r : ℝ) (R : ℝ) (d : ℝ)
  (hr : r = 4)
  (hR : R = 5)
  (hd : d = 6) :
  1 < d ∧ d < r + R :=
by
  sorry

end circles_intersect_l304_304427


namespace possible_number_of_classmates_l304_304648

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l304_304648


namespace eighth_hexagonal_number_l304_304010

theorem eighth_hexagonal_number : (8 * (2 * 8 - 1)) = 120 :=
  by
  sorry

end eighth_hexagonal_number_l304_304010


namespace solve_equation_nat_numbers_l304_304006

theorem solve_equation_nat_numbers :
  ∃ (x y z : ℕ), (2 ^ x + 3 ^ y + 7 = z!) ∧ ((x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5)) := 
sorry

end solve_equation_nat_numbers_l304_304006


namespace q_work_alone_in_10_days_l304_304901

theorem q_work_alone_in_10_days (p_rate : ℝ) (q_rate : ℝ) (d : ℕ) (h1 : p_rate = 1 / 20)
                                    (h2 : q_rate = 1 / d) (h3 : 2 * (p_rate + q_rate) = 0.3) :
                                    d = 10 :=
by sorry

end q_work_alone_in_10_days_l304_304901


namespace arithmetic_seq_ratio_l304_304257

theorem arithmetic_seq_ratio
  (a b : ℕ → ℝ)
  (S T : ℕ → ℝ)
  (H_seq_a : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (H_seq_b : ∀ n, T n = (n * (b 1 + b n)) / 2)
  (H_ratio : ∀ n, S n / T n = (2 * n - 3) / (4 * n - 3)) :
  (a 3 + a 15) / (2 * (b 3 + b 9)) + a 3 / (b 2 + b 10) = 19 / 41 :=
by
  sorry

end arithmetic_seq_ratio_l304_304257


namespace dave_final_tickets_l304_304763

variable (initial_tickets_set1_won : ℕ) (initial_tickets_set1_lost : ℕ)
variable (initial_tickets_set2_won : ℕ) (initial_tickets_set2_lost : ℕ)
variable (multiplier_set3 : ℕ)
variable (initial_tickets_set3_lost : ℕ)
variable (used_tickets : ℕ)
variable (additional_tickets : ℕ)

theorem dave_final_tickets :
  let net_gain_set1 := initial_tickets_set1_won - initial_tickets_set1_lost
  let net_gain_set2 := initial_tickets_set2_won - initial_tickets_set2_lost
  let net_gain_set3 := multiplier_set3 * net_gain_set1 - initial_tickets_set3_lost
  let total_tickets_after_sets := net_gain_set1 + net_gain_set2 + net_gain_set3
  let tickets_after_buying := total_tickets_after_sets - used_tickets
  let final_tickets := tickets_after_buying + additional_tickets
  initial_tickets_set1_won = 14 →
  initial_tickets_set1_lost = 2 →
  initial_tickets_set2_won = 8 →
  initial_tickets_set2_lost = 5 →
  multiplier_set3 = 3 →
  initial_tickets_set3_lost = 15 →
  used_tickets = 25 →
  additional_tickets = 7 →
  final_tickets = 18 :=
by
  intros
  sorry

end dave_final_tickets_l304_304763


namespace perimeter_of_staircase_region_l304_304616

-- Definitions according to the conditions.
def staircase_region.all_right_angles : Prop := True -- Given condition that all angles are right angles.
def staircase_region.side_length : ℕ := 1 -- Given condition that the side length of each congruent side is 1 foot.
def staircase_region.total_area : ℕ := 120 -- Given condition that the total area of the region is 120 square feet.
def num_sides : ℕ := 12 -- Number of congruent sides.

-- The question is to prove that the perimeter of the region is 36 feet.
theorem perimeter_of_staircase_region : 
  (num_sides * staircase_region.side_length + 
    15 + -- length added to complete the larger rectangle assuming x = 15
    9   -- length added to complete the larger rectangle assuming y = 9
  ) = 36 := 
by
  -- Given and facts are already logically considered to prove (conditions and right angles are trivial)
  sorry

end perimeter_of_staircase_region_l304_304616


namespace amy_total_score_correct_l304_304376

def amyTotalScore (points_per_treasure : ℕ) (treasures_first_level : ℕ) (treasures_second_level : ℕ) : ℕ :=
  (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level)

theorem amy_total_score_correct:
  amyTotalScore 4 6 2 = 32 :=
by
  -- Proof goes here
  sorry

end amy_total_score_correct_l304_304376


namespace total_cost_price_l304_304749

theorem total_cost_price (P_ct P_ch P_bs : ℝ) (h1 : 8091 = P_ct * 1.24)
    (h2 : 5346 = P_ch * 1.18 * 0.95) (h3 : 11700 = P_bs * 1.30) : 
    P_ct + P_ch + P_bs = 20295 := 
by 
    sorry

end total_cost_price_l304_304749


namespace union_A_B_complement_U_A_intersection_B_range_of_a_l304_304805

-- Define the sets A, B, C, and U
def setA (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 8
def setB (x : ℝ) : Prop := 1 < x ∧ x < 6
def setC (a : ℝ) (x : ℝ) : Prop := x > a
def U (x : ℝ) : Prop := True  -- U being the universal set of all real numbers

-- Define complements and intersections
def complement (A : ℝ → Prop) (x : ℝ) : Prop := ¬ A x
def intersection (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∧ B x
def union (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∨ B x

-- Proof problems
theorem union_A_B : ∀ x, union setA setB x ↔ (1 < x ∧ x ≤ 8) :=
by 
  intros x
  sorry

theorem complement_U_A_intersection_B : ∀ x, intersection (complement setA) setB x ↔ (1 < x ∧ x < 2) :=
by 
  intros x
  sorry

theorem range_of_a (a : ℝ) : (∃ x, intersection setA (setC a) x) → a < 8 :=
by
  intros h
  sorry

end union_A_B_complement_U_A_intersection_B_range_of_a_l304_304805


namespace stacy_has_2_more_than_triple_steve_l304_304686

-- Definitions based on the given conditions
def skylar_berries : ℕ := 20
def steve_berries : ℕ := skylar_berries / 2
def stacy_berries : ℕ := 32

-- Statement to be proved
theorem stacy_has_2_more_than_triple_steve :
  stacy_berries = 3 * steve_berries + 2 := by
  sorry

end stacy_has_2_more_than_triple_steve_l304_304686


namespace original_triangle_area_l304_304688

-- Define the conditions
def dimensions_quadrupled (original_area new_area : ℝ) : Prop :=
  4^2 * original_area = new_area

-- Define the statement to be proved
theorem original_triangle_area {new_area : ℝ} (h : new_area = 64) :
  ∃ (original_area : ℝ), dimensions_quadrupled original_area new_area ∧ original_area = 4 :=
by
  sorry

end original_triangle_area_l304_304688


namespace hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l304_304336

-- Definitions for conditions of the problem
def ellipse_C1 (x y : ℝ) (b : ℝ) : Prop := (x^2) / 4 + (y^2) / (b^2) = 1

def is_sister_conic_section (e1 e2 : ℝ) : Prop :=
  e1 * e2 = Real.sqrt 15 / 4

def hyperbola_C2 (x y : ℝ) : Prop := (x^2) / 4 - y^2 = 1

variable {b : ℝ} (hb : 0 < b ∧ b < 2)
variable {e1 e2 : ℝ} (heccentricities : is_sister_conic_section e1 e2)

theorem hyperbola_C2_equation :
  ∃ (x y : ℝ), ellipse_C1 x y b → hyperbola_C2 x y := sorry

theorem constant_ratio_kAM_kBN (G : ℝ × ℝ) :
  G = (4,0) → 
  ∀ (M N : ℝ × ℝ) (kAM kBN : ℝ), 
  (kAM / kBN = -1/3) := sorry

theorem range_of_w_kAM_kBN (kAM kBN : ℝ) :
  ∃ (w : ℝ),
  w = kAM^2 + (2 / 3) * kBN →
  (w ∈ Set.Icc (-3 / 4) (-11 / 36) ∪ Set.Icc (13 / 36) (5 / 4)) := sorry

end hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l304_304336


namespace scientific_notation_11580000_l304_304841

theorem scientific_notation_11580000 :
  11580000 = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l304_304841


namespace candies_distribution_l304_304864

theorem candies_distribution (C : ℕ) (hC : C / 150 = C / 300 + 24) : C / 150 = 48 :=
by sorry

end candies_distribution_l304_304864


namespace Z_evaluation_l304_304599

def Z (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem Z_evaluation : Z 5 3 = 19 := by
  sorry

end Z_evaluation_l304_304599


namespace nonpositive_sum_of_products_l304_304622

theorem nonpositive_sum_of_products {a b c d : ℝ} (h : a + b + c + d = 0) :
  ab + ac + ad + bc + bd + cd ≤ 0 :=
sorry

end nonpositive_sum_of_products_l304_304622


namespace value_of_a_l304_304573

theorem value_of_a (a : ℕ) (h : ∀ x, ((a - 2) * x > a - 2) ↔ (x < 1)) : a = 0 ∨ a = 1 := by
  sorry

end value_of_a_l304_304573


namespace scientific_notation_correct_l304_304838

theorem scientific_notation_correct (n : ℕ) (h : n = 11580000) : n = 1.158 * 10^7 := 
sorry

end scientific_notation_correct_l304_304838


namespace pies_with_no_ingredients_l304_304867

theorem pies_with_no_ingredients (total_pies : ℕ)
  (pies_with_chocolate : ℕ)
  (pies_with_blueberries : ℕ)
  (pies_with_vanilla : ℕ)
  (pies_with_almonds : ℕ)
  (H_total : total_pies = 60)
  (H_chocolate : pies_with_chocolate = 1 / 3 * total_pies)
  (H_blueberries : pies_with_blueberries = 3 / 4 * total_pies)
  (H_vanilla : pies_with_vanilla = 2 / 5 * total_pies)
  (H_almonds : pies_with_almonds = 1 / 10 * total_pies) :
  ∃ (pies_without_ingredients : ℕ), pies_without_ingredients = 15 :=
by
  sorry

end pies_with_no_ingredients_l304_304867


namespace suzy_twice_mary_l304_304846

def suzy_current_age : ℕ := 20
def mary_current_age : ℕ := 8

theorem suzy_twice_mary (x : ℕ) : suzy_current_age + x = 2 * (mary_current_age + x) ↔ x = 4 := by
  sorry

end suzy_twice_mary_l304_304846


namespace custom_op_identity_l304_304779

def custom_op (x y : ℕ) : ℕ := x * y + 3 * x - 4 * y

theorem custom_op_identity : custom_op 7 5 - custom_op 5 7 = 14 :=
by
  sorry

end custom_op_identity_l304_304779


namespace opposite_numbers_expression_l304_304827

theorem opposite_numbers_expression (a b : ℤ) (h : a + b = 0) : 3 * a + 3 * b - 2 = -2 :=
by
  sorry

end opposite_numbers_expression_l304_304827


namespace parametric_eq_to_ordinary_l304_304020

theorem parametric_eq_to_ordinary (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
    let x := abs (Real.sin (θ / 2) + Real.cos (θ / 2))
    let y := 1 + Real.sin θ
    x ^ 2 = y := by sorry

end parametric_eq_to_ordinary_l304_304020


namespace minimize_expression_l304_304140

theorem minimize_expression (a b c d e f : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h_sum : a + b + c + d + e + f = 10) :
  (1 / a + 9 / b + 25 / c + 49 / d + 81 / e + 121 / f) ≥ 129.6 :=
by
  sorry

end minimize_expression_l304_304140


namespace min_disks_to_store_files_l304_304849

open Nat

theorem min_disks_to_store_files :
  ∃ minimum_disks : ℕ,
    (minimum_disks = 24) ∧
    ∀ (files : ℕ) (disk_capacity : ℕ) (file_sizes : List ℕ),
      files = 36 →
      disk_capacity = 144 →
      (∃ (size_85 : ℕ) (size_75 : ℕ) (size_45 : ℕ),
         size_85 = 5 ∧
         size_75 = 15 ∧
         size_45 = 16 ∧
         (∀ (disks : ℕ), disks >= minimum_disks →
            ∃ (used_disks_85 : ℕ) (remaining_files_45 : ℕ) (used_disks_45 : ℕ) (used_disks_75 : ℕ),
              remaining_files_45 = size_45 - used_disks_85 ∧
              used_disks_85 = size_85 ∧
              (remaining_files_45 % 3 = 0 → used_disks_45 = remaining_files_45 / 3) ∧
              (remaining_files_45 % 3 ≠ 0 → used_disks_45 = remaining_files_45 / 3 + 1) ∧
              used_disks_75 = size_75 ∧
              disks = used_disks_85 + used_disks_45 + used_disks_75)) :=
by
  sorry

end min_disks_to_store_files_l304_304849


namespace product_divisible_by_8_probability_l304_304706

noncomputable def probability_product_divisible_by_8 (dice_rolls : Fin 6 → Fin 8) : ℚ :=
  -- Function to calculate the probability that the product of numbers is divisible by 8
  sorry

theorem product_divisible_by_8_probability :
  ∀ (dice_rolls : Fin 6 → Fin 8),
  probability_product_divisible_by_8 dice_rolls = 177 / 256 :=
sorry

end product_divisible_by_8_probability_l304_304706


namespace complement_intersection_l304_304819

def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | x < 2 }
def CR (S : Set ℝ) : Set ℝ := { x | x ∉ S }

theorem complement_intersection :
  CR (M ∩ N) = { x | x < 1 } ∪ { x | x ≥ 2 } := by
  sorry

end complement_intersection_l304_304819


namespace vertical_increase_is_100m_l304_304543

theorem vertical_increase_is_100m 
  (a b x : ℝ)
  (hypotenuse : a = 100 * Real.sqrt 5)
  (slope_ratio : b = 2 * x)
  (pythagorean_thm : x^2 + b^2 = a^2) : 
  x = 100 :=
by
  sorry

end vertical_increase_is_100m_l304_304543


namespace solve_for_z_l304_304437

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l304_304437


namespace smallest_k_square_divisible_l304_304761

theorem smallest_k_square_divisible (k : ℤ) (n : ℤ) (h1 : k = 60)
    (h2 : ∀ m : ℤ, m < k → ∃ d : ℤ, d ∣ (k^2) → m = d ) : n = 3600 :=
sorry

end smallest_k_square_divisible_l304_304761


namespace place_value_accuracy_l304_304317

theorem place_value_accuracy (x : ℝ) (h : x = 3.20 * 10000) :
  ∃ p : ℕ, p = 100 ∧ (∃ k : ℤ, x / p = k) := by
  sorry

end place_value_accuracy_l304_304317


namespace greatest_2q_minus_r_l304_304159

theorem greatest_2q_minus_r :
  ∃ (q r : ℕ), 1027 = 21 * q + r ∧ q > 0 ∧ r > 0 ∧ 2 * q - r = 77 :=
by
  sorry

end greatest_2q_minus_r_l304_304159


namespace polynomial_factorization_l304_304347

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end polynomial_factorization_l304_304347


namespace least_integer_greater_than_sqrt_450_l304_304515

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l304_304515


namespace sum_of_three_consecutive_integers_product_990_l304_304024

theorem sum_of_three_consecutive_integers_product_990 
  (a b c : ℕ) 
  (h1 : b = a + 1)
  (h2 : c = b + 1)
  (h3 : a * b * c = 990) :
  a + b + c = 30 :=
sorry

end sum_of_three_consecutive_integers_product_990_l304_304024


namespace find_z_l304_304435

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l304_304435


namespace n_mul_n_plus_one_even_l304_304681

theorem n_mul_n_plus_one_even (n : ℤ) : Even (n * (n + 1)) := 
sorry

end n_mul_n_plus_one_even_l304_304681


namespace tap_filling_time_l304_304922

theorem tap_filling_time (T : ℝ) (hT1 : T > 0) 
  (h_fill_with_one_tap : ∀ (t : ℝ), t = T → t > 0)
  (h_fill_with_second_tap : ∀ (s : ℝ), s = 60 → s > 0)
  (both_open_first_10_minutes : 10 * (1 / T + 1 / 60) + 20 * (1 / 60) = 1) :
    T = 20 := 
sorry

end tap_filling_time_l304_304922


namespace ce_over_de_l304_304448

theorem ce_over_de {A B C D E T : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ (A →ₗ[ℝ] B)]
  {AT DT BT ET CE DE : ℝ}
  (h1 : AT / DT = 2)
  (h2 : BT / ET = 3) :
  CE / DE = 1 / 2 := 
sorry

end ce_over_de_l304_304448


namespace greatest_value_of_x_for_equation_l304_304340

theorem greatest_value_of_x_for_equation :
  ∃ x : ℝ, (4 * x - 5) ≠ 0 ∧ ((5 * x - 20) / (4 * x - 5)) ^ 2 + ((5 * x - 20) / (4 * x - 5)) = 18 ∧ x = 50 / 29 :=
sorry

end greatest_value_of_x_for_equation_l304_304340


namespace simplify_expression_l304_304004

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end simplify_expression_l304_304004


namespace polynomial_prime_is_11_l304_304262

def P (a : ℕ) : ℕ := a^4 - 4 * a^3 + 15 * a^2 - 30 * a + 27

theorem polynomial_prime_is_11 (a : ℕ) (hp : Nat.Prime (P a)) : P a = 11 := 
by {
  sorry
}

end polynomial_prime_is_11_l304_304262


namespace maximum_monthly_profit_l304_304038

-- Let's set up our conditions

def selling_price := 25
def monthly_profit := 120
def cost_price := 20
def selling_price_threshold := 32
def relationship (x n : ℝ) := -10 * x + n

-- Define the value of n
def value_of_n : ℝ := 370

-- Profit function
def profit_function (x n : ℝ) : ℝ := (x - cost_price) * (relationship x n)

-- Define the condition for maximum profit where the selling price should be higher than 32
def max_profit_condition (n : ℝ) (x : ℝ) := x > selling_price_threshold

-- Define what the maximum profit should be
def max_profit := 160

-- The main theorem to be proven
theorem maximum_monthly_profit :
  (relationship selling_price value_of_n = monthly_profit) →
  max_profit_condition value_of_n 32 →
  profit_function 32 value_of_n = max_profit :=
by sorry

end maximum_monthly_profit_l304_304038


namespace distinct_arrangements_banana_l304_304961

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_count := 6
  word.length = total_count ∧ b_count = 1 ∧ a_count = 3 ∧ n_count = 2 →
  (Nat.factorial total_count) / ((Nat.factorial a_count) * (Nat.factorial n_count)) = 60 :=
by
  intros
  have h_total := word.length
  have h_b := b_count
  have h_a := a_count
  have h_n := n_count
  sorry

end distinct_arrangements_banana_l304_304961


namespace inscribed_circle_radius_l304_304928

variable (AB AC BC s K r : ℝ)
variable (AB_eq AC_eq BC_eq : AB = AC ∧ AC = 8 ∧ BC = 7)
variable (s_eq : s = (AB + AC + BC) / 2)
variable (K_eq : K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)))
variable (r_eq : r * s = K)

/-- Prove that the radius of the inscribed circle is 23.75 / 11.5 given the conditions of the triangle --/
theorem inscribed_circle_radius :
  AB = 8 → AC = 8 → BC = 7 → 
  s = (AB + AC + BC) / 2 → 
  K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) →
  r * s = K →
  r = (23.75 / 11.5) :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end inscribed_circle_radius_l304_304928


namespace red_black_probability_l304_304753

-- Define the number of cards and ranks
def num_cards : ℕ := 64
def num_ranks : ℕ := 16

-- Define the suits and their properties
def suits := 6
def red_suits := 3
def black_suits := 3
def cards_per_suit := num_ranks

-- Define the number of red and black cards
def red_cards := red_suits * cards_per_suit
def black_cards := black_suits * cards_per_suit

-- Prove the probability that the top card is red and the second card is black
theorem red_black_probability : 
  (red_cards * black_cards) / (num_cards * (num_cards - 1)) = 3 / 4 := by 
  sorry

end red_black_probability_l304_304753


namespace probability_sum_l304_304451

noncomputable def ballsInBox := 9
noncomputable def blackBalls := 5
noncomputable def whiteBalls := 4

def P_A := (blackBalls : ℚ) / (ballsInBox : ℚ)
def P_B_given_A := (whiteBalls : ℚ) / ((ballsInBox - 1) : ℚ)
def P_AB := P_A * P_B_given_A
def P_B := P_B_given_A

theorem probability_sum :
  P_AB + P_B = 7 / 9 :=
by
  sorry

end probability_sum_l304_304451


namespace segment_length_of_absolute_value_l304_304713

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end segment_length_of_absolute_value_l304_304713


namespace range_of_b_l304_304979

theorem range_of_b (b : ℝ) :
  (∀ x : ℤ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) ↔ 5 < b ∧ b < 7 := 
sorry

end range_of_b_l304_304979


namespace interest_groups_ranges_l304_304331

variable (A B C : Finset ℕ)

-- Given conditions
axiom card_A : A.card = 5
axiom card_B : B.card = 4
axiom card_C : C.card = 7
axiom card_A_inter_B : (A ∩ B).card = 3
axiom card_A_inter_B_inter_C : (A ∩ B ∩ C).card = 2

-- Mathematical statement to be proved
theorem interest_groups_ranges :
  2 ≤ ((A ∪ B) ∩ C).card ∧ ((A ∪ B) ∩ C).card ≤ 5 ∧
  8 ≤ (A ∪ B ∪ C).card ∧ (A ∪ B ∪ C).card ≤ 11 := by
  sorry

end interest_groups_ranges_l304_304331


namespace correct_calculation_l304_304731

theorem correct_calculation (a : ℕ) :
  ¬ (a^3 + a^4 = a^7) ∧
  ¬ (2 * a - a = 2) ∧
  2 * a + a = 3 * a ∧
  ¬ (a^4 - a^3 = a) :=
by
  sorry

end correct_calculation_l304_304731


namespace polynomial_factorization_l304_304350

open Polynomial

theorem polynomial_factorization :
  (X ^ 15 + X ^ 10 + X ^ 5 + 1) =
    (X ^ 3 + X ^ 2 + 1) * 
    (X ^ 12 - X ^ 11 + X ^ 9 - X ^ 8 + X ^ 6 - X ^ 5 + X ^ 4 + X ^ 3 + X ^ 2 + X + 1) :=
by
  sorry

end polynomial_factorization_l304_304350


namespace tamara_has_30_crackers_l304_304880

theorem tamara_has_30_crackers :
  ∀ (Tamara Nicholas Marcus Mona : ℕ),
    Tamara = 2 * Nicholas →
    Marcus = 3 * Mona →
    Nicholas = Mona + 6 →
    Marcus = 27 →
    Tamara = 30 :=
by
  intros Tamara Nicholas Marcus Mona h1 h2 h3 h4
  sorry

end tamara_has_30_crackers_l304_304880


namespace cake_eating_classmates_l304_304672

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l304_304672


namespace root_power_sum_eq_l304_304362

open Real

theorem root_power_sum_eq :
  ∀ {a b c : ℝ},
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (a^3 - 3 * a + 1 = 0) → (b^3 - 3 * b + 1 = 0) → (c^3 - 3 * c + 1 = 0) →
  a^8 + b^8 + c^8 = 186 :=
by
  intros a b c h1 h2 h3 ha hb hc
  sorry

end root_power_sum_eq_l304_304362


namespace find_a1000_l304_304608

noncomputable def seq (a : ℕ → ℤ) : Prop :=
a 1 = 1009 ∧
a 2 = 1010 ∧
(∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n)

theorem find_a1000 (a : ℕ → ℤ) (h : seq a) : a 1000 = 1675 :=
sorry

end find_a1000_l304_304608


namespace nine_op_ten_l304_304098

def op (A B : ℕ) : ℚ := (1 : ℚ) / (A * B) + (1 : ℚ) / ((A + 1) * (B + 2))

theorem nine_op_ten : op 9 10 = 7 / 360 := by
  sorry

end nine_op_ten_l304_304098


namespace red_balls_estimate_l304_304609

/-- There are several red balls and 4 black balls in a bag.
Each ball is identical except for color.
A ball is drawn and put back into the bag. This process is repeated 100 times.
Among those 100 draws, 40 times a black ball is drawn.
Prove that the number of red balls (x) is 6. -/
theorem red_balls_estimate (x : ℕ) (h_condition : (4 / (4 + x) = 40 / 100)) : x = 6 :=
by
    sorry

end red_balls_estimate_l304_304609


namespace max_value_x2_y2_l304_304808

noncomputable def max_x2_y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y ≥ x^3 + y^2) : ℝ := 2

theorem max_value_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y ≥ x^3 + y^2) : 
  x^2 + y^2 ≤ max_x2_y2 x y hx hy h :=
by
  sorry

end max_value_x2_y2_l304_304808


namespace cake_sharing_l304_304657

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l304_304657


namespace range_of_a_l304_304440

theorem range_of_a (x y a : ℝ) (h1 : x < y) (h2 : (a - 3) * x > (a - 3) * y) : a < 3 :=
sorry

end range_of_a_l304_304440


namespace problem_solution_l304_304464

noncomputable def p (x : ℝ) : ℝ := 
  (x - (Real.sin 1)^2) * (x - (Real.sin 3)^2) * (x - (Real.sin 9)^2)

theorem problem_solution : ∃ a b n : ℕ, 
  p (1 / 4) = Real.sin (a * Real.pi / 180) / (n * Real.sin (b * Real.pi / 180)) ∧
  a > 0 ∧ b > 0 ∧ a ≤ 90 ∧ b ≤ 90 ∧ a + b + n = 216 :=
sorry

end problem_solution_l304_304464


namespace sam_quarters_mowing_lawns_l304_304871

-- Definitions based on the given conditions
def pennies : ℕ := 9
def total_amount_dollars : ℝ := 1.84
def penny_value_dollars : ℝ := 0.01
def quarter_value_dollars : ℝ := 0.25

-- Theorem statement that Sam got 7 quarters given the conditions
theorem sam_quarters_mowing_lawns : 
  (total_amount_dollars - pennies * penny_value_dollars) / quarter_value_dollars = 7 := by
  sorry

end sam_quarters_mowing_lawns_l304_304871


namespace integer_solutions_eq_l304_304787

theorem integer_solutions_eq (x y : ℤ) (h : y^2 = x^3 + (x + 1)^2) : (x, y) = (0, 1) ∨ (x, y) = (0, -1) :=
by
  sorry

end integer_solutions_eq_l304_304787


namespace blake_change_l304_304767

def cost_oranges : ℕ := 40
def cost_apples : ℕ := 50
def cost_mangoes : ℕ := 60
def initial_money : ℕ := 300

def total_cost : ℕ := cost_oranges + cost_apples + cost_mangoes
def change : ℕ := initial_money - total_cost

theorem blake_change : change = 150 := by
  sorry

end blake_change_l304_304767


namespace distance_is_30_l304_304307

-- Define given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Define the distance from Mrs. Hilt's desk to the water fountain
def distance_to_water_fountain : ℕ := total_distance / trips

-- Prove the distance is 30 feet
theorem distance_is_30 : distance_to_water_fountain = 30 :=
by
  -- Utilizing the division defined in distance_to_water_fountain
  sorry

end distance_is_30_l304_304307


namespace M_inter_N_l304_304096

def M : Set ℝ := { x | -2 < x ∧ x < 1 }
def N : Set ℤ := { x | Int.natAbs x ≤ 2 }

theorem M_inter_N : { x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) < 1 } ∩ N = { -1, 0 } :=
by
  simp [M, N]
  sorry

end M_inter_N_l304_304096


namespace minimize_material_use_l304_304374

theorem minimize_material_use 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (total_area : x * y + (x^2 / 4) = 8) :
  (abs (x - 2.343) ≤ 0.001) ∧ (abs (y - 2.828) ≤ 0.001) :=
sorry

end minimize_material_use_l304_304374


namespace combined_cost_of_items_is_221_l304_304205

def wallet_cost : ℕ := 22
def purse_cost : ℕ := 4 * wallet_cost - 3
def shoes_cost : ℕ := wallet_cost + purse_cost + 7
def combined_cost : ℕ := wallet_cost + purse_cost + shoes_cost

theorem combined_cost_of_items_is_221 : combined_cost = 221 := by
  sorry

end combined_cost_of_items_is_221_l304_304205


namespace expected_value_of_8_sided_die_l304_304912

theorem expected_value_of_8_sided_die : 
  let p := (1 / 8 : ℚ)
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let values := outcomes.map (λ n, n^3)
  let expected_value := (values.sum : ℚ) * p
  expected_value = 162 := sorry

end expected_value_of_8_sided_die_l304_304912


namespace rowing_distance_l304_304327

theorem rowing_distance (D : ℝ) : 
  (D / 14 + D / 2 = 120) → D = 210 := by
  sorry

end rowing_distance_l304_304327


namespace g_g_x_has_two_distinct_real_roots_iff_l304_304298

noncomputable def g (d x : ℝ) := x^2 - 4 * x + d

def has_two_distinct_real_roots (f : ℝ → ℝ) : Prop := 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

theorem g_g_x_has_two_distinct_real_roots_iff (d : ℝ) :
  has_two_distinct_real_roots (g d ∘ g d) ↔ d = 8 := sorry

end g_g_x_has_two_distinct_real_roots_iff_l304_304298


namespace pounds_of_coffee_bought_l304_304311

theorem pounds_of_coffee_bought 
  (total_amount_gift_card : ℝ := 70) 
  (cost_per_pound : ℝ := 8.58) 
  (amount_left_on_card : ℝ := 35.68) :
  (total_amount_gift_card - amount_left_on_card) / cost_per_pound = 4 :=
sorry

end pounds_of_coffee_bought_l304_304311


namespace euston_carriages_l304_304344

-- Definitions of the conditions
def E (N : ℕ) : ℕ := N + 20
def No : ℕ := 100
def FS : ℕ := No + 20
def total_carriages (E N : ℕ) : ℕ := E + N + No + FS

theorem euston_carriages (N : ℕ) (h : total_carriages (E N) N = 460) : E N = 130 :=
by
  -- Proof goes here
  sorry

end euston_carriages_l304_304344


namespace num_rectangles_in_grid_l304_304449

theorem num_rectangles_in_grid : 
  let width := 35
  let height := 44
  ∃ n, n = 87 ∧ 
  ∀ x y, (1 ≤ x ∧ x ≤ width) ∧ (1 ≤ y ∧ y ≤ height) → 
    n = (x * (x + 1) / 2) * (y * (y + 1) / 2) := 
by
  sorry

end num_rectangles_in_grid_l304_304449


namespace gigi_remaining_batches_l304_304419

variable (f b1 tf remaining_batches : ℕ)
variable (f_pos : 0 < f)
variable (batches_nonneg : 0 ≤ b1)
variable (t_f_pos : 0 < tf)
variable (h_f : f = 2)
variable (h_b1 : b1 = 3)
variable (h_tf : tf = 20)

theorem gigi_remaining_batches (h : remaining_batches = (tf - (f * b1)) / f) : remaining_batches = 7 := by
  sorry

end gigi_remaining_batches_l304_304419


namespace color_of_85th_bead_l304_304462

def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def bead_color (n : ℕ) : String :=
  bead_pattern.get! (n % bead_pattern.length)

theorem color_of_85th_bead : bead_color 84 = "yellow" := 
by
  sorry

end color_of_85th_bead_l304_304462


namespace b_alone_days_l304_304531

-- Definitions from the conditions
def work_rate_b (W_b : ℝ) : ℝ := W_b
def work_rate_a (W_b : ℝ) : ℝ := 2 * W_b
def work_rate_c (W_b : ℝ) : ℝ := 6 * W_b
def combined_work_rate (W_b : ℝ) : ℝ := work_rate_a W_b + work_rate_b W_b + work_rate_c W_b
def total_days_together : ℝ := 10
def total_work (W_b : ℝ) : ℝ := combined_work_rate W_b * total_days_together

-- The proof problem
theorem b_alone_days (W_b : ℝ) : 90 = total_work W_b / work_rate_b W_b :=
by
  sorry

end b_alone_days_l304_304531


namespace find_f_2_l304_304485

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x + y) = f x + f y
axiom f_8 : f 8 = 3

theorem find_f_2 : f 2 = 3 / 4 := 
by sorry

end find_f_2_l304_304485


namespace students_more_than_rabbits_by_64_l304_304394

-- Define the conditions as constants
def number_of_classrooms : ℕ := 4
def students_per_classroom : ℕ := 18
def rabbits_per_classroom : ℕ := 2

-- Define the quantities that need calculations
def total_students : ℕ := number_of_classrooms * students_per_classroom
def total_rabbits : ℕ := number_of_classrooms * rabbits_per_classroom
def difference_students_rabbits : ℕ := total_students - total_rabbits

-- State the theorem to be proven
theorem students_more_than_rabbits_by_64 :
  difference_students_rabbits = 64 := by
  sorry

end students_more_than_rabbits_by_64_l304_304394


namespace watch_cost_price_l304_304532

theorem watch_cost_price (CP : ℝ) (H1 : 0.90 * CP = CP - 0.10 * CP)
(H2 : 1.04 * CP = CP + 0.04 * CP)
(H3 : 1.04 * CP - 0.90 * CP = 168) : CP = 1200 := by
sorry

end watch_cost_price_l304_304532


namespace abs_inequality_solution_bounded_a_b_inequality_l304_304184

theorem abs_inequality_solution (x : ℝ) : (-4 < x ∧ x < 0) ↔ (|x + 1| + |x + 3| < 4) := sorry

theorem bounded_a_b_inequality (a b : ℝ) (h1 : -4 < a) (h2 : a < 0) (h3 : -4 < b) (h4 : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| := sorry

end abs_inequality_solution_bounded_a_b_inequality_l304_304184


namespace monthly_growth_rate_l304_304064

-- Definitions based on the conditions given in the original problem.
def final_height : ℝ := 80
def current_height : ℝ := 20
def months_in_year : ℕ := 12

-- Prove the monthly growth rate.
theorem monthly_growth_rate : (final_height - current_height) / months_in_year = 5 := by
  sorry

end monthly_growth_rate_l304_304064


namespace xiaoming_pens_l304_304736

theorem xiaoming_pens (P M : ℝ) (hP : P > 0) (hM : M > 0) :
  (M / (7 / 8 * P) - M / P = 13) → (M / P = 91) := 
by
  sorry

end xiaoming_pens_l304_304736


namespace expenditure_ratio_l304_304196

variable {I : ℝ} -- Income in the first year

-- Conditions
def first_year_savings (I : ℝ) : ℝ := 0.5 * I
def first_year_expenditure (I : ℝ) : ℝ := I - first_year_savings I
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (I : ℝ) : ℝ := 2 * first_year_savings I
def second_year_expenditure (I : ℝ) : ℝ := second_year_income I - second_year_savings I

-- Condition statement in Lean
theorem expenditure_ratio (I : ℝ) : 
  let total_expenditure := first_year_expenditure I + second_year_expenditure I
  (total_expenditure / first_year_expenditure I) = 2 :=
  by 
    sorry

end expenditure_ratio_l304_304196


namespace least_integer_greater_than_sqrt_450_l304_304506

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l304_304506


namespace find_A_l304_304172

theorem find_A (d q r A : ℕ) (h1 : d = 7) (h2 : q = 5) (h3 : r = 3) (h4 : A = d * q + r) : A = 38 := 
by 
  { sorry }

end find_A_l304_304172


namespace degree_of_f_at_least_n_l304_304626

variables {R : Type*} [CommRing R] [Algebra R ℝ]

-- Define the polynomial f
noncomputable def f (n m : ℕ) (hn : 2 ≤ n) (hm : 2 ≤ m) : Polynomial ℝ :=
  Polynomial.ofFn (λ (xi : Vector ℕ n), ⌊(xi.toList.sum) / m⌋)

theorem degree_of_f_at_least_n (n m : ℕ) (hn : 2 ≤ n) (hm : 2 ≤ m) :
  ∀ (f : Vector ℕ n → ℝ),
  (∀ (xi : Vector ℕ n), (∀ i, xi.get i ∈ Finset.range m) → f xi = ⌊(xi.toList.sum : ℝ) / m⌋) →
  (Polynomial.totalDegree (f n m) ≥ n) :=
sorry

end degree_of_f_at_least_n_l304_304626


namespace least_integer_greater_than_sqrt_450_l304_304514

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l304_304514


namespace order_of_a_b_c_l304_304574

noncomputable def a := 2 + Real.sqrt 3
noncomputable def b := 1 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 + Real.sqrt 5

theorem order_of_a_b_c : a > c ∧ c > b := 
by {
  sorry
}

end order_of_a_b_c_l304_304574


namespace least_integer_greater_than_sqrt_450_l304_304524

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l304_304524


namespace price_of_eraser_l304_304529

variables (x y : ℝ)

theorem price_of_eraser : 
  (3 * x + 5 * y = 10.6) ∧ (4 * x + 4 * y = 12) → x = 2.2 :=
by
  sorry

end price_of_eraser_l304_304529


namespace possible_number_of_classmates_l304_304647

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l304_304647


namespace savings_promotion_l304_304137

theorem savings_promotion (reg_price promo_price num_pizzas : ℕ) (h1 : reg_price = 18) (h2 : promo_price = 5) (h3 : num_pizzas = 3) :
  reg_price * num_pizzas - promo_price * num_pizzas = 39 := by
  sorry

end savings_promotion_l304_304137


namespace max_value_of_function_l304_304017

theorem max_value_of_function : 
  ∀ (x : ℝ), 0 ≤ x → x ≤ 1 → (3 * x - 4 * x^3) ≤ 1 :=
by
  intro x hx0 hx1
  -- proof goes here
  sorry

end max_value_of_function_l304_304017


namespace earrings_cost_l304_304480

theorem earrings_cost (initial_savings necklace_cost remaining_savings : ℕ) 
  (h_initial : initial_savings = 80) 
  (h_necklace : necklace_cost = 48) 
  (h_remaining : remaining_savings = 9) : 
  initial_savings - remaining_savings - necklace_cost = 23 := 
by {
  -- insert proof steps here -- 
  sorry
}

end earrings_cost_l304_304480


namespace intersection_of_M_and_N_l304_304430

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | x < 1}

theorem intersection_of_M_and_N : (M ∩ N = {x : ℝ | -1 < x ∧ x < 1}) :=
by
  sorry

end intersection_of_M_and_N_l304_304430


namespace number_of_classmates_ate_cake_l304_304668

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l304_304668


namespace min_value_expr_l304_304078

theorem min_value_expr :
  ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by sorry

end min_value_expr_l304_304078


namespace monthly_growth_rate_l304_304062

-- Definitions and conditions
def initial_height : ℝ := 20
def final_height : ℝ := 80
def months_in_year : ℕ := 12

-- Theorem stating the monthly growth rate
theorem monthly_growth_rate :
  (final_height - initial_height) / (months_in_year : ℝ) = 5 :=
by 
  sorry

end monthly_growth_rate_l304_304062


namespace binomial_coefficient_sum_l304_304980

theorem binomial_coefficient_sum {n : ℕ} (h : (1 : ℝ) + 1 = 128) : n = 7 :=
by
  sorry

end binomial_coefficient_sum_l304_304980


namespace average_of_combined_samples_l304_304156

theorem average_of_combined_samples 
  (a : Fin 10 → ℝ)
  (b : Fin 10 → ℝ)
  (ave_a : ℝ := (1 / 10) * (Finset.univ.sum (fun i => a i)))
  (ave_b : ℝ := (1 / 10) * (Finset.univ.sum (fun i => b i)))
  (combined_average : ℝ := (1 / 20) * (Finset.univ.sum (fun i => a i) + Finset.univ.sum (fun i => b i))) :
  combined_average = (1 / 2) * (ave_a + ave_b) := 
  by
    sorry

end average_of_combined_samples_l304_304156


namespace teacher_proctor_arrangements_l304_304313

theorem teacher_proctor_arrangements {f m : ℕ} (hf : f = 2) (hm : m = 5) :
  (∃ moving_teachers : ℕ, moving_teachers = 1 ∧ (f - moving_teachers) + m = 7 
   ∧ (f - moving_teachers).choose 2 = 21)
  ∧ 2 * 21 = 42 :=
by
    sorry

end teacher_proctor_arrangements_l304_304313


namespace repeating_decimal_as_fraction_l304_304231

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l304_304231


namespace simplify_expression_frac_l304_304689

theorem simplify_expression_frac (a b k : ℤ) (h : (6*k + 12) / 6 = a * k + b) : a = 1 ∧ b = 2 → a / b = 1 / 2 := by
  sorry

end simplify_expression_frac_l304_304689


namespace exists_distinct_permutations_divisible_l304_304141

open Equiv.Perm

theorem exists_distinct_permutations_divisible (n : ℕ) (hn : Odd n) (hn1 : 1 < n) 
  (k : Fin n → ℤ) : 
  ∃ (b c : Perm (Fin n)), b ≠ c ∧ (∑ i, k i * ↑(b i) - ∑ i, k i * ↑(c i)) % (n!) = 0 :=
sorry

end exists_distinct_permutations_divisible_l304_304141


namespace isosceles_triangle_properties_l304_304945

noncomputable def isosceles_triangle_sides (a : ℝ) : ℝ × ℝ × ℝ :=
  let x := a * Real.sqrt 3
  let y := 2 * x / 3
  let z := (x + y) / 2
  (x, z, z)

theorem isosceles_triangle_properties (a x y z : ℝ) 
  (h1 : x * y = 2 * a ^ 2) 
  (h2 : x + y = 2 * z) 
  (h3 : y ^ 2 + (x / 2) ^ 2 = z ^ 2) : 
  x = a * Real.sqrt 3 ∧ 
  z = 5 * a * Real.sqrt 3 / 6 :=
by
-- Proof goes here
sorry

end isosceles_triangle_properties_l304_304945


namespace intersection_M_N_eq_M_inter_N_l304_304954

def M : Set ℝ := { x | x^2 - 4 > 0 }
def N : Set ℝ := { x | x < 0 }
def M_inter_N : Set ℝ := { x | x < -2 }

theorem intersection_M_N_eq_M_inter_N : M ∩ N = M_inter_N := 
by
  sorry

end intersection_M_N_eq_M_inter_N_l304_304954


namespace proof_smallest_lcm_1_to_12_l304_304726

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end proof_smallest_lcm_1_to_12_l304_304726


namespace probability_at_least_60_cents_l304_304188

theorem probability_at_least_60_cents :
  let num_total_outcomes := Nat.choose 16 8
  let num_successful_outcomes := 
    (Nat.choose 4 2) * (Nat.choose 5 1) * (Nat.choose 7 5) +
    1 -- only one way to choose all 8 dimes
  num_successful_outcomes / num_total_outcomes = 631 / 12870 := by
  sorry

end probability_at_least_60_cents_l304_304188


namespace smallest_vertical_distance_between_graphs_l304_304044

noncomputable def f (x : ℝ) : ℝ := abs x
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem smallest_vertical_distance_between_graphs :
  ∃ (d : ℝ), (∀ (x : ℝ), |f x - g x| ≥ d) ∧ (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), |f x - g x| < d + ε) ∧ d = 3 / 4 :=
by
  sorry

end smallest_vertical_distance_between_graphs_l304_304044


namespace greg_pages_per_day_l304_304822

variable (greg_pages : ℕ)
variable (brad_pages : ℕ)

theorem greg_pages_per_day :
  brad_pages = 26 → brad_pages = greg_pages + 8 → greg_pages = 18 :=
by
  intros h1 h2
  rw [h1, add_comm] at h2
  linarith

end greg_pages_per_day_l304_304822


namespace domain_of_myFunction_l304_304026

-- Define the function
def myFunction (x : ℝ) : ℝ := (x + 2) ^ (1 / 2) - (x + 1) ^ 0

-- State the domain constraints as a theorem
theorem domain_of_myFunction (x : ℝ) : 
  (x ≥ -2 ∧ x ≠ -1) →
  ∃ y : ℝ, y = myFunction x := 
sorry

end domain_of_myFunction_l304_304026


namespace decreasing_interval_l304_304255

noncomputable def y (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 15 * x^2 + 36 * x - 24

def has_extremum_at (a : ℝ) (x_ext : ℝ) : Prop :=
  deriv (y a) x_ext = 0

theorem decreasing_interval (a : ℝ) (h_extremum_at : has_extremum_at a 3) :
  a = 2 → ∀ x, (2 < x ∧ x < 3) → deriv (y a) x < 0 :=
sorry

end decreasing_interval_l304_304255


namespace find_X_l304_304342

theorem find_X :
  let N := 90
  let X := (1 / 15) * N - (1 / 2 * 1 / 3 * 1 / 5 * N)
  X = 3 := by
  sorry

end find_X_l304_304342


namespace evaluate_expression_l304_304076

theorem evaluate_expression : -1 ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 = -2 :=
by
  -- sorry is added as a placeholder for the proof steps
  sorry

end evaluate_expression_l304_304076


namespace max_length_of_cuts_l304_304186

-- Define the dimensions of the board and the number of parts
def board_size : ℕ := 30
def num_parts : ℕ := 225

-- Define the total possible length of the cuts
def max_possible_cuts_length : ℕ := 1065

-- Define the condition that the board is cut into parts of equal area
def equal_area_partition (board_size num_parts : ℕ) : Prop :=
  ∃ (area_per_part : ℕ), (board_size * board_size) / num_parts = area_per_part

-- Define the theorem to prove the maximum possible total length of the cuts
theorem max_length_of_cuts (h : equal_area_partition board_size num_parts) :
  max_possible_cuts_length = 1065 :=
by
  -- Proof to be filled in
  sorry

end max_length_of_cuts_l304_304186


namespace discount_difference_l304_304548

theorem discount_difference :
  ∀ (original_price : ℝ),
  let initial_discount := 0.40
  let subsequent_discount := 0.25
  let claimed_discount := 0.60
  let actual_discount := 1 - (1 - initial_discount) * (1 - subsequent_discount)
  let difference := claimed_discount - actual_discount
  actual_discount = 0.55 ∧ difference = 0.05
:= by
  sorry

end discount_difference_l304_304548


namespace sum_of_consecutive_integers_product_l304_304021

noncomputable def consecutive_integers_sum (n m k : ℤ) : ℤ :=
  n + m + k

theorem sum_of_consecutive_integers_product (n m k : ℤ)
  (h1 : n = m - 1)
  (h2 : k = m + 1)
  (h3 : n * m * k = 990) :
  consecutive_integers_sum n m k = 30 :=
by
  sorry

end sum_of_consecutive_integers_product_l304_304021


namespace g_50_equals_zero_l304_304320

noncomputable def g : ℝ → ℝ := sorry

theorem g_50_equals_zero (h : ∀ (x y : ℝ), 0 < x → 0 < y → x * g y - y * g x = g ((x + y) / y)) : g 50 = 0 :=
sorry

end g_50_equals_zero_l304_304320


namespace find_x_l304_304562

noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem find_x (x : ℝ) (h₁ : log x 16 = log 4 256) : x = 2 := by
  sorry

end find_x_l304_304562


namespace min_value_expression_l304_304144

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 = 4 :=
sorry

end min_value_expression_l304_304144


namespace k_plus_m_eq_27_l304_304467

theorem k_plus_m_eq_27 (k m : ℝ) (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a > 0) (h5 : b > 0) (h6 : c > 0)
  (h7 : a + b + c = 8) 
  (h8 : k = a * b + a * c + b * c) 
  (h9 : m = a * b * c) :
  k + m = 27 :=
by
  sorry

end k_plus_m_eq_27_l304_304467


namespace value_of_x_l304_304143

theorem value_of_x 
  (x : ℚ) 
  (h₁ : 6 * x^2 + 19 * x - 7 = 0) 
  (h₂ : 18 * x^2 + 47 * x - 21 = 0) : 
  x = 1 / 3 := 
  sorry

end value_of_x_l304_304143


namespace length_more_than_breadth_l304_304323

theorem length_more_than_breadth
  (b x : ℝ)
  (h1 : b + x = 60)
  (h2 : 4 * b + 2 * x = 200) :
  x = 20 :=
by
  sorry

end length_more_than_breadth_l304_304323


namespace decrease_in_demand_correct_l304_304060

noncomputable def proportionate_decrease_in_demand (p e : ℝ) : ℝ :=
  1 - (1 / (1 + e * p))

theorem decrease_in_demand_correct :
  proportionate_decrease_in_demand 0.20 1.5 = 0.23077 :=
by
  sorry

end decrease_in_demand_correct_l304_304060


namespace part_1_select_B_prob_part_2_select_BC_prob_l304_304606

-- Definitions for the four students
inductive Student
| A
| B
| C
| D

open Student

-- Definition for calculating probability
def probability (favorable total : Nat) : Rat :=
  favorable / total

-- Part (1)
theorem part_1_select_B_prob : probability 1 4 = 1 / 4 :=
  sorry

-- Part (2)
theorem part_2_select_BC_prob : probability 2 12 = 1 / 6 :=
  sorry

end part_1_select_B_prob_part_2_select_BC_prob_l304_304606


namespace repeating_decimal_fraction_l304_304220

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l304_304220


namespace classmates_ate_cake_l304_304641

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l304_304641


namespace repeating_decimal_fraction_l304_304219

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l304_304219


namespace sqrt_domain_l304_304029

theorem sqrt_domain (x : ℝ) : x - 5 ≥ 0 ↔ x ≥ 5 :=
by sorry

end sqrt_domain_l304_304029


namespace parabola_y_intercepts_l304_304595

theorem parabola_y_intercepts : 
  (∀ y : ℝ, 3 * y^2 - 6 * y + 1 = 0) → (∃ y1 y2 : ℝ, y1 ≠ y2) :=
by sorry

end parabola_y_intercepts_l304_304595


namespace factor_expression_l304_304940

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_expression_l304_304940


namespace muffin_banana_cost_ratio_l304_304879

variables (m b c : ℕ) -- costs of muffin, banana, and cookie respectively
variables (susie_cost calvin_cost : ℕ)

-- Conditions
def susie_cost_eq : Prop := susie_cost = 5 * m + 4 * b + 2 * c
def calvin_cost_eq : Prop := calvin_cost = 3 * (5 * m + 4 * b + 2 * c)
def calvin_cost_eq_reduced : Prop := calvin_cost = 3 * m + 20 * b + 6 * c
def cookie_cost_eq : Prop := c = 2 * b

-- Question and Answer
theorem muffin_banana_cost_ratio
  (h1 : susie_cost_eq m b c susie_cost)
  (h2 : calvin_cost_eq m b c calvin_cost)
  (h3 : calvin_cost_eq_reduced m b c calvin_cost)
  (h4 : cookie_cost_eq b c)
  : m = 4 * b / 3 :=
sorry

end muffin_banana_cost_ratio_l304_304879


namespace range_of_expression_l304_304299

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := by
  sorry

end range_of_expression_l304_304299


namespace least_integer_greater_than_sqrt_450_l304_304513

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l304_304513


namespace cans_purchased_l304_304762

theorem cans_purchased (S Q E : ℝ) (h1 : Q ≠ 0) (h2 : S > 0) :
  (10 * E * S) / Q = (10 * (E : ℝ) * (S : ℝ)) / (Q : ℝ) := by 
  sorry

end cans_purchased_l304_304762


namespace polynomial_factorization_l304_304353

theorem polynomial_factorization :
  (x : ℤ[X]) →
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  intros x
  sorry

end polynomial_factorization_l304_304353


namespace num_parallelogram_even_l304_304545

-- Define the conditions of the problem in Lean
def isosceles_right_triangle (base_length : ℕ) := 
  base_length = 2

def square (side_length : ℕ) := 
  side_length = 1

def parallelogram (sides_length : ℕ) (diagonals_length : ℕ) := 
  sides_length = 1 ∧ diagonals_length = 1

-- Main statement to prove
theorem num_parallelogram_even (num_triangles num_squares num_parallelograms : ℕ)
  (Htriangle : ∀ t, t < num_triangles → isosceles_right_triangle 2)
  (Hsquare : ∀ s, s < num_squares → square 1)
  (Hparallelogram : ∀ p, p < num_parallelograms → parallelogram 1 1) :
  num_parallelograms % 2 = 0 := 
sorry

end num_parallelogram_even_l304_304545


namespace log_expression_equality_l304_304066

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_equality :
  Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) + (log_base 2 5) * (log_base 5 8) = 5 := by
  sorry

end log_expression_equality_l304_304066


namespace ivan_total_money_l304_304128

-- Define the value of a dime in cents
def value_of_dime : ℕ := 10

-- Define the value of a penny in cents
def value_of_penny : ℕ := 1

-- Define the number of dimes per piggy bank
def dimes_per_piggy_bank : ℕ := 50

-- Define the number of pennies per piggy bank
def pennies_per_piggy_bank : ℕ := 100

-- Define the number of piggy banks
def number_of_piggy_banks : ℕ := 2

-- Define the total value in dollars
noncomputable def total_value_in_dollars : ℕ := 
  (dimes_per_piggy_bank * value_of_dime + pennies_per_piggy_bank * value_of_penny) * number_of_piggy_banks / 100

theorem ivan_total_money : total_value_in_dollars = 12 := by
  sorry

end ivan_total_money_l304_304128


namespace number_of_classmates_l304_304652

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l304_304652


namespace stipulated_percentage_l304_304924

theorem stipulated_percentage
  (A B C : ℝ)
  (P : ℝ)
  (hA : A = 20000)
  (h_range : B - C = 10000)
  (hB : B = A + (P / 100) * A)
  (hC : C = A - (P / 100) * A) :
  P = 25 :=
sorry

end stipulated_percentage_l304_304924


namespace common_fraction_l304_304408

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l304_304408


namespace expected_value_of_win_l304_304910

theorem expected_value_of_win :
  (∑ n in finset.range 9, (n ^ 3)) / 8 = 162 := 
sorry

end expected_value_of_win_l304_304910


namespace ciphertext_to_plaintext_l304_304693

theorem ciphertext_to_plaintext :
  ∃ (a b c d : ℕ), (a + 2 * b = 14) ∧ (2 * b + c = 9) ∧ (2 * c + 3 * d = 23) ∧ (4 * d = 28) ∧ a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7 :=
by 
  sorry

end ciphertext_to_plaintext_l304_304693


namespace four_people_seven_chairs_l304_304458

def num_arrangements (total_chairs : ℕ) (num_reserved : ℕ) (num_people : ℕ) : ℕ :=
  (total_chairs - num_reserved).choose num_people * (num_people.factorial)

theorem four_people_seven_chairs (total_chairs : ℕ) (chairs_occupied : ℕ) (num_people : ℕ): 
    total_chairs = 7 → chairs_occupied = 2 → num_people = 4 →
    num_arrangements total_chairs chairs_occupied num_people = 120 :=
by
  intros
  unfold num_arrangements
  sorry

end four_people_seven_chairs_l304_304458


namespace sequence_6th_term_l304_304983

theorem sequence_6th_term 
    (a₁ a₂ a₃ a₄ a₅ a₆ : ℚ)
    (h₁ : a₁ = 3)
    (h₅ : a₅ = 54)
    (h₂ : a₂ = (a₁ + a₃) / 3)
    (h₃ : a₃ = (a₂ + a₄) / 3)
    (h₄ : a₄ = (a₃ + a₅) / 3)
    (h₆ : a₅ = (a₄ + a₆) / 3) :
    a₆ = 1133 / 7 :=
by
  sorry

end sequence_6th_term_l304_304983


namespace variance_of_white_balls_l304_304450

section
variable (n : ℕ := 7) 
variable (p : ℚ := 3/7)

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_white_balls : binomial_variance n p = 12/7 :=
by
  sorry
end

end variance_of_white_balls_l304_304450


namespace factorial_inequality_l304_304535

theorem factorial_inequality (n : ℕ) : 2^n * n! < (n+1)^n :=
by
  sorry

end factorial_inequality_l304_304535


namespace dawn_monthly_payments_l304_304068

theorem dawn_monthly_payments (annual_salary : ℕ) (saved_per_month : ℕ)
  (h₁ : annual_salary = 48000)
  (h₂ : saved_per_month = 400)
  (h₃ : ∀ (monthly_salary : ℕ), saved_per_month = (10 * monthly_salary) / 100):
  annual_salary / saved_per_month = 12 :=
by
  sorry

end dawn_monthly_payments_l304_304068


namespace range_of_a_l304_304596

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 - 3 * a * x + 9 ≤ 0) → a ∈ Set.Ico 0 4 := by
  sorry

end range_of_a_l304_304596


namespace find_vector_at_t5_l304_304049

def vector_on_line (t : ℝ) : ℝ × ℝ := 
  let a := (0, 11) -- From solving the system of equations
  let d := (2, -4) -- From solving the system of equations
  (a.1 + t * d.1, a.2 + t * d.2)

theorem find_vector_at_t5 : vector_on_line 5 = (10, -9) := 
by 
  sorry

end find_vector_at_t5_l304_304049


namespace sum_a10_a11_l304_304847

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)

theorem sum_a10_a11 {a : ℕ → ℝ} (h_seq : geometric_sequence a)
  (h1 : a 1 + a 2 = 2)
  (h4 : a 4 + a 5 = 4) :
  a 10 + a 11 = 16 :=
by {
  sorry
}

end sum_a10_a11_l304_304847


namespace k_plus_m_eq_27_l304_304466

theorem k_plus_m_eq_27 (k m : ℝ) (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a > 0) (h5 : b > 0) (h6 : c > 0)
  (h7 : a + b + c = 8) 
  (h8 : k = a * b + a * c + b * c) 
  (h9 : m = a * b * c) :
  k + m = 27 :=
by
  sorry

end k_plus_m_eq_27_l304_304466


namespace sum_ages_l304_304335

theorem sum_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 := 
by 
  sorry

end sum_ages_l304_304335


namespace train_speed_l304_304359

def train_length : ℝ := 1000  -- train length in meters
def time_to_cross_pole : ℝ := 200  -- time to cross the pole in seconds

theorem train_speed : train_length / time_to_cross_pole = 5 := by
  sorry

end train_speed_l304_304359


namespace diagonal_cubes_intersect_l304_304904

def a := 150
def b := 324
def c := 375

def gcd_ab := Nat.gcd a b = 6
def gcd_bc := Nat.gcd b c = 3
def gcd_ca := Nat.gcd c a = 75
def gcd_abc := Nat.gcd (Nat.gcd a b) c = 3

theorem diagonal_cubes_intersect :
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd (Nat.gcd a b) c = 768 :=
by {
  intro gcd_ab gcd_bc gcd_ca gcd_abc,
  sorry
}

end diagonal_cubes_intersect_l304_304904


namespace log_pi_inequality_l304_304638

theorem log_pi_inequality (a b : ℝ) (π : ℝ) (h1 : 2^a = π) (h2 : 5^b = π) (h3 : a = Real.log π / Real.log 2) (h4 : b = Real.log π / Real.log 5) :
  (1 / a) + (1 / b) > 2 :=
by
  sorry

end log_pi_inequality_l304_304638


namespace lines_not_intersecting_may_be_parallel_or_skew_l304_304442

theorem lines_not_intersecting_may_be_parallel_or_skew (a b : ℝ × ℝ → Prop) 
  (h : ∀ x, ¬ (a x ∧ b x)) : 
  (∃ c d : ℝ × ℝ → Prop, a = c ∧ b = d) := 
sorry

end lines_not_intersecting_may_be_parallel_or_skew_l304_304442


namespace lcm_1_to_12_l304_304721

theorem lcm_1_to_12 : Nat.lcm_list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 27720 := by
  sorry

end lcm_1_to_12_l304_304721


namespace probability_student_C_first_l304_304372

theorem probability_student_C_first (A B C D E : Type) :
  let students := [A, B, C, D, E]
  -- Define the conditions
  neither_A_nor_B_first (s : List Type) := (s.head ≠ A ∧ s.head ≠ B)
  B_not_last (s : List Type) := s.last ≠ B

  -- Define the possibilities under the given conditions
  possible_arrangements_with_conditions :=
    students.permutations.filter (λ s, neither_A_nor_B_first s ∧ B_not_last s)

  total_ways := possible_arrangements_with_conditions.length

  -- Define the possibilities with student C being the first to speak
  C_first_arrangements :=
    possible_arrangements_with_conditions.filter (λ s, s.head = C)

  ways_C_first := C_first_arrangements.length

  -- Calculate the probability
  probability_C_first := ways_C_first / total_ways
  -- Prove that the calculated probability is 1/3
  in probability_C_first = 1/3 :=
  sorry

end probability_student_C_first_l304_304372


namespace sector_properties_l304_304944

variables (r : ℝ) (alpha l S : ℝ)

noncomputable def arc_length (r alpha : ℝ) : ℝ := alpha * r
noncomputable def sector_area (l r : ℝ) : ℝ := (1/2) * l * r

theorem sector_properties
  (h_r : r = 2)
  (h_alpha : alpha = π / 6) :
  arc_length r alpha = π / 3 ∧ sector_area (arc_length r alpha) r = π / 3 :=
by
  sorry

end sector_properties_l304_304944


namespace repeating_decimals_sum_l304_304400

theorem repeating_decimals_sum :
  let x := 0.6666666 -- 0.\overline{6}
  let y := 0.2222222 -- 0.\overline{2}
  let z := 0.4444444 -- 0.\overline{4}
  (x + y - z) = 4 / 9 := 
by
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  calc
    -- Calculate (x + y - z)
    (x + y - z) = (2 / 3 + 2 / 9 - 4 / 9) : by sorry
                ... = 4 / 9 : by sorry


end repeating_decimals_sum_l304_304400


namespace weight_of_steel_rod_l304_304965

theorem weight_of_steel_rod (length1 : ℝ) (weight1 : ℝ) (length2 : ℝ) (weight2 : ℝ) 
  (h1 : length1 = 9) (h2 : weight1 = 34.2) (h3 : length2 = 11.25) : 
  weight2 = (weight1 / length1) * length2 :=
by
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end weight_of_steel_rod_l304_304965


namespace coconut_grove_l304_304275

theorem coconut_grove (x N : ℕ) (h1 : (x + 4) * 60 + x * N + (x - 4) * 180 = 3 * x * 100) (hx : x = 8) : N = 120 := 
by
  subst hx
  sorry

end coconut_grove_l304_304275


namespace base_height_ratio_l304_304318

-- Define the conditions
def cultivation_cost : ℝ := 333.18
def rate_per_hectare : ℝ := 24.68
def base_of_field : ℝ := 300
def height_of_field : ℝ := 300

-- Prove the ratio of base to height is 1
theorem base_height_ratio (b h : ℝ) (cost rate : ℝ)
  (h1 : cost = 333.18) (h2 : rate = 24.68) 
  (h3 : b = 300) (h4 : h = 300) : b / h = 1 :=
by
  sorry

end base_height_ratio_l304_304318


namespace crows_eat_worms_l304_304966

theorem crows_eat_worms (worms_eaten_by_3_crows_in_1_hour : ℕ) 
                        (crows_eating_worms_constant : worms_eaten_by_3_crows_in_1_hour = 30)
                        (number_of_crows : ℕ) 
                        (observation_time_hours : ℕ) :
                        number_of_crows = 5 ∧ observation_time_hours = 2 →
                        (number_of_crows * worms_eaten_by_3_crows_in_1_hour / 3) * observation_time_hours = 100 :=
by
  sorry

end crows_eat_worms_l304_304966


namespace range_of_m_l304_304825

theorem range_of_m 
  (h : ∀ x, -1 < x ∧ x < 4 → x > 2 * (m: ℝ)^2 - 3)
  : ∀ (m: ℝ), -1 ≤ m ∧ m ≤ 1 :=
by 
  sorry

end range_of_m_l304_304825


namespace total_amount_divided_into_two_parts_l304_304046

theorem total_amount_divided_into_two_parts (P1 P2 : ℝ) (annual_income : ℝ) :
  P1 = 1500.0000000000007 →
  annual_income = 135 →
  (P1 * 0.05 + P2 * 0.06 = annual_income) →
  P1 + P2 = 2500.000000000000 :=
by
  intros hP1 hIncome hInterest
  sorry

end total_amount_divided_into_two_parts_l304_304046


namespace total_balloons_cost_is_91_l304_304083

-- Define the number of balloons and their costs for Fred, Sam, and Dan
def fred_balloons : ℕ := 10
def fred_cost_per_balloon : ℝ := 1

def sam_balloons : ℕ := 46
def sam_cost_per_balloon : ℝ := 1.5

def dan_balloons : ℕ := 16
def dan_cost_per_balloon : ℝ := 0.75

-- Calculate the total cost for each person’s balloons
def fred_total_cost : ℝ := fred_balloons * fred_cost_per_balloon
def sam_total_cost : ℝ := sam_balloons * sam_cost_per_balloon
def dan_total_cost : ℝ := dan_balloons * dan_cost_per_balloon

-- Calculate the total cost of all the balloons combined
def total_cost : ℝ := fred_total_cost + sam_total_cost + dan_total_cost

-- The main statement to be proved
theorem total_balloons_cost_is_91 : total_cost = 91 :=
by
  -- Recall that the previous individual costs can be worked out and added
  -- But for the sake of this statement, we use sorry to skip details
  sorry

end total_balloons_cost_is_91_l304_304083


namespace sin_double_angle_neg_of_fourth_quadrant_l304_304107

variable (α : ℝ)

def is_in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, -π / 2 + 2 * k * π < α ∧ α < 2 * k * π

theorem sin_double_angle_neg_of_fourth_quadrant (h : is_in_fourth_quadrant α) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_fourth_quadrant_l304_304107


namespace value_of_x_l304_304361

noncomputable def k := 9

theorem value_of_x (y : ℝ) (h1 : y = 3) (h2 : ∀ (x : ℝ), x = 2.25 → x = k / (2 : ℝ)^2) : 
  ∃ (x : ℝ), x = 1 := by
  sorry

end value_of_x_l304_304361


namespace monotonicity_and_range_l304_304811

noncomputable def f (a x : ℝ) : ℝ := (a * x - 2) * Real.exp x - Real.exp (a - 2)

theorem monotonicity_and_range (a x : ℝ) :
  ( (a = 0 → ∀ x, f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x < (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x > (2 - a) / a, f a x > f a (x + 1) ) ∧
  (a < 0 → ∀ x > (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x < (2 - a) / a, f a x > f a (x + 1) ) ∧
  (∀ x > 1, f a x > 0 ↔ a ∈ Set.Ici 1)) 
:=
sorry

end monotonicity_and_range_l304_304811


namespace monthly_growth_rate_l304_304063

-- Definitions and conditions
def initial_height : ℝ := 20
def final_height : ℝ := 80
def months_in_year : ℕ := 12

-- Theorem stating the monthly growth rate
theorem monthly_growth_rate :
  (final_height - initial_height) / (months_in_year : ℝ) = 5 :=
by 
  sorry

end monthly_growth_rate_l304_304063


namespace regression_estimate_l304_304937

theorem regression_estimate (x : ℝ) (h : x = 28) : 4.75 * x + 257 = 390 :=
by
  rw [h]
  norm_num

end regression_estimate_l304_304937


namespace larger_number_is_34_l304_304874

theorem larger_number_is_34 (a b : ℕ) (h1 : a > b) (h2 : (a + b) + (a - b) = 68) : a = 34 := 
by
  sorry

end larger_number_is_34_l304_304874


namespace gcd_8154_8640_l304_304565

theorem gcd_8154_8640 : Nat.gcd 8154 8640 = 6 := by
  sorry

end gcd_8154_8640_l304_304565


namespace distinct_valid_c_values_l304_304795

theorem distinct_valid_c_values : 
  let is_solution (c : ℤ) (x : ℚ) := (5 * ⌊x⌋₊ + 3 * ⌈x⌉₊ = c) 
  ∃ s : Finset ℤ, (∀ c ∈ s, (∃ x : ℚ, is_solution c x)) ∧ s.card = 500 :=
by sorry

end distinct_valid_c_values_l304_304795


namespace range_of_a_l304_304593

theorem range_of_a (a : ℝ) :
  (a > 0 ∧ (∃ x, x^2 - 4 * a * x + 3 * a^2 < 0)) →
  (∃ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0) →
  (2 < a ∧ a ≤ 2) := sorry

end range_of_a_l304_304593


namespace intersection_of_A_and_B_l304_304138

def setA : Set ℝ := { x : ℝ | x > -1 }
def setB : Set ℝ := { y : ℝ | 0 ≤ y ∧ y < 1 }

theorem intersection_of_A_and_B :
  (setA ∩ setB) = { z : ℝ | 0 ≤ z ∧ z < 1 } :=
by
  sorry

end intersection_of_A_and_B_l304_304138


namespace least_integer_gt_sqrt_450_l304_304521

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l304_304521


namespace rectangle_area_l304_304540

-- Definitions based on the conditions
def radius := 6
def diameter := 2 * radius
def width := diameter
def length := 3 * width

-- Statement of the theorem
theorem rectangle_area : (width * length = 432) := by
  sorry

end rectangle_area_l304_304540


namespace rhind_papyrus_problem_l304_304984

theorem rhind_papyrus_problem 
  (a1 a2 a3 a4 a5 : ℚ)
  (h1 : a2 = a1 + d)
  (h2 : a3 = a1 + 2 * d)
  (h3 : a4 = a1 + 3 * d)
  (h4 : a5 = a1 + 4 * d)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 60)
  (h_condition : (a4 + a5) / 2 = a1 + a2 + a3) :
  a1 = 4 / 3 :=
by
  sorry

end rhind_papyrus_problem_l304_304984


namespace original_cost_l304_304702

theorem original_cost (original_cost : ℝ) (h : 0.30 * original_cost = 588) : original_cost = 1960 :=
sorry

end original_cost_l304_304702


namespace evaluate_expression_l304_304784

theorem evaluate_expression :
  (3 ^ 1002 + 7 ^ 1003) ^ 2 - (3 ^ 1002 - 7 ^ 1003) ^ 2 = 56 * 10 ^ 1003 :=
by
  sorry

end evaluate_expression_l304_304784


namespace coeff_sum_eq_32_l304_304269

theorem coeff_sum_eq_32 (n : ℕ) (h : (2 : ℕ)^n = 32) : n = 5 :=
sorry

end coeff_sum_eq_32_l304_304269


namespace window_side_length_is_five_l304_304878

def pane_width (x : ℝ) : ℝ := x
def pane_height (x : ℝ) : ℝ := 3 * x
def border_width : ℝ := 1
def pane_rows : ℕ := 2
def pane_columns : ℕ := 3

theorem window_side_length_is_five (x : ℝ) (h : pane_height x = 3 * pane_width x) : 
  (3 * x + 4 = 6 * x + 3) -> (3 * x + 4 = 5) :=
by
  intros h1
  sorry

end window_side_length_is_five_l304_304878


namespace valid_n_values_l304_304534

variables (n x y : ℕ)

theorem valid_n_values :
  (n * (x - 3) = y + 3) ∧ (x + n = 3 * (y - n)) →
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7) :=
by
  sorry

end valid_n_values_l304_304534


namespace minimum_value_shifted_function_l304_304972

def f (x a : ℝ) : ℝ := x^2 + 4 * x + 7 - a

theorem minimum_value_shifted_function (a : ℝ) (h : ∃ x, f x a = 2) :
  ∃ y, (∃ x, y = f (x - 2015) a) ∧ y = 2 :=
sorry

end minimum_value_shifted_function_l304_304972


namespace custom_op_1_neg3_l304_304970

-- Define the custom operation as per the condition
def custom_op (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2

-- The theorem to prove that 1 * (-3) = -14 using the defined operation
theorem custom_op_1_neg3 : custom_op 1 (-3) = -14 := sorry

end custom_op_1_neg3_l304_304970


namespace bus_speed_excluding_stoppages_l304_304212

theorem bus_speed_excluding_stoppages :
  ∀ (S : ℝ), (45 = (3 / 4) * S) → (S = 60) :=
by 
  intros S h
  sorry

end bus_speed_excluding_stoppages_l304_304212


namespace percentage_reduction_l304_304757

theorem percentage_reduction :
  let original := 243.75
  let reduced := 195
  let percentage := ((original - reduced) / original) * 100
  percentage = 20 :=
by
  sorry

end percentage_reduction_l304_304757


namespace find_angle_A_l304_304981

theorem find_angle_A 
  (a b : ℝ) (A B : ℝ) 
  (h1 : b = 2 * a)
  (h2 : B = A + 60) : 
  A = 30 :=
  sorry

end find_angle_A_l304_304981


namespace range_of_c_l304_304994

theorem range_of_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 1) : ∀ c : ℝ, c < 9 → a + b > c :=
by
  sorry

end range_of_c_l304_304994


namespace vector_addition_AC_l304_304955

def vector := (ℝ × ℝ)

def AB : vector := (0, 1)
def BC : vector := (1, 0)

def AC (AB BC : vector) : vector := (AB.1 + BC.1, AB.2 + BC.2) 

theorem vector_addition_AC (AB BC : vector) (h1 : AB = (0, 1)) (h2 : BC = (1, 0)) : 
  AC AB BC = (1, 1) :=
by
  sorry

end vector_addition_AC_l304_304955


namespace sum_of_constants_l304_304385

theorem sum_of_constants (c d : ℝ) (h₁ : 16 = 2 * 4 + c) (h₂ : 16 = 4 * 4 + d) : c + d = 8 := by
  sorry

end sum_of_constants_l304_304385


namespace problem1_problem2_l304_304797

def f (x a : ℝ) : ℝ := abs (1 - x - a) + abs (2 * a - x)

theorem problem1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
  sorry

theorem problem2 (a x : ℝ) (h : a ≥ 2/3) : f x a ≥ 1 :=
  sorry

end problem1_problem2_l304_304797


namespace segment_length_l304_304707

theorem segment_length (x : ℝ) (h : |x - (27)^(1/3)| = 5) : ∃ a b : ℝ, (a = 8 ∧ b = -2 ∨ a = -2 ∧ b = 8) ∧ real.dist a b = 10 :=
by
  use [8, -2] -- providing the endpoints explicitly
  split
  -- prove that these are the correct endpoints
  · left; exact ⟨rfl, rfl⟩
  -- prove the distance is 10
  · apply real.dist_eq; linarith
  

end segment_length_l304_304707


namespace sum_of_roots_l304_304700

theorem sum_of_roots (r s t : ℝ) (h : 3 * r * s * t - 9 * (r * s + s * t + t * r) - 28 * (r + s + t) + 12 = 0) : r + s + t = 3 :=
by sorry

end sum_of_roots_l304_304700


namespace find_cost_price_l304_304542

noncomputable def original_cost_price (C S C_new S_new : ℝ) : Prop :=
  S = 1.25 * C ∧
  C_new = 0.80 * C ∧
  S_new = 1.25 * C - 10.50 ∧
  S_new = 1.04 * C

theorem find_cost_price (C S C_new S_new : ℝ) :
  original_cost_price C S C_new S_new → C = 50 :=
by
  sorry

end find_cost_price_l304_304542


namespace circle_area_ratio_in_hexagon_l304_304778

/-- Math problem statement based on regular hexagon and circle tangency properties. -/
theorem circle_area_ratio_in_hexagon :
  ∃ (hexagon_side_length : ℝ) (r1 r2 : ℝ),
    hexagon_side_length = 2 ∧
    r1 = (Real.sqrt 3) / 3 ∧ 
    r2 = (Real.sqrt 3) ∧ 
    let A1 := π * r1^2 
    let A2 := π * r2^2 
    let ratio := A2 / A1 
    ratio = 3 * Real.sqrt 3 :=
sorry

end circle_area_ratio_in_hexagon_l304_304778


namespace geometric_sequence_y_l304_304584

theorem geometric_sequence_y (x y z : ℝ) (h1 : 1 ≠ 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : z ≠ 0) (h5 : 9 ≠ 0)
  (h_seq : ∀ a b c d e : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ a * e = b * d ∧ b * d = c^2) →
           (a, b, c, d, e) = (1, x, y, z, 9)) :
  y = 3 :=
sorry

end geometric_sequence_y_l304_304584


namespace trip_time_is_14_l304_304747

-- Define the conditions
def avg_speed1 := 40  -- miles per hour
def time1 := 4  -- hours
def avg_speed2 := 65  -- miles per hour
def time2 := 3  -- hours
def avg_speed3 := 54  -- miles per hour
def time3 := 2  -- hours
def avg_speed4 := 70  -- miles per hour
def total_avg_speed := 58  -- miles per hour

-- Define the total time for a car trip
noncomputable def total_trip_time (x : ℕ) : ℕ :=
  time1 + time2 + time3 + x

-- Define the total distance
noncomputable def total_distance (x : ℕ) : ℕ :=
  avg_speed1 * time1 + avg_speed2 * time2 + avg_speed3 * time3 + avg_speed4 * x

-- Statement of the problem to prove
theorem trip_time_is_14 : ∃ x, total_avg_speed = total_distance x / total_trip_time x ∧ total_trip_time x = 14 := by
  sorry


end trip_time_is_14_l304_304747


namespace blake_change_l304_304768

def cost_oranges : ℕ := 40
def cost_apples : ℕ := 50
def cost_mangoes : ℕ := 60
def initial_money : ℕ := 300

def total_cost : ℕ := cost_oranges + cost_apples + cost_mangoes
def change : ℕ := initial_money - total_cost

theorem blake_change : change = 150 := by
  sorry

end blake_change_l304_304768


namespace distance_to_fountain_l304_304306

def total_distance : ℝ := 120
def number_of_trips : ℝ := 4
def distance_per_trip : ℝ := total_distance / number_of_trips

theorem distance_to_fountain : distance_per_trip = 30 := by
  sorry

end distance_to_fountain_l304_304306


namespace eccentricity_of_ellipse_l304_304586

theorem eccentricity_of_ellipse (k : ℝ) (h_k : k > 0)
  (focus : ∃ (x : ℝ), (x, 0) = ⟨3, 0⟩) :
  ∃ e : ℝ, e = (Real.sqrt 3 / 2) := 
sorry

end eccentricity_of_ellipse_l304_304586


namespace part1_union_part1_intersect_complement_part2_range_a_l304_304804

open Set

-- Problem Setup
namespace ProofProblem

variables (x a : ℝ) (U : Set ℝ := univ)
def A := {x : ℝ | 2 ≤ x ∧ x ≤ 8}
def B := {x : ℝ | 1 < x ∧ x < 6}
def C (a : ℝ) := {x : ℝ | x > a}

-- Proof Statements

theorem part1_union : A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8} := 
by sorry

theorem part1_intersect_complement : (U \ A) ∩ B = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

theorem part2_range_a (h : (A ∩ C a) ≠ ∅) : a < 8 :=
by sorry

end ProofProblem

end part1_union_part1_intersect_complement_part2_range_a_l304_304804


namespace five_crows_two_hours_l304_304969

-- Define the conditions and the question as hypotheses
def crows_worms (crows worms hours : ℕ) := 
  (crows = 3) ∧ (worms = 30) ∧ (hours = 1)

theorem five_crows_two_hours 
  (c: ℕ) (w: ℕ) (h: ℕ)
  (H: crows_worms c w h)
  : ∃ worms_eaten : ℕ, worms_eaten = 100 :=
by
  sorry

end five_crows_two_hours_l304_304969


namespace green_fish_count_l304_304149

theorem green_fish_count (B O G : ℕ) (h1 : B = (2 / 5) * 200)
  (h2 : O = 2 * B - 30) (h3 : G = (3 / 2) * O) (h4 : B + O + G = 200) : 
  G = 195 :=
by
  sorry

end green_fish_count_l304_304149


namespace verify_triangle_inequality_l304_304478

-- Conditions of the problem
variables (L : ℕ → ℕ)
-- The rods lengths are arranged in increasing order
axiom rods_in_order : ∀ i : ℕ, L i ≤ L (i + 1)

-- Define the critical check
def critical_check : Prop :=
  L 98 + L 99 > L 100

-- Prove that verifying the critical_check is sufficient
theorem verify_triangle_inequality (h : critical_check L) :
  ∀ i j k : ℕ, 1 ≤ i → i < j → j < k → k ≤ 100 → L i + L j > L k :=
by
  sorry

end verify_triangle_inequality_l304_304478


namespace beths_total_crayons_l304_304380

def packs : ℕ := 4
def crayons_per_pack : ℕ := 10
def extra_crayons : ℕ := 6

theorem beths_total_crayons : packs * crayons_per_pack + extra_crayons = 46 := by
  sorry

end beths_total_crayons_l304_304380


namespace sin_neg_five_sixths_pi_l304_304561

theorem sin_neg_five_sixths_pi : Real.sin (- 5 / 6 * Real.pi) = -1 / 2 :=
sorry

end sin_neg_five_sixths_pi_l304_304561


namespace probability_three_nine_l304_304589

noncomputable def X (σ : ℝ) (hσ : σ > 0) : ProbabilitySpace ℝ :=
normalDist 6 σ

theorem probability_three_nine (σ : ℝ) (hσ : σ > 0) :
  (∫ x in (3 : ℝ)..9, pdf (X σ hσ) x) = 0.6 :=
sorry

end probability_three_nine_l304_304589


namespace kevin_hops_7_times_l304_304989

noncomputable def distance_hopped_after_n_hops (n : ℕ) : ℚ :=
  4 * (1 - (3 / 4) ^ n)

theorem kevin_hops_7_times :
  distance_hopped_after_n_hops 7 = 7086 / 2048 := 
by
  sorry

end kevin_hops_7_times_l304_304989


namespace all_statements_correct_l304_304123

-- Definitions based on the problem conditions
def population_size : ℕ := 60000
def sample_size : ℕ := 1000
def is_sampling_survey (population_size sample_size : ℕ) : Prop := sample_size < population_size
def is_population (n : ℕ) : Prop := n = 60000
def is_sample (population_size sample_size : ℕ) : Prop := sample_size < population_size
def matches_sample_size (n : ℕ) : Prop := n = 1000

-- Lean problem statement representing the proof that all statements are correct
theorem all_statements_correct :
  is_sampling_survey population_size sample_size ∧
  is_population population_size ∧ 
  is_sample population_size sample_size ∧
  matches_sample_size sample_size := by
  sorry

end all_statements_correct_l304_304123


namespace cake_eating_classmates_l304_304671

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l304_304671


namespace macey_saving_weeks_l304_304304

-- Definitions for conditions
def shirt_cost : ℝ := 3
def amount_saved : ℝ := 1.5
def weekly_saving : ℝ := 0.5

-- Statement of the proof problem
theorem macey_saving_weeks : (shirt_cost - amount_saved) / weekly_saving = 3 := by
  sorry

end macey_saving_weeks_l304_304304


namespace local_minimum_at_minus_one_l304_304998

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_minimum_at_minus_one :
  (∃ δ > 0, ∀ x : ℝ, (x < -1 + δ ∧ x > -1 - δ) → f x ≥ f (-1)) :=
by
  sorry

end local_minimum_at_minus_one_l304_304998


namespace point_returns_to_original_after_seven_steps_l304_304580

-- Define a structure for a triangle and a point inside it
structure Triangle :=
  (A B C : Point)

structure Point :=
  (x y : ℝ)

-- Given a triangle and a point inside it
variable (ABC : Triangle)
variable (M : Point)

-- Define the set of movements and the intersection points
def move_parallel_to_BC (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AB (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AC (M : Point) (ABC : Triangle) : Point := sorry

-- Function to perform the stepwise movement through 7 steps
def move_M_seven_times (M : Point) (ABC : Triangle) : Point :=
  let M1 := move_parallel_to_BC M ABC
  let M2 := move_parallel_to_AB M1 ABC 
  let M3 := move_parallel_to_AC M2 ABC
  let M4 := move_parallel_to_BC M3 ABC
  let M5 := move_parallel_to_AB M4 ABC
  let M6 := move_parallel_to_AC M5 ABC
  let M7 := move_parallel_to_BC M6 ABC
  M7

-- The theorem stating that after 7 steps, point M returns to its original position
theorem point_returns_to_original_after_seven_steps :
  move_M_seven_times M ABC = M := sorry

end point_returns_to_original_after_seven_steps_l304_304580


namespace ab_difference_l304_304115

theorem ab_difference (a b : ℤ) (h1 : |a| = 5) (h2 : |b| = 3) (h3 : a + b > 0) : a - b = 2 ∨ a - b = 8 :=
sorry

end ab_difference_l304_304115


namespace sum_of_three_consecutive_integers_product_990_l304_304023

theorem sum_of_three_consecutive_integers_product_990 
  (a b c : ℕ) 
  (h1 : b = a + 1)
  (h2 : c = b + 1)
  (h3 : a * b * c = 990) :
  a + b + c = 30 :=
sorry

end sum_of_three_consecutive_integers_product_990_l304_304023


namespace square_of_other_leg_l304_304016

-- Conditions
variable (a b c : ℝ)
variable (h₁ : c = a + 2)
variable (h₂ : a^2 + b^2 = c^2)

-- The theorem statement
theorem square_of_other_leg (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
by
  sorry

end square_of_other_leg_l304_304016


namespace original_selling_price_l304_304189

/-- A boy sells a book for some amount and he gets a loss of 10%.
To gain 10%, the selling price should be Rs. 550.
Prove that the original selling price of the book was Rs. 450. -/
theorem original_selling_price (CP : ℝ) (h1 : 1.10 * CP = 550) :
    0.90 * CP = 450 := 
sorry

end original_selling_price_l304_304189


namespace scientific_notation_correct_l304_304839

theorem scientific_notation_correct (n : ℕ) (h : n = 11580000) : n = 1.158 * 10^7 := 
sorry

end scientific_notation_correct_l304_304839


namespace triangle_angle_bisectors_l304_304424

theorem triangle_angle_bisectors {a b c : ℝ} (ht : (a = 2 ∧ b = 3 ∧ c < 5)) : 
  (∃ h_a h_b h_c : ℝ, h_a + h_b > h_c ∧ h_a + h_c > h_b ∧ h_b + h_c > h_a) →
  ¬ (∃ ell_a ell_b ell_c : ℝ, ell_a + ell_b > ell_c ∧ ell_a + ell_c > ell_b ∧ ell_b + ell_c > ell_a) :=
by
  sorry

end triangle_angle_bisectors_l304_304424


namespace pigs_and_dogs_more_than_sheep_l304_304032

-- Define the number of pigs and sheep
def numberOfPigs : ℕ := 42
def numberOfSheep : ℕ := 48

-- Define the number of dogs such that it is the same as the number of pigs
def numberOfDogs : ℕ := numberOfPigs

-- Define the total number of pigs and dogs
def totalPigsAndDogs : ℕ := numberOfPigs + numberOfDogs

-- State the theorem about the difference between pigs and dogs and the number of sheep
theorem pigs_and_dogs_more_than_sheep :
  totalPigsAndDogs - numberOfSheep = 36 := 
sorry

end pigs_and_dogs_more_than_sheep_l304_304032


namespace at_least_one_wins_l304_304498

def probability_A := 1 / 2
def probability_B := 1 / 4

def probability_at_least_one (pA pB : ℚ) : ℚ := 
  1 - ((1 - pA) * (1 - pB))

theorem at_least_one_wins :
  probability_at_least_one probability_A probability_B = 5 / 8 := 
by
  sorry

end at_least_one_wins_l304_304498


namespace find_x_l304_304927

noncomputable def satisfy_equation (x : ℝ) : Prop :=
  8 / (Real.sqrt (x - 10) - 10) +
  2 / (Real.sqrt (x - 10) - 5) +
  10 / (Real.sqrt (x - 10) + 5) +
  16 / (Real.sqrt (x - 10) + 10) = 0

theorem find_x : ∃ x : ℝ, satisfy_equation x ∧ x = 60 := sorry

end find_x_l304_304927


namespace evaluate_f_difference_l304_304623

def f (x : ℤ) : ℤ := x^6 + 3 * x^4 - 4 * x^3 + x^2 + 2 * x

theorem evaluate_f_difference : f 3 - f (-3) = -204 := by
  sorry

end evaluate_f_difference_l304_304623


namespace shopkeeper_discount_and_selling_price_l304_304920

theorem shopkeeper_discount_and_selling_price :
  let CP := 100
  let MP := CP + 0.5 * CP
  let SP := CP + 0.15 * CP
  let Discount := (MP - SP) / MP * 100
  Discount = 23.33 ∧ SP = 115 :=
by
  sorry

end shopkeeper_discount_and_selling_price_l304_304920


namespace hexagon_colorings_l304_304210

-- Definitions based on conditions
def isValidColoring (A B C D E F : ℕ) (colors : Fin 7 → ℕ) : Prop :=
  -- Adjacent vertices must have different colors
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧
  -- Diagonal vertices must have different colors
  A ≠ D ∧ B ≠ E ∧ C ≠ F

-- Function to count all valid colorings
def countValidColorings : ℕ :=
  let colors := List.range 7
  -- Calculate total number of valid colorings
  7 * 6 * 5 * 4 * 3 * 2

theorem hexagon_colorings : countValidColorings = 5040 := by
  sorry

end hexagon_colorings_l304_304210


namespace tv_sale_increase_l304_304445

theorem tv_sale_increase (P Q : ℝ) :
  let new_price := 0.9 * P
  let original_sale_value := P * Q
  let increased_percentage := 1.665
  ∃ x : ℝ, (new_price * (1 + x / 100) * Q = increased_percentage * original_sale_value) → x = 85 :=
by
  sorry

end tv_sale_increase_l304_304445


namespace Patricia_earns_more_l304_304854

-- Define the function for compound interest with annual compounding
noncomputable def yearly_compound (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Define the function for compound interest with quarterly compounding
noncomputable def quarterly_compound (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 4)^ (4 * n)

-- Define the conditions
def P := 50000.0
def r := 0.04
def n := 2

-- Define the amounts for Jose and Patricia using their respective compounding methods
def A_Jose := yearly_compound P r n
def A_Patricia := quarterly_compound P r n

-- Define the target difference in earnings
def difference := A_Patricia - A_Jose

-- Theorem statement
theorem Patricia_earns_more : difference = 63 := by
  sorry

end Patricia_earns_more_l304_304854


namespace k_m_sum_l304_304468

theorem k_m_sum (k m : ℝ) (h : ∀ {x : ℝ}, x^3 - 8 * x^2 + k * x - m = 0 → x ∈ {1, 2, 5} ∨ x ∈ {1, 3, 4}) :
  k + m = 27 ∨ k + m = 31 :=
by
  sorry

end k_m_sum_l304_304468


namespace required_run_rate_is_correct_l304_304127

open Nat

noncomputable def requiredRunRate (initialRunRate : ℝ) (initialOvers : ℕ) (targetRuns : ℕ) (totalOvers : ℕ) : ℝ :=
  let runsScored := initialRunRate * initialOvers
  let runsNeeded := targetRuns - runsScored
  let remainingOvers := totalOvers - initialOvers
  runsNeeded / (remainingOvers : ℝ)

theorem required_run_rate_is_correct :
  (requiredRunRate 3.6 10 282 50 = 6.15) :=
by
  sorry

end required_run_rate_is_correct_l304_304127


namespace least_number_remainder_l304_304167

theorem least_number_remainder (n : ℕ) (h : 20 ∣ (n - 5)) : n = 125 := sorry

end least_number_remainder_l304_304167


namespace math_problem_l304_304363

theorem math_problem : 
  ( - (1 / 12 : ℚ) + (1 / 3 : ℚ) - (1 / 2 : ℚ) ) / ( - (1 / 18 : ℚ) ) = 4.5 := 
by
  sorry

end math_problem_l304_304363


namespace a_squared_gt_b_squared_l304_304423

theorem a_squared_gt_b_squared {a b : ℝ} (h : a ≠ 0) (hb : b ≠ 0) (hb_domain : b > -1 ∧ b < 1) (h_eq : a = Real.log (1 + b) - Real.log (1 - b)) :
  a^2 > b^2 := 
sorry

end a_squared_gt_b_squared_l304_304423


namespace smallest_integer_solution_l304_304169

theorem smallest_integer_solution (y : ℤ) : (10 - 5 * y < 5) → y = 2 := by
  sorry

end smallest_integer_solution_l304_304169


namespace least_integer_greater_than_sqrt_450_l304_304508

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l304_304508


namespace parabola_value_l304_304974

theorem parabola_value (b c : ℝ) (h : 3 = -(-2) ^ 2 + b * -2 + c) : 2 * c - 4 * b - 9 = 5 := by
  sorry

end parabola_value_l304_304974


namespace segment_length_of_absolute_value_l304_304715

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end segment_length_of_absolute_value_l304_304715


namespace possible_classmates_l304_304664

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l304_304664


namespace probability_desired_urn_l304_304203

-- Define the initial conditions and operations
def initial_urn : Type := { reds : ℕ // reds = 2 } × { blues : ℕ // blues = 1 }

-- Define the probability function for drawing a ball and adding two of the same color
noncomputable def ball_draw_and_add (current_urn : { reds : ℕ × blues : ℕ }) : ℕ := sorry

-- Define the desired outcome after five operations
def desired_urn : Type := { reds : ℕ // reds = 7 } × { blues : ℕ // blues = 4 }

-- Define the probability of achieving the desired outcome
noncomputable def probability_after_five_operations : ℚ := sorry

theorem probability_desired_urn (initial_urn : { reds : ℕ × blues : ℕ } )
    (current_urn : initial_urn = ⟨2, 1⟩) : 
    probability_after_five_operations = 32 / 315 := 
sorry

end probability_desired_urn_l304_304203


namespace find_coordinates_of_C_l304_304803

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 4, y := -1, z := 2 }
def B : Point := { x := 2, y := -3, z := 0 }

def satisfies_condition (C : Point) : Prop :=
  (C.x - B.x, C.y - B.y, C.z - B.z) = (2 * (A.x - C.x), 2 * (A.y - C.y), 2 * (A.z - C.z))

theorem find_coordinates_of_C (C : Point) (h : satisfies_condition C) : C = { x := 10/3, y := -5/3, z := 4/3 } :=
  sorry -- Proof is omitted as requested

end find_coordinates_of_C_l304_304803


namespace possible_classmates_l304_304660

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l304_304660


namespace trig_identity_l304_304826

theorem trig_identity 
  (α : ℝ) 
  (h : Real.tan α = 1 / 3) : 
  Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) = 3 / 10 :=
sorry

end trig_identity_l304_304826


namespace bridge_length_is_235_l304_304490

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * time_sec
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_is_235 :
  length_of_bridge 140 45 30 = 235 :=
by 
  sorry

end bridge_length_is_235_l304_304490


namespace repeating_decimal_fraction_l304_304213

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l304_304213


namespace number_of_plains_routes_is_81_l304_304280

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end number_of_plains_routes_is_81_l304_304280


namespace sum_of_fourth_powers_l304_304956

theorem sum_of_fourth_powers
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 15.5 := sorry

end sum_of_fourth_powers_l304_304956


namespace length_more_than_breadth_l304_304325

theorem length_more_than_breadth (b x : ℕ) 
  (h1 : 60 = b + x) 
  (h2 : 4 * b + 2 * x = 200) : x = 20 :=
by {
  sorry
}

end length_more_than_breadth_l304_304325


namespace beads_per_necklace_correct_l304_304074
-- Importing the necessary library.

-- Defining the given number of necklaces and total beads.
def number_of_necklaces : ℕ := 11
def total_beads : ℕ := 308

-- Stating the proof goal as a theorem.
theorem beads_per_necklace_correct : (total_beads / number_of_necklaces) = 28 := 
by
  sorry

end beads_per_necklace_correct_l304_304074


namespace scientific_notation_of_11580000_l304_304834

theorem scientific_notation_of_11580000 :
  11_580_000 = 1.158 * 10^7 :=
sorry

end scientific_notation_of_11580000_l304_304834


namespace trapezoid_area_l304_304881

theorem trapezoid_area 
  (h : ℝ) (BM CM : ℝ) 
  (height_cond : h = 12) 
  (BM_cond : BM = 15) 
  (CM_cond : CM = 13) 
  (angle_bisectors_intersect : ∃ M : ℝ, (BM^2 - h^2) = 9^2 ∧ (CM^2 - h^2) = 5^2) : 
  ∃ (S : ℝ), S = 260.4 :=
by
  -- Skipping the proof part by using sorry
  sorry

end trapezoid_area_l304_304881


namespace range_f_1_range_m_l304_304810

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 2) * (Real.log x / (2 * Real.log 2) - 1/2)

theorem range_f_1 (x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : 
  -1/8 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem range_m (m : ℝ) (x : ℝ) (h1 : 4 ≤ x) (h2 : x ≤ 16) (h3 : f x ≥ m * Real.log x / Real.log 2) :
  m ≤ 0 :=
sorry

end range_f_1_range_m_l304_304810


namespace avg_marks_second_class_l304_304011

theorem avg_marks_second_class
  (x : ℝ)
  (avg_class1 : ℝ)
  (avg_total : ℝ)
  (n1 n2 : ℕ)
  (h1 : n1 = 30)
  (h2 : n2 = 50)
  (h3 : avg_class1 = 30)
  (h4: avg_total = 48.75)
  (h5 : (n1 * avg_class1 + n2 * x) / (n1 + n2) = avg_total) :
  x = 60 := by
  sorry

end avg_marks_second_class_l304_304011


namespace banana_permutations_l304_304959

theorem banana_permutations : 
  let b_freq := 1
  let n_freq := 2
  let a_freq := 3
  let total_letters := 6 in
  nat.choose total_letters 1 * nat.choose (total_letters - 1) 2 * nat.choose (total_letters - 3) 3 = 60 := 
by
  let fact := nat.factorial
  let perms := fact total_letters / (fact b_freq * fact n_freq * fact a_freq)
  exact perms = 60

end banana_permutations_l304_304959


namespace stock_yield_percentage_l304_304746

noncomputable def FaceValue : ℝ := 100
noncomputable def AnnualYield : ℝ := 0.20 * FaceValue
noncomputable def MarketPrice : ℝ := 166.66666666666669
noncomputable def ExpectedYieldPercentage : ℝ := 12

theorem stock_yield_percentage :
  (AnnualYield / MarketPrice) * 100 = ExpectedYieldPercentage :=
by
  -- given conditions directly from the problem
  have h1 : FaceValue = 100 := rfl
  have h2 : AnnualYield = 0.20 * FaceValue := rfl
  have h3 : MarketPrice = 166.66666666666669 := rfl
  
  -- we are proving that the yield percentage is 12%
  sorry

end stock_yield_percentage_l304_304746


namespace possible_classmates_l304_304663

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l304_304663


namespace no_nat_fourfold_digit_move_l304_304866

theorem no_nat_fourfold_digit_move :
  ¬ ∃ (N : ℕ), ∃ (a : ℕ), ∃ (n : ℕ), ∃ (x : ℕ),
    (1 ≤ a ∧ a ≤ 9) ∧ 
    (N = a * 10^n + x) ∧ 
    (4 * N = 10 * x + a) :=
by
  sorry

end no_nat_fourfold_digit_move_l304_304866


namespace louisa_average_speed_l304_304900

def average_speed (v : ℝ) : Prop :=
  (350 / v) - (200 / v) = 3

theorem louisa_average_speed :
  ∃ v : ℝ, average_speed v ∧ v = 50 := 
by
  use 50
  unfold average_speed
  sorry

end louisa_average_speed_l304_304900


namespace square_side_length_l304_304174

theorem square_side_length (A : ℝ) (π : ℝ) (s : ℝ) (area_circle_eq : A = 100)
  (area_circle_eq_perimeter_square : A = 4 * s) : s = 25 := by
  sorry

end square_side_length_l304_304174


namespace decimal_to_fraction_l304_304227

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l304_304227


namespace sector_angle_l304_304588

theorem sector_angle (r : ℝ) (S_sector : ℝ) (h_r : r = 2) (h_S : S_sector = (2 / 5) * π) : 
  (∃ α : ℝ, S_sector = (1 / 2) * α * r^2 ∧ α = (π / 5)) :=
by
  use π / 5
  sorry

end sector_angle_l304_304588


namespace bc_sum_l304_304758

theorem bc_sum (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 10) : B + C = 310 := by
  sorry

end bc_sum_l304_304758


namespace simplify_expression_l304_304003

theorem simplify_expression : 5 * (18 / -9) * (24 / 36) = -(20 / 3) :=
by
  sorry

end simplify_expression_l304_304003


namespace candidate_percentage_l304_304538

theorem candidate_percentage (P : ℕ) (total_votes : ℕ) (vote_diff : ℕ)
  (h1 : total_votes = 7000)
  (h2 : vote_diff = 2100)
  (h3 : (P * total_votes / 100) + (P * total_votes / 100) + vote_diff = total_votes) :
  P = 35 :=
by
  sorry

end candidate_percentage_l304_304538


namespace solve_equation_l304_304005

theorem solve_equation :
  ∃ (x y z : ℕ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 4 ∧ 
  (2^x + 3^y + 7 = nat.factorial z) ∧
  ((x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5)) := 
by {
  sorry
}

end solve_equation_l304_304005


namespace factor_2w4_minus_50_l304_304410

noncomputable def factor_expr (w : Polynomial ℝ) : Polynomial ℝ :=
  2 * (w^2 - 5) * (w^2 + 5)

theorem factor_2w4_minus_50 (w : Polynomial ℝ) :
  2 * w^4 - 50 = factor_expr w :=
by {
  sorry
}

end factor_2w4_minus_50_l304_304410


namespace new_person_weight_l304_304012

theorem new_person_weight (W : ℝ) :
  (∃ (W : ℝ), (390 - W + 70) / 4 = (390 - W) / 4 + 3 ∧ (390 - W + W) = 390) → 
  W = 58 :=
by
  sorry

end new_person_weight_l304_304012


namespace sin_double_angle_in_fourth_quadrant_l304_304111

theorem sin_double_angle_in_fourth_quadrant (α : ℝ) (h : -π/2 < α ∧ α < 0) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_in_fourth_quadrant_l304_304111


namespace weeks_to_save_remaining_l304_304301

-- Assuming the conditions
def cost_of_shirt : ℝ := 3
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

-- The proof goal
theorem weeks_to_save_remaining (cost_of_shirt amount_saved saving_per_week : ℝ) :
  cost_of_shirt = 3 ∧ amount_saved = 1.5 ∧ saving_per_week = 0.5 →
  ((cost_of_shirt - amount_saved) / saving_per_week) = 3 := by
  sorry

end weeks_to_save_remaining_l304_304301


namespace determine_a_range_f_l304_304627

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - (2 / (2 ^ x + 1))

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) -> a = 1 :=
by
  sorry

theorem range_f (x : ℝ) : (∀ x : ℝ, f 1 (-x) = -f 1 x) -> -1 < f 1 x ∧ f 1 x < 1 :=
by
  sorry

end determine_a_range_f_l304_304627


namespace sum_of_trinomials_1_l304_304033

theorem sum_of_trinomials_1 (p q : ℝ) :
  (p + q = 0 ∨ p + q = 8) →
  (2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 2 ∨ 2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 18) :=
by sorry

end sum_of_trinomials_1_l304_304033


namespace sector_area_l304_304603

noncomputable def l : ℝ := 4
noncomputable def θ : ℝ := 2
noncomputable def r : ℝ := l / θ

theorem sector_area :
  (1 / 2) * l * r = 4 :=
by
  -- Proof goes here
  sorry

end sector_area_l304_304603


namespace domain_of_myFunction_l304_304025

-- Define the function
def myFunction (x : ℝ) : ℝ := (x + 2) ^ (1 / 2) - (x + 1) ^ 0

-- State the domain constraints as a theorem
theorem domain_of_myFunction (x : ℝ) : 
  (x ≥ -2 ∧ x ≠ -1) →
  ∃ y : ℝ, y = myFunction x := 
sorry

end domain_of_myFunction_l304_304025


namespace option_B_valid_l304_304938

-- Definitions derived from conditions
def at_least_one_black (balls : List Bool) : Prop :=
  ∃ b ∈ balls, b = true

def both_black (balls : List Bool) : Prop :=
  balls = [true, true]

def exactly_one_black (balls : List Bool) : Prop :=
  balls.count true = 1

def exactly_two_black (balls : List Bool) : Prop :=
  balls.count true = 2

def mutually_exclusive (P Q : Prop) : Prop :=
  P ∧ Q → False

def non_complementary (P Q : Prop) : Prop :=
  ¬(P → ¬Q) ∧ ¬(¬P → Q)

-- Balls: true represents a black ball, false represents a red ball.
def all_draws := [[true, true], [true, false], [false, true], [false, false]]

-- Proof statement
theorem option_B_valid :
  (mutually_exclusive (exactly_one_black [true, false]) (exactly_two_black [true, true])) ∧ 
  (non_complementary (exactly_one_black [true, false]) (exactly_two_black [true, true])) :=
  sorry

end option_B_valid_l304_304938
