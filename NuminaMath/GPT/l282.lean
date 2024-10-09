import Mathlib

namespace sin_double_angle_condition_l282_28203

theorem sin_double_angle_condition (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1 / 3) : Real.sin (2 * θ) = -8 / 9 := 
sorry

end sin_double_angle_condition_l282_28203


namespace items_in_bags_l282_28217

def calculateWaysToPlaceItems (n_items : ℕ) (n_bags : ℕ) : ℕ :=
  sorry

theorem items_in_bags :
  calculateWaysToPlaceItems 5 3 = 41 :=
by sorry

end items_in_bags_l282_28217


namespace antonella_purchase_l282_28237

theorem antonella_purchase
  (total_coins : ℕ)
  (coin_value : ℕ → ℕ)
  (num_toonies : ℕ)
  (initial_loonies : ℕ)
  (initial_toonies : ℕ)
  (total_value : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (H1 : total_coins = 10)
  (H2 : coin_value 1 = 1)
  (H3 : coin_value 2 = 2)
  (H4 : initial_toonies = 4)
  (H5 : initial_loonies = total_coins - initial_toonies)
  (H6 : total_value = initial_loonies * coin_value 1 + initial_toonies * coin_value 2)
  (H7 : amount_spent = 3)
  (H8 : amount_left = total_value - amount_spent)
  (H9 : amount_left = 11) :
  ∃ (used_loonies used_toonies : ℕ), used_loonies = 1 ∧ used_toonies = 1 ∧ (used_loonies * coin_value 1 + used_toonies * coin_value 2 = amount_spent) :=
by
  sorry

end antonella_purchase_l282_28237


namespace math_problem_l282_28281

theorem math_problem
  (m : ℕ) (h₁ : m = 8^126) :
  (m * 16) / 64 = 16^94 :=
by
  sorry

end math_problem_l282_28281


namespace total_students_in_faculty_l282_28208

theorem total_students_in_faculty :
  (let sec_year_num := 230
   let sec_year_auto := 423
   let both_subj := 134
   let sec_year_total := 0.80
   let at_least_one_subj := sec_year_num + sec_year_auto - both_subj
   ∃ (T : ℝ), sec_year_total * T = at_least_one_subj ∧ T = 649) := by
  sorry

end total_students_in_faculty_l282_28208


namespace norb_age_is_47_l282_28298

section NorbAge

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def exactlyHalfGuessesTooLow (guesses : List ℕ) (age : ℕ) : Prop :=
  (guesses.filter (λ x => x < age)).length = (guesses.length / 2)

def oneGuessOffByTwo (guesses : List ℕ) (age : ℕ) : Prop :=
  guesses.any (λ x => x = age + 2 ∨ x = age - 2)

def validAge (guesses : List ℕ) (age : ℕ) : Prop :=
  exactlyHalfGuessesTooLow guesses age ∧ oneGuessOffByTwo guesses age ∧ isPrime age

theorem norb_age_is_47 : validAge [23, 29, 33, 35, 39, 41, 46, 48, 50, 54] 47 :=
sorry

end NorbAge

end norb_age_is_47_l282_28298


namespace solution_set_of_inequality_l282_28255

theorem solution_set_of_inequality (x : ℝ) : 
  (x ≠ 0 ∧ (x * (x - 1)) ≤ 0) ↔ 0 < x ∧ x ≤ 1 :=
sorry

end solution_set_of_inequality_l282_28255


namespace fraction_to_decimal_l282_28275

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l282_28275


namespace trains_clear_each_other_in_11_seconds_l282_28225

-- Define the lengths of the trains
def length_train1 := 100  -- in meters
def length_train2 := 120  -- in meters

-- Define the speeds of the trains (in km/h), converted to m/s
def speed_train1 := 42 * 1000 / 3600  -- 42 km/h to m/s
def speed_train2 := 30 * 1000 / 3600  -- 30 km/h to m/s

-- Calculate the total distance to be covered
def total_distance := length_train1 + length_train2  -- in meters

-- Calculate the relative speed when they are moving towards each other
def relative_speed := speed_train1 + speed_train2  -- in m/s

-- Calculate the time required for the trains to be clear of each other (in seconds)
noncomputable def clear_time := total_distance / relative_speed

-- Theorem stating the above
theorem trains_clear_each_other_in_11_seconds :
  clear_time = 11 :=
by
  -- Proof would go here
  sorry

end trains_clear_each_other_in_11_seconds_l282_28225


namespace percentage_less_than_l282_28286

theorem percentage_less_than (T F S : ℝ) 
  (hF : F = 0.70 * T) 
  (hS : S = 0.63 * T) : 
  ((T - S) / T) * 100 = 37 := 
by
  sorry

end percentage_less_than_l282_28286


namespace product_mod_eq_l282_28212

theorem product_mod_eq :
  (1497 * 2003) % 600 = 291 := 
sorry

end product_mod_eq_l282_28212


namespace john_free_throws_l282_28230

theorem john_free_throws 
  (hit_rate : ℝ) 
  (shots_per_foul : ℕ) 
  (fouls_per_game : ℕ) 
  (total_games : ℕ) 
  (percentage_played : ℝ) 
  : hit_rate = 0.7 → 
    shots_per_foul = 2 → 
    fouls_per_game = 5 → 
    total_games = 20 → 
    percentage_played = 0.8 → 
    ∃ (total_free_throws : ℕ), total_free_throws = 112 := 
by
  intros
  sorry

end john_free_throws_l282_28230


namespace initial_books_count_l282_28224

-- Definitions of the given conditions
def shelves : ℕ := 9
def books_per_shelf : ℕ := 9
def books_remaining : ℕ := shelves * books_per_shelf
def books_sold : ℕ := 39

-- Statement of the proof problem
theorem initial_books_count : books_remaining + books_sold = 120 := 
by {
  sorry
}

end initial_books_count_l282_28224


namespace sin_alpha_cos_squared_beta_range_l282_28251

theorem sin_alpha_cos_squared_beta_range (α β : ℝ) 
  (h : Real.sin α + Real.sin β = 1) : 
  ∃ y, y = Real.sin α - Real.cos β ^ 2 ∧ (-1/4 ≤ y ∧ y ≤ 0) :=
sorry

end sin_alpha_cos_squared_beta_range_l282_28251


namespace radius_of_inscribed_circle_l282_28246

noncomputable def radius_inscribed_circle (AB BC AC : ℝ) (s : ℝ) (K : ℝ) : ℝ := K / s

theorem radius_of_inscribed_circle (AB BC AC : ℝ) (h1: AB = 8) (h2: BC = 8) (h3: AC = 10) :
  radius_inscribed_circle AB BC AC 13 (5 * Real.sqrt 39) = (5 * Real.sqrt 39) / 13 :=
  by
  sorry

end radius_of_inscribed_circle_l282_28246


namespace sum_x_y_eq_two_l282_28241

theorem sum_x_y_eq_two (x y : ℝ) 
  (h1 : (x-1)^3 + 2003*(x-1) = -1) 
  (h2 : (y-1)^3 + 2003*(y-1) = 1) : 
  x + y = 2 :=
by
  sorry

end sum_x_y_eq_two_l282_28241


namespace unique_third_rectangle_exists_l282_28263

-- Define the given rectangles.
def rect1_length : ℕ := 3
def rect1_width : ℕ := 8
def rect2_length : ℕ := 2
def rect2_width : ℕ := 5

-- Define the areas of the given rectangles.
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width

-- Define the total area covered by the two given rectangles.
def total_area_without_third : ℕ := area_rect1 + area_rect2

-- We need to prove that there exists one unique configuration for the third rectangle.
theorem unique_third_rectangle_exists (a b : ℕ) : 
  (total_area_without_third + a * b = 34) → 
  (a * b = 4) → 
  (a = 4 ∧ b = 1 ∨ a = 1 ∧ b = 4) :=
by sorry

end unique_third_rectangle_exists_l282_28263


namespace group4_equations_groupN_equations_find_k_pos_l282_28214

-- Conditions from the problem
def group1_fractions := (1 : ℚ) / 1 + (1 : ℚ) / 3 = 4 / 3
def group1_pythagorean := 4^2 + 3^2 = 5^2

def group2_fractions := (1 : ℚ) / 3 + (1 : ℚ) / 5 = 8 / 15
def group2_pythagorean := 8^2 + 15^2 = 17^2

def group3_fractions := (1 : ℚ) / 5 + (1 : ℚ) / 7 = 12 / 35
def group3_pythagorean := 12^2 + 35^2 = 37^2

-- Proof Statements
theorem group4_equations :
  ((1 : ℚ) / 7 + (1 : ℚ) / 9 = 16 / 63) ∧ (16^2 + 63^2 = 65^2) := 
  sorry

theorem groupN_equations (n : ℕ) :
  ((1 : ℚ) / (2 * n - 1) + (1 : ℚ) / (2 * n + 1) = 4 * n / (4 * n^2 - 1)) ∧
  ((4 * n)^2 + (4 * n^2 - 1)^2 = (4 * n^2 + 1)^2) :=
  sorry

theorem find_k_pos (k : ℕ) : 
  k^2 + 9603^2 = 9605^2 → k = 196 := 
  sorry

end group4_equations_groupN_equations_find_k_pos_l282_28214


namespace question_solution_l282_28278

variable (a b : ℝ)

theorem question_solution : 2 * a - 3 * (a - b) = -a + 3 * b := by
  sorry

end question_solution_l282_28278


namespace zongzi_unit_price_l282_28297

theorem zongzi_unit_price (uA uB : ℝ) (pA pB : ℝ) : 
  pA = 1200 → pB = 800 → uA = 2 * uB → pA / uA = pB / uB - 50 → uB = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end zongzi_unit_price_l282_28297


namespace line_through_points_l282_28238

theorem line_through_points (a b : ℝ) (h₁ : 1 = a * 3 + b) (h₂ : 13 = a * 7 + b) : a - b = 11 := 
  sorry

end line_through_points_l282_28238


namespace inheritance_division_l282_28222

variables {M P Q R : ℝ} {p q r : ℕ}

theorem inheritance_division (hP : P < 99 * (p : ℝ))
                             (hR : R > 10000 * (r : ℝ))
                             (hM : M = P + Q + R)
                             (hRichPoor : R ≥ P) : 
                             R ≥ 100 * P := 
sorry

end inheritance_division_l282_28222


namespace right_triangle_side_length_l282_28221

theorem right_triangle_side_length (x : ℝ) (hx : x > 0) (h_area : (1 / 2) * x * (3 * x) = 108) :
  x = 6 * Real.sqrt 2 :=
sorry

end right_triangle_side_length_l282_28221


namespace smallest_n_terminating_decimal_l282_28276

-- Define the given condition: n + 150 must be expressible as 2^a * 5^b.
def has_terminating_decimal_property (n : ℕ) := ∃ a b : ℕ, n + 150 = 2^a * 5^b

-- We want to prove that the smallest n satisfying the property is 50.
theorem smallest_n_terminating_decimal :
  (∀ n : ℕ, n > 0 ∧ has_terminating_decimal_property n → n ≥ 50) ∧ (has_terminating_decimal_property 50) :=
by
  sorry

end smallest_n_terminating_decimal_l282_28276


namespace gcd_8154_8640_l282_28247

theorem gcd_8154_8640 : Nat.gcd 8154 8640 = 6 := by
  sorry

end gcd_8154_8640_l282_28247


namespace john_total_feet_climbed_l282_28218

def first_stair_steps : ℕ := 20
def second_stair_steps : ℕ := 2 * first_stair_steps
def third_stair_steps : ℕ := second_stair_steps - 10
def step_height : ℝ := 0.5

theorem john_total_feet_climbed : 
  (first_stair_steps + second_stair_steps + third_stair_steps) * step_height = 45 :=
by
  sorry

end john_total_feet_climbed_l282_28218


namespace inequality_l282_28215

theorem inequality (a b c : ℝ) (h₀ : 0 < c) (h₁ : c < b) (h₂ : b < a) :
  a^4 * b + b^4 * c + c^4 * a > a * b^4 + b * c^4 + c * a^4 :=
by sorry

end inequality_l282_28215


namespace probability_of_adjacent_vertices_in_dodecagon_l282_28210

def probability_at_least_two_adjacent_vertices (n : ℕ) : ℚ :=
  if n = 12 then 24 / 55 else 0  -- Only considering the dodecagon case

theorem probability_of_adjacent_vertices_in_dodecagon :
  probability_at_least_two_adjacent_vertices 12 = 24 / 55 :=
by
  sorry

end probability_of_adjacent_vertices_in_dodecagon_l282_28210


namespace dexter_total_cards_l282_28248

theorem dexter_total_cards 
  (boxes_basketball : ℕ) 
  (cards_per_basketball_box : ℕ) 
  (boxes_football : ℕ) 
  (cards_per_football_box : ℕ) 
   (h1 : boxes_basketball = 15)
   (h2 : cards_per_basketball_box = 20)
   (h3 : boxes_football = boxes_basketball - 7)
   (h4 : cards_per_football_box = 25) 
   : boxes_basketball * cards_per_basketball_box + boxes_football * cards_per_football_box = 500 := by 
sorry

end dexter_total_cards_l282_28248


namespace ratio_won_to_lost_l282_28242

-- Define the total number of games and the number of games won
def total_games : Nat := 30
def games_won : Nat := 18

-- Define the number of games lost
def games_lost : Nat := total_games - games_won

-- Define the ratio of games won to games lost as a pair
def ratio : Nat × Nat := (games_won / Nat.gcd games_won games_lost, games_lost / Nat.gcd games_won games_lost)

-- The theorem to be proved
theorem ratio_won_to_lost : ratio = (3, 2) :=
  by
    -- Skipping the proof here
    sorry

end ratio_won_to_lost_l282_28242


namespace city_grid_sinks_l282_28233

-- Define the main conditions of the grid city
def cell_side_meter : Int := 500
def max_travel_km : Int := 1

-- Total number of intersections in a 100x100 grid
def total_intersections : Int := (100 + 1) * (100 + 1)

-- Number of sinks that need to be proven
def required_sinks : Int := 1300

-- Lean theorem statement to prove that given the conditions,
-- there are at least 1300 sinks (intersections that act as sinks)
theorem city_grid_sinks :
  ∀ (city_grid : Matrix (Fin 101) (Fin 101) IntersectionType),
  (∀ i j, i < 100 → j < 100 → cell_side_meter ≤ max_travel_km * 1000) →
  ∃ (sinks : Finset (Fin 101 × Fin 101)), 
  (sinks.card ≥ required_sinks) := sorry

end city_grid_sinks_l282_28233


namespace max_3cosx_4sinx_l282_28239

theorem max_3cosx_4sinx (x : ℝ) : (3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧ (∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5) :=
  sorry

end max_3cosx_4sinx_l282_28239


namespace complex_round_quadrant_l282_28216

open Complex

theorem complex_round_quadrant (z : ℂ) (i : ℂ) (h : i = Complex.I) (h1 : z * i = 2 - i):
  z.re < 0 ∧ z.im < 0 := 
sorry

end complex_round_quadrant_l282_28216


namespace smallest_angle_product_l282_28204

-- Define an isosceles triangle with angle at B being the smallest angle
def isosceles_triangle (α : ℝ) : Prop :=
  α < 90 ∧ α = 180 / 7

-- Proof that the smallest angle multiplied by 6006 is 154440
theorem smallest_angle_product : 
  isosceles_triangle α → (180 / 7) * 6006 = 154440 :=
by
  intros
  sorry

end smallest_angle_product_l282_28204


namespace parabola_directrix_l282_28234

theorem parabola_directrix (x y : ℝ) (h_parabola : x^2 = (1/2) * y) : y = - (1/8) :=
sorry

end parabola_directrix_l282_28234


namespace river_length_GSA_AWRA_l282_28220

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end river_length_GSA_AWRA_l282_28220


namespace problem1_problem2_l282_28285

-- Define the base types and expressions
variables (x m : ℝ)

-- Proofs of the given expressions
theorem problem1 : (x^7 / x^3) * x^4 = x^8 :=
by sorry

theorem problem2 : m * m^3 + ((-m^2)^3 / m^2) = 0 :=
by sorry

end problem1_problem2_l282_28285


namespace probability_l282_28266

def total_chips : ℕ := 15
def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

def probability_of_different_colors : ℚ :=
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips)

theorem probability : probability_of_different_colors = 148 / 225 :=
by
  unfold probability_of_different_colors
  sorry

end probability_l282_28266


namespace mn_min_l282_28260

noncomputable def min_mn_value (m n : ℝ) : ℝ := m * n

theorem mn_min : 
  (∃ m n, m = Real.sin (2 * (π / 12)) ∧ n > 0 ∧ 
            Real.cos (2 * (π / 12 + n) - π / 4) = m ∧ 
            min_mn_value m n = π * 5 / 48) := by
  sorry

end mn_min_l282_28260


namespace max_sum_unit_hexagons_l282_28287

theorem max_sum_unit_hexagons (k : ℕ) (hk : k ≥ 3) : 
  ∃ S, S = 6 + (3 * k - 9) * k * (k + 1) / 2 + (3 * (k^2 - 2)) * (k * (k + 1) * (2 * k + 1) / 6) / 6 ∧
       S = 3 * (k * k - 14 * k + 33 * k - 28) / 2 :=
by
  sorry

end max_sum_unit_hexagons_l282_28287


namespace buns_problem_l282_28227

theorem buns_problem (N : ℕ) (x y u v : ℕ) 
  (h1 : 3 * x + 5 * y = 25)
  (h2 : 3 * u + 5 * v = 35)
  (h3 : x + y = N)
  (h4 : u + v = N) : 
  N = 7 := 
sorry

end buns_problem_l282_28227


namespace tan_sum_identity_l282_28270

-- Definitions
def quadratic_eq (x : ℝ) : Prop := 6 * x^2 - 5 * x + 1 = 0
def tan_roots (α β : ℝ) : Prop := quadratic_eq (Real.tan α) ∧ quadratic_eq (Real.tan β)

-- Problem statement
theorem tan_sum_identity (α β : ℝ) (hαβ : tan_roots α β) : Real.tan (α + β) = 1 :=
sorry

end tan_sum_identity_l282_28270


namespace smallest_base_converted_l282_28265

def convert_to_decimal_base_3 (n : ℕ) : ℕ :=
  1 * 3^3 + 0 * 3^2 + 0 * 3^1 + 2 * 3^0

def convert_to_decimal_base_6 (n : ℕ) : ℕ :=
  2 * 6^2 + 1 * 6^1 + 0 * 6^0

def convert_to_decimal_base_4 (n : ℕ) : ℕ :=
  1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0

def convert_to_decimal_base_2 (n : ℕ) : ℕ :=
  1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem smallest_base_converted :
  min (convert_to_decimal_base_3 1002) 
      (min (convert_to_decimal_base_6 210) 
           (min (convert_to_decimal_base_4 1000) 
                (convert_to_decimal_base_2 111111))) = convert_to_decimal_base_3 1002 :=
by sorry

end smallest_base_converted_l282_28265


namespace age_problem_lean4_l282_28202

/-
Conditions:
1. Mr. Bernard's age in eight years will be 60.
2. Luke's age in eight years will be 28.
3. Sarah's age in eight years will be 48.
4. The sum of their ages in eight years will be 136.

Question (translated to proof problem):
Prove that 10 years less than the average age of all three of them is approximately 35.33.

The Lean 4 statement below formalizes this:
-/

theorem age_problem_lean4 :
  let bernard_age := 60
  let luke_age := 28
  let sarah_age := 48
  let total_age := bernard_age + luke_age + sarah_age
  total_age = 136 → ((total_age / 3.0) - 10.0 = 35.33) :=
by
  intros
  sorry

end age_problem_lean4_l282_28202


namespace train_crossing_time_l282_28245

noncomputable def relative_speed_kmh (speed_train : ℕ) (speed_man : ℕ) : ℕ := speed_train + speed_man

noncomputable def kmh_to_mps (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def crossing_time (length_train : ℕ) (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) : ℝ :=
  let relative_speed_kmh := relative_speed_kmh speed_train_kmh speed_man_kmh
  let relative_speed_mps := kmh_to_mps relative_speed_kmh
  length_train / relative_speed_mps

theorem train_crossing_time :
  crossing_time 210 25 2 = 28 :=
  by
  sorry

end train_crossing_time_l282_28245


namespace values_of_j_for_exactly_one_real_solution_l282_28201

open Real

theorem values_of_j_for_exactly_one_real_solution :
  ∀ j : ℝ, (∀ x : ℝ, (3 * x + 4) * (x - 6) = -51 + j * x) → (j = 0 ∨ j = -36) := by
sorry

end values_of_j_for_exactly_one_real_solution_l282_28201


namespace sum_of_reciprocals_is_3_over_8_l282_28268

theorem sum_of_reciprocals_is_3_over_8 (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  (1 / x + 1 / y) = 3 / 8 := 
by 
  sorry

end sum_of_reciprocals_is_3_over_8_l282_28268


namespace probability_of_pink_flower_is_five_over_nine_l282_28235

-- Definitions as per the conditions
def flowersInBagA := 9
def pinkFlowersInBagA := 3
def flowersInBagB := 9
def pinkFlowersInBagB := 7
def probChoosingBag := (1:ℚ) / 2

-- Definition of the probabilities
def probPinkFromA := pinkFlowersInBagA / flowersInBagA
def probPinkFromB := pinkFlowersInBagB / flowersInBagB

-- Total probability calculation using the law of total probability
def probPink := probPinkFromA * probChoosingBag + probPinkFromB * probChoosingBag

-- Statement to be proved
theorem probability_of_pink_flower_is_five_over_nine : probPink = (5:ℚ) / 9 := 
by
  sorry

end probability_of_pink_flower_is_five_over_nine_l282_28235


namespace people_per_pizza_l282_28256

def pizza_cost := 12 -- dollars per pizza
def babysitting_earnings_per_night := 4 -- dollars per night
def nights_babysitting := 15
def total_people := 15

theorem people_per_pizza : (babysitting_earnings_per_night * nights_babysitting / pizza_cost) = (total_people / ((babysitting_earnings_per_night * nights_babysitting / pizza_cost))) := 
by
  sorry

end people_per_pizza_l282_28256


namespace find_sin_cos_of_perpendicular_vectors_l282_28254

theorem find_sin_cos_of_perpendicular_vectors 
  (θ : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (h_a : a = (Real.sin θ, -2)) 
  (h_b : b = (1, Real.cos θ)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) 
  (h_theta_range : 0 < θ ∧ θ < Real.pi / 2) : 
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ Real.cos θ = Real.sqrt 5 / 5 := 
by 
  sorry

end find_sin_cos_of_perpendicular_vectors_l282_28254


namespace no_square_remainder_2_infinitely_many_squares_remainder_3_l282_28296

theorem no_square_remainder_2 :
  ∀ n : ℤ, (n * n) % 6 ≠ 2 :=
by sorry

theorem infinitely_many_squares_remainder_3 :
  ∀ k : ℤ, ∃ n : ℤ, n = 6 * k + 3 ∧ (n * n) % 6 = 3 :=
by sorry

end no_square_remainder_2_infinitely_many_squares_remainder_3_l282_28296


namespace spring_festival_scientific_notation_l282_28273

noncomputable def scientific_notation := (260000000: ℝ) = (2.6 * 10^8)

theorem spring_festival_scientific_notation : scientific_notation :=
by
  -- proof logic goes here
  sorry

end spring_festival_scientific_notation_l282_28273


namespace range_of_x2_plus_y2_l282_28258

theorem range_of_x2_plus_y2 (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f x = -f (-x))
  (h_increasing : ∀ x y : ℝ, x < y → f x < f y)
  (x y : ℝ)
  (h_inequality : f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) :
  16 < x^2 + y^2 ∧ x^2 + y^2 < 36 :=
sorry

end range_of_x2_plus_y2_l282_28258


namespace remainder_of_x_l282_28231

theorem remainder_of_x (x : ℤ) (h : 2 * x - 3 = 7) : x % 2 = 1 := by
  sorry

end remainder_of_x_l282_28231


namespace factorization_6x2_minus_24x_plus_18_l282_28292

theorem factorization_6x2_minus_24x_plus_18 :
    ∀ x : ℝ, 6 * x^2 - 24 * x + 18 = 6 * (x - 1) * (x - 3) :=
by
  intro x
  sorry

end factorization_6x2_minus_24x_plus_18_l282_28292


namespace smallest_x_plus_y_l282_28200

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l282_28200


namespace larger_cookie_raisins_l282_28211

theorem larger_cookie_raisins : ∃ n r, 5 ≤ n ∧ n ≤ 10 ∧ (n - 1) * r + (r + 1) = 100 ∧ r + 1 = 12 :=
by
  sorry

end larger_cookie_raisins_l282_28211


namespace min_value_l282_28299

noncomputable def min_res (a b c : ℝ) : ℝ := 
  if h : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
  then (1 / a + 2 / b + 3 / c) 
  else 0

theorem min_value (a b c : ℝ) : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
    → min_res a b c = 6 := 
sorry

end min_value_l282_28299


namespace smallest_interior_angle_l282_28293

open Real

theorem smallest_interior_angle (A B C : ℝ) (hA : 0 < A ∧ A < π)
    (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π)
    (h_sum_angles : A + B + C = π)
    (h_ratio : sin A / sin B = 2 / sqrt 6 ∧ sin A / sin C = 2 / (sqrt 3 + 1)) :
    min A (min B C) = π / 4 := 
  by sorry

end smallest_interior_angle_l282_28293


namespace total_leaves_correct_l282_28226

-- Definitions based on conditions
def basil_pots := 3
def rosemary_pots := 9
def thyme_pots := 6

def basil_leaves_per_pot := 4
def rosemary_leaves_per_pot := 18
def thyme_leaves_per_pot := 30

-- Calculate the total number of leaves
def total_leaves : Nat :=
  (basil_pots * basil_leaves_per_pot) +
  (rosemary_pots * rosemary_leaves_per_pot) +
  (thyme_pots * thyme_leaves_per_pot)

-- The statement to prove
theorem total_leaves_correct : total_leaves = 354 := by
  sorry

end total_leaves_correct_l282_28226


namespace solve_x_minus_y_l282_28250

theorem solve_x_minus_y :
  (2 = 0.25 * x) → (2 = 0.1 * y) → (x - y = -12) :=
by
  sorry

end solve_x_minus_y_l282_28250


namespace tower_surface_area_l282_28290

noncomputable def total_visible_surface_area (volumes : List ℕ) : ℕ := sorry

theorem tower_surface_area :
  total_visible_surface_area [512, 343, 216, 125, 64, 27, 8, 1] = 882 :=
sorry

end tower_surface_area_l282_28290


namespace dogs_sold_l282_28243

theorem dogs_sold (cats_sold : ℕ) (h1 : cats_sold = 16) (ratio : ℕ × ℕ) (h2 : ratio = (2, 1)) : ∃ dogs_sold : ℕ, dogs_sold = 8 := by
  sorry

end dogs_sold_l282_28243


namespace painting_two_sides_time_l282_28213

-- Definitions for the conditions
def time_to_paint_one_side_per_board : Nat := 1
def drying_time_per_board : Nat := 5

-- Definitions for the problem
def total_boards : Nat := 6

-- Main theorem statement
theorem painting_two_sides_time :
  (total_boards * time_to_paint_one_side_per_board) + drying_time_per_board + (total_boards * time_to_paint_one_side_per_board) = 12 :=
sorry

end painting_two_sides_time_l282_28213


namespace heroes_on_the_back_l282_28264

theorem heroes_on_the_back (total_heroes front_heroes : ℕ) (h1 : total_heroes = 9) (h2 : front_heroes = 2) :
  total_heroes - front_heroes = 7 := by
  sorry

end heroes_on_the_back_l282_28264


namespace maximum_profit_l282_28253

noncomputable def profit (x : ℝ) : ℝ :=
  5.06 * x - 0.15 * x^2 + 2 * (15 - x)

theorem maximum_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 :=
by
  sorry

end maximum_profit_l282_28253


namespace collinearity_necessary_but_not_sufficient_l282_28272

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u

def equal (u v : V) : Prop := u = v

theorem collinearity_necessary_but_not_sufficient (u v : V) :
  (collinear u v → equal u v) ∧ (equal u v → collinear u v) → collinear u v ∧ ¬(collinear u v ↔ equal u v) :=
sorry

end collinearity_necessary_but_not_sufficient_l282_28272


namespace gcd_fx_x_l282_28206

def f (x: ℕ) := (5 * x + 4) * (9 * x + 7) * (11 * x + 3) * (x + 12)

theorem gcd_fx_x (x: ℕ) (h: x % 54896 = 0) : Nat.gcd (f x) x = 112 :=
  sorry

end gcd_fx_x_l282_28206


namespace proof_inequality_l282_28288

noncomputable def proof_problem (a b c d : ℝ) (h_ab : a * b + b * c + c * d + d * a = 1) : Prop :=
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1 / 3

theorem proof_inequality (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_ab : a * b + b * c + c * d + d * a = 1) : 
  proof_problem a b c d h_ab := 
by
  sorry

end proof_inequality_l282_28288


namespace sum_of_intercepts_l282_28282

theorem sum_of_intercepts (x y : ℝ) (h : 3 * x - 4 * y - 12 = 0) :
    (y = -3 ∧ x = 4) → x + y = 1 :=
by
  intro h'
  obtain ⟨hy, hx⟩ := h'
  rw [hy, hx]
  norm_num
  done

end sum_of_intercepts_l282_28282


namespace total_onions_l282_28249

-- Define the number of onions grown by each individual
def nancy_onions : ℕ := 2
def dan_onions : ℕ := 9
def mike_onions : ℕ := 4

-- Proposition: The total number of onions grown is 15
theorem total_onions : (nancy_onions + dan_onions + mike_onions) = 15 := 
by sorry

end total_onions_l282_28249


namespace find_m_l282_28261

theorem find_m (x y m : ℝ) (hx : x = 1) (hy : y = 2) (h : m * x + 2 * y = 6) : m = 2 :=
by sorry

end find_m_l282_28261


namespace capacity_of_other_bottle_l282_28207

theorem capacity_of_other_bottle (C : ℝ) :
  (∀ (total_milk c1 c2 : ℝ), total_milk = 8 ∧ c1 = 5.333333333333333 ∧ c2 = C ∧ 
  (c1 / 8 = (c2 / C))) → C = 4 :=
by
  intros h
  sorry

end capacity_of_other_bottle_l282_28207


namespace initial_dolphins_l282_28205

variable (D : ℕ)

theorem initial_dolphins (h1 : 3 * D + D = 260) : D = 65 :=
by
  sorry

end initial_dolphins_l282_28205


namespace expression_for_f_in_positive_domain_l282_28232

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def given_f (x : ℝ) : ℝ :=
  if x < 0 then 3 * Real.sin x + 4 * Real.cos x + 1 else 0 -- temp def for Lean proof

theorem expression_for_f_in_positive_domain (f : ℝ → ℝ) (h_odd : is_odd_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = 3 * Real.sin x + 4 * Real.cos x + 1) :
  ∀ x : ℝ, x > 0 → f x = 3 * Real.sin x - 4 * Real.cos x - 1 :=
by
  intros x hx_pos
  sorry

end expression_for_f_in_positive_domain_l282_28232


namespace suzie_reads_pages_hour_l282_28236

-- Declaration of the variables and conditions
variables (S : ℕ) -- S is the number of pages Suzie reads in an hour
variables (L : ℕ) -- L is the number of pages Liza reads in an hour

-- Conditions given in the problem
def reads_per_hour_Liza : L = 20 := sorry
def reads_more_pages : L * 3 = S * 3 + 15 := sorry

-- The statement we want to prove:
theorem suzie_reads_pages_hour : S = 15 :=
by
  -- Proof steps needed here (omitted due to the instruction)
  sorry

end suzie_reads_pages_hour_l282_28236


namespace find_c_l282_28277

def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

theorem find_c (c : ℝ) :
  (∀ x, f x c ≤ f 2 c) → c = 6 :=
sorry

end find_c_l282_28277


namespace kate_retirement_fund_value_l282_28295

theorem kate_retirement_fund_value 
(initial_value decrease final_value : ℝ) 
(h1 : initial_value = 1472)
(h2 : decrease = 12)
(h3 : final_value = initial_value - decrease) : 
final_value = 1460 := 
by
  sorry

end kate_retirement_fund_value_l282_28295


namespace least_possible_lcm_l282_28252

-- Definitions of the least common multiples given the conditions
variable (a b c : ℕ)
variable (h₁ : Nat.lcm a b = 20)
variable (h₂ : Nat.lcm b c = 28)

-- The goal is to prove the least possible value of lcm(a, c) given the conditions
theorem least_possible_lcm (a b c : ℕ) (h₁ : Nat.lcm a b = 20) (h₂ : Nat.lcm b c = 28) : Nat.lcm a c = 35 :=
by
  sorry

end least_possible_lcm_l282_28252


namespace sample_size_l282_28262

-- Define the given conditions
def number_of_male_athletes : Nat := 42
def number_of_female_athletes : Nat := 30
def sampled_female_athletes : Nat := 5

-- Define the target total sample size
def total_sample_size (male_athletes female_athletes sample_females : Nat) : Nat :=
  sample_females * male_athletes / female_athletes + sample_females

-- State the theorem to prove
theorem sample_size (h1: number_of_male_athletes = 42) 
                    (h2: number_of_female_athletes = 30)
                    (h3: sampled_female_athletes = 5) :
  total_sample_size number_of_male_athletes number_of_female_athletes sampled_female_athletes = 12 :=
by
  -- Proof is omitted
  sorry

end sample_size_l282_28262


namespace simplify_expression_l282_28283

-- We need to prove that the simplified expression is equal to the expected form
theorem simplify_expression (y : ℝ) : (3 * y - 7 * y^2 + 4 - (5 + 3 * y - 7 * y^2)) = (0 * y^2 + 0 * y - 1) :=
by
  -- The detailed proof steps will go here
  sorry

end simplify_expression_l282_28283


namespace correct_option_is_B_l282_28271

-- Define the Pythagorean theorem condition for right-angled triangles
def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Conditions given in the problem
def option_A : Prop := ¬is_right_angled_triangle 1 2 2
def option_B : Prop := is_right_angled_triangle 1 (Real.sqrt 3) 2
def option_C : Prop := ¬is_right_angled_triangle 4 5 6
def option_D : Prop := ¬is_right_angled_triangle 1 1 (Real.sqrt 3)

-- The formal proof problem statement
theorem correct_option_is_B : option_A ∧ option_B ∧ option_C ∧ option_D :=
by
  sorry

end correct_option_is_B_l282_28271


namespace tabitha_honey_nights_l282_28244

def servings_per_cup := 1
def cups_per_night := 2
def ounces_per_container := 16
def servings_per_ounce := 6
def total_servings := servings_per_ounce * ounces_per_container
def servings_per_night := servings_per_cup * cups_per_night
def number_of_nights := total_servings / servings_per_night

theorem tabitha_honey_nights : number_of_nights = 48 :=
by
  -- Proof to be provided.
  sorry

end tabitha_honey_nights_l282_28244


namespace general_term_formula_sum_of_geometric_sequence_l282_28240

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = 3

def conditions_1 (a : ℕ → ℤ) : Prop :=
  a 2 + a 4 = 14

-- Definitions for the geometric sequence
def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q, ∀ n, b (n + 1) = b n * q

def conditions_2 (a b : ℕ → ℤ) : Prop := 
  b 2 = a 2 ∧ 
  b 4 = a 6

-- The main theorem statements for part (I) and part (II)
theorem general_term_formula (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : conditions_1 a) : 
  ∀ n, a n = 3 * n - 2 := 
sorry

theorem sum_of_geometric_sequence (a b : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) - a n = 3)
  (h2 : a 2 + a 4 = 14)
  (h3 : b 2 = a 2)
  (h4 : b 4 = a 6)
  (h5 : geometric_sequence b) :
  ∃ (S7 : ℤ), S7 = 254 ∨ S7 = -86 :=
sorry

end general_term_formula_sum_of_geometric_sequence_l282_28240


namespace dance_pairs_exist_l282_28284

variable {Boy Girl : Type} 

-- Define danced_with relation
variable (danced_with : Boy → Girl → Prop)

-- Given conditions
variable (H1 : ∀ (b : Boy), ∃ (g : Girl), ¬ danced_with b g)
variable (H2 : ∀ (g : Girl), ∃ (b : Boy), danced_with b g)

-- Proof that desired pairs exist
theorem dance_pairs_exist :
  ∃ (M1 M2 : Boy) (D1 D2 : Girl),
    danced_with M1 D1 ∧
    danced_with M2 D2 ∧
    ¬ danced_with M1 D2 ∧
    ¬ danced_with M2 D1 :=
sorry

end dance_pairs_exist_l282_28284


namespace total_amount_l282_28267

theorem total_amount {B C : ℝ} 
  (h1 : C = 1600) 
  (h2 : 4 * B = 16 * C) : 
  B + C = 2000 :=
sorry

end total_amount_l282_28267


namespace find_larger_number_l282_28259

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1365) (h2 : y = 4 * x + 15) : y = 1815 :=
sorry

end find_larger_number_l282_28259


namespace rectangle_ratio_l282_28294

theorem rectangle_ratio (s x y : ℝ) 
  (h1 : 4 * (x * y) + s^2 = 9 * s^2)
  (h2 : x + s = 3 * s)
  (h3 : s + 2 * y = 3 * s) :
  x / y = 2 :=
by
  sorry

end rectangle_ratio_l282_28294


namespace ab_value_l282_28274

theorem ab_value (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by sorry

end ab_value_l282_28274


namespace minimize_theta_abs_theta_val_l282_28229

noncomputable def theta (k : ℤ) : ℝ := -11 / 4 * Real.pi + 2 * k * Real.pi

theorem minimize_theta_abs (k : ℤ) :
  ∃ θ : ℝ, (θ = -11 / 4 * Real.pi + 2 * k * Real.pi) ∧
           (∀ η : ℝ, (η = -11 / 4 * Real.pi + 2 * (k + 1) * Real.pi) →
             |θ| ≤ |η|) :=
  sorry

theorem theta_val : ∃ θ : ℝ, θ = -3 / 4 * Real.pi :=
  ⟨ -3 / 4 * Real.pi, rfl ⟩

end minimize_theta_abs_theta_val_l282_28229


namespace initial_number_of_angelfish_l282_28269

/-- The initial number of fish in the tank. -/
def initial_total_fish (A : ℕ) := 94 + A + 89 + 58

/-- The remaining number of fish for each species after sale. -/
def remaining_fish (A : ℕ) := 64 + (A - 48) + 72 + 34

/-- Given: 
1. The total number of remaining fish in the tank is 198.
2. The initial number of fish for each species: 94 guppies, A angelfish, 89 tiger sharks, 58 Oscar fish.
3. The number of fish sold: 30 guppies, 48 angelfish, 17 tiger sharks, 24 Oscar fish.
Prove: The initial number of angelfish is 76. -/
theorem initial_number_of_angelfish (A : ℕ) (h : remaining_fish A = 198) : A = 76 :=
sorry

end initial_number_of_angelfish_l282_28269


namespace tan_alpha_eq_3_l282_28280

theorem tan_alpha_eq_3 (α : ℝ) (h1 : 0 < α ∧ α < (π / 2))
  (h2 : (Real.sin α)^2 + Real.cos ((π / 2) + 2 * α) = 3 / 10) : Real.tan α = 3 := by
  sorry

end tan_alpha_eq_3_l282_28280


namespace moving_circle_passes_focus_l282_28257

noncomputable def parabola (x : ℝ) : Set (ℝ × ℝ) := {p | p.2 ^ 2 = 8 * p.1}
def is_tangent (c : ℝ × ℝ) (r : ℝ) : Prop := c.1 = -2 ∨ c.1 = -2 + 2 * r

theorem moving_circle_passes_focus
  (center : ℝ × ℝ) (H1 : center ∈ parabola center.1)
  (H2 : is_tangent center 2) :
  ∃ focus : ℝ × ℝ, focus = (2, 0) ∧ ∃ r : ℝ, ∀ p ∈ parabola center.1, dist center p = r := sorry

end moving_circle_passes_focus_l282_28257


namespace noah_class_size_l282_28291

theorem noah_class_size :
  ∀ n : ℕ, (n = 39 + 39 + 1) → n = 79 :=
by
  intro n
  intro h
  exact h

end noah_class_size_l282_28291


namespace polygon_sides_from_diagonals_l282_28219

theorem polygon_sides_from_diagonals (n D : ℕ) (h1 : D = 15) (h2 : D = n * (n - 3) / 2) : n = 8 :=
by
  -- skipping proof
  sorry

end polygon_sides_from_diagonals_l282_28219


namespace katherine_fruit_count_l282_28279

variables (apples pears bananas total_fruit : ℕ)

theorem katherine_fruit_count (h1 : apples = 4) 
  (h2 : pears = 3 * apples)
  (h3 : total_fruit = 21) 
  (h4 : total_fruit = apples + pears + bananas) : bananas = 5 := 
by sorry

end katherine_fruit_count_l282_28279


namespace range_of_a_for_min_value_at_x_eq_1_l282_28209

noncomputable def f (a x : ℝ) : ℝ := a*x^3 + (a-1)*x^2 - x + 2

theorem range_of_a_for_min_value_at_x_eq_1 :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a 1 ≤ f a x) → a ≤ 3 / 5 :=
by
  sorry

end range_of_a_for_min_value_at_x_eq_1_l282_28209


namespace farm_cows_l282_28223

theorem farm_cows (c h : ℕ) 
  (legs_eq : 5 * c + 2 * h = 20 + 2 * (c + h)) : 
  c = 6 :=
by 
  sorry

end farm_cows_l282_28223


namespace reciprocal_of_neg_two_thirds_l282_28228

-- Definition for finding the reciprocal
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The proof problem statement
theorem reciprocal_of_neg_two_thirds : reciprocal (-2 / 3) = -3 / 2 :=
sorry

end reciprocal_of_neg_two_thirds_l282_28228


namespace coin_flip_probability_l282_28289

theorem coin_flip_probability :
  let total_outcomes := 2^5
  let successful_outcomes := 2 * 2^2
  total_outcomes > 0 → (successful_outcomes / total_outcomes) = (1 / 4) :=
by
  intros
  sorry

end coin_flip_probability_l282_28289
