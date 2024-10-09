import Mathlib

namespace original_average_weight_l1591_159142

theorem original_average_weight (W : ℝ) (h : (7 * W + 110 + 60) / 9 = 113) : W = 121 :=
by
  sorry

end original_average_weight_l1591_159142


namespace women_in_room_l1591_159132

theorem women_in_room (x q : ℕ) (h1 : 4 * x + 2 = 14) (h2 : q = 2 * (5 * x - 3)) : q = 24 :=
by sorry

end women_in_room_l1591_159132


namespace train_speed_correct_l1591_159163

noncomputable def jogger_speed_km_per_hr := 9
noncomputable def jogger_speed_m_per_s := 9 * 1000 / 3600
noncomputable def train_speed_km_per_hr := 45
noncomputable def distance_ahead_m := 270
noncomputable def train_length_m := 120
noncomputable def total_distance_m := distance_ahead_m + train_length_m
noncomputable def time_seconds := 39

theorem train_speed_correct :
  let relative_speed_m_per_s := total_distance_m / time_seconds
  let train_speed_m_per_s := relative_speed_m_per_s + jogger_speed_m_per_s
  let train_speed_km_per_hr_calculated := train_speed_m_per_s * 3600 / 1000
  train_speed_km_per_hr_calculated = train_speed_km_per_hr :=
by
  sorry

end train_speed_correct_l1591_159163


namespace algebra_expression_solution_l1591_159159

theorem algebra_expression_solution
  (m : ℝ)
  (h : m^2 + m - 1 = 0) :
  m^3 + 2 * m^2 - 2001 = -2000 := by
  sorry

end algebra_expression_solution_l1591_159159


namespace probability_of_winning_second_lawsuit_l1591_159108

theorem probability_of_winning_second_lawsuit
  (P_W1 P_L1 P_W2 P_L2 : ℝ)
  (h1 : P_W1 = 0.30)
  (h2 : P_L1 = 0.70)
  (h3 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20)
  (h4 : P_L2 = 1 - P_W2) :
  P_W2 = 0.50 :=
by
  sorry

end probability_of_winning_second_lawsuit_l1591_159108


namespace factorize_l1591_159146

theorem factorize (m : ℝ) : m^3 - 4 * m = m * (m + 2) * (m - 2) :=
by
  sorry

end factorize_l1591_159146


namespace calc_value_l1591_159119

theorem calc_value : 2 + 3 * 4 - 5 + 6 = 15 := 
by 
  sorry

end calc_value_l1591_159119


namespace W_555_2_last_three_digits_l1591_159170

noncomputable def W : ℕ → ℕ → ℕ
| n, 0     => n ^ n
| n, (k+1) => W (W n k) k

theorem W_555_2_last_three_digits :
  (W 555 2) % 1000 = 875 :=
sorry

end W_555_2_last_three_digits_l1591_159170


namespace length_of_ladder_l1591_159107

theorem length_of_ladder (a b : ℝ) (ha : a = 20) (hb : b = 15) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 25 := by
  sorry

end length_of_ladder_l1591_159107


namespace minimum_other_sales_met_l1591_159168

-- Define the sales percentages for pens, pencils, and the condition for other items
def pens_sales : ℝ := 40
def pencils_sales : ℝ := 28
def minimum_other_sales : ℝ := 20

-- Define the total percentage and calculate the required percentage for other items
def total_sales : ℝ := 100
def required_other_sales : ℝ := total_sales - (pens_sales + pencils_sales)

-- The Lean4 statement to prove the percentage of sales for other items
theorem minimum_other_sales_met 
  (pens_sales_eq : pens_sales = 40)
  (pencils_sales_eq : pencils_sales = 28)
  (total_sales_eq : total_sales = 100)
  (minimum_other_sales_eq : minimum_other_sales = 20)
  (required_other_sales_eq : required_other_sales = total_sales - (pens_sales + pencils_sales)) 
  : required_other_sales = 32 ∧ pens_sales + pencils_sales + required_other_sales = 100 := 
by
  sorry

end minimum_other_sales_met_l1591_159168


namespace XiaoMing_team_award_l1591_159138

def points (x : ℕ) : ℕ := 2 * x + (8 - x)

theorem XiaoMing_team_award (x : ℕ) : 2 * x + (8 - x) ≥ 12 := 
by 
  sorry

end XiaoMing_team_award_l1591_159138


namespace no_tetrahedron_with_given_heights_l1591_159144

theorem no_tetrahedron_with_given_heights (h1 h2 h3 h4 : ℝ) (V : ℝ) (V_pos : V > 0)
    (S1 : ℝ := 3*V) (S2 : ℝ := (3/2)*V) (S3 : ℝ := V) (S4 : ℝ := V/2) :
    (h1 = 1) → (h2 = 2) → (h3 = 3) → (h4 = 6) → ¬ ∃ (S1 S2 S3 S4 : ℝ), S1 < S2 + S3 + S4 := by
  intros
  sorry

end no_tetrahedron_with_given_heights_l1591_159144


namespace regular_polygon_sides_l1591_159181

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 160) : n = 18 :=
by
  sorry

end regular_polygon_sides_l1591_159181


namespace floor_e_is_two_l1591_159179

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_is_two : ⌊e⌋ = 2 := by
  sorry

end floor_e_is_two_l1591_159179


namespace find_n_l1591_159186

theorem find_n (n : ℤ) (h1 : 1 ≤ n) (h2 : n ≤ 9) (h3 : n % 10 = -245 % 10) : n = 5 := 
  sorry

end find_n_l1591_159186


namespace EventB_is_random_l1591_159167

-- Define the events A, B, C, and D as propositions
def EventA : Prop := ∀ (x : ℕ), true -- A coin thrown will fall due to gravity (certain event)
def EventB : Prop := ∃ (n : ℕ), n > 0 -- Hitting the target with a score of 10 points (random event)
def EventC : Prop := ∀ (x : ℕ), true -- The sun rises from the east (certain event)
def EventD : Prop := ∀ (x : ℕ), false -- Horse runs at 70 meters per second (impossible event)

-- Prove that EventB is random, we can use a custom predicate for random events
def is_random_event (e : Prop) : Prop := (∃ (n : ℕ), n > 1) ∧ ¬ ∀ (x : ℕ), e

-- Main statement
theorem EventB_is_random :
  is_random_event EventB :=
by sorry -- The proof will be written here

end EventB_is_random_l1591_159167


namespace min_occupied_seats_l1591_159160

theorem min_occupied_seats (n : ℕ) (h_n : n = 150) : 
  ∃ k : ℕ, k = 37 ∧ ∀ (occupied : Finset ℕ), 
    occupied.card < k → ∃ i : ℕ, i ∉ occupied ∧ ∀ j : ℕ, j ∈ occupied → j + 1 ≠ i ∧ j - 1 ≠ i :=
by
  sorry

end min_occupied_seats_l1591_159160


namespace linear_regression_decrease_l1591_159197

theorem linear_regression_decrease (x : ℝ) (y : ℝ) :
  (h : ∃ c₀ c₁, (c₀ = 2) ∧ (c₁ = -1.5) ∧ y = c₀ - c₁ * x) →
  ( ∃ Δx, Δx = 1 → ∃ Δy, Δy = -1.5) :=
by 
  sorry

end linear_regression_decrease_l1591_159197


namespace original_number_of_professors_l1591_159102

theorem original_number_of_professors (p : ℕ) 
  (h1 : 6480 % p = 0) 
  (h2 : 11200 % (p + 3) = 0) 
  (h3 : 6480 / p < 11200 / (p + 3))
  (h4 : 5 ≤ p) : 
  p = 5 :=
by {
  -- The body of the proof goes here.
  sorry
}

end original_number_of_professors_l1591_159102


namespace at_least_one_lands_l1591_159178

def p : Prop := sorry -- Proposition that Person A lands in the designated area
def q : Prop := sorry -- Proposition that Person B lands in the designated area

theorem at_least_one_lands : p ∨ q := sorry

end at_least_one_lands_l1591_159178


namespace tank_capacity_l1591_159111

theorem tank_capacity (C : ℝ) (h : (3 / 4) * C + 9 = (7 / 8) * C) : C = 72 :=
sorry

end tank_capacity_l1591_159111


namespace alice_sold_20_pears_l1591_159135

-- Definitions (Conditions)
def canned_more_than_poached (C P : ℝ) : Prop := C = P + 0.2 * P
def poached_less_than_sold (P S : ℝ) : Prop := P = 0.5 * S
def total_pears (S C P : ℝ) : Prop := S + C + P = 42

-- Theorem statement
theorem alice_sold_20_pears (S C P : ℝ) (h1 : canned_more_than_poached C P) (h2 : poached_less_than_sold P S) (h3 : total_pears S C P) : S = 20 :=
by 
  -- This is where the proof would go, but for now, we use sorry to signify it's omitted.
  sorry

end alice_sold_20_pears_l1591_159135


namespace verify_incorrect_operation_l1591_159104

theorem verify_incorrect_operation (a : ℝ) :
  ¬ ((-a^2)^3 = -a^5) :=
by
  sorry

end verify_incorrect_operation_l1591_159104


namespace question1_solution_question2_solution_l1591_159194

noncomputable def f (x m : ℝ) : ℝ := x^2 - m * x + m - 1

theorem question1_solution (x : ℝ) :
  ∀ x, f x 3 ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 :=
sorry

theorem question2_solution (m : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f x m ≥ -1) ↔ m ≤ 4 :=
sorry

end question1_solution_question2_solution_l1591_159194


namespace prism_pyramid_sum_l1591_159109

theorem prism_pyramid_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  let new_faces := faces - 1 + 4
  let new_edges := edges + 4
  let new_vertices := vertices + 1
  new_faces + new_edges + new_vertices = 34 :=
by
  sorry

end prism_pyramid_sum_l1591_159109


namespace train_crossing_time_l1591_159120

noncomputable def length_train : ℝ := 250
noncomputable def length_bridge : ℝ := 150
noncomputable def speed_train_kmh : ℝ := 57.6
noncomputable def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

theorem train_crossing_time : 
  let total_length := length_train + length_bridge 
  let time := total_length / speed_train_ms 
  time = 25 := 
by 
  -- Convert all necessary units and parameters
  let length_train := (250 : ℝ)
  let length_bridge := (150 : ℝ)
  let speed_train_ms := (57.6 * (1000 / 3600) : ℝ)
  
  -- Compute the total length and time
  let total_length := length_train + length_bridge
  let time := total_length / speed_train_ms
  
  -- State the proof
  show time = 25
  { sorry }

end train_crossing_time_l1591_159120


namespace player_A_success_l1591_159172

/-- Representation of the problem conditions --/
structure GameState where
  coins : ℕ
  boxes : ℕ
  n_coins : ℕ 
  n_boxes : ℕ 
  arrangement: ℕ → ℕ 
  (h_coins : coins ≥ 2012)
  (h_boxes : boxes = 2012)
  (h_initial_distribution : (∀ b, arrangement b ≥ 1))
  
/-- The main theorem for player A to ensure at least 1 coin in each box --/
theorem player_A_success (s : GameState) : 
  s.coins ≥ 4022 → (∀ b, s.arrangement b ≥ 1) :=
by
  sorry

end player_A_success_l1591_159172


namespace value_of_2alpha_minus_beta_l1591_159195

theorem value_of_2alpha_minus_beta (a β : ℝ) (h1 : 3 * Real.sin a - Real.cos a = 0) 
    (h2 : 7 * Real.sin β + Real.cos β = 0) (h3 : 0 < a ∧ a < Real.pi / 2) 
    (h4 : Real.pi / 2 < β ∧ β < Real.pi) : 
    2 * a - β = -3 * Real.pi / 4 := 
sorry

end value_of_2alpha_minus_beta_l1591_159195


namespace original_cost_l1591_159134

theorem original_cost (C : ℝ) (h : 670 = C + 0.35 * C) : C = 496.30 :=
by
  -- The proof is omitted
  sorry

end original_cost_l1591_159134


namespace points_lie_on_hyperbola_l1591_159124

theorem points_lie_on_hyperbola (s : ℝ) :
  let x := 2 * (Real.exp s + Real.exp (-s))
  let y := 4 * (Real.exp s - Real.exp (-s))
  (x^2) / 16 - (y^2) / 64 = 1 :=
by
  sorry

end points_lie_on_hyperbola_l1591_159124


namespace negation_of_universal_proposition_l1591_159100
open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 :=
by
  sorry

end negation_of_universal_proposition_l1591_159100


namespace find_y_l1591_159143

theorem find_y (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -5) : y = 18 := by
  -- Proof can go here
  sorry

end find_y_l1591_159143


namespace john_days_ran_l1591_159106

theorem john_days_ran 
  (total_distance : ℕ) (daily_distance : ℕ) 
  (h1 : total_distance = 10200) (h2 : daily_distance = 1700) :
  total_distance / daily_distance = 6 :=
by
  sorry

end john_days_ran_l1591_159106


namespace purple_chip_count_l1591_159117

theorem purple_chip_count :
  ∃ (x : ℕ), (x > 5) ∧ (x < 11) ∧
  (∃ (blue green purple red : ℕ),
    (2^6) * (5^2) * 11 * 7 = (blue * 1) * (green * 5) * (purple * x) * (red * 11) ∧ purple = 1) :=
sorry

end purple_chip_count_l1591_159117


namespace find_position_of_2017_l1591_159154

theorem find_position_of_2017 :
  ∃ (row col : ℕ), row = 45 ∧ col = 81 ∧ 2017 = (row - 1)^2 + col :=
by
  sorry

end find_position_of_2017_l1591_159154


namespace part1_part2_l1591_159145

-- Part (1)
theorem part1 (a : ℝ) (P Q : Set ℝ) (hP : P = {x | 4 <= x ∧ x <= 7})
              (hQ : Q = {x | -2 <= x ∧ x <= 5}) :
  (Set.compl P ∩ Q) = {x | -2 <= x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (P Q : Set ℝ)
              (hP : P = {x | a + 1 <= x ∧ x <= 2 * a + 1})
              (hQ : Q = {x | -2 <= x ∧ x <= 5})
              (h_sufficient : ∀ x, x ∈ P → x ∈ Q) 
              (h_not_necessary : ∃ x, x ∈ Q ∧ x ∉ P) :
  (0 <= a ∧ a <= 2) :=
by
  sorry

end part1_part2_l1591_159145


namespace find_initial_cards_l1591_159158

theorem find_initial_cards (B : ℕ) :
  let Tim_initial := 20
  let Sarah_initial := 15
  let Tim_after_give_to_Sarah := Tim_initial - 5
  let Sarah_after_give_to_Sarah := Sarah_initial + 5
  let Tim_after_receive_from_Sarah := Tim_after_give_to_Sarah + 2
  let Sarah_after_receive_from_Sarah := Sarah_after_give_to_Sarah - 2
  let Tim_after_exchange_with_Ben := Tim_after_receive_from_Sarah - 3
  let Ben_after_exchange := B + 13
  let Ben_after_all_transactions := 3 * Tim_after_exchange_with_Ben
  Ben_after_exchange = Ben_after_all_transactions -> B = 29 := by
  sorry

end find_initial_cards_l1591_159158


namespace frac_e_a_l1591_159114

variable (a b c d e : ℚ)

theorem frac_e_a (h1 : a / b = 5) (h2 : b / c = 1 / 4) (h3 : c / d = 7) (h4 : d / e = 1 / 2) :
  e / a = 8 / 35 :=
sorry

end frac_e_a_l1591_159114


namespace sale_price_after_discounts_l1591_159136

/-- The sale price of the television as a percentage of its original price after successive discounts of 25% followed by 10%. -/
theorem sale_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 350 → discount1 = 0.25 → discount2 = 0.10 →
  (original_price * (1 - discount1) * (1 - discount2) / original_price) * 100 = 67.5 :=
by
  intro h_price h_discount1 h_discount2
  sorry

end sale_price_after_discounts_l1591_159136


namespace age_difference_l1591_159125

theorem age_difference :
  ∃ a b : ℕ, (a < 10) ∧ (b < 10) ∧
    (∀ x y : ℕ, (x = 10 * a + b) ∧ (y = 10 * b + a) → 
    (x + 5 = 2 * (y + 5)) ∧ ((10 * a + b) - (10 * b + a) = 18)) :=
by
  sorry

end age_difference_l1591_159125


namespace find_interest_rate_l1591_159150

noncomputable def compoundInterestRate (P A : ℝ) (t : ℕ) : ℝ := 
  ((A / P) ^ (1 / t)) - 1

theorem find_interest_rate :
  ∀ (P A : ℝ) (t : ℕ),
    P = 1200 → 
    A = 1200 + 873.60 →
    t = 3 →
    compoundInterestRate P A t = 0.2 :=
by
  intros P A t hP hA ht
  sorry

end find_interest_rate_l1591_159150


namespace square_division_rectangles_l1591_159118

theorem square_division_rectangles (k l : ℕ) (h_square : exists s : ℝ, 0 < s) 
(segment_division : ∀ (p q : ℝ), exists r : ℕ, r = s * k ∧ r = s * l) :
  ∃ n : ℕ, n = k * l :=
sorry

end square_division_rectangles_l1591_159118


namespace ice_cream_ratio_l1591_159151

theorem ice_cream_ratio :
  ∃ (B C : ℕ), 
    C = 1 ∧
    (∃ (W D : ℕ), 
      D = 2 ∧
      W = B + 1 ∧
      B + W + C + D = 10 ∧
      B / C = 3
    ) := sorry

end ice_cream_ratio_l1591_159151


namespace largest_integer_dividing_consecutive_product_l1591_159174

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l1591_159174


namespace construction_cost_is_correct_l1591_159152

def land_cost (cost_per_sqm : ℕ) (area : ℕ) : ℕ :=
  cost_per_sqm * area

def bricks_cost (cost_per_1000 : ℕ) (quantity : ℕ) : ℕ :=
  (cost_per_1000 * quantity) / 1000

def roof_tiles_cost (cost_per_tile : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_tile * quantity

def cement_bags_cost (cost_per_bag : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_bag * quantity

def wooden_beams_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def steel_bars_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def electrical_wiring_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def plumbing_pipes_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def total_cost : ℕ :=
  land_cost 60 2500 +
  bricks_cost 120 15000 +
  roof_tiles_cost 12 800 +
  cement_bags_cost 8 250 +
  wooden_beams_cost 25 1000 +
  steel_bars_cost 15 500 +
  electrical_wiring_cost 2 2000 +
  plumbing_pipes_cost 4 3000

theorem construction_cost_is_correct : total_cost = 212900 :=
  by
    sorry

end construction_cost_is_correct_l1591_159152


namespace rectangle_area_error_l1591_159113

theorem rectangle_area_error (A B : ℝ) :
  let A' := 1.08 * A
  let B' := 1.08 * B
  let actual_area := A * B
  let measured_area := A' * B'
  let percentage_error := ((measured_area - actual_area) / actual_area) * 100
  percentage_error = 16.64 :=
by
  sorry

end rectangle_area_error_l1591_159113


namespace simplify_expression_l1591_159101

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 3 * (2 - i) + i * (3 + 2 * i) = 4 :=
by
  sorry

end simplify_expression_l1591_159101


namespace expression_evaluation_l1591_159121

theorem expression_evaluation:
  ( (1/3)^2000 * 27^669 + Real.sin (60 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) + (2009 + Real.sin (25 * Real.pi / 180))^0 ) = 
  (2 + 29/54) := by
  sorry

end expression_evaluation_l1591_159121


namespace shaded_area_fraction_l1591_159122

theorem shaded_area_fraction :
  let A := (0, 0)
  let B := (4, 0)
  let C := (4, 4)
  let D := (0, 4)
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let Q := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let R := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let S := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  let area_triangle := 1 / 2 * 2 * 2
  let shaded_area := 2 * area_triangle
  let total_area := 4 * 4
  shaded_area / total_area = 1 / 4 :=
by
  sorry

end shaded_area_fraction_l1591_159122


namespace composite_divisible_by_six_l1591_159149

theorem composite_divisible_by_six (n : ℤ) (h : ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n = a * b) : 6 ∣ (n^4 - n) :=
sorry

end composite_divisible_by_six_l1591_159149


namespace flight_time_is_approximately_50_hours_l1591_159199

noncomputable def flightTime (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  circumference / speed

theorem flight_time_is_approximately_50_hours :
  let radius := 4200
  let speed := 525
  abs (flightTime radius speed - 50) < 1 :=
by
  sorry

end flight_time_is_approximately_50_hours_l1591_159199


namespace max_a_plus_b_l1591_159182

/-- Given real numbers a and b such that 5a + 3b <= 11 and 3a + 6b <= 12,
    the largest possible value of a + b is 23/9. -/
theorem max_a_plus_b (a b : ℝ) (h1 : 5 * a + 3 * b ≤ 11) (h2 : 3 * a + 6 * b ≤ 12) :
  a + b ≤ 23 / 9 :=
sorry

end max_a_plus_b_l1591_159182


namespace contrapositive_equiv_l1591_159162

variable {α : Type}  -- Type of elements
variable (P : Set α) (a b : α)

theorem contrapositive_equiv (h : a ∈ P → b ∉ P) : b ∈ P → a ∉ P :=
by
  sorry

end contrapositive_equiv_l1591_159162


namespace cyclic_quadrilateral_XF_XG_l1591_159180

/-- 
Given:
- A cyclic quadrilateral ABCD inscribed in a circle O,
- Side lengths: AB = 4, BC = 3, CD = 7, DA = 9,
- Points X and Y such that DX/BD = 1/3 and BY/BD = 1/4,
- E is the intersection of line AX and the line through Y parallel to BC,
- F is the intersection of line CX and the line through E parallel to AB,
- G is the other intersection of line CX with circle O,
Prove:
- XF * XG = 36.5.
-/
theorem cyclic_quadrilateral_XF_XG (AB BC CD DA DX BD BY : ℝ) 
  (h_AB : AB = 4) (h_BC : BC = 3) (h_CD : CD = 7) (h_DA : DA = 9)
  (h_ratio1 : DX / BD = 1 / 3) (h_ratio2 : BY / BD = 1 / 4)
  (BD := Real.sqrt 73) :
  ∃ (XF XG : ℝ), XF * XG = 36.5 :=
by
  sorry

end cyclic_quadrilateral_XF_XG_l1591_159180


namespace equal_savings_l1591_159176

theorem equal_savings (A B AE BE AS BS : ℕ) 
  (hA : A = 2000)
  (hA_B : 5 * B = 4 * A)
  (hAE_BE : 3 * BE = 2 * AE)
  (hSavings : AS = A - AE ∧ BS = B - BE ∧ AS = BS) :
  AS = 800 ∧ BS = 800 :=
by
  -- Placeholders for definitions and calculations
  sorry

end equal_savings_l1591_159176


namespace find_b_l1591_159127

def f (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ (b : ℝ), f b = 3 :=
by
  use 2
  show f 2 = 3
  sorry

end find_b_l1591_159127


namespace sum_of_perimeters_l1591_159177

theorem sum_of_perimeters (s : ℝ) : (∀ n : ℕ, n >= 0) → 
  (∑' n : ℕ, (4 * s) / (2 ^ n)) = 8 * s :=
by
  sorry

end sum_of_perimeters_l1591_159177


namespace sum_of_squares_of_roots_eq_213_l1591_159157

theorem sum_of_squares_of_roots_eq_213
  {a b : ℝ}
  (h1 : a + b = 15)
  (h2 : a * b = 6) :
  a^2 + b^2 = 213 :=
by
  sorry

end sum_of_squares_of_roots_eq_213_l1591_159157


namespace no_real_solution_f_of_f_f_eq_x_l1591_159183

-- Defining the quadratic polynomial f(x) = ax^2 + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Stating the main theorem
theorem no_real_solution_f_of_f_f_eq_x (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
by 
  -- Proof will go here
  sorry

end no_real_solution_f_of_f_f_eq_x_l1591_159183


namespace cost_of_each_adult_meal_is_8_l1591_159115

/- Define the basic parameters and conditions -/
def total_people : ℕ := 11
def kids : ℕ := 2
def total_cost : ℕ := 72
def kids_eat_free (k : ℕ) := k = 0

/- The number of adults is derived from the total people minus kids -/
def num_adults : ℕ := total_people - kids

/- The cost per adult meal can be defined and we need to prove it equals to $8 -/
def cost_per_adult (total_cost : ℕ) (num_adults : ℕ) : ℕ := total_cost / num_adults

/- The statement to prove that the cost per adult meal is $8 -/
theorem cost_of_each_adult_meal_is_8 : cost_per_adult total_cost num_adults = 8 := by
  sorry

end cost_of_each_adult_meal_is_8_l1591_159115


namespace part1_relationship_range_part2_maximize_profit_l1591_159123

variables {x y a : ℝ}
noncomputable def zongzi_profit (x : ℝ) : ℝ := -5 * x + 6000

-- Given conditions
def conditions (x : ℝ) : Prop :=
  100 ≤ x ∧ x ≤ 150

-- Part 1: Prove the functional relationship and range of x
theorem part1_relationship_range (x : ℝ) (h : conditions x) :
  zongzi_profit x = -5 * x + 6000 :=
  sorry

-- Part 2: Profit maximization given modified purchase price condition
noncomputable def modified_zongzi_profit (x : ℝ) (a : ℝ) : ℝ :=
  (a - 5) * x + 6000

def maximize_strategy (x a : ℝ) : Prop :=
  (0 < a ∧ a < 5 → x = 100) ∧ (5 ≤ a ∧ a < 10 → x = 150)

theorem part2_maximize_profit (a : ℝ) (ha : 0 < a ∧ a < 10) :
  ∃ x, conditions x ∧ maximize_strategy x a :=
  sorry

end part1_relationship_range_part2_maximize_profit_l1591_159123


namespace find_a_given_coefficient_l1591_159131

theorem find_a_given_coefficient (a : ℝ) :
  (∀ x : ℝ, a ≠ 0 → x ≠ 0 → a^4 * x^4 + 4 * a^3 * x^2 * (1/x) + 6 * a^2 * (1/x)^2 * x^4 + 4 * a * (1/x)^3 * x^6 + (1/x)^4 * x^8 = (ax + 1/x)^4) → (4 * a^3 = 32) → a = 2 :=
by
  intros H1 H2
  sorry

end find_a_given_coefficient_l1591_159131


namespace rate_of_stream_equation_l1591_159161

theorem rate_of_stream_equation 
  (v : ℝ) 
  (boat_speed : ℝ) 
  (travel_time : ℝ) 
  (distance : ℝ)
  (h_boat_speed : boat_speed = 16)
  (h_travel_time : travel_time = 5)
  (h_distance : distance = 105)
  (h_equation : distance = (boat_speed + v) * travel_time) : v = 5 :=
by 
  sorry

end rate_of_stream_equation_l1591_159161


namespace point_on_circle_l1591_159147

theorem point_on_circle (a b : ℝ) 
  (h1 : (b + 2) * x + a * y + 4 = 0) 
  (h2 : a * x + (2 - b) * y - 3 = 0) 
  (parallel_lines : ∀ x y : ℝ, ∀ C1 C2 : ℝ, 
    (b + 2) * x + a * y + C1 = 0 ∧ a * x + (2 - b) * y + C2 = 0 → 
    - (b + 2) / a = - a / (2 - b)
  ) : a^2 + b^2 = 4 :=
sorry

end point_on_circle_l1591_159147


namespace determine_p_l1591_159116

variable (x y z p : ℝ)

theorem determine_p (h1 : 8 / (x + y) = p / (x + z)) (h2 : p / (x + z) = 12 / (z - y)) : p = 20 :=
sorry

end determine_p_l1591_159116


namespace specific_n_values_l1591_159191

theorem specific_n_values (n : ℕ) : 
  ∃ m : ℕ, 
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → m % k = 0) ∧ 
    (m % (n + 1) ≠ 0) ∧ 
    (m % (n + 2) ≠ 0) ∧ 
    (m % (n + 3) ≠ 0) ↔ n = 1 ∨ n = 2 ∨ n = 6 := 
by
  sorry

end specific_n_values_l1591_159191


namespace interest_rate_decrease_l1591_159112

theorem interest_rate_decrease (initial_rate final_rate : ℝ) (x : ℝ) 
  (h_initial_rate : initial_rate = 2.25 * 0.01)
  (h_final_rate : final_rate = 1.98 * 0.01) :
  final_rate = initial_rate * (1 - x)^2 := 
  sorry

end interest_rate_decrease_l1591_159112


namespace find_focus_with_larger_x_coordinate_l1591_159126

noncomputable def focus_of_hyperbola_with_larger_x_coordinate : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 7
  let b := 9
  let c := Real.sqrt (a^2 + b^2)
  (h + c, k)

theorem find_focus_with_larger_x_coordinate :
  focus_of_hyperbola_with_larger_x_coordinate = (5 + Real.sqrt 130, 20) := by
  sorry

end find_focus_with_larger_x_coordinate_l1591_159126


namespace remainder_problem_l1591_159198

theorem remainder_problem (n : ℤ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end remainder_problem_l1591_159198


namespace find_number_of_cups_l1591_159140

theorem find_number_of_cups (a C B : ℝ) (h1 : a * C + 2 * B = 12.75) (h2 : 2 * C + 5 * B = 14.00) (h3 : B = 1.5) : a = 3 :=
by
  sorry

end find_number_of_cups_l1591_159140


namespace inscribed_rectangle_area_correct_l1591_159137

noncomputable def area_of_inscribed_rectangle : Prop := 
  let AD : ℝ := 15 / (12 / (1 / 3) + 3)
  let AB : ℝ := 1 / 3 * AD
  AD * AB = 25 / 12

theorem inscribed_rectangle_area_correct :
  area_of_inscribed_rectangle
  := by
  let hf : ℝ := 12
  let eg : ℝ := 15
  let ad : ℝ := 15 / (hf / (1 / 3) + 3)
  let ab : ℝ := 1 / 3 * ad
  have area : ad * ab = 25 / 12 := by sorry
  exact area

end inscribed_rectangle_area_correct_l1591_159137


namespace radius_of_circle_l1591_159139

theorem radius_of_circle
  (AC BD : ℝ) (h_perpendicular : AC * BD = 0)
  (h_intersect_center : AC / 2 = BD / 2)
  (AB : ℝ) (h_AB : AB = 3)
  (CD : ℝ) (h_CD : CD = 4) :
  (∃ R : ℝ, R = 5 / 2) :=
by
  sorry

end radius_of_circle_l1591_159139


namespace members_playing_badminton_l1591_159165

theorem members_playing_badminton
  (total_members : ℕ := 42)
  (tennis_players : ℕ := 23)
  (neither_players : ℕ := 6)
  (both_players : ℕ := 7) :
  ∃ (badminton_players : ℕ), badminton_players = 20 :=
by
  have union_players := total_members - neither_players
  have badminton_players := union_players - (tennis_players - both_players)
  use badminton_players
  sorry

end members_playing_badminton_l1591_159165


namespace sin_minus_cos_eq_minus_1_l1591_159155

theorem sin_minus_cos_eq_minus_1 (x : ℝ) 
  (h : Real.sin x ^ 3 - Real.cos x ^ 3 = -1) :
  Real.sin x - Real.cos x = -1 := by
  sorry

end sin_minus_cos_eq_minus_1_l1591_159155


namespace sheila_weekly_earnings_l1591_159110

-- Definitions for conditions
def hours_per_day_on_MWF : ℕ := 8
def days_worked_on_MWF : ℕ := 3
def hours_per_day_on_TT : ℕ := 6
def days_worked_on_TT : ℕ := 2
def hourly_rate : ℕ := 10

-- Total weekly hours worked
def total_weekly_hours : ℕ :=
  (hours_per_day_on_MWF * days_worked_on_MWF) + (hours_per_day_on_TT * days_worked_on_TT)

-- Total weekly earnings
def weekly_earnings : ℕ :=
  total_weekly_hours * hourly_rate

-- Lean statement for the proof
theorem sheila_weekly_earnings : weekly_earnings = 360 :=
  sorry

end sheila_weekly_earnings_l1591_159110


namespace sum_of_legs_is_43_l1591_159171

theorem sum_of_legs_is_43 (x : ℕ) (h1 : x * x + (x + 1) * (x + 1) = 31 * 31) :
  x + (x + 1) = 43 :=
sorry

end sum_of_legs_is_43_l1591_159171


namespace students_not_skating_nor_skiing_l1591_159193

theorem students_not_skating_nor_skiing (total_students skating_students skiing_students both_students : ℕ)
  (h_total : total_students = 30)
  (h_skating : skating_students = 20)
  (h_skiing : skiing_students = 9)
  (h_both : both_students = 5) :
  total_students - (skating_students + skiing_students - both_students) = 6 :=
by
  sorry

end students_not_skating_nor_skiing_l1591_159193


namespace solve_trig_eq_l1591_159187

noncomputable def rad (d : ℝ) := d * (Real.pi / 180)

theorem solve_trig_eq (z : ℝ) (k : ℤ) :
  (7 * Real.cos (z) ^ 3 - 6 * Real.cos (z) = 3 * Real.cos (3 * z)) ↔
  (z = rad 90 + k * rad 180 ∨
   z = rad 39.2333 + k * rad 180 ∨
   z = rad 140.7667 + k * rad 180) :=
sorry

end solve_trig_eq_l1591_159187


namespace geometric_monotonic_condition_l1591_159190

-- Definition of a geometrically increasing sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Definition of a monotonically increasing sequence
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

-- The theorem statement
theorem geometric_monotonic_condition (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 1 < a 2 ∧ a 2 < a 3) ↔ monotonically_increasing a :=
sorry

end geometric_monotonic_condition_l1591_159190


namespace tan_of_angle_in_second_quadrant_l1591_159103

theorem tan_of_angle_in_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.cos (π / 2 - α) = 4 / 5) : Real.tan α = -4 / 3 :=
by
  sorry

end tan_of_angle_in_second_quadrant_l1591_159103


namespace total_comics_in_box_l1591_159130

theorem total_comics_in_box 
  (pages_per_comic : ℕ)
  (total_pages_found : ℕ)
  (untorn_comics : ℕ)
  (comics_fixed : ℕ := total_pages_found / pages_per_comic)
  (total_comics : ℕ := comics_fixed + untorn_comics)
  (h_pages_per_comic : pages_per_comic = 25)
  (h_total_pages_found : total_pages_found = 150)
  (h_untorn_comics : untorn_comics = 5) :
  total_comics = 11 :=
by
  sorry

end total_comics_in_box_l1591_159130


namespace problem1_solution_problem2_solution_l1591_159153

-- Problem 1: System of Equations
theorem problem1_solution (x y : ℝ) (h_eq1 : x - y = 2) (h_eq2 : 2 * x + y = 7) : x = 3 ∧ y = 1 :=
by {
  sorry -- Proof to be filled in
}

-- Problem 2: Fractional Equation
theorem problem2_solution (y : ℝ) (h_eq : 3 / (1 - y) = y / (y - 1) - 5) : y = 2 :=
by {
  sorry -- Proof to be filled in
}

end problem1_solution_problem2_solution_l1591_159153


namespace purely_imaginary_has_specific_a_l1591_159184

theorem purely_imaginary_has_specific_a (a : ℝ) :
  (a^2 - 1 + (a - 1 : ℂ) * Complex.I) = (a - 1 : ℂ) * Complex.I → a = -1 := 
by
  sorry

end purely_imaginary_has_specific_a_l1591_159184


namespace sum_of_other_endpoint_coordinates_l1591_159148

theorem sum_of_other_endpoint_coordinates {x y : ℝ} :
  let P1 := (1, 2)
  let M := (5, 6)
  let P2 := (x, y)
  (M.1 = (P1.1 + P2.1) / 2 ∧ M.2 = (P1.2 + P2.2) / 2) → (x + y) = 19 :=
by
  intros P1 M P2 h
  sorry

end sum_of_other_endpoint_coordinates_l1591_159148


namespace find_x_parallel_vectors_l1591_159169

theorem find_x_parallel_vectors
   (x : ℝ)
   (ha : (x, 2) = (x, 2))
   (hb : (-2, 4) = (-2, 4))
   (hparallel : ∀ (k : ℝ), (x, 2) = (k * -2, k * 4)) :
   x = -1 :=
by
  sorry

end find_x_parallel_vectors_l1591_159169


namespace sample_size_l1591_159189

theorem sample_size (f r n : ℕ) (freq_def : f = 36) (rate_def : r = 25 / 100) (relation : r = f / n) : n = 144 :=
sorry

end sample_size_l1591_159189


namespace decreasing_function_range_l1591_159188

theorem decreasing_function_range (f : ℝ → ℝ) (a : ℝ) (h_decreasing : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < 1 → -1 < x2 ∧ x2 < 1 ∧ x1 > x2 → f x1 < f x2)
  (h_ineq: f (1 - a) < f (3 * a - 1)) : 0 < a ∧ a < 1 / 2 :=
by
  sorry

end decreasing_function_range_l1591_159188


namespace total_number_of_tiles_l1591_159164

theorem total_number_of_tiles {s : ℕ} 
  (h1 : ∃ s : ℕ, (s^2 - 4*s + 896 = 0))
  (h2 : 225 = 2*s - 1 + s^2 / 4 - s / 2) :
  s^2 = 1024 := by
  sorry

end total_number_of_tiles_l1591_159164


namespace find_varphi_l1591_159196

theorem find_varphi (ϕ : ℝ) (h1 : 0 < ϕ) (h2 : ϕ < π)
(h_symm : ∃ k : ℤ, ϕ = k * π + 2 * π / 3) :
ϕ = 2 * π / 3 :=
sorry

end find_varphi_l1591_159196


namespace turnip_total_correct_l1591_159166

def turnips_left (melanie benny sarah david m_sold d_sold : ℕ) : ℕ :=
  let melanie_left := melanie - m_sold
  let david_left := david - d_sold
  benny + sarah + melanie_left + david_left

theorem turnip_total_correct :
  turnips_left 139 113 195 87 32 15 = 487 :=
by
  sorry

end turnip_total_correct_l1591_159166


namespace p_p_eq_twenty_l1591_159141

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x ≥ 0 ∧ y < 0 then 4 * x + 2 * y
  else 3 * x + 2 * y

theorem p_p_eq_twenty : p (p 2 (-3)) (p (-3) (-4)) = 20 :=
by
  sorry

end p_p_eq_twenty_l1591_159141


namespace fraction_of_income_from_tips_l1591_159105

variable (S T I : ℝ)

-- Conditions
def tips_as_fraction_of_salary : Prop := T = (3/4) * S
def total_income : Prop := I = S + T

-- Theorem stating the proof problem
theorem fraction_of_income_from_tips 
  (h1 : tips_as_fraction_of_salary S T)
  (h2 : total_income S T I) : (T / I) = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l1591_159105


namespace value_at_x12_l1591_159173

def quadratic_function (d e f x : ℝ) : ℝ :=
  d * x^2 + e * x + f

def axis_of_symmetry (d e f : ℝ) : ℝ := 10.5

def point_on_graph (d e f : ℝ) : Prop :=
  quadratic_function d e f 3 = -5

theorem value_at_x12 (d e f : ℝ)
  (Hsymm : axis_of_symmetry d e f = 10.5)
  (Hpoint : point_on_graph d e f) :
  quadratic_function d e f 12 = -5 :=
sorry

end value_at_x12_l1591_159173


namespace blackboard_final_number_lower_bound_l1591_159185

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def L (c : ℝ) : ℝ := 1 + Real.log c / Real.log phi

theorem blackboard_final_number_lower_bound (c : ℝ) (n : ℕ) (h_pos_c : c > 1) (h_pos_n : n > 0) :
  ∃ x, x ≥ ((c^(n / (L c)) - 1) / (c^(1 / (L c)) - 1))^(L c) :=
sorry

end blackboard_final_number_lower_bound_l1591_159185


namespace calculate_expression_l1591_159129

theorem calculate_expression :
  3 ^ 3 * 2 ^ 2 * 7 ^ 2 * 11 = 58212 :=
by
  sorry

end calculate_expression_l1591_159129


namespace value_of_f_2_pow_100_l1591_159156

def f : ℕ → ℕ :=
sorry

axiom f_base : f 1 = 1
axiom f_recursive : ∀ n : ℕ, f (2 * n) = n * f n

theorem value_of_f_2_pow_100 : f (2^100) = 2^4950 :=
sorry

end value_of_f_2_pow_100_l1591_159156


namespace distance_between_A_and_B_l1591_159128

theorem distance_between_A_and_B (x : ℝ) (boat_speed : ℝ) (flow_speed : ℝ) (dist_AC : ℝ) (total_time : ℝ) :
  (boat_speed = 8) →
  (flow_speed = 2) →
  (dist_AC = 2) →
  (total_time = 3) →
  (x = 10 ∨ x = 12.5) :=
by {
  sorry
}

end distance_between_A_and_B_l1591_159128


namespace roof_area_l1591_159175

theorem roof_area (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : l - w = 28) : 
  l * w = 3136 / 9 := 
by 
  sorry

end roof_area_l1591_159175


namespace student_distribution_l1591_159133

-- Definition to check the number of ways to distribute 7 students into two dormitories A and B
-- with each dormitory having at least 2 students equals 56.
theorem student_distribution (students dorms : Nat) (min_students : Nat) (dist_plans : Nat) :
  students = 7 → dorms = 2 → min_students = 2 → dist_plans = 56 → 
  true := sorry

end student_distribution_l1591_159133


namespace cosine_value_of_angle_between_vectors_l1591_159192

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def cosine_angle (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_value_of_angle_between_vectors :
  cosine_angle a b = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end cosine_value_of_angle_between_vectors_l1591_159192
