import Mathlib

namespace log_inequality_l122_12208

theorem log_inequality (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ 1) (h4 : y ≠ 1) :
    (Real.log y / Real.log x + Real.log x / Real.log y > 2) →
    (x ≠ y ∧ ((x > 1 ∧ y > 1) ∨ (x < 1 ∧ y < 1))) :=
by
    sorry

end log_inequality_l122_12208


namespace second_number_is_correct_l122_12292

theorem second_number_is_correct (A B C : ℝ) 
  (h1 : A + B + C = 157.5)
  (h2 : A / B = 14 / 17)
  (h3 : B / C = 2 / 3)
  (h4 : A - C = 12.75) : 
  B = 18.75 := 
sorry

end second_number_is_correct_l122_12292


namespace find_x_l122_12278

theorem find_x (x : ℝ) (hx : x > 0) (h : Real.sqrt (12*x) * Real.sqrt (5*x) * Real.sqrt (7*x) * Real.sqrt (21*x) = 21) : 
  x = 21 / 97 :=
by
  sorry

end find_x_l122_12278


namespace expand_polynomials_l122_12204

-- Define the given polynomials
def poly1 (x : ℝ) : ℝ := 12 * x^2 + 5 * x - 3
def poly2 (x : ℝ) : ℝ := 3 * x^3 + 2

-- Define the expected result of the polynomial multiplication
def expected (x : ℝ) : ℝ := 36 * x^5 + 15 * x^4 - 9 * x^3 + 24 * x^2 + 10 * x - 6

-- State the theorem
theorem expand_polynomials (x : ℝ) :
  (poly1 x) * (poly2 x) = expected x :=
by
  sorry

end expand_polynomials_l122_12204


namespace incorrect_statement_d_l122_12252

noncomputable def x := Complex.mk (-1/2) (Real.sqrt 3 / 2)
noncomputable def y := Complex.mk (-1/2) (-Real.sqrt 3 / 2)

theorem incorrect_statement_d : (x^12 + y^12) ≠ 1 := by
  sorry

end incorrect_statement_d_l122_12252


namespace number_of_matching_pages_l122_12254

theorem number_of_matching_pages : 
  ∃ (n : Nat), n = 13 ∧ ∀ x, 1 ≤ x ∧ x ≤ 63 → (x % 10 = (64 - x) % 10) ↔ x % 10 = 2 ∨ x % 10 = 7 :=
by
  sorry

end number_of_matching_pages_l122_12254


namespace xiao_ming_correct_answers_l122_12220

theorem xiao_ming_correct_answers :
  ∃ (m n : ℕ), m + n = 20 ∧ 5 * m - n = 76 ∧ m = 16 := 
by
  -- Definitions of points for correct and wrong answers
  let points_per_correct := 5 
  let points_deducted_per_wrong := 1

  -- Contestant's Scores and Conditions
  have contestant_a : 20 * points_per_correct - 0 * points_deducted_per_wrong = 100 := by sorry
  have contestant_b : 19 * points_per_correct - 1 * points_deducted_per_wrong = 94 := by sorry
  have contestant_c : 18 * points_per_correct - 2 * points_deducted_per_wrong = 88 := by sorry
  have contestant_d : 14 * points_per_correct - 6 * points_deducted_per_wrong = 64 := by sorry
  have contestant_e : 10 * points_per_correct - 10 * points_deducted_per_wrong = 40 := by sorry

  -- Xiao Ming's conditions translated to variables m and n
  have xiao_ming_conditions : (∃ m n : ℕ, m + n = 20 ∧ 5 * m - n = 76) := by sorry

  exact ⟨16, 4, rfl, rfl, rfl⟩

end xiao_ming_correct_answers_l122_12220


namespace cost_price_of_watch_l122_12200

theorem cost_price_of_watch
  (C : ℝ)
  (h1 : 0.9 * C + 225 = 1.05 * C) :
  C = 1500 :=
by sorry

end cost_price_of_watch_l122_12200


namespace find_x_l122_12274

theorem find_x (x : ℝ) (h : 0.25 * x = 0.10 * 500 - 5) : x = 180 :=
by
  sorry

end find_x_l122_12274


namespace find_natural_triples_l122_12227

open Nat

noncomputable def satisfies_conditions (a b c : ℕ) : Prop :=
  (a + b) % c = 0 ∧ (b + c) % a = 0 ∧ (c + a) % b = 0

theorem find_natural_triples :
  ∀ (a b c : ℕ), satisfies_conditions a b c ↔
    (∃ a, (a = b ∧ b = c) ∨ 
          (a = b ∧ c = 2 * a) ∨ 
          (b = 2 * a ∧ c = 3 * a) ∨ 
          (b = 3 * a ∧ c = 2 * a) ∨ 
          (a = 2 * b ∧ c = 3 * b) ∨ 
          (a = 3 * b ∧ c = 2 * b)) :=
sorry

end find_natural_triples_l122_12227


namespace shifted_parabola_is_correct_l122_12251

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  -((x - 1) ^ 2) + 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ :=
  -((x + 1 - 1) ^ 2) + 4

-- State the theorem
theorem shifted_parabola_is_correct :
  ∀ x : ℝ, shifted_parabola x = -x^2 + 4 :=
by
  -- Proof would go here
  sorry

end shifted_parabola_is_correct_l122_12251


namespace denise_spent_l122_12296

theorem denise_spent (price_simple : ℕ) (price_meat : ℕ) (price_fish : ℕ)
  (price_milk_smoothie : ℕ) (price_fruit_smoothie : ℕ) (price_special_smoothie : ℕ)
  (julio_spent_more : ℕ) :
  price_simple = 7 →
  price_meat = 11 →
  price_fish = 14 →
  price_milk_smoothie = 6 →
  price_fruit_smoothie = 7 →
  price_special_smoothie = 9 →
  julio_spent_more = 6 →
  ∃ (d_price : ℕ), (d_price = 14 ∨ d_price = 17) :=
by
  sorry

end denise_spent_l122_12296


namespace first_place_beat_joe_l122_12271

theorem first_place_beat_joe (joe_won joe_draw first_place_won first_place_draw points_win points_draw : ℕ) 
    (h1 : joe_won = 1) (h2 : joe_draw = 3) (h3 : first_place_won = 2) (h4 : first_place_draw = 2)
    (h5 : points_win = 3) (h6 : points_draw = 1) : 
    (first_place_won * points_win + first_place_draw * points_draw) - (joe_won * points_win + joe_draw * points_draw) = 2 :=
by
   sorry

end first_place_beat_joe_l122_12271


namespace maciek_total_cost_l122_12247

theorem maciek_total_cost :
  let p := 4
  let cost_of_chips := 1.75 * p
  let pretzels_cost := 2 * p
  let chips_cost := 2 * cost_of_chips
  let t := pretzels_cost + chips_cost
  t = 22 :=
by
  sorry

end maciek_total_cost_l122_12247


namespace negation_of_proposition_l122_12253

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 1 < x → (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) > 4)) ↔
  (∃ x : ℝ, 1 < x ∧ (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) ≤ 4) :=
sorry

end negation_of_proposition_l122_12253


namespace tamia_bell_pepper_pieces_l122_12235

theorem tamia_bell_pepper_pieces :
  let bell_peppers := 5
  let slices_per_pepper := 20
  let initial_slices := bell_peppers * slices_per_pepper
  let half_slices_cut := initial_slices / 2
  let small_pieces := half_slices_cut * 3
  let total_pieces := (initial_slices - half_slices_cut) + small_pieces
  total_pieces = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l122_12235


namespace set_inter_complement_l122_12212

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem set_inter_complement :
  U = {1, 2, 3, 4, 5, 6, 7} ∧ A = {1, 2, 3, 4} ∧ B = {3, 5, 6} →
  A ∩ (U \ B) = {1, 2, 4} :=
by
  sorry

end set_inter_complement_l122_12212


namespace number_of_answer_choices_l122_12279

theorem number_of_answer_choices (n : ℕ) (H1 : (n + 1)^4 = 625) : n = 4 :=
sorry

end number_of_answer_choices_l122_12279


namespace find_a_l122_12214

theorem find_a (a k : ℝ) (h1 : ∀ x, a * x^2 + 3 * x - k = 0 → x = 7) (h2 : k = 119) : a = 2 :=
by
  sorry

end find_a_l122_12214


namespace sarah_min_days_l122_12289

theorem sarah_min_days (r P B : ℝ) (x : ℕ) (h_r : r = 0.1) (h_P : P = 20) (h_B : B = 60) :
  (P + r * P * x ≥ B) → (x ≥ 20) :=
by
  sorry

end sarah_min_days_l122_12289


namespace production_time_l122_12231

-- Define the conditions
def machineProductionRate (machines: ℕ) (units: ℕ) (hours: ℕ): ℕ := units / machines / hours

-- The question we need to answer: How long will it take for 10 machines to produce 100 units?
theorem production_time (h1 : machineProductionRate 5 20 10 = 4 / 10)
  : 10 * 0.4 * 25 = 100 :=
by sorry

end production_time_l122_12231


namespace twelve_times_reciprocal_sum_l122_12249

theorem twelve_times_reciprocal_sum (a b c : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/4) (h₃ : c = 1/6) :
  12 * (a + b + c)⁻¹ = 16 := 
by
  sorry

end twelve_times_reciprocal_sum_l122_12249


namespace main_l122_12236

def prop_p (x0 : ℝ) : Prop := x0 > -2 ∧ 6 + abs x0 = 5
def p : Prop := ∃ x : ℝ, prop_p x

def q : Prop := ∀ x : ℝ, x < 0 → x^2 + 4 / x^2 ≥ 4

def r : Prop := ∀ x y : ℝ, abs x + abs y ≤ 1 → abs y / (abs x + 2) ≤ 1 / 2
def not_r : Prop := ∃ x y : ℝ, abs x + abs y > 1 ∧ abs y / (abs x + 2) > 1 / 2

theorem main : ¬ p ∧ ¬ p ∨ r ∧ (p ∧ q) := by
  sorry

end main_l122_12236


namespace sqrt_meaningful_real_domain_l122_12219

theorem sqrt_meaningful_real_domain (x : ℝ) (h : 6 - 4 * x ≥ 0) : x ≤ 3 / 2 :=
by sorry

end sqrt_meaningful_real_domain_l122_12219


namespace ny_mets_fans_count_l122_12210

-- Define the known ratios and total fans
def ratio_Y_to_M (Y M : ℕ) : Prop := 3 * M = 2 * Y
def ratio_M_to_R (M R : ℕ) : Prop := 4 * R = 5 * M
def total_fans (Y M R : ℕ) : Prop := Y + M + R = 330

-- Define what we want to prove
theorem ny_mets_fans_count (Y M R : ℕ) (h1 : ratio_Y_to_M Y M) (h2 : ratio_M_to_R M R) (h3 : total_fans Y M R) : M = 88 :=
sorry

end ny_mets_fans_count_l122_12210


namespace maria_earnings_l122_12240

def cost_of_brushes : ℕ := 20
def cost_of_canvas : ℕ := 3 * cost_of_brushes
def cost_per_liter_of_paint : ℕ := 8
def liters_of_paint : ℕ := 5
def cost_of_paint : ℕ := liters_of_paint * cost_per_liter_of_paint
def total_cost : ℕ := cost_of_brushes + cost_of_canvas + cost_of_paint
def selling_price : ℕ := 200

theorem maria_earnings : (selling_price - total_cost) = 80 := by
  sorry

end maria_earnings_l122_12240


namespace remainder_when_added_then_divided_l122_12259

def num1 : ℕ := 2058167
def num2 : ℕ := 934
def divisor : ℕ := 8

theorem remainder_when_added_then_divided :
  (num1 + num2) % divisor = 5 := 
sorry

end remainder_when_added_then_divided_l122_12259


namespace intersection_A_B_l122_12209

def A : Set ℤ := {-2, 0, 1, 2}
def B : Set ℤ := { x | -2 ≤ x ∧ x ≤ 1 }

theorem intersection_A_B : A ∩ B = {-2, 0, 1} := by
  sorry

end intersection_A_B_l122_12209


namespace shortest_path_correct_l122_12238

noncomputable def shortest_path_length (length width height : ℕ) : ℝ :=
  let diagonal := Real.sqrt ((length + height)^2 + width^2)
  Real.sqrt 145

theorem shortest_path_correct :
  ∀ (length width height : ℕ),
    length = 4 → width = 5 → height = 4 →
    shortest_path_length length width height = Real.sqrt 145 :=
by
  intros length width height h1 h2 h3
  rw [h1, h2, h3]
  sorry

end shortest_path_correct_l122_12238


namespace isosceles_triangle_construction_l122_12223

noncomputable def isosceles_triangle_construction_impossible 
  (hb lb : ℝ) : Prop :=
  ∀ (α β : ℝ), 
  3 * β ≠ α

theorem isosceles_triangle_construction : 
  ∃ (hb lb : ℝ), isosceles_triangle_construction_impossible hb lb :=
sorry

end isosceles_triangle_construction_l122_12223


namespace players_quit_game_l122_12288

variable (total_players initial num_lives players_left players_quit : Nat)
variable (each_player_lives : Nat)

theorem players_quit_game :
  (initial = 8) →
  (each_player_lives = 3) →
  (num_lives = 15) →
  players_left = num_lives / each_player_lives →
  players_quit = initial - players_left →
  players_quit = 3 :=
by
  intros h_initial h_each_player_lives h_num_lives h_players_left h_players_quit
  sorry

end players_quit_game_l122_12288


namespace range_of_m_l122_12216

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define A as the set of real numbers satisfying 2x^2 - x = 0
def A : Set ℝ := {x | 2 * x^2 - x = 0}

-- Define B based on the parameter m as the set of real numbers satisfying mx^2 - mx - 1 = 0
def B (m : ℝ) : Set ℝ := {x | m * x^2 - m * x - 1 = 0}

-- Define the condition (¬U A) ∩ B = ∅
def condition (m : ℝ) : Prop := (U \ A) ∩ B m = ∅

theorem range_of_m : ∀ m : ℝ, condition m → -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end range_of_m_l122_12216


namespace spinner_probability_l122_12237

-- Define the game board conditions
def total_regions : ℕ := 12  -- The triangle is divided into 12 smaller regions
def shaded_regions : ℕ := 3  -- Three regions are shaded

-- Define the probability calculation
def probability (total : ℕ) (shaded : ℕ): ℚ := shaded / total

-- State the proof problem
theorem spinner_probability :
  probability total_regions shaded_regions = 1 / 4 :=
by
  sorry

end spinner_probability_l122_12237


namespace length_of_single_row_l122_12272

-- Define smaller cube properties and larger cube properties
def side_length_smaller_cube : ℕ := 5  -- in cm
def side_length_larger_cube : ℕ := 100  -- converted from 1 meter to cm

-- Prove that the row of smaller cubes is 400 meters long
theorem length_of_single_row :
  let num_smaller_cubes := (side_length_larger_cube / side_length_smaller_cube) ^ 3
  let length_in_cm := num_smaller_cubes * side_length_smaller_cube
  let length_in_m := length_in_cm / 100
  length_in_m = 400 :=
by
  sorry

end length_of_single_row_l122_12272


namespace area_of_triangle_HFG_l122_12293

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

end area_of_triangle_HFG_l122_12293


namespace ab_sum_l122_12263

theorem ab_sum (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 7) (h3 : |a - b| = b - a) : a + b = 10 ∨ a + b = 4 :=
by
  sorry

end ab_sum_l122_12263


namespace exists_acute_triangle_l122_12245

-- Define the segments as a list of positive real numbers
variables (a b c d e : ℝ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0) (h3 : d > 0) (h4 : e > 0)
(h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e)

-- Conditions: Any three segments can form a triangle
variables (h_triangle_1 : a + b > c ∧ a + c > b ∧ b + c > a)
variables (h_triangle_2 : a + b > d ∧ a + d > b ∧ b + d > a)
variables (h_triangle_3 : a + b > e ∧ a + e > b ∧ b + e > a)
variables (h_triangle_4 : a + c > d ∧ a + d > c ∧ c + d > a)
variables (h_triangle_5 : a + c > e ∧ a + e > c ∧ c + e > a)
variables (h_triangle_6 : a + d > e ∧ a + e > d ∧ d + e > a)
variables (h_triangle_7 : b + c > d ∧ b + d > c ∧ c + d > b)
variables (h_triangle_8 : b + c > e ∧ b + e > c ∧ c + e > b)
variables (h_triangle_9 : b + d > e ∧ b + e > d ∧ d + e > b)
variables (h_triangle_10 : c + d > e ∧ c + e > d ∧ d + e > c)

-- Prove that there exists an acute-angled triangle 
theorem exists_acute_triangle : ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                                        (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                                        (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
                                        x + y > z ∧ x + z > y ∧ y + z > x ∧ 
                                        x^2 < y^2 + z^2 := 
sorry

end exists_acute_triangle_l122_12245


namespace problem_part1_problem_part2_l122_12291

noncomputable def f (a : ℝ) (x : ℝ) := 2 * Real.log x + a / x
noncomputable def g (a : ℝ) (x : ℝ) := (x / 2) * f a x - a * x^2 - x

theorem problem_part1 (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x > 0) ↔ 0 < a ∧ a < 2/Real.exp 1 := sorry

theorem problem_part2 (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : g a x₁ = 0) (h₃ : g a x₂ = 0) :
  0 < a ∧ a < 2/Real.exp 1 → Real.log x₁ + 2 * Real.log x₂ > 3 := sorry

end problem_part1_problem_part2_l122_12291


namespace solve_for_x_l122_12257

-- Lean 4 statement for the problem
theorem solve_for_x (x : ℝ) (h : (x + 1)^3 = -27) : x = -4 := by
  sorry

end solve_for_x_l122_12257


namespace diff_of_squares_not_2018_l122_12294

theorem diff_of_squares_not_2018 (a b : ℕ) (h : a > b) : ¬(a^2 - b^2 = 2018) :=
by {
  -- proof goes here
  sorry
}

end diff_of_squares_not_2018_l122_12294


namespace problem_1_problem_2_l122_12299

noncomputable def f (x : ℝ) (a : ℝ) := Real.sqrt (a - x^2)

-- First proof problem statement: 
theorem problem_1 (a : ℝ) (x : ℝ) (A B : Set ℝ) (h1 : a = 4) (h2 : A = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) (h3 : B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) : 
  (A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) :=
sorry

-- Second proof problem statement:
theorem problem_2 (a : ℝ) (h : 1 ∈ {y : ℝ | 0 ≤ y ∧ y ≤ Real.sqrt a}) : a ≥ 1 :=
sorry

end problem_1_problem_2_l122_12299


namespace all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l122_12290

-- Conditions
def investment := 13500  -- in yuan
def total_yield := 19000 -- in kg
def price_orchard := 4   -- in yuan/kg
def price_market (x : ℝ) := x -- in yuan/kg
def market_daily_sale := 1000 -- in kg/day

-- Part 1: Days to sell all fruits in the market
theorem all_fruits_sold_in_market (x : ℝ) (h : x > 4) : total_yield / market_daily_sale = 19 :=
by
  sorry

-- Part 2: Income difference between market and orchard sales
theorem market_vs_orchard_income_diff (x : ℝ) (h : x > 4) : total_yield * price_market x - total_yield * price_orchard = 19000 * x - 76000 :=
by
  sorry

-- Part 3: Total profit from selling partly in the orchard and partly in the market
theorem total_profit (x : ℝ) (h : x > 4) : 6000 * price_orchard + (total_yield - 6000) * price_market x - investment = 13000 * x + 10500 :=
by
  sorry

end all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l122_12290


namespace age_difference_l122_12265

variable (Patrick_age Michael_age Monica_age : ℕ)

theorem age_difference 
  (h1 : ∃ x : ℕ, Patrick_age = 3 * x ∧ Michael_age = 5 * x)
  (h2 : ∃ y : ℕ, Michael_age = 3 * y ∧ Monica_age = 5 * y)
  (h3 : Patrick_age + Michael_age + Monica_age = 245) :
  Monica_age - Patrick_age = 80 := by 
sorry

end age_difference_l122_12265


namespace find_coefficient_b_l122_12213

variable (a b c p : ℝ)

def parabola (x : ℝ) := a * x^2 + b * x + c

theorem find_coefficient_b (h_vertex : ∀ x, parabola a b c x = a * (x - p)^2 + p)
                           (h_y_intercept : parabola a b c 0 = -3 * p)
                           (hp_nonzero : p ≠ 0) :
  b = 8 / p :=
by
  sorry

end find_coefficient_b_l122_12213


namespace parabola_intersection_sum_l122_12270

theorem parabola_intersection_sum : 
  ∃ x_0 y_0 : ℝ, (y_0 = x_0^2 + 15 * x_0 + 32) ∧ (x_0 = y_0^2 + 49 * y_0 + 593) ∧ (x_0 + y_0 = -33) :=
by
  sorry

end parabola_intersection_sum_l122_12270


namespace weekly_crab_meat_cost_l122_12211

-- Declare conditions as definitions
def dishes_per_day : ℕ := 40
def pounds_per_dish : ℝ := 1.5
def cost_per_pound : ℝ := 8
def closed_days_per_week : ℕ := 3
def days_per_week : ℕ := 7

-- Define the Lean statement to prove the weekly cost
theorem weekly_crab_meat_cost :
  let days_open_per_week := days_per_week - closed_days_per_week
  let pounds_per_day := dishes_per_day * pounds_per_dish
  let daily_cost := pounds_per_day * cost_per_pound
  let weekly_cost := daily_cost * (days_open_per_week : ℝ)
  weekly_cost = 1920 :=
by
  sorry

end weekly_crab_meat_cost_l122_12211


namespace grazing_time_for_36_cows_l122_12234

-- Defining the problem conditions and the question in Lean 4
theorem grazing_time_for_36_cows :
  ∀ (g r b : ℕ), 
    (24 * 6 * b = g + 6 * r) →
    (21 * 8 * b = g + 8 * r) →
    36 * 3 * b = g + 3 * r :=
by
  intros
  sorry

end grazing_time_for_36_cows_l122_12234


namespace highest_a_value_l122_12201

theorem highest_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 143) : a = 23 :=
sorry

end highest_a_value_l122_12201


namespace find_multiple_l122_12222

theorem find_multiple (x m : ℕ) (h₁ : x = 69) (h₂ : x - 18 = m * (86 - x)) : m = 3 :=
by
  sorry

end find_multiple_l122_12222


namespace g_sum_even_function_l122_12256

def g (a b c d x : ℝ) : ℝ := a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + d * x ^ 2 + 5

theorem g_sum_even_function 
  (a b c d : ℝ) 
  (h : g a b c d 2 = 4)
  : g a b c d 2 + g a b c d (-2) = 8 :=
by
  sorry

end g_sum_even_function_l122_12256


namespace find_k_of_collinear_points_l122_12243

theorem find_k_of_collinear_points :
  ∃ k : ℚ, ∀ (x1 y1 x2 y2 x3 y3 : ℚ), (x1, y1) = (4, 10) → (x2, y2) = (-3, k) → (x3, y3) = (-8, 5) → 
  ((y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)) → k = 85 / 12 :=
by
  sorry

end find_k_of_collinear_points_l122_12243


namespace find_ratio_l122_12261

-- Definitions and conditions
def sides_form_right_triangle (x d : ℝ) : Prop :=
  x > d ∧ d > 0 ∧ (x^2 + (x^2 - d)^2 = (x^2 + d)^2)

-- The theorem stating the required ratio
theorem find_ratio (x d : ℝ) (h : sides_form_right_triangle x d) : 
  x / d = 8 :=
by
  sorry

end find_ratio_l122_12261


namespace grabbed_books_l122_12283

-- Definitions from conditions
def initial_books : ℕ := 99
def boxed_books : ℕ := 3 * 15
def room_books : ℕ := 21
def table_books : ℕ := 4
def kitchen_books : ℕ := 18
def current_books : ℕ := 23

-- Proof statement
theorem grabbed_books : (boxed_books + room_books + table_books + kitchen_books = initial_books - (23 - current_books)) → true := sorry

end grabbed_books_l122_12283


namespace find_first_hour_speed_l122_12275

variable (x : ℝ)

-- Conditions
def speed_second_hour : ℝ := 60
def average_speed_two_hours : ℝ := 102.5

theorem find_first_hour_speed (h1 : average_speed_two_hours = (x + speed_second_hour) / 2) : 
  x = 145 := 
by
  sorry

end find_first_hour_speed_l122_12275


namespace speed_in_still_water_l122_12277

/-- A man can row upstream at 37 km/h and downstream at 53 km/h, 
    prove that the speed of the man in still water is 45 km/h. --/
theorem speed_in_still_water 
  (upstream_speed : ℕ) 
  (downstream_speed : ℕ)
  (h1 : upstream_speed = 37)
  (h2 : downstream_speed = 53) : 
  (upstream_speed + downstream_speed) / 2 = 45 := 
by 
  sorry

end speed_in_still_water_l122_12277


namespace gcd_of_11121_and_12012_l122_12225

def gcd_problem : Prop :=
  gcd 11121 12012 = 1

theorem gcd_of_11121_and_12012 : gcd_problem :=
by
  -- Proof omitted
  sorry

end gcd_of_11121_and_12012_l122_12225


namespace max_value_f_l122_12276

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x - Real.tan x

theorem max_value_f : 
  ∃ x ∈ Set.Ioo 0 (Real.pi / 2), ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≤ f x ∧ f x = 3 * Real.sqrt 3 :=
by
  sorry

end max_value_f_l122_12276


namespace expression_evaluation_l122_12268

theorem expression_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
sorry

end expression_evaluation_l122_12268


namespace initial_earning_members_l122_12260

theorem initial_earning_members (average_income_before: ℝ) (average_income_after: ℝ) (income_deceased: ℝ) (n: ℝ)
    (H1: average_income_before = 735)
    (H2: average_income_after = 650)
    (H3: income_deceased = 990)
    (H4: n * average_income_before - (n - 1) * average_income_after = income_deceased)
    : n = 4 := 
by 
  rw [H1, H2, H3] at H4
  linarith


end initial_earning_members_l122_12260


namespace decreasing_function_positive_l122_12282

variable {f : ℝ → ℝ}

axiom decreasing (h : ℝ → ℝ) : ∀ x1 x2, x1 < x2 → h x1 > h x2

theorem decreasing_function_positive (h_decreasing: ∀ x1 x2: ℝ, x1 < x2 → f x1 > f x2)
    (h_condition: ∀ x: ℝ, f x / (deriv^[2] f x) + x < 1) :
  ∀ x : ℝ, f x > 0 := 
by
  sorry

end decreasing_function_positive_l122_12282


namespace interest_rate_is_4_l122_12287

-- Define the conditions based on the problem statement
def principal : ℕ := 500
def time : ℕ := 8
def simple_interest : ℕ := 160

-- Assuming the formula for simple interest
def simple_interest_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- The interest rate we aim to prove
def interest_rate : ℕ := 4

-- The statement we want to prove: Given the conditions, the interest rate is 4%
theorem interest_rate_is_4 : simple_interest_formula principal interest_rate time = simple_interest := by
  -- The proof steps would go here
  sorry

end interest_rate_is_4_l122_12287


namespace spending_on_clothes_transport_per_month_l122_12218

noncomputable def monthly_spending_on_clothes_transport (S : ℝ) : ℝ :=
  0.2 * S

theorem spending_on_clothes_transport_per_month :
  ∃ (S : ℝ), (monthly_spending_on_clothes_transport S = 1584) ∧
             (12 * S - (12 * 0.6 * S + 12 * monthly_spending_on_clothes_transport S) = 19008) :=
by
  sorry

end spending_on_clothes_transport_per_month_l122_12218


namespace total_oranges_in_buckets_l122_12286

theorem total_oranges_in_buckets (a b c : ℕ) 
  (h1 : a = 22) 
  (h2 : b = a + 17) 
  (h3 : c = b - 11) : 
  a + b + c = 89 := 
by {
  sorry
}

end total_oranges_in_buckets_l122_12286


namespace fill_tank_with_two_pipes_l122_12258

def Pipe (Rate : Type) := Rate

theorem fill_tank_with_two_pipes
  (capacity : ℝ)
  (three_pipes_fill_time : ℝ)
  (h1 : three_pipes_fill_time = 12)
  (pipe_rate : ℝ)
  (h2 : pipe_rate = capacity / 36) :
  2 * pipe_rate * 18 = capacity := 
by 
  sorry

end fill_tank_with_two_pipes_l122_12258


namespace sum_of_absolute_values_of_coefficients_l122_12233

theorem sum_of_absolute_values_of_coefficients :
  ∀ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 - 3 * x) ^ 9 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9) →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 4 ^ 9 :=
by
  intro a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 h
  sorry

end sum_of_absolute_values_of_coefficients_l122_12233


namespace ratio_a3_b3_l122_12228

theorem ratio_a3_b3 (a : ℝ) (ha : a ≠ 0)
  (h1 : a = b₁)
  (h2 : a * q * b = 2)
  (h3 : b₄ = 8 * a * q^3) :
  (∃ r : ℝ, r = -5 ∨ r = -3.2) :=
by
  sorry

end ratio_a3_b3_l122_12228


namespace part1_a2_part1_a3_part2_general_formula_l122_12221

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| n + 1 => (n + 1) * n / 2

noncomputable def S (n : ℕ) : ℚ := (n + 2) * a n / 3

theorem part1_a2 : a 2 = 3 := sorry

theorem part1_a3 : a 3 = 6 := sorry

theorem part2_general_formula (n : ℕ) (h : n > 0) : a n = n * (n + 1) / 2 := sorry

end part1_a2_part1_a3_part2_general_formula_l122_12221


namespace graph_location_l122_12206

theorem graph_location (k : ℝ) (H : k > 0) :
    (∀ x : ℝ, (0 < x → 0 < y) → (y = 2/x) → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
    sorry

end graph_location_l122_12206


namespace yuna_has_biggest_number_l122_12241

-- Define the collections
def yoongi_collected : ℕ := 4
def jungkook_collected : ℕ := 6 - 3
def yuna_collected : ℕ := 5

-- State the theorem
theorem yuna_has_biggest_number :
  yuna_collected > yoongi_collected ∧ yuna_collected > jungkook_collected :=
by
  sorry

end yuna_has_biggest_number_l122_12241


namespace intersection_M_N_l122_12242

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 + 2 * x - 3 ≤ 0}
def intersection : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l122_12242


namespace find_pairs_l122_12230

theorem find_pairs (x y : ℕ) (h1 : 0 < x ∧ 0 < y)
  (h2 : ∃ p : ℕ, Prime p ∧ (x + y = 2 * p))
  (h3 : (x! + y!) % (x + y) = 0) : ∃ p : ℕ, Prime p ∧ x = p ∧ y = p :=
by
  sorry

end find_pairs_l122_12230


namespace find_divisor_l122_12266

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h_dividend : dividend = 125) 
  (h_quotient : quotient = 8) 
  (h_remainder : remainder = 5) 
  (h_equation : dividend = (divisor * quotient) + remainder) : 
  divisor = 15 := by
  sorry

end find_divisor_l122_12266


namespace intersection_at_one_point_l122_12202

-- Define the quadratic equation derived from the intersection condition
def quadratic (y k : ℝ) : ℝ :=
  3 * y^2 - 2 * y + (k - 4)

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ :=
  (-2)^2 - 4 * 3 * (k - 4)

-- The statement of the problem in Lean
theorem intersection_at_one_point (k : ℝ) :
  (∃ y : ℝ, quadratic y k = 0 ∧ discriminant k = 0) ↔ k = 13 / 3 :=
by 
  sorry

end intersection_at_one_point_l122_12202


namespace find_m_parallel_l122_12269

def vector_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k • v) ∨ v = (k • u)

theorem find_m_parallel (m : ℝ) (a b : ℝ × ℝ) (h_a : a = (-1, 1)) (h_b : b = (3, m)) 
  (h_parallel : vector_parallel a (a.1 + b.1, a.2 + b.2)) : m = -3 := 
by 
  sorry

end find_m_parallel_l122_12269


namespace sum_of_n_values_l122_12267

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l122_12267


namespace math_problem_l122_12281

theorem math_problem :
  3 ^ (2 + 4 + 6) - (3 ^ 2 + 3 ^ 4 + 3 ^ 6) + (3 ^ 2 * 3 ^ 4 * 3 ^ 6) = 1062242 :=
by
  sorry

end math_problem_l122_12281


namespace ratio_fourth_to_third_l122_12250

theorem ratio_fourth_to_third (third_graders fifth_graders fourth_graders : ℕ) (H1 : third_graders = 20) (H2 : fifth_graders = third_graders / 2) (H3 : third_graders + fifth_graders + fourth_graders = 70) : fourth_graders / third_graders = 2 := by
  sorry

end ratio_fourth_to_third_l122_12250


namespace markus_more_marbles_l122_12226

theorem markus_more_marbles :
  let mara_bags := 12
  let marbles_per_mara_bag := 2
  let markus_bags := 2
  let marbles_per_markus_bag := 13
  let mara_marbles := mara_bags * marbles_per_mara_bag
  let markus_marbles := markus_bags * marbles_per_markus_bag
  mara_marbles + 2 = markus_marbles := 
by
  sorry

end markus_more_marbles_l122_12226


namespace concave_sequence_count_l122_12203

   theorem concave_sequence_count (m : ℕ) (h : 2 ≤ m) :
     ∀ b_0, (b_0 = 1 ∨ b_0 = 2) → 
     (∃ b : ℕ → ℕ, (∀ k, 2 ≤ k ∧ k ≤ m → b k + b (k - 2) ≤ 2 * b (k - 1)) → 
     (∃ S : ℕ, S ≤ 2^m)) :=
   by 
     sorry
   
end concave_sequence_count_l122_12203


namespace concentric_circles_ratio_l122_12297

theorem concentric_circles_ratio (R r k : ℝ) (hr : r > 0) (hRr : R > r) (hk : k > 0)
  (area_condition : π * (R^2 - r^2) = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
by
  sorry

end concentric_circles_ratio_l122_12297


namespace base_four_to_base_ten_of_20314_eq_568_l122_12264

-- Define what it means to convert a base-four number to base-ten
def base_four_to_base_ten (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldr (λ ⟨index, digit⟩ acc => acc + digit * 4^index) 0

-- Define the specific base-four number 20314_4 as a list of its digits
def num_20314_base_four : List ℕ := [2, 0, 3, 1, 4]

-- Theorem stating that the base-ten equivalent of 20314_4 is 568
theorem base_four_to_base_ten_of_20314_eq_568 : base_four_to_base_ten num_20314_base_four = 568 := sorry

end base_four_to_base_ten_of_20314_eq_568_l122_12264


namespace john_loses_probability_eq_3_over_5_l122_12255

-- Definitions used directly from the conditions in a)
def probability_win := 2 / 5
def probability_lose := 1 - probability_win

-- The theorem statement
theorem john_loses_probability_eq_3_over_5 : 
  probability_lose = 3 / 5 := 
by
  sorry -- proof is to be filled in later

end john_loses_probability_eq_3_over_5_l122_12255


namespace c_zero_roots_arithmetic_seq_range_f1_l122_12248

section problem

variable (b : ℝ)
def f (x : ℝ) := x^3 + 3 * b * x^2 + 0 * x + (-2 * b^3)
def f' (x : ℝ) := 3 * x^2 + 6 * b * x + 0

-- Proving c = 0 if f(x) is increasing on (-∞, 0) and decreasing on (0, 2)
theorem c_zero (h_inc : ∀ x < 0, f' b x > 0) (h_dec : ∀ x > 0, f' b x < 0) : 0 = 0 := sorry

-- Proving f(x) = 0 has two other distinct real roots x1 and x2 different from -b, forming an arithmetic sequence
theorem roots_arithmetic_seq (hb : ∀ x : ℝ, f b x = 0 → (x = -b ∨ -b ≠ x)) : 
    ∃ (x1 x2 : ℝ), x1 ≠ -b ∧ x2 ≠ -b ∧ x1 + x2 = -2 * b := sorry

-- Proving the range of values for f(1) when the maximum value of f(x) is less than 16
theorem range_f1 (h_max : ∀ x : ℝ, f b x < 16 ) : 0 ≤ f b 1 ∧ f b 1 < 11 := sorry

end problem

end c_zero_roots_arithmetic_seq_range_f1_l122_12248


namespace smallest_z_l122_12246

-- Given conditions
def distinct_consecutive_even_positive_perfect_cubes (w x y z : ℕ) : Prop :=
  w^3 + x^3 + y^3 = z^3 ∧
  ∃ a b c d : ℕ, 
    a < b ∧ b < c ∧ c < d ∧
    2 * a = w ∧ 2 * b = x ∧ 2 * c = y ∧ 2 * d = z

-- The smallest value of z proving the equation holds
theorem smallest_z (w x y z : ℕ) (h : distinct_consecutive_even_positive_perfect_cubes w x y z) : z = 12 :=
  sorry

end smallest_z_l122_12246


namespace geometric_series_sum_l122_12285

theorem geometric_series_sum :
  ∀ (a r : ℤ) (n : ℕ),
  a = 3 → r = -2 → n = 10 →
  (a * ((r ^ n - 1) / (r - 1))) = -1024 :=
by
  intros a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l122_12285


namespace LCM_of_18_and_27_l122_12205

theorem LCM_of_18_and_27 : Nat.lcm 18 27 = 54 := by
  sorry

end LCM_of_18_and_27_l122_12205


namespace sandbox_area_l122_12229

def sandbox_length : ℕ := 312
def sandbox_width : ℕ := 146

theorem sandbox_area : sandbox_length * sandbox_width = 45552 := by
  sorry

end sandbox_area_l122_12229


namespace max_magnitude_value_is_4_l122_12284

noncomputable def max_value_vector_magnitude (θ : ℝ) : ℝ :=
  let a := (Real.cos θ, Real.sin θ)
  let b := (Real.sqrt 3, -1)
  let vector := (2 * a.1 - b.1, 2 * a.2 + 1)
  Real.sqrt (vector.1 ^ 2 + vector.2 ^ 2)

theorem max_magnitude_value_is_4 (θ : ℝ) : 
  ∃ θ : ℝ, max_value_vector_magnitude θ = 4 :=
sorry

end max_magnitude_value_is_4_l122_12284


namespace green_flowers_count_l122_12244

theorem green_flowers_count :
  ∀ (G R B Y T : ℕ),
    T = 96 →
    R = 3 * G →
    B = 48 →
    Y = 12 →
    G + R + B + Y = T →
    G = 9 :=
by
  intros G R B Y T
  intro hT
  intro hR
  intro hB
  intro hY
  intro hSum
  sorry

end green_flowers_count_l122_12244


namespace sum_of_interior_edges_l122_12232

-- Define the problem parameters
def width_of_frame : ℝ := 2 -- width of the frame pieces in inches
def exposed_area : ℝ := 30 -- exposed area of the frame in square inches
def outer_edge_length : ℝ := 6 -- one of the outer edge length in inches

-- Define the statement to prove
theorem sum_of_interior_edges :
  ∃ (y : ℝ), (6 * y - 2 * (y - width_of_frame * 2) = exposed_area) ∧
  (2 * (6 - width_of_frame * 2) + 2 * (y - width_of_frame * 2) = 7) :=
sorry

end sum_of_interior_edges_l122_12232


namespace opponent_final_score_l122_12207

theorem opponent_final_score (x : ℕ) (h : x + 29 = 39) : x = 10 :=
by {
  sorry
}

end opponent_final_score_l122_12207


namespace constant_term_in_expansion_l122_12273

theorem constant_term_in_expansion : 
  let a := (x : ℝ)
  let b := - (2 / Real.sqrt x)
  let n := 6
  let general_term (r : Nat) : ℝ := Nat.choose n r * a * (b ^ (n - r))
  (∀ x : ℝ, ∃ (r : Nat), r = 4 ∧ (1 - (n - r) / 2 = 0) →
  general_term 4 = 60) :=
by
  sorry

end constant_term_in_expansion_l122_12273


namespace productivity_increase_correct_l122_12280

def productivity_increase (that: ℝ) :=
  ∃ x : ℝ, (x + 1) * (x + 1) * 2500 = 2809

theorem productivity_increase_correct :
  productivity_increase (0.06) :=
by
  sorry

end productivity_increase_correct_l122_12280


namespace jenny_investment_l122_12215

theorem jenny_investment :
  ∃ (m r : ℝ), m + r = 240000 ∧ r = 6 * m ∧ r = 205714.29 :=
by
  sorry

end jenny_investment_l122_12215


namespace distribute_positions_l122_12262

theorem distribute_positions :
  let positions := 11
  let classes := 6
  ∃ total_ways : ℕ, total_ways = Nat.choose (positions - 1) (classes - 1) ∧ total_ways = 252 :=
by
  let positions := 11
  let classes := 6
  have : Nat.choose (positions - 1) (classes - 1) = 252 := by sorry
  exact ⟨Nat.choose (positions - 1) (classes - 1), this, this⟩

end distribute_positions_l122_12262


namespace wrench_force_l122_12239

theorem wrench_force (F L k: ℝ) (h_inv: ∀ F L, F * L = k) (h_given: F * 12 = 240 * 12) : 
  (∀ L, (L = 16) → (F = 180)) ∧ (∀ L, (L = 8) → (F = 360)) := by 
sorry

end wrench_force_l122_12239


namespace complement_intersection_l122_12224

open Set

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (M_def : M = {2, 3})
variable (N_def : N = {1, 4})

theorem complement_intersection (U M N : Set ℕ) (U_def : U = {1, 2, 3, 4, 5, 6}) (M_def : M = {2, 3}) (N_def : N = {1, 4}) :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  sorry

end complement_intersection_l122_12224


namespace correct_conclusion_l122_12298

theorem correct_conclusion (x : ℝ) (hx : x > 1/2) : -2 * x + 1 < 0 :=
by
  -- sorry placeholder
  sorry

end correct_conclusion_l122_12298


namespace david_recreation_l122_12217

theorem david_recreation (W : ℝ) (P : ℝ) 
  (h1 : 0.95 * W = this_week_wages) 
  (h2 : 0.5 * this_week_wages = recreation_this_week)
  (h3 : 1.1875 * (P / 100) * W = recreation_this_week) : P = 40 :=
sorry

end david_recreation_l122_12217


namespace ken_got_1750_l122_12295

theorem ken_got_1750 (K : ℝ) (h : K + 2 * K = 5250) : K = 1750 :=
sorry

end ken_got_1750_l122_12295
