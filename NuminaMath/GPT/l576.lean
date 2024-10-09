import Mathlib

namespace seq_geom_seq_of_geom_and_arith_l576_57623

theorem seq_geom_seq_of_geom_and_arith (a : ℕ → ℕ) (b : ℕ → ℕ) 
  (h1 : ∃ a₁ : ℕ, ∀ n : ℕ, a n = a₁ * 2^(n-1))
  (h2 : ∃ b₁ d : ℕ, d = 3 ∧ ∀ n : ℕ, b (n + 1) = b₁ + n * d ∧ b₁ > 0) :
  ∃ r : ℕ, r = 8 ∧ ∃ a₁ : ℕ, ∀ n : ℕ, a (b (n + 1)) = a₁ * r^n :=
by
  sorry

end seq_geom_seq_of_geom_and_arith_l576_57623


namespace power_division_simplify_l576_57665

theorem power_division_simplify :
  ((9^9 / 9^8)^2 * 3^4) / 2^4 = 410 + 1/16 := by
  sorry

end power_division_simplify_l576_57665


namespace certain_number_divisibility_l576_57681

theorem certain_number_divisibility {n : ℕ} (h : ∃ count : ℕ, count = 50 ∧ (count = (300 / (2 * n)))) : n = 3 :=
by
  sorry

end certain_number_divisibility_l576_57681


namespace minimize_sum_areas_l576_57651

theorem minimize_sum_areas (x : ℝ) (h_wire_length : 0 < x ∧ x < 1) :
    let side_length := x / 4
    let square_area := (side_length ^ 2)
    let circle_radius := (1 - x) / (2 * Real.pi)
    let circle_area := Real.pi * (circle_radius ^ 2)
    let total_area := square_area + circle_area
    total_area = (x^2 / 16 + (1 - x)^2 / (4 * Real.pi)) -> 
    x = Real.pi / (Real.pi + 4) :=
by
  sorry

end minimize_sum_areas_l576_57651


namespace xiao_dong_not_both_understand_english_and_french_l576_57654

variables (P Q : Prop)

theorem xiao_dong_not_both_understand_english_and_french (h : ¬ (P ∧ Q)) : P → ¬ Q :=
sorry

end xiao_dong_not_both_understand_english_and_french_l576_57654


namespace min_value_of_f_solution_set_of_inequality_l576_57639

-- Define the given function f
def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- (1) Prove that the minimum value of y = f(x) is 3
theorem min_value_of_f : ∃ x : ℝ, f x = 3 := 
sorry

-- (2) Prove that the solution set of the inequality |f(x) - 6| ≤ 1 is [-10/3, -8/3] ∪ [0, 4/3]
theorem solution_set_of_inequality : 
  {x | |f x - 6| ≤ 1} = {x | -(10/3) ≤ x ∧ x ≤ -(8/3) ∨ 0 ≤ x ∧ x ≤ (4/3)} :=
sorry

end min_value_of_f_solution_set_of_inequality_l576_57639


namespace neighbor_packs_l576_57672

theorem neighbor_packs (n : ℕ) :
  let milly_balloons := 3 * 6 -- Milly and Floretta use 3 packs of their own
  let neighbor_balloons := n * 6 -- some packs of the neighbor's balloons, each contains 6 balloons
  let total_balloons := milly_balloons + neighbor_balloons -- total balloons
  -- They split balloons evenly; Milly takes 7 extra, then Floretta has 8 left
  total_balloons / 2 + 7 = total_balloons - 15
  → n = 2 := sorry

end neighbor_packs_l576_57672


namespace geometric_sequence_first_term_l576_57613

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) -- sequence a_n
  (r : ℝ) -- common ratio
  (h1 : r = 2) -- given common ratio
  (h2 : a 4 = 16) -- given a_4 = 16
  (h3 : ∀ n, a n = a 1 * r^(n-1)) -- definition of geometric sequence
  : a 1 = 2 := 
sorry

end geometric_sequence_first_term_l576_57613


namespace range_of_a_l576_57652

variable (a : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≥ 0)

def proposition_q : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + (2 - a) = 0

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : a ≤ -2 ∨ a = 1 :=
  sorry

end range_of_a_l576_57652


namespace remainder_when_divided_by_x_add_1_l576_57677

def q (x : ℝ) : ℝ := 2 * x^4 - 3 * x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_when_divided_by_x_add_1 :
  q 2 = 6 → q (-1) = 20 :=
by
  intro hq2
  sorry

end remainder_when_divided_by_x_add_1_l576_57677


namespace ceil_sqrt_product_l576_57686

noncomputable def ceil_sqrt_3 : ℕ := ⌈Real.sqrt 3⌉₊
noncomputable def ceil_sqrt_12 : ℕ := ⌈Real.sqrt 12⌉₊
noncomputable def ceil_sqrt_120 : ℕ := ⌈Real.sqrt 120⌉₊

theorem ceil_sqrt_product :
  ceil_sqrt_3 * ceil_sqrt_12 * ceil_sqrt_120 = 88 :=
by
  sorry

end ceil_sqrt_product_l576_57686


namespace quadratic_has_distinct_real_roots_l576_57619

theorem quadratic_has_distinct_real_roots : 
  ∀ (x : ℝ), x^2 - 3 * x + 1 = 0 → ∀ (a b c : ℝ), a = 1 ∧ b = -3 ∧ c = 1 →
  (b^2 - 4 * a * c) > 0 := 
by
  sorry

end quadratic_has_distinct_real_roots_l576_57619


namespace number_of_roots_of_unity_l576_57660

theorem number_of_roots_of_unity (n : ℕ) (z : ℂ) (c d : ℤ) (h1 : n ≥ 3) (h2 : z^n = 1) (h3 : z^3 + (c : ℂ) * z + (d : ℂ) = 0) : 
  ∃ k : ℕ, k = 4 :=
by sorry

end number_of_roots_of_unity_l576_57660


namespace cube_volume_is_216_l576_57602

-- Define the conditions
def total_edge_length : ℕ := 72
def num_edges_of_cube : ℕ := 12

-- The side length of the cube can be calculated as
def side_length (E : ℕ) (n : ℕ) : ℕ := E / n

-- The volume of the cube is the cube of its side length
def volume (s : ℕ) : ℕ := s ^ 3

theorem cube_volume_is_216 (E : ℕ) (n : ℕ) (V : ℕ) 
  (hE : E = total_edge_length) 
  (hn : n = num_edges_of_cube) 
  (hv : V = volume (side_length E n)) : 
  V = 216 := by
  sorry

end cube_volume_is_216_l576_57602


namespace tim_total_points_l576_57630

-- Definitions based on the conditions
def points_single : ℕ := 1000
def points_tetris : ℕ := 8 * points_single
def singles_scored : ℕ := 6
def tetrises_scored : ℕ := 4

-- Theorem stating the total points scored by Tim
theorem tim_total_points : singles_scored * points_single + tetrises_scored * points_tetris = 38000 := by
  sorry

end tim_total_points_l576_57630


namespace ratio_EG_FH_l576_57638

theorem ratio_EG_FH (EF FG EH : ℝ) (hEF : EF = 3) (hFG : FG = 7) (hEH : EH = 20) :
  (EF + FG) / (EH - EF) = 10 / 17 :=
by
  sorry

end ratio_EG_FH_l576_57638


namespace sum_f_84_eq_1764_l576_57608

theorem sum_f_84_eq_1764 (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, 0 < n → f n < f (n + 1))
  (h2 : ∀ m n : ℕ, 0 < m → 0 < n → f (m * n) = f m * f n)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → m ≠ n → m^n = n^m → (f m = n ∨ f n = m)) :
  f 84 = 1764 :=
by
  sorry

end sum_f_84_eq_1764_l576_57608


namespace cost_per_scarf_l576_57626

-- Define the cost of each earring
def cost_of_earring : ℕ := 6000

-- Define the number of earrings
def num_earrings : ℕ := 2

-- Define the cost of the iPhone
def cost_of_iphone : ℕ := 2000

-- Define the number of scarves
def num_scarves : ℕ := 4

-- Define the total value of the swag bag
def total_swag_bag_value : ℕ := 20000

-- Define the total value of diamond earrings and the iPhone
def total_value_of_earrings_and_iphone : ℕ := (num_earrings * cost_of_earring) + cost_of_iphone

-- Define the total value of the scarves
def total_value_of_scarves : ℕ := total_swag_bag_value - total_value_of_earrings_and_iphone

-- Define the cost of each designer scarf
def cost_of_each_scarf : ℕ := total_value_of_scarves / num_scarves

-- Prove that each designer scarf costs $1,500
theorem cost_per_scarf : cost_of_each_scarf = 1500 := by
  sorry

end cost_per_scarf_l576_57626


namespace find_point_coordinates_l576_57648

theorem find_point_coordinates (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so x < 0
  (h2 : P.2 > 0) -- Point P is in the second quadrant, so y > 0
  (h3 : abs P.2 = 4) -- distance from P to x-axis is 4
  (h4 : abs P.1 = 5) -- distance from P to y-axis is 5
  : P = (-5, 4) :=
by {
  -- point P is in the second quadrant, so x < 0 and y > 0
  -- |y| = 4 -> y = 4 
  -- |x| = 5 -> x = -5
  sorry
}

end find_point_coordinates_l576_57648


namespace better_offer_saves_800_l576_57655

theorem better_offer_saves_800 :
  let initial_order := 20000
  let discount1 (x : ℝ) := x * 0.70 * 0.90 - 800
  let discount2 (x : ℝ) := x * 0.75 * 0.80 - 1000
  discount1 initial_order - discount2 initial_order = 800 :=
by
  sorry

end better_offer_saves_800_l576_57655


namespace tile_difference_is_42_l576_57616

def original_blue_tiles : ℕ := 14
def original_green_tiles : ℕ := 8
def green_tiles_first_border : ℕ := 18
def green_tiles_second_border : ℕ := 30

theorem tile_difference_is_42 :
  (original_green_tiles + green_tiles_first_border + green_tiles_second_border) - original_blue_tiles = 42 :=
by
  sorry

end tile_difference_is_42_l576_57616


namespace eccentricity_of_hyperbola_l576_57683

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : ℝ :=
  (3 * Real.sqrt 7) / 7

-- Ensure the function returns the correct eccentricity
theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : hyperbola_eccentricity a b c ha hb h = (3 * Real.sqrt 7) / 7 :=
sorry

end eccentricity_of_hyperbola_l576_57683


namespace remaining_customers_l576_57684

theorem remaining_customers (initial: ℕ) (left: ℕ) (remaining: ℕ) 
  (h1: initial = 14) (h2: left = 11) : remaining = initial - left → remaining = 3 :=
by {
  sorry
}

end remaining_customers_l576_57684


namespace range_of_m_l576_57641

def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + b

def has_local_extremum_at (a b x : ℝ) : Prop :=
  f_prime a b x = 0 ∧ f a b x = 0

def h (a b m x : ℝ) : ℝ := f a b x - m + 1

theorem range_of_m (a b m : ℝ) :
  (has_local_extremum_at 2 9 (-1) ∧
   ∀ x, f 2 9 x = x^3 + 6 * x^2 + 9 * x + 4) →
  (∀ x, (x^3 + 6 * x^2 + 9 * x + 4 - m + 1 = 0) → 
  1 < m ∧ m < 5) := 
sorry

end range_of_m_l576_57641


namespace tournament_byes_and_games_l576_57670

/-- In a single-elimination tournament with 300 players initially registered,
- if the number of players in each subsequent round must be a power of 2,
- then 44 players must receive a bye in the first round, and 255 total games
- must be played to determine the champion. -/
theorem tournament_byes_and_games :
  let initial_players := 300
  let pow2_players := 256
  44 = initial_players - pow2_players ∧
  255 = pow2_players - 1 :=
by
  let initial_players := 300
  let pow2_players := 256
  have h_byes : 44 = initial_players - pow2_players := by sorry
  have h_games : 255 = pow2_players - 1 := by sorry
  exact ⟨h_byes, h_games⟩

end tournament_byes_and_games_l576_57670


namespace expected_sides_of_red_polygon_l576_57611

-- Define the conditions
def isChosenWithinSquare (F : ℝ × ℝ) (side_length: ℝ) : Prop :=
  0 ≤ F.1 ∧ F.1 ≤ side_length ∧ 0 ≤ F.2 ∧ F.2 ≤ side_length

def pointF (side_length: ℝ) : ℝ × ℝ := sorry
def foldToF (vertex: ℝ × ℝ) (F: ℝ × ℝ) : ℝ := sorry

-- Define the expected number of sides of the resulting red polygon
noncomputable def expected_sides (side_length : ℝ) : ℝ :=
  let P_g := 2 - (Real.pi / 2)
  let P_o := (Real.pi / 2) - 1 
  (3 * P_o) + (4 * P_g)

-- Prove the expected number of sides equals 5 - π / 2
theorem expected_sides_of_red_polygon (side_length : ℝ) :
  expected_sides side_length = 5 - (Real.pi / 2) := 
  by sorry

end expected_sides_of_red_polygon_l576_57611


namespace market_value_correct_l576_57646

noncomputable def market_value : ℝ :=
  let dividend_income (M : ℝ) := 0.12 * M
  let fees (M : ℝ) := 0.01 * M
  let taxes (M : ℝ) := 0.15 * dividend_income M
  have yield_after_fees_and_taxes : ∀ M, 0.08 * M = dividend_income M - fees M - taxes M := 
    by sorry
  86.96

theorem market_value_correct :
  market_value = 86.96 := 
by
  sorry

end market_value_correct_l576_57646


namespace simple_interest_years_l576_57612

theorem simple_interest_years (P : ℝ) (R : ℝ) (N : ℝ) (higher_interest_amount : ℝ) (additional_rate : ℝ) (initial_sum : ℝ) :
  (initial_sum * (R + additional_rate) * N) / 100 - (initial_sum * R * N) / 100 = higher_interest_amount →
  initial_sum = 3000 →
  higher_interest_amount = 1350 →
  additional_rate = 5 →
  N = 9 :=
by
  sorry

end simple_interest_years_l576_57612


namespace lcm_pairs_count_l576_57662

noncomputable def distinct_pairs_lcm_count : ℕ :=
  sorry

theorem lcm_pairs_count :
  distinct_pairs_lcm_count = 1502 :=
  sorry

end lcm_pairs_count_l576_57662


namespace Donny_change_l576_57658

/-- The change Donny will receive after filling up his truck. -/
theorem Donny_change
  (capacity : ℝ)
  (initial_fuel : ℝ)
  (cost_per_liter : ℝ)
  (money_available : ℝ)
  (change : ℝ) :
  capacity = 150 →
  initial_fuel = 38 →
  cost_per_liter = 3 →
  money_available = 350 →
  change = money_available - cost_per_liter * (capacity - initial_fuel) →
  change = 14 :=
by
  intros h_capacity h_initial_fuel h_cost_per_liter h_money_available h_change
  rw [h_capacity, h_initial_fuel, h_cost_per_liter, h_money_available] at h_change
  sorry

end Donny_change_l576_57658


namespace altitude_segments_of_acute_triangle_l576_57647

/-- If two altitudes of an acute triangle divide the sides into segments of lengths 5, 3, 2, and x units,
then x is equal to 10. -/
theorem altitude_segments_of_acute_triangle (a b c d e : ℝ) (h1 : a = 5) (h2 : b = 3) (h3 : c = 2) (h4 : d = x) :
  x = 10 :=
by
  sorry

end altitude_segments_of_acute_triangle_l576_57647


namespace solution_exists_iff_divisor_form_l576_57691

theorem solution_exists_iff_divisor_form (n : ℕ) (hn_pos : 0 < n) (hn_odd : n % 2 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 4 * x * y = n * (x + y)) ↔
    (∃ k : ℕ, n % (4 * k + 3) = 0) :=
by
  sorry

end solution_exists_iff_divisor_form_l576_57691


namespace range_of_a_l576_57657

theorem range_of_a (a : ℝ) :
  (a > 0 ∧ (∃ x, x^2 - 4 * a * x + 3 * a^2 < 0)) →
  (∃ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0) →
  (2 < a ∧ a ≤ 2) := sorry

end range_of_a_l576_57657


namespace wyatt_envelopes_fewer_l576_57676

-- Define assets for envelopes
variables (blue_envelopes yellow_envelopes : ℕ)

-- Conditions from the problem
def wyatt_conditions :=
  blue_envelopes = 10 ∧ yellow_envelopes < blue_envelopes ∧ blue_envelopes + yellow_envelopes = 16

-- Theorem: How many fewer yellow envelopes Wyatt has compared to blue envelopes?
theorem wyatt_envelopes_fewer (hb : blue_envelopes = 10) (ht : blue_envelopes + yellow_envelopes = 16) : 
  blue_envelopes - yellow_envelopes = 4 := 
by sorry

end wyatt_envelopes_fewer_l576_57676


namespace polar_eq_to_cartesian_l576_57649

-- Define the conditions
def polar_to_cartesian_eq (ρ : ℝ) : Prop :=
  ρ = 2 → (∃ x y : ℝ, x^2 + y^2 = ρ^2)

-- State the main theorem/proof problem
theorem polar_eq_to_cartesian : polar_to_cartesian_eq 2 :=
by
  -- Proof sketch:
  --   Given ρ = 2
  --   We have ρ^2 = 4
  --   By converting to Cartesian coordinates: x^2 + y^2 = ρ^2
  --   Result: x^2 + y^2 = 4
  sorry

end polar_eq_to_cartesian_l576_57649


namespace mechanic_worked_hours_l576_57601

theorem mechanic_worked_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (part_count : ℕ) 
  (labor_cost_per_minute : ℝ) 
  (parts_total_cost : ℝ) 
  (labor_total_cost : ℝ) 
  (hours_worked : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_count = 2) 
  (h3 : part_cost = 20) 
  (h4 : labor_cost_per_minute = 0.5) 
  (h5 : parts_total_cost = part_count * part_cost) 
  (h6 : labor_total_cost = total_cost - parts_total_cost) 
  (labor_cost_per_hour := labor_cost_per_minute * 60) 
  (h7 : hours_worked = labor_total_cost / labor_cost_per_hour) : 
  hours_worked = 6 := 
sorry

end mechanic_worked_hours_l576_57601


namespace cost_to_paint_cube_l576_57645

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) (total_cost : ℝ) :
  cost_per_kg = 36.50 →
  coverage_per_kg = 16 →
  side_length = 8 →
  total_cost = (6 * side_length^2 / coverage_per_kg) * cost_per_kg →
  total_cost = 876 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_to_paint_cube_l576_57645


namespace circle_S_radius_properties_l576_57656

theorem circle_S_radius_properties :
  let DE := 120
  let DF := 120
  let EF := 68
  let R_radius := 20
  let S_radius := 52 - 6 * Real.sqrt 35
  let m := 52
  let n := 6
  let k := 35
  m + n * k = 262 := by
  sorry

end circle_S_radius_properties_l576_57656


namespace nina_ants_count_l576_57693

theorem nina_ants_count 
  (spiders : ℕ) 
  (eyes_per_spider : ℕ) 
  (eyes_per_ant : ℕ) 
  (total_eyes : ℕ) 
  (total_spider_eyes : ℕ) 
  (total_ant_eyes : ℕ) 
  (ants : ℕ) 
  (h1 : spiders = 3) 
  (h2 : eyes_per_spider = 8) 
  (h3 : eyes_per_ant = 2) 
  (h4 : total_eyes = 124) 
  (h5 : total_spider_eyes = spiders * eyes_per_spider) 
  (h6 : total_ant_eyes = total_eyes - total_spider_eyes) 
  (h7 : ants = total_ant_eyes / eyes_per_ant) : 
  ants = 50 := by
  sorry

end nina_ants_count_l576_57693


namespace y2_over_x2_plus_x2_over_y2_eq_9_over_4_l576_57689

theorem y2_over_x2_plus_x2_over_y2_eq_9_over_4 (x y : ℝ) 
  (h : (1 / x) - (1 / (2 * y)) = (1 / (2 * x + y))) : 
  (y^2 / x^2) + (x^2 / y^2) = 9 / 4 := 
by 
  sorry

end y2_over_x2_plus_x2_over_y2_eq_9_over_4_l576_57689


namespace bruce_three_times_son_in_six_years_l576_57669

-- Define the current ages of Bruce and his son
def bruce_age : ℕ := 36
def son_age : ℕ := 8

-- Define the statement to be proved
theorem bruce_three_times_son_in_six_years :
  ∃ (x : ℕ), x = 6 ∧ ∀ t, (t = x) → (bruce_age + t = 3 * (son_age + t)) :=
by
  sorry

end bruce_three_times_son_in_six_years_l576_57669


namespace inequality_abc_l576_57642

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^3) / (a^2 + a * b + b^2) + (b^3) / (b^2 + b * c + c^2) + (c^3) / (c^2 + c * a + a^2) ≥ (a + b + c) / 3 := 
by
    sorry

end inequality_abc_l576_57642


namespace jimmy_more_sheets_than_tommy_l576_57664

-- Definitions for the conditions
def initial_jimmy_sheets : ℕ := 58
def initial_tommy_sheets : ℕ := initial_jimmy_sheets + 25
def ashton_gives_jimmy : ℕ := 85
def jessica_gives_jimmy : ℕ := 47
def cousin_gives_tommy : ℕ := 30
def aunt_gives_tommy : ℕ := 19

-- Lean 4 statement for the proof problem
theorem jimmy_more_sheets_than_tommy :
  let final_jimmy_sheets := initial_jimmy_sheets + ashton_gives_jimmy + jessica_gives_jimmy;
  let final_tommy_sheets := initial_tommy_sheets + cousin_gives_tommy + aunt_gives_tommy;
  final_jimmy_sheets - final_tommy_sheets = 58 :=
by sorry

end jimmy_more_sheets_than_tommy_l576_57664


namespace math_problem_l576_57609

theorem math_problem (f_star f_ast : ℕ → ℕ → ℕ) (h₁ : f_star 20 5 = 15) (h₂ : f_ast 15 5 = 75) :
  (f_star 8 4) / (f_ast 10 2) = (1:ℚ) / 5 := by
  sorry

end math_problem_l576_57609


namespace simplify_expression_l576_57653

theorem simplify_expression : 
  3 * Real.sqrt 48 - 9 * Real.sqrt (1 / 3) - Real.sqrt 3 * (2 - Real.sqrt 27) = 7 * Real.sqrt 3 + 9 :=
by
  -- The proof is omitted as per the instructions
  sorry

end simplify_expression_l576_57653


namespace hyperbola_asymptotes_angle_l576_57666

theorem hyperbola_asymptotes_angle {a b : ℝ} (h₁ : a > b) 
  (h₂ : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h₃ : ∀ θ : ℝ, θ = Real.pi / 4) : a / b = Real.sqrt 2 :=
by
  sorry

end hyperbola_asymptotes_angle_l576_57666


namespace ethanol_solution_exists_l576_57659

noncomputable def ethanol_problem : Prop :=
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 204 ∧ 0.12 * x + 0.16 * (204 - x) = 30

theorem ethanol_solution_exists : ethanol_problem :=
sorry

end ethanol_solution_exists_l576_57659


namespace sequence_formula_l576_57629

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (diff : ∀ n, a (n + 1) - a n = 3^n) :
  ∀ n, a n = (3^n - 1) / 2 :=
by
  sorry

end sequence_formula_l576_57629


namespace number_of_chicks_is_8_l576_57673

-- Define the number of total chickens
def total_chickens : ℕ := 15

-- Define the number of hens
def hens : ℕ := 3

-- Define the number of roosters
def roosters : ℕ := total_chickens - hens

-- Define the number of chicks
def chicks : ℕ := roosters - 4

-- State the main theorem
theorem number_of_chicks_is_8 : chicks = 8 := 
by
  -- the solution follows from the given definitions and conditions
  sorry

end number_of_chicks_is_8_l576_57673


namespace no_solution_A_eq_B_l576_57674

theorem no_solution_A_eq_B (a : ℝ) (h1 : a = 2 * a) (h2 : a ≠ 2) : false := by
  sorry

end no_solution_A_eq_B_l576_57674


namespace product_scaled_areas_l576_57617

variable (a b c k V : ℝ)

def volume (a b c : ℝ) : ℝ := a * b * c

theorem product_scaled_areas (a b c k : ℝ) (V : ℝ) (hV : V = volume a b c) :
  (k * a * b) * (k * b * c) * (k * c * a) = k^3 * (V^2) := 
by
  -- Proof steps would go here, but we use sorry to skip the proof
  sorry

end product_scaled_areas_l576_57617


namespace spherical_circle_radius_l576_57620

theorem spherical_circle_radius:
  (∀ (θ : Real), ∃ (r : Real), r = 1 * Real.sin (Real.pi / 6)) → ∀ (θ : Real), r = 1 / 2 := by
  sorry

end spherical_circle_radius_l576_57620


namespace Jungkook_has_most_apples_l576_57622

-- Conditions
def Yoongi_apples : ℕ := 4
def Jungkook_apples_initial : ℕ := 6
def Jungkook_apples_additional : ℕ := 3
def Jungkook_total_apples : ℕ := Jungkook_apples_initial + Jungkook_apples_additional
def Yuna_apples : ℕ := 5

-- Statement (to prove)
theorem Jungkook_has_most_apples : Jungkook_total_apples > Yoongi_apples ∧ Jungkook_total_apples > Yuna_apples := by
  sorry

end Jungkook_has_most_apples_l576_57622


namespace mean_steps_per_day_l576_57661

theorem mean_steps_per_day (total_steps : ℕ) (days_in_april : ℕ) (h_total : total_steps = 243000) (h_days : days_in_april = 30) :
  (total_steps / days_in_april) = 8100 :=
by
  sorry

end mean_steps_per_day_l576_57661


namespace initial_population_l576_57625

theorem initial_population (P : ℝ) (h1 : 0.76 * P = 3553) : P = 4678 :=
by
  sorry

end initial_population_l576_57625


namespace triangle_inequality_check_triangle_sets_l576_57643

theorem triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem check_triangle_sets :
  ¬triangle_inequality 1 2 3 ∧
  triangle_inequality 2 2 2 ∧
  ¬triangle_inequality 2 2 4 ∧
  ¬triangle_inequality 1 3 5 :=
by
  sorry

end triangle_inequality_check_triangle_sets_l576_57643


namespace new_computer_price_l576_57607

-- Define the initial conditions
def initial_price_condition (x : ℝ) : Prop := 2 * x = 540

-- Define the calculation for the new price after a 30% increase
def new_price (x : ℝ) : ℝ := x * 1.30

-- Define the final proof problem statement
theorem new_computer_price : ∃ x : ℝ, initial_price_condition x ∧ new_price x = 351 :=
by sorry

end new_computer_price_l576_57607


namespace math_problem_l576_57680

theorem math_problem : (4 + 6 + 7) * 2 - 2 + (3 / 3) = 33 := 
by
  sorry

end math_problem_l576_57680


namespace exist_abc_l576_57635

theorem exist_abc (n k : ℕ) (h1 : 20 < n) (h2 : 1 < k) (h3 : n % k^2 = 0) :
  ∃ a b c : ℕ, n = a * b + b * c + c * a :=
sorry

end exist_abc_l576_57635


namespace base8_357_plus_base13_4CD_eq_1084_l576_57687

def C := 12
def D := 13

def base8_357 := 3 * (8^2) + 5 * (8^1) + 7 * (8^0)
def base13_4CD := 4 * (13^2) + C * (13^1) + D * (13^0)

theorem base8_357_plus_base13_4CD_eq_1084 :
  base8_357 + base13_4CD = 1084 :=
by
  sorry

end base8_357_plus_base13_4CD_eq_1084_l576_57687


namespace alice_operations_terminate_l576_57695

theorem alice_operations_terminate (a : List ℕ) (h_pos : ∀ x ∈ a, x > 0) : 
(∀ x y z, (x, y) = (y + 1, x) ∨ (x, y) = (x - 1, x) → ∃ n, (x :: y :: z).sum ≤ n) :=
by sorry

end alice_operations_terminate_l576_57695


namespace sum_of_x_y_z_l576_57679

theorem sum_of_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 2 * y) : x + y + z = 10 * x := by
  sorry

end sum_of_x_y_z_l576_57679


namespace ticket_price_l576_57640

theorem ticket_price (Olivia_money : ℕ) (Nigel_money : ℕ) (left_money : ℕ) (total_tickets : ℕ)
  (h1 : Olivia_money = 112)
  (h2 : Nigel_money = 139)
  (h3 : left_money = 83)
  (h4 : total_tickets = 6) :
  (Olivia_money + Nigel_money - left_money) / total_tickets = 28 :=
by
  sorry

end ticket_price_l576_57640


namespace perimeter_of_square_36_l576_57667

variable (a s P : ℕ)

def is_square_area : Prop := a = s * s
def is_square_perimeter : Prop := P = 4 * s
def condition : Prop := 5 * a = 10 * P + 45

theorem perimeter_of_square_36 (h1 : is_square_area a s) (h2 : is_square_perimeter P s) (h3 : condition a P) : P = 36 := 
by
  sorry

end perimeter_of_square_36_l576_57667


namespace remainder_of_B_is_4_l576_57610

theorem remainder_of_B_is_4 (A B : ℕ) (h : B = A * 9 + 13) : B % 9 = 4 :=
by {
  sorry
}

end remainder_of_B_is_4_l576_57610


namespace smallest_integer_for_inequality_l576_57671

theorem smallest_integer_for_inequality :
  ∃ x : ℤ, x^2 < 2 * x + 1 ∧ ∀ y : ℤ, y^2 < 2 * y + 1 → x ≤ y := sorry

end smallest_integer_for_inequality_l576_57671


namespace correct_equation_option_l576_57668

theorem correct_equation_option :
  (∀ (x : ℝ), (x = 4 → false) ∧ (x = -4 → false)) →
  (∀ (y : ℝ), (y = 12 → true) ∧ (y = -12 → false)) →
  (∀ (z : ℝ), (z = -7 → false) ∧ (z = 7 → true)) →
  (∀ (w : ℝ), (w = 2 → true)) →
  ∃ (option : ℕ), option = 4 := 
by
  sorry

end correct_equation_option_l576_57668


namespace club_boys_count_l576_57688

theorem club_boys_count (B G : ℕ) (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 18) : B = 12 :=
by
  -- We would proceed with the steps here, but add 'sorry' to indicate incomplete proof
  sorry

end club_boys_count_l576_57688


namespace cost_price_percentage_l576_57631

theorem cost_price_percentage (MP CP SP : ℝ) (h1 : SP = 0.88 * MP) (h2 : SP = 1.375 * CP) :
  (CP / MP) * 100 = 64 :=
by
  sorry

end cost_price_percentage_l576_57631


namespace percentage_salt_in_mixture_l576_57636

-- Conditions
def volume_pure_water : ℝ := 1
def volume_salt_solution : ℝ := 2
def salt_concentration : ℝ := 0.30
def total_volume : ℝ := volume_pure_water + volume_salt_solution
def amount_of_salt_in_solution : ℝ := salt_concentration * volume_salt_solution

-- Theorem
theorem percentage_salt_in_mixture :
  (amount_of_salt_in_solution / total_volume) * 100 = 20 :=
by
  sorry

end percentage_salt_in_mixture_l576_57636


namespace fourth_powers_sum_is_8432_l576_57678

def sum_fourth_powers (x y : ℝ) : ℝ := x^4 + y^4

theorem fourth_powers_sum_is_8432 (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 4) : 
  sum_fourth_powers x y = 8432 :=
by
  sorry

end fourth_powers_sum_is_8432_l576_57678


namespace trig_identity_l576_57692

theorem trig_identity : 
  (Real.sin (12 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (72 * Real.pi / 180)) * (Real.sin (84 * Real.pi / 180)) = 1 / 32 :=
by sorry

end trig_identity_l576_57692


namespace phase_shift_of_cosine_l576_57627

theorem phase_shift_of_cosine (a b c : ℝ) (h : c = -π / 4 ∧ b = 3) :
  (-c / b) = π / 12 :=
by
  sorry

end phase_shift_of_cosine_l576_57627


namespace payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l576_57624

namespace ShoppingMall

def tea_set_price : ℕ := 200
def tea_bowl_price : ℕ := 20
def discount_option_1 (x : ℕ) : ℕ := 20 * x + 5400
def discount_option_2 (x : ℕ) : ℕ := 19 * x + 5700
def combined_option_40 : ℕ := 6000 + 190

theorem payment_equation_1 (x : ℕ) (hx : x > 30) : 
  discount_option_1 x = 20 * x + 5400 :=
by sorry

theorem payment_equation_2 (x : ℕ) (hx : x > 30) : 
  discount_option_2 x = 19 * x + 5700 :=
by sorry

theorem cost_effective_40 : discount_option_1 40 < discount_option_2 40 :=
by sorry

theorem combined_cost_effective_40 : combined_option_40 < discount_option_1 40 ∧ combined_option_40 < discount_option_2 40 :=
by sorry

end ShoppingMall

end payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l576_57624


namespace problem_solution_l576_57618

theorem problem_solution :
  (2200 - 2089)^2 / 196 = 63 :=
sorry

end problem_solution_l576_57618


namespace find_n_from_A_k_l576_57663

theorem find_n_from_A_k (n : ℕ) (A : ℕ → ℕ) (h1 : A 1 = Int.natAbs (n + 1))
  (h2 : ∀ k : ℕ, k > 0 → A k = Int.natAbs (n + (2 * k - 1)))
  (h3 : A 100 = 2005) : n = 1806 :=
sorry

end find_n_from_A_k_l576_57663


namespace area_of_square_with_adjacent_points_l576_57600

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l576_57600


namespace relationship_abc_l576_57694

variables {a b c : ℝ}

-- Given conditions
def condition1 (a b c : ℝ) : Prop := 0 < a ∧ 0 < b ∧ 0 < c ∧ (11/6 : ℝ) * c < a + b ∧ a + b < 2 * c
def condition2 (a b c : ℝ) : Prop := (3/2 : ℝ) * a < b + c ∧ b + c < (5/3 : ℝ) * a
def condition3 (a b c : ℝ) : Prop := (5/2 : ℝ) * b < a + c ∧ a + c < (11/4 : ℝ) * b

-- Proof statement
theorem relationship_abc (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  b < c ∧ c < a :=
by
  sorry

end relationship_abc_l576_57694


namespace girls_attending_event_l576_57632

theorem girls_attending_event (g b : ℕ) 
  (h1 : g + b = 1500)
  (h2 : 3 / 4 * g + 2 / 5 * b = 900) :
  3 / 4 * g = 643 := 
by
  sorry

end girls_attending_event_l576_57632


namespace tickets_difference_l576_57606

-- Define the conditions
def tickets_friday : ℕ := 181
def tickets_sunday : ℕ := 78
def tickets_saturday : ℕ := 2 * tickets_friday

-- The theorem to prove
theorem tickets_difference : tickets_saturday - tickets_sunday = 284 := by
  sorry

end tickets_difference_l576_57606


namespace happy_dictionary_problem_l576_57690

def smallest_positive_integer : ℕ := 1
def largest_negative_integer : ℤ := -1
def smallest_abs_rational : ℚ := 0

theorem happy_dictionary_problem : 
  smallest_positive_integer - largest_negative_integer + smallest_abs_rational = 2 := 
by
  sorry

end happy_dictionary_problem_l576_57690


namespace triangles_needed_for_hexagon_with_perimeter_19_l576_57614

def num_triangles_to_construct_hexagon (perimeter : ℕ) : ℕ :=
  match perimeter with
  | 19 => 59
  | _ => 0  -- We handle only the case where perimeter is 19

theorem triangles_needed_for_hexagon_with_perimeter_19 :
  num_triangles_to_construct_hexagon 19 = 59 :=
by
  -- Here we assert that the number of triangles to construct the hexagon with perimeter 19 is 59
  sorry

end triangles_needed_for_hexagon_with_perimeter_19_l576_57614


namespace candy_from_sister_l576_57603

variable (total_neighbors : Nat) (pieces_per_day : Nat) (days : Nat) (total_pieces : Nat)
variable (pieces_per_day_eq : pieces_per_day = 9)
variable (days_eq : days = 9)
variable (total_neighbors_eq : total_neighbors = 66)
variable (total_pieces_eq : total_pieces = 81)

theorem candy_from_sister : 
  total_pieces = total_neighbors + 15 :=
by
  sorry

end candy_from_sister_l576_57603


namespace MargaretsMeanScore_l576_57605

theorem MargaretsMeanScore :
  ∀ (scores : List ℕ)
    (cyprian_mean : ℝ)
    (highest_lowest_different : Prop),
    scores = [82, 85, 88, 90, 92, 95, 97, 99] →
    cyprian_mean = 88.5 →
    highest_lowest_different →
    ∃ (margaret_mean : ℝ), margaret_mean = 93.5 := by
  sorry

end MargaretsMeanScore_l576_57605


namespace total_pencils_l576_57633

theorem total_pencils (initial_additional1 initial_additional2 : ℕ) (h₁ : initial_additional1 = 37) (h₂ : initial_additional2 = 17) : (initial_additional1 + initial_additional2) = 54 :=
by sorry

end total_pencils_l576_57633


namespace shelves_needed_number_of_shelves_l576_57628

-- Define the initial number of books
def initial_books : Float := 46.0

-- Define the number of additional books added by the librarian
def additional_books : Float := 10.0

-- Define the number of books each shelf can hold
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_books + additional_books

-- The mathematical proof statement for the number of shelves needed
theorem shelves_needed : Float := total_books / books_per_shelf

-- The required statement proving that the number of shelves needed is 14.0
theorem number_of_shelves : shelves_needed = 14.0 := by
  sorry

end shelves_needed_number_of_shelves_l576_57628


namespace different_prime_factors_of_factorial_eq_10_l576_57699

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l576_57699


namespace polynomial_expansion_a6_l576_57634

theorem polynomial_expansion_a6 :
  let p := x^2 + x^7
  ∃ (a : ℕ → ℝ), p = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 ∧ a 6 = -7 := 
sorry

end polynomial_expansion_a6_l576_57634


namespace teams_B_and_C_worked_together_days_l576_57644

def workload_project_B := 5/4
def time_team_A_project_A := 20
def time_team_B_project_A := 24
def time_team_C_project_A := 30

def equation1 (x y : ℕ) : Prop := 
  3 * x + 5 * y = 60

def equation2 (x y : ℕ) : Prop := 
  9 * x + 5 * y = 150

theorem teams_B_and_C_worked_together_days (x : ℕ) (y : ℕ) :
  equation1 x y ∧ equation2 x y → x = 15 := 
by 
  sorry

end teams_B_and_C_worked_together_days_l576_57644


namespace torn_pages_are_112_and_113_l576_57697

theorem torn_pages_are_112_and_113 (n k : ℕ) (S S' : ℕ) 
  (h1 : S = n * (n + 1) / 2)
  (h2 : S' = S - (k - 1) - k)
  (h3 : S' = 15000) :
  (k = 113) ∧ (k - 1 = 112) :=
by
  sorry

end torn_pages_are_112_and_113_l576_57697


namespace count_integers_between_sqrt5_and_sqrt50_l576_57696

theorem count_integers_between_sqrt5_and_sqrt50 
  (h1 : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3)
  (h2 : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8) : 
  ∃ n : ℕ, n = 5 := 
sorry

end count_integers_between_sqrt5_and_sqrt50_l576_57696


namespace ratio_of_josh_to_brad_l576_57698

theorem ratio_of_josh_to_brad (J D B : ℝ) (h1 : J + D + B = 68) (h2 : J = (3 / 4) * D) (h3 : D = 32) :
  (J / B) = 2 :=
by
  sorry

end ratio_of_josh_to_brad_l576_57698


namespace no_solutions_to_equation_l576_57675

theorem no_solutions_to_equation (a b c : ℤ) : a^2 + b^2 - 8 * c ≠ 6 := 
by 
-- sorry to skip the proof part
sorry

end no_solutions_to_equation_l576_57675


namespace exists_k_square_congruent_neg_one_iff_l576_57621

theorem exists_k_square_congruent_neg_one_iff (p : ℕ) [Fact p.Prime] :
  (∃ k : ℤ, (k^2 ≡ -1 [ZMOD p])) ↔ (p = 2 ∨ p % 4 = 1) :=
sorry

end exists_k_square_congruent_neg_one_iff_l576_57621


namespace cos_neg_pi_div_3_l576_57604

theorem cos_neg_pi_div_3 : Real.cos (-π / 3) = 1 / 2 := 
by 
  sorry

end cos_neg_pi_div_3_l576_57604


namespace kate_change_is_correct_l576_57685

-- Define prices of items
def gum_price : ℝ := 0.89
def chocolate_price : ℝ := 1.25
def chips_price : ℝ := 2.49

-- Define sales tax rate
def tax_rate : ℝ := 0.06

-- Define the total money Kate gave to the clerk
def payment : ℝ := 10.00

-- Define total cost of items before tax
def total_before_tax := gum_price + chocolate_price + chips_price

-- Define the sales tax
def sales_tax := tax_rate * total_before_tax

-- Define the correct answer for total cost
def total_cost := total_before_tax + sales_tax

-- Define the correct amount of change Kate should get back
def change := payment - total_cost

theorem kate_change_is_correct : abs (change - 5.09) < 0.01 :=
by
  sorry

end kate_change_is_correct_l576_57685


namespace ella_savings_l576_57615

theorem ella_savings
  (initial_cost_per_lamp : ℝ)
  (num_lamps : ℕ)
  (discount_rate : ℝ)
  (additional_discount : ℝ)
  (initial_total_cost : ℝ := num_lamps * initial_cost_per_lamp)
  (discounted_lamp_cost : ℝ := initial_cost_per_lamp - (initial_cost_per_lamp * discount_rate))
  (total_cost_with_discount : ℝ := num_lamps * discounted_lamp_cost)
  (total_cost_after_additional_discount : ℝ := total_cost_with_discount - additional_discount) :
  initial_cost_per_lamp = 15 →
  num_lamps = 3 →
  discount_rate = 0.25 →
  additional_discount = 5 →
  initial_total_cost - total_cost_after_additional_discount = 16.25 :=
by
  intros
  sorry

end ella_savings_l576_57615


namespace largest_int_less_150_gcd_18_eq_6_l576_57682

theorem largest_int_less_150_gcd_18_eq_6 : ∃ (n : ℕ), n < 150 ∧ gcd n 18 = 6 ∧ ∀ (m : ℕ), m < 150 ∧ gcd m 18 = 6 → m ≤ n ∧ n = 138 := 
by
  sorry

end largest_int_less_150_gcd_18_eq_6_l576_57682


namespace apple_box_weights_l576_57637

theorem apple_box_weights (a b c d : ℤ) 
  (h1 : a + b + c = 70)
  (h2 : a + b + d = 80)
  (h3 : a + c + d = 73)
  (h4 : b + c + d = 77) : 
  a = 23 ∧ b = 27 ∧ c = 20 ∧ d = 30 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end apple_box_weights_l576_57637


namespace preimage_of_point_l576_57650

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

-- Define the statement of the problem
theorem preimage_of_point {x y : ℝ} (h1 : f x y = (3, 1)) : (x = 2 ∧ y = 1) :=
by
  sorry

end preimage_of_point_l576_57650
