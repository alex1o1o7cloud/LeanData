import Mathlib

namespace parabola_points_relation_l1114_111451

theorem parabola_points_relation {a b c y1 y2 y3 : ℝ} 
  (hA : y1 = a * (1 / 2)^2 + b * (1 / 2) + c)
  (hB : y2 = a * (0)^2 + b * (0) + c)
  (hC : y3 = a * (-1)^2 + b * (-1) + c)
  (h_cond : 0 < 2 * a ∧ 2 * a < b) : 
  y1 > y2 ∧ y2 > y3 :=
by 
  sorry

end parabola_points_relation_l1114_111451


namespace numer_greater_than_denom_iff_l1114_111447

theorem numer_greater_than_denom_iff (x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) : 
  (4 * x - 3 > 9 - 2 * x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

end numer_greater_than_denom_iff_l1114_111447


namespace riley_mistakes_l1114_111410

theorem riley_mistakes :
  ∃ R O : ℕ, R + O = 17 ∧ O = 35 - ((35 - R) / 2 + 5) ∧ R = 3 := by
  sorry

end riley_mistakes_l1114_111410


namespace phase_shift_of_sine_function_l1114_111457

theorem phase_shift_of_sine_function :
  ∀ x : ℝ, y = 3 * Real.sin (3 * x + π / 4) → (∃ φ : ℝ, φ = -π / 12) :=
by sorry

end phase_shift_of_sine_function_l1114_111457


namespace perpendicular_slope_l1114_111485

-- Define the line equation and the result we want to prove about its perpendicular slope
def line_eq (x y : ℝ) := 5 * x - 2 * y = 10

theorem perpendicular_slope : ∀ (m : ℝ), 
  (∀ (x y : ℝ), line_eq x y → y = (5 / 2) * x - 5) →
  m = -(2 / 5) :=
by
  intros m H
  -- Additional logical steps would go here
  sorry

end perpendicular_slope_l1114_111485


namespace determine_s_value_l1114_111459

def f (x : ℚ) : ℚ := abs (x - 1) - abs x

def u : ℚ := f (5 / 16)
def v : ℚ := f u
def s : ℚ := f v

theorem determine_s_value : s = 1 / 2 :=
by
  -- Proof needed here
  sorry

end determine_s_value_l1114_111459


namespace power_function_point_l1114_111409

theorem power_function_point (m n: ℝ) (h: (m - 1) * m^n = 8) : n^(-m) = 1/9 := 
  sorry

end power_function_point_l1114_111409


namespace volume_removed_percentage_l1114_111433

noncomputable def volume_of_box (length width height : ℝ) : ℝ := 
  length * width * height

noncomputable def volume_of_cube (side : ℝ) : ℝ := 
  side ^ 3

noncomputable def volume_removed (length width height side : ℝ) : ℝ :=
  8 * (volume_of_cube side)

noncomputable def percentage_removed (length width height side : ℝ) : ℝ :=
  (volume_removed length width height side) / (volume_of_box length width height) * 100

theorem volume_removed_percentage :
  percentage_removed 20 15 12 4 = 14.22 := 
by
  sorry

end volume_removed_percentage_l1114_111433


namespace range_of_a_l1114_111491

-- Define the propositions
def Proposition_p (a : ℝ) := ∀ x : ℝ, x > 0 → x + 1/x > a
def Proposition_q (a : ℝ) := ∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0

-- Define the main theorem
theorem range_of_a (a : ℝ) (h1 : ¬ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) 
(h2 : (∀ x : ℝ, x > 0 → x + 1/x > a) ∧ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) :
a ≥ 2 :=
sorry

end range_of_a_l1114_111491


namespace blue_hat_cost_is_6_l1114_111476

-- Total number of hats is 85
def total_hats : ℕ := 85

-- Number of green hats
def green_hats : ℕ := 20

-- Number of blue hats
def blue_hats : ℕ := total_hats - green_hats

-- Cost of each green hat
def cost_per_green_hat : ℕ := 7

-- Total cost for all hats
def total_cost : ℕ := 530

-- Total cost of green hats
def total_cost_green_hats : ℕ := green_hats * cost_per_green_hat

-- Total cost of blue hats
def total_cost_blue_hats : ℕ := total_cost - total_cost_green_hats

-- Cost per blue hat
def cost_per_blue_hat : ℕ := total_cost_blue_hats / blue_hats 

-- Prove that the cost of each blue hat is $6
theorem blue_hat_cost_is_6 : cost_per_blue_hat = 6 :=
by
  sorry

end blue_hat_cost_is_6_l1114_111476


namespace CE_length_l1114_111420

theorem CE_length (AF ED AE area : ℝ) (hAF : AF = 30) (hED : ED = 50) (hAE : AE = 120) (h_area : area = 7200) : 
  ∃ CE : ℝ, CE = 138 :=
by
  -- omitted proof steps
  sorry

end CE_length_l1114_111420


namespace ryan_lamps_probability_l1114_111412

theorem ryan_lamps_probability :
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_ways_to_arrange := Nat.choose total_lamps red_lamps
  let total_ways_to_turn_on := Nat.choose total_lamps 4
  let remaining_blue := blue_lamps - 1 -- Due to leftmost lamp being blue and off
  let remaining_red := red_lamps - 1 -- Due to rightmost lamp being red and on
  let remaining_red_after_middle := remaining_red - 1 -- Due to middle lamp being red and off
  let remaining_lamps := remaining_blue + remaining_red_after_middle
  let ways_to_assign_remaining_red := Nat.choose remaining_lamps remaining_red_after_middle
  let ways_to_turn_on_remaining_lamps := Nat.choose remaining_lamps 2
  let favorable_ways := ways_to_assign_remaining_red * ways_to_turn_on_remaining_lamps
  let total_possibilities := total_ways_to_arrange * total_ways_to_turn_on
  favorable_ways / total_possibilities = (10 / 490) := by
  sorry

end ryan_lamps_probability_l1114_111412


namespace Linda_total_amount_at_21_years_l1114_111481

theorem Linda_total_amount_at_21_years (P : ℝ) (r : ℝ) (n : ℕ) (initial_principal : P = 1500) (annual_rate : r = 0.03) (years : n = 21):
    P * (1 + r)^n = 2709.17 :=
by
  sorry

end Linda_total_amount_at_21_years_l1114_111481


namespace find_c_plus_d_l1114_111470

theorem find_c_plus_d (a b c d : ℤ) (h1 : a + b = 14) (h2 : b + c = 9) (h3 : a + d = 8) : c + d = 3 := 
by
  sorry

end find_c_plus_d_l1114_111470


namespace stratified_sampling_third_year_students_l1114_111425

theorem stratified_sampling_third_year_students :
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  (third_year_students : ℚ) * sampling_ratio = 20 :=
by 
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  show (third_year_students : ℚ) * sampling_ratio = 20
  sorry

end stratified_sampling_third_year_students_l1114_111425


namespace relationship_ab_c_l1114_111490

def a := 0.8 ^ 0.8
def b := 0.8 ^ 0.9
def c := 1.2 ^ 0.8

theorem relationship_ab_c : c > a ∧ a > b := 
by
  -- The proof would go here
  sorry

end relationship_ab_c_l1114_111490


namespace range_of_m_l1114_111418

noncomputable def f (m x : ℝ) : ℝ := x^3 + m * x^2 + (m + 6) * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, ∃ c : ℝ, (f m x) ≤ c ∧ (f m y) ≥ (f m x) ∧ ∀ z : ℝ, f m z ≥ f m x ∧ f m z ≤ c) ↔ (m < -3 ∨ m > 6) :=
by
  sorry

end range_of_m_l1114_111418


namespace yuna_initial_pieces_l1114_111495

variable (Y : ℕ)

theorem yuna_initial_pieces
  (namjoon_initial : ℕ := 250)
  (given_pieces : ℕ := 60)
  (namjoon_after : namjoon_initial - given_pieces = Y + given_pieces - 20) :
  Y = 150 :=
by
  sorry

end yuna_initial_pieces_l1114_111495


namespace three_2x2_squares_exceed_100_l1114_111452

open BigOperators

noncomputable def sum_of_1_to_64 : ℕ :=
  (64 * (64 + 1)) / 2

theorem three_2x2_squares_exceed_100 :
  ∀ (s : Fin 16 → ℕ),
    (∑ i, s i = sum_of_1_to_64) →
    (∀ i j, i ≠ j → s i = s j ∨ s i > s j ∨ s i < s j) →
    (∃ i₁ i₂ i₃, i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₁ ≠ i₃ ∧ s i₁ > 100 ∧ s i₂ > 100 ∧ s i₃ > 100) := sorry

end three_2x2_squares_exceed_100_l1114_111452


namespace pradeep_passing_percentage_l1114_111443

theorem pradeep_passing_percentage (score failed_by max_marks : ℕ) :
  score = 185 → failed_by = 25 → max_marks = 600 →
  ((score + failed_by) / max_marks : ℚ) * 100 = 35 :=
by
  intros h_score h_failed_by h_max_marks
  sorry

end pradeep_passing_percentage_l1114_111443


namespace green_pill_cost_is_21_l1114_111427

-- Definitions based on conditions
def number_of_days : ℕ := 21
def total_cost : ℕ := 819
def daily_cost : ℕ := total_cost / number_of_days
def green_pill_cost (pink_pill_cost : ℕ) : ℕ := pink_pill_cost + 3

-- Given pink pill cost is x, then green pill cost is x + 3
-- We need to prove that for some x, the daily cost of the pills equals 39, and thus green pill cost is 21

theorem green_pill_cost_is_21 (pink_pill_cost : ℕ) (h : daily_cost = (green_pill_cost pink_pill_cost) + pink_pill_cost) :
    green_pill_cost pink_pill_cost = 21 :=
by
  sorry

end green_pill_cost_is_21_l1114_111427


namespace sin_cos_sixth_power_l1114_111417

theorem sin_cos_sixth_power (θ : ℝ) 
  (h : Real.sin (3 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
  sorry

end sin_cos_sixth_power_l1114_111417


namespace cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l1114_111411

theorem cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2 : 
  Real.cos (- (11 / 4) * Real.pi) = - Real.sqrt 2 / 2 := 
sorry

end cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l1114_111411


namespace items_from_B_l1114_111471

noncomputable def totalItems : ℕ := 1200
noncomputable def ratioA : ℕ := 3
noncomputable def ratioB : ℕ := 4
noncomputable def ratioC : ℕ := 5
noncomputable def totalRatio : ℕ := ratioA + ratioB + ratioC
noncomputable def sampledItems : ℕ := 60
noncomputable def numberB := sampledItems * ratioB / totalRatio

theorem items_from_B :
  numberB = 20 :=
by
  sorry

end items_from_B_l1114_111471


namespace subtract_fractions_l1114_111468

theorem subtract_fractions (p q : ℚ) (h₁ : 4 / p = 8) (h₂ : 4 / q = 18) : p - q = 5 / 18 := 
by 
  sorry

end subtract_fractions_l1114_111468


namespace abc_is_cube_l1114_111464

theorem abc_is_cube (a b c : ℤ) (h : (a:ℚ) / (b:ℚ) + (b:ℚ) / (c:ℚ) + (c:ℚ) / (a:ℚ) = 3) : ∃ x : ℤ, abc = x^3 :=
by
  sorry

end abc_is_cube_l1114_111464


namespace cost_of_camel_is_6000_l1114_111400

noncomputable def cost_of_camel : ℕ := 6000

variables (C H O E : ℕ)
variables (cost_of_camel_rs cost_of_horses cost_of_oxen cost_of_elephants : ℕ)

-- Conditions
axiom cond1 : 10 * C = 24 * H
axiom cond2 : 16 * H = 4 * O
axiom cond3 : 6 * O = 4 * E
axiom cond4 : 10 * E = 150000

theorem cost_of_camel_is_6000
    (cond1 : 10 * C = 24 * H)
    (cond2 : 16 * H = 4 * O)
    (cond3 : 6 * O = 4 * E)
    (cond4 : 10 * E = 150000) :
  cost_of_camel = 6000 := 
sorry

end cost_of_camel_is_6000_l1114_111400


namespace leak_empty_time_l1114_111426

-- Define the given conditions
def tank_volume := 2160 -- Tank volume in litres
def inlet_rate := 6 * 60 -- Inlet rate in litres per hour
def combined_empty_time := 12 -- Time in hours to empty the tank with the inlet on

-- Define the derived conditions
def net_rate := tank_volume / combined_empty_time -- Net rate of emptying in litres per hour

-- Define the rate of leakage
def leak_rate := inlet_rate - net_rate -- Rate of leak in litres per hour

-- Prove the main statement
theorem leak_empty_time : (2160 / leak_rate) = 12 :=
by
  unfold leak_rate
  exact sorry

end leak_empty_time_l1114_111426


namespace wheel_speed_is_12_mph_l1114_111465

theorem wheel_speed_is_12_mph
  (r : ℝ) -- speed in miles per hour
  (C : ℝ := 15 / 5280) -- circumference in miles
  (H1 : ∃ t, r * t = C * 3600) -- initial condition that speed times time for one rotation equals 15/5280 miles in seconds
  (H2 : ∃ t, (r + 7) * (t - 1/21600) = C * 3600) -- condition that speed increases by 7 mph when time shortens by 1/6 second
  : r = 12 :=
sorry

end wheel_speed_is_12_mph_l1114_111465


namespace compare_exponential_functions_l1114_111488

theorem compare_exponential_functions (x : ℝ) (hx1 : 0 < x) :
  0.4^4 < 1 ∧ 1 < 4^0.4 :=
by sorry

end compare_exponential_functions_l1114_111488


namespace cost_of_five_dozen_l1114_111421

noncomputable def price_per_dozen (total_cost : ℝ) (num_dozen : ℕ) : ℝ :=
  total_cost / num_dozen

noncomputable def total_cost (price_per_dozen : ℝ) (num_dozen : ℕ) : ℝ :=
  price_per_dozen * num_dozen

theorem cost_of_five_dozen (total_cost_threedozens : ℝ := 28.20) (num_threedozens : ℕ := 3) (num_fivedozens : ℕ := 5) :
  total_cost (price_per_dozen total_cost_threedozens num_threedozens) num_fivedozens = 47.00 :=
  by sorry

end cost_of_five_dozen_l1114_111421


namespace find_x_solution_l1114_111407

theorem find_x_solution (x : ℚ) : (∀ y : ℚ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by
  sorry

end find_x_solution_l1114_111407


namespace sock_pair_selection_l1114_111482

def num_white_socks : Nat := 5
def num_brown_socks : Nat := 5
def num_blue_socks : Nat := 3

def white_odd_positions : List Nat := [1, 3, 5]
def white_even_positions : List Nat := [2, 4]

def brown_odd_positions : List Nat := [1, 3, 5]
def brown_even_positions : List Nat := [2, 4]

def blue_odd_positions : List Nat := [1, 3]
def blue_even_positions : List Nat := [2]

noncomputable def count_pairs : Nat :=
  let white_brown := (white_odd_positions.length * brown_odd_positions.length) +
                     (white_even_positions.length * brown_even_positions.length)
  
  let brown_blue := (brown_odd_positions.length * blue_odd_positions.length) +
                    (brown_even_positions.length * blue_even_positions.length)

  let white_blue := (white_odd_positions.length * blue_odd_positions.length) +
                    (white_even_positions.length * blue_even_positions.length)

  white_brown + brown_blue + white_blue

theorem sock_pair_selection :
  count_pairs = 29 :=
by
  sorry

end sock_pair_selection_l1114_111482


namespace part1_part2_l1114_111422

-- Definition of the function f
def f (a x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- First Proof Statement: Inequality for a = 2
theorem part1 : ∀ x : ℝ, - (1 : ℝ) / 3 ≤ x ∧ x ≤ 5 → f 2 x ≤ 1 :=
by
  sorry

-- Second Proof Statement: Range for a such that -4 ≤ f(x) ≤ 4 for all x ∈ ℝ
theorem part2 : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4 ↔ a = 1 ∨ a = -1 :=
by
  sorry

end part1_part2_l1114_111422


namespace find_number_of_women_l1114_111439

-- Define the work rate variables and the equations from conditions
variables (m w : ℝ) (x : ℝ)

-- Define the first condition
def condition1 : Prop := 3 * m + x * w = 6 * m + 2 * w

-- Define the second condition
def condition2 : Prop := 4 * m + 2 * w = (5 / 7) * (3 * m + x * w)

-- The theorem stating that, given the above conditions, x must be 23
theorem find_number_of_women (hmw : m = 7 * w) (h1 : condition1 m w x) (h2 : condition2 m w x) : x = 23 :=
sorry

end find_number_of_women_l1114_111439


namespace gcd_n3_plus_16_n_plus_4_l1114_111438

/-- For a given positive integer n > 2^4, the greatest common divisor of n^3 + 16 and n + 4 is 1. -/
theorem gcd_n3_plus_16_n_plus_4 (n : ℕ) (h : n > 2^4) : Nat.gcd (n^3 + 16) (n + 4) = 1 := by
  sorry

end gcd_n3_plus_16_n_plus_4_l1114_111438


namespace total_cost_pencils_l1114_111461

theorem total_cost_pencils
  (boxes : ℕ)
  (cost_per_box : ℕ → ℕ → ℕ)
  (price_regular : ℕ)
  (price_bulk : ℕ)
  (box_size : ℕ)
  (bulk_threshold : ℕ)
  (total_pencils : ℕ) :
  total_pencils = 3150 →
  box_size = 150 →
  price_regular = 40 →
  price_bulk = 35 →
  bulk_threshold = 2000 →
  boxes = (total_pencils + box_size - 1) / box_size →
  (total_pencils > bulk_threshold → cost_per_box boxes price_bulk = boxes * price_bulk) →
  (total_pencils ≤ bulk_threshold → cost_per_box boxes price_regular = boxes * price_regular) →
  total_pencils > bulk_threshold →
  cost_per_box boxes price_bulk = 735 :=
by
  intro h_total_pencils
  intro h_box_size
  intro h_price_regular
  intro h_price_bulk
  intro h_bulk_threshold
  intro h_boxes
  intro h_cost_bulk
  intro h_cost_regular
  intro h_bulk_discount_passt
  -- sorry statement as we don't provide the actual proof here
  sorry

end total_cost_pencils_l1114_111461


namespace Kayla_total_items_l1114_111478

-- Define the given conditions
def Theresa_chocolate_bars := 12
def Theresa_soda_cans := 18
def twice (x : Nat) := 2 * x

-- Define the unknowns
def Kayla_chocolate_bars := Theresa_chocolate_bars / 2
def Kayla_soda_cans := Theresa_soda_cans / 2

-- Final proof statement to be shown
theorem Kayla_total_items :
  Kayla_chocolate_bars + Kayla_soda_cans = 15 := 
by
  simp [Kayla_chocolate_bars, Kayla_soda_cans]
  norm_num
  sorry

end Kayla_total_items_l1114_111478


namespace girls_first_half_l1114_111423

theorem girls_first_half (total_students boys_first_half girls_first_half boys_second_half girls_second_half boys_whole_year : ℕ)
  (h1: total_students = 56)
  (h2: boys_first_half = 25)
  (h3: girls_first_half = 15)
  (h4: boys_second_half = 26)
  (h5: girls_second_half = 25)
  (h6: boys_whole_year = 23) : 
  ∃ girls_first_half_only : ℕ, girls_first_half_only = 3 :=
by {
  sorry
}

end girls_first_half_l1114_111423


namespace ratio_monkeys_camels_l1114_111455

-- Definitions corresponding to conditions
variables (zebras camels monkeys giraffes : ℕ)
variables (multiple : ℕ)

-- Conditions
def condition1 := zebras = 12
def condition2 := camels = zebras / 2
def condition3 := monkeys = camels * multiple
def condition4 := giraffes = 2
def condition5 := monkeys = giraffes + 22

-- Question: What is the ratio of monkeys to camels? Prove it is 4:1 given the conditions.
theorem ratio_monkeys_camels (zebras camels monkeys giraffes multiple : ℕ) 
  (h1 : condition1 zebras) 
  (h2 : condition2 zebras camels)
  (h3 : condition3 camels monkeys multiple)
  (h4 : condition4 giraffes)
  (h5 : condition5 monkeys giraffes) :
  multiple = 4 :=
sorry

end ratio_monkeys_camels_l1114_111455


namespace avg_weight_of_children_is_138_l1114_111424

-- Define the average weight of boys and girls
def average_weight_of_boys := 150
def number_of_boys := 6
def average_weight_of_girls := 120
def number_of_girls := 4

-- Calculate total weights and average weight of all children
noncomputable def total_weight_of_boys := number_of_boys * average_weight_of_boys
noncomputable def total_weight_of_girls := number_of_girls * average_weight_of_girls
noncomputable def total_weight_of_children := total_weight_of_boys + total_weight_of_girls
noncomputable def number_of_children := number_of_boys + number_of_girls
noncomputable def average_weight_of_children := total_weight_of_children / number_of_children

-- Lean statement to prove the average weight of all children is 138 pounds
theorem avg_weight_of_children_is_138 : average_weight_of_children = 138 := by
    sorry

end avg_weight_of_children_is_138_l1114_111424


namespace toy_sword_cost_l1114_111484

theorem toy_sword_cost (L S : ℕ) (play_dough_cost total_cost : ℕ) :
    L = 250 →
    play_dough_cost = 35 →
    total_cost = 1940 →
    3 * L + 7 * S + 10 * play_dough_cost = total_cost →
    S = 120 :=
by
  intros hL h_play_dough_cost h_total_cost h_eq
  sorry

end toy_sword_cost_l1114_111484


namespace expand_expression_l1114_111402

theorem expand_expression (x : ℝ) : (15 * x + 17 + 3) * (3 * x) = 45 * x^2 + 60 * x :=
by
  sorry

end expand_expression_l1114_111402


namespace minnie_penny_time_difference_l1114_111408

noncomputable def minnie_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def penny_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def break_time (minutes: ℝ) := minutes / 60

noncomputable def minnie_total_time :=
  minnie_time_uphill 12 6 + minnie_time_downhill 18 25 + minnie_time_flat 25 18

noncomputable def penny_total_time :=
  penny_time_flat 25 25 + penny_time_downhill 12 35 + 
  penny_time_uphill 18 12 + break_time 10

noncomputable def time_difference := (minnie_total_time - penny_total_time) * 60

theorem minnie_penny_time_difference :
  time_difference = 66 := by
  sorry

end minnie_penny_time_difference_l1114_111408


namespace difference_in_cents_l1114_111450

theorem difference_in_cents (pennies dimes : ℕ) (h : pennies + dimes = 5050) (hpennies : 1 ≤ pennies) (hdimes : 1 ≤ dimes) : 
  let total_value := pennies + 10 * dimes
  let max_value := 50500 - 9 * 1
  let min_value := 50500 - 9 * 5049
  max_value - min_value = 45432 := 
by 
  -- proof goes here
  sorry

end difference_in_cents_l1114_111450


namespace length_of_chord_l1114_111475

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the line y = x - 1 with slope 1 passing through the focus (1, 0)
def line (x y : ℝ) : Prop :=
  y = x - 1

-- Prove that the length of the chord |AB| is 8
theorem length_of_chord 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h3 : line x1 y1) 
  (h4 : line x2 y2) : 
  abs (x2 - x1) = 8 :=
sorry

end length_of_chord_l1114_111475


namespace speed_difference_l1114_111406

theorem speed_difference (distance : ℕ) (time_jordan time_alex : ℕ) (h_distance : distance = 12) (h_time_jordan : time_jordan = 10) (h_time_alex : time_alex = 15) :
  (distance / (time_jordan / 60) - distance / (time_alex / 60) = 24) := by
  -- Lean code to correctly parse and understand the natural numbers, division, and maintain the theorem structure.
  sorry

end speed_difference_l1114_111406


namespace ray_reflection_and_distance_l1114_111486

-- Define the initial conditions
def pointA : ℝ × ℝ := (-3, 3)
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Definitions of the lines for incident and reflected rays
def incident_ray_line (x y : ℝ) : Prop := 4*x + 3*y + 3 = 0
def reflected_ray_line (x y : ℝ) : Prop := 3*x + 4*y - 3 = 0

-- Distance traveled by the ray
def distance_traveled (A T : ℝ × ℝ) := 7

theorem ray_reflection_and_distance :
  ∃ (x₁ y₁ : ℝ), incident_ray_line x₁ y₁ ∧ reflected_ray_line x₁ y₁ ∧ circleC_eq x₁ y₁ ∧ 
  (∀ (P : ℝ × ℝ), P = pointA → distance_traveled P (x₁, y₁) = 7) :=
sorry

end ray_reflection_and_distance_l1114_111486


namespace smallest_YZ_minus_XZ_l1114_111498

theorem smallest_YZ_minus_XZ 
  (XZ YZ XY : ℕ)
  (h_sum : XZ + YZ + XY = 3001)
  (h_order : XZ < YZ ∧ YZ ≤ XY)
  (h_triangle_ineq1 : XZ + YZ > XY)
  (h_triangle_ineq2 : XZ + XY > YZ)
  (h_triangle_ineq3 : YZ + XY > XZ) :
  ∃ XZ YZ XY : ℕ, YZ - XZ = 1 := sorry

end smallest_YZ_minus_XZ_l1114_111498


namespace steve_family_time_l1114_111445

theorem steve_family_time :
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours * assignments_fraction
  let total_activity_hours := sleeping_hours + school_hours + assignments_hours
  day_hours - total_activity_hours = 10 :=
by
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours *  assignments_fraction
  let total_activity_hours := sleeping_hours +
                              school_hours + 
                              assignments_hours
  show day_hours - total_activity_hours = 10
  sorry

end steve_family_time_l1114_111445


namespace anna_score_correct_l1114_111449

-- Given conditions
def correct_answers : ℕ := 17
def incorrect_answers : ℕ := 6
def unanswered_questions : ℕ := 7
def point_per_correct : ℕ := 1
def point_per_incorrect : ℕ := 0
def deduction_per_unanswered : ℤ := -1 / 2

-- Proving the score
theorem anna_score_correct : 
  correct_answers * point_per_correct + incorrect_answers * point_per_incorrect + unanswered_questions * deduction_per_unanswered = 27 / 2 :=
by
  sorry

end anna_score_correct_l1114_111449


namespace simplify_and_evaluate_expression_l1114_111436

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = 2 ∨ x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  ((1 / (x:ℚ) - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1))) = (1 / 2) :=
by
  -- Skipping the proof
  sorry

end simplify_and_evaluate_expression_l1114_111436


namespace graph_shift_cos_function_l1114_111413

theorem graph_shift_cos_function (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = 2 * Real.cos (π * x / 3 + φ)) ∧ 
  (∃ x, f x = 0 ∧ x = 2) ∧ 
  (f 1 > f 3) →
  (∀ x, f x = 2 * Real.cos (π * (x - 1/2) / 3)) :=
by
  sorry

end graph_shift_cos_function_l1114_111413


namespace no_int_solutions_5x2_minus_4y2_eq_2017_l1114_111467

theorem no_int_solutions_5x2_minus_4y2_eq_2017 :
  ¬ ∃ x y : ℤ, 5 * x^2 - 4 * y^2 = 2017 :=
by
  -- The detailed proof goes here
  sorry

end no_int_solutions_5x2_minus_4y2_eq_2017_l1114_111467


namespace fraction_of_students_received_As_l1114_111419

/-- Assume A is the fraction of students who received A's,
and B is the fraction of students who received B's,
and T is the total fraction of students who received either A's or B's. -/
theorem fraction_of_students_received_As
  (A B T : ℝ)
  (hB : B = 0.2)
  (hT : T = 0.9)
  (h : A + B = T) :
  A = 0.7 := 
by
  -- establishing the proof steps
  sorry

end fraction_of_students_received_As_l1114_111419


namespace possible_values_of_a_l1114_111479

-- Declare the sets M and N based on given conditions.
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define a proof where the set of possible values for a is {-1, 0, 2/3}
theorem possible_values_of_a : 
  {a : ℝ | N a ⊆ M} = {-1, 0, 2 / 3} := 
by 
  sorry

end possible_values_of_a_l1114_111479


namespace Tom_runs_60_miles_in_a_week_l1114_111453

variable (days_per_week : ℕ) (hours_per_day : ℝ) (speed : ℝ)
variable (h_days_per_week : days_per_week = 5)
variable (h_hours_per_day : hours_per_day = 1.5)
variable (h_speed : speed = 8)

theorem Tom_runs_60_miles_in_a_week : (days_per_week * hours_per_day * speed) = 60 := by
  sorry

end Tom_runs_60_miles_in_a_week_l1114_111453


namespace solve_for_s_l1114_111431

theorem solve_for_s (s t : ℚ) (h1 : 15 * s + 7 * t = 210) (h2 : t = 3 * s) : s = 35 / 6 := 
by
  sorry

end solve_for_s_l1114_111431


namespace number_of_minibusses_l1114_111466

def total_students := 156
def students_per_van := 10
def students_per_minibus := 24
def number_of_vans := 6

theorem number_of_minibusses : (total_students - number_of_vans * students_per_van) / students_per_minibus = 4 :=
by
  sorry

end number_of_minibusses_l1114_111466


namespace average_age_of_adults_l1114_111472

theorem average_age_of_adults 
  (total_members : ℕ)
  (avg_age_total : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_adults : ℕ)
  (avg_age_girls : ℕ)
  (avg_age_boys : ℕ)
  (total_sum_ages : ℕ := total_members * avg_age_total)
  (sum_ages_girls : ℕ := num_girls * avg_age_girls)
  (sum_ages_boys : ℕ := num_boys * avg_age_boys)
  (sum_ages_adults : ℕ := total_sum_ages - sum_ages_girls - sum_ages_boys)
  : (num_adults = 10) → (avg_age_total = 20) → (num_girls = 30) → (avg_age_girls = 18) → (num_boys = 20) → (avg_age_boys = 22) → (total_sum_ages = 1200) → (sum_ages_girls = 540) → (sum_ages_boys = 440) → (sum_ages_adults = 220) → (sum_ages_adults / num_adults = 22) :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end average_age_of_adults_l1114_111472


namespace nth_equation_l1114_111414

theorem nth_equation (n : ℕ) (h : 0 < n) : (- (n : ℤ)) * (n : ℝ) / (n + 1) = - (n : ℤ) + (n : ℝ) / (n + 1) :=
sorry

end nth_equation_l1114_111414


namespace capital_payment_l1114_111463

theorem capital_payment (m : ℕ) (hm : m ≥ 3) : 
  ∃ d : ℕ, d = (1000 * (3^m - 2^(m-1))) / (3^m - 2^m) 
  ∧ (∃ a : ℕ, a = 4000 ∧ a = ((3/2)^(m-1) * (3000 - 3 * d) + 2 * d)) := 
by
  sorry

end capital_payment_l1114_111463


namespace find_x_l1114_111499

theorem find_x (x : ℚ) : (8 + 10 + 22) / 3 = (15 + x) / 2 → x = 35 / 3 :=
by
  sorry

end find_x_l1114_111499


namespace bikes_in_parking_lot_l1114_111415

theorem bikes_in_parking_lot (C : ℕ) (Total_Wheels : ℕ) (Wheels_per_car : ℕ) (Wheels_per_bike : ℕ) (h1 : C = 14) (h2 : Total_Wheels = 76) (h3 : Wheels_per_car = 4) (h4 : Wheels_per_bike = 2) : 
  ∃ B : ℕ, 4 * C + 2 * B = Total_Wheels ∧ B = 10 :=
by
  sorry

end bikes_in_parking_lot_l1114_111415


namespace smallest_two_digit_number_l1114_111474

theorem smallest_two_digit_number :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧
            n % 12 = 0 ∧
            n % 5 = 4 ∧
            ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ m % 12 = 0 ∧ m % 5 = 4 → n ≤ m :=
  by {
  -- proof shows the mathematical statement is true
  sorry
}

end smallest_two_digit_number_l1114_111474


namespace total_skips_correct_l1114_111469

def S (n : ℕ) : ℕ := n^2 + n

def TotalSkips5 : ℕ :=
  S 1 + S 2 + S 3 + S 4 + S 5

def Skips6 : ℕ :=
  2 * S 6

theorem total_skips_correct : TotalSkips5 + Skips6 = 154 :=
by
  -- proof goes here
  sorry

end total_skips_correct_l1114_111469


namespace intersection_point_of_lines_l1114_111437

theorem intersection_point_of_lines :
  ∃ x y : ℚ, 
    (y = -3 * x + 4) ∧ 
    (y = (1 / 3) * x + 1) ∧ 
    x = 9 / 10 ∧ 
    y = 13 / 10 :=
by sorry

end intersection_point_of_lines_l1114_111437


namespace bacteria_growth_time_l1114_111458

theorem bacteria_growth_time (initial_bacteria : ℕ) (final_bacteria : ℕ) (doubling_time : ℕ) :
  initial_bacteria = 1000 →
  final_bacteria = 128000 →
  doubling_time = 3 →
  (∃ t : ℕ, final_bacteria = initial_bacteria * 2 ^ (t / doubling_time) ∧ t = 21) :=
by
  sorry

end bacteria_growth_time_l1114_111458


namespace students_neither_correct_l1114_111483

-- Define the total number of students and the numbers for chemistry, biology, and both
def total_students := 75
def chemistry_students := 42
def biology_students := 33
def both_subject_students := 18

-- Define a function to calculate the number of students taking neither chemistry nor biology
def students_neither : ℕ :=
  total_students - ((chemistry_students - both_subject_students) 
                    + (biology_students - both_subject_students) 
                    + both_subject_students)

-- Theorem stating that the number of students taking neither chemistry nor biology is as expected
theorem students_neither_correct : students_neither = 18 :=
  sorry

end students_neither_correct_l1114_111483


namespace negation_of_p_l1114_111456

open Real

-- Define the statement to be negated
def p := ∀ x : ℝ, -π/2 < x ∧ x < π/2 → tan x > 0

-- Define the negation of the statement
def not_p := ∃ x_0 : ℝ, -π/2 < x_0 ∧ x_0 < π/2 ∧ tan x_0 ≤ 0

-- Theorem stating that the negation of p is not_p
theorem negation_of_p : ¬ p ↔ not_p :=
sorry

end negation_of_p_l1114_111456


namespace chairs_problem_l1114_111493

theorem chairs_problem (B G W : ℕ) 
  (h1 : G = 3 * B) 
  (h2 : W = B + G - 13) 
  (h3 : B + G + W = 67) : 
  B = 10 :=
by
  sorry

end chairs_problem_l1114_111493


namespace cos_sum_formula_l1114_111480

open Real

theorem cos_sum_formula (A B C : ℝ) (h1 : sin A + sin B + sin C = 0) (h2 : cos A + cos B + cos C = 0) :
  cos (A - B) + cos (B - C) + cos (C - A) = -3 / 2 :=
by
  sorry

end cos_sum_formula_l1114_111480


namespace two_students_follow_all_celebrities_l1114_111430

theorem two_students_follow_all_celebrities :
  ∀ (students : Finset ℕ) (celebrities_followers : ℕ → Finset ℕ),
    (students.card = 120) →
    (∀ c : ℕ, c < 10 → (celebrities_followers c).card ≥ 85 ∧ (celebrities_followers c) ⊆ students) →
    ∃ (s1 s2 : ℕ), s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧
      (∀ c : ℕ, c < 10 → (s1 ∈ celebrities_followers c ∨ s2 ∈ celebrities_followers c)) :=
by
  intros students celebrities_followers h_students_card h_followers_cond
  sorry

end two_students_follow_all_celebrities_l1114_111430


namespace range_of_k_l1114_111448

theorem range_of_k
  (x y k : ℝ)
  (h1 : 3 * x + y = k + 1)
  (h2 : x + 3 * y = 3)
  (h3 : 0 < x + y)
  (h4 : x + y < 1) :
  -4 < k ∧ k < 0 :=
sorry

end range_of_k_l1114_111448


namespace find_x_squared_inv_x_squared_l1114_111440

theorem find_x_squared_inv_x_squared (x : ℝ) (h : x^3 + 1/x^3 = 110) : x^2 + 1/x^2 = 23 :=
sorry

end find_x_squared_inv_x_squared_l1114_111440


namespace sum_of_all_possible_values_of_g_11_l1114_111497

def f (x : ℝ) : ℝ := x^2 - 6 * x + 14

def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g_11 :
  (∀ x : ℝ, f x = 11 → g x = 13 ∨ g x = 7) →
  (13 + 7 = 20) := by
  intros h
  sorry

end sum_of_all_possible_values_of_g_11_l1114_111497


namespace range_of_x_l1114_111496

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) :
  -1 ≤ x ∧ x < 5 / 4 :=
sorry

end range_of_x_l1114_111496


namespace Jung_age_is_26_l1114_111416

-- Define the ages of Li, Zhang, and Jung
def Li : ℕ := 12
def Zhang : ℕ := 2 * Li
def Jung : ℕ := Zhang + 2

-- The goal is to prove Jung's age is 26 years
theorem Jung_age_is_26 : Jung = 26 :=
by
  -- Placeholder for the proof
  sorry

end Jung_age_is_26_l1114_111416


namespace three_digit_decimal_bounds_l1114_111444

def is_rounded_half_up (x : ℝ) (y : ℝ) : Prop :=
  (y - 0.005 ≤ x) ∧ (x < y + 0.005)

theorem three_digit_decimal_bounds :
  ∃ (x : ℝ), (8.725 ≤ x) ∧ (x ≤ 8.734) ∧ is_rounded_half_up x 8.73 :=
by
  sorry

end three_digit_decimal_bounds_l1114_111444


namespace like_terms_sum_l1114_111435

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 2) (h2 : n = 3) : m + n = 4 :=
sorry

end like_terms_sum_l1114_111435


namespace num_adult_tickets_l1114_111487

variables (A C : ℕ)

def num_tickets (A C : ℕ) : Prop := A + C = 900
def total_revenue (A C : ℕ) : Prop := 7 * A + 4 * C = 5100

theorem num_adult_tickets : ∃ A, ∃ C, num_tickets A C ∧ total_revenue A C ∧ A = 500 := 
by
  sorry

end num_adult_tickets_l1114_111487


namespace propA_necessary_but_not_sufficient_l1114_111405

variable {a : ℝ}

-- Proposition A: ∀ x ∈ ℝ, ax² + 2ax + 1 > 0
def propA (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0

-- Proposition B: 0 < a < 1
def propB (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem statement: Proposition A is necessary but not sufficient for Proposition B
theorem propA_necessary_but_not_sufficient (a : ℝ) :
  (propB a → propA a) ∧
  (propA a → propB a → False) :=
by
  sorry

end propA_necessary_but_not_sufficient_l1114_111405


namespace total_airflow_correct_l1114_111429

def airflow_fan_A : ℕ := 10 * 10 * 60 * 7
def airflow_fan_B : ℕ := 15 * 20 * 60 * 5
def airflow_fan_C : ℕ := 25 * 30 * 60 * 5
def airflow_fan_D : ℕ := 20 * 15 * 60 * 2
def airflow_fan_E : ℕ := 30 * 60 * 60 * 6

def total_airflow : ℕ :=
  airflow_fan_A + airflow_fan_B + airflow_fan_C + airflow_fan_D + airflow_fan_E

theorem total_airflow_correct : total_airflow = 1041000 := by
  sorry

end total_airflow_correct_l1114_111429


namespace find_largest_number_l1114_111434

theorem find_largest_number
  (a b c d : ℕ)
  (h1 : a + b + c = 222)
  (h2 : a + b + d = 208)
  (h3 : a + c + d = 197)
  (h4 : b + c + d = 180) :
  max a (max b (max c d)) = 89 :=
by
  sorry

end find_largest_number_l1114_111434


namespace decagon_diagonals_intersect_probability_l1114_111460

theorem decagon_diagonals_intersect_probability :
  let n := 10  -- number of vertices in decagon
  let diagonals := n * (n - 3) / 2  -- number of diagonals in decagon
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2  -- ways to choose 2 diagonals from diagonals
  let ways_choose_4 := Nat.choose 10 4  -- ways to choose 4 vertices from 10
  let probability := (4 * ways_choose_4) / pairs_diagonals  -- four vertices chosen determine two intersecting diagonals forming a convex quadrilateral
  probability = (210 / 595) := by
  -- Definitions (diagonals, pairs_diagonals, ways_choose_4) are directly used as hypothesis

  sorry  -- skipping the proof

end decagon_diagonals_intersect_probability_l1114_111460


namespace flight_distance_l1114_111441

theorem flight_distance (D : ℝ) :
  let t_out := D / 300
  let t_return := D / 500
  t_out + t_return = 8 -> D = 1500 :=
by
  intro h
  sorry

end flight_distance_l1114_111441


namespace judson_contribution_l1114_111489

theorem judson_contribution (J K C : ℝ) (hK : K = 1.20 * J) (hC : C = K + 200) (h_total : J + K + C = 1900) : J = 500 :=
by
  -- This is where the proof would go, but we are skipping it as per the instructions.
  sorry

end judson_contribution_l1114_111489


namespace imo_2007_p6_l1114_111454

theorem imo_2007_p6 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ∃ k : ℕ, (x = 11 * k^2) ∧ (y = 11 * k) ↔
  ∃ k : ℕ, (∃ k₁ : ℤ, k₁ = (x^2 * y + x + y) / (x * y^2 + y + 11)) :=
sorry

end imo_2007_p6_l1114_111454


namespace perfect_square_trinomial_l1114_111401

variable (x y : ℝ)

theorem perfect_square_trinomial (a : ℝ) :
  (∃ b c : ℝ, 4 * x^2 - (a - 1) * x * y + 9 * y^2 = (b * x + c * y) ^ 2) ↔ 
  (a = 13 ∨ a = -11) := 
by
  sorry

end perfect_square_trinomial_l1114_111401


namespace david_profit_l1114_111432

def weight : ℝ := 50
def cost : ℝ := 50
def price_per_kg : ℝ := 1.20
def total_earnings : ℝ := weight * price_per_kg
def profit : ℝ := total_earnings - cost

theorem david_profit : profit = 10 := by
  sorry

end david_profit_l1114_111432


namespace problem_1_part1_problem_1_part2_problem_2_l1114_111403

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 + 2 * cos (x) ^ 2

theorem problem_1_part1 : (∃ T > 0, ∀ x, f (x + T) = f x) := sorry

theorem problem_1_part2 : (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), ∀ y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), x < y → f x > f y) := sorry

noncomputable def S_triangle (A B C : ℝ) (a b c : ℝ) : ℝ := 1 / 2 * b * c * sin A

theorem problem_2 :
  ∀ (A B C a b c : ℝ), f A = 4 → b = 1 → S_triangle A B C a b c = sqrt 3 / 2 →
    a^2 = b^2 + c^2 - 2 * b * c * cos A → a = sqrt 3 := sorry

end problem_1_part1_problem_1_part2_problem_2_l1114_111403


namespace proof_statement_l1114_111428

-- Define the initial dimensions and areas
def initial_length : ℕ := 7
def initial_width : ℕ := 5

-- Shortened dimensions by one side and the corresponding area condition
def shortened_new_width : ℕ := 3
def shortened_area : ℕ := 21

-- Define the task
def task_statement : Prop :=
  (initial_length - 2) * initial_width = shortened_area ∧
  (initial_width - 2) * initial_length = shortened_area →
  (initial_length - 2) * (initial_width - 2) = 25

theorem proof_statement : task_statement :=
by {
  sorry -- Proof goes here
}

end proof_statement_l1114_111428


namespace function_decreasing_iff_l1114_111462

theorem function_decreasing_iff (a : ℝ) :
  (0 < a ∧ a < 1) ∧ a ≤ 1/4 ↔ (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end function_decreasing_iff_l1114_111462


namespace inequality_solution_l1114_111494

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem inequality_solution (a b : ℝ) 
  (h1 : ∀ (x : ℝ), f a b x > 0 ↔ -1 < x ∧ x < 3) :
  ∀ (x : ℝ), f a b (-2 * x) < 0 ↔ x < -3 / 2 ∨ x > 1 / 2 :=
sorry

end inequality_solution_l1114_111494


namespace least_number_remainder_l1114_111492

theorem least_number_remainder (n : ℕ) (hn : n = 115) : n % 38 = 1 ∧ n % 3 = 1 := by
  sorry

end least_number_remainder_l1114_111492


namespace inequality_solution_set_range_of_a_l1114_111442

def f (x : ℝ) : ℝ := abs (3*x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x < 4 - abs (x - 1) } = { x : ℝ | -5/4 < x ∧ x < 1/2 } :=
by 
  sorry

theorem range_of_a (a : ℝ) (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) 
  (h4 : ∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) : 
  0 < a ∧ a ≤ 10/3 :=
by 
  sorry

end inequality_solution_set_range_of_a_l1114_111442


namespace ratio_of_terms_l1114_111404

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem ratio_of_terms
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (S T : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = geometric_sum (a 1) (a 2) n)
  (h₁ : ∀ n : ℕ, T n = geometric_sum (b 1) (b 2) n)
  (h₂ : ∀ n : ℕ, n > 0 → S n / T n = (3 ^ n + 1) / 4) :
  a 3 / b 4 = 3 := 
sorry

end ratio_of_terms_l1114_111404


namespace diameter_of_larger_circle_l1114_111477

theorem diameter_of_larger_circle (R r D : ℝ) 
  (h1 : R^2 - r^2 = 25) 
  (h2 : D = 2 * R) : 
  D = Real.sqrt (100 + 4 * r^2) := 
by 
  sorry

end diameter_of_larger_circle_l1114_111477


namespace shopkeeper_total_cards_l1114_111446

-- Definition of the number of cards in a complete deck
def cards_in_deck : Nat := 52

-- Definition of the number of complete decks the shopkeeper has
def number_of_decks : Nat := 3

-- Definition of the additional cards the shopkeeper has
def additional_cards : Nat := 4

-- The total number of cards the shopkeeper should have
def total_cards : Nat := number_of_decks * cards_in_deck + additional_cards

-- Theorem statement to prove the total number of cards is 160
theorem shopkeeper_total_cards : total_cards = 160 := by
  sorry

end shopkeeper_total_cards_l1114_111446


namespace distance_proof_l1114_111473

-- Define the speeds of Alice and Bob
def aliceSpeed : ℚ := 1 / 20 -- Alice's speed in miles per minute
def bobSpeed : ℚ := 3 / 40 -- Bob's speed in miles per minute

-- Define the time they walk/jog
def time : ℚ := 120 -- Time in minutes (2 hours)

-- Calculate the distances
def aliceDistance : ℚ := aliceSpeed * time -- Distance Alice walked
def bobDistance : ℚ := bobSpeed * time -- Distance Bob jogged

-- The total distance between Alice and Bob after 2 hours
def totalDistance : ℚ := aliceDistance + bobDistance

-- Prove that the total distance is 15 miles
theorem distance_proof : totalDistance = 15 := by
  sorry

end distance_proof_l1114_111473
