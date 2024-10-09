import Mathlib

namespace find_integers_l1747_174773

theorem find_integers (x y : ℕ) (h : 2 * x * y = 21 + 2 * x + y) : (x = 1 ∧ y = 23) ∨ (x = 6 ∧ y = 3) :=
by
  sorry

end find_integers_l1747_174773


namespace room_tiling_problem_correct_l1747_174775

noncomputable def room_tiling_problem : Prop :=
  let room_length := 6.72
  let room_width := 4.32
  let tile_size := 0.3
  let room_area := room_length * room_width
  let tile_area := tile_size * tile_size
  let num_tiles := (room_area / tile_area).ceil
  num_tiles = 323

theorem room_tiling_problem_correct : room_tiling_problem := 
  sorry

end room_tiling_problem_correct_l1747_174775


namespace number_exceeds_by_35_l1747_174752

theorem number_exceeds_by_35 (x : ℤ) (h : x = (3 / 8 : ℚ) * x + 35) : x = 56 :=
by
  sorry

end number_exceeds_by_35_l1747_174752


namespace y_in_interval_l1747_174792

theorem y_in_interval :
  ∃ (y : ℝ), y = 5 + (1/y) * -y ∧ 2 < y ∧ y ≤ 4 :=
by
  sorry

end y_in_interval_l1747_174792


namespace total_boxes_correct_l1747_174740

noncomputable def initial_boxes : ℕ := 400
noncomputable def cost_per_box : ℕ := 80 + 165
noncomputable def initial_spent : ℕ := initial_boxes * cost_per_box
noncomputable def donor_amount : ℕ := 4 * initial_spent
noncomputable def additional_boxes : ℕ := donor_amount / cost_per_box
noncomputable def total_boxes : ℕ := initial_boxes + additional_boxes

theorem total_boxes_correct : total_boxes = 2000 := by
  sorry

end total_boxes_correct_l1747_174740


namespace juice_water_ratio_l1747_174794

theorem juice_water_ratio (V : ℝ) :
  let glass_juice_ratio := (2, 1)
  let mug_volume := 2 * V
  let mug_juice_ratio := (4, 1)
  let glass_juice_vol := (2 / 3) * V
  let glass_water_vol := (1 / 3) * V
  let mug_juice_vol := (8 / 5) * V
  let mug_water_vol := (2 / 5) * V
  let total_juice := glass_juice_vol + mug_juice_vol
  let total_water := glass_water_vol + mug_water_vol
  let ratio := total_juice / total_water
  ratio = 34 / 11 :=
by
  sorry

end juice_water_ratio_l1747_174794


namespace find_ratio_EG_ES_l1747_174757

variables (EF GH EH EG ES QR : ℝ) -- lengths of the segments
variables (x y : ℝ) -- unknowns for parts of the segments
variables (Q R S : Point) -- points

-- Define conditions based on the problem
def parallelogram_EFGH (EF GH EH EG : ℝ) : Prop :=
  ∀ (x y : ℝ), EF = 8 * x ∧ EH = 9 * y

def point_on_segment_Q (Q : Point) (EF EQ : ℝ) : Prop :=
  ∃ x : ℝ, EQ = (1 / 8) * EF

def point_on_segment_R (R : Point) (EH ER : ℝ) : Prop :=
  ∃ y : ℝ, ER = (1 / 9) * EH

def intersection_at_S (EG QR ES : ℝ) : Prop :=
  ∃ x y : ℝ, ES = (1 / 8) * EG + (1 / 9) * EG

theorem find_ratio_EG_ES :
  parallelogram_EFGH EF GH EH EG →
  point_on_segment_Q Q EF (1/8 * EF) →
  point_on_segment_R R EH (1/9 * EH) →
  intersection_at_S EG QR ES →
  EG / ES = 72 / 17 :=
by
  intros h_parallelogram h_pointQ h_pointR h_intersection
  sorry

end find_ratio_EG_ES_l1747_174757


namespace total_hotdogs_brought_l1747_174715

-- Define the number of hotdogs brought by the first and second neighbors based on given conditions.

def first_neighbor_hotdogs : Nat := 75
def second_neighbor_hotdogs : Nat := first_neighbor_hotdogs - 25

-- Prove that the total hotdogs brought by the neighbors equals 125.
theorem total_hotdogs_brought :
  first_neighbor_hotdogs + second_neighbor_hotdogs = 125 :=
by
  -- statement only, proof not required
  sorry

end total_hotdogs_brought_l1747_174715


namespace profit_distribution_l1747_174763

noncomputable def profit_sharing (investment_a investment_d profit: ℝ) : ℝ × ℝ :=
  let total_investment := investment_a + investment_d
  let share_a := investment_a / total_investment
  let share_d := investment_d / total_investment
  (share_a * profit, share_d * profit)

theorem profit_distribution :
  let investment_a := 22500
  let investment_d := 35000
  let first_period_profit := 9600
  let second_period_profit := 12800
  let third_period_profit := 18000
  profit_sharing investment_a investment_d first_period_profit = (3600, 6000) ∧
  profit_sharing investment_a investment_d second_period_profit = (5040, 7760) ∧
  profit_sharing investment_a investment_d third_period_profit = (7040, 10960) :=
sorry

end profit_distribution_l1747_174763


namespace pilot_fish_speed_is_30_l1747_174720

-- Define the initial conditions
def keanu_speed : ℝ := 20
def shark_initial_speed : ℝ := keanu_speed
def shark_speed_increase_factor : ℝ := 2
def pilot_fish_speed_increase_factor : ℝ := 0.5

-- Calculating final speeds
def shark_final_speed : ℝ := shark_initial_speed * shark_speed_increase_factor
def shark_speed_increase : ℝ := shark_final_speed - shark_initial_speed
def pilot_fish_speed_increase : ℝ := shark_speed_increase * pilot_fish_speed_increase_factor
def pilot_fish_final_speed : ℝ := keanu_speed + pilot_fish_speed_increase

-- The statement to prove
theorem pilot_fish_speed_is_30 : pilot_fish_final_speed = 30 := by
  sorry

end pilot_fish_speed_is_30_l1747_174720


namespace parabola_equation_l1747_174774

-- Definitions of the conditions
def parabola_passes_through (x y : ℝ) : Prop :=
  y^2 = -2 * (3 * x)

def focus_on_line (x y : ℝ) : Prop :=
  3 * x - 2 * y - 6 = 0

theorem parabola_equation (x y : ℝ) (hM : x = -6 ∧ y = 6) (hF : ∃ (x y : ℝ), focus_on_line x y) :
  parabola_passes_through x y = (y^2 = -6 * x) :=
by 
  sorry

end parabola_equation_l1747_174774


namespace find_m_l1747_174778

theorem find_m (m : ℝ) : 
  (∃ α β : ℝ, (α + β = 2 * (m + 1)) ∧ (α * β = m + 4) ∧ ((1 / α) + (1 / β) = 1)) → m = 2 :=
by
  sorry

end find_m_l1747_174778


namespace mike_total_rose_bushes_l1747_174732

-- Definitions based on the conditions
def costPerRoseBush : ℕ := 75
def costPerTigerToothAloe : ℕ := 100
def numberOfRoseBushesForFriend : ℕ := 2
def totalExpenseByMike : ℕ := 500
def numberOfTigerToothAloe : ℕ := 2

-- The total number of rose bushes Mike bought
noncomputable def totalNumberOfRoseBushes : ℕ :=
  let totalSpentOnAloes := numberOfTigerToothAloe * costPerTigerToothAloe
  let amountSpentOnRoseBushes := totalExpenseByMike - totalSpentOnAloes
  let numberOfRoseBushesForMike := amountSpentOnRoseBushes / costPerRoseBush
  numberOfRoseBushesForMike + numberOfRoseBushesForFriend

-- The theorem to prove
theorem mike_total_rose_bushes : totalNumberOfRoseBushes = 6 :=
  by
    sorry

end mike_total_rose_bushes_l1747_174732


namespace calculation_error_l1747_174718

def percentage_error (actual expected : ℚ) : ℚ :=
  (actual - expected) / expected * 100

theorem calculation_error :
  let correct_result := (5 / 3) * 3
  let incorrect_result := (5 / 3) / 3
  percentage_error incorrect_result correct_result = 88.89 := by
  sorry

end calculation_error_l1747_174718


namespace sqrt_one_fourth_l1747_174726

theorem sqrt_one_fourth :
  {x : ℚ | x^2 = 1/4} = {1/2, -1/2} :=
by sorry

end sqrt_one_fourth_l1747_174726


namespace problem_solution_l1747_174738

theorem problem_solution : (3127 - 2972) ^ 3 / 343 = 125 := by
  sorry

end problem_solution_l1747_174738


namespace cricket_team_average_age_difference_l1747_174799

theorem cricket_team_average_age_difference :
  let team_size := 11
  let captain_age := 26
  let keeper_age := captain_age + 3
  let avg_whole_team := 23
  let total_team_age := avg_whole_team * team_size
  let combined_age := captain_age + keeper_age
  let remaining_players := team_size - 2
  let total_remaining_age := total_team_age - combined_age
  let avg_remaining_players := total_remaining_age / remaining_players
  avg_whole_team - avg_remaining_players = 1 :=
by
  -- Proof omitted
  sorry

end cricket_team_average_age_difference_l1747_174799


namespace annual_growth_rate_l1747_174749

theorem annual_growth_rate (P₁ P₂ : ℝ) (y : ℕ) (r : ℝ)
  (h₁ : P₁ = 1) 
  (h₂ : P₂ = 1.21)
  (h₃ : y = 2)
  (h_growth : P₂ = P₁ * (1 + r) ^ y) :
  r = 0.1 :=
by {
  sorry
}

end annual_growth_rate_l1747_174749


namespace trajectory_midpoint_l1747_174760

-- Defining the point A(-2, 0)
def A : ℝ × ℝ := (-2, 0)

-- Defining the curve equation
def curve (x y : ℝ) : Prop := 2 * y^2 = x

-- Coordinates of P based on the midpoint formula
def P (x y : ℝ) : ℝ × ℝ := (2 * x + 2, 2 * y)

-- The target trajectory equation
def trajectory_eqn (x y : ℝ) : Prop := x = 4 * y^2 - 1

-- The theorem to be proved
theorem trajectory_midpoint (x y : ℝ) :
  curve (2 * y) (2 * x + 2) → 
  trajectory_eqn x y :=
sorry

end trajectory_midpoint_l1747_174760


namespace consecutive_integer_product_sum_l1747_174755

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l1747_174755


namespace balloon_count_l1747_174734

-- Conditions
def Fred_balloons : ℕ := 5
def Sam_balloons : ℕ := 6
def Mary_balloons : ℕ := 7
def total_balloons : ℕ := 18

-- Proof statement
theorem balloon_count :
  Fred_balloons + Sam_balloons + Mary_balloons = total_balloons :=
by
  exact Nat.add_assoc 5 6 7 ▸ rfl

end balloon_count_l1747_174734


namespace probability_one_even_dice_l1747_174753

noncomputable def probability_exactly_one_even (p : ℚ) : Prop :=
  ∃ (n : ℕ), (p = (4 * (1/2)^4 )) ∧ (n = 1) → p = 1/4

theorem probability_one_even_dice : probability_exactly_one_even (1/4) :=
by
  unfold probability_exactly_one_even
  sorry

end probability_one_even_dice_l1747_174753


namespace part1_part2_l1747_174724

noncomputable section
def g1 (x : ℝ) : ℝ := Real.log x

noncomputable def f (t : ℝ) : ℝ := 
  if g1 t = t then 1 else sorry  -- Assuming g1(x) = t has exactly one root.

theorem part1 (t : ℝ) : f t = 1 :=
by sorry

def g2 (x : ℝ) (a : ℝ) : ℝ := 
  if x ≤ 0 then x else -x^2 + 2*a*x + a

theorem part2 (a : ℝ) (h : ∃ t : ℝ, f (t + 2) > f t) : a > 1 :=
by sorry

end part1_part2_l1747_174724


namespace harrys_age_l1747_174782

-- Definitions of the ages
variable (Kiarra Bea Job Figaro Harry : ℕ)

-- Given conditions
variable (h1 : Kiarra = 2 * Bea)
variable (h2 : Job = 3 * Bea)
variable (h3 : Figaro = Job + 7)
variable (h4 : Harry = Figaro / 2)
variable (h5 : Kiarra = 30)

-- The statement to prove
theorem harrys_age : Harry = 26 := sorry

end harrys_age_l1747_174782


namespace largest_fraction_l1747_174747

theorem largest_fraction :
  let f1 := (2 : ℚ) / 3
  let f2 := (3 : ℚ) / 4
  let f3 := (2 : ℚ) / 5
  let f4 := (11 : ℚ) / 15
  f2 > f1 ∧ f2 > f3 ∧ f2 > f4 :=
by
  sorry

end largest_fraction_l1747_174747


namespace compute_fraction_eq_2410_l1747_174723

theorem compute_fraction_eq_2410 (x : ℕ) (hx : x = 7) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 2410 := 
by
  -- proof steps go here
  sorry

end compute_fraction_eq_2410_l1747_174723


namespace prob_neither_snow_nor_windy_l1747_174783

-- Define the probabilities.
def prob_snow : ℚ := 1 / 4
def prob_windy : ℚ := 1 / 3

-- Define the complementary probabilities.
def prob_not_snow : ℚ := 1 - prob_snow
def prob_not_windy : ℚ := 1 - prob_windy

-- State that the events are independent and calculate the combined probability.
theorem prob_neither_snow_nor_windy :
  prob_not_snow * prob_not_windy = 1 / 2 := by
  sorry

end prob_neither_snow_nor_windy_l1747_174783


namespace book_pages_read_l1747_174769

theorem book_pages_read (pages_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) :
  (pages_per_day = 100) →
  (days_per_week = 3) →
  (weeks = 7) →
  total_pages = pages_per_day * days_per_week * weeks →
  total_pages = 2100 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end book_pages_read_l1747_174769


namespace sequence_explicit_formula_l1747_174742

noncomputable def sequence_a : ℕ → ℝ
| 0     => 0  -- Not used, but needed for definition completeness
| 1     => 3
| (n+1) => n / (n + 1) * sequence_a n

theorem sequence_explicit_formula (n : ℕ) (h : n ≠ 0) :
  sequence_a n = 3 / n :=
by sorry

end sequence_explicit_formula_l1747_174742


namespace Jean_money_l1747_174728

theorem Jean_money (x : ℝ) (h1 : 3 * x + x = 76): 
  3 * x = 57 := 
by
  sorry

end Jean_money_l1747_174728


namespace cubic_has_one_real_root_l1747_174788

theorem cubic_has_one_real_root :
  (∃ x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0) ∧ ∀ x y : ℝ, (x^3 - 6*x^2 + 9*x - 10 = 0) ∧ (y^3 - 6*y^2 + 9*y - 10 = 0) → x = y :=
by
  sorry

end cubic_has_one_real_root_l1747_174788


namespace min_colors_for_distance_six_l1747_174795

/-
Definitions and conditions:
- The board is an infinite checkered paper with a cell side of one unit.
- The distance between two cells is the length of the shortest path of a rook from one cell to another.

Statement:
- Prove that the minimum number of colors needed to color the board such that two cells that are a distance of 6 apart are always painted different colors is 4.
-/

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  |c1.1 - c2.1| + |c1.2 - c2.2|

theorem min_colors_for_distance_six : ∃ (n : ℕ), (∀ (f : cell → ℕ), (∀ c1 c2, rook_distance c1 c2 = 6 → f c1 ≠ f c2) → n ≤ 4) :=
by
  sorry

end min_colors_for_distance_six_l1747_174795


namespace transformed_passes_through_l1747_174776

def original_parabola (x : ℝ) : ℝ :=
  -x^2 - 2*x + 3

def transformed_parabola (x : ℝ) : ℝ :=
  -(x - 1)^2 + 2

theorem transformed_passes_through : transformed_parabola (-1) = 1 :=
  by sorry

end transformed_passes_through_l1747_174776


namespace basketball_game_l1747_174770

theorem basketball_game (E H : ℕ) (h1 : E = H + 18) (h2 : E + H = 50) : H = 16 :=
by
  sorry

end basketball_game_l1747_174770


namespace gcd_lcm_product_l1747_174751

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [h₁, h₂]
  unfold Nat.gcd Nat.lcm
  sorry

end gcd_lcm_product_l1747_174751


namespace min_value_f_prime_at_2_l1747_174705

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1/a) * x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 4*a*x + (1/a)

theorem min_value_f_prime_at_2 (a : ℝ) (h : a > 0) : 
  f_prime a 2 >= 12 + 4 * Real.sqrt 2 := 
by
  -- proof will be written here
  sorry

end min_value_f_prime_at_2_l1747_174705


namespace father_son_age_problem_l1747_174716

theorem father_son_age_problem
  (F S Y : ℕ)
  (h1 : F = 3 * S)
  (h2 : F = 45)
  (h3 : F + Y = 2 * (S + Y)) :
  Y = 15 :=
sorry

end father_son_age_problem_l1747_174716


namespace trigonometric_inequality_1_l1747_174748

theorem trigonometric_inequality_1 {n : ℕ} 
  (h1 : 0 < n) (x : ℝ) (h2 : 0 < x) (h3 : x < (Real.pi / (2 * n))) :
  (1 / 2) * (Real.tan x + Real.tan (n * x) - Real.tan ((n - 1) * x)) > (1 / n) * Real.tan (n * x) := 
sorry

end trigonometric_inequality_1_l1747_174748


namespace geo_series_sum_l1747_174784

theorem geo_series_sum (a r : ℚ) (n: ℕ) (ha : a = 1/3) (hr : r = 1/2) (hn : n = 8) : 
    (a * (1 - r^n) / (1 - r)) = 85 / 128 := 
by
  sorry

end geo_series_sum_l1747_174784


namespace class_scores_mean_l1747_174707

theorem class_scores_mean 
  (F S : ℕ) (Rf Rs : ℚ)
  (hF : F = 90)
  (hS : S = 75)
  (hRatio : Rf / Rs = 2 / 3) :
  (F * (2/3 * Rs) + S * Rs) / (2/3 * Rs + Rs) = 81 := by
    sorry

end class_scores_mean_l1747_174707


namespace investment_ratio_l1747_174736

variable (x : ℝ)
variable (p q t : ℝ)

theorem investment_ratio (h1 : 7 * p = 5 * q) (h2 : (7 * p * 8) / (5 * q * t) = 7 / 10) : t = 16 :=
by
  sorry

end investment_ratio_l1747_174736


namespace zeros_of_f_l1747_174777

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x ^ 2 - 2 * x - 3)

theorem zeros_of_f :
  { x : ℝ | f x = 0 } = {1, -1, 3} :=
sorry

end zeros_of_f_l1747_174777


namespace matchsticks_20th_stage_l1747_174765

theorem matchsticks_20th_stage :
  let a1 := 3
  let d := 3
  let a20 := a1 + 19 * d
  a20 = 60 := by
  sorry

end matchsticks_20th_stage_l1747_174765


namespace sequence_geometric_l1747_174754

theorem sequence_geometric (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = r * a n) (h2 : a 4 = 2) : a 2 * a 6 = 4 :=
by
  sorry

end sequence_geometric_l1747_174754


namespace simple_interest_l1747_174771

theorem simple_interest (TD : ℝ) (Sum : ℝ) (SI : ℝ) 
  (h1 : TD = 78) 
  (h2 : Sum = 947.1428571428571) 
  (h3 : SI = Sum - (Sum - TD)) : 
  SI = 78 := 
by 
  sorry

end simple_interest_l1747_174771


namespace sine_triangle_l1747_174709

theorem sine_triangle (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_perimeter : a + b + c ≤ 2 * Real.pi)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (ha_pi : a < Real.pi) (hb_pi : b < Real.pi) (hc_pi : c < Real.pi):
  ∃ (x y z : ℝ), x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ x + z > y :=
by
  sorry

end sine_triangle_l1747_174709


namespace compare_fractions_l1747_174746

theorem compare_fractions :
  (111110 / 111111) < (333331 / 333334) ∧ (333331 / 333334) < (222221 / 222223) :=
by
  sorry

end compare_fractions_l1747_174746


namespace cubic_expression_l1747_174790

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * abc = 1027 :=
sorry

end cubic_expression_l1747_174790


namespace profit_of_150_cents_requires_120_oranges_l1747_174714

def cost_price_per_orange := 15 / 4  -- cost price per orange in cents
def selling_price_per_orange := 30 / 6  -- selling price per orange in cents
def profit_per_orange := selling_price_per_orange - cost_price_per_orange  -- profit per orange in cents
def required_oranges_to_make_profit := 150 / profit_per_orange  -- number of oranges to get 150 cents of profit

theorem profit_of_150_cents_requires_120_oranges :
  required_oranges_to_make_profit = 120 :=
by
  -- the actual proof will follow here
  sorry

end profit_of_150_cents_requires_120_oranges_l1747_174714


namespace sufficient_but_not_necessary_l1747_174789

theorem sufficient_but_not_necessary (p q : Prop) :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ (¬ (p ∧ q) → p ∨ q → False) :=
by
  sorry

end sufficient_but_not_necessary_l1747_174789


namespace total_amount_spent_l1747_174750

namespace KeithSpending

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tires_cost : ℝ := 112.46
def total_cost : ℝ := 387.85

theorem total_amount_spent : speakers_cost + cd_player_cost + tires_cost = total_cost :=
by sorry

end KeithSpending

end total_amount_spent_l1747_174750


namespace tank_capacity_l1747_174737

theorem tank_capacity (T : ℝ) (h : (3 / 4) * T + 7 = (7 / 8) * T) : T = 56 := 
sorry

end tank_capacity_l1747_174737


namespace find_function_f_l1747_174711

theorem find_function_f
  (f : ℝ → ℝ)
  (H : ∀ x y, f x ^ 2 + f y ^ 2 = f (x + y) ^ 2) :
  ∀ x, f x = 0 := 
by 
  sorry

end find_function_f_l1747_174711


namespace ratio_lena_kevin_after_5_more_l1747_174727

variables (L K N : ℕ)

def lena_initial_candy : ℕ := 16
def lena_gets_more : ℕ := 5
def kevin_candy_less_than_nicole : ℕ := 4
def lena_more_than_nicole : ℕ := 5

theorem ratio_lena_kevin_after_5_more
  (lena_initial : L = lena_initial_candy)
  (lena_to_multiple_of_kevin : L + lena_gets_more = K * 3) 
  (kevin_less_than_nicole : K = N - kevin_candy_less_than_nicole)
  (lena_more_than_nicole_condition : L = N + lena_more_than_nicole) :
  (L + lena_gets_more) / K = 3 :=
sorry

end ratio_lena_kevin_after_5_more_l1747_174727


namespace circumcircle_equation_l1747_174729

theorem circumcircle_equation :
  ∃ (a b r : ℝ), 
    (∀ {x y : ℝ}, (x, y) = (2, 2) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (5, 3) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (3, -1) → (x - a)^2 + (y - b)^2 = r^2) ∧
    ((x - 4)^2 + (y - 1)^2 = 5) :=
sorry

end circumcircle_equation_l1747_174729


namespace number_of_monkeys_l1747_174704

theorem number_of_monkeys (X : ℕ) : 
  10 * 10 = 10 →
  1 * 1 = 1 →
  1 * 70 / 10 = 7 →
  (X / 7) = X / 7 :=
by
  intros h1 h2 h3
  sorry

end number_of_monkeys_l1747_174704


namespace isosceles_triangle_sides_l1747_174706

/-
  Given: 
  - An isosceles triangle with a perimeter of 60 cm.
  - The intersection point of the medians lies on the inscribed circle.
  Prove:
  - The sides of the triangle are 25 cm, 25 cm, and 10 cm.
-/

theorem isosceles_triangle_sides (AB BC AC : ℝ) 
  (h1 : AB = BC)
  (h2 : AB + BC + AC = 60) 
  (h3 : ∃ r : ℝ, r > 0 ∧ 6 * r = AC ∧ 3 * r * AC = 30 * r) :
  AB = 25 ∧ BC = 25 ∧ AC = 10 :=
sorry

end isosceles_triangle_sides_l1747_174706


namespace translate_function_down_l1747_174700

theorem translate_function_down 
  (f : ℝ → ℝ) 
  (a k : ℝ) 
  (h : ∀ x, f x = a * x) 
  : ∀ x, (f x - k) = a * x - k :=
by
  sorry

end translate_function_down_l1747_174700


namespace profit_function_simplified_maximize_profit_l1747_174735

-- Define the given conditions
def cost_per_product : ℝ := 3
def management_fee_per_product : ℝ := 3
def annual_sales_volume (x : ℝ) : ℝ := (12 - x) ^ 2

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - (cost_per_product + management_fee_per_product)) * annual_sales_volume x

-- Define the bounds for x
def x_bounds (x : ℝ) : Prop := 9 ≤ x ∧ x ≤ 11

-- Prove the profit function in simplified form
theorem profit_function_simplified (x : ℝ) (h : x_bounds x) :
    profit x = x ^ 3 - 30 * x ^ 2 + 288 * x - 864 :=
by
  sorry

-- Prove the maximum profit and the corresponding x value
theorem maximize_profit (x : ℝ) (h : x_bounds x) :
    (∀ y, (∃ x', x_bounds x' ∧ y = profit x') → y ≤ 27) ∧ profit 9 = 27 :=
by
  sorry

end profit_function_simplified_maximize_profit_l1747_174735


namespace problem1_problem2_l1747_174793

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : (3 * x - y)^2 - (3 * x + 2 * y) * (3 * x - 2 * y) = 5 * y^2 - 6 * x * y :=
by
  sorry

end problem1_problem2_l1747_174793


namespace point_on_line_l1747_174780

theorem point_on_line (x : ℝ) : 
    (∃ k : ℝ, (-4) = k * (-4) + 8) → 
    (-4 = 2 * x + 8) → 
    x = -6 := 
sorry

end point_on_line_l1747_174780


namespace find_ck_l1747_174786

theorem find_ck (d r k : ℕ) (a_n b_n c_n : ℕ → ℕ) 
  (h_an : ∀ n, a_n n = 1 + (n - 1) * d)
  (h_bn : ∀ n, b_n n = r ^ (n - 1))
  (h_cn : ∀ n, c_n n = a_n n + b_n n)
  (h_ckm1 : c_n (k - 1) = 30)
  (h_ckp1 : c_n (k + 1) = 300) :
  c_n k = 83 := 
sorry

end find_ck_l1747_174786


namespace diff_cubes_square_of_squares_l1747_174731

theorem diff_cubes_square_of_squares {x y : ℤ} (h1 : (x + 1) ^ 3 - x ^ 3 = y ^ 2) :
  ∃ (a b : ℤ), y = a ^ 2 + b ^ 2 ∧ a = b + 1 :=
sorry

end diff_cubes_square_of_squares_l1747_174731


namespace all_terms_are_integers_l1747_174730

open Nat

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 143 ∧ ∀ n ≥ 2, a (n + 1) = 5 * (Finset.range n).sum a / n

theorem all_terms_are_integers (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, 1 ≤ n → ∃ k : ℕ, a n = k := 
by
  sorry

end all_terms_are_integers_l1747_174730


namespace alex_candles_left_l1747_174701

theorem alex_candles_left (candles_start used_candles : ℕ) (h1 : candles_start = 44) (h2 : used_candles = 32) :
  candles_start - used_candles = 12 :=
by
  sorry

end alex_candles_left_l1747_174701


namespace ratio_Pat_Mark_l1747_174797

-- Definitions inferred from the conditions
def total_hours : ℕ := 135
def Kate_hours (K : ℕ) : ℕ := K
def Pat_hours (K : ℕ) : ℕ := 2 * K
def Mark_hours (K : ℕ) : ℕ := K + 75

-- The main statement
theorem ratio_Pat_Mark (K : ℕ) (h : Kate_hours K + Pat_hours K + Mark_hours K = total_hours) :
  (Pat_hours K) / (Mark_hours K) = 1 / 3 := by
  sorry

end ratio_Pat_Mark_l1747_174797


namespace sum_of_dimensions_l1747_174756

theorem sum_of_dimensions
  (X Y Z : ℝ)
  (h1 : X * Y = 24)
  (h2 : X * Z = 48)
  (h3 : Y * Z = 72) :
  X + Y + Z = 22 := 
sorry

end sum_of_dimensions_l1747_174756


namespace hens_egg_laying_l1747_174717

theorem hens_egg_laying :
  ∀ (hens: ℕ) (price_per_dozen: ℝ) (total_revenue: ℝ) (weeks: ℕ) (total_hens: ℕ),
  hens = 10 →
  price_per_dozen = 3 →
  total_revenue = 120 →
  weeks = 4 →
  total_hens = hens →
  (total_revenue / price_per_dozen / 12) * 12 = 480 →
  (480 / weeks) = 120 →
  (120 / hens) = 12 :=
by sorry

end hens_egg_laying_l1747_174717


namespace sum_of_powers_l1747_174772

theorem sum_of_powers (m n : ℤ)
  (h1 : m + n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 4)
  (h4 : m^4 + n^4 = 7)
  (h5 : m^5 + n^5 = 11) :
  m^9 + n^9 = 76 :=
sorry

end sum_of_powers_l1747_174772


namespace inequality_proof_l1747_174739

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (2 * x) + 1 / (2 * y) + 1 / (2 * z)) > 
  (1 / (y + z) + 1 / (z + x) + 1 / (x + y)) :=
  by
    let a := y + z
    let b := z + x
    let c := x + y
    have x_def : x = (a + c - b) / 2 := sorry
    have y_def : y = (a + b - c) / 2 := sorry
    have z_def : z = (b + c - a) / 2 := sorry
    sorry

end inequality_proof_l1747_174739


namespace min_days_to_sun_l1747_174719

def active_days_for_level (N : ℕ) : ℕ :=
  N * (N + 4)

def days_needed_for_upgrade (current_days future_days : ℕ) : ℕ :=
  future_days - current_days

theorem min_days_to_sun (current_level future_level : ℕ) :
  current_level = 9 →
  future_level = 16 →
  days_needed_for_upgrade (active_days_for_level current_level) (active_days_for_level future_level) = 203 :=
by
  intros h1 h2
  rw [h1, h2, active_days_for_level, active_days_for_level]
  sorry

end min_days_to_sun_l1747_174719


namespace populations_equal_in_years_l1747_174779

-- Definitions
def populationX (n : ℕ) : ℤ := 68000 - 1200 * n
def populationY (n : ℕ) : ℤ := 42000 + 800 * n

-- Statement to prove
theorem populations_equal_in_years : ∃ n : ℕ, populationX n = populationY n ∧ n = 13 :=
sorry

end populations_equal_in_years_l1747_174779


namespace cyclic_sum_inequality_l1747_174766

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (h_product : a * b * c = 1) :
  (a^6 / ((a - b) * (a - c)) + b^6 / ((b - c) * (b - a)) + c^6 / ((c - a) * (c - b)) > 15) := 
by sorry

end cyclic_sum_inequality_l1747_174766


namespace final_cost_l1747_174744

-- Definitions of initial conditions
def initial_cart_total : ℝ := 54.00
def discounted_item_original_price : ℝ := 20.00
def discount_rate1 : ℝ := 0.20
def coupon_rate : ℝ := 0.10

-- Prove the final cost after applying discounts
theorem final_cost (initial_cart_total discounted_item_original_price discount_rate1 coupon_rate : ℝ) :
  let discounted_price := discounted_item_original_price * (1 - discount_rate1)
  let total_after_first_discount := initial_cart_total - discounted_price
  let final_total := total_after_first_discount * (1 - coupon_rate)
  final_total = 45.00 :=
by 
  sorry

end final_cost_l1747_174744


namespace solve_system_eqs_l1747_174708

theorem solve_system_eqs : 
    ∃ (x y z : ℚ), 
    4 * x - 3 * y + z = -10 ∧
    3 * x + 5 * y - 2 * z = 8 ∧
    x - 2 * y + 7 * z = 5 ∧
    x = -51 / 61 ∧ 
    y = 378 / 61 ∧ 
    z = 728 / 61 := by
  sorry

end solve_system_eqs_l1747_174708


namespace common_difference_of_sequence_l1747_174745

variable (a : ℕ → ℚ)

def is_arithmetic_sequence (a : ℕ → ℚ) :=
  ∃ d : ℚ, ∀ n m : ℕ, a n = a m + d * (n - m)

theorem common_difference_of_sequence 
  (h : a 2015 = a 2013 + 6) 
  (ha : is_arithmetic_sequence a) :
  ∃ d : ℚ, d = 3 :=
by
  sorry

end common_difference_of_sequence_l1747_174745


namespace pine_taller_than_maple_l1747_174758

def height_maple : ℚ := 13 + 1 / 4
def height_pine : ℚ := 19 + 3 / 8

theorem pine_taller_than_maple :
  (height_pine - height_maple = 6 + 1 / 8) :=
sorry

end pine_taller_than_maple_l1747_174758


namespace sum_of_squares_l1747_174768

theorem sum_of_squares (n m : ℕ) (h : 2 * m = n^2 + 1) : ∃ k : ℕ, m = k^2 + (k - 1)^2 :=
sorry

end sum_of_squares_l1747_174768


namespace number_of_diagonals_25_sides_l1747_174791

theorem number_of_diagonals_25_sides (n : ℕ) (h : n = 25) : 
    (n * (n - 3)) / 2 = 275 := by
  sorry

end number_of_diagonals_25_sides_l1747_174791


namespace productivity_after_repair_l1747_174743

-- Define the initial productivity and the increase factor.
def original_productivity : ℕ := 10
def increase_factor : ℝ := 1.5

-- Define the expected productivity after the improvement.
def expected_productivity : ℝ := 25

-- The theorem we need to prove.
theorem productivity_after_repair :
  original_productivity * (1 + increase_factor) = expected_productivity := by
  sorry

end productivity_after_repair_l1747_174743


namespace maximum_value_of_a_l1747_174725

theorem maximum_value_of_a :
  (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) → a ≤ 6 :=
by
  sorry

end maximum_value_of_a_l1747_174725


namespace find_a₉_l1747_174787

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom S_6_eq : S 6 = 3
axiom S_11_eq : S 11 = 18

noncomputable def a₉ : ℝ := sorry -- Define a₉ here, proof skipped by "sorry"

theorem find_a₉ (a : ℕ → ℝ) (S : ℕ → ℝ) :
  S 6 = 3 →
  S 11 = 18 →
  a₉ = 3 :=
by
  intros S_6_eq S_11_eq
  sorry -- Proof goes here

end find_a₉_l1747_174787


namespace quadratic_transformation_l1747_174712

theorem quadratic_transformation (y m n : ℝ) 
  (h1 : 2 * y^2 - 2 = 4 * y) 
  (h2 : (y - m)^2 = n) : 
  (m - n)^2023 = -1 := 
  sorry

end quadratic_transformation_l1747_174712


namespace find_a_b_find_A_l1747_174767

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2 * (Real.log x / Real.log 2) ^ 2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b

theorem find_a_b : (∀ x : ℝ, 0 < x → f x a b = 2 * (Real.log x / Real.log 2)^2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b) 
                     → f (1/2) a b = -8 
                     ∧ ∀ x : ℝ, 0 < x → x ≠ 1/2 → f x a b ≥ f (1 / 2) a b
                     → a = -2 ∧ b = -6 := 
sorry

theorem find_A (a b : ℝ) (h₁ : a = -2) (h₂ : b = -6) : 
  { x : ℝ | 0 < x ∧ f x a b > 0 } = {x | 0 < x ∧ (x < 1/8 ∨ x > 2)} :=
sorry

end find_a_b_find_A_l1747_174767


namespace time_to_paint_one_house_l1747_174741

theorem time_to_paint_one_house (houses : ℕ) (total_time_hours : ℕ) (total_time_minutes : ℕ) 
  (minutes_per_hour : ℕ) (h1 : houses = 9) (h2 : total_time_hours = 3) 
  (h3 : minutes_per_hour = 60) (h4 : total_time_minutes = total_time_hours * minutes_per_hour) : 
  (total_time_minutes / houses) = 20 :=
by
  sorry

end time_to_paint_one_house_l1747_174741


namespace jelly_bean_ratio_l1747_174703

theorem jelly_bean_ratio
  (initial_jelly_beans : ℕ)
  (num_people : ℕ)
  (remaining_jelly_beans : ℕ)
  (amount_taken_by_each_of_last_four : ℕ)
  (total_taken_by_last_four : ℕ)
  (total_jelly_beans_taken : ℕ)
  (X : ℕ)
  (ratio : ℕ)
  (h0 : initial_jelly_beans = 8000)
  (h1 : num_people = 10)
  (h2 : remaining_jelly_beans = 1600)
  (h3 : amount_taken_by_each_of_last_four = 400)
  (h4 : total_taken_by_last_four = 4 * amount_taken_by_each_of_last_four)
  (h5 : total_jelly_beans_taken = initial_jelly_beans - remaining_jelly_beans)
  (h6 : X = total_jelly_beans_taken - total_taken_by_last_four)
  (h7 : ratio = X / total_taken_by_last_four)
  : ratio = 3 :=
by sorry

end jelly_bean_ratio_l1747_174703


namespace sum_of_three_numbers_l1747_174762

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 36) 
  (h2 : b + c = 55) 
  (h3 : c + a = 60) : 
  a + b + c = 75.5 := 
by 
  sorry

end sum_of_three_numbers_l1747_174762


namespace x_intercept_is_neg_three_halves_l1747_174764

-- Definition of the points
def pointA : ℝ × ℝ := (-1, 1)
def pointB : ℝ × ℝ := (3, 9)

-- Statement of the theorem: The x-intercept of the line passing through the points is -3/2.
theorem x_intercept_is_neg_three_halves (A B : ℝ × ℝ)
    (hA : A = pointA)
    (hB : B = pointB) :
    ∃ x_intercept : ℝ, x_intercept = -3 / 2 := 
by
    sorry

end x_intercept_is_neg_three_halves_l1747_174764


namespace overall_average_output_l1747_174796

theorem overall_average_output 
  (initial_cogs : ℕ := 60) 
  (rate_1 : ℕ := 36) 
  (rate_2 : ℕ := 60) 
  (second_batch_cogs : ℕ := 60) :
  (initial_cogs + second_batch_cogs) / ((initial_cogs / rate_1) + (second_batch_cogs / rate_2)) = 45 := 
  sorry

end overall_average_output_l1747_174796


namespace bloodPressureFriday_l1747_174713

def bloodPressureSunday : ℕ := 120
def bpChangeMonday : ℤ := 20
def bpChangeTuesday : ℤ := -30
def bpChangeWednesday : ℤ := -25
def bpChangeThursday : ℤ := 15
def bpChangeFriday : ℤ := 30

theorem bloodPressureFriday : bloodPressureSunday + bpChangeMonday + bpChangeTuesday + bpChangeWednesday + bpChangeThursday + bpChangeFriday = 130 := by {
  -- Placeholder for the proof
  sorry
}

end bloodPressureFriday_l1747_174713


namespace probability_adjacent_points_l1747_174781

open Finset

-- Define the hexagon points and adjacency relationship
def hexagon_points : Finset ℕ := {0, 1, 2, 3, 4, 5}

def adjacent (a b : ℕ) : Prop :=
  (a = b + 1 ∨ a = b - 1 ∨ (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0))

-- Total number of ways to choose 2 points from 6 points
def total_pairs := (hexagon_points.card.choose 2)

-- Number of pairs that are adjacent
def favorable_pairs := (6 : ℕ) -- Each point has exactly 2 adjacent points, counted twice

-- The probability of selecting two adjacent points
theorem probability_adjacent_points : (favorable_pairs : ℚ) / total_pairs = 2 / 5 :=
by {
  sorry
}

end probability_adjacent_points_l1747_174781


namespace arthur_bought_hamburgers_on_first_day_l1747_174785

-- Define the constants and parameters
def D : ℕ := 1
def H : ℕ := 2
def total_cost_day1 : ℕ := 10
def total_cost_day2 : ℕ := 7

-- Define the equation representing the transactions
def equation_day1 (h : ℕ) := H * h + 4 * D = total_cost_day1
def equation_day2 := 2 * H + 3 * D = total_cost_day2

-- The theorem we need to prove: the number of hamburgers h bought on the first day is 3
theorem arthur_bought_hamburgers_on_first_day (h : ℕ) (hd1 : equation_day1 h) (hd2 : equation_day2) : h = 3 := 
by 
  sorry

end arthur_bought_hamburgers_on_first_day_l1747_174785


namespace total_limes_picked_l1747_174721

-- Define the number of limes each person picked
def fred_limes : Nat := 36
def alyssa_limes : Nat := 32
def nancy_limes : Nat := 35
def david_limes : Nat := 42
def eileen_limes : Nat := 50

-- Formal statement of the problem
theorem total_limes_picked : 
  fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes = 195 := by
  -- Add proof
  sorry

end total_limes_picked_l1747_174721


namespace range_of_m_l1747_174733

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ x y : ℝ, 0 < x → 0 < y → (2 * y / x + 9 * x / (2 * y) ≥ m^2 + m))
  ↔ (-3 ≤ m ∧ m ≤ 2) := sorry

end range_of_m_l1747_174733


namespace f_five_eq_three_f_three_x_inv_f_243_l1747_174722

-- Define the function f satisfying the given conditions.
def f (x : ℕ) : ℕ :=
  if x = 5 then 3
  else if x = 15 then 9
  else if x = 45 then 27
  else if x = 135 then 81
  else if x = 405 then 243
  else 0

-- Define the condition f(5) = 3
theorem f_five_eq_three : f 5 = 3 := rfl

-- Define the condition f(3x) = 3f(x) for all x
theorem f_three_x (x : ℕ) : f (3 * x) = 3 * f x :=
sorry

-- Prove that f⁻¹(243) = 405.
theorem inv_f_243 : f (405) = 243 :=
by sorry

-- Concluding the proof statement using the concluded theorems.
example : f (405) = 243 :=
by apply inv_f_243

end f_five_eq_three_f_three_x_inv_f_243_l1747_174722


namespace remainder_of_2365487_div_3_l1747_174759

theorem remainder_of_2365487_div_3 : (2365487 % 3) = 2 := by
  sorry

end remainder_of_2365487_div_3_l1747_174759


namespace base_eight_to_base_ten_l1747_174798

theorem base_eight_to_base_ten : (4 * 8^1 + 5 * 8^0 = 37) := by
  sorry

end base_eight_to_base_ten_l1747_174798


namespace propositions_correct_l1747_174710

variable {R : Type} [LinearOrderedField R] {A B : Set R}

theorem propositions_correct :
  (¬ ∃ x : R, x^2 + x + 1 = 0) ∧
  (¬ (∃ x : R, x + 1 ≤ 2) → ∀ x : R, x + 1 > 2) ∧
  (∀ x : R, x ∈ A ∩ B → x ∈ A) ∧
  (∀ x : R, x > 3 → x^2 > 9 ∧ ∃ y : R, y^2 > 9 ∧ y < 3) :=
by
  sorry

end propositions_correct_l1747_174710


namespace find_k_l1747_174702

theorem find_k (k b : ℤ) (h1 : -x^2 - (k + 10) * x - b = -(x - 2) * (x - 4))
  (h2 : b = 8) : k = -16 :=
sorry

end find_k_l1747_174702


namespace tan_theta_sqrt3_l1747_174761

theorem tan_theta_sqrt3 (θ : ℝ) 
  (h : Real.cos (40 * (π / 180) - θ) 
     + Real.cos (40 * (π / 180) + θ) 
     + Real.cos (80 * (π / 180) - θ) = 0) 
  : Real.tan θ = -Real.sqrt 3 := 
by
  sorry

end tan_theta_sqrt3_l1747_174761
