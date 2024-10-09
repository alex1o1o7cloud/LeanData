import Mathlib

namespace part1_part2_l709_70962

def f (x : ℝ) (t : ℝ) : ℝ := x^2 + 2 * t * x + t - 1

theorem part1 (hf : ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3) : 
  ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3 :=
by 
  sorry
  
theorem part2 (ht : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f x t > 0) : 
  t ∈ Set.Ioi (0 : ℝ) :=
by 
  sorry

end part1_part2_l709_70962


namespace total_tiles_144_l709_70951

-- Define the dimensions of the dining room
def diningRoomLength : ℕ := 15
def diningRoomWidth : ℕ := 20

-- Define the border width using 1x1 tiles
def borderWidth : ℕ := 2

-- Area of each 3x3 tile
def tileArea : ℕ := 9

-- Calculate the dimensions of the inner area after the border
def innerAreaLength : ℕ := diningRoomLength - 2 * borderWidth
def innerAreaWidth : ℕ := diningRoomWidth - 2 * borderWidth

-- Calculate the area of the inner region
def innerArea : ℕ := innerAreaLength * innerAreaWidth

-- Calculate the number of 3x3 tiles
def numThreeByThreeTiles : ℕ := (innerArea + tileArea - 1) / tileArea -- rounded up division

-- Calculate the number of 1x1 tiles for the border
def numOneByOneTiles : ℕ :=
  2 * (innerAreaLength + innerAreaWidth + 4 * borderWidth)

-- Total number of tiles
def totalTiles : ℕ := numOneByOneTiles + numThreeByThreeTiles

-- Prove that the total number of tiles is 144
theorem total_tiles_144 : totalTiles = 144 := by
  sorry

end total_tiles_144_l709_70951


namespace good_apples_count_l709_70996

def total_apples : ℕ := 14
def unripe_apples : ℕ := 6

theorem good_apples_count : total_apples - unripe_apples = 8 :=
by
  unfold total_apples unripe_apples
  sorry

end good_apples_count_l709_70996


namespace troll_problem_l709_70907

theorem troll_problem (T : ℕ) (h : 6 + T + T / 2 = 33) : 4 * 6 - T = 6 :=
by sorry

end troll_problem_l709_70907


namespace ways_to_select_at_least_one_defective_l709_70995

open Finset

-- Define basic combinatorial selection functions
def combination (n k : ℕ) := Nat.choose n k

-- Given conditions
def total_products : ℕ := 100
def defective_products : ℕ := 6
def selected_products : ℕ := 3
def non_defective_products : ℕ := total_products - defective_products

-- The question to prove: the number of ways to select at least one defective product
theorem ways_to_select_at_least_one_defective :
  (combination total_products selected_products) - (combination non_defective_products selected_products) =
  (combination 100 3) - (combination 94 3) := by
  sorry

end ways_to_select_at_least_one_defective_l709_70995


namespace remainder_of_h_x6_l709_70936

def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

noncomputable def remainder_when_h_x6_divided_by_h (x : ℝ) : ℝ :=
  let hx := h x
  let hx6 := h (x^6)
  hx6 - 6 * hx

theorem remainder_of_h_x6 (x : ℝ) : remainder_when_h_x6_divided_by_h x = 6 :=
  sorry

end remainder_of_h_x6_l709_70936


namespace daily_sales_volume_80_sales_volume_function_price_for_profit_l709_70968

-- Define all relevant conditions
def cost_price : ℝ := 70
def max_price : ℝ := 99
def initial_price : ℝ := 95
def initial_sales : ℕ := 50
def price_reduction_effect : ℕ := 2

-- Part 1: Proving daily sales volume at 80 yuan
theorem daily_sales_volume_80 : 
  (initial_price - 80) * price_reduction_effect + initial_sales = 80 := 
by sorry

-- Part 2: Proving functional relationship
theorem sales_volume_function (x : ℝ) (h₁ : 70 ≤ x) (h₂ : x ≤ 99) : 
  (initial_sales + price_reduction_effect * (initial_price - x) = -2 * x + 240) :=
by sorry

-- Part 3: Proving price for 1200 yuan daily profit
theorem price_for_profit (profit_target : ℝ) (h : profit_target = 1200) :
  ∃ x, (x - cost_price) * (initial_sales + price_reduction_effect * (initial_price - x)) = profit_target ∧ x ≤ max_price :=
by sorry

end daily_sales_volume_80_sales_volume_function_price_for_profit_l709_70968


namespace quadrilateral_area_inequality_l709_70941

theorem quadrilateral_area_inequality (a b c d : ℝ) :
  ∃ (S_ABCD : ℝ), S_ABCD ≤ (1 / 4) * (a + c) ^ 2 + b * d :=
sorry

end quadrilateral_area_inequality_l709_70941


namespace number_of_children_l709_70945

def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6
def number_of_adults := 2
def total_cost := 77

theorem number_of_children : 
  ∃ (x : ℕ), cost_of_child_ticket * x + cost_of_adult_ticket * number_of_adults = total_cost ∧ x = 3 :=
by
  sorry

end number_of_children_l709_70945


namespace mobius_total_trip_time_l709_70982

theorem mobius_total_trip_time :
  ∀ (d1 d2 v1 v2 : ℝ) (n r : ℕ),
  d1 = 143 → d2 = 143 → 
  v1 = 11 → v2 = 13 → 
  n = 4 → r = (30:ℝ)/60 →
  d1 / v1 + d2 / v2 + n * r = 26 :=
by
  intros d1 d2 v1 v2 n r h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num

end mobius_total_trip_time_l709_70982


namespace num_br_atoms_l709_70956

theorem num_br_atoms (num_br : ℕ) : 
  (1 * 1 + num_br * 80 + 3 * 16 = 129) → num_br = 1 :=
  by
    intro h
    sorry

end num_br_atoms_l709_70956


namespace x_pow_n_plus_inv_x_pow_n_l709_70935

theorem x_pow_n_plus_inv_x_pow_n (θ : ℝ) (x : ℝ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : x + 1 / x = 2 * Real.sin θ) (hn_pos : 0 < n) : 
  x^n + (1 / x)^n = 2 * Real.cos (n * θ) := 
by
  sorry

end x_pow_n_plus_inv_x_pow_n_l709_70935


namespace incorrect_statement_is_A_l709_70949

theorem incorrect_statement_is_A :
  (∀ (w h : ℝ), w * (2 * h) ≠ 3 * (w * h)) ∧
  (∀ (s : ℝ), (2 * s) ^ 2 = 4 * (s ^ 2)) ∧
  (∀ (s : ℝ), (2 * s) ^ 3 = 8 * (s ^ 3)) ∧
  (∀ (w h : ℝ), (w / 2) * (3 * h) = (3 / 2) * (w * h)) ∧
  (∀ (l w : ℝ), (2 * l) * (3 * w) = 6 * (l * w)) →
  ∃ (incorrect_statement : String), incorrect_statement = "A" := 
by 
  sorry

end incorrect_statement_is_A_l709_70949


namespace ratio_seniors_to_juniors_l709_70927

variable (j s : ℕ)

-- Condition: \(\frac{3}{7}\) of the juniors participated is equal to \(\frac{6}{7}\) of the seniors participated
def participation_condition (j s : ℕ) : Prop :=
  3 * j = 6 * s

-- Theorem to be proved: the ratio of seniors to juniors is \( \frac{1}{2} \)
theorem ratio_seniors_to_juniors (j s : ℕ) (h : participation_condition j s) : s / j = 1 / 2 :=
  sorry

end ratio_seniors_to_juniors_l709_70927


namespace point_in_first_or_third_quadrant_l709_70980

-- Definitions based on conditions
variables {x y : ℝ}

-- The proof statement
theorem point_in_first_or_third_quadrant (h : x * y > 0) : 
  (0 < x ∧ 0 < y) ∨ (x < 0 ∧ y < 0) :=
  sorry

end point_in_first_or_third_quadrant_l709_70980


namespace mowed_times_in_spring_l709_70953

-- Definition of the problem conditions
def total_mowed_times : ℕ := 11
def summer_mowed_times : ℕ := 5

-- The theorem to prove
theorem mowed_times_in_spring : (total_mowed_times - summer_mowed_times = 6) :=
by
  sorry

end mowed_times_in_spring_l709_70953


namespace solve_quadratic_equation_l709_70992

theorem solve_quadratic_equation (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
sorry

end solve_quadratic_equation_l709_70992


namespace overall_percent_change_in_stock_l709_70974

noncomputable def stock_change (initial_value : ℝ) : ℝ :=
  let value_after_first_day := 0.85 * initial_value
  let value_after_second_day := 1.25 * value_after_first_day
  (value_after_second_day - initial_value) / initial_value * 100

theorem overall_percent_change_in_stock (x : ℝ) : stock_change x = 6.25 :=
by
  sorry

end overall_percent_change_in_stock_l709_70974


namespace calculate_P_AB_l709_70916

section Probability
-- Define the given probabilities
variables (P_B_given_A : ℚ) (P_A : ℚ)
-- Given conditions
def given_conditions := P_B_given_A = 3/10 ∧ P_A = 1/5

-- Prove that P(AB) = 3/50
theorem calculate_P_AB (h : given_conditions P_B_given_A P_A) : (P_A * P_B_given_A) = 3/50 :=
by
  rcases h with ⟨h1, h2⟩
  simp [h1, h2]
  -- Here we would include the steps leading to the conclusion; this part just states the theorem
  sorry

end Probability

end calculate_P_AB_l709_70916


namespace find_a_l709_70994

theorem find_a (a : ℝ) :
  (∃x y : ℝ, x^2 + y^2 + 2 * x - 2 * y + a = 0 ∧ x + y + 4 = 0) →
  ∃c : ℝ, c = 2 ∧ a = -7 :=
by
  -- proof to be filled in
  sorry

end find_a_l709_70994


namespace no_real_k_for_distinct_roots_l709_70979

theorem no_real_k_for_distinct_roots (k : ℝ) : ¬ ( -8 * k^2 > 0 ) := 
by
  sorry

end no_real_k_for_distinct_roots_l709_70979


namespace coins_in_pockets_l709_70985

theorem coins_in_pockets : (Nat.choose (5 + 3 - 1) (3 - 1)) = 21 := by
  sorry

end coins_in_pockets_l709_70985


namespace sum_of_three_consecutive_even_nums_l709_70911

theorem sum_of_three_consecutive_even_nums : 80 + 82 + 84 = 246 := by
  sorry

end sum_of_three_consecutive_even_nums_l709_70911


namespace circle_area_pi_div_2_l709_70938

open Real EuclideanGeometry

variable (x y : ℝ)

def circleEquation : Prop := 3 * x^2 + 3 * y^2 - 15 * x + 9 * y + 27 = 0

theorem circle_area_pi_div_2
  (h : circleEquation x y) : 
  ∃ (r : ℝ), r = sqrt 0.5 ∧ π * r * r = π / 2 :=
by
  sorry

end circle_area_pi_div_2_l709_70938


namespace number_of_students_increased_l709_70901

theorem number_of_students_increased
  (original_number_of_students : ℕ) (increase_in_expenses : ℕ) (diminshed_average_expenditure : ℕ)
  (original_expenditure : ℕ) (increase_in_students : ℕ) :
  original_number_of_students = 35 →
  increase_in_expenses = 42 →
  diminshed_average_expenditure = 1 →
  original_expenditure = 420 →
  (35 + increase_in_students) * (12 - 1) - 420 = 42 →
  increase_in_students = 7 :=
by
  intros
  sorry

end number_of_students_increased_l709_70901


namespace multiply_binomials_l709_70963

theorem multiply_binomials (x : ℝ) : (4 * x + 3) * (2 * x - 7) = 8 * x^2 - 22 * x - 21 :=
by 
  -- Proof is to be filled here
  sorry

end multiply_binomials_l709_70963


namespace courier_cost_formula_l709_70991

def cost (P : ℕ) : ℕ :=
if P = 0 then 0 else max 50 (30 + 7 * (P - 1))

theorem courier_cost_formula (P : ℕ) : cost P = 
  if P = 0 then 0 else max 50 (30 + 7 * (P - 1)) :=
by
  sorry

end courier_cost_formula_l709_70991


namespace total_cost_accurate_l709_70954

def price_iphone: ℝ := 800
def price_iwatch: ℝ := 300
def price_ipad: ℝ := 500

def discount_iphone: ℝ := 0.15
def discount_iwatch: ℝ := 0.10
def discount_ipad: ℝ := 0.05

def tax_iphone: ℝ := 0.07
def tax_iwatch: ℝ := 0.05
def tax_ipad: ℝ := 0.06

def cashback: ℝ := 0.02

theorem total_cost_accurate:
  let discounted_auction (price: ℝ) (discount: ℝ) := price * (1 - discount)
  let taxed_auction (price: ℝ) (tax: ℝ) := price * (1 + tax)
  let total_cost :=
    let discount_iphone_cost := discounted_auction price_iphone discount_iphone
    let discount_iwatch_cost := discounted_auction price_iwatch discount_iwatch
    let discount_ipad_cost := discounted_auction price_ipad discount_ipad
    
    let tax_iphone_cost := taxed_auction discount_iphone_cost tax_iphone
    let tax_iwatch_cost := taxed_auction discount_iwatch_cost tax_iwatch
    let tax_ipad_cost := taxed_auction discount_ipad_cost tax_ipad
    
    let total_price := tax_iphone_cost + tax_iwatch_cost + tax_ipad_cost
    total_price * (1 - cashback)
  total_cost = 1484.31 := 
  by sorry

end total_cost_accurate_l709_70954


namespace trains_meet_at_10_am_l709_70924

def distance (speed time : ℝ) : ℝ := speed * time

theorem trains_meet_at_10_am
  (distance_pq : ℝ)
  (speed_train_from_p : ℝ)
  (start_time_from_p : ℝ)
  (speed_train_from_q : ℝ)
  (start_time_from_q : ℝ)
  (meeting_time : ℝ) :
  distance_pq = 110 → 
  speed_train_from_p = 20 → 
  start_time_from_p = 7 → 
  speed_train_from_q = 25 → 
  start_time_from_q = 8 → 
  meeting_time = 10 :=
by
  sorry

end trains_meet_at_10_am_l709_70924


namespace largest_number_of_hcf_lcm_l709_70946

theorem largest_number_of_hcf_lcm (a b c : ℕ) (h : Nat.gcd (Nat.gcd a b) c = 42)
  (factor1 : 10 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor2 : 20 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor3 : 25 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor4 : 30 ∣ Nat.lcm (Nat.lcm a b) c) :
  max (max a b) c = 1260 := 
  sorry

end largest_number_of_hcf_lcm_l709_70946


namespace min_value_a_l709_70923

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ (Real.sqrt 2) / 2 → x^3 - 2 * x * Real.log x / Real.log a ≤ 0) ↔ a ≥ 1 / 4 := 
sorry

end min_value_a_l709_70923


namespace diagonal_of_square_l709_70950

theorem diagonal_of_square (length_rect width_rect : ℝ) (h1 : length_rect = 45) (h2 : width_rect = 40)
  (area_rect : ℝ) (h3 : area_rect = length_rect * width_rect) (area_square : ℝ) (h4 : area_square = area_rect)
  (side_square : ℝ) (h5 : side_square^2 = area_square) (diagonal_square : ℝ) (h6 : diagonal_square = side_square * Real.sqrt 2) :
  diagonal_square = 60 := by
  sorry

end diagonal_of_square_l709_70950


namespace numerator_in_second_fraction_l709_70981

theorem numerator_in_second_fraction (p q x: ℚ) (h1 : p / q = 4 / 5) (h2 : 11 / 7 + x / (2 * q + p) = 2) : x = 6 :=
sorry

end numerator_in_second_fraction_l709_70981


namespace number_of_men_at_picnic_l709_70976

theorem number_of_men_at_picnic (total persons W M A C : ℕ) (h1 : total = 200) 
  (h2 : M = W + 20) (h3 : A = C + 20) (h4 : A = M + W) : M = 65 :=
by
  -- Proof can be filled in here
  sorry

end number_of_men_at_picnic_l709_70976


namespace jed_change_l709_70966

theorem jed_change :
  ∀ (num_games : ℕ) (cost_per_game : ℕ) (payment : ℕ) (bill_value : ℕ),
  num_games = 6 →
  cost_per_game = 15 →
  payment = 100 →
  bill_value = 5 →
  (payment - num_games * cost_per_game) / bill_value = 2 :=
by
  intros num_games cost_per_game payment bill_value
  sorry

end jed_change_l709_70966


namespace sufficient_but_not_necessary_l709_70978

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x > 1) : x < 1 := by
  sorry

end sufficient_but_not_necessary_l709_70978


namespace roses_in_centerpiece_l709_70957

variable (r : ℕ)

theorem roses_in_centerpiece (h : 6 * 15 * (3 * r + 6) = 2700) : r = 8 := 
  sorry

end roses_in_centerpiece_l709_70957


namespace contrapositive_of_proposition_l709_70959

theorem contrapositive_of_proposition :
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by
  sorry

end contrapositive_of_proposition_l709_70959


namespace min_value_l709_70931

variable (a b c : ℝ)

theorem min_value (h1 : a > b) (h2 : b > c) (h3 : a - c = 5) : 
  (a - b) ^ 2 + (b - c) ^ 2 = 25 / 2 := 
sorry

end min_value_l709_70931


namespace symmetric_line_l709_70958

theorem symmetric_line (y : ℝ → ℝ) (h : ∀ x, y x = 2 * x + 1) :
  ∀ x, y (-x) = -2 * x + 1 :=
by
  -- Proof skipped
  sorry

end symmetric_line_l709_70958


namespace group_value_21_le_a_lt_41_l709_70970

theorem group_value_21_le_a_lt_41 : 
  (∀ a: ℤ, 21 ≤ a ∧ a < 41 → (21 + 41) / 2 = 31) :=
by 
  sorry

end group_value_21_le_a_lt_41_l709_70970


namespace grace_wins_probability_l709_70919

def probability_grace_wins : ℚ :=
  let total_possible_outcomes := 36
  let losing_combinations := 6
  let winning_combinations := total_possible_outcomes - losing_combinations
  winning_combinations / total_possible_outcomes

theorem grace_wins_probability :
    probability_grace_wins = 5 / 6 := by
  sorry

end grace_wins_probability_l709_70919


namespace basketball_games_l709_70973

theorem basketball_games (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 3 * N + 4 * M = 88) : 3 * N = 48 :=
by sorry

end basketball_games_l709_70973


namespace sequence_general_formula_l709_70929

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- because sequences in the solution are 1-indexed.
  | 1 => 2
  | k+2 => sequence (k+1) + 3 * (k+1)

theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : 
  sequence n = 2 + 3 * n * (n - 1) / 2 :=
by
  sorry

#eval sequence 1  -- should output 2
#eval sequence 2  -- should output 5
#eval sequence 3  -- should output 11
#eval sequence 4  -- should output 20
#eval sequence 5  -- should output 32
#eval sequence 6  -- should output 47

end sequence_general_formula_l709_70929


namespace customer_paid_correct_amount_l709_70961

noncomputable def cost_price : ℝ := 5565.217391304348
noncomputable def markup_percentage : ℝ := 0.15
noncomputable def markup_amount (cost : ℝ) : ℝ := cost * markup_percentage
noncomputable def final_price (cost : ℝ) (markup : ℝ) : ℝ := cost + markup

theorem customer_paid_correct_amount :
  final_price cost_price (markup_amount cost_price) = 6400 := sorry

end customer_paid_correct_amount_l709_70961


namespace solve_stamps_l709_70964

noncomputable def stamps_problem : Prop :=
  ∃ (A B C D : ℝ), 
    A + B + C + D = 251 ∧
    A = 2 * B + 2 ∧
    A = 3 * C + 6 ∧
    A = 4 * D - 16 ∧
    D = 32

theorem solve_stamps : stamps_problem :=
sorry

end solve_stamps_l709_70964


namespace rearrange_rooks_possible_l709_70952

theorem rearrange_rooks_possible (board : Fin 8 × Fin 8 → Prop) (rooks : Fin 8 → Fin 8 × Fin 8) (painted : Fin 8 × Fin 8 → Prop) :
  (∀ i j : Fin 8, i ≠ j → (rooks i).1 ≠ (rooks j).1 ∧ (rooks i).2 ≠ (rooks j).2) → -- no two rooks are in the same row or column
  (∃ (unpainted_count : ℕ), (unpainted_count = 64 - 27)) → -- 27 squares are painted red
  (∃ new_rooks : Fin 8 → Fin 8 × Fin 8,
    (∀ i : Fin 8, ¬painted (new_rooks i)) ∧ -- all rooks are on unpainted squares
    (∀ i j : Fin 8, i ≠ j → (new_rooks i).1 ≠ (new_rooks j).1 ∧ (new_rooks i).2 ≠ (new_rooks j).2) ∧ -- no two rooks are in the same row or column
    (∃ i : Fin 8, rooks i ≠ new_rooks i)) -- at least one rook has moved
:=
sorry

end rearrange_rooks_possible_l709_70952


namespace base_6_conversion_l709_70986

-- Define the conditions given in the problem
def base_6_to_10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

-- given that 524_6 = 2cd_10 and c, d are base-10 digits, prove that (c * d) / 12 = 3/4
theorem base_6_conversion (c d : ℕ) (h1 : base_6_to_10 5 2 4 = 196) (h2 : 2 * 10 * c + d = 196) :
  (c * d) / 12 = 3 / 4 :=
sorry

end base_6_conversion_l709_70986


namespace cannot_form_right_triangle_l709_70913

theorem cannot_form_right_triangle : ¬∃ a b c : ℕ, a = 4 ∧ b = 6 ∧ c = 11 ∧ (a^2 + b^2 = c^2) :=
by
  sorry

end cannot_form_right_triangle_l709_70913


namespace x_squared_y_plus_xy_squared_l709_70988

-- Define the variables and their conditions
variables {x y : ℝ}

-- Define the theorem stating that if xy = 3 and x + y = 5, then x^2y + xy^2 = 15
theorem x_squared_y_plus_xy_squared (h1 : x * y = 3) (h2 : x + y = 5) : x^2 * y + x * y^2 = 15 :=
by {
  sorry
}

end x_squared_y_plus_xy_squared_l709_70988


namespace Namjoon_walk_extra_l709_70969

-- Define the usual distance Namjoon walks to school
def usual_distance := 1.2

-- Define the distance Namjoon walked to the intermediate point
def intermediate_distance := 0.3

-- Define the total distance Namjoon walked today
def total_distance_today := (intermediate_distance * 2) + usual_distance

-- Define the extra distance walked today compared to usual
def extra_distance := total_distance_today - usual_distance

-- State the theorem to prove that the extra distance walked today is 0.6 km
theorem Namjoon_walk_extra : extra_distance = 0.6 := 
by
  sorry

end Namjoon_walk_extra_l709_70969


namespace sufficient_condition_for_perpendicular_l709_70925

variables {Plane : Type} {Line : Type} 
variables (α β γ : Plane) (m n : Line)

-- Definitions based on conditions
variables (perpendicular : Plane → Plane → Prop)
variables (perpendicular_line : Line → Plane → Prop)
variables (intersection : Plane → Plane → Line)

-- Conditions from option D
variable (h1 : perpendicular_line n α)
variable (h2 : perpendicular_line n β)
variable (h3 : perpendicular_line m α)

-- Statement to prove
theorem sufficient_condition_for_perpendicular (h1 : perpendicular_line n α)
  (h2 : perpendicular_line n β) (h3 : perpendicular_line m α) : 
  perpendicular_line m β := 
sorry

end sufficient_condition_for_perpendicular_l709_70925


namespace average_book_width_correct_l709_70933

noncomputable def average_book_width 
  (widths : List ℚ) (number_of_books : ℕ) : ℚ :=
(widths.sum) / number_of_books

theorem average_book_width_correct :
  average_book_width [5, 3/4, 1.5, 3, 7.25, 12] 6 = 59 / 12 := 
  by 
  sorry

end average_book_width_correct_l709_70933


namespace namjoon_rank_l709_70999

theorem namjoon_rank (total_students : ℕ) (fewer_than_namjoon : ℕ) (rank_of_namjoon : ℕ) 
  (h1 : total_students = 13) (h2 : fewer_than_namjoon = 4) : rank_of_namjoon = 9 :=
sorry

end namjoon_rank_l709_70999


namespace largest_among_abc_l709_70928

theorem largest_among_abc
  (x : ℝ) 
  (hx : 0 < x) 
  (hx1 : x < 1)
  (a : ℝ)
  (ha : a = 2 * Real.sqrt x )
  (b : ℝ)
  (hb : b = 1 + x)
  (c : ℝ)
  (hc : c = 1 / (1 - x)) 
  : a < b ∧ b < c :=
by
  sorry

end largest_among_abc_l709_70928


namespace min_shift_for_even_function_l709_70967

theorem min_shift_for_even_function :
  ∃ (m : ℝ), (m > 0) ∧ (∀ x : ℝ, (Real.sin (x + m) + Real.cos (x + m)) = (Real.sin (-x + m) + Real.cos (-x + m))) ∧ m = π / 4 :=
by
  sorry

end min_shift_for_even_function_l709_70967


namespace sum_of_first_three_cards_l709_70906

theorem sum_of_first_three_cards :
  ∀ (G Y : ℕ → ℕ) (cards : ℕ → ℕ),
  (∀ n, G n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) →
  (∀ n, Y n ∈ ({4, 5, 6, 7, 8} : Set ℕ)) →
  (∀ n, cards (2 * n) = G (cards n) → cards (2 * n + 1) = Y (cards n + 1)) →
  (∀ n, Y n = G (n + 1) ∨ ∃ k, Y n = k * G (n + 1)) →
  (cards 0 + cards 1 + cards 2 = 14) :=
by
  sorry

end sum_of_first_three_cards_l709_70906


namespace sqrt_81_eq_9_l709_70908

theorem sqrt_81_eq_9 : Real.sqrt 81 = 9 :=
by
  sorry

end sqrt_81_eq_9_l709_70908


namespace distinct_solutions_difference_eq_sqrt29_l709_70904

theorem distinct_solutions_difference_eq_sqrt29 :
  (∃ a b : ℝ, a > b ∧
    (∀ x : ℝ, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ 
      x = a ∨ x = b) ∧ 
    a - b = Real.sqrt 29) :=
sorry

end distinct_solutions_difference_eq_sqrt29_l709_70904


namespace largest_constant_c_l709_70902

theorem largest_constant_c (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y^2 = 1) : 
  x^6 + y^6 ≥ (1 / 2) * x * y :=
sorry

end largest_constant_c_l709_70902


namespace product_of_digits_of_nondivisible_by_5_number_is_30_l709_70972

-- Define the four-digit numbers
def numbers : List ℕ := [4825, 4835, 4845, 4855, 4865]

-- Define units and tens digit function
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

-- Assertion that 4865 is the number that is not divisible by 5
def not_divisible_by_5 (n : ℕ) : Prop := ¬ (units_digit n = 5 ∨ units_digit n = 0)

-- Lean 4 statement to prove the product of units and tens digit of the number not divisible by 5 is 30
theorem product_of_digits_of_nondivisible_by_5_number_is_30 :
  ∃ n ∈ numbers, not_divisible_by_5 n ∧ (units_digit n) * (tens_digit n) = 30 :=
by
  sorry

end product_of_digits_of_nondivisible_by_5_number_is_30_l709_70972


namespace ratio_of_bubbles_l709_70918

def bubbles_dawn_per_ounce : ℕ := 200000

def mixture_bubbles (bubbles_other_per_ounce : ℕ) : ℕ :=
  let half_ounce_dawn := bubbles_dawn_per_ounce / 2
  let half_ounce_other := bubbles_other_per_ounce / 2
  half_ounce_dawn + half_ounce_other

noncomputable def find_ratio (bubbles_other_per_ounce : ℕ) : ℚ :=
  (bubbles_other_per_ounce : ℚ) / bubbles_dawn_per_ounce

theorem ratio_of_bubbles
  (bubbles_other_per_ounce : ℕ)
  (h_mixture : mixture_bubbles bubbles_other_per_ounce = 150000) :
  find_ratio bubbles_other_per_ounce = 1 / 2 :=
by
  sorry

end ratio_of_bubbles_l709_70918


namespace total_cost_after_discounts_l709_70942

-- Definition of the cost function with applicable discounts
def pencil_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

def pen_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

-- The statement to be proved
theorem total_cost_after_discounts :
  let pencil_price := 2.50
  let pen_price := 3.50
  let pencil_count := 38
  let pen_count := 56
  let pencil_discount_threshold := 30
  let pencil_discount_rate := 0.10
  let pen_discount_threshold := 50
  let pen_discount_rate := 0.15
  let total_cost := pencil_cost pencil_price pencil_count pencil_discount_threshold pencil_discount_rate
                   + pen_cost pen_price pen_count pen_discount_threshold pen_discount_rate
  total_cost = 252.10 := 
by 
  sorry

end total_cost_after_discounts_l709_70942


namespace cauliflower_area_l709_70944

theorem cauliflower_area
  (s : ℕ) (a : ℕ) 
  (H1 : s * s / a = 40401)
  (H2 : s * s / a = 40000) :
  a = 1 :=
sorry

end cauliflower_area_l709_70944


namespace function_domain_l709_70993

noncomputable def domain_function (x : ℝ) : Prop :=
  x > 0 ∧ (Real.log x / Real.log 2)^2 - 1 > 0

theorem function_domain :
  { x : ℝ | domain_function x } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end function_domain_l709_70993


namespace valentine_problem_l709_70971

def initial_valentines : ℕ := 30
def given_valentines : ℕ := 8
def remaining_valentines : ℕ := 22

theorem valentine_problem : initial_valentines - given_valentines = remaining_valentines := by
  sorry

end valentine_problem_l709_70971


namespace weight_of_steel_rod_l709_70955

theorem weight_of_steel_rod (length1 : ℝ) (weight1 : ℝ) (length2 : ℝ) (weight2 : ℝ) 
  (h1 : length1 = 9) (h2 : weight1 = 34.2) (h3 : length2 = 11.25) : 
  weight2 = (weight1 / length1) * length2 :=
by
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end weight_of_steel_rod_l709_70955


namespace quadratic_other_x_intercept_l709_70934

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → (a * x^2 + b * x + c) = -3)
  (h_intercept : ∀ x, x = 1 → (a * x^2 + b * x + c) = 0) : 
  ∃ x : ℝ, x = 9 ∧ (a * x^2 + b * x + c) = 0 :=
sorry

end quadratic_other_x_intercept_l709_70934


namespace Ponchik_week_day_l709_70939

theorem Ponchik_week_day (n s : ℕ) (h1 : s = 20) (h2 : s * (4 * n + 1) = 1360) : n = 4 :=
by
  sorry

end Ponchik_week_day_l709_70939


namespace total_revenue_l709_70943

def price_federal := 50
def price_state := 30
def price_quarterly := 80
def num_federal := 60
def num_state := 20
def num_quarterly := 10

theorem total_revenue : (num_federal * price_federal + num_state * price_state + num_quarterly * price_quarterly) = 4400 := by
  sorry

end total_revenue_l709_70943


namespace find_n_positive_integers_l709_70905

theorem find_n_positive_integers :
  ∀ n : ℕ, 0 < n →
  (∃ k : ℕ, (n^2 + 11 * n - 4) * n! + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by
  sorry

end find_n_positive_integers_l709_70905


namespace simplest_radical_form_l709_70910

def is_simplest_radical_form (r : ℝ) : Prop :=
  ∀ x : ℝ, x * x = r → ∃ y : ℝ, y * y ≠ r

theorem simplest_radical_form :
   (is_simplest_radical_form 6) :=
by
  sorry

end simplest_radical_form_l709_70910


namespace rose_age_l709_70926

variable {R M : ℝ}

theorem rose_age (h1 : R = (1/3) * M) (h2 : R + M = 100) : R = 25 :=
sorry

end rose_age_l709_70926


namespace proof_problem_l709_70947

/-- 
  Given:
  - r, j, z are Ryan's, Jason's, and Zachary's earnings respectively.
  - Zachary sold 40 games at $5 each.
  - Jason received 30% more money than Zachary.
  - The total amount of money received by all three is $770.
  Prove:
  - Ryan received $50 more than Jason.
--/
def problem_statement : Prop :=
  ∃ (r j z : ℕ), 
    z = 40 * 5 ∧
    j = z + z * 30 / 100 ∧
    r + j + z = 770 ∧ 
    r - j = 50

theorem proof_problem : problem_statement :=
by 
  sorry

end proof_problem_l709_70947


namespace fraction_of_jam_eaten_for_dinner_l709_70987

-- Define the problem
theorem fraction_of_jam_eaten_for_dinner :
  ∃ (J : ℝ) (x : ℝ), 
  J > 0 ∧
  (1 / 3) * J + (x * (2 / 3) * J) + (4 / 7) * J = J ∧
  x = 1 / 7 :=
by
  sorry

end fraction_of_jam_eaten_for_dinner_l709_70987


namespace polynomial_root_l709_70922

theorem polynomial_root (x0 : ℝ) (z : ℝ) 
  (h1 : x0^3 - x0 - 1 = 0) 
  (h2 : z = x0^2 + 3 * x0 + 1) : 
  z^3 - 5 * z^2 - 10 * z - 11 = 0 := 
sorry

end polynomial_root_l709_70922


namespace permutation_6_4_l709_70997

theorem permutation_6_4 : (Nat.factorial 6) / (Nat.factorial (6 - 4)) = 360 := by
  sorry

end permutation_6_4_l709_70997


namespace probability_two_red_books_l709_70975

theorem probability_two_red_books (total_books red_books blue_books selected_books : ℕ)
  (h_total: total_books = 8)
  (h_red: red_books = 4)
  (h_blue: blue_books = 4)
  (h_selected: selected_books = 2) :
  (Nat.choose red_books selected_books : ℚ) / (Nat.choose total_books selected_books) = 3 / 14 := by
  sorry

end probability_two_red_books_l709_70975


namespace hoseok_subtraction_result_l709_70915

theorem hoseok_subtraction_result:
  ∃ x : ℤ, 15 * x = 45 ∧ x - 1 = 2 :=
by
  sorry

end hoseok_subtraction_result_l709_70915


namespace find_larger_number_l709_70984

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1390)
  (h2 : L = 6 * S + 15) : 
  L = 1665 :=
sorry

end find_larger_number_l709_70984


namespace simplify_expression_l709_70909

theorem simplify_expression :
  (120^2 - 9^2) / (90^2 - 18^2) * ((90 - 18) * (90 + 18)) / ((120 - 9) * (120 + 9)) = 1 := by
  sorry

end simplify_expression_l709_70909


namespace least_subtracted_number_correct_l709_70932

noncomputable def least_subtracted_number (n : ℕ) : ℕ :=
  n - 13

theorem least_subtracted_number_correct (n : ℕ) : 
  least_subtracted_number 997 = 997 - 13 ∧
  (least_subtracted_number 997 % 5 = 3) ∧
  (least_subtracted_number 997 % 9 = 3) ∧
  (least_subtracted_number 997 % 11 = 3) :=
by
  let x := 997 - 13
  have : x = 984 := rfl
  have h5 : x % 5 = 3 := by sorry
  have h9 : x % 9 = 3 := by sorry
  have h11 : x % 11 = 3 := by sorry
  exact ⟨rfl, h5, h9, h11⟩

end least_subtracted_number_correct_l709_70932


namespace roots_sum_product_l709_70921

variable {a b : ℝ}

theorem roots_sum_product (ha : a + b = 6) (hp : a * b = 8) : 
  a^4 + b^4 + a^3 * b + a * b^3 = 432 :=
by
  sorry

end roots_sum_product_l709_70921


namespace angle_PDO_45_degrees_l709_70940

-- Define the square configuration
variables (A B C D L P Q M N O : Type)
variables (a : ℝ) -- side length of the square ABCD

-- Conditions as hypothesized in the problem
def is_square (v₁ v₂ v₃ v₄ : Type) := true -- Placeholder for the square property
def on_diagonal_AC (L : Type) := true -- Placeholder for L being on diagonal AC
def common_vertex_L (sq1_v1 sq1_v2 sq1_v3 sq1_v4 sq2_v1 sq2_v2 sq2_v3 sq2_v4 : Type) := true -- Placeholder for common vertex L
def point_on_side (P AB_side: Type) := true -- Placeholder for P on side AB of ABCD
def square_center (center sq_v1 sq_v2 sq_v3 sq_v4 : Type) := true -- Placeholder for square's center

-- Prove the angle PDO is 45 degrees
theorem angle_PDO_45_degrees 
  (h₁ : is_square A B C D)
  (h₂ : on_diagonal_AC L)
  (h₃ : is_square A P L Q)
  (h₄ : is_square C M L N)
  (h₅ : common_vertex_L A P L Q C M L N)
  (h₆ : point_on_side P B)
  (h₇ : square_center O C M L N)
  : ∃ θ : ℝ, θ = 45 := 
  sorry

end angle_PDO_45_degrees_l709_70940


namespace fraction_sum_identity_l709_70920

theorem fraction_sum_identity (p q r : ℝ) (h₀ : p ≠ q) (h₁ : p ≠ r) (h₂ : q ≠ r) 
(h : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 1 / (q - r) + 1 / (r - p) + 1 / (p - q) - 1 := 
sorry

end fraction_sum_identity_l709_70920


namespace remainder_approx_l709_70998

def x : ℝ := 74.99999999999716 * 96
def y : ℝ := 74.99999999999716
def quotient : ℝ := 96
def expected_remainder : ℝ := 0.4096

theorem remainder_approx (x y : ℝ) (quotient : ℝ) (h1 : y = 74.99999999999716)
  (h2 : quotient = 96) (h3 : x = y * quotient) :
  x - y * quotient = expected_remainder :=
by
  sorry

end remainder_approx_l709_70998


namespace speed_of_second_part_l709_70930

theorem speed_of_second_part
  (total_distance : ℝ)
  (distance_part1 : ℝ)
  (speed_part1 : ℝ)
  (average_speed : ℝ)
  (speed_part2 : ℝ) :
  total_distance = 70 →
  distance_part1 = 35 →
  speed_part1 = 48 →
  average_speed = 32 →
  speed_part2 = 24 :=
by
  sorry

end speed_of_second_part_l709_70930


namespace production_today_l709_70903

theorem production_today (n : ℕ) (P T : ℕ) 
  (h1 : n = 4) 
  (h2 : (P + T) / (n + 1) = 58) 
  (h3 : P = n * 50) : 
  T = 90 := 
by
  sorry

end production_today_l709_70903


namespace minimum_bottles_needed_l709_70960

theorem minimum_bottles_needed (fl_oz_needed : ℝ) (bottle_size_ml : ℝ) (fl_oz_per_liter : ℝ) (ml_per_liter : ℝ)
  (h1 : fl_oz_needed = 60)
  (h2 : bottle_size_ml = 250)
  (h3 : fl_oz_per_liter = 33.8)
  (h4 : ml_per_liter = 1000) :
  ∃ n : ℕ, n = 8 ∧ fl_oz_needed * ml_per_liter / fl_oz_per_liter / bottle_size_ml ≤ n :=
by
  sorry

end minimum_bottles_needed_l709_70960


namespace problem_a_lt_c_lt_b_l709_70948

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem problem_a_lt_c_lt_b : a < c ∧ c < b := 
by {
  sorry
}

end problem_a_lt_c_lt_b_l709_70948


namespace days_at_grandparents_l709_70965

theorem days_at_grandparents
  (total_vacation_days : ℕ)
  (travel_to_gp : ℕ)
  (travel_to_brother : ℕ)
  (days_at_brother : ℕ)
  (travel_to_sister : ℕ)
  (days_at_sister : ℕ)
  (travel_home : ℕ)
  (total_days : total_vacation_days = 21) :
  total_vacation_days - (travel_to_gp + travel_to_brother + days_at_brother + travel_to_sister + days_at_sister + travel_home) = 5 :=
by
  sorry -- proof to be constructed

end days_at_grandparents_l709_70965


namespace cone_central_angle_l709_70917

theorem cone_central_angle (l : ℝ) (α : ℝ) (h : (30 : ℝ) * π / 180 > 0) :
  α = π := 
sorry

end cone_central_angle_l709_70917


namespace max_elements_of_S_l709_70983

-- Define the relation on set S and the conditions given
variable {S : Type} (R : S → S → Prop)

-- Lean translation of the conditions
def condition_1 (a b : S) : Prop :=
  (R a b ∨ R b a) ∧ ¬ (R a b ∧ R b a)

def condition_2 (a b c : S) : Prop :=
  R a b ∧ R b c → R c a

-- Define the problem statement:
theorem max_elements_of_S (h1 : ∀ a b : S, condition_1 R a b)
                          (h2 : ∀ a b c : S, condition_2 R a b c) :
  ∃ (n : ℕ), (∀ T : Finset S, T.card ≤ n) ∧ (∃ T : Finset S, T.card = 3) :=
sorry

end max_elements_of_S_l709_70983


namespace time_after_2023_minutes_l709_70912

def start_time : Nat := 1 * 60 -- Start time is 1:00 a.m. in minutes from midnight, which is 60 minutes.
def elapsed_time : Nat := 2023 -- The elapsed time is 2023 minutes.

theorem time_after_2023_minutes : (start_time + elapsed_time) % 1440 = 643 := 
by
  -- 1440 represents the total minutes in a day (24 hours * 60 minutes).
  -- 643 represents the time 10:43 a.m. in minutes from midnight. This is obtained as 10 * 60 + 43 = 643.
  sorry

end time_after_2023_minutes_l709_70912


namespace inverse_of_f_at_2_l709_70977

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem inverse_of_f_at_2 : ∀ x, x ≥ 0 → f x = 2 → x = Real.sqrt 3 :=
by
  intro x hx heq
  sorry

end inverse_of_f_at_2_l709_70977


namespace triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l709_70900

-- Define the properties and variables of the given obtuse triangle
variables (a b c : ℝ) (A C : ℝ)
-- Given conditions
axiom ha : a = 7
axiom hb : b = 3
axiom hcosC : Real.cos C = 11 / 14

-- Prove the values of c and angle A
theorem triangle_ABC_c_and_A_value (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : c = 5 ∧ A = 2 * Real.pi / 3 :=
sorry

-- Prove the value of sin(2C - π / 6)
theorem sin_2C_minus_pi_6 (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : Real.sin (2 * C - Real.pi / 6) = 71 / 98 :=
sorry

end triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l709_70900


namespace eval_expr_l709_70990

-- Define the expression
def expr : ℚ := 2 + 3 / (2 + 1 / (2 + 1 / 2))

-- The theorem to prove the evaluation of the expression
theorem eval_expr : expr = 13 / 4 :=
by
  sorry

end eval_expr_l709_70990


namespace cylinder_height_decrease_l709_70914

/--
Two right circular cylinders have the same volume. The radius of the second cylinder is 20% more than the radius
of the first. Prove that the height of the second cylinder is approximately 30.56% less than the first one's height.
-/
theorem cylinder_height_decrease (r1 h1 r2 h2 : ℝ) (hradius : r2 = 1.2 * r1) (hvolumes : π * r1^2 * h1 = π * r2^2 * h2) :
  h2 = 25 / 36 * h1 :=
by
  sorry

end cylinder_height_decrease_l709_70914


namespace slope_parallel_line_l709_70989

theorem slope_parallel_line (x y : ℝ) (a b c : ℝ) (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = 1 / 2 :=
by 
  sorry

end slope_parallel_line_l709_70989


namespace neg_p_l709_70937

-- Define the sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- Define the proposition p
def p : Prop := ∀ x : ℤ, is_odd x → is_even (2 * x)

-- State the negation of proposition p
theorem neg_p : ¬ p ↔ ∃ x : ℤ, is_odd x ∧ ¬ is_even (2 * x) := by sorry

end neg_p_l709_70937
