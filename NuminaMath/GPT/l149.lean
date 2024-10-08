import Mathlib

namespace husband_monthly_savings_l149_149233

theorem husband_monthly_savings :
  let wife_weekly_savings := 100
  let weeks_in_month := 4
  let months := 4
  let total_weeks := weeks_in_month * months
  let wife_savings := wife_weekly_savings * total_weeks
  let stock_price := 50
  let number_of_shares := 25
  let invested_half := stock_price * number_of_shares
  let total_savings := invested_half * 2
  let husband_savings := total_savings - wife_savings
  let monthly_husband_savings := husband_savings / months
  monthly_husband_savings = 225 := 
by 
  sorry

end husband_monthly_savings_l149_149233


namespace ratio_of_selling_to_buying_l149_149752

noncomputable def natasha_has_3_times_carla (N C : ℕ) : Prop :=
  N = 3 * C

noncomputable def carla_has_2_times_cosima (C S : ℕ) : Prop :=
  C = 2 * S

noncomputable def total_buying_price (N C S : ℕ) : ℕ :=
  N + C + S

noncomputable def total_selling_price (buying_price profit : ℕ) : ℕ :=
  buying_price + profit

theorem ratio_of_selling_to_buying (N C S buying_price selling_price ratio : ℕ) 
  (h1 : natasha_has_3_times_carla N C)
  (h2 : carla_has_2_times_cosima C S)
  (h3 : N = 60)
  (h4 : buying_price = total_buying_price N C S)
  (h5 : total_selling_price buying_price 36 = selling_price)
  (h6 : 18 * ratio = selling_price * 5): ratio = 7 :=
by
  sorry

end ratio_of_selling_to_buying_l149_149752


namespace linear_eq_find_m_l149_149637

theorem linear_eq_find_m (m : ℤ) (x : ℝ) 
  (h : (m - 5) * x^(|m| - 4) + 5 = 0) 
  (h_linear : |m| - 4 = 1) 
  (h_nonzero : m - 5 ≠ 0) : m = -5 :=
by
  sorry

end linear_eq_find_m_l149_149637


namespace empty_seats_after_second_stop_l149_149029

-- Definitions for the conditions described in the problem
def bus_seats : Nat := 23 * 4
def initial_people : Nat := 16
def first_stop_people_on : Nat := 15
def first_stop_people_off : Nat := 3
def second_stop_people_on : Nat := 17
def second_stop_people_off : Nat := 10

-- The theorem statement proving the number of empty seats
theorem empty_seats_after_second_stop : 
  (bus_seats - (initial_people + first_stop_people_on - first_stop_people_off + second_stop_people_on - second_stop_people_off)) = 57 :=
by
  sorry

end empty_seats_after_second_stop_l149_149029


namespace radius_any_positive_real_l149_149797

theorem radius_any_positive_real (r : ℝ) (h₁ : r > 0) 
    (h₂ : r * (2 * Real.pi * r) = 2 * Real.pi * r^2) : True :=
by
  sorry

end radius_any_positive_real_l149_149797


namespace sequence_third_order_and_nth_term_l149_149871

-- Define the given sequence
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 4
  | 1 => 6
  | 2 => 13
  | 3 => 27
  | 4 => 50
  | 5 => 84
  | _ => sorry -- let’s define the general form for other terms later

-- Define first differences
def first_diff (n : ℕ) : ℤ := a (n + 1) - a n

-- Define second differences
def second_diff (n : ℕ) : ℤ := first_diff (n + 1) - first_diff n

-- Define third differences
def third_diff (n : ℕ) : ℤ := second_diff (n + 1) - second_diff n

-- Define the nth term formula
noncomputable def nth_term (n : ℕ) : ℚ := (1 / 6) * (2 * n^3 + 3 * n^2 - 11 * n + 30)

-- Theorem stating the least possible order is 3 and the nth term formula
theorem sequence_third_order_and_nth_term :
  (∀ n, third_diff n = 2) ∧ (∀ n, a n = nth_term n) :=
by
  sorry

end sequence_third_order_and_nth_term_l149_149871


namespace half_abs_diff_of_squares_l149_149246

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l149_149246


namespace loss_percentage_eq_100_div_9_l149_149068

theorem loss_percentage_eq_100_div_9 :
  ( ∀ C : ℝ,
    (11 * C > 1) ∧ 
    (8.25 * (1 + 0.20) * C = 1) →
    ((C - 1/11) / C * 100) = 100 / 9) 
  :=
by sorry

end loss_percentage_eq_100_div_9_l149_149068


namespace son_l149_149531

variable (M S : ℕ)

theorem son's_age (h1 : M = 4 * S) (h2 : (M - 3) + (S - 3) = 49) : S = 11 :=
by
  sorry

end son_l149_149531


namespace janet_initial_action_figures_l149_149694

theorem janet_initial_action_figures (x : ℕ) :
  (x - 2 + 2 * (x - 2) = 24) -> x = 10 := 
by
  sorry

end janet_initial_action_figures_l149_149694


namespace simplify_expression_l149_149670

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ((x + y) ^ 2 - (x - y) ^ 2) / (4 * x * y) = 1 := 
by sorry

end simplify_expression_l149_149670


namespace exist_common_divisor_l149_149857

theorem exist_common_divisor (a : ℕ → ℕ) (m : ℕ) (h_positive : ∀ i, 1 ≤ i ∧ i ≤ m → 0 < a i)
  (p : ℕ → ℤ) (h_poly : ∀ n : ℕ, ∃ i, 1 ≤ i ∧ i ≤ m ∧ (a i : ℤ) ∣ p n) :
  ∃ j, 1 ≤ j ∧ j ≤ m ∧ ∀ n, (a j : ℤ) ∣ p n :=
by
  sorry

end exist_common_divisor_l149_149857


namespace twice_abs_difference_of_squares_is_4000_l149_149609

theorem twice_abs_difference_of_squares_is_4000 :
  2 * |(105:ℤ)^2 - (95:ℤ)^2| = 4000 :=
by sorry

end twice_abs_difference_of_squares_is_4000_l149_149609


namespace polygon_sides_l149_149671

theorem polygon_sides (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n > 2) : n = 8 :=
sorry

end polygon_sides_l149_149671


namespace odd_power_preserves_order_l149_149917

theorem odd_power_preserves_order {n : ℤ} (h1 : n > 0) (h2 : n % 2 = 1) :
  ∀ (a b : ℝ), a > b → a^n > b^n :=
by
  sorry

end odd_power_preserves_order_l149_149917


namespace total_games_played_l149_149529

theorem total_games_played (won lost total_games : ℕ) 
  (h1 : won = 18)
  (h2 : lost = won + 21)
  (h3 : total_games = won + lost) : total_games = 57 :=
by sorry

end total_games_played_l149_149529


namespace Will_old_cards_l149_149910

theorem Will_old_cards (new_cards pages cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : pages = 6) (h3 : cards_per_page = 3) :
  (pages * cards_per_page) - new_cards = 10 :=
by
  sorry

end Will_old_cards_l149_149910


namespace perimeter_of_first_square_l149_149134

theorem perimeter_of_first_square (p1 p2 p3 : ℕ) (h1 : p1 = 40) (h2 : p2 = 32) (h3 : p3 = 24) :
  p1 = 40 := 
  sorry

end perimeter_of_first_square_l149_149134


namespace christopher_more_money_l149_149305

-- Define the conditions provided in the problem

def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64
def quarter_value : ℝ := 0.25

-- Define the question as a theorem

theorem christopher_more_money : (christopher_quarters - karen_quarters) * quarter_value = 8 :=
by sorry

end christopher_more_money_l149_149305


namespace problem_statement_l149_149012

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a + b + c + 2 = a * b * c) :
  (a+1) * (b+1) * (c+1) ≥ 27 ∧ ((a+1) * (b+1) * (c+1) = 27 → a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end problem_statement_l149_149012


namespace triangle_area_ratio_l149_149416

open Set 

variables {X Y Z W : Type} 
variable [LinearOrder X]

noncomputable def ratio_areas (XW WZ : ℕ) (h : ℕ) : ℚ :=
  (8 * h : ℚ) / (12 * h)

theorem triangle_area_ratio (XW WZ : ℕ) (h : ℕ)
  (hXW : XW = 8)
  (hWZ : WZ = 12) :
  ratio_areas XW WZ h = 2 / 3 :=
by
  rw [hXW, hWZ]
  unfold ratio_areas
  norm_num
  sorry

end triangle_area_ratio_l149_149416


namespace find_x_l149_149166

theorem find_x
  (x : ℤ)
  (h1 : 71 * x % 9 = 8) :
  x = 1 :=
sorry

end find_x_l149_149166


namespace volume_of_released_gas_l149_149644

def mol_co2 : ℝ := 2.4
def molar_volume : ℝ := 22.4

theorem volume_of_released_gas : mol_co2 * molar_volume = 53.76 := by
  sorry -- proof to be filled in

end volume_of_released_gas_l149_149644


namespace vector_calculation_l149_149697

def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (-1, 6)
def v3 : ℝ × ℝ := (2, -1)

theorem vector_calculation :
  (5:ℝ) • v1 - (3:ℝ) • v2 + v3 = (20, -44) :=
by
  sorry

end vector_calculation_l149_149697


namespace estate_value_l149_149109

theorem estate_value (x : ℕ) (E : ℕ) (cook_share : ℕ := 500) 
  (daughter_share : ℕ := 4 * x) (son_share : ℕ := 3 * x) 
  (wife_share : ℕ := 6 * x) (estate_eqn : E = 14 * x) : 
  2 * (daughter_share + son_share) = E ∧ wife_share = 2 * son_share ∧ E = 13 * x + cook_share → 
  E = 7000 :=
by
  sorry

end estate_value_l149_149109


namespace methane_production_proof_l149_149272

noncomputable def methane_production
  (C H : ℕ)
  (methane_formed : ℕ)
  (h_formula : ∀ c h, c = 1 ∧ h = 2)
  (h_initial_conditions : C = 3 ∧ H = 6)
  (h_reaction : ∀ (c h m : ℕ), c = 1 ∧ h = 2 → m = 1) : Prop :=
  methane_formed = 3

theorem methane_production_proof 
  (C H : ℕ)
  (methane_formed : ℕ)
  (h_formula : ∀ c h, c = 1 ∧ h = 2)
  (h_initial_conditions : C = 3 ∧ H = 6)
  (h_reaction : ∀ (c h m : ℕ), c = 1 ∧ h = 2 → m = 1) : methane_production C H methane_formed h_formula h_initial_conditions h_reaction :=
by {
  sorry
}

end methane_production_proof_l149_149272


namespace janet_used_clips_correct_l149_149904

-- Define the initial number of paper clips
def initial_clips : ℕ := 85

-- Define the remaining number of paper clips
def remaining_clips : ℕ := 26

-- Define the number of clips Janet used
def used_clips (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

-- The theorem to state the correctness of the calculation
theorem janet_used_clips_correct : used_clips initial_clips remaining_clips = 59 :=
by
  -- Lean proof goes here
  sorry

end janet_used_clips_correct_l149_149904


namespace largest_multiple_of_15_under_500_l149_149208

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l149_149208


namespace fifth_term_arithmetic_sequence_l149_149700

theorem fifth_term_arithmetic_sequence (a d : ℤ) 
  (h_twentieth : a + 19 * d = 12) 
  (h_twenty_first : a + 20 * d = 16) : 
  a + 4 * d = -48 := 
by sorry

end fifth_term_arithmetic_sequence_l149_149700


namespace regular_decagon_interior_angle_degree_measure_l149_149802

theorem regular_decagon_interior_angle_degree_measure :
  ∀ (n : ℕ), n = 10 → (2 * 180 / n : ℝ) = 144 :=
by
  sorry

end regular_decagon_interior_angle_degree_measure_l149_149802


namespace find_m_if_perpendicular_l149_149716

theorem find_m_if_perpendicular 
  (m : ℝ)
  (h : ∀ m (slope1 : ℝ) (slope2 : ℝ), 
    (slope1 = -m) → 
    (slope2 = (-1) / (3 - 2 * m)) → 
    slope1 * slope2 = -1)
  : m = 3 := 
by
  sorry

end find_m_if_perpendicular_l149_149716


namespace drive_time_from_city_B_to_city_A_l149_149040

theorem drive_time_from_city_B_to_city_A
  (t : ℝ)
  (round_trip_distance : ℝ := 360)
  (saved_time_per_trip : ℝ := 0.5)
  (average_speed : ℝ := 80) :
  (80 * ((3 + t) - 2 * 0.5)) = 360 → t = 2.5 :=
by
  intro h
  sorry

end drive_time_from_city_B_to_city_A_l149_149040


namespace jill_peaches_l149_149569

-- Definitions based on conditions in a
def Steven_has_peaches : ℕ := 19
def Steven_more_than_Jill : ℕ := 13

-- Statement to prove Jill's peaches
theorem jill_peaches : (Steven_has_peaches - Steven_more_than_Jill = 6) :=
by
  sorry

end jill_peaches_l149_149569


namespace f_is_32x5_l149_149582

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

-- State the theorem to be proved
theorem f_is_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  sorry

end f_is_32x5_l149_149582


namespace power_fraction_example_l149_149010

theorem power_fraction_example : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := 
by
  sorry

end power_fraction_example_l149_149010


namespace portion_spent_in_second_store_l149_149252

theorem portion_spent_in_second_store (M : ℕ) (X : ℕ) (H : M = 180)
  (H1 : M - (M / 2 + 14) = 76)
  (H2 : X + 16 = 76)
  (H3 : M = (M / 2 + 14) + (X + 16)) :
  (X : ℚ) / M = 1 / 3 :=
by 
  sorry

end portion_spent_in_second_store_l149_149252


namespace find_line_equation_l149_149245

open Real

noncomputable def line_equation (x y : ℝ) (k : ℝ) : ℝ := k * x - y + 4 - 3 * k

noncomputable def distance_to_line (x1 y1 k : ℝ) : ℝ :=
  abs (k * x1 - y1 + 4 - 3 * k) / sqrt (k^2 + 1)

theorem find_line_equation :
  (∃ k : ℝ, (k = 2 ∨ k = -2 / 3) ∧
    (∀ x y, (x, y) = (3, 4) → (2 * x - y - 2 = 0 ∨ 2 * x + 3 * y - 18 = 0)))
    ∧ (line_equation (-2) 2 2 = line_equation 4 (-2) 2)
    ∧ (line_equation (-2) 2 (-2 / 3) = line_equation 4 (-2) (-2 / 3)) :=
sorry

end find_line_equation_l149_149245


namespace white_ball_probability_l149_149783

theorem white_ball_probability (m : ℕ) 
  (initial_black : ℕ := 6) 
  (initial_white : ℕ := 10) 
  (added_white := 14) 
  (probability := 0.8) :
  (10 + added_white) / (16 + added_white) = probability :=
by
  -- no proof required
  sorry

end white_ball_probability_l149_149783


namespace two_digit_number_l149_149103

theorem two_digit_number (x : ℕ) (h1 : x ≥ 10 ∧ x < 100)
  (h2 : ∃ k : ℤ, 3 * x - 4 = 10 * k)
  (h3 : 60 < 4 * x - 15 ∧ 4 * x - 15 < 100) :
  x = 28 :=
by
  sorry

end two_digit_number_l149_149103


namespace person_picking_number_who_announced_6_is_1_l149_149449

theorem person_picking_number_who_announced_6_is_1
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ)
  (h₁ : a₁₀ + a₂ = 2)
  (h₂ : a₁ + a₃ = 4)
  (h₃ : a₂ + a₄ = 6)
  (h₄ : a₃ + a₅ = 8)
  (h₅ : a₄ + a₆ = 10)
  (h₆ : a₅ + a₇ = 12)
  (h₇ : a₆ + a₈ = 14)
  (h₈ : a₇ + a₉ = 16)
  (h₉ : a₈ + a₁₀ = 18)
  (h₁₀ : a₉ + a₁ = 20) :
  a₆ = 1 :=
by
  sorry

end person_picking_number_who_announced_6_is_1_l149_149449


namespace determine_x_l149_149300

theorem determine_x (x : ℝ) (hx : 0 < x) (h : x * ⌊x⌋ = 72) : x = 9 :=
sorry

end determine_x_l149_149300


namespace ratio_of_houses_second_to_first_day_l149_149368

theorem ratio_of_houses_second_to_first_day 
    (houses_day1 : ℕ)
    (houses_day2 : ℕ)
    (sales_per_house : ℕ)
    (sold_pct_day2 : ℝ) 
    (total_sales_day1 : ℕ)
    (total_sales_day2 : ℝ) :
    houses_day1 = 20 →
    sales_per_house = 2 →
    sold_pct_day2 = 0.8 →
    total_sales_day1 = houses_day1 * sales_per_house →
    total_sales_day2 = sold_pct_day2 * houses_day2 * sales_per_house →
    total_sales_day1 = total_sales_day2 →
    (houses_day2 : ℝ) / houses_day1 = 5 / 4 :=
by
    intro h1 h2 h3 h4 h5 h6
    sorry

end ratio_of_houses_second_to_first_day_l149_149368


namespace total_movies_purchased_l149_149804

theorem total_movies_purchased (x : ℕ) (h1 : 17 * x > 0) (h2 : 4 * x > 0) (h3 : 4 * x - 4 > 0) :
  (17 * x) / (4 * x - 4) = 9 / 2 → 17 * x + 4 * x = 378 :=
by 
  intro hab
  sorry

end total_movies_purchased_l149_149804


namespace chocolate_bars_in_small_box_l149_149597

-- Given conditions
def num_small_boxes : ℕ := 21
def total_chocolate_bars : ℕ := 525

-- Statement to prove
theorem chocolate_bars_in_small_box : total_chocolate_bars / num_small_boxes = 25 := by
  sorry

end chocolate_bars_in_small_box_l149_149597


namespace range_of_a_l149_149584

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a < x ∧ x < a + 1) → (-2 ≤ x ∧ x ≤ 2)) ↔ -2 ≤ a ∧ a ≤ 1 :=
by 
  sorry

end range_of_a_l149_149584


namespace number_of_terms_in_arithmetic_sequence_l149_149401

-- Definitions and conditions
def a : ℤ := -58  -- First term
def d : ℤ := 7   -- Common difference
def l : ℤ := 78  -- Last term

-- Statement of the problem
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 20 := 
by
  sorry

end number_of_terms_in_arithmetic_sequence_l149_149401


namespace seconds_in_part_of_day_l149_149362

theorem seconds_in_part_of_day : (1 / 4) * (1 / 6) * (1 / 8) * 24 * 60 * 60 = 450 := by
  sorry

end seconds_in_part_of_day_l149_149362


namespace optimal_floor_optimal_floor_achieved_at_three_l149_149539

theorem optimal_floor : ∀ (n : ℕ), n > 0 → (n + 9 / n : ℝ) ≥ 6 := sorry

theorem optimal_floor_achieved_at_three : ∃ n : ℕ, (n > 0 ∧ (n + 9 / n : ℝ) = 6) := sorry

end optimal_floor_optimal_floor_achieved_at_three_l149_149539


namespace total_cards_square_l149_149683

theorem total_cards_square (s : ℕ) (h_perim : 4 * s - 4 = 240) : s * s = 3721 := by
  sorry

end total_cards_square_l149_149683


namespace solve_for_x_l149_149875

theorem solve_for_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 := 
sorry

end solve_for_x_l149_149875


namespace jim_gas_tank_capacity_l149_149886

/-- Jim has 2/3 of a tank left after a round-trip of 20 miles where he gets 5 miles per gallon.
    Prove that the capacity of Jim's gas tank is 12 gallons. --/
theorem jim_gas_tank_capacity
    (remaining_fraction : ℚ)
    (round_trip_distance : ℚ)
    (fuel_efficiency : ℚ)
    (used_fraction : ℚ)
    (used_gallons : ℚ)
    (total_capacity : ℚ)
    (h1 : remaining_fraction = 2/3)
    (h2 : round_trip_distance = 20)
    (h3 : fuel_efficiency = 5)
    (h4 : used_fraction = 1 - remaining_fraction)
    (h5 : used_gallons = round_trip_distance / fuel_efficiency)
    (h6 : used_gallons = used_fraction * total_capacity) :
  total_capacity = 12 :=
sorry

end jim_gas_tank_capacity_l149_149886


namespace triangle_area_of_parabola_hyperbola_l149_149284

-- Definitions for parabola and hyperbola
def parabola_directrix (a : ℕ) (x y : ℝ) : Prop := x^2 = 16 * y
def hyperbola_asymptotes (a b : ℕ) (x y : ℝ) : Prop := x^2 / (a^2) - y^2 / (b^2) = 1

-- Theorem stating the area of the triangle formed by the intersections of the asymptotes with the directrix
theorem triangle_area_of_parabola_hyperbola (a b : ℕ) (h : a = 1) (h' : b = 1) : 
  ∃ (area : ℝ), area = 16 :=
sorry

end triangle_area_of_parabola_hyperbola_l149_149284


namespace original_length_wire_l149_149617

-- Define the conditions.
def length_cut_off_parts : ℕ := 10
def remaining_length_relation (L_remaining : ℕ) : Prop :=
  L_remaining = 4 * (2 * length_cut_off_parts) + 10

-- Define the theorem to prove the original length of the wire.
theorem original_length_wire (L_remaining : ℕ) (H : remaining_length_relation L_remaining) : 
  L_remaining + 2 * length_cut_off_parts = 110 :=
by 
  -- Use the given conditions
  unfold remaining_length_relation at H
  -- The proof would show that the equation holds true.
  sorry

end original_length_wire_l149_149617


namespace parabola_b_value_l149_149730

variable (a b c p : ℝ)
variable (h1 : p ≠ 0)
variable (h2 : ∀ x, y = a*x^2 + b*x + c)
variable (h3 : vertex' y = (p, -p))
variable (h4 : y-intercept' y = (0, p))

theorem parabola_b_value : b = -4 :=
sorry

end parabola_b_value_l149_149730


namespace larger_number_is_25_l149_149942

-- Let x and y be real numbers, with x being the larger number
variables (x y : ℝ)

-- The sum of the two numbers is 45
axiom sum_eq_45 : x + y = 45

-- The difference of the two numbers is 5
axiom diff_eq_5 : x - y = 5

-- We need to prove that the larger number x is 25
theorem larger_number_is_25 : x = 25 :=
by
  sorry

end larger_number_is_25_l149_149942


namespace abc_le_one_eighth_l149_149792

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : 
  a * b * c ≤ 1 / 8 := 
by
  sorry

end abc_le_one_eighth_l149_149792


namespace ratio_sum_eq_seven_eight_l149_149052

theorem ratio_sum_eq_seven_eight 
  (a b c x y z : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7/8 :=
by
  sorry

end ratio_sum_eq_seven_eight_l149_149052


namespace number_of_tangent_lines_l149_149733

def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 + a

def on_line (a x y : ℝ) : Prop := 3 * x + y = a + 1

theorem number_of_tangent_lines (a m : ℝ) (h1 : on_line a m (a + 1 - 3 * m)) :
  ∃ n : ℤ, n = 1 ∨ n = 2 :=
sorry

end number_of_tangent_lines_l149_149733


namespace greatest_x_plus_z_l149_149501

theorem greatest_x_plus_z (x y z c d : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 1 ≤ z ∧ z ≤ 9)
  (h4 : 700 - c = 700)
  (h5 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 693)
  (h6 : x > z) :
  d = 11 :=
by
  sorry

end greatest_x_plus_z_l149_149501


namespace draw_odds_l149_149889

theorem draw_odds (x : ℝ) (bet_Zubilo bet_Shaiba bet_Draw payout : ℝ) (h1 : bet_Zubilo = 3 * x) (h2 : bet_Shaiba = 2 * x) (h3 : payout = 6 * x) : 
  bet_Draw * 6 = payout :=
by
  sorry

end draw_odds_l149_149889


namespace geometric_sequence_terms_sum_l149_149905

theorem geometric_sequence_terms_sum :
  ∀ (a_n : ℕ → ℝ) (q : ℝ),
    (∀ n, a_n (n + 1) = a_n n * q) ∧ a_n 1 = 3 ∧
    (a_n 1 + a_n 2 + a_n 3) = 21 →
    (a_n (1 + 2) + a_n (1 + 3) + a_n (1 + 4)) = 84 :=
by
  intros a_n q h
  sorry

end geometric_sequence_terms_sum_l149_149905


namespace BoxC_in_BoxA_l149_149212

-- Define the relationship between the boxes
def BoxA_has_BoxB (A B : ℕ) : Prop := A = 4 * B
def BoxB_has_BoxC (B C : ℕ) : Prop := B = 6 * C

-- Define the proof problem
theorem BoxC_in_BoxA {A B C : ℕ} (h1 : BoxA_has_BoxB A B) (h2 : BoxB_has_BoxC B C) : A = 24 * C :=
by
  sorry

end BoxC_in_BoxA_l149_149212


namespace max_consecutive_sum_le_1000_l149_149281

theorem max_consecutive_sum_le_1000 : 
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → (m * (m + 1)) / 2 < 1000) ∧ ¬∃ n' : ℕ, n < n' ∧ (n' * (n' + 1)) / 2 < 1000 :=
sorry

end max_consecutive_sum_le_1000_l149_149281


namespace compare_magnitudes_l149_149813

theorem compare_magnitudes : -0.5 > -0.75 :=
by
  have h1 : |(-0.5: ℝ)| = 0.5 := by norm_num
  have h2 : |(-0.75: ℝ)| = 0.75 := by norm_num
  have h3 : (0.5: ℝ) < 0.75 := by norm_num
  sorry

end compare_magnitudes_l149_149813


namespace hyperbola_eccentricity_l149_149432

-- Definitions based on conditions
def hyperbola (x y : ℝ) (a : ℝ) := x^2 / a^2 - y^2 / 5 = 1

-- Main theorem
theorem hyperbola_eccentricity (a : ℝ) (c : ℝ) (h_focus : c = 3) (h_hyperbola : hyperbola 0 0 a) (focus_condition : c^2 = a^2 + 5) :
  c / a = 3 / 2 :=
by
  sorry

end hyperbola_eccentricity_l149_149432


namespace average_weight_of_11_children_l149_149939

theorem average_weight_of_11_children (b: ℕ) (g: ℕ) (avg_b: ℕ) (avg_g: ℕ) (hb: b = 8) (hg: g = 3) (havg_b: avg_b = 155) (havg_g: avg_g = 115) : 
  (b * avg_b + g * avg_g) / (b + g) = 144 :=
by {
  sorry
}

end average_weight_of_11_children_l149_149939


namespace total_games_played_l149_149724

-- Definition of the number of teams
def num_teams : ℕ := 20

-- Definition of the number of games each pair plays
def games_per_pair : ℕ := 10

-- Theorem stating the total number of games played
theorem total_games_played : (num_teams * (num_teams - 1) / 2) * games_per_pair = 1900 :=
by sorry

end total_games_played_l149_149724


namespace equivalence_statement_l149_149588

open Complex

noncomputable def distinct_complex (a b c d : ℂ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem equivalence_statement (a b c d : ℂ) (h : distinct_complex a b c d) :
  (∀ (z : ℂ), (abs (z - a) + abs (z - b) ≥ abs (z - c) + abs (z - d)))
  ↔ (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ c = t * a + (1 - t) * b ∧ d = (1 - t) * a + t * b) :=
sorry

end equivalence_statement_l149_149588


namespace logical_equivalence_l149_149893

variables (P Q : Prop)

theorem logical_equivalence :
  (¬P → ¬Q) ↔ (Q → P) :=
sorry

end logical_equivalence_l149_149893


namespace find_initial_pens_l149_149532

-- Conditions in the form of definitions
def initial_pens (P : ℕ) : ℕ := P
def after_mike (P : ℕ) : ℕ := P + 20
def after_cindy (P : ℕ) : ℕ := 2 * after_mike P
def after_sharon (P : ℕ) : ℕ := after_cindy P - 19

-- The final condition
def final_pens (P : ℕ) : ℕ := 31

-- The goal is to prove that the initial number of pens is 5
theorem find_initial_pens : 
  ∃ (P : ℕ), after_sharon P = final_pens P → P = 5 :=
by 
  sorry

end find_initial_pens_l149_149532


namespace sum_odd_even_50_l149_149338

def sum_first_n_odd (n : ℕ) : ℕ := n * n

def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

theorem sum_odd_even_50 : 
  sum_first_n_odd 50 + sum_first_n_even 50 = 5050 := by
  sorry

end sum_odd_even_50_l149_149338


namespace largest_non_sum_l149_149222

theorem largest_non_sum (n : ℕ) : 
  ¬ (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b ∣ 2 ∧ n = 36 * a + b) ↔ n = 104 :=
by
  sorry

end largest_non_sum_l149_149222


namespace union_of_A_and_B_l149_149467

open Set

theorem union_of_A_and_B : 
  let A := {x : ℝ | x + 2 > 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = Real.cos x}
  A ∪ B = {z : ℝ | z > -2} := 
by
  intros
  sorry

end union_of_A_and_B_l149_149467


namespace cost_of_a_pen_l149_149877

theorem cost_of_a_pen:
  ∃ x y : ℕ, 5 * x + 4 * y = 345 ∧ 3 * x + 6 * y = 285 ∧ x = 52 :=
by
  sorry

end cost_of_a_pen_l149_149877


namespace perimeter_of_triangle_l149_149764

theorem perimeter_of_triangle (r a : ℝ) (p : ℝ) (h1 : r = 3.5) (h2 : a = 56) :
  p = 32 :=
by
  sorry

end perimeter_of_triangle_l149_149764


namespace seq_a_n_a_4_l149_149241

theorem seq_a_n_a_4 :
  ∃ a : ℕ → ℕ, (a 1 = 1) ∧ (∀ n : ℕ, a (n+1) = 2 * a n) ∧ (a 4 = 8) :=
sorry

end seq_a_n_a_4_l149_149241


namespace max_liters_l149_149201

-- Declare the problem conditions as constants
def initialHeat : ℕ := 480 -- kJ for the first 5 min
def reductionRate : ℚ := 0.25 -- 25%
def initialTemp : ℚ := 20 -- °C
def boilingTemp : ℚ := 100 -- °C
def specificHeatCapacity : ℚ := 4.2 -- kJ/kg°C

-- Define the heat required to raise m kg of water from initialTemp to boilingTemp
def heatRequired (m : ℚ) : ℚ := m * specificHeatCapacity * (boilingTemp - initialTemp)

-- Define the total available heat from the fuel
noncomputable def totalAvailableHeat : ℚ :=
  initialHeat / (1 - (1 - reductionRate))

-- The proof problem statement
theorem max_liters :
  ∃ (m : ℕ), m ≤ 5 ∧ heatRequired m ≤ totalAvailableHeat :=
sorry

end max_liters_l149_149201


namespace multiple_for_snack_cost_l149_149080

-- Define the conditions
def kyle_time_to_work : ℕ := 2 -- Kyle bikes for 2 hours to work every day.
def cost_of_snacks (total_cost packs : ℕ) : ℕ := total_cost / packs -- Ryan will pay $2000 to buy 50 packs of snacks.

-- Ryan pays $2000 for 50 packs of snacks.
def cost_per_pack := cost_of_snacks 2000 50

-- The time for a round trip (to work and back)
def round_trip_time (h : ℕ) : ℕ := 2 * h

-- The multiple of the time taken to travel to work and back that equals the cost of a pack of snacks
def multiple (cost time : ℕ) : ℕ := cost / time

-- Statement we need to prove
theorem multiple_for_snack_cost : 
  multiple cost_per_pack (round_trip_time kyle_time_to_work) = 10 :=
  by
  sorry

end multiple_for_snack_cost_l149_149080


namespace cubic_root_relationship_l149_149468

theorem cubic_root_relationship 
  (r : ℝ) (h : r^3 - r + 3 = 0) : 
  (r^2)^3 - 2 * (r^2)^2 + (r^2) - 9 = 0 := 
by 
  sorry

end cubic_root_relationship_l149_149468


namespace number_of_intersections_l149_149848

/-- 
  Define the two curves as provided in the problem:
  curve1 is defined by the equation 3x² + 2y² = 6,
  curve2 is defined by the equation x² - 2y² = 1.
  We aim to prove that there are exactly 4 distinct intersection points.
--/
def curve1 (x y : ℝ) : Prop := 3 * x^2 + 2 * y^2 = 6

def curve2 (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1

theorem number_of_intersections : ∃ (points : Finset (ℝ × ℝ)), (∀ p ∈ points, curve1 p.1 p.2 ∧ curve2 p.1 p.2) ∧ points.card = 4 :=
sorry

end number_of_intersections_l149_149848


namespace average_score_is_8_9_l149_149675

-- Define the scores and their frequencies
def scores : List ℝ := [7.5, 8.5, 9, 10]
def frequencies : List ℕ := [2, 2, 3, 3]

-- Express the condition that the total number of shots is 10
def total_shots : ℕ := frequencies.sum

-- Calculate the weighted sum of the scores
def weighted_sum (scores : List ℝ) (frequencies : List ℕ) : ℝ :=
  (List.zip scores frequencies).foldl (λ acc (sc, freq) => acc + (sc * freq)) 0

-- Prove that the average score is 8.9
theorem average_score_is_8_9 :
  total_shots = 10 →
  weighted_sum scores frequencies / total_shots = 8.9 :=
by
  intros h_total_shots
  sorry

end average_score_is_8_9_l149_149675


namespace simplify_power_of_product_l149_149169

theorem simplify_power_of_product (x : ℝ) : (5 * x^2)^4 = 625 * x^8 :=
by
  sorry

end simplify_power_of_product_l149_149169


namespace son_l149_149849

def woman's_age (W S : ℕ) : Prop := W = 2 * S + 3
def sum_of_ages (W S : ℕ) : Prop := W + S = 84

theorem son's_age_is_27 (W S : ℕ) (h1: woman's_age W S) (h2: sum_of_ages W S) : S = 27 :=
by
  sorry

end son_l149_149849


namespace circle_symmetric_line_l149_149056

theorem circle_symmetric_line (m : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) → (3*x + y + m = 0)) →
  m = 1 :=
by
  intro h
  sorry

end circle_symmetric_line_l149_149056


namespace V_product_is_V_form_l149_149725

noncomputable def V (a b c : ℝ) : ℝ := a^3 + b^3 + c^3 - 3 * a * b * c

theorem V_product_is_V_form (a b c x y z : ℝ) :
  V a b c * V x y z = V (a * x + b * y + c * z) (b * x + c * y + a * z) (c * x + a * y + b * z) := by
  sorry

end V_product_is_V_form_l149_149725


namespace alex_buys_17_1_pounds_of_corn_l149_149706

-- Definitions based on conditions
def corn_cost_per_pound : ℝ := 1.20
def bean_cost_per_pound : ℝ := 0.50
def total_pounds : ℝ := 30
def total_cost : ℝ := 27.00

-- Define the variables
variables (c b : ℝ)

-- Theorem statement to prove the number of pounds of corn Alex buys
theorem alex_buys_17_1_pounds_of_corn (h1 : b + c = total_pounds) (h2 : bean_cost_per_pound * b + corn_cost_per_pound * c = total_cost) :
  c = 17.1 :=
sorry

end alex_buys_17_1_pounds_of_corn_l149_149706


namespace exponent_division_simplification_l149_149447

theorem exponent_division_simplification :
  ((18^18 / 18^17)^2 * 9^2) / 3^4 = 324 :=
by
  sorry

end exponent_division_simplification_l149_149447


namespace find_y_given_conditions_l149_149433

theorem find_y_given_conditions (x y : ℝ) (h1 : x^(3 * y) = 27) (h2 : x = 3) : y = 1 := 
by
  sorry

end find_y_given_conditions_l149_149433


namespace carrie_remaining_money_l149_149261

def initial_money : ℝ := 200
def sweater_cost : ℝ := 36
def tshirt_cost : ℝ := 12
def tshirt_discount : ℝ := 0.10
def shoes_cost : ℝ := 45
def jeans_cost : ℝ := 52
def scarf_cost : ℝ := 18
def sales_tax_rate : ℝ := 0.05

-- Calculate tshirt price after discount
def tshirt_final_price : ℝ := tshirt_cost * (1 - tshirt_discount)

-- Sum all the item costs before tax
def total_cost_before_tax : ℝ := sweater_cost + tshirt_final_price + shoes_cost + jeans_cost + scarf_cost

-- Calculate the total sales tax
def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

-- Calculate total cost after tax
def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

-- Calculate the remaining money
def remaining_money (initial : ℝ) (total : ℝ) : ℝ := initial - total

theorem carrie_remaining_money
  (initial_money : ℝ)
  (sweater_cost : ℝ)
  (tshirt_cost : ℝ)
  (tshirt_discount : ℝ)
  (shoes_cost : ℝ)
  (jeans_cost : ℝ)
  (scarf_cost : ℝ)
  (sales_tax_rate : ℝ)
  (h₁ : initial_money = 200)
  (h₂ : sweater_cost = 36)
  (h₃ : tshirt_cost = 12)
  (h₄ : tshirt_discount = 0.10)
  (h₅ : shoes_cost = 45)
  (h₆ : jeans_cost = 52)
  (h₇ : scarf_cost = 18)
  (h₈ : sales_tax_rate = 0.05) :
  remaining_money initial_money (total_cost_after_tax) = 30.11 := 
by 
  simp only [remaining_money, total_cost_after_tax, total_cost_before_tax, tshirt_final_price, sales_tax];
  sorry

end carrie_remaining_money_l149_149261


namespace negation_of_proposition_l149_149235

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x < 0 → x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x < 0 ∧ x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_of_proposition_l149_149235


namespace sum_and_product_of_roots_l149_149551

theorem sum_and_product_of_roots (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b = 0 → x = -2 ∨ x = 3) → a + b = -7 :=
by
  sorry

end sum_and_product_of_roots_l149_149551


namespace correct_statement_l149_149673

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + (Real.pi / 2))
noncomputable def g (x : ℝ) : ℝ := Real.cos (x + (3 * Real.pi / 2))

theorem correct_statement (x : ℝ) : f (x - (Real.pi / 2)) = g x :=
by sorry

end correct_statement_l149_149673


namespace find_a_l149_149081

variable (y : ℝ) (a : ℝ)

theorem find_a (hy : y > 0) (h_expr : (a * y / 20) + (3 * y / 10) = 0.7 * y) : a = 8 :=
by
  sorry

end find_a_l149_149081


namespace expression_values_l149_149435

variable {a b c : ℚ}

theorem expression_values (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = 2) ∨ 
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = -2) := 
sorry

end expression_values_l149_149435


namespace quadratic_one_solution_m_value_l149_149182

theorem quadratic_one_solution_m_value (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) → (b^2 - 4 * a * m = 0) → m = 3 :=
by
  sorry

end quadratic_one_solution_m_value_l149_149182


namespace find_m_range_a_l149_149698

noncomputable def f (x m : ℝ) : ℝ :=
  m - |x - 3|

theorem find_m (m : ℝ) (h : ∀ x, 2 < f x m ↔ 2 < x ∧ x < 4) : m = 3 :=
  sorry

theorem range_a (a : ℝ) (h : ∀ x, |x - a| ≥ f x 3) : a ≤ 0 ∨ 6 ≤ a :=
  sorry

end find_m_range_a_l149_149698


namespace purchase_price_of_furniture_l149_149464

theorem purchase_price_of_furniture (marked_price discount_rate profit_rate : ℝ) 
(h_marked_price : marked_price = 132) 
(h_discount_rate : discount_rate = 0.1)
(h_profit_rate : profit_rate = 0.1)
: ∃ a : ℝ, (marked_price * (1 - discount_rate) - a = profit_rate * a) ∧ a = 108 := by
  sorry

end purchase_price_of_furniture_l149_149464


namespace rate_calculation_l149_149138

def principal : ℝ := 910
def simple_interest : ℝ := 260
def time : ℝ := 4
def rate : ℝ := 7.14

theorem rate_calculation :
  (simple_interest / (principal * time)) * 100 = rate :=
by
  sorry

end rate_calculation_l149_149138


namespace john_task_completion_time_l149_149759

/-- John can complete a task alone in 18 days given the conditions. -/
theorem john_task_completion_time :
  ∀ (John Jane taskDays : ℝ), 
    Jane = 12 → 
    taskDays = 10.8 → 
    (10.8 - 6) * (1 / 12) + 10.8 * (1 / John) = 1 → 
    John = 18 :=
by
  intros John Jane taskDays hJane hTaskDays hWorkDone
  sorry

end john_task_completion_time_l149_149759


namespace range_of_a_l149_149649

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) := ∀ x₁ x₂ : ℝ, x₁ < x₂ → -(5 - 2 * a)^x₁ > -(5 - 2 * a)^x₂

theorem range_of_a (a : ℝ) : (p a ∨ q a) → ¬ (p a ∧ q a) → a ≤ -2 := by 
  sorry

end range_of_a_l149_149649


namespace notification_probability_l149_149812

theorem notification_probability
  (num_students : ℕ)
  (num_notified_Li : ℕ)
  (num_notified_Zhang : ℕ)
  (prob_Li : ℚ)
  (prob_Zhang : ℚ)
  (h1 : num_students = 10)
  (h2 : num_notified_Li = 4)
  (h3 : num_notified_Zhang = 4)
  (h4 : prob_Li = (4 : ℚ) / 10)
  (h5 : prob_Zhang = (4 : ℚ) / 10) :
  prob_Li + prob_Zhang - prob_Li * prob_Zhang = (16 : ℚ) / 25 := 
by 
  sorry

end notification_probability_l149_149812


namespace complex_number_equality_l149_149051

theorem complex_number_equality (i : ℂ) (h : i^2 = -1) : 1 + i + i^2 = i :=
by
  sorry

end complex_number_equality_l149_149051


namespace polynomial_evaluation_l149_149699

theorem polynomial_evaluation :
  101^4 - 4 * 101^3 + 6 * 101^2 - 4 * 101 + 1 = 100000000 := sorry

end polynomial_evaluation_l149_149699


namespace side_lengths_le_sqrt3_probability_is_1_over_3_l149_149189

open Real

noncomputable def probability_side_lengths_le_sqrt3 : ℝ :=
  let total_area : ℝ := 2 * π^2
  let satisfactory_area : ℝ := 2 * π^2 / 3
  satisfactory_area / total_area

theorem side_lengths_le_sqrt3_probability_is_1_over_3 :
  probability_side_lengths_le_sqrt3 = 1 / 3 :=
by
  sorry

end side_lengths_le_sqrt3_probability_is_1_over_3_l149_149189


namespace number_of_functions_l149_149844

open Nat

theorem number_of_functions (f : Fin 15 → Fin 15)
  (h : ∀ x, (f (f x) - 2 * f x + x : Int) % 15 = 0) :
  ∃! n : Nat, n = 375 := sorry

end number_of_functions_l149_149844


namespace f_at_neg_one_l149_149062

def f (x : ℝ) : ℝ := x^2 - 1

theorem f_at_neg_one : f (-1) = 0 := by
  sorry

end f_at_neg_one_l149_149062


namespace individual_weights_l149_149154

theorem individual_weights (A P : ℕ) 
    (h1 : 12 * A + 14 * P = 692)
    (h2 : P = A - 10) : 
    A = 32 ∧ P = 22 :=
by
  sorry

end individual_weights_l149_149154


namespace find_common_difference_l149_149323

-- Define the arithmetic series sum formula
def arithmetic_series_sum (a₁ : ℕ) (d : ℚ) (n : ℕ) :=
  (n / 2) * (2 * a₁ + (n - 1) * d)

-- Define the first day's production, total days, and total fabric
def first_day := 5
def total_days := 30
def total_fabric := 390

-- The proof statement
theorem find_common_difference : 
  ∃ d : ℚ, arithmetic_series_sum first_day d total_days = total_fabric ∧ d = 16 / 29 :=
by
  sorry

end find_common_difference_l149_149323


namespace find_integer_l149_149984

-- Definition of the given conditions
def conditions (x : ℤ) (r : ℤ) : Prop :=
  (0 ≤ r ∧ r < 7) ∧ ((x - 77) * 8 = 259 + r)

-- Statement of the theorem to be proved
theorem find_integer : ∃ x : ℤ, ∃ r : ℤ, conditions x r ∧ (x = 110) :=
by
  sorry

end find_integer_l149_149984


namespace fraction_power_l149_149901

theorem fraction_power : (2 / 5 : ℚ) ^ 3 = 8 / 125 := by
  sorry

end fraction_power_l149_149901


namespace probability_calculations_l149_149791

-- Define the number of students
def total_students : ℕ := 2006

-- Number of students eliminated in the first step
def eliminated_students : ℕ := 6

-- Number of students remaining after elimination
def remaining_students : ℕ := total_students - eliminated_students

-- Number of students to be selected in the second step
def selected_students : ℕ := 50

-- Calculate the probability of a specific student being eliminated
def elimination_probability := (6 : ℚ) / total_students

-- Calculate the probability of a specific student being selected from the remaining students
def selection_probability := (50 : ℚ) / remaining_students

-- The theorem to prove our equivalent proof problem
theorem probability_calculations :
  elimination_probability = (3 : ℚ) / 1003 ∧
  selection_probability = (25 : ℚ) / 1003 :=
by
  sorry

end probability_calculations_l149_149791


namespace avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l149_149119

-- Average fuel consumption per kilometer
noncomputable def avgFuelConsumption (initial_fuel: ℝ) (final_fuel: ℝ) (distance: ℝ) : ℝ :=
  (initial_fuel - final_fuel) / distance

-- Relationship between remaining fuel Q and distance x
noncomputable def remainingFuel (initial_fuel: ℝ) (consumption_rate: ℝ) (distance: ℝ) : ℝ :=
  initial_fuel - consumption_rate * distance

-- Check if the car can return home without refueling
noncomputable def canReturnHome (initial_fuel: ℝ) (consumption_rate: ℝ) (round_trip_distance: ℝ) (alarm_fuel_level: ℝ) : Bool :=
  initial_fuel - consumption_rate * round_trip_distance ≥ alarm_fuel_level

-- Theorem statements to prove
theorem avg_fuel_consumption_correct :
  avgFuelConsumption 45 27 180 = 0.1 :=
sorry

theorem remaining_fuel_correct :
  ∀ x, remainingFuel 45 0.1 x = 45 - 0.1 * x :=
sorry

theorem cannot_return_home_without_refueling :
  ¬canReturnHome 45 0.1 (220 * 2) 3 :=
sorry

end avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l149_149119


namespace median_length_of_right_triangle_l149_149325

theorem median_length_of_right_triangle (DE EF : ℝ) (hDE : DE = 5) (hEF : EF = 12) :
  let DF := Real.sqrt (DE^2 + EF^2)
  let N := (EF / 2)
  let DN := DF / 2
  DN = 6.5 :=
by
  sorry

end median_length_of_right_triangle_l149_149325


namespace converse_proposition_l149_149265

-- Define a proposition for vertical angles
def vertical_angles (α β : ℕ) : Prop := α = β

-- Define the converse of the vertical angle proposition
def converse_vertical_angles (α β : ℕ) : Prop := β = α

-- Prove that the converse of "Vertical angles are equal" is 
-- "Angles that are equal are vertical angles"
theorem converse_proposition (α β : ℕ) : vertical_angles α β ↔ converse_vertical_angles α β :=
by
  sorry

end converse_proposition_l149_149265


namespace digit_for_divisibility_by_5_l149_149101

theorem digit_for_divisibility_by_5 (B : ℕ) (B_digit_condition : B < 10) :
  (∃ k : ℕ, 6470 + B = 5 * k) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end digit_for_divisibility_by_5_l149_149101


namespace natalie_bushes_needed_l149_149909

theorem natalie_bushes_needed (b c p : ℕ) 
  (h1 : ∀ b, b * 10 = c) 
  (h2 : ∀ c, c * 2 = p)
  (target_p : p = 36) :
  ∃ b, b * 10 ≥ 72 :=
by
  sorry

end natalie_bushes_needed_l149_149909


namespace find_speed_l149_149504

theorem find_speed (v d : ℝ) (h1 : d > 0) (h2 : 1.10 * v > 0) (h3 : 84 = 2 * d / (d / v + d / (1.10 * v))) : v = 80.18 := 
sorry

end find_speed_l149_149504


namespace number_of_students_in_class_l149_149555

theorem number_of_students_in_class (S : ℕ) 
  (h1 : ∀ n : ℕ, 4 * n ≠ 0 → S % 4 = 0) -- S is divisible by 4
  (h2 : ∀ G : ℕ, 3 * G ≠ 0 → (S * 3) % 4 = G) -- Number of students who went to the playground (3/4 * S) is integer
  (h3 : ∀ B : ℕ, G - B ≠ 0 → (G * 2) / 3 = 10) -- Number of girls on the playground
  : S = 20 := sorry

end number_of_students_in_class_l149_149555


namespace michael_card_count_l149_149228

variable (Lloyd Mark Michael : ℕ)
variable (L : ℕ)

-- Conditions from the problem
axiom condition1 : Mark = 3 * Lloyd
axiom condition2 : Mark + 10 = Michael
axiom condition3 : Lloyd + Mark + (Michael + 80) = 300

-- The correct answer we want to prove
theorem michael_card_count : Michael = 100 :=
by
  -- Proof will be here.
  sorry

end michael_card_count_l149_149228


namespace positive_integer_pairs_count_l149_149963

theorem positive_integer_pairs_count :
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs → a > 0 ∧ b > 0 ∧ (1 : ℚ) / a - (1 : ℚ) / b = (1 : ℚ) / 2021) ∧ 
    pairs.length = 4 :=
by sorry

end positive_integer_pairs_count_l149_149963


namespace hearty_beads_count_l149_149020

theorem hearty_beads_count :
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  total_beads = 320 :=
by
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  show total_beads = 320
  sorry

end hearty_beads_count_l149_149020


namespace fewer_bronze_stickers_l149_149199

theorem fewer_bronze_stickers
  (gold_stickers : ℕ)
  (silver_stickers : ℕ)
  (each_student_stickers : ℕ)
  (students : ℕ)
  (total_stickers_given : ℕ)
  (bronze_stickers : ℕ)
  (total_gold_and_silver_stickers : ℕ)
  (gold_stickers_eq : gold_stickers = 50)
  (silver_stickers_eq : silver_stickers = 2 * gold_stickers)
  (each_student_stickers_eq : each_student_stickers = 46)
  (students_eq : students = 5)
  (total_stickers_given_eq : total_stickers_given = students * each_student_stickers)
  (total_gold_and_silver_stickers_eq : total_gold_and_silver_stickers = gold_stickers + silver_stickers)
  (bronze_stickers_eq : bronze_stickers = total_stickers_given - total_gold_and_silver_stickers) :
  silver_stickers - bronze_stickers = 20 :=
by
  sorry

end fewer_bronze_stickers_l149_149199


namespace bridge_length_l149_149375

theorem bridge_length
  (train_length : ℝ) (train_speed : ℝ) (time_taken : ℝ)
  (h_train_length : train_length = 280)
  (h_train_speed : train_speed = 18)
  (h_time_taken : time_taken = 20) : ∃ L : ℝ, L = 80 :=
by
  let distance_covered := train_speed * time_taken
  have h_distance_covered : distance_covered = 360 := by sorry
  let bridge_length := distance_covered - train_length
  have h_bridge_length : bridge_length = 80 := by sorry
  existsi bridge_length
  exact h_bridge_length

end bridge_length_l149_149375


namespace boat_distance_l149_149186

theorem boat_distance (v_b : ℝ) (v_s : ℝ) (t_downstream : ℝ) (t_upstream : ℝ) (d : ℝ) :
  v_b = 7 ∧ t_downstream = 2 ∧ t_upstream = 5 ∧ d = (v_b + v_s) * t_downstream ∧ d = (v_b - v_s) * t_upstream → d = 20 :=
by {
  sorry
}

end boat_distance_l149_149186


namespace find_x_plus_y_l149_149067

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c p : V) (x y : ℝ)

-- Conditions: Definitions as the given problem requires
-- Basis definitions
def basis1 := [a, b, c]
def basis2 := [a + b, a - b, c]

-- Conditions on p
def condition1 : p = 3 • a + b + c := sorry
def condition2 : p = x • (a + b) + y • (a - b) + c := sorry

-- The proof statement
theorem find_x_plus_y (h1 : p = 3 • a + b + c) (h2 : p = x • (a + b) + y • (a - b) + c) :
  x + y = 3 :=
sorry

end find_x_plus_y_l149_149067


namespace Daniella_savings_l149_149118

def initial_savings_of_Daniella (D : ℤ) := D
def initial_savings_of_Ariella (D : ℤ) := D + 200
def interest_rate : ℚ := 0.10
def time_years : ℚ := 2
def total_amount_after_two_years (initial_amount : ℤ) : ℚ :=
  initial_amount + initial_amount * interest_rate * time_years
def final_amount_of_Ariella : ℚ := 720

theorem Daniella_savings :
  ∃ D : ℤ, total_amount_after_two_years (initial_savings_of_Ariella D) = final_amount_of_Ariella ∧ initial_savings_of_Daniella D = 400 :=
by
  sorry

end Daniella_savings_l149_149118


namespace find_all_solutions_l149_149988

def is_solution (f : ℕ → ℝ) : Prop :=
  (∀ n ≥ 1, f (n + 1) ≥ f n) ∧
  (∀ m n, Nat.gcd m n = 1 → f (m * n) = f m * f n)

theorem find_all_solutions :
  ∀ f : ℕ → ℝ, is_solution f →
    (∀ n, f n = 0) ∨ (∃ a ≥ 0, ∀ n, f n = n ^ a) :=
sorry

end find_all_solutions_l149_149988


namespace king_chessboard_strategy_king_chessboard_strategy_odd_l149_149398

theorem king_chessboard_strategy (m n : ℕ) : 
  (m * n) % 2 = 0 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) := 
sorry

theorem king_chessboard_strategy_odd (m n : ℕ) : 
  (m * n) % 2 = 1 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) :=
sorry

end king_chessboard_strategy_king_chessboard_strategy_odd_l149_149398


namespace boys_girls_students_l149_149846

theorem boys_girls_students (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
  (h1 : total_students = 100)
  (h2 : ratio_boys = 3)
  (h3 : ratio_girls = 2) :
  3 * (total_students / (ratio_boys + ratio_girls)) - 2 * (total_students / (ratio_boys + ratio_girls)) = 20 :=
by
  sorry

end boys_girls_students_l149_149846


namespace problem_1_problem_2_l149_149414

namespace ProofProblems

def U : Set ℝ := {y | true}

def E : Set ℝ := {y | y > 2}

def F : Set ℝ := {y | ∃ (x : ℝ), (-1 < x ∧ x < 2 ∧ y = x^2 - 2*x)}

def complement (A : Set ℝ) : Set ℝ := {y | y ∉ A}

theorem problem_1 : 
  (complement E ∩ F) = {y | -1 ≤ y ∧ y ≤ 2} := 
  sorry

def G (a : ℝ) : Set ℝ := {y | ∃ (x : ℝ), (0 < x ∧ x < a ∧ y = Real.log x / Real.log 2)}

theorem problem_2 (a : ℝ) :
  (∀ y, (y ∈ G a → y < 3)) → a ≥ 8 :=
  sorry

end ProofProblems

end problem_1_problem_2_l149_149414


namespace problem1_l149_149114

theorem problem1 : 1361 + 972 + 693 + 28 = 3000 :=
by
  sorry

end problem1_l149_149114


namespace dot_product_a_b_l149_149782

open Real

noncomputable def cos_deg (x : ℝ) := cos (x * π / 180)
noncomputable def sin_deg (x : ℝ) := sin (x * π / 180)

theorem dot_product_a_b :
  let a_magnitude := 2 * cos_deg 15
  let b_magnitude := 4 * sin_deg 15
  let angle_ab := 30
  a_magnitude * b_magnitude * cos_deg angle_ab = sqrt 3 :=
by
  -- proof omitted
  sorry

end dot_product_a_b_l149_149782


namespace greatest_power_of_two_factor_l149_149496

theorem greatest_power_of_two_factor (n m : ℕ) (h1 : n = 12) (h2 : m = 8) :
  ∃ k, k = 1209 ∧ 2^k ∣ n^603 - m^402 :=
by
  sorry

end greatest_power_of_two_factor_l149_149496


namespace anya_age_l149_149684

theorem anya_age (n : ℕ) (h : 110 ≤ (n * (n + 1)) / 2 ∧ (n * (n + 1)) / 2 ≤ 130) : n = 15 :=
sorry

end anya_age_l149_149684


namespace find_factorial_number_l149_149908

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_factorial_number (n : ℕ) : Prop :=
  ∃ x y z : ℕ, (0 ≤ x ∧ x ≤ 5) ∧
               (0 ≤ y ∧ y ≤ 5) ∧
               (0 ≤ z ∧ z ≤ 5) ∧
               n = 100 * x + 10 * y + z ∧
               n = x.factorial + y.factorial + z.factorial

theorem find_factorial_number : ∃ n, is_three_digit_number n ∧ is_factorial_number n ∧ n = 145 :=
by {
  sorry
}

end find_factorial_number_l149_149908


namespace remainder_of_2_pow_87_plus_3_mod_7_l149_149781

theorem remainder_of_2_pow_87_plus_3_mod_7 : (2^87 + 3) % 7 = 4 := by
  sorry

end remainder_of_2_pow_87_plus_3_mod_7_l149_149781


namespace max_value_abcd_l149_149037

-- Define the digits and constraints on them
def distinct_digits (a b c d e : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

-- Encode the given problem as a Lean theorem
theorem max_value_abcd (a b c d e : ℕ) 
  (h₀ : distinct_digits a b c d e)
  (h₁ : 0 ≤ a ∧ a ≤ 9) 
  (h₂ : 0 ≤ b ∧ b ≤ 9) 
  (h₃ : 0 ≤ c ∧ c ≤ 9) 
  (h₄ : 0 ≤ d ∧ d ≤ 9)
  (h₅ : 0 ≤ e ∧ e ≤ 9)
  (h₆ : e ≠ 0)
  (h₇ : a * 1000 + b * 100 + c * 10 + d = (a * 100 + a * 10 + d) * e) :
  a * 1000 + b * 100 + c * 10 + d = 3015 :=
by {
  sorry
}

end max_value_abcd_l149_149037


namespace problem1_l149_149277

theorem problem1 (a : ℝ) (x : ℝ) (h : a > 0) : |x - (1/a)| + |x + a| ≥ 2 :=
sorry

end problem1_l149_149277


namespace sequence_an_general_formula_sequence_bn_sum_l149_149518

theorem sequence_an_general_formula
  (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : ∀ n, S n = (3 / 2) * a n - (1 / 2) * a 1)
  (h2 : 2 * (a 2 + 6) = a 1 + a 3) :
  ∀ n, a n = 3 ^ n := sorry

theorem sequence_bn_sum
  (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, S n = (3 / 2) * a n - (1 / 2) * a 1)
  (h2 : 2 * (a 2 + 6) = a 1 + a 3)
  (h3 : ∀ n, b n = a (n + 1) / (S n * S (n + 1))) :
  ∀ n, T n = (2 / 3) * (1 / 2 - 1 / (3 ^ (n + 1) - 1)) := sorry

end sequence_an_general_formula_sequence_bn_sum_l149_149518


namespace reservoir_solution_l149_149042

theorem reservoir_solution (x y z : ℝ) :
  8 * (1 / x - 1 / y) = 1 →
  24 * (1 / x - 1 / y - 1 / z) = 1 →
  8 * (1 / y + 1 / z) = 1 →
  x = 8 ∧ y = 24 ∧ z = 12 :=
by
  intros h1 h2 h3
  sorry

end reservoir_solution_l149_149042


namespace marbles_left_l149_149366

def initial_marbles : ℝ := 9.0
def given_marbles : ℝ := 3.0

theorem marbles_left : initial_marbles - given_marbles = 6.0 := 
by
  sorry

end marbles_left_l149_149366


namespace arrangements_count_l149_149360

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of positions
def num_positions : ℕ := 3

-- Define a type for the students
inductive Student
| A | B | C | D | E

-- Define the positions
inductive Position
| athletics | swimming | ball_games

-- Constraint: student A cannot be the swimming volunteer
def cannot_be_swimming_volunteer (s : Student) (p : Position) : Prop :=
  (s = Student.A → p ≠ Position.swimming)

-- Define the function to count the arrangements given the constraints
noncomputable def count_arrangements : ℕ :=
  (num_students.choose num_positions) - 1 -- Placeholder for the actual count based on given conditions

-- The theorem statement
theorem arrangements_count : count_arrangements = 16 :=
by
  sorry

end arrangements_count_l149_149360


namespace veronica_cans_of_food_is_multiple_of_4_l149_149923

-- Definitions of the given conditions
def number_of_water_bottles : ℕ := 20
def number_of_kits : ℕ := 4

-- Proof statement
theorem veronica_cans_of_food_is_multiple_of_4 (F : ℕ) :
  F % number_of_kits = 0 :=
sorry

end veronica_cans_of_food_is_multiple_of_4_l149_149923


namespace road_completion_days_l149_149244

variable (L : ℕ) (M_1 : ℕ) (W_1 : ℕ) (t1 : ℕ) (M_2 : ℕ)

theorem road_completion_days : L = 10 ∧ M_1 = 30 ∧ W_1 = 2 ∧ t1 = 5 ∧ M_2 = 60 → D = 15 :=
by
  sorry

end road_completion_days_l149_149244


namespace father_seven_times_as_old_l149_149115

theorem father_seven_times_as_old (x : ℕ) (father_age : ℕ) (son_age : ℕ) :
  father_age = 38 → son_age = 14 → (father_age - x = 7 * (son_age - x) → x = 10) :=
by
  intros h_father_age h_son_age h_equation
  rw [h_father_age, h_son_age] at h_equation
  sorry

end father_seven_times_as_old_l149_149115


namespace F_at_2_eq_minus_22_l149_149767

variable (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x^5 + c * x^3 + d * x

def F (x : ℝ) : ℝ := f a b c d x - 6

theorem F_at_2_eq_minus_22 (h : F a b c d (-2) = 10) : F a b c d 2 = -22 :=
by
  sorry

end F_at_2_eq_minus_22_l149_149767


namespace relationship_y1_y2_l149_149133

theorem relationship_y1_y2 (k b y1 y2 : ℝ) (h₀ : k < 0) (h₁ : y1 = k * (-1) + b) (h₂ : y2 = k * 1 + b) : y1 > y2 := 
by
  sorry

end relationship_y1_y2_l149_149133


namespace alice_savings_l149_149884

def sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.10

theorem alice_savings :
  (basic_salary + (sales * commission_rate)) * savings_rate = 29 :=
by
  sorry

end alice_savings_l149_149884


namespace find_square_number_divisible_by_9_between_40_and_90_l149_149013

theorem find_square_number_divisible_by_9_between_40_and_90 :
  ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (9 ∣ x) ∧ 40 < x ∧ x < 90 ∧ x = 81 :=
by
  sorry

end find_square_number_divisible_by_9_between_40_and_90_l149_149013


namespace function_is_increasing_l149_149098

theorem function_is_increasing : ∀ (x1 x2 : ℝ), x1 < x2 → (2 * x1 + 1) < (2 * x2 + 1) :=
by sorry

end function_is_increasing_l149_149098


namespace min_AB_DE_l149_149860

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_focus (k x y : ℝ) : Prop := y = k * (x - 1)

theorem min_AB_DE 
(F : (ℝ × ℝ)) 
(A B D E : ℝ × ℝ) 
(k1 k2 : ℝ) 
(hF : F = (1, 0)) 
(hk : k1^2 + k2^2 = 1) 
(hAB : ∀ x y, parabola x y → line_through_focus k1 x y → A = (x, y) ∨ B = (x, y)) 
(hDE : ∀ x y, parabola x y → line_through_focus k2 x y → D = (x, y) ∨ E = (x, y)) 
: |(A.1 - B.1)| + |(D.1 - E.1)| ≥ 24 := 
sorry

end min_AB_DE_l149_149860


namespace Isabela_spent_l149_149102

theorem Isabela_spent (num_pencils : ℕ) (cost_per_item : ℕ) (num_cucumbers : ℕ)
  (h1 : cost_per_item = 20)
  (h2 : num_cucumbers = 100)
  (h3 : num_cucumbers = 2 * num_pencils)
  (discount : ℚ := 0.20) :
  let pencil_cost := num_pencils * cost_per_item
  let cucumber_cost := num_cucumbers * cost_per_item
  let discounted_pencil_cost := pencil_cost * (1 - discount)
  let total_cost := cucumber_cost + discounted_pencil_cost
  total_cost = 2800 := by
  -- Begin proof. We will add actual proof here later.
  sorry

end Isabela_spent_l149_149102


namespace max_AC_not_RS_l149_149530

theorem max_AC_not_RS (TotalCars NoACCars MinRS MaxACnotRS : ℕ)
  (h1 : TotalCars = 100)
  (h2 : NoACCars = 49)
  (h3 : MinRS >= 51)
  (h4 : (TotalCars - NoACCars) - MinRS = MaxACnotRS)
  : MaxACnotRS = 0 :=
by
  sorry

end max_AC_not_RS_l149_149530


namespace max_non_triangulated_segments_correct_l149_149203

open Classical

/-
Problem description:
Given an equilateral triangle divided into smaller equilateral triangles with side length 1, 
we need to define the maximum number of 1-unit segments that can be marked such that no 
triangular subregion has all its sides marked.
-/

def total_segments (n : ℕ) : ℕ :=
  (3 * n * (n + 1)) / 2

def max_non_triangular_segments (n : ℕ) : ℕ :=
  n * (n + 1)

theorem max_non_triangulated_segments_correct (n : ℕ) :
  max_non_triangular_segments n = n * (n + 1) := by sorry

end max_non_triangulated_segments_correct_l149_149203


namespace tangent_parabola_line_l149_149239

theorem tangent_parabola_line (a : ℝ) :
  (∃ x0 : ℝ, ax0^2 + 3 = 2 * x0 + 1) ∧ (∀ x : ℝ, a * x^2 - 2 * x + 2 = 0 → x = x0) → a = 1/2 :=
by
  intro h
  sorry

end tangent_parabola_line_l149_149239


namespace square_fits_in_unit_cube_l149_149821

theorem square_fits_in_unit_cube (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) :
  let PQ := Real.sqrt (2 * (1 - x) ^ 2)
  let PS := Real.sqrt (1 + 2 * x ^ 2)
  (PQ > 1.05 ∧ PS > 1.05) :=
by
  sorry

end square_fits_in_unit_cube_l149_149821


namespace traffic_light_probability_change_l149_149959

theorem traffic_light_probability_change :
  let cycle_time := 100
  let intervals := [(0, 50), (50, 55), (55, 100)]
  let time_changing := [((45, 50), 5), ((50, 55), 5), ((95, 100), 5)]
  let total_change_time := time_changing.map Prod.snd |>.sum
  let probability := (total_change_time : ℚ) / cycle_time
  probability = 3 / 20 := sorry

end traffic_light_probability_change_l149_149959


namespace fixed_point_l149_149340

theorem fixed_point (m : ℝ) : (2 * m - 1) * 2 - (m + 3) * 3 - (m - 11) = 0 :=
by {
  sorry
}

end fixed_point_l149_149340


namespace translate_one_chapter_in_three_hours_l149_149256

-- Definitions representing the conditions:
def jun_seok_time : ℝ := 4
def yoon_yeol_time : ℝ := 12

-- Question and Correct answer as a statement:
theorem translate_one_chapter_in_three_hours :
  (1 / (1 / jun_seok_time + 1 / yoon_yeol_time)) = 3 := by
sorry

end translate_one_chapter_in_three_hours_l149_149256


namespace average_age_of_students_l149_149282

theorem average_age_of_students :
  (8 * 14 + 6 * 16 + 17) / 15 = 15 :=
by
  sorry

end average_age_of_students_l149_149282


namespace probability_A_fires_proof_l149_149291

noncomputable def probability_A_fires (prob_A : ℝ) (prob_B : ℝ) : ℝ :=
  1 / 6 + (5 / 6) * (5 / 6) * prob_A

theorem probability_A_fires_proof :
  ∀ prob_A prob_B : ℝ,
  prob_A = (1 / 6 + (5 / 6) * (5 / 6) * prob_A) →
  prob_A = 6 / 11 :=
by
  sorry

end probability_A_fires_proof_l149_149291


namespace store_credit_percentage_l149_149773

theorem store_credit_percentage (SN NES cash_given change_back game_value : ℕ) (P : ℚ)
  (hSN : SN = 150)
  (hNES : NES = 160)
  (hcash_given : cash_given = 80)
  (hchange_back : change_back = 10)
  (hgame_value : game_value = 30)
  (hP_def : NES = P * SN + (cash_given - change_back) + game_value) :
  P = 0.4 :=
  sorry

end store_credit_percentage_l149_149773


namespace compare_y1_y2_l149_149460

-- Define the function
def f (x : ℝ) : ℝ := -3 * x + 1

-- Define the points
def y1 := f 1
def y2 := f 3

-- The theorem to be proved
theorem compare_y1_y2 : y1 > y2 :=
by
  -- Proof placeholder
  sorry

end compare_y1_y2_l149_149460


namespace quadratic_solutions_l149_149952

theorem quadratic_solutions :
  ∀ x : ℝ, (x^2 - 4 * x = 0) → (x = 0 ∨ x = 4) :=
by sorry

end quadratic_solutions_l149_149952


namespace max_min_values_of_f_l149_149721

noncomputable def f (x : ℝ) : ℝ :=
  4^x - 2^(x+1) - 3

theorem max_min_values_of_f :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → (∀ y, y = f x → y ≤ 5) ∧ (∃ y, y = f 2 ∧ y = 5) ∧ (∀ y, y = f x → y ≥ -4) ∧ (∃ y, y = f 0 ∧ y = -4) :=
by
  sorry

end max_min_values_of_f_l149_149721


namespace katie_baked_5_cookies_l149_149145

theorem katie_baked_5_cookies (cupcakes cookies sold left : ℕ) 
  (h1 : cupcakes = 7) 
  (h2 : sold = 4) 
  (h3 : left = 8) 
  (h4 : cupcakes + cookies = sold + left) : 
  cookies = 5 :=
by sorry

end katie_baked_5_cookies_l149_149145


namespace feathers_before_crossing_road_l149_149369

theorem feathers_before_crossing_road : 
  ∀ (F : ℕ), 
  (F - (2 * 23) = 5217) → 
  F = 5263 :=
by
  intros F h
  sorry

end feathers_before_crossing_road_l149_149369


namespace kids_meals_sold_l149_149055

theorem kids_meals_sold (x y : ℕ) (h1 : x / y = 2) (h2 : x + y = 12) : x = 8 :=
by
  sorry

end kids_meals_sold_l149_149055


namespace cheesecake_total_calories_l149_149221

-- Define the conditions
def slice_calories : ℕ := 350

def percent_eaten : ℕ := 25
def slices_eaten : ℕ := 2

-- Define the total number of slices in a cheesecake
def total_slices (percent_eaten slices_eaten : ℕ) : ℕ :=
  slices_eaten * (100 / percent_eaten)

-- Define the total calories in a cheesecake given the above conditions
def total_calories (slice_calories slices : ℕ) : ℕ :=
  slice_calories * slices

-- State the theorem
theorem cheesecake_total_calories :
  total_calories slice_calories (total_slices percent_eaten slices_eaten) = 2800 :=
by
  sorry

end cheesecake_total_calories_l149_149221


namespace shortest_ribbon_length_is_10_l149_149358

noncomputable def shortest_ribbon_length (L : ℕ) : Prop :=
  (∃ k1 : ℕ, L = 2 * k1) ∧ (∃ k2 : ℕ, L = 5 * k2)

theorem shortest_ribbon_length_is_10 : shortest_ribbon_length 10 :=
by
  sorry

end shortest_ribbon_length_is_10_l149_149358


namespace greatest_x_value_l149_149309

theorem greatest_x_value :
  ∃ x : ℝ, (x ≠ 2 ∧ (x^2 - 5 * x - 14) / (x - 2) = 4 / (x + 4)) ∧ x = -2 ∧ 
           ∀ y, (y ≠ 2 ∧ (y^2 - 5 * y - 14) / (y - 2) = 4 / (y + 4)) → y ≤ x :=
by
  sorry

end greatest_x_value_l149_149309


namespace number_of_friends_with_pears_l149_149452

-- Each friend either carries pears or oranges
def total_friends : Nat := 15
def friends_with_oranges : Nat := 6
def friends_with_pears : Nat := total_friends - friends_with_oranges

theorem number_of_friends_with_pears :
  friends_with_pears = 9 := by
  -- Proof steps would go here
  sorry

end number_of_friends_with_pears_l149_149452


namespace calculate_expression_l149_149688

theorem calculate_expression : 1^345 + 5^10 / 5^7 = 126 := by
  sorry

end calculate_expression_l149_149688


namespace find_wrong_observation_value_l149_149695

-- Defining the given conditions
def original_mean : ℝ := 36
def corrected_mean : ℝ := 36.5
def num_observations : ℕ := 50
def correct_value : ℝ := 30

-- Defining the given sums based on means
def original_sum : ℝ := num_observations * original_mean
def corrected_sum : ℝ := num_observations * corrected_mean

-- The wrong value can be calculated based on the difference
def wrong_value : ℝ := correct_value + (corrected_sum - original_sum)

-- The theorem to prove
theorem find_wrong_observation_value (h : original_sum = 1800) (h' : corrected_sum = 1825) :
  wrong_value = 55 :=
sorry

end find_wrong_observation_value_l149_149695


namespace divide_400_l149_149334

theorem divide_400 (a b c d : ℕ) (h1 : a + b + c + d = 400) 
  (h2 : a + 1 = b - 2) (h3 : a + 1 = 3 * c) (h4 : a + 1 = d / 4) 
  : a = 62 ∧ b = 65 ∧ c = 21 ∧ d = 252 :=
sorry

end divide_400_l149_149334


namespace no_integer_solution_for_system_l149_149147

theorem no_integer_solution_for_system :
  (¬ ∃ x y : ℤ, 18 * x + 27 * y = 21 ∧ 27 * x + 18 * y = 69) :=
by
  sorry

end no_integer_solution_for_system_l149_149147


namespace correct_divisor_l149_149301

-- Definitions of variables and conditions
variables (X D : ℕ)

-- Stating the theorem
theorem correct_divisor (h1 : X = 49 * 12) (h2 : X = 28 * D) : D = 21 :=
by
  sorry

end correct_divisor_l149_149301


namespace parabola_y_coordinate_l149_149512

theorem parabola_y_coordinate (x y : ℝ) :
  x^2 = 4 * y ∧ (x - 0)^2 + (y - 1)^2 = 16 → y = 3 :=
by
  sorry

end parabola_y_coordinate_l149_149512


namespace rectangular_solid_sum_of_edges_l149_149175

noncomputable def sum_of_edges (x y z : ℝ) := 4 * (x + y + z)

theorem rectangular_solid_sum_of_edges :
  ∃ (x y z : ℝ), (x * y * z = 512) ∧ (2 * (x * y + y * z + z * x) = 384) ∧
  (∃ (r a : ℝ), x = a / r ∧ y = a ∧ z = a * r) ∧ sum_of_edges x y z = 96 :=
by
  sorry

end rectangular_solid_sum_of_edges_l149_149175


namespace cuboid_breadth_l149_149251

theorem cuboid_breadth (l h A : ℝ) (w : ℝ) :
  l = 8 ∧ h = 12 ∧ A = 960 → 2 * (l * w + l * h + w * h) = A → w = 19.2 :=
by
  intros h1 h2
  sorry

end cuboid_breadth_l149_149251


namespace length_more_than_breadth_l149_149559

theorem length_more_than_breadth (length cost_per_metre total_cost : ℝ) (breadth : ℝ) :
  length = 60 → cost_per_metre = 26.50 → total_cost = 5300 → 
  (total_cost = (2 * length + 2 * breadth) * cost_per_metre) → length - breadth = 20 :=
by
  intros hlength hcost_per_metre htotal_cost hperimeter_cost
  rw [hlength, hcost_per_metre] at hperimeter_cost
  sorry

end length_more_than_breadth_l149_149559


namespace evaluate_expression_l149_149500

theorem evaluate_expression (a b : ℝ) (h1 : a = 4) (h2 : b = -1) : -2 * a ^ 2 - 3 * b ^ 2 + 2 * a * b = -43 :=
by
  sorry

end evaluate_expression_l149_149500


namespace solution_set_of_inequality_l149_149598

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_of_inequality_l149_149598


namespace rhombuses_in_grid_l149_149178

def number_of_rhombuses (n : ℕ) : ℕ :=
(n - 1) * n + (n - 1) * n

theorem rhombuses_in_grid :
  number_of_rhombuses 5 = 30 :=
by
  sorry

end rhombuses_in_grid_l149_149178


namespace distinct_positive_integers_criteria_l149_149655

theorem distinct_positive_integers_criteria (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)
  (hxyz_div : x * y * z ∣ (x * y - 1) * (y * z - 1) * (z * x - 1)) :
  (x, y, z) = (2, 3, 5) ∨ (x, y, z) = (2, 5, 3) ∨ (x, y, z) = (3, 2, 5) ∨
  (x, y, z) = (3, 5, 2) ∨ (x, y, z) = (5, 2, 3) ∨ (x, y, z) = (5, 3, 2) :=
by sorry

end distinct_positive_integers_criteria_l149_149655


namespace obtuse_angle_at_515_l149_149442

-- Definitions derived from conditions
def minuteHandDegrees (minute: ℕ) : ℝ := minute * 6.0
def hourHandDegrees (hour: ℕ) (minute: ℕ) : ℝ := hour * 30.0 + (minute * 0.5)

-- Main statement to be proved
theorem obtuse_angle_at_515 : 
  let hour := 5
  let minute := 15
  let minute_pos := minuteHandDegrees minute
  let hour_pos := hourHandDegrees hour minute
  let angle := abs (minute_pos - hour_pos)
  angle = 67.5 :=
by
  sorry

end obtuse_angle_at_515_l149_149442


namespace average_A_B_l149_149553

variables (A B C : ℝ)

def conditions (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧
  (B + C) / 2 = 43 ∧
  B = 31

theorem average_A_B (A B C : ℝ) (h : conditions A B C) : (A + B) / 2 = 40 :=
by
  sorry

end average_A_B_l149_149553


namespace minimum_type_A_tickets_value_of_m_l149_149341

theorem minimum_type_A_tickets (x : ℕ) (h1 : x + (500 - x) = 500) (h2 : x ≥ 3 * (500 - x)) : x = 375 := by
  sorry

theorem value_of_m (m : ℕ) (h : 500 * (1 + (m + 10) / 100) * (m + 20) = 56000) : m = 50 := by
  sorry

end minimum_type_A_tickets_value_of_m_l149_149341


namespace expression_evaluation_l149_149287

theorem expression_evaluation : 1 + 3 + 5 + 7 - (2 + 4 + 6) + 3^2 + 5^2 = 38 := by
  sorry

end expression_evaluation_l149_149287


namespace cost_of_one_dozen_pens_l149_149817

variable (x : ℝ)

-- Conditions 1 and 2 as assumptions
def pen_cost := 5 * x
def pencil_cost := x

axiom cost_equation  : 3 * pen_cost + 5 * pencil_cost = 200
axiom cost_ratio     : pen_cost / pencil_cost = 5 / 1 -- ratio is given

-- Question and target statement
theorem cost_of_one_dozen_pens : 12 * pen_cost = 600 :=
by
  sorry

end cost_of_one_dozen_pens_l149_149817


namespace product_telescope_l149_149624

theorem product_telescope : ((1 + (1 / 1)) * 
                             (1 + (1 / 2)) * 
                             (1 + (1 / 3)) * 
                             (1 + (1 / 4)) * 
                             (1 + (1 / 5)) * 
                             (1 + (1 / 6)) * 
                             (1 + (1 / 7)) * 
                             (1 + (1 / 8)) * 
                             (1 + (1 / 9)) * 
                             (1 + (1 / 10))) = 11 := 
by
  sorry

end product_telescope_l149_149624


namespace radian_to_degree_equivalent_l149_149156

theorem radian_to_degree_equivalent : 
  (7 / 12) * (180 : ℝ) = 105 :=
by
  sorry

end radian_to_degree_equivalent_l149_149156


namespace max_4x_3y_l149_149455

theorem max_4x_3y (x y : ℝ) (h : x^2 + y^2 = 16 * x + 8 * y + 8) : 4 * x + 3 * y ≤ 63 :=
sorry

end max_4x_3y_l149_149455


namespace unique_solution_integer_equation_l149_149785

theorem unique_solution_integer_equation : 
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 = x^2 * y^2 :=
by sorry

end unique_solution_integer_equation_l149_149785


namespace stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l149_149814

-- Definitions of fixed points and stable points
def is_fixed_point(f : ℝ → ℝ) (x : ℝ) : Prop := f x = x
def is_stable_point(f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x 

-- Problem 1: Stable points of g(x) = 2x - 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem stable_points_of_g : {x : ℝ | is_stable_point g x} = {1} :=
sorry

-- Problem 2: Prove A ⊂ B for any function f
theorem fixed_points_subset_stable_points (f : ℝ → ℝ) : 
  {x : ℝ | is_fixed_point f x} ⊆ {x : ℝ | is_stable_point f x} :=
sorry

-- Problem 3: Range of a for f(x) = ax^2 - 1 when A = B ≠ ∅
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

theorem range_of_a (a : ℝ) (h : ∃ x, is_fixed_point (f a) x ∧ is_stable_point (f a) x):
  - (1/4 : ℝ) ≤ a ∧ a ≤ (3/4 : ℝ) :=
sorry

end stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l149_149814


namespace jelly_beans_total_l149_149499

-- Definitions from the conditions
def vanilla : Nat := 120
def grape : Nat := 5 * vanilla + 50
def total : Nat := vanilla + grape

-- Statement to prove
theorem jelly_beans_total :
  total = 770 := 
by 
  sorry

end jelly_beans_total_l149_149499


namespace sum_of_squares_not_divisible_by_4_or_8_l149_149816

theorem sum_of_squares_not_divisible_by_4_or_8 (n : ℤ) (h : n % 2 = 1) :
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  ¬(4 ∣ sum_squares ∨ 8 ∣ sum_squares) :=
by
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  sorry

end sum_of_squares_not_divisible_by_4_or_8_l149_149816


namespace only_solutions_l149_149705

theorem only_solutions (m n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (condition : (Nat.choose m 2) - 1 = p^n) :
  (m = 5 ∧ n = 2 ∧ p = 3) ∨ (m = 8 ∧ n = 3 ∧ p = 3) :=
by
  sorry

end only_solutions_l149_149705


namespace Albert_cabbage_count_l149_149520

-- Define the conditions
def rows := 12
def heads_per_row := 15

-- State the theorem
theorem Albert_cabbage_count : rows * heads_per_row = 180 := 
by sorry

end Albert_cabbage_count_l149_149520


namespace problem_l149_149174

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 4
noncomputable def c : ℝ := Real.log 9 / Real.log 4

theorem problem : a = c ∧ a > b :=
by
  sorry

end problem_l149_149174


namespace parabola_units_shift_l149_149057

noncomputable def parabola_expression (A B : ℝ × ℝ) (x : ℝ) : ℝ :=
  let b := -5
  let c := 6
  x^2 + b * x + c

theorem parabola_units_shift (A B : ℝ × ℝ) (x : ℝ) (y : ℝ) :
  A = (2, 0) → B = (0, 6) → parabola_expression A B 4 = 2 →
  (y - 2 = 0) → true :=
by
  intro hA hB h4 hy
  sorry

end parabola_units_shift_l149_149057


namespace sugar_water_sweeter_l149_149274

variable (a b m : ℝ)
variable (a_pos : a > 0) (b_gt_a : b > a) (m_pos : m > 0)

theorem sugar_water_sweeter : (a + m) / (b + m) > a / b :=
by
  sorry

end sugar_water_sweeter_l149_149274


namespace foci_coordinates_l149_149095

-- Define the parameters for the hyperbola
def a_squared : ℝ := 3
def b_squared : ℝ := 1
def c_squared : ℝ := a_squared + b_squared

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

-- State the theorem about the coordinates of the foci
theorem foci_coordinates : {foci : ℝ × ℝ // foci = (-2, 0) ∨ foci = (2, 0)} :=
by 
  have ha : a_squared = 3 := rfl
  have hb : b_squared = 1 := rfl
  have hc : c_squared = a_squared + b_squared := rfl
  have c := Real.sqrt c_squared
  have hc' : c = 2 := 
  -- sqrt part can be filled if detailed, for now, just direct conclusion
  sorry
  exact ⟨(2, 0), Or.inr rfl⟩

end foci_coordinates_l149_149095


namespace quad_root_l149_149784

theorem quad_root (m : ℝ) (β : ℝ) (root_condition : ∃ α : ℝ, α = -5 ∧ (α + β) * (α * β) = x^2 + m * x - 10) : β = 2 :=
by
  sorry

end quad_root_l149_149784


namespace distance_min_value_l149_149679

theorem distance_min_value (a b c d : ℝ) 
  (h₁ : |b - (Real.log a) / a| + |c - d + 2| = 0) : 
  (a - c)^2 + (b - d)^2 = 9 / 2 :=
by {
  sorry
}

end distance_min_value_l149_149679


namespace savings_is_22_77_cents_per_egg_l149_149257

-- Defining the costs and discount condition
def cost_per_large_egg_StoreA : ℚ := 0.55
def cost_per_extra_large_egg_StoreA : ℚ := 0.65
def discounted_cost_of_three_trays_large_StoreB : ℚ := 38
def total_eggs_in_three_trays : ℕ := 90

-- Savings calculation
def savings_per_egg : ℚ := (cost_per_extra_large_egg_StoreA - (discounted_cost_of_three_trays_large_StoreB / total_eggs_in_three_trays)) * 100

-- The statement to prove
theorem savings_is_22_77_cents_per_egg : savings_per_egg = 22.77 :=
by
  -- Here the proof would go, but we are omitting it with sorry
  sorry

end savings_is_22_77_cents_per_egg_l149_149257


namespace second_hand_degree_per_minute_l149_149009

theorem second_hand_degree_per_minute :
  (∀ (t : ℝ), t = 60 → 360 / t = 6) :=
by
  intro t
  intro ht
  rw [ht]
  norm_num

end second_hand_degree_per_minute_l149_149009


namespace solve_for_x_l149_149900

theorem solve_for_x (x : ℚ) : ((1/3 - x) ^ 2 = 4) → (x = -5/3 ∨ x = 7/3) :=
by
  sorry

end solve_for_x_l149_149900


namespace truck_speed_kmph_l149_149556

theorem truck_speed_kmph (d : ℕ) (t : ℕ) (km_m : ℕ) (hr_s : ℕ) 
  (h1 : d = 600) (h2 : t = 20) (h3 : km_m = 1000) (h4 : hr_s = 3600) : 
  (d / t) * (hr_s / km_m) = 108 := by
  sorry

end truck_speed_kmph_l149_149556


namespace range_of_m_l149_149944

def A (x : ℝ) : Prop := x^2 - x - 6 > 0
def B (x m : ℝ) : Prop := (x - m) * (x - 2 * m) ≤ 0
def is_disjoint (A B : ℝ → Prop) : Prop := ∀ x, ¬ (A x ∧ B x)

theorem range_of_m (m : ℝ) : 
  is_disjoint (A) (B m) ↔ -1 ≤ m ∧ m ≤ 3 / 2 := by
  sorry

end range_of_m_l149_149944


namespace geometric_sequence_sum_l149_149830

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = a n * r) 
    (h1 : a 1 + a 2 = 40) 
    (h2 : a 3 + a 4 = 60) : 
    a 7 + a 8 = 135 :=
sorry

end geometric_sequence_sum_l149_149830


namespace sum_is_correct_l149_149289

noncomputable def calculate_sum : ℚ :=
  (4 / 3) + (13 / 9) + (40 / 27) + (121 / 81) - (8 / 3)

theorem sum_is_correct : calculate_sum = 171 / 81 := 
by {
  sorry
}

end sum_is_correct_l149_149289


namespace ratio_bob_to_jason_l149_149311

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := 35

theorem ratio_bob_to_jason : bob_grade / jason_grade = 1 / 2 := by
  sorry

end ratio_bob_to_jason_l149_149311


namespace jill_earnings_l149_149502

theorem jill_earnings :
  ∀ (hourly_wage : ℝ) (tip_rate : ℝ) (num_shifts : ℕ) (hours_per_shift : ℕ) (avg_orders_per_hour : ℝ),
  hourly_wage = 4.00 →
  tip_rate = 0.15 →
  num_shifts = 3 →
  hours_per_shift = 8 →
  avg_orders_per_hour = 40 →
  (num_shifts * hours_per_shift * hourly_wage + num_shifts * hours_per_shift * avg_orders_per_hour * tip_rate = 240) :=
by
  intros hourly_wage tip_rate num_shifts hours_per_shift avg_orders_per_hour
  intros hwage_eq trip_rate_eq nshifts_eq hshift_eq avgorder_eq
  sorry

end jill_earnings_l149_149502


namespace smallest_m_plus_n_l149_149307

theorem smallest_m_plus_n (m n : ℕ) (h1 : m > n) (h2 : n ≥ 1) 
(h3 : 1000 ∣ 1978^m - 1978^n) : m + n = 106 :=
sorry

end smallest_m_plus_n_l149_149307


namespace unique_number_l149_149718

theorem unique_number (a : ℕ) (h1 : 1 < a) 
  (h2 : ∀ p : ℕ, Prime p → p ∣ a^6 - 1 → p ∣ a^3 - 1 ∨ p ∣ a^2 - 1) : a = 2 :=
by
  sorry

end unique_number_l149_149718


namespace oldest_sibling_age_difference_l149_149005

theorem oldest_sibling_age_difference 
  (D : ℝ) 
  (avg_age : ℝ) 
  (hD : D = 25.75) 
  (h_avg : avg_age = 30) :
  ∃ A : ℝ, (A - D ≥ 17) :=
by
  sorry

end oldest_sibling_age_difference_l149_149005


namespace set_M_real_l149_149503

noncomputable def set_M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem set_M_real :
  set_M = {z : ℂ | ∃ x : ℝ, z = x} :=
by
  sorry

end set_M_real_l149_149503


namespace find_angle_A_max_perimeter_incircle_l149_149856

-- Definition of the triangle and the conditions
variables {A B C : Real} {a b c : Real} 

-- The conditions given in the problem
def triangle_conditions (a b c A B C : Real) : Prop :=
  (b + c = a * (Real.cos C + Real.sqrt 3 * Real.sin C)) ∧
  A + B + C = Real.pi

-- Part 1: Prove the value of angle A
theorem find_angle_A (a b c A B C : Real) 
(h : triangle_conditions a b c A B C) : 
A = Real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter of the incircle when a=2
theorem max_perimeter_incircle (b c A B C : Real) 
(h : triangle_conditions 2 b c A B C) : 
2 * Real.pi * (Real.sqrt 3 / 6 * (b + c - 2)) ≤ (2 * Real.sqrt 3 / 3) * Real.pi := sorry

end find_angle_A_max_perimeter_incircle_l149_149856


namespace passengers_taken_second_station_l149_149108

def initial_passengers : ℕ := 288
def passengers_dropped_first_station : ℕ := initial_passengers / 3
def passengers_after_first_station : ℕ := initial_passengers - passengers_dropped_first_station
def passengers_taken_first_station : ℕ := 280
def total_passengers_after_first_station : ℕ := passengers_after_first_station + passengers_taken_first_station
def passengers_dropped_second_station : ℕ := total_passengers_after_first_station / 2
def passengers_left_after_second_station : ℕ := total_passengers_after_first_station - passengers_dropped_second_station
def passengers_at_third_station : ℕ := 248

theorem passengers_taken_second_station : 
  ∃ (x : ℕ), passengers_left_after_second_station + x = passengers_at_third_station ∧ x = 12 :=
by 
  sorry

end passengers_taken_second_station_l149_149108


namespace range_of_g_l149_149771

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + 2 * Real.pi * Real.arcsin (x / 3) -
  (Real.arcsin (x / 3))^2 + (Real.pi^2 / 18) * (x^2 + 12 * x + 27)

lemma arccos_arcsin_identity (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.arccos x + Real.arcsin x = Real.pi / 2 := sorry

theorem range_of_g : ∀ (x : ℝ), -3 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, g x = y ∧ y ∈ Set.Icc (Real.pi^2 / 4) (5 * Real.pi^2 / 2) :=
sorry

end range_of_g_l149_149771


namespace gcd_2952_1386_l149_149647

theorem gcd_2952_1386 : Nat.gcd 2952 1386 = 18 := by
  sorry

end gcd_2952_1386_l149_149647


namespace each_partner_percentage_l149_149436

-- Defining the conditions as variables
variables (total_profit majority_share combined_amount : ℝ) (num_partners : ℕ)

-- Given conditions
def majority_owner_received_25_percent_of_total : total_profit * 0.25 = majority_share := sorry
def remaining_profit_distribution : total_profit - majority_share = 60000 := sorry
def combined_share_of_three : majority_share + 30000 = combined_amount := sorry
def total_profit_amount : total_profit = 80000 := sorry
def number_of_partners : num_partners = 4 := sorry

-- The theorem to be proven
theorem each_partner_percentage :
  ∃ (percent : ℝ), percent = 25 :=
sorry

end each_partner_percentage_l149_149436


namespace expression_value_l149_149748

theorem expression_value
  (x y : ℝ) 
  (h : x - 3 * y = 4) : 
  (x - 3 * y)^2 + 2 * x - 6 * y - 10 = 14 :=
by
  sorry

end expression_value_l149_149748


namespace num_ways_to_tile_3x5_is_40_l149_149870

-- Definition of the problem
def numTilings (tiles : List (ℕ × ℕ)) (m n : ℕ) : ℕ :=
  sorry -- Placeholder for actual tiling computation

-- Condition specific to this problem
def specificTiles : List (ℕ × ℕ) :=
  [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

-- Problem statement in Lean 4
theorem num_ways_to_tile_3x5_is_40 :
  numTilings specificTiles 3 5 = 40 :=
sorry

end num_ways_to_tile_3x5_is_40_l149_149870


namespace birds_per_cup_l149_149427

theorem birds_per_cup :
  ∀ (C B S T : ℕ) (H1 : C = 2) (H2 : S = 1 / 2 * C) (H3 : T = 21) (H4 : B = 14),
    ((C - S) * B = T) :=
by
  sorry

end birds_per_cup_l149_149427


namespace domain_of_inverse_function_l149_149234

noncomputable def log_inverse_domain : Set ℝ :=
  {y | y ≥ 5}

theorem domain_of_inverse_function :
  ∀ y, y ∈ log_inverse_domain ↔ ∃ x, x ≥ 3 ∧ y = 4 + Real.logb 2 (x - 1) :=
by
  sorry

end domain_of_inverse_function_l149_149234


namespace triangle_inequality_l149_149829

noncomputable def f (K : ℝ) (x : ℝ) : ℝ :=
  (x^4 + K * x^2 + 1) / (x^4 + x^2 + 1)

theorem triangle_inequality (K : ℝ) (a b c : ℝ) :
  (-1 / 2) < K ∧ K < 4 → ∃ (A B C : ℝ), A = f K a ∧ B = f K b ∧ C = f K c ∧ A + B > C ∧ A + C > B ∧ B + C > A :=
by
  sorry

end triangle_inequality_l149_149829


namespace christian_age_in_eight_years_l149_149899

-- Definitions from the conditions
def christian_current_age : ℕ := 72
def brian_age_in_eight_years : ℕ := 40

-- Theorem to prove
theorem christian_age_in_eight_years : ∃ (age : ℕ), age = christian_current_age + 8 ∧ age = 80 := by
  sorry

end christian_age_in_eight_years_l149_149899


namespace area_decreases_by_28_l149_149396

def decrease_in_area (s h : ℤ) (h_eq : h = s + 3) : ℤ :=
  let new_area := (s - 4) * (s + 7)
  let original_area := s * h
  new_area - original_area

theorem area_decreases_by_28 (s h : ℤ) (h_eq : h = s + 3) : decrease_in_area s h h_eq = -28 :=
sorry

end area_decreases_by_28_l149_149396


namespace vectors_form_basis_l149_149997

-- Define the vectors in set B
def e1 : ℝ × ℝ := (-1, 2)
def e2 : ℝ × ℝ := (3, 7)

-- Define a function that checks if two vectors form a basis
def form_basis (v1 v2 : ℝ × ℝ) : Prop :=
  let det := v1.1 * v2.2 - v1.2 * v2.1
  det ≠ 0

-- State the theorem that vectors e1 and e2 form a basis
theorem vectors_form_basis : form_basis e1 e2 :=
by
  -- Add the proof here
  sorry

end vectors_form_basis_l149_149997


namespace roadsters_paving_company_total_cement_l149_149950

noncomputable def cement_lexi : ℝ := 10
noncomputable def cement_tess : ℝ := cement_lexi + 0.20 * cement_lexi
noncomputable def cement_ben : ℝ := cement_tess - 0.10 * cement_tess
noncomputable def cement_olivia : ℝ := 2 * cement_ben

theorem roadsters_paving_company_total_cement :
  cement_lexi + cement_tess + cement_ben + cement_olivia = 54.4 := by
  sorry

end roadsters_paving_company_total_cement_l149_149950


namespace profit_in_december_l149_149359

variable (a : ℝ)

theorem profit_in_december (h_a: a > 0):
  (1 - 0.06) * (1 + 0.10) * a = (1 - 0.06) * (1 + 0.10) * a :=
by
  sorry

end profit_in_december_l149_149359


namespace find_naturals_for_divisibility_l149_149285

theorem find_naturals_for_divisibility (n : ℕ) (h1 : 3 * n ≠ 1) :
  (∃ k : ℤ, 7 * n + 5 = k * (3 * n - 1)) ↔ n = 1 ∨ n = 4 := 
by
  sorry

end find_naturals_for_divisibility_l149_149285


namespace charlene_sold_necklaces_l149_149183

theorem charlene_sold_necklaces 
  (initial_necklaces : ℕ) 
  (given_away : ℕ) 
  (remaining : ℕ) 
  (total_made : initial_necklaces = 60) 
  (given_to_friends : given_away = 18) 
  (left_with : remaining = 26) : 
  initial_necklaces - given_away - remaining = 16 := 
by
  sorry

end charlene_sold_necklaces_l149_149183


namespace correct_statements_count_l149_149589

-- Definitions
def proper_fraction (x : ℚ) : Prop := (0 < x) ∧ (x < 1)
def improper_fraction (x : ℚ) : Prop := (x ≥ 1)

-- Statements as conditions
def statement1 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a + b)
def statement2 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a * b)
def statement3 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a + b)
def statement4 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a * b)

-- The main theorem stating the correct answer
theorem correct_statements_count : 
  (¬ (∀ a b, statement1 a b)) ∧ 
  (∀ a b, statement2 a b) ∧ 
  (∀ a b, statement3 a b) ∧ 
  (¬ (∀ a b, statement4 a b)) → 
  (2 = 2)
:= by sorry

end correct_statements_count_l149_149589


namespace find_y_l149_149077

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 10) (h3 : x = 1) : y = 13 := by
  sorry

end find_y_l149_149077


namespace value_of_3x_plus_5y_l149_149826

variable (x y : ℚ)

theorem value_of_3x_plus_5y
  (h1 : x + 4 * y = 5) 
  (h2 : 5 * x + 6 * y = 7) : 3 * x + 5 * y = 6 := 
sorry

end value_of_3x_plus_5y_l149_149826


namespace population_decrease_is_25_percent_l149_149918

def initial_population : ℕ := 20000
def final_population_first_year : ℕ := initial_population + (initial_population * 25 / 100)
def final_population_second_year : ℕ := 18750

def percentage_decrease (initial final : ℕ) : ℚ :=
  ((initial - final : ℚ) * 100) / initial 

theorem population_decrease_is_25_percent :
  percentage_decrease final_population_first_year final_population_second_year = 25 :=
by
  sorry

end population_decrease_is_25_percent_l149_149918


namespace find_X_in_rectangle_diagram_l149_149596

theorem find_X_in_rectangle_diagram :
  ∀ (X : ℝ),
  (1 + 1 + 1 + 2 + X = 1 + 2 + 1 + 6) → X = 5 :=
by
  intros X h
  sorry

end find_X_in_rectangle_diagram_l149_149596


namespace total_white_papers_l149_149188

-- Define the given conditions
def papers_per_envelope : ℕ := 10
def number_of_envelopes : ℕ := 12

-- The theorem statement
theorem total_white_papers : (papers_per_envelope * number_of_envelopes) = 120 :=
by
  sorry

end total_white_papers_l149_149188


namespace find_e_l149_149004

theorem find_e 
  (a b c d e : ℕ) 
  (h1 : a = 16)
  (h2 : b = 2)
  (h3 : c = 3)
  (h4 : d = 12)
  (h5 : 32 / e = 288 / e) 
  : e = 9 := 
by
  sorry

end find_e_l149_149004


namespace smallest_factor_to_end_with_four_zeros_l149_149003

theorem smallest_factor_to_end_with_four_zeros :
  ∃ x : ℕ, (975 * 935 * 972 * x) % 10000 = 0 ∧
           (∀ y : ℕ, (975 * 935 * 972 * y) % 10000 = 0 → x ≤ y) ∧
           x = 20 := by
  -- The proof would go here.
  sorry

end smallest_factor_to_end_with_four_zeros_l149_149003


namespace weight_of_new_person_l149_149793

theorem weight_of_new_person 
  (average_weight_first_20 : ℕ → ℕ → ℕ)
  (new_average_weight : ℕ → ℕ → ℕ) 
  (total_weight_21 : ℕ): 
  (average_weight_first_20 1200 20 = 60) → 
  (new_average_weight (1200 + total_weight_21) 21 = 55) → 
  total_weight_21 = 55 := 
by 
  intros 
  sorry

end weight_of_new_person_l149_149793


namespace train_cross_signal_pole_in_18_seconds_l149_149253

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 550
noncomputable def crossing_time_platform : ℝ := 51
noncomputable def signal_pole_crossing_time : ℝ := 18

theorem train_cross_signal_pole_in_18_seconds (t l_p t_p t_s : ℝ)
    (h1 : t = train_length)
    (h2 : l_p = platform_length)
    (h3 : t_p = crossing_time_platform)
    (h4 : t_s = signal_pole_crossing_time) : 
    (t + l_p) / t_p = train_length / signal_pole_crossing_time :=
by
  unfold train_length platform_length crossing_time_platform signal_pole_crossing_time at *
  -- proof will go here
  sorry

end train_cross_signal_pole_in_18_seconds_l149_149253


namespace loot_box_cost_l149_149961

variable (C : ℝ) -- Declare cost of each loot box as a real number

-- Conditions (average value of items, money spent, loss)
def avg_value : ℝ := 3.5
def money_spent : ℝ := 40
def avg_loss : ℝ := 12

-- Derived equation
def equation := avg_value * (money_spent / C) = money_spent - avg_loss

-- Statement to prove
theorem loot_box_cost : equation C → C = 5 := by
  sorry

end loot_box_cost_l149_149961


namespace volunteer_org_percentage_change_l149_149839

theorem volunteer_org_percentage_change 
  (initial_membership : ℝ)
  (fall_increase_rate : ℝ)
  (spring_decrease_rate : ℝ) :
  (initial_membership = 100) →
  (fall_increase_rate = 0.05) →
  (spring_decrease_rate = 0.19) →
  (14.95 : ℝ) =
  ((initial_membership * (1 + fall_increase_rate)) * (1 - spring_decrease_rate)
  - initial_membership) / initial_membership * 100 := by
  sorry

end volunteer_org_percentage_change_l149_149839


namespace concert_total_cost_l149_149230

noncomputable def total_cost (ticket_cost : ℕ) (processing_fee_rate : ℚ) (parking_fee : ℕ)
  (entrance_fee_per_person : ℕ) (num_persons : ℕ) (refreshments_cost : ℕ) 
  (merchandise_cost : ℕ) : ℚ :=
  let ticket_total := ticket_cost * num_persons
  let processing_fee := processing_fee_rate * (ticket_total : ℚ)
  ticket_total + processing_fee + (parking_fee + entrance_fee_per_person * num_persons 
  + refreshments_cost + merchandise_cost)

theorem concert_total_cost :
  total_cost 75 0.15 10 5 2 20 40 = 252.50 := by 
  sorry

end concert_total_cost_l149_149230


namespace simplify_expression_l149_149691

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  (3 * x - 1 - 5 * x) / 3 = -(2 / 3) * x - (1 / 3) := 
by
  sorry

end simplify_expression_l149_149691


namespace days_in_first_quarter_2010_l149_149194

theorem days_in_first_quarter_2010 : 
  let not_leap_year := ¬ (2010 % 4 = 0)
  let days_in_february := 28
  let days_in_january_and_march := 31
  not_leap_year → days_in_february = 28 → days_in_january_and_march = 31 → (31 + 28 + 31 = 90)
:= 
sorry

end days_in_first_quarter_2010_l149_149194


namespace evaluate_expression_l149_149614

theorem evaluate_expression (x : ℕ) (h : x = 3) : x + x^2 * (x^(x^2)) = 177150 := by
  sorry

end evaluate_expression_l149_149614


namespace right_triangle_legs_sum_l149_149479

-- Definitions
def sum_of_legs (a b : ℕ) : ℕ := a + b

-- Main theorem statement
theorem right_triangle_legs_sum (x : ℕ) (h : x^2 + (x + 1)^2 = 53^2) :
  sum_of_legs x (x + 1) = 75 :=
sorry

end right_triangle_legs_sum_l149_149479


namespace range_a_ineq_value_of_a_plus_b_l149_149672

open Real

def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)
def g (a x : ℝ) : ℝ := a - abs (x - 2)

noncomputable def range_a (a : ℝ) : Prop :=
  ∃ x : ℝ, f x < g a x

theorem range_a_ineq (a : ℝ) : range_a a ↔ 4 < a := sorry

def solution_set (b : ℝ) : Prop :=
  ∀ x : ℝ, f x < g ((13/2) : ℝ) x ↔ (b < x ∧ x < 7/2)

theorem value_of_a_plus_b (b : ℝ) (h : solution_set b) : (13/2) + b = 6 := sorry

end range_a_ineq_value_of_a_plus_b_l149_149672


namespace fraction_product_l149_149262

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l149_149262


namespace books_per_continent_l149_149392

-- Definition of the given conditions
def total_books := 488
def continents_visited := 4

-- The theorem we need to prove
theorem books_per_continent : total_books / continents_visited = 122 :=
sorry

end books_per_continent_l149_149392


namespace trigonometric_identity_l149_149050

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (2 * Real.sin θ - 4 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 / 2 := 
by
  sorry

end trigonometric_identity_l149_149050


namespace ImpossibleNonConformists_l149_149916

open Int

def BadPairCondition (f : ℤ → ℤ) (n : ℤ) :=
  ∃ (pairs : Finset (ℤ × ℤ)), 
    pairs.card ≤ ⌊0.001 * (n.natAbs^2 : ℝ)⌋₊ ∧ 
    ∀ (x y : ℤ), (x, y) ∈ pairs → max (abs x) (abs y) ≤ n ∧ f (x + y) ≠ f x + f y

def NonConformistCondition (f : ℤ → ℤ) (n : ℤ) :=
  ∃ (conformists : Finset ℤ), 
    conformists.card > n ∧ 
    ∀ (a : ℤ), abs a ≤ n → (f a ≠ a * f 1 → a ∈ conformists)

theorem ImpossibleNonConformists (f : ℤ → ℤ) :
  (∀ (n : ℤ), n ≥ 0 → BadPairCondition f n) → 
  ¬ ∃ (n : ℤ), n ≥ 0 ∧ NonConformistCondition f n :=
  by 
    intros h_cond h_ex
    sorry

end ImpossibleNonConformists_l149_149916


namespace problem_conditions_l149_149607

noncomputable def f (a b c x : ℝ) := 3 * a * x^2 + 2 * b * x + c

theorem problem_conditions (a b c : ℝ) (h0 : a + b + c = 0)
  (h1 : f a b c 0 > 0) (h2 : f a b c 1 > 0) :
    (a > 0 ∧ -2 < b / a ∧ b / a < -1) ∧
    (∃ z1 z2 : ℝ, 0 < z1 ∧ z1 < 1 ∧ 0 < z2 ∧ z2 < 1 ∧ z1 ≠ z2 ∧ f a b c z1 = 0 ∧ f a b c z2 = 0) :=
by
  sorry

end problem_conditions_l149_149607


namespace ticket_costs_l149_149096

theorem ticket_costs (ticket_price : ℕ) (number_of_tickets : ℕ) : ticket_price = 44 ∧ number_of_tickets = 7 → ticket_price * number_of_tickets = 308 :=
by
  intros h
  cases h
  sorry

end ticket_costs_l149_149096


namespace min_value_of_x_plus_y_l149_149708

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)  
  (h : 19 / x + 98 / y = 1) : x + y ≥ 203 :=
sorry

end min_value_of_x_plus_y_l149_149708


namespace johns_share_is_1100_l149_149026

def total_amount : ℕ := 6600
def ratio_john : ℕ := 2
def ratio_jose : ℕ := 4
def ratio_binoy : ℕ := 6
def total_parts : ℕ := ratio_john + ratio_jose + ratio_binoy
def value_per_part : ℚ := total_amount / total_parts
def amount_received_by_john : ℚ := value_per_part * ratio_john

theorem johns_share_is_1100 : amount_received_by_john = 1100 := by
  sorry

end johns_share_is_1100_l149_149026


namespace graph_not_pass_through_second_quadrant_l149_149590

theorem graph_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = 2 * x - 3 ∧ x < 0 ∧ y > 0 :=
by sorry

end graph_not_pass_through_second_quadrant_l149_149590


namespace circle_symmetry_l149_149507

theorem circle_symmetry (a b : ℝ) 
  (h1 : ∀ x y : ℝ, (x - a)^2 + (y - b)^2 = 1 ↔ (x - 1)^2 + (y - 3)^2 = 1) 
  (symm_line : ∀ x y : ℝ, y = x + 1) : a + b = 2 :=
sorry

end circle_symmetry_l149_149507


namespace work_duration_l149_149985

theorem work_duration (work_rate_x work_rate_y : ℚ) (time_x : ℕ) (total_work : ℚ) :
  work_rate_x = (1 / 20) → 
  work_rate_y = (1 / 12) → 
  time_x = 4 → 
  total_work = 1 →
  ((time_x * work_rate_x) + ((total_work - (time_x * work_rate_x)) / (work_rate_x + work_rate_y))) = 10 := 
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end work_duration_l149_149985


namespace determine_a_perpendicular_l149_149015

theorem determine_a_perpendicular 
  (a : ℝ)
  (h1 : 2 * x + 3 * y + 5 = 0)
  (h2 : a * x + 3 * y - 4 = 0) 
  (h_perpendicular : ∀ x y, (2 * x + 3 * y + 5 = 0) → ∀ x y, (a * x + 3 * y - 4 = 0) → (-(2 : ℝ) / (3 : ℝ)) * (-(a : ℝ) / (3 : ℝ)) = -1) :
  a = -9 / 2 :=
by
  sorry

end determine_a_perpendicular_l149_149015


namespace tile_count_difference_l149_149374

theorem tile_count_difference (W : ℕ) (B : ℕ) (B' : ℕ) (added_black_tiles : ℕ)
  (hW : W = 16) (hB : B = 9) (h_add : added_black_tiles = 8) (hB' : B' = B + added_black_tiles) :
  B' - W = 1 :=
by
  sorry

end tile_count_difference_l149_149374


namespace arithmetic_sequence_20th_term_l149_149693

-- Definitions for the first term and common difference
def first_term : ℤ := 8
def common_difference : ℤ := -3

-- Define the general term for an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- The specific property we seek to prove: the 20th term is -49
theorem arithmetic_sequence_20th_term : arithmetic_sequence 20 = -49 := by
  -- Proof is omitted, filled with sorry
  sorry

end arithmetic_sequence_20th_term_l149_149693


namespace problem1_problem2_l149_149043

def prop_p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem problem1 (a : ℝ) (h_a : a = 1) (h_pq : ∃ x, prop_p x a ∧ prop_q x) :
  ∃ x, 2 < x ∧ x < 3 :=
by sorry

theorem problem2 (h_qp : ∀ x (a : ℝ), prop_q x → prop_p x a) :
  ∃ a, 1 < a ∧ a ≤ 2 :=
by sorry

end problem1_problem2_l149_149043


namespace proposition_3_true_proposition_4_true_l149_149896

def exp_pos (x : ℝ) : Prop := Real.exp x > 0

def two_power_gt_xsq (x : ℝ) : Prop := 2^x > x^2

def prod_gt_one (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop := a * b > 1

def geom_seq_nec_suff (a b c : ℝ) : Prop := ¬(b = Real.sqrt (a * c) ∨ (a * b = c * b ∧ b^2 = a * c))

theorem proposition_3_true (a b : ℝ) (ha : a > 1) (hb : b > 1) : prod_gt_one a b ha hb :=
sorry

theorem proposition_4_true (a b c : ℝ) : geom_seq_nec_suff a b c :=
sorry

end proposition_3_true_proposition_4_true_l149_149896


namespace ethanol_combustion_heat_l149_149279

theorem ethanol_combustion_heat (Q : Real) :
  (∃ (m : Real), m = 0.1 ∧ (∀ (n : Real), n = 1 → Q * n / m = 10 * Q)) :=
by
  sorry

end ethanol_combustion_heat_l149_149279


namespace determine_n_l149_149535

theorem determine_n (n : ℕ) (x : ℤ) (h : x^n + (2 + x)^n + (2 - x)^n = 0) : n = 1 :=
sorry

end determine_n_l149_149535


namespace intersection_M_N_l149_149780

noncomputable def M := {x : ℝ | x > 1}
noncomputable def N := {x : ℝ | x < 2}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l149_149780


namespace houses_with_animals_l149_149818

theorem houses_with_animals (n A B C x y : ℕ) (h1 : n = 2017) (h2 : A = 1820) (h3 : B = 1651) (h4 : C = 1182) 
    (hx : x = 1182) (hy : y = 619) : x - y = 563 := 
by {
  sorry
}

end houses_with_animals_l149_149818


namespace initial_number_of_orchids_l149_149533

theorem initial_number_of_orchids 
  (initial_orchids : ℕ)
  (cut_orchids : ℕ)
  (final_orchids : ℕ)
  (h_cut : cut_orchids = 19)
  (h_final : final_orchids = 21) :
  initial_orchids + cut_orchids = final_orchids → initial_orchids = 2 :=
by
  sorry

end initial_number_of_orchids_l149_149533


namespace total_people_present_l149_149389

def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698
def total_people : ℕ := number_of_parents + number_of_pupils

theorem total_people_present : total_people = 803 :=
by
  sorry

end total_people_present_l149_149389


namespace expand_expression_l149_149441

theorem expand_expression (x y : ℤ) : (x + 7) * (3 * y + 8) = 3 * x * y + 8 * x + 21 * y + 56 :=
by
  sorry

end expand_expression_l149_149441


namespace ratio_c_d_l149_149072

theorem ratio_c_d (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
    (h1 : 8 * x - 5 * y = c) (h2 : 10 * y - 12 * x = d) 
  : c / d = 2 / 3 := by
  sorry

end ratio_c_d_l149_149072


namespace total_embroidery_time_l149_149412

-- Defining the constants as given in the problem
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_flowers : ℕ := 50
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1 -- Implicitly from the problem statement

-- Total time calculation as a Lean theorem
theorem total_embroidery_time : 
  (stitches_per_godzilla * num_godzillas + 
   stitches_per_unicorn * num_unicorns + 
   stitches_per_flower * num_flowers) / stitches_per_minute = 1085 := 
by
  sorry

end total_embroidery_time_l149_149412


namespace abs_neg_2035_l149_149833

theorem abs_neg_2035 : abs (-2035) = 2035 := 
by {
  sorry
}

end abs_neg_2035_l149_149833


namespace original_amount_l149_149573

variable (M : ℕ)

def initialAmountAfterFirstLoss := M - M / 3
def amountAfterFirstWin := initialAmountAfterFirstLoss M + 10
def amountAfterSecondLoss := amountAfterFirstWin M - (amountAfterFirstWin M) / 3
def amountAfterSecondWin := amountAfterSecondLoss M + 20
def finalAmount := amountAfterSecondWin M - (amountAfterSecondWin M) / 4

theorem original_amount : finalAmount M = M → M = 30 :=
by
  sorry

end original_amount_l149_149573


namespace two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l149_149922

theorem two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one :
  (2.85 = 2850 * 0.001) := by
  sorry

end two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l149_149922


namespace width_of_canal_at_bottom_l149_149919

theorem width_of_canal_at_bottom (h : Real) (b : Real) : 
  (A = 1/2 * (top_width + b) * d) ∧ 
  (A = 840) ∧ 
  (top_width = 12) ∧ 
  (d = 84) 
  → b = 8 := 
by
  intros
  sorry

end width_of_canal_at_bottom_l149_149919


namespace f_is_odd_l149_149669

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem f_is_odd : ∀ x, f (-x) = -f x := by
  sorry

end f_is_odd_l149_149669


namespace geometric_sequence_iff_arithmetic_sequence_l149_149561

/-
  Suppose that {a_n} is an infinite geometric sequence with common ratio q, where q^2 ≠ 1.
  Also suppose that {b_n} is a sequence of positive natural numbers (ℕ).
  Prove that {a_{b_n}} forms a geometric sequence if and only if {b_n} forms an arithmetic sequence.
-/

theorem geometric_sequence_iff_arithmetic_sequence
  (a : ℕ → ℕ) (b : ℕ → ℕ) (q : ℝ)
  (h_geom_a : ∃ a1, ∀ n, a n = a1 * q ^ (n - 1))
  (h_q_squared_ne_one : q^2 ≠ 1)
  (h_bn_positive : ∀ n, 0 < b n) :
  (∃ a1, ∃ q', ∀ n, a (b n) = a1 * q' ^ n) ↔ (∃ d, ∀ n, b (n + 1) - b n = d) := 
sorry

end geometric_sequence_iff_arithmetic_sequence_l149_149561


namespace problem_statement_l149_149024

def T (m : ℕ) : ℕ := sorry
def H (m : ℕ) : ℕ := sorry

def p (m k : ℕ) : ℝ := 
  if k % 2 = 1 then 0 else sorry

theorem problem_statement (m : ℕ) : p m 0 ≥ p (m + 1) 0 := sorry

end problem_statement_l149_149024


namespace toothpick_count_l149_149747

theorem toothpick_count (length width : ℕ) (h_len : length = 20) (h_width : width = 10) : 
  2 * (length * (width + 1) + width * (length + 1)) = 430 :=
by
  sorry

end toothpick_count_l149_149747


namespace line_hyperbola_unique_intersection_l149_149451

theorem line_hyperbola_unique_intersection (k : ℝ) :
  (∃ (x y : ℝ), k * x - y - 2 * k = 0 ∧ x^2 - y^2 = 2 ∧ 
  ∀ y₁, y₁ ≠ y → k * x - y₁ - 2 * k ≠ 0 ∧ x^2 - y₁^2 ≠ 2) ↔ (k = 1 ∨ k = -1) :=
by
  sorry

end line_hyperbola_unique_intersection_l149_149451


namespace compare_powers_l149_149165

theorem compare_powers (a b c : ℕ) (h1 : a = 81^31) (h2 : b = 27^41) (h3 : c = 9^61) : a > b ∧ b > c := by
  sorry

end compare_powers_l149_149165


namespace fifteen_power_ab_l149_149381

theorem fifteen_power_ab (a b : ℤ) (R S : ℝ) 
  (hR : R = 3^a) 
  (hS : S = 5^b) : 
  15^(a * b) = R^b * S^a :=
by sorry

end fifteen_power_ab_l149_149381


namespace time_to_finish_furniture_l149_149578

-- Define the problem's conditions
def chairs : ℕ := 7
def tables : ℕ := 3
def minutes_per_piece : ℕ := 4

-- Define total furniture
def total_furniture : ℕ := chairs + tables

-- Define the function to calculate total time
def total_time (pieces : ℕ) (time_per_piece: ℕ) : ℕ :=
  pieces * time_per_piece

-- Theorem statement to be proven
theorem time_to_finish_furniture : total_time total_furniture minutes_per_piece = 40 := 
by
  -- Provide a placeholder for the proof
  sorry

end time_to_finish_furniture_l149_149578


namespace difference_face_local_value_8_l149_149729

theorem difference_face_local_value_8 :
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3  -- 0-indexed place for thousands
  let local_value := digit * 10^position
  local_value - face_value = 7992 :=
by
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3
  let local_value := digit * 10^position
  show local_value - face_value = 7992
  sorry

end difference_face_local_value_8_l149_149729


namespace possible_vertex_angles_of_isosceles_triangle_l149_149465

def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (β = γ) ∨ (γ = α)

def altitude_half_side (α β γ a b c : ℝ) : Prop :=
  (a = α / 2) ∨ (b = β / 2) ∨ (c = γ / 2)

theorem possible_vertex_angles_of_isosceles_triangle (α β γ a b c : ℝ) :
  isosceles_triangle α β γ →
  altitude_half_side α β γ a b c →
  α = 30 ∨ α = 120 ∨ α = 150 :=
by
  sorry

end possible_vertex_angles_of_isosceles_triangle_l149_149465


namespace boys_in_other_communities_l149_149339

def percentage_of_other_communities (p_M p_H p_S : ℕ) : ℕ :=
  100 - (p_M + p_H + p_S)

def number_of_boys_other_communities (total_boys : ℕ) (percentage_other : ℕ) : ℕ :=
  (percentage_other * total_boys) / 100

theorem boys_in_other_communities (N p_M p_H p_S : ℕ) (hN : N = 650) (hpM : p_M = 44) (hpH : p_H = 28) (hpS : p_S = 10) :
  number_of_boys_other_communities N (percentage_of_other_communities p_M p_H p_S) = 117 :=
by
  -- Steps to prove the theorem would go here
  sorry

end boys_in_other_communities_l149_149339


namespace number_of_valid_b_l149_149121

theorem number_of_valid_b : ∃ (bs : Finset ℂ), bs.card = 2 ∧ ∀ b ∈ bs, ∃ (x : ℂ), (x + b = b^2) :=
by
  sorry

end number_of_valid_b_l149_149121


namespace arithmetic_mean_15_23_37_45_l149_149318

def arithmetic_mean (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem arithmetic_mean_15_23_37_45 :
  arithmetic_mean 15 23 37 45 = 30 :=
by {
  sorry
}

end arithmetic_mean_15_23_37_45_l149_149318


namespace some_athletes_not_members_honor_society_l149_149996

universe u

variable {U : Type u} -- Assume U is our universe of discourse, e.g., individuals.
variables (Athletes Disciplined HonorSociety : U → Prop)

-- Conditions
def some_athletes_not_disciplined := ∃ x, Athletes x ∧ ¬Disciplined x
def all_honor_society_disciplined := ∀ x, HonorSociety x → Disciplined x

-- Correct Answer
theorem some_athletes_not_members_honor_society :
  some_athletes_not_disciplined Athletes Disciplined →
  all_honor_society_disciplined HonorSociety Disciplined →
  ∃ y, Athletes y ∧ ¬HonorSociety y :=
by
  intros h1 h2
  sorry

end some_athletes_not_members_honor_society_l149_149996


namespace factorize_x_squared_minus_one_l149_149964

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l149_149964


namespace lloyd_hourly_rate_l149_149714

variable (R : ℝ)  -- Lloyd's regular hourly rate

-- Conditions
def lloyd_works_regular_hours_per_day : Prop := R > 0
def lloyd_earns_excess_rate : Prop := 1.5 * R > 0
def lloyd_worked_hours : Prop := 10.5 > 7.5
def lloyd_earned_amount : Prop := 7.5 * R + 3 * 1.5 * R = 66

-- Theorem statement
theorem lloyd_hourly_rate (hr_pos : lloyd_works_regular_hours_per_day R)
                           (excess_rate : lloyd_earns_excess_rate R)
                           (worked_hours : lloyd_worked_hours)
                           (earned_amount : lloyd_earned_amount R) : 
    R = 5.5 :=
by sorry

end lloyd_hourly_rate_l149_149714


namespace solve_for_y_l149_149682

theorem solve_for_y (y : ℕ) (h : 5 * (2 ^ y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l149_149682


namespace max_single_player_salary_is_426000_l149_149295

noncomputable def max_single_player_salary (total_salary_cap : ℤ) (min_salary : ℤ) (num_players : ℤ) : ℤ :=
  total_salary_cap - (num_players - 1) * min_salary

theorem max_single_player_salary_is_426000 :
  ∃ y, max_single_player_salary 800000 17000 23 = y ∧ y = 426000 :=
by
  sorry

end max_single_player_salary_is_426000_l149_149295


namespace simplify_radicals_l149_149023

theorem simplify_radicals (q : ℝ) (hq : 0 < q) :
  (Real.sqrt (42 * q)) * (Real.sqrt (7 * q)) * (Real.sqrt (14 * q)) = 98 * q * Real.sqrt (3 * q) :=
by
  sorry

end simplify_radicals_l149_149023


namespace greatest_integer_property_l149_149155

theorem greatest_integer_property :
  ∃ n : ℤ, n < 1000 ∧ (∃ m : ℤ, 4 * n^3 - 3 * n = (2 * m - 1) * (2 * m + 1)) ∧ 
  (∀ k : ℤ, k < 1000 ∧ (∃ m : ℤ, 4 * k^3 - 3 * k = (2 * m - 1) * (2 * m + 1)) → k ≤ n) := by
  -- skipped the proof with sorry
  sorry

end greatest_integer_property_l149_149155


namespace cups_per_girl_l149_149627

noncomputable def numStudents := 30
noncomputable def numBoys := 10
noncomputable def numCupsByBoys := numBoys * 5
noncomputable def totalCups := 90
noncomputable def numGirls := 2 * numBoys
noncomputable def numCupsByGirls := totalCups - numCupsByBoys

theorem cups_per_girl : (numCupsByGirls / numGirls) = 2 := by
  sorry

end cups_per_girl_l149_149627


namespace largest_final_number_l149_149745

-- Define the sequence and conditions
def initial_number := List.replicate 40 [3, 1, 1, 2, 3] |> List.join

-- The transformation rule
def valid_transform (a b : ℕ) : ℕ := if a + b <= 9 then a + b else 0

-- Sum of digits of a number
def sum_digits : List ℕ → ℕ := List.foldr (· + ·) 0

-- Define the final valid number pattern
def valid_final_pattern (n : ℕ) : Prop := n = 77

-- The main theorem statement
theorem largest_final_number (seq : List ℕ) (h_seq : seq = initial_number) :
  valid_final_pattern (sum_digits seq) := sorry

end largest_final_number_l149_149745


namespace heptagonal_prism_faces_and_vertices_l149_149132

structure HeptagonalPrism where
  heptagonal_basis : ℕ
  lateral_faces : ℕ
  basis_vertices : ℕ

noncomputable def faces (h : HeptagonalPrism) : ℕ :=
  2 + h.lateral_faces

noncomputable def vertices (h : HeptagonalPrism) : ℕ :=
  h.basis_vertices * 2

theorem heptagonal_prism_faces_and_vertices : ∀ h : HeptagonalPrism,
  (h.heptagonal_basis = 2) →
  (h.lateral_faces = 7) →
  (h.basis_vertices = 7) →
  faces h = 9 ∧ vertices h = 14 :=
by
  intros
  simp [faces, vertices]
  sorry

end heptagonal_prism_faces_and_vertices_l149_149132


namespace largest_prime_factor_2999_l149_149552

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  -- Note: This would require actual computation logic to find the largest prime factor.
  sorry

theorem largest_prime_factor_2999 :
  largest_prime_factor 2999 = 103 :=
by 
  -- Given conditions:
  -- 1. 2999 is an odd number (doesn't need explicit condition in proof).
  -- 2. Sum of digits is 29, thus not divisible by 3.
  -- 3. 2999 is not divisible by 11.
  -- 4. 2999 is not divisible by 7, 13, 17, 19.
  -- 5. Prime factorization of 2999 is 29 * 103.
  admit -- actual proof will need detailed prime factor test results 

end largest_prime_factor_2999_l149_149552


namespace star_example_l149_149937

def star (a b : ℤ) : ℤ := a * b^3 - 2 * b + 2

theorem star_example : star 2 3 = 50 := by
  sorry

end star_example_l149_149937


namespace problem1_problem2_l149_149430

theorem problem1 (m : ℝ) (H : m > 0) (p : ∀ x : ℝ, (x+1)*(x-5) ≤ 0 → 1 - m ≤ x ∧ x ≤ 1 + m) : m ≥ 4 :=
sorry

theorem problem2 (x : ℝ) (m : ℝ) (H : m = 5) (disj : ∀ x : ℝ, ((x+1)*(x-5) ≤ 0 ∨ (1 - m ≤ x ∧ x ≤ 1 + m))
) (conj : ¬ ∃ x : ℝ, (x+1)*(x-5) ≤ 0 ∧ (1 - m ≤ x ∧ x ≤ 1 + m)) : (-4 ≤ x ∧ x < -1) ∨ (5 < x ∧ x < 6) :=
sorry

end problem1_problem2_l149_149430


namespace price_of_book_l149_149993

variables (D B : ℝ)

def younger_brother : ℝ := 10

theorem price_of_book 
  (h1 : D = 1/2 * (B + younger_brother))
  (h2 : B = 1/3 * (D + younger_brother)) : 
  D + B + younger_brother = 24 := 
sorry

end price_of_book_l149_149993


namespace pizza_payment_difference_l149_149879

theorem pizza_payment_difference
  (total_slices : ℕ := 12)
  (plain_cost : ℝ := 12)
  (onion_cost : ℝ := 3)
  (jack_onion_slices : ℕ := 4)
  (jack_plain_slices : ℕ := 3)
  (carl_plain_slices : ℕ := 5) :
  let total_cost := plain_cost + onion_cost
  let cost_per_slice := total_cost / total_slices
  let jack_onion_payment := jack_onion_slices * cost_per_slice
  let jack_plain_payment := jack_plain_slices * cost_per_slice
  let jack_total_payment := jack_onion_payment + jack_plain_payment
  let carl_total_payment := carl_plain_slices * cost_per_slice
  jack_total_payment - carl_total_payment = 2.5 :=
by
  sorry

end pizza_payment_difference_l149_149879


namespace ramu_profit_percent_l149_149936

-- Definitions of the given conditions
def usd_to_inr (usd : ℤ) : ℤ := usd * 45 / 10
def eur_to_inr (eur : ℤ) : ℤ := eur * 567 / 100
def jpy_to_inr (jpy : ℤ) : ℤ := jpy * 1667 / 10000

def cost_of_car_in_inr := usd_to_inr 10000
def engine_repair_cost_in_inr := eur_to_inr 3000
def bodywork_repair_cost_in_inr := jpy_to_inr 150000
def total_cost_in_inr := cost_of_car_in_inr + engine_repair_cost_in_inr + bodywork_repair_cost_in_inr

def selling_price_in_inr : ℤ := 80000
def profit_or_loss_in_inr : ℤ := selling_price_in_inr - total_cost_in_inr

-- Profit percent calculation
def profit_percent (profit_or_loss total_cost : ℤ) : ℚ := (profit_or_loss : ℚ) / (total_cost : ℚ) * 100

-- The theorem stating the mathematically equivalent problem
theorem ramu_profit_percent :
  profit_percent profit_or_loss_in_inr total_cost_in_inr = -8.06 := by
  sorry

end ramu_profit_percent_l149_149936


namespace company_sales_difference_l149_149197

theorem company_sales_difference:
  let price_A := 4
  let quantity_A := 300
  let total_sales_A := price_A * quantity_A
  let price_B := 3.5
  let quantity_B := 350
  let total_sales_B := price_B * quantity_B
  total_sales_B - total_sales_A = 25 :=
by
  sorry

end company_sales_difference_l149_149197


namespace find_r_of_tangential_cones_l149_149622

theorem find_r_of_tangential_cones (r : ℝ) : 
  (∃ (r1 r2 r3 R : ℝ), r1 = 2 * r ∧ r2 = 3 * r ∧ r3 = 10 * r ∧ R = 15 ∧
  -- Additional conditions to ensure the three cones touch and share a slant height
  -- with the truncated cone of radius R
  true) → r = 29 :=
by
  intro h
  sorry

end find_r_of_tangential_cones_l149_149622


namespace incorrect_expressions_l149_149336

-- Definitions for the conditions
def F : ℝ := sorry   -- F represents a repeating decimal
def X : ℝ := sorry   -- X represents the t digits of F that are non-repeating
def Y : ℝ := sorry   -- Y represents the u digits of F that repeat
def t : ℕ := sorry   -- t is the number of non-repeating digits
def u : ℕ := sorry   -- u is the number of repeating digits

-- Statement that expressions (C) and (D) are incorrect
theorem incorrect_expressions : 
  ¬ (10^(t + 2 * u) * F = X + Y / 10 ^ u) ∧ ¬ (10^t * (10^u - 1) * F = Y * (X - 1)) :=
sorry

end incorrect_expressions_l149_149336


namespace find_A_l149_149945

theorem find_A (A7B : ℕ) (H1 : (A7B % 100) / 10 = 7) (H2 : A7B + 23 = 695) : (A7B / 100) = 6 := 
  sorry

end find_A_l149_149945


namespace fibonacci_factorial_sum_l149_149229

def factorial_last_two_digits(n: ℕ) : ℕ :=
  if n > 10 then 0 else 
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 24
  | 5 => 120 % 100
  | 6 => 720 % 100
  | 7 => 5040 % 100
  | 8 => 40320 % 100
  | 9 => 362880 % 100
  | 10 => 3628800 % 100
  | _ => 0

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

noncomputable def sum_last_two_digits (l: List ℕ) : ℕ :=
  l.map factorial_last_two_digits |>.sum

theorem fibonacci_factorial_sum:
  sum_last_two_digits fibonacci_factorial_series = 50 := by
  sorry

end fibonacci_factorial_sum_l149_149229


namespace possible_values_of_a_l149_149036

theorem possible_values_of_a (a : ℝ) : (2 < a ∧ a < 3 ∨ 3 < a ∧ a < 5) → (a = 5/2 ∨ a = 4) := 
by
  sorry

end possible_values_of_a_l149_149036


namespace river_current_speed_l149_149892

variable (c : ℝ)

def boat_speed_still_water : ℝ := 20
def round_trip_distance : ℝ := 182
def round_trip_time : ℝ := 10

theorem river_current_speed (h : (91 / (boat_speed_still_water - c)) + (91 / (boat_speed_still_water + c)) = round_trip_time) : c = 6 :=
sorry

end river_current_speed_l149_149892


namespace triangle_to_pentagon_ratio_l149_149906

theorem triangle_to_pentagon_ratio (t p : ℕ) 
  (h1 : 3 * t = 15) 
  (h2 : 5 * p = 15) : (t : ℚ) / (p : ℚ) = 5 / 3 :=
by
  sorry

end triangle_to_pentagon_ratio_l149_149906


namespace cars_sold_first_day_l149_149269

theorem cars_sold_first_day (c_2 c_3 : ℕ) (total : ℕ) (h1 : c_2 = 16) (h2 : c_3 = 27) (h3 : total = 57) :
  ∃ c_1 : ℕ, c_1 + c_2 + c_3 = total ∧ c_1 = 14 :=
by
  sorry

end cars_sold_first_day_l149_149269


namespace intersection_M_N_l149_149011

def is_M (x : ℝ) : Prop := x^2 + x - 6 < 0
def is_N (x : ℝ) : Prop := abs (x - 1) <= 2

theorem intersection_M_N : {x : ℝ | is_M x} ∩ {x : ℝ | is_N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l149_149011


namespace parabola_vertex_l149_149411

theorem parabola_vertex :
  (∃ x y : ℝ, y^2 + 6 * y + 4 * x - 7 = 0 ∧ (x, y) = (4, -3)) :=
sorry

end parabola_vertex_l149_149411


namespace complement_of_M_with_respect_to_U_l149_149185

open Set

def U : Set ℤ := {-1, -2, -3, -4}
def M : Set ℤ := {-2, -3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {-1, -4} :=
by
  sorry

end complement_of_M_with_respect_to_U_l149_149185


namespace find_m_geq_9_l149_149795

-- Define the real numbers
variables {x m : ℝ}

-- Define the conditions
def p (x : ℝ) := x ≤ 2
def q (x m : ℝ) := x^2 - 2*x + 1 - m^2 ≤ 0

-- Main theorem statement based on the given problem
theorem find_m_geq_9 (m : ℝ) (hm : m > 0) :
  (¬ p x → ¬ q x m) → (p x → q x m) → m ≥ 9 :=
  sorry

end find_m_geq_9_l149_149795


namespace number_of_symmetric_subsets_l149_149271

def has_integer_solutions (m : ℤ) : Prop :=
  ∃ x y : ℤ, x * y = -36 ∧ x + y = -m

def M : Set ℤ :=
  {m | has_integer_solutions m}

def is_symmetric_subset (A : Set ℤ) : Prop :=
  A ⊆ M ∧ ∀ a ∈ A, -a ∈ A

theorem number_of_symmetric_subsets :
  (∃ A : Set ℤ, is_symmetric_subset A ∧ A ≠ ∅) →
  (∃ n : ℕ, n = 31) :=
by
  sorry

end number_of_symmetric_subsets_l149_149271


namespace smaller_circle_radius_l149_149019

theorem smaller_circle_radius (A1 A2 : ℝ) 
  (h1 : A1 + 2 * A2 = 25 * Real.pi) 
  (h2 : ∃ d : ℝ, A1 + d = A2 ∧ A2 + d = A1 + 2 * A2) : 
  ∃ r : ℝ, r^2 = 5 ∧ Real.pi * r^2 = A1 :=
by
  sorry

end smaller_circle_radius_l149_149019


namespace length_of_platform_l149_149585

variables (t L T_p T_s : ℝ)
def train_length := 200  -- length of the train in meters
def platform_cross_time := 50  -- time in seconds to cross the platform
def pole_cross_time := 42  -- time in seconds to cross the signal pole

theorem length_of_platform :
  T_p = platform_cross_time ->
  T_s = pole_cross_time ->
  t = train_length ->
  (L = 38) :=
by
  intros hp hsp ht
  sorry  -- proof goes here

end length_of_platform_l149_149585


namespace minimize_intercepts_line_eqn_l149_149192

theorem minimize_intercepts_line_eqn (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : (1:ℝ)/a + (1:ℝ)/b = 1)
  (h2 : ∃ a b, a + b = 4 ∧ a = 2 ∧ b = 2) :
  ∀ (x y : ℝ), x + y - 2 = 0 :=
by 
  sorry

end minimize_intercepts_line_eqn_l149_149192


namespace absent_children_count_l149_149220

-- Definition of conditions
def total_children := 700
def bananas_per_child := 2
def bananas_extra := 2
def total_bananas := total_children * bananas_per_child

-- The proof goal
theorem absent_children_count (A P : ℕ) (h_P : P = total_children - A)
    (h_bananas : total_bananas = P * (bananas_per_child + bananas_extra)) : A = 350 :=
by
  -- Since this is a statement only, we place a sorry here to skip the proof.
  sorry

end absent_children_count_l149_149220


namespace quadratic_roots_problem_l149_149218

theorem quadratic_roots_problem (m n : ℝ) (h1 : m^2 - 2 * m - 2025 = 0) (h2 : n^2 - 2 * n - 2025 = 0) (h3 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 :=
sorry

end quadratic_roots_problem_l149_149218


namespace smallest_number_of_students_l149_149696

theorem smallest_number_of_students 
  (n : ℕ) 
  (h1 : 4 * 80 + (n - 4) * 50 ≤ 65 * n) :
  n = 8 :=
by sorry

end smallest_number_of_students_l149_149696


namespace dad_steps_l149_149544

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l149_149544


namespace total_nails_l149_149409

def num_planks : Nat := 1
def nails_per_plank : Nat := 3
def additional_nails : Nat := 8

theorem total_nails : (num_planks * nails_per_plank + additional_nails) = 11 :=
by
  sorry

end total_nails_l149_149409


namespace unique_positive_real_solution_of_polynomial_l149_149018

theorem unique_positive_real_solution_of_polynomial :
  ∃! x : ℝ, x > 0 ∧ (x^11 + 8 * x^10 + 15 * x^9 + 1000 * x^8 - 1200 * x^7 = 0) :=
by
  sorry

end unique_positive_real_solution_of_polynomial_l149_149018


namespace outer_circle_increase_l149_149481

theorem outer_circle_increase : 
  let R_o := 6
  let R_i := 4
  let R_i_new := (3 : ℝ)  -- 4 * (3/4)
  let A_original := 20 * Real.pi  -- π * (6^2 - 4^2)
  let A_new := 72 * Real.pi  -- 3.6 * A_original
  ∃ (x : ℝ), 
    let R_o_new := R_o * (1 + x / 100)
    π * R_o_new^2 - π * R_i_new^2 = A_new →
    x = 50 := 
sorry

end outer_circle_increase_l149_149481


namespace benjamin_weekly_walks_l149_149248

def walking_miles_in_week
  (work_days_per_week : ℕ)
  (work_distance_per_day : ℕ)
  (dog_walks_per_day : ℕ)
  (dog_walk_distance : ℕ)
  (best_friend_visits_per_week : ℕ)
  (best_friend_distance : ℕ)
  (store_visits_per_week : ℕ)
  (store_distance : ℕ)
  (hike_distance_per_week : ℕ) : ℕ :=
  (work_days_per_week * work_distance_per_day) +
  (dog_walks_per_day * dog_walk_distance * 7) +
  (best_friend_visits_per_week * (best_friend_distance * 2)) +
  (store_visits_per_week * (store_distance * 2)) +
  hike_distance_per_week

theorem benjamin_weekly_walks :
  walking_miles_in_week 5 (8 * 2) 2 3 1 5 2 4 10 = 158 := 
  by
    sorry

end benjamin_weekly_walks_l149_149248


namespace find_years_ago_twice_age_l149_149692

-- Definitions of given conditions
def age_sum (H J : ℕ) : Prop := H + J = 43
def henry_age : ℕ := 27
def jill_age : ℕ := 16

-- Definition of the problem to be proved
theorem find_years_ago_twice_age (X : ℕ) 
  (h1 : age_sum henry_age jill_age) 
  (h2 : henry_age = 27) 
  (h3 : jill_age = 16) : (27 - X = 2 * (16 - X)) → X = 5 := 
by 
  sorry

end find_years_ago_twice_age_l149_149692


namespace number_of_perfect_squares_criteria_l149_149211

noncomputable def number_of_multiples_of_40_squares_lt_4e6 : ℕ :=
  let upper_limit := 2000
  let multiple := 40
  let largest_multiple := upper_limit - (upper_limit % multiple)
  largest_multiple / multiple

theorem number_of_perfect_squares_criteria :
  number_of_multiples_of_40_squares_lt_4e6 = 49 :=
sorry

end number_of_perfect_squares_criteria_l149_149211


namespace sum_of_roots_of_quadratic_l149_149929

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l149_149929


namespace solve_system_of_equations_l149_149367

theorem solve_system_of_equations :
  ∃ x y : ℝ, (3 * x - 5 * y = -1.5) ∧ (7 * x + 2 * y = 4.7) ∧ x = 0.5 ∧ y = 0.6 :=
by
  sorry -- Proof to be completed

end solve_system_of_equations_l149_149367


namespace students_drawn_in_sample_l149_149712

def total_people : ℕ := 1600
def number_of_teachers : ℕ := 100
def sample_size : ℕ := 80
def number_of_students : ℕ := total_people - number_of_teachers
def expected_students_sample : ℕ := 75

theorem students_drawn_in_sample : (sample_size * number_of_students) / total_people = expected_students_sample :=
by
  -- The proof steps would go here
  sorry

end students_drawn_in_sample_l149_149712


namespace jacoby_needs_l149_149378

-- Given conditions
def total_goal : ℤ := 5000
def job_earnings_per_hour : ℤ := 20
def total_job_hours : ℤ := 10
def cookie_price_each : ℤ := 4
def total_cookies_sold : ℤ := 24
def lottery_ticket_cost : ℤ := 10
def lottery_winning : ℤ := 500
def gift_from_sister_one : ℤ := 500
def gift_from_sister_two : ℤ := 500

-- Total money Jacoby has so far
def current_total_money : ℤ := 
  job_earnings_per_hour * total_job_hours +
  cookie_price_each * total_cookies_sold +
  lottery_winning +
  gift_from_sister_one + gift_from_sister_two -
  lottery_ticket_cost

-- The amount Jacoby needs to reach his goal
def amount_needed : ℤ := total_goal - current_total_money

-- The main statement to be proved
theorem jacoby_needs : amount_needed = 3214 := by
  -- The proof is skipped
  sorry

end jacoby_needs_l149_149378


namespace correct_option_is_B_l149_149259

-- Define the operations as hypotheses
def option_A (a : ℤ) : Prop := (a^2 + a^3 = a^5)
def option_B (a : ℤ) : Prop := ((a^2)^3 = a^6)
def option_C (a : ℤ) : Prop := (a^2 * a^3 = a^6)
def option_D (a : ℤ) : Prop := (6 * a^6 - 2 * a^3 = 3 * a^3)

-- Prove that option B is correct
theorem correct_option_is_B (a : ℤ) : option_B a :=
by
  unfold option_B
  sorry

end correct_option_is_B_l149_149259


namespace curve_not_parabola_l149_149346

theorem curve_not_parabola (k : ℝ) : ¬(∃ a b c : ℝ, a ≠ 0 ∧ x^2 + ky^2 = a*x^2 + b*y + c) :=
sorry

end curve_not_parabola_l149_149346


namespace max_product_of_xy_on_circle_l149_149488

theorem max_product_of_xy_on_circle (x y : ℤ) (h : x^2 + y^2 = 100) : 
  ∃ (x y : ℤ), (x^2 + y^2 = 100) ∧ (∀ x y : ℤ, x^2 + y^2 = 100 → x * y ≤ 48) ∧ x * y = 48 := by
  sorry

end max_product_of_xy_on_circle_l149_149488


namespace repeating_decimal_to_fraction_l149_149128

theorem repeating_decimal_to_fraction : (let a := (0.28282828 : ℚ); a = 28/99) := sorry

end repeating_decimal_to_fraction_l149_149128


namespace num_games_last_year_l149_149728

-- Definitions from conditions
def num_games_this_year : ℕ := 14
def total_num_games : ℕ := 43

-- Theorem to prove
theorem num_games_last_year (num_games_last_year : ℕ) : 
  total_num_games - num_games_this_year = num_games_last_year ↔ num_games_last_year = 29 :=
by
  sorry

end num_games_last_year_l149_149728


namespace percentage_runs_by_running_l149_149395

theorem percentage_runs_by_running 
  (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) 
  (runs_per_boundary : ℕ) (runs_per_six : ℕ)
  (H_total_runs : total_runs = 120)
  (H_boundaries : boundaries = 3)
  (H_sixes : sixes = 8)
  (H_runs_per_boundary : runs_per_boundary = 4)
  (H_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs : ℚ) * 100 = 50 := 
by
  sorry

end percentage_runs_by_running_l149_149395


namespace range_of_a_l149_149577

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end range_of_a_l149_149577


namespace price_after_two_reductions_l149_149946

variable (orig_price : ℝ) (m : ℝ)

def current_price (orig_price : ℝ) (m : ℝ) : ℝ :=
  orig_price * (1 - m) * (1 - m)

theorem price_after_two_reductions (h1 : orig_price = 100) (h2 : 0 ≤ m ∧ m ≤ 1) :
  current_price orig_price m = 100 * (1 - m) ^ 2 := by
    sorry

end price_after_two_reductions_l149_149946


namespace initial_units_of_phones_l149_149734

theorem initial_units_of_phones
  (X : ℕ) 
  (h1 : 5 = 5) 
  (h2 : X - 5 = 3 + 5 + 7) : 
  X = 20 := 
by
  sorry

end initial_units_of_phones_l149_149734


namespace right_triangle_example_find_inverse_450_mod_3599_l149_149958

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a b m : ℕ) : Prop :=
  (a * b) % m = 1

theorem right_triangle_example : is_right_triangle 60 221 229 :=
by
  sorry

theorem find_inverse_450_mod_3599 : ∃ n, 0 ≤ n ∧ n < 3599 ∧ multiplicative_inverse 450 n 3599 :=
by
  use 8
  sorry

end right_triangle_example_find_inverse_450_mod_3599_l149_149958


namespace both_locks_stall_time_l149_149149

-- Definitions of the conditions
def first_lock_time : ℕ := 5
def second_lock_time : ℕ := 3 * first_lock_time - 3
def both_locks_time : ℕ := 5 * second_lock_time

-- The proof statement
theorem both_locks_stall_time : both_locks_time = 60 := by
  sorry

end both_locks_stall_time_l149_149149


namespace find_profits_maximize_profit_week3_l149_149978

-- Defining the conditions of the problems
def week1_sales_A := 10
def week1_sales_B := 12
def week1_profit := 2000

def week2_sales_A := 20
def week2_sales_B := 15
def week2_profit := 3100

def total_sales_week3 := 25

-- Condition: Sales of type B exceed sales of type A but do not exceed twice the sales of type A
def sales_condition (x : ℕ) := (total_sales_week3 - x) > x ∧ (total_sales_week3 - x) ≤ 2 * x

-- Define the profits for types A and B
def profit_A (a b : ℕ) := week1_sales_A * a + week1_sales_B * b = week1_profit
def profit_B (a b : ℕ) := week2_sales_A * a + week2_sales_B * b = week2_profit

-- Define the profit function for week 3
def profit_week3 (a b x : ℕ) := a * x + b * (total_sales_week3 - x)

theorem find_profits : ∃ a b, profit_A a b ∧ profit_B a b :=
by
  use 80, 100
  sorry

theorem maximize_profit_week3 : 
  ∃ x y, 
  sales_condition x ∧ 
  x + y = total_sales_week3 ∧ 
  profit_week3 80 100 x = 2320 :=
by
  use 9, 16
  sorry

end find_profits_maximize_profit_week3_l149_149978


namespace radical_conjugate_sum_l149_149224

theorem radical_conjugate_sum:
  let a := 15 - Real.sqrt 500
  let b := 15 + Real.sqrt 500
  3 * (a + b) = 90 :=
by
  sorry

end radical_conjugate_sum_l149_149224


namespace arithmetic_sequence_15th_term_l149_149466

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem arithmetic_sequence_15th_term :
  arithmetic_sequence (-3) 4 15 = 53 :=
by
  sorry

end arithmetic_sequence_15th_term_l149_149466


namespace count_triples_satisfying_conditions_l149_149326

theorem count_triples_satisfying_conditions :
  (∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ ab + bc = 72 ∧ ac + bc = 35) → 
  ∃! t : (ℕ × ℕ × ℕ), 0 < t.1 ∧ 0 < t.2.1 ∧ 0 < t.2.2 ∧ 
                     t.1 * t.2.1 + t.2.1 * t.2.2 = 72 ∧ 
                     t.1 * t.2.2 + t.2.1 * t.2.2 = 35 :=
by sorry

end count_triples_satisfying_conditions_l149_149326


namespace allocate_teaching_positions_l149_149887

theorem allocate_teaching_positions :
  ∃ (ways : ℕ), ways = 10 ∧ 
    (∃ (a b c : ℕ), a + b + c = 8 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 2 ≤ a) := 
sorry

end allocate_teaching_positions_l149_149887


namespace stable_equilibrium_condition_l149_149086

theorem stable_equilibrium_condition
  (a b : ℝ)
  (h_condition1 : a > b)
  (h_condition2 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  : (b / a) < (1 / Real.sqrt 2) :=
sorry

end stable_equilibrium_condition_l149_149086


namespace final_amount_simple_interest_l149_149290

theorem final_amount_simple_interest (P R T : ℕ) (hP : P = 12500) (hR : R = 6) (hT : T = 4) : 
  P + (P * R * T) / 100 = 13250 :=
by
  rw [hP, hR, hT]
  norm_num
  sorry

end final_amount_simple_interest_l149_149290


namespace xy_sufficient_not_necessary_l149_149021

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy_lt_zero : x * y < 0) → abs (x - y) = abs x + abs y ∧ (abs (x - y) = abs x + abs y → x * y ≥ 0) := 
by
  sorry

end xy_sufficient_not_necessary_l149_149021


namespace average_speed_is_five_l149_149319

-- Define the speeds for each segment
def swimming_speed : ℝ := 2 -- km/h
def biking_speed : ℝ := 15 -- km/h
def running_speed : ℝ := 9 -- km/h
def kayaking_speed : ℝ := 6 -- km/h

-- Define the problem to prove the average speed
theorem average_speed_is_five :
  let segments := [swimming_speed, biking_speed, running_speed, kayaking_speed]
  let harmonic_mean (speeds : List ℝ) : ℝ :=
    let n := speeds.length
    n / (speeds.foldl (fun acc s => acc + 1 / s) 0)
  harmonic_mean segments = 5 := by
  sorry

end average_speed_is_five_l149_149319


namespace average_roots_of_quadratic_l149_149665

open Real

theorem average_roots_of_quadratic (a b : ℝ) (h_eq : ∃ x1 x2 : ℝ, a * x1^2 - 2 * a * x1 + b = 0 ∧ a * x2^2 - 2 * a * x2 + b = 0):
  (b = b) → (a ≠ 0) → (h_discriminant : (2 * a)^2 - 4 * a * b ≥ 0) → (x1 + x2) / 2 = 1 :=
by
  sorry

end average_roots_of_quadratic_l149_149665


namespace prob_A_winning_l149_149878

variable (P_draw P_B : ℚ)

def P_A_winning := 1 - P_draw - P_B

theorem prob_A_winning (h1 : P_draw = 1 / 2) (h2 : P_B = 1 / 3) :
  P_A_winning P_draw P_B = 1 / 6 :=
by
  rw [P_A_winning, h1, h2]
  norm_num
  done

end prob_A_winning_l149_149878


namespace oranges_in_shop_l149_149475

-- Define the problem conditions
def ratio (M O A : ℕ) : Prop := (10 * O = 2 * M) ∧ (10 * A = 3 * M)

noncomputable def numMangoes : ℕ := 120
noncomputable def numApples : ℕ := 36

-- Statement of the problem
theorem oranges_in_shop (ratio_factor : ℕ) (h_ratio : ratio numMangoes (2 * ratio_factor) numApples) :
  (2 * ratio_factor) = 24 := by
  sorry

end oranges_in_shop_l149_149475


namespace light_year_scientific_notation_l149_149491

def sci_not_eq : Prop := 
  let x := 9500000000000
  let y := 9.5 * 10^12
  x = y

theorem light_year_scientific_notation : sci_not_eq :=
  by sorry

end light_year_scientific_notation_l149_149491


namespace difference_in_profit_l149_149384

def records := 300
def price_sammy := 4
def price_bryan_two_thirds := 6
def price_bryan_one_third := 1
def price_christine_thirty := 10
def price_christine_remaining := 3

def profit_sammy := records * price_sammy
def profit_bryan := ((records * 2 / 3) * price_bryan_two_thirds) + ((records * 1 / 3) * price_bryan_one_third)
def profit_christine := (30 * price_christine_thirty) + ((records - 30) * price_christine_remaining)

theorem difference_in_profit : 
  max profit_sammy (max profit_bryan profit_christine) - min profit_sammy (min profit_bryan profit_christine) = 190 :=
by
  sorry

end difference_in_profit_l149_149384


namespace n_gon_angle_condition_l149_149425

theorem n_gon_angle_condition (n : ℕ) (h1 : 150 * (n-1) + (30 * n - 210) = 180 * (n-2)) (h2 : 30 * n - 210 < 150) (h3 : 30 * n - 210 > 0) :
  n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 :=
by
  sorry

end n_gon_angle_condition_l149_149425


namespace problem1_problem2_l149_149554

-- Problem 1
theorem problem1 : (-2) ^ 2 + (Real.sqrt 2 - 1) ^ 0 - 1 = 4 := by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) (A : ℝ) (B : ℝ) (h1 : A = a - 1) (h2 : B = -a + 3) (h3 : A > B) : a > 2 := by
  sorry

end problem1_problem2_l149_149554


namespace heating_time_correct_l149_149934

def initial_temp : ℤ := 20

def desired_temp : ℤ := 100

def heating_rate : ℤ := 5

def time_to_heat (initial desired rate : ℤ) : ℤ :=
  (desired - initial) / rate

theorem heating_time_correct :
  time_to_heat initial_temp desired_temp heating_rate = 16 :=
by
  sorry

end heating_time_correct_l149_149934


namespace xiaoming_grandfather_age_l149_149991

def grandfather_age (x xm_diff : ℕ) :=
  xm_diff = 60 ∧ x > 7 * (x - xm_diff) ∧ x < 70

theorem xiaoming_grandfather_age (x : ℕ) (h_cond : grandfather_age x 60) : x = 69 :=
by
  sorry

end xiaoming_grandfather_age_l149_149991


namespace mixing_paint_l149_149630

theorem mixing_paint (total_parts : ℕ) (blue_parts : ℕ) (red_parts : ℕ) (white_parts : ℕ) (blue_ounces : ℕ) (max_mixture : ℕ) (ounces_per_part : ℕ) :
  total_parts = blue_parts + red_parts + white_parts →
  blue_parts = 7 →
  red_parts = 2 →
  white_parts = 1 →
  blue_ounces = 140 →
  max_mixture = 180 →
  ounces_per_part = blue_ounces / blue_parts →
  max_mixture / ounces_per_part = 9 →
  white_ounces = white_parts * ounces_per_part →
  white_ounces = 20 :=
sorry

end mixing_paint_l149_149630


namespace equivalent_fraction_l149_149458

theorem equivalent_fraction : (8 / (5 * 46)) = (0.8 / 23) := 
by sorry

end equivalent_fraction_l149_149458


namespace ellipse_standard_equation_l149_149586

theorem ellipse_standard_equation (a c : ℝ) (h1 : a^2 = 13) (h2 : c^2 = 12) :
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ 
    ((∀ x y : ℝ, (x^2 / 13 + y^2 = 1)) ∨ (∀ x y : ℝ, (x^2 + y^2 / 13 = 1)))) :=
by
  sorry

end ellipse_standard_equation_l149_149586


namespace lcm_of_2_4_5_6_l149_149538

theorem lcm_of_2_4_5_6 : Nat.lcm (Nat.lcm (Nat.lcm 2 4) 5) 6 = 60 :=
by
  sorry

end lcm_of_2_4_5_6_l149_149538


namespace handshake_count_l149_149935

def total_employees : ℕ := 50
def dept_X : ℕ := 30
def dept_Y : ℕ := 20
def handshakes_between_departments : ℕ := dept_X * dept_Y

theorem handshake_count : handshakes_between_departments = 600 :=
by
  sorry

end handshake_count_l149_149935


namespace largest_divisor_of_expression_l149_149583

theorem largest_divisor_of_expression
  (x : ℤ) (h_odd : x % 2 = 1) : 
  ∃ k : ℤ, k = 40 ∧ 40 ∣ (12 * x + 2) * (8 * x + 14) * (10 * x + 10) :=
by
  sorry

end largest_divisor_of_expression_l149_149583


namespace roots_opposite_k_eq_2_l149_149794

theorem roots_opposite_k_eq_2 (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 + x2 = 0 ∧ x1 * x2 = -1 ∧ x1 ≠ x2 ∧ x1*x1 + (k-2)*x1 - 1 = 0 ∧ x2*x2 + (k-2)*x2 - 1 = 0) → k = 2 :=
by
  sorry

end roots_opposite_k_eq_2_l149_149794


namespace solution_contains_non_zero_arrays_l149_149264

noncomputable def verify_non_zero_array (x y z w : ℝ) : Prop :=
  1 + (1 / x) + (2 * (x + 1) / (x * y)) + (3 * (x + 1) * (y + 2) / (x * y * z)) + 
  (4 * (x + 1) * (y + 2) * (z + 3) / (x * y * z * w)) = 0

theorem solution_contains_non_zero_arrays (x y z w : ℝ) (non_zero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0) :
  verify_non_zero_array x y z w ↔ 
  (x = -1 ∨ y = -2 ∨ z = -3 ∨ w = -4) ∧
  (if x = -1 then y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 else 
   if y = -2 then x ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 else 
   if z = -3 then x ≠ 0 ∧ y ≠ 0 ∧ w ≠ 0 else 
   x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :=
sorry

end solution_contains_non_zero_arrays_l149_149264


namespace construction_better_than_logistics_l149_149418

theorem construction_better_than_logistics 
  (applications_computer : ℕ := 215830)
  (applications_mechanical : ℕ := 200250)
  (applications_marketing : ℕ := 154676)
  (applications_logistics : ℕ := 74570)
  (applications_trade : ℕ := 65280)
  (recruitments_computer : ℕ := 124620)
  (recruitments_marketing : ℕ := 102935)
  (recruitments_mechanical : ℕ := 89115)
  (recruitments_construction : ℕ := 76516)
  (recruitments_chemical : ℕ := 70436) :
  applications_construction / recruitments_construction < applications_logistics / recruitments_logistics→ 
  (applications_computer / recruitments_computer < applications_chemical / recruitments_chemical) :=
sorry

end construction_better_than_logistics_l149_149418


namespace swimming_speed_l149_149210

variable (v s : ℝ)

-- Given conditions
def stream_speed : Prop := s = 0.5
def time_relationship : Prop := ∀ d : ℝ, d > 0 → d / (v - s) = 2 * (d / (v + s))

-- The theorem to prove
theorem swimming_speed (h1 : stream_speed s) (h2 : time_relationship v s) : v = 1.5 :=
  sorry

end swimming_speed_l149_149210


namespace new_dressing_contains_12_percent_vinegar_l149_149521

-- Definitions
def new_dressing_vinegar_percentage (p_vinegar q_vinegar p_fraction q_fraction : ℝ) : ℝ :=
  p_vinegar * p_fraction + q_vinegar * q_fraction

-- Conditions
def p_vinegar : ℝ := 0.30
def q_vinegar : ℝ := 0.10
def p_fraction : ℝ := 0.10
def q_fraction : ℝ := 0.90

-- The theorem to be proven
theorem new_dressing_contains_12_percent_vinegar :
  new_dressing_vinegar_percentage p_vinegar q_vinegar p_fraction q_fraction = 0.12 := 
by
  -- The proof is omitted here
  sorry

end new_dressing_contains_12_percent_vinegar_l149_149521


namespace sin_cos_sixth_power_l149_149522

theorem sin_cos_sixth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1/2) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 :=
by
  sorry

end sin_cos_sixth_power_l149_149522


namespace pool_depths_l149_149774

theorem pool_depths (J S Su : ℝ) 
  (h1 : J = 15) 
  (h2 : J = 2 * S + 5) 
  (h3 : Su = J + S - 3) : 
  S = 5 ∧ Su = 17 := 
by 
  -- proof steps go here
  sorry

end pool_depths_l149_149774


namespace sheets_borrowed_l149_149850

theorem sheets_borrowed (pages sheets borrowed remaining_sheets : ℕ) 
  (h1 : pages = 70) 
  (h2 : sheets = 35)
  (h3 : remaining_sheets = sheets - borrowed)
  (h4 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ remaining_sheets -> 2*i-1 <= pages) 
  (h5 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ remaining_sheets -> i + 1 != borrowed ∧ i <= remaining_sheets)
  (avg : ℕ) (h6 : avg = 28)
  : borrowed = 17 := by
  sorry

end sheets_borrowed_l149_149850


namespace point_of_tangency_of_circles_l149_149509

/--
Given two circles defined by the following equations:
1. \( x^2 - 2x + y^2 - 10y + 17 = 0 \)
2. \( x^2 - 8x + y^2 - 10y + 49 = 0 \)
Prove that the coordinates of the point of tangency of these circles are \( (2.5, 5) \).
-/
theorem point_of_tangency_of_circles :
  (∃ x y : ℝ, (x^2 - 2*x + y^2 - 10*y + 17 = 0) ∧ (x = 2.5) ∧ (y = 5)) ∧ 
  (∃ x' y' : ℝ, (x'^2 - 8*x' + y'^2 - 10*y' + 49 = 0) ∧ (x' = 2.5) ∧ (y' = 5)) :=
sorry

end point_of_tangency_of_circles_l149_149509


namespace ordered_pairs_count_l149_149966

theorem ordered_pairs_count : 
  (∃ s : Finset (ℕ × ℕ), (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 6) ∧ s.card = 15) :=
by
  -- The proof would go here
  sorry

end ordered_pairs_count_l149_149966


namespace sum_of_fractions_equals_three_l149_149634

-- Definitions according to the conditions
def proper_fraction (a b : ℕ) := 1 ≤ a ∧ a < b
def improper_fraction (a b : ℕ) := a ≥ b
def mixed_number (a b c : ℕ) := a + b / c

-- Constants according to the given problem
def n := 8
def d := 9
def improper_n := 9

-- Values for elements in the conditions
def largest_proper_fraction := n / d
def smallest_improper_fraction := improper_n / d
def smallest_mixed_number := 1 + 1 / d

-- Theorem statement with the correct answer
theorem sum_of_fractions_equals_three :
  largest_proper_fraction + smallest_improper_fraction + smallest_mixed_number = 3 :=
sorry

end sum_of_fractions_equals_three_l149_149634


namespace total_lunch_bill_l149_149809

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h1 : cost_hotdog = 5.36) (h2 : cost_salad = 5.10) : 
  cost_hotdog + cost_salad = 10.46 := 
by 
  sorry

end total_lunch_bill_l149_149809


namespace original_wage_l149_149304

theorem original_wage (W : ℝ) (h : 1.5 * W = 42) : W = 28 :=
by
  sorry

end original_wage_l149_149304


namespace sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l149_149227

def original_price : ℝ := 150
def discount_monday_to_wednesday : ℝ := 0.20
def tax_monday_to_wednesday : ℝ := 0.05
def discount_thursday_to_saturday : ℝ := 0.15
def tax_thursday_to_saturday : ℝ := 0.04
def discount_super_saver_sunday1 : ℝ := 0.25
def discount_super_saver_sunday2 : ℝ := 0.10
def tax_super_saver_sunday : ℝ := 0.03
def discount_festive_friday : ℝ := 0.20
def tax_festive_friday : ℝ := 0.04
def additional_discount_festive_friday : ℝ := 0.05

theorem sale_price_monday_to_wednesday : (original_price * (1 - discount_monday_to_wednesday)) * (1 + tax_monday_to_wednesday) = 126 :=
by sorry

theorem sale_price_thursday_to_saturday : (original_price * (1 - discount_thursday_to_saturday)) * (1 + tax_thursday_to_saturday) = 132.60 :=
by sorry

theorem sale_price_super_saver_sunday : ((original_price * (1 - discount_super_saver_sunday1)) * (1 - discount_super_saver_sunday2)) * (1 + tax_super_saver_sunday) = 104.29 :=
by sorry

theorem sale_price_festive_friday_selected : ((original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday)) * (1 - additional_discount_festive_friday) = 118.56 :=
by sorry

theorem sale_price_festive_friday_non_selected : (original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday) = 124.80 :=
by sorry

end sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l149_149227


namespace coin_ratio_l149_149386

theorem coin_ratio (n₁ n₅ n₂₅ : ℕ) (total_value : ℕ) 
  (h₁ : n₁ = 40) 
  (h₅ : n₅ = 40) 
  (h₂₅ : n₂₅ = 40) 
  (hv : total_value = 70) 
  (hv_calc : n₁ * 1 + n₅ * (50 / 100) + n₂₅ * (25 / 100) = total_value) : 
  n₁ = n₅ ∧ n₁ = n₂₅ :=
by
  sorry

end coin_ratio_l149_149386


namespace no_integral_points_on_AB_l149_149686

theorem no_integral_points_on_AB (k m n : ℤ) (h1: ((m^3 - m)^2 + (n^3 - n)^2 > (3*k + 1)^2)) :
  ¬ ∃ (x y : ℤ), (m^3 - m) * x + (n^3 - n) * y = (3*k + 1)^2 :=
by {
  sorry
}

end no_integral_points_on_AB_l149_149686


namespace sets_are_equal_l149_149636

def int : Type := ℤ  -- Redefine integer as ℤ for clarity

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem sets_are_equal : SetA = SetB := by
  -- implement the proof here
  sorry

end sets_are_equal_l149_149636


namespace positive_root_of_cubic_eq_l149_149882

theorem positive_root_of_cubic_eq : ∃ (x : ℝ), x > 0 ∧ x^3 - 3 * x^2 - x - Real.sqrt 2 = 0 ∧ x = 2 + Real.sqrt 2 := by
  sorry

end positive_root_of_cubic_eq_l149_149882


namespace composite_numbers_with_same_main_divisors_are_equal_l149_149735

theorem composite_numbers_with_same_main_divisors_are_equal (a b : ℕ) 
  (h_a_not_prime : ¬ Prime a)
  (h_b_not_prime : ¬ Prime b)
  (h_a_comp : 1 < a ∧ ∃ p, p ∣ a ∧ p ≠ a)
  (h_b_comp : 1 < b ∧ ∃ p, p ∣ b ∧ p ≠ b)
  (main_divisors : {d : ℕ // d ∣ a ∧ d ≠ a} = {d : ℕ // d ∣ b ∧ d ≠ b}) :
  a = b := 
sorry

end composite_numbers_with_same_main_divisors_are_equal_l149_149735


namespace set_non_neg_even_set_primes_up_to_10_eq_sol_set_l149_149434

noncomputable def non_neg_even (x : ℕ) : Prop := x % 2 = 0 ∧ x ≤ 10
def primes_up_to_10 (x : ℕ) : Prop := Nat.Prime x ∧ x ≤ 10
def eq_sol (x : ℤ) : Prop := x^2 + 2*x - 15 = 0

theorem set_non_neg_even :
  {x : ℕ | non_neg_even x} = {0, 2, 4, 6, 8, 10} := by
  sorry

theorem set_primes_up_to_10 :
  {x : ℕ | primes_up_to_10 x} = {2, 3, 5, 7} := by
  sorry

theorem eq_sol_set :
  {x : ℤ | eq_sol x} = {-5, 3} := by
  sorry

end set_non_neg_even_set_primes_up_to_10_eq_sol_set_l149_149434


namespace determine_values_l149_149885

theorem determine_values (A B : ℚ) :
  (A + B = 4) ∧ (2 * A - 7 * B = 3) →
  A = 31 / 9 ∧ B = 5 / 9 :=
by
  sorry

end determine_values_l149_149885


namespace factorial_divisibility_l149_149276

theorem factorial_divisibility 
  (n k : ℕ) 
  (p : ℕ) 
  [hp : Fact (Nat.Prime p)] 
  (h1 : 0 < n) 
  (h2 : 0 < k) 
  (h3 : p ^ k ∣ n!) : 
  (p! ^ k ∣ n!) :=
sorry

end factorial_divisibility_l149_149276


namespace probability_blue_then_red_l149_149676

/--
A box contains 15 balls, of which 5 are blue and 10 are red.
Two balls are drawn sequentially from the box without returning the first ball to the box.
Prove that the probability that the first ball drawn is blue and the second ball is red is 5 / 21.
-/
theorem probability_blue_then_red :
  let total_balls := 15
  let blue_balls := 5
  let red_balls := 10
  let first_is_blue := (blue_balls : ℚ) / total_balls
  let second_is_red_given_blue := (red_balls : ℚ) / (total_balls - 1)
  first_is_blue * second_is_red_given_blue = 5 / 21 := by
  sorry

end probability_blue_then_red_l149_149676


namespace paul_lives_on_story_5_l149_149968

/-- 
Given:
1. Each story is 10 feet tall.
2. Paul makes 3 trips out from and back to his apartment each day.
3. Over a week (7 days), he travels 2100 feet vertically in total.

Prove that the story on which Paul lives \( S \) is 5.
-/
theorem paul_lives_on_story_5 (height_per_story : ℕ)
  (trips_per_day : ℕ)
  (number_of_days : ℕ)
  (total_feet_travelled : ℕ)
  (S : ℕ) :
  height_per_story = 10 → 
  trips_per_day = 3 → 
  number_of_days = 7 → 
  total_feet_travelled = 2100 → 
  2 * height_per_story * trips_per_day * number_of_days * S = total_feet_travelled → 
  S = 5 :=
by
  intros
  sorry

end paul_lives_on_story_5_l149_149968


namespace distinct_products_count_is_26_l149_149487

open Finset

def set_numbers : Finset ℕ := {2, 3, 5, 7, 11}

def distinct_products_count (s : Finset ℕ) : ℕ :=
  let pairs := s.powerset.filter (λ t => 2 ≤ t.card)
  pairs.card

theorem distinct_products_count_is_26 : distinct_products_count set_numbers = 26 := by
  sorry

end distinct_products_count_is_26_l149_149487


namespace ellipse_hyperbola_tangent_n_value_l149_149045

theorem ellipse_hyperbola_tangent_n_value :
  (∃ n : ℝ, (∀ x y : ℝ, 4 * x^2 + y^2 = 4 ∧ x^2 - n * (y - 1)^2 = 1) ↔ n = 3 / 2) :=
by
  sorry

end ellipse_hyperbola_tangent_n_value_l149_149045


namespace natural_number_triplets_l149_149799

theorem natural_number_triplets (x y z : ℕ) : 
  3^x + 4^y = 5^z → 
  (x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by 
  sorry

end natural_number_triplets_l149_149799


namespace min_max_x_l149_149980

theorem min_max_x (n : ℕ) (hn : 0 < n) (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = n * x + n * y) : 
  n + 1 ≤ x ∧ x ≤ n * (n + 1) :=
by {
  sorry  -- Proof goes here
}

end min_max_x_l149_149980


namespace neg_p_equiv_l149_149140

theorem neg_p_equiv :
  (¬ (∀ x : ℝ, x > 0 → x - Real.log x > 0)) ↔ (∃ x_0 : ℝ, x_0 > 0 ∧ x_0 - Real.log x_0 ≤ 0) :=
by
  sorry

end neg_p_equiv_l149_149140


namespace div_condition_for_lcm_l149_149975

theorem div_condition_for_lcm (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h : Nat.lcm (x + 2) (y + 2) - Nat.lcm (x + 1) (y + 1) = Nat.lcm (x + 1) (y + 1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x :=
sorry

end div_condition_for_lcm_l149_149975


namespace lunks_needed_for_20_apples_l149_149911

-- Define the conditions as given in the problem
def lunks_to_kunks (lunks : ℤ) : ℤ := (4 * lunks) / 7
def kunks_to_apples (kunks : ℤ) : ℤ := (5 * kunks) / 3

-- Define the target function to calculate the number of lunks needed for given apples
def apples_to_lunks (apples : ℤ) : ℤ := 
  let kunks := (3 * apples) / 5
  let lunks := (7 * kunks) / 4
  lunks

-- Prove the given problem
theorem lunks_needed_for_20_apples : apples_to_lunks 20 = 21 := by
  sorry

end lunks_needed_for_20_apples_l149_149911


namespace mod_inverse_17_1200_l149_149605

theorem mod_inverse_17_1200 : ∃ x : ℕ, x < 1200 ∧ 17 * x % 1200 = 1 := 
by
  use 353
  sorry

end mod_inverse_17_1200_l149_149605


namespace biking_distance_l149_149385

/-- Mathematical equivalent proof problem for the distance biked -/
theorem biking_distance
  (x t d : ℕ)
  (h1 : d = (x + 1) * (3 * t / 4))
  (h2 : d = (x - 1) * (t + 3)) :
  d = 36 :=
by
  -- The proof goes here
  sorry

end biking_distance_l149_149385


namespace rational_solutions_k_values_l149_149270

theorem rational_solutions_k_values (k : ℕ) (h₁ : k > 0) 
    (h₂ : ∃ (m : ℤ), 900 - 4 * (k:ℤ)^2 = m^2) : k = 9 ∨ k = 15 := 
by
  sorry

end rational_solutions_k_values_l149_149270


namespace sequence_a113_l149_149168

theorem sequence_a113 {a : ℕ → ℝ} 
  (h1 : ∀ n, a n > 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n, (a (n+1))^2 + (a n)^2 = 2 * n * ((a (n+1))^2 - (a n)^2)) :
  a 113 = 15 :=
sorry

end sequence_a113_l149_149168


namespace maximum_value_of_d_l149_149477

theorem maximum_value_of_d 
  (d e : ℕ) 
  (h1 : 0 ≤ d ∧ d < 10) 
  (h2: 0 ≤ e ∧ e < 10) 
  (h3 : (18 + d + e) % 3 = 0) 
  (h4 : (15 - (d + e)) % 11 = 0) 
  : d ≤ 0 := 
sorry

end maximum_value_of_d_l149_149477


namespace hours_between_dates_not_thirteen_l149_149450

def total_hours (start_date: ℕ × ℕ × ℕ × ℕ) (end_date: ℕ × ℕ × ℕ × ℕ) (days_in_dec: ℕ) : ℕ :=
  let (start_year, start_month, start_day, start_hour) := start_date
  let (end_year, end_month, end_day, end_hour) := end_date
  (days_in_dec - start_day) * 24 - start_hour + end_day * 24 + end_hour

theorem hours_between_dates_not_thirteen :
  let start_date := (2015, 12, 30, 23)
  let end_date := (2016, 1, 1, 12)
  let days_in_dec := 31
  total_hours start_date end_date days_in_dec ≠ 13 :=
by
  sorry

end hours_between_dates_not_thirteen_l149_149450


namespace fans_who_received_all_three_l149_149424

theorem fans_who_received_all_three (n : ℕ) :
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ n)) ∧
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ 8)) :=
by
  sorry

end fans_who_received_all_three_l149_149424


namespace trees_in_one_row_l149_149611

variable (total_trees_cleaned : ℕ)
variable (trees_per_row : ℕ)

theorem trees_in_one_row (h1 : total_trees_cleaned = 20) (h2 : trees_per_row = 5) :
  (total_trees_cleaned / trees_per_row) = 4 :=
by
  sorry

end trees_in_one_row_l149_149611


namespace f_g_g_f_l149_149331

noncomputable def f (x: ℝ) := 1 - 2 * x
noncomputable def g (x: ℝ) := x^2 + 3

theorem f_g (x : ℝ) : f (g x) = -2 * x^2 - 5 :=
by
  sorry

theorem g_f (x : ℝ) : g (f x) = 4 * x^2 - 4 * x + 4 :=
by
  sorry

end f_g_g_f_l149_149331


namespace number_of_senior_citizen_tickets_l149_149483

theorem number_of_senior_citizen_tickets 
    (A S : ℕ)
    (h1 : A + S = 529)
    (h2 : 25 * A + 15 * S = 9745) 
    : S = 348 := 
by
  sorry

end number_of_senior_citizen_tickets_l149_149483


namespace intersection_M_N_l149_149083

def M : Set ℕ := {1, 2, 4, 8}
def N : Set ℕ := {x | x ∣ 4 ∧ 0 < x}

theorem intersection_M_N :
  M ∩ N = {1, 2, 4} :=
sorry

end intersection_M_N_l149_149083


namespace prob_students_both_days_l149_149394

def num_scenarios (students : ℕ) (choices : ℕ) : ℕ :=
  choices ^ students

def scenarios_sat_sun (total_scenarios : ℕ) (both_days_empty : ℕ) : ℕ :=
  total_scenarios - both_days_empty

theorem prob_students_both_days :
  let students := 3
  let choices := 2
  let total_scenarios := num_scenarios students choices
  let both_days_empty := 2 -- When all choose Saturday or all choose Sunday
  let scenarios_both := scenarios_sat_sun total_scenarios both_days_empty
  let probability := scenarios_both / total_scenarios
  probability = 3 / 4 :=
by
  sorry

end prob_students_both_days_l149_149394


namespace part1_part2_part3_l149_149933

-- Part 1
theorem part1 (x : ℝ) :
  (2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4) ↔ x = 2 :=
sorry

-- Part 2
theorem part2 (x : ℤ) :
  (x - 1 / 4 < 1 ∧ 4 + 2 * x > -7 * x + 5) ↔ x = 1 :=
sorry

-- Part 3
theorem part3 (m : ℝ) :
  (∀ x, m < x ∧ x <= m + 2 → (x = 3 ∨ x = 2)) ↔ 1 ≤ m ∧ m < 2 :=
sorry

end part1_part2_part3_l149_149933


namespace first_discount_is_20_percent_l149_149987

-- Define the problem parameters
def original_price : ℝ := 200
def final_price : ℝ := 152
def second_discount : ℝ := 0.05

-- Define the function to compute the price after two discounts
def price_after_discounts (first_discount : ℝ) : ℝ := 
  original_price * (1 - first_discount) * (1 - second_discount)

-- Define the statement that we need to prove
theorem first_discount_is_20_percent : 
  ∃ (first_discount : ℝ), price_after_discounts first_discount = final_price ∧ first_discount = 0.20 :=
by
  sorry

end first_discount_is_20_percent_l149_149987


namespace isosceles_triangle_perimeter_l149_149320

theorem isosceles_triangle_perimeter 
    (a b : ℕ) (h_iso : a = 3 ∨ a = 5) (h_other : b = 3 ∨ b = 5) 
    (h_distinct : a ≠ b) : 
    ∃ p : ℕ, p = (3 + 3 + 5) ∨ p = (5 + 5 + 3) :=
by
  sorry

end isosceles_triangle_perimeter_l149_149320


namespace son_age_is_10_l149_149070

-- Define the conditions
variables (S F : ℕ)
axiom condition1 : F = S + 30
axiom condition2 : F + 5 = 3 * (S + 5)

-- State the theorem to prove the son's age
theorem son_age_is_10 : S = 10 :=
by
  sorry

end son_age_is_10_l149_149070


namespace math_problem_l149_149049

variable {f : ℝ → ℝ}

theorem math_problem (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                     (h2 : ∀ x : ℝ, x > 0 → f x > 0)
                     (h3 : f 1 = 2) :
                     f 0 = 0 ∧
                     (∀ x : ℝ, f (-x) = -f x) ∧
                     (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ∧
                     (∃ a : ℝ, f (2 - a) = 6 ∧ a = -1) := 
by
  sorry

end math_problem_l149_149049


namespace sarah_apples_calc_l149_149788

variable (brother_apples : ℕ)
variable (sarah_apples : ℕ)
variable (multiplier : ℕ)

theorem sarah_apples_calc
  (h1 : brother_apples = 9)
  (h2 : multiplier = 5)
  (h3 : sarah_apples = multiplier * brother_apples) : sarah_apples = 45 := by
  sorry

end sarah_apples_calc_l149_149788


namespace modified_full_house_probability_l149_149755

def total_choices : ℕ := Nat.choose 52 6

def ways_rank1 : ℕ := 13
def ways_3_cards : ℕ := Nat.choose 4 3
def ways_rank2 : ℕ := 12
def ways_2_cards : ℕ := Nat.choose 4 2
def ways_additional_card : ℕ := 11 * 4

def ways_modified_full_house : ℕ := ways_rank1 * ways_3_cards * ways_rank2 * ways_2_cards * ways_additional_card

def probability_modified_full_house : ℚ := ways_modified_full_house / total_choices

theorem modified_full_house_probability : probability_modified_full_house = 24 / 2977 := 
by sorry

end modified_full_house_probability_l149_149755


namespace max_discardable_grapes_l149_149874

theorem max_discardable_grapes (n : ℕ) (k : ℕ) (h : k = 8) : 
  ∃ m : ℕ, m < k ∧ (∀ q : ℕ, q * k + m = n) ∧ m = 7 :=
by
  sorry

end max_discardable_grapes_l149_149874


namespace average_marks_l149_149732

theorem average_marks (D I T : ℕ) 
  (hD : D = 90)
  (hI : I = (3 * D) / 5)
  (hT : T = 2 * I) : 
  (D + I + T) / 3 = 84 :=
by
  sorry

end average_marks_l149_149732


namespace base6_sub_base9_to_base10_l149_149664

theorem base6_sub_base9_to_base10 :
  (3 * 6^2 + 2 * 6^1 + 5 * 6^0) - (2 * 9^2 + 1 * 9^1 + 5 * 9^0) = -51 :=
by
  sorry

end base6_sub_base9_to_base10_l149_149664


namespace fish_caught_300_l149_149100

def fish_caught_at_dawn (F : ℕ) : Prop :=
  (3 * F / 5) = 180

theorem fish_caught_300 : ∃ F, fish_caught_at_dawn F ∧ F = 300 := 
by 
  use 300 
  have h1 : 3 * 300 / 5 = 180 := by norm_num 
  exact ⟨h1, rfl⟩

end fish_caught_300_l149_149100


namespace faster_by_airplane_l149_149008

theorem faster_by_airplane : 
  let driving_time := 3 * 60 + 15 
  let airport_drive := 10
  let wait_to_board := 20
  let flight_duration := driving_time / 3
  let exit_plane := 10
  driving_time - (airport_drive + wait_to_board + flight_duration + exit_plane) = 90 := 
by
  let driving_time : ℕ := 3 * 60 + 15
  let airport_drive : ℕ := 10
  let wait_to_board : ℕ := 20
  let flight_duration : ℕ := driving_time / 3
  let exit_plane : ℕ := 10
  have h1 : driving_time = 195 := rfl
  have h2 : flight_duration = 65 := by norm_num [h1]
  have h3 : 195 - (10 + 20 + 65 + 10) = 195 - 105 := by norm_num
  have h4 : 195 - 105 = 90 := by norm_num
  exact h4

end faster_by_airplane_l149_149008


namespace replace_asterisk_l149_149090

theorem replace_asterisk :
  ∃ x : ℤ, (x / 21) * (63 / 189) = 1 ∧ x = 63 := sorry

end replace_asterisk_l149_149090


namespace bottles_left_l149_149834

variable (initial_bottles : ℕ) (jason_bottles : ℕ) (harry_bottles : ℕ)

theorem bottles_left (h1 : initial_bottles = 35) (h2 : jason_bottles = 5) (h3 : harry_bottles = 6) :
    initial_bottles - (jason_bottles + harry_bottles) = 24 := by
  sorry

end bottles_left_l149_149834


namespace denomination_of_checks_l149_149190

-- Definitions based on the conditions.
def total_checks := 30
def total_worth := 1800
def checks_spent := 24
def average_remaining := 100

-- Statement to be proven.
theorem denomination_of_checks :
  ∃ x : ℝ, (total_checks - checks_spent) * average_remaining + checks_spent * x = total_worth ∧ x = 40 :=
by
  sorry

end denomination_of_checks_l149_149190


namespace work_completion_time_l149_149240

noncomputable def work_done (hours : ℕ) (a_rate : ℚ) (b_rate : ℚ) : ℚ :=
  if hours % 2 = 0 then (hours / 2) * (a_rate + b_rate)
  else ((hours - 1) / 2) * (a_rate + b_rate) + a_rate

theorem work_completion_time :
  let a_rate := 1/4
  let b_rate := 1/12
  (∃ t, work_done t a_rate b_rate = 1) → t = 6 := 
by
  intro h
  sorry

end work_completion_time_l149_149240


namespace sum_of_other_two_angles_is_108_l149_149741

theorem sum_of_other_two_angles_is_108 (A B C : Type) (angleA angleB angleC : ℝ) 
  (h_angle_sum : angleA + angleB + angleC = 180) (h_angleB : angleB = 72) :
  angleA + angleC = 108 := 
by
  sorry

end sum_of_other_two_angles_is_108_l149_149741


namespace pencils_ordered_l149_149594

theorem pencils_ordered (pencils_per_student : ℕ) (number_of_students : ℕ) (total_pencils : ℕ) :
  pencils_per_student = 3 →
  number_of_students = 65 →
  total_pencils = pencils_per_student * number_of_students →
  total_pencils = 195 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end pencils_ordered_l149_149594


namespace total_cakes_needed_l149_149823

theorem total_cakes_needed (C : ℕ) (h : C / 4 - C / 12 = 10) : C = 60 := by
  sorry

end total_cakes_needed_l149_149823


namespace exists_constant_not_geometric_l149_149659

-- Definitions for constant and geometric sequences
def is_constant_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, seq n = c

def is_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, seq (n + 1) = r * seq n

-- The negation problem statement
theorem exists_constant_not_geometric :
  ∃ seq : ℕ → ℝ, is_constant_sequence seq ∧ ¬is_geometric_sequence seq :=
sorry

end exists_constant_not_geometric_l149_149659


namespace combination_exists_l149_149415

theorem combination_exists 
  (S T Ti : ℝ) (x y z : ℝ)
  (h : 3 * S + 4 * T + 2 * Ti = 40) :
  ∃ x y z : ℝ, x * S + y * T + z * Ti = 60 :=
sorry

end combination_exists_l149_149415


namespace rhombus_area_l149_149864

theorem rhombus_area (x y : ℝ) (h : |x - 1| + |y - 1| = 1) : 
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end rhombus_area_l149_149864


namespace second_candidate_marks_l149_149810

variable (T : ℝ) (pass_mark : ℝ := 160)

-- Conditions
def condition1 : Prop := 0.20 * T + 40 = pass_mark
def condition2 : Prop := 0.30 * T - pass_mark > 0 

-- The statement we want to prove
theorem second_candidate_marks (h1 : condition1 T) (h2 : condition2 T) : 
  (0.30 * T - pass_mark = 20) :=
by 
  -- Skipping proof steps as per the guidelines
  sorry

end second_candidate_marks_l149_149810


namespace prob1_prob2_l149_149540

-- Definitions and conditions for Problem 1
def U : Set ℝ := {x | x ≤ 4}
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof Problem 1: Equivalent Lean proof statement
theorem prob1 : (U \ A) ∩ B = {-3, -2, 3} := by
  sorry

-- Definitions and conditions for Problem 2
def tan_alpha_eq_3 (α : ℝ) : Prop := Real.tan α = 3

-- Proof Problem 2: Equivalent Lean proof statement
theorem prob2 (α : ℝ) (h : tan_alpha_eq_3 α) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 ∧
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = -4 / 5 := by
  sorry

end prob1_prob2_l149_149540


namespace sam_walking_speed_l149_149267

variable (s : ℝ)
variable (t : ℝ)
variable (fred_speed : ℝ := 2)
variable (sam_distance : ℝ := 25)
variable (total_distance : ℝ := 35)

theorem sam_walking_speed :
  (total_distance - sam_distance) = fred_speed * t ∧
  sam_distance = s * t →
  s = 5 := 
by
  intros
  sorry

end sam_walking_speed_l149_149267


namespace product_of_b_l149_149925

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

noncomputable def g_inv (b : ℝ) (y : ℝ) : ℝ := (y + 4) / 3

theorem product_of_b (b : ℝ) :
  g b 3 = g_inv b (b + 2) → b = 3 := 
by
  sorry

end product_of_b_l149_149925


namespace breadth_halved_of_percentage_change_area_l149_149078

theorem breadth_halved_of_percentage_change_area {L B B' : ℝ} (h : 0 < L ∧ 0 < B) 
  (h1 : L / 2 * B' = 0.5 * (L * B)) : B' = 0.5 * B :=
sorry

end breadth_halved_of_percentage_change_area_l149_149078


namespace math_problem_l149_149557

theorem math_problem : 8 / 4 - 3 - 10 + 3 * 7 = 10 := by
  sorry

end math_problem_l149_149557


namespace sum_of_interior_edges_l149_149525

theorem sum_of_interior_edges (frame_width : ℕ) (frame_area : ℕ) (outer_edge : ℕ) 
  (H1 : frame_width = 2) (H2 : frame_area = 32) (H3 : outer_edge = 7) : 
  2 * (outer_edge - 2 * frame_width) + 2 * (x : ℕ) = 8 :=
by
  sorry

end sum_of_interior_edges_l149_149525


namespace sequence_23rd_term_is_45_l149_149390

def sequence_game (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * n - 1 else 2 * n + 1

theorem sequence_23rd_term_is_45 :
  sequence_game 23 = 45 :=
by
  -- Proving the 23rd term in the sequence as given by the game rules
  sorry

end sequence_23rd_term_is_45_l149_149390


namespace shift_parabola_upwards_l149_149122

theorem shift_parabola_upwards (y x : ℝ) (h : y = x^2) : y + 5 = (x^2 + 5) := by 
  sorry

end shift_parabola_upwards_l149_149122


namespace contrapositive_ex_l149_149642

theorem contrapositive_ex (x y : ℝ)
  (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) :
  ¬ (x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0 :=
by
  sorry

end contrapositive_ex_l149_149642


namespace shortest_total_distance_piglet_by_noon_l149_149153

-- Define the distances
def distance_fs : ℕ := 1300  -- Distance through the forest (Piglet to Winnie-the-Pooh)
def distance_pr : ℕ := 600   -- Distance (Piglet to Rabbit)
def distance_rw : ℕ := 500   -- Distance (Rabbit to Winnie-the-Pooh)

-- Define the total distance via Rabbit and via forest
def total_distance_rabbit_path : ℕ := distance_pr + distance_rw + distance_rw
def total_distance_forest_path : ℕ := distance_fs + distance_rw

-- Prove that shortest distance Piglet covers by noon
theorem shortest_total_distance_piglet_by_noon : 
  min (total_distance_forest_path) (total_distance_rabbit_path) = 1600 := by
  sorry

end shortest_total_distance_piglet_by_noon_l149_149153


namespace soda_cost_proof_l149_149608

theorem soda_cost_proof (b s : ℤ) (h1 : 4 * b + 3 * s = 440) (h2 : 3 * b + 2 * s = 310) : s = 80 :=
by
  sorry

end soda_cost_proof_l149_149608


namespace simplify_fraction_part1_simplify_fraction_part2_l149_149566

-- Part 1
theorem simplify_fraction_part1 (x : ℝ) (h1 : x ≠ -2) :
  (x^2 / (x + 2)) + ((4 * x + 4) / (x + 2)) = x + 2 :=
sorry

-- Part 2
theorem simplify_fraction_part2 (x : ℝ) (h1 : x ≠ 1) :
  (x^2 / ((x - 1)^2)) / ((1 - 2 * x) / (x - 1) - (x - 1)) = -1 / (x - 1) :=
sorry

end simplify_fraction_part1_simplify_fraction_part2_l149_149566


namespace combined_time_to_finish_cereal_l149_149142

theorem combined_time_to_finish_cereal : 
  let rate_fat := 1 / 15
  let rate_thin := 1 / 45
  let combined_rate := rate_fat + rate_thin
  let time_needed := 4 / combined_rate
  time_needed = 45 := 
by 
  sorry

end combined_time_to_finish_cereal_l149_149142


namespace probability_heads_at_least_9_l149_149219

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l149_149219


namespace sale_in_third_month_l149_149749

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (avg_sale_months : ℕ) 
  (total_sales : ℕ)
  (h1 : sale1 = 6435) 
  (h2 : sale2 = 6927) 
  (h4 : sale4 = 7230) 
  (h5 : sale5 = 6562) 
  (h6 : sale6 = 7991) 
  (h_avg : avg_sale_months = 7000) 
  (h_total : total_sales = 6 * avg_sale_months) 
  : (total_sales - (sale1 + sale2 + sale4 + sale5 + sale6)) = 6855 :=
by
  have sales_sum := sale1 + sale2 + sale4 + sale5 + sale6
  have required_sales := total_sales - sales_sum
  sorry

end sale_in_third_month_l149_149749


namespace tangent_circles_t_value_l149_149861

theorem tangent_circles_t_value (t : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = t^2 → x^2 + y^2 + 6 * x - 8 * y + 24 = 0 → dist (0, 0) (-3, 4) = t + 1) → t = 4 :=
by
  sorry

end tangent_circles_t_value_l149_149861


namespace profit_correct_l149_149493

-- Conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def sets : ℕ := 500

-- Definitions used in the problem
def manufacturing_cost : ℕ := initial_outlay + (sets * cost_per_set)
def revenue : ℕ := sets * selling_price_per_set
def profit : ℕ := revenue - manufacturing_cost

-- The theorem statement
theorem profit_correct : profit = 5000 := by
  sorry

end profit_correct_l149_149493


namespace drawings_per_neighbor_l149_149306

theorem drawings_per_neighbor (n_neighbors animals : ℕ) (h1 : n_neighbors = 6) (h2 : animals = 54) : animals / n_neighbors = 9 :=
by
  sorry

end drawings_per_neighbor_l149_149306


namespace max_correct_answers_l149_149106

theorem max_correct_answers :
  ∀ (a b c : ℕ), a + b + c = 60 ∧ 4 * a - c = 112 → a ≤ 34 :=
by
  sorry

end max_correct_answers_l149_149106


namespace domain_of_function_l149_149302

theorem domain_of_function :
  {x : ℝ | 3 - x > 0 ∧ x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end domain_of_function_l149_149302


namespace seven_digit_number_insertion_l149_149456

theorem seven_digit_number_insertion (num : ℕ) (h : num = 52115) : (∃ (count : ℕ), count = 21) :=
by 
  sorry

end seven_digit_number_insertion_l149_149456


namespace find_px_value_l149_149027

noncomputable def p (a b c x : ℤ) := a * x^2 + b * x + c

theorem find_px_value {a b c : ℤ} 
  (h1 : p a b c 2 = 2) 
  (h2 : p a b c (-2) = -2) 
  (h3 : p a b c 9 = 3) 
  (h : a = -2 / 11) 
  (h4 : b = 1)
  (h5 : c = 8 / 11) :
  p a b c 14 = -230 / 11 :=
by
  sorry

end find_px_value_l149_149027


namespace fraction_of_selected_films_in_color_l149_149576

variables (x y : ℕ)

theorem fraction_of_selected_films_in_color (B C : ℕ) (e : ℚ)
  (h1 : B = 20 * x)
  (h2 : C = 6 * y)
  (h3 : e = (6 * y : ℚ) / (((y / 5 : ℚ) + 6 * y))) :
  e = 30 / 31 :=
by {
  sorry
}

end fraction_of_selected_films_in_color_l149_149576


namespace rajeev_share_of_profit_l149_149662

open Nat

theorem rajeev_share_of_profit (profit : ℕ) (ramesh_xyz_ratio1 ramesh_xyz_ratio2 xyz_rajeev_ratio1 xyz_rajeev_ratio2 : ℕ) (rajeev_ratio_part : ℕ) (total_parts : ℕ) (individual_part_value : ℕ) :
  profit = 36000 →
  ramesh_xyz_ratio1 = 5 →
  ramesh_xyz_ratio2 = 4 →
  xyz_rajeev_ratio1 = 8 →
  xyz_rajeev_ratio2 = 9 →
  rajeev_ratio_part = 9 →
  total_parts = ramesh_xyz_ratio1 * (xyz_rajeev_ratio1 / ramesh_xyz_ratio2) + xyz_rajeev_ratio1 + xyz_rajeev_ratio2 →
  individual_part_value = profit / total_parts →
  rajeev_ratio_part * individual_part_value = 12000 := 
sorry

end rajeev_share_of_profit_l149_149662


namespace perfect_number_divisibility_l149_149704

theorem perfect_number_divisibility (P : ℕ) (h1 : P > 28) (h2 : Nat.Perfect P) (h3 : 7 ∣ P) : 49 ∣ P := 
sorry

end perfect_number_divisibility_l149_149704


namespace proof_mn_proof_expr_l149_149615

variables (m n : ℚ)
-- Conditions
def condition1 : Prop := (m + n)^2 = 9
def condition2 : Prop := (m - n)^2 = 1

-- Expected results
def expected_mn : ℚ := 2
def expected_expr : ℚ := 3

-- The theorem to be proved
theorem proof_mn : condition1 m n → condition2 m n → m * n = expected_mn :=
by
  sorry

theorem proof_expr : condition1 m n → condition2 m n → m^2 + n^2 - m * n = expected_expr :=
by
  sorry

end proof_mn_proof_expr_l149_149615


namespace c_payment_l149_149960

theorem c_payment 
  (A_rate : ℝ) (B_rate : ℝ) (days : ℝ) (total_payment : ℝ) (C_fraction : ℝ) 
  (hA : A_rate = 1 / 6) 
  (hB : B_rate = 1 / 8) 
  (hdays : days = 3) 
  (hpayment : total_payment = 3200)
  (hC_fraction : C_fraction = 1 / 8) :
  total_payment * C_fraction = 400 :=
by {
  -- The proof would go here
  sorry
}

end c_payment_l149_149960


namespace joint_purchases_popular_l149_149740

-- Define the conditions stating what makes joint purchases feasible
structure Conditions where
  cost_saving : Prop  -- Joint purchases allow significant cost savings.
  shared_overhead : Prop  -- Overhead costs are distributed among all members.
  collective_quality_assessment : Prop  -- Enhanced quality assessment via collective feedback.
  community_trust : Prop  -- Trust within the community encourages honest feedback.

-- Define the proposition stating the popularity of joint purchases
theorem joint_purchases_popular (cond : Conditions) : 
  cond.cost_saving ∧ cond.shared_overhead ∧ cond.collective_quality_assessment ∧ cond.community_trust → 
  Prop := 
by 
  intro h
  sorry

end joint_purchases_popular_l149_149740


namespace product_of_divisors_sum_l149_149786

theorem product_of_divisors_sum :
  ∃ (a b c : ℕ), (a ∣ 11^3) ∧ (b ∣ 11^3) ∧ (c ∣ 11^3) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a * b * c = 11^3) ∧ (a + b + c = 133) :=
sorry

end product_of_divisors_sum_l149_149786


namespace travel_cost_is_correct_l149_149035

-- Definitions of the conditions
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 60
def road_width : ℝ := 15
def cost_per_sq_m : ℝ := 3

-- Areas of individual roads
def area_road_length := road_width * lawn_breadth
def area_road_breadth := road_width * lawn_length
def intersection_area := road_width * road_width

-- Adjusted area for roads discounting intersection area
def total_area_roads := area_road_length + area_road_breadth - intersection_area

-- Total cost of traveling the roads
def total_cost := total_area_roads * cost_per_sq_m

theorem travel_cost_is_correct : total_cost = 5625 := by
  sorry

end travel_cost_is_correct_l149_149035


namespace Robie_chocolates_left_l149_149894

def initial_bags : ℕ := 3
def given_away : ℕ := 2
def additional_bags : ℕ := 3

theorem Robie_chocolates_left : (initial_bags - given_away) + additional_bags = 4 :=
by
  sorry

end Robie_chocolates_left_l149_149894


namespace simplify_expression_l149_149854

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : a ≠ 1)
variable (h3 : 0 < b)

theorem simplify_expression : a ^ Real.log (1 / b ^ Real.log a) = 1 / b ^ (Real.log a) ^ 2 :=
by
  sorry

end simplify_expression_l149_149854


namespace find_F_l149_149601

theorem find_F (F C : ℝ) (h1 : C = 30) (h2 : C = (5 / 9) * (F - 30)) : F = 84 := by
  sorry

end find_F_l149_149601


namespace balance_blue_balls_l149_149025

noncomputable def weight_balance (G B Y W : ℝ) : ℝ :=
  3 * G + 3 * Y + 5 * W

theorem balance_blue_balls (G B Y W : ℝ)
  (hG : G = 2 * B)
  (hY : Y = 2 * B)
  (hW : W = (5 / 3) * B) :
  weight_balance G B Y W = (61 / 3) * B :=
by
  sorry

end balance_blue_balls_l149_149025


namespace part1_part2_l149_149463

noncomputable def seq (n : ℕ) : ℚ :=
  match n with
  | 0     => 0  -- since there is no a_0 (we use ℕ*), we set it to 0
  | 1     => 1/3
  | n + 1 => seq n + (seq n) ^ 2 / (n : ℚ) ^ 2

theorem part1 (n : ℕ) (h : 0 < n) :
  seq n < seq (n + 1) ∧ seq (n + 1) < 1 :=
sorry

theorem part2 (n : ℕ) (h : 0 < n) :
  seq n > 1/2 - 1/(4 * n) :=
sorry

end part1_part2_l149_149463


namespace no_positive_int_solutions_l149_149065

theorem no_positive_int_solutions
  (x y z t : ℕ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (ht : 0 < t)
  (h1 : x^2 + 2 * y^2 = z^2)
  (h2 : 2 * x^2 + y^2 = t^2) : false :=
by
  sorry

end no_positive_int_solutions_l149_149065


namespace quadratic_roots_algebraic_expression_value_l149_149526

-- Part 1: Proof statement for the roots of the quadratic equation
theorem quadratic_roots : (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 7 ∧ x₂ = 2 - Real.sqrt 7 ∧ (∀ x : ℝ, x^2 - 4 * x - 3 = 0 → x = x₁ ∨ x = x₂)) :=
by
  sorry

-- Part 2: Proof statement for the algebraic expression value
theorem algebraic_expression_value (a : ℝ) (h : a^2 = 3 * a + 10) :
  (a + 4) * (a - 4) - 3 * (a - 1) = -3 :=
by
  sorry

end quadratic_roots_algebraic_expression_value_l149_149526


namespace find_xy_l149_149000

theorem find_xy (x y : ℝ) :
  0.75 * x - 0.40 * y = 0.20 * 422.50 →
  0.30 * x + 0.50 * y = 0.35 * 530 →
  x = 52.816 ∧ y = -112.222 :=
by
  intro h1 h2
  sorry

end find_xy_l149_149000


namespace max_chain_triangles_l149_149408

theorem max_chain_triangles (n : ℕ) (h : n > 0) : 
  ∃ k, k = n^2 - n + 1 := 
sorry

end max_chain_triangles_l149_149408


namespace sum_excluding_multiples_l149_149738

theorem sum_excluding_multiples (S_total S_2 S_3 S_6 : ℕ) 
  (hS_total : S_total = (100 * (1 + 100)) / 2) 
  (hS_2 : S_2 = (50 * (2 + 100)) / 2) 
  (hS_3 : S_3 = (33 * (3 + 99)) / 2) 
  (hS_6 : S_6 = (16 * (6 + 96)) / 2) :
  S_total - S_2 - S_3 + S_6 = 1633 :=
by
  sorry

end sum_excluding_multiples_l149_149738


namespace min_f_value_l149_149956

open Real

theorem min_f_value (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) :
    ∃ (x : ℝ), (∀ y : ℝ, (|y - a| + |y - b| + |y - c| + |y - d| + |y - e|) ≥ -a - b + d + e) ∧ 
    (|x - a| + |x - b| + |x - c| + |x - d| + |x - e| = -a - b + d + e) :=
sorry

end min_f_value_l149_149956


namespace hens_count_l149_149299

theorem hens_count (H C : ℕ) (heads_eq : H + C = 44) (feet_eq : 2 * H + 4 * C = 140) : H = 18 := by
  sorry

end hens_count_l149_149299


namespace proof_problem_l149_149620

theorem proof_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b + (3/4)) * (b^2 + c + (3/4)) * (c^2 + a + (3/4)) ≥ (2 * a + (1/2)) * (2 * b + (1/2)) * (2 * c + (1/2)) := 
by
  sorry

end proof_problem_l149_149620


namespace children_left_on_bus_l149_149940

-- Definitions based on the conditions
def initial_children := 43
def children_got_off := 22

-- The theorem we want to prove
theorem children_left_on_bus (initial_children children_got_off : ℕ) : 
  initial_children - children_got_off = 21 :=
by
  sorry

end children_left_on_bus_l149_149940


namespace division_correct_l149_149173

theorem division_correct :
  250 / (15 + 13 * 3^2) = 125 / 66 :=
by
  -- The proof steps can be filled in here.
  sorry

end division_correct_l149_149173


namespace twenty_five_question_test_l149_149703

def not_possible_score (score total_questions correct_points unanswered_points incorrect_points : ℕ) : Prop :=
  ∀ correct unanswered incorrect : ℕ,
    correct + unanswered + incorrect = total_questions →
    correct * correct_points + unanswered * unanswered_points + incorrect * incorrect_points ≠ score

theorem twenty_five_question_test :
  not_possible_score 96 25 4 2 0 :=
by
  sorry

end twenty_five_question_test_l149_149703


namespace half_lake_covered_day_l149_149616

theorem half_lake_covered_day
  (N : ℕ) -- the total number of flowers needed to cover the entire lake
  (flowers_on_day : ℕ → ℕ) -- a function that gives the number of flowers on a specific day
  (h1 : flowers_on_day 20 = N) -- on the 20th day, the number of flowers is N
  (h2 : ∀ d, flowers_on_day (d + 1) = 2 * flowers_on_day d) -- the number of flowers doubles each day
  : flowers_on_day 19 = N / 2 :=
by
  sorry

end half_lake_covered_day_l149_149616


namespace probability_single_shot_l149_149924

-- Define the event and probability given
def event_A := "shooter hits the target at least once out of three shots"
def probability_event_A : ℝ := 0.875

-- The probability of missing in one shot is q, and missing all three is q^3, 
-- which leads to hitting at least once being 1 - q^3
theorem probability_single_shot (q : ℝ) (h : 1 - q^3 = 0.875) : 1 - q = 0.5 :=
by
  sorry

end probability_single_shot_l149_149924


namespace scientific_notation_29150000_l149_149974

theorem scientific_notation_29150000 :
  29150000 = 2.915 * 10^7 := sorry

end scientific_notation_29150000_l149_149974


namespace minimum_and_maximum_attendees_more_than_one_reunion_l149_149516

noncomputable def minimum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  let total_unique_attendees := oates_attendees + hall_attendees + brown_attendees
  total_unique_attendees - total_guests

noncomputable def maximum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  oates_attendees

theorem minimum_and_maximum_attendees_more_than_one_reunion
  (total_guests oates_attendees hall_attendees brown_attendees : ℕ)
  (H1 : total_guests = 200)
  (H2 : oates_attendees = 60)
  (H3 : hall_attendees = 90)
  (H4 : brown_attendees = 80) :
  minimum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 30 ∧
  maximum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 60 :=
by
  sorry

end minimum_and_maximum_attendees_more_than_one_reunion_l149_149516


namespace measure_of_angle_C_l149_149751

variable (A B C : ℕ)

theorem measure_of_angle_C :
  (A = B - 20) →
  (C = A + 40) →
  (A + B + C = 180) →
  C = 80 :=
by
  intros h1 h2 h3
  sorry

end measure_of_angle_C_l149_149751


namespace number_of_children_l149_149641

theorem number_of_children (C A : ℕ) (h1 : C = 2 * A) (h2 : C + A = 120) : C = 80 :=
by
  sorry

end number_of_children_l149_149641


namespace binomial_mod_prime_eq_floor_l149_149129

-- Define the problem's conditions and goal in Lean.
theorem binomial_mod_prime_eq_floor (n p : ℕ) (hp : Nat.Prime p) : (Nat.choose n p) % p = n / p := by
  sorry

end binomial_mod_prime_eq_floor_l149_149129


namespace symmetric_about_y_axis_l149_149629

noncomputable def f (x : ℝ) : ℝ := (4^x + 1) / 2^x

theorem symmetric_about_y_axis : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  unfold f
  sorry

end symmetric_about_y_axis_l149_149629


namespace natasha_time_reach_top_l149_149457

variable (t : ℝ) (d_up d_total T : ℝ)

def time_to_reach_top (T d_up d_total t : ℝ) : Prop :=
  d_total = 2 * d_up ∧
  d_up = 1.5 * t ∧
  T = t + 2 ∧
  2 = d_total / T

theorem natasha_time_reach_top (T : ℝ) (h : time_to_reach_top T (1.5 * 4) (3 * 4) 4) : T = 4 :=
by
  sorry

end natasha_time_reach_top_l149_149457


namespace arithmetic_sequence_a6_l149_149317

theorem arithmetic_sequence_a6 {a : ℕ → ℤ}
  (h1 : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h2 : a 2 + a 8 = 16)
  (h3 : a 4 = 6) :
  a 6 = 10 :=
by
  sorry

end arithmetic_sequence_a6_l149_149317


namespace experiment_implies_101_sq_1_equals_10200_l149_149883

theorem experiment_implies_101_sq_1_equals_10200 :
    (5^2 - 1 = 24) →
    (7^2 - 1 = 48) →
    (11^2 - 1 = 120) →
    (13^2 - 1 = 168) →
    (101^2 - 1 = 10200) :=
by
  repeat { intro }
  sorry

end experiment_implies_101_sq_1_equals_10200_l149_149883


namespace puppies_per_female_dog_l149_149731

theorem puppies_per_female_dog
  (number_of_dogs : ℕ)
  (percent_female : ℝ)
  (fraction_female_giving_birth : ℝ)
  (remaining_puppies : ℕ)
  (donated_puppies : ℕ)
  (total_puppies : ℕ)
  (number_of_female_dogs : ℕ)
  (number_female_giving_birth : ℕ)
  (puppies_per_dog : ℕ) :
  number_of_dogs = 40 →
  percent_female = 0.60 →
  fraction_female_giving_birth = 0.75 →
  remaining_puppies = 50 →
  donated_puppies = 130 →
  total_puppies = remaining_puppies + donated_puppies →
  number_of_female_dogs = percent_female * number_of_dogs →
  number_female_giving_birth = fraction_female_giving_birth * number_of_female_dogs →
  puppies_per_dog = total_puppies / number_female_giving_birth →
  puppies_per_dog = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end puppies_per_female_dog_l149_149731


namespace marc_average_speed_l149_149510

theorem marc_average_speed 
  (d : ℝ) -- Define d as a real number representing distance
  (chantal_speed1 : ℝ := 3) -- Chantal's speed for the first half
  (chantal_speed2 : ℝ := 1.5) -- Chantal's speed for the second half
  (chantal_speed3 : ℝ := 2) -- Chantal's speed while descending
  (marc_meeting_point : ℝ := (2 / 3) * d) -- One-third point from the trailhead
  (chantal_time1 : ℝ := d / chantal_speed1) 
  (chantal_time2 : ℝ := (d / chantal_speed2))
  (chantal_time3 : ℝ := (d / 6)) -- Chantal's time for the descent from peak to one-third point
  (total_time : ℝ := chantal_time1 + chantal_time2 + chantal_time3) : 
  marc_meeting_point / total_time = 12 / 13 := 
  by 
  -- Leaving the proof as sorry to indicate where the proof would be
  sorry

end marc_average_speed_l149_149510


namespace balls_into_boxes_l149_149517

theorem balls_into_boxes :
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1) 
  combination = 15 :=
by
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1)
  show combination = 15
  sorry

end balls_into_boxes_l149_149517


namespace population_total_l149_149628

variable (x y : ℕ)

theorem population_total (h1 : 20 * y = 12 * y * (x + y)) : x + y = 240 :=
  by
  -- Proceed with solving the provided conditions.
  sorry

end population_total_l149_149628


namespace Granger_payment_correct_l149_149800

noncomputable def Granger_total_payment : ℝ :=
  let spam_per_can := 3.0
  let peanut_butter_per_jar := 5.0
  let bread_per_loaf := 2.0
  let spam_quantity := 12
  let peanut_butter_quantity := 3
  let bread_quantity := 4
  let spam_dis := 0.1
  let peanut_butter_tax := 0.05
  let spam_cost := spam_quantity * spam_per_can
  let peanut_butter_cost := peanut_butter_quantity * peanut_butter_per_jar
  let bread_cost := bread_quantity * bread_per_loaf
  let spam_discount := spam_dis * spam_cost
  let peanut_butter_tax_amount := peanut_butter_tax * peanut_butter_cost
  let spam_final_cost := spam_cost - spam_discount
  let peanut_butter_final_cost := peanut_butter_cost + peanut_butter_tax_amount
  let total := spam_final_cost + peanut_butter_final_cost + bread_cost
  total

theorem Granger_payment_correct :
  Granger_total_payment = 56.15 :=
by
  sorry

end Granger_payment_correct_l149_149800


namespace A_on_curve_slope_at_A_l149_149485

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x ^ 2

-- Define the point A on the curve f
def A : ℝ × ℝ := (2, 8)

-- Define the condition that A is on the curve f
theorem A_on_curve : A.2 = f A.1 := by
  -- * left as a proof placeholder
  sorry

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x

-- State and prove the main theorem
theorem slope_at_A : (deriv f) 2 = 8 := by
  -- * left as a proof placeholder
  sorry

end A_on_curve_slope_at_A_l149_149485


namespace tel_aviv_rain_days_l149_149470

-- Define the conditions
def chance_of_rain : ℝ := 0.5
def days_considered : ℕ := 6
def given_probability : ℝ := 0.234375

-- Helper function to compute binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function P(X = k)
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- The main theorem to prove
theorem tel_aviv_rain_days :
  ∃ k, binomial_probability days_considered k chance_of_rain = given_probability ∧ k = 2 := by
  sorry

end tel_aviv_rain_days_l149_149470


namespace minimum_n_required_l149_149558

def A_0 : (ℝ × ℝ) := (0, 0)

def is_on_x_axis (A : ℝ × ℝ) : Prop := A.snd = 0
def is_on_y_equals_x_squared (B : ℝ × ℝ) : Prop := B.snd = B.fst ^ 2
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop := sorry

def A_n (n : ℕ) : ℝ × ℝ := sorry
def B_n (n : ℕ) : ℝ × ℝ := sorry

def euclidean_distance (P Q : ℝ × ℝ) : ℝ :=
  ((Q.fst - P.fst) ^ 2 + (Q.snd - P.snd) ^ 2) ^ (1/2)

theorem minimum_n_required (n : ℕ) (h1 : ∀ n, is_on_x_axis (A_n n))
    (h2 : ∀ n, is_on_y_equals_x_squared (B_n n))
    (h3 : ∀ n, is_equilateral_triangle (A_n (n-1)) (B_n n) (A_n n)) :
    (euclidean_distance A_0 (A_n n) ≥ 50) → n ≥ 17 :=
by sorry

end minimum_n_required_l149_149558


namespace sum_first_60_terms_l149_149515

theorem sum_first_60_terms {a : ℕ → ℤ}
  (h : ∀ n, a (n + 1) + (-1)^n * a n = 2 * n - 1) :
  (Finset.range 60).sum a = 1830 :=
sorry

end sum_first_60_terms_l149_149515


namespace gcd_60_90_l149_149511

theorem gcd_60_90 : Nat.gcd 60 90 = 30 := by
  let f1 : 60 = 2 ^ 2 * 3 * 5 := by sorry
  let f2 : 90 = 2 * 3 ^ 2 * 5 := by sorry
  sorry

end gcd_60_90_l149_149511


namespace total_area_of_combined_figure_l149_149152

noncomputable def combined_area (A_triangle : ℕ) (b : ℕ) : ℕ :=
  let h := (2 * A_triangle) / b
  let A_square := b * b
  A_square + A_triangle

theorem total_area_of_combined_figure :
  combined_area 720 40 = 2320 := by
  sorry

end total_area_of_combined_figure_l149_149152


namespace total_precious_stones_l149_149085

theorem total_precious_stones (agate olivine diamond : ℕ)
  (h1 : olivine = agate + 5)
  (h2 : diamond = olivine + 11)
  (h3 : agate = 30) : 
  agate + olivine + diamond = 111 :=
by
  sorry

end total_precious_stones_l149_149085


namespace towel_decrease_percentage_l149_149754

variable (L B : ℝ)
variable (h1 : 0.70 * L = L - (0.30 * L))
variable (h2 : 0.60 * B = B - (0.40 * B))

theorem towel_decrease_percentage (L B : ℝ) 
  (h1 : 0.70 * L = L - (0.30 * L))
  (h2 : 0.60 * B = B - (0.40 * B)) :
  ((L * B - (0.70 * L) * (0.60 * B)) / (L * B)) * 100 = 58 := 
by
  sorry

end towel_decrease_percentage_l149_149754


namespace machine_present_value_l149_149213

theorem machine_present_value
  (r : ℝ)  -- the depletion rate
  (t : ℝ)  -- the time in years
  (V_t : ℝ)  -- the value of the machine after time t
  (V_0 : ℝ)  -- the present value of the machine
  (h1 : r = 0.10)  -- condition for depletion rate
  (h2 : t = 2)  -- condition for time
  (h3 : V_t = 729)  -- condition for machine's value after time t
  (h4 : V_t = V_0 * (1 - r) ^ t)  -- exponential decay formula
  : V_0 = 900 :=
sorry

end machine_present_value_l149_149213


namespace evaluate_expression_l149_149495

def acbd (a b c d : ℝ) : ℝ := a * d - b * c

theorem evaluate_expression (x : ℝ) (h : x^2 - 3 * x + 1 = 0) :
  acbd (x + 1) (x - 2) (3 * x) (x - 1) = 1 := 
by
  sorry

end evaluate_expression_l149_149495


namespace polar_to_cartesian_correct_l149_149288

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_correct : polar_to_cartesian 2 (5 * Real.pi / 6) = (-Real.sqrt 3, 1) :=
by
  sorry -- We are not required to provide the proof here

end polar_to_cartesian_correct_l149_149288


namespace jerry_age_is_10_l149_149805

-- Define the ages of Mickey and Jerry
def MickeyAge : ℝ := 20
def mickey_eq_jerry (JerryAge : ℝ) : Prop := MickeyAge = 2.5 * JerryAge - 5

theorem jerry_age_is_10 : ∃ JerryAge : ℝ, mickey_eq_jerry JerryAge ∧ JerryAge = 10 :=
by
  -- By solving the equation MickeyAge = 2.5 * JerryAge - 5,
  -- we can find that Jerry's age must be 10.
  use 10
  sorry

end jerry_age_is_10_l149_149805


namespace circle_parabola_intersections_l149_149356

theorem circle_parabola_intersections : 
  ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, (p.1 ^ 2 + p.2 ^ 2 = 16) ∧ (p.2 = p.1 ^ 2 - 4)) ∧
  points.card = 3 := 
sorry

end circle_parabola_intersections_l149_149356


namespace socks_expected_value_l149_149770

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l149_149770


namespace andre_max_points_visited_l149_149383
noncomputable def largest_points_to_visit_in_alphabetical_order : ℕ :=
  10

theorem andre_max_points_visited : largest_points_to_visit_in_alphabetical_order = 10 := 
by
  sorry

end andre_max_points_visited_l149_149383


namespace medicine_types_count_l149_149335

theorem medicine_types_count (n : ℕ) (hn : n = 5) : (Nat.choose n 2 = 10) :=
by
  sorry

end medicine_types_count_l149_149335


namespace value_of_F_l149_149313

   variables (B G P Q F : ℕ)

   -- Define the main hypothesis stating that the total lengths of the books are equal.
   def fill_shelf := 
     (∃ d a : ℕ, d = B * a + 2 * G * a ∧ d = P * a + 2 * Q * a ∧ d = F * a)

   -- Prove that F equals B + 2G and P + 2Q under the hypothesis.
   theorem value_of_F (h : fill_shelf B G P Q F) : F = B + 2 * G ∧ F = P + 2 * Q :=
   sorry
   
end value_of_F_l149_149313


namespace even_function_a_is_0_l149_149444

def f (a : ℝ) (x : ℝ) : ℝ := (a+1) * x^2 + 3 * a * x + 1

theorem even_function_a_is_0 (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 :=
by sorry

end even_function_a_is_0_l149_149444


namespace dhoni_initial_toys_l149_149920

theorem dhoni_initial_toys (x : ℕ) (T : ℕ) 
    (h1 : T = 10 * x) 
    (h2 : T + 16 = 66) : x = 5 := by
  sorry

end dhoni_initial_toys_l149_149920


namespace distance_problem_l149_149690

theorem distance_problem (x y n : ℝ) (h1 : y = 15) (h2 : Real.sqrt ((x - 2) ^ 2 + (15 - 7) ^ 2) = 13) (h3 : x > 2) :
  n = Real.sqrt ((2 + Real.sqrt 105) ^ 2 + 15 ^ 2) := by
  sorry

end distance_problem_l149_149690


namespace fraction_of_height_of_head_l149_149170

theorem fraction_of_height_of_head (h_leg: ℝ) (h_total: ℝ) (h_rest: ℝ) (h_head: ℝ):
  h_leg = 1 / 3 ∧ h_total = 60 ∧ h_rest = 25 ∧ h_head = h_total - (h_leg * h_total + h_rest) 
  → h_head / h_total = 1 / 4 :=
by sorry

end fraction_of_height_of_head_l149_149170


namespace solve_system_equations_l149_149663

theorem solve_system_equations (x y : ℝ) :
  x + y = 0 ∧ 2 * x + 3 * y = 3 → x = -3 ∧ y = 3 :=
by {
  -- Leave the proof as a placeholder with "sorry".
  sorry
}

end solve_system_equations_l149_149663


namespace tan_sum_identity_l149_149420

theorem tan_sum_identity (theta : Real) (h : Real.tan theta = 1 / 3) :
  Real.tan (theta + Real.pi / 4) = 2 :=
by
  sorry

end tan_sum_identity_l149_149420


namespace perimeter_of_new_figure_is_correct_l149_149361

-- Define the given conditions
def original_horizontal_segments := 16
def original_vertical_segments := 10
def original_side_length := 1
def new_side_length := 2

-- Define total lengths calculations
def total_horizontal_length (new_side_length original_horizontal_segments : ℕ) : ℕ :=
  original_horizontal_segments * new_side_length

def total_vertical_length (new_side_length original_vertical_segments : ℕ) : ℕ :=
  original_vertical_segments * new_side_length

-- Formulate the main theorem
theorem perimeter_of_new_figure_is_correct :
  total_horizontal_length new_side_length original_horizontal_segments + 
  total_vertical_length new_side_length original_vertical_segments = 52 := by
  sorry

end perimeter_of_new_figure_is_correct_l149_149361


namespace crayons_remaining_l149_149117

def initial_crayons : ℕ := 87
def eaten_crayons : ℕ := 7

theorem crayons_remaining : (initial_crayons - eaten_crayons) = 80 := by
  sorry

end crayons_remaining_l149_149117


namespace m_plus_n_eq_47_l149_149646

theorem m_plus_n_eq_47 (m n : ℕ)
  (h1 : m + 8 < n - 1)
  (h2 : (m + m + 3 + m + 8 + n - 1 + n + 3 + 2 * n - 2) / 6 = n)
  (h3 : (m + 8 + (n - 1)) / 2 = n) :
  m + n = 47 :=
sorry

end m_plus_n_eq_47_l149_149646


namespace range_of_a_l149_149872

theorem range_of_a (a : ℝ) (x1 x2 : ℝ)
  (h_poly: ∀ x, x * x + (a * a - 1) * x + (a - 2) = 0 → x = x1 ∨ x = x2)
  (h_order: x1 < 1 ∧ 1 < x2) : 
  -2 < a ∧ a < 1 := 
sorry

end range_of_a_l149_149872


namespace estimated_number_of_red_balls_l149_149159

theorem estimated_number_of_red_balls (total_balls : ℕ) (red_draws : ℕ) (total_draws : ℕ)
    (h_total_balls : total_balls = 8) (h_red_draws : red_draws = 75) (h_total_draws : total_draws = 100) :
    total_balls * (red_draws / total_draws : ℚ) = 6 := 
by
  sorry

end estimated_number_of_red_balls_l149_149159


namespace smallest_n_for_modulo_eq_l149_149902

theorem smallest_n_for_modulo_eq :
  ∃ (n : ℕ), (3^n % 4 = n^3 % 4) ∧ (∀ m : ℕ, m < n → 3^m % 4 ≠ m^3 % 4) ∧ n = 7 :=
by
  sorry

end smallest_n_for_modulo_eq_l149_149902


namespace brad_age_proof_l149_149536

theorem brad_age_proof :
  ∀ (Shara_age Jaymee_age Average_age Brad_age : ℕ),
  Jaymee_age = 2 * Shara_age + 2 →
  Average_age = (Shara_age + Jaymee_age) / 2 →
  Brad_age = Average_age - 3 →
  Shara_age = 10 →
  Brad_age = 13 :=
by
  intros Shara_age Jaymee_age Average_age Brad_age
  intro h1 h2 h3 h4
  sorry

end brad_age_proof_l149_149536


namespace mark_purchased_cans_l149_149423

theorem mark_purchased_cans : ∀ (J M : ℕ), 
    (J = 40) → 
    (100 - J = 6 * M / 5) → 
    M = 27 := by
  sorry

end mark_purchased_cans_l149_149423


namespace fraction_difference_in_simplest_form_l149_149766

noncomputable def difference_fraction : ℚ := (5 / 19) - (2 / 23)

theorem fraction_difference_in_simplest_form :
  difference_fraction = 77 / 437 := by sorry

end fraction_difference_in_simplest_form_l149_149766


namespace fraction_of_ripe_oranges_eaten_l149_149321

theorem fraction_of_ripe_oranges_eaten :
  ∀ (total_oranges ripe_oranges unripe_oranges uneaten_oranges eaten_unripe_oranges eaten_ripe_oranges : ℕ),
    total_oranges = 96 →
    ripe_oranges = total_oranges / 2 →
    unripe_oranges = total_oranges / 2 →
    eaten_unripe_oranges = unripe_oranges / 8 →
    uneaten_oranges = 78 →
    eaten_ripe_oranges = (total_oranges - uneaten_oranges) - eaten_unripe_oranges →
    (eaten_ripe_oranges : ℚ) / ripe_oranges = 1 / 4 :=
by
  intros total_oranges ripe_oranges unripe_oranges uneaten_oranges eaten_unripe_oranges eaten_ripe_oranges
  intros h_total h_ripe h_unripe h_eaten_unripe h_uneaten h_eaten_ripe
  sorry

end fraction_of_ripe_oranges_eaten_l149_149321


namespace max_value_fraction_l149_149838

theorem max_value_fraction {a b c : ℝ} (h1 : c = Real.sqrt (a^2 + b^2)) 
  (h2 : a > 0) (h3 : b > 0) (A : ℝ) (hA : A = 1 / 2 * a * b) :
  ∃ x : ℝ, x = (a + b + A) / c ∧ x ≤ (5 / 4) * Real.sqrt 2 :=
by
  sorry

end max_value_fraction_l149_149838


namespace max_f_on_interval_l149_149995

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + (Real.sqrt 3) * Real.sin x * Real.cos x

theorem max_f_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ ∀ y ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f y ≤ f x ∧ f x = 3 / 2 :=
  sorry

end max_f_on_interval_l149_149995


namespace quilt_square_side_length_l149_149382

theorem quilt_square_side_length (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ s : ℝ, (length * width = s * s) ∧ s = 12 :=
by {
  sorry
}

end quilt_square_side_length_l149_149382


namespace spending_example_l149_149268

theorem spending_example (X : ℝ) (h₁ : X + 2 * X + 3 * X = 120) : X = 20 := by
  sorry

end spending_example_l149_149268


namespace same_number_assigned_to_each_point_l149_149832

namespace EqualNumberAssignment

def is_arithmetic_mean (f : ℤ × ℤ → ℕ) (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4

theorem same_number_assigned_to_each_point (f : ℤ × ℤ → ℕ) :
  (∀ p : ℤ × ℤ, is_arithmetic_mean f p) → ∃ m : ℕ, ∀ p : ℤ × ℤ, f p = m :=
by
  intros h
  sorry

end EqualNumberAssignment

end same_number_assigned_to_each_point_l149_149832


namespace product_lcm_gcd_eq_2160_l149_149038

theorem product_lcm_gcd_eq_2160 :
  let a := 36
  let b := 60
  lcm a b * gcd a b = 2160 := by
  sorry

end product_lcm_gcd_eq_2160_l149_149038


namespace sin_105_value_cos_75_value_trigonometric_identity_l149_149461

noncomputable def sin_105_eq : Real := Real.sin (105 * Real.pi / 180)
noncomputable def cos_75_eq : Real := Real.cos (75 * Real.pi / 180)
noncomputable def cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq : Real := 
  Real.cos (Real.pi / 5) * Real.cos (3 * Real.pi / 10) - Real.sin (Real.pi / 5) * Real.sin (3 * Real.pi / 10)

theorem sin_105_value : sin_105_eq = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
  by sorry

theorem cos_75_value : cos_75_eq = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
  by sorry

theorem trigonometric_identity : cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq = 0 := 
  by sorry

end sin_105_value_cos_75_value_trigonometric_identity_l149_149461


namespace probability_all_same_color_l149_149534

open scoped Classical

noncomputable def num_black : ℕ := 5
noncomputable def num_red : ℕ := 4
noncomputable def num_green : ℕ := 6
noncomputable def num_blue : ℕ := 3
noncomputable def num_yellow : ℕ := 2

noncomputable def total_marbles : ℕ :=
  num_black + num_red + num_green + num_blue + num_yellow

noncomputable def prob_all_same_color : ℚ :=
  let p_black := if num_black >= 4 then 
      (num_black / total_marbles) * ((num_black - 1) / (total_marbles - 1)) *
      ((num_black - 2) / (total_marbles - 2)) * ((num_black - 3) / (total_marbles - 3)) else 0
  let p_green := if num_green >= 4 then 
      (num_green / total_marbles) * ((num_green - 1) / (total_marbles - 1)) *
      ((num_green - 2) / (total_marbles - 2)) * ((num_green - 3) / (total_marbles - 3)) else 0
  p_black + p_green

theorem probability_all_same_color :
  prob_all_same_color = 0.004128 :=
sorry

end probability_all_same_color_l149_149534


namespace tom_needs_noodle_packages_l149_149543

def beef_pounds : ℕ := 10
def noodle_multiplier : ℕ := 2
def initial_noodles : ℕ := 4
def package_weight : ℕ := 2

theorem tom_needs_noodle_packages :
  (noodle_multiplier * beef_pounds - initial_noodles) / package_weight = 8 := 
by 
  -- Faithfully skipping the solution steps
  sorry

end tom_needs_noodle_packages_l149_149543


namespace sibling_age_difference_l149_149006

theorem sibling_age_difference (Y : ℝ) (Y_eq : Y = 25.75) (avg_age_eq : (Y + (Y + 3) + (Y + 6) + (Y + x)) / 4 = 30) : (Y + 6) - Y = 6 :=
by
  sorry

end sibling_age_difference_l149_149006


namespace retail_price_increase_l149_149087

theorem retail_price_increase (R W : ℝ) (h1 : 0.80 * R = 1.44000000000000014 * W)
  : ((R - W) / W) * 100 = 80 :=
by 
  sorry

end retail_price_increase_l149_149087


namespace sheila_earning_per_hour_l149_149352

theorem sheila_earning_per_hour :
  (252 / ((8 * 3) + (6 * 2)) = 7) := 
by
  -- Prove that sheila earns $7 per hour
  
  sorry

end sheila_earning_per_hour_l149_149352


namespace sum_is_square_l149_149136

theorem sum_is_square (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : Nat.gcd a b = 1) (h5 : Nat.gcd b c = 1) (h6 : Nat.gcd c a = 1) 
  (h7 : (1:ℚ)/a + (1:ℚ)/b = (1:ℚ)/c) : ∃ k : ℕ, a + b = k ^ 2 := 
by 
  sorry

end sum_is_square_l149_149136


namespace correct_transformation_l149_149405

theorem correct_transformation (x : ℝ) :
  (6 * ((2 * x + 1) / 3) - 6 * ((10 * x + 1) / 6) = 6) ↔ (4 * x + 2 - 10 * x - 1 = 6) :=
by
  sorry

end correct_transformation_l149_149405


namespace minimum_value_of_a_l149_149180

theorem minimum_value_of_a (x y a : ℝ) (h1 : y = (1 / (x - 2)) * (x^2))
(h2 : x = a * y) : a = 3 :=
sorry

end minimum_value_of_a_l149_149180


namespace find_first_part_l149_149736

variable (x y : ℕ)

theorem find_first_part (h₁ : x + y = 24) (h₂ : 7 * x + 5 * y = 146) : x = 13 :=
by
  -- The proof is omitted
  sorry

end find_first_part_l149_149736


namespace min_direction_changes_l149_149379

theorem min_direction_changes (n : ℕ) : 
  ∀ (path : Finset (ℕ × ℕ)), 
    (path.card = (n + 1) * (n + 2) / 2) → 
    (∀ (v : ℕ × ℕ), v ∈ path) →
    ∃ changes, (changes ≥ n) :=
by sorry

end min_direction_changes_l149_149379


namespace total_container_weight_is_correct_l149_149898

-- Definitions based on the conditions
def copper_bar_weight : ℕ := 90
def steel_bar_weight : ℕ := copper_bar_weight + 20
def tin_bar_weight : ℕ := steel_bar_weight / 2
def aluminum_bar_weight : ℕ := tin_bar_weight + 10

-- Number of bars in the container
def count_steel_bars : ℕ := 10
def count_tin_bars : ℕ := 15
def count_copper_bars : ℕ := 12
def count_aluminum_bars : ℕ := 8

-- Total weight of each type of bar
def total_steel_weight : ℕ := count_steel_bars * steel_bar_weight
def total_tin_weight : ℕ := count_tin_bars * tin_bar_weight
def total_copper_weight : ℕ := count_copper_bars * copper_bar_weight
def total_aluminum_weight : ℕ := count_aluminum_bars * aluminum_bar_weight

-- Total weight of the container
def total_container_weight : ℕ := total_steel_weight + total_tin_weight + total_copper_weight + total_aluminum_weight

-- Theorem to prove
theorem total_container_weight_is_correct : total_container_weight = 3525 := by
  sorry

end total_container_weight_is_correct_l149_149898


namespace infinite_series_sum_l149_149494

theorem infinite_series_sum :
  (∑' n : ℕ, (2 * (n + 1) * (n + 1) + (n + 1) + 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))) = 5 / 6 := by
  sorry

end infinite_series_sum_l149_149494


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l149_149677

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l149_149677


namespace math_problem_l149_149406

noncomputable def f (x : ℝ) : ℝ := sorry

theorem math_problem (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (1 - x) = f (1 + x))
  (h2 : ∀ x : ℝ, f (-x) = -f x)
  (h3 : ∀ {x y : ℝ}, (0 ≤ x → x < y → y ≤ 1 → f x < f y)) :
  (f 0 = 0) ∧ 
  (∀ x : ℝ, f (x + 2) = f (-x)) ∧ 
  (∀ x : ℝ, x = -1 ∨ ∀ ε > 0, ε ≠ (x + 1))
:= sorry

end math_problem_l149_149406


namespace mars_mission_cost_per_person_l149_149028

theorem mars_mission_cost_per_person
  (total_cost : ℕ) (number_of_people : ℕ)
  (h1 : total_cost = 50000000000) (h2 : number_of_people = 500000000) :
  (total_cost / number_of_people) = 100 := 
by
  sorry

end mars_mission_cost_per_person_l149_149028


namespace ratio_tends_to_zero_as_n_tends_to_infinity_l149_149855

def smallest_prime_not_dividing (n : ℕ) : ℕ :=
  -- Function to find the smallest prime not dividing n
  sorry

theorem ratio_tends_to_zero_as_n_tends_to_infinity :
  ∀ ε > 0, ∃ N, ∀ n > N, (smallest_prime_not_dividing n : ℝ) / (n : ℝ) < ε := by
  sorry

end ratio_tends_to_zero_as_n_tends_to_infinity_l149_149855


namespace least_distance_between_ticks_l149_149680

theorem least_distance_between_ticks (x : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, k = n * 11 ∨ k = n * 13) →
  x = 1 / 143 :=
by
  sorry

end least_distance_between_ticks_l149_149680


namespace complex_addition_l149_149949

namespace ComplexProof

def B := (3 : ℂ) + (2 * Complex.I)
def Q := (-5 : ℂ)
def R := (2 * Complex.I)
def T := (3 : ℂ) + (5 * Complex.I)

theorem complex_addition :
  B - Q + R + T = (1 : ℂ) + (9 * Complex.I) := 
by
  sorry

end ComplexProof

end complex_addition_l149_149949


namespace problem1_problem2_l149_149947

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem problem1 (m : ℝ) (h₀ : m > 3) (h₁ : ∃ m, (1/2) * (((m - 1) / 2) - (-(m + 1) / 2) + 3) * (m - 3) = 7 / 2) : m = 4 := by
  sorry

theorem problem2 (a : ℝ) (h₂ : ∃ x, (0 ≤ x ∧ x ≤ 2) ∧ f x ≥ abs (a - 3)) : -2 ≤ a ∧ a ≤ 8 := by
  sorry

end problem1_problem2_l149_149947


namespace find_a_l149_149064

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h : f = fun x => a * x ^ 3 - 3 * x) (h1 : f (-1) = 4) : a = -1 :=
by
  sorry

end find_a_l149_149064


namespace optionC_is_correct_l149_149076

theorem optionC_is_correct (x : ℝ) : (x^2)^3 = x^6 :=
by sorry

end optionC_is_correct_l149_149076


namespace sqrt_diff_eq_neg_four_sqrt_five_l149_149769

theorem sqrt_diff_eq_neg_four_sqrt_five : 
  (Real.sqrt (16 - 8 * Real.sqrt 5) - Real.sqrt (16 + 8 * Real.sqrt 5)) = -4 * Real.sqrt 5 := 
sorry

end sqrt_diff_eq_neg_four_sqrt_five_l149_149769


namespace quadratic_vertex_form_l149_149373

theorem quadratic_vertex_form (a h k x: ℝ) (h_a : a = 3) (hx : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by {
  sorry
}

end quadratic_vertex_form_l149_149373


namespace find_m_l149_149039

theorem find_m (m : ℝ) : (m + 2) * (m - 2) + 3 * m * (m + 2) = 0 ↔ m = 1/2 ∨ m = -2 :=
by
  sorry

end find_m_l149_149039


namespace correct_transformation_l149_149462

-- Given conditions
variables {a b : ℝ}
variable (h : 3 * a = 4 * b)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)

-- Statement of the problem
theorem correct_transformation : (a / 4) = (b / 3) :=
sorry

end correct_transformation_l149_149462


namespace sum_equals_120_l149_149921

def rectangular_parallelepiped := (3, 4, 5)

def face_dimensions : List (ℕ × ℕ) := [(4, 5), (3, 5), (3, 4)]

def number_assignment (d : ℕ × ℕ) : ℕ :=
  if d = (4, 5) then 9
  else if d = (3, 5) then 8
  else if d = (3, 4) then 5
  else 0

def sum_checkerboard_ring_one_width (rect_dims : ℕ × ℕ × ℕ) (number_assignment : ℕ × ℕ → ℕ) : ℕ :=
  let (x, y, z) := rect_dims
  let l1 := number_assignment (4, 5) * 2 * (4 * 5)
  let l2 := number_assignment (3, 5) * 2 * (3 * 5)
  let l3 := number_assignment (3, 4) * 2 * (3 * 4) 
  l1 + l2 + l3

theorem sum_equals_120 : ∀ rect_dims number_assignment,
  rect_dims = rectangular_parallelepiped → sum_checkerboard_ring_one_width rect_dims number_assignment = 720 := sorry

end sum_equals_120_l149_149921


namespace unique_rhombus_property_not_in_rectangle_l149_149653

-- Definitions of properties for a rhombus and a rectangle
def is_rhombus (sides_equal : Prop) (opposite_sides_parallel : Prop) (opposite_angles_equal : Prop)
  (diagonals_perpendicular_and_bisect : Prop) : Prop :=
  sides_equal ∧ opposite_sides_parallel ∧ opposite_angles_equal ∧ diagonals_perpendicular_and_bisect

def is_rectangle (opposite_sides_equal_and_parallel : Prop) (all_angles_right : Prop)
  (diagonals_equal_and_bisect : Prop) : Prop :=
  opposite_sides_equal_and_parallel ∧ all_angles_right ∧ diagonals_equal_and_bisect

-- Proof objective: Prove that the unique property of a rhombus is the perpendicular and bisecting nature of its diagonals
theorem unique_rhombus_property_not_in_rectangle :
  ∀ (sides_equal opposite_sides_parallel opposite_angles_equal
      diagonals_perpendicular_and_bisect opposite_sides_equal_and_parallel
      all_angles_right diagonals_equal_and_bisect : Prop),
  is_rhombus sides_equal opposite_sides_parallel opposite_angles_equal diagonals_perpendicular_and_bisect →
  is_rectangle opposite_sides_equal_and_parallel all_angles_right diagonals_equal_and_bisect →
  diagonals_perpendicular_and_bisect ∧ ¬diagonals_equal_and_bisect :=
by
  sorry

end unique_rhombus_property_not_in_rectangle_l149_149653


namespace part1_part2_l149_149570

namespace VectorProblem

def vector_a : ℝ × ℝ := (3, 2)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (4, 1)

def m := 5 / 9
def n := 8 / 9

def k := -16 / 13

-- Statement 1: Prove vectors satisfy the linear combination
theorem part1 : vector_a = (m * vector_b.1 + n * vector_c.1, m * vector_b.2 + n * vector_c.2) :=
by {
  sorry
}

-- Statement 2: Prove vectors are parallel
theorem part2 : (3 + 4 * k) * 2 + (2 + k) * 5 = 0 :=
by {
  sorry
}

end VectorProblem

end part1_part2_l149_149570


namespace rectangle_dimensions_l149_149990

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : 2 * l + 2 * w = 150) 
  (h2 : l = w + 15) : 
  w = 30 ∧ l = 45 := 
  by 
  sorry

end rectangle_dimensions_l149_149990


namespace smallest_unwritable_number_l149_149497

theorem smallest_unwritable_number :
  ∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d) := sorry

end smallest_unwritable_number_l149_149497


namespace hypotenuse_length_l149_149196

theorem hypotenuse_length
  (a b : ℝ)
  (V1 : ℝ := (1/3) * Real.pi * a * b^2)
  (V2 : ℝ := (1/3) * Real.pi * b * a^2)
  (hV1 : V1 = 800 * Real.pi)
  (hV2 : V2 = 1920 * Real.pi) :
  Real.sqrt (a^2 + b^2) = 26 :=
by
  sorry

end hypotenuse_length_l149_149196


namespace sufficient_not_necessary_condition_l149_149972

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x^2 > 1 → 1 / x < 1) ∧ (¬(1 / x < 1 → x^2 > 1)) :=
by sorry

end sufficient_not_necessary_condition_l149_149972


namespace find_two_numbers_l149_149970

theorem find_two_numbers (x y : ℕ) : 
  (x + y = 20) ∧
  (x * y = 96) ↔ 
  ((x = 12 ∧ y = 8) ∨ (x = 8 ∧ y = 12)) := 
by
  sorry

end find_two_numbers_l149_149970


namespace avg_cost_apple_tv_200_l149_149779

noncomputable def average_cost_apple_tv (iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost: ℝ) : ℝ :=
  (overall_avg_cost * (iphones_sold + ipads_sold + apple_tvs_sold) - (iphones_sold * iphone_cost + ipads_sold * ipad_cost)) / apple_tvs_sold

theorem avg_cost_apple_tv_200 :
  let iphones_sold := 100
  let ipads_sold := 20
  let apple_tvs_sold := 80
  let iphone_cost := 1000
  let ipad_cost := 900
  let overall_avg_cost := 670
  average_cost_apple_tv iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost = 200 :=
by
  sorry

end avg_cost_apple_tv_200_l149_149779


namespace range_of_a_l149_149054

-- Definitions of sets A and B
def A (x : ℝ) : Prop := 1 < |x - 2| ∧ |x - 2| < 2
def B (x a : ℝ) : Prop := x^2 - (a + 1) * x + a < 0

-- The condition A ∩ B ≠ ∅
def nonempty_intersection (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a

-- Proving the required range of a
theorem range_of_a : {a : ℝ | nonempty_intersection a} = {a : ℝ | a < 1 ∨ a > 3} := by
  sorry

end range_of_a_l149_149054


namespace sequence_diff_ge_abs_m_l149_149651

-- Define the conditions and theorem in Lean

theorem sequence_diff_ge_abs_m
    (m : ℤ) (h_m : |m| ≥ 2)
    (a : ℕ → ℤ)
    (h_seq_not_zero : ¬ (a 1 = 0 ∧ a 2 = 0))
    (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - m * a n)
    (r s : ℕ) (h_r : r > s) (h_s : s ≥ 2)
    (h_equal : a r = a 1 ∧ a s = a 1) :
    r - s ≥ |m| :=
by
  sorry

end sequence_diff_ge_abs_m_l149_149651


namespace square_areas_l149_149063

variables (a b : ℝ)

def is_perimeter_difference (a b : ℝ) : Prop :=
  4 * a - 4 * b = 12

def is_area_difference (a b : ℝ) : Prop :=
  a^2 - b^2 = 69

theorem square_areas (a b : ℝ) (h1 : is_perimeter_difference a b) (h2 : is_area_difference a b) :
  a^2 = 169 ∧ b^2 = 100 :=
by {
  sorry
}

end square_areas_l149_149063


namespace price_of_pants_l149_149713

theorem price_of_pants (P : ℝ) (h1 : 4 * 33 = 132) (h2 : 2 * P + 132 = 240) : P = 54 :=
sorry

end price_of_pants_l149_149713


namespace maximum_contribution_l149_149440

theorem maximum_contribution (total_contribution : ℕ) (num_people : ℕ) (individual_min_contribution : ℕ) :
  total_contribution = 20 → num_people = 10 → individual_min_contribution = 1 → 
  ∃ (max_contribution : ℕ), max_contribution = 11 := by
  intro h1 h2 h3
  existsi 11
  sorry

end maximum_contribution_l149_149440


namespace prob_at_least_one_l149_149626

-- Defining the probabilities of the alarms going off on time
def prob_A : ℝ := 0.80
def prob_B : ℝ := 0.90

-- Define the complementary event (neither alarm goes off on time)
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

-- The main theorem statement we need to prove
theorem prob_at_least_one : 1 - prob_neither = 0.98 :=
by
  sorry

end prob_at_least_one_l149_149626


namespace expand_expression_l149_149312

variable (x y z : ℝ)

theorem expand_expression :
  (x + 8) * (3 * y + 12) * (2 * z + 4) =
  6 * x * y * z + 12 * x * z + 24 * y * z + 12 * x * y + 48 * x + 96 * y + 96 * z + 384 := 
  sorry

end expand_expression_l149_149312


namespace new_volume_of_balloon_l149_149139

def initial_volume : ℝ := 2.00  -- Initial volume in liters
def initial_pressure : ℝ := 745  -- Initial pressure in mmHg
def initial_temperature : ℝ := 293.15  -- Initial temperature in Kelvin
def final_pressure : ℝ := 700  -- Final pressure in mmHg
def final_temperature : ℝ := 283.15  -- Final temperature in Kelvin
def final_volume : ℝ := 2.06  -- Expected final volume in liters

theorem new_volume_of_balloon :
  (initial_pressure * initial_volume / initial_temperature) = (final_pressure * final_volume / final_temperature) :=
  sorry  -- Proof to be filled in later

end new_volume_of_balloon_l149_149139


namespace coffee_ratio_correct_l149_149034

noncomputable def ratio_of_guests (cups_weak : ℕ) (cups_strong : ℕ) (tablespoons_weak : ℕ) (tablespoons_strong : ℕ) (total_tablespoons : ℕ) : ℤ :=
  if (cups_weak * tablespoons_weak + cups_strong * tablespoons_strong = total_tablespoons) then
    (cups_weak * tablespoons_weak / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong)) /
    (cups_strong * tablespoons_strong / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong))
  else 0

theorem coffee_ratio_correct :
  ratio_of_guests 12 12 1 2 36 = 1 / 2 :=
by
  sorry

end coffee_ratio_correct_l149_149034


namespace difference_of_numbers_l149_149089

theorem difference_of_numbers (x y : ℝ) (h1 : x * y = 23) (h2 : x + y = 24) : |x - y| = 22 :=
sorry

end difference_of_numbers_l149_149089


namespace intersection_of_M_N_equals_0_1_open_interval_l149_149453

def M : Set ℝ := { x | x ≥ 0 }
def N : Set ℝ := { x | x^2 < 1 }

theorem intersection_of_M_N_equals_0_1_open_interval :
  M ∩ N = { x | 0 ≤ x ∧ x < 1 } := 
sorry

end intersection_of_M_N_equals_0_1_open_interval_l149_149453


namespace inequality_solution_l149_149177

noncomputable def inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : Prop :=
  (x^4 + y^4 + z^4) ≥ (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) ∧ (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) ≥ (x * y * z * (x + y + z))

theorem inequality_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  inequality_proof x y z hx hy hz :=
by 
  sorry

end inequality_solution_l149_149177


namespace find_f_of_2_l149_149007

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x - b

theorem find_f_of_2 (a b : ℝ) (h_pos : 0 < a)
  (h1 : ∀ x : ℝ, a * f x a b - b = 4 * x - 3)
  : f 2 a b = 3 := 
sorry

end find_f_of_2_l149_149007


namespace servings_in_box_l149_149216

-- Define amounts
def total_cereal : ℕ := 18
def per_serving : ℕ := 2

-- Define the statement to prove
theorem servings_in_box : total_cereal / per_serving = 9 :=
by
  sorry

end servings_in_box_l149_149216


namespace regular_hexagon_interior_angle_l149_149357

theorem regular_hexagon_interior_angle : ∀ (n : ℕ), n = 6 → ∀ (angle_sum : ℕ), angle_sum = (n - 2) * 180 → (∀ (angle : ℕ), angle = angle_sum / n → angle = 120) :=
by sorry

end regular_hexagon_interior_angle_l149_149357


namespace find_room_length_l149_149845

variable (w : ℝ) (C : ℝ) (r : ℝ)

theorem find_room_length (h_w : w = 4.75) (h_C : C = 29925) (h_r : r = 900) : (C / r) / w = 7 := by
  sorry

end find_room_length_l149_149845


namespace complement_of_A_in_U_l149_149388

-- Define the universal set U
def U : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 1 / x ≥ 1}

-- Define the complement of A within U
def complement_U_A : Set ℝ := {x | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_of_A_in_U : complement_U_A = {x | -1 < x ∧ x ≤ 0} :=
  sorry

end complement_of_A_in_U_l149_149388


namespace highway_speed_l149_149150

theorem highway_speed 
  (local_distance : ℝ) (local_speed : ℝ)
  (highway_distance : ℝ) (avg_speed : ℝ)
  (h_local : local_distance = 90) 
  (h_local_speed : local_speed = 30)
  (h_highway : highway_distance = 75)
  (h_avg : avg_speed = 38.82) :
  ∃ v : ℝ, v = 60 := 
sorry

end highway_speed_l149_149150


namespace randys_trip_length_l149_149110

theorem randys_trip_length
  (trip_length : ℚ)
  (fraction_gravel : trip_length = (1 / 4) * trip_length)
  (middle_miles : 30 = (7 / 12) * trip_length)
  (fraction_dirt : trip_length = (1 / 6) * trip_length) :
  trip_length = 360 / 7 :=
by
  sorry

end randys_trip_length_l149_149110


namespace max_value_eq_two_l149_149198

noncomputable def max_value (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) : ℝ :=
  a + b^3 + c^4

theorem max_value_eq_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) :
  max_value a b c h1 h2 h3 h4 ≤ 2 :=
sorry

end max_value_eq_two_l149_149198


namespace natural_numbers_satisfying_conditions_l149_149798

variable (a b : ℕ)

theorem natural_numbers_satisfying_conditions :
  (90 < a + b ∧ a + b < 100) ∧ (0.9 < (a : ℝ) / b ∧ (a : ℝ) / b < 0.91) ↔ (a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52) := by
  sorry

end natural_numbers_satisfying_conditions_l149_149798


namespace triangle_is_obtuse_l149_149146

noncomputable def is_exterior_smaller (exterior_angle interior_angle : ℝ) : Prop :=
  exterior_angle < interior_angle

noncomputable def sum_of_angles (exterior_angle interior_angle : ℝ) : Prop :=
  exterior_angle + interior_angle = 180

theorem triangle_is_obtuse (exterior_angle interior_angle : ℝ) (h1 : is_exterior_smaller exterior_angle interior_angle) 
  (h2 : sum_of_angles exterior_angle interior_angle) : ∃ b, 90 < b ∧ b = interior_angle :=
sorry

end triangle_is_obtuse_l149_149146


namespace min_max_expression_l149_149342

theorem min_max_expression (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 19) (h2 : b^2 + b * c + c^2 = 19) :
  ∃ (min_val max_val : ℝ), 
    min_val = 0 ∧ max_val = 57 ∧ 
    (∀ x, x = c^2 + c * a + a^2 → min_val ≤ x ∧ x ≤ max_val) :=
by sorry

end min_max_expression_l149_149342


namespace eval_32_pow_5_div_2_l149_149066

theorem eval_32_pow_5_div_2 :
  32^(5/2) = 4096 * Real.sqrt 2 :=
by
  sorry

end eval_32_pow_5_div_2_l149_149066


namespace total_points_scored_l149_149523

theorem total_points_scored 
  (darius_score : ℕ) 
  (marius_score : ℕ) 
  (matt_score : ℕ) 
  (h1 : marius_score = darius_score + 3) 
  (h2 : darius_score = matt_score - 5)
  (h3 : darius_score = 10) : darius_score + marius_score + matt_score = 38 :=
by
  sorry

end total_points_scored_l149_149523


namespace units_digit_47_4_plus_28_4_l149_149954

theorem units_digit_47_4_plus_28_4 (units_digit_47 : Nat := 7) (units_digit_28 : Nat := 8) :
  (47^4 + 28^4) % 10 = 7 :=
by
  sorry

end units_digit_47_4_plus_28_4_l149_149954


namespace chosen_numbers_rel_prime_l149_149171

theorem chosen_numbers_rel_prime :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 2003 → s.card = 1002 → ∃ (x y : ℕ), x ∈ s ∧ y ∈ s ∧ Nat.gcd x y = 1 :=
by
  sorry

end chosen_numbers_rel_prime_l149_149171


namespace find_distance_between_sides_l149_149648

-- Define the given conditions
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area_trapezium : ℝ := 247

-- Define the distance h between parallel sides
def distance_between_sides (h : ℝ) : Prop :=
  area_trapezium = (1 / 2) * (length_side1 + length_side2) * h

-- Define the theorem we want to prove
theorem find_distance_between_sides : ∃ h : ℝ, distance_between_sides h ∧ h = 13 := by
  sorry

end find_distance_between_sides_l149_149648


namespace problem_l149_149292

theorem problem (a b c d : ℝ) (h₁ : a + b = 0) (h₂ : c * d = 1) : 
  (5 * a + 5 * b - 7 * c * d) / (-(c * d) ^ 3) = 7 := 
by
  sorry

end problem_l149_149292


namespace max_net_income_is_50000_l149_149658

def tax_rate (y : ℝ) : ℝ :=
  10 * y ^ 2

def net_income (y : ℝ) : ℝ :=
  1000 * y - tax_rate y

theorem max_net_income_is_50000 :
  ∃ y : ℝ, (net_income y = 25000 ∧ 1000 * y = 50000) :=
by
  use 50
  sorry

end max_net_income_is_50000_l149_149658


namespace negation_of_universal_l149_149547
-- Import the Mathlib library to provide the necessary mathematical background

-- State the theorem that we want to prove. This will state that the negation of the universal proposition is an existential proposition
theorem negation_of_universal :
  (¬ (∀ x : ℝ, x > 0)) ↔ (∃ x : ℝ, x ≤ 0) :=
sorry

end negation_of_universal_l149_149547


namespace marks_chemistry_l149_149971

-- Definitions based on conditions
def marks_english : ℕ := 96
def marks_math : ℕ := 98
def marks_physics : ℕ := 99
def marks_biology : ℕ := 98
def average_marks : ℝ := 98.2
def num_subjects : ℕ := 5

-- Statement to prove
theorem marks_chemistry :
  ((marks_english + marks_math + marks_physics + marks_biology : ℕ) + (x : ℕ)) / num_subjects = average_marks →
  x = 100 :=
by
  sorry

end marks_chemistry_l149_149971


namespace functional_equation_true_l149_149932

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : f x > 0
axiom f_property (a b : ℝ) : f a * f b = f (a + b)

theorem functional_equation_true :
  (f 0 = 1) ∧ 
  (∀ a, f (-a) = 1 / f a) ∧ 
  (∀ a, f a = (f (4 * a)) ^ (1 / 4)) ∧ 
  (∀ a, f (a^2) = (f a)^2) :=
by {
  sorry
}

end functional_equation_true_l149_149932


namespace total_number_of_boys_in_all_class_sections_is_380_l149_149825

theorem total_number_of_boys_in_all_class_sections_is_380 :
  let students_section1 := 160
  let students_section2 := 200
  let students_section3 := 240
  let girls_section1 := students_section1 / 4
  let boys_section1 := students_section1 - girls_section1
  let boys_section2 := (3 / 5) * students_section2
  let total_parts := 7 + 5
  let boys_section3 := (7 / total_parts) * students_section3
  boys_section1 + boys_section2 + boys_section3 = 380 :=
sorry

end total_number_of_boys_in_all_class_sections_is_380_l149_149825


namespace total_fencing_needed_l149_149343

def width1 : ℕ := 4
def length1 : ℕ := 2 * width1 - 1

def length2 : ℕ := length1 + 3
def width2 : ℕ := width1 - 2

def width3 : ℕ := (width1 + width2) / 2
def length3 : ℚ := (length1 + length2) / 2

def perimeter (w l : ℚ) : ℚ := 2 * (w + l)

def P1 : ℚ := perimeter width1 length1
def P2 : ℚ := perimeter width2 length2
def P3 : ℚ := perimeter width3 length3

def total_fence : ℚ := P1 + P2 + P3

theorem total_fencing_needed : total_fence = 69 := 
  sorry

end total_fencing_needed_l149_149343


namespace range_of_m_l149_149753

noncomputable def f (a x: ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

theorem range_of_m (a m x₁ x₂: ℝ) (h₁: a ∈ Set.Icc (-3) (0)) (h₂: x₁ ∈ Set.Icc (0) (2)) (h₃: x₂ ∈ Set.Icc (0) (2)) : m ∈ Set.Ici (5) → m - a * m^2 ≥ |f a x₁ - f a x₂| :=
sorry

end range_of_m_l149_149753


namespace sum_complex_l149_149498

-- Define the given complex numbers
def z1 : ℂ := ⟨2, 5⟩
def z2 : ℂ := ⟨3, -7⟩

-- State the theorem to prove the sum
theorem sum_complex : z1 + z2 = ⟨5, -2⟩ :=
by
  sorry

end sum_complex_l149_149498


namespace asymptotes_equation_l149_149876

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  x^2 / 64 - y^2 / 36 = 1

theorem asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y → (y = (3/4) * x ∨ y = - (3/4) * x) :=
by
  intro x y
  intro h
  sorry

end asymptotes_equation_l149_149876


namespace find_costs_l149_149801

theorem find_costs (a b : ℝ) (h1 : a - b = 3) (h2 : 3 * b - 2 * a = 3) : a = 12 ∧ b = 9 :=
sorry

end find_costs_l149_149801


namespace longest_side_obtuse_triangle_l149_149126

theorem longest_side_obtuse_triangle (a b c : ℝ) (h₀ : a = 2) (h₁ : b = 4) 
  (h₂ : a^2 + b^2 < c^2) : 
  2 * Real.sqrt 5 < c ∧ c < 6 :=
by 
  sorry

end longest_side_obtuse_triangle_l149_149126


namespace order_of_even_function_l149_149914

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem order_of_even_function {f : ℝ → ℝ}
  (h_even : is_even f)
  (h_mono_inc : is_monotonically_increasing_on_nonneg f) :
  f (-π) > f (3) ∧ f (3) > f (-2) :=
sorry

end order_of_even_function_l149_149914


namespace pyramid_height_l149_149881

theorem pyramid_height (perimeter_side_base : ℝ) (apex_distance_to_vertex : ℝ) (height_peak_to_center_base : ℝ) : 
  (perimeter_side_base = 32) → (apex_distance_to_vertex = 12) → 
  height_peak_to_center_base = 4 * Real.sqrt 7 := 
  by
    sorry

end pyramid_height_l149_149881


namespace gcd_m_n_eq_one_l149_149941

/-- Mathematical definitions of m and n. --/
def m : ℕ := 123^2 + 235^2 + 347^2
def n : ℕ := 122^2 + 234^2 + 348^2

/-- Listing the conditions and deriving the result that gcd(m, n) = 1. --/
theorem gcd_m_n_eq_one : gcd m n = 1 :=
by sorry

end gcd_m_n_eq_one_l149_149941


namespace perpendicular_slope_l149_149720

-- Conditions
def slope_of_given_line : ℚ := 5 / 2

-- The statement
theorem perpendicular_slope (slope_of_given_line : ℚ) : (-1 / slope_of_given_line = -2 / 5) :=
by
  sorry

end perpendicular_slope_l149_149720


namespace common_property_of_rectangles_rhombuses_and_squares_l149_149976

-- Definitions of shapes and properties

-- Assume properties P1 = "Diagonals are equal", P2 = "Diagonals bisect each other", 
-- P3 = "Diagonals are perpendicular to each other", and P4 = "Diagonals bisect each other and are equal"

def is_rectangle (R : Type) : Prop := sorry
def is_rhombus (R : Type) : Prop := sorry
def is_square (R : Type) : Prop := sorry

def diagonals_bisect_each_other (R : Type) : Prop := sorry

-- Theorem stating the common property
theorem common_property_of_rectangles_rhombuses_and_squares 
  (R : Type)
  (H_rect : is_rectangle R)
  (H_rhomb : is_rhombus R)
  (H_square : is_square R) :
  diagonals_bisect_each_other R := 
  sorry

end common_property_of_rectangles_rhombuses_and_squares_l149_149976


namespace M_gt_N_l149_149237

variable (x y : ℝ)

def M := x^2 + y^2 + 1
def N := 2*x + 2*y - 2

theorem M_gt_N : M x y > N x y :=
by
  sorry

end M_gt_N_l149_149237


namespace circular_garden_radius_l149_149473

theorem circular_garden_radius
  (r : ℝ) -- radius of the circular garden
  (h : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) :
  r = 12 := 
by {
  sorry
}

end circular_garden_radius_l149_149473


namespace Cameron_list_count_l149_149519

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l149_149519


namespace inscribed_square_properties_l149_149587

theorem inscribed_square_properties (r : ℝ) (s : ℝ) (d : ℝ) (A_circle : ℝ) (A_square : ℝ) (total_diagonals : ℝ) (hA_circle : A_circle = 324 * Real.pi) (hr : r = Real.sqrt 324) (hd : d = 2 * r) (hs : s = d / Real.sqrt 2) (hA_square : A_square = s ^ 2) (htotal_diagonals : total_diagonals = 2 * d) :
  A_square = 648 ∧ total_diagonals = 72 :=
by sorry

end inscribed_square_properties_l149_149587


namespace intersection_of_sets_l149_149859
-- Define the sets and the proof statement
theorem intersection_of_sets : 
  let A := { x : ℝ | x^2 - 3 * x - 4 < 0 }
  let B := {-4, 1, 3, 5}
  A ∩ B = {1, 3} :=
by
  sorry

end intersection_of_sets_l149_149859


namespace parallel_lines_slope_eq_l149_149869

theorem parallel_lines_slope_eq {a : ℝ} : (∀ x : ℝ, 2*x - 1 = a*x + 1) → a = 2 :=
by
  sorry

end parallel_lines_slope_eq_l149_149869


namespace circle_equation_through_points_l149_149353

-- Definitions of the points A, B, and C
def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (-1, 1)

-- Prove that the equation of the circle passing through A, B, and C is (x - 1)^2 + y^2 = 5
theorem circle_equation_through_points :
  ∃ (D E F : ℝ), (∀ x y : ℝ, 
  x^2 + y^2 + D * x + E * y + F = 0 ↔
  x = -1 ∧ y = -1 ∨ 
  x = 2 ∧ y = 2 ∨ 
  x = -1 ∧ y = 1) ∧ 
  ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x - 1)^2 + y^2 = 5 :=
by
  sorry

end circle_equation_through_points_l149_149353


namespace find_m_l149_149592

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def f' (x : ℝ) : ℝ := -1 / (x^2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * x

theorem find_m (m : ℝ) :
  g 2 m = 1 / (f' 2) →
  m = -2 :=
by
  sorry

end find_m_l149_149592


namespace hyperbola_standard_equation_l149_149727

theorem hyperbola_standard_equation (a b : ℝ) :
  (∃ (P Q : ℝ × ℝ), P = (-3, 2 * Real.sqrt 7) ∧ Q = (-6 * Real.sqrt 2, -7) ∧
    (∀ x y b, y^2 / b^2 - x^2 / a^2 = 1 ∧ (2 * Real.sqrt 7)^2 / b^2 - (-3)^2 / a^2 = 1
    ∧ (-7)^2 / b^2 - (-6 * Real.sqrt 2)^2 / a^2 = 1)) →
  b^2 = 25 → a^2 = 75 →
  (∀ x y, y^2 / (25:ℝ) - x^2 / (75:ℝ) = 1) :=
sorry

end hyperbola_standard_equation_l149_149727


namespace tom_dimes_count_l149_149710

def originalDimes := 15
def dimesFromDad := 33
def dimesSpent := 11

theorem tom_dimes_count : originalDimes + dimesFromDad - dimesSpent = 37 := by
  sorry

end tom_dimes_count_l149_149710


namespace greatest_num_consecutive_integers_l149_149623

theorem greatest_num_consecutive_integers (N a : ℤ) (h : (N * (2*a + N - 1) = 210)) :
  ∃ N, N = 210 :=
sorry

end greatest_num_consecutive_integers_l149_149623


namespace problem_statement_l149_149604

def op (x y : ℝ) : ℝ := (x + 3) * (y - 1)

theorem problem_statement (a : ℝ) : (∀ x : ℝ, op (x - a) (x + a) > -16) ↔ -2 < a ∧ a < 6 :=
by
  sorry

end problem_statement_l149_149604


namespace area_of_parallelogram_l149_149618

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 18) (h_height : height = 16) : 
  base * height = 288 := 
by
  sorry

end area_of_parallelogram_l149_149618


namespace minimum_d_value_l149_149344

theorem minimum_d_value :
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  distance = 4 * d :=
by
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  sorry

end minimum_d_value_l149_149344


namespace inscribed_circle_radius_squared_l149_149084

theorem inscribed_circle_radius_squared 
  (X Y Z W R S : Type) 
  (XR RY WS SZ : ℝ)
  (hXR : XR = 23) 
  (hRY : RY = 29)
  (hWS : WS = 41) 
  (hSZ : SZ = 31)
  (tangent_at_XY : true) (tangent_at_WZ : true) -- since tangents are assumed by problem
  : ∃ (r : ℝ), r^2 = 905 :=
by sorry

end inscribed_circle_radius_squared_l149_149084


namespace inequality_proof_l149_149775

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * b * (b + 1) * (c + 1))) + 
  (1 / (b * c * (c + 1) * (a + 1))) + 
  (1 / (c * a * (a + 1) * (b + 1))) ≥ 
  (3 / (1 + a * b * c)^2) :=
sorry

end inequality_proof_l149_149775


namespace celsius_to_fahrenheit_conversion_l149_149867

theorem celsius_to_fahrenheit_conversion (k b : ℝ) :
  (∀ C : ℝ, (C * k + b = C * 1.8 + 32)) → (k = 1.8 ∧ b = 32) :=
by
  intro h
  sorry

end celsius_to_fahrenheit_conversion_l149_149867


namespace total_travel_time_is_correct_l149_149563

-- Conditions as definitions
def total_distance : ℕ := 200
def initial_fraction : ℚ := 1 / 4
def initial_time : ℚ := 1 -- in hours
def lunch_time : ℚ := 1 -- in hours
def remaining_fraction : ℚ := 1 / 2
def pit_stop_time : ℚ := 0.5 -- in hours
def speed_increase : ℚ := 10

-- Derived/Calculated values needed for the problem statement
def initial_distance : ℚ := initial_fraction * total_distance
def initial_speed : ℚ := initial_distance / initial_time
def remaining_distance : ℚ := total_distance - initial_distance
def half_remaining_distance : ℚ := remaining_fraction * remaining_distance
def second_drive_time : ℚ := half_remaining_distance / initial_speed
def last_distance : ℚ := remaining_distance - half_remaining_distance
def last_speed : ℚ := initial_speed + speed_increase
def last_drive_time : ℚ := last_distance / last_speed

-- Total time calculation
def total_time : ℚ :=
  initial_time + lunch_time + second_drive_time + pit_stop_time + last_drive_time

-- Lean theorem statement
theorem total_travel_time_is_correct : total_time = 5.25 :=
  sorry

end total_travel_time_is_correct_l149_149563


namespace solve_Mary_height_l149_149370

theorem solve_Mary_height :
  ∃ (m s : ℝ), 
  s = 150 ∧ 
  s * 1.2 = 180 ∧ 
  m = s + (180 - s) / 2 ∧ 
  m = 165 :=
by
  sorry

end solve_Mary_height_l149_149370


namespace find_m_l149_149891

theorem find_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = m) : m = 4 :=
sorry

end find_m_l149_149891


namespace more_birds_than_nests_l149_149639

theorem more_birds_than_nests (birds nests : Nat) (h_birds : birds = 6) (h_nests : nests = 3) : birds - nests = 3 :=
by
  sorry

end more_birds_than_nests_l149_149639


namespace determine_m_for_value_range_l149_149564

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + m

theorem determine_m_for_value_range :
  ∀ m : ℝ, (∀ x : ℝ, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end determine_m_for_value_range_l149_149564


namespace find_max_a_l149_149737

def f (a x : ℝ) := a * x^3 - x

theorem find_max_a (a : ℝ) (h : ∃ t : ℝ, |f a (t + 2) - f a t| ≤ 2 / 3) :
  a ≤ 4 / 3 :=
sorry

end find_max_a_l149_149737


namespace find_K_l149_149428

def satisfies_conditions (K m n h : ℕ) : Prop :=
  K ∣ (m^h - 1) ∧ K ∣ (n ^ ((m^h - 1) / K) + 1)

def odd (n : ℕ) : Prop := n % 2 = 1

theorem find_K (r : ℕ) (h : ℕ := 2^r) :
    ∀ K : ℕ, (∃ (m : ℕ), odd m ∧ m > 1 ∧ ∃ (n : ℕ), satisfies_conditions K m n h) ↔
    (∃ s t : ℕ, K = 2^(r + s) * t ∧ 2 ∣ t) := sorry

end find_K_l149_149428


namespace numbers_with_special_remainder_property_l149_149717

theorem numbers_with_special_remainder_property (n : ℕ) :
  (∀ q : ℕ, q > 0 → n % (q ^ 2) < (q ^ 2) / 2) ↔ (n = 1 ∨ n = 4) := 
by
  sorry

end numbers_with_special_remainder_property_l149_149717


namespace sequence_squared_l149_149047

theorem sequence_squared (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = a (n - 1) + 2 * (n - 1)) 
  : ∀ n, a n = n^2 := 
by
  sorry

end sequence_squared_l149_149047


namespace distribute_teachers_l149_149739

theorem distribute_teachers :
  let math_teachers := 3
  let lang_teachers := 3 
  let schools := 2
  let teachers_each_school := 3
  let distribution_plans := 
    (math_teachers.choose 2) * (lang_teachers.choose 1) + 
    (math_teachers.choose 1) * (lang_teachers.choose 2)
  distribution_plans = 18 := 
by
  sorry

end distribute_teachers_l149_149739


namespace hemisphere_surface_area_l149_149822

theorem hemisphere_surface_area (base_area : ℝ) (r : ℝ) (total_surface_area : ℝ) 
(h1: base_area = 64 * Real.pi) 
(h2: r^2 = 64)
(h3: total_surface_area = base_area + 2 * Real.pi * r^2) : 
total_surface_area = 192 * Real.pi := 
sorry

end hemisphere_surface_area_l149_149822


namespace inequality_proof_l149_149542

theorem inequality_proof (a b m n p : ℝ) (h1 : a > b) (h2 : m > n) (h3 : p > 0) : n - a * p < m - b * p :=
sorry

end inequality_proof_l149_149542


namespace distance_between_cars_after_third_checkpoint_l149_149657

theorem distance_between_cars_after_third_checkpoint
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (speed_after_first : ℝ)
  (speed_after_second : ℝ)
  (speed_after_third : ℝ)
  (distance_travelled : ℝ) :
  initial_distance = 100 →
  initial_speed = 60 →
  speed_after_first = 80 →
  speed_after_second = 100 →
  speed_after_third = 120 →
  distance_travelled = 200 :=
by
  sorry

end distance_between_cars_after_third_checkpoint_l149_149657


namespace coronavirus_diameter_scientific_notation_l149_149880

theorem coronavirus_diameter_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), a = 1.1 ∧ n = -7 ∧ 0.00000011 = a * 10^n := by
sorry

end coronavirus_diameter_scientific_notation_l149_149880


namespace total_visitors_over_two_days_l149_149981

-- Definitions of the conditions
def visitors_on_Saturday : ℕ := 200
def additional_visitors_on_Sunday : ℕ := 40

-- Statement of the problem
theorem total_visitors_over_two_days :
  let visitors_on_Sunday := visitors_on_Saturday + additional_visitors_on_Sunday
  let total_visitors := visitors_on_Saturday + visitors_on_Sunday
  total_visitors = 440 :=
by
  let visitors_on_Sunday := visitors_on_Saturday + additional_visitors_on_Sunday
  let total_visitors := visitors_on_Saturday + visitors_on_Sunday
  sorry

end total_visitors_over_two_days_l149_149981


namespace distinct_students_count_l149_149322

open Set

theorem distinct_students_count 
  (germain_students : ℕ := 15) 
  (newton_students : ℕ := 12) 
  (young_students : ℕ := 9)
  (overlap_students : ℕ := 3) :
  (germain_students + newton_students + young_students - overlap_students) = 33 := 
by
  sorry

end distinct_students_count_l149_149322


namespace collinear_example_l149_149600

structure Vector2D where
  x : ℝ
  y : ℝ

def collinear (u v : Vector2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.x = k * u.x ∧ v.y = k * u.y

def a : Vector2D := ⟨1, 2⟩
def b : Vector2D := ⟨2, 4⟩

theorem collinear_example :
  collinear a b :=
by
  sorry

end collinear_example_l149_149600


namespace vampire_pints_per_person_l149_149835

-- Definitions based on conditions
def gallons_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8
def days_per_week : ℕ := 7
def people_per_day : ℕ := 4

-- The statement to be proven
theorem vampire_pints_per_person :
  (gallons_per_week * pints_per_gallon) / (days_per_week * people_per_day) = 2 :=
by
  sorry

end vampire_pints_per_person_l149_149835


namespace elder_age_is_twenty_l149_149092

-- Let e be the present age of the elder person
-- Let y be the present age of the younger person

def ages_diff_by_twelve (e y : ℕ) : Prop :=
  e = y + 12

def elder_five_years_ago (e y : ℕ) : Prop :=
  e - 5 = 5 * (y - 5)

theorem elder_age_is_twenty (e y : ℕ) (h1 : ages_diff_by_twelve e y) (h2 : elder_five_years_ago e y) :
  e = 20 :=
by
  sorry

end elder_age_is_twenty_l149_149092


namespace rectangular_prism_dimensions_l149_149840

theorem rectangular_prism_dimensions 
    (a b c : ℝ) -- edges of the rectangular prism
    (h_increase_volume : (2 * a * b = 90)) -- condition 2: increasing height increases volume by 90 cm³ 
    (h_volume_proportion : (a * (c + 2)) / 2 = (3 / 5) * (a * b * c)) -- condition 3: height change results in 3/5 of original volume
    (h_edge_relation : (a = 5 * b ∨ b = 5 * a ∨ a * b = 45)) -- condition 1: one edge 5 times longer
    : 
    (a = 0.9 ∧ b = 50 ∧ c = 10) ∨ (a = 2 ∧ b = 22.5 ∧ c = 10) ∨ (a = 3 ∧ b = 15 ∧ c = 10) :=
sorry

end rectangular_prism_dimensions_l149_149840


namespace true_propositions_3_and_4_l149_149890

-- Define the condition for Proposition ③
def prop3_statement (m : ℝ) : Prop :=
  (m > 2) → ∀ x : ℝ, (x^2 - 2*x + m > 0)

def prop3_contrapositive (m : ℝ) : Prop :=
  (∀ x : ℝ, (x^2 - 2*x + m > 0)) → (m > 2)

-- Define the condition for Proposition ④
def prop4_condition (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (1 + x) = f (1 - x))

def prop4_period_4 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 4) = f (x))

-- Theorem to prove Propositions ③ and ④ are true
theorem true_propositions_3_and_4
  (m : ℝ) (f : ℝ → ℝ)
  (h3 : ∀ (m : ℝ), prop3_contrapositive m)
  (h4 : prop4_condition f): 
  prop3_statement m ∧ prop4_period_4 f :=
by {
  sorry
}

end true_propositions_3_and_4_l149_149890


namespace find_k1_over_k2_plus_k2_over_k1_l149_149545

theorem find_k1_over_k2_plus_k2_over_k1 (p q k k1 k2 : ℚ)
  (h1 : k * (p^2) - (2 * k - 3) * p + 7 = 0)
  (h2 : k * (q^2) - (2 * k - 3) * q + 7 = 0)
  (h3 : p ≠ 0)
  (h4 : q ≠ 0)
  (h5 : k ≠ 0)
  (h6 : k1 ≠ 0)
  (h7 : k2 ≠ 0)
  (h8 : p / q + q / p = 6 / 7)
  (h9 : (p + q) = (2 * k - 3) / k)
  (h10 : p * q = 7 / k)
  (h11 : k1 + k2 = 6)
  (h12 : k1 * k2 = 9 / 4) :
  (k1 / k2 + k2 / k1 = 14) :=
  sorry

end find_k1_over_k2_plus_k2_over_k1_l149_149545


namespace least_positive_linear_combination_l149_149223

theorem least_positive_linear_combination :
  ∃ x y z : ℤ, 0 < 24 * x + 20 * y + 12 * z ∧ ∀ n : ℤ, (∃ x y z : ℤ, n = 24 * x + 20 * y + 12 * z) → 0 < n → 4 ≤ n :=
by
  sorry

end least_positive_linear_combination_l149_149223


namespace problem_proof_l149_149075

theorem problem_proof (c d : ℝ) 
  (h1 : 5 + c = 6 - d) 
  (h2 : 6 + d = 9 + c) : 
  5 - c = 6 := 
sorry

end problem_proof_l149_149075


namespace sequence_integers_l149_149157

theorem sequence_integers (a : ℕ → ℤ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, n ≥ 3 → a n = (a (n-1)) ^ 2 + 2 / a (n-2)) : 
  ∀ n, ∃ k : ℤ, a n = k := 
by 
  sorry

end sequence_integers_l149_149157


namespace sum_lent_l149_149437

theorem sum_lent (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ) 
  (h1 : r = 0.06) (h2 : t = 8) (h3 : I = P - 520) : P * r * t = I → P = 1000 :=
by
  -- Given conditions
  intros
  -- Sorry placeholder
  sorry

end sum_lent_l149_149437


namespace average_speed_difference_l149_149568

noncomputable def v_R : Float := 56.44102863722254
noncomputable def distance : Float := 750
noncomputable def t_R : Float := distance / v_R
noncomputable def t_P : Float := t_R - 2
noncomputable def v_P : Float := distance / t_P

theorem average_speed_difference : v_P - v_R = 10 := by
  sorry

end average_speed_difference_l149_149568


namespace calculate_fraction_product_l149_149574

noncomputable def b8 := 2 * (8^2) + 6 * (8^1) + 2 * (8^0) -- 262_8 in base 10
noncomputable def b4 := 1 * (4^1) + 3 * (4^0) -- 13_4 in base 10
noncomputable def b7 := 1 * (7^2) + 4 * (7^1) + 4 * (7^0) -- 144_7 in base 10
noncomputable def b5 := 2 * (5^1) + 4 * (5^0) -- 24_5 in base 10

theorem calculate_fraction_product : 
  ((b8 : ℕ) / (b4 : ℕ)) * ((b7 : ℕ) / (b5 : ℕ)) = 147 :=
by
  sorry

end calculate_fraction_product_l149_149574


namespace option_A_option_B_option_D_l149_149048

-- Definitions of sequences
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a_1 + n * d

def geometric_seq (b_1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  b_1 * q ^ n

-- Option A: Prove that there exist d and q such that a_n = b_n
theorem option_A : ∃ (d q : ℤ), ∀ (a_1 b_1 : ℤ) (n : ℕ), 
  (arithmetic_seq a_1 d n = geometric_seq b_1 q n) := sorry

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Option B: Prove the differences form an arithmetic sequence
theorem option_B (a_1 : ℤ) (d : ℤ) :
  ∀ n k : ℕ, k > 0 → 
  (sum_arithmetic_seq a_1 d ((k + 1) * n) - sum_arithmetic_seq a_1 d (k * n) =
   (sum_arithmetic_seq a_1 d n + k * n * n * d)) := sorry

-- Option D: Prove there exist real numbers A and a such that A * a^a_n = b_n
theorem option_D (a_1 : ℤ) (d : ℤ) (b_1 : ℤ) (q : ℤ) :
  ∀ n : ℕ, b_1 > 0 → q > 0 → 
  ∃ A a : ℝ, A * a^ (arithmetic_seq a_1 d n) = (geometric_seq b_1 q n) := sorry

end option_A_option_B_option_D_l149_149048


namespace probability_of_same_suit_l149_149017

-- Definitions for the conditions
def total_cards : ℕ := 52
def suits : ℕ := 4
def cards_per_suit : ℕ := 13
def total_draws : ℕ := 2

-- Definition of factorial for binomial coefficient calculation
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Binomial coefficient calculation
def binomial_coeff (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Calculation of the probability
def prob_same_suit : ℚ :=
  let ways_to_choose_2_cards_from_52 := binomial_coeff total_cards total_draws
  let ways_to_choose_2_cards_per_suit := binomial_coeff cards_per_suit total_draws
  let total_ways_to_choose_2_same_suit := suits * ways_to_choose_2_cards_per_suit
  total_ways_to_choose_2_same_suit / ways_to_choose_2_cards_from_52

theorem probability_of_same_suit :
  prob_same_suit = 4 / 17 :=
by
  sorry

end probability_of_same_suit_l149_149017


namespace cuboid_length_l149_149445

variable (L W H V : ℝ)

theorem cuboid_length (W_eq : W = 4) (H_eq : H = 6) (V_eq : V = 96) (Volume_eq : V = L * W * H) : L = 4 :=
by
  sorry

end cuboid_length_l149_149445


namespace rabbits_initially_bought_l149_149743

theorem rabbits_initially_bought (R : ℕ) (h : ∃ (k : ℕ), R + 6 = 17 * k) : R = 28 :=
sorry

end rabbits_initially_bought_l149_149743


namespace B_work_time_l149_149280

theorem B_work_time :
  (∀ A_efficiency : ℝ, A_efficiency = 1 / 12 → ∀ B_efficiency : ℝ, B_efficiency = A_efficiency * 1.2 → (1 / B_efficiency = 10)) :=
by
  intros A_efficiency A_efficiency_eq B_efficiency B_efficiency_eq
  sorry

end B_work_time_l149_149280


namespace john_marks_wrongly_entered_as_l149_149858

-- Definitions based on the conditions
def john_correct_marks : ℤ := 62
def num_students : ℤ := 80
def avg_increase : ℤ := 1/2
def total_increase : ℤ := num_students * avg_increase

-- Statement to prove
theorem john_marks_wrongly_entered_as (x : ℤ) :
  (total_increase = (x - john_correct_marks)) → x = 102 :=
by {
  -- Placeholder for proof
  sorry
}

end john_marks_wrongly_entered_as_l149_149858


namespace range_of_m_l149_149796

noncomputable def unique_zero_point (m : ℝ) : Prop :=
  ∀ x : ℝ, m * (1/4)^x - (1/2)^x + 1 = 0 → ∀ x' : ℝ, m * (1/4)^x' - (1/2)^x' + 1 = 0 → x = x'

theorem range_of_m (m : ℝ) : unique_zero_point m → (m ≤ 0 ∨ m = 1/4) :=
sorry

end range_of_m_l149_149796


namespace area_of_rectangle_l149_149402

theorem area_of_rectangle (length : ℝ) (width : ℝ) (h_length : length = 47.3) (h_width : width = 24) : 
  length * width = 1135.2 := 
by 
  sorry

end area_of_rectangle_l149_149402


namespace ethel_subtracts_l149_149143

theorem ethel_subtracts (h : 50^2 = 2500) : 2500 - 99 = 49^2 :=
by
  sorry

end ethel_subtracts_l149_149143


namespace monthly_rent_of_shop_l149_149828

theorem monthly_rent_of_shop
  (length width : ℕ)
  (annual_rent_per_sq_ft : ℕ)
  (length_def : length = 18)
  (width_def : width = 22)
  (annual_rent_per_sq_ft_def : annual_rent_per_sq_ft = 68) :
  (18 * 22 * 68) / 12 = 2244 := 
by
  sorry

end monthly_rent_of_shop_l149_149828


namespace lateral_surface_area_of_cone_l149_149238

-- Definitions of the given conditions
def base_radius_cm : ℝ := 3
def slant_height_cm : ℝ := 5

-- The theorem to prove
theorem lateral_surface_area_of_cone :
  let r := base_radius_cm
  let l := slant_height_cm
  π * r * l = 15 * π := 
by
  sorry

end lateral_surface_area_of_cone_l149_149238


namespace find_divisor_l149_149209

theorem find_divisor
  (D dividend quotient remainder : ℤ)
  (h_dividend : dividend = 13787)
  (h_quotient : quotient = 89)
  (h_remainder : remainder = 14)
  (h_relation : dividend = (D * quotient) + remainder) :
  D = 155 :=
by
  sorry

end find_divisor_l149_149209


namespace train_speed_fraction_l149_149232

theorem train_speed_fraction (T : ℝ) (hT : T = 3) : T / (T + 0.5) = 6 / 7 := by
  sorry

end train_speed_fraction_l149_149232


namespace find_first_number_in_list_l149_149722

theorem find_first_number_in_list
  (x : ℕ)
  (h1 : x < 10)
  (h2 : ∃ n : ℕ, 2012 = x + 9 * n)
  : x = 5 :=
by
  sorry

end find_first_number_in_list_l149_149722


namespace remainder_5310_mod8_l149_149707

theorem remainder_5310_mod8 : (53 ^ 10) % 8 = 1 := 
by 
  sorry

end remainder_5310_mod8_l149_149707


namespace linear_polynomial_divisible_49_l149_149619

theorem linear_polynomial_divisible_49 {P : ℕ → Polynomial ℚ} :
    let Q := Polynomial.C 1 * (Polynomial.X ^ 8) + Polynomial.C 1 * (Polynomial.X ^ 7)
    ∃ a b x, (P x) = Polynomial.C a * Polynomial.X + Polynomial.C b ∧ a ≠ 0 ∧ 
              (∀ i, P (i + 1) = (Polynomial.C 1 * Polynomial.X + Polynomial.C 1) * P i ∨ 
                            P (i + 1) = Polynomial.derivative (P i)) →
              (a - b) % 49 = 0 :=
by
  sorry

end linear_polynomial_divisible_49_l149_149619


namespace numbers_to_be_left_out_l149_149044

axiom problem_conditions :
  let numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]
  let grid_numbers := [1, 9, 14, 5]
  numbers.sum + grid_numbers.sum = 106 ∧
  ∃ (left_out : ℕ) (remaining_numbers : List ℕ),
    numbers.erase left_out = remaining_numbers ∧
    (numbers.sum + grid_numbers.sum - left_out) = 96 ∧
    remaining_numbers.length = 8

theorem numbers_to_be_left_out :
  let numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]
  10 ∈ numbers ∧
  let grid_numbers := [1, 9, 14, 5]
  let total_sum := numbers.sum + grid_numbers.sum
  let grid_sum := total_sum - 10
  grid_sum % 12 = 0 ∧
  grid_sum = 96 :=
sorry

end numbers_to_be_left_out_l149_149044


namespace cos_double_angle_l149_149363

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 1 / 2) : Real.cos (2 * α) = 3 / 5 :=
by sorry

end cos_double_angle_l149_149363


namespace katy_read_books_l149_149742

theorem katy_read_books (juneBooks : ℕ) (julyBooks : ℕ) (augustBooks : ℕ)
  (H1 : juneBooks = 8)
  (H2 : julyBooks = 2 * juneBooks)
  (H3 : augustBooks = julyBooks - 3) :
  juneBooks + julyBooks + augustBooks = 37 := by
  -- Proof goes here
  sorry

end katy_read_books_l149_149742


namespace find_f_half_l149_149082

theorem find_f_half (f : ℝ → ℝ) (h : ∀ x, f (2 * x / (x + 1)) = x^2 - 1) : f (1 / 2) = -8 / 9 :=
by
  sorry

end find_f_half_l149_149082


namespace sum_of_roots_eq_neg3_l149_149327

theorem sum_of_roots_eq_neg3
  (a b c : ℝ)
  (h_eq : 2 * x^2 + 6 * x - 1 = 0)
  (h_a : a = 2)
  (h_b : b = 6) :
  (x1 x2 : ℝ) → x1 + x2 = -b / a :=
by
  sorry

end sum_of_roots_eq_neg3_l149_149327


namespace tan_alpha_fraction_value_l149_149161

theorem tan_alpha_fraction_value {α : Real} (h : Real.tan α = 2) : 
  (3 * Real.sin α + Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = 7 / 12 :=
by
  sorry

end tan_alpha_fraction_value_l149_149161


namespace find_gamma_k_l149_149399

noncomputable def alpha (n d : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def beta (n r : ℕ) : ℕ := r^(n - 1)
noncomputable def gamma (n d r : ℕ) : ℕ := alpha n d + beta n r

theorem find_gamma_k (k d r : ℕ) (hk1 : gamma (k-1) d r = 200) (hk2 : gamma (k+1) d r = 2000) :
    gamma k d r = 387 :=
sorry

end find_gamma_k_l149_149399


namespace opposite_of_2023_l149_149843

def opposite (n : Int) : Int := -n

theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l149_149843


namespace union_of_M_and_N_is_correct_l149_149120

def M : Set ℤ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 ≤ n ∧ n ≤ 3 }

theorem union_of_M_and_N_is_correct : M ∪ N = { -2, -1, 0, 1, 2, 3 } := 
by
  sorry

end union_of_M_and_N_is_correct_l149_149120


namespace math_students_but_not_science_l149_149355

theorem math_students_but_not_science (total_students : ℕ) (students_math : ℕ) (students_science : ℕ)
  (students_both : ℕ) (students_math_three_times : ℕ) :
  total_students = 30 ∧ students_both = 2 ∧ students_math = 3 * students_science ∧ 
  students_math = students_both + (22 - 2) → (students_math - students_both = 20) :=
by
  sorry

end math_students_but_not_science_l149_149355


namespace same_side_of_line_l149_149419

open Real

theorem same_side_of_line (a : ℝ) :
  let O := (0, 0)
  let A := (1, 1)
  (O.1 + O.2 < a ↔ A.1 + A.2 < a) →
  a < 0 ∨ a > 2 := by
  sorry

end same_side_of_line_l149_149419


namespace cos_F_l149_149492

theorem cos_F (D E F : ℝ) (hDEF : D + E + F = 180)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = -16 / 65 :=
  sorry

end cos_F_l149_149492


namespace train_length_at_constant_acceleration_l149_149112

variables (u : ℝ) (t : ℝ) (a : ℝ) (s : ℝ)

theorem train_length_at_constant_acceleration (h₁ : u = 16.67) (h₂ : t = 30) : 
  s = u * t + 0.5 * a * t^2 :=
sorry

end train_length_at_constant_acceleration_l149_149112


namespace product_formula_l149_149862

theorem product_formula :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) *
  (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) *
  (3^128 + 5^128) = 3^256 - 5^256 := by
  sorry

end product_formula_l149_149862


namespace sum_of_possible_values_l149_149602

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) : (∃ x y : ℝ, x * (x - 4) = -21 ∧ y * (y - 4) = -21 ∧ x + y = 4) :=
sorry

end sum_of_possible_values_l149_149602


namespace parallelogram_area_l149_149151

open Real

def line1 (p : ℝ × ℝ) : Prop := p.2 = 2
def line2 (p : ℝ × ℝ) : Prop := p.2 = -2
def line3 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 - 10 = 0
def line4 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 + 20 = 0

theorem parallelogram_area :
  ∃ D : ℝ, D = 30 ∧
  (∀ p : ℝ × ℝ, line1 p ∨ line2 p ∨ line3 p ∨ line4 p) :=
sorry

end parallelogram_area_l149_149151


namespace area_of_triangle_KDC_l149_149016

open Real

noncomputable def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

theorem area_of_triangle_KDC
  (radius : ℝ) (chord_length : ℝ) (seg_KA : ℝ)
  (OX distance_DY : ℝ)
  (parallel : ∀ (PA PB : ℝ), PA = PB)
  (collinear : ∀ (PK PA PQ PB : ℝ), PK + PA + PQ + PB = PK + PQ + PA + PB)
  (hyp_radius : radius = 10)
  (hyp_chord_length : chord_length = 12)
  (hyp_seg_KA : seg_KA = 24)
  (hyp_OX : OX = 8)
  (hyp_distance_DY : distance_DY = 8) :
  triangle_area chord_length distance_DY = 48 :=
  by
  sorry

end area_of_triangle_KDC_l149_149016


namespace shaded_fraction_is_one_fourth_l149_149687

def quilt_block_shaded_fraction : ℚ :=
  let total_unit_squares := 16
  let triangles_per_unit_square := 2
  let shaded_triangles := 8
  let shaded_unit_squares := shaded_triangles / triangles_per_unit_square
  shaded_unit_squares / total_unit_squares

theorem shaded_fraction_is_one_fourth :
  quilt_block_shaded_fraction = 1 / 4 :=
sorry

end shaded_fraction_is_one_fourth_l149_149687


namespace face_opposite_to_A_l149_149851

-- Define the faces and their relationships
inductive Face : Type
| A | B | C | D | E | F
open Face

def adjacent (x y : Face) : Prop :=
  match x, y with
  | A, B => true
  | B, A => true
  | C, A => true
  | A, C => true
  | D, A => true
  | A, D => true
  | C, D => true
  | D, C => true
  | E, F => true
  | F, E => true
  | _, _ => false

-- Theorem stating that "F" is opposite to "A" given the provided conditions.
theorem face_opposite_to_A : ∀ x : Face, (adjacent A x = false) → (x = B ∨ x = C ∨ x = D → false) → (x = E ∨ x = F) → x = F := 
  by
    intros x h1 h2 h3
    sorry

end face_opposite_to_A_l149_149851


namespace weight_of_replaced_student_l149_149058

theorem weight_of_replaced_student (W : ℝ) : 
  (W - 12 = 5 * 12) → W = 72 :=
by
  intro hyp
  linarith

end weight_of_replaced_student_l149_149058


namespace largest_reciprocal_l149_149606

theorem largest_reciprocal: 
  let A := -(1 / 4)
  let B := 2 / 7
  let C := -2
  let D := 3
  let E := -(3 / 2)
  let reciprocal (x : ℚ) := 1 / x
  reciprocal B > reciprocal A ∧
  reciprocal B > reciprocal C ∧
  reciprocal B > reciprocal D ∧
  reciprocal B > reciprocal E :=
by
  sorry

end largest_reciprocal_l149_149606


namespace raft_capacity_l149_149059

theorem raft_capacity (total_without_life_jackets : ℕ) (reduction_with_life_jackets : ℕ)
  (people_needing_life_jackets : ℕ) (total_capacity_with_life_jackets : ℕ)
  (no_life_jackets_capacity : total_without_life_jackets = 21)
  (life_jackets_reduction : reduction_with_life_jackets = 7)
  (life_jackets_needed : people_needing_life_jackets = 8) :
  total_capacity_with_life_jackets = 14 :=
by
  -- Proof should be here
  sorry

end raft_capacity_l149_149059


namespace sandwiches_per_person_l149_149837

open Nat

theorem sandwiches_per_person (total_sandwiches : ℕ) (total_people : ℕ) (h1 : total_sandwiches = 657) (h2 : total_people = 219) : 
(total_sandwiches / total_people) = 3 :=
by
  -- a proof would go here
  sorry

end sandwiches_per_person_l149_149837


namespace minimum_value_x_plus_3y_plus_6z_l149_149701

theorem minimum_value_x_plus_3y_plus_6z 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h4 : x * y * z = 18) : 
  x + 3 * y + 6 * z ≥ 3 * (2 * Real.sqrt 6 + 1) :=
sorry

end minimum_value_x_plus_3y_plus_6z_l149_149701


namespace rearrangement_count_correct_l149_149105

def original_number := "1234567890"

def is_valid_rearrangement (n : String) : Prop :=
  n.length = 10 ∧ n.front ≠ '0'
  
def count_rearrangements (n : String) : ℕ :=
  if n = original_number 
  then 232
  else 0

theorem rearrangement_count_correct :
  count_rearrangements original_number = 232 :=
sorry


end rearrangement_count_correct_l149_149105


namespace x_y_sum_cube_proof_l149_149141

noncomputable def x_y_sum_cube (x y : ℝ) : ℝ := x^3 + y^3

theorem x_y_sum_cube_proof (x y : ℝ) (hx : 1 < x) (hy : 1 < y)
  (h_eq : (Real.log x / Real.log 2)^3 + (Real.log y / Real.log 3)^3 = 3 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x_y_sum_cube x y = 307 :=
sorry

end x_y_sum_cube_proof_l149_149141


namespace minimum_prime_factorization_sum_l149_149349

theorem minimum_prime_factorization_sum (x y a b c d : ℕ) (hx : x > 0) (hy : y > 0)
  (h : 5 * x^7 = 13 * y^17) (h_pf: x = a ^ c * b ^ d) :
  a + b + c + d = 33 :=
sorry

end minimum_prime_factorization_sum_l149_149349


namespace general_term_of_sequence_l149_149768

variable (a : ℕ → ℕ)
variable (h1 : ∀ m : ℕ, a (m^2) = a m ^ 2)
variable (h2 : ∀ m k : ℕ, a (m^2 + k^2) = a m * a k)

theorem general_term_of_sequence : ∀ n : ℕ, n > 0 → a n = 1 :=
by
  intros n hn
  sorry

end general_term_of_sequence_l149_149768


namespace flour_needed_l149_149548

-- Definitions
def cups_per_loaf := 2.5
def loaves := 2

-- Statement we want to prove
theorem flour_needed {cups_per_loaf loaves : ℝ} (h : cups_per_loaf = 2.5) (l : loaves = 2) : 
  cups_per_loaf * loaves = 5 :=
sorry

end flour_needed_l149_149548


namespace brendan_yards_per_week_l149_149250

def original_speed_flat : ℝ := 8  -- Brendan's speed on flat terrain in yards/day
def improvement_flat : ℝ := 0.5   -- Lawn mower improvement on flat terrain (50%)
def reduction_uneven : ℝ := 0.35  -- Speed reduction on uneven terrain (35%)
def days_flat : ℝ := 4            -- Days on flat terrain
def days_uneven : ℝ := 3          -- Days on uneven terrain

def improved_speed_flat : ℝ := original_speed_flat * (1 + improvement_flat)
def speed_uneven : ℝ := improved_speed_flat * (1 - reduction_uneven)

def total_yards_week : ℝ := (improved_speed_flat * days_flat) + (speed_uneven * days_uneven)

theorem brendan_yards_per_week : total_yards_week = 71.4 :=
sorry

end brendan_yards_per_week_l149_149250


namespace find_period_l149_149046

theorem find_period (A P R : ℕ) (I : ℕ) (T : ℚ) 
  (hA : A = 1120) 
  (hP : P = 896) 
  (hR : R = 5) 
  (hSI : I = A - P) 
  (hT : I = (P * R * T) / 100) :
  T = 5 := by 
  sorry

end find_period_l149_149046


namespace part1_part2_l149_149079

-- Definition of points and given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Conditions for part 1
def A1 (a : ℝ) : Point := { x := -2, y := a + 1 }
def B1 (a : ℝ) : Point := { x := a - 1, y := 4 }

-- Definition for distance calculation
def distance (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)

-- Problem 1 Statement
theorem part1 (a : ℝ) (h : a = 3) : distance (A1 a) (B1 a) = 4 :=
by 
  sorry

-- Conditions for part 2
def C2 (b : ℝ) : Point := { x := b - 2, y := b }

-- Problem 2 Statement
theorem part2 (b : ℝ) (h : abs b = 1) :
  (C2 b = { x := -1, y := 1 } ∨ C2 b = { x := -3, y := -1 }) :=
by
  sorry

end part1_part2_l149_149079


namespace derivative_f_l149_149333

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by
  sorry

end derivative_f_l149_149333


namespace range_of_b_l149_149214

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x - a) / (2^x + 1)
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := Real.log (x^2 - b)

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x1 x2 : ℝ, f x1 a ≤ g x2 b) → b ≤ -Real.exp 1 :=
by
  sorry

end range_of_b_l149_149214


namespace third_rectangle_area_l149_149377

-- Definitions for dimensions of the first two rectangles
def rect1_length := 3
def rect1_width := 8

def rect2_length := 2
def rect2_width := 5

-- Total area of the first two rectangles
def total_area := (rect1_length * rect1_width) + (rect2_length * rect2_width)

-- Declaration of the theorem to be proven
theorem third_rectangle_area :
  ∃ a b : ℝ, a * b = 4 ∧ total_area + a * b = total_area + 4 :=
by
  sorry

end third_rectangle_area_l149_149377


namespace air_conditioner_sales_l149_149226

/-- Represent the conditions -/
def conditions (x y m : ℕ) : Prop :=
  (3 * x + 5 * y = 23500) ∧
  (4 * x + 10 * y = 42000) ∧
  (x = 2500) ∧
  (y = 3200) ∧
  (700 * (50 - m) + 800 * m ≥ 38000)

/-- Prove that the unit selling prices of models A and B are 2500 yuan and 3200 yuan respectively,
    and at least 30 units of model B need to be purchased for a profit of at least 38000 yuan,
    given the conditions. -/
theorem air_conditioner_sales :
  ∃ (x y m : ℕ), conditions x y m ∧ m ≥ 30 := by
  sorry

end air_conditioner_sales_l149_149226


namespace longest_side_of_rectangle_l149_149541

theorem longest_side_of_rectangle 
    (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 2400) : 
    max l w = 80 :=
by sorry

end longest_side_of_rectangle_l149_149541


namespace similar_triangles_height_ratio_l149_149807

-- Given condition: two similar triangles have a similarity ratio of 3:5
def similar_triangles (ratio : ℕ) : Prop := ratio = 3 ∧ ratio = 5

-- Goal: What is the ratio of their corresponding heights?
theorem similar_triangles_height_ratio (r : ℕ) (h : similar_triangles r) :
  r = 3 / 5 :=
sorry

end similar_triangles_height_ratio_l149_149807


namespace range_of_a_l149_149668

variable (a : ℝ)

-- Definitions of propositions p and q
def p := ∀ x : ℝ, x^2 - 2*x - a ≥ 0
def q := ∃ x : ℝ, x^2 + x + 2*a - 1 ≤ 0

-- Lean 4 statement of the proof problem
theorem range_of_a : ¬ p a ∧ q a → -1 < a ∧ a ≤ 5/8 := by
  sorry

end range_of_a_l149_149668


namespace store_discount_difference_l149_149603

theorem store_discount_difference 
  (p : ℝ) -- original price
  (p1 : ℝ := p * 0.60) -- price after initial discount
  (p2 : ℝ := p1 * 0.90) -- price after additional discount
  (claimed_discount : ℝ := 0.55) -- store's claimed discount
  (true_discount : ℝ := (p - p2) / p) -- calculated true discount
  (difference : ℝ := claimed_discount - true_discount)
  : difference = 0.09 :=
sorry

end store_discount_difference_l149_149603


namespace part1_solution_sets_part2_solution_set_l149_149074

-- Define the function f(x)
def f (a x : ℝ) := x^2 + (1 - a) * x - a

-- Statement for part (1)
theorem part1_solution_sets (a x : ℝ) :
  (a < -1 → f a x < 0 ↔ a < x ∧ x < -1) ∧
  (a = -1 → ¬ (f a x < 0)) ∧
  (a > -1 → f a x < 0 ↔ -1 < x ∧ x < a) :=
sorry

-- Statement for part (2)
theorem part2_solution_set (x : ℝ) :
  (f 2 x) > 0 → (x^3 * f 2 x > 0 ↔ (-1 < x ∧ x < 0) ∨ 2 < x) :=
sorry

end part1_solution_sets_part2_solution_set_l149_149074


namespace car_wash_cost_l149_149097

-- Definitions based on the conditions
def washes_per_bottle : ℕ := 4
def bottle_cost : ℕ := 4   -- Assuming cost is recorded in dollars
def total_weeks : ℕ := 20

-- Stating the problem
theorem car_wash_cost : (total_weeks / washes_per_bottle) * bottle_cost = 20 := 
by
  -- Placeholder for the proof
  sorry

end car_wash_cost_l149_149097


namespace friend_decks_l149_149772

noncomputable def cost_per_deck : ℕ := 8
noncomputable def victor_decks : ℕ := 6
noncomputable def total_amount_spent : ℕ := 64

theorem friend_decks :
  ∃ x : ℕ, (victor_decks * cost_per_deck) + (x * cost_per_deck) = total_amount_spent ∧ x = 2 :=
by
  sorry

end friend_decks_l149_149772


namespace fraction_calls_processed_by_team_B_l149_149762

variable (A B C_A C_B : ℕ)

theorem fraction_calls_processed_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : C_A = (2 / 5) * C_B) :
  (B * C_B) / ((A * C_A) + (B * C_B)) = 8 / 9 := by
  sorry

end fraction_calls_processed_by_team_B_l149_149762


namespace obtuse_angles_in_second_quadrant_l149_149014

theorem obtuse_angles_in_second_quadrant
  (θ : ℝ) 
  (is_obtuse : θ > 90 ∧ θ < 180) :
  90 < θ ∧ θ < 180 :=
by sorry

end obtuse_angles_in_second_quadrant_l149_149014


namespace solution_exists_l149_149719

theorem solution_exists :
  ∃ x : ℝ, x = 2 ∧ (-2 * x + 4 = 0) :=
sorry

end solution_exists_l149_149719


namespace polynomial_evaluation_l149_149957

noncomputable def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem polynomial_evaluation : f 2 = 123 := by
  sorry

end polynomial_evaluation_l149_149957


namespace total_value_of_coins_l149_149231

theorem total_value_of_coins (h1 : ∀ (q d : ℕ), q + d = 23)
                             (h2 : ∀ q, q = 16)
                             (h3 : ∀ d, d = 23 - 16)
                             (h4 : ∀ q, q * 0.25 = 4.00)
                             (h5 : ∀ d, d * 0.10 = 0.70)
                             : 4.00 + 0.70 = 4.70 :=
by
  sorry

end total_value_of_coins_l149_149231


namespace problem_statement_l149_149278

def S (a b : ℤ) : ℤ := 4 * a + 6 * b
def T (a b : ℤ) : ℤ := 2 * a - 3 * b

theorem problem_statement : T (S 8 3) 4 = 88 := by
  sorry

end problem_statement_l149_149278


namespace recipe_total_cups_l149_149591

noncomputable def total_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℕ) : ℕ :=
  let part := sugar_cups / sugar_ratio
  let butter_cups := butter_ratio * part
  let flour_cups := flour_ratio * part
  butter_cups + flour_cups + sugar_cups

theorem recipe_total_cups : 
  total_cups 2 7 5 10 = 28 :=
by
  sorry

end recipe_total_cups_l149_149591


namespace parabola_chord_length_l149_149030

theorem parabola_chord_length (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) (h2 : y2^2 = 4 * x2) 
  (hx : x1 + x2 = 9) 
  (focus_line : ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b → y^2 = 4 * x) :
  |(x1 - 1, y1) - (x2 - 1, y2)| = 11 := 
sorry

end parabola_chord_length_l149_149030


namespace sum_ages_l149_149580

variable (Bob_age Carol_age : ℕ)

theorem sum_ages (h1 : Bob_age = 16) (h2 : Carol_age = 50) (h3 : Carol_age = 3 * Bob_age + 2) :
  Bob_age + Carol_age = 66 :=
by
  sorry

end sum_ages_l149_149580


namespace min_value_problem_l149_149033

theorem min_value_problem (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 57 * a + 88 * b + 125 * c ≥ 1148) :
  240 ≤ a^3 + b^3 + c^3 + 5 * a^2 + 5 * b^2 + 5 * c^2 :=
sorry

end min_value_problem_l149_149033


namespace determine_x_l149_149524

theorem determine_x (x : ℚ) : 
  x + 5 / 8 = 2 + 3 / 16 - 2 / 3 → 
  x = 43 / 48 := 
by
  intro h
  sorry

end determine_x_l149_149524


namespace reduced_price_after_discount_l149_149041

theorem reduced_price_after_discount (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 1500 / R - 1500 / P = 10) :
  R = 30 := 
by
  sorry

end reduced_price_after_discount_l149_149041


namespace driver_speed_ratio_l149_149948

theorem driver_speed_ratio (V1 V2 x : ℝ) (h : V1 > 0 ∧ V2 > 0 ∧ x > 0)
  (meet_halfway : ∀ t1 t2, t1 = x / (2 * V1) ∧ t2 = x / (2 * V2))
  (earlier_start : ∀ t1 t2, t1 = t2 + x / (2 * (V1 + V2))) :
  V2 / V1 = (1 + Real.sqrt 5) / 2 := by
  sorry

end driver_speed_ratio_l149_149948


namespace trig_eq_solution_l149_149868

open Real

theorem trig_eq_solution (x : ℝ) : 
  (cos (7 * x) + cos (3 * x) + sin (7 * x) - sin (3 * x) + sqrt 2 * cos (4 * x) = 0) ↔ 
  (∃ k : ℤ, 
    (x = -π / 8 + π * k / 2) ∨ 
    (x = -π / 4 + 2 * π * k / 3) ∨ 
    (x = 3 * π / 28 + 2 * π * k / 7)) :=
by sorry

end trig_eq_solution_l149_149868


namespace number_of_children_per_seat_l149_149842

variable (children : ℕ) (seats : ℕ)

theorem number_of_children_per_seat (h1 : children = 58) (h2 : seats = 29) :
  children / seats = 2 := by
  sorry

end number_of_children_per_seat_l149_149842


namespace habitat_limits_are_correct_l149_149446

-- Definitions of the conditions
def colonyA_doubling_days : ℕ := 22
def colonyB_tripling_days : ℕ := 30
def tripling_interval : ℕ := 2

-- Definitions to confirm they grow as described
def is_colonyA_habitat_limit_reached (days : ℕ) : Prop := days = colonyA_doubling_days
def is_colonyB_habitat_limit_reached (days : ℕ) : Prop := days = colonyB_tripling_days

-- Proof statement
theorem habitat_limits_are_correct :
  (is_colonyA_habitat_limit_reached colonyA_doubling_days) ∧ (is_colonyB_habitat_limit_reached colonyB_tripling_days) :=
by
  sorry

end habitat_limits_are_correct_l149_149446


namespace rebus_solution_l149_149852

-- We state the conditions:
variables (A B Γ D : ℤ)

-- Define the correct values
def A_correct := 2
def B_correct := 7
def Γ_correct := 1
def D_correct := 0

-- State the conditions as assumptions
axiom cond1 : A * B + 8 = 3 * B
axiom cond2 : Γ * D + B = 5  -- Adjusted assuming V = 5 from problem data
axiom cond3 : Γ * B + 3 = A * D

-- State the goal to be proved
theorem rebus_solution : A = A_correct ∧ B = B_correct ∧ Γ = Γ_correct ∧ D = D_correct :=
by
  sorry

end rebus_solution_l149_149852


namespace speed_of_train_is_correct_l149_149283

noncomputable def speedOfTrain := 
  let lengthOfTrain : ℝ := 800 -- length of the train in meters
  let timeToCrossMan : ℝ := 47.99616030717543 -- time in seconds to cross the man
  let speedOfMan : ℝ := 5 * (1000 / 3600) -- speed of the man in m/s (conversion from km/hr to m/s)
  let relativeSpeed : ℝ := lengthOfTrain / timeToCrossMan -- relative speed of the train
  let speedOfTrainInMS : ℝ := relativeSpeed + speedOfMan -- speed of the train in m/s
  let speedOfTrainInKMHR : ℝ := speedOfTrainInMS * (3600 / 1000) -- speed in km/hr
  64.9848 -- result is approximately 64.9848 km/hr

theorem speed_of_train_is_correct :
  speedOfTrain = 64.9848 :=
by
  sorry

end speed_of_train_is_correct_l149_149283


namespace sum_powers_l149_149982

open Complex

theorem sum_powers (ω : ℂ) (h₁ : ω^5 = 1) (h₂ : ω ≠ 1) : 
  ω^10 + ω^12 + ω^14 + ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 = 1 := sorry

end sum_powers_l149_149982


namespace problem_1_problem_2_problem_3_l149_149073

open Real

theorem problem_1 : (1 * (-12)) - (-20) + (-8) - 15 = -15 := by
  sorry

theorem problem_2 : -3^2 + ((2/3) - (1/2) + (5/8)) * (-24) = -28 := by
  sorry

theorem problem_3 : -1^(2023) + 3 * (-2)^2 - (-6) / ((-1/3)^2) = 65 := by
  sorry

end problem_1_problem_2_problem_3_l149_149073


namespace pages_per_inch_l149_149986

theorem pages_per_inch (number_of_books : ℕ) (average_pages_per_book : ℕ) (total_thickness : ℕ) 
                        (H1 : number_of_books = 6)
                        (H2 : average_pages_per_book = 160)
                        (H3 : total_thickness = 12) :
  (number_of_books * average_pages_per_book) / total_thickness = 80 :=
by
  -- Placeholder for proof
  sorry

end pages_per_inch_l149_149986


namespace ellipse_semi_minor_axis_l149_149549

theorem ellipse_semi_minor_axis (b : ℝ) 
    (h1 : 0 < b) 
    (h2 : b < 5)
    (h_ellipse : ∀ x y : ℝ, x^2 / 25 + y^2 / b^2 = 1) 
    (h_eccentricity : 4 / 5 = 4 / 5) : b = 3 := 
sorry

end ellipse_semi_minor_axis_l149_149549


namespace exponentiation_rule_l149_149329

variable {a b : ℝ}

theorem exponentiation_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end exponentiation_rule_l149_149329


namespace roots_of_polynomial_l149_149635

theorem roots_of_polynomial :
  ∀ x : ℝ, (2 * x^3 - 3 * x^2 - 13 * x + 10) * (x - 1) = 0 → x = 1 :=
by
  sorry

end roots_of_polynomial_l149_149635


namespace gcd_g_x_l149_149410

def g (x : ℕ) : ℕ := (5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)

theorem gcd_g_x (x : ℕ) (hx : 17280 ∣ x) : Nat.gcd (g x) x = 120 :=
by sorry

end gcd_g_x_l149_149410


namespace inequality_problem_l149_149429

theorem inequality_problem (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by {
  sorry
}

end inequality_problem_l149_149429


namespace ivy_has_20_collectors_dolls_l149_149969

theorem ivy_has_20_collectors_dolls
  (D : ℕ) (I : ℕ) (C : ℕ)
  (h1 : D = 60)
  (h2 : D = 2 * I)
  (h3 : C = 2 * I / 3) 
  : C = 20 :=
by sorry

end ivy_has_20_collectors_dolls_l149_149969


namespace quotient_remainder_scaled_l149_149247

theorem quotient_remainder_scaled (a b q r k : ℤ) (hb : b > 0) (hk : k ≠ 0) (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) :
  a * k = (b * k) * q + (r * k) ∧ (k ∣ r → (a / k = (b / k) * q + (r / k) ∧ 0 ≤ (r / k) ∧ (r / k) < (b / k))) :=
by
  sorry

end quotient_remainder_scaled_l149_149247


namespace unique_solutions_xy_l149_149640

theorem unique_solutions_xy (x y : ℝ) : 
  x^3 + y^3 = 1 ∧ x^4 + y^4 = 1 ↔ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end unique_solutions_xy_l149_149640


namespace new_tax_rate_is_correct_l149_149575

noncomputable def new_tax_rate (old_rate : ℝ) (income : ℝ) (savings : ℝ) : ℝ := 
  let old_tax := old_rate * income / 100
  let new_tax := (income - savings) / income * old_tax
  let rate := new_tax / income * 100
  rate

theorem new_tax_rate_is_correct :
  ∀ (income : ℝ) (old_rate : ℝ) (savings : ℝ),
    old_rate = 42 →
    income = 34500 →
    savings = 4830 →
    new_tax_rate old_rate income savings = 28 := 
by
  intros income old_rate savings h1 h2 h3
  sorry

end new_tax_rate_is_correct_l149_149575


namespace initial_ratio_milk_water_l149_149439

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 60) 
  (h2 : ∀ k, k = M → M * 2 = W + 60) : (M:ℚ) / (W:ℚ) = 4 / 1 :=
by
  sorry

end initial_ratio_milk_water_l149_149439


namespace simplify_144_over_1296_times_36_l149_149413

theorem simplify_144_over_1296_times_36 :
  (144 / 1296) * 36 = 4 :=
by
  sorry

end simplify_144_over_1296_times_36_l149_149413


namespace solve_complex_problem_l149_149032

-- Define the problem
def complex_sum_eq_two (a b : ℝ) (i : ℂ) : Prop :=
  a + b = 2

-- Define the conditions
def conditions (a b : ℝ) (i : ℂ) : Prop :=
  a + b * i = (1 - i) * (2 + i)

-- State the theorem
theorem solve_complex_problem (a b : ℝ) (i : ℂ) (h : conditions a b i) : complex_sum_eq_two a b i :=
by
  sorry -- Proof goes here

end solve_complex_problem_l149_149032


namespace total_time_taken_l149_149938

theorem total_time_taken 
  (R : ℝ) -- Rickey's speed
  (T_R : ℝ := 40) -- Rickey's time
  (T_P : ℝ := (40 * (4 / 3))) -- Prejean's time derived from given conditions
  (P : ℝ := (3 / 4) * R) -- Prejean's speed
  (k : ℝ := 40 * R) -- constant k for distance
 
  (h1 : T_R = 40)
  (h2 : T_P = 40 * (4 / 3))
  -- Main goal: Prove total time taken equals 93.33 minutes
  : (T_R + T_P) = 93.33 := 
  sorry

end total_time_taken_l149_149938


namespace max_distance_curve_line_l149_149979

noncomputable def curve_param_x (θ : ℝ) : ℝ := 1 + Real.cos θ
noncomputable def curve_param_y (θ : ℝ) : ℝ := Real.sin θ
noncomputable def line (x y : ℝ) : Prop := x + y + 2 = 0

theorem max_distance_curve_line 
  (θ : ℝ) 
  (x := curve_param_x θ) 
  (y := curve_param_y θ) :
  ∃ (d : ℝ), 
    (∀ t : ℝ, curve_param_x t = x ∧ curve_param_y t = y → d ≤ (abs (x + y + 2)) / Real.sqrt (1^2 + 1^2)) 
    ∧ d = (3 * Real.sqrt 2) / 2 + 1 :=
sorry

end max_distance_curve_line_l149_149979


namespace geometric_sequence_ratio_l149_149472

theorem geometric_sequence_ratio (a b : ℕ → ℝ) (A B : ℕ → ℝ)
  (hA9 : A 9 = (a 5) ^ 9)
  (hB9 : B 9 = (b 5) ^ 9)
  (h_ratio : a 5 / b 5 = 2) :
  (A 9 / B 9) = 512 := by
  sorry

end geometric_sequence_ratio_l149_149472


namespace graph_shift_correct_l149_149953

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x) - Real.sqrt 3 * Real.cos (3 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (3 * x)

theorem graph_shift_correct :
  ∀ (x : ℝ), f x = g (x - (5 * Real.pi / 18)) :=
sorry

end graph_shift_correct_l149_149953


namespace smallest_number_divisible_l149_149702

theorem smallest_number_divisible (x : ℕ) :
  (∃ n : ℕ, x = n * 5 + 24) ∧
  (∃ n : ℕ, x = n * 10 + 24) ∧
  (∃ n : ℕ, x = n * 15 + 24) ∧
  (∃ n : ℕ, x = n * 20 + 24) →
  x = 84 :=
by
  sorry

end smallest_number_divisible_l149_149702


namespace third_chapter_is_24_pages_l149_149565

-- Define the total number of pages in the book
def total_pages : ℕ := 125

-- Define the number of pages in the first chapter
def first_chapter_pages : ℕ := 66

-- Define the number of pages in the second chapter
def second_chapter_pages : ℕ := 35

-- Define the number of pages in the third chapter
def third_chapter_pages : ℕ := total_pages - (first_chapter_pages + second_chapter_pages)

-- Prove that the number of pages in the third chapter is 24
theorem third_chapter_is_24_pages : third_chapter_pages = 24 := by
  sorry

end third_chapter_is_24_pages_l149_149565


namespace note_relationship_l149_149397

theorem note_relationship
  (x y z : ℕ) 
  (h1 : x + 5 * y + 10 * z = 480)
  (h2 : x + y + z = 90)
  (h3 : y = 2 * x)
  (h4 : z = 3 * x) : 
  x = 15 ∧ y = 30 ∧ z = 45 :=
by 
  sorry

end note_relationship_l149_149397


namespace symmetric_line_equation_l149_149158

theorem symmetric_line_equation (l : ℝ × ℝ → Prop)
  (h1 : ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0)
  (h2 : ∀ p : ℝ × ℝ, l p ↔ p = (0, 2) ∨ p = ⟨-3, 2⟩) :
  ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0 :=
by
  sorry

end symmetric_line_equation_l149_149158


namespace real_roots_exactly_three_l149_149426

theorem real_roots_exactly_three (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * |x| + 2 = m) → (∃ a b c : ℝ, 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (a^2 - 2 * |a| + 2 = m) ∧ 
  (b^2 - 2 * |b| + 2 = m) ∧ 
  (c^2 - 2 * |c| + 2 = m)) → 
  m = 2 := 
sorry

end real_roots_exactly_three_l149_149426


namespace sugar_amount_l149_149723

noncomputable def mixed_to_improper (a : ℕ) (b c : ℕ) : ℚ :=
  a + b / c

theorem sugar_amount (a : ℚ) (h : a = mixed_to_improper 7 3 4) : 1 / 3 * a = 2 + 7 / 12 :=
by
  rw [h]
  simp
  sorry

end sugar_amount_l149_149723


namespace savings_percentage_first_year_l149_149962

noncomputable def savings_percentage (I S : ℝ) : ℝ := (S / I) * 100

theorem savings_percentage_first_year (I S : ℝ) (h1 : S = 0.20 * I) :
  savings_percentage I S = 20 :=
by
  unfold savings_percentage
  rw [h1]
  field_simp
  norm_num
  sorry

end savings_percentage_first_year_l149_149962


namespace quadratic_min_value_l149_149711

theorem quadratic_min_value (f : ℕ → ℚ) (n : ℕ)
  (h₁ : f n = 6)
  (h₂ : f (n + 1) = 5)
  (h₃ : f (n + 2) = 5) :
  ∃ c : ℚ, c = 39 / 8 ∧ ∀ x : ℕ, f x ≥ c :=
by
  sorry

end quadratic_min_value_l149_149711


namespace problem_1_problem_2_l149_149709

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 3|

theorem problem_1 (a x : ℝ) (h1 : a < 3) (h2 : (∀ x, f x a >= 4 ↔ x ≤ 1/2 ∨ x ≥ 9/2)) : 
  a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h1 : ∀ x : ℝ, f x a + |x - 3| ≥ 1) : 
  a ≤ 2 :=
sorry

end problem_1_problem_2_l149_149709


namespace tea_mixture_price_l149_149184

theorem tea_mixture_price :
  ∃ P Q : ℝ, (62 * P + 72 * Q) / (3 * P + Q) = 64.5 :=
by
  sorry

end tea_mixture_price_l149_149184


namespace hours_worked_l149_149689

theorem hours_worked (w e : ℝ) (hw : w = 6.75) (he : e = 67.5) 
  : e / w = 10 := by
  sorry

end hours_worked_l149_149689


namespace sum_of_fractions_l149_149107

theorem sum_of_fractions : (3/7 : ℚ) + (5/14 : ℚ) = 11/14 :=
by
  sorry

end sum_of_fractions_l149_149107


namespace five_to_one_ratio_to_eleven_is_fifty_five_l149_149480

theorem five_to_one_ratio_to_eleven_is_fifty_five (y : ℚ) (h : 5 / 1 = y / 11) : y = 55 :=
by
  sorry

end five_to_one_ratio_to_eleven_is_fifty_five_l149_149480


namespace minimum_reciprocal_sum_l149_149610

theorem minimum_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  4 ≤ (1 / a) + (1 / b) :=
sorry

end minimum_reciprocal_sum_l149_149610


namespace insect_population_calculations_l149_149328

theorem insect_population_calculations :
  (let ants_1 := 100
   let ants_2 := ants_1 - 20 * ants_1 / 100
   let ants_3 := ants_2 - 25 * ants_2 / 100
   let bees_1 := 150
   let bees_2 := bees_1 - 30 * bees_1 / 100
   let termites_1 := 200
   let termites_2 := termites_1 - 10 * termites_1 / 100
   ants_3 = 60 ∧ bees_2 = 105 ∧ termites_2 = 180) :=
by
  sorry

end insect_population_calculations_l149_149328


namespace domain_of_function_l149_149508

def domain_sqrt_log : Set ℝ :=
  {x | (2 - x ≥ 0) ∧ ((2 * x - 1) / (3 - x) > 0)}

theorem domain_of_function :
  domain_sqrt_log = {x | (1/2 < x) ∧ (x ≤ 2)} :=
by
  sorry

end domain_of_function_l149_149508


namespace inequality_proof_l149_149163

variable (a b c d : ℝ)

theorem inequality_proof (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (1 / (1 / a + 1 / b)) + (1 / (1 / c + 1 / d)) ≤ (1 / (1 / (a + c) + 1 / (b + d))) :=
by
  sorry

end inequality_proof_l149_149163


namespace age_sum_l149_149123

theorem age_sum (P Q : ℕ) (h1 : P - 12 = (1 / 2 : ℚ) * (Q - 12)) (h2 : (P : ℚ) / Q = (3 / 4 : ℚ)) : P + Q = 42 :=
sorry

end age_sum_l149_149123


namespace distance_of_course_l149_149242

-- Definitions
def teamESpeed : ℕ := 20
def teamASpeed : ℕ := teamESpeed + 5

-- Time taken by Team E
variable (tE : ℕ)

-- Distance calculation
def teamEDistance : ℕ := teamESpeed * tE
def teamADistance : ℕ := teamASpeed * (tE - 3)

-- Proof statement
theorem distance_of_course (tE : ℕ) (h : teamEDistance tE = teamADistance tE) : teamEDistance tE = 300 :=
sorry

end distance_of_course_l149_149242


namespace tangent_line_eqn_unique_local_minimum_l149_149187

noncomputable def f (x : ℝ) : ℝ := (Real.exp x + 2) / x

def tangent_line_at_1 (x y : ℝ) : Prop :=
  2 * x + y - Real.exp 1 - 4 = 0

theorem tangent_line_eqn :
  tangent_line_at_1 1 (f 1) :=
sorry

noncomputable def h (x : ℝ) : ℝ := Real.exp x * (x - 1) - 2

theorem unique_local_minimum :
  ∃! c : ℝ, 1 < c ∧ c < 2 ∧ (∀ x < c, f x > f c) ∧ (∀ x > c, f c < f x) :=
sorry

end tangent_line_eqn_unique_local_minimum_l149_149187


namespace diameter_circle_inscribed_triangle_l149_149071

noncomputable def diameter_of_inscribed_circle (XY XZ YZ : ℝ) : ℝ :=
  let s := (XY + XZ + YZ) / 2
  let K := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  let r := K / s
  2 * r

theorem diameter_circle_inscribed_triangle (XY XZ YZ : ℝ) (hXY : XY = 13) (hXZ : XZ = 8) (hYZ : YZ = 9) :
  diameter_of_inscribed_circle XY XZ YZ = 2 * Real.sqrt 210 / 5 := by
{
  rw [hXY, hXZ, hYZ]
  sorry
}

end diameter_circle_inscribed_triangle_l149_149071


namespace ratio_of_black_to_blue_l149_149927

universe u

-- Define the types of black and red pens
variables (B R : ℕ)

-- Define the conditions
def condition1 : Prop := 2 + B + R = 12
def condition2 : Prop := R = 2 * B - 2

-- Define the proof statement
theorem ratio_of_black_to_blue (h1 : condition1 B R) (h2 : condition2 B R) : B / 2 = 1 :=
by
  sorry

end ratio_of_black_to_blue_l149_149927


namespace probability_at_least_one_white_ball_l149_149113

noncomputable def total_combinations : ℕ := (Nat.choose 5 3)
noncomputable def no_white_combinations : ℕ := (Nat.choose 3 3)
noncomputable def prob_no_white_balls : ℚ := no_white_combinations / total_combinations
noncomputable def prob_at_least_one_white_ball : ℚ := 1 - prob_no_white_balls

theorem probability_at_least_one_white_ball :
  prob_at_least_one_white_ball = 9 / 10 :=
by
  have h : total_combinations = 10 := by sorry
  have h1 : no_white_combinations = 1 := by sorry
  have h2 : prob_no_white_balls = 1 / 10 := by sorry
  have h3 : prob_at_least_one_white_ball = 1 - prob_no_white_balls := by sorry
  norm_num [prob_no_white_balls, prob_at_least_one_white_ball, h, h1, h2, h3]

end probability_at_least_one_white_ball_l149_149113


namespace initial_tomatoes_l149_149907

theorem initial_tomatoes (T : ℕ) (picked : ℕ) (remaining_total : ℕ) (potatoes : ℕ) :
  potatoes = 12 →
  picked = 53 →
  remaining_total = 136 →
  T + picked = remaining_total - potatoes →
  T = 71 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_tomatoes_l149_149907


namespace subcommittee_count_l149_149332

theorem subcommittee_count :
  (Nat.choose 10 4) * (Nat.choose 8 3) = 11760 :=
by
  sorry

end subcommittee_count_l149_149332


namespace solve_system_of_equations_l149_149681

theorem solve_system_of_equations (x y : ℝ) :
  (3 * x^2 + 4 * x * y + 12 * y^2 + 16 * y = -6) ∧
  (x^2 - 12 * x * y + 4 * y^2 - 10 * x + 12 * y = -7) →
  (x = 1 / 2) ∧ (y = -3 / 4) :=
by
  sorry

end solve_system_of_equations_l149_149681


namespace correct_sample_in_survey_l149_149001

-- Definitions based on conditions:
def total_population := 1500
def surveyed_population := 150
def sample_description := "the national security knowledge of the selected 150 teachers and students"

-- Hypotheses: conditions
variables (pop : ℕ) (surveyed : ℕ) (description : String)
  (h1 : pop = total_population)
  (h2 : surveyed = surveyed_population)
  (h3 : description = sample_description)

-- Theorem we want to prove
theorem correct_sample_in_survey : description = sample_description :=
  by sorry

end correct_sample_in_survey_l149_149001


namespace first_dig_site_date_difference_l149_149643

-- Definitions for the conditions
def F : Int := sorry  -- The age of the first dig site
def S : Int := sorry  -- The age of the second dig site
def T : Int := sorry  -- The age of the third dig site
def Fo : Int := 8400  -- The age of the fourth dig site
def x : Int := (S - F)

-- The conditions
axiom condition1 : F = S + x
axiom condition2 : T = F + 3700
axiom condition3 : Fo = 2 * T
axiom condition4 : S = 852
axiom condition5 : S > F  -- Ensuring S is older than F for meaningfulness

-- The theorem to prove
theorem first_dig_site_date_difference : x = 352 :=
by
  -- Proof goes here
  sorry

end first_dig_site_date_difference_l149_149643


namespace faction_with_more_liars_than_truth_tellers_l149_149205

theorem faction_with_more_liars_than_truth_tellers 
  (r1 r2 r3 l1 l2 l3 : ℕ) 
  (H1 : r1 + r2 + r3 + l1 + l2 + l3 = 2016)
  (H2 : r1 + l2 + l3 = 1208)
  (H3 : r2 + l1 + l3 = 908)
  (H4 : r3 + l1 + l2 = 608) :
  l3 - r3 = 100 :=
by
  sorry

end faction_with_more_liars_than_truth_tellers_l149_149205


namespace bicycle_total_distance_l149_149757

noncomputable def front_wheel_circumference : ℚ := 4/3
noncomputable def rear_wheel_circumference : ℚ := 3/2
noncomputable def extra_revolutions : ℕ := 25

theorem bicycle_total_distance :
  (front_wheel_circumference * extra_revolutions + (rear_wheel_circumference * 
  ((front_wheel_circumference * extra_revolutions) / (rear_wheel_circumference - front_wheel_circumference))) = 300) := sorry

end bicycle_total_distance_l149_149757


namespace circumference_of_base_l149_149489

-- Definitions used for the problem
def radius : ℝ := 6
def sector_angle : ℝ := 300
def full_circle_angle : ℝ := 360

-- Ask for the circumference of the base of the cone formed by the sector
theorem circumference_of_base (r : ℝ) (theta_sector : ℝ) (theta_full : ℝ) :
  (theta_sector / theta_full) * (2 * π * r) = 10 * π :=
by
  sorry

end circumference_of_base_l149_149489


namespace A_finish_time_l149_149650

theorem A_finish_time {A_work B_work C_work : ℝ} 
  (h1 : A_work + B_work + C_work = 1/4)
  (h2 : B_work = 1/24)
  (h3 : C_work = 1/8) :
  1 / A_work = 12 := by
  sorry

end A_finish_time_l149_149650


namespace problem_statement_l149_149803

noncomputable def square : ℝ := sorry -- We define a placeholder
noncomputable def pentagon : ℝ := sorry -- We define a placeholder

axiom eq1 : 2 * square + 4 * pentagon = 25
axiom eq2 : 3 * square + 3 * pentagon = 22

theorem problem_statement : 4 * pentagon = 20.67 := 
by
  sorry

end problem_statement_l149_149803


namespace min_value_frac_inv_sum_l149_149403

theorem min_value_frac_inv_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + 3 * y = 1) : 
  ∃ (minimum_value : ℝ), minimum_value = 4 + 2 * Real.sqrt 3 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + 3 * b = 1 → (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3) := 
sorry

end min_value_frac_inv_sum_l149_149403


namespace difference_between_c_and_a_l149_149787

variable (a b c : ℝ)

theorem difference_between_c_and_a (h1 : (a + b) / 2 = 30) (h2 : c - a = 60) : c - a = 60 :=
by
  exact h2

end difference_between_c_and_a_l149_149787


namespace area_ratio_l149_149595

-- Definitions corresponding to the conditions
variable {A B C P Q R : Type}
variable (t : ℝ)
variable (h_pos : 0 < t) (h_lt_one : t < 1)

-- Define the areas in terms of provided conditions
noncomputable def area_AP : ℝ := sorry
noncomputable def area_BQ : ℝ := sorry
noncomputable def area_CR : ℝ := sorry
noncomputable def K : ℝ := area_AP * area_BQ * area_CR
noncomputable def L : ℝ := sorry -- Area of triangle ABC

-- The statement to be proved
theorem area_ratio (h_pos : 0 < t) (h_lt_one : t < 1) :
  (K / L) = (1 - t + t^2)^2 :=
sorry

end area_ratio_l149_149595


namespace find_k_l149_149625

def A (a b : ℤ) : Prop := 3 * a + b - 2 = 0
def B (a b : ℤ) (k : ℤ) : Prop := k * (a^2 - a + 1) - b = 0

theorem find_k (k : ℤ) (h : ∃ a b : ℤ, A a b ∧ B a b k ∧ a > 0) : k = -1 ∨ k = 2 :=
by
  sorry

end find_k_l149_149625


namespace rose_tom_profit_difference_l149_149776

def investment_months (amount: ℕ) (months: ℕ) : ℕ :=
  amount * months

def total_investment_months (john_inv: ℕ) (rose_inv: ℕ) (tom_inv: ℕ) : ℕ :=
  john_inv + rose_inv + tom_inv

def profit_share (investment: ℕ) (total_investment: ℕ) (total_profit: ℕ) : ℤ :=
  (investment * total_profit) / total_investment

theorem rose_tom_profit_difference
  (john_inv rs_per_year: ℕ := 18000 * 12)
  (rose_inv rs_per_9_months: ℕ := 12000 * 9)
  (tom_inv rs_per_8_months: ℕ := 9000 * 8)
  (total_profit: ℕ := 4070):
  profit_share rose_inv (total_investment_months john_inv rose_inv tom_inv) total_profit -
  profit_share tom_inv (total_investment_months john_inv rose_inv tom_inv) total_profit = 370 := 
by
  sorry

end rose_tom_profit_difference_l149_149776


namespace min_value_x_plus_y_l149_149131

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y + 9 * x = x * y) :
  x + y ≥ 16 :=
by
  sorry

end min_value_x_plus_y_l149_149131


namespace lcm_second_factor_l149_149726

theorem lcm_second_factor (A B : ℕ) (hcf : ℕ) (f1 f2 : ℕ) 
  (h₁ : hcf = 25) 
  (h₂ : A = 350) 
  (h₃ : Nat.gcd A B = hcf) 
  (h₄ : Nat.lcm A B = hcf * f1 * f2) 
  (h₅ : f1 = 13)
  : f2 = 14 := 
sorry

end lcm_second_factor_l149_149726


namespace intersection_A_complement_B_l149_149195

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { y | 0 ≤ y }

theorem intersection_A_complement_B : A ∩ -B = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end intersection_A_complement_B_l149_149195


namespace total_students_shook_hands_l149_149330

theorem total_students_shook_hands (S3 S2 S1 : ℕ) (h1 : S3 = 200) (h2 : S2 = S3 + 40) (h3 : S1 = 2 * S2) : 
  S1 + S2 + S3 = 920 :=
by
  sorry

end total_students_shook_hands_l149_149330


namespace find_m_l149_149866

noncomputable def inverse_proportion (x : ℝ) : ℝ := 4 / x

theorem find_m (m n : ℝ) (h1 : ∀ x, -4 ≤ x ∧ x ≤ m → inverse_proportion x = 4 / x ∧ n ≤ inverse_proportion x ∧ inverse_proportion x ≤ n + 3) :
  m = -1 :=
by
  sorry

end find_m_l149_149866


namespace lowest_score_l149_149581

-- Define the conditions
def test_scores (s1 s2 s3 : ℕ) := s1 = 86 ∧ s2 = 112 ∧ s3 = 91
def max_score := 120
def target_average := 95
def num_tests := 5
def total_points_needed := target_average * num_tests

-- Define the proof statement
theorem lowest_score 
  (s1 s2 s3 : ℕ)
  (condition1 : test_scores s1 s2 s3)
  (max_pts : ℕ := max_score) 
  (target_avg : ℕ := target_average) 
  (num_tests : ℕ := num_tests)
  (total_needed : ℕ := total_points_needed) :
  ∃ s4 s5 : ℕ, s4 ≤ max_pts ∧ s5 ≤ max_pts ∧ s4 + s5 + s1 + s2 + s3 = total_needed ∧ (s4 = 66 ∨ s5 = 66) :=
by
  sorry

end lowest_score_l149_149581


namespace three_five_seven_sum_fraction_l149_149505

theorem three_five_seven_sum_fraction :
  (3 * 5 * 7) * ((1 / 3) + (1 / 5) + (1 / 7)) = 71 :=
by
  sorry

end three_five_seven_sum_fraction_l149_149505


namespace remainder_of_square_l149_149983

variable (N X : Set ℤ)
variable (k : ℤ)

/-- Given any n in set N and any x in set X, where dividing n by x gives a remainder of 3,
prove that the remainder of n^2 divided by x is 9 mod x. -/
theorem remainder_of_square (n x : ℤ) (hn : n ∈ N) (hx : x ∈ X)
  (h : ∃ k, n = k * x + 3) : (n^2) % x = 9 % x :=
by
  sorry

end remainder_of_square_l149_149983


namespace geometric_sum_l149_149254

theorem geometric_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
    (h1 : S 3 = 8)
    (h2 : S 6 = 7)
    (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 7 + a 8 + a 9 = 1 / 8 :=
by
  sorry

end geometric_sum_l149_149254


namespace max_y_diff_eq_0_l149_149831

-- Definitions for the given conditions
def eq1 (x : ℝ) : ℝ := 4 - 2 * x + x^2
def eq2 (x : ℝ) : ℝ := 2 + 2 * x + x^2

-- Statement of the proof problem
theorem max_y_diff_eq_0 : 
  (∀ x y, eq1 x = y ∧ eq2 x = y → y = (13 / 4)) →
  ∀ (x1 x2 : ℝ), (∃ y1 y2, eq1 x1 = y1 ∧ eq2 x1 = y1 ∧ eq1 x2 = y2 ∧ eq2 x2 = y2) → 
  (x1 = x2) → (y1 = y2) →
  0 = 0 := 
by
  sorry

end max_y_diff_eq_0_l149_149831


namespace trapezoid_not_isosceles_l149_149977

noncomputable def is_trapezoid (BC AD AC : ℝ) : Prop :=
BC = 3 ∧ AD = 4 ∧ AC = 6

def is_isosceles_trapezoid_not_possible (BC AD AC : ℝ) : Prop :=
is_trapezoid BC AD AC → ¬(BC = AD)

theorem trapezoid_not_isosceles (BC AD AC : ℝ) :
  is_isosceles_trapezoid_not_possible BC AD AC :=
sorry

end trapezoid_not_isosceles_l149_149977


namespace quadratic_real_roots_range_l149_149104

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 3 * x - 9 / 4 = 0) →
  (k >= -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_range_l149_149104


namespace root_interval_l149_149819

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 2 * x - 1

theorem root_interval : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
by
  have h_decreasing : ∀ x y : ℝ, x < y → f x < f y :=
    sorry -- Proof that f is increasing on (-1, +∞)
  have h_f0 : f 0 = -1 := by
    sorry -- Calculation that f(0) = -1
  have h_f1 : f 1 = Real.log 2 + 1 := by
    sorry -- Calculation that f(1) = ln(2) + 1
  have h_exist_root : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
    by
      sorry -- Existence of a root in (0,1)
  exact h_exist_root

end root_interval_l149_149819


namespace simplify_expression_l149_149144

theorem simplify_expression(x : ℝ) : 2 * x * (4 * x^2 - 3 * x + 1) - 7 * (2 * x^2 - 3 * x + 4) = 8 * x^3 - 20 * x^2 + 23 * x - 28 :=
by
  sorry

end simplify_expression_l149_149144


namespace basketball_players_taking_chemistry_l149_149454

variable (total_players : ℕ) (taking_biology : ℕ) (taking_both : ℕ)

theorem basketball_players_taking_chemistry (h1 : total_players = 20) 
                                           (h2 : taking_biology = 8) 
                                           (h3 : taking_both = 4) 
                                           (h4 : ∀p, p ≤ total_players) :
  total_players - taking_biology + taking_both = 16 :=
by sorry

end basketball_players_taking_chemistry_l149_149454


namespace rational_powers_implies_rational_a_rational_powers_implies_rational_b_l149_149912

open Real

theorem rational_powers_implies_rational_a (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^7 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

theorem rational_powers_implies_rational_b (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^9 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

end rational_powers_implies_rational_a_rational_powers_implies_rational_b_l149_149912


namespace determine_b_perpendicular_l149_149417

theorem determine_b_perpendicular :
  ∀ (b : ℝ),
  (b * 2 + (-3) * (-1) + 2 * 4 = 0) → 
  b = -11/2 :=
by
  intros b h
  sorry

end determine_b_perpendicular_l149_149417


namespace solve_for_x_l149_149484

variable {x : ℝ}

def is_positive (x : ℝ) : Prop := x > 0

def area_of_triangle_is_150 (x : ℝ) : Prop :=
  let base := 2 * x
  let height := 3 * x
  (1/2) * base * height = 150

theorem solve_for_x (hx : is_positive x) (ha : area_of_triangle_is_150 x) : x = 5 * Real.sqrt 2 := by
  sorry

end solve_for_x_l149_149484


namespace find_num_trumpet_players_l149_149527

namespace OprahWinfreyHighSchoolMarchingBand

def num_trumpet_players (total_weight : ℕ) 
  (num_clarinet : ℕ) (num_trombone : ℕ) 
  (num_tuba : ℕ) (num_drum : ℕ) : ℕ :=
(total_weight - 
  ((num_clarinet * 5) + 
  (num_trombone * 10) + 
  (num_tuba * 20) + 
  (num_drum * 15)))
  / 5

theorem find_num_trumpet_players :
  num_trumpet_players 245 9 8 3 2 = 6 :=
by
  -- calculation and reasoning steps would go here
  sorry

end OprahWinfreyHighSchoolMarchingBand

end find_num_trumpet_players_l149_149527


namespace problem_b_c_constants_l149_149656

theorem problem_b_c_constants (b c : ℝ) (h : ∀ x : ℝ, (x + 2) * (x + b) = x^2 + c * x + 6) : c = 5 := 
by sorry

end problem_b_c_constants_l149_149656


namespace savings_account_amount_l149_149895

-- Definitions and conditions from the problem
def checking_account_yen : ℕ := 6359
def total_yen : ℕ := 9844

-- Question we aim to prove - the amount in the savings account
def savings_account_yen : ℕ := total_yen - checking_account_yen

-- Lean statement to prove the equality
theorem savings_account_amount : savings_account_yen = 3485 :=
by
  sorry

end savings_account_amount_l149_149895


namespace find_f_three_l149_149973

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l149_149973


namespace negation_equiv_l149_149806

-- Define the proposition that the square of all real numbers is positive
def pos_of_all_squares : Prop := ∀ x : ℝ, x^2 > 0

-- Define the negation of the proposition
def neg_pos_of_all_squares : Prop := ∃ x : ℝ, x^2 ≤ 0

theorem negation_equiv (h : ¬ pos_of_all_squares) : neg_pos_of_all_squares :=
  sorry

end negation_equiv_l149_149806


namespace smallest_value_z_minus_x_l149_149571

theorem smallest_value_z_minus_x 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hmul : x * y * z = 5040) 
  (hxy : x < y) 
  (hyz : y < z) : 
  z - x = 9 := 
  sorry

end smallest_value_z_minus_x_l149_149571


namespace find_a_l149_149293

theorem find_a (a : ℝ) (k_l : ℝ) (h1 : k_l = -1)
  (h2 : a ≠ 3) 
  (h3 : (2 - (-1)) / (3 - a) * k_l = -1) : a = 6 :=
by
  sorry

end find_a_l149_149293


namespace goose_eggs_count_l149_149632

theorem goose_eggs_count (E : ℝ) (h1 : 1 / 4 * E = (1 / 4) * E)
  (h2 : 4 / 5 * (1 / 4) * E = (4 / 5) * (1 / 4) * E)
  (h3 : 3 / 5 * (4 / 5) * (1 / 4) * E = 120)
  (h4 : 120 = 120)
  : E = 800 :=
by
  sorry

end goose_eggs_count_l149_149632


namespace arithmetic_seq_a2_l149_149093

theorem arithmetic_seq_a2 (a : ℕ → ℤ) (d : ℤ) (h1 : d = -2) 
  (h2 : (a 1 + a 5) / 2 = -1) : 
  a 2 = 1 :=
by
  sorry

end arithmetic_seq_a2_l149_149093


namespace neg_proposition_P_l149_149476

theorem neg_proposition_P : 
  (¬ (∀ x : ℝ, 2^x + x^2 > 0)) ↔ (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
by
  sorry

end neg_proposition_P_l149_149476


namespace sum_a5_a6_a7_l149_149091

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = q * a n

variables (a : ℕ → ℤ)
variables (h_geo : geometric_sequence a)
variables (h1 : a 2 + a 3 = 1)
variables (h2 : a 3 + a 4 = -2)

theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 24 :=
by
  sorry

end sum_a5_a6_a7_l149_149091


namespace equal_papers_per_cousin_l149_149438

-- Given conditions
def haley_origami_papers : Float := 48.0
def cousins_count : Float := 6.0

-- Question and expected answer
def papers_per_cousin (total_papers : Float) (cousins : Float) : Float :=
  total_papers / cousins

-- Proof statement asserting the correct answer
theorem equal_papers_per_cousin :
  papers_per_cousin haley_origami_papers cousins_count = 8.0 :=
sorry

end equal_papers_per_cousin_l149_149438


namespace distinct_real_roots_absolute_sum_l149_149777

theorem distinct_real_roots_absolute_sum {r1 r2 p : ℝ} (h_root1 : r1 ^ 2 + p * r1 + 7 = 0) 
(h_root2 : r2 ^ 2 + p * r2 + 7 = 0) (h_distinct : r1 ≠ r2) : 
|r1 + r2| > 2 * Real.sqrt 7 := 
sorry

end distinct_real_roots_absolute_sum_l149_149777


namespace max_wrappers_l149_149258

-- Definitions for the conditions
def total_wrappers : ℕ := 49
def andy_wrappers : ℕ := 34

-- The problem statement to prove
theorem max_wrappers : total_wrappers - andy_wrappers = 15 :=
by
  sorry

end max_wrappers_l149_149258


namespace two_digit_number_problem_l149_149111

theorem two_digit_number_problem (a b : ℕ) :
  let M := 10 * b + a
  let N := 10 * a + b
  2 * M - N = 19 * b - 8 * a := by
  sorry

end two_digit_number_problem_l149_149111


namespace largest_partner_share_l149_149928

def total_profit : ℕ := 48000
def partner_ratios : List ℕ := [3, 4, 4, 6, 7]
def value_per_part : ℕ := total_profit / partner_ratios.sum
def largest_share : ℕ := 7 * value_per_part

theorem largest_partner_share :
  largest_share = 14000 := by
  sorry

end largest_partner_share_l149_149928


namespace solve_quadratic_substitution_l149_149422

theorem solve_quadratic_substitution : 
  (∀ x : ℝ, (2 * x - 5) ^ 2 - 2 * (2 * x - 5) - 3 = 0 ↔ x = 2 ∨ x = 4) :=
by
  sorry

end solve_quadratic_substitution_l149_149422


namespace total_pictures_480_l149_149298

noncomputable def total_pictures (pictures_per_album : ℕ) (num_albums : ℕ) : ℕ :=
  pictures_per_album * num_albums

theorem total_pictures_480 : total_pictures 20 24 = 480 :=
  by
    sorry

end total_pictures_480_l149_149298


namespace find_number_l149_149225

theorem find_number (number : ℝ) (h : 0.75 / 100 * number = 0.06) : number = 8 := 
by
  sorry

end find_number_l149_149225


namespace arithmetic_seq_common_difference_l149_149391

theorem arithmetic_seq_common_difference (a1 d : ℝ) (h1 : a1 + 2 * d = 10) (h2 : 4 * a1 + 6 * d = 36) : d = 2 :=
by
  sorry

end arithmetic_seq_common_difference_l149_149391


namespace intersection_of_S_and_T_l149_149125

open Set

def setS : Set ℝ := { x | (x-2)*(x+3) > 0 }
def setT : Set ℝ := { x | 3 - x ≥ 0 }

theorem intersection_of_S_and_T : setS ∩ setT = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_S_and_T_l149_149125


namespace equilibrium_possible_l149_149965

variables {a b θ : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : (b / 2) < a) (h4 : a ≤ b)

theorem equilibrium_possible :
  θ = 0 ∨ θ = Real.arccos ((b^2 + 2 * a^2) / (3 * a * b)) → 
  (b / 2) < a ∧ a ≤ b ∧ (0 ≤ θ ∧ θ ≤ π) :=
sorry

end equilibrium_possible_l149_149965


namespace no_32_people_class_exists_30_people_class_l149_149421

-- Definition of the conditions: relationship between boys and girls
def friends_condition (B G : ℕ) : Prop :=
  3 * B = 2 * G

-- The first problem statement: No 32 people class
theorem no_32_people_class : ¬ ∃ (B G : ℕ), friends_condition B G ∧ B + G = 32 := 
sorry

-- The second problem statement: There is a 30 people class
theorem exists_30_people_class : ∃ (B G : ℕ), friends_condition B G ∧ B + G = 30 := 
sorry

end no_32_people_class_exists_30_people_class_l149_149421


namespace quadratic_max_m_l149_149308

theorem quadratic_max_m (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (m * x^2 - 2 * m * x + 2) ≤ 4) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ (m * x^2 - 2 * m * x + 2) = 4) ∧ 
  m ≠ 0 → 
  (m = 2 / 3 ∨ m = -2) := 
by
  sorry

end quadratic_max_m_l149_149308


namespace intersection_one_point_l149_149191

def quadratic_function (x : ℝ) : ℝ := -x^2 + 5 * x
def linear_function (x : ℝ) (t : ℝ) : ℝ := -3 * x + t
def quadratic_combined_function (x : ℝ) (t : ℝ) : ℝ := x^2 - 8 * x + t

theorem intersection_one_point (t : ℝ) : 
  (64 - 4 * t = 0) → t = 16 :=
by
  intro h
  sorry

end intersection_one_point_l149_149191


namespace barnyard_owl_hoots_per_minute_l149_149031

theorem barnyard_owl_hoots_per_minute :
  (20 - 5) / 3 = 5 := 
by
  sorry

end barnyard_owl_hoots_per_minute_l149_149031


namespace total_monthly_cost_l149_149756

theorem total_monthly_cost (volume_per_box : ℕ := 1800) 
                          (total_volume : ℕ := 1080000)
                          (cost_per_box_per_month : ℝ := 0.8) 
                          (expected_cost : ℝ := 480) : 
                          (total_volume / volume_per_box) * cost_per_box_per_month = expected_cost :=
by
  sorry

end total_monthly_cost_l149_149756


namespace dave_apps_added_l149_149474

theorem dave_apps_added (initial_apps : ℕ) (total_apps_after_adding : ℕ) (apps_added : ℕ) 
  (h1 : initial_apps = 17) (h2 : total_apps_after_adding = 18) 
  (h3 : total_apps_after_adding = initial_apps + apps_added) : 
  apps_added = 1 := 
by
  -- proof omitted
  sorry

end dave_apps_added_l149_149474


namespace ice_cream_to_afford_games_l149_149763

theorem ice_cream_to_afford_games :
  let game_cost := 60
  let ice_cream_price := 5
  (game_cost * 2) / ice_cream_price = 24 :=
by
  let game_cost := 60
  let ice_cream_price := 5
  show (game_cost * 2) / ice_cream_price = 24
  sorry

end ice_cream_to_afford_games_l149_149763


namespace friends_popcorn_l149_149249

theorem friends_popcorn (pieces_per_serving : ℕ) (jared_count : ℕ) (total_servings : ℕ) (jared_friends : ℕ)
  (h1 : pieces_per_serving = 30)
  (h2 : jared_count = 90)
  (h3 : total_servings = 9)
  (h4 : jared_friends = 3) :
  (total_servings * pieces_per_serving - jared_count) / jared_friends = 60 := by
  sorry

end friends_popcorn_l149_149249


namespace m_value_for_perfect_square_l149_149316

theorem m_value_for_perfect_square (m : ℤ) (x y : ℤ) :
  (∃ k : ℤ, 4 * x^2 - m * x * y + 9 * y^2 = k^2) → m = 12 ∨ m = -12 :=
by
  sorry

end m_value_for_perfect_square_l149_149316


namespace inappropriate_expression_is_D_l149_149303

-- Definitions of each expression as constants
def expr_A : String := "Recently, I have had the honor to read your masterpiece, and I felt enlightened."
def expr_B : String := "Your visit has brought glory to my humble abode."
def expr_C : String := "It's the first time you honor my place with a visit, and I apologize for any lack of hospitality."
def expr_D : String := "My mother has been slightly unwell recently, I hope you won't bother her."

-- Definition of the problem context
def is_inappropriate (expr : String) : Prop := 
  expr = expr_D

-- The theorem statement
theorem inappropriate_expression_is_D : is_inappropriate expr_D := 
by
  sorry

end inappropriate_expression_is_D_l149_149303


namespace distinct_values_for_D_l149_149400

-- Define distinct digits
def distinct_digits (a b c d e : ℕ) :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10

-- Declare the problem statement
theorem distinct_values_for_D : 
  ∃ D_values : Finset ℕ, 
    (∀ (A B C D E : ℕ), 
      distinct_digits A B C D E → 
      E + C = D ∧
      B + C = E ∧
      B + D = E) →
    D_values.card = 7 := 
by 
  sorry

end distinct_values_for_D_l149_149400


namespace vector_equation_solution_l149_149572

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) (h : 3 • a + 4 • (b - x) = 0) : 
  x = (3 / 4) • a + b := 
sorry

end vector_equation_solution_l149_149572


namespace circle_properties_l149_149790

theorem circle_properties (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 4 * y + 5 * m = 0) →
  (m < 1 ∨ m > 4) ∧
  (m = -2 → ∃ d : ℝ, d = 2 * Real.sqrt (18 - 5)) :=
by
  sorry

end circle_properties_l149_149790


namespace number_of_groups_is_correct_l149_149760

-- Defining the conditions
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6
def total_players : Nat := new_players + returning_players

-- Theorem to prove the number of groups
theorem number_of_groups_is_correct : total_players / players_per_group = 9 := by
  sorry

end number_of_groups_is_correct_l149_149760


namespace arith_seq_sum_signs_l149_149371

variable {α : Type*} [LinearOrderedField α]
variable {a : ℕ → α} {S : ℕ → α} {d : α}

noncomputable def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  n * (a 1 + a n) / 2

-- Given conditions
variable (a_8_neg : a 8 < 0)
variable (a_9_pos : a 9 > 0)
variable (a_9_greater_abs_a_8 : a 9 > abs (a 8))

-- The theorem to prove
theorem arith_seq_sum_signs (h : is_arith_seq a) :
  (∀ n, n ≤ 15 → sum_first_n_terms a n < 0) ∧ (∀ n, n ≥ 16 → sum_first_n_terms a n > 0) :=
sorry

end arith_seq_sum_signs_l149_149371


namespace range_of_a_l149_149613

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
    sorry

end range_of_a_l149_149613


namespace vector_c_solution_l149_149127

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def vector_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_c_solution
  (a b c : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (2, -3))
  (h3 : vector_parallel (c.1 + 1, c.2 + 2) b)
  (h4 : vector_perpendicular c (3, -1)) :
  c = (-7/9, -7/3) :=
sorry

end vector_c_solution_l149_149127


namespace unique_solution_of_pair_of_equations_l149_149645

-- Definitions and conditions
def pair_of_equations (x k : ℝ) : Prop :=
  (x^2 + 1 = 4 * x + k)

-- Theorem to prove
theorem unique_solution_of_pair_of_equations :
  ∃ k : ℝ, (∀ x : ℝ, pair_of_equations x k -> x = 2) ∧ k = 0 :=
by
  -- Proof omitted
  sorry

end unique_solution_of_pair_of_equations_l149_149645


namespace number_of_pizzas_ordered_l149_149567

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of slices per pizza
def slices_per_pizza : ℕ := 8

-- Define the number of slices each person ate
def slices_per_person : ℕ := 4

-- Define the total number of slices eaten
def total_slices_eaten : ℕ := total_people * slices_per_person

-- Prove that the number of pizzas needed is 3
theorem number_of_pizzas_ordered : total_slices_eaten / slices_per_pizza = 3 := by
  sorry

end number_of_pizzas_ordered_l149_149567


namespace gail_working_hours_x_l149_149989

theorem gail_working_hours_x (x : ℕ) (hx : x < 12) : 
  let hours_am := 12 - x
  let hours_pm := x
  hours_am + hours_pm = 12 := 
by {
  sorry
}

end gail_working_hours_x_l149_149989


namespace sum_value_l149_149217

variable (T R S PV : ℝ)
variable (TD SI : ℝ) (h_td : TD = 80) (h_si : SI = 88)
variable (h1 : SI = TD + (TD * R * T) / 100)
variable (h2 : (PV * R * T) / 100 = TD)
variable (h3 : PV = S - TD)
variable (h4 : R * T = 10)

theorem sum_value : S = 880 := by
  sorry

end sum_value_l149_149217


namespace number_of_solutions_sine_exponential_l149_149116

theorem number_of_solutions_sine_exponential :
  let f := λ x => Real.sin x
  let g := λ x => (1 / 3) ^ x
  ∃ n, n = 150 ∧ ∀ k ∈ Set.Icc (0 : ℝ) (150 * Real.pi), f k = g k → (k : ℝ) ∈ {n : ℝ | n ∈ Set.Icc (0 : ℝ) (150 * Real.pi)} :=
sorry

end number_of_solutions_sine_exponential_l149_149116


namespace max_x_minus_y_l149_149215

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l149_149215


namespace coeff_z_in_third_eq_l149_149162

-- Definitions for the conditions
def eq1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z = 22
def eq2 (x y z : ℝ) : Prop := 4 * x + 8 * y - 11 * z = 7
def eq3 (x y z : ℝ) : Prop := 5 * x - 6 * y + z = 6
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coeff_z_in_third_eq : ∀ (x y z : ℝ), eq1 x y z → eq2 x y z → eq3 x y z → sum_condition x y z → (1 = 1) :=
by
  intros
  sorry

end coeff_z_in_third_eq_l149_149162


namespace rectangle_width_length_ratio_l149_149897

theorem rectangle_width_length_ratio (w l P : ℕ) (hP : P = 30) (hl : l = 10) (h_perimeter : P = 2*l + 2*w) :
  w / l = 1 / 2 :=
by
  sorry

end rectangle_width_length_ratio_l149_149897


namespace area_of_ABCD_l149_149297

noncomputable def quadrilateral_area (AB BC AD DC : ℝ) : ℝ :=
  let area_ABC := 1 / 2 * AB * BC
  let area_ADC := 1 / 2 * AD * DC
  area_ABC + area_ADC

theorem area_of_ABCD {AB BC AD DC AC : ℝ}
  (h1 : AC = 5)
  (h2 : AB * AB + BC * BC = 25)
  (h3 : AD * AD + DC * DC = 25)
  (h4 : AB ≠ AD)
  (h5 : BC ≠ DC) :
  quadrilateral_area AB BC AD DC = 12 :=
sorry

end area_of_ABCD_l149_149297


namespace partI_partII_l149_149137

theorem partI (m : ℝ) (h1 : ∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2) :
  1 ≤ m ∧ m ≤ 5 :=
sorry

noncomputable def lambda : ℝ := 5

theorem partII (x y z : ℝ) (h2 : 3 * x + 4 * y + 5 * z = lambda) :
  x^2 + y^2 + z^2 ≥ 1/2 :=
sorry

end partI_partII_l149_149137


namespace polynomial_value_at_4_l149_149654

def f (x : ℝ) : ℝ := 9 + 15 * x - 8 * x^2 - 20 * x^3 + 6 * x^4 + 3 * x^5

theorem polynomial_value_at_4 :
  f 4 = 3269 :=
by
  sorry

end polynomial_value_at_4_l149_149654


namespace find_k_of_geometric_mean_l149_149200

-- Let {a_n} be an arithmetic sequence with common difference d and a_1 = 9d.
-- Prove that if a_k is the geometric mean of a_1 and a_{2k}, then k = 4.
theorem find_k_of_geometric_mean
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : ∀ n, a n = 9 * d + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a k ^ 2 = a 1 * a (2 * k)) : k = 4 :=
sorry

end find_k_of_geometric_mean_l149_149200


namespace horse_food_per_day_l149_149350

-- Given conditions
def sheep_count : ℕ := 48
def horse_food_total : ℕ := 12880
def sheep_horse_ratio : ℚ := 6 / 7

-- Definition of the number of horses based on the ratio
def horse_count : ℕ := (sheep_count * 7) / 6

-- Statement to prove: each horse needs 230 ounces of food per day
theorem horse_food_per_day : horse_food_total / horse_count = 230 := by
  -- proof here
  sorry

end horse_food_per_day_l149_149350


namespace composite_dice_product_probability_l149_149286

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l149_149286


namespace circumcircle_radius_of_sector_l149_149206

theorem circumcircle_radius_of_sector (θ : Real) (r : Real) (cos_val : Real) (R : Real) :
  θ = 30 * Real.pi / 180 ∧ r = 8 ∧ cos_val = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ R = 8 * (Real.sqrt 6 - Real.sqrt 2) →
  R = 8 * (Real.sqrt 6 - Real.sqrt 2) :=
by
  sorry

end circumcircle_radius_of_sector_l149_149206


namespace find_common_difference_l149_149994

variable (a an Sn d : ℚ)
variable (n : ℕ)

def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def sum_arithmetic_sequence (a : ℚ) (an : ℚ) (n : ℕ) : ℚ :=
  n * (a + an) / 2

theorem find_common_difference
  (h1 : a = 3)
  (h2 : an = 50)
  (h3 : Sn = 318)
  (h4 : an = arithmetic_sequence a d n)
  (h5 : Sn = sum_arithmetic_sequence a an n) :
  d = 47 / 11 :=
by
  sorry

end find_common_difference_l149_149994


namespace ral_current_age_l149_149951

variable (ral suri : ℕ)

-- Conditions
axiom age_relation : ral = 3 * suri
axiom suri_future_age : suri + 3 = 16

-- Statement
theorem ral_current_age : ral = 39 := by
  sorry

end ral_current_age_l149_149951


namespace solve_equation_l149_149380

-- Definitions based on the conditions
def equation (x : ℝ) : Prop :=
  1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 5) + 1 / (x^2 - 17*x - 10) = 0

-- Theorem stating that the solutions of the given equation are the expected values
theorem solve_equation :
  {x : ℝ | equation x} = {-2 + 2 * Real.sqrt 14, -2 - 2 * Real.sqrt 14, (7 + Real.sqrt 89) / 2, (7 - Real.sqrt 89) / 2} :=
by
  sorry

end solve_equation_l149_149380


namespace true_propositions_l149_149638

def p : Prop :=
  ∀ a b : ℝ, (a > 2 ∧ b > 2) → a + b > 4

def q : Prop :=
  ¬ ∃ x : ℝ, x^2 - x > 0 → ∀ x : ℝ, x^2 - x ≤ 0

theorem true_propositions :
  (¬ p ∨ ¬ q) ∧ (p ∨ ¬ q) := by
  sorry

end true_propositions_l149_149638


namespace delta_zeta_finish_time_l149_149824

noncomputable def delta_epsilon_zeta_proof_problem (D E Z : ℝ) (k : ℝ) : Prop :=
  (1 / D + 1 / E + 1 / Z = 1 / (D - 4)) ∧
  (1 / D + 1 / E + 1 / Z = 1 / (E - 3.5)) ∧
  (1 / E + 1 / Z = 2 / E) → 
  k = 2

-- Now we prepare the theorem statement
theorem delta_zeta_finish_time (D E Z k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z = 1 / (D - 4))
                                (h2 : 1 / D + 1 / E + 1 / Z = 1 / (E - 3.5))
                                (h3 : 1 / E + 1 / Z = 2 / E) 
                                (h4 : E = 6) :
  k = 2 := 
sorry

end delta_zeta_finish_time_l149_149824


namespace smallest_integer_k_l149_149354

theorem smallest_integer_k (k : ℤ) : k > 2 ∧ k % 19 = 2 ∧ k % 7 = 2 ∧ k % 4 = 2 ↔ k = 534 :=
by
  sorry

end smallest_integer_k_l149_149354


namespace income_of_A_l149_149999

theorem income_of_A (x y : ℝ) (hx₁ : 5 * x - 3 * y = 1600) (hx₂ : 4 * x - 2 * y = 1600) : 
  5 * x = 4000 :=
by
  sorry

end income_of_A_l149_149999


namespace son_and_daughter_current_ages_l149_149275

theorem son_and_daughter_current_ages
  (father_age_now : ℕ)
  (son_age_5_years_ago : ℕ)
  (daughter_age_5_years_ago : ℝ)
  (h_father_son_birth : father_age_now - (son_age_5_years_ago + 5) = (son_age_5_years_ago + 5))
  (h_father_daughter_birth : father_age_now - (daughter_age_5_years_ago + 5) = (daughter_age_5_years_ago + 5))
  (h_daughter_half_son_5_years_ago : daughter_age_5_years_ago = son_age_5_years_ago / 2) :
  son_age_5_years_ago + 5 = 12 ∧ daughter_age_5_years_ago + 5 = 8.5 :=
by
  sorry

end son_and_daughter_current_ages_l149_149275


namespace range_of_a_l149_149579

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + 2 * x + a ≤ 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_l149_149579


namespace sum_of_proper_divisors_30_is_42_l149_149715

def is_proper_divisor (n d : ℕ) : Prop := d ∣ n ∧ d ≠ n

-- The set of proper divisors of 30.
def proper_divisors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15}

-- The sum of all proper divisors of 30.
def sum_proper_divisors_30 : ℕ := proper_divisors_30.sum id

theorem sum_of_proper_divisors_30_is_42 : sum_proper_divisors_30 = 42 := 
by
  -- Proof can be filled in here
  sorry

end sum_of_proper_divisors_30_is_42_l149_149715


namespace base8_base6_positive_integer_l149_149528

theorem base8_base6_positive_integer (C D N : ℕ)
  (base8: N = 8 * C + D)
  (base6: N = 6 * D + C)
  (valid_C_base8: C < 8)
  (valid_D_base6: D < 6)
  (valid_C_D: 7 * C = 5 * D)
: N = 43 := by
  sorry

end base8_base6_positive_integer_l149_149528


namespace factor_difference_of_squares_l149_149310

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l149_149310


namespace smallest_k_l149_149263

def arith_seq_sum (k n : ℕ) : ℕ :=
  (n + 1) * (2 * k + n) / 2

theorem smallest_k (k n : ℕ) (h_sum : arith_seq_sum k n = 100) :
  k = 9 :=
by
  sorry

end smallest_k_l149_149263


namespace friend_initial_marbles_l149_149160

theorem friend_initial_marbles (total_games : ℕ) (bids_per_game : ℕ) (games_lost : ℕ) (final_marbles : ℕ) 
  (h_games_eq : total_games = 9) (h_bids_eq : bids_per_game = 10) 
  (h_lost_eq : games_lost = 1) (h_final_eq : final_marbles = 90) : 
  ∃ initial_marbles : ℕ, initial_marbles = 20 := by
  sorry

end friend_initial_marbles_l149_149160


namespace adrian_water_amount_l149_149022

theorem adrian_water_amount
  (O S W : ℕ) 
  (h1 : S = 3 * O)
  (h2 : W = 5 * S)
  (h3 : O = 4) : W = 60 :=
by
  sorry

end adrian_water_amount_l149_149022


namespace contradiction_proof_l149_149337

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) :=
by
  sorry

end contradiction_proof_l149_149337


namespace count_valid_48_tuples_l149_149873

open BigOperators

theorem count_valid_48_tuples : 
  ∃ n : ℕ, n = 54 ^ 48 ∧ 
  ( ∃ a : Fin 48 → ℕ, 
    (∀ i : Fin 48, 0 ≤ a i ∧ a i ≤ 100) ∧ 
    (∀ (i j : Fin 48), i < j → a i ≠ a j ∧ a i ≠ a j + 1) 
  ) :=
by
  sorry

end count_valid_48_tuples_l149_149873


namespace general_formula_a_S_n_no_arithmetic_sequence_in_b_l149_149348

def sequence_a (a : ℕ → ℚ) :=
  (a 1 = 1 / 4) ∧ (∀ n : ℕ, n > 0 → 3 * a (n + 1) - 2 * a n = 1)

def sequence_b (b : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n : ℕ, n > 0 → b n = a (n + 1) - a n

theorem general_formula_a_S_n (a : ℕ → ℚ) (S : ℕ → ℚ) :
  sequence_a a →
  (∀ n : ℕ, n > 0 → a n = 1 - (3 / 4) * (2 / 3)^(n - 1)) →
  (∀ n : ℕ, n > 0 → S n = (2 / 3)^(n - 2) + n - 9 / 4) →
  True := sorry

theorem no_arithmetic_sequence_in_b (b : ℕ → ℚ) (a : ℕ → ℚ) :
  sequence_b b a →
  (∀ n : ℕ, n > 0 → b n = (1 / 4) * (2 / 3)^(n - 1)) →
  (∀ r s t : ℕ, r < s ∧ s < t → ¬ (b s - b r = b t - b s)) :=
  sorry

end general_formula_a_S_n_no_arithmetic_sequence_in_b_l149_149348


namespace problem_HMMT_before_HMT_l149_149827
noncomputable def probability_of_sequence (seq: List Char) : ℚ := sorry
def probability_H : ℚ := 1 / 3
def probability_M : ℚ := 1 / 3
def probability_T : ℚ := 1 / 3

theorem problem_HMMT_before_HMT : probability_of_sequence ['H', 'M', 'M', 'T'] = 1 / 4 :=
sorry

end problem_HMMT_before_HMT_l149_149827


namespace radar_placement_coverage_l149_149815

noncomputable def max_distance_radars (r : ℝ) (n : ℕ) : ℝ :=
  r / Real.sin (Real.pi / n)

noncomputable def coverage_ring_area (r : ℝ) (width : ℝ) (n : ℕ) : ℝ :=
  (1440 * Real.pi) / Real.tan (Real.pi / n)

theorem radar_placement_coverage :
  let r := 41
  let width := 18
  let n := 7
  max_distance_radars r n = 40 / Real.sin (Real.pi / 7) ∧
  coverage_ring_area r width n = (1440 * Real.pi) / Real.tan (Real.pi / 7) :=
by
  sorry

end radar_placement_coverage_l149_149815


namespace new_person_weight_l149_149631

/-- 
The average weight of 10 persons increases by 6.3 kg when a new person comes 
in place of one of them weighing 65 kg. Prove that the weight of the new person 
is 128 kg.
-/
theorem new_person_weight (w_old : ℝ) (n : ℝ) (delta_w : ℝ) (w_new : ℝ) 
  (h1 : w_old = 65) 
  (h2 : n = 10) 
  (h3 : delta_w = 6.3) 
  (h4 : w_new = w_old + n * delta_w) : 
  w_new = 128 :=
by 
  rw [h1, h2, h3] at h4 
  rw [h4]
  norm_num

end new_person_weight_l149_149631


namespace find_angle_four_l149_149667

theorem find_angle_four (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle1 + angle3 + 60 = 180)
  (h3 : angle3 = angle4) :
  angle4 = 60 :=
by sorry

end find_angle_four_l149_149667


namespace geometric_sequence_a1_cannot_be_2_l149_149124

theorem geometric_sequence_a1_cannot_be_2
  (a : ℕ → ℕ)
  (q : ℕ)
  (h1 : 2 * a 2 + a 3 = a 4)
  (h2 : (a 2 + 1) * (a 3 + 1) = a 5 - 1)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 1 ≠ 2 :=
by sorry

end geometric_sequence_a1_cannot_be_2_l149_149124


namespace quadratic_has_minimum_l149_149243

theorem quadratic_has_minimum 
  (a b : ℝ) (h : a ≠ 0) (g : ℝ → ℝ) 
  (H : ∀ x, g x = a * x^2 + b * x + (b^2 / a)) :
  ∃ x₀, ∀ x, g x ≥ g x₀ :=
by sorry

end quadratic_has_minimum_l149_149243


namespace positive_integers_congruent_to_2_mod_7_lt_500_count_l149_149546

theorem positive_integers_congruent_to_2_mod_7_lt_500_count : 
  ∃ n : ℕ, n = 72 ∧ ∀ k : ℕ, (k < n → (∃ m : ℕ, (m < 500 ∧ m % 7 = 2) ∧ m = 2 + 7 * k)) := 
by
  sorry

end positive_integers_congruent_to_2_mod_7_lt_500_count_l149_149546


namespace range_of_a_l149_149135

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a + 3) * Real.exp (a * x)

theorem range_of_a (a : ℝ) : 
  (∀ x y, x ≤ y → f a x ≤ f a y) ∨ (∀ x y, x ≤ y → f a x ≥ f a y) → 
  a ∈ Set.Ico (-2 : ℝ) 0 :=
sorry

end range_of_a_l149_149135


namespace perimeter_rectangles_l149_149345

theorem perimeter_rectangles (a b : ℕ) (p_rect1 p_rect2 : ℕ) (p_photo : ℕ) (h1 : 2 * (a + b) = p_photo) (h2 : a + b = 10) (h3 : p_rect1 = 40) (h4 : p_rect2 = 44) : 
p_rect1 ≠ p_rect2 -> (p_rect1 = 40 ∧ p_rect2 = 44) := 
by 
  sorry

end perimeter_rectangles_l149_149345


namespace total_interest_correct_l149_149407

-- Definitions
def total_amount : ℝ := 3500
def P1 : ℝ := 1550
def P2 : ℝ := total_amount - P1
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Total interest calculation
noncomputable def interest1 : ℝ := P1 * rate1
noncomputable def interest2 : ℝ := P2 * rate2
noncomputable def total_interest : ℝ := interest1 + interest2

-- Theorem statement
theorem total_interest_correct : total_interest = 144 := 
by
  -- Proof steps would go here
  sorry

end total_interest_correct_l149_149407


namespace calculate_expression_l149_149765

theorem calculate_expression :
  -1 ^ 2023 + (Real.pi - 3.14) ^ 0 + |(-2 : ℝ)| = 2 :=
by
  sorry

end calculate_expression_l149_149765


namespace new_cooks_waiters_ratio_l149_149172

-- Definitions based on the conditions
variables (cooks waiters new_waiters : ℕ)

-- Given conditions
def ratio := 3
def initial_waiters := (ratio * cooks) / 3 -- Derived from 3 cooks / 11 waiters = 9 cooks / x waiters
def hired_waiters := 12
def total_waiters := initial_waiters + hired_waiters

-- The restaurant has 9 cooks
def restaurant_cooks := 9

-- Conclusion to prove
theorem new_cooks_waiters_ratio :
  (ratio = 3) →
  (restaurant_cooks = 9) →
  (initial_waiters = (ratio * restaurant_cooks) / 3) →
  (cooks = restaurant_cooks) →
  (waiters = initial_waiters) →
  (new_waiters = waiters + hired_waiters) →
  (new_waiters = 45) →
  (cooks / new_waiters = 1 / 5) :=
by
  intros
  sorry

end new_cooks_waiters_ratio_l149_149172


namespace parking_cost_savings_l149_149459

theorem parking_cost_savings
  (weekly_rate : ℕ := 10)
  (monthly_rate : ℕ := 24)
  (weeks_in_year : ℕ := 52)
  (months_in_year : ℕ := 12) :
  (weekly_rate * weeks_in_year) - (monthly_rate * months_in_year) = 232 :=
by
  sorry

end parking_cost_savings_l149_149459


namespace terminating_decimal_expansion_l149_149490

theorem terminating_decimal_expansion (a b : ℕ) (h : 1600 = 2^6 * 5^2) :
  (13 : ℚ) / 1600 = 65 / 1000 :=
by
  sorry

end terminating_decimal_expansion_l149_149490


namespace shopkeeper_loss_percentage_l149_149315

theorem shopkeeper_loss_percentage
    (CP : ℝ) (profit_rate loss_percent : ℝ) 
    (SP : ℝ := CP * (1 + profit_rate)) 
    (value_after_theft : ℝ := SP * (1 - loss_percent)) 
    (goods_loss : ℝ := 100 * (1 - (value_after_theft / CP))) :
    goods_loss = 51.6 :=
by
    sorry

end shopkeeper_loss_percentage_l149_149315


namespace circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l149_149685

theorem circumscribe_quadrilateral_a : 
  ∃ (x : ℝ), 2 * x + 4 * x + 5 * x + 3 * x = 360 
          ∧ (2 * x + 5 * x = 180) 
          ∧ (4 * x + 3 * x = 180) := sorry

theorem circumscribe_quadrilateral_b : 
  ∃ (x : ℝ), 5 * x + 7 * x + 8 * x + 9 * x = 360 
          ∧ (5 * x + 8 * x ≠ 180) 
          ∧ (7 * x + 9 * x ≠ 180) := sorry

end circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l149_149685


namespace train_speed_proof_l149_149820

theorem train_speed_proof :
  (∀ (speed : ℝ), 
    let train_length := 120
    let cross_time := 16
    let total_distance := 240
    let relative_speed := total_distance / cross_time
    let individual_speed := relative_speed / 2
    let speed_kmh := individual_speed * 3.6
    (speed_kmh = 27) → speed = 27
  ) :=
by
  sorry

end train_speed_proof_l149_149820


namespace probability_no_success_l149_149176

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l149_149176


namespace representable_as_product_l149_149486

theorem representable_as_product (n : ℤ) (p q : ℚ) (h1 : n > 1995) (h2 : 0 < p) (h3 : p < 1) :
  ∃ (terms : List ℚ), p = terms.prod ∧ ∀ t ∈ terms, ∃ n, t = (n^2 - 1995^2) / (n^2 - 1994^2) ∧ n > 1995 :=
sorry

end representable_as_product_l149_149486


namespace eggs_in_nests_l149_149652

theorem eggs_in_nests (x : ℕ) (h1 : 2 * x + 3 + 4 = 17) : x = 5 :=
by
  /- This is where the proof would go, but the problem only requires the statement -/
  sorry

end eggs_in_nests_l149_149652


namespace person_income_l149_149660

theorem person_income 
    (income expenditure savings : ℕ) 
    (h1 : income = 3 * (income / 3)) 
    (h2 : expenditure = 2 * (income / 3)) 
    (h3 : savings = 7000) 
    (h4 : income = expenditure + savings) : 
    income = 21000 := 
by 
  sorry

end person_income_l149_149660


namespace suraj_new_average_l149_149255

noncomputable def suraj_average (A : ℝ) : ℝ := A + 8

theorem suraj_new_average (A : ℝ) (h_conditions : 14 * A + 140 = 15 * (A + 8)) :
  suraj_average A = 28 :=
by
  sorry

end suraj_new_average_l149_149255


namespace birds_left_after_a_week_l149_149404

def initial_chickens := 300
def initial_turkeys := 200
def initial_guinea_fowls := 80
def daily_chicken_loss := 20
def daily_turkey_loss := 8
def daily_guinea_fowl_loss := 5
def days_in_a_week := 7

def remaining_chickens := initial_chickens - daily_chicken_loss * days_in_a_week
def remaining_turkeys := initial_turkeys - daily_turkey_loss * days_in_a_week
def remaining_guinea_fowls := initial_guinea_fowls - daily_guinea_fowl_loss * days_in_a_week

def total_remaining_birds := remaining_chickens + remaining_turkeys + remaining_guinea_fowls

theorem birds_left_after_a_week : total_remaining_birds = 349 := by
  sorry

end birds_left_after_a_week_l149_149404


namespace multiplication_difference_is_1242_l149_149746

theorem multiplication_difference_is_1242 (a b c : ℕ) (h1 : a = 138) (h2 : b = 43) (h3 : c = 34) :
  a * b - a * c = 1242 :=
by
  sorry

end multiplication_difference_is_1242_l149_149746


namespace rate_per_square_meter_is_3_l149_149296

def floor_painting_rate 
  (length : ℝ) 
  (total_cost : ℝ)
  (length_more_than_breadth_by_percentage : ℝ)
  (expected_rate : ℝ) : Prop :=
  ∃ (breadth : ℝ) (rate : ℝ),
    length = (1 + length_more_than_breadth_by_percentage / 100) * breadth ∧
    total_cost = length * breadth * rate ∧
    rate = expected_rate

-- Given conditions
theorem rate_per_square_meter_is_3 :
  floor_painting_rate 15.491933384829668 240 200 3 :=
by
  sorry

end rate_per_square_meter_is_3_l149_149296


namespace store_A_profit_margin_l149_149808

theorem store_A_profit_margin
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > x)
  (h : (y - x) / x + 0.12 = (y - 0.9 * x) / (0.9 * x)) :
  (y - x) / x = 0.08 :=
by {
  sorry
}

end store_A_profit_margin_l149_149808


namespace rank_from_right_l149_149612

theorem rank_from_right (n total rank_left : ℕ) (h1 : rank_left = 5) (h2 : total = 21) : n = total - (rank_left - 1) :=
by {
  sorry
}

end rank_from_right_l149_149612


namespace dave_deleted_apps_l149_149593

theorem dave_deleted_apps :
  ∃ d : ℕ, d = 150 - 65 :=
sorry

end dave_deleted_apps_l149_149593


namespace center_of_circle_is_2_1_l149_149789

-- Definition of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y - 5 = 0

-- Theorem stating the center of the circle
theorem center_of_circle_is_2_1 (x y : ℝ) (h : circle_eq x y) : (x, y) = (2, 1) := sorry

end center_of_circle_is_2_1_l149_149789


namespace peter_total_pizza_eaten_l149_149324

def slices_total : Nat := 16
def peter_slices_eaten_alone : ℚ := 2 / 16
def shared_slice_total : ℚ := 1 / (3 * 16)

theorem peter_total_pizza_eaten : peter_slices_eaten_alone + shared_slice_total = 7 / 48 := by
  sorry

end peter_total_pizza_eaten_l149_149324


namespace logarithmic_inequality_l149_149130

noncomputable def log_a_b (a b : ℝ) := Real.log b / Real.log a

theorem logarithmic_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  log_a_b a b + log_a_b b c + log_a_b a c ≥ 3 :=
by
  sorry

end logarithmic_inequality_l149_149130


namespace baseball_team_earnings_l149_149847

theorem baseball_team_earnings (S : ℝ) (W : ℝ) (Total : ℝ) 
    (h1 : S = 2662.50) 
    (h2 : W = S - 142.50) 
    (h3 : Total = W + S) : 
  Total = 5182.50 :=
sorry

end baseball_team_earnings_l149_149847


namespace laura_bought_4_shirts_l149_149621

-- Definitions for the conditions
def pants_price : ℕ := 54
def num_pants : ℕ := 2
def shirt_price : ℕ := 33
def given_money : ℕ := 250
def change_received : ℕ := 10

-- Proving the number of shirts bought is 4
theorem laura_bought_4_shirts :
  (num_pants * pants_price) + (shirt_price * 4) + change_received = given_money :=
by
  sorry

end laura_bought_4_shirts_l149_149621


namespace range_of_p_l149_149750

noncomputable def a_n (p : ℝ) (n : ℕ) : ℝ := -2 * n + p
noncomputable def b_n (n : ℕ) : ℝ := 2 ^ (n - 7)

noncomputable def c_n (p : ℝ) (n : ℕ) : ℝ :=
if a_n p n <= b_n n then a_n p n else b_n n

theorem range_of_p (p : ℝ) :
  (∀ n : ℕ, n ≠ 10 → c_n p 10 > c_n p n) ↔ 24 < p ∧ p < 30 :=
sorry

end range_of_p_l149_149750


namespace bill_has_correct_final_amount_l149_149069

def initial_amount : ℕ := 42
def pizza_cost : ℕ := 11
def pizzas_bought : ℕ := 3
def bill_initial_amount : ℕ := 30
def amount_spent := pizzas_bought * pizza_cost
def frank_remaining_amount := initial_amount - amount_spent
def bill_final_amount := bill_initial_amount + frank_remaining_amount

theorem bill_has_correct_final_amount : bill_final_amount = 39 := by
  sorry

end bill_has_correct_final_amount_l149_149069


namespace triangular_region_area_l149_149148

theorem triangular_region_area : 
  ∀ (x y : ℝ),  (3 * x + 4 * y = 12) →
  (0 ≤ x ∧ 0 ≤ y) →
  ∃ (A : ℝ), A = 6 := 
by 
  sorry

end triangular_region_area_l149_149148


namespace trapezium_side_length_l149_149744

variable (length1 length2 height area : ℕ)

theorem trapezium_side_length
  (h1 : length1 = 20)
  (h2 : height = 15)
  (h3 : area = 270)
  (h4 : area = (length1 + length2) * height / 2) :
  length2 = 16 :=
by
  sorry

end trapezium_side_length_l149_149744


namespace quotient_is_10_l149_149094

theorem quotient_is_10 (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 161)
  (h2 : divisor = 16)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) : 
  quotient = 10 := 
by
  sorry

end quotient_is_10_l149_149094


namespace geometric_sequence_problem_l149_149513

variable {α : Type*} [LinearOrder α] [Field α]

def is_geometric_sequence (a : ℕ → α) :=
  ∀ n : ℕ, a (n + 1) * a (n - 1) = a n ^ 2

theorem geometric_sequence_problem (a : ℕ → α) (r : α) (h1 : a 1 = 1) (h2 : is_geometric_sequence a) (h3 : a 3 * a 5 = 4 * (a 4 - 1)) : 
  a 7 = 4 :=
by
  sorry

end geometric_sequence_problem_l149_149513


namespace skyler_total_songs_skyler_success_breakdown_l149_149678

noncomputable def skyler_songs : ℕ :=
  let hit_songs := 25
  let top_100_songs := hit_songs + 10
  let unreleased_songs := hit_songs - 5
  let duets_total := 12
  let duets_top_20 := duets_total / 2
  let duets_not_top_200 := duets_total / 2
  let soundtracks_total := 18
  let soundtracks_extremely := 3
  let soundtracks_moderate := 8
  let soundtracks_lukewarm := 7
  let projects_total := 22
  let projects_global := 1
  let projects_regional := 7
  let projects_overlooked := 14
  hit_songs + top_100_songs + unreleased_songs + duets_total + soundtracks_total + projects_total

theorem skyler_total_songs : skyler_songs = 132 := by
  sorry

theorem skyler_success_breakdown :
  let extremely_successful := 25 + 1
  let successful := 35 + 6 + 3
  let moderately_successful := 8 + 7
  let less_successful := 7 + 14 + 6
  let unreleased := 20
  (extremely_successful, successful, moderately_successful, less_successful, unreleased) =
  (26, 44, 15, 27, 20) := by
  sorry

end skyler_total_songs_skyler_success_breakdown_l149_149678


namespace sum_faces_edges_vertices_eq_26_l149_149888

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l149_149888


namespace find_ab_l149_149761
-- Import the necessary Lean libraries 

-- Define the statement for the proof problem
theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : ab = 9 :=
by {
    sorry
}

end find_ab_l149_149761


namespace max_brownie_cakes_l149_149060

theorem max_brownie_cakes (m n : ℕ) (h : (m-2)*(n-2) = (1/2)*m*n) :  m * n ≤ 60 :=
sorry

end max_brownie_cakes_l149_149060


namespace roses_problem_l149_149661

variable (R B C : ℕ)

theorem roses_problem
    (h1 : R = B + 10)
    (h2 : C = 10)
    (h3 : 16 - 6 = C)
    (h4 : B = R - C):
  R = B + 10 ∧ R - C = B := 
by 
  have hC: C = 10 := by linarith
  have hR: R = B + 10 := by linarith
  have hRC: R - C = B := by linarith
  exact ⟨hR, hRC⟩

end roses_problem_l149_149661


namespace sub_of_neg_l149_149294

theorem sub_of_neg : -3 - 2 = -5 :=
by 
  sorry

end sub_of_neg_l149_149294


namespace find_fraction_value_l149_149913

noncomputable section

open Real

theorem find_fraction_value (α : ℝ) (h : sin (α / 2) - 2 * cos (α / 2) = 1) :
  (1 + sin α + cos α) / (1 + sin α - cos α) = 1 :=
sorry

end find_fraction_value_l149_149913


namespace tuesday_snow_correct_l149_149674

-- Define the snowfall amounts as given in the conditions
def monday_snow : ℝ := 0.32
def total_snow : ℝ := 0.53

-- Define the amount of snow on Tuesday as per the question to be proved
def tuesday_snow : ℝ := total_snow - monday_snow

-- State the theorem to prove that the snowfall on Tuesday is 0.21 inches
theorem tuesday_snow_correct : tuesday_snow = 0.21 := by
  -- Proof skipped with sorry
  sorry

end tuesday_snow_correct_l149_149674


namespace votes_lost_by_l149_149314

theorem votes_lost_by (total_votes : ℕ) (candidate_percentage : ℕ) : total_votes = 20000 → candidate_percentage = 10 → 
  (total_votes * candidate_percentage / 100 - total_votes * (100 - candidate_percentage) / 100 = 16000) :=
by
  intros h_total_votes h_candidate_percentage
  have vote_candidate := total_votes * candidate_percentage / 100
  have vote_rival := total_votes * (100 - candidate_percentage) / 100
  have votes_diff := vote_rival - vote_candidate
  rw [h_total_votes, h_candidate_percentage] at *
  sorry

end votes_lost_by_l149_149314


namespace minimum_value_is_138_l149_149266

-- Definition of problem conditions and question
def is_digit (n : ℕ) : Prop := n < 10
def digits (A : ℕ) : List ℕ := A.digits 10

def multiple_of_3_not_9 (A : ℕ) : Prop :=
  A % 3 = 0 ∧ A % 9 ≠ 0

def product_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· * ·) 1

def sum_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· + ·) 0

def given_condition (A : ℕ) : Prop :=
  A % 9 = 0 → False ∧
  (A + product_of_digits A) % 9 = 0

-- Main goal: Prove that the minimum value A == 138 satisfies the given conditions
theorem minimum_value_is_138 : ∃ A, A = 138 ∧
  multiple_of_3_not_9 A ∧
  given_condition A :=
sorry

end minimum_value_is_138_l149_149266


namespace trains_crossing_time_l149_149599

noncomputable def TrainA_length := 200  -- meters
noncomputable def TrainA_time := 15  -- seconds
noncomputable def TrainB_length := 300  -- meters
noncomputable def TrainB_time := 25  -- seconds

noncomputable def Speed (length : ℕ) (time : ℕ) := (length : ℝ) / (time : ℝ)

noncomputable def TrainA_speed := Speed TrainA_length TrainA_time
noncomputable def TrainB_speed := Speed TrainB_length TrainB_time

noncomputable def relative_speed := TrainA_speed + TrainB_speed
noncomputable def total_distance := (TrainA_length : ℝ) + (TrainB_length : ℝ)

noncomputable def crossing_time := total_distance / relative_speed

theorem trains_crossing_time :
  (crossing_time : ℝ) = 500 / 25.33 :=
sorry

end trains_crossing_time_l149_149599


namespace goldfish_cost_graph_is_finite_set_of_points_l149_149967

theorem goldfish_cost_graph_is_finite_set_of_points :
  ∀ (n : ℤ), (1 ≤ n ∧ n ≤ 12) → ∃ (C : ℤ), C = 15 * n ∧ ∀ m ≠ n, C ≠ 15 * m :=
by
  -- The proof goes here
  sorry

end goldfish_cost_graph_is_finite_set_of_points_l149_149967


namespace calc_abc_squares_l149_149633

theorem calc_abc_squares :
  ∀ (a b c : ℝ),
  a^2 + 3 * b = 14 →
  b^2 + 5 * c = -13 →
  c^2 + 7 * a = -26 →
  a^2 + b^2 + c^2 = 20.75 :=
by
  intros a b c h1 h2 h3
  -- The proof is omitted; reasoning is provided in the solution.
  sorry

end calc_abc_squares_l149_149633


namespace gathering_gift_exchange_l149_149364

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l149_149364


namespace bike_cost_l149_149903

theorem bike_cost (price_per_apple repairs_share remaining_share apples_sold earnings repairs_cost bike_cost : ℝ) :
  price_per_apple = 1.25 →
  repairs_share = 0.25 →
  remaining_share = 1/5 →
  apples_sold = 20 →
  earnings = apples_sold * price_per_apple →
  repairs_cost = earnings * 4/5 →
  repairs_cost = bike_cost * repairs_share →
  bike_cost = 80 :=
by
  intros;
  sorry

end bike_cost_l149_149903


namespace distance_to_school_l149_149372

variable (T D : ℕ)

/-- Given the conditions, prove the distance from the child's home to the school is 630 meters --/
theorem distance_to_school :
  (5 * (T + 6) = D) →
  (7 * (T - 30) = D) →
  D = 630 :=
by
  intros h1 h2
  sorry

end distance_to_school_l149_149372


namespace harvest_weeks_l149_149478

/-- Lewis earns $403 every week during a certain number of weeks of harvest. 
If he has to pay $49 rent every week, and he earns $93,899 during the harvest season, 
we need to prove that the number of weeks in the harvest season is 265. --/
theorem harvest_weeks 
  (E : ℕ) (R : ℕ) (T : ℕ) (W : ℕ) 
  (hE : E = 403) (hR : R = 49) (hT : T = 93899) 
  (hW : W = 265) : 
  W = (T / (E - R)) := 
by sorry

end harvest_weeks_l149_149478


namespace rectangle_area_l149_149930

variable (a b : ℝ)

-- Given conditions
axiom h1 : (a + b)^2 = 16 
axiom h2 : (a - b)^2 = 4

-- Objective: Prove that the area of the rectangle ab equals 3
theorem rectangle_area : a * b = 3 := by
  sorry

end rectangle_area_l149_149930


namespace age_sum_proof_l149_149002

theorem age_sum_proof (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 20) : a + b + c = 52 :=
by
  sorry

end age_sum_proof_l149_149002


namespace minimum_lines_for_regions_l149_149099

theorem minimum_lines_for_regions (n : ℕ) : 1 + n * (n + 1) / 2 ≥ 1000 ↔ n ≥ 45 :=
sorry

end minimum_lines_for_regions_l149_149099


namespace ratio_of_waist_to_hem_l149_149562

theorem ratio_of_waist_to_hem
  (cuffs_length : ℕ)
  (hem_length : ℕ)
  (ruffles_length : ℕ)
  (num_ruffles : ℕ)
  (lace_cost_per_meter : ℕ)
  (total_spent : ℕ)
  (waist_length : ℕ) :
  cuffs_length = 50 →
  hem_length = 300 →
  ruffles_length = 20 →
  num_ruffles = 5 →
  lace_cost_per_meter = 6 →
  total_spent = 36 →
  waist_length = (total_spent / lace_cost_per_meter * 100) -
                (2 * cuffs_length + hem_length + num_ruffles * ruffles_length) →
  waist_length / hem_length = 1 / 3 :=
by
  sorry

end ratio_of_waist_to_hem_l149_149562


namespace simplify_fraction_expression_l149_149666

theorem simplify_fraction_expression : 
  (18 / 42 - 3 / 8 - 1 / 12 : ℚ) = -5 / 168 :=
by
  sorry

end simplify_fraction_expression_l149_149666


namespace probability_triangle_l149_149514

noncomputable def points : List (ℕ × ℕ) := [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2), (3, 3)]

def collinear (p1 p2 p3 : (ℕ × ℕ)) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

def is_triangle (p1 p2 p3 : (ℕ × ℕ)) : Prop := ¬ collinear p1 p2 p3

axiom collinear_ACEF : collinear (0, 0) (1, 1) (2, 2) ∧ collinear (0, 0) (1, 1) (3, 3) ∧ collinear (1, 1) (2, 2) (3, 3)
axiom collinear_BCD : collinear (2, 0) (1, 1) (0, 2)

theorem probability_triangle : 
  let total := 20
  let collinear_ACEF := 4
  let collinear_BCD := 1
  (total - collinear_ACEF - collinear_BCD) / total = 3 / 4 :=
by
  sorry

end probability_triangle_l149_149514


namespace parrot_initial_phrases_l149_149167

theorem parrot_initial_phrases (current_phrases : ℕ) (days_with_parrot : ℕ) (phrases_per_week : ℕ) (initial_phrases : ℕ) :
  current_phrases = 17 →
  days_with_parrot = 49 →
  phrases_per_week = 2 →
  initial_phrases = current_phrases - phrases_per_week * (days_with_parrot / 7) :=
by
  sorry

end parrot_initial_phrases_l149_149167


namespace p_necessary_not_sufficient_for_q_l149_149202

variables (a b c : ℝ) (p q : Prop)

def condition_p : Prop := a * b * c = 0
def condition_q : Prop := a = 0

theorem p_necessary_not_sufficient_for_q : (q → p) ∧ ¬ (p → q) :=
by
  let p := condition_p a b c
  let q := condition_q a
  sorry

end p_necessary_not_sufficient_for_q_l149_149202


namespace number_of_rectangles_on_3x3_grid_l149_149088

-- Define the grid and its properties
structure Grid3x3 where
  sides_are_2_units_apart : Bool
  diagonal_connections_allowed : Bool
  condition : sides_are_2_units_apart = true ∧ diagonal_connections_allowed = true

-- Define the number_rectangles function
def number_rectangles (g : Grid3x3) : Nat := 60

-- Define the theorem to prove the number of rectangles
theorem number_of_rectangles_on_3x3_grid : ∀ (g : Grid3x3), g.sides_are_2_units_apart = true ∧ g.diagonal_connections_allowed = true → number_rectangles g = 60 := by
  intro g
  intro h
  -- proof goes here
  sorry

end number_of_rectangles_on_3x3_grid_l149_149088


namespace minimum_value_of_expression_l149_149865

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  a^2 + b^2 + c^2 + (3 / (a + b + c)^2) ≥ 2 :=
sorry

end minimum_value_of_expression_l149_149865


namespace find_y_l149_149236

theorem find_y (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  y = 0.25 := 
sorry

end find_y_l149_149236


namespace num_girls_went_to_spa_l149_149778

-- Define the condition that each girl has 20 nails
def nails_per_girl : ℕ := 20

-- Define the total number of nails polished
def total_nails_polished : ℕ := 40

-- Define the number of girls
def number_of_girls : ℕ := total_nails_polished / nails_per_girl

-- The theorem we want to prove
theorem num_girls_went_to_spa : number_of_girls = 2 :=
by
  unfold number_of_girls
  unfold total_nails_polished
  unfold nails_per_girl
  sorry

end num_girls_went_to_spa_l149_149778


namespace famous_quote_author_l149_149351

-- conditions
def statement_date := "July 20, 1969"
def mission := "Apollo 11"
def astronauts := ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]
def first_to_moon := "Neil Armstrong"

-- goal
theorem famous_quote_author : (statement_date = "July 20, 1969") ∧ (mission = "Apollo 11") ∧ (astronauts = ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]) ∧ (first_to_moon = "Neil Armstrong") → "Neil Armstrong" = "Neil Armstrong" :=
by 
  intros _; 
  exact rfl

end famous_quote_author_l149_149351


namespace num_integer_pairs_l149_149365

theorem num_integer_pairs (m n : ℤ) :
  0 < m ∧ m < n ∧ n < 53 ∧ 53^2 + m^2 = 52^2 + n^2 →
  ∃ k, k = 3 := 
sorry

end num_integer_pairs_l149_149365


namespace circle_arc_sum_bounds_l149_149181

open Nat

theorem circle_arc_sum_bounds :
  let red_points := 40
  let blue_points := 30
  let green_points := 20
  let total_arcs := 90
  let T := 0 * red_points + 1 * blue_points + 2 * green_points
  let S_min := 6
  let S_max := 140
  (∀ S, (S = 2 * T - A) → (0 ≤ A ∧ A ≤ 134) → (S_min ≤ S ∧ S ≤ S_max))
  → ∃ S_min S_max, S_min = 6 ∧ S_max = 140 :=
by
  intros
  sorry

end circle_arc_sum_bounds_l149_149181


namespace zed_to_wyes_l149_149992

theorem zed_to_wyes (value_ex: ℝ) (value_wye: ℝ) (value_zed: ℝ)
  (h1: 2 * value_ex = 29 * value_wye)
  (h2: value_zed = 16 * value_ex) : value_zed = 232 * value_wye := by
  sorry

end zed_to_wyes_l149_149992


namespace area_of_figure_l149_149811
-- Import necessary libraries

-- Define the conditions as functions/constants
def length_left : ℕ := 7
def width_top : ℕ := 6
def height_middle : ℕ := 3
def width_middle : ℕ := 4
def height_right : ℕ := 5
def width_right : ℕ := 5

-- State the problem as a theorem
theorem area_of_figure : 
  (length_left * width_top) + 
  (width_middle * height_middle) + 
  (width_right * height_right) = 79 := 
  by
  sorry

end area_of_figure_l149_149811


namespace infinite_series_sum_eq_l149_149915

theorem infinite_series_sum_eq : 
  (∑' n : ℕ, if n = 0 then 0 else ((1 : ℝ) / (n * (n + 3)))) = (11 / 18 : ℝ) :=
sorry

end infinite_series_sum_eq_l149_149915


namespace gcd_lcm_identity_l149_149443

theorem gcd_lcm_identity (a b: ℕ) (h_lcm: (Nat.lcm a b) = 4620) (h_gcd: Nat.gcd a b = 33) (h_a: a = 231) : b = 660 := by
  sorry

end gcd_lcm_identity_l149_149443


namespace inv_sum_mod_l149_149164

theorem inv_sum_mod (x y : ℤ) (h1 : 5 * x ≡ 1 [ZMOD 23]) (h2 : 25 * y ≡ 1 [ZMOD 23]) : (x + y) ≡ 3 [ZMOD 23] := by
  sorry

end inv_sum_mod_l149_149164


namespace domain_of_log_sqrt_l149_149998

noncomputable def domain_of_function := {x : ℝ | (2 * x - 1 > 0) ∧ (2 * x - 1 ≠ 1) ∧ (3 * x - 2 > 0)}

theorem domain_of_log_sqrt : domain_of_function = {x : ℝ | (2 / 3 < x ∧ x < 1) ∨ (1 < x)} :=
by sorry

end domain_of_log_sqrt_l149_149998


namespace Pat_worked_days_eq_57_l149_149431

def Pat_earnings (x : ℕ) : ℤ := 100 * x
def Pat_food_costs (x : ℕ) : ℤ := 20 * (70 - x)
def total_balance (x : ℕ) : ℤ := Pat_earnings x - Pat_food_costs x

theorem Pat_worked_days_eq_57 (x : ℕ) (h : total_balance x = 5440) : x = 57 :=
by
  sorry

end Pat_worked_days_eq_57_l149_149431


namespace withdrawal_amount_in_2008_l149_149053

noncomputable def total_withdrawal (a : ℕ) (p : ℝ) : ℝ :=
  (a / p) * ((1 + p) - (1 + p)^8)

theorem withdrawal_amount_in_2008 (a : ℕ) (p : ℝ) (h_pos : 0 < p) (h_neg_one_lt : -1 < p) :
  total_withdrawal a p = (a / p) * ((1 + p) - (1 + p)^8) :=
by
  -- Conditions
  -- Starting from May 10th, 2001, multiple annual deposits.
  -- Annual interest rate p > 0 and p > -1.
  sorry

end withdrawal_amount_in_2008_l149_149053


namespace median_number_of_children_is_three_l149_149469

/-- Define the context of the problem with total number of families. -/
def total_families : Nat := 15

/-- Prove that given the conditions, the median number of children is 3. -/
theorem median_number_of_children_is_three 
  (h : total_families = 15) : 
  ∃ median : Nat, median = 3 :=
by
  sorry

end median_number_of_children_is_three_l149_149469


namespace intersection_eq_l149_149061

def setA : Set ℕ := {0, 1, 2, 3, 4, 5 }
def setB : Set ℕ := { x | |(x : ℤ) - 2| ≤ 1 }

theorem intersection_eq :
  setA ∩ setB = {1, 2, 3} := by
  sorry

end intersection_eq_l149_149061


namespace windows_ways_l149_149560

theorem windows_ways (n : ℕ) (h : n = 8) : (n * (n - 1)) = 56 :=
by
  sorry

end windows_ways_l149_149560


namespace average_mark_first_class_l149_149260

theorem average_mark_first_class (A : ℝ)
  (class1_students class2_students : ℝ)
  (avg2 combined_avg total_students total_marks_combined : ℝ)
  (h1 : class1_students = 22)
  (h2 : class2_students = 28)
  (h3 : avg2 = 60)
  (h4 : combined_avg = 51.2)
  (h5 : total_students = class1_students + class2_students)
  (h6 : total_marks_combined = total_students * combined_avg)
  (h7 : 22 * A + 28 * avg2 = total_marks_combined) :
  A = 40 :=
by
  sorry

end average_mark_first_class_l149_149260


namespace no_value_of_b_valid_l149_149387

theorem no_value_of_b_valid (b n : ℤ) : b^2 + 3 * b + 1 ≠ n^2 := by
  sorry

end no_value_of_b_valid_l149_149387


namespace relationship_between_a_and_b_l149_149448

variable (a b : ℝ)

-- Conditions: Points lie on the line y = 2x + 1
def point_M (a : ℝ) : Prop := a = 2 * 2 + 1
def point_N (b : ℝ) : Prop := b = 2 * 3 + 1

-- Prove that a < b given the conditions
theorem relationship_between_a_and_b (hM : point_M a) (hN : point_N b) : a < b := 
sorry

end relationship_between_a_and_b_l149_149448


namespace days_left_in_year_is_100_l149_149931

noncomputable def days_left_in_year 
    (daily_average_rain_before : ℝ) 
    (total_rainfall_so_far : ℝ) 
    (average_rain_needed : ℝ) 
    (total_days_in_year : ℕ) : ℕ :=
    sorry

theorem days_left_in_year_is_100 :
    days_left_in_year 2 430 3 365 = 100 := 
sorry

end days_left_in_year_is_100_l149_149931


namespace find_number_with_21_multiples_of_4_l149_149550

theorem find_number_with_21_multiples_of_4 (n : ℕ) (h₁ : ∀ k : ℕ, n + k * 4 ≤ 92 → k < 21) : n = 80 :=
sorry

end find_number_with_21_multiples_of_4_l149_149550


namespace root_exponent_equiv_l149_149943

theorem root_exponent_equiv :
  (7 ^ (1 / 2)) / (7 ^ (1 / 4)) = 7 ^ (1 / 4) := by
  sorry

end root_exponent_equiv_l149_149943


namespace treasure_chest_coins_l149_149179

theorem treasure_chest_coins :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 5) ∧ (n ≥ 0) ∧
  (∀ m : ℕ, (m % 8 = 6) ∧ (m % 9 = 5) → m ≥ 0 → n ≤ m) ∧
  (∃ r : ℕ, n = 11 * (n / 11) + r ∧ r = 3) :=
by
  sorry

end treasure_chest_coins_l149_149179


namespace middle_card_is_five_l149_149926

section card_numbers

variables {a b c : ℕ}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def sum_fifteen (a b c : ℕ) : Prop := a + b + c = 15
def sum_two_smallest_less_than_ten (a b : ℕ) : Prop := a + b < 10
def ascending_order (a b c : ℕ) : Prop := a < b ∧ b < c 

-- Main theorem statement
theorem middle_card_is_five 
  (h1 : distinct a b c)
  (h2 : sum_fifteen a b c)
  (h3 : sum_two_smallest_less_than_ten a b) 
  (h4 : ascending_order a b c)
  (h5 : ∀ x, (x = a → (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten x b ∧ ascending_order x b c ∧ ¬ (b = 5 ∧ c = 10))) →
           (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten b c ∧ ascending_order x b c ∧ ¬ (b = 2 ∧ c = 7)))
  (h6 : ∀ x, (x = c → (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 1 ∧ b = 4))) →
           (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 2 ∧ b = 6)))
  (h7 : ∀ x, (x = b → (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 9 ∨ a = 2 ∧ c = 8))) →
           (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 6 ∨ a = 2 ∧ c = 5)))
  : b = 5 := sorry

end card_numbers

end middle_card_is_five_l149_149926


namespace total_flowers_l149_149955

noncomputable def yellow_flowers : ℕ := 10
noncomputable def purple_flowers : ℕ := yellow_flowers + (80 * yellow_flowers) / 100
noncomputable def green_flowers : ℕ := (25 * (yellow_flowers + purple_flowers)) / 100
noncomputable def red_flowers : ℕ := (35 * (yellow_flowers + purple_flowers + green_flowers)) / 100

theorem total_flowers :
  yellow_flowers + purple_flowers + green_flowers + red_flowers = 47 :=
by
  -- Insert proof here
  sorry

end total_flowers_l149_149955


namespace even_of_even_square_sqrt_two_irrational_l149_149204

-- Problem 1: Let p ∈ ℤ. Show that if p² is even, then p is even.
theorem even_of_even_square (p : ℤ) (h : p^2 % 2 = 0) : p % 2 = 0 :=
by
  sorry

-- Problem 2: Show that √2 is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ a * a = 2 * b * b :=
by
  sorry

end even_of_even_square_sqrt_two_irrational_l149_149204


namespace solve_quadratic_eq_l149_149853

theorem solve_quadratic_eq (x : ℝ) :
  x^2 - 7 * x + 6 = 0 ↔ x = 1 ∨ x = 6 :=
by
  sorry

end solve_quadratic_eq_l149_149853


namespace next_consecutive_time_l149_149537

theorem next_consecutive_time (current_hour : ℕ) (current_minute : ℕ) 
  (valid_minutes : 0 ≤ current_minute ∧ current_minute < 60) 
  (valid_hours : 0 ≤ current_hour ∧ current_hour < 24) : 
  current_hour = 4 ∧ current_minute = 56 →
  ∃ next_hour next_minute : ℕ, 
    (0 ≤ next_minute ∧ next_minute < 60) ∧ 
    (0 ≤ next_hour ∧ next_hour < 24) ∧
    (next_hour, next_minute) = (12, 34) ∧ 
    (next_hour * 60 + next_minute) - (current_hour * 60 + current_minute) = 458 := 
by sorry

end next_consecutive_time_l149_149537


namespace coordinates_of_C_l149_149841

theorem coordinates_of_C (A B : ℝ × ℝ) (hA : A = (-2, -1)) (hB : B = (4, 9)) :
    ∃ C : ℝ × ℝ, (dist C A) = 4 * dist C B ∧ C = (-0.8, 1) :=
sorry

end coordinates_of_C_l149_149841


namespace find_other_endpoint_diameter_l149_149482

-- Define the given conditions
def center : ℝ × ℝ := (1, 2)
def endpoint_A : ℝ × ℝ := (4, 6)

-- Define a function to find the other endpoint
def other_endpoint (center endpoint_A : ℝ × ℝ) : ℝ × ℝ := 
  let vector_CA := (center.1 - endpoint_A.1, center.2 - endpoint_A.2)
  let vector_CB := (-vector_CA.1, -vector_CA.2)
  (center.1 + vector_CB.1, center.2 + vector_CB.2)

-- State the theorem
theorem find_other_endpoint_diameter : 
  ∀ center endpoint_A, other_endpoint center endpoint_A = (4, 6) :=
by
  intro center endpoint_A
  -- Proof would go here
  sorry

end find_other_endpoint_diameter_l149_149482


namespace equation_solution_l149_149273

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (2 * x / (x - 2) - 2 = 1 / (x * (x - 2))) ↔ x = 1 / 4 :=
by
  sorry

end equation_solution_l149_149273


namespace smallest_prime_with_digit_sum_23_l149_149347

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l149_149347


namespace area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l149_149836

-- (a) Proving the area of triangle ABC
theorem area_triangle_ABC (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 3) (h_right : true) : 
  (1 / 2) * AB * BC = 3 := sorry

-- (b) Proving the area of figure DEFGH
theorem area_figure_DEFGH (DH HG : ℝ) (hDH : DH = 5) (hHG : HG = 5) (triangle_area : ℝ) (hEPF : triangle_area = 3) : 
  DH * HG - triangle_area = 22 := sorry

-- (c) Proving the area of triangle JKL 
theorem area_triangle_JKL (side_area : ℝ) (h_side : side_area = 25) 
  (area_JSK : ℝ) (h_JSK : area_JSK = 3) 
  (area_LQJ : ℝ) (h_LQJ : area_LQJ = 15/2) 
  (area_LRK : ℝ) (h_LRK : area_LRK = 5) : 
  side_area - area_JSK - area_LQJ - area_LRK = 19/2 := sorry

end area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l149_149836


namespace conditional_probability_correct_l149_149193

noncomputable def total_products : ℕ := 8
noncomputable def first_class_products : ℕ := 6
noncomputable def chosen_products : ℕ := 2

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def P_A : ℚ := 1 - (combination first_class_products chosen_products) / (combination total_products chosen_products)
noncomputable def P_AB : ℚ := (combination 2 1 * combination first_class_products 1) / (combination total_products chosen_products)

noncomputable def conditional_probability : ℚ := P_AB / P_A

theorem conditional_probability_correct :
  conditional_probability = 12 / 13 :=
  sorry

end conditional_probability_correct_l149_149193


namespace incorrect_statement_among_given_options_l149_149471

theorem incorrect_statement_among_given_options :
  (∀ (b h : ℝ), 3 * (b * h) = (3 * b) * h) ∧
  (∀ (b h : ℝ), 3 * (1 / 2 * b * h) = 1 / 2 * b * (3 * h)) ∧
  (∀ (π r : ℝ), 9 * (π * r * r) ≠ (π * (3 * r) * (3 * r))) ∧
  (∀ (a b : ℝ), (3 * a) / (2 * b) ≠ a / b) ∧
  (∀ (x : ℝ), x < 0 → 3 * x < x) →
  false :=
by
  sorry

end incorrect_statement_among_given_options_l149_149471


namespace sum_of_digits_of_d_l149_149207

theorem sum_of_digits_of_d (d : ℕ) (h₁ : ∃ d_ca : ℕ, d_ca = (8 * d) / 5) (h₂ : d_ca - 75 = d) :
  (1 + 2 + 5 = 8) :=
by
  sorry

end sum_of_digits_of_d_l149_149207


namespace radius_inscribed_sphere_quadrilateral_pyramid_l149_149506

noncomputable def radius_of_inscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt 5 - 1) / 4

theorem radius_inscribed_sphere_quadrilateral_pyramid (a : ℝ) :
  r = radius_of_inscribed_sphere a :=
by
  -- problem conditions:
  -- side of the base a
  -- height a
  -- result: r = a * (Real.sqrt 5 - 1) / 4
  sorry

end radius_inscribed_sphere_quadrilateral_pyramid_l149_149506


namespace trig_identity_l149_149376

theorem trig_identity (θ : ℝ) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
  sorry

end trig_identity_l149_149376


namespace total_cases_l149_149393

def NY : ℕ := 2000
def CA : ℕ := NY / 2
def TX : ℕ := CA - 400

theorem total_cases : NY + CA + TX = 3600 :=
by
  -- use sorry placeholder to indicate the solution is omitted
  sorry

end total_cases_l149_149393


namespace sock_pairs_count_l149_149758

theorem sock_pairs_count :
  let white_socks := 5
  let brown_socks := 3
  let blue_socks := 4
  let blue_white_pairs := blue_socks * white_socks
  let blue_brown_pairs := blue_socks * brown_socks
  let total_pairs := blue_white_pairs + blue_brown_pairs
  total_pairs = 32 :=
by
  sorry

end sock_pairs_count_l149_149758


namespace find_excluded_number_l149_149863

-- Definition of the problem conditions
def avg (nums : List ℕ) : ℕ := (nums.sum / nums.length)

-- Problem condition: the average of 5 numbers is 27
def condition1 (nums : List ℕ) : Prop :=
  nums.length = 5 ∧ avg nums = 27

-- Problem condition: excluding one number, the average of remaining 4 numbers is 25
def condition2 (nums : List ℕ) (x : ℕ) : Prop :=
  let nums' := nums.filter (λ n => n ≠ x)
  nums.length = 5 ∧ nums'.length = 4 ∧ avg nums' = 25

-- Proof statement: finding the excluded number
theorem find_excluded_number (nums : List ℕ) (x : ℕ) (h1 : condition1 nums) (h2 : condition2 nums x) : x = 35 := 
by
  sorry

end find_excluded_number_l149_149863
