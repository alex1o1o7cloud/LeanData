import Mathlib

namespace who_made_statements_and_fate_l759_75984

namespace IvanTsarevichProblem

-- Define the characters and their behaviors
inductive Animal
| Bear : Animal
| Fox : Animal
| Wolf : Animal

def always_true (s : Prop) : Prop := s
def always_false (s : Prop) : Prop := ¬s
def alternates (s1 s2 : Prop) : Prop := s1 ∧ ¬s2

-- Statements made by the animals
def statement1 (save_die : Bool) : Prop := save_die = true
def statement2 (safe_sound_save : Bool) : Prop := safe_sound_save = true
def statement3 (safe_lose : Bool) : Prop := safe_lose = true

-- Analyze truth based on behaviors
noncomputable def belongs_to (a : Animal) (s : Prop) : Prop :=
  match a with
  | Animal.Bear => always_true s
  | Animal.Fox => always_false s
  | Animal.Wolf =>
    match s with
    | ss => alternates (ss = true) (ss = false)

-- Given conditions
axiom h1 : statement1 false -- Fox lies, so "You will save the horse. But you will die." is false
axiom h2 : statement2 false -- Wolf alternates, so "You will stay safe and sound. And you will save the horse." is a mix
axiom h3 : statement3 true  -- Bear tells the truth, so "You will survive. But you will lose the horse." is true

-- Conclusion: Animal who made each statement
theorem who_made_statements_and_fate : 
  belongs_to Animal.Fox (statement1 false) ∧ 
  belongs_to Animal.Wolf (statement2 false) ∧ 
  belongs_to Animal.Bear (statement3 true) ∧ 
  (¬safe_lose) := sorry

end IvanTsarevichProblem

end who_made_statements_and_fate_l759_75984


namespace sin_thirty_degrees_l759_75933

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l759_75933


namespace expected_value_is_correct_l759_75951

-- Define the monetary outcomes associated with each side
def monetaryOutcome (X : String) : ℚ :=
  if X = "A" then 2 else 
  if X = "B" then -4 else 
  if X = "C" then 6 else 
  0

-- Define the probabilities associated with each side
def probability (X : String) : ℚ :=
  if X = "A" then 1/3 else 
  if X = "B" then 1/2 else 
  if X = "C" then 1/6 else 
  0

-- Compute the expected value
def expectedMonetaryOutcome : ℚ := (probability "A" * monetaryOutcome "A") 
                                + (probability "B" * monetaryOutcome "B") 
                                + (probability "C" * monetaryOutcome "C")

theorem expected_value_is_correct : 
  expectedMonetaryOutcome = -2/3 := by
  sorry

end expected_value_is_correct_l759_75951


namespace find_second_dimension_l759_75955

theorem find_second_dimension (x : ℕ) 
    (h1 : 12 * x * 16 / (3 * 7 * 2) = 64) : 
    x = 14 := by
    sorry

end find_second_dimension_l759_75955


namespace smallest_value_of_a_l759_75937

theorem smallest_value_of_a (a b c d : ℤ) (h1 : (a - 2 * b) > 0) (h2 : (b - 3 * c) > 0) (h3 : (c - 4 * d) > 0) (h4 : d > 100) : a ≥ 2433 := sorry

end smallest_value_of_a_l759_75937


namespace max_remaining_grapes_l759_75927

theorem max_remaining_grapes (x : ℕ) : x % 7 ≤ 6 :=
  sorry

end max_remaining_grapes_l759_75927


namespace solve_for_x_l759_75964

theorem solve_for_x (x : ℝ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 := 
  sorry

end solve_for_x_l759_75964


namespace length_of_platform_l759_75907

-- Definitions based on the problem conditions
def train_length : ℝ := 300
def platform_crossing_time : ℝ := 39
def signal_pole_crossing_time : ℝ := 18

-- The main theorem statement
theorem length_of_platform : ∀ (L : ℝ), train_length + L = (train_length / signal_pole_crossing_time) * platform_crossing_time → L = 350.13 :=
by
  intro L h
  sorry

end length_of_platform_l759_75907


namespace rectangular_prism_volume_l759_75916

theorem rectangular_prism_volume (h : ℝ) : 
  ∃ (V : ℝ), V = 120 * h :=
by
  sorry

end rectangular_prism_volume_l759_75916


namespace remainder_17_pow_77_mod_7_l759_75919

theorem remainder_17_pow_77_mod_7 : (17^77) % 7 = 5 := 
by sorry

end remainder_17_pow_77_mod_7_l759_75919


namespace total_spears_is_78_l759_75944

-- Define the spear production rates for each type of wood
def spears_from_sapling := 3
def spears_from_log := 9
def spears_from_bundle := 7
def spears_from_trunk := 15

-- Define the quantity of each type of wood
def saplings := 6
def logs := 1
def bundles := 3
def trunks := 2

-- Prove that the total number of spears is 78
theorem total_spears_is_78 : (saplings * spears_from_sapling) + (logs * spears_from_log) + (bundles * spears_from_bundle) + (trunks * spears_from_trunk) = 78 :=
by 
  -- Calculation can be filled here
  sorry

end total_spears_is_78_l759_75944


namespace cannot_factorize_using_difference_of_squares_l759_75977

theorem cannot_factorize_using_difference_of_squares (x y : ℝ) :
  ¬ ∃ a b : ℝ, -x^2 - y^2 = a^2 - b^2 :=
sorry

end cannot_factorize_using_difference_of_squares_l759_75977


namespace sin_1035_eq_neg_sqrt2_div_2_l759_75900

theorem sin_1035_eq_neg_sqrt2_div_2 : Real.sin (1035 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
    sorry

end sin_1035_eq_neg_sqrt2_div_2_l759_75900


namespace days_to_fill_tank_l759_75948

-- Definitions based on the problem conditions
def tank_capacity_liters : ℕ := 50
def liters_to_milliliters : ℕ := 1000
def rain_collection_per_day : ℕ := 800
def river_collection_per_day : ℕ := 1700
def total_collection_per_day : ℕ := rain_collection_per_day + river_collection_per_day
def tank_capacity_milliliters : ℕ := tank_capacity_liters * liters_to_milliliters

-- Statement of the proof that Jacob needs 20 days to fill the tank
theorem days_to_fill_tank : tank_capacity_milliliters / total_collection_per_day = 20 := by
  sorry

end days_to_fill_tank_l759_75948


namespace lowest_possible_sale_price_percentage_l759_75910

noncomputable def list_price : ℝ := 80
noncomputable def max_initial_discount_percent : ℝ := 0.5
noncomputable def summer_sale_discount_percent : ℝ := 0.2
noncomputable def membership_discount_percent : ℝ := 0.1
noncomputable def coupon_discount_percent : ℝ := 0.05

theorem lowest_possible_sale_price_percentage :
  let max_initial_discount := max_initial_discount_percent * list_price
  let summer_sale_discount := summer_sale_discount_percent * list_price
  let membership_discount := membership_discount_percent * list_price
  let coupon_discount := coupon_discount_percent * list_price
  let lowest_sale_price := list_price * (1 - max_initial_discount_percent) - summer_sale_discount - membership_discount - coupon_discount
  (lowest_sale_price / list_price) * 100 = 15 :=
by
  sorry

end lowest_possible_sale_price_percentage_l759_75910


namespace parabola_equation_maximum_area_of_triangle_l759_75915

-- Definitions of the conditions
def parabola_eq (x y : ℝ) (p : ℝ) : Prop := x^2 = 2 * p * y ∧ p > 0
def distances_equal (AO AF : ℝ) : Prop := AO = 3 / 2 ∧ AF = 3 / 2
def line_eq (x k b y : ℝ) : Prop := y = k * x + b
def midpoint_y (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 1

-- Part (I)
theorem parabola_equation (p : ℝ) (x y AO AF : ℝ) (h1 : parabola_eq x y p)
  (h2 : distances_equal AO AF) :
  x^2 = 4 * y :=
sorry

-- Part (II)
theorem maximum_area_of_triangle (p k b AO AF x1 y1 x2 y2 : ℝ)
  (h1 : parabola_eq x1 y1 p) (h2 : parabola_eq x2 y2 p)
  (h3 : distances_equal AO AF) (h4 : line_eq x1 k b y1) 
  (h5 : line_eq x2 k b y2) (h6 : midpoint_y y1 y2)
  : ∃ (area : ℝ), area = 2 :=
sorry

end parabola_equation_maximum_area_of_triangle_l759_75915


namespace factor_cubic_expression_l759_75980

theorem factor_cubic_expression :
  ∃ a b c : ℕ, 
  a > b ∧ b > c ∧ 
  x^3 - 16 * x^2 + 65 * x - 80 = (x - a) * (x - b) * (x - c) ∧ 
  3 * b - c = 12 := 
sorry

end factor_cubic_expression_l759_75980


namespace intersection_A_B_l759_75939

theorem intersection_A_B :
  let A := {1, 3, 5, 7}
  let B := {x | x^2 - 2 * x - 5 ≤ 0}
  A ∩ B = {1, 3} := by
sorry

end intersection_A_B_l759_75939


namespace min_value_problem1_l759_75985

theorem min_value_problem1 (x : ℝ) (hx : x > -1) : 
  ∃ m, m = 2 * Real.sqrt 2 + 1 ∧ (∀ y, y = (x^2 + 3 * x + 4) / (x + 1) ∧ x > -1 → y ≥ m) :=
sorry

end min_value_problem1_l759_75985


namespace mass_percentage_Ca_in_mixture_l759_75956

theorem mass_percentage_Ca_in_mixture :
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  percentage_Ca = 26.69 :=
by
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  have : percentage_Ca = 26.69 := by sorry
  exact this

end mass_percentage_Ca_in_mixture_l759_75956


namespace q1_q2_q3_l759_75942

noncomputable def quadratic_function (a x: ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem q1 (a : ℝ) : (∀ {x : ℝ}, quadratic_function a x = 0 → x < 2) ∧ (quadratic_function a 2 > 0) ∧ (2 * a ≠ 0) → a < -1 := 
by 
  sorry

theorem q2 (a : ℝ) : (∀ x : ℝ, quadratic_function a x ≥ -1 - a * x) → -2 ≤ a ∧ a ≤ 6 := 
by 
  sorry
  
theorem q3 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → quadratic_function a x ≤ 4) → a = 2 ∨ a = 2 / 3 := 
by 
  sorry

end q1_q2_q3_l759_75942


namespace calculate_difference_of_squares_l759_75986

theorem calculate_difference_of_squares : (153^2 - 147^2) = 1800 := by
  sorry

end calculate_difference_of_squares_l759_75986


namespace neither_sufficient_nor_necessary_condition_l759_75921

theorem neither_sufficient_nor_necessary_condition (a b : ℝ) :
  ¬ ((a < 0 ∧ b < 0) → (a * b * (a - b) > 0)) ∧
  ¬ ((a * b * (a - b) > 0) → (a < 0 ∧ b < 0)) :=
by
  sorry

end neither_sufficient_nor_necessary_condition_l759_75921


namespace estimate_red_balls_l759_75957

-- Definitions based on conditions
def total_balls : ℕ := 20
def total_draws : ℕ := 100
def red_draws : ℕ := 30

-- The theorem statement
theorem estimate_red_balls (h1 : total_balls = 20) (h2 : total_draws = 100) (h3 : red_draws = 30) :
  (total_balls * (red_draws / total_draws) : ℤ) = 6 := 
by
  sorry

end estimate_red_balls_l759_75957


namespace find_smallest_n_l759_75982

theorem find_smallest_n 
  (n : ℕ) 
  (hn : 23 * n ≡ 789 [MOD 8]) : 
  ∃ n : ℕ, n > 0 ∧ n ≡ 3 [MOD 8] :=
sorry

end find_smallest_n_l759_75982


namespace intersect_points_count_l759_75965

open Classical
open Real

noncomputable def f : ℝ → ℝ := sorry
def f_inv : ℝ → ℝ := sorry

axiom f_invertible : ∀ x y : ℝ, f x = f y ↔ x = y

theorem intersect_points_count : ∃ (count : ℕ), count = 3 ∧ ∀ x : ℝ, (f (x ^ 3) = f (x ^ 5)) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by sorry

end intersect_points_count_l759_75965


namespace no_four_consecutive_powers_l759_75990

/-- 
  There do not exist four consecutive natural numbers 
  such that each of them is a power (greater than 1) of another natural number.
-/
theorem no_four_consecutive_powers : 
  ¬ ∃ (n : ℕ), (∀ (i : ℕ), i < 4 → ∃ (a k : ℕ), k > 1 ∧ n + i = a^k) := sorry

end no_four_consecutive_powers_l759_75990


namespace estimate_total_height_l759_75978

theorem estimate_total_height :
  let middle_height := 100
  let left_height := 0.80 * middle_height
  let right_height := (left_height + middle_height) - 20
  left_height + middle_height + right_height = 340 := 
by
  sorry

end estimate_total_height_l759_75978


namespace sum_greater_than_four_l759_75997

theorem sum_greater_than_four (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hprod : x * y > x + y) : x + y > 4 :=
by
  sorry

end sum_greater_than_four_l759_75997


namespace octal_742_to_decimal_l759_75999

theorem octal_742_to_decimal : (7 * 8^2 + 4 * 8^1 + 2 * 8^0 = 482) :=
by
  sorry

end octal_742_to_decimal_l759_75999


namespace probability_green_ball_l759_75935

theorem probability_green_ball 
  (total_balls : ℕ) 
  (green_balls : ℕ) 
  (white_balls : ℕ) 
  (h_total : total_balls = 9) 
  (h_green : green_balls = 7)
  (h_white : white_balls = 2)
  (h_total_eq : total_balls = green_balls + white_balls) : 
  (green_balls / total_balls : ℚ) = 7 / 9 := 
by
  sorry

end probability_green_ball_l759_75935


namespace cos_squared_identity_l759_75979

variable (θ : ℝ)

-- Given condition
def tan_theta : Prop := Real.tan θ = 2

-- Question: Find the value of cos²(θ + π/4)
theorem cos_squared_identity (h : tan_theta θ) : Real.cos (θ + Real.pi / 4) ^ 2 = 1 / 10 := 
  sorry

end cos_squared_identity_l759_75979


namespace total_cost_is_716_mom_has_enough_money_l759_75953

/-- Definition of the price of the table lamp -/
def table_lamp_price : ℕ := 86

/-- Definition of the price of the electric fan -/
def electric_fan_price : ℕ := 185

/-- Definition of the price of the bicycle -/
def bicycle_price : ℕ := 445

/-- The total cost of buying all three items -/
def total_cost : ℕ := table_lamp_price + electric_fan_price + bicycle_price

/-- Mom's money -/
def mom_money : ℕ := 300

/-- Problem 1: Prove that the total cost equals 716 -/
theorem total_cost_is_716 : total_cost = 716 := 
by 
  sorry

/-- Problem 2: Prove that Mom has enough money to buy a table lamp and an electric fan -/
theorem mom_has_enough_money : table_lamp_price + electric_fan_price ≤ mom_money :=
by 
  sorry

end total_cost_is_716_mom_has_enough_money_l759_75953


namespace number_of_sides_l759_75934

theorem number_of_sides (n : ℕ) : 
  (2 / 9) * (n - 2) * 180 = 360 → n = 11 := 
by
  intro h
  sorry

end number_of_sides_l759_75934


namespace xyz_inequality_l759_75912

theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end xyz_inequality_l759_75912


namespace min_value_expression_l759_75923

theorem min_value_expression (x : ℚ) : ∃ x : ℚ, (2 * x - 5)^2 + 18 = 18 :=
by {
  use 2.5,
  sorry
}

end min_value_expression_l759_75923


namespace total_number_of_trees_l759_75973

theorem total_number_of_trees (D P : ℕ) (cost_D cost_P total_cost : ℕ)
  (hD : D = 350)
  (h_cost_D : cost_D = 300)
  (h_cost_P : cost_P = 225)
  (h_total_cost : total_cost = 217500)
  (h_cost_equation : cost_D * D + cost_P * P = total_cost) :
  D + P = 850 :=
by
  rw [hD, h_cost_D, h_cost_P, h_total_cost] at h_cost_equation
  sorry

end total_number_of_trees_l759_75973


namespace winning_percentage_l759_75903

/-- In an election with two candidates, wherein the winner received 490 votes and won by 280 votes,
we aim to prove that the winner received 70% of the total votes. -/

theorem winning_percentage (votes_winner : ℕ) (votes_margin : ℕ) (total_votes : ℕ)
  (h1 : votes_winner = 490) (h2 : votes_margin = 280)
  (h3 : total_votes = votes_winner + (votes_winner - votes_margin)) :
  (votes_winner * 100 / total_votes) = 70 :=
by
  -- Skipping the proof for now
  sorry

end winning_percentage_l759_75903


namespace triangle_is_right_triangle_l759_75981

theorem triangle_is_right_triangle 
  {A B C : ℝ} {a b c : ℝ} 
  (h₁ : b - a * Real.cos B = a * Real.cos C - c) 
  (h₂ : ∀ (angle : ℝ), 0 < angle ∧ angle < π) : A = π / 2 := 
sorry

end triangle_is_right_triangle_l759_75981


namespace ratio_of_surface_areas_of_spheres_l759_75932

theorem ratio_of_surface_areas_of_spheres (V1 V2 S1 S2 : ℝ) 
(h : V1 / V2 = 8 / 27) 
(h1 : S1 = 4 * π * (V1^(2/3)) / (2 * π)^(2/3))
(h2 : S2 = 4 * π * (V2^(2/3)) / (3 * π)^(2/3)) :
S1 / S2 = 4 / 9 :=
sorry

end ratio_of_surface_areas_of_spheres_l759_75932


namespace antonio_age_in_months_l759_75906

-- Definitions based on the conditions
def is_twice_as_old (isabella_age antonio_age : ℕ) : Prop :=
  isabella_age = 2 * antonio_age

def future_age (current_age months_future : ℕ) : ℕ :=
  current_age + months_future

-- Given the conditions
variables (isabella_age antonio_age : ℕ)
variables (future_age_18months target_age : ℕ)

-- Conditions
axiom condition1 : is_twice_as_old isabella_age antonio_age
axiom condition2 : future_age_18months = 18
axiom condition3 : target_age = 10 * 12

-- Assertion that we need to prove
theorem antonio_age_in_months :
  ∃ (antonio_age : ℕ), future_age isabella_age future_age_18months = target_age → antonio_age = 51 :=
by
  sorry

end antonio_age_in_months_l759_75906


namespace compute_u2_plus_v2_l759_75940

theorem compute_u2_plus_v2 (u v : ℝ) (hu : 1 < u) (hv : 1 < v)
  (h : (Real.log u / Real.log 3)^4 + (Real.log v / Real.log 7)^4 = 10 * (Real.log u / Real.log 3) * (Real.log v / Real.log 7)) :
  u^2 + v^2 = 3^(Real.sqrt 5) + 7^(Real.sqrt 5) :=
by
  sorry

end compute_u2_plus_v2_l759_75940


namespace average_price_per_racket_l759_75929

theorem average_price_per_racket (total_amount : ℕ) (pairs_sold : ℕ) (expected_average : ℚ) 
  (h1 : total_amount = 637) (h2 : pairs_sold = 65) : 
  expected_average = total_amount / pairs_sold := 
by
  sorry

end average_price_per_racket_l759_75929


namespace canonical_equations_of_line_l759_75949

-- Definitions for the normal vectors of the planes
def n1 : ℝ × ℝ × ℝ := (2, 3, -2)
def n2 : ℝ × ℝ × ℝ := (1, -3, 1)

-- Define the equations of the planes
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y - 2 * z + 6 = 0
def plane2 (x y z : ℝ) : Prop := x - 3 * y + z + 3 = 0

-- The canonical equations of the line of intersection
def canonical_eq (x y z : ℝ) : Prop := (z * (-4)) = (y * (-9)) ∧ (z * (-3)) = (x + 3) * (-9)

theorem canonical_equations_of_line :
  ∀ x y z : ℝ, (plane1 x y z) ∧ (plane2 x y z) → canonical_eq x y z :=
by
  sorry

end canonical_equations_of_line_l759_75949


namespace jane_evening_pages_l759_75958

theorem jane_evening_pages :
  ∀ (P : ℕ), (7 * (5 + P) = 105) → P = 10 :=
by
  intros P h
  sorry

end jane_evening_pages_l759_75958


namespace square_roots_N_l759_75994

theorem square_roots_N (m N : ℤ) (h1 : (3 * m - 4) ^ 2 = N) (h2 : (7 - 4 * m) ^ 2 = N) : N = 25 := 
by
  sorry

end square_roots_N_l759_75994


namespace alyssa_photos_vacation_l759_75905

theorem alyssa_photos_vacation
  (pages_first_section : ℕ)
  (photos_per_page_first_section : ℕ)
  (pages_second_section : ℕ)
  (photos_per_page_second_section : ℕ)
  (pages_total : ℕ)
  (photos_per_page_remaining : ℕ)
  (pages_remaining : ℕ)
  (h_total_pages : pages_first_section + pages_second_section + pages_remaining = pages_total)
  (h_photos_first_section : photos_per_page_first_section = 3)
  (h_photos_second_section : photos_per_page_second_section = 4)
  (h_pages_first_section : pages_first_section = 10)
  (h_pages_second_section : pages_second_section = 10)
  (h_photos_remaining : photos_per_page_remaining = 3)
  (h_pages_total : pages_total = 30)
  (h_pages_remaining : pages_remaining = 10) :
  pages_first_section * photos_per_page_first_section +
  pages_second_section * photos_per_page_second_section +
  pages_remaining * photos_per_page_remaining = 100 := by
sorry

end alyssa_photos_vacation_l759_75905


namespace doughnuts_in_each_box_l759_75920

theorem doughnuts_in_each_box (total_doughnuts : ℕ) (boxes : ℕ) (h1 : total_doughnuts = 48) (h2 : boxes = 4) : total_doughnuts / boxes = 12 :=
by
  sorry

end doughnuts_in_each_box_l759_75920


namespace area_percentage_increase_l759_75993

theorem area_percentage_increase (r₁ r₂ : ℝ) (π : ℝ) :
  r₁ = 6 ∧ r₂ = 4 ∧ π > 0 →
  (π * r₁^2 - π * r₂^2) / (π * r₂^2) * 100 = 125 := 
by {
  sorry
}

end area_percentage_increase_l759_75993


namespace disjoint_subsets_same_sum_l759_75959

/-- 
Given a set of 10 distinct integers between 1 and 100, 
there exist two disjoint subsets of this set that have the same sum.
-/
theorem disjoint_subsets_same_sum : ∃ (x : Finset ℤ), (x.card = 10) ∧ (∀ i ∈ x, 1 ≤ i ∧ i ≤ 100) → 
  ∃ (A B : Finset ℤ), (A ⊆ x) ∧ (B ⊆ x) ∧ (A ∩ B = ∅) ∧ (A.sum id = B.sum id) :=
by
  sorry

end disjoint_subsets_same_sum_l759_75959


namespace Eva_needs_weeks_l759_75966

theorem Eva_needs_weeks (apples : ℕ) (days_in_week : ℕ) (weeks : ℕ) 
  (h1 : apples = 14)
  (h2 : days_in_week = 7) 
  (h3 : apples = weeks * days_in_week) : 
  weeks = 2 := 
by 
  sorry

end Eva_needs_weeks_l759_75966


namespace capital_growth_rate_l759_75918

theorem capital_growth_rate
  (loan_amount : ℝ) (interest_rate : ℝ) (repayment_period : ℝ) (surplus : ℝ) (growth_rate : ℝ) :
  loan_amount = 2000000 ∧ interest_rate = 0.08 ∧ repayment_period = 2 ∧ surplus = 720000 ∧
  (loan_amount * (1 + growth_rate)^repayment_period = loan_amount * (1 + interest_rate) + surplus) →
  growth_rate = 0.2 :=
by
  sorry

end capital_growth_rate_l759_75918


namespace cartesian_to_polar_circle_l759_75943

open Real

theorem cartesian_to_polar_circle (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = ρ * cos θ) 
  (h2 : y = ρ * sin θ) 
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * cos θ :=
sorry

end cartesian_to_polar_circle_l759_75943


namespace xy_square_difference_l759_75928

variable (x y : ℚ)

theorem xy_square_difference (h1 : x + y = 8/15) (h2 : x - y = 1/45) : 
  x^2 - y^2 = 8/675 := by
  sorry

end xy_square_difference_l759_75928


namespace certain_event_among_options_l759_75931

-- Definition of the proof problem
theorem certain_event_among_options (is_random_A : Prop) (is_random_C : Prop) (is_random_D : Prop) (is_certain_B : Prop) :
  (is_random_A → (¬is_certain_B)) ∧
  (is_random_C → (¬is_certain_B)) ∧
  (is_random_D → (¬is_certain_B)) ∧
  (is_certain_B ∧ ((¬is_random_A) ∧ (¬is_random_C) ∧ (¬is_random_D))) :=
by
  sorry

end certain_event_among_options_l759_75931


namespace rectangular_prism_volume_l759_75970

theorem rectangular_prism_volume (w : ℝ) (w_pos : 0 < w) 
    (h_edges_sum : 4 * w + 8 * (2 * w) + 4 * (w / 2) = 88) :
    (2 * w) * w * (w / 2) = 85184 / 343 :=
by
  sorry

end rectangular_prism_volume_l759_75970


namespace zain_coin_total_l759_75936

def zain_coins (q d n : ℕ) := q + d + n
def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def zain_quarters : ℕ := emerie_quarters + 10
def zain_dimes : ℕ := emerie_dimes + 10
def zain_nickels : ℕ := emerie_nickels + 10

theorem zain_coin_total : zain_coins zain_quarters zain_dimes zain_nickels = 48 := 
by
  unfold zain_coins zain_quarters zain_dimes zain_nickels emerie_quarters emerie_dimes emerie_nickels
  rfl

end zain_coin_total_l759_75936


namespace largest_consecutive_integer_product_2520_l759_75992

theorem largest_consecutive_integer_product_2520 :
  ∃ (n : ℕ), n * (n + 1) * (n + 2) * (n + 3) = 2520 ∧ (n + 3) = 8 :=
by {
  sorry
}

end largest_consecutive_integer_product_2520_l759_75992


namespace count_paths_to_form_2005_l759_75908

/-- Define the structure of a circle label. -/
inductive CircleLabel
| two
| zero
| five

open CircleLabel

/-- Define the number of possible moves from each circle. -/
def moves_from_two : Nat := 6
def moves_from_zero_to_zero : Nat := 2
def moves_from_zero_to_five : Nat := 3

/-- Define the total number of paths to form 2005. -/
def total_paths : Nat := moves_from_two * moves_from_zero_to_zero * moves_from_zero_to_five

/-- The proof statement: The total number of different paths to form the number 2005 is 36. -/
theorem count_paths_to_form_2005 : total_paths = 36 :=
by
  sorry

end count_paths_to_form_2005_l759_75908


namespace intersection_M_N_l759_75913

open Set

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by 
sorry

end intersection_M_N_l759_75913


namespace ratio_girls_total_members_l759_75917

theorem ratio_girls_total_members {p_boy p_girl : ℚ} (h_prob_ratio : p_girl = (3/5) * p_boy) (h_total_prob : p_boy + p_girl = 1) :
  p_girl / (p_boy + p_girl) = 3 / 8 :=
by
  sorry

end ratio_girls_total_members_l759_75917


namespace gabrielle_saw_more_birds_l759_75941

def birds_seen (robins cardinals blue_jays : Nat) : Nat :=
  robins + cardinals + blue_jays

def percentage_difference (g c : Nat) : Nat :=
  ((g - c) * 100) / c

theorem gabrielle_saw_more_birds :
  let gabrielle := birds_seen 5 4 3
  let chase := birds_seen 2 5 3
  percentage_difference gabrielle chase = 20 := 
by
  sorry

end gabrielle_saw_more_birds_l759_75941


namespace boys_number_is_60_l759_75988

-- Definitions based on the conditions
variables (x y : ℕ)

def sum_boys_girls (x y : ℕ) : Prop := 
  x + y = 150

def girls_percentage (x y : ℕ) : Prop := 
  y = (x * 150) / 100

-- Prove that the number of boys equals 60
theorem boys_number_is_60 (x y : ℕ) 
  (h1 : sum_boys_girls x y) 
  (h2 : girls_percentage x y) : 
  x = 60 := by
  sorry

end boys_number_is_60_l759_75988


namespace pyramid_volume_is_one_sixth_l759_75930

noncomputable def volume_of_pyramid_in_cube : ℝ :=
  let edge_length := 1
  let base_area := (1 / 2) * edge_length * edge_length
  let height := edge_length
  (1 / 3) * base_area * height

theorem pyramid_volume_is_one_sixth : volume_of_pyramid_in_cube = 1 / 6 :=
by
  -- Let edge_length = 1, base_area = 1 / 2 * edge_length * edge_length = 1 / 2, 
  -- height = edge_length = 1. Then volume = 1 / 3 * base_area * height = 1 / 6.
  sorry

end pyramid_volume_is_one_sixth_l759_75930


namespace total_balls_in_bag_l759_75962

theorem total_balls_in_bag (x : ℕ) (H : 3/(4 + x) = x/(4 + x)) : 3 + 1 + x = 7 :=
by
  -- We would provide the proof here, but it's not required as per the instructions.
  sorry

end total_balls_in_bag_l759_75962


namespace tan_value_l759_75938

theorem tan_value (x : ℝ) (hx : x ∈ Set.Ioo (-π / 2) 0) (hcos : Real.cos x = 4 / 5) : Real.tan x = -3 / 4 :=
sorry

end tan_value_l759_75938


namespace peaches_left_l759_75972

/-- Brenda picks 3600 peaches, 37.5% are fresh, and 250 are disposed of. Prove that Brenda has 1100 peaches left. -/
theorem peaches_left (total_peaches : ℕ) (percent_fresh : ℚ) (peaches_disposed : ℕ) (h1 : total_peaches = 3600) (h2 : percent_fresh = 3 / 8) (h3 : peaches_disposed = 250) : 
  total_peaches * percent_fresh - peaches_disposed = 1100 := 
by
  sorry

end peaches_left_l759_75972


namespace neg_of_proposition_l759_75983

variable (a : ℝ)

def proposition := ∀ x : ℝ, 0 < a^x

theorem neg_of_proposition (h₀ : 0 < a) (h₁ : a ≠ 1) : ¬proposition a ↔ ∃ x : ℝ, a^x ≤ 0 :=
by
  sorry

end neg_of_proposition_l759_75983


namespace length_of_parallel_at_60N_l759_75974

noncomputable def parallel_length (R : ℝ) (lat_deg : ℝ) : ℝ :=
  2 * Real.pi * R * Real.cos (Real.pi * lat_deg / 180)

theorem length_of_parallel_at_60N :
  parallel_length 20 60 = 20 * Real.pi :=
by
  sorry

end length_of_parallel_at_60N_l759_75974


namespace smallest_positive_k_l759_75946

theorem smallest_positive_k (k a n : ℕ) (h_pos : k > 0) (h_cond : 3^3 + 4^3 + 5^3 = 216) (h_eq : k * 216 = a^n) (h_n : n > 1) : k = 1 :=
by {
    sorry
}

end smallest_positive_k_l759_75946


namespace set_of_values_a_l759_75901

theorem set_of_values_a (a : ℝ) : (2 ∉ {x : ℝ | x - a < 0}) ↔ (a ≤ 2) :=
by
  sorry

end set_of_values_a_l759_75901


namespace I_consecutive_integers_I_consecutive_even_integers_II_consecutive_integers_II_consecutive_even_integers_II_consecutive_odd_integers_l759_75922

-- Define the problems
theorem I_consecutive_integers:
  ∃ (x y z : ℕ), 2 * x + y + z = 47 ∧ y = x + 1 ∧ z = x + 2 :=
sorry

theorem I_consecutive_even_integers:
  ¬ ∃ (x y z : ℕ), 2 * x + y + z = 47 ∧ y = x + 2 ∧ z = x + 4 :=
sorry

theorem II_consecutive_integers:
  ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 1 ∧ z = x + 2 ∧ w = x + 3 :=
sorry

theorem II_consecutive_even_integers:
  ¬ ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 2 ∧ z = x + 4 ∧ w = x + 6 :=
sorry

theorem II_consecutive_odd_integers:
  ¬ ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 2 ∧ z = x + 4 ∧ w = x + 6 ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧ w % 2 = 1 :=
sorry

end I_consecutive_integers_I_consecutive_even_integers_II_consecutive_integers_II_consecutive_even_integers_II_consecutive_odd_integers_l759_75922


namespace polynomial_divisible_l759_75960

theorem polynomial_divisible (A B : ℝ) (h : ∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + A * x + B = 0) : A + B = -1 :=
by
  sorry

end polynomial_divisible_l759_75960


namespace find_math_marks_l759_75998

theorem find_math_marks :
  ∀ (english marks physics chemistry biology : ℕ) (average : ℕ),
  average = 78 →
  english = 91 →
  physics = 82 →
  chemistry = 67 →
  biology = 85 →
  (english + marks + physics + chemistry + biology) / 5 = average →
  marks = 65 :=
by
  intros english marks physics chemistry biology average h_average h_english h_physics h_chemistry h_biology h_avg_eq
  sorry

end find_math_marks_l759_75998


namespace min_groups_required_l759_75925

-- Define the conditions
def total_children : ℕ := 30
def max_children_per_group : ℕ := 12
def largest_divisor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ d ≤ max_children_per_group

-- Define the property that we are interested in: the minimum number of groups required
def min_num_groups (total : ℕ) (group_size : ℕ) : ℕ := total / group_size

-- Prove the minimum number of groups is 3 given the conditions
theorem min_groups_required : ∃ d, largest_divisor total_children d ∧ min_num_groups total_children d = 3 :=
sorry

end min_groups_required_l759_75925


namespace find_m_l759_75989

theorem find_m (x y m : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : m * x - y = 3) : m = 6 := 
by
  sorry

end find_m_l759_75989


namespace girls_attending_ball_l759_75926

theorem girls_attending_ball (g b : ℕ) 
    (h1 : g + b = 1500) 
    (h2 : 3 * g / 4 + 2 * b / 3 = 900) : 
    g = 1200 ∧ 3 * 1200 / 4 = 900 := 
by
  sorry

end girls_attending_ball_l759_75926


namespace largest_angle_of_consecutive_odd_int_angles_is_125_l759_75924

-- Definitions for a convex hexagon with six consecutive odd integer interior angles
def is_consecutive_odd_integers (xs : List ℕ) : Prop :=
  ∀ n, 0 ≤ n ∧ n < 5 → xs.get! n + 2 = xs.get! (n + 1)

def hexagon_angles_sum_720 (xs : List ℕ) : Prop :=
  xs.length = 6 ∧ xs.sum = 720

-- Main theorem statement
theorem largest_angle_of_consecutive_odd_int_angles_is_125 (xs : List ℕ) 
(h1 : is_consecutive_odd_integers xs) 
(h2 : hexagon_angles_sum_720 xs) : 
  xs.maximum = 125 := 
sorry

end largest_angle_of_consecutive_odd_int_angles_is_125_l759_75924


namespace xyz_inequality_l759_75950

-- Definitions for the conditions and the statement of the problem
theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h_ineq : x * y * z ≥ x * y + y * z + z * x) : 
  x * y * z ≥ 3 * (x + y + z) :=
by
  sorry

end xyz_inequality_l759_75950


namespace proof_problem_l759_75968

noncomputable def expr (a b : ℚ) : ℚ :=
  ((a / b + b / a + 2) * ((a + b) / (2 * a) - (b / (a + b)))) /
  ((a + 2 * b + b^2 / a) * (a / (a + b) + b / (a - b)))

theorem proof_problem : expr (3/4 : ℚ) (4/3 : ℚ) = -7/24 :=
by
  sorry

end proof_problem_l759_75968


namespace smallest_k_l759_75947

theorem smallest_k (k : ℕ) : 
  (k > 0 ∧ (k*(k+1)*(2*k+1)/6) % 400 = 0) → k = 800 :=
by
  sorry

end smallest_k_l759_75947


namespace arithmetic_seq_value_zero_l759_75969

theorem arithmetic_seq_value_zero (a b c : ℝ) (a_seq : ℕ → ℝ)
    (l m n : ℕ) (h_arith : ∀ k, a_seq (k + 1) - a_seq k = a_seq 1 - a_seq 0)
    (h_l : a_seq l = 1 / a)
    (h_m : a_seq m = 1 / b)
    (h_n : a_seq n = 1 / c) :
    (l - m) * a * b + (m - n) * b * c + (n - l) * c * a = 0 := 
sorry

end arithmetic_seq_value_zero_l759_75969


namespace minimum_x_plus_y_l759_75945

theorem minimum_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
    (h1 : x - y < 1) (h2 : 2 * x - y > 2) (h3 : x < 5) : 
    x + y ≥ 6 :=
sorry

end minimum_x_plus_y_l759_75945


namespace average_weight_of_all_boys_l759_75996

theorem average_weight_of_all_boys 
  (n₁ n₂ : ℕ) (w₁ w₂ : ℝ) 
  (h₁ : n₁ = 20) (h₂ : w₁ = 50.25) 
  (h₃ : n₂ = 8) (h₄ : w₂ = 45.15) :
  (n₁ * w₁ + n₂ * w₂) / (n₁ + n₂) = 48.79 := 
by
  sorry

end average_weight_of_all_boys_l759_75996


namespace functional_equation_solution_l759_75952

theorem functional_equation_solution (f : ℝ → ℝ) (t : ℝ) (h : t ≠ -1) :
  (∀ x y : ℝ, (t + 1) * f (1 + x * y) - f (x + y) = f (x + 1) * f (y + 1)) →
  (∀ x, f x = 0) ∨ (∀ x, f x = t) ∨ (∀ x, f x = (t + 1) * x - (t + 2)) :=
by
  sorry

end functional_equation_solution_l759_75952


namespace people_left_first_hour_l759_75909

theorem people_left_first_hour 
  (X : ℕ)
  (h1 : X ≥ 0)
  (h2 : 94 - X + 18 - 9 = 76) :
  X = 27 := 
sorry

end people_left_first_hour_l759_75909


namespace perpendicular_unit_vector_exists_l759_75991

theorem perpendicular_unit_vector_exists :
  ∃ (m n : ℝ), (2 * m + n = 0) ∧ (m^2 + n^2 = 1) ∧ (m = (Real.sqrt 5) / 5) ∧ (n = -(2 * (Real.sqrt 5)) / 5) :=
by
  sorry

end perpendicular_unit_vector_exists_l759_75991


namespace total_cost_with_discounts_l759_75911

theorem total_cost_with_discounts :
  let red_roses := 2 * 12
  let white_roses := 1 * 12
  let yellow_roses := 2 * 12
  let cost_red := red_roses * 6
  let cost_white := white_roses * 7
  let cost_yellow := yellow_roses * 5
  let total_cost_before_discount := cost_red + cost_white + cost_yellow
  let first_discount := 0.15 * total_cost_before_discount
  let cost_after_first_discount := total_cost_before_discount - first_discount
  let additional_discount := 0.10 * cost_after_first_discount
  let total_cost := cost_after_first_discount - additional_discount
  total_cost = 266.22 := by
  sorry

end total_cost_with_discounts_l759_75911


namespace math_problem_mod_1001_l759_75975

theorem math_problem_mod_1001 :
  (2^6 * 3^10 * 5^12 - 75^4 * (26^2 - 1)^2 + 3^10 - 50^6 + 5^12) % 1001 = 400 := by
  sorry

end math_problem_mod_1001_l759_75975


namespace max_value_k_l759_75963

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 4
  | (n+1) => 3 * seq n - 2

theorem max_value_k (k : ℝ) :
  (∀ n : ℕ, n > 0 → k * (seq n) ≤ 9^n) → k ≤ 9 / 4 :=
sorry

end max_value_k_l759_75963


namespace median_of_100_numbers_l759_75954

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end median_of_100_numbers_l759_75954


namespace final_coordinates_of_A_l759_75976

-- Define the initial points
def A : ℝ × ℝ := (3, -2)
def B : ℝ × ℝ := (5, -5)
def C : ℝ × ℝ := (2, -4)

-- Define the translation operation
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

-- Define the rotation operation (180 degrees around a point (h, k))
def rotate180 (p : ℝ × ℝ) (h k : ℝ) : ℝ × ℝ :=
  (2 * h - p.1, 2 * k - p.2)

-- Translate point A
def A' := translate A 4 3

-- Rotate the translated point A' 180 degrees around the point (4, 0)
def A'' := rotate180 A' 4 0

-- The final coordinates of point A after transformations should be (1, -1)
theorem final_coordinates_of_A : A'' = (1, -1) :=
  sorry

end final_coordinates_of_A_l759_75976


namespace min_benches_l759_75902
-- Import the necessary library

-- Defining the problem in Lean statement
theorem min_benches (N : ℕ) :
  (∀ a c : ℕ, (8 * N = a) ∧ (12 * N = c) ∧ (a = c)) → N = 6 :=
by
  sorry

end min_benches_l759_75902


namespace correct_regression_eq_l759_75967

-- Definitions related to the conditions
def negative_correlation (y x : ℝ) : Prop :=
  -- y is negatively correlated with x implies a negative slope in regression
  ∃ a b : ℝ, a < 0 ∧ ∀ x, y = a * x + b

-- The potential regression equations
def regression_eq1 (x : ℝ) : ℝ := -10 * x + 200
def regression_eq2 (x : ℝ) : ℝ := 10 * x + 200
def regression_eq3 (x : ℝ) : ℝ := -10 * x - 200
def regression_eq4 (x : ℝ) : ℝ := 10 * x - 200

-- Prove that the correct regression equation is selected given the conditions
theorem correct_regression_eq (y x : ℝ) (h : negative_correlation y x) : 
  (∀ x : ℝ, y = regression_eq1 x) ∨ (∀ x : ℝ, y = regression_eq2 x) ∨ 
  (∀ x : ℝ, y = regression_eq3 x) ∨ (∀ x : ℝ, y = regression_eq4 x) →
  ∀ x : ℝ, y = regression_eq1 x := by
  -- This theorem states that given negative correlation and the possible options, 
  -- the correct regression equation consistent with all conditions must be regression_eq1.
  sorry

end correct_regression_eq_l759_75967


namespace problem1_l759_75904

theorem problem1 : 13 + (-24) - (-40) = 29 := by
  sorry

end problem1_l759_75904


namespace inscribed_squares_ratio_l759_75987

theorem inscribed_squares_ratio (x y : ℝ) 
  (h₁ : 5^2 + 12^2 = 13^2)
  (h₂ : x = 144 / 17)
  (h₃ : y = 5) :
  x / y = 144 / 85 :=
by
  sorry

end inscribed_squares_ratio_l759_75987


namespace math_problem_l759_75961

variable {R : Type} [LinearOrderedField R]

theorem math_problem
  (a b : R) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
by
  sorry

end math_problem_l759_75961


namespace max_area_of_rectangle_l759_75914

theorem max_area_of_rectangle (L : ℝ) (hL : L = 16) :
  ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 8 → A = x * (8 - x)) ∧ A = 16 :=
by
  sorry

end max_area_of_rectangle_l759_75914


namespace largest_value_of_b_l759_75995

theorem largest_value_of_b (b : ℚ) (h : (2 * b + 5) * (b - 1) = 6 * b) : b = 5 / 2 :=
by
  sorry

end largest_value_of_b_l759_75995


namespace mariel_dogs_count_l759_75971

theorem mariel_dogs_count
  (num_dogs_other: Nat)
  (num_legs_tangled: Nat)
  (num_legs_per_dog: Nat)
  (num_legs_per_human: Nat)
  (num_dog_walkers: Nat)
  (num_dogs_mariel: Nat):
  num_dogs_other = 3 →
  num_legs_tangled = 36 →
  num_legs_per_dog = 4 →
  num_legs_per_human = 2 →
  num_dog_walkers = 2 →
  4*num_dogs_mariel + 4*num_dogs_other + 2*num_dog_walkers = num_legs_tangled →
  num_dogs_mariel = 5 :=
by 
  intros h_other h_tangled h_legs_dog h_legs_human h_walkers h_eq
  sorry

end mariel_dogs_count_l759_75971
