import Mathlib

namespace xy_diff_l1100_110065

theorem xy_diff {x y : ℝ} (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end xy_diff_l1100_110065


namespace runs_in_last_match_l1100_110081

theorem runs_in_last_match (W : ℕ) (R x : ℝ) 
    (hW : W = 85) 
    (hR : R = 12.4 * W) 
    (new_average : (R + x) / (W + 5) = 12) : 
    x = 26 := 
by 
  sorry

end runs_in_last_match_l1100_110081


namespace minimum_problem_l1100_110091

open BigOperators

theorem minimum_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1 / y) * (x + 1 / y - 2020) + (y + 1 / x) * (y + 1 / x - 2020) ≥ -2040200 := 
sorry

end minimum_problem_l1100_110091


namespace speed_in_still_water_l1100_110054

/-- Conditions -/
def upstream_speed : ℝ := 30
def downstream_speed : ℝ := 40

/-- Theorem: The speed of the man in still water is 35 kmph. -/
theorem speed_in_still_water : 
  (upstream_speed + downstream_speed) / 2 = 35 := 
by 
  sorry

end speed_in_still_water_l1100_110054


namespace domain_of_myFunction_l1100_110042

-- Define the function
def myFunction (x : ℝ) : ℝ := (x + 2) ^ (1 / 2) - (x + 1) ^ 0

-- State the domain constraints as a theorem
theorem domain_of_myFunction (x : ℝ) : 
  (x ≥ -2 ∧ x ≠ -1) →
  ∃ y : ℝ, y = myFunction x := 
sorry

end domain_of_myFunction_l1100_110042


namespace nicholas_bottle_caps_l1100_110026

theorem nicholas_bottle_caps (initial : ℕ) (additional : ℕ) (final : ℕ) (h1 : initial = 8) (h2 : additional = 85) :
  final = 93 :=
by
  sorry

end nicholas_bottle_caps_l1100_110026


namespace blackjack_payment_l1100_110001

def casino_payout (b: ℤ) (r: ℤ): ℤ := b + r
def blackjack_payout (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ): ℤ :=
  (ratio_numerator * bet) / ratio_denominator

theorem blackjack_payment (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ) (payout: ℤ):
  ratio_numerator = 3 → 
  ratio_denominator = 2 → 
  bet = 40 →
  payout = blackjack_payout bet ratio_numerator ratio_denominator → 
  casino_payout bet payout = 100 :=
by
  sorry

end blackjack_payment_l1100_110001


namespace total_cost_train_and_bus_l1100_110013

noncomputable def trainFare := 3.75 + 2.35
noncomputable def busFare := 3.75
noncomputable def totalFare := trainFare + busFare

theorem total_cost_train_and_bus : totalFare = 9.85 :=
by
  -- We'll need a proof here if required.
  sorry

end total_cost_train_and_bus_l1100_110013


namespace find_initial_average_price_l1100_110084

noncomputable def average_initial_price (P : ℚ) : Prop :=
  let total_cost_of_4_cans := 120
  let total_cost_of_returned_cans := 99
  let total_cost_of_6_cans := 6 * P
  total_cost_of_6_cans - total_cost_of_4_cans = total_cost_of_returned_cans

theorem find_initial_average_price (P : ℚ) :
    average_initial_price P → 
    P = 36.5 := sorry

end find_initial_average_price_l1100_110084


namespace find_quotient_from_conditions_l1100_110088

variable (x y : ℕ)
variable (k : ℕ)

theorem find_quotient_from_conditions :
  y - x = 1360 ∧ y = 1614 ∧ y % x = 15 → y / x = 6 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end find_quotient_from_conditions_l1100_110088


namespace total_cost_of_plates_and_cups_l1100_110080

theorem total_cost_of_plates_and_cups (P C : ℝ) 
  (h : 20 * P + 40 * C = 1.50) : 
  100 * P + 200 * C = 7.50 :=
by
  -- proof here
  sorry

end total_cost_of_plates_and_cups_l1100_110080


namespace billy_trays_l1100_110030

def trays_needed (total_ice_cubes : ℕ) (ice_cubes_per_tray : ℕ) : ℕ :=
  total_ice_cubes / ice_cubes_per_tray

theorem billy_trays (total_ice_cubes ice_cubes_per_tray : ℕ) (h1 : total_ice_cubes = 72) (h2 : ice_cubes_per_tray = 9) :
  trays_needed total_ice_cubes ice_cubes_per_tray = 8 :=
by
  sorry

end billy_trays_l1100_110030


namespace monotonically_increasing_intervals_sin_value_l1100_110045

noncomputable def f (x : Real) : Real := 2 * Real.cos (x - Real.pi / 3) * Real.cos x + 1

theorem monotonically_increasing_intervals :
  ∀ (k : Int), ∃ (a b : Real), a = k * Real.pi - Real.pi / 3 ∧ b = k * Real.pi + Real.pi / 6 ∧
                 ∀ (x y : Real), a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y :=
sorry

theorem sin_value 
  (α : Real) (hα : 0 < α ∧ α < Real.pi / 2) 
  (h : f (α + Real.pi / 12) = 7 / 6) : 
  Real.sin (7 * Real.pi / 6 - 2 * α) = 2 * Real.sqrt 2 / 3 :=
sorry

end monotonically_increasing_intervals_sin_value_l1100_110045


namespace problem1_problem2_l1100_110051

-- Problem 1
theorem problem1 (b : ℝ) :
  4 * b^2 * (b^3 - 1) - 3 * (1 - 2 * b^2) > 4 * (b^5 - 1) :=
by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) :
  a - a * abs (-a^2 - 1) < 1 - a^2 * (a - 1) :=
by
  sorry

end problem1_problem2_l1100_110051


namespace find_xyz_l1100_110007

theorem find_xyz (x y z : ℝ) 
  (h1: 3 * x - y + z = 8)
  (h2: x + 3 * y - z = 2) 
  (h3: x - y + 3 * z = 6) :
  x = 1 ∧ y = 3 ∧ z = 8 := by
  sorry

end find_xyz_l1100_110007


namespace cross_product_scaled_v_and_w_l1100_110048

-- Assume the vectors and their scalar multiple
def v : ℝ × ℝ × ℝ := (3, 1, 4)
def w : ℝ × ℝ × ℝ := (-2, 2, -3)
def v_scaled : ℝ × ℝ × ℝ := (6, 2, 8)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.1 * b.2.2 - a.2.2 * b.1,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_scaled_v_and_w :
  cross_product v_scaled w = (-22, -2, 16) :=
by
  sorry

end cross_product_scaled_v_and_w_l1100_110048


namespace line_parabola_intersection_l1100_110052

noncomputable def intersection_range (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + m * x - 1 = 2 * x - 2 * m → -1 ≤ x ∧ x ≤ 3

theorem line_parabola_intersection (m : ℝ) :
  intersection_range m ↔ -3 / 5 < m ∧ m < 5 :=
by
  sorry

end line_parabola_intersection_l1100_110052


namespace train_crossing_time_l1100_110064

noncomputable def time_to_cross_bridge (l_train : ℕ) (v_train_kmh : ℕ) (l_bridge : ℕ) : ℚ :=
  let total_distance := l_train + l_bridge
  let v_train_ms := (v_train_kmh * 1000 : ℚ) / 3600
  total_distance / v_train_ms

theorem train_crossing_time :
  time_to_cross_bridge 110 72 136 = 12.3 := 
by
  sorry

end train_crossing_time_l1100_110064


namespace eval_to_one_l1100_110039

noncomputable def evalExpression (a b c : ℝ) : ℝ :=
  let numerator := (1 / a + 1 / b - 2 * c / (a * b)) * (a + b + 2 * c)
  let denominator := 1 / a^2 + 1 / b^2 + 2 / (a * b) - 4 * c^2 / (a^2 * b^2)
  numerator / denominator

theorem eval_to_one : 
  evalExpression 7.4 (5 / 37) c = 1 := 
by 
  sorry

end eval_to_one_l1100_110039


namespace barbara_wins_iff_multiple_of_6_l1100_110075

-- Define the conditions and the statement to be proved
theorem barbara_wins_iff_multiple_of_6 (n : ℕ) (h : n > 1) :
  (∃ a b : ℕ, a > 0 ∧ b > 1 ∧ (b ∣ a ∨ a ∣ b) ∧ ∀ k ≤ 50, (b + k = n ∨ b - k = n)) ↔ 6 ∣ n :=
sorry

end barbara_wins_iff_multiple_of_6_l1100_110075


namespace inequality_solution_set_l1100_110008

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono_dec : is_monotonically_decreasing_on_nonneg f) :
  { x : ℝ | f 1 - f (1 / x) < 0 } = { x : ℝ | x < -1 ∨ x > 1 } :=
by
  sorry

end inequality_solution_set_l1100_110008


namespace ways_to_select_computers_l1100_110094

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the number of Type A and Type B computers
def num_type_a := 4
def num_type_b := 5

-- Define the total number of computers to select
def total_selected := 3

-- Define the calculation for number of ways to select the computers ensuring both types are included
def ways_to_select := binomial num_type_a 2 * binomial num_type_b 1 + binomial num_type_a 1 * binomial num_type_b 2

-- State the theorem
theorem ways_to_select_computers : ways_to_select = 70 :=
by
  -- Proof will be provided here
  sorry

end ways_to_select_computers_l1100_110094


namespace ratio_of_chocolate_to_regular_milk_l1100_110074

def total_cartons : Nat := 24
def regular_milk_cartons : Nat := 3
def chocolate_milk_cartons : Nat := total_cartons - regular_milk_cartons

theorem ratio_of_chocolate_to_regular_milk (h1 : total_cartons = 24) (h2 : regular_milk_cartons = 3) :
  chocolate_milk_cartons / regular_milk_cartons = 7 :=
by 
  -- Skipping proof with sorry
  sorry

end ratio_of_chocolate_to_regular_milk_l1100_110074


namespace part_a_impossible_part_b_possible_l1100_110092

-- Part (a)
theorem part_a_impossible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ¬ ∀ (x : ℝ), (1 < x ∧ x < a) ∧ (a < 2*x ∧ 2*x < a^2) :=
sorry

-- Part (b)
theorem part_b_possible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ∃ (x : ℝ), (a < 2*x ∧ 2*x < a^2) ∧ ¬ (1 < x ∧ x < a) :=
sorry

end part_a_impossible_part_b_possible_l1100_110092


namespace find_ab_l1100_110019

theorem find_ab (a b : ℝ) 
  (H_period : (1 : ℝ) * (π / b) = π / 2)
  (H_point : a * Real.tan (b * (π / 8)) = 4) :
  a * b = 8 :=
sorry

end find_ab_l1100_110019


namespace isosceles_triangle_perimeter_l1100_110040

-- Define what it means to be a root of the equation x^2 - 5x + 6 = 0
def is_root (x : ℝ) : Prop := x^2 - 5 * x + 6 = 0

-- Define the perimeter based on given conditions
theorem isosceles_triangle_perimeter (x : ℝ) (base : ℝ) (h_base : base = 4) (h_root : is_root x) :
    2 * x + base = 10 :=
by
  -- Insert proof here
  sorry

end isosceles_triangle_perimeter_l1100_110040


namespace boys_laps_eq_27_l1100_110079

noncomputable def miles_per_lap : ℝ := 3 / 4
noncomputable def girls_miles : ℝ := 27
noncomputable def girls_extra_laps : ℝ := 9

theorem boys_laps_eq_27 :
  (∃ boys_laps girls_laps : ℝ, 
    girls_laps = girls_miles / miles_per_lap ∧ 
    boys_laps = girls_laps - girls_extra_laps ∧ 
    boys_laps = 27) :=
by
  sorry

end boys_laps_eq_27_l1100_110079


namespace earnings_correct_l1100_110062

def phonePrice : Nat := 11
def laptopPrice : Nat := 15
def computerPrice : Nat := 18
def tabletPrice : Nat := 12
def smartwatchPrice : Nat := 8

def phoneRepairs : Nat := 9
def laptopRepairs : Nat := 5
def computerRepairs : Nat := 4
def tabletRepairs : Nat := 6
def smartwatchRepairs : Nat := 8

def totalEarnings : Nat := 
  phoneRepairs * phonePrice + 
  laptopRepairs * laptopPrice + 
  computerRepairs * computerPrice + 
  tabletRepairs * tabletPrice + 
  smartwatchRepairs * smartwatchPrice

theorem earnings_correct : totalEarnings = 382 := by
  sorry

end earnings_correct_l1100_110062


namespace zombies_count_decrease_l1100_110032

theorem zombies_count_decrease (z : ℕ) (d : ℕ) : z = 480 → (∀ n, d = 2^n * z) → ∃ t, d / t < 50 :=
by
  intros hz hdz
  let initial_count := 480
  have := 480 / (2 ^ 4)
  sorry

end zombies_count_decrease_l1100_110032


namespace first_digit_base8_of_473_l1100_110085

theorem first_digit_base8_of_473 : 
  ∃ (d : ℕ), (d < 8) ∧ (473 = d * 64 + r ∧ r < 64) ∧ 473 = 7 * 64 + 25 :=
sorry

end first_digit_base8_of_473_l1100_110085


namespace initial_bags_l1100_110000

variable (b : ℕ)

theorem initial_bags (h : 5 * (b - 2) = 45) : b = 11 := 
by 
  sorry

end initial_bags_l1100_110000


namespace no_symmetric_a_l1100_110020

noncomputable def f (a x : ℝ) : ℝ := Real.log (((x + 1) / (x - 1)) * (x - 1) * (a - x))

theorem no_symmetric_a (a : ℝ) (h_a : 1 < a) : ¬ ∃ c : ℝ, ∀ d : ℝ, 1 < c - d ∧ c - d < a ∧ 1 < c + d ∧ c + d < a → f a (c - d) = f a (c + d) :=
sorry

end no_symmetric_a_l1100_110020


namespace quadratic_floor_eq_solutions_count_l1100_110076

theorem quadratic_floor_eq_solutions_count : 
  ∃ s : Finset ℝ, (∀ x : ℝ, x^2 - 4 * ⌊x⌋ + 3 = 0 → x ∈ s) ∧ s.card = 3 :=
by 
  sorry

end quadratic_floor_eq_solutions_count_l1100_110076


namespace total_time_before_playing_game_l1100_110046

theorem total_time_before_playing_game : 
  ∀ (d i t_t t : ℕ), 
  d = 10 → 
  i = d / 2 → 
  t_t = 3 * (d + i) → 
  t = d + i + t_t → 
  t = 60 := 
by
  intros d i t_t t h1 h2 h3 h4
  sorry

end total_time_before_playing_game_l1100_110046


namespace trigonometric_inequality_l1100_110035

open Real

theorem trigonometric_inequality 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < y) 
  (h3 : y < z) 
  (h4 : z < π / 2) : 
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
  sorry

end trigonometric_inequality_l1100_110035


namespace coin_count_l1100_110037

theorem coin_count (x : ℝ) (h₁ : x + 0.50 * x + 0.25 * x = 35) : x = 20 :=
by
  sorry

end coin_count_l1100_110037


namespace total_hangers_l1100_110036

def pink_hangers : ℕ := 7
def green_hangers : ℕ := 4
def blue_hangers : ℕ := green_hangers - 1
def yellow_hangers : ℕ := blue_hangers - 1

theorem total_hangers :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 := by
  sorry

end total_hangers_l1100_110036


namespace chord_length_of_circle_l1100_110057

theorem chord_length_of_circle (x y : ℝ) (h1 : (x - 0)^2 + (y - 2)^2 = 4) (h2 : y = x) : 
  length_of_chord_intercepted_by_line_eq_2sqrt2 :=
sorry

end chord_length_of_circle_l1100_110057


namespace divisible_by_five_l1100_110025

theorem divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
sorry

end divisible_by_five_l1100_110025


namespace find_A_and_area_l1100_110089

open Real

variable (A B C a b c : ℝ)
variable (h1 : 2 * sin A * cos B = 2 * sin C - sin B)
variable (h2 : a = 4 * sqrt 3)
variable (h3 : b + c = 8)
variable (h4 : a^2 = b^2 + c^2 - 2*b*c* cos A)

theorem find_A_and_area :
  A = π / 3 ∧ (1/2 * b * c * sin A = 4 * sqrt 3 / 3) :=
by
  sorry

end find_A_and_area_l1100_110089


namespace equality_am_bn_l1100_110050

theorem equality_am_bn (m n : ℝ) (x : ℝ) (a b : ℝ) (hmn : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0) :
  ((x + m) ^ 2 - (x + n) ^ 2 = (m - n) ^ 2) → (x = am + bn) → (a = 0 ∧ b = -1) :=
by
  intro h1 h2
  sorry

end equality_am_bn_l1100_110050


namespace find_k_for_line_l1100_110060

theorem find_k_for_line : 
  ∃ k : ℚ, (∀ x y : ℚ, (-1 / 3 - 3 * k * x = 4 * y) ∧ (x = 1 / 3) ∧ (y = -8)) → k = 95 / 3 :=
by
  sorry

end find_k_for_line_l1100_110060


namespace cakes_served_for_lunch_l1100_110004

theorem cakes_served_for_lunch (total_cakes: ℕ) (dinner_cakes: ℕ) (lunch_cakes: ℕ) 
  (h1: total_cakes = 15) 
  (h2: dinner_cakes = 9) 
  (h3: total_cakes = lunch_cakes + dinner_cakes) : 
  lunch_cakes = 6 := 
by 
  sorry

end cakes_served_for_lunch_l1100_110004


namespace garden_area_increase_l1100_110072

-- Problem: Prove that changing a 40 ft by 10 ft rectangular garden into a square,
-- using the same fencing, increases the area by 225 sq ft.

theorem garden_area_increase :
  let length_orig := 40
  let width_orig := 10
  let perimeter := 2 * (length_orig + width_orig)
  let side_square := perimeter / 4
  let area_orig := length_orig * width_orig
  let area_square := side_square * side_square
  (area_square - area_orig) = 225 := 
sorry

end garden_area_increase_l1100_110072


namespace geometric_seq_property_l1100_110053

noncomputable def a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

theorem geometric_seq_property (n : ℕ) (h_arith : S (n + 1) + S (n + 1) = 2 * S (n)) (h_condition : a 2 = -2) :
  a 7 = 64 := 
by sorry

end geometric_seq_property_l1100_110053


namespace find_wall_width_l1100_110044

-- Define the volume of one brick
def volume_of_one_brick : ℚ := 100 * 11.25 * 6

-- Define the total number of bricks
def number_of_bricks : ℕ := 1600

-- Define the volume of all bricks combined
def total_volume_of_bricks : ℚ := volume_of_one_brick * number_of_bricks

-- Define dimensions of the wall
def wall_height : ℚ := 800 -- in cm (since 8 meters = 800 cm)
def wall_depth : ℚ := 22.5 -- in cm

-- Theorem to prove the width of the wall
theorem find_wall_width : ∃ width : ℚ, total_volume_of_bricks = wall_height * width * wall_depth ∧ width = 600 :=
by
  -- skipping the actual proof
  sorry

end find_wall_width_l1100_110044


namespace circle_through_points_and_intercepts_l1100_110022

noncomputable def circle_eq (x y D E F : ℝ) : ℝ := x^2 + y^2 + D * x + E * y + F

theorem circle_through_points_and_intercepts :
  ∃ (D E F : ℝ), 
    circle_eq 4 2 D E F = 0 ∧
    circle_eq (-1) 3 D E F = 0 ∧ 
    D + E = -2 ∧
    circle_eq x y (-2) 0 (-12) = 0 :=
by
  unfold circle_eq
  sorry

end circle_through_points_and_intercepts_l1100_110022


namespace three_digit_number_is_657_l1100_110070

theorem three_digit_number_is_657 :
  ∃ (a b c : ℕ), (100 * a + 10 * b + c = 657) ∧ (a + b + c = 18) ∧ (a = b + 1) ∧ (c = b + 2) :=
by
  sorry

end three_digit_number_is_657_l1100_110070


namespace CarriesJellybeanCount_l1100_110027

-- Definitions based on conditions in part a)
def BertBoxJellybeans : ℕ := 150
def BertBoxVolume : ℕ := 6
def CarriesBoxVolume : ℕ := 3 * 2 * 4 * BertBoxVolume -- (3 * height, 2 * width, 4 * length)

-- Theorem statement in Lean based on part c)
theorem CarriesJellybeanCount : (CarriesBoxVolume / BertBoxVolume) * BertBoxJellybeans = 3600 := by 
  sorry

end CarriesJellybeanCount_l1100_110027


namespace change_back_l1100_110098

theorem change_back (price_laptop : ℤ) (price_smartphone : ℤ) (qty_laptops : ℤ) (qty_smartphones : ℤ) (initial_amount : ℤ) (total_cost : ℤ) (change : ℤ) :
  price_laptop = 600 →
  price_smartphone = 400 →
  qty_laptops = 2 →
  qty_smartphones = 4 →
  initial_amount = 3000 →
  total_cost = (price_laptop * qty_laptops) + (price_smartphone * qty_smartphones) →
  change = initial_amount - total_cost →
  change = 200 := by
  sorry

end change_back_l1100_110098


namespace little_john_height_l1100_110068

theorem little_john_height :
  let m := 2 
  let cm_to_m := 8 * 0.01
  let mm_to_m := 3 * 0.001
  m + cm_to_m + mm_to_m = 2.083 := 
by
  sorry

end little_john_height_l1100_110068


namespace candy_pack_cost_l1100_110017

theorem candy_pack_cost (c : ℝ) (h1 : 20 + 78 = 98) (h2 : 2 * c = 98) : c = 49 :=
by {
  sorry
}

end candy_pack_cost_l1100_110017


namespace largest_of_20_consecutive_even_integers_l1100_110099

theorem largest_of_20_consecutive_even_integers (x : ℕ) 
  (h : 20 * (x + 19) = 8000) : (x + 38) = 419 :=
  sorry

end largest_of_20_consecutive_even_integers_l1100_110099


namespace gross_profit_percentage_is_correct_l1100_110038

def selling_price : ℝ := 28
def wholesale_cost : ℝ := 24.56
def gross_profit : ℝ := selling_price - wholesale_cost

-- Define the expected profit percentage as a constant value.
def expected_profit_percentage : ℝ := 14.01

theorem gross_profit_percentage_is_correct :
  ((gross_profit / wholesale_cost) * 100) = expected_profit_percentage :=
by
  -- Placeholder for proof
  sorry

end gross_profit_percentage_is_correct_l1100_110038


namespace eight_S_three_l1100_110097

def custom_operation_S (a b : ℤ) : ℤ := 4 * a + 6 * b + 3

theorem eight_S_three : custom_operation_S 8 3 = 53 := by
  sorry

end eight_S_three_l1100_110097


namespace total_people_after_one_hour_l1100_110049

variable (x y Z : ℕ)

def ferris_wheel_line_initial := 50
def bumper_cars_line_initial := 50
def roller_coaster_line_initial := 50

def ferris_wheel_line_after_half_hour := ferris_wheel_line_initial - x
def bumper_cars_line_after_half_hour := bumper_cars_line_initial + y

axiom Z_eq : Z = ferris_wheel_line_after_half_hour + bumper_cars_line_after_half_hour

theorem total_people_after_one_hour : (Z = (50 - x) + (50 + y)) -> (Z + 100) = ((50 - x) + (50 + y) + 100) :=
by {
  sorry
}

end total_people_after_one_hour_l1100_110049


namespace negation_of_universal_l1100_110006

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 :=
by
  sorry    -- Proof is not required, just the statement.

end negation_of_universal_l1100_110006


namespace negation_of_prop_l1100_110087

theorem negation_of_prop :
  (¬ ∀ (x y : ℝ), x^2 + y^2 ≥ 0) ↔ (∃ (x y : ℝ), x^2 + y^2 < 0) :=
by
  sorry

end negation_of_prop_l1100_110087


namespace remaining_soup_feeds_adults_l1100_110055

theorem remaining_soup_feeds_adults :
  (∀ (cans : ℕ), cans ≥ 8 ∧ cans / 6 ≥ 24) → (∃ (adults : ℕ), adults = 16) :=
by
  sorry

end remaining_soup_feeds_adults_l1100_110055


namespace sin_cos_relation_l1100_110011

theorem sin_cos_relation (α : ℝ) (h : Real.tan (π / 4 + α) = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1 / 2 :=
by
  sorry

end sin_cos_relation_l1100_110011


namespace arithmetic_sequence_num_terms_l1100_110090

theorem arithmetic_sequence_num_terms 
  (a : ℕ) (d : ℕ) (l : ℕ) (n : ℕ)
  (h1 : a = 20)
  (h2 : d = 5)
  (h3 : l = 150)
  (h4 : 150 = 20 + (n-1) * 5) :
  n = 27 :=
by sorry

end arithmetic_sequence_num_terms_l1100_110090


namespace contradiction_example_l1100_110034

theorem contradiction_example (x y : ℝ) (h1 : x + y > 2) (h2 : x ≤ 1) (h3 : y ≤ 1) : False :=
by
  sorry

end contradiction_example_l1100_110034


namespace number_of_bad_cards_l1100_110043

-- Define the initial conditions
def janessa_initial_cards : ℕ := 4
def father_given_cards : ℕ := 13
def ordered_cards : ℕ := 36
def cards_given_to_dexter : ℕ := 29
def cards_kept_for_herself : ℕ := 20

-- Define the total cards and cards in bad shape calculation
theorem number_of_bad_cards : 
  let total_initial_cards := janessa_initial_cards + father_given_cards;
  let total_cards := total_initial_cards + ordered_cards;
  let total_distributed_cards := cards_given_to_dexter + cards_kept_for_herself;
  total_cards - total_distributed_cards = 4 :=
by {
  sorry
}

end number_of_bad_cards_l1100_110043


namespace intersection_of_sets_l1100_110010

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x | x > 2 ∨ x < 1}

theorem intersection_of_sets :
  (A ∪ B) ∩ C = {0, 3, 4} :=
by
  sorry

end intersection_of_sets_l1100_110010


namespace number_of_cooks_l1100_110015

variable (C W : ℕ)

-- Conditions
def initial_ratio := 3 * W = 8 * C
def new_ratio := 4 * C = W + 12

theorem number_of_cooks (h1 : initial_ratio W C) (h2 : new_ratio W C) : C = 9 := by
  sorry

end number_of_cooks_l1100_110015


namespace XAXAXA_divisible_by_seven_l1100_110033

theorem XAXAXA_divisible_by_seven (X A : ℕ) (hX : X < 10) (hA : A < 10) : 
  (101010 * X + 10101 * A) % 7 = 0 := 
by 
  sorry

end XAXAXA_divisible_by_seven_l1100_110033


namespace min_value_xy_l1100_110061

theorem min_value_xy (x y : ℝ) (h1 : x + y = -1) (h2 : x < 0) (h3 : y < 0) :
  ∃ (xy_min : ℝ), (∀ (xy : ℝ), xy = x * y → xy + 1 / xy ≥ xy_min) ∧ xy_min = 17 / 4 :=
by
  sorry

end min_value_xy_l1100_110061


namespace borrow_years_l1100_110083

/-- A person borrows Rs. 5000 at 4% p.a simple interest and lends it at 6% p.a simple interest.
His gain in the transaction per year is Rs. 100. Prove that he borrowed the money for 1 year. --/
theorem borrow_years
  (principal : ℝ)
  (borrow_rate : ℝ)
  (lend_rate : ℝ)
  (gain : ℝ)
  (interest_paid_per_year : ℝ)
  (interest_earned_per_year : ℝ) :
  (principal = 5000) →
  (borrow_rate = 0.04) →
  (lend_rate = 0.06) →
  (gain = 100) →
  (interest_paid_per_year = principal * borrow_rate) →
  (interest_earned_per_year = principal * lend_rate) →
  (interest_earned_per_year - interest_paid_per_year = gain) →
  1 = 1 := 
by
  -- Placeholder for the proof
  sorry

end borrow_years_l1100_110083


namespace vasya_days_without_purchases_l1100_110093

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l1100_110093


namespace problem_statement_l1100_110021

def f (x : ℕ) : ℝ := sorry

theorem problem_statement (h_cond : ∀ k : ℕ, f k ≤ (k : ℝ) ^ 2 → f (k + 1) ≤ (k + 1 : ℝ) ^ 2)
    (h_f7 : f 7 = 50) : ∀ k : ℕ, k ≤ 7 → f k > (k : ℝ) ^ 2 :=
sorry

end problem_statement_l1100_110021


namespace possible_orange_cells_l1100_110014

theorem possible_orange_cells :
  ∃ (n : ℕ), n = 2021 * 2020 ∨ n = 2022 * 2020 := 
sorry

end possible_orange_cells_l1100_110014


namespace rainfall_second_week_l1100_110066

theorem rainfall_second_week (x : ℝ) (h1 : x + 1.5 * x = 20) : 1.5 * x = 12 := 
by {
  sorry
}

end rainfall_second_week_l1100_110066


namespace caleb_trip_duration_l1100_110018

-- Define the times when the clock hands meet
def startTime := 7 * 60 + 38 -- 7:38 a.m. in minutes from midnight
def endTime := 13 * 60 + 5 -- 1:05 p.m. in minutes from midnight

def duration := endTime - startTime

theorem caleb_trip_duration :
  duration = 5 * 60 + 27 := by
sorry

end caleb_trip_duration_l1100_110018


namespace sum_of_squares_ne_sum_of_fourth_powers_l1100_110086

theorem sum_of_squares_ne_sum_of_fourth_powers :
  ∀ (a b : ℤ), a^2 + (a + 1)^2 ≠ b^4 + (b + 1)^4 :=
by 
  sorry

end sum_of_squares_ne_sum_of_fourth_powers_l1100_110086


namespace top_card_is_queen_probability_l1100_110058

theorem top_card_is_queen_probability :
  let num_queens := 4
  let total_cards := 52
  let prob := num_queens / total_cards
  prob = 1 / 13 :=
by 
  sorry

end top_card_is_queen_probability_l1100_110058


namespace cube_side_length_increase_20_percent_l1100_110067

variable {s : ℝ} (initial_side_length_increase : ℝ) (percentage_surface_area_increase : ℝ) (percentage_volume_increase : ℝ)
variable (new_surface_area : ℝ) (new_volume : ℝ)

theorem cube_side_length_increase_20_percent :
  ∀ (s : ℝ),
  (initial_side_length_increase = 1.2 * s) →
  (new_surface_area = 6 * (1.2 * s)^2) →
  (new_volume = (1.2 * s)^3) →
  (percentage_surface_area_increase = ((new_surface_area - (6 * s^2)) / (6 * s^2)) * 100) →
  (percentage_volume_increase = ((new_volume - s^3) / s^3) * 100) →
  5 * (percentage_volume_increase - percentage_surface_area_increase) = 144 := by
  sorry

end cube_side_length_increase_20_percent_l1100_110067


namespace christopher_sword_length_l1100_110031

variable (C J U : ℤ)

def jameson_sword (C : ℤ) : ℤ := 2 * C + 3
def june_sword (J : ℤ) : ℤ := J + 5
def june_sword_christopher (C : ℤ) : ℤ := C + 23

theorem christopher_sword_length (h1 : J = jameson_sword C)
                                (h2 : U = june_sword J)
                                (h3 : U = june_sword_christopher C) :
                                C = 15 :=
by
  sorry

end christopher_sword_length_l1100_110031


namespace eval_sum_l1100_110077

theorem eval_sum : 333 + 33 + 3 = 369 :=
by
  sorry

end eval_sum_l1100_110077


namespace hare_race_l1100_110024

theorem hare_race :
  ∃ (total_jumps: ℕ) (final_jump_leg: String), total_jumps = 548 ∧ final_jump_leg = "right leg" :=
by
  sorry

end hare_race_l1100_110024


namespace distinct_complex_numbers_count_l1100_110005

theorem distinct_complex_numbers_count :
  let real_choices := 10
  let imag_choices := 9
  let distinct_complex_numbers := real_choices * imag_choices
  distinct_complex_numbers = 90 :=
by
  sorry

end distinct_complex_numbers_count_l1100_110005


namespace pow_gt_of_gt_l1100_110012

variable {a x1 x2 : ℝ}

theorem pow_gt_of_gt (ha : a > 1) (hx : x1 > x2) : a^x1 > a^x2 :=
by sorry

end pow_gt_of_gt_l1100_110012


namespace orange_ratio_l1100_110056

variable {R U : ℕ}

theorem orange_ratio (h1 : R + U = 96) 
                    (h2 : (3 / 4 : ℝ) * R + (7 / 8 : ℝ) * U = 78) :
  (R : ℝ) / (R + U : ℝ) = 1 / 2 := 
by
  sorry

end orange_ratio_l1100_110056


namespace convert_units_l1100_110078

theorem convert_units :
  (0.56 * 10 = 5.6 ∧ 0.6 * 10 = 6) ∧
  (2.05 = 2 + 0.05 ∧ 0.05 * 100 = 5) :=
by 
  sorry

end convert_units_l1100_110078


namespace proof_probability_and_expectations_l1100_110041

/-- Number of white balls drawn from two boxes --/
def X : ℕ := 1

/-- Number of red balls drawn from two boxes --/
def Y : ℕ := 1

/-- Given the conditions, the probability of drawing one white ball is 1/2, and
the expected value of white balls drawn is greater than the expected value of red balls drawn --/
theorem proof_probability_and_expectations :
  (∃ (P_X : ℚ), P_X = 1 / 2) ∧ (∃ (E_X E_Y : ℚ), E_X > E_Y) :=
by {
  sorry
}

end proof_probability_and_expectations_l1100_110041


namespace solve_floor_equation_l1100_110095

theorem solve_floor_equation (x : ℚ) 
  (h : ⌊(5 + 6 * x) / 8⌋ = (15 * x - 7) / 5) : 
  x = 7 / 15 ∨ x = 4 / 5 := 
sorry

end solve_floor_equation_l1100_110095


namespace hexagon_circle_ratio_correct_l1100_110069

noncomputable def hexagon_circle_area_ratio (s r : ℝ) (h : 6 * s = 2 * π * r) : ℝ :=
  let A_hex := (3 * Real.sqrt 3 / 2) * s^2
  let A_circ := π * r^2
  (A_hex / A_circ)

theorem hexagon_circle_ratio_correct (s r : ℝ) (h : 6 * s = 2 * π * r) :
    hexagon_circle_area_ratio s r h = (π * Real.sqrt 3 / 6) :=
sorry

end hexagon_circle_ratio_correct_l1100_110069


namespace bobby_has_candy_left_l1100_110082

def initial_candy := 36
def candy_eaten_first := 17
def candy_eaten_second := 15

theorem bobby_has_candy_left : 
  initial_candy - (candy_eaten_first + candy_eaten_second) = 4 := 
by
  sorry


end bobby_has_candy_left_l1100_110082


namespace f_neg_a_l1100_110096

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end f_neg_a_l1100_110096


namespace cube_volume_l1100_110023

theorem cube_volume (a : ℝ) (h : (a - 1) * (a - 1) * (a + 1) = a^3 - 7) : a^3 = 8 :=
  sorry

end cube_volume_l1100_110023


namespace fraction_of_money_left_l1100_110073

theorem fraction_of_money_left (m : ℝ) (b : ℝ) (h1 : (1 / 4) * m = (1 / 2) * b) :
  m - b - 50 = m / 2 - 50 → (m - b - 50) / m = 1 / 2 - 50 / m :=
by sorry

end fraction_of_money_left_l1100_110073


namespace stickers_initial_count_l1100_110003

variable (initial : ℕ) (lost : ℕ)

theorem stickers_initial_count (lost_stickers : lost = 6) (remaining_stickers : initial - lost = 87) : initial = 93 :=
by {
  sorry
}

end stickers_initial_count_l1100_110003


namespace ice_cream_stall_difference_l1100_110009

theorem ice_cream_stall_difference (d : ℕ) 
  (h1 : ∃ d, 10 + (10 + d) + (10 + 2*d) + (10 + 3*d) + (10 + 4*d) = 90) : 
  d = 4 :=
by
  sorry

end ice_cream_stall_difference_l1100_110009


namespace function_maximum_at_1_l1100_110029

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x ^ 2

theorem function_maximum_at_1 :
  ∀ x > 0, (f x ≤ f 1) :=
by
  intro x hx
  have hx_pos : 0 < x := hx
  sorry

end function_maximum_at_1_l1100_110029


namespace golden_section_BC_length_l1100_110016

-- Definition of a golden section point
def is_golden_section_point (A B C : ℝ) : Prop :=
  ∃ (φ : ℝ), φ = (1 + Real.sqrt 5) / 2 ∧ B = φ * C

-- The given problem translated to Lean
theorem golden_section_BC_length (A B C : ℝ) (h1 : is_golden_section_point A B C) (h2 : B - A = 6) : 
  C - B = 3 * Real.sqrt 5 - 3 ∨ C - B = 9 - 3 * Real.sqrt 5 :=
by
  sorry

end golden_section_BC_length_l1100_110016


namespace abs_sum_condition_l1100_110047

theorem abs_sum_condition (a b : ℝ) (h1 : |a| = 7) (h2 : |b| = 3) (h3 : a * b > 0) : a + b = 10 ∨ a + b = -10 :=
by { sorry }

end abs_sum_condition_l1100_110047


namespace square_difference_l1100_110002

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x + 1) * (x - 1) = 9800 :=
by {
  sorry
}

end square_difference_l1100_110002


namespace fg_of_neg2_l1100_110063

def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x + 5

theorem fg_of_neg2 : f (g (-2)) = 1 := by
  sorry

end fg_of_neg2_l1100_110063


namespace gas_usage_l1100_110028

def distance_dermatologist : ℕ := 30
def distance_gynecologist : ℕ := 50
def car_efficiency : ℕ := 20

theorem gas_usage (d_1 d_2 e : ℕ) (H1 : d_1 = distance_dermatologist) (H2 : d_2 = distance_gynecologist) (H3 : e = car_efficiency) :
  (2 * d_1 + 2 * d_2) / e = 8 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end gas_usage_l1100_110028


namespace basketball_team_selection_l1100_110059

theorem basketball_team_selection :
  (Nat.choose 4 2) * (Nat.choose 14 6) = 18018 := 
by
  -- number of ways to choose 2 out of 4 quadruplets
  -- number of ways to choose 6 out of the remaining 14 players
  -- the product of these combinations equals the required number of ways
  sorry

end basketball_team_selection_l1100_110059


namespace alex_correct_percentage_l1100_110071

theorem alex_correct_percentage 
  (score_quiz : ℤ) (problems_quiz : ℤ)
  (score_test : ℤ) (problems_test : ℤ)
  (score_exam : ℤ) (problems_exam : ℤ)
  (h1 : score_quiz = 75) (h2 : problems_quiz = 30)
  (h3 : score_test = 85) (h4 : problems_test = 50)
  (h5 : score_exam = 80) (h6 : problems_exam = 20) :
  (75 * 30 + 85 * 50 + 80 * 20) / (30 + 50 + 20) = 81 := 
sorry

end alex_correct_percentage_l1100_110071
