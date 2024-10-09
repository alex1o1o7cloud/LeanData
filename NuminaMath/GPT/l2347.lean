import Mathlib

namespace james_spends_252_per_week_l2347_234711

noncomputable def cost_pistachios_per_ounce := 10 / 5
noncomputable def cost_almonds_per_ounce := 8 / 4
noncomputable def cost_walnuts_per_ounce := 12 / 6

noncomputable def daily_consumption_pistachios := 30 / 5
noncomputable def daily_consumption_almonds := 24 / 4
noncomputable def daily_consumption_walnuts := 18 / 3

noncomputable def weekly_consumption_pistachios := daily_consumption_pistachios * 7
noncomputable def weekly_consumption_almonds := daily_consumption_almonds * 7
noncomputable def weekly_consumption_walnuts := daily_consumption_walnuts * 7

noncomputable def weekly_cost_pistachios := weekly_consumption_pistachios * cost_pistachios_per_ounce
noncomputable def weekly_cost_almonds := weekly_consumption_almonds * cost_almonds_per_ounce
noncomputable def weekly_cost_walnuts := weekly_consumption_walnuts * cost_walnuts_per_ounce

noncomputable def total_weekly_cost := weekly_cost_pistachios + weekly_cost_almonds + weekly_cost_walnuts

theorem james_spends_252_per_week :
  total_weekly_cost = 252 := by
  sorry

end james_spends_252_per_week_l2347_234711


namespace oil_flow_relationship_l2347_234750

theorem oil_flow_relationship (t : ℝ) (Q : ℝ) (initial_quantity : ℝ) (flow_rate : ℝ)
  (h_initial : initial_quantity = 20) (h_flow : flow_rate = 0.2) :
  Q = initial_quantity - flow_rate * t :=
by
  -- proof to be filled in
  sorry

end oil_flow_relationship_l2347_234750


namespace Nick_sister_age_l2347_234779

theorem Nick_sister_age
  (Nick_age : ℕ := 13)
  (Bro_in_5_years : ℕ := 21)
  (H : ∃ S : ℕ, (Nick_age + S) / 2 + 5 = Bro_in_5_years) :
  ∃ S : ℕ, S = 19 :=
by
  sorry

end Nick_sister_age_l2347_234779


namespace fraction_exponentiation_multiplication_l2347_234786

theorem fraction_exponentiation_multiplication :
  (1 / 3) ^ 4 * (1 / 8) = 1 / 648 :=
by
  sorry

end fraction_exponentiation_multiplication_l2347_234786


namespace union_sets_l2347_234778

def A : Set ℝ := { x | (2 / x) > 1 }
def B : Set ℝ := { x | Real.log x < 0 }

theorem union_sets : (A ∪ B) = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end union_sets_l2347_234778


namespace train_crossing_time_l2347_234729

theorem train_crossing_time (length_of_train : ℕ) (speed_kmh : ℕ) (speed_ms : ℕ) 
  (conversion_factor : speed_kmh * 1000 / 3600 = speed_ms) 
  (H1 : length_of_train = 180) 
  (H2 : speed_kmh = 72) 
  (H3 : speed_ms = 20) 
  : length_of_train / speed_ms = 9 := by
  sorry

end train_crossing_time_l2347_234729


namespace circumference_irrational_l2347_234726

theorem circumference_irrational (d : ℚ) : ¬ ∃ (r : ℚ), r = π * d :=
sorry

end circumference_irrational_l2347_234726


namespace son_age_is_eight_l2347_234712

theorem son_age_is_eight (F S : ℕ) (h1 : F + 6 + S + 6 = 68) (h2 : F = 6 * S) : S = 8 :=
by
  sorry

end son_age_is_eight_l2347_234712


namespace algebraic_expression_value_l2347_234700

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 = 3 * y) :
  x^2 - 6 * x * y + 9 * y^2 = 4 :=
sorry

end algebraic_expression_value_l2347_234700


namespace spiderCanEatAllFlies_l2347_234788

-- Define the number of nodes in the grid.
def numNodes := 100

-- Define initial conditions.
def cornerStart := true
def numFlies := 100
def fliesAtNodes (nodes : ℕ) : Prop := nodes = numFlies

-- Define the predicate for whether the spider can eat all flies within a certain number of moves.
def canEatAllFliesWithinMoves (maxMoves : ℕ) : Prop :=
  ∃ (moves : ℕ), moves ≤ maxMoves

-- The theorem we need to prove in Lean 4.
theorem spiderCanEatAllFlies (h1 : cornerStart) (h2 : fliesAtNodes numFlies) : canEatAllFliesWithinMoves 2000 :=
by
  sorry

end spiderCanEatAllFlies_l2347_234788


namespace mary_needs_more_cups_l2347_234787

theorem mary_needs_more_cups (total_cups required_cups added_cups : ℕ) (h1 : required_cups = 8) (h2 : added_cups = 2) : total_cups = 6 :=
by
  sorry

end mary_needs_more_cups_l2347_234787


namespace combined_tax_rate_l2347_234732

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 3 * Mork_income)
  (tax_Mork tax_Mindy : ℝ) (h2 : tax_Mork = 0.10 * Mork_income) (h3 : tax_Mindy = 0.20 * Mindy_income)
  : (tax_Mork + tax_Mindy) / (Mork_income + Mindy_income) = 0.175 :=
by
  sorry

end combined_tax_rate_l2347_234732


namespace range_of_a_decreasing_function_l2347_234797

theorem range_of_a_decreasing_function (a : ℝ) :
  (∀ x < 1, ∀ y < x, (3 * a - 1) * x + 4 * a ≥ (3 * a - 1) * y + 4 * a) ∧ 
  (∀ x ≥ 1, ∀ y > x, -a * x ≤ -a * y) ∧
  (∀ x < 1, ∀ y ≥ 1, (3 * a - 1) * x + 4 * a ≥ -a * y)  →
  (1 / 8 : ℝ) ≤ a ∧ a < (1 / 3 : ℝ) :=
sorry

end range_of_a_decreasing_function_l2347_234797


namespace problem_statement_l2347_234751

-- Definitions of the events as described in the problem conditions.
def event1 (a b : ℝ) : Prop := a * b < 0 → a + b < 0
def event2 (a b : ℝ) : Prop := a * b < 0 → a - b > 0
def event3 (a b : ℝ) : Prop := a * b < 0 → a * b > 0
def event4 (a b : ℝ) : Prop := a * b < 0 → a / b < 0

-- The problem statement combining the conditions and the conclusion.
theorem problem_statement (a b : ℝ) (h1 : a * b < 0):
  (event4 a b) ∧ ¬(event3 a b) ∧ (event1 a b ∨ ¬(event1 a b)) ∧ (event2 a b ∨ ¬(event2 a b)) :=
by
  sorry

end problem_statement_l2347_234751


namespace peak_infection_day_l2347_234757

-- Given conditions
def initial_cases : Nat := 20
def increase_rate : Nat := 50
def decrease_rate : Nat := 30
def total_infections : Nat := 8670
def total_days : Nat := 30

-- Peak Day and infections on that day
def peak_day : Nat := 12

-- Theorem stating what we want to prove
theorem peak_infection_day :
  ∃ n : Nat, n = initial_cases + increase_rate * (peak_day - 1) - decrease_rate * (30 - peak_day) :=
sorry

end peak_infection_day_l2347_234757


namespace train_length_l2347_234760

theorem train_length 
  (t1 t2 : ℕ) 
  (d2 : ℕ) 
  (V L : ℝ) 
  (h1 : t1 = 11)
  (h2 : t2 = 22)
  (h3 : d2 = 120)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) : 
  L = 120 := 
by 
  sorry

end train_length_l2347_234760


namespace coeff_x4_in_expansion_correct_l2347_234782

noncomputable def coeff_x4_in_expansion (f g : ℕ → ℤ) := 
  ∀ (c : ℤ), c = 80 → f 4 + g 1 * g 3 = c

-- Definitions of the individual polynomials
def poly1 (x : ℤ) : ℤ := 4 * x^2 - 2 * x + 1
def poly2 (x : ℤ) : ℤ := 2 * x + 1

-- Expanded form coefficients
def coeff_poly1 : ℕ → ℤ
  | 0       => 1
  | 1       => -2
  | 2       => 4
  | _       => 0

def coeff_poly2_pow4 : ℕ → ℤ
  | 0       => 1
  | 1       => 8
  | 2       => 24
  | 3       => 32
  | 4       => 16
  | _       => 0

-- The theorem we want to prove
theorem coeff_x4_in_expansion_correct :
  coeff_x4_in_expansion coeff_poly1 coeff_poly2_pow4 := 
by
  sorry

end coeff_x4_in_expansion_correct_l2347_234782


namespace find_other_number_l2347_234781

theorem find_other_number (x y : ℕ) (h_gcd : Nat.gcd x y = 22) (h_lcm : Nat.lcm x y = 5940) (h_x : x = 220) :
  y = 594 :=
sorry

end find_other_number_l2347_234781


namespace value_of_a_l2347_234736

theorem value_of_a {a x : ℝ} (h1 : x > 0) (h2 : 2 * x + 1 > a * x) : a ≤ 2 :=
sorry

end value_of_a_l2347_234736


namespace real_values_satisfying_inequality_l2347_234770

theorem real_values_satisfying_inequality :
  { x : ℝ | (x^2 + 2*x^3 - 3*x^4) / (2*x + 3*x^2 - 4*x^3) ≥ -1 } =
  Set.Icc (-1 : ℝ) ((-3 - Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 - Real.sqrt 41) / -8) ((-3 + Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 + Real.sqrt 41) / -8) 0 ∪ 
  Set.Ioi 0 :=
by
  sorry

end real_values_satisfying_inequality_l2347_234770


namespace work_rate_problem_l2347_234727

theorem work_rate_problem :
  ∃ (x : ℝ), 
    (0 < x) ∧ 
    (10 * (1 / x + 1 / 40) = 0.5833333333333334) ∧ 
    (x = 30) :=
by
  sorry

end work_rate_problem_l2347_234727


namespace steve_height_end_second_year_l2347_234752

noncomputable def initial_height_ft : ℝ := 5
noncomputable def initial_height_inch : ℝ := 6
noncomputable def inch_to_cm : ℝ := 2.54

noncomputable def initial_height_cm : ℝ :=
  (initial_height_ft * 12 + initial_height_inch) * inch_to_cm

noncomputable def first_growth_spurt : ℝ := 0.15
noncomputable def second_growth_spurt : ℝ := 0.07
noncomputable def height_decrease : ℝ := 0.04

noncomputable def height_after_growths : ℝ :=
  let height_after_first_growth := initial_height_cm * (1 + first_growth_spurt)
  height_after_first_growth * (1 + second_growth_spurt)

noncomputable def final_height_cm : ℝ :=
  height_after_growths * (1 - height_decrease)

theorem steve_height_end_second_year : final_height_cm = 198.03 :=
  sorry

end steve_height_end_second_year_l2347_234752


namespace smallest_number_of_fruits_l2347_234795

theorem smallest_number_of_fruits 
  (n_apple_slices : ℕ) (n_grapes : ℕ) (n_orange_wedges : ℕ) (n_cherries : ℕ)
  (h_apple : n_apple_slices = 18)
  (h_grape : n_grapes = 9)
  (h_orange : n_orange_wedges = 12)
  (h_cherry : n_cherries = 6)
  : ∃ (n : ℕ), n = 36 ∧ (n % n_apple_slices = 0) ∧ (n % n_grapes = 0) ∧ (n % n_orange_wedges = 0) ∧ (n % n_cherries = 0) :=
sorry

end smallest_number_of_fruits_l2347_234795


namespace time_to_sell_all_cars_l2347_234721

/-- Conditions: -/
def total_cars : ℕ := 500
def number_of_sales_professionals : ℕ := 10
def cars_per_salesperson_per_month : ℕ := 10

/-- Proof Statement: -/
theorem time_to_sell_all_cars 
  (total_cars : ℕ) 
  (number_of_sales_professionals : ℕ) 
  (cars_per_salesperson_per_month : ℕ) : 
  ((number_of_sales_professionals * cars_per_salesperson_per_month) > 0) →
  (total_cars / (number_of_sales_professionals * cars_per_salesperson_per_month)) = 5 :=
by
  sorry

end time_to_sell_all_cars_l2347_234721


namespace shaded_area_of_square_with_circles_l2347_234745

theorem shaded_area_of_square_with_circles :
  let side_length_square := 12
  let radius_quarter_circle := 6
  let radius_center_circle := 3
  let area_square := side_length_square * side_length_square
  let area_quarter_circles := 4 * (1 / 4) * Real.pi * (radius_quarter_circle ^ 2)
  let area_center_circle := Real.pi * (radius_center_circle ^ 2)
  area_square - area_quarter_circles - area_center_circle = 144 - 45 * Real.pi :=
by
  sorry

end shaded_area_of_square_with_circles_l2347_234745


namespace solve_quadratic_l2347_234734

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
sorry

end solve_quadratic_l2347_234734


namespace Dana_Colin_relationship_l2347_234792

variable (C : ℝ) -- Let C be the number of cards Colin has.

def Ben_cards (C : ℝ) : ℝ := 1.20 * C -- Ben has 20% more cards than Colin
def Dana_cards (C : ℝ) : ℝ := 1.40 * Ben_cards C + Ben_cards C -- Dana has 40% more cards than Ben

theorem Dana_Colin_relationship : Dana_cards C = 1.68 * C := by
  sorry

end Dana_Colin_relationship_l2347_234792


namespace min_value_of_f_range_of_a_l2347_234716

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x - 1

theorem min_value_of_f : ∃ x ∈ Set.Ioi 0, ∀ y ∈ Set.Ioi 0, f y ≥ f x ∧ f x = -2 * Real.exp (-1) - 1 := 
  sorry

theorem range_of_a {a : ℝ} : (∀ x > 0, f x ≤ 3 * x^2 + 2 * a * x) ↔ a ∈ Set.Ici (-2) := 
  sorry

end min_value_of_f_range_of_a_l2347_234716


namespace angle_intersecting_lines_l2347_234767

/-- 
Given three lines intersecting at a point forming six equal angles 
around the point, each angle equals 60 degrees.
-/
theorem angle_intersecting_lines (x : ℝ) (h : 6 * x = 360) : x = 60 := by
  sorry

end angle_intersecting_lines_l2347_234767


namespace martian_calendar_months_l2347_234704

theorem martian_calendar_months (x y : ℕ) 
  (h1 : 100 * x + 77 * y = 5882) : x + y = 74 :=
sorry

end martian_calendar_months_l2347_234704


namespace Q_eq_G_l2347_234789

def P := {y | ∃ x, y = x^2 + 1}
def Q := {y : ℝ | ∃ x, y = x^2 + 1}
def E := {x : ℝ | ∃ y, y = x^2 + 1}
def F := {(x, y) | y = x^2 + 1}
def G := {x : ℝ | x ≥ 1}

theorem Q_eq_G : Q = G := by
  sorry

end Q_eq_G_l2347_234789


namespace find_a_from_inclination_l2347_234768

open Real

theorem find_a_from_inclination (a : ℝ) :
  (∃ (k : ℝ), k = (2 - (-3)) / (1 - a) ∧ k = tan (135 * pi / 180)) → a = 6 :=
by
  sorry

end find_a_from_inclination_l2347_234768


namespace max_value_of_linear_combination_l2347_234743

theorem max_value_of_linear_combination (x y : ℝ) (h : x^2 - 3 * x + 4 * y = 7) : 
  3 * x + 4 * y ≤ 16 :=
sorry

end max_value_of_linear_combination_l2347_234743


namespace min_value_l2347_234714

-- Conditions
variables {x y : ℝ}
variable (hx : x > 0)
variable (hy : y > 0)
variable (hxy : x + y = 2)

-- Theorem
theorem min_value (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  ∃ x y, (x > 0) ∧ (y > 0) ∧ (x + y = 2) ∧ (1/x + 4/y = 9/2) := 
by
  sorry

end min_value_l2347_234714


namespace black_car_overtakes_red_car_in_one_hour_l2347_234706

-- Define the speeds of the cars
def red_car_speed := 30 -- in miles per hour
def black_car_speed := 50 -- in miles per hour

-- Define the initial distance between the cars
def initial_distance := 20 -- in miles

-- Calculate the time required for the black car to overtake the red car
theorem black_car_overtakes_red_car_in_one_hour : initial_distance / (black_car_speed - red_car_speed) = 1 := by
  sorry

end black_car_overtakes_red_car_in_one_hour_l2347_234706


namespace sqrt_10_bounds_l2347_234710

theorem sqrt_10_bounds : 10 > 9 ∧ 10 < 16 → 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := 
by 
  sorry

end sqrt_10_bounds_l2347_234710


namespace numDifferentSignals_l2347_234785

-- Number of indicator lights in a row
def numLights : Nat := 6

-- Number of lights that light up each time
def lightsLit : Nat := 3

-- Number of colors each light can show
def numColors : Nat := 3

-- Function to calculate number of different signals
noncomputable def calculateSignals (n m k : Nat) : Nat :=
  -- Number of possible arrangements of "adjacent, adjacent, separate" and "separate, adjacent, adjacent"
  let arrangements := 4 + 4
  -- Number of color combinations for the lit lights
  let colors := k * k * k
  arrangements * colors

-- Theorem stating the total number of different signals is 324
theorem numDifferentSignals : calculateSignals numLights lightsLit numColors = 324 := 
by
  sorry

end numDifferentSignals_l2347_234785


namespace congruent_triangle_sides_l2347_234783

variable {x y : ℕ}

theorem congruent_triangle_sides (h_congruent : ∃ (a b c d e f : ℕ), (a = x) ∧ (b = 2) ∧ (c = 6) ∧ (d = 5) ∧ (e = 6) ∧ (f = y) ∧ (a = d) ∧ (b = f) ∧ (c = e)) : 
  x + y = 7 :=
sorry

end congruent_triangle_sides_l2347_234783


namespace circle_diameter_l2347_234758

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ (d : ℝ), d = 16 :=
by
  sorry

end circle_diameter_l2347_234758


namespace option_a_is_correct_l2347_234740

theorem option_a_is_correct (a b : ℝ) : 
  (a^2 + a * b) / a = a + b := 
by sorry

end option_a_is_correct_l2347_234740


namespace range_of_m_l2347_234793

-- Definitions used to state conditions of the problem.
def fractional_equation (m x : ℝ) : Prop := (m / (2 * x - 1)) + 2 = 0
def positive_solution (x : ℝ) : Prop := x > 0

-- The Lean 4 theorem statement
theorem range_of_m (m x : ℝ) (h : fractional_equation m x) (hx : positive_solution x) : m < 2 ∧ m ≠ 0 :=
by
  sorry

end range_of_m_l2347_234793


namespace sector_properties_l2347_234737

noncomputable def central_angle (l R : ℝ) : ℝ := l / R

noncomputable def area_of_sector (l R : ℝ) : ℝ := (1 / 2) * l * R

theorem sector_properties (R l : ℝ) (hR : R = 8) (hl : l = 12) :
  central_angle l R = 3 / 2 ∧ area_of_sector l R = 48 :=
by
  sorry

end sector_properties_l2347_234737


namespace problem_a_b_n_l2347_234701

theorem problem_a_b_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ 0 → (b - k) ∣ (a - k^n)) : a = b^n := 
sorry

end problem_a_b_n_l2347_234701


namespace repetitions_today_l2347_234719

theorem repetitions_today (yesterday_reps : ℕ) (deficit : ℤ) (today_reps : ℕ) : 
  yesterday_reps = 86 ∧ deficit = -13 → 
  today_reps = yesterday_reps + deficit →
  today_reps = 73 :=
by
  intros
  sorry

end repetitions_today_l2347_234719


namespace find_a_and_solve_inequalities_l2347_234755

-- Definitions as per conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a*x^2 + 5*x - 2 > 0
def inequality2 (a : ℝ) (x : ℝ) : Prop := a*x^2 - 5*x + a^2 - 1 > 0

-- Statement of the theorem
theorem find_a_and_solve_inequalities :
  ∀ (a : ℝ),
    (∀ x, (1/2 < x ∧ x < 2) ↔ inequality1 a x) →
    a = -2 ∧
    (∀ x, (-1/2 < x ∧ x < 3) ↔ inequality2 (-2) x) :=
by
  intros a h
  sorry

end find_a_and_solve_inequalities_l2347_234755


namespace factor_expression_l2347_234715

variable {a : ℝ}

theorem factor_expression :
  ((10 * a^4 - 160 * a^3 - 32) - (-2 * a^4 - 16 * a^3 + 32)) = 4 * (3 * a^3 * (a - 12) - 16) :=
by
  sorry

end factor_expression_l2347_234715


namespace ben_marble_count_l2347_234746

theorem ben_marble_count :
  ∃ k : ℕ, 5 * 2^k > 200 ∧ ∀ m < k, 5 * 2^m ≤ 200 :=
sorry

end ben_marble_count_l2347_234746


namespace find_x_l2347_234749

theorem find_x (x : ℕ) : 
  (∃ (students : ℕ), students = 10) ∧ 
  (∃ (selected : ℕ), selected = 6) ∧ 
  (¬ (∃ (k : ℕ), k = 5 ∧ k = x) ) ∧ 
  (1 ≤ 10 - x) ∧
  (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

end find_x_l2347_234749


namespace total_fireworks_l2347_234707

-- Define the conditions
def fireworks_per_number := 6
def fireworks_per_letter := 5
def numbers_in_year := 4
def letters_in_phrase := 12
def number_of_boxes := 50
def fireworks_per_box := 8

-- Main statement: Prove the total number of fireworks lit during the display
theorem total_fireworks : fireworks_per_number * numbers_in_year + fireworks_per_letter * letters_in_phrase + number_of_boxes * fireworks_per_box = 484 :=
by
  sorry

end total_fireworks_l2347_234707


namespace max_fraction_l2347_234799

theorem max_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) : 
  ∃ k, k = (x + y) / x ∧ k ≤ -2 := 
sorry

end max_fraction_l2347_234799


namespace germination_percentage_l2347_234733

theorem germination_percentage (total_seeds_plot1 total_seeds_plot2 germinated_plot2_percentage total_germinated_percentage germinated_plot1_percentage : ℝ) 
  (plant1 : total_seeds_plot1 = 300) 
  (plant2 : total_seeds_plot2 = 200) 
  (germination2 : germinated_plot2_percentage = 0.35) 
  (total_germination : total_germinated_percentage = 0.23)
  (germinated_plot1 : germinated_plot1_percentage = 0.15) :
  (total_germinated_percentage * (total_seeds_plot1 + total_seeds_plot2) = 
    (germinated_plot2_percentage * total_seeds_plot2) + (germinated_plot1_percentage * total_seeds_plot1)) :=
by
  sorry

end germination_percentage_l2347_234733


namespace wall_bricks_count_l2347_234774

def alice_rate (y : ℕ) : ℕ := y / 8
def bob_rate (y : ℕ) : ℕ := y / 12
def combined_rate (y : ℕ) : ℕ := (5 * y) / 24 - 12
def effective_working_time : ℕ := 6

theorem wall_bricks_count :
  ∃ y : ℕ, (combined_rate y * effective_working_time = y) ∧ y = 288 :=
by
  sorry

end wall_bricks_count_l2347_234774


namespace correct_sampling_methods_l2347_234723

-- Define the surveys with their corresponding conditions
structure Survey1 where
  high_income : Nat
  middle_income : Nat
  low_income : Nat
  total_households : Nat

structure Survey2 where
  total_students : Nat
  sample_students : Nat
  differences_small : Bool
  sizes_small : Bool

-- Define the conditions
def survey1_conditions (s : Survey1) : Prop :=
  s.high_income = 125 ∧ s.middle_income = 280 ∧ s.low_income = 95 ∧ s.total_households = 100

def survey2_conditions (s : Survey2) : Prop :=
  s.total_students = 15 ∧ s.sample_students = 3 ∧ s.differences_small = true ∧ s.sizes_small = true

-- Define the answer predicate
def correct_answer (method1 method2 : String) : Prop :=
  method1 = "stratified sampling" ∧ method2 = "simple random sampling"

-- The theorem statement
theorem correct_sampling_methods (s1 : Survey1) (s2 : Survey2) :
  survey1_conditions s1 → survey2_conditions s2 → correct_answer "stratified sampling" "simple random sampling" :=
by
  -- Proof skipped for problem statement purpose
  sorry

end correct_sampling_methods_l2347_234723


namespace divide_milk_l2347_234735

theorem divide_milk : (3 / 5 : ℚ) = 3 / 5 := by {
    sorry
}

end divide_milk_l2347_234735


namespace true_propositions_count_l2347_234761

theorem true_propositions_count (a : ℝ) :
  ((a > -3 → a > -6) ∧ (a > -6 → ¬(a ≤ -3)) ∧ (a ≤ -3 → ¬(a > -6)) ∧ (a ≤ -6 → a ≤ -3)) → 
  2 = 2 := 
by
  sorry

end true_propositions_count_l2347_234761


namespace log_eval_l2347_234731

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end log_eval_l2347_234731


namespace next_shared_meeting_day_l2347_234709

-- Definitions based on the conditions:
def dramaClubMeetingInterval : ℕ := 3
def choirMeetingInterval : ℕ := 5
def debateTeamMeetingInterval : ℕ := 7

-- Statement to prove:
theorem next_shared_meeting_day : Nat.lcm (Nat.lcm dramaClubMeetingInterval choirMeetingInterval) debateTeamMeetingInterval = 105 := by
  sorry

end next_shared_meeting_day_l2347_234709


namespace trader_sold_meters_l2347_234738

-- Defining the context and conditions
def cost_price_per_meter : ℝ := 100
def profit_per_meter : ℝ := 5
def total_selling_price : ℝ := 8925

-- Calculating the selling price per meter
def selling_price_per_meter : ℝ := cost_price_per_meter + profit_per_meter

-- The problem statement: proving the number of meters sold is 85
theorem trader_sold_meters : (total_selling_price / selling_price_per_meter) = 85 :=
by
  sorry

end trader_sold_meters_l2347_234738


namespace irrigation_tank_final_amount_l2347_234741

theorem irrigation_tank_final_amount : 
  let initial_amount := 300.0
  let evaporation := 1.0
  let addition := 0.3
  let days := 45
  let daily_change := addition - evaporation
  let total_change := daily_change * days
  initial_amount + total_change = 268.5 := 
by {
  -- Proof goes here
  sorry
}

end irrigation_tank_final_amount_l2347_234741


namespace remainder_when_four_times_n_minus_nine_divided_by_7_l2347_234754

theorem remainder_when_four_times_n_minus_nine_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (4 * n - 9) % 7 = 3 := by
  sorry

end remainder_when_four_times_n_minus_nine_divided_by_7_l2347_234754


namespace range_of_abs_function_l2347_234775

theorem range_of_abs_function:
  (∀ y, ∃ x : ℝ, y = |x + 3| - |x - 5|) → ∀ y, y ≤ 8 :=
by
  sorry

end range_of_abs_function_l2347_234775


namespace relay_team_orders_l2347_234766

noncomputable def jordan_relay_orders : Nat :=
  let friends := [1, 2, 3] -- Differentiate friends; let's represent A by 1, B by 2, C by 3
  let choices_for_jordan_third := 2 -- Ways if Jordan runs third
  let choices_for_jordan_fourth := 2 -- Ways if Jordan runs fourth
  choices_for_jordan_third + choices_for_jordan_fourth

theorem relay_team_orders :
  jordan_relay_orders = 4 :=
by
  sorry

end relay_team_orders_l2347_234766


namespace no_solution_if_n_eq_neg_one_l2347_234720

theorem no_solution_if_n_eq_neg_one (n x y z : ℝ) :
  (n * x + y + z = 2) ∧ (x + n * y + z = 2) ∧ (x + y + n * z = 2) ↔ n = -1 → false :=
by
  sorry

end no_solution_if_n_eq_neg_one_l2347_234720


namespace find_annual_interest_rate_l2347_234753

noncomputable def compound_interest (P A : ℝ) (r : ℝ) (n t : ℕ) :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_interest_rate
  (P A : ℝ) (t n : ℕ) (r : ℝ)
  (hP : P = 6000)
  (hA : A = 6615)
  (ht : t = 2)
  (hn : n = 1)
  (hr : compound_interest P A r n t) :
  r = 0.05 :=
sorry

end find_annual_interest_rate_l2347_234753


namespace percentage_of_literate_females_is_32_5_l2347_234772

noncomputable def percentage_literate_females (inhabitants : ℕ) (percent_male : ℝ) (percent_literate_males : ℝ) (percent_literate_total : ℝ) : ℝ :=
  let males := (percent_male / 100) * inhabitants
  let females := inhabitants - males
  let literate_males := (percent_literate_males / 100) * males
  let literate_total := (percent_literate_total / 100) * inhabitants
  let literate_females := literate_total - literate_males
  (literate_females / females) * 100

theorem percentage_of_literate_females_is_32_5 :
  percentage_literate_females 1000 60 20 25 = 32.5 := 
by 
  unfold percentage_literate_females
  sorry

end percentage_of_literate_females_is_32_5_l2347_234772


namespace find_x_plus_y_l2347_234747

theorem find_x_plus_y (x y : ℚ) (h1 : 3 * x - 4 * y = 18) (h2 : x + 3 * y = -1) :
  x + y = 29 / 13 :=
sorry

end find_x_plus_y_l2347_234747


namespace find_constant_l2347_234717

theorem find_constant (n : ℤ) (c : ℝ) (h1 : ∀ n ≤ 10, c * (n : ℝ)^2 ≤ 12100) : c ≤ 121 :=
sorry

end find_constant_l2347_234717


namespace garden_area_l2347_234718

-- Given conditions:
def width := 16
def length (W : ℕ) := 3 * W

-- Proof statement:
theorem garden_area (W : ℕ) (hW : W = width) : length W * W = 768 :=
by
  rw [hW]
  exact rfl

end garden_area_l2347_234718


namespace smallest_a_l2347_234764

theorem smallest_a (a : ℕ) (h1 : a > 0) (h2 : (∀ b : ℕ, b > 0 → b < a → ∀ h3 : b > 0, ¬ (gcd b 72 > 1 ∧ gcd b 90 > 1)))
  (h3 : gcd a 72 > 1) (h4 : gcd a 90 > 1) : a = 2 :=
by
  sorry

end smallest_a_l2347_234764


namespace complex_number_solution_l2347_234771

theorem complex_number_solution {i z : ℂ} (h : (2 : ℂ) / (1 + i) = z + i) : z = 1 + 2 * i :=
sorry

end complex_number_solution_l2347_234771


namespace right_triangle_distance_l2347_234744

theorem right_triangle_distance (x h d : ℝ) :
  x + Real.sqrt ((x + 2 * h) ^ 2 + d ^ 2) = 2 * h + d → 
  x = (h * d) / (2 * h + d) :=
by
  intros h_eq_d
  sorry

end right_triangle_distance_l2347_234744


namespace outstanding_student_awards_l2347_234708

theorem outstanding_student_awards :
  ∃ n : ℕ, 
  (n = Nat.choose 9 7) ∧ 
  (∀ (awards : ℕ) (classes : ℕ), awards = 10 → classes = 8 → n = 36) := 
by
  sorry

end outstanding_student_awards_l2347_234708


namespace conic_curve_focus_eccentricity_l2347_234777

theorem conic_curve_focus_eccentricity (m : ℝ) 
  (h : ∀ x y : ℝ, x^2 + m * y^2 = 1)
  (eccentricity_eq : ∀ a b : ℝ, a > b → m = 4/3) : m = 4/3 :=
by
  sorry

end conic_curve_focus_eccentricity_l2347_234777


namespace no_sol_x_y_pos_int_eq_2015_l2347_234756

theorem no_sol_x_y_pos_int_eq_2015 (x y : ℕ) (hx : x > 0) (hy : y > 0) : ¬ (x^2 - y! = 2015) :=
sorry

end no_sol_x_y_pos_int_eq_2015_l2347_234756


namespace sum_of_ages_is_59_l2347_234724

variable (juliet maggie ralph nicky lucy lily alex : ℕ)

def juliet_age := 10
def maggie_age := juliet_age - 3
def ralph_age := juliet_age + 2
def nicky_age := ralph_age / 2
def lucy_age := ralph_age + 1
def lily_age := ralph_age + 1
def alex_age := lucy_age - 5

theorem sum_of_ages_is_59 :
  maggie_age + ralph_age + nicky_age + lucy_age + lily_age + alex_age = 59 :=
by
  let maggie := 7
  let ralph := 12
  let nicky := 6
  let lucy := 13
  let lily := 13
  let alex := 8
  show maggie + ralph + nicky + lucy + lily + alex = 59
  sorry

end sum_of_ages_is_59_l2347_234724


namespace pairs_of_values_l2347_234769

theorem pairs_of_values (x y : ℂ) :
  (y = (x + 2)^3 ∧ x * y + 2 * y = 2) →
  (∃ (r1 r2 i1 i2 : ℂ), (r1.im = 0 ∧ r2.im = 0) ∧ (i1.im ≠ 0 ∧ i2.im ≠ 0) ∧ 
    ((r1, (r1 + 2)^3) = (x, y) ∨ (r2, (r2 + 2)^3) = (x, y) ∨
     (i1, (i1 + 2)^3) = (x, y) ∨ (i2, (i2 + 2)^3) = (x, y))) :=
sorry

end pairs_of_values_l2347_234769


namespace fraction_of_sum_l2347_234798

theorem fraction_of_sum (numbers : List ℝ) (h_len : numbers.length = 21)
  (n : ℝ) (h_n : n ∈ numbers)
  (h_avg : n = 5 * ((numbers.sum - n) / 20)) :
  n / numbers.sum = 1 / 5 :=
by
  sorry

end fraction_of_sum_l2347_234798


namespace rectangular_prism_sides_multiples_of_5_l2347_234765

noncomputable def rectangular_prism_sides_multiples_product_condition 
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) (prod_eq_450 : l * w = 450) : Prop :=
  l ∣ 450 ∧ w ∣ 450

theorem rectangular_prism_sides_multiples_of_5
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) :
  rectangular_prism_sides_multiples_product_condition l w hl hw (by sorry) :=
sorry

end rectangular_prism_sides_multiples_of_5_l2347_234765


namespace C_investment_l2347_234773

def A_investment_eq : Prop :=
  ∀ (C T : ℝ), (C * T) / 36 = (1 / 6 : ℝ) * C * (1 / 6 : ℝ) * T

def B_investment_eq : Prop :=
  ∀ (C T : ℝ), (C * T) / 9 = (1 / 3 : ℝ) * C * (1 / 3 : ℝ) * T

def C_investment_eq (x : ℝ) : Prop :=
  ∀ (C T : ℝ), x * C * T = (x : ℝ) * C * T

theorem C_investment (x : ℝ) :
  (∀ (C T : ℝ), A_investment_eq) ∧
  (∀ (C T : ℝ), B_investment_eq) ∧
  (∀ (C T : ℝ), C_investment_eq x) ∧
  (∀ (C T : ℝ), 100 / 2300 = (C * T / 36) / ((C * T / 36) + (C * T / 9) + (x * C * T))) →
  x = 1 / 2 :=
by
  intros
  sorry

end C_investment_l2347_234773


namespace probability_not_blue_marble_l2347_234703

-- Define the conditions
def odds_for_blue_marble : ℕ := 5
def odds_for_not_blue_marble : ℕ := 6
def total_outcomes := odds_for_blue_marble + odds_for_not_blue_marble

-- Define the question and statement to be proven
theorem probability_not_blue_marble :
  (odds_for_not_blue_marble : ℚ) / total_outcomes = 6 / 11 :=
by
  -- skipping the proof step as per instruction
  sorry

end probability_not_blue_marble_l2347_234703


namespace valid_numbers_l2347_234742

def is_valid_100_digit_number (N N' : ℕ) (k m n : ℕ) (a : ℕ) : Prop :=
  0 ≤ a ∧ a < 100 ∧ 0 ≤ m ∧ m < 10^k ∧ 
  N = m + 10^k * a + 10^(k + 2) * n ∧ 
  N' = m + 10^k * n ∧
  N = 87 * N'

theorem valid_numbers : ∀ (N : ℕ), (∃ N' k m n a, is_valid_100_digit_number N N' k m n a) →
  N = 435 * 10^97 ∨ 
  N = 1305 * 10^96 ∨ 
  N = 2175 * 10^96 ∨ 
  N = 3045 * 10^96 :=
by
  sorry

end valid_numbers_l2347_234742


namespace hiking_rate_l2347_234702

theorem hiking_rate (rate_uphill: ℝ) (time_total: ℝ) (time_uphill: ℝ) (rate_downhill: ℝ) 
  (h1: rate_uphill = 4) (h2: time_total = 3) (h3: time_uphill = 1.2) : rate_downhill = 4.8 / (time_total - time_uphill) :=
by
  sorry

end hiking_rate_l2347_234702


namespace sufficient_and_necessary_condition_l2347_234705

theorem sufficient_and_necessary_condition (x : ℝ) : 
  2 * x - 4 ≥ 0 ↔ x ≥ 2 :=
sorry

end sufficient_and_necessary_condition_l2347_234705


namespace a_13_eq_30_l2347_234730

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Define arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a_5_eq_6 : a 5 = 6
axiom a_8_eq_15 : a 8 = 15

-- Required proof
theorem a_13_eq_30 (h : arithmetic_sequence a d) : a 13 = 30 :=
  sorry

end a_13_eq_30_l2347_234730


namespace total_ice_cream_sales_l2347_234790

theorem total_ice_cream_sales (tuesday_sales : ℕ) (h1 : tuesday_sales = 12000)
    (wednesday_sales : ℕ) (h2 : wednesday_sales = 2 * tuesday_sales) :
    tuesday_sales + wednesday_sales = 36000 := by
  -- This is the proof statement
  sorry

end total_ice_cream_sales_l2347_234790


namespace find_k_and_general_term_l2347_234759

noncomputable def sum_of_first_n_terms (n k : ℝ) : ℝ :=
  -n^2 + (10 + k) * n + (k - 1)

noncomputable def general_term (n : ℕ) : ℝ :=
  -2 * n + 12

theorem find_k_and_general_term :
  (∀ n k : ℝ, sum_of_first_n_terms n k = sum_of_first_n_terms n (1 : ℝ)) ∧
  (∀ n : ℕ, ∃ an : ℝ, an = general_term n) :=
by
  sorry

end find_k_and_general_term_l2347_234759


namespace Caitlin_correct_age_l2347_234722

def Aunt_Anna_age := 48
def Brianna_age := Aunt_Anna_age / 2
def Caitlin_age := Brianna_age - 7

theorem Caitlin_correct_age : Caitlin_age = 17 := by
  /- Condon: Aunt Anna is 48 years old. -/
  let ha := Aunt_Anna_age
  /- Condon: Brianna is half as old as Aunt Anna. -/
  let hb := Brianna_age
  /- Condon: Caitlin is 7 years younger than Brianna. -/
  let hc := Caitlin_age
  /- Question: How old is Caitlin? Proof: -/
  sorry

end Caitlin_correct_age_l2347_234722


namespace find_a7_of_arithmetic_sequence_l2347_234784

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + d * (n - 1)

theorem find_a7_of_arithmetic_sequence (a d : ℤ)
  (h : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 +
       arithmetic_sequence a d 12 + arithmetic_sequence a d 13 = 24) :
  arithmetic_sequence a d 7 = 6 :=
by
  sorry

end find_a7_of_arithmetic_sequence_l2347_234784


namespace sum_of_extreme_a_l2347_234725

theorem sum_of_extreme_a (a : ℝ) (h : ∀ x, x^2 - a*x - 20*a^2 < 0) (h_diff : |5*a - (-4*a)| ≤ 9) : 
  -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → a_min + a_max = 0 :=
by 
  sorry

end sum_of_extreme_a_l2347_234725


namespace radius_of_circle_l2347_234791

theorem radius_of_circle (A C : ℝ) (h1 : A = π * (r : ℝ)^2) (h2 : C = 2 * π * r) (h3 : A / C = 10) :
  r = 20 :=
by
  sorry

end radius_of_circle_l2347_234791


namespace satisfies_equation_l2347_234763

theorem satisfies_equation : 
  { (x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y } = 
  { (0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2), (-6, 2) } :=
by
  sorry

end satisfies_equation_l2347_234763


namespace brads_running_speed_proof_l2347_234739

noncomputable def brads_speed (distance_between_homes : ℕ) (maxwells_speed : ℕ) (maxwells_time : ℕ) (brad_start_delay : ℕ) : ℕ :=
  let distance_covered_by_maxwell := maxwells_speed * maxwells_time
  let distance_covered_by_brad := distance_between_homes - distance_covered_by_maxwell
  let brads_time := maxwells_time - brad_start_delay
  distance_covered_by_brad / brads_time

theorem brads_running_speed_proof :
  brads_speed 54 4 6 1 = 6 := 
by
  unfold brads_speed
  rfl

end brads_running_speed_proof_l2347_234739


namespace population_multiple_of_seven_l2347_234796

theorem population_multiple_of_seven 
  (a b c : ℕ) 
  (h1 : a^2 + 100 = b^2 + 1) 
  (h2 : b^2 + 1 + 100 = c^2) : 
  (∃ k : ℕ, a = 7 * k) :=
sorry

end population_multiple_of_seven_l2347_234796


namespace group_product_number_l2347_234776

theorem group_product_number (a : ℕ) (group_size : ℕ) (interval : ℕ) (fifth_group_product : ℕ) :
  fifth_group_product = a + 4 * interval → fifth_group_product = 94 → group_size = 5 → interval = 20 →
  (a + (1 - 1) * interval + 1 * interval) = 34 :=
by
  intros fifth_group_eq fifth_group_is_94 group_size_is_5 interval_is_20
  -- Missing steps are handled by sorry
  sorry

end group_product_number_l2347_234776


namespace river_ratio_l2347_234728

theorem river_ratio (total_length straight_length crooked_length : ℕ) 
  (h1 : total_length = 80) (h2 : straight_length = 20) 
  (h3 : crooked_length = total_length - straight_length) : 
  (straight_length / Nat.gcd straight_length crooked_length) = 1 ∧ (crooked_length / Nat.gcd straight_length crooked_length) = 3 := 
by
  sorry

end river_ratio_l2347_234728


namespace number_of_people_for_cheaper_second_caterer_l2347_234762

theorem number_of_people_for_cheaper_second_caterer : 
  ∃ (x : ℕ), (150 + 20 * x > 250 + 15 * x + 50) ∧ 
  ∀ (y : ℕ), (y < x → ¬ (150 + 20 * y > 250 + 15 * y + 50)) :=
by
  sorry

end number_of_people_for_cheaper_second_caterer_l2347_234762


namespace cube_root_of_64_is_4_l2347_234713

theorem cube_root_of_64_is_4 (x : ℝ) (h1 : 0 < x) (h2 : x^3 = 64) : x = 4 :=
by
  sorry

end cube_root_of_64_is_4_l2347_234713


namespace shirts_per_minute_l2347_234794

theorem shirts_per_minute (S : ℕ) 
  (h1 : 12 * S + 14 = 156) : S = 11 := 
by
  sorry

end shirts_per_minute_l2347_234794


namespace ten_percent_of_x_l2347_234780

theorem ten_percent_of_x
  (x : ℝ)
  (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 17.85 :=
by
  -- theorem proof goes here
  sorry

end ten_percent_of_x_l2347_234780


namespace units_digit_of_result_is_7_l2347_234748

theorem units_digit_of_result_is_7 (a b c : ℕ) (h : a = c + 3) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  (original - reversed) % 10 = 7 :=
by
  sorry

end units_digit_of_result_is_7_l2347_234748
