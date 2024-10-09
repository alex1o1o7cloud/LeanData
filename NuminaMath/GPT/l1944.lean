import Mathlib

namespace phone_sales_total_amount_l1944_194457

theorem phone_sales_total_amount
  (vivienne_phones : ℕ)
  (aliyah_more_phones : ℕ)
  (price_per_phone : ℕ)
  (aliyah_phones : ℕ := vivienne_phones + aliyah_more_phones)
  (total_phones : ℕ := vivienne_phones + aliyah_phones)
  (total_amount : ℕ := total_phones * price_per_phone) :
  vivienne_phones = 40 → aliyah_more_phones = 10 → price_per_phone = 400 → total_amount = 36000 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end phone_sales_total_amount_l1944_194457


namespace james_fraction_of_pizza_slices_l1944_194475

theorem james_fraction_of_pizza_slices :
  (2 * 6 = 12) ∧ (8 / 12 = 2 / 3) :=
by
  sorry

end james_fraction_of_pizza_slices_l1944_194475


namespace sam_total_pennies_l1944_194498

theorem sam_total_pennies : 
  ∀ (initial_pennies found_pennies total_pennies : ℕ),
  initial_pennies = 98 → 
  found_pennies = 93 → 
  total_pennies = initial_pennies + found_pennies → 
  total_pennies = 191 := by
  intros
  sorry

end sam_total_pennies_l1944_194498


namespace brian_total_distance_l1944_194430

noncomputable def miles_per_gallon : ℝ := 20
noncomputable def tank_capacity : ℝ := 15
noncomputable def tank_fraction_remaining : ℝ := 3 / 7

noncomputable def total_miles_traveled (miles_per_gallon tank_capacity tank_fraction_remaining : ℝ) : ℝ :=
  let total_miles := miles_per_gallon * tank_capacity
  let fuel_used := tank_capacity * (1 - tank_fraction_remaining)
  let miles_traveled := fuel_used * miles_per_gallon
  miles_traveled

theorem brian_total_distance : 
  total_miles_traveled miles_per_gallon tank_capacity tank_fraction_remaining = 171.4 := 
by
  sorry

end brian_total_distance_l1944_194430


namespace find_other_number_l1944_194428

theorem find_other_number (x y : ℤ) (h1 : 3 * x + 2 * y = 145) (h2 : x = 35 ∨ y = 35) : y = 20 :=
sorry

end find_other_number_l1944_194428


namespace coefficients_identity_l1944_194412

def coefficients_of_quadratic (a b c : ℤ) (x : ℤ) : Prop :=
  a * x^2 + b * x + c = 0

theorem coefficients_identity : ∀ x : ℤ,
  coefficients_of_quadratic 3 (-4) 1 x :=
by
  sorry

end coefficients_identity_l1944_194412


namespace jake_buys_packages_l1944_194484

theorem jake_buys_packages:
  ∀ (pkg_weight cost_per_pound total_paid : ℕ),
    pkg_weight = 2 →
    cost_per_pound = 4 →
    total_paid = 24 →
    (total_paid / (pkg_weight * cost_per_pound)) = 3 :=
by
  intros pkg_weight cost_per_pound total_paid hw_cp ht
  sorry

end jake_buys_packages_l1944_194484


namespace zhuzhuxia_defeats_monsters_l1944_194406

theorem zhuzhuxia_defeats_monsters {a : ℕ} (H1 : zhuzhuxia_total_defeated_monsters = 20) :
  zhuzhuxia_total_defeated_by_monsters = 8 :=
sorry

end zhuzhuxia_defeats_monsters_l1944_194406


namespace percentage_relationship_l1944_194426

variable {x y z : ℝ}

theorem percentage_relationship (h1 : x = 1.30 * y) (h2 : y = 0.50 * z) : x = 0.65 * z :=
by
  sorry

end percentage_relationship_l1944_194426


namespace total_money_is_correct_l1944_194433

-- Define conditions as constants
def numChocolateCookies : ℕ := 220
def pricePerChocolateCookie : ℕ := 1
def numVanillaCookies : ℕ := 70
def pricePerVanillaCookie : ℕ := 2

-- Total money made from selling chocolate cookies
def moneyFromChocolateCookies : ℕ := numChocolateCookies * pricePerChocolateCookie

-- Total money made from selling vanilla cookies
def moneyFromVanillaCookies : ℕ := numVanillaCookies * pricePerVanillaCookie

-- Total money made from selling all cookies
def totalMoneyMade : ℕ := moneyFromChocolateCookies + moneyFromVanillaCookies

-- The statement to prove, with the expected result
theorem total_money_is_correct : totalMoneyMade = 360 := by
  sorry

end total_money_is_correct_l1944_194433


namespace rationalize_denominator_l1944_194415

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end rationalize_denominator_l1944_194415


namespace episodes_per_monday_l1944_194425

theorem episodes_per_monday (M : ℕ) (h : 67 * (M + 2) = 201) : M = 1 :=
sorry

end episodes_per_monday_l1944_194425


namespace min_days_to_triple_loan_l1944_194418

theorem min_days_to_triple_loan (amount_borrowed : ℕ) (interest_rate : ℝ) :
  ∀ x : ℕ, x ≥ 20 ↔ amount_borrowed + (amount_borrowed * (interest_rate / 10)) * x ≥ 3 * amount_borrowed :=
sorry

end min_days_to_triple_loan_l1944_194418


namespace dan_balloons_correct_l1944_194477

-- Define the initial conditions
def sam_initial_balloons : Float := 46.0
def sam_given_fred_balloons : Float := 10.0
def total_balloons : Float := 52.0

-- Calculate Sam's remaining balloons
def sam_current_balloons : Float := sam_initial_balloons - sam_given_fred_balloons

-- Define the target: Dan's balloons
def dan_balloons := total_balloons - sam_current_balloons

-- Statement to prove
theorem dan_balloons_correct : dan_balloons = 16.0 := sorry

end dan_balloons_correct_l1944_194477


namespace expected_coins_100_rounds_l1944_194404

noncomputable def expectedCoinsAfterGame (rounds : ℕ) (initialCoins : ℕ) : ℝ :=
  initialCoins * (101 / 100) ^ rounds

theorem expected_coins_100_rounds :
  expectedCoinsAfterGame 100 1 = (101 / 100 : ℝ) ^ 100 :=
by
  sorry

end expected_coins_100_rounds_l1944_194404


namespace integer_solutions_l1944_194419

theorem integer_solutions (x y : ℤ) : 2 * (x + y) = x * y + 7 ↔ (x, y) = (3, -1) ∨ (x, y) = (5, 1) ∨ (x, y) = (1, 5) ∨ (x, y) = (-1, 3) := by
  sorry

end integer_solutions_l1944_194419


namespace stars_per_classmate_is_correct_l1944_194416

-- Define the given conditions
def total_stars : ℕ := 45
def num_classmates : ℕ := 9

-- Define the expected number of stars per classmate
def stars_per_classmate : ℕ := 5

-- Prove that the number of stars per classmate is 5 given the conditions
theorem stars_per_classmate_is_correct :
  total_stars / num_classmates = stars_per_classmate :=
sorry

end stars_per_classmate_is_correct_l1944_194416


namespace exponential_graph_passes_through_point_l1944_194471

variable (a : ℝ) (hx1 : a > 0) (hx2 : a ≠ 1)

theorem exponential_graph_passes_through_point :
  ∃ y : ℝ, (y = a^0 + 1) ∧ (y = 2) :=
sorry

end exponential_graph_passes_through_point_l1944_194471


namespace present_age_of_son_l1944_194468

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 29) (h2 : M + 2 = 2 * (S + 2)) : S = 27 :=
sorry

end present_age_of_son_l1944_194468


namespace solve_for_x_l1944_194400

theorem solve_for_x : ∃ x : ℚ, 24 - 4 = 3 * (1 + x) ∧ x = 17 / 3 :=
by
  sorry

end solve_for_x_l1944_194400


namespace find_range_of_a_l1944_194417

variable (x a : ℝ)

/-- Given p: 2 * x^2 - 9 * x + a < 0 and q: the negation of p is sufficient 
condition for the negation of q,
prove to find the range of the real number a. -/
theorem find_range_of_a (hp: 2 * x^2 - 9 * x + a < 0) (hq: ¬ (2 * x^2 - 9 * x + a < 0) → ¬ q) :
  ∃ a : ℝ, sorry := sorry

end find_range_of_a_l1944_194417


namespace coordinates_of_P_l1944_194427

-- Definitions of conditions
def inFourthQuadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def absEqSeven (x : ℝ) : Prop := |x| = 7
def ysquareEqNine (y : ℝ) : Prop := y^2 = 9

-- Main theorem
theorem coordinates_of_P (x y : ℝ) (hx : absEqSeven x) (hy : ysquareEqNine y) (hq : inFourthQuadrant x y) :
  (x, y) = (7, -3) :=
  sorry

end coordinates_of_P_l1944_194427


namespace commute_time_late_l1944_194459

theorem commute_time_late (S : ℝ) (T : ℝ) (T' : ℝ) (H1 : T = 1) (H2 : T' = (4/3)) :
  T' - T = 20 / 60 :=
by
  sorry

end commute_time_late_l1944_194459


namespace solve_system_l1944_194403

theorem solve_system (a b c x y z : ℝ) (h₀ : a = (a * x + c * y) / (b * z + 1))
  (h₁ : b = (b * x + y) / (b * z + 1)) 
  (h₂ : c = (a * z + c) / (b * z + 1)) 
  (h₃ : ¬ a = b * c) :
  x = 1 ∧ y = 0 ∧ z = 0 :=
sorry

end solve_system_l1944_194403


namespace highest_value_of_a_l1944_194470

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def highest_a : Nat :=
  7

theorem highest_value_of_a (a : Nat) 
  (last_three_digits := a * 100 + 53)
  (number := 4 * 10^8 + 3 * 10^7 + 7 * 10^6 + 5 * 10^5 + 2 * 10^4 + a * 10^3 + 5 * 10^2 + 3 * 10^1 + 9) :
  (∃ a, last_three_digits % 8 = 0 ∧ sum_of_digits number % 9 = 0 ∧ number % 12 = 0 ∧ a <= 9) → a = highest_a :=
by
  intros
  sorry

end highest_value_of_a_l1944_194470


namespace penguins_count_l1944_194405

theorem penguins_count (fish_total penguins_fed penguins_require : ℕ) (h1 : fish_total = 68) (h2 : penguins_fed = 19) (h3 : penguins_require = 17) : penguins_fed + penguins_require = 36 :=
by
  sorry

end penguins_count_l1944_194405


namespace miles_left_to_drive_l1944_194449

theorem miles_left_to_drive 
  (total_distance : ℕ) 
  (distance_covered : ℕ) 
  (remaining_distance : ℕ) 
  (h1 : total_distance = 78) 
  (h2 : distance_covered = 32) 
  : remaining_distance = total_distance - distance_covered -> remaining_distance = 46 :=
by
  sorry

end miles_left_to_drive_l1944_194449


namespace expression_evaluation_l1944_194414

theorem expression_evaluation : 
  ( ((2 + 2)^2 / 2^2) * ((3 + 3 + 3 + 3)^3 / (3 + 3 + 3)^3) * ((6 + 6 + 6 + 6 + 6 + 6)^6 / (6 + 6 + 6 + 6)^6) = 108 ) := 
by 
  sorry

end expression_evaluation_l1944_194414


namespace math_problem_l1944_194434

theorem math_problem (a : ℝ) (h : a = 1/3) : (3 * a⁻¹ + 2 / 3 * a⁻¹) / a = 33 := by
  sorry

end math_problem_l1944_194434


namespace graph_passes_through_point_l1944_194439

theorem graph_passes_through_point (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) :
  ∃ x y, (x, y) = (0, 3) ∧ (∀ f : ℝ → ℝ, (∀ y, (f y = a ^ y) → (0, f 0 + 2) = (0, 3))) :=
by
  sorry

end graph_passes_through_point_l1944_194439


namespace more_calories_per_dollar_l1944_194465

-- The conditions given in the problem as definitions
def price_burritos : ℕ := 6
def price_burgers : ℕ := 8
def calories_per_burrito : ℕ := 120
def calories_per_burger : ℕ := 400
def num_burritos : ℕ := 10
def num_burgers : ℕ := 5

-- The theorem stating the mathematically equivalent proof problem
theorem more_calories_per_dollar : 
  (num_burgers * calories_per_burger / price_burgers) - (num_burritos * calories_per_burrito / price_burritos) = 50 :=
by
  sorry

end more_calories_per_dollar_l1944_194465


namespace lulu_cash_left_l1944_194437

theorem lulu_cash_left :
  ∀ (initial money spentIceCream spentTshirt deposited finalCash: ℝ),
    initial = 65 →
    spentIceCream = 5 →
    spentTshirt = 0.5 * (initial - spentIceCream) →
    deposited = (initial - spentIceCream - spentTshirt) / 5 →
    finalCash = initial - spentIceCream - spentTshirt - deposited →
    finalCash = 24 :=
by
  intros
  sorry

end lulu_cash_left_l1944_194437


namespace exists_list_with_all_players_l1944_194402

-- Definitions and assumptions
variable {Player : Type} 

-- Each player plays against every other player exactly once, and there are no ties.
-- Defining defeats relationship
def defeats (p1 p2 : Player) : Prop :=
  sorry -- Assume some ordering or wins relationship

-- Defining the list of defeats
def list_of_defeats (p : Player) : Set Player :=
  { q | defeats p q ∨ (∃ r, defeats p r ∧ defeats r q) }

-- Main theorem to be proven
theorem exists_list_with_all_players (players : Set Player) :
  (∀ p q : Player, p ∈ players → q ∈ players → p ≠ q → (defeats p q ∨ defeats q p)) →
  ∃ p : Player, (list_of_defeats p) = players \ {p} :=
by
  sorry

end exists_list_with_all_players_l1944_194402


namespace tod_driving_time_l1944_194463
noncomputable def total_driving_time (distance_north distance_west speed : ℕ) : ℕ :=
  (distance_north + distance_west) / speed

theorem tod_driving_time :
  total_driving_time 55 95 25 = 6 :=
by
  sorry

end tod_driving_time_l1944_194463


namespace cube_volume_from_surface_area_l1944_194481

theorem cube_volume_from_surface_area (A : ℝ) (h : A = 54) :
  ∃ V : ℝ, V = 27 := by
  sorry

end cube_volume_from_surface_area_l1944_194481


namespace calculation_l1944_194422

theorem calculation : 8 - (7.14 * (1 / 3) - (20 / 9) / (5 / 2)) + 0.1 = 6.62 :=
by
  sorry

end calculation_l1944_194422


namespace ellipse_hyperbola_tangent_l1944_194450

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m) → m = 2 :=
by sorry

end ellipse_hyperbola_tangent_l1944_194450


namespace real_solution_exists_l1944_194494

theorem real_solution_exists (x : ℝ) (h1: x ≠ 5) (h2: x ≠ 6) : 
  (x = 1 ∨ x = 2 ∨ x = 3) ↔ 
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1) /
  ((x - 5) * (x - 6) * (x - 5)) = 1) := 
by 
  sorry

end real_solution_exists_l1944_194494


namespace circles_exceeding_n_squared_l1944_194469

noncomputable def num_circles (n : ℕ) : ℕ :=
  if n >= 8 then 
    5 * n + 4 * (n - 1)
  else 
    n * n

theorem circles_exceeding_n_squared (n : ℕ) (hn : n ≥ 8) : num_circles n > n^2 := 
by {
  sorry
}

end circles_exceeding_n_squared_l1944_194469


namespace trajectory_sufficient_not_necessary_l1944_194489

-- Define for any point P if its trajectory is y = |x|
def trajectory (P : ℝ × ℝ) : Prop :=
  P.2 = abs P.1

-- Define for any point P if its distances to the coordinate axes are equal
def equal_distances (P : ℝ × ℝ) : Prop :=
  abs P.1 = abs P.2

-- The main statement: prove that the trajectory is a sufficient but not necessary condition for equal_distances
theorem trajectory_sufficient_not_necessary (P : ℝ × ℝ) :
  trajectory P → equal_distances P ∧ ¬(equal_distances P → trajectory P) := 
sorry

end trajectory_sufficient_not_necessary_l1944_194489


namespace average_of_last_three_numbers_l1944_194447

theorem average_of_last_three_numbers (A B C D E F : ℕ) 
  (h1 : (A + B + C + D + E + F) / 6 = 30)
  (h2 : (A + B + C + D) / 4 = 25)
  (h3 : D = 25) :
  (D + E + F) / 3 = 35 :=
by
  sorry

end average_of_last_three_numbers_l1944_194447


namespace maximum_value_of_x2y3z_l1944_194454

theorem maximum_value_of_x2y3z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) : 
  x + 2 * y + 3 * z ≤ Real.sqrt 70 :=
by 
  sorry

end maximum_value_of_x2y3z_l1944_194454


namespace Tina_independent_work_hours_l1944_194480

-- Defining conditions as Lean constants
def Tina_work_rate := 1 / 12
def Ann_work_rate := 1 / 9
def Ann_work_hours := 3

-- Declaring the theorem to be proven
theorem Tina_independent_work_hours : 
  (Ann_work_hours * Ann_work_rate = 1/3) →
  ((1 : ℚ) - (Ann_work_hours * Ann_work_rate)) / Tina_work_rate = 8 :=
by {
  sorry
}

end Tina_independent_work_hours_l1944_194480


namespace children_vehicle_wheels_l1944_194408

theorem children_vehicle_wheels:
  ∀ (x : ℕ),
    (6 * 2) + (15 * x) = 57 →
    x = 3 :=
by
  intros x h
  sorry

end children_vehicle_wheels_l1944_194408


namespace eighth_term_sum_of_first_15_terms_l1944_194409

-- Given definitions from the conditions
def a1 : ℚ := 5
def a30 : ℚ := 100
def n8 : ℕ := 8
def n15 : ℕ := 15
def n30 : ℕ := 30

-- Formulate the arithmetic sequence properties
def common_difference : ℚ := (a30 - a1) / (n30 - 1)

def nth_term (n : ℕ) : ℚ :=
  a1 + (n - 1) * common_difference

def sum_of_first_n_terms (n : ℕ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * common_difference)

-- Statements to be proven
theorem eighth_term :
  nth_term n8 = 25 + 1/29 := by sorry

theorem sum_of_first_15_terms :
  sum_of_first_n_terms n15 = 393 + 2/29 := by sorry

end eighth_term_sum_of_first_15_terms_l1944_194409


namespace discount_coupon_value_l1944_194436

theorem discount_coupon_value :
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  total_cost - amount_paid = 4 := by
  intros
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  show total_cost - amount_paid = 4
  sorry

end discount_coupon_value_l1944_194436


namespace simplify_expression_l1944_194413

theorem simplify_expression :
  (1 / 2^2 + (2 / 3^3 * (3 / 2)^2) + 4^(1/2)) - 8 / (4^2 - 3^2) = 107 / 84 :=
by
  -- Skip the proof
  sorry

end simplify_expression_l1944_194413


namespace number_of_shoes_outside_library_l1944_194476

-- Define the conditions
def number_of_people : ℕ := 10
def shoes_per_person : ℕ := 2

-- Define the proof that the number of shoes kept outside the library is 20.
theorem number_of_shoes_outside_library : number_of_people * shoes_per_person = 20 :=
by
  -- Proof left as sorry because the proof steps are not required
  sorry

end number_of_shoes_outside_library_l1944_194476


namespace total_dog_legs_l1944_194473

theorem total_dog_legs (total_animals cats dogs: ℕ) (h1: total_animals = 300) 
  (h2: cats = 2 / 3 * total_animals) 
  (h3: dogs = 1 / 3 * total_animals): (dogs * 4) = 400 :=
by
  sorry

end total_dog_legs_l1944_194473


namespace speed_of_man_proof_l1944_194445

noncomputable def speed_of_man (train_length : ℝ) (crossing_time : ℝ) (train_speed_kph : ℝ) : ℝ :=
  let train_speed_mps := (train_speed_kph * 1000) / 3600
  let relative_speed := train_length / crossing_time
  train_speed_mps - relative_speed

theorem speed_of_man_proof 
  (train_length : ℝ := 600) 
  (crossing_time : ℝ := 35.99712023038157) 
  (train_speed_kph : ℝ := 64) :
  speed_of_man train_length crossing_time train_speed_kph = 1.10977777777778 :=
by
  -- Proof goes here
  sorry

end speed_of_man_proof_l1944_194445


namespace safe_trip_possible_l1944_194442

-- Define the time intervals and eruption cycles
def total_round_trip_time := 16
def trail_time := 8
def crater1_cycle := 18
def crater2_cycle := 10
def crater1_erupt := 1
def crater1_quiet := 17
def crater2_erupt := 1
def crater2_quiet := 9

-- Ivan wants to safely reach the summit and return
theorem safe_trip_possible : ∃ t, 
  -- t is a valid start time where both craters are quiet
  ((t % crater1_cycle) ≥ crater1_erupt ∧ (t % crater2_cycle) ≥ crater2_erupt) ∧
  -- t + total_round_trip_time is also safe for both craters
  (((t + total_round_trip_time) % crater1_cycle) ≥ crater1_erupt ∧ ((t + total_round_trip_time) % crater2_cycle) ≥ crater2_erupt) :=
sorry

end safe_trip_possible_l1944_194442


namespace complex_imaginary_unit_theorem_l1944_194440

def complex_imaginary_unit_equality : Prop :=
  let i := Complex.I
  i * (i + 1) = -1 + i

theorem complex_imaginary_unit_theorem : complex_imaginary_unit_equality :=
by
  sorry

end complex_imaginary_unit_theorem_l1944_194440


namespace number_of_BMWs_sold_l1944_194448

theorem number_of_BMWs_sold (total_cars : ℕ) (ford_percentage nissan_percentage volkswagen_percentage : ℝ) 
    (h1 : total_cars = 300)
    (h2 : ford_percentage = 0.2)
    (h3 : nissan_percentage = 0.25)
    (h4 : volkswagen_percentage = 0.1) :
    ∃ (bmw_percentage : ℝ) (bmw_cars : ℕ), bmw_percentage = 0.45 ∧ bmw_cars = 135 :=
by 
    sorry

end number_of_BMWs_sold_l1944_194448


namespace proof_problem_l1944_194485

-- Definitions coming from the conditions
def num_large_divisions := 12
def num_small_divisions_per_large := 5
def seconds_per_small_division := 1
def seconds_per_large_division := num_small_divisions_per_large * seconds_per_small_division
def start_position := 5
def end_position := 9
def divisions_moved := end_position - start_position
def total_seconds_actual := divisions_moved * seconds_per_large_division
def total_seconds_claimed := 4

-- The theorem stating the false claim
theorem proof_problem : total_seconds_actual ≠ total_seconds_claimed :=
by {
  -- We skip the actual proof as instructed
  sorry
}

end proof_problem_l1944_194485


namespace polynomial_expression_value_l1944_194438

theorem polynomial_expression_value
  (p q r s : ℂ)
  (h1 : p + q + r + s = 0)
  (h2 : p*q + p*r + p*s + q*r + q*s + r*s = -1)
  (h3 : p*q*r + p*q*s + p*r*s + q*r*s = -1)
  (h4 : p*q*r*s = 2) :
  p*(q - r)^2 + q*(r - s)^2 + r*(s - p)^2 + s*(p - q)^2 = -6 :=
by sorry

end polynomial_expression_value_l1944_194438


namespace leftover_yarn_after_square_l1944_194435

theorem leftover_yarn_after_square (total_yarn : ℕ) (side_length : ℕ) (left_yarn : ℕ) :
  total_yarn = 35 →
  (4 * side_length ≤ total_yarn ∧ (∀ s : ℕ, s > side_length → 4 * s > total_yarn)) →
  left_yarn = total_yarn - 4 * side_length →
  left_yarn = 3 :=
by
  sorry

end leftover_yarn_after_square_l1944_194435


namespace quadratic_function_example_l1944_194474

theorem quadratic_function_example : ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x^2 + b * x + c = 0) ↔ (x = 1 ∨ x = 5)) ∧ 
  (a * 3^2 + b * 3 + c = 8) ∧ 
  (a = -2 ∧ b = 12 ∧ c = -10) :=
by
  sorry

end quadratic_function_example_l1944_194474


namespace most_cost_effective_payment_l1944_194410

theorem most_cost_effective_payment :
  let worker_days := 5 * 10
  let hourly_rate_per_worker := 8 * 10 * 4
  let paint_cost := 4800
  let area_painted := 150
  let cost_option_1 := worker_days * 30
  let cost_option_2 := paint_cost * 0.30
  let cost_option_3 := area_painted * 12
  let cost_option_4 := 5 * hourly_rate_per_worker
  (cost_option_2 < cost_option_1) ∧ (cost_option_2 < cost_option_3) ∧ (cost_option_2 < cost_option_4) :=
by
  sorry

end most_cost_effective_payment_l1944_194410


namespace polygon_sides_sum_l1944_194441

theorem polygon_sides_sum
  (area_ABCDEF : ℕ) (AB BC FA DE EF : ℕ)
  (h1 : area_ABCDEF = 78)
  (h2 : AB = 10)
  (h3 : BC = 11)
  (h4 : FA = 7)
  (h5 : DE = 4)
  (h6 : EF = 8) :
  DE + EF = 12 := 
by
  sorry

end polygon_sides_sum_l1944_194441


namespace songs_owned_initially_l1944_194453

theorem songs_owned_initially (a b c : ℕ) (hc : c = a + b) (hb : b = 7) (hc_total : c = 13) :
  a = 6 :=
by
  -- Direct usage of the given conditions to conclude the proof goes here.
  sorry

end songs_owned_initially_l1944_194453


namespace arcsin_neg_one_eq_neg_pi_div_two_l1944_194431

theorem arcsin_neg_one_eq_neg_pi_div_two : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_neg_one_eq_neg_pi_div_two_l1944_194431


namespace x_minus_y_solution_l1944_194492

theorem x_minus_y_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end x_minus_y_solution_l1944_194492


namespace cherry_tree_leaves_l1944_194456

theorem cherry_tree_leaves (original_plan : ℕ) (multiplier : ℕ) (leaves_per_tree : ℕ) 
  (h1 : original_plan = 7) (h2 : multiplier = 2) (h3 : leaves_per_tree = 100) : 
  (original_plan * multiplier * leaves_per_tree = 1400) :=
by
  sorry

end cherry_tree_leaves_l1944_194456


namespace weight_of_2019_is_correct_l1944_194467

-- Declare the conditions as definitions to be used in Lean 4
def stick_weight : Real := 0.5
def digit_to_sticks (n : Nat) : Nat :=
  match n with
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0  -- other digits aren't considered in this problem

-- Calculate the total weight of the number 2019
def weight_of_2019 : Real :=
  (digit_to_sticks 2 + digit_to_sticks 0 + digit_to_sticks 1 + digit_to_sticks 9) * stick_weight

-- Statement to prove the weight of the number 2019
theorem weight_of_2019_is_correct : weight_of_2019 = 9.5 := by
  sorry

end weight_of_2019_is_correct_l1944_194467


namespace largest_integer_n_neg_quad_expr_l1944_194407

theorem largest_integer_n_neg_quad_expr :
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, ((n^2 - 11 * n + 28 < 0) → (m < 7 ∧ m > 4) → m ≤ n) :=
by
  sorry

end largest_integer_n_neg_quad_expr_l1944_194407


namespace equations_have_same_solution_l1944_194488

theorem equations_have_same_solution (x c : ℝ) 
  (h1 : 3 * x + 9 = 0) (h2 : c * x + 15 = 3) : c = 4 :=
by
  sorry

end equations_have_same_solution_l1944_194488


namespace total_papers_delivered_l1944_194420

-- Definitions based on given conditions
def papers_saturday : ℕ := 45
def papers_sunday : ℕ := 65
def total_papers := papers_saturday + papers_sunday

-- The statement we need to prove
theorem total_papers_delivered : total_papers = 110 := by
  -- Proof steps would go here
  sorry

end total_papers_delivered_l1944_194420


namespace arrange_magnitudes_l1944_194490

theorem arrange_magnitudes (x : ℝ) (h1 : 0.85 < x) (h2 : x < 1.1)
  (y : ℝ := x + Real.sin x) (z : ℝ := x ^ (x ^ x)) : x < y ∧ y < z := 
sorry

end arrange_magnitudes_l1944_194490


namespace mark_charged_more_hours_than_kate_l1944_194401

variables (K P M : ℝ)
variables (h1 : K + P + M = 198) (h2 : P = 2 * K) (h3 : M = 3 * P)

theorem mark_charged_more_hours_than_kate : M - K = 110 :=
by
  sorry

end mark_charged_more_hours_than_kate_l1944_194401


namespace triangle_angle_proof_l1944_194496

theorem triangle_angle_proof (α β γ : ℝ) (hα : α > 60) (hβ : β > 60) (hγ : γ > 60) (h_sum : α + β + γ = 180) : false :=
by
  sorry

end triangle_angle_proof_l1944_194496


namespace sin_180_degrees_l1944_194497

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l1944_194497


namespace Robert_salary_loss_l1944_194432

theorem Robert_salary_loss (S : ℝ) (x : ℝ) (h : x ≠ 0) (h1 : (S - (x/100) * S + (x/100) * (S - (x/100) * S) = (96/100) * S)) : x = 20 :=
by sorry

end Robert_salary_loss_l1944_194432


namespace find_divisor_l1944_194462

theorem find_divisor 
    (x : ℕ) 
    (h : 83 = 9 * x + 2) : 
    x = 9 := 
  sorry

end find_divisor_l1944_194462


namespace triangle_area_l1944_194421

theorem triangle_area :
  let A := (2, -3)
  let B := (2, 4)
  let C := (8, 0) 
  let base := (4 - (-3))
  let height := (8 - 2)
  let area := (1 / 2) * base * height
  area = 21 := 
by 
  sorry

end triangle_area_l1944_194421


namespace find_x_in_interval_l1944_194424

theorem find_x_in_interval (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
  abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2 → 
  Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4 :=
by 
  sorry

end find_x_in_interval_l1944_194424


namespace kenya_peanuts_count_l1944_194499

def peanuts_jose : ℕ := 85
def diff_kenya_jose : ℕ := 48
def peanuts_kenya : ℕ := peanuts_jose + diff_kenya_jose

theorem kenya_peanuts_count : peanuts_kenya = 133 := 
by
  -- proof goes here
  sorry

end kenya_peanuts_count_l1944_194499


namespace max_marks_l1944_194411

theorem max_marks (M : ℕ) (h1 : M * 33 / 100 = 175 + 56) : M = 700 :=
by
  sorry

end max_marks_l1944_194411


namespace probability_20_correct_l1944_194491

noncomputable def probability_sum_20_dodecahedral : ℚ :=
  let num_faces := 12
  let total_outcomes := num_faces * num_faces
  let favorable_outcomes := 5
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_20_correct : probability_sum_20_dodecahedral = 5 / 144 := 
by 
  sorry

end probability_20_correct_l1944_194491


namespace profit_distribution_l1944_194452

theorem profit_distribution (x : ℕ) (hx : 2 * x = 4000) :
  let A := 2 * x
  let B := 3 * x
  let C := 5 * x
  A + B + C = 20000 := by
  sorry

end profit_distribution_l1944_194452


namespace xiaohua_final_score_l1944_194482

-- Definitions for conditions
def education_score : ℝ := 9
def experience_score : ℝ := 7
def work_attitude_score : ℝ := 8
def weight_education : ℝ := 1
def weight_experience : ℝ := 2
def weight_attitude : ℝ := 2

-- Computation of the final score
noncomputable def final_score : ℝ :=
  education_score * (weight_education / (weight_education + weight_experience + weight_attitude)) +
  experience_score * (weight_experience / (weight_education + weight_experience + weight_attitude)) +
  work_attitude_score * (weight_attitude / (weight_education + weight_experience + weight_attitude))

-- The statement we want to prove
theorem xiaohua_final_score :
  final_score = 7.8 :=
sorry

end xiaohua_final_score_l1944_194482


namespace no_consecutive_positive_integers_with_no_real_solutions_l1944_194478

theorem no_consecutive_positive_integers_with_no_real_solutions :
  ∀ b c : ℕ, (c = b + 1) → (b^2 - 4 * c < 0) → (c^2 - 4 * b < 0) → false :=
by
  intro b c
  sorry

end no_consecutive_positive_integers_with_no_real_solutions_l1944_194478


namespace probability_is_zero_l1944_194464

noncomputable def probability_same_number (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : ℝ :=
  0

theorem probability_is_zero (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : 
    probability_same_number b d h_b h_d h_b_multiple h_d_multiple h_square = 0 :=
  sorry

end probability_is_zero_l1944_194464


namespace johns_raw_squat_weight_l1944_194460

variable (R : ℝ)

def sleeves_lift := R + 30
def wraps_lift := 1.25 * R
def wraps_more_than_sleeves := wraps_lift R - sleeves_lift R = 120

theorem johns_raw_squat_weight : wraps_more_than_sleeves R → R = 600 :=
by
  intro h
  sorry

end johns_raw_squat_weight_l1944_194460


namespace bicycle_cost_price_l1944_194472

-- Definitions of conditions
def profit_22_5_percent (x : ℝ) : ℝ := 1.225 * x
def loss_14_3_percent (x : ℝ) : ℝ := 0.857 * x
def profit_32_4_percent (x : ℝ) : ℝ := 1.324 * x
def loss_7_8_percent (x : ℝ) : ℝ := 0.922 * x
def discount_5_percent (x : ℝ) : ℝ := 0.95 * x
def tax_6_percent (x : ℝ) : ℝ := 1.06 * x

theorem bicycle_cost_price (CP_A : ℝ) (TP_E : ℝ) (h : TP_E = 295.88) : 
  CP_A = 295.88 / 1.29058890594 :=
by
  sorry

end bicycle_cost_price_l1944_194472


namespace vasya_days_without_purchase_l1944_194495

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l1944_194495


namespace jim_net_paycheck_l1944_194429

-- Let’s state the problem conditions:
def biweekly_gross_pay : ℝ := 1120
def retirement_percentage : ℝ := 0.25
def tax_deduction : ℝ := 100

-- Define the amount deduction for the retirement account
def retirement_deduction (gross : ℝ) (percentage : ℝ) : ℝ := gross * percentage

-- Define the remaining paycheck after all deductions
def net_paycheck (gross : ℝ) (retirement : ℝ) (tax : ℝ) : ℝ :=
  gross - retirement - tax

-- The theorem to prove:
theorem jim_net_paycheck :
  net_paycheck biweekly_gross_pay (retirement_deduction biweekly_gross_pay retirement_percentage) tax_deduction = 740 :=
by
  sorry

end jim_net_paycheck_l1944_194429


namespace number_of_balls_sold_l1944_194423

theorem number_of_balls_sold 
  (selling_price : ℤ) (loss_per_5_balls : ℤ) (cost_price_per_ball : ℤ) (n : ℤ) 
  (h1 : selling_price = 720)
  (h2 : loss_per_5_balls = 5 * cost_price_per_ball)
  (h3 : cost_price_per_ball = 48)
  (h4 : (n * cost_price_per_ball) - selling_price = loss_per_5_balls) :
  n = 20 := 
by
  sorry

end number_of_balls_sold_l1944_194423


namespace total_amount_leaked_l1944_194461

def amount_leaked_before_start : ℕ := 2475
def amount_leaked_while_fixing : ℕ := 3731

theorem total_amount_leaked : amount_leaked_before_start + amount_leaked_while_fixing = 6206 := by
  sorry

end total_amount_leaked_l1944_194461


namespace natalia_crates_l1944_194483

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end natalia_crates_l1944_194483


namespace both_games_players_l1944_194486

theorem both_games_players (kabadi_players kho_kho_only total_players both_games : ℕ)
  (h_kabadi : kabadi_players = 10)
  (h_kho_kho_only : kho_kho_only = 15)
  (h_total : total_players = 25)
  (h_equation : kabadi_players + kho_kho_only + both_games = total_players) :
  both_games = 0 :=
by
  -- question == answer given conditions
  sorry

end both_games_players_l1944_194486


namespace value_of_y_l1944_194493

theorem value_of_y : 
  ∀ y : ℚ, y = (2010^2 - 2010 + 1 : ℚ) / 2010 → y = (2009 + 1 / 2010 : ℚ) := by
  sorry

end value_of_y_l1944_194493


namespace range_of_p_l1944_194444

theorem range_of_p (p : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → x^2 + p * x + 1 > 2 * x + p) → p > -1 := 
by
  sorry

end range_of_p_l1944_194444


namespace appropriate_sampling_methods_l1944_194446

structure Region :=
  (total_households : ℕ)
  (farmer_households : ℕ)
  (worker_households : ℕ)
  (sample_size : ℕ)

theorem appropriate_sampling_methods (r : Region) 
  (h_total: r.total_households = 2004)
  (h_farmers: r.farmer_households = 1600)
  (h_workers: r.worker_households = 303)
  (h_sample: r.sample_size = 40) :
  ("Simple random sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) ∧
  ("Systematic sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) ∧
  ("Stratified sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) :=
by
  sorry

end appropriate_sampling_methods_l1944_194446


namespace solve_for_x_l1944_194451

theorem solve_for_x (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 6)
  (h : (x + 10) / (x - 4) = (x - 3) / (x + 6)) : x = -48 / 23 :=
sorry

end solve_for_x_l1944_194451


namespace no_such_triples_l1944_194466

theorem no_such_triples 
  (a b c : ℕ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : ¬ ∃ k, k ∣ a + c ∧ k ∣ b + c ∧ k ∣ a + b) 
  (h₃ : c^2 ∣ a + b) 
  (h₄ : b^2 ∣ a + c) 
  (h₅ : a^2 ∣ b + c) : 
  false :=
sorry

end no_such_triples_l1944_194466


namespace faster_train_passes_slower_l1944_194479

theorem faster_train_passes_slower (v_fast v_slow : ℝ) (length_fast : ℝ) 
  (hv_fast : v_fast = 50) (hv_slow : v_slow = 32) (hl_length_fast : length_fast = 75) :
  ∃ t : ℝ, t = 15 := 
by
  sorry

end faster_train_passes_slower_l1944_194479


namespace unique_digit_sum_is_21_l1944_194487

theorem unique_digit_sum_is_21
  (Y E M T : ℕ)
  (YE ME : ℕ)
  (HT0 : YE = 10 * Y + E)
  (HT1 : ME = 10 * M + E)
  (H1 : YE * ME = 999)
  (H2 : Y ≠ E)
  (H3 : Y ≠ M)
  (H4 : Y ≠ T)
  (H5 : E ≠ M)
  (H6 : E ≠ T)
  (H7 : M ≠ T)
  (H8 : Y < 10)
  (H9 : E < 10)
  (H10 : M < 10)
  (H11 : T < 10) :
  Y + E + M + T = 21 :=
sorry

end unique_digit_sum_is_21_l1944_194487


namespace sandy_has_32_fish_l1944_194443

-- Define the initial number of pet fish Sandy has
def initial_fish : Nat := 26

-- Define the number of fish Sandy bought
def fish_bought : Nat := 6

-- Define the total number of pet fish Sandy has now
def total_fish : Nat := initial_fish + fish_bought

-- Prove that Sandy now has 32 pet fish
theorem sandy_has_32_fish : total_fish = 32 :=
by
  sorry

end sandy_has_32_fish_l1944_194443


namespace clothes_add_percentage_l1944_194455

theorem clothes_add_percentage (W : ℝ) (C : ℝ) (h1 : W > 0) 
  (h2 : C = 0.0174 * W) : 
  ((C / (0.87 * W)) * 100) = 2 :=
by
  sorry

end clothes_add_percentage_l1944_194455


namespace problem_part1_problem_part2_l1944_194458

variable (a m : ℝ)

def prop_p (a m : ℝ) : Prop := (m - a) * (m - 3 * a) ≤ 0
def prop_q (m : ℝ) : Prop := (m + 2) * (m + 1) < 0

theorem problem_part1 (h₁ : a = -1) (h₂ : prop_p a m ∨ prop_q m) : -3 ≤ m ∧ m ≤ -1 :=
sorry

theorem problem_part2 (h₁ : ∀ m, prop_p a m → ¬prop_q m) :
  -1 / 3 ≤ a ∧ a < 0 ∨ a ≤ -2 :=
sorry

end problem_part1_problem_part2_l1944_194458
