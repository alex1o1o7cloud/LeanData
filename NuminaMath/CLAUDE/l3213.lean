import Mathlib

namespace absolute_value_and_roots_l3213_321330

theorem absolute_value_and_roots : |-3| + (Real.sqrt 2 - 1)^0 - (Real.sqrt 3)^2 = 1 := by
  sorry

end absolute_value_and_roots_l3213_321330


namespace long_tennis_players_l3213_321391

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 35 →
  football = 26 →
  both = 17 →
  neither = 6 →
  ∃ long_tennis : ℕ, long_tennis = 20 ∧ 
    long_tennis = total - (football - both) - neither :=
by sorry

end long_tennis_players_l3213_321391


namespace prime_sum_divisibility_l3213_321365

theorem prime_sum_divisibility (p q : ℕ) : 
  Prime p → Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
  sorry

end prime_sum_divisibility_l3213_321365


namespace total_balls_is_six_l3213_321384

/-- The number of balls in each box -/
def balls_per_box : ℕ := 3

/-- The number of boxes -/
def num_boxes : ℕ := 2

/-- The total number of balls -/
def total_balls : ℕ := balls_per_box * num_boxes

theorem total_balls_is_six : total_balls = 6 := by
  sorry

end total_balls_is_six_l3213_321384


namespace temperature_difference_l3213_321349

theorem temperature_difference (highest lowest : Int) 
  (h1 : highest = 2) 
  (h2 : lowest = -8) : 
  highest - lowest = 10 := by
  sorry

end temperature_difference_l3213_321349


namespace same_color_ratio_property_l3213_321346

/-- A coloring of natural numbers using 2017 colors -/
def Coloring := ℕ → Fin 2017

/-- The theorem stating that for any coloring of natural numbers using 2017 colors,
    there exist two natural numbers of the same color with a specific ratio property -/
theorem same_color_ratio_property (c : Coloring) :
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ c a = c b ∧ 
  ∃ (k : ℕ), k ≠ 0 ∧ b = k * a ∧ 2016 ∣ k := by
  sorry

end same_color_ratio_property_l3213_321346


namespace divisibility_problem_l3213_321336

theorem divisibility_problem (a b : ℕ) 
  (h1 : b ∣ (5 * a - 1))
  (h2 : b ∣ (a - 10))
  (h3 : ¬(b ∣ (3 * a + 5))) :
  b = 49 := by
sorry

end divisibility_problem_l3213_321336


namespace cookies_per_box_type3_is_16_l3213_321311

/-- The number of cookies in each box of the third type -/
def cookies_per_box_type3 (
  cookies_per_box_type1 : ℕ)
  (cookies_per_box_type2 : ℕ)
  (boxes_sold_type1 : ℕ)
  (boxes_sold_type2 : ℕ)
  (boxes_sold_type3 : ℕ)
  (total_cookies_sold : ℕ) : ℕ :=
  (total_cookies_sold - (cookies_per_box_type1 * boxes_sold_type1 + cookies_per_box_type2 * boxes_sold_type2)) / boxes_sold_type3

theorem cookies_per_box_type3_is_16 :
  cookies_per_box_type3 12 20 50 80 70 3320 = 16 := by
  sorry

end cookies_per_box_type3_is_16_l3213_321311


namespace recipe_total_cups_l3213_321386

theorem recipe_total_cups (butter baking_soda flour sugar : ℚ)
  (ratio : butter = 1 ∧ baking_soda = 2 ∧ flour = 5 ∧ sugar = 3)
  (flour_cups : flour * 3 = 15) :
  butter * 3 + baking_soda * 3 + 15 + sugar * 3 = 33 := by
sorry

end recipe_total_cups_l3213_321386


namespace moles_of_H2O_formed_l3213_321395

-- Define the chemical reaction
structure ChemicalReaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String
  ratio : ℕ → ℕ → ℕ

-- Define the problem setup
def reaction : ChemicalReaction := {
  reactant1 := "NaHCO3"
  reactant2 := "HC2H3O2"
  product1 := "NaC2H3O2"
  product2 := "CO2"
  product3 := "H2O"
  ratio := λ x y => min x y
}

def initial_moles_NaHCO3 : ℕ := 3
def initial_moles_HC2H3O2 : ℕ := 3

-- State the theorem
theorem moles_of_H2O_formed (r : ChemicalReaction) 
  (h1 : r = reaction) 
  (h2 : initial_moles_NaHCO3 = 3) 
  (h3 : initial_moles_HC2H3O2 = 3) : 
  r.ratio initial_moles_NaHCO3 initial_moles_HC2H3O2 = 3 := by
  sorry

end moles_of_H2O_formed_l3213_321395


namespace water_sales_profit_profit_for_240_barrels_barrels_for_760_profit_l3213_321369

/-- Represents the daily sales and profit of a water sales department -/
structure WaterSales where
  fixed_costs : ℕ := 200
  cost_price : ℕ := 5
  selling_price : ℕ := 8

/-- Calculates the daily profit based on the number of barrels sold -/
def daily_profit (ws : WaterSales) (x : ℕ) : ℤ :=
  (ws.selling_price * x : ℤ) - (ws.cost_price * x : ℤ) - ws.fixed_costs

theorem water_sales_profit (ws : WaterSales) :
  ∀ x : ℕ, daily_profit ws x = 3 * x - 200 := by sorry

theorem profit_for_240_barrels (ws : WaterSales) :
  daily_profit ws 240 = 520 := by sorry

theorem barrels_for_760_profit (ws : WaterSales) :
  ∃ x : ℕ, daily_profit ws x = 760 ∧ x = 320 := by sorry

end water_sales_profit_profit_for_240_barrels_barrels_for_760_profit_l3213_321369


namespace unique_positive_number_l3213_321307

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x^2 + x = 210 :=
by
  -- Proof goes here
  sorry

end unique_positive_number_l3213_321307


namespace milk_fraction_after_pouring_l3213_321359

/-- Represents a cup containing a mixture of tea and milk -/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- The pouring process described in the problem -/
def pour_process (initial_tea_cup : Cup) (initial_milk_cup : Cup) : Cup :=
  let first_pour := Cup.mk (initial_tea_cup.tea - 2) initial_milk_cup.milk
  let second_cup_total := initial_milk_cup.tea + initial_milk_cup.milk + 2
  let milk_ratio := initial_milk_cup.milk / second_cup_total
  let tea_ratio := (initial_milk_cup.tea + 2) / second_cup_total
  Cup.mk (first_pour.tea + 2 * tea_ratio) (first_pour.milk + 2 * milk_ratio)

theorem milk_fraction_after_pouring 
  (initial_tea_cup : Cup) 
  (initial_milk_cup : Cup) 
  (h1 : initial_tea_cup.tea = 6) 
  (h2 : initial_tea_cup.milk = 0) 
  (h3 : initial_milk_cup.tea = 0) 
  (h4 : initial_milk_cup.milk = 6) :
  let final_cup := pour_process initial_tea_cup initial_milk_cup
  (final_cup.milk / (final_cup.tea + final_cup.milk)) = 1/4 := by
  sorry

end milk_fraction_after_pouring_l3213_321359


namespace legs_walking_theorem_l3213_321301

/-- The number of legs walking on the ground given the conditions of the problem -/
def legs_walking_on_ground (num_horses : ℕ) : ℕ :=
  let num_men := num_horses
  let num_riding := num_horses / 2
  let num_walking_men := num_men - num_riding
  let men_legs := num_walking_men * 2
  let horse_legs := num_horses * 4
  let walking_horse_legs := horse_legs / 2
  men_legs + walking_horse_legs

/-- Theorem stating that given 14 horses, the number of legs walking on the ground is 42 -/
theorem legs_walking_theorem : legs_walking_on_ground 14 = 42 := by
  sorry

end legs_walking_theorem_l3213_321301


namespace tissue_packs_per_box_l3213_321379

/-- Proves that the number of packs in each box is 20 given the specified conditions -/
theorem tissue_packs_per_box :
  ∀ (total_boxes : ℕ) 
    (tissues_per_pack : ℕ) 
    (cost_per_tissue : ℚ) 
    (total_cost : ℚ),
  total_boxes = 10 →
  tissues_per_pack = 100 →
  cost_per_tissue = 5 / 100 →
  total_cost = 1000 →
  (total_cost / total_boxes) / (tissues_per_pack * cost_per_tissue) = 20 := by
sorry

end tissue_packs_per_box_l3213_321379


namespace intersection_of_sets_l3213_321339

theorem intersection_of_sets : 
  let M : Set Int := {-1, 0}
  let N : Set Int := {0, 1}
  M ∩ N = {0} := by sorry

end intersection_of_sets_l3213_321339


namespace derivative_of_sin_cubed_inverse_l3213_321348

noncomputable def f (x : ℝ) : ℝ := Real.sin (1 / x) ^ 3

theorem derivative_of_sin_cubed_inverse (x : ℝ) (hx : x ≠ 0) :
  deriv f x = -3 / x^2 * Real.sin (1 / x)^2 * Real.cos (1 / x) := by
  sorry

end derivative_of_sin_cubed_inverse_l3213_321348


namespace problem_1_problem_2_l3213_321390

-- Part 1
theorem problem_1 : 2023^2 - 2022 * 2024 = 1 := by sorry

-- Part 2
theorem problem_2 (m : ℝ) (h1 : m ≠ 1) (h2 : m ≠ -1) (h3 : m ≠ 0) :
  (m / (m^2 - 1)) / ((m^2 - m) / (m^2 - 2*m + 1)) = 1 / (m + 1) := by sorry

end problem_1_problem_2_l3213_321390


namespace fraction_1800_1809_equals_4_13_l3213_321324

/-- The number of states that joined the union during 1800-1809. -/
def states_1800_1809 : ℕ := 8

/-- The total number of states in Jennifer's collection. -/
def total_states : ℕ := 26

/-- The fraction of states that joined during 1800-1809 out of the first 26 states. -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_states

theorem fraction_1800_1809_equals_4_13 : fraction_1800_1809 = 4 / 13 := by sorry

end fraction_1800_1809_equals_4_13_l3213_321324


namespace hyperbola_eccentricity_l3213_321367

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from its right vertex to one of its asymptotes is b/2,
    then its eccentricity is 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_distance : (a * b) / Real.sqrt (a^2 + b^2) = b / 2) : 
  (Real.sqrt (a^2 + b^2)) / a = 2 := by
  sorry

end hyperbola_eccentricity_l3213_321367


namespace ring_arrangement_count_l3213_321374

def number_of_rings : ℕ := 10
def rings_to_arrange : ℕ := 6
def number_of_fingers : ℕ := 4

def ring_arrangements (n k : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial k * Nat.choose (k + number_of_fingers) number_of_fingers

theorem ring_arrangement_count :
  ring_arrangements number_of_rings rings_to_arrange = 31752000 :=
by sorry

end ring_arrangement_count_l3213_321374


namespace journey_takes_eight_hours_l3213_321319

/-- Represents the journey with three people A, B, and C --/
structure Journey where
  totalDistance : ℝ
  carSpeed : ℝ
  walkSpeed : ℝ
  t1 : ℝ  -- time A and C drive together
  t2 : ℝ  -- time A drives back
  t3 : ℝ  -- time A and B drive while C walks

/-- The conditions of the journey --/
def journeyConditions (j : Journey) : Prop :=
  j.totalDistance = 100 ∧
  j.carSpeed = 25 ∧
  j.walkSpeed = 5 ∧
  j.carSpeed * j.t1 - j.carSpeed * j.t2 + j.carSpeed * j.t3 = j.totalDistance ∧
  j.walkSpeed * j.t1 + j.walkSpeed * j.t2 + j.carSpeed * j.t3 = j.totalDistance ∧
  j.carSpeed * j.t1 + j.walkSpeed * j.t2 + j.walkSpeed * j.t3 = j.totalDistance

/-- The theorem stating that the journey takes 8 hours --/
theorem journey_takes_eight_hours (j : Journey) (h : journeyConditions j) :
  j.t1 + j.t2 + j.t3 = 8 := by
  sorry

end journey_takes_eight_hours_l3213_321319


namespace airplane_capacity_theorem_l3213_321303

/-- The total luggage weight an airplane can hold -/
def airplane_luggage_capacity 
  (num_people : ℕ) 
  (bags_per_person : ℕ) 
  (bag_weight : ℕ) 
  (additional_bags : ℕ) : ℕ :=
  (num_people * bags_per_person * bag_weight) + (additional_bags * bag_weight)

/-- Theorem stating the total luggage weight the airplane can hold -/
theorem airplane_capacity_theorem :
  airplane_luggage_capacity 6 5 50 90 = 6000 := by
  sorry

end airplane_capacity_theorem_l3213_321303


namespace dam_building_problem_l3213_321377

/-- Represents the work rate of beavers building a dam -/
def work_rate (beavers : ℕ) (hours : ℝ) : ℝ := beavers * hours

/-- The number of beavers in the second group -/
def second_group_beavers : ℕ := 12

theorem dam_building_problem :
  let first_group_beavers : ℕ := 20
  let first_group_hours : ℝ := 3
  let second_group_hours : ℝ := 5
  work_rate first_group_beavers first_group_hours = work_rate second_group_beavers second_group_hours :=
by
  sorry

end dam_building_problem_l3213_321377


namespace expression_simplification_l3213_321310

theorem expression_simplification (x y : ℝ) :
  7 * y - 3 * x + 8 + 2 * y^2 - x + 12 = 2 * y^2 + 7 * y - 4 * x + 20 := by
  sorry

end expression_simplification_l3213_321310


namespace simplify_and_evaluate_l3213_321315

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) :
  (1 + 1 / (a + 1)) / ((a^2 - 4) / (2 * a + 2)) = 2 := by
  sorry

end simplify_and_evaluate_l3213_321315


namespace max_subset_with_distinct_sums_l3213_321360

def S (A : Finset ℕ) : Finset ℕ :=
  Finset.powerset A \ {∅} |>.image (λ B => B.sum id)

theorem max_subset_with_distinct_sums :
  (∃ (A : Finset ℕ), A ⊆ Finset.range 16 ∧ A.card = 5 ∧ S A.toSet.toFinset = S A) ∧
  ¬(∃ (A : Finset ℕ), A ⊆ Finset.range 16 ∧ A.card = 6 ∧ S A.toSet.toFinset = S A) :=
sorry

end max_subset_with_distinct_sums_l3213_321360


namespace tank_filling_time_l3213_321394

/-- Given a tank and three hoses X, Y, and Z, prove that they together fill the tank in 24/13 hours. -/
theorem tank_filling_time (T X Y Z : ℝ) (hxy : T = 2 * (X + Y)) (hxz : T = 3 * (X + Z)) (hyz : T = 4 * (Y + Z)) :
  T / (X + Y + Z) = 24 / 13 := by
  sorry

end tank_filling_time_l3213_321394


namespace bacteria_growth_l3213_321362

theorem bacteria_growth (n : ℕ) : 
  (∀ t : ℕ, t ≤ 10 → n * (4 ^ t) = n * 4 ^ t) →
  n * 4 ^ 10 = 1048576 ↔ n = 1 :=
by sorry

end bacteria_growth_l3213_321362


namespace season_length_l3213_321343

/-- The number of games in the entire season -/
def total_games : ℕ := 20

/-- Donovan Mitchell's current average points per game -/
def current_average : ℕ := 26

/-- Number of games played so far -/
def games_played : ℕ := 15

/-- Donovan Mitchell's goal average for the entire season -/
def goal_average : ℕ := 30

/-- Required average for remaining games to reach the goal -/
def required_average : ℕ := 42

/-- Theorem stating that the total number of games is 20 -/
theorem season_length : 
  current_average * games_played + required_average * (total_games - games_played) = 
  goal_average * total_games := by sorry

end season_length_l3213_321343


namespace frank_second_half_correct_l3213_321321

/-- Represents the number of questions Frank answered correctly in the first half -/
def first_half_correct : ℕ := 3

/-- Represents the points awarded for each correct answer -/
def points_per_question : ℕ := 3

/-- Represents Frank's final score -/
def final_score : ℕ := 15

/-- Calculates the number of questions Frank answered correctly in the second half -/
def second_half_correct : ℕ :=
  (final_score - first_half_correct * points_per_question) / points_per_question

theorem frank_second_half_correct :
  second_half_correct = 2 := by sorry

end frank_second_half_correct_l3213_321321


namespace sum_of_arithmetic_progressions_l3213_321383

/-- The sum of the first 40 terms of an arithmetic progression -/
def S (p : ℕ) : ℕ :=
  let a := p  -- first term
  let d := 2 * p + 2  -- common difference
  let n := 40  -- number of terms
  n * (2 * a + (n - 1) * d) / 2

/-- The sum of S_p for p from 1 to 10 -/
def total_sum : ℕ :=
  (Finset.range 10).sum (fun i => S (i + 1))

theorem sum_of_arithmetic_progressions : total_sum = 103600 := by
  sorry

end sum_of_arithmetic_progressions_l3213_321383


namespace increasing_linear_function_l3213_321337

def linearFunction (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem increasing_linear_function (k b : ℝ) (h : k > 0) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → linearFunction k b x₁ < linearFunction k b x₂ :=
by
  sorry

end increasing_linear_function_l3213_321337


namespace min_max_cubic_linear_exists_y_min_max_zero_min_max_value_is_zero_l3213_321305

theorem min_max_cubic_linear (y : ℝ) : 
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ |x^3 - x*y| = 0) ∨ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y| > 0) :=
sorry

theorem exists_y_min_max_zero : 
  ∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y| ≤ 0 :=
sorry

theorem min_max_value_is_zero : 
  ∃ (y : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y| ≤ 0) ∧ 
  (∀ (y' : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y'| ≤ 0) → y' = y) :=
sorry

end min_max_cubic_linear_exists_y_min_max_zero_min_max_value_is_zero_l3213_321305


namespace alien_home_planet_abductees_l3213_321332

def total_abducted : ℕ := 1000
def return_percentage : ℚ := 528 / 1000
def to_zog : ℕ := 135
def to_xelbor : ℕ := 88
def to_qyruis : ℕ := 45

theorem alien_home_planet_abductees :
  total_abducted - 
  (↑(total_abducted) * return_percentage).floor - 
  to_zog - 
  to_xelbor - 
  to_qyruis = 204 := by
  sorry

end alien_home_planet_abductees_l3213_321332


namespace conic_touches_square_l3213_321309

/-- The conic equation derived from the differential equation -/
def conic (h : ℝ) (x y : ℝ) : Prop :=
  y^2 + 2*h*x*y + x^2 = 9*(1 - h^2)

/-- The square with sides touching the conic -/
def square (x y : ℝ) : Prop :=
  (x = 3 ∨ x = -3 ∨ y = 3 ∨ y = -3) ∧ (abs x ≤ 3 ∧ abs y ≤ 3)

/-- The theorem stating that the conic touches the sides of the square -/
theorem conic_touches_square (h : ℝ) (h_bounds : 0 ≤ h ∧ h ≤ 1) :
  ∃ (x y : ℝ), conic h x y ∧ square x y :=
sorry

end conic_touches_square_l3213_321309


namespace tan_alpha_sqrt_three_l3213_321304

theorem tan_alpha_sqrt_three (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin α ^ 2 + Real.cos (2 * α) = 1 / 4) : 
  Real.tan α = Real.sqrt 3 := by
  sorry

end tan_alpha_sqrt_three_l3213_321304


namespace tailor_trim_problem_l3213_321322

theorem tailor_trim_problem (original_side : ℝ) (trimmed_other_side : ℝ) (remaining_area : ℝ) 
  (h1 : original_side = 18)
  (h2 : trimmed_other_side = 3)
  (h3 : remaining_area = 120) :
  ∃ x : ℝ, x = 10 ∧ (original_side - x) * (original_side - trimmed_other_side) = remaining_area :=
by
  sorry

end tailor_trim_problem_l3213_321322


namespace insert_zeros_is_perfect_cube_l3213_321378

/-- Given a non-negative integer n, the function calculates the number
    obtained by inserting n zeros between each digit of 1331. -/
def insert_zeros (n : ℕ) : ℕ :=
  10^(3*n+3) + 3 * 10^(2*n+2) + 3 * 10^(n+1) + 1

/-- Theorem stating that for any non-negative integer n,
    the number obtained by inserting n zeros between each digit of 1331
    is equal to (10^(n+1) + 1)^3. -/
theorem insert_zeros_is_perfect_cube (n : ℕ) :
  insert_zeros n = (10^(n+1) + 1)^3 := by
  sorry

end insert_zeros_is_perfect_cube_l3213_321378


namespace sqrt_square_eq_abs_l3213_321382

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end sqrt_square_eq_abs_l3213_321382


namespace not_parabola_l3213_321338

-- Define the equation
def equation (α : Real) (x y : Real) : Prop :=
  x^2 * Real.sin α + y^2 * Real.cos α = 1

-- Theorem statement
theorem not_parabola (α : Real) (h : α ∈ Set.Icc 0 Real.pi) :
  ¬∃ (a b c : Real), ∀ (x y : Real),
    equation α x y ↔ y = a*x^2 + b*x + c :=
sorry

end not_parabola_l3213_321338


namespace four_zeros_when_a_positive_l3213_321312

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * x + 1 else Real.log x / Real.log 3

def F (a : ℝ) (x : ℝ) : ℝ :=
  f a (f a x) + 1

theorem four_zeros_when_a_positive (a : ℝ) (h : a > 0) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    F a x₁ = 0 ∧ F a x₂ = 0 ∧ F a x₃ = 0 ∧ F a x₄ = 0 ∧
    ∀ x, F a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ :=
sorry

end

end four_zeros_when_a_positive_l3213_321312


namespace probability_factor_less_than_8_of_90_l3213_321352

def positive_factors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ x => x > 0 ∧ n % x = 0)

def factors_less_than (n k : ℕ) : Finset ℕ :=
  (positive_factors n).filter (λ x => x < k)

theorem probability_factor_less_than_8_of_90 :
  (factors_less_than 90 8).card / (positive_factors 90).card = 5 / 12 := by
  sorry

end probability_factor_less_than_8_of_90_l3213_321352


namespace apple_bags_count_l3213_321317

/-- The number of bags of apples loaded onto a lorry -/
def number_of_bags (empty_weight loaded_weight bag_weight : ℕ) : ℕ :=
  (loaded_weight - empty_weight) / bag_weight

/-- Theorem stating that the number of bags of apples is 20 -/
theorem apple_bags_count : 
  let empty_weight : ℕ := 500
  let loaded_weight : ℕ := 1700
  let bag_weight : ℕ := 60
  number_of_bags empty_weight loaded_weight bag_weight = 20 := by
  sorry

end apple_bags_count_l3213_321317


namespace wheat_bread_served_l3213_321327

/-- The number of loaves of wheat bread served at a restaurant -/
def wheat_bread : ℝ := 0.9 - 0.4

/-- The total number of loaves served at the restaurant -/
def total_loaves : ℝ := 0.9

/-- The number of loaves of white bread served at the restaurant -/
def white_bread : ℝ := 0.4

/-- Theorem stating that the number of loaves of wheat bread served is 0.5 -/
theorem wheat_bread_served : wheat_bread = 0.5 := by
  sorry

end wheat_bread_served_l3213_321327


namespace sin_cos_identity_l3213_321325

theorem sin_cos_identity (α : Real) (h : Real.sin α ^ 2 + Real.sin α = 1) :
  Real.cos α ^ 4 + Real.cos α ^ 2 = 1 := by
  sorry

end sin_cos_identity_l3213_321325


namespace equation_2010_l3213_321342

theorem equation_2010 (digits : Finset Nat) : digits = {2, 3, 5, 6, 7} →
  ∃ (a b c d : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  ((d = 67 ∧ (6 ∈ digits ∧ 7 ∈ digits)) ∨ d ∈ digits) ∧
  a * b * c * d = 2010 :=
by
  sorry

#check equation_2010

end equation_2010_l3213_321342


namespace pool_draining_rate_l3213_321300

/-- Given a rectangular pool with specified dimensions, capacity, and draining time,
    calculate the rate of water removal in cubic feet per minute. -/
theorem pool_draining_rate
  (width : ℝ) (length : ℝ) (depth : ℝ) (capacity : ℝ) (drain_time : ℝ)
  (h_width : width = 50)
  (h_length : length = 150)
  (h_depth : depth = 10)
  (h_capacity : capacity = 0.8)
  (h_drain_time : drain_time = 1000)
  : (width * length * depth * capacity) / drain_time = 60 := by
  sorry

end pool_draining_rate_l3213_321300


namespace division_reduction_l3213_321366

theorem division_reduction (x : ℝ) : 
  (63 / x = 63 - 42) → x = 3 := by
  sorry

end division_reduction_l3213_321366


namespace acai_berry_juice_cost_per_litre_l3213_321326

/-- The cost per litre of açaí berry juice given the following conditions:
  * The superfruit juice cocktail costs $1399.45 per litre to make.
  * The mixed fruit juice costs $262.85 per litre.
  * 37 litres of mixed fruit juice are used.
  * 24.666666666666668 litres of açaí berry juice are used.
-/
theorem acai_berry_juice_cost_per_litre :
  let cocktail_cost_per_litre : ℝ := 1399.45
  let mixed_fruit_juice_cost_per_litre : ℝ := 262.85
  let mixed_fruit_juice_volume : ℝ := 37
  let acai_berry_juice_volume : ℝ := 24.666666666666668
  let total_volume : ℝ := mixed_fruit_juice_volume + acai_berry_juice_volume
  let mixed_fruit_juice_total_cost : ℝ := mixed_fruit_juice_cost_per_litre * mixed_fruit_juice_volume
  let cocktail_total_cost : ℝ := cocktail_cost_per_litre * total_volume
  let acai_berry_juice_total_cost : ℝ := cocktail_total_cost - mixed_fruit_juice_total_cost
  let acai_berry_juice_cost_per_litre : ℝ := acai_berry_juice_total_cost / acai_berry_juice_volume
  acai_berry_juice_cost_per_litre = 3105.99 :=
by
  sorry


end acai_berry_juice_cost_per_litre_l3213_321326


namespace orange_count_correct_l3213_321320

/-- The number of oranges in the box -/
def num_oranges : ℕ := 24

/-- The initial number of kiwis in the box -/
def initial_kiwis : ℕ := 30

/-- The number of kiwis added to the box -/
def added_kiwis : ℕ := 26

/-- The percentage of oranges after adding kiwis -/
def orange_percentage : ℚ := 30 / 100

theorem orange_count_correct :
  (orange_percentage * (num_oranges + initial_kiwis + added_kiwis) : ℚ) = num_oranges := by
  sorry

end orange_count_correct_l3213_321320


namespace necessary_not_sufficient_condition_l3213_321351

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ a, a^2 > 2*a → (a > 2 ∨ a < 0)) ∧
  (∃ a, a > 2 ∧ a^2 ≤ 2*a) :=
sorry

end necessary_not_sufficient_condition_l3213_321351


namespace polynomial_value_equality_l3213_321350

theorem polynomial_value_equality (x : ℝ) : 
  x^2 - (5/2)*x = 6 → 2*x^2 - 5*x + 6 = 18 := by
sorry

end polynomial_value_equality_l3213_321350


namespace line_passes_through_K_min_distance_AC_dot_product_range_l3213_321323

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 12

-- Define the line l
def line_l (m x y : ℝ) : Prop := m*x - y - 2*m + 3 = 0

-- Define the point K
def point_K : ℝ × ℝ := (2, 3)

-- Define the intersection points A and C
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ circle_M x y ∧ line_l m x y}

-- Theorem 1: Line l passes through point K for all m
theorem line_passes_through_K (m : ℝ) : line_l m (point_K.1) (point_K.2) :=
sorry

-- Theorem 2: Minimum distance between intersection points is 4
theorem min_distance_AC :
  ∃ (m : ℝ), ∀ (A C : ℝ × ℝ), A ∈ intersection_points m → C ∈ intersection_points m →
  A ≠ C → ‖A - C‖ ≥ 4 :=
sorry

-- Theorem 3: Range of dot product MA · MC
theorem dot_product_range (M : ℝ × ℝ) (m : ℝ) :
  M = (4, 5) →
  ∀ (A C : ℝ × ℝ), A ∈ intersection_points m → C ∈ intersection_points m →
  -12 ≤ (A - M) • (C - M) ∧ (A - M) • (C - M) ≤ 4 :=
sorry

end line_passes_through_K_min_distance_AC_dot_product_range_l3213_321323


namespace rooster_to_hen_ratio_l3213_321372

/-- Given a chicken farm with roosters and hens, prove the ratio of roosters to hens. -/
theorem rooster_to_hen_ratio 
  (total_chickens : ℕ) 
  (roosters : ℕ) 
  (h_total : total_chickens = 9000)
  (h_roosters : roosters = 6000) : 
  (roosters : ℚ) / (total_chickens - roosters) = 2 := by
  sorry

end rooster_to_hen_ratio_l3213_321372


namespace winner_takes_eight_l3213_321345

/-- Represents the game state and rules --/
structure Game where
  winner_candies : ℕ
  loser_candies : ℕ
  total_rounds : ℕ
  tim_wins : ℕ
  nick_total : ℕ
  tim_total : ℕ

/-- The game satisfies the given conditions --/
def valid_game (g : Game) : Prop :=
  g.winner_candies > g.loser_candies ∧
  g.loser_candies > 0 ∧
  g.tim_wins = 2 ∧
  g.nick_total = 30 ∧
  g.tim_total = 25 ∧
  g.total_rounds * (g.winner_candies + g.loser_candies) = g.nick_total + g.tim_total

/-- The theorem to be proved --/
theorem winner_takes_eight (g : Game) (h : valid_game g) : g.winner_candies = 8 := by
  sorry

end winner_takes_eight_l3213_321345


namespace intersection_properties_l3213_321361

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Define the intersection points M and N (existence assumed)
axiom exists_intersection_points : ∃ (M N : ℝ × ℝ), 
  curve_C M.1 M.2 ∧ line_l M.1 M.2 ∧
  curve_C N.1 N.2 ∧ line_l N.1 N.2 ∧
  M ≠ N

-- State the theorem
theorem intersection_properties :
  ∃ (M N : ℝ × ℝ), 
    curve_C M.1 M.2 ∧ line_l M.1 M.2 ∧
    curve_C N.1 N.2 ∧ line_l N.1 N.2 ∧
    M ≠ N ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 8 ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) *
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) = 14 := by
  sorry


end intersection_properties_l3213_321361


namespace tangent_line_at_one_l3213_321393

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x^2 - 4 * x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = f x) →
    (y = m * (x - 1) + f 1) →
    (m * x - y - b = 0) →
    (m = Real.exp 1) ∧
    (b = 2) :=
sorry

end tangent_line_at_one_l3213_321393


namespace smallest_four_digit_divisible_by_53_l3213_321335

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end smallest_four_digit_divisible_by_53_l3213_321335


namespace quadratic_solution_l3213_321389

theorem quadratic_solution (c : ℝ) : 
  ((-9 : ℝ)^2 + c * (-9 : ℝ) + 45 = 0) → c = 14 := by
  sorry

end quadratic_solution_l3213_321389


namespace binary_to_quaternary_conversion_l3213_321344

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal [false, true, true, true, false, true, true, false, true]) = [1, 1, 2, 3, 2] := by
  sorry

end binary_to_quaternary_conversion_l3213_321344


namespace inverse_proposition_reciprocals_l3213_321333

/-- The inverse proposition of "If ab = 1, then a and b are reciprocals" -/
theorem inverse_proposition_reciprocals (a b : ℝ) :
  (a ≠ 0 ∧ b ≠ 0 ∧ a * b = 1 → a = 1 / b ∧ b = 1 / a) →
  (a = 1 / b ∧ b = 1 / a → a * b = 1) :=
sorry

end inverse_proposition_reciprocals_l3213_321333


namespace reflection_line_sum_l3213_321363

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), (x, y) = (10, 7) ∧ 
   (x - 2)^2 + (y - 3)^2 = (10 - 2)^2 + (7 - 3)^2 ∧
   (y - 3) = m * (x - 2) ∧
   y = m * x + b) →
  m + b = 15 := by
sorry

end reflection_line_sum_l3213_321363


namespace average_of_six_numbers_l3213_321357

theorem average_of_six_numbers (a b c d e f : ℝ) :
  (a + b + c + d + e + f) / 6 = 4.60 →
  (c + d) / 2 = 3.8 →
  (e + f) / 2 = 6.6 →
  (a + b) / 2 = 3.4 :=
by sorry

end average_of_six_numbers_l3213_321357


namespace rip3_properties_l3213_321385

-- Define the basic concepts
def Cell : Type := sorry
def RIP3 : Type := sorry
def Gene : Type := sorry

-- Define the properties and relationships
def can_convert_apoptosis_to_necrosis (r : RIP3) : Prop := sorry
def controls_synthesis_of (g : Gene) (r : RIP3) : Prop := sorry
def exists_in_human_body (r : RIP3) : Prop := sorry
def can_regulate_cell_death_mode (r : RIP3) : Prop := sorry
def has_gene (c : Cell) (g : Gene) : Prop := sorry

-- State the theorem
theorem rip3_properties :
  ∃ (r : RIP3) (g : Gene),
    exists_in_human_body r ∧
    can_convert_apoptosis_to_necrosis r ∧
    controls_synthesis_of g r ∧
    can_regulate_cell_death_mode r ∧
    ∀ (c : Cell), has_gene c g :=
sorry

-- Note: This theorem encapsulates the main points about RIP3 from the problem statement,
-- without making claims about the correctness or incorrectness of the given statements.

end rip3_properties_l3213_321385


namespace range_of_a_solution_set_l3213_321302

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for part I
theorem range_of_a (a : ℝ) :
  (∃ x, f x < 2 * a - 1) ↔ a > 2 :=
sorry

-- Theorem for part II
theorem solution_set :
  {x : ℝ | f x ≥ x^2 - 2*x} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 + Real.sqrt 3} :=
sorry

end range_of_a_solution_set_l3213_321302


namespace sports_meeting_score_l3213_321396

/-- Represents the score for a single placement --/
inductive Placement
| first
| second
| third

/-- Calculates the score for a given placement --/
def score (p : Placement) : Nat :=
  match p with
  | .first => 5
  | .second => 3
  | .third => 1

/-- Represents the placements of a class --/
structure ClassPlacements where
  first : Nat
  second : Nat
  third : Nat

/-- Calculates the total score for a class given its placements --/
def totalScore (cp : ClassPlacements) : Nat :=
  cp.first * score Placement.first +
  cp.second * score Placement.second +
  cp.third * score Placement.third

/-- Calculates the total number of placements for a class --/
def totalPlacements (cp : ClassPlacements) : Nat :=
  cp.first + cp.second + cp.third

theorem sports_meeting_score (class1 class2 : ClassPlacements) :
  totalPlacements class1 = 2 →
  totalPlacements class2 = 4 →
  totalScore class1 = totalScore class2 →
  totalScore class1 + totalScore class2 + 7 = 27 :=
by sorry

end sports_meeting_score_l3213_321396


namespace g_three_sixteenths_l3213_321388

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1) ∧
  (g 0 = 0) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2)

-- Theorem statement
theorem g_three_sixteenths (g : ℝ → ℝ) (h : g_properties g) : g (3/16) = 1/4 := by
  sorry

end g_three_sixteenths_l3213_321388


namespace square_equation_solve_l3213_321381

theorem square_equation_solve (x y : ℝ) (h1 : x^2 = y + 4) (h2 : x = 7) : y = 45 := by
  sorry

end square_equation_solve_l3213_321381


namespace total_price_increase_l3213_321373

-- Define the sequence of price increases
def price_increases : List Real := [0.375, 0.31, 0.427, 0.523, 0.272]

-- Function to calculate the total price increase factor
def total_increase_factor (increases : List Real) : Real :=
  List.foldl (fun acc x => acc * (1 + x)) 1 increases

-- Theorem stating the total equivalent percent increase
theorem total_price_increase : 
  ∀ ε > 0, 
  |total_increase_factor price_increases - 1 - 3.9799| < ε := by
  sorry

end total_price_increase_l3213_321373


namespace minimum_trips_moscow_l3213_321328

theorem minimum_trips_moscow (x y : ℕ) : 
  (31 * x + 32 * y = 5000) → 
  (∀ a b : ℕ, 31 * a + 32 * b = 5000 → x + y ≤ a + b) →
  x + y = 157 := by
sorry

end minimum_trips_moscow_l3213_321328


namespace fraction_to_decimal_l3213_321353

theorem fraction_to_decimal : (63 : ℚ) / (2^3 * 5^4) = 0.0126 := by
  sorry

end fraction_to_decimal_l3213_321353


namespace fifteen_plus_neg_twentythree_l3213_321354

-- Define the operation for adding a positive and negative rational number
def add_pos_neg (a b : ℚ) : ℚ := -(b - a)

-- State the theorem
theorem fifteen_plus_neg_twentythree :
  15 + (-23) = add_pos_neg 15 23 :=
by sorry

end fifteen_plus_neg_twentythree_l3213_321354


namespace mathematics_puzzle_solution_l3213_321392

/-- Represents a mapping from characters to either digits or arithmetic operations -/
def LetterMapping := Char → Option (Nat ⊕ Bool)

/-- The word to be mapped -/
def word : List Char := ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S']

/-- Evaluates an expression given a mapping -/
def evalExpression (mapping : LetterMapping) (expr : List Char) : Option Int := sorry

/-- Checks if a mapping is valid according to the problem constraints -/
def isValidMapping (mapping : LetterMapping) : Prop := sorry

theorem mathematics_puzzle_solution :
  ∃ (mapping : LetterMapping),
    isValidMapping mapping ∧
    evalExpression mapping word = some 2014 := by sorry

end mathematics_puzzle_solution_l3213_321392


namespace find_divisor_l3213_321387

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 140)
  (h2 : quotient = 9)
  (h3 : remainder = 5)
  : ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 15 := by
  sorry

end find_divisor_l3213_321387


namespace area_is_seven_and_half_l3213_321368

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf_continuous : Continuous f)
variable (hf_monotone : Monotone f)
variable (hf_0 : f 0 = 0)
variable (hf_1 : f 1 = 1)

-- Define the area calculation function
def area_bounded (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem area_is_seven_and_half :
  area_bounded f = 7.5 :=
sorry

end area_is_seven_and_half_l3213_321368


namespace b_approximation_l3213_321398

/-- Given that a = 2.68 * 0.74, prove that b = a^2 + cos(a) is approximately 2.96535 -/
theorem b_approximation (a : ℝ) (h : a = 2.68 * 0.74) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |a^2 + Real.cos a - 2.96535| < ε :=
sorry

end b_approximation_l3213_321398


namespace bowling_ball_weight_is_correct_l3213_321314

/-- The weight of a single bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18.75

/-- The weight of a single canoe in pounds -/
def canoe_weight : ℝ := 30

/-- Theorem stating that the weight of one bowling ball is 18.75 pounds -/
theorem bowling_ball_weight_is_correct :
  (8 * bowling_ball_weight = 5 * canoe_weight) ∧
  (4 * canoe_weight = 120) →
  bowling_ball_weight = 18.75 :=
by
  sorry

end bowling_ball_weight_is_correct_l3213_321314


namespace arithmetic_sequence_15_to_100_l3213_321376

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceTerms (a : ℕ) (d : ℕ) (lastTerm : ℕ) : ℕ :=
  (lastTerm - a) / d + 1

/-- Theorem: The arithmetic sequence with first term 15, last term 100, and common difference 5 has 18 terms -/
theorem arithmetic_sequence_15_to_100 :
  arithmeticSequenceTerms 15 5 100 = 18 := by
  sorry

#eval arithmeticSequenceTerms 15 5 100

end arithmetic_sequence_15_to_100_l3213_321376


namespace polynomial_division_remainder_l3213_321306

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 = (x^2 + 3*x - 4) * q + (-51*x + 52) :=
sorry

end polynomial_division_remainder_l3213_321306


namespace operation_result_l3213_321318

def operation (n : ℕ) : ℕ := 2 * n + 1

def iterate_operation (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => operation (iterate_operation n k)

theorem operation_result (x : ℕ) :
  ¬(∃ (y : ℕ), iterate_operation x 100 = 1980 * y) ∧
  (∃ (x : ℕ), ∃ (y : ℕ), iterate_operation x 100 = 1981 * y) := by
  sorry


end operation_result_l3213_321318


namespace min_value_xy_minus_2x_l3213_321340

theorem min_value_xy_minus_2x (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y * Real.log x + y * Real.log y = Real.exp x) :
  ∃ (m : ℝ), m = 2 - 2 * Real.log 2 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 
  y' * Real.log x' + y' * Real.log y' = Real.exp x' → x' * y' - 2 * x' ≥ m :=
sorry

end min_value_xy_minus_2x_l3213_321340


namespace representable_set_l3213_321331

def representable (k : ℕ) : Prop :=
  ∃ x y z : ℕ+, k = (x + y + z)^2 / (x * y * z)

theorem representable_set : 
  {k : ℕ | representable k} = {1, 2, 3, 4, 5, 6, 8, 9} :=
by sorry

end representable_set_l3213_321331


namespace lcm_problem_l3213_321329

theorem lcm_problem (a b c : ℕ+) (h1 : a = 15) (h2 : b = 25) (h3 : Nat.lcm (Nat.lcm a b) c = 525) : c = 7 := by
  sorry

end lcm_problem_l3213_321329


namespace proposition_and_variants_are_false_l3213_321399

theorem proposition_and_variants_are_false :
  (¬ ∀ (a b : ℝ), ab ≤ 0 → a ≤ 0 ∨ b ≤ 0) ∧
  (¬ ∀ (a b : ℝ), (a ≤ 0 ∨ b ≤ 0) → ab ≤ 0) ∧
  (¬ ∀ (a b : ℝ), ab > 0 → a > 0 ∧ b > 0) ∧
  (¬ ∀ (a b : ℝ), (a > 0 ∧ b > 0) → ab > 0) :=
by sorry

end proposition_and_variants_are_false_l3213_321399


namespace factorization_equality_l3213_321380

theorem factorization_equality (a b : ℝ) : 4 * a^2 * b - b = b * (2*a + 1) * (2*a - 1) := by
  sorry

end factorization_equality_l3213_321380


namespace area_under_curve_l3213_321316

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the bounds of integration
def a : ℝ := 0
def b : ℝ := 1

-- State the theorem
theorem area_under_curve : ∫ x in a..b, f x = 4/3 := by
  sorry

end area_under_curve_l3213_321316


namespace car_speed_proof_l3213_321358

/-- Proves that a car's speed is 60 km/h if it takes 12 seconds longer to travel 1 km compared to 75 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v * 3600 = 1 / 75 * 3600 + 12) ↔ v = 60 :=
by sorry

end car_speed_proof_l3213_321358


namespace derivatives_correct_l3213_321313

-- Function 1
def f₁ (x : ℝ) : ℝ := 3 * x^3 - 4 * x

-- Function 2
def f₂ (x : ℝ) : ℝ := (2 * x - 1) * (3 * x + 2)

-- Function 3
def f₃ (x : ℝ) : ℝ := x^2 * (x^3 - 4)

theorem derivatives_correct :
  (∀ x, deriv f₁ x = 9 * x^2 - 4) ∧
  (∀ x, deriv f₂ x = 12 * x + 1) ∧
  (∀ x, deriv f₃ x = 5 * x^4 - 8 * x) := by
  sorry

end derivatives_correct_l3213_321313


namespace square_of_binomial_constant_l3213_321334

/-- If 16x^2 + 40x + b is the square of a binomial, then b = 25 -/
theorem square_of_binomial_constant (b : ℝ) : 
  (∃ (p q : ℝ), ∀ x, 16 * x^2 + 40 * x + b = (p * x + q)^2) → b = 25 := by
  sorry

end square_of_binomial_constant_l3213_321334


namespace fishing_problem_l3213_321364

/-- Represents the number of fish caught by each person --/
structure FishCaught where
  jason : ℕ
  ryan : ℕ
  jeffery : ℕ

/-- The fishing problem statement --/
theorem fishing_problem (f : FishCaught) 
  (h1 : f.jason + f.ryan + f.jeffery = 100)
  (h2 : f.jeffery = 2 * f.ryan)
  (h3 : f.jeffery = 60) : 
  f.ryan = 30 := by
sorry


end fishing_problem_l3213_321364


namespace mary_flour_needed_l3213_321370

/-- The number of cups of flour required by the recipe -/
def recipe_flour : ℕ := 9

/-- The number of cups of flour Mary has already added -/
def added_flour : ℕ := 2

/-- The number of cups of flour Mary needs to add -/
def flour_needed : ℕ := recipe_flour - added_flour

theorem mary_flour_needed : flour_needed = 7 := by sorry

end mary_flour_needed_l3213_321370


namespace sqrt_5_greater_than_2_l3213_321341

theorem sqrt_5_greater_than_2 : Real.sqrt 5 > 2 := by
  sorry

end sqrt_5_greater_than_2_l3213_321341


namespace lemonade_stand_profit_l3213_321347

/-- Calculates the net profit from a lemonade stand --/
theorem lemonade_stand_profit
  (glasses_per_gallon : ℕ)
  (cost_per_gallon : ℚ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ)
  (h1 : glasses_per_gallon = 16)
  (h2 : cost_per_gallon = 7/2)
  (h3 : gallons_made = 2)
  (h4 : price_per_glass = 1)
  (h5 : glasses_drunk = 5)
  (h6 : glasses_unsold = 6) :
  (gallons_made * glasses_per_gallon - glasses_drunk - glasses_unsold) * price_per_glass -
  (gallons_made * cost_per_gallon) = 14 :=
by sorry

end lemonade_stand_profit_l3213_321347


namespace infinitely_many_special_n_l3213_321397

theorem infinitely_many_special_n : ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 
  (∃ m : ℕ, n * m = 2^(2^n + 1) + 1) ∧ 
  (∀ m : ℕ, n * m ≠ 2^n + 1) := by
  sorry

end infinitely_many_special_n_l3213_321397


namespace members_playing_both_sports_l3213_321371

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  neither : ℕ

/-- Calculates the number of members playing both badminton and tennis -/
def both_sports (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - (club.total - club.neither)

/-- Theorem stating the number of members playing both sports in the given club -/
theorem members_playing_both_sports (club : SportsClub)
  (h1 : club.total = 30)
  (h2 : club.badminton = 16)
  (h3 : club.tennis = 19)
  (h4 : club.neither = 2) :
  both_sports club = 7 := by
  sorry

end members_playing_both_sports_l3213_321371


namespace triple_overlap_area_is_six_l3213_321355

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the hall and the carpets placed in it -/
structure CarpetLayout where
  hallWidth : ℝ
  hallHeight : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area covered by all three carpets in the given layout -/
def tripleOverlapArea (layout : CarpetLayout) : ℝ :=
  sorry

/-- Theorem stating that the area covered by all three carpets is 6 square meters -/
theorem triple_overlap_area_is_six (layout : CarpetLayout) 
  (h1 : layout.hallWidth = 10 ∧ layout.hallHeight = 10)
  (h2 : layout.carpet1.width = 6 ∧ layout.carpet1.height = 8)
  (h3 : layout.carpet2.width = 6 ∧ layout.carpet2.height = 6)
  (h4 : layout.carpet3.width = 5 ∧ layout.carpet3.height = 7) :
  tripleOverlapArea layout = 6 :=
sorry

end triple_overlap_area_is_six_l3213_321355


namespace min_cars_with_racing_stripes_l3213_321308

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 37)
  (h3 : max_ac_no_stripes = 59) :
  ∃ (min_cars_with_stripes : ℕ), 
    min_cars_with_stripes = 4 ∧ 
    min_cars_with_stripes ≤ total_cars - cars_without_ac ∧
    min_cars_with_stripes = total_cars - cars_without_ac - max_ac_no_stripes :=
by
  sorry

#check min_cars_with_racing_stripes

end min_cars_with_racing_stripes_l3213_321308


namespace train_crossing_time_train_crossing_platform_time_l3213_321356

/-- Calculates the time required for a train to cross a platform --/
theorem train_crossing_time 
  (train_speed_kmph : ℝ) 
  (man_crossing_time : ℝ) 
  (platform_length : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let train_length := train_speed_mps * man_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

/-- Proves that the train takes 30 seconds to cross the platform --/
theorem train_crossing_platform_time : 
  train_crossing_time 72 19 220 = 30 := by
  sorry

end train_crossing_time_train_crossing_platform_time_l3213_321356


namespace increasing_order_x_z_y_l3213_321375

theorem increasing_order_x_z_y (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  x < x^(x^x) ∧ x^(x^x) < x^x := by sorry

end increasing_order_x_z_y_l3213_321375
