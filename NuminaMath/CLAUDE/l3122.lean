import Mathlib

namespace semi_circle_perimeter_l3122_312293

/-- The perimeter of a semi-circle with radius 38.50946843518593 cm is 198.03029487037186 cm. -/
theorem semi_circle_perimeter :
  let r : ℝ := 38.50946843518593
  let π : ℝ := Real.pi
  let perimeter : ℝ := π * r + 2 * r
  perimeter = 198.03029487037186 := by sorry

end semi_circle_perimeter_l3122_312293


namespace pool_filling_rate_l3122_312269

theorem pool_filling_rate (jim_rate sue_rate tony_rate : ℚ) 
  (h_jim : jim_rate = 1 / 30)
  (h_sue : sue_rate = 1 / 45)
  (h_tony : tony_rate = 1 / 90) :
  jim_rate + sue_rate + tony_rate = 1 / 15 := by
  sorry

end pool_filling_rate_l3122_312269


namespace right_trapezoid_base_difference_l3122_312260

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The length of the longer leg -/
  longer_leg : ℝ
  /-- The measure of the largest angle in degrees -/
  largest_angle : ℝ
  /-- The length of the longer base -/
  longer_base : ℝ
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- The longer leg is positive -/
  longer_leg_pos : longer_leg > 0
  /-- The largest angle is between 90° and 180° -/
  largest_angle_range : 90 < largest_angle ∧ largest_angle < 180
  /-- The longer base is longer than the shorter base -/
  base_order : longer_base > shorter_base

/-- The theorem stating the difference between bases of the specific right trapezoid -/
theorem right_trapezoid_base_difference (t : RightTrapezoid) 
    (h1 : t.longer_leg = 12)
    (h2 : t.largest_angle = 120) :
    t.longer_base - t.shorter_base = 6 := by
  sorry

end right_trapezoid_base_difference_l3122_312260


namespace dropped_student_score_l3122_312270

theorem dropped_student_score 
  (initial_students : ℕ) 
  (initial_average : ℝ) 
  (remaining_students : ℕ) 
  (new_average : ℝ) : ℝ :=
  by
  have h1 : initial_students = 16 := by sorry
  have h2 : initial_average = 60.5 := by sorry
  have h3 : remaining_students = 15 := by sorry
  have h4 : new_average = 64 := by sorry

  -- The score of the dropped student
  let dropped_score := initial_students * initial_average - remaining_students * new_average

  -- Prove that the dropped score is 8
  have h5 : dropped_score = 8 := by sorry

  exact dropped_score

end dropped_student_score_l3122_312270


namespace fraction_to_decimal_conversion_l3122_312209

theorem fraction_to_decimal_conversion :
  ∃ (d : ℚ), (d.num / d.den = 7 / 12) ∧ (abs (d - 0.58333) < 0.000005) := by
  sorry

end fraction_to_decimal_conversion_l3122_312209


namespace quadratic_two_distinct_roots_l3122_312229

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
by sorry

end quadratic_two_distinct_roots_l3122_312229


namespace root_product_sum_l3122_312208

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x, Real.sqrt 2021 * x^3 - 4043 * x^2 + x + 2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 2 * Real.sqrt 2021 :=
by sorry

end root_product_sum_l3122_312208


namespace min_value_theorem_min_value_is_glb_l3122_312219

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := by sorry

theorem min_value_is_glb : ∃ (seq : ℕ → ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
  ((seq n)^2 + 9) / Real.sqrt ((seq n)^2 + 5) < 4 + ε := by sorry

end min_value_theorem_min_value_is_glb_l3122_312219


namespace parallel_lines_k_value_l3122_312204

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 3x + 5 and y = (5k)x + 7 are parallel -/
theorem parallel_lines_k_value : 
  (∀ x y : ℝ, y = 3 * x + 5 ↔ y = (5 * k) * x + 7) → k = 3/5 := by
  sorry

end parallel_lines_k_value_l3122_312204


namespace mashed_potatoes_tomatoes_difference_l3122_312261

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 144

/-- The number of students who suggested bacon -/
def bacon : ℕ := 467

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 79

/-- The difference between the number of students who suggested mashed potatoes and tomatoes -/
def difference : ℕ := mashed_potatoes - tomatoes

theorem mashed_potatoes_tomatoes_difference : difference = 65 := by sorry

end mashed_potatoes_tomatoes_difference_l3122_312261


namespace smallest_leftover_four_boxes_l3122_312280

/-- The number of kids among whom the Snackies are distributed -/
def num_kids : ℕ := 8

/-- The number of Snackies left over when one box is divided among the kids -/
def leftover_one_box : ℕ := 5

/-- The number of boxes used in the final distribution -/
def num_boxes : ℕ := 4

/-- Represents the number of Snackies in one box -/
def snackies_per_box : ℕ := num_kids * leftover_one_box + leftover_one_box

theorem smallest_leftover_four_boxes :
  ∃ (leftover : ℕ), leftover < num_kids ∧
  ∃ (pieces_per_kid : ℕ),
    num_boxes * snackies_per_box = num_kids * pieces_per_kid + leftover ∧
    ∀ (smaller_leftover : ℕ),
      smaller_leftover < leftover →
      ¬∃ (alt_pieces_per_kid : ℕ),
        num_boxes * snackies_per_box = num_kids * alt_pieces_per_kid + smaller_leftover :=
by sorry

end smallest_leftover_four_boxes_l3122_312280


namespace square_root_49_squared_l3122_312214

theorem square_root_49_squared : Real.sqrt 49 ^ 2 = 49 := by
  sorry

end square_root_49_squared_l3122_312214


namespace existence_of_unique_representation_sets_l3122_312296

-- Define the property of being an infinite set of non-negative integers
def IsInfiniteNonNegSet (S : Set ℕ) : Prop :=
  Set.Infinite S ∧ ∀ x ∈ S, x ≥ 0

-- Define the property that every non-negative integer has a unique representation
def HasUniqueRepresentation (A B : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b

-- The main theorem
theorem existence_of_unique_representation_sets :
  ∃ A B : Set ℕ, IsInfiniteNonNegSet A ∧ IsInfiniteNonNegSet B ∧ HasUniqueRepresentation A B :=
sorry

end existence_of_unique_representation_sets_l3122_312296


namespace terrys_breakfast_spending_l3122_312220

theorem terrys_breakfast_spending (x : ℝ) : 
  x > 0 ∧ x + 2*x + 6*x = 54 → x = 6 := by
  sorry

end terrys_breakfast_spending_l3122_312220


namespace painted_cube_problem_l3122_312239

theorem painted_cube_problem (n : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (2 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 6 → 
  n = 2 := by
  sorry

end painted_cube_problem_l3122_312239


namespace cosine_inequality_l3122_312249

theorem cosine_inequality (y : Real) :
  y ∈ Set.Icc 0 Real.pi →
  (∀ x : Real, Real.cos (x + y) ≤ Real.cos x * Real.cos y) ↔
  (y = 0 ∨ y = Real.pi) := by
  sorry

end cosine_inequality_l3122_312249


namespace swap_result_l3122_312227

def swap_values (a b : ℕ) : ℕ × ℕ :=
  let t := a
  let a := b
  let b := t
  (a, b)

theorem swap_result : swap_values 3 2 = (2, 3) := by sorry

end swap_result_l3122_312227


namespace circle_equation_problem1_circle_equation_problem2_l3122_312287

-- Problem 1
theorem circle_equation_problem1 (x y : ℝ) :
  (∃ (h : ℝ), x - 2*y - 2 = 0 ∧ 
    (x - 0)^2 + (y - 4)^2 = (x - 4)^2 + (y - 6)^2) →
  (x - 4)^2 + (y - 1)^2 = 25 :=
sorry

-- Problem 2
theorem circle_equation_problem2 (x y : ℝ) :
  (2*2 + 3*2 - 10 = 0 ∧
    ((x - 2)^2 + (y - 2)^2 = 13 ∧
     (y - 2)/(x - 2) * (-2/3) = -1)) →
  ((x - 4)^2 + (y - 5)^2 = 13 ∨ x^2 + (y + 1)^2 = 13) :=
sorry

end circle_equation_problem1_circle_equation_problem2_l3122_312287


namespace quadratic_function_properties_l3122_312279

/-- A quadratic function f(x) = ax² + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties :
  ∃ (a b c : ℝ),
    (∀ x : ℝ, quadratic_function a b c (-1) = 0 ∧
              quadratic_function a b c (x + 1) - quadratic_function a b c x = 2 * x) →
    (∀ x : ℝ, quadratic_function a b c x = x^2 - x - 2) ∧
    (∀ x : ℝ, quadratic_function a b c x ≥ 0) ∧
    (∀ x : ℝ, quadratic_function a b c (x - 4) = quadratic_function a b c (2 - x)) ∧
    (∀ x : ℝ, 0 ≤ quadratic_function a b c x - x ∧
              quadratic_function a b c x - x ≤ (1/2) * (x - 1)^2) ∧
    a = 1/4 ∧ b = 1/2 ∧ c = 1/4 := by
  sorry

end quadratic_function_properties_l3122_312279


namespace water_requirement_l3122_312245

/-- Represents the amount of a substance in moles -/
def Moles : Type := ℝ

/-- Represents a chemical reaction between ammonium chloride and water -/
structure Reaction where
  nh4cl : Moles  -- Amount of ammonium chloride
  h2o : Moles    -- Amount of water
  hcl : Moles    -- Amount of hydrochloric acid produced
  nh4oh : Moles  -- Amount of ammonium hydroxide produced

/-- The reaction is balanced when the amounts of reactants and products are in the correct proportion -/
def is_balanced (r : Reaction) : Prop :=
  r.nh4cl = r.h2o ∧ r.nh4cl = r.hcl ∧ r.nh4cl = r.nh4oh

/-- The amount of water required is equal to the amount of ammonium chloride when the reaction is balanced -/
theorem water_requirement (r : Reaction) (h : is_balanced r) : r.h2o = r.nh4cl := by
  sorry

end water_requirement_l3122_312245


namespace digit_equation_sum_l3122_312294

/-- Represents a base-10 digit -/
def Digit := Fin 10

/-- Checks if all digits in a natural number are the same -/
def allDigitsSame (n : ℕ) : Prop :=
  ∃ d : Digit, n = d.val * 100 + d.val * 10 + d.val

/-- The main theorem -/
theorem digit_equation_sum :
  ∀ (Y E M L : Digit),
    Y ≠ E → Y ≠ M → Y ≠ L → E ≠ M → E ≠ L → M ≠ L →
    (Y.val * 10 + E.val) * (M.val * 10 + E.val) = L.val * 100 + L.val * 10 + L.val →
    E.val + M.val + L.val + Y.val = 15 := by
  sorry


end digit_equation_sum_l3122_312294


namespace last_locker_opened_l3122_312256

/-- Represents the process of opening lockers in a hall -/
def openLockers (n : ℕ) : ℕ :=
  n - 2

/-- Theorem stating that the last locker opened is number 727 -/
theorem last_locker_opened (total_lockers : ℕ) (h : total_lockers = 729) :
  openLockers total_lockers = 727 :=
by sorry

end last_locker_opened_l3122_312256


namespace justin_reading_problem_l3122_312225

/-- Justin's reading problem -/
theorem justin_reading_problem (pages_first_day : ℕ) (remaining_days : ℕ) :
  pages_first_day = 10 →
  remaining_days = 6 →
  pages_first_day + remaining_days * (2 * pages_first_day) = 130 :=
by
  sorry

end justin_reading_problem_l3122_312225


namespace midpoint_property_l3122_312247

/-- Given two points A and B in a 2D plane, if C is their midpoint,
    then 3 times the x-coordinate of C minus 2 times the y-coordinate of C equals -3. -/
theorem midpoint_property (A B C : ℝ × ℝ) :
  A = (-6, 9) →
  B = (8, -3) →
  C.1 = (A.1 + B.1) / 2 →
  C.2 = (A.2 + B.2) / 2 →
  3 * C.1 - 2 * C.2 = -3 := by
  sorry

end midpoint_property_l3122_312247


namespace unique_coin_configuration_l3122_312297

/-- Represents the different types of coins -/
inductive CoinType
  | Penny
  | Nickel
  | Dime

/-- The value of each coin type in cents -/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10

/-- A configuration of coins -/
structure CoinConfiguration where
  pennies : Nat
  nickels : Nat
  dimes : Nat

/-- The total number of coins in a configuration -/
def CoinConfiguration.totalCoins (c : CoinConfiguration) : Nat :=
  c.pennies + c.nickels + c.dimes

/-- The total value of coins in a configuration in cents -/
def CoinConfiguration.totalValue (c : CoinConfiguration) : Nat :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime

/-- Theorem: There is a unique coin configuration with 8 coins, 53 cents total value,
    and at least one of each coin type, which must have exactly 3 nickels -/
theorem unique_coin_configuration :
  ∃! c : CoinConfiguration,
    c.totalCoins = 8 ∧
    c.totalValue = 53 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.nickels = 3 := by
  sorry


end unique_coin_configuration_l3122_312297


namespace f_sum_symmetric_l3122_312274

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 2

-- State the theorem
theorem f_sum_symmetric (a b m : ℝ) : 
  f a b (-2) = m → f a b 2 + f a b (-2) = -4 := by
sorry

end f_sum_symmetric_l3122_312274


namespace certain_number_problem_l3122_312234

theorem certain_number_problem : ∃! x : ℝ, x / 9 + x + 9 = 69 := by sorry

end certain_number_problem_l3122_312234


namespace divisibility_of_consecutive_numbers_l3122_312285

theorem divisibility_of_consecutive_numbers (n : ℕ) 
  (h1 : ∀ p : ℕ, Prime p → p ∣ n → p^2 ∣ n)
  (h2 : ∀ p : ℕ, Prime p → p ∣ (n + 1) → p^2 ∣ (n + 1))
  (h3 : ∀ p : ℕ, Prime p → p ∣ (n + 2) → p^2 ∣ (n + 2)) :
  ∃ p : ℕ, Prime p ∧ p^3 ∣ n :=
sorry

end divisibility_of_consecutive_numbers_l3122_312285


namespace child_ticket_cost_l3122_312202

theorem child_ticket_cost (adult_price : ℕ) (total_people : ℕ) (total_revenue : ℕ) (num_children : ℕ) :
  adult_price = 11 →
  total_people = 23 →
  total_revenue = 246 →
  num_children = 7 →
  ∃ (child_price : ℕ), child_price = 10 ∧ 
    adult_price * (total_people - num_children) + child_price * num_children = total_revenue :=
by sorry

end child_ticket_cost_l3122_312202


namespace bobs_number_l3122_312218

theorem bobs_number (alice bob : ℂ) : 
  alice * bob = 48 - 16 * I ∧ alice = 7 + 4 * I → 
  bob = 272/65 - 304/65 * I := by
sorry

end bobs_number_l3122_312218


namespace simple_random_sampling_probability_l3122_312237

theorem simple_random_sampling_probability 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 8)
  (h2 : sample_size = 4) :
  (sample_size : ℚ) / population_size = 1 / 2 := by
  sorry

end simple_random_sampling_probability_l3122_312237


namespace fraction_subtraction_proof_l3122_312213

theorem fraction_subtraction_proof :
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end fraction_subtraction_proof_l3122_312213


namespace rational_operations_l3122_312284

-- Define the new operation
def star (x y : ℚ) : ℚ := x + y - x * y

-- Theorem statement
theorem rational_operations :
  -- Unit elements
  (∀ a : ℚ, a + 0 = a) ∧
  (∀ a : ℚ, a * 1 = a) ∧
  -- Inverse element of 3 under addition
  (3 + (-3) = 0) ∧
  -- 0 has no multiplicative inverse
  (∀ x : ℚ, x ≠ 0 → ∃ y : ℚ, x * y = 1) ∧
  -- Properties of the new operation
  (∀ x : ℚ, star x 0 = x) ∧
  (∀ m : ℚ, m ≠ 1 → star m (m / (m - 1)) = 0) :=
by sorry

end rational_operations_l3122_312284


namespace greater_eighteen_league_games_l3122_312243

/-- Represents a hockey league with the given specifications -/
structure HockeyLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of games in the hockey league -/
def total_games (league : HockeyLeague) : Nat :=
  let total_teams := league.divisions * league.teams_per_division
  let games_per_team := (league.teams_per_division - 1) * league.intra_division_games + 
                        (total_teams - league.teams_per_division) * league.inter_division_games
  total_teams * games_per_team / 2

/-- Theorem stating that the total number of games in the specified league is 351 -/
theorem greater_eighteen_league_games : 
  total_games { divisions := 3
              , teams_per_division := 6
              , intra_division_games := 3
              , inter_division_games := 2 } = 351 := by
  sorry

end greater_eighteen_league_games_l3122_312243


namespace zookeeper_excess_fish_l3122_312236

/-- The number of penguins in the zoo -/
def total_penguins : ℕ := 48

/-- The ratio of Emperor to Adelie penguins -/
def emperor_ratio : ℕ := 3
def adelie_ratio : ℕ := 5

/-- The amount of fish needed for each type of penguin -/
def emperor_fish_need : ℚ := 3/2
def adelie_fish_need : ℕ := 2

/-- The percentage of additional fish the zookeeper has -/
def additional_fish_percentage : ℕ := 150

theorem zookeeper_excess_fish :
  let emperor_count : ℕ := (emperor_ratio * total_penguins) / (emperor_ratio + adelie_ratio)
  let adelie_count : ℕ := (adelie_ratio * total_penguins) / (emperor_ratio + adelie_ratio)
  let total_fish_needed : ℚ := emperor_count * emperor_fish_need + adelie_count * adelie_fish_need
  let zookeeper_fish : ℕ := total_penguins + (additional_fish_percentage * total_penguins) / 100
  (zookeeper_fish : ℚ) - total_fish_needed = 33 := by
  sorry

end zookeeper_excess_fish_l3122_312236


namespace total_vitamins_in_box_vitamins_in_half_bag_l3122_312254

-- Define the number of bags in a box
def bags_per_box : ℕ := 9

-- Define the grams of vitamins per bag
def vitamins_per_bag : ℝ := 0.2

-- Theorem for total vitamins in a box
theorem total_vitamins_in_box :
  bags_per_box * vitamins_per_bag = 1.8 := by sorry

-- Theorem for vitamins in half a bag
theorem vitamins_in_half_bag :
  vitamins_per_bag / 2 = 0.1 := by sorry

end total_vitamins_in_box_vitamins_in_half_bag_l3122_312254


namespace model_airplane_competition_l3122_312262

/-- Represents a model airplane -/
structure ModelAirplane where
  speed : ℝ
  flightTime : ℝ

/-- Theorem about model airplane competition -/
theorem model_airplane_competition 
  (m h c : ℝ) 
  (model1 model2 : ModelAirplane) 
  (h_positive : h > 0)
  (m_positive : m > 0)
  (c_positive : c > 0)
  (time_diff : model2.flightTime = model1.flightTime + m)
  (headwind_distance : 
    (model1.speed - c) * model1.flightTime = 
    (model2.speed - c) * model2.flightTime + h) :
  (h > c * m → 
    model1.speed * model1.flightTime > 
    model2.speed * model2.flightTime) ∧
  (h < c * m → 
    model1.speed * model1.flightTime < 
    model2.speed * model2.flightTime) ∧
  (h = c * m → 
    model1.speed * model1.flightTime = 
    model2.speed * model2.flightTime) := by
  sorry

end model_airplane_competition_l3122_312262


namespace total_vehicles_is_2800_l3122_312230

/-- Calculates the total number of vehicles on a road with given conditions -/
def totalVehicles (lanes : ℕ) (trucksPerLane : ℕ) (busesPerLane : ℕ) : ℕ :=
  let totalTrucks := lanes * trucksPerLane
  let carsPerLane := 2 * totalTrucks
  let totalCars := lanes * carsPerLane
  let totalBuses := lanes * busesPerLane
  let motorcyclesPerLane := 3 * busesPerLane
  let totalMotorcycles := lanes * motorcyclesPerLane
  totalTrucks + totalCars + totalBuses + totalMotorcycles

/-- Theorem stating that under the given conditions, the total number of vehicles is 2800 -/
theorem total_vehicles_is_2800 : totalVehicles 4 60 40 = 2800 := by
  sorry

#eval totalVehicles 4 60 40

end total_vehicles_is_2800_l3122_312230


namespace equidistant_points_on_line_in_quadrants_I_and_IV_l3122_312283

/-- The line equation 4x + 3y = 12 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

/-- A point (x, y) is equidistant from coordinate axes if |x| = |y| -/
def equidistant_from_axes (x y : ℝ) : Prop := abs x = abs y

/-- Quadrant I: x > 0 and y > 0 -/
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Quadrant IV: x > 0 and y < 0 -/
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The main theorem: points on the line 4x + 3y = 12 that are equidistant 
    from coordinate axes exist only in quadrants I and IV -/
theorem equidistant_points_on_line_in_quadrants_I_and_IV :
  ∀ x y : ℝ, line_equation x y → equidistant_from_axes x y →
  (in_quadrant_I x y ∨ in_quadrant_IV x y) ∧
  ¬(∃ x y : ℝ, line_equation x y ∧ equidistant_from_axes x y ∧ 
    ¬(in_quadrant_I x y ∨ in_quadrant_IV x y)) :=
sorry

end equidistant_points_on_line_in_quadrants_I_and_IV_l3122_312283


namespace distance_squared_is_53_l3122_312241

/-- A notched circle with specific measurements -/
structure NotchedCircle where
  radius : ℝ
  AB : ℝ
  BC : ℝ
  right_angle : Bool

/-- The square of the distance from point B to the center of the circle -/
def distance_squared (nc : NotchedCircle) : ℝ :=
  sorry

/-- Theorem stating the square of the distance from B to the center is 53 -/
theorem distance_squared_is_53 (nc : NotchedCircle) 
  (h1 : nc.radius = Real.sqrt 72)
  (h2 : nc.AB = 8)
  (h3 : nc.BC = 3)
  (h4 : nc.right_angle = true) :
  distance_squared nc = 53 :=
sorry

end distance_squared_is_53_l3122_312241


namespace garden_perimeter_is_700_l3122_312255

/-- The perimeter of a rectangular garden with given length and breadth -/
def garden_perimeter (length : ℝ) (breadth : ℝ) : ℝ :=
  2 * (length + breadth)

theorem garden_perimeter_is_700 :
  garden_perimeter 250 100 = 700 := by
  sorry

end garden_perimeter_is_700_l3122_312255


namespace husband_catch_up_time_and_distance_l3122_312265

-- Define the problem parameters
def yolanda_initial_speed : ℝ := 20
def yolanda_second_speed : ℝ := 22
def yolanda_final_speed : ℝ := 18
def yolanda_first_distance : ℝ := 5
def yolanda_second_distance : ℝ := 8
def yolanda_third_distance : ℝ := 7
def yolanda_stop_time : ℝ := 12
def husband_speed : ℝ := 40
def husband_delay : ℝ := 15
def route_difference : ℝ := 10

-- Define the theorem
theorem husband_catch_up_time_and_distance : 
  let yolanda_total_distance := yolanda_first_distance + yolanda_second_distance + yolanda_third_distance
  let husband_distance := yolanda_total_distance - route_difference
  let yolanda_travel_time := yolanda_first_distance / yolanda_initial_speed * 60 + 
                             yolanda_second_distance / yolanda_second_speed * 60 + 
                             yolanda_third_distance / yolanda_final_speed * 60 + 
                             yolanda_stop_time
  let husband_travel_time := husband_distance / husband_speed * 60
  husband_distance = 10 ∧ husband_travel_time + husband_delay = 30 := by
    sorry


end husband_catch_up_time_and_distance_l3122_312265


namespace min_value_x_plus_y_l3122_312250

theorem min_value_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) : 
  x + y ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x + 1) * (y + 1) = 9 ∧ x + y = 4 :=
by sorry

end min_value_x_plus_y_l3122_312250


namespace quadratic_coefficient_l3122_312231

/-- Given a quadratic of the form x^2 + bx + 50 where b is positive,
    if it can be written as (x+n)^2 + 8, then b = 2√42. -/
theorem quadratic_coefficient (b n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 50 = (x+n)^2 + 8) → 
  b = 2 * Real.sqrt 42 := by
sorry

end quadratic_coefficient_l3122_312231


namespace valid_numbers_l3122_312263

def is_valid_number (n : ℕ) : Prop :=
  500 < n ∧ n < 2500 ∧ n % 180 = 0 ∧ n % 75 = 0

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 900 ∨ n = 1800 :=
sorry

end valid_numbers_l3122_312263


namespace ring_arrangements_count_l3122_312206

/-- The number of ways to distribute n distinct objects into m distinct containers,
    where each container can hold multiple objects. -/
def distribute (n m : ℕ) : ℕ := m^n

/-- The number of distinct arrangements for wearing 5 different rings
    on the 5 fingers of the right hand. -/
def ringArrangements : ℕ := distribute 5 5

theorem ring_arrangements_count :
  ringArrangements = 5^5 := by sorry

end ring_arrangements_count_l3122_312206


namespace tims_trip_duration_l3122_312201

/-- Calculates the total duration of Tim's trip given the specified conditions -/
theorem tims_trip_duration :
  let total_driving_time : ℝ := 5
  let num_traffic_jams : ℕ := 3
  let first_jam_multiplier : ℝ := 1.5
  let second_jam_multiplier : ℝ := 2
  let third_jam_multiplier : ℝ := 3
  let num_pit_stops : ℕ := 2
  let pit_stop_duration : ℝ := 0.5
  let time_before_first_jam : ℝ := 1
  let time_between_first_and_second_jam : ℝ := 1.5

  let first_jam_duration : ℝ := first_jam_multiplier * time_before_first_jam
  let second_jam_duration : ℝ := second_jam_multiplier * time_between_first_and_second_jam
  let time_before_third_jam : ℝ := total_driving_time - time_before_first_jam - time_between_first_and_second_jam
  let third_jam_duration : ℝ := third_jam_multiplier * time_before_third_jam
  let total_pit_stop_time : ℝ := num_pit_stops * pit_stop_duration

  let total_duration : ℝ := total_driving_time + first_jam_duration + second_jam_duration + third_jam_duration + total_pit_stop_time

  total_duration = 18 := by sorry

end tims_trip_duration_l3122_312201


namespace symmetrical_circles_sum_l3122_312207

/-- Two circles are symmetrical with respect to the line y = x + 1 -/
def symmetrical_circles (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - a)^2 + (y - b)^2 = 1 ∧ 
                (x - 1)^2 + (y - 3)^2 = 1 ∧
                y = x + 1

/-- If two circles are symmetrical with respect to the line y = x + 1,
    then the sum of their center coordinates is 2 -/
theorem symmetrical_circles_sum (a b : ℝ) :
  symmetrical_circles a b → a + b = 2 :=
by
  sorry

end symmetrical_circles_sum_l3122_312207


namespace parallel_planes_condition_l3122_312288

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations and relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Set Point)
variable (planes_parallel : Plane → Plane → Prop)

-- Define the given lines and planes
variable (m n l₁ l₂ : Line)
variable (α β : Plane)
variable (M : Point)

theorem parallel_planes_condition
  (h1 : subset m α)
  (h2 : subset n α)
  (h3 : subset l₁ β)
  (h4 : subset l₂ β)
  (h5 : intersect l₁ l₂ = {M})
  (h6 : parallel m l₁)
  (h7 : parallel n l₂) :
  planes_parallel α β :=
sorry

end parallel_planes_condition_l3122_312288


namespace difference_of_products_equals_one_l3122_312246

theorem difference_of_products_equals_one : (1011 : ℕ) * 1011 - 1010 * 1012 = 1 := by
  sorry

end difference_of_products_equals_one_l3122_312246


namespace three_digit_not_mult_4_or_6_eq_600_l3122_312275

/-- The number of three-digit numbers that are multiples of neither 4 nor 6 -/
def three_digit_not_mult_4_or_6 : ℕ :=
  let three_digit_count := 999 - 100 + 1
  let mult_4_count := (996 / 4) - (100 / 4) + 1
  let mult_6_count := (996 / 6) - (102 / 6) + 1
  let mult_12_count := (996 / 12) - (108 / 12) + 1
  three_digit_count - (mult_4_count + mult_6_count - mult_12_count)

theorem three_digit_not_mult_4_or_6_eq_600 :
  three_digit_not_mult_4_or_6 = 600 := by
  sorry

end three_digit_not_mult_4_or_6_eq_600_l3122_312275


namespace invalid_votes_percentage_l3122_312259

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 70 / 100)
  (h3 : candidate_a_votes = 333200) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 :=
by sorry

end invalid_votes_percentage_l3122_312259


namespace impossible_all_positive_l3122_312268

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → Int

/-- The initial grid configuration -/
def initial_grid : Grid :=
  fun i j => if i = 2 ∧ j = 3 then -1 else 1

/-- Represents an operation on the grid -/
inductive Operation
  | row (i : Fin 4)
  | col (j : Fin 4)
  | diag (d : Fin 7)

/-- Applies an operation to a grid -/
def apply_operation (g : Grid) (op : Operation) : Grid :=
  match op with
  | Operation.row i => fun x y => if x = i then -g x y else g x y
  | Operation.col j => fun x y => if y = j then -g x y else g x y
  | Operation.diag d => fun x y => if x + y = d then -g x y else g x y

/-- Applies a sequence of operations to a grid -/
def apply_operations (g : Grid) (ops : List Operation) : Grid :=
  ops.foldl apply_operation g

/-- Predicate to check if all cells in a grid are positive -/
def all_positive (g : Grid) : Prop :=
  ∀ i j, g i j > 0

/-- The main theorem -/
theorem impossible_all_positive (ops : List Operation) :
  ¬(all_positive (apply_operations initial_grid ops)) :=
sorry

end impossible_all_positive_l3122_312268


namespace number_999_in_column_C_l3122_312291

/-- Represents the columns in which numbers are arranged --/
inductive Column
  | A | B | C | D | E | F | G

/-- Determines the column for a given positive integer greater than 1 --/
def column_for_number (n : ℕ) : Column :=
  sorry

/-- The main theorem stating that 999 is in column C --/
theorem number_999_in_column_C : column_for_number 999 = Column.C := by
  sorry

end number_999_in_column_C_l3122_312291


namespace compare_data_fluctuation_l3122_312267

def group_mean (g : String) : ℝ :=
  match g with
  | "A" => 80
  | "B" => 90
  | _ => 0

def group_variance (g : String) : ℝ :=
  match g with
  | "A" => 10
  | "B" => 5
  | _ => 0

def less_fluctuation (g1 g2 : String) : Prop :=
  group_variance g1 < group_variance g2

theorem compare_data_fluctuation (g1 g2 : String) :
  less_fluctuation g1 g2 → group_variance g1 < group_variance g2 :=
by sorry

end compare_data_fluctuation_l3122_312267


namespace number_of_students_l3122_312299

theorem number_of_students (possible_outcomes : ℕ) (total_results : ℕ) : 
  possible_outcomes = 3 → total_results = 59049 → 
  ∃ n : ℕ, possible_outcomes ^ n = total_results ∧ n = 10 :=
by sorry

end number_of_students_l3122_312299


namespace origin_fixed_under_dilation_l3122_312232

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square defined by its four vertices -/
structure Square where
  s : Point
  t : Point
  u : Point
  v : Point

/-- Defines a dilation transformation -/
def dilation (center : Point) (k : ℝ) (p : Point) : Point :=
  { x := center.x + k * (p.x - center.x)
  , y := center.y + k * (p.y - center.y) }

theorem origin_fixed_under_dilation (original : Square) (dilated : Square) :
  original.s = Point.mk 3 3 ∧
  original.t = Point.mk 7 3 ∧
  original.u = Point.mk 7 7 ∧
  original.v = Point.mk 3 7 ∧
  dilated.s = Point.mk 6 6 ∧
  dilated.t = Point.mk 12 6 ∧
  dilated.u = Point.mk 12 12 ∧
  dilated.v = Point.mk 6 12 →
  ∃ (k : ℝ), ∀ (p : Point),
    dilation (Point.mk 0 0) k original.s = dilated.s ∧
    dilation (Point.mk 0 0) k original.t = dilated.t ∧
    dilation (Point.mk 0 0) k original.u = dilated.u ∧
    dilation (Point.mk 0 0) k original.v = dilated.v :=
by sorry

end origin_fixed_under_dilation_l3122_312232


namespace initial_cookies_l3122_312212

/-- The number of cookies remaining after the first day -/
def remaining_after_first_day (C : ℚ) : ℚ :=
  C * (1/4) * (4/5)

/-- The number of cookies remaining after the second day -/
def remaining_after_second_day (C : ℚ) : ℚ :=
  remaining_after_first_day C * (1/2)

/-- Theorem stating the initial number of cookies -/
theorem initial_cookies : ∃ C : ℚ, C > 0 ∧ remaining_after_second_day C = 10 := by
  sorry

end initial_cookies_l3122_312212


namespace solution_satisfies_inequalities_inequalities_imply_solution_solution_is_correct_l3122_312277

-- Define the system of inequalities
def inequality1 (x : ℝ) : Prop := x + 2 < 3 + 2*x
def inequality2 (x : ℝ) : Prop := 4*x - 3 < 3*x - 1
def inequality3 (x : ℝ) : Prop := 8 + 5*x ≥ 6*x + 7

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

-- Theorem stating that the solution set satisfies all inequalities
theorem solution_satisfies_inequalities :
  ∀ x ∈ solution_set, inequality1 x ∧ inequality2 x ∧ inequality3 x :=
sorry

-- Theorem stating that any real number satisfying all inequalities is in the solution set
theorem inequalities_imply_solution :
  ∀ x : ℝ, inequality1 x ∧ inequality2 x ∧ inequality3 x → x ∈ solution_set :=
sorry

-- Main theorem: The solution set is exactly (-1, 1]
theorem solution_is_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality1 x ∧ inequality2 x ∧ inequality3 x :=
sorry

end solution_satisfies_inequalities_inequalities_imply_solution_solution_is_correct_l3122_312277


namespace prob_not_both_white_l3122_312228

theorem prob_not_both_white (prob_white_A prob_white_B : ℚ) 
  (h1 : prob_white_A = 1/3)
  (h2 : prob_white_B = 1/2) :
  1 - prob_white_A * prob_white_B = 5/6 := by
  sorry

end prob_not_both_white_l3122_312228


namespace obtain_11_from_1_l3122_312221

/-- Represents the allowed operations on the calculator -/
inductive Operation
  | Multiply3
  | Add3
  | Divide3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Multiply3 => n * 3
  | Operation.Add3 => n + 3
  | Operation.Divide3 => if n % 3 = 0 then n / 3 else n

/-- Applies a sequence of operations to a number -/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Theorem: It's possible to obtain 11 from 1 using the given calculator operations -/
theorem obtain_11_from_1 : ∃ (ops : List Operation), applyOperations 1 ops = 11 := by
  sorry

end obtain_11_from_1_l3122_312221


namespace original_denominator_proof_l3122_312276

theorem original_denominator_proof (d : ℚ) : 
  (3 : ℚ) / d ≠ (1 : ℚ) / 3 ∧ (3 + 7 : ℚ) / (d + 7) = (1 : ℚ) / 3 → d = 23 := by
  sorry

end original_denominator_proof_l3122_312276


namespace subscription_difference_is_5000_l3122_312253

/-- Represents the subscription amounts and profit distribution in a business venture -/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  a_profit : ℕ
  a_extra : ℕ

/-- Calculates the difference between B's and C's subscriptions -/
def subscription_difference (bv : BusinessVenture) : ℕ :=
  let b_subscription := (bv.total_subscription * bv.a_profit * 2) / (bv.total_profit * 3) - bv.a_extra
  let c_subscription := bv.total_subscription - b_subscription - (b_subscription + bv.a_extra)
  b_subscription - c_subscription

/-- Theorem stating that the difference between B's and C's subscriptions is 5000 -/
theorem subscription_difference_is_5000 (bv : BusinessVenture) 
    (h1 : bv.total_subscription = 50000)
    (h2 : bv.total_profit = 35000)
    (h3 : bv.a_profit = 14700)
    (h4 : bv.a_extra = 4000) :
  subscription_difference bv = 5000 := by
  sorry

#eval subscription_difference ⟨50000, 35000, 14700, 4000⟩

end subscription_difference_is_5000_l3122_312253


namespace correct_travel_times_l3122_312205

/-- Represents the travel times of Winnie-the-Pooh and Piglet -/
structure TravelTimes where
  pooh : ℝ
  piglet : ℝ

/-- Calculates the travel times based on the given conditions -/
def calculate_travel_times (time_after_meeting_pooh : ℝ) (time_after_meeting_piglet : ℝ) : TravelTimes :=
  let speed_ratio := time_after_meeting_pooh / time_after_meeting_piglet
  let time_before_meeting := time_after_meeting_piglet
  { pooh := time_before_meeting + time_after_meeting_piglet
  , piglet := time_before_meeting + time_after_meeting_pooh }

/-- Theorem stating that the calculated travel times are correct -/
theorem correct_travel_times :
  let result := calculate_travel_times 4 1
  result.pooh = 2 ∧ result.piglet = 6 := by sorry

end correct_travel_times_l3122_312205


namespace polar_to_cartesian_conversion_l3122_312282

theorem polar_to_cartesian_conversion :
  let r : ℝ := 2
  let θ : ℝ := π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by sorry

end polar_to_cartesian_conversion_l3122_312282


namespace season_games_count_l3122_312289

/-- Represents a sports league with the given structure -/
structure SportsLeague where
  num_divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of games in a complete season -/
def total_games (league : SportsLeague) : Nat :=
  let total_teams := league.num_divisions * league.teams_per_division
  let intra_division_total := league.num_divisions * (league.teams_per_division * (league.teams_per_division - 1) / 2) * league.intra_division_games
  let inter_division_total := (total_teams * (total_teams - league.teams_per_division) / 2) * league.inter_division_games
  intra_division_total + inter_division_total

/-- The theorem to be proved -/
theorem season_games_count : 
  let league := SportsLeague.mk 3 6 3 2
  total_games league = 351 := by
  sorry

end season_games_count_l3122_312289


namespace multiplication_addition_equality_l3122_312210

theorem multiplication_addition_equality : 21 * 47 + 21 * 53 = 2100 := by
  sorry

end multiplication_addition_equality_l3122_312210


namespace tom_trade_amount_l3122_312258

/-- The amount Tom initially gave to trade his Super Nintendo for an NES -/
def trade_amount (super_nintendo_value : ℝ) (credit_percentage : ℝ) 
  (nes_price : ℝ) (game_value : ℝ) (change : ℝ) : ℝ :=
  let credit := super_nintendo_value * credit_percentage
  let remaining := nes_price - credit
  let total_needed := remaining + game_value
  total_needed + change

/-- Theorem stating the amount Tom initially gave -/
theorem tom_trade_amount : 
  trade_amount 150 0.8 160 30 10 = 80 := by
  sorry

end tom_trade_amount_l3122_312258


namespace tangent_line_equation_l3122_312298

-- Define the parabola function
def f (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the line parallel to 4x - y + 3 = 0
def m : ℝ := 4

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the parabola
    y₀ = f x₀ ∧
    -- The slope of the tangent line at (x₀, y₀) is m
    (deriv f) x₀ = m ∧
    -- The equation of the tangent line is 4x - y - 2 = 0
    ∀ (x y : ℝ), y - y₀ = m * (x - x₀) ↔ 4 * x - y - 2 = 0 :=
sorry

end tangent_line_equation_l3122_312298


namespace function_positive_l3122_312200

/-- Given a function f: ℝ → ℝ with derivative f', 
    if 2f(x) + xf'(x) > x² for all x ∈ ℝ, 
    then f(x) > 0 for all x ∈ ℝ. -/
theorem function_positive 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, 2 * f x + x * deriv f x > x^2) : 
  ∀ x, f x > 0 := by
  sorry

end function_positive_l3122_312200


namespace square_of_negative_two_i_l3122_312235

theorem square_of_negative_two_i (i : ℂ) (hi : i^2 = -1) : (-2 * i)^2 = -4 := by
  sorry

end square_of_negative_two_i_l3122_312235


namespace shaded_area_theorem_l3122_312242

/-- Represents a rectangle on a grid --/
structure Rectangle where
  width : ℕ
  height : ℕ
  is_not_square : width ≠ height

/-- Represents the configuration of rectangles in the problem --/
structure Configuration where
  abcd : Rectangle
  qrsc : Rectangle
  ap : ℕ
  qr : ℕ
  bp : ℕ
  br : ℕ
  sc : ℕ

/-- The main theorem statement --/
theorem shaded_area_theorem (config : Configuration) :
  config.abcd.width * config.abcd.height = 35 →
  config.ap < config.qr →
  (config.abcd.width * config.abcd.height - 
   (config.bp * config.br + config.ap * config.sc) = 24) ∨
  (config.abcd.width * config.abcd.height - 
   (config.bp * config.br + config.ap * config.sc) = 26) := by
  sorry

end shaded_area_theorem_l3122_312242


namespace fraction_of_fraction_of_fraction_problem_solution_l3122_312217

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem problem_solution : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 180 = 5 := by sorry

end fraction_of_fraction_of_fraction_problem_solution_l3122_312217


namespace slope_of_line_parallel_lines_solution_l3122_312266

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
theorem slope_of_line (a b c : ℝ) (hb : b ≠ 0) :
  ∀ x y, a * x + b * y + c = 0 ↔ y = (-a/b) * x + (-c/b) :=
  sorry

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y, a * x + y - 4 = 0 ↔ x + (a + 3/2) * y + 2 = 0) → a = 1/2 :=
  sorry

end slope_of_line_parallel_lines_solution_l3122_312266


namespace sum_product_bounds_l3122_312244

theorem sum_product_bounds (a b c d : ℝ) (h : a + b + c + d = 1) :
  0 ≤ a * b + a * c + a * d + b * c + b * d + c * d ∧
  a * b + a * c + a * d + b * c + b * d + c * d ≤ 3 / 8 := by
  sorry

end sum_product_bounds_l3122_312244


namespace sum_difference_multiples_l3122_312264

theorem sum_difference_multiples (m n : ℕ+) : 
  (∃ x : ℕ+, m = 101 * x) → 
  (∃ y : ℕ+, n = 63 * y) → 
  m + n = 2018 → 
  m - n = 2 := by
sorry

end sum_difference_multiples_l3122_312264


namespace zip_code_sum_l3122_312224

theorem zip_code_sum (a b c d e : ℕ) : 
  a + b + c + d + e = 10 →
  a = b →
  c = 0 →
  d = 2 * a →
  d + e = 8 := by
sorry

end zip_code_sum_l3122_312224


namespace reflection_of_C_l3122_312226

/-- Reflects a point over the line y = x -/
def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem reflection_of_C : 
  let C : ℝ × ℝ := (2, 2)
  reflect_over_y_eq_x C = C :=
by sorry

end reflection_of_C_l3122_312226


namespace divide_plot_with_fences_l3122_312278

/-- Represents a rectangular plot with length and width -/
structure Plot where
  length : ℝ
  width : ℝ

/-- Represents a fence with a length -/
structure Fence where
  length : ℝ

/-- Represents a section of the plot -/
structure Section where
  area : ℝ

theorem divide_plot_with_fences (p : Plot) (f : Fence) :
  p.length = 80 →
  p.width = 50 →
  ∃ (sections : Finset Section),
    sections.card = 5 ∧
    (∀ s ∈ sections, s.area = (p.length * p.width) / 5) ∧
    f.length = 40 := by
  sorry

end divide_plot_with_fences_l3122_312278


namespace max_remainder_eleven_l3122_312271

theorem max_remainder_eleven (A B C : ℕ) (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h2 : A = 11 * B + C) : C ≤ 10 :=
sorry

end max_remainder_eleven_l3122_312271


namespace cube_sum_equality_l3122_312290

theorem cube_sum_equality (a b : ℝ) (h : a + b = 4) : a^3 + 12*a*b + b^3 = 64 := by
  sorry

end cube_sum_equality_l3122_312290


namespace solution_to_inequality_l3122_312222

theorem solution_to_inequality : 1 - 1 ≥ 0 := by sorry

end solution_to_inequality_l3122_312222


namespace chemical_mixture_problem_l3122_312233

/-- Represents the composition of a chemical solution -/
structure Solution where
  a : ℝ  -- Percentage of chemical a
  b : ℝ  -- Percentage of chemical b

/-- Represents a mixture of two solutions -/
structure Mixture where
  x : Solution  -- First solution
  y : Solution  -- Second solution
  x_ratio : ℝ   -- Ratio of solution x in the mixture

/-- The problem statement -/
theorem chemical_mixture_problem (x y : Solution) (m : Mixture) :
  x.a = 0.1 ∧                 -- Solution x is 10% chemical a
  x.b = 0.9 ∧                 -- Solution x is 90% chemical b
  y.b = 0.8 ∧                 -- Solution y is 80% chemical b
  m.x = x ∧                   -- Mixture contains solution x
  m.y = y ∧                   -- Mixture contains solution y
  m.x_ratio = 0.8 ∧           -- 80% of the mixture is solution x
  m.x_ratio * x.a + (1 - m.x_ratio) * y.a = 0.12  -- Mixture is 12% chemical a
  →
  y.a = 0.2                   -- Percentage of chemical a in solution y is 20%
  := by sorry

end chemical_mixture_problem_l3122_312233


namespace class_composition_l3122_312286

theorem class_composition (total_students : ℕ) (total_planes : ℕ) (girls_planes : ℕ) (boys_planes : ℕ) 
  (h1 : total_students = 21)
  (h2 : total_planes = 69)
  (h3 : girls_planes = 2)
  (h4 : boys_planes = 5) :
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧
    boys * boys_planes + girls * girls_planes = total_planes ∧
    boys = 9 ∧
    girls = 12 := by
  sorry

end class_composition_l3122_312286


namespace bhupathi_amount_l3122_312292

theorem bhupathi_amount (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A + B = 1210) (h4 : (4/15) * A = (2/5) * B) : B = 484 := by
  sorry

end bhupathi_amount_l3122_312292


namespace john_tax_difference_l3122_312257

/-- Calculates the difference in taxes paid given old and new tax rates and incomes -/
def tax_difference (old_rate new_rate old_income new_income : ℚ) : ℚ :=
  new_rate * new_income - old_rate * old_income

/-- Proves that the difference in taxes paid by John is $250,000 -/
theorem john_tax_difference :
  tax_difference (20 / 100) (30 / 100) 1000000 1500000 = 250000 := by
  sorry

end john_tax_difference_l3122_312257


namespace square_binomial_coefficient_l3122_312281

theorem square_binomial_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 20 * x + 16 = (r * x + s)^2) → 
  a = 25 / 4 :=
by sorry

end square_binomial_coefficient_l3122_312281


namespace triangle_inequality_l3122_312203

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) := by
  sorry

end triangle_inequality_l3122_312203


namespace experimental_plans_count_l3122_312223

/-- The number of ways to select an even number of elements from a set of 6 elements -/
def evenSelectionA : ℕ := Finset.sum (Finset.filter (λ k => k % 2 = 0) (Finset.range 7)) (λ k => Nat.choose 6 k)

/-- The number of ways to select at least 2 elements from a set of 4 elements -/
def atLeastTwoSelectionB : ℕ := Finset.sum (Finset.range 3) (λ k => Nat.choose 4 (k + 2))

/-- The total number of experimental plans -/
def totalExperimentalPlans : ℕ := evenSelectionA * atLeastTwoSelectionB

theorem experimental_plans_count : totalExperimentalPlans = 352 := by
  sorry

end experimental_plans_count_l3122_312223


namespace m_greater_than_n_l3122_312272

theorem m_greater_than_n (a b m n : ℝ) 
  (ha : 0 < a) (hb : 0 < b)
  (h : m^2 * n^2 > a^2 * m^2 + b^2 * n^2) : 
  Real.sqrt (m^2 + n^2) > a + b := by
  sorry

end m_greater_than_n_l3122_312272


namespace increase_by_fraction_l3122_312295

theorem increase_by_fraction (initial : ℝ) (increase : ℚ) (result : ℝ) :
  initial = 120 →
  increase = 5/6 →
  result = initial * (1 + increase) →
  result = 220 := by
  sorry

end increase_by_fraction_l3122_312295


namespace people_in_room_l3122_312251

theorem people_in_room (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (4 : ℚ) / 5 * chairs ∧ 
  chairs - (4 : ℚ) / 5 * chairs = 10 →
  people = 67 := by
sorry

end people_in_room_l3122_312251


namespace power_sum_zero_l3122_312248

theorem power_sum_zero : (-2 : ℤ) ^ (3^2) + 2 ^ (3^2) = 0 := by
  sorry

end power_sum_zero_l3122_312248


namespace power_of_two_subset_bound_l3122_312211

/-- The set of powers of 2 from 2^7 to 2^n -/
def S (n : ℕ) : Set ℕ := {x | ∃ k, 7 ≤ k ∧ k ≤ n ∧ x = 2^k}

/-- The subset of S where the sum of the last three digits is 8 -/
def A (n : ℕ) : Set ℕ := {x ∈ S n | (x % 1000 / 100 + x % 100 / 10 + x % 10) = 8}

/-- The number of elements in a finite set -/
def card {α : Type*} (s : Set α) : ℕ := sorry

theorem power_of_two_subset_bound (n : ℕ) (h : n ≥ 2009) :
  (28 : ℚ) / 2009 < (card (A n) : ℚ) / (card (S n)) ∧
  (card (A n) : ℚ) / (card (S n)) < 82 / 2009 := by
  sorry

end power_of_two_subset_bound_l3122_312211


namespace ceramic_firing_probabilities_l3122_312273

/-- Represents the probability of success for a craft in each firing process -/
structure CraftProbabilities where
  first : Float
  second : Float

/-- Calculates the probability of exactly one success out of three independent events -/
def probExactlyOne (p1 p2 p3 : Float) : Float :=
  p1 * (1 - p2) * (1 - p3) + (1 - p1) * p2 * (1 - p3) + (1 - p1) * (1 - p2) * p3

/-- Calculates the expected value of a binomial distribution -/
def binomialExpectedValue (n : Nat) (p : Float) : Float :=
  n.toFloat * p

/-- Theorem about ceramic firing probabilities -/
theorem ceramic_firing_probabilities
  (craftA craftB craftC : CraftProbabilities)
  (h1 : craftA.first = 0.5)
  (h2 : craftB.first = 0.6)
  (h3 : craftC.first = 0.4)
  (h4 : craftA.second = 0.6)
  (h5 : craftB.second = 0.5)
  (h6 : craftC.second = 0.75) :
  (probExactlyOne craftA.first craftB.first craftC.first = 0.38) ∧
  (binomialExpectedValue 3 (craftA.first * craftA.second) = 0.9) := by
  sorry


end ceramic_firing_probabilities_l3122_312273


namespace star_properties_l3122_312215

/-- The star operation -/
def star (a b : ℝ) : ℝ := a + b + a * b

/-- The prime operation -/
noncomputable def prime (a : ℝ) : ℝ := -a / (a + 1)

theorem star_properties :
  ∀ (a b c : ℝ),
  (a ≠ -1) →
  (b ≠ -1) →
  (c ≠ -1) →
  (∀ (x : ℝ), star a x = c ↔ x = (c - a) / (a + 1)) ∧
  (star a (prime a) = 0) ∧
  (prime (prime a) = a) ∧
  (prime (star a b) = star (prime a) (prime b)) ∧
  (prime (star (prime a) b) = star (prime a) (prime b)) ∧
  (prime (star (prime a) (prime b)) = star a b) :=
by sorry

end star_properties_l3122_312215


namespace toms_lifting_capacity_l3122_312238

/-- Calculates the total weight Tom can lift after training -/
def totalWeightAfterTraining (initialCapacity : ℝ) : ℝ :=
  let afterIntensiveTraining := initialCapacity * (1 + 1.5)
  let afterSpecialization := afterIntensiveTraining * (1 + 0.25)
  let afterNewGripTechnique := afterSpecialization * (1 + 0.1)
  2 * afterNewGripTechnique

/-- Theorem stating that Tom's final lifting capacity is 687.5 kg -/
theorem toms_lifting_capacity :
  totalWeightAfterTraining 100 = 687.5 := by
  sorry

#eval totalWeightAfterTraining 100

end toms_lifting_capacity_l3122_312238


namespace midnight_temperature_l3122_312216

/-- Given an initial temperature, a temperature rise, and a temperature drop,
    calculate the final temperature. -/
def final_temperature (initial : Int) (rise : Int) (drop : Int) : Int :=
  initial + rise - drop

/-- Theorem stating that given the specific temperature changes in the problem,
    the final temperature is -8°C. -/
theorem midnight_temperature :
  final_temperature (-5) 5 8 = -8 := by
  sorry

end midnight_temperature_l3122_312216


namespace annual_savings_l3122_312240

/-- Given monthly income and expenses, calculate annual savings --/
theorem annual_savings (monthly_income monthly_expenses : ℕ) : 
  monthly_income = 5000 → 
  monthly_expenses = 4600 → 
  (monthly_income - monthly_expenses) * 12 = 4800 := by
  sorry

end annual_savings_l3122_312240


namespace fraction_simplification_l3122_312252

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) :
  (x^2 - 4*x + 3) / (x^2 - 7*x + 12) / ((x^2 - 4*x + 4) / (x^2 - 6*x + 9)) = 
  ((x - 1) * (x - 3)^2) / ((x - 4) * (x - 2)^2) := by
  sorry

end fraction_simplification_l3122_312252
