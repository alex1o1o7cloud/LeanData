import Mathlib

namespace hyperbola_eccentricity_ratio_l3262_326275

/-- Theorem: For a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where a > 0, b > 0,
    and eccentricity = 2, the ratio b/a equals √3. -/
theorem hyperbola_eccentricity_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c / a = 2) →
  b / a = Real.sqrt 3 := by
  sorry

end hyperbola_eccentricity_ratio_l3262_326275


namespace grapes_count_l3262_326242

/-- The number of grapes in Rob's bowl -/
def rob_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allie_grapes : ℕ := rob_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyn_grapes : ℕ := allie_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := rob_grapes + allie_grapes + allyn_grapes

theorem grapes_count : total_grapes = 83 := by
  sorry

end grapes_count_l3262_326242


namespace unique_modular_equivalence_l3262_326215

theorem unique_modular_equivalence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 14525 [ZMOD 16] ∧ n = 13 := by
  sorry

end unique_modular_equivalence_l3262_326215


namespace consecutive_integers_average_l3262_326285

theorem consecutive_integers_average (c : ℕ) (d : ℚ) : 
  (c > 0) →
  (d = (2 * c + (2 * c + 1) + (2 * c + 2) + (2 * c + 3) + (2 * c + 4) + (2 * c + 5) + (2 * c + 6)) / 7) →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = 2 * c + 6) :=
by sorry

end consecutive_integers_average_l3262_326285


namespace storm_encounter_average_time_l3262_326289

/-- Represents the position of an object in a 2D plane -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents a moving circular storm -/
structure Storm where
  center : Position
  radius : ℝ
  velocity : Position

/-- Represents a car moving in a straight line -/
structure Car where
  position : Position
  velocity : ℝ

/-- Calculates the position of an object after time t -/
def position_at_time (initial : Position) (velocity : Position) (t : ℝ) : Position :=
  { x := initial.x + velocity.x * t
  , y := initial.y + velocity.y * t }

/-- Determines if a point is inside a circle -/
def is_inside_circle (point : Position) (center : Position) (radius : ℝ) : Prop :=
  (point.x - center.x)^2 + (point.y - center.y)^2 ≤ radius^2

/-- The main theorem to be proved -/
theorem storm_encounter_average_time 
  (car : Car)
  (storm : Storm)
  (t₁ t₂ : ℝ) :
  car.position = { x := 0, y := 0 } →
  car.velocity = 3/4 →
  storm.center = { x := 0, y := 150 } →
  storm.radius = 75 →
  storm.velocity = { x := 3/4, y := -3/4 } →
  (is_inside_circle (position_at_time car.position { x := car.velocity, y := 0 } t₁) 
                    (position_at_time storm.center storm.velocity t₁) 
                    storm.radius) →
  (is_inside_circle (position_at_time car.position { x := car.velocity, y := 0 } t₂) 
                    (position_at_time storm.center storm.velocity t₂) 
                    storm.radius) →
  (∀ t, t₁ < t ∧ t < t₂ → 
    is_inside_circle (position_at_time car.position { x := car.velocity, y := 0 } t) 
                     (position_at_time storm.center storm.velocity t) 
                     storm.radius) →
  (t₁ + t₂) / 2 = 400 :=
by sorry

end storm_encounter_average_time_l3262_326289


namespace fractional_equation_solution_l3262_326204

theorem fractional_equation_solution : 
  ∃ x : ℝ, (3 / (x^2 - x) + 1 = x / (x - 1)) ∧ x = 3 := by
  sorry

end fractional_equation_solution_l3262_326204


namespace max_x_over_y_l3262_326298

theorem max_x_over_y (x y a b : ℝ) (h1 : x ≥ y) (h2 : y > 0)
  (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y)
  (h7 : (x - a)^2 + (y - b)^2 = x^2 + b^2)
  (h8 : x^2 + b^2 = y^2 + a^2) :
  ∃ (x' y' : ℝ), x' ≥ y' ∧ y' > 0 ∧
  ∃ (a' b' : ℝ), 0 ≤ a' ∧ a' ≤ x' ∧ 0 ≤ b' ∧ b' ≤ y' ∧
  (x' - a')^2 + (y' - b')^2 = x'^2 + b'^2 ∧ x'^2 + b'^2 = y'^2 + a'^2 ∧
  x' / y' = 2 * Real.sqrt 3 / 3 ∧
  ∀ (x'' y'' : ℝ), x'' ≥ y'' → y'' > 0 →
  ∃ (a'' b'' : ℝ), 0 ≤ a'' ∧ a'' ≤ x'' ∧ 0 ≤ b'' ∧ b'' ≤ y'' ∧
  (x'' - a'')^2 + (y'' - b'')^2 = x''^2 + b''^2 ∧ x''^2 + b''^2 = y''^2 + a''^2 →
  x'' / y'' ≤ 2 * Real.sqrt 3 / 3 :=
by sorry

end max_x_over_y_l3262_326298


namespace inequality_solution_set_l3262_326288

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  1 / x ≤ x ↔ (-1 ≤ x ∧ x < 0) ∨ x ≥ 1 := by
  sorry

end inequality_solution_set_l3262_326288


namespace bobby_pancakes_left_l3262_326249

/-- The number of pancakes Bobby has left after making and serving breakfast -/
def pancakes_left (standard_batch : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) (friends_ate : ℕ) : ℕ :=
  let total_made := standard_batch + 2 * standard_batch + standard_batch
  let total_eaten := bobby_ate + dog_ate + friends_ate
  total_made - total_eaten

/-- Theorem stating that Bobby has 50 pancakes left -/
theorem bobby_pancakes_left : 
  pancakes_left 21 5 7 22 = 50 := by
  sorry

end bobby_pancakes_left_l3262_326249


namespace closest_to_target_l3262_326210

def options : List ℝ := [-4, -3, 0, 3, 4]

def target : ℝ := -3.4

def distance (x y : ℝ) : ℝ := |x - y|

theorem closest_to_target :
  ∃ (closest : ℝ), closest ∈ options ∧
    (∀ x ∈ options, distance target closest ≤ distance target x) ∧
    closest = -3 := by
  sorry

end closest_to_target_l3262_326210


namespace part_one_part_two_l3262_326256

-- Part 1
theorem part_one (a b : ℤ) (h1 : a = 4) (h2 : b = 5) : a - b = -1 := by
  sorry

-- Part 2
theorem part_two (a b m n s : ℝ) 
  (h1 : a + b = 0) 
  (h2 : m * n = 1) 
  (h3 : |s| = 3) : 
  a + b + m * n + s = 4 ∨ a + b + m * n + s = -2 := by
  sorry

end part_one_part_two_l3262_326256


namespace dart_board_probability_l3262_326243

theorem dart_board_probability (r : ℝ) (h : r = 10) :
  let circle_area := π * r^2
  let square_side := r * Real.sqrt 2
  let square_area := square_side^2
  square_area / circle_area = 2 / π := by sorry

end dart_board_probability_l3262_326243


namespace flagpole_height_l3262_326260

/-- Given a 3-meter pole with a 1.2-meter shadow and a flagpole with a 4.8-meter shadow,
    the height of the flagpole is 12 meters. -/
theorem flagpole_height
  (pole_height : Real)
  (pole_shadow : Real)
  (flagpole_shadow : Real)
  (h_pole_height : pole_height = 3)
  (h_pole_shadow : pole_shadow = 1.2)
  (h_flagpole_shadow : flagpole_shadow = 4.8) :
  pole_height / pole_shadow = 12 / flagpole_shadow := by
  sorry

end flagpole_height_l3262_326260


namespace attractions_permutations_l3262_326268

theorem attractions_permutations : Nat.factorial 4 = 24 := by
  sorry

end attractions_permutations_l3262_326268


namespace red_blood_cell_diameter_scientific_notation_l3262_326253

/-- Expresses a given decimal number in scientific notation -/
def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem red_blood_cell_diameter_scientific_notation :
  scientific_notation 0.00077 = (7.7, -4) :=
sorry

end red_blood_cell_diameter_scientific_notation_l3262_326253


namespace car_distance_proof_l3262_326245

theorem car_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 30 →
  (∃ (initial_speed : ℝ), 
    initial_speed * initial_time = new_speed * (initial_time * (2/3))) →
  (∃ (distance : ℝ), distance = 120) :=
by
  sorry

end car_distance_proof_l3262_326245


namespace xyz_value_l3262_326222

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 17)
  (h3 : x^3 + y^3 + z^3 = 27) :
  x * y * z = 32 / 3 := by
sorry

end xyz_value_l3262_326222


namespace cake_and_muffin_buyers_l3262_326296

theorem cake_and_muffin_buyers (total : ℕ) (cake : ℕ) (muffin : ℕ) (neither_prob : ℚ) :
  total = 100 →
  cake = 50 →
  muffin = 40 →
  neither_prob = 26 / 100 →
  ∃ (both : ℕ), both = 16 ∧ 
    (total : ℚ) * (1 - neither_prob) = (cake + muffin - both : ℚ) :=
by sorry

end cake_and_muffin_buyers_l3262_326296


namespace candy_bar_sales_l3262_326223

theorem candy_bar_sales (members : ℕ) (price : ℚ) (total_earnings : ℚ) 
  (h1 : members = 20)
  (h2 : price = 1/2)
  (h3 : total_earnings = 80) :
  (total_earnings / price) / members = 8 := by
  sorry

end candy_bar_sales_l3262_326223


namespace trig_identity_1_trig_identity_2_l3262_326219

-- Problem 1
theorem trig_identity_1 (θ : ℝ) : (Real.sin θ - Real.cos θ) / (Real.tan θ - 1) = Real.cos θ := by
  sorry

-- Problem 2
theorem trig_identity_2 (α : ℝ) : Real.sin α ^ 4 - Real.cos α ^ 4 = 2 * Real.sin α ^ 2 - 1 := by
  sorry

end trig_identity_1_trig_identity_2_l3262_326219


namespace plums_given_to_sam_l3262_326213

/-- Given Melanie's plum picking and sharing scenario, prove the number of plums given to Sam. -/
theorem plums_given_to_sam 
  (original_plums : ℕ) 
  (plums_left : ℕ) 
  (h1 : original_plums = 7)
  (h2 : plums_left = 4)
  : original_plums - plums_left = 3 := by
  sorry

end plums_given_to_sam_l3262_326213


namespace contractor_problem_solution_correctness_l3262_326284

/-- Represents the number of days required to complete the work -/
def original_days : ℕ := 9

/-- Represents the number of absent laborers -/
def absent_laborers : ℕ := 6

/-- Represents the number of days required to complete the work with absent laborers -/
def new_days : ℕ := 15

/-- Represents the original number of laborers -/
def original_laborers : ℕ := 15

theorem contractor_problem :
  original_laborers * original_days = (original_laborers - absent_laborers) * new_days :=
by sorry

theorem solution_correctness :
  original_laborers = 15 :=
by sorry

end contractor_problem_solution_correctness_l3262_326284


namespace line_equation_through_points_line_equation_specific_points_l3262_326235

/-- The equation of a line passing through two points -/
theorem line_equation_through_points (x₁ y₁ x₂ y₂ : ℝ) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (x₂ ≠ x₁) →
  (∀ x y : ℝ, y = m * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)) :=
by sorry

/-- The equation of the line passing through (0, -5) and (1, 0) is y = 5x - 5 -/
theorem line_equation_specific_points :
  ∀ x y : ℝ, y = 5 * x - 5 ↔ (x = 0 ∧ y = -5) ∨ (x = 1 ∧ y = 0) ∨ (y - (-5)) * (1 - 0) = (x - 0) * (0 - (-5)) :=
by sorry

end line_equation_through_points_line_equation_specific_points_l3262_326235


namespace right_triangle_area_l3262_326236

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 90 →
  a^2 + b^2 + c^2 = 3362 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 180 := by
sorry

end right_triangle_area_l3262_326236


namespace factor_x4_plus_64_l3262_326202

theorem factor_x4_plus_64 (x : ℝ) : 
  x^4 + 64 = (x^2 + 4*x + 8) * (x^2 - 4*x + 8) := by
sorry

end factor_x4_plus_64_l3262_326202


namespace initial_investment_solution_exists_l3262_326299

/-- Simple interest calculation function -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the initial investment given the conditions -/
theorem initial_investment (P : ℝ) (r : ℝ) :
  (simpleInterest P r 2 = 480) →
  (simpleInterest P r 7 = 680) →
  P = 400 := by
  sorry

/-- Proof of the existence of a solution -/
theorem solution_exists : ∃ (P r : ℝ),
  (simpleInterest P r 2 = 480) ∧
  (simpleInterest P r 7 = 680) ∧
  P = 400 := by
  sorry

end initial_investment_solution_exists_l3262_326299


namespace fraction_pattern_sum_of_fractions_sum_of_irrational_fractions_l3262_326263

-- Part 1: Pattern for positive integers
theorem fraction_pattern (n : ℕ+) : 
  1 / n.val * (1 / (n.val + 1)) = 1 / n.val - 1 / (n.val + 1) := by sorry

-- Part 2: Sum of fractions
theorem sum_of_fractions (x : ℝ) : 
  1 / (x * (x + 1)) + 1 / ((x + 1) * (x + 2)) + 1 / ((x + 2) * (x + 3)) + 1 / ((x + 3) * (x + 4)) = 
  4 / (x^2 + 4*x) := by sorry

-- Part 3: Sum of irrational fractions
theorem sum_of_irrational_fractions : 
  1 / (1 + Real.sqrt 2) + 1 / (Real.sqrt 2 + Real.sqrt 3) + 1 / (Real.sqrt 3 + 2) + 1 / (2 + Real.sqrt 5) +
  1 / (Real.sqrt 5 + Real.sqrt 6) + 1 / (Real.sqrt 6 + 3) + 1 / (3 + Real.sqrt 10) = 
  -1 + Real.sqrt 10 := by sorry

end fraction_pattern_sum_of_fractions_sum_of_irrational_fractions_l3262_326263


namespace square_perimeter_l3262_326294

theorem square_perimeter (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 125)
  (h2 : rectangle_width = 64)
  (h3 : rectangle_length > 0)
  (h4 : rectangle_width > 0) :
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  square_side * 4 = 800 := by sorry

end square_perimeter_l3262_326294


namespace equation_result_is_55_l3262_326266

/-- The result of 4 times a number plus 7 times the same number, given the number is 5.0 -/
def equation_result (n : ℝ) : ℝ := 4 * n + 7 * n

/-- Theorem stating that the result of the equation is 55.0 when the number is 5.0 -/
theorem equation_result_is_55 : equation_result 5.0 = 55.0 := by
  sorry

end equation_result_is_55_l3262_326266


namespace part1_part2_l3262_326214

-- Part 1
def U1 : Set ℕ := {2, 3, 4}
def A1 : Set ℕ := {4, 3}
def B1 : Set ℕ := ∅

theorem part1 :
  (U1 \ A1 = {2}) ∧ (U1 \ B1 = U1) := by sorry

-- Part 2
def U2 : Set ℝ := {x | x ≤ 4}
def A2 : Set ℝ := {x | -2 < x ∧ x < 3}
def B2 : Set ℝ := {x | -3 < x ∧ x ≤ 3}

theorem part2 :
  (U2 \ A2 = {x | x ≤ -2 ∨ (3 ≤ x ∧ x ≤ 4)}) ∧
  (A2 ∩ B2 = {x | -2 < x ∧ x < 3}) ∧
  (U2 \ (A2 ∩ B2) = {x | x ≤ -2 ∨ (3 ≤ x ∧ x ≤ 4)}) ∧
  ((U2 \ A2) ∩ B2 = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3}) := by sorry

end part1_part2_l3262_326214


namespace inequality_comparison_l3262_326264

theorem inequality_comparison (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  (∀ c, a + c > b + c) ∧
  (∀ c, a - 3*c > b - 3*c) ∧
  (¬∀ c, a*c > b*c) ∧
  (∀ c, a/c^2 > b/c^2) ∧
  (∀ c, a*c^3 > b*c^3) :=
by sorry

end inequality_comparison_l3262_326264


namespace rational_equation_solution_l3262_326295

theorem rational_equation_solution (x : ℝ) :
  (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 8*x + 15) / (x^2 - 10*x + 24) →
  x = (13 + Real.sqrt 5) / 2 ∨ x = (13 - Real.sqrt 5) / 2 := by
  sorry

end rational_equation_solution_l3262_326295


namespace nested_square_root_18_l3262_326274

theorem nested_square_root_18 :
  ∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x = (1 + Real.sqrt 73) / 2 := by
  sorry

end nested_square_root_18_l3262_326274


namespace sum_of_digits_seven_to_seventeen_l3262_326255

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_seven_to_seventeen (h : ℕ) :
  h = 7^17 →
  tens_digit h + ones_digit h = 7 :=
sorry

end sum_of_digits_seven_to_seventeen_l3262_326255


namespace negation_of_proposition_l3262_326282

theorem negation_of_proposition (x y : ℝ) :
  ¬(((x - 1)^2 + (y - 2)^2 = 0) → (x = 1 ∧ y = 2)) ↔
  (((x - 1)^2 + (y - 2)^2 ≠ 0) → (x ≠ 1 ∨ y ≠ 2)) :=
by sorry

end negation_of_proposition_l3262_326282


namespace cube_root_of_256_l3262_326237

theorem cube_root_of_256 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 256) : x = 4 * Real.rpow 2 (1/3) := by
  sorry

end cube_root_of_256_l3262_326237


namespace printer_equation_l3262_326212

/-- The equation for determining the time of the second printer to print 1000 flyers -/
theorem printer_equation (x : ℝ) : 
  (1000 : ℝ) > 0 → x > 0 → (
    (1000 / 10 + 1000 / x = 1000 / 4) ↔ 
    (1 / 10 + 1 / x = 1 / 4)
  ) := by sorry

end printer_equation_l3262_326212


namespace max_product_constraint_l3262_326269

theorem max_product_constraint (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  (x * y) + z^2 = (x + z) * (y + z) →
  x + y + z = 3 →
  x * y * z ≤ 1 := by sorry

end max_product_constraint_l3262_326269


namespace set_membership_implies_m_values_l3262_326221

theorem set_membership_implies_m_values (m : ℝ) :
  let A : Set ℝ := {1, m + 2, m^2 + 4}
  5 ∈ A → m = 3 ∨ m = 1 := by
sorry

end set_membership_implies_m_values_l3262_326221


namespace sequence_growth_l3262_326277

theorem sequence_growth (a : ℕ → ℕ) 
  (h1 : ∀ n, a n > 1) 
  (h2 : ∀ m n, m ≠ n → a m ≠ a n) : 
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ a n > n :=
sorry

end sequence_growth_l3262_326277


namespace age_ratio_sandy_molly_l3262_326234

/-- Given that Sandy is 42 years old and Molly is 12 years older than Sandy,
    prove that the ratio of their ages is 7:9. -/
theorem age_ratio_sandy_molly :
  let sandy_age : ℕ := 42
  let molly_age : ℕ := sandy_age + 12
  (sandy_age : ℚ) / molly_age = 7 / 9 := by
  sorry

end age_ratio_sandy_molly_l3262_326234


namespace car_trading_problem_l3262_326200

-- Define the profit per car for models A and B (in thousand yuan)
variable (profit_A profit_B : ℚ)

-- Define the number of cars of each model
variable (num_A num_B : ℕ)

-- Define the given conditions
axiom profit_condition_1 : 3 * profit_A + 2 * profit_B = 34
axiom profit_condition_2 : profit_A + 4 * profit_B = 28

-- Define the purchase prices (in thousand yuan)
def price_A : ℚ := 160
def price_B : ℚ := 140

-- Define the total number of cars and budget (in thousand yuan)
def total_cars : ℕ := 30
def max_budget : ℚ := 4400

-- Define the minimum profit (in thousand yuan)
def min_profit : ℚ := 177

-- Theorem statement
theorem car_trading_problem :
  (profit_A = 8 ∧ profit_B = 5) ∧
  ((num_A = 9 ∧ num_B = 21) ∨ (num_A = 10 ∧ num_B = 20)) ∧
  (num_A + num_B = total_cars) ∧
  (num_A * price_A + num_B * price_B ≤ max_budget) ∧
  (num_A * profit_A + num_B * profit_B ≥ min_profit) :=
sorry

end car_trading_problem_l3262_326200


namespace janet_return_time_l3262_326216

/-- Represents the number of blocks Janet walks in each direction --/
structure WalkingDistance where
  north : ℕ
  west : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the time taken to walk a given distance at a given speed --/
def timeToWalk (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

/-- Janet's walking pattern and speed --/
def janet : WalkingDistance × ℕ :=
  ({ north := 3
   , west := 3 * 7
   , south := 3
   , east := 3 * 2
   }, 2)

/-- Theorem: Janet takes 9 minutes to return home --/
theorem janet_return_time : 
  let (walk, speed) := janet
  timeToWalk (walk.south + (walk.west - walk.east)) speed = 9 := by
  sorry

end janet_return_time_l3262_326216


namespace valid_parameterization_l3262_326278

/-- A structure representing a vector parameterization of a line -/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ
  b : ℝ

/-- Checks if a given vector parameterization is valid for the line y = 2x - 4 -/
def isValidParam (p : VectorParam) : Prop :=
  p.y₀ = 2 * p.x₀ - 4 ∧ ∃ k : ℝ, p.a = k * 1 ∧ p.b = k * 2

/-- The theorem stating the conditions for a valid vector parameterization -/
theorem valid_parameterization (p : VectorParam) : 
  isValidParam p ↔ 
  (∀ t : ℝ, (p.x₀ + t * p.a, p.y₀ + t * p.b) ∈ {(x, y) : ℝ × ℝ | y = 2 * x - 4}) :=
sorry

end valid_parameterization_l3262_326278


namespace odd_prime_sum_iff_floor_sum_odd_l3262_326267

theorem odd_prime_sum_iff_floor_sum_odd (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (a b : ℕ) (ha : 0 < a ∧ a < p) (hb : 0 < b ∧ b < p) :
  a + b = p ↔
  ∀ n : ℕ, 0 < n → n < p →
    ∃ k : ℕ, Int.floor ((2 * a * n : ℚ) / p) + Int.floor ((2 * b * n : ℚ) / p) = 2 * k + 1 :=
by sorry

end odd_prime_sum_iff_floor_sum_odd_l3262_326267


namespace permutations_formula_l3262_326229

def factorial (n : ℕ) : ℕ := Nat.factorial n

def permutations_with_repetition (n : ℕ) (k : List ℕ) : ℚ :=
  (factorial n) / (k.map factorial).prod

theorem permutations_formula (n : ℕ) (k : List ℕ) 
  (h : k.sum = n) : 
  permutations_with_repetition n k = 
    (factorial n) / (k.map factorial).prod := by
  sorry

#eval permutations_with_repetition 5 [5]  -- for "замок"
#eval permutations_with_repetition 5 [1, 2, 2]  -- for "ротор"
#eval permutations_with_repetition 5 [3, 2]  -- for "топор"
#eval permutations_with_repetition 7 [1, 2, 2, 3]  -- for "колокол"

end permutations_formula_l3262_326229


namespace lily_painting_time_l3262_326206

/-- The time it takes to paint a lily -/
def time_for_lily : ℕ := sorry

/-- The time it takes to paint a rose -/
def time_for_rose : ℕ := 7

/-- The time it takes to paint an orchid -/
def time_for_orchid : ℕ := 3

/-- The time it takes to paint a vine -/
def time_for_vine : ℕ := 2

/-- The total time taken to paint all flowers and vines -/
def total_time : ℕ := 213

/-- The number of lilies painted -/
def num_lilies : ℕ := 17

/-- The number of roses painted -/
def num_roses : ℕ := 10

/-- The number of orchids painted -/
def num_orchids : ℕ := 6

/-- The number of vines painted -/
def num_vines : ℕ := 20

theorem lily_painting_time : time_for_lily = 5 := by
  sorry

end lily_painting_time_l3262_326206


namespace cube_volume_l3262_326246

theorem cube_volume (edge : ℝ) (h : edge = 7) : edge^3 = 343 := by
  sorry

end cube_volume_l3262_326246


namespace triangle_area_l3262_326227

theorem triangle_area (base height : ℝ) (h1 : base = 8.4) (h2 : height = 5.8) :
  (base * height) / 2 = 24.36 := by
  sorry

end triangle_area_l3262_326227


namespace right_triangle_hypotenuse_l3262_326217

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 80 → 
  b = 150 → 
  c^2 = a^2 + b^2 → 
  c = 170 := by sorry

end right_triangle_hypotenuse_l3262_326217


namespace problems_left_to_grade_l3262_326250

theorem problems_left_to_grade 
  (problems_per_worksheet : ℕ) 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (h1 : problems_per_worksheet = 4)
  (h2 : total_worksheets = 9)
  (h3 : graded_worksheets = 5) :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 :=
by sorry

end problems_left_to_grade_l3262_326250


namespace sphere_equal_volume_surface_area_l3262_326244

theorem sphere_equal_volume_surface_area (r k S : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = S ∧ 
  4 * Real.pi * r^2 = S ∧ 
  k * r = S → 
  r = 3 ∧ k = 12 * Real.pi := by
sorry

end sphere_equal_volume_surface_area_l3262_326244


namespace moving_circle_trajectory_l3262_326231

/-- A moving circle in a plane passing through (-2, 0) and tangent to x = 2 -/
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_A : center.1 ^ 2 + center.2 ^ 2 = (-2 - center.1) ^ 2 + center.2 ^ 2
  tangent_to_line : (2 - center.1) ^ 2 + center.2 ^ 2 = (2 - (-2)) ^ 2

/-- The trajectory of the center of the moving circle -/
def trajectory_equation (x y : ℝ) : Prop :=
  y ^ 2 = -8 * x

theorem moving_circle_trajectory :
  ∀ (c : MovingCircle), trajectory_equation c.center.1 c.center.2 :=
sorry

end moving_circle_trajectory_l3262_326231


namespace second_element_of_sequence_l3262_326228

theorem second_element_of_sequence (n : ℕ) : 
  n > 1 → (n * (n + 1)) / 2 = 78 → 2 = 2 := by
  sorry

end second_element_of_sequence_l3262_326228


namespace max_value_of_sum_l3262_326283

theorem max_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 1 ∧
    1 / (a' + b' + 1) + 1 / (b' + c' + 1) + 1 / (c' + a' + 1) = 1 :=
by sorry

end max_value_of_sum_l3262_326283


namespace square_of_nilpotent_matrix_is_zero_l3262_326226

theorem square_of_nilpotent_matrix_is_zero (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : A ^ 2 = 0 := by
  sorry

end square_of_nilpotent_matrix_is_zero_l3262_326226


namespace larger_number_problem_l3262_326201

theorem larger_number_problem (x y : ℝ) 
  (sum : x + y = 40)
  (diff : x - y = 10)
  (prod : x * y = 375)
  (greater : x > y) : x = 25 := by
sorry

end larger_number_problem_l3262_326201


namespace least_cans_required_l3262_326286

theorem least_cans_required (a b c d e f g h : ℕ+) : 
  a = 139 → b = 223 → c = 179 → d = 199 → e = 173 → f = 211 → g = 131 → h = 257 →
  (∃ (x : ℕ+), x = a + b + c + d + e + f + g + h ∧ 
   x = Nat.gcd a (Nat.gcd b (Nat.gcd c (Nat.gcd d (Nat.gcd e (Nat.gcd f (Nat.gcd g h))))))) :=
by sorry

end least_cans_required_l3262_326286


namespace derivative_cos_minus_cube_l3262_326261

/-- The derivative of f(x) = cos x - x^3 is f'(x) = -sin x - 3x^2 -/
theorem derivative_cos_minus_cube (x : ℝ) : 
  deriv (λ x : ℝ => Real.cos x - x^3) x = -Real.sin x - 3 * x^2 := by
  sorry

end derivative_cos_minus_cube_l3262_326261


namespace jake_peaches_count_l3262_326205

-- Define the number of apples and peaches for Steven and Jake
def steven_apples : ℕ := 52
def steven_peaches : ℕ := 13
def jake_apples : ℕ := steven_apples + 84
def jake_peaches : ℕ := steven_peaches - 10

-- Theorem to prove
theorem jake_peaches_count : jake_peaches = 3 := by
  sorry

end jake_peaches_count_l3262_326205


namespace irrigation_flux_theorem_l3262_326252

-- Define the irrigation system
structure IrrigationSystem where
  channels : List Char
  entry : Char
  exit : Char
  flux : Char → Char → ℝ

-- Define the properties of the irrigation system
def has_constant_flux_sum (sys : IrrigationSystem) : Prop :=
  ∀ (p q r : Char), p ∈ sys.channels → q ∈ sys.channels → r ∈ sys.channels →
    sys.flux p q + sys.flux q r = sys.flux p r

-- Define the theorem
theorem irrigation_flux_theorem (sys : IrrigationSystem) 
  (h_channels : sys.channels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
  (h_entry : sys.entry = 'A')
  (h_exit : sys.exit = 'E')
  (h_constant_flux : has_constant_flux_sum sys)
  (h_flux_bc : sys.flux 'B' 'C' = q₀) :
  sys.flux 'A' 'B' = 2 * q₀ ∧ 
  sys.flux 'A' 'H' = 3/2 * q₀ ∧ 
  sys.flux 'A' 'B' + sys.flux 'A' 'H' = 7/2 * q₀ := by
  sorry

end irrigation_flux_theorem_l3262_326252


namespace sum_equals_1998_l3262_326257

theorem sum_equals_1998 (a b c d : ℕ) (h : a * c + b * d + a * d + b * c = 1997) :
  a + b + c + d = 1998 := by sorry

end sum_equals_1998_l3262_326257


namespace calculations_proof_l3262_326240

-- Define the calculations
def calc1 : ℝ := 70.8 - 1.25 - 1.75
def calc2 : ℝ := (8 + 0.8) * 1.25
def calc3 : ℝ := 125 * 0.48
def calc4 : ℝ := 6.7 * (9.3 * (6.2 + 1.7))

-- Theorem to prove the calculations
theorem calculations_proof :
  calc1 = 67.8 ∧
  calc2 = 11 ∧
  calc3 = 600 ∧
  calc4 = 554.559 := by
  sorry

#eval calc1
#eval calc2
#eval calc3
#eval calc4

end calculations_proof_l3262_326240


namespace quadratic_distinct_roots_l3262_326273

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
by sorry

end quadratic_distinct_roots_l3262_326273


namespace parabola_intersection_theorem_l3262_326225

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line l
def line_l (p m : ℝ) (x y : ℝ) : Prop := x = m*y + p/2

-- Define points A and B on the parabola and line
def point_on_parabola_and_line (p m : ℝ) (x y : ℝ) : Prop :=
  parabola p x y ∧ line_l p m x y

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁*x₂ + y₁*y₂ = -3

-- Define the distance between two points
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Define the minimization condition
def is_minimum (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∀ x₁' y₁' x₂' y₂', point_on_parabola_and_line p ((x₁' + x₂')/2) x₁' y₁' →
    point_on_parabola_and_line p ((x₁' + x₂')/2) x₂' y₂' →
    dot_product_condition x₁' y₁' x₂' y₂' →
    (|x₁' - p/2| + 4*|x₂' - p/2| ≥ |x₁ - p/2| + 4*|x₂ - p/2|)

-- The main theorem
theorem parabola_intersection_theorem (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabola p x₁ y₁ →
  parabola p x₂ y₂ →
  (∃ m : ℝ, line_l p m x₁ y₁ ∧ line_l p m x₂ y₂) →
  dot_product_condition x₁ y₁ x₂ y₂ →
  is_minimum p x₁ y₁ x₂ y₂ →
  distance x₁ y₁ x₂ y₂ = 9/2 :=
sorry

end parabola_intersection_theorem_l3262_326225


namespace puzzle_pieces_left_l3262_326262

theorem puzzle_pieces_left (total_pieces : ℕ) (num_sons : ℕ) (reyn_pieces : ℕ) : 
  total_pieces = 300 →
  num_sons = 3 →
  reyn_pieces = 25 →
  total_pieces - (reyn_pieces + 2 * reyn_pieces + 3 * reyn_pieces) = 150 :=
by
  sorry

#check puzzle_pieces_left

end puzzle_pieces_left_l3262_326262


namespace A_P_parity_uniformity_l3262_326239

-- Define the set A_P
def A_P : Set ℤ := sorry

-- Define a property for elements of A_P related to positioning in a function or polynomial
def has_positioning_property (n : ℤ) : Prop := sorry

-- Axiom: All elements in A_P have the positioning property
axiom A_P_property : ∀ n ∈ A_P, has_positioning_property n

-- Define parity
def same_parity (a b : ℤ) : Prop := a % 2 = b % 2

-- Theorem: The smallest and largest elements of A_P have the same parity
theorem A_P_parity_uniformity :
  ∀ (min max : ℤ), min ∈ A_P → max ∈ A_P →
  (∀ x ∈ A_P, min ≤ x ∧ x ≤ max) →
  same_parity min max :=
sorry

end A_P_parity_uniformity_l3262_326239


namespace great_circle_to_surface_area_ratio_l3262_326270

theorem great_circle_to_surface_area_ratio (O : Type*) [MetricSpace O] [NormedAddCommGroup O] 
  [InnerProductSpace ℝ O] [FiniteDimensional ℝ O] [ProperSpace O] (S₁ S₂ : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ S₁ = π * r^2 ∧ S₂ = 4 * π * r^2) → 
  S₁ / S₂ = 1 / 4 := by
sorry

end great_circle_to_surface_area_ratio_l3262_326270


namespace solution_product_l3262_326272

theorem solution_product (a b : ℝ) : 
  (a - 3) * (3 * a + 7) = a^2 - 16 * a + 55 →
  (b - 3) * (3 * b + 7) = b^2 - 16 * b + 55 →
  a ≠ b →
  (a + 2) * (b + 2) = -54 := by
sorry

end solution_product_l3262_326272


namespace equation_solution_l3262_326259

theorem equation_solution : ∃ x : ℝ, 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ∧ 
  x = 31 := by
  sorry

end equation_solution_l3262_326259


namespace triangle_trigonometric_identities_l3262_326279

theorem triangle_trigonometric_identities
  (α β γ : Real) (p r R : Real)
  (h_triangle : α + β + γ = Real.pi)
  (h_positive : 0 < p ∧ 0 < r ∧ 0 < R)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_inradius : r = area / p)
  (h_circumradius : R = (a * b * c) / (4 * area)) :
  (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = (p^2 - r^2 - 4*r*R) / (2*R^2) ∧
  4 * R^2 * Real.cos α * Real.cos β * Real.cos γ = p^2 - (2*R + r)^2 :=
by sorry

end triangle_trigonometric_identities_l3262_326279


namespace tangent_lines_to_circle_l3262_326209

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  abs (l.a * x₀ + l.b * y₀ + l.c) / Real.sqrt (l.a^2 + l.b^2) = c.radius

/-- The main theorem -/
theorem tangent_lines_to_circle 
  (c : Circle) 
  (p : ℝ × ℝ) 
  (h_circle : c.center = (1, 1) ∧ c.radius = 1) 
  (h_point : p = (2, 3)) :
  ∃ (l₁ l₂ : Line),
    isTangent l₁ c ∧ isTangent l₂ c ∧
    (l₁.a = 1 ∧ l₁.b = 0 ∧ l₁.c = -2) ∧
    (l₂.a = 3 ∧ l₂.b = -4 ∧ l₂.c = 6) :=
sorry

end tangent_lines_to_circle_l3262_326209


namespace value_of_B_l3262_326218

/-- Given the value assignments for letters and words, prove the value of B --/
theorem value_of_B (T L A B : ℤ) : 
  T = 15 →
  B + A + L + L = 40 →
  L + A + B = 25 →
  A + L + L = 30 →
  B = 10 := by
sorry

end value_of_B_l3262_326218


namespace triangles_from_points_l3262_326251

/-- Represents a triangular paper with points -/
structure TriangularPaper where
  n : ℕ  -- number of points inside the triangle

/-- Condition that no three points are collinear -/
axiom not_collinear (paper : TriangularPaper) : True

/-- Function to calculate the number of smaller triangles -/
def num_triangles (paper : TriangularPaper) : ℕ :=
  2 * paper.n + 1

/-- Theorem stating the relationship between points and triangles -/
theorem triangles_from_points (paper : TriangularPaper) :
  num_triangles paper = 2 * paper.n + 1 :=
sorry

end triangles_from_points_l3262_326251


namespace white_washing_cost_per_square_foot_l3262_326297

/-- Calculates the cost per square foot for white washing a room --/
theorem white_washing_cost_per_square_foot
  (room_length room_width room_height : ℝ)
  (door_length door_width : ℝ)
  (window_length window_width : ℝ)
  (num_windows : ℕ)
  (total_cost : ℝ)
  (h_room_length : room_length = 25)
  (h_room_width : room_width = 15)
  (h_room_height : room_height = 12)
  (h_door_length : door_length = 6)
  (h_door_width : door_width = 3)
  (h_window_length : window_length = 4)
  (h_window_width : window_width = 3)
  (h_num_windows : num_windows = 3)
  (h_total_cost : total_cost = 8154) :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_length * door_width
  let window_area := num_windows * (window_length * window_width)
  let net_area := wall_area - door_area - window_area
  let cost_per_square_foot := total_cost / net_area
  cost_per_square_foot = 9 :=
sorry

end white_washing_cost_per_square_foot_l3262_326297


namespace two_stretches_to_similar_triangle_l3262_326248

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define a stretch transformation
structure Stretch where
  center : Point2D
  coefficient : ℝ

-- Define similarity between triangles
def Similar (t1 t2 : Triangle) : Prop := sorry

-- Define the application of a stretch to a triangle
def ApplyStretch (s : Stretch) (t : Triangle) : Triangle := sorry

-- Theorem statement
theorem two_stretches_to_similar_triangle 
  (ABC : Triangle) (DEF : Triangle) (h : DEF.A.x = DEF.A.y ∧ DEF.B.x = DEF.B.y) :
  ∃ (S1 S2 : Stretch), Similar (ApplyStretch S2 (ApplyStretch S1 ABC)) DEF := by
  sorry

end two_stretches_to_similar_triangle_l3262_326248


namespace sqrt_sum_inequality_l3262_326276

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (Real.sqrt a + Real.sqrt b)^8 ≥ 64 * a * b * (a + b)^2 := by
  sorry

end sqrt_sum_inequality_l3262_326276


namespace solution_value_l3262_326241

theorem solution_value (a b t : ℝ) : 
  a^2 + 4*b = t^2 →
  a^2 - b^2 = 4 →
  b > 0 →
  b = t - 2 := by
sorry

end solution_value_l3262_326241


namespace prime_sum_problem_l3262_326233

theorem prime_sum_problem (a b c : ℕ) 
  (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (hab : a + b = 49) (hbc : b + c = 60) : c = 13 := by
  sorry

end prime_sum_problem_l3262_326233


namespace inverse_of_3_mod_221_l3262_326293

theorem inverse_of_3_mod_221 : ∃ x : ℕ, x < 221 ∧ (3 * x) % 221 = 1 :=
by
  use 74
  sorry

end inverse_of_3_mod_221_l3262_326293


namespace five_line_regions_l3262_326238

/-- Number of regions formed by n lines in a plane -/
def num_regions (n : ℕ) : ℕ := 1 + n + n.choose 2

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  not_parallel : Prop
  not_concurrent : Prop

/-- The number of regions formed by a line configuration -/
def regions_formed (config : LineConfiguration) : ℕ := num_regions config.num_lines

theorem five_line_regions (config : LineConfiguration) :
  config.num_lines = 5 →
  config.not_parallel →
  config.not_concurrent →
  regions_formed config = 16 := by
  sorry

end five_line_regions_l3262_326238


namespace polygon_sides_diagonals_l3262_326208

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon has 11 sides if the number of its diagonals is 33 more than the number of its sides -/
theorem polygon_sides_diagonals : 
  ∃ (n : ℕ), n > 3 ∧ num_diagonals n = n + 33 ∧ n = 11 := by
  sorry

end polygon_sides_diagonals_l3262_326208


namespace inequality_proof_l3262_326211

theorem inequality_proof (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end inequality_proof_l3262_326211


namespace inequality_proof_l3262_326280

theorem inequality_proof (a b c : ℝ) (k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 2) (habc : a * b * c = 1) :
  (a^k / (a + b)) + (b^k / (b + c)) + (c^k / (c + a)) ≥ 3/2 := by
  sorry

end inequality_proof_l3262_326280


namespace expected_white_balls_after_transfer_l3262_326232

/-- Represents a bag of colored balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- Represents the process of transferring balls between bags -/
def transfer (a b : Bag) : ℝ → Bag × Bag
  | p => sorry

/-- Calculates the expected number of white balls in the first bag after transfers -/
noncomputable def expected_white_balls (a b : Bag) : ℝ :=
  sorry

theorem expected_white_balls_after_transfer :
  let a : Bag := { red := 2, white := 3 }
  let b : Bag := { red := 3, white := 3 }
  expected_white_balls a b = 102 / 35 := by sorry

end expected_white_balls_after_transfer_l3262_326232


namespace triangle_angle_sum_l3262_326271

theorem triangle_angle_sum (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- Ensures positive angles
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A = 25 →  -- Given angle A
  B = 55 →  -- Given angle B
  C = 100 :=  -- Conclusion: angle C
by sorry

end triangle_angle_sum_l3262_326271


namespace class_savings_theorem_l3262_326265

/-- Calculates the total amount saved by a class for a field trip over a given period. -/
def total_savings (num_students : ℕ) (contribution_per_student : ℕ) (weeks_per_month : ℕ) (num_months : ℕ) : ℕ :=
  num_students * contribution_per_student * weeks_per_month * num_months

/-- Theorem stating that a class of 30 students contributing $2 each week will save $480 in 2 months. -/
theorem class_savings_theorem :
  total_savings 30 2 4 2 = 480 := by
  sorry

#eval total_savings 30 2 4 2

end class_savings_theorem_l3262_326265


namespace triangle_division_theorem_l3262_326203

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def pointInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Represent a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Calculate the area of a part of the triangle cut by a line -/
def areaPartition (t : Triangle) (l : Line) : ℝ := sorry

theorem triangle_division_theorem (t : Triangle) (P : ℝ × ℝ) (m n : ℝ) 
  (h_point : pointInTriangle P t) (h_positive : m > 0 ∧ n > 0) :
  ∃ (l : Line), 
    pointOnLine P l ∧ 
    areaPartition t l / (triangleArea t - areaPartition t l) = m / n :=
sorry

end triangle_division_theorem_l3262_326203


namespace min_value_of_g_l3262_326258

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define the maximum value M(a) on the interval [1,3]
noncomputable def M (a : ℝ) : ℝ := 
  ⨆ (x : ℝ) (h : x ∈ Set.Icc 1 3), f a x

-- Define the minimum value N(a) on the interval [1,3]
noncomputable def N (a : ℝ) : ℝ := 
  ⨅ (x : ℝ) (h : x ∈ Set.Icc 1 3), f a x

-- Define g(a) as M(a) - N(a)
noncomputable def g (a : ℝ) : ℝ := M a - N a

-- State the theorem
theorem min_value_of_g :
  ∀ a : ℝ, a ∈ Set.Icc (1/3) 1 → 
  ∃ min_g : ℝ, min_g = (⨅ (a : ℝ) (h : a ∈ Set.Icc (1/3) 1), g a) ∧ min_g = 1/2 := by
  sorry

end min_value_of_g_l3262_326258


namespace todd_ate_five_cupcakes_l3262_326207

def cupcake_problem (initial_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  initial_cupcakes - packages * cupcakes_per_package

theorem todd_ate_five_cupcakes :
  cupcake_problem 50 9 5 = 5 :=
by sorry

end todd_ate_five_cupcakes_l3262_326207


namespace cookie_and_game_cost_l3262_326287

/-- Represents the cost and profit information for an item --/
structure ItemInfo where
  cost : ℚ
  price : ℚ
  profit : ℚ
  makeTime : ℚ

/-- Represents the sales quota for each item --/
structure SalesQuota where
  bracelets : ℕ
  necklaces : ℕ
  rings : ℕ

def bracelet : ItemInfo := ⟨1, 1.5, 0.5, 10/60⟩
def necklace : ItemInfo := ⟨2, 3, 1, 15/60⟩
def ring : ItemInfo := ⟨0.5, 1, 0.5, 5/60⟩

def salesQuota : SalesQuota := ⟨5, 3, 10⟩

def profitMargin : ℚ := 0.5
def workingHoursPerDay : ℚ := 2
def daysInWeek : ℕ := 7
def remainingMoney : ℚ := 5

theorem cookie_and_game_cost (totalSales totalCost : ℚ) :
  totalSales = (bracelet.price * salesQuota.bracelets + 
                necklace.price * salesQuota.necklaces + 
                ring.price * salesQuota.rings) →
  totalCost = (bracelet.cost * salesQuota.bracelets + 
               necklace.cost * salesQuota.necklaces + 
               ring.cost * salesQuota.rings) →
  totalSales = totalCost * (1 + profitMargin) →
  (bracelet.makeTime * salesQuota.bracelets + 
   necklace.makeTime * salesQuota.necklaces + 
   ring.makeTime * salesQuota.rings) ≤ workingHoursPerDay * daysInWeek →
  totalSales - remainingMoney = 24 := by
  sorry

end cookie_and_game_cost_l3262_326287


namespace remaining_fruits_count_l3262_326292

/-- Represents the number of fruits on each tree type -/
structure FruitTrees :=
  (apples : ℕ)
  (plums : ℕ)
  (pears : ℕ)
  (cherries : ℕ)

/-- Represents the fraction of fruits picked from each tree -/
structure PickedFractions :=
  (apples : ℚ)
  (plums : ℚ)
  (pears : ℚ)
  (cherries : ℚ)

def original_fruits : FruitTrees :=
  { apples := 180
  , plums := 60
  , pears := 120
  , cherries := 720 }

def picked_fractions : PickedFractions :=
  { apples := 3/5
  , plums := 2/3
  , pears := 3/4
  , cherries := 7/10 }

theorem remaining_fruits_count 
  (orig : FruitTrees) 
  (picked : PickedFractions) 
  (h1 : orig.apples = 3 * orig.plums)
  (h2 : orig.pears = 2 * orig.plums)
  (h3 : orig.cherries = 4 * orig.apples)
  (h4 : orig = original_fruits)
  (h5 : picked = picked_fractions) :
  (orig.apples - (picked.apples * orig.apples).num) +
  (orig.plums - (picked.plums * orig.plums).num) +
  (orig.pears - (picked.pears * orig.pears).num) +
  (orig.cherries - (picked.cherries * orig.cherries).num) = 338 :=
by sorry

end remaining_fruits_count_l3262_326292


namespace fraction_addition_l3262_326220

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l3262_326220


namespace a_range_l3262_326281

/-- Proposition p: For all real x, ax^2 + 2ax + 3 > 0 -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * a * x + 3 > 0

/-- Proposition q: There exists a real x such that x^2 + 2ax + a + 2 = 0 -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

/-- The theorem stating that if both p and q are true, then a is in the range [2, 3) -/
theorem a_range (a : ℝ) (hp : p a) (hq : q a) : a ∈ Set.Ici 2 ∩ Set.Iio 3 := by
  sorry

end a_range_l3262_326281


namespace octagon_circle_circumference_l3262_326291

/-- The circumference of a circle containing an inscribed regular octagon -/
theorem octagon_circle_circumference (side_length : ℝ) (h : side_length = 5) :
  ∃ (circumference : ℝ), circumference = (5 * π) / Real.sin (22.5 * π / 180) := by
  sorry

end octagon_circle_circumference_l3262_326291


namespace parallel_lines_a_equals_one_l3262_326230

/-- Two lines in the xy-plane -/
structure ParallelLines where
  /-- The first line equation: x + 2y - 4 = 0 -/
  line1 : ℝ → ℝ → Prop := fun x y => x + 2*y - 4 = 0
  /-- The second line equation: ax + 2y + 6 = 0 -/
  line2 : ℝ → ℝ → ℝ → Prop := fun a x y => a*x + 2*y + 6 = 0
  /-- The lines are parallel -/
  parallel : ∀ (a : ℝ), (∀ x y, line1 x y ↔ ∃ k, line2 a (x + k) (y + k))

/-- If two lines are parallel as defined, then a = 1 -/
theorem parallel_lines_a_equals_one (pl : ParallelLines) : ∃ a, ∀ x y, pl.line2 a x y ↔ pl.line2 1 x y := by
  sorry

end parallel_lines_a_equals_one_l3262_326230


namespace sum_of_fours_and_fives_l3262_326290

/-- The number of ways to write 1800 as the sum of fours and fives -/
def ways_to_sum_1800 : ℕ :=
  (Finset.range 201).card

/-- Theorem: There are exactly 201 ways to write 1800 as the sum of fours and fives -/
theorem sum_of_fours_and_fives :
  ways_to_sum_1800 = 201 := by
  sorry

#eval ways_to_sum_1800  -- Should output 201

end sum_of_fours_and_fives_l3262_326290


namespace square_side_difference_l3262_326247

theorem square_side_difference (a b : ℝ) 
  (h1 : a + b = 20) 
  (h2 : a^2 - b^2 = 40) : 
  a - b = 2 := by
sorry

end square_side_difference_l3262_326247


namespace sqrt_t4_4t2_4_l3262_326224

theorem sqrt_t4_4t2_4 (t : ℝ) : Real.sqrt (t^4 + 4*t^2 + 4) = |t^2 + 2| := by
  sorry

end sqrt_t4_4t2_4_l3262_326224


namespace quadruple_solutions_l3262_326254

theorem quadruple_solutions : 
  ∀ a b c d : ℝ, 
    (a * b + c * d = 6) ∧ 
    (a * c + b * d = 3) ∧ 
    (a * d + b * c = 2) ∧ 
    (a + b + c + d = 6) → 
    ((a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
     (a = 2 ∧ b = 3 ∧ c = 0 ∧ d = 1) ∨
     (a = 1 ∧ b = 0 ∧ c = 3 ∧ d = 2) ∨
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 0)) :=
by sorry

end quadruple_solutions_l3262_326254
