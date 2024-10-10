import Mathlib

namespace division_remainder_l3292_329277

theorem division_remainder (N : ℕ) : 
  (∃ R : ℕ, N = 44 * 432 + R) ∧ 
  (∃ Q : ℕ, N = 39 * Q + 15) → 
  ∃ Q' : ℕ, N = 44 * Q' + 0 := by
sorry

end division_remainder_l3292_329277


namespace polynomial_perfect_square_count_l3292_329228

def p (x : ℤ) : ℤ := 4*x^4 - 12*x^3 + 17*x^2 - 6*x - 14

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

theorem polynomial_perfect_square_count :
  ∃! (S : Finset ℤ), S.card = 2 ∧ ∀ x : ℤ, x ∈ S ↔ is_perfect_square (p x) :=
sorry

end polynomial_perfect_square_count_l3292_329228


namespace geometric_sequence_problem_l3292_329294

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_positive_geometric_sequence a)
  (h_sum : a 2 + a 6 = 10)
  (h_prod : a 4 * a 8 = 64) :
  a 4 = 4 := by
sorry

end geometric_sequence_problem_l3292_329294


namespace train_distance_l3292_329215

/-- Represents the speed of a train in miles per minute -/
def train_speed : ℚ := 3 / 2.25

/-- Represents the duration of the journey in minutes -/
def journey_duration : ℚ := 120

/-- Theorem stating that the train will travel 160 miles in 2 hours -/
theorem train_distance : train_speed * journey_duration = 160 := by
  sorry

end train_distance_l3292_329215


namespace intersection_point_l3292_329209

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The y-intercept of the first line -/
def b₁ : ℚ := -2

/-- The x-coordinate of the given point -/
def x₀ : ℚ := 2

/-- The y-coordinate of the given point -/
def y₀ : ℚ := 2

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The y-intercept of the perpendicular line -/
def b₂ : ℚ := y₀ - m₂ * x₀

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := (b₂ - b₁) / (m₁ - m₂)

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := m₁ * x_intersect + b₁

theorem intersection_point :
  (x_intersect = 7/5) ∧ (y_intersect = 11/5) := by
  sorry

end intersection_point_l3292_329209


namespace base8_subtraction_l3292_329201

-- Define a function to convert base 8 numbers to natural numbers
def base8ToNat (x : ℕ) : ℕ := sorry

-- Define a function to convert natural numbers to base 8
def natToBase8 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_subtraction :
  natToBase8 (base8ToNat 546 - base8ToNat 321 - base8ToNat 105) = 120 := by sorry

end base8_subtraction_l3292_329201


namespace dress_designs_count_l3292_329284

/-- The number of different fabric colors available -/
def num_colors : ℕ := 5

/-- The number of different patterns available -/
def num_patterns : ℕ := 4

/-- The number of different sizes available -/
def num_sizes : ℕ := 3

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_patterns * num_sizes

/-- Theorem stating that the total number of possible dress designs is 60 -/
theorem dress_designs_count : total_designs = 60 := by
  sorry

end dress_designs_count_l3292_329284


namespace eldest_boy_age_l3292_329292

/-- Given three boys whose ages are in proportion 3 : 5 : 7 and have an average age of 15 years,
    the age of the eldest boy is 21 years. -/
theorem eldest_boy_age (age1 age2 age3 : ℕ) : 
  age1 + age2 + age3 = 45 →  -- average age is 15
  ∃ (k : ℕ), age1 = 3 * k ∧ age2 = 5 * k ∧ age3 = 7 * k →  -- ages are in proportion 3 : 5 : 7
  age3 = 21 :=
by sorry

end eldest_boy_age_l3292_329292


namespace smallest_pair_with_six_coins_l3292_329219

/-- Represents the value of a coin in half-pennies -/
inductive Coin : Nat → Type where
  | halfpenny : Coin 1
  | penny : Coin 2
  | threepence : Coin 6
  | fourpence : Coin 8
  | sixpence : Coin 12
  | shilling : Coin 24

/-- Checks if an amount can be represented with exactly 6 coins -/
def representableWithSixCoins (amount : Nat) : Prop :=
  ∃ (c₁ c₂ c₃ c₄ c₅ c₆ : Nat),
    (∃ (coin₁ : Coin c₁) (coin₂ : Coin c₂) (coin₃ : Coin c₃)
        (coin₄ : Coin c₄) (coin₅ : Coin c₅) (coin₆ : Coin c₆),
      c₁ + c₂ + c₃ + c₄ + c₅ + c₆ = amount)

/-- The main theorem to prove -/
theorem smallest_pair_with_six_coins :
  ∀ (a b : Nat),
    a < 60 ∧ b < 60 ∧ a < b ∧
    representableWithSixCoins a ∧
    representableWithSixCoins b ∧
    representableWithSixCoins (a + b) →
    a ≥ 23 ∧ b ≥ 47 :=
sorry

end smallest_pair_with_six_coins_l3292_329219


namespace geometric_series_sum_five_terms_quarter_l3292_329236

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_five_terms_quarter :
  geometric_series_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end geometric_series_sum_five_terms_quarter_l3292_329236


namespace working_partner_receives_6000_l3292_329256

/-- Calculates the amount received by the working partner in a business partnership --/
def amount_received_by_working_partner (total_profit management_fee_percentage a_capital b_capital : ℚ) : ℚ :=
  let management_fee := management_fee_percentage * total_profit
  let remaining_profit := total_profit - management_fee
  let total_capital := a_capital + b_capital
  let a_share := (a_capital / total_capital) * remaining_profit
  management_fee + a_share

/-- Theorem stating that the working partner receives 6000 Rs given the specified conditions --/
theorem working_partner_receives_6000 :
  let total_profit : ℚ := 9600
  let management_fee_percentage : ℚ := 1/10
  let a_capital : ℚ := 3500
  let b_capital : ℚ := 2500
  amount_received_by_working_partner total_profit management_fee_percentage a_capital b_capital = 6000 := by
  sorry

end working_partner_receives_6000_l3292_329256


namespace profit_percentage_is_twenty_percent_l3292_329235

/-- Calculates the profit percentage given wholesale price, retail price, and discount percentage. -/
def profit_percentage (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that under the given conditions, the profit percentage is 20%. -/
theorem profit_percentage_is_twenty_percent :
  profit_percentage 90 120 (10/100) = 20 := by
  sorry

end profit_percentage_is_twenty_percent_l3292_329235


namespace function_inequality_l3292_329202

/-- Given a continuous function f: ℝ → ℝ such that xf'(x) < 0 for all x in ℝ,
    prove that f(-1) + f(1) < 2f(0). -/
theorem function_inequality (f : ℝ → ℝ) 
    (hf_cont : Continuous f) 
    (hf_deriv : ∀ x : ℝ, x * (deriv f x) < 0) : 
    f (-1) + f 1 < 2 * f 0 := by
  sorry

end function_inequality_l3292_329202


namespace ratio_value_l3292_329238

theorem ratio_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end ratio_value_l3292_329238


namespace factorization_of_cubic_l3292_329272

theorem factorization_of_cubic (x : ℝ) : 6 * x^3 - 24 = 6 * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end factorization_of_cubic_l3292_329272


namespace expand_and_simplify_l3292_329293

theorem expand_and_simplify (a b : ℝ) : (3*a + b) * (a - b) = 3*a^2 - 2*a*b - b^2 := by
  sorry

end expand_and_simplify_l3292_329293


namespace nested_fraction_equation_l3292_329237

theorem nested_fraction_equation (x : ℚ) : 
  3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 53/16 → x = -3/2 := by
  sorry

end nested_fraction_equation_l3292_329237


namespace kingdom_cats_and_hogs_l3292_329285

theorem kingdom_cats_and_hogs (num_hogs : ℕ) (num_cats : ℕ) : 
  num_hogs = 630 → 
  num_hogs = 7 * num_cats → 
  15 < (0.8 * (num_cats^2 : ℝ)) → 
  (0.8 * (num_cats^2 : ℝ)) - 15 = 6465 := by
sorry

end kingdom_cats_and_hogs_l3292_329285


namespace xy_sum_theorem_l3292_329216

theorem xy_sum_theorem (x y : ℤ) (h : 2*x*y + x + y = 83) : 
  x + y = 83 ∨ x + y = -85 := by
sorry

end xy_sum_theorem_l3292_329216


namespace justin_tim_games_count_l3292_329265

/-- The number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of players to choose after Justin and Tim are already selected -/
def players_to_choose : ℕ := players_per_game - 2

/-- The number of remaining players after Justin and Tim are excluded -/
def remaining_players : ℕ := total_players - 2

/-- Theorem stating that the number of games Justin and Tim play together
    is equal to the number of ways to choose the remaining players -/
theorem justin_tim_games_count :
  Nat.choose remaining_players players_to_choose = 210 := by
  sorry

end justin_tim_games_count_l3292_329265


namespace non_increasing_iff_exists_greater_l3292_329257

open Set

-- Define the property of being an increasing function
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

-- Define the property of being a non-increasing function
def IsNonIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ f x > f y

-- Theorem statement
theorem non_increasing_iff_exists_greater (f : ℝ → ℝ) (a b : ℝ) :
  IsNonIncreasing f a b ↔ ¬(IsIncreasing f a b) :=
sorry

end non_increasing_iff_exists_greater_l3292_329257


namespace inequality_proof_l3292_329244

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
sorry

end inequality_proof_l3292_329244


namespace highest_power_of_three_dividing_N_l3292_329253

def N : ℕ := sorry

theorem highest_power_of_three_dividing_N : 
  (∃ m : ℕ, N = 3 * m) ∧ ¬(∃ m : ℕ, N = 9 * m) := by sorry

end highest_power_of_three_dividing_N_l3292_329253


namespace intersection_of_M_and_N_l3292_329276

def M : Set ℝ := {x | |x| < 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l3292_329276


namespace new_city_building_count_l3292_329227

/-- Represents the number of buildings of each type in Pittsburgh -/
structure PittsburghBuildings where
  stores : Nat
  hospitals : Nat
  schools : Nat
  police_stations : Nat

/-- Calculates the total number of buildings for the new city based on Pittsburgh's data -/
def new_city_buildings (p : PittsburghBuildings) : Nat :=
  p.stores / 2 + p.hospitals * 2 + (p.schools - 50) + (p.police_stations + 5)

/-- The theorem stating that given Pittsburgh's building numbers, the new city will require 2175 buildings -/
theorem new_city_building_count (p : PittsburghBuildings) 
  (h1 : p.stores = 2000)
  (h2 : p.hospitals = 500)
  (h3 : p.schools = 200)
  (h4 : p.police_stations = 20) :
  new_city_buildings p = 2175 := by
  sorry

end new_city_building_count_l3292_329227


namespace rational_point_coloring_l3292_329270

/-- A coloring function for rational points in the plane -/
def coloringFunction (n : ℕ) (p : ℚ × ℚ) : Fin n :=
  sorry

/-- A predicate to check if a point is on a line segment -/
def isOnLineSegment (p q r : ℚ × ℚ) : Prop :=
  sorry

theorem rational_point_coloring (n : ℕ) (hn : n > 0) :
  ∃ (f : ℚ × ℚ → Fin n),
    ∀ (p q : ℚ × ℚ) (c : Fin n),
      ∃ (r : ℚ × ℚ), isOnLineSegment p q r ∧ f r = c :=
sorry

end rational_point_coloring_l3292_329270


namespace max_profit_is_4900_l3292_329245

/-- A transportation problem with two types of trucks --/
structure TransportProblem where
  driversAvailable : ℕ
  workersAvailable : ℕ
  typeATrucks : ℕ
  typeBTrucks : ℕ
  typeATruckCapacity : ℕ
  typeBTruckCapacity : ℕ
  minTonsToTransport : ℕ
  typeAWorkersRequired : ℕ
  typeBWorkersRequired : ℕ
  typeAProfit : ℕ
  typeBProfit : ℕ

/-- The solution to the transportation problem --/
structure TransportSolution where
  typeATrucksUsed : ℕ
  typeBTrucksUsed : ℕ

/-- Calculate the profit for a given solution --/
def calculateProfit (p : TransportProblem) (s : TransportSolution) : ℕ :=
  p.typeAProfit * s.typeATrucksUsed + p.typeBProfit * s.typeBTrucksUsed

/-- Check if a solution is valid for a given problem --/
def isValidSolution (p : TransportProblem) (s : TransportSolution) : Prop :=
  s.typeATrucksUsed ≤ p.typeATrucks ∧
  s.typeBTrucksUsed ≤ p.typeBTrucks ∧
  s.typeATrucksUsed * p.typeAWorkersRequired + s.typeBTrucksUsed * p.typeBWorkersRequired ≤ p.workersAvailable ∧
  s.typeATrucksUsed * p.typeATruckCapacity + s.typeBTrucksUsed * p.typeBTruckCapacity ≥ p.minTonsToTransport

/-- The main theorem stating that the maximum profit is 4900 yuan --/
theorem max_profit_is_4900 (p : TransportProblem)
  (h1 : p.driversAvailable = 12)
  (h2 : p.workersAvailable = 19)
  (h3 : p.typeATrucks = 8)
  (h4 : p.typeBTrucks = 7)
  (h5 : p.typeATruckCapacity = 10)
  (h6 : p.typeBTruckCapacity = 6)
  (h7 : p.minTonsToTransport = 72)
  (h8 : p.typeAWorkersRequired = 2)
  (h9 : p.typeBWorkersRequired = 1)
  (h10 : p.typeAProfit = 450)
  (h11 : p.typeBProfit = 350) :
  ∃ (s : TransportSolution), isValidSolution p s ∧ 
  calculateProfit p s = 4900 ∧ 
  ∀ (s' : TransportSolution), isValidSolution p s' → calculateProfit p s' ≤ 4900 := by
  sorry


end max_profit_is_4900_l3292_329245


namespace function_equality_l3292_329262

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x ≤ x) 
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) : 
  ∀ x : ℝ, f x = x := by
sorry

end function_equality_l3292_329262


namespace numerator_increase_l3292_329254

theorem numerator_increase (x y a : ℝ) : 
  x / y = 2 / 5 → 
  x + y = 5.25 → 
  (x + a) / (2 * y) = 1 / 3 → 
  a = 1 := by
sorry

end numerator_increase_l3292_329254


namespace expression_equality_l3292_329298

theorem expression_equality (y b : ℝ) (h1 : y > 0) 
  (h2 : (4 * y) / b + (3 * y) / 10 = 0.5 * y) : b = 20 := by
  sorry

end expression_equality_l3292_329298


namespace circle_area_not_tripled_l3292_329281

/-- Tripling the radius of a circle does not triple its area -/
theorem circle_area_not_tripled (r : ℝ) (h : r > 0) : π * (3 * r)^2 ≠ 3 * (π * r^2) := by
  sorry

end circle_area_not_tripled_l3292_329281


namespace solution_equation_one_solution_equation_two_solution_system_equations_l3292_329224

-- Problem 1
theorem solution_equation_one (x : ℝ) : 4 - 3 * x = 6 - 5 * x ↔ x = 1 := by sorry

-- Problem 2
theorem solution_equation_two (x : ℝ) : (x + 1) / 2 - 1 = (2 - x) / 3 ↔ x = 7 / 5 := by sorry

-- Problem 3
theorem solution_system_equations (x y : ℝ) : 3 * x - y = 7 ∧ x + 3 * y = -1 ↔ x = 2 ∧ y = -1 := by sorry

end solution_equation_one_solution_equation_two_solution_system_equations_l3292_329224


namespace cylinder_radius_with_prisms_l3292_329291

theorem cylinder_radius_with_prisms (h₁ h₂ d : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (d_pos : d > 0) 
  (h₁_eq : h₁ = 9) (h₂_eq : h₂ = 2) (d_eq : d = 23) : ∃ R : ℝ, 
  R > 0 ∧ 
  R^2 = (R - h₁)^2 + (d - x)^2 ∧ 
  R^2 = (R - h₂)^2 + x^2 ∧ 
  R = 17 :=
sorry

end cylinder_radius_with_prisms_l3292_329291


namespace sixth_diagram_shaded_fraction_l3292_329258

/-- Represents the number of shaded triangles in the nth diagram -/
def shaded_triangles (n : ℕ) : ℕ := (n - 1) ^ 2

/-- Represents the total number of triangles in the nth diagram -/
def total_triangles (n : ℕ) : ℕ := n ^ 2

/-- The fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ := shaded_triangles n / total_triangles n

theorem sixth_diagram_shaded_fraction :
  shaded_fraction 6 = 25 / 36 := by sorry

end sixth_diagram_shaded_fraction_l3292_329258


namespace beka_flew_more_than_jackson_l3292_329200

/-- The difference in miles flown between Beka and Jackson -/
def miles_difference (beka_miles jackson_miles : ℕ) : ℕ :=
  beka_miles - jackson_miles

/-- Theorem stating that Beka flew 310 miles more than Jackson -/
theorem beka_flew_more_than_jackson :
  miles_difference 873 563 = 310 := by
  sorry

end beka_flew_more_than_jackson_l3292_329200


namespace common_tangents_of_circles_l3292_329260

/-- Circle C1 with equation x² + y² - 2x - 4y - 4 = 0 -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- Circle C2 with equation x² + y² - 6x - 10y - 2 = 0 -/
def C2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 10*y - 2 = 0

/-- The number of common tangents between C1 and C2 -/
def num_common_tangents : ℕ := 2

theorem common_tangents_of_circles :
  num_common_tangents = 2 :=
sorry

end common_tangents_of_circles_l3292_329260


namespace original_decimal_l3292_329282

theorem original_decimal (x : ℝ) : (100 * x = x + 29.7) → x = 0.3 := by
  sorry

end original_decimal_l3292_329282


namespace tuesday_is_only_valid_start_day_l3292_329267

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

def advance_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (advance_days d m)

def voucher_days (start : DayOfWeek) : List DayOfWeek :=
  List.map (fun i => advance_days start (i * 7)) [0, 1, 2, 3, 4]

theorem tuesday_is_only_valid_start_day :
  ∀ (start : DayOfWeek),
    (∀ (d : DayOfWeek), d ∈ voucher_days start → d ≠ DayOfWeek.Monday) ↔
    start = DayOfWeek.Tuesday :=
sorry

end tuesday_is_only_valid_start_day_l3292_329267


namespace range_of_a_l3292_329280

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - 2 * x

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, x^a ≥ 2 * Real.exp (2*x) * f a x + Real.exp (2*x)) →
  a ≤ 2 * Real.exp 1 :=
by sorry

end range_of_a_l3292_329280


namespace sqrt_product_equals_product_l3292_329274

theorem sqrt_product_equals_product : Real.sqrt (4 * 9) = 2 * 3 := by
  sorry

end sqrt_product_equals_product_l3292_329274


namespace march_first_is_friday_l3292_329210

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Counts the number of occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

theorem march_first_is_friday (m : Month) : 
  m.days = 31 ∧ 
  countDayOccurrences m DayOfWeek.Friday = 5 ∧ 
  countDayOccurrences m DayOfWeek.Sunday = 4 → 
  m.firstDay = DayOfWeek.Friday :=
sorry

end march_first_is_friday_l3292_329210


namespace correct_addition_with_digit_change_l3292_329269

theorem correct_addition_with_digit_change :
  ∃ (d e : ℕ), d ≠ e ∧ d < 10 ∧ e < 10 ∧
  ((853697 + 930541 = 1383238 ∧ d = 8 ∧ e = 4) ∨
   (453697 + 930541 = 1383238 ∧ d = 8 ∧ e = 4)) ∧
  d + e = 12 := by
sorry

end correct_addition_with_digit_change_l3292_329269


namespace transaction_result_l3292_329212

def initial_x : ℝ := 15000
def initial_y : ℝ := 18000
def painting_value : ℝ := 15000
def first_sale_price : ℝ := 20000
def second_sale_price : ℝ := 14000
def commission_rate : ℝ := 0.05

def first_transaction_x (initial : ℝ) (sale_price : ℝ) (commission : ℝ) : ℝ :=
  initial + sale_price * (1 - commission)

def first_transaction_y (initial : ℝ) (purchase_price : ℝ) : ℝ :=
  initial - purchase_price

def second_transaction_x (cash : ℝ) (purchase_price : ℝ) : ℝ :=
  cash - purchase_price

def second_transaction_y (cash : ℝ) (sale_price : ℝ) (commission : ℝ) : ℝ :=
  cash + sale_price * (1 - commission)

theorem transaction_result :
  let x_final := second_transaction_x (first_transaction_x initial_x first_sale_price commission_rate) second_sale_price
  let y_final := second_transaction_y (first_transaction_y initial_y first_sale_price) second_sale_price commission_rate
  (x_final - initial_x = 5000) ∧ (y_final - initial_y = -6700) :=
by sorry

end transaction_result_l3292_329212


namespace mary_found_two_seashells_l3292_329278

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 7 - 5

/-- The number of seashells Keith found -/
def keith_seashells : ℕ := 5

/-- The total number of seashells Mary and Keith found together -/
def total_seashells : ℕ := 7

theorem mary_found_two_seashells :
  mary_seashells = 2 :=
sorry

end mary_found_two_seashells_l3292_329278


namespace quadratic_inequality_l3292_329203

theorem quadratic_inequality (x : ℝ) : 2 * x^2 - 6 * x - 56 > 0 ↔ x < -4 ∨ x > 7 := by
  sorry

end quadratic_inequality_l3292_329203


namespace min_guesses_bound_one_guess_sufficient_two_guesses_necessary_l3292_329266

/-- Given positive integers n and k with n > k, this function returns the minimum number
    of guesses required to determine a binary string of length n, given all binary strings
    that differ from it in exactly k positions. -/
def min_guesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

/-- Theorem stating that the minimum number of guesses is at most 2 and at least 1. -/
theorem min_guesses_bound (n k : ℕ) (h : n > k) :
  min_guesses n k = max 1 2 := by
  sorry

/-- Theorem stating that when n ≠ 2k, one guess is sufficient. -/
theorem one_guess_sufficient (n k : ℕ) (h1 : n > k) (h2 : n ≠ 2 * k) :
  min_guesses n k = 1 := by
  sorry

/-- Theorem stating that when n = 2k, two guesses are necessary and sufficient. -/
theorem two_guesses_necessary (n k : ℕ) (h1 : n > k) (h2 : n = 2 * k) :
  min_guesses n k = 2 := by
  sorry

end min_guesses_bound_one_guess_sufficient_two_guesses_necessary_l3292_329266


namespace cake_cross_section_is_rectangle_l3292_329208

/-- A cylindrical cake -/
structure Cake where
  base_diameter : ℝ
  height : ℝ

/-- The cross-section of a cake when cut along its diameter -/
inductive CrossSection
  | Rectangle
  | Circle
  | Square
  | Undetermined

/-- The shape of the cross-section when a cylindrical cake is cut along its diameter -/
def cross_section_shape (c : Cake) : CrossSection :=
  CrossSection.Rectangle

/-- Theorem: The cross-section of a cylindrical cake with base diameter 3 cm and height 9 cm, 
    when cut along its diameter, is a rectangle -/
theorem cake_cross_section_is_rectangle :
  let c : Cake := { base_diameter := 3, height := 9 }
  cross_section_shape c = CrossSection.Rectangle := by
  sorry

end cake_cross_section_is_rectangle_l3292_329208


namespace meeting_2015_same_as_first_l3292_329286

/-- Represents a point on a line segment --/
structure Point :=
  (position : ℝ)

/-- Represents a person moving on a line segment --/
structure Person :=
  (speed : ℝ)
  (startPosition : Point)
  (startTime : ℝ)

/-- Represents a meeting between two people --/
structure Meeting :=
  (position : Point)
  (time : ℝ)

/-- The theorem stating that the 2015th meeting occurs at the same point as the first meeting --/
theorem meeting_2015_same_as_first 
  (a b : Person) 
  (segment : Set Point) 
  (first_meeting last_meeting : Meeting) :
  first_meeting.position = last_meeting.position :=
sorry

end meeting_2015_same_as_first_l3292_329286


namespace negative_correlation_implies_negative_slope_l3292_329243

/-- A linear regression model with two variables -/
structure LinearRegressionModel where
  x : ℝ → ℝ  -- Independent variable
  y : ℝ → ℝ  -- Dependent variable
  a : ℝ       -- Intercept
  b : ℝ       -- Slope

/-- Definition of negative correlation between two variables -/
def NegativelyCorrelated (model : LinearRegressionModel) : Prop :=
  ∀ x1 x2, x1 < x2 → model.y x1 > model.y x2

/-- Theorem: In a linear regression model, if two variables are negatively correlated, then the slope b is negative -/
theorem negative_correlation_implies_negative_slope (model : LinearRegressionModel) 
  (h : NegativelyCorrelated model) : model.b < 0 := by
  sorry

end negative_correlation_implies_negative_slope_l3292_329243


namespace min_value_linear_program_l3292_329240

theorem min_value_linear_program :
  ∀ x y : ℝ,
  (2 * x + y - 2 ≥ 0) →
  (x - 2 * y + 4 ≥ 0) →
  (x - 1 ≤ 0) →
  ∃ (z : ℝ), z = 3 * x + 2 * y ∧ z ≥ 3 ∧ (∀ x' y' : ℝ, 
    (2 * x' + y' - 2 ≥ 0) →
    (x' - 2 * y' + 4 ≥ 0) →
    (x' - 1 ≤ 0) →
    3 * x' + 2 * y' ≥ z) :=
by sorry

end min_value_linear_program_l3292_329240


namespace gcd_of_specific_squares_l3292_329246

theorem gcd_of_specific_squares : Nat.gcd (130^2 + 240^2 + 350^2) (131^2 + 241^2 + 349^2) = 1 := by
  sorry

end gcd_of_specific_squares_l3292_329246


namespace yoongis_answer_l3292_329273

theorem yoongis_answer : ∃ x : ℝ, 5 * x = 100 ∧ x / 10 = 2 := by
  sorry

end yoongis_answer_l3292_329273


namespace constant_term_is_integer_coefficients_not_necessarily_integer_l3292_329268

/-- A real quadratic polynomial that takes integer values for all integer inputs -/
structure IntegerValuedQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  integer_valued : ∀ (x : ℤ), ∃ (y : ℤ), a * x^2 + b * x + c = y

theorem constant_term_is_integer (p : IntegerValuedQuadratic) : ∃ (n : ℤ), p.c = n := by
  sorry

theorem coefficients_not_necessarily_integer : 
  ∃ (p : IntegerValuedQuadratic), ¬(∃ (m n : ℤ), p.a = m ∧ p.b = n) := by
  sorry

end constant_term_is_integer_coefficients_not_necessarily_integer_l3292_329268


namespace amount_A_to_B_plus_ratio_l3292_329249

/-- The amount promised for a B+ grade -/
def amount_B_plus : ℝ := 5

/-- The number of courses in Paul's scorecard -/
def num_courses : ℕ := 10

/-- The flat amount received for each A+ grade -/
def amount_A_plus : ℝ := 15

/-- The maximum amount Paul could receive -/
def max_amount : ℝ := 190

/-- The amount promised for an A grade -/
noncomputable def amount_A : ℝ := 
  (max_amount - 2 * amount_A_plus) / (2 * (num_courses - 2))

/-- Theorem stating that the ratio of amount promised for an A to a B+ is 2:1 -/
theorem amount_A_to_B_plus_ratio : 
  amount_A / amount_B_plus = 2 := by sorry

end amount_A_to_B_plus_ratio_l3292_329249


namespace town_literacy_distribution_l3292_329230

theorem town_literacy_distribution :
  ∀ (T : ℝ) (M F : ℝ),
    T > 0 →
    M + F = 100 →
    0.20 * M * T + 0.325 * F * T = 0.25 * T →
    M = 60 ∧ F = 40 := by
  sorry

end town_literacy_distribution_l3292_329230


namespace prime_triplet_l3292_329213

theorem prime_triplet (p : ℤ) : 
  Prime p ∧ Prime (p + 2) ∧ Prime (p + 4) → p = 3 :=
by sorry

end prime_triplet_l3292_329213


namespace product_eleven_cubed_sum_l3292_329250

theorem product_eleven_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 11^3 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 133 := by sorry

end product_eleven_cubed_sum_l3292_329250


namespace at_most_one_root_l3292_329220

-- Define a monotonically increasing function on an interval
def MonoIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Theorem statement
theorem at_most_one_root (f : ℝ → ℝ) (a b : ℝ) (h : MonoIncreasing f a b) :
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 :=
by
  sorry

end at_most_one_root_l3292_329220


namespace scores_mode_and_median_l3292_329247

def scores : List ℕ := [95, 97, 96, 97, 99, 98]

/-- The mode of a list of natural numbers -/
def mode (l : List ℕ) : ℕ := sorry

/-- The median of a list of natural numbers -/
def median (l : List ℕ) : ℚ := sorry

theorem scores_mode_and_median :
  mode scores = 97 ∧ median scores = 97 := by sorry

end scores_mode_and_median_l3292_329247


namespace zeros_of_continuous_function_l3292_329233

theorem zeros_of_continuous_function 
  (f : ℝ → ℝ) (hf : Continuous f) 
  (a b c : ℝ) (hab : a < b) (hbc : b < c)
  (hab_sign : f a * f b < 0) (hbc_sign : f b * f c < 0) :
  ∃ (n : ℕ), n > 0 ∧ Even n ∧ 
  (∃ (S : Finset ℝ), S.card = n ∧ 
    (∀ x ∈ S, a < x ∧ x < c ∧ f x = 0)) :=
sorry

end zeros_of_continuous_function_l3292_329233


namespace sum_of_cubes_roots_l3292_329299

theorem sum_of_cubes_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - a*x₁ + a + 2 = 0 ∧ 
                x₂^2 - a*x₂ + a + 2 = 0 ∧ 
                x₁^3 + x₂^3 = -8) ↔ 
  a = -2 := by
sorry

end sum_of_cubes_roots_l3292_329299


namespace distance_is_sqrt_152_l3292_329214

/-- The distance between two adjacent parallel lines intersecting a circle -/
def distance_between_lines (r : ℝ) (d : ℝ) : Prop :=
  ∃ (chord1 chord2 chord3 : ℝ),
    chord1 = 40 ∧ chord2 = 36 ∧ chord3 = 34 ∧
    40 * r^2 = 800 + 10 * d^2 ∧
    36 * r^2 = 648 + 9 * d^2 ∧
    d = Real.sqrt 152

/-- Theorem stating that the distance between two adjacent parallel lines is √152 -/
theorem distance_is_sqrt_152 :
  ∃ (r : ℝ), distance_between_lines r (Real.sqrt 152) :=
sorry

end distance_is_sqrt_152_l3292_329214


namespace pentagon_ink_length_l3292_329218

/-- Ink length of a regular pentagon with side length n -/
def inkLength (n : ℕ) : ℕ := 5 * n

theorem pentagon_ink_length :
  (inkLength 4 = 20) ∧
  (inkLength 9 - inkLength 8 = 5) ∧
  (inkLength 100 = 500) := by
  sorry

end pentagon_ink_length_l3292_329218


namespace somu_age_problem_l3292_329287

/-- Proves that Somu was one-fifth of his father's age 6 years ago -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) : 
  somu_age = 12 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 6 := by
sorry

end somu_age_problem_l3292_329287


namespace diagonal_length_range_l3292_329275

/-- Represents a quadrilateral with given side lengths and an integer diagonal -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (EG : ℤ)

/-- The theorem stating the possible values for the diagonal EG -/
theorem diagonal_length_range (q : Quadrilateral)
  (h1 : q.EF = 7)
  (h2 : q.FG = 12)
  (h3 : q.GH = 7)
  (h4 : q.HE = 15) :
  9 ≤ q.EG ∧ q.EG ≤ 18 :=
by sorry

end diagonal_length_range_l3292_329275


namespace water_consumption_difference_l3292_329217

/-- The yearly water consumption difference between two schools -/
theorem water_consumption_difference 
  (chunlei_daily : ℕ) -- Daily water consumption of Chunlei Central Elementary School
  (days_per_year : ℕ) -- Number of days in a year
  (h1 : chunlei_daily = 111) -- Chunlei's daily consumption is 111 kg
  (h2 : days_per_year = 365) -- A year has 365 days
  : 
  chunlei_daily * days_per_year - (chunlei_daily / 3) * days_per_year = 26910 :=
by sorry

end water_consumption_difference_l3292_329217


namespace tenth_term_of_sequence_l3292_329223

theorem tenth_term_of_sequence (a : ℕ → ℚ) :
  (∀ n : ℕ, a n = (-1)^(n+1) * (2*n) / (2*n+1)) →
  a 10 = -20 / 21 :=
by
  sorry

end tenth_term_of_sequence_l3292_329223


namespace part_i_part_ii_l3292_329241

-- Define the function f
def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

-- Part I
theorem part_i : 
  let m : ℝ := -1
  let a : ℝ := -1
  {x : ℝ | f x a m ≥ x} = {x : ℝ | x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 2)} := by sorry

-- Part II
theorem part_ii (m : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2) 
  (h4 : a ≤ -3 ∨ a ≥ 3) : 
  m = 1/3 := by sorry

end part_i_part_ii_l3292_329241


namespace total_accidents_across_highways_l3292_329288

/-- Represents the accident rate and traffic data for a highway -/
structure HighwayData where
  accidents : ℕ
  vehicles : ℕ
  totalTraffic : ℕ

/-- Calculates the number of accidents for a given highway -/
def calculateAccidents (data : HighwayData) : ℕ :=
  (data.accidents * data.totalTraffic) / data.vehicles

/-- The data for Highway A -/
def highwayA : HighwayData :=
  { accidents := 75, vehicles := 100000000, totalTraffic := 2500000000 }

/-- The data for Highway B -/
def highwayB : HighwayData :=
  { accidents := 50, vehicles := 80000000, totalTraffic := 1600000000 }

/-- The data for Highway C -/
def highwayC : HighwayData :=
  { accidents := 90, vehicles := 200000000, totalTraffic := 1900000000 }

/-- Theorem stating that the total number of accidents across all three highways is 3730 -/
theorem total_accidents_across_highways :
  calculateAccidents highwayA + calculateAccidents highwayB + calculateAccidents highwayC = 3730 :=
by
  sorry

end total_accidents_across_highways_l3292_329288


namespace even_function_implies_cubic_l3292_329295

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (m+1)x^2 + (m-2)x -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m + 1) * x^2 + (m - 2) * x

theorem even_function_implies_cubic (m : ℝ) :
  IsEven (f m) → f m = fun x ↦ 3 * x^2 := by
  sorry

end even_function_implies_cubic_l3292_329295


namespace gcd_problem_l3292_329211

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ b = k * 1177) :
  Nat.gcd (Int.natAbs (2 * b^2 + 31 * b + 71)) (Int.natAbs (b + 15)) = 1 := by
sorry

end gcd_problem_l3292_329211


namespace butterflies_in_garden_l3292_329290

theorem butterflies_in_garden (initial : ℕ) (fraction : ℚ) (remaining : ℕ) : 
  initial = 9 → fraction = 1/3 → remaining = initial - (initial * fraction).floor → remaining = 6 := by
  sorry

end butterflies_in_garden_l3292_329290


namespace bicycle_has_four_wheels_l3292_329271

-- Define the universe of objects
variable (Object : Type)

-- Define predicates
variable (isCar : Object → Prop)
variable (hasFourWheels : Object → Prop)

-- Define a specific object
variable (bicycle : Object)

-- Theorem statement
theorem bicycle_has_four_wheels 
  (all_cars_have_four_wheels : ∀ x, isCar x → hasFourWheels x)
  (bicycle_is_car : isCar bicycle) :
  hasFourWheels bicycle :=
by
  sorry


end bicycle_has_four_wheels_l3292_329271


namespace percentage_subtraction_equivalence_l3292_329222

theorem percentage_subtraction_equivalence (a : ℝ) : 
  a - (0.05 * a) = 0.95 * a := by sorry

end percentage_subtraction_equivalence_l3292_329222


namespace inequality_proof_l3292_329297

theorem inequality_proof (a b : ℝ) (h1 : 0 < b) (h2 : b < 1) (h3 : 1 < a) :
  a * b^2 < a * b ∧ a * b < a :=
by sorry

end inequality_proof_l3292_329297


namespace three_numbers_product_sum_l3292_329242

theorem three_numbers_product_sum (x y z : ℝ) : 
  (x * y + x + y = 8) ∧ 
  (y * z + y + z = 15) ∧ 
  (x * z + x + z = 24) → 
  x = 209 / 25 ∧ y = 7 ∧ z = 17 := by
sorry

end three_numbers_product_sum_l3292_329242


namespace samantha_routes_l3292_329279

/-- The number of ways to arrange n blocks in two directions --/
def arrangeBlocks (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- The number of diagonal paths through the park --/
def diagonalPaths : ℕ := 2

/-- The total number of routes Samantha can take --/
def totalRoutes : ℕ := arrangeBlocks 3 * diagonalPaths * arrangeBlocks 3

theorem samantha_routes :
  totalRoutes = 800 := by
  sorry

end samantha_routes_l3292_329279


namespace two_digit_number_problem_l3292_329283

theorem two_digit_number_problem : ∃ (n : ℕ), 
  (n ≥ 10 ∧ n < 100) ∧  -- two-digit number
  (n % 10 = (n / 10) + 3) ∧  -- units digit is 3 greater than tens digit
  ((n % 10)^2 + (n / 10)^2 = (n % 10) + (n / 10) + 18) ∧  -- sum of squares condition
  n = 47 :=
by sorry

end two_digit_number_problem_l3292_329283


namespace five_single_beds_weight_l3292_329252

/-- The weight of a single bed in kg -/
def single_bed_weight : ℝ := sorry

/-- The weight of a double bed in kg -/
def double_bed_weight : ℝ := sorry

/-- A double bed is 10 kg heavier than a single bed -/
axiom double_bed_heavier : double_bed_weight = single_bed_weight + 10

/-- The total weight of 2 single beds and 4 double beds is 100 kg -/
axiom total_weight : 2 * single_bed_weight + 4 * double_bed_weight = 100

theorem five_single_beds_weight :
  5 * single_bed_weight = 50 := by sorry

end five_single_beds_weight_l3292_329252


namespace combinatorial_identities_l3292_329263

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of k-permutations of n -/
def permutation (n k : ℕ) : ℕ := sorry

theorem combinatorial_identities :
  (∀ n k : ℕ, k > 0 → k * binomial n k = n * binomial (n - 1) (k - 1)) ∧
  binomial 2014 2013 + permutation 5 3 = 2074 := by sorry

end combinatorial_identities_l3292_329263


namespace three_student_committees_from_eight_l3292_329231

theorem three_student_committees_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end three_student_committees_from_eight_l3292_329231


namespace probability_point_in_circle_l3292_329225

/-- The probability of a randomly selected point in a square with side length 6 
    being within 2 units of the center is π/9. -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 6 → circle_radius = 2 → 
  (π * circle_radius^2) / (square_side^2) = π / 9 := by
  sorry

end probability_point_in_circle_l3292_329225


namespace cos_135_degrees_l3292_329207

theorem cos_135_degrees : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_degrees_l3292_329207


namespace butterfat_percentage_in_cream_l3292_329239

/-- The percentage of butterfat in cream when mixed with skim milk to achieve a target butterfat percentage -/
theorem butterfat_percentage_in_cream 
  (cream_volume : ℝ) 
  (skim_milk_volume : ℝ) 
  (skim_milk_butterfat : ℝ) 
  (final_mixture_butterfat : ℝ) 
  (h1 : cream_volume = 1)
  (h2 : skim_milk_volume = 3)
  (h3 : skim_milk_butterfat = 5.5)
  (h4 : final_mixture_butterfat = 6.5)
  (h5 : cream_volume + skim_milk_volume = 4) :
  ∃ (cream_butterfat : ℝ), 
    cream_butterfat = 9.5 ∧ 
    cream_butterfat * cream_volume + skim_milk_butterfat * skim_milk_volume = 
    final_mixture_butterfat * (cream_volume + skim_milk_volume) := by
  sorry


end butterfat_percentage_in_cream_l3292_329239


namespace circle_passes_through_points_l3292_329248

/-- A circle passing through three points -/
structure Circle where
  D : ℝ
  E : ℝ

/-- Check if a point lies on the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y = 0

/-- The specific circle we're interested in -/
def our_circle : Circle := { D := -4, E := -6 }

/-- Theorem stating that our_circle passes through the given points -/
theorem circle_passes_through_points : 
  (our_circle.contains 0 0) ∧ 
  (our_circle.contains 4 0) ∧ 
  (our_circle.contains (-1) 1) := by
  sorry


end circle_passes_through_points_l3292_329248


namespace unique_divisible_by_18_l3292_329205

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem unique_divisible_by_18 : 
  ∀ n : ℕ, n < 10 → 
    (is_divisible_by (3140 + n) 18 ↔ n = 4) :=
by sorry

end unique_divisible_by_18_l3292_329205


namespace parallel_vectors_result_symmetric_function_range_l3292_329234

-- Part 1
theorem parallel_vectors_result (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, Real.cos x)
  let b : ℝ × ℝ := (3, -1)
  (∃ (k : ℝ), a = k • b) →
  2 * (Real.sin x)^2 - 3 * (Real.cos x)^2 = 3/2 := by sorry

-- Part 2
theorem symmetric_function_range (x m : ℝ) :
  let a : ℝ → ℝ × ℝ := λ t => (Real.sin t, m * Real.cos t)
  let b : ℝ × ℝ := (3, -1)
  let f : ℝ → ℝ := λ t => (a t).1 * b.1 + (a t).2 * b.2
  (∀ t, f (2*π/3 - t) = f (2*π/3 + t)) →
  ∃ y ∈ Set.Icc (-Real.sqrt 3) (2 * Real.sqrt 3),
    ∃ x ∈ Set.Icc (π/8) (2*π/3), f (2*x) = y := by sorry

end parallel_vectors_result_symmetric_function_range_l3292_329234


namespace art_arrangement_count_l3292_329232

/-- Represents the number of calligraphy works -/
def calligraphy_count : ℕ := 2

/-- Represents the number of painting works -/
def painting_count : ℕ := 2

/-- Represents the number of architectural designs -/
def architecture_count : ℕ := 1

/-- Represents the total number of art pieces -/
def total_art_pieces : ℕ := calligraphy_count + painting_count + architecture_count

/-- Calculates the number of arrangements of art pieces -/
def calculate_arrangements : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements is 24 -/
theorem art_arrangement_count : calculate_arrangements = 24 := by
  sorry

end art_arrangement_count_l3292_329232


namespace sin_120_degrees_l3292_329259

theorem sin_120_degrees : Real.sin (2 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l3292_329259


namespace factorial_calculation_l3292_329251

theorem factorial_calculation : (Nat.factorial 11) / (Nat.factorial 10) * 12 = 132 := by
  sorry

end factorial_calculation_l3292_329251


namespace arithmetic_computation_l3292_329264

theorem arithmetic_computation : 143 - 13 + 31 + 17 = 178 := by
  sorry

end arithmetic_computation_l3292_329264


namespace ellipse_hyperbola_ab_product_l3292_329261

-- Define the ellipse and hyperbola equations
def ellipse_equation (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def hyperbola_equation (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the theorem
theorem ellipse_hyperbola_ab_product 
  (a b : ℝ) 
  (h_ellipse : ∃ (x y : ℝ), ellipse_equation x y a b ∧ (x = 0 ∧ y = 5 ∨ x = 0 ∧ y = -5))
  (h_hyperbola : ∃ (x y : ℝ), hyperbola_equation x y a b ∧ (x = 7 ∧ y = 0 ∨ x = -7 ∧ y = 0)) :
  |a * b| = 2 * Real.sqrt 111 := by
  sorry

end ellipse_hyperbola_ab_product_l3292_329261


namespace max_side_length_triangle_l3292_329255

theorem max_side_length_triangle (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Different side lengths
  a + b + c = 24 →  -- Perimeter is 24
  a ≤ 11 ∧ b ≤ 11 ∧ c ≤ 11 →  -- Maximum side length is 11
  (a + b > c ∧ b + c > a ∧ a + c > b) →  -- Triangle inequality
  ∃ (x y z : ℕ), x + y + z = 24 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x ≤ 11 ∧ y ≤ 11 ∧ z ≤ 11 ∧ 
    (x + y > z ∧ y + z > x ∧ x + z > y) ∧
    (∀ w : ℕ, w > 11 → ¬(∃ u v : ℕ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
      u + v + w = 24 ∧ u + v > w ∧ v + w > u ∧ u + w > v)) :=
by sorry

end max_side_length_triangle_l3292_329255


namespace u_2002_equals_2_l3292_329296

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 4
| 4 => 2
| 5 => 1
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence u
def u : ℕ → ℕ
| 0 => 3
| (n + 1) => g (u n)

-- State the theorem
theorem u_2002_equals_2 : u 2002 = 2 := by
  sorry

end u_2002_equals_2_l3292_329296


namespace stock_price_change_l3292_329289

theorem stock_price_change (initial_price : ℝ) (h_pos : initial_price > 0) : 
  let price_after_decrease := initial_price * (1 - 0.08)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 1.2 := by
sorry

end stock_price_change_l3292_329289


namespace unique_solution_l3292_329226

theorem unique_solution : ∃! n : ℕ, n > 0 ∧ Nat.lcm n 150 = Nat.gcd n 150 + 600 ∧ n = 675 := by
  sorry

end unique_solution_l3292_329226


namespace vasyas_numbers_l3292_329206

theorem vasyas_numbers (x y : ℝ) : 
  x + y = x * y ∧ x + y = x / y → x = (1 : ℝ) / 2 ∧ y = -1 := by sorry

end vasyas_numbers_l3292_329206


namespace johns_cost_per_minute_l3292_329204

/-- Calculates the cost per minute for long distance calls -/
def cost_per_minute (monthly_fee : ℚ) (total_bill : ℚ) (minutes_billed : ℚ) : ℚ :=
  (total_bill - monthly_fee) / minutes_billed

/-- Theorem stating that John's cost per minute for long distance calls is $0.25 -/
theorem johns_cost_per_minute :
  let monthly_fee : ℚ := 5
  let total_bill : ℚ := 12.02
  let minutes_billed : ℚ := 28.08
  cost_per_minute monthly_fee total_bill minutes_billed = 0.25 := by
sorry

end johns_cost_per_minute_l3292_329204


namespace quadruple_equation_solutions_l3292_329221

theorem quadruple_equation_solutions :
  ∀ (a b c d : ℝ),
  (b + c + d)^2010 = 3 * a ∧
  (a + c + d)^2010 = 3 * b ∧
  (a + b + d)^2010 = 3 * c ∧
  (a + b + c)^2010 = 3 * d →
  ((a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
   (a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3)) :=
by sorry

end quadruple_equation_solutions_l3292_329221


namespace set_equality_l3292_329229

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {3, 5}

-- Theorem statement
theorem set_equality : U = A ∪ (U \ B) := by sorry

end set_equality_l3292_329229
