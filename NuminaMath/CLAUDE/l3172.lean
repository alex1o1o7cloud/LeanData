import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_of_x_is_negative_one_l3172_317290

-- Define the expression as a polynomial
def expression (x : ℝ) : ℝ := 5 * (x - 6) + 3 * (8 - 3 * x^2 + 7 * x) - 9 * (3 * x - 2)

-- Theorem stating that the coefficient of x in the expression is -1
theorem coefficient_of_x_is_negative_one :
  ∃ (a b c : ℝ), expression = fun x => a * x^2 + (-1) * x + c :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_negative_one_l3172_317290


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3172_317219

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- Theorem: In an arithmetic sequence, if 3a₉ - a₁₅ - a₃ = 20, then 2a₈ - a₇ = 20 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) 
  (eq : 3 * a 9 - a 15 - a 3 = 20) : 
  2 * a 8 - a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3172_317219


namespace NUMINAMATH_CALUDE_square_root_of_one_fourth_l3172_317249

theorem square_root_of_one_fourth : ∃ x : ℚ, x^2 = (1/4 : ℚ) ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_one_fourth_l3172_317249


namespace NUMINAMATH_CALUDE_positive_solution_sum_l3172_317278

theorem positive_solution_sum (a b : ℕ+) (x : ℤ) : 
  x > 0 → x = Int.sqrt a - b → x^2 - 10*x = 39 → a + b = 69 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_sum_l3172_317278


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3172_317235

theorem rectangle_area_increase (x y : ℝ) 
  (area_eq : x * y = 180)
  (perimeter_eq : 2 * x + 2 * y = 54) :
  (x + 6) * (y + 6) = 378 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3172_317235


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l3172_317255

theorem complex_on_real_axis (a : ℝ) : 
  Complex.im ((1 + Complex.I) * (a + Complex.I)) = 0 ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l3172_317255


namespace NUMINAMATH_CALUDE_domain_of_f_l3172_317242

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arccos (Real.sin x))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l3172_317242


namespace NUMINAMATH_CALUDE_grocer_purchase_price_l3172_317287

/-- Represents the price at which the grocer purchased 3 pounds of bananas -/
def purchase_price : ℝ := sorry

/-- Represents the total quantity of bananas purchased in pounds -/
def total_quantity : ℝ := 72

/-- Represents the profit made by the grocer -/
def profit : ℝ := 6

/-- Represents the selling price of 4 pounds of bananas -/
def selling_price : ℝ := 1

/-- Theorem stating that the purchase price for 3 pounds of bananas is $0.50 -/
theorem grocer_purchase_price : purchase_price = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_grocer_purchase_price_l3172_317287


namespace NUMINAMATH_CALUDE_gcd_repeated_six_digit_l3172_317213

def is_repeated_six_digit (n : ℕ) : Prop :=
  ∃ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ n = 1001 * m

theorem gcd_repeated_six_digit :
  ∃ g : ℕ, ∀ n : ℕ, is_repeated_six_digit n → Nat.gcd n g = g ∧ g = 1001 :=
sorry

end NUMINAMATH_CALUDE_gcd_repeated_six_digit_l3172_317213


namespace NUMINAMATH_CALUDE_group_size_problem_l3172_317259

theorem group_size_problem (x : ℕ) : 
  (5 * x + 45 = 7 * x + 3) → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l3172_317259


namespace NUMINAMATH_CALUDE_speed_ratio_is_four_thirds_l3172_317229

/-- Two runners in a race where one gets a head start -/
structure Race where
  length : ℝ
  speed_a : ℝ
  speed_b : ℝ
  head_start : ℝ

/-- The race ends in a dead heat -/
def dead_heat (r : Race) : Prop :=
  r.length / r.speed_a = (r.length - r.head_start) / r.speed_b

/-- The head start is 0.25 of the race length -/
def quarter_head_start (r : Race) : Prop :=
  r.head_start = 0.25 * r.length

theorem speed_ratio_is_four_thirds (r : Race) 
  (h1 : dead_heat r) (h2 : quarter_head_start r) : 
  r.speed_a / r.speed_b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_is_four_thirds_l3172_317229


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3172_317271

theorem quadratic_minimum (x : ℝ) (h : x > 0) : x^2 - 2*x + 3 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3172_317271


namespace NUMINAMATH_CALUDE_atlantis_population_growth_l3172_317291

def initial_year : Nat := 2000
def initial_population : Nat := 400
def island_capacity : Nat := 15000
def years_to_check : Nat := 200

def population_after_n_cycles (n : Nat) : Nat :=
  initial_population * 2^n

theorem atlantis_population_growth :
  ∃ (y : Nat), y ≤ years_to_check ∧ 
  population_after_n_cycles (y / 40) ≥ island_capacity :=
sorry

end NUMINAMATH_CALUDE_atlantis_population_growth_l3172_317291


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l3172_317269

/-- The minimum distance between a point on the line x - y - 4 = 0 and a point on the parabola x² = 4y is (3√2)/2 -/
theorem min_distance_line_parabola :
  let line := {p : ℝ × ℝ | p.1 - p.2 = 4}
  let parabola := {p : ℝ × ℝ | p.1^2 = 4 * p.2}
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ parabola ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ parabola →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l3172_317269


namespace NUMINAMATH_CALUDE_parallelogram_grid_non_congruent_triangles_l3172_317251

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the parallelogram grid array -/
def ParallelogramGrid : List Point := [
  ⟨0, 0⟩,   -- Point 1
  ⟨1, 0⟩,   -- Point 2
  ⟨1.5, 0.5⟩, -- Point 3
  ⟨2.5, 0.5⟩, -- Point 4
  ⟨0.5, 0.25⟩, -- Point 5 (midpoint)
  ⟨1.75, 0.25⟩, -- Point 6 (midpoint)
  ⟨1.75, 0⟩, -- Point 7 (midpoint)
  ⟨1.25, 0.25⟩  -- Point 8 (center)
]

/-- Determines if two triangles are congruent -/
def areTrianglesCongruent (t1 t2 : List Point) : Bool :=
  sorry -- Implementation details omitted

/-- Counts the number of non-congruent triangles in the grid -/
def countNonCongruentTriangles (grid : List Point) : Nat :=
  sorry -- Implementation details omitted

/-- Theorem: The number of non-congruent triangles in the parallelogram grid is 9 -/
theorem parallelogram_grid_non_congruent_triangles :
  countNonCongruentTriangles ParallelogramGrid = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_grid_non_congruent_triangles_l3172_317251


namespace NUMINAMATH_CALUDE_candy_game_solution_l3172_317248

/-- 
Given a game where:
- 50 questions are asked
- Correct answers result in gaining 7 candies
- Incorrect answers result in losing 3 candies
- The net change in candies is zero

Prove that the number of correctly answered questions is 15.
-/
theorem candy_game_solution (total_questions : Nat) 
  (correct_reward : Nat) (incorrect_penalty : Nat) 
  (x : Nat) : 
  total_questions = 50 → 
  correct_reward = 7 → 
  incorrect_penalty = 3 → 
  x * correct_reward = (total_questions - x) * incorrect_penalty → 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_game_solution_l3172_317248


namespace NUMINAMATH_CALUDE_movie_attendees_l3172_317275

theorem movie_attendees (people_per_car : ℕ) (cars_needed : ℕ) (h1 : people_per_car = 6) (h2 : cars_needed = 18) : 
  people_per_car * cars_needed = 108 := by
  sorry

end NUMINAMATH_CALUDE_movie_attendees_l3172_317275


namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l3172_317299

/-- The number of peaches Mike initially had -/
def initial_peaches : ℕ := 34

/-- The total number of peaches Mike has after picking more -/
def total_peaches : ℕ := 86

/-- The number of peaches Mike picked -/
def picked_peaches : ℕ := total_peaches - initial_peaches

theorem mike_picked_52_peaches : picked_peaches = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l3172_317299


namespace NUMINAMATH_CALUDE_f_has_two_real_roots_l3172_317296

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 3

/-- Theorem stating that f has exactly two real roots -/
theorem f_has_two_real_roots : ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_two_real_roots_l3172_317296


namespace NUMINAMATH_CALUDE_sin_increasing_in_interval_l3172_317288

theorem sin_increasing_in_interval :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x - π / 6)
  ∀ x y, -π/6 < x ∧ x < y ∧ y < π/3 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_sin_increasing_in_interval_l3172_317288


namespace NUMINAMATH_CALUDE_choose_two_from_ten_l3172_317293

/-- The number of ways to choose 2 colors out of 10 colors -/
def choose_colors (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: Choosing 2 colors out of 10 results in 45 combinations -/
theorem choose_two_from_ten :
  choose_colors 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_ten_l3172_317293


namespace NUMINAMATH_CALUDE_max_almonds_in_mixture_l3172_317284

/-- Represents the composition of the nut mixture -/
structure NutMixture where
  almonds : ℕ
  walnuts : ℕ
  cashews : ℕ
  pistachios : ℕ

/-- Represents the cost per pound of each nut type -/
structure NutCosts where
  almonds : ℚ
  walnuts : ℚ
  cashews : ℚ
  pistachios : ℚ

/-- Calculates the total cost of a given nut mixture -/
def totalCost (mixture : NutMixture) (costs : NutCosts) : ℚ :=
  mixture.almonds * costs.almonds +
  mixture.walnuts * costs.walnuts +
  mixture.cashews * costs.cashews +
  mixture.pistachios * costs.pistachios

/-- Theorem stating the maximum possible pounds of almonds in the mixture -/
theorem max_almonds_in_mixture
  (mixture : NutMixture)
  (costs : NutCosts)
  (budget : ℚ)
  (total_weight : ℕ)
  (h_composition : mixture.almonds = 5 ∧ mixture.walnuts = 2 ∧ mixture.cashews = 3 ∧ mixture.pistachios = 4)
  (h_costs : costs.almonds = 6 ∧ costs.walnuts = 5 ∧ costs.cashews = 8 ∧ costs.pistachios = 10)
  (h_budget : budget = 1500)
  (h_total_weight : total_weight = 800)
  (h_almond_percentage : (mixture.almonds : ℚ) / (mixture.almonds + mixture.walnuts + mixture.cashews + mixture.pistachios) ≥ 30 / 100) :
  (mixture.almonds : ℚ) / (mixture.almonds + mixture.walnuts + mixture.cashews + mixture.pistachios) * total_weight ≤ 240 ∧
  totalCost mixture costs ≤ budget :=
sorry

end NUMINAMATH_CALUDE_max_almonds_in_mixture_l3172_317284


namespace NUMINAMATH_CALUDE_max_duck_moves_l3172_317214

/-- 
Given positive integers a, b, and c representing the number of ducks 
picking rock, paper, and scissors respectively in a circular arrangement, 
the maximum number of possible moves according to the rock-paper-scissors 
switching rules is max(a × b, b × c, c × a).
-/
theorem max_duck_moves (a b c : ℕ+) : 
  ∃ (max_moves : ℕ), max_moves = max (a * b) (max (b * c) (c * a)) ∧
  ∀ (moves : ℕ), moves ≤ max_moves := by
sorry


end NUMINAMATH_CALUDE_max_duck_moves_l3172_317214


namespace NUMINAMATH_CALUDE_range_of_m_l3172_317220

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x - 2 - m < 0) → m > -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3172_317220


namespace NUMINAMATH_CALUDE_shopkeeper_card_decks_l3172_317292

theorem shopkeeper_card_decks 
  (total_cards : ℕ) 
  (additional_cards : ℕ) 
  (cards_per_deck : ℕ) 
  (h1 : total_cards = 160)
  (h2 : additional_cards = 4)
  (h3 : cards_per_deck = 52) :
  (total_cards - additional_cards) / cards_per_deck = 3 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_card_decks_l3172_317292


namespace NUMINAMATH_CALUDE_range_of_a_l3172_317295

-- Define the conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x a : ℝ) : Prop := x < a

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x)

-- Theorem statement
theorem range_of_a (h : sufficient_not_necessary p (q · a)) :
  ∀ y : ℝ, y ≥ 2 ↔ ∃ x : ℝ, a = y := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3172_317295


namespace NUMINAMATH_CALUDE_basketball_shooting_improvement_l3172_317207

theorem basketball_shooting_improvement (initial_made initial_total fifth_game_shots : ℕ) 
  (new_average : ℚ) : 
  initial_made = 15 →
  initial_total = 45 →
  fifth_game_shots = 15 →
  new_average = 2/5 →
  (initial_made : ℚ) / initial_total = 1/3 →
  ∃ (fifth_game_made : ℕ), 
    (initial_made + fifth_game_made : ℚ) / (initial_total + fifth_game_shots) = new_average ∧
    fifth_game_made = 9 := by
  sorry

end NUMINAMATH_CALUDE_basketball_shooting_improvement_l3172_317207


namespace NUMINAMATH_CALUDE_train_distance_theorem_l3172_317241

/-- The distance between two stations given the conditions of two trains meeting --/
theorem train_distance_theorem (v₁ v₂ : ℝ) (d : ℝ) :
  v₁ > 0 → v₂ > 0 →
  v₁ = 20 →
  v₂ = 25 →
  d = 75 →
  (∃ (t : ℝ), t > 0 ∧ v₁ * t + (v₂ * t - d) = v₁ * t + v₂ * t) →
  v₁ * t + v₂ * t = 675 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l3172_317241


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l3172_317243

theorem rhombus_longest_diagonal (area : ℝ) (ratio_long : ℝ) (ratio_short : ℝ) :
  area = 108 →
  ratio_long = 3 →
  ratio_short = 2 →
  let diagonal_long := ratio_long * (2 * area / (ratio_long * ratio_short)) ^ (1/2 : ℝ)
  diagonal_long = 18 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l3172_317243


namespace NUMINAMATH_CALUDE_two_thirds_plus_six_l3172_317221

theorem two_thirds_plus_six (x : ℝ) : x = 6 → (2 / 3 * x) + 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_plus_six_l3172_317221


namespace NUMINAMATH_CALUDE_pump_out_time_for_specific_basement_l3172_317234

/-- Represents the dimensions and flooding of a basement -/
structure Basement :=
  (length : ℝ)
  (width : ℝ)
  (depth_inches : ℝ)

/-- Represents a water pump -/
structure Pump :=
  (rate : ℝ)  -- gallons per minute

/-- Calculates the time required to pump out a flooded basement -/
def pump_out_time (b : Basement) (pumps : List Pump) (cubic_foot_to_gallon : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time required to pump out the specific basement -/
theorem pump_out_time_for_specific_basement :
  let basement := Basement.mk 40 20 24
  let pumps := [Pump.mk 10, Pump.mk 10, Pump.mk 10]
  pump_out_time basement pumps 7.5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_pump_out_time_for_specific_basement_l3172_317234


namespace NUMINAMATH_CALUDE_equality_condition_for_squared_sum_equals_product_sum_l3172_317227

theorem equality_condition_for_squared_sum_equals_product_sum (a b c : ℝ) :
  (a^2 + b^2 + c^2 = a*b + b*c + c*a) ↔ (a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_equality_condition_for_squared_sum_equals_product_sum_l3172_317227


namespace NUMINAMATH_CALUDE_circle_condition_l3172_317257

/-- A circle in the xy-plane can be represented by the equation x^2 + y^2 - x + y + m = 0,
    where m is a real number. This theorem states that for the equation to represent a circle,
    the value of m must be less than 1/4. -/
theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ∧ 
   ∀ (a b : ℝ), (a - (1/2))^2 + (b - (1/2))^2 = ((1/2)^2 + (1/2)^2 - m)) →
  m < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l3172_317257


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l3172_317237

/-- A linear function y = mx - 1 passing through the second, third, and fourth quadrants implies m < 0 -/
theorem linear_function_quadrants (m : ℝ) : 
  (∀ x y : ℝ, y = m * x - 1 →
    ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) →
  m < 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l3172_317237


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3172_317260

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3172_317260


namespace NUMINAMATH_CALUDE_optimal_strategy_result_l3172_317226

/-- Represents a strategy for placing numbers in quadratic expressions -/
structure Strategy where
  place : Nat → Nat → ℝ → ℝ
  
/-- The game state -/
structure GameState where
  n : Nat
  turn : Nat
  equations : List (Option ℝ × Option ℝ × Option ℝ)

/-- The result of playing the game -/
def playGame (n : Nat) (s1 s2 : Strategy) : Nat :=
  sorry

/-- Theorem statement -/
theorem optimal_strategy_result (n : Nat) (h : Odd n) :
  ∃ (s1 : Strategy), ∀ (s2 : Strategy), 
    playGame n s1 s2 = (n + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_result_l3172_317226


namespace NUMINAMATH_CALUDE_father_son_age_difference_l3172_317253

theorem father_son_age_difference (son_age father_age : ℕ) : 
  father_age = 33 →
  father_age = 3 * son_age + (father_age - 3 * son_age) →
  father_age + 3 = 2 * (son_age + 3) + 10 →
  father_age - 3 * son_age = 3 :=
by sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l3172_317253


namespace NUMINAMATH_CALUDE_count_seating_arrangements_l3172_317239

/-- Represents a seating arrangement in a 5x5 classroom -/
def SeatingArrangement := Fin 5 → Fin 5 → Bool

/-- A seating arrangement is valid if for each occupied desk, either its row or column is full -/
def is_valid (arrangement : SeatingArrangement) : Prop :=
  ∀ i j, arrangement i j → 
    (∀ k, arrangement i k) ∨ (∀ k, arrangement k j)

/-- The total number of valid seating arrangements -/
def total_arrangements : ℕ := sorry

theorem count_seating_arrangements :
  total_arrangements = 962 := by sorry

end NUMINAMATH_CALUDE_count_seating_arrangements_l3172_317239


namespace NUMINAMATH_CALUDE_garden_length_l3172_317258

/-- Given a square playground and a rectangular garden, proves the length of the garden
    when the total fencing is known. -/
theorem garden_length
  (playground_side : ℕ)
  (garden_width : ℕ)
  (total_fencing : ℕ)
  (h1 : playground_side = 27)
  (h2 : garden_width = 9)
  (h3 : total_fencing = 150)
  (h4 : 4 * playground_side + 2 * garden_width + 2 * (total_fencing - 4 * playground_side - 2 * garden_width) / 2 = total_fencing) :
  (total_fencing - 4 * playground_side - 2 * garden_width) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_garden_length_l3172_317258


namespace NUMINAMATH_CALUDE_car_travel_distance_l3172_317294

/-- Proves that two cars traveling at different speeds for different times cover the same distance of 600 miles -/
theorem car_travel_distance :
  ∀ (distance : ℝ) (time_R : ℝ) (speed_R : ℝ),
    speed_R = 50 →
    distance = speed_R * time_R →
    distance = (speed_R + 10) * (time_R - 2) →
    distance = 600 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l3172_317294


namespace NUMINAMATH_CALUDE_inequality_proof_l3172_317238

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3172_317238


namespace NUMINAMATH_CALUDE_correct_ball_placement_count_l3172_317225

/-- The number of ways to place four distinct balls into three boxes, leaving exactly one box empty -/
def ball_placement_count : ℕ := 42

/-- The number of distinct balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 3

/-- Theorem stating that the number of ways to place the balls is correct -/
theorem correct_ball_placement_count :
  (∃ (f : Fin num_balls → Fin num_boxes),
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∃ (empty_box : Fin num_boxes), ∀ i, f i ≠ empty_box) ∧
    (∀ box : Fin num_boxes, box ≠ empty_box → ∃ i, f i = box)) →
  ball_placement_count = 42 :=
by sorry

end NUMINAMATH_CALUDE_correct_ball_placement_count_l3172_317225


namespace NUMINAMATH_CALUDE_divisors_of_180_l3172_317273

/-- The number of positive divisors of 180 is 18. -/
theorem divisors_of_180 : Finset.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_180_l3172_317273


namespace NUMINAMATH_CALUDE_twin_birthday_product_l3172_317261

theorem twin_birthday_product (age : ℕ) (h : age = 5) :
  (age + 1) * (age + 1) - age * age = 11 := by
  sorry

end NUMINAMATH_CALUDE_twin_birthday_product_l3172_317261


namespace NUMINAMATH_CALUDE_dima_guarantee_win_or_draw_l3172_317272

/-- Represents a player in the game -/
inductive Player : Type
| Gosha : Player
| Dima : Player

/-- Represents a cell on the board -/
structure Cell :=
(row : Nat)
(col : Nat)

/-- Represents the game board -/
def Board := List Cell

/-- Represents a game state -/
structure GameState :=
(board : Board)
(currentPlayer : Player)

/-- Checks if a sequence of 7 consecutive cells is occupied -/
def isWinningSequence (sequence : List Cell) (board : Board) : Bool :=
  sorry

/-- Checks if the game is in a winning state for the current player -/
def isWinningState (state : GameState) : Bool :=
  sorry

/-- Represents a game strategy -/
def Strategy := GameState → Cell

/-- Theorem: Dima can guarantee a win or draw -/
theorem dima_guarantee_win_or_draw :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      game.currentPlayer = Player.Dima →
      (∃ (future_game : GameState), 
        isWinningState future_game ∧ future_game.currentPlayer = Player.Dima) ∨
      (∀ (future_game : GameState), ¬isWinningState future_game) :=
sorry

end NUMINAMATH_CALUDE_dima_guarantee_win_or_draw_l3172_317272


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l3172_317222

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (25 - x) = 9) →
  ((10 + x) * (25 - x) = 529) := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l3172_317222


namespace NUMINAMATH_CALUDE_treehouse_planks_l3172_317216

theorem treehouse_planks :
  ∀ (T : ℕ),
  (T / 4 : ℚ) + (T / 2 : ℚ) + 20 + 30 = T →
  T = 200 := by
sorry

end NUMINAMATH_CALUDE_treehouse_planks_l3172_317216


namespace NUMINAMATH_CALUDE_cora_cookie_expense_l3172_317208

/-- Calculates the total amount spent on cookies in April given the daily purchase, cost per cookie, and number of days. -/
def cookie_expense (daily_purchase : ℕ) (cost_per_cookie : ℕ) (days_in_april : ℕ) : ℕ :=
  daily_purchase * cost_per_cookie * days_in_april

/-- Proves that Cora spent 1620 dollars on cookies in April. -/
theorem cora_cookie_expense :
  cookie_expense 3 18 30 = 1620 := by
  sorry

end NUMINAMATH_CALUDE_cora_cookie_expense_l3172_317208


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l3172_317297

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 5040 → 
  Nat.gcd a b = 24 → 
  a = 240 → 
  b = 504 := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l3172_317297


namespace NUMINAMATH_CALUDE_base_8_4512_equals_2378_l3172_317200

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4512_equals_2378 :
  base_8_to_10 [2, 1, 5, 4] = 2378 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4512_equals_2378_l3172_317200


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l3172_317231

/-- A parabola defined by y = 2(x-1)² + c passing through three points -/
structure Parabola where
  c : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  eq_y₁ : y₁ = 2 * (-2 - 1)^2 + c
  eq_y₂ : y₂ = 2 * (0 - 1)^2 + c
  eq_y₃ : y₃ = 2 * (5/3 - 1)^2 + c

/-- Theorem stating the relationship between y₁, y₂, and y₃ for the given parabola -/
theorem parabola_y_relationship (p : Parabola) : p.y₁ > p.y₂ ∧ p.y₂ > p.y₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_relationship_l3172_317231


namespace NUMINAMATH_CALUDE_f_negative_2014_l3172_317256

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_negative_2014 (h1 : ∀ x, f x = -f (-x))  -- f is odd
                        (h2 : ∀ x, f (x + 3) = f x)  -- f has period 3
                        (h3 : ∀ x ∈ Set.Icc 0 1, f x = x^2 - x + 2)  -- f on [0,1]
                        : f (-2014) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_2014_l3172_317256


namespace NUMINAMATH_CALUDE_sum_of_s_r_at_points_l3172_317262

def r (x : ℝ) : ℝ := |x| + 3

def s (x : ℝ) : ℝ := -|x|

def evaluation_points : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_s_r_at_points :
  (evaluation_points.map (λ x => s (r x))).sum = -63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_s_r_at_points_l3172_317262


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_theorem_l3172_317289

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

def geometric_sum_reciprocals (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  (1 / a) * (1 - (1 / r)^n) / (1 - (1 / r))

theorem geometric_sequence_ratio_theorem :
  let a : ℚ := 1 / 4
  let r : ℚ := 2
  let n : ℕ := 10
  let S := geometric_sum a r n
  let S' := geometric_sum_reciprocals a r n
  S / S' = 32 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_theorem_l3172_317289


namespace NUMINAMATH_CALUDE_leftover_sets_problem_l3172_317203

/-- Given a total number of crayons, number of friends, and crayons per set,
    calculate the number of complete sets left over after distributing one set to each friend. -/
def leftover_sets (total_crayons : ℕ) (num_friends : ℕ) (crayons_per_set : ℕ) : ℕ :=
  (total_crayons / crayons_per_set) % num_friends

theorem leftover_sets_problem :
  leftover_sets 210 30 5 = 12 := by
  sorry

#eval leftover_sets 210 30 5

end NUMINAMATH_CALUDE_leftover_sets_problem_l3172_317203


namespace NUMINAMATH_CALUDE_square_not_always_positive_l3172_317210

theorem square_not_always_positive : ¬ ∀ a : ℝ, a^2 > 0 := by sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l3172_317210


namespace NUMINAMATH_CALUDE_parabola_point_M_x_coordinate_l3172_317250

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define line l passing through F and intersecting the parabola at A and B
def line_l (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  ∃ k : ℝ, A.2 = k * (A.1 - 1) ∧ B.2 = k * (B.1 - 1)

-- Define point M as the midpoint of A and B
def point_M (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define point P on the parabola
def point_P (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define the distance between P and F is 2
def PF_distance (P : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 4

theorem parabola_point_M_x_coordinate 
  (A B M P : ℝ × ℝ) 
  (h1 : line_l A B) 
  (h2 : point_M A B M) 
  (h3 : point_P P) 
  (h4 : PF_distance P) :
  M.1 = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_point_M_x_coordinate_l3172_317250


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3172_317247

def M : Set ℝ := {x | 2 * x - 1 > 0}
def N : Set ℝ := {x | Real.sqrt x < 2}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 1/2 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3172_317247


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_one_l3172_317215

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

theorem M_intersect_N_eq_one : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_one_l3172_317215


namespace NUMINAMATH_CALUDE_simplify_fraction_sum_powers_of_half_l3172_317212

theorem simplify_fraction_sum_powers_of_half :
  1 / (1 / ((1/2)^0) + 1 / ((1/2)^1) + 1 / ((1/2)^2) + 1 / ((1/2)^3)) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_sum_powers_of_half_l3172_317212


namespace NUMINAMATH_CALUDE_binary_10110011_equals_179_l3172_317228

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_10110011_equals_179 :
  binary_to_decimal [true, true, false, false, true, true, false, true] = 179 := by
  sorry

end NUMINAMATH_CALUDE_binary_10110011_equals_179_l3172_317228


namespace NUMINAMATH_CALUDE_trapezoid_area_equality_l3172_317298

/-- Represents a trapezoid divided into triangles and a pentagon as described in the problem -/
structure DividedTrapezoid where
  /-- Area of the central pentagon -/
  Q : ℝ
  /-- Area of the triangle adjacent to one lateral side -/
  s₁ : ℝ
  /-- Area of the triangle adjacent to the shorter base -/
  s₂ : ℝ
  /-- Area of the triangle adjacent to the other lateral side -/
  s₃ : ℝ
  /-- Area of the triangle between s₁ and s₂ -/
  x : ℝ
  /-- Area of the triangle between s₂ and s₃ -/
  y : ℝ
  /-- The sum of areas of triangles adjacent to one side and the shorter base equals half the sum of x, y, s₂, and Q -/
  h₁ : s₁ + x + s₂ = (x + y + s₂ + Q) / 2
  /-- The sum of areas of triangles adjacent to the shorter base and the other side equals half the sum of x, y, s₂, and Q -/
  h₂ : s₂ + y + s₃ = (x + y + s₂ + Q) / 2

/-- The sum of the areas of the three triangles adjacent to the lateral sides and the shorter base 
    of the trapezoid is equal to the area of the pentagon -/
theorem trapezoid_area_equality (t : DividedTrapezoid) : t.s₁ + t.s₂ + t.s₃ = t.Q := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_equality_l3172_317298


namespace NUMINAMATH_CALUDE_diana_work_hours_l3172_317280

/-- Represents Diana's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ  -- Hours worked on Monday, Wednesday, and Friday combined
  tue_thu_hours : ℕ      -- Hours worked on Tuesday and Thursday combined
  weekly_earnings : ℕ    -- Weekly earnings in dollars
  hourly_rate : ℕ        -- Hourly rate in dollars

/-- Theorem stating Diana's work hours on Monday, Wednesday, and Friday --/
theorem diana_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.tue_thu_hours = 30)  -- 15 hours each on Tuesday and Thursday
  (h2 : schedule.weekly_earnings = 1800)
  (h3 : schedule.hourly_rate = 30)
  : schedule.mon_wed_fri_hours = 30 := by
  sorry


end NUMINAMATH_CALUDE_diana_work_hours_l3172_317280


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l3172_317232

/-- Given a quadratic equation 2Ax^2 + 3Bx + 4C = 0 with roots r and s,
    prove that the value of p in the equation x^2 + px + q = 0 with roots r^2 and s^2
    is equal to (16AC - 9B^2) / (4A^2) -/
theorem quadratic_root_transformation (A B C : ℝ) (r s : ℝ) :
  (2 * A * r ^ 2 + 3 * B * r + 4 * C = 0) →
  (2 * A * s ^ 2 + 3 * B * s + 4 * C = 0) →
  ∃ q : ℝ, r ^ 2 ^ 2 + ((16 * A * C - 9 * B ^ 2) / (4 * A ^ 2)) * r ^ 2 + q = 0 ∧
           s ^ 2 ^ 2 + ((16 * A * C - 9 * B ^ 2) / (4 * A ^ 2)) * s ^ 2 + q = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l3172_317232


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3172_317270

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3172_317270


namespace NUMINAMATH_CALUDE_largest_last_digit_is_two_l3172_317277

/-- A function that checks if a two-digit number is divisible by 17 or 23 -/
def isDivisibleBy17Or23 (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n % 17 = 0 ∨ n % 23 = 0)

/-- A function that represents a valid digit string according to the problem conditions -/
def isValidDigitString (s : List Nat) : Prop :=
  s.length = 1001 ∧
  s.head? = some 2 ∧
  ∀ i, i < 1000 → isDivisibleBy17Or23 (s[i]! * 10 + s[i+1]!)

/-- The theorem stating that the largest possible last digit is 2 -/
theorem largest_last_digit_is_two (s : List Nat) (h : isValidDigitString s) :
  s[1000]! ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_largest_last_digit_is_two_l3172_317277


namespace NUMINAMATH_CALUDE_equation_solution_l3172_317276

theorem equation_solution (x y : ℝ) : 
  x^2 + (1-y)^2 + (x-y)^2 = 1/3 ↔ x = 1/3 ∧ y = 2/3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3172_317276


namespace NUMINAMATH_CALUDE_largest_special_number_l3172_317246

def has_distinct_digits (n : ℕ) : Prop := sorry

def divisible_by_digits (n : ℕ) : Prop := sorry

def contains_digit (n : ℕ) (d : ℕ) : Prop := sorry

theorem largest_special_number :
  ∀ n : ℕ,
    has_distinct_digits n ∧
    divisible_by_digits n ∧
    contains_digit n 5 →
    n ≤ 9735 :=
by sorry

end NUMINAMATH_CALUDE_largest_special_number_l3172_317246


namespace NUMINAMATH_CALUDE_equation_solution_l3172_317201

theorem equation_solution : ∃ x : ℝ, x * 15 - x * (2/3) + 1.4 = 10 ∧ x = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3172_317201


namespace NUMINAMATH_CALUDE_permutations_count_l3172_317202

def original_number : List Nat := [1, 1, 2, 3, 4, 5, 6, 7]
def target_number : List Nat := [4, 6, 7, 5, 3, 2, 1, 1]

def count_permutations_less_than_or_equal (original : List Nat) (target : List Nat) : Nat :=
  sorry

theorem permutations_count :
  count_permutations_less_than_or_equal original_number target_number = 12240 :=
sorry

end NUMINAMATH_CALUDE_permutations_count_l3172_317202


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_18_times_5_smallest_positive_multiple_is_90_l3172_317286

theorem smallest_positive_multiple_of_18_times_5 :
  ∀ n : ℕ+, n * (18 * 5) ≥ 18 * 5 :=
by
  sorry

theorem smallest_positive_multiple_is_90 :
  ∃ (n : ℕ+), n * (18 * 5) = 90 ∧ ∀ (m : ℕ+), m * (18 * 5) ≥ 90 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_18_times_5_smallest_positive_multiple_is_90_l3172_317286


namespace NUMINAMATH_CALUDE_square_side_length_l3172_317279

theorem square_side_length : ∃ (s : ℝ), s > 0 ∧ s^2 + s - 4*s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3172_317279


namespace NUMINAMATH_CALUDE_one_third_green_faces_iff_three_l3172_317245

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- The number of green faces on unit cubes after cutting a large painted cube -/
def green_faces (c : Cube n) : ℕ := 6 * n^2

/-- The total number of faces on all unit cubes after cutting -/
def total_faces (c : Cube n) : ℕ := 6 * n^3

/-- Theorem stating that exactly one-third of faces are green iff n = 3 -/
theorem one_third_green_faces_iff_three (c : Cube n) :
  3 * green_faces c = total_faces c ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_one_third_green_faces_iff_three_l3172_317245


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l3172_317236

/-- The line equation: 5y + 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point on the x-axis has y-coordinate equal to 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (5, 0)

theorem line_x_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_x_axis x y :=
by sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l3172_317236


namespace NUMINAMATH_CALUDE_girls_with_rulers_girls_with_rulers_proof_l3172_317274

theorem girls_with_rulers (total_students : ℕ) (students_with_rulers : ℕ) (boys_with_set_squares : ℕ) (total_girls : ℕ) : ℕ :=
  let students_with_set_squares := total_students - students_with_rulers
  let girls_with_set_squares := students_with_set_squares - boys_with_set_squares
  let girls_with_rulers := total_girls - girls_with_set_squares
  girls_with_rulers

#check girls_with_rulers 50 28 14 31 = 23

theorem girls_with_rulers_proof :
  girls_with_rulers 50 28 14 31 = 23 := by
  sorry

end NUMINAMATH_CALUDE_girls_with_rulers_girls_with_rulers_proof_l3172_317274


namespace NUMINAMATH_CALUDE_overtime_to_regular_pay_ratio_l3172_317285

/-- Proves that the ratio of overtime to regular pay rate is 2:1 given the problem conditions --/
theorem overtime_to_regular_pay_ratio :
  ∀ (regular_rate overtime_rate total_pay : ℚ) (regular_hours overtime_hours : ℕ),
    regular_rate = 3 →
    regular_hours = 40 →
    overtime_hours = 12 →
    total_pay = 192 →
    total_pay = regular_rate * regular_hours + overtime_rate * overtime_hours →
    overtime_rate / regular_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_overtime_to_regular_pay_ratio_l3172_317285


namespace NUMINAMATH_CALUDE_sticker_difference_l3172_317218

theorem sticker_difference (belle_stickers carolyn_stickers : ℕ) 
  (h1 : belle_stickers = 97)
  (h2 : carolyn_stickers = 79)
  (h3 : carolyn_stickers < belle_stickers) : 
  belle_stickers - carolyn_stickers = 18 := by
  sorry

end NUMINAMATH_CALUDE_sticker_difference_l3172_317218


namespace NUMINAMATH_CALUDE_group_size_proof_l3172_317267

theorem group_size_proof (n : ℕ) (W : ℝ) : 
  (W + 25) / n - W / n = 2.5 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l3172_317267


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3172_317252

theorem simplify_trig_expression :
  let tan60 : ℝ := Real.sqrt 3
  let cot60 : ℝ := 1 / Real.sqrt 3
  (tan60^3 + cot60^3) / (tan60 + cot60) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3172_317252


namespace NUMINAMATH_CALUDE_widget_production_difference_l3172_317263

/-- Represents the number of widgets produced by David on Tuesday and Wednesday -/
def widget_difference (t : ℝ) : ℝ :=
  let w := 3 * t  -- Tuesday's production rate
  let tuesday_production := w * t
  let wednesday_production := (w + 3) * (t - 3) * 0.9
  tuesday_production - wednesday_production

/-- Theorem stating the difference in widget production between Tuesday and Wednesday -/
theorem widget_production_difference (t : ℝ) :
  widget_difference t = 0.3 * t^2 + 5.4 * t + 8.1 := by
  sorry


end NUMINAMATH_CALUDE_widget_production_difference_l3172_317263


namespace NUMINAMATH_CALUDE_current_age_problem_l3172_317281

theorem current_age_problem (my_age brother_age : ℕ) : 
  (my_age + 10 = 2 * (brother_age + 10)) →
  ((my_age + 10) + (brother_age + 10) = 45) →
  my_age = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_current_age_problem_l3172_317281


namespace NUMINAMATH_CALUDE_quadratic_polynomial_problem_l3172_317224

theorem quadratic_polynomial_problem :
  ∃ (p : ℝ → ℝ),
    (∀ x, p x = (20/9) * x^2 + (20/3) * x - 40) ∧
    p (-6) = 0 ∧
    p 3 = 0 ∧
    p (-3) = -40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_problem_l3172_317224


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3172_317209

def f (x : ℝ) : ℝ := -x + 1

theorem f_satisfies_conditions :
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂) :=
sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3172_317209


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l3172_317282

/-- The number of players in the volleyball team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def num_starters : ℕ := 7

/-- The number of ways to choose starters with the given conditions -/
def valid_lineups : ℕ := Nat.choose total_players num_starters - Nat.choose (total_players - num_quadruplets) (num_starters - num_quadruplets)

theorem volleyball_lineup_count :
  valid_lineups = 11220 :=
sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l3172_317282


namespace NUMINAMATH_CALUDE_product_less_than_sum_plus_one_l3172_317233

theorem product_less_than_sum_plus_one (a₁ a₂ : ℝ) 
  (h₁ : 0 < a₁) (h₂ : a₁ < 1) (h₃ : 0 < a₂) (h₄ : a₂ < 1) : 
  a₁ * a₂ < a₁ + a₂ + 1 := by
  sorry

#check product_less_than_sum_plus_one

end NUMINAMATH_CALUDE_product_less_than_sum_plus_one_l3172_317233


namespace NUMINAMATH_CALUDE_hannah_seashell_distribution_l3172_317268

theorem hannah_seashell_distribution (noah liam hannah : ℕ) : 
  hannah = 4 * liam ∧ 
  liam = 3 * noah → 
  (7 : ℚ) / 36 = (hannah + liam + noah) / 3 - liam / hannah :=
by sorry

end NUMINAMATH_CALUDE_hannah_seashell_distribution_l3172_317268


namespace NUMINAMATH_CALUDE_star_difference_l3172_317211

def star (x y : ℝ) : ℝ := x * y - 3 * x + y

theorem star_difference : (star 5 8) - (star 8 5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l3172_317211


namespace NUMINAMATH_CALUDE_marjs_wallet_after_purchase_l3172_317205

/-- The amount of money left in Marj's wallet after buying a cake -/
def money_left_in_wallet (twenty_bills : ℕ) (five_bills : ℕ) (loose_coins : ℚ) (cake_cost : ℚ) : ℚ :=
  (twenty_bills * 20 + five_bills * 5 : ℚ) + loose_coins - cake_cost

/-- Theorem stating the amount of money left in Marj's wallet -/
theorem marjs_wallet_after_purchase :
  money_left_in_wallet 2 3 4.5 17.5 = 42 := by sorry

end NUMINAMATH_CALUDE_marjs_wallet_after_purchase_l3172_317205


namespace NUMINAMATH_CALUDE_initial_trees_per_row_garden_problem_l3172_317223

theorem initial_trees_per_row : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun initial_rows added_rows final_trees_per_row result =>
    let final_rows := initial_rows + added_rows
    (initial_rows * result = final_rows * final_trees_per_row) →
    (result = 42)

/-- Given the initial number of rows, added rows, and final trees per row,
    prove that the initial number of trees per row is 42. -/
theorem garden_problem (initial_rows added_rows final_trees_per_row : ℕ)
    (h1 : initial_rows = 24)
    (h2 : added_rows = 12)
    (h3 : final_trees_per_row = 28) :
    initial_trees_per_row initial_rows added_rows final_trees_per_row 42 := by
  sorry

end NUMINAMATH_CALUDE_initial_trees_per_row_garden_problem_l3172_317223


namespace NUMINAMATH_CALUDE_trigonometric_equation_l3172_317266

theorem trigonometric_equation (α : Real) 
  (h : (5 * Real.sin α - Real.cos α) / (Real.cos α + Real.sin α) = 1) : 
  Real.tan α = 1/2 ∧ 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) + Real.sin α * Real.cos α = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l3172_317266


namespace NUMINAMATH_CALUDE_frog_jump_distance_l3172_317264

/-- The jumping contest problem -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (frog_extra_jump : ℕ) :
  grasshopper_jump = 36 →
  frog_extra_jump = 17 →
  grasshopper_jump + frog_extra_jump = 53 :=
by
  sorry

#check frog_jump_distance

end NUMINAMATH_CALUDE_frog_jump_distance_l3172_317264


namespace NUMINAMATH_CALUDE_product_properties_l3172_317244

-- Define a function to count trailing zeros
def trailingZeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + trailingZeros (n / 10)
  else 0

theorem product_properties :
  (trailingZeros (360 * 5) = 2) ∧ (250 * 4 = 1000) := by
  sorry

end NUMINAMATH_CALUDE_product_properties_l3172_317244


namespace NUMINAMATH_CALUDE_chess_game_probability_l3172_317206

theorem chess_game_probability (prob_A_win prob_draw : ℝ) 
  (h1 : prob_A_win = 0.2)
  (h2 : prob_draw = 0.5)
  (h3 : 0 ≤ prob_A_win ∧ prob_A_win ≤ 1)
  (h4 : 0 ≤ prob_draw ∧ prob_draw ≤ 1) :
  1 - (prob_A_win + prob_draw) = 0.3 := by
sorry

end NUMINAMATH_CALUDE_chess_game_probability_l3172_317206


namespace NUMINAMATH_CALUDE_jake_weight_loss_l3172_317240

theorem jake_weight_loss (total_weight jake_weight : ℝ) 
  (h1 : total_weight = 290)
  (h2 : jake_weight = 196) : 
  jake_weight - 2 * (total_weight - jake_weight) = 8 :=
sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l3172_317240


namespace NUMINAMATH_CALUDE_inequality_proof_l3172_317265

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) : 
  (a * b ≤ 1/8) ∧ (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3172_317265


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3172_317217

theorem complex_equation_solution :
  ∀ (z : ℂ), z * (Complex.I - 1) = 2 * Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3172_317217


namespace NUMINAMATH_CALUDE_final_walnuts_count_l3172_317204

-- Define the initial conditions and actions
def initial_walnuts : ℕ := 25
def boy_gathered : ℕ := 15
def boy_dropped : ℕ := 3
def boy_hidden : ℕ := 5
def girl_brought : ℕ := 12
def girl_eaten : ℕ := 4
def girl_given : ℕ := 3
def girl_lost : ℕ := 2

-- Theorem to prove
theorem final_walnuts_count :
  initial_walnuts + 
  (boy_gathered - boy_dropped - boy_hidden) + 
  (girl_brought - girl_eaten - girl_given - girl_lost) = 35 := by
  sorry

end NUMINAMATH_CALUDE_final_walnuts_count_l3172_317204


namespace NUMINAMATH_CALUDE_problem_solution_l3172_317254

theorem problem_solution : 4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31 = 470 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3172_317254


namespace NUMINAMATH_CALUDE_only_one_divides_l3172_317283

theorem only_one_divides (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_l3172_317283


namespace NUMINAMATH_CALUDE_pentagon_covers_half_l3172_317230

/-- Represents a tiling of a plane with large squares -/
structure PlaneTiling where
  /-- The number of smaller squares in each row/column of a large square -/
  grid_size : ℕ
  /-- The number of smaller squares that are part of pentagons in each large square -/
  pentagon_squares : ℕ

/-- The percentage of the plane enclosed by pentagons -/
def pentagon_percentage (tiling : PlaneTiling) : ℚ :=
  (tiling.pentagon_squares : ℚ) / (tiling.grid_size^2 : ℚ) * 100

/-- Theorem stating that the percentage of the plane enclosed by pentagons is 50% -/
theorem pentagon_covers_half (tiling : PlaneTiling) 
  (h1 : tiling.grid_size = 4)
  (h2 : tiling.pentagon_squares = 8) : 
  pentagon_percentage tiling = 50 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_covers_half_l3172_317230
