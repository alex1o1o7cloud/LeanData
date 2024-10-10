import Mathlib

namespace unique_positive_solution_l4051_405155

/-- The polynomial function f(x) = x^8 + 6x^7 + 13x^6 + 256x^5 - 684x^4 -/
def f (x : ℝ) : ℝ := x^8 + 6*x^7 + 13*x^6 + 256*x^5 - 684*x^4

/-- The theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end unique_positive_solution_l4051_405155


namespace homothety_composition_l4051_405129

-- Define a homothety
structure Homothety (α : Type*) [AddCommGroup α] :=
  (center : α)
  (coefficient : ℝ)

-- Define a parallel translation
structure ParallelTranslation (α : Type*) [AddCommGroup α] :=
  (vector : α)

-- Define the composition of two homotheties
def compose_homotheties {α : Type*} [AddCommGroup α] [Module ℝ α]
  (h1 h2 : Homothety α) : (ParallelTranslation α) ⊕ (Homothety α) :=
  sorry

-- Theorem statement
theorem homothety_composition {α : Type*} [AddCommGroup α] [Module ℝ α]
  (h1 h2 : Homothety α) :
  (∃ (t : ParallelTranslation α), compose_homotheties h1 h2 = Sum.inl t ∧
    ∃ (v : α), t.vector = v ∧ (∃ (c : ℝ), v = c • (h2.center - h1.center)) ∧
    h1.coefficient * h2.coefficient = 1) ∨
  (∃ (h : Homothety α), compose_homotheties h1 h2 = Sum.inr h ∧
    ∃ (c : ℝ), h.center = h1.center + c • (h2.center - h1.center) ∧
    h.coefficient = h1.coefficient * h2.coefficient ∧
    h1.coefficient * h2.coefficient ≠ 1) :=
  sorry

end homothety_composition_l4051_405129


namespace coffee_shop_solution_l4051_405189

/-- Represents the coffee shop scenario with Alice and Bob -/
def coffee_shop_scenario (x : ℝ) : Prop :=
  let alice_initial := x
  let bob_initial := 1.25 * x
  let alice_consumed := 0.75 * x
  let bob_consumed := 0.75 * (1.25 * x)
  let alice_remaining := 0.25 * x
  let bob_remaining := 1.25 * x - 0.75 * (1.25 * x)
  let alice_gives := 0.5 * alice_remaining + 1
  let alice_final := alice_consumed - alice_gives
  let bob_final := bob_consumed + alice_gives
  (alice_final = bob_final) ∧
  (alice_initial + bob_initial = 9)

/-- Theorem stating that there exists a solution to the coffee shop scenario -/
theorem coffee_shop_solution : ∃ x : ℝ, coffee_shop_scenario x := by
  sorry


end coffee_shop_solution_l4051_405189


namespace two_week_riding_time_l4051_405121

-- Define the riding schedule
def riding_schedule : List (String × Float) := [
  ("Monday", 1),
  ("Tuesday", 0.5),
  ("Wednesday", 1),
  ("Thursday", 0.5),
  ("Friday", 1),
  ("Saturday", 2),
  ("Sunday", 0)
]

-- Calculate the total riding time for one week
def weekly_riding_time : Float :=
  (riding_schedule.map (λ (_, time) => time)).sum

-- Theorem: The total riding time for a 2-week period is 12 hours
theorem two_week_riding_time :
  weekly_riding_time * 2 = 12 := by
  sorry

end two_week_riding_time_l4051_405121


namespace bills_toilet_paper_supply_l4051_405163

/-- Theorem: Bill's Toilet Paper Supply

Given:
- Bill uses the bathroom 3 times a day
- Bill uses 5 squares of toilet paper each time
- Each roll has 300 squares of toilet paper
- Bill's toilet paper supply will last for 20000 days

Prove that Bill has 1000 rolls of toilet paper.
-/
theorem bills_toilet_paper_supply 
  (bathroom_visits_per_day : ℕ) 
  (squares_per_visit : ℕ) 
  (squares_per_roll : ℕ) 
  (supply_duration_days : ℕ) 
  (h1 : bathroom_visits_per_day = 3)
  (h2 : squares_per_visit = 5)
  (h3 : squares_per_roll = 300)
  (h4 : supply_duration_days = 20000) :
  (bathroom_visits_per_day * squares_per_visit * supply_duration_days) / squares_per_roll = 1000 := by
  sorry

#check bills_toilet_paper_supply

end bills_toilet_paper_supply_l4051_405163


namespace range_of_a_l4051_405176

/-- The function f(x) = ax^2 - 2x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x + 1

/-- f is decreasing on [1, +∞) -/
def f_decreasing (a : ℝ) : Prop := 
  ∀ x y, 1 ≤ x → x < y → f a y < f a x

/-- The range of a is (-∞, 0] -/
theorem range_of_a : 
  (∃ a, f_decreasing a) ↔ (∀ a, f_decreasing a → a ≤ 0) ∧ (∃ a ≤ 0, f_decreasing a) :=
sorry

end range_of_a_l4051_405176


namespace binary_110101_is_53_l4051_405180

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101 -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_is_53 : binary_to_decimal binary_110101 = 53 := by
  sorry

end binary_110101_is_53_l4051_405180


namespace quincy_peter_difference_l4051_405122

/-- The number of pictures Randy drew -/
def randy_pictures : ℕ := 5

/-- The number of additional pictures Peter drew compared to Randy -/
def peter_additional : ℕ := 3

/-- The total number of pictures drawn by all three -/
def total_pictures : ℕ := 41

/-- The number of pictures Peter drew -/
def peter_pictures : ℕ := randy_pictures + peter_additional

/-- The number of pictures Quincy drew -/
def quincy_pictures : ℕ := total_pictures - randy_pictures - peter_pictures

theorem quincy_peter_difference : quincy_pictures - peter_pictures = 20 := by
  sorry

end quincy_peter_difference_l4051_405122


namespace ivan_piggy_bank_l4051_405140

/-- Represents the contents of Ivan's piggy bank -/
structure PiggyBank where
  dimes : Nat
  pennies : Nat

/-- The value of the piggy bank in cents -/
def PiggyBank.value (pb : PiggyBank) : Nat :=
  pb.dimes * 10 + pb.pennies

theorem ivan_piggy_bank :
  ∀ (pb : PiggyBank),
    pb.dimes = 50 →
    pb.value = 1200 →
    pb.pennies = 700 := by
  sorry

end ivan_piggy_bank_l4051_405140


namespace ice_cream_cost_l4051_405154

/-- Proves that the cost of a scoop of ice cream is $5 given the problem conditions -/
theorem ice_cream_cost (people : ℕ) (meal_cost : ℕ) (total_money : ℕ) 
  (h1 : people = 3)
  (h2 : meal_cost = 10)
  (h3 : total_money = 45)
  (h4 : ∃ (ice_cream_cost : ℕ), total_money = people * meal_cost + people * ice_cream_cost) :
  ∃ (ice_cream_cost : ℕ), ice_cream_cost = 5 := by
sorry

end ice_cream_cost_l4051_405154


namespace original_price_satisfies_conditions_l4051_405166

/-- The original price of merchandise satisfying given conditions -/
def original_price : ℝ := 175

/-- The loss when sold at 60% of the original price -/
def loss_at_60_percent : ℝ := 20

/-- The gain when sold at 80% of the original price -/
def gain_at_80_percent : ℝ := 15

/-- Theorem stating that the original price satisfies the given conditions -/
theorem original_price_satisfies_conditions : 
  (0.6 * original_price + loss_at_60_percent = 0.8 * original_price - gain_at_80_percent) := by
  sorry

end original_price_satisfies_conditions_l4051_405166


namespace unique_function_solution_l4051_405142

/-- Given a positive real number c, prove that the only function f: ℝ₊ → ℝ₊ 
    satisfying f((c+1)x + f(y)) = f(x + 2y) + 2cx for all x, y ∈ ℝ₊ is f(x) = 2x. -/
theorem unique_function_solution (c : ℝ) (hc : c > 0) :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
  (∀ x y, x > 0 → y > 0 → f ((c + 1) * x + f y) = f (x + 2 * y) + 2 * c * x) →
  ∀ x, x > 0 → f x = 2 * x :=
by sorry

end unique_function_solution_l4051_405142


namespace alligators_count_l4051_405145

/-- Given the number of alligators seen by Samara and her friends, prove the total number of alligators seen. -/
theorem alligators_count (samara_count : ℕ) (friend_count : ℕ) (friend_average : ℕ) : 
  samara_count = 20 → friend_count = 3 → friend_average = 10 →
  samara_count + friend_count * friend_average = 50 := by
  sorry


end alligators_count_l4051_405145


namespace square_side_length_equal_area_l4051_405109

theorem square_side_length_equal_area (rectangle_length rectangle_width : ℝ) :
  rectangle_length = 72 ∧ rectangle_width = 18 →
  ∃ (square_side : ℝ), square_side ^ 2 = rectangle_length * rectangle_width ∧ square_side = 36 := by
  sorry

end square_side_length_equal_area_l4051_405109


namespace arithmetic_mean_geq_geometric_mean_l4051_405102

theorem arithmetic_mean_geq_geometric_mean (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end arithmetic_mean_geq_geometric_mean_l4051_405102


namespace pencil_cost_l4051_405173

/-- Proves that the cost of each pencil Cindi bought is $0.50 --/
theorem pencil_cost (cindi_pencils : ℕ) (marcia_pencils : ℕ) (donna_pencils : ℕ) 
  (h1 : marcia_pencils = 2 * cindi_pencils)
  (h2 : donna_pencils = 3 * marcia_pencils)
  (h3 : donna_pencils + marcia_pencils = 480)
  (h4 : cindi_pencils * (cost_per_pencil : ℚ) = 30) : 
  cost_per_pencil = 1/2 := by
  sorry

#check pencil_cost

end pencil_cost_l4051_405173


namespace component_is_unqualified_l4051_405164

-- Define the nominal diameter and tolerance
def nominal_diameter : ℝ := 20
def tolerance : ℝ := 0.02

-- Define the measured diameter
def measured_diameter : ℝ := 19.9

-- Define what it means for a component to be qualified
def is_qualified (d : ℝ) : Prop :=
  nominal_diameter - tolerance ≤ d ∧ d ≤ nominal_diameter + tolerance

-- Theorem stating that the component is unqualified
theorem component_is_unqualified : ¬ is_qualified measured_diameter := by
  sorry

end component_is_unqualified_l4051_405164


namespace angle_abc_measure_l4051_405141

/-- A configuration with a square inscribed in a regular pentagon sharing a side -/
structure SquareInPentagon where
  /-- The measure of an interior angle of the regular pentagon in degrees -/
  pentagon_angle : ℝ
  /-- The measure of an interior angle of the square in degrees -/
  square_angle : ℝ
  /-- The angle ABC formed by the vertex of the pentagon adjacent to the shared side
      and the two nearest vertices of the square -/
  angle_abc : ℝ
  /-- The pentagon_angle is 108 degrees -/
  pentagon_angle_eq : pentagon_angle = 108
  /-- The square_angle is 90 degrees -/
  square_angle_eq : square_angle = 90

/-- The angle ABC in a SquareInPentagon configuration is 27 degrees -/
theorem angle_abc_measure (config : SquareInPentagon) : config.angle_abc = 27 :=
  sorry

end angle_abc_measure_l4051_405141


namespace train_crossing_time_l4051_405134

/-- Proves that a train 400 meters long, traveling at 36 km/h, takes 40 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 400 →
  train_speed_kmh = 36 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 40 := by
  sorry

#check train_crossing_time

end train_crossing_time_l4051_405134


namespace quadratic_inequality_condition_l4051_405114

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 2 > 0) ↔ -2*Real.sqrt 2 < m ∧ m < 2*Real.sqrt 2 := by
  sorry

end quadratic_inequality_condition_l4051_405114


namespace somu_age_problem_l4051_405165

theorem somu_age_problem (somu_age father_age : ℕ) : 
  somu_age = father_age / 3 →
  somu_age - 6 = (father_age - 6) / 5 →
  somu_age = 12 := by
sorry

end somu_age_problem_l4051_405165


namespace system_solutions_l4051_405188

theorem system_solutions :
  let solutions := [(2, 1), (0, -3), (-6, 9)]
  ∀ (x y : ℝ),
    (x + |y| = 3 ∧ 2*|x| - y = 3) ↔ (x, y) ∈ solutions := by
  sorry

end system_solutions_l4051_405188


namespace fifteenth_number_with_digit_sum_14_l4051_405118

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 14 -/
def nth_number_with_digit_sum_14 (n : ℕ+) : ℕ+ := sorry

/-- The main theorem -/
theorem fifteenth_number_with_digit_sum_14 :
  nth_number_with_digit_sum_14 15 = 266 := by sorry

end fifteenth_number_with_digit_sum_14_l4051_405118


namespace solution_characterization_l4051_405183

/-- The set of polynomials that satisfy the given condition -/
def SolutionSet : Set (Polynomial ℤ) :=
  {f | f = Polynomial.monomial 3 1 + Polynomial.monomial 2 1 + Polynomial.monomial 1 1 + Polynomial.monomial 0 1 ∨
       f = Polynomial.monomial 3 1 + Polynomial.monomial 2 2 + Polynomial.monomial 1 2 + Polynomial.monomial 0 2 ∨
       f = Polynomial.monomial 3 2 + Polynomial.monomial 2 1 + Polynomial.monomial 1 2 + Polynomial.monomial 0 1 ∨
       f = Polynomial.monomial 3 2 + Polynomial.monomial 2 2 + Polynomial.monomial 1 1 + Polynomial.monomial 0 2}

/-- The condition that f must satisfy -/
def SatisfiesCondition (f : Polynomial ℤ) : Prop :=
  ∃ g h : Polynomial ℤ, f^4 + 2*f + 2 = (Polynomial.monomial 4 1 + 2*Polynomial.monomial 2 1 + 2)*g + 3*h

theorem solution_characterization :
  ∀ f : Polynomial ℤ, (f ∈ SolutionSet ↔ (SatisfiesCondition f ∧ 
    ∀ f' : Polynomial ℤ, SatisfiesCondition f' → (Polynomial.degree f' ≥ Polynomial.degree f))) :=
sorry

end solution_characterization_l4051_405183


namespace problem_statement_l4051_405135

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → a * b ≤ m) → m ≥ 1/4) ∧
  (∀ x : ℝ, (1/a + 1/b ≥ |2*x - 1| - |x + 1|) ↔ -2 ≤ x ∧ x ≤ 6) :=
by sorry

end problem_statement_l4051_405135


namespace magnitude_of_BC_l4051_405150

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, -2)
def AC : ℝ × ℝ := (4, -1)

theorem magnitude_of_BC : 
  let C : ℝ × ℝ := (A.1 + AC.1, A.2 + AC.2)
  let BC : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)
  Real.sqrt ((BC.1)^2 + (BC.2)^2) = Real.sqrt 13 := by
  sorry

end magnitude_of_BC_l4051_405150


namespace quadratic_function_form_l4051_405123

/-- A quadratic function with two equal real roots and a specific derivative -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧ 
  (∃! r : ℝ, f r = 0) ∧
  (∀ x, deriv f x = 2 * x + 2)

/-- The theorem stating the specific form of the quadratic function -/
theorem quadratic_function_form (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∀ x, f x = x^2 + 2*x + 1 :=
sorry

end quadratic_function_form_l4051_405123


namespace perpendicular_line_through_point_l4051_405191

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using the general form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b + l1.b * l2.a = 0

-- Define a point being on a line
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- The main theorem
theorem perpendicular_line_through_point :
  let P : Point2D := ⟨1, -2⟩
  let given_line : Line2D := ⟨1, -3, 2⟩
  let result_line : Line2D := ⟨3, 1, -1⟩
  perpendicular given_line result_line ∧ point_on_line P result_line := by
  sorry


end perpendicular_line_through_point_l4051_405191


namespace player1_wins_l4051_405124

/-- Represents the state of the game -/
structure GameState :=
  (coins : ℕ)

/-- Represents a player's move -/
structure Move :=
  (coins_taken : ℕ)

/-- Defines a valid move for Player 1 -/
def valid_move_player1 (m : Move) : Prop :=
  m.coins_taken % 2 = 1 ∧ 1 ≤ m.coins_taken ∧ m.coins_taken ≤ 99

/-- Defines a valid move for Player 2 -/
def valid_move_player2 (m : Move) : Prop :=
  m.coins_taken % 2 = 0 ∧ 2 ≤ m.coins_taken ∧ m.coins_taken ≤ 100

/-- Defines the game transition for a player's move -/
def make_move (s : GameState) (m : Move) : GameState :=
  ⟨s.coins - m.coins_taken⟩

/-- Defines a winning strategy for Player 1 -/
def winning_strategy (initial_coins : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    (∀ s : GameState, valid_move_player1 (strategy s)) ∧
    (∀ s : GameState, ∀ m : Move, 
      valid_move_player2 m → 
      ∃ (next_move : Move), 
        valid_move_player1 next_move ∧
        make_move (make_move s m) next_move = ⟨0⟩)

theorem player1_wins : winning_strategy 2001 := by
  sorry


end player1_wins_l4051_405124


namespace interest_rate_equation_l4051_405178

/-- Given the following conditions:
  - Manoj borrowed Rs. 3900 from Anwar
  - The loan is for 3 years
  - Manoj lent Rs. 5655 to Ramu for 3 years at 9% p.a. simple interest
  - Manoj gains Rs. 824.85 from the whole transaction
Prove that the interest rate r at which Manoj borrowed from Anwar satisfies the equation:
5655 * 0.09 * 3 - 3900 * (r / 100) * 3 = 824.85 -/
theorem interest_rate_equation (borrowed : ℝ) (lent : ℝ) (duration : ℝ) (ramu_rate : ℝ) (gain : ℝ) (r : ℝ) 
    (h1 : borrowed = 3900)
    (h2 : lent = 5655)
    (h3 : duration = 3)
    (h4 : ramu_rate = 0.09)
    (h5 : gain = 824.85) :
  lent * ramu_rate * duration - borrowed * (r / 100) * duration = gain := by
  sorry

end interest_rate_equation_l4051_405178


namespace absolute_value_equation_l4051_405187

theorem absolute_value_equation (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 0 ∧ c = 1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = -1)) := by sorry

end absolute_value_equation_l4051_405187


namespace village_population_equality_l4051_405111

/-- The number of years it takes for two village populations to be equal -/
def years_to_equal_population (x_initial : ℕ) (x_rate : ℕ) (y_initial : ℕ) (y_rate : ℕ) : ℕ :=
  (x_initial - y_initial) / (y_rate + x_rate)

/-- Theorem stating that the populations of Village X and Village Y will be equal after 16 years -/
theorem village_population_equality :
  years_to_equal_population 74000 1200 42000 800 = 16 := by
  sorry

end village_population_equality_l4051_405111


namespace pencils_left_l4051_405127

/-- Calculates the number of pencils Steve has left after giving some to Matt and Lauren -/
theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (lauren_pencils : ℕ) (matt_extra : ℕ) : 
  boxes * pencils_per_box - (lauren_pencils + (lauren_pencils + matt_extra)) = 9 :=
by
  sorry

#check pencils_left 2 12 6 3

end pencils_left_l4051_405127


namespace sixth_term_before_three_l4051_405138

def fibonacci_like_sequence (a : ℤ → ℤ) : Prop :=
  ∀ n, a (n + 2) = a (n + 1) + a n

theorem sixth_term_before_three (a : ℤ → ℤ) :
  fibonacci_like_sequence a →
  a 0 = 3 ∧ a 1 = 5 ∧ a 2 = 8 ∧ a 3 = 13 ∧ a 4 = 21 →
  a (-6) = -1 := by
sorry

end sixth_term_before_three_l4051_405138


namespace combined_mean_of_two_sets_l4051_405126

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℝ) 
                                  (set2_count : ℕ) (set2_mean : ℝ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 27 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℝ) = 21.4 := by
sorry

end combined_mean_of_two_sets_l4051_405126


namespace display_board_sides_l4051_405175

/-- A polygonal display board with given perimeter and side ribbon length has a specific number of sides. -/
theorem display_board_sides (perimeter : ℝ) (side_ribbon_length : ℝ) (num_sides : ℕ) : 
  perimeter = 42 → side_ribbon_length = 7 → num_sides * side_ribbon_length = perimeter → num_sides = 6 := by
  sorry

end display_board_sides_l4051_405175


namespace root_equation_consequence_l4051_405112

theorem root_equation_consequence (m : ℝ) : 
  m^2 - 2*m - 7 = 0 → m^2 - 2*m + 1 = 8 := by
  sorry

end root_equation_consequence_l4051_405112


namespace mistaken_multiplication_l4051_405159

theorem mistaken_multiplication (x y : ℕ) : 
  x ≥ 1000000 ∧ x ≤ 9999999 ∧
  y ≥ 1000000 ∧ y ≤ 9999999 ∧
  (10^7 : ℕ) * x + y = 3 * x * y →
  x = 3333333 ∧ y = 3333334 := by
sorry

end mistaken_multiplication_l4051_405159


namespace ramon_age_l4051_405132

/-- Ramon's age problem -/
theorem ramon_age (loui_age : ℕ) (ramon_future_age : ℕ) : 
  loui_age = 23 →
  ramon_future_age = 2 * loui_age →
  ramon_future_age - 20 = 26 :=
by
  sorry

end ramon_age_l4051_405132


namespace joel_laps_l4051_405199

/-- Given that Yvonne swims 10 laps in 5 minutes, her younger sister swims half as many laps,
    and Joel swims three times as many laps as the younger sister,
    prove that Joel swims 15 laps in 5 minutes. -/
theorem joel_laps (yvonne_laps : ℕ) (younger_sister_ratio : ℚ) (joel_ratio : ℕ) :
  yvonne_laps = 10 →
  younger_sister_ratio = 1 / 2 →
  joel_ratio = 3 →
  (yvonne_laps : ℚ) * younger_sister_ratio * joel_ratio = 15 := by
  sorry

end joel_laps_l4051_405199


namespace parabola_y_axis_intersection_l4051_405156

/-- The parabola y = x^2 - 4 intersects the y-axis at the point (0, -4) -/
theorem parabola_y_axis_intersection :
  let f : ℝ → ℝ := fun x ↦ x^2 - 4
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, -4) := by
  sorry

end parabola_y_axis_intersection_l4051_405156


namespace square_plus_fourth_power_equality_l4051_405196

theorem square_plus_fourth_power_equality (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : n > 3) 
  (h4 : m^2 + n^4 = 2*(m - 6)^2 + 2*(n + 1)^2) : 
  m^2 + n^4 = 1994 := by
sorry

end square_plus_fourth_power_equality_l4051_405196


namespace nonnegative_difference_of_roots_l4051_405195

theorem nonnegative_difference_of_roots (x : ℝ) : 
  let roots := {r : ℝ | r^2 + 6*r + 8 = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 2 :=
by
  sorry

end nonnegative_difference_of_roots_l4051_405195


namespace f_2017_equals_3_l4051_405108

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_2017_equals_3 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_value : f (-1) = -3) :
  f 2017 = 3 := by
sorry

end f_2017_equals_3_l4051_405108


namespace unique_intersection_point_l4051_405168

/-- The function g(x) = x^3 + 5x^2 + 12x + 20 -/
def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 12*x + 20

/-- Theorem: The unique intersection point of g(x) and its inverse is (-4, -4) -/
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-4, -4) := by sorry

end unique_intersection_point_l4051_405168


namespace pineapples_cost_theorem_l4051_405162

/-- The cost relationship between bananas, apples, and pineapples -/
structure FruitCosts where
  banana_to_apple : ℚ    -- 5 bananas = 3 apples
  apple_to_pineapple : ℚ  -- 9 apples = 6 pineapples

/-- The number of pineapples that cost the same as 30 bananas -/
def pineapples_equal_to_30_bananas (costs : FruitCosts) : ℚ :=
  30 * (costs.apple_to_pineapple / 9) * (3 / 5)

theorem pineapples_cost_theorem (costs : FruitCosts) 
  (h1 : costs.banana_to_apple = 3 / 5)
  (h2 : costs.apple_to_pineapple = 6 / 9) :
  pineapples_equal_to_30_bananas costs = 12 := by
  sorry

end pineapples_cost_theorem_l4051_405162


namespace hyperbola_intersection_perpendicular_l4051_405167

-- Define the hyperbola C₁
def C₁ (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a line with slope 1
def Line (x y b : ℝ) : Prop := y = x + b

-- Define the tangency condition
def IsTangent (b : ℝ) : Prop := b^2 = 2

-- Define the perpendicularity of two vectors
def Perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_intersection_perpendicular 
  (x₁ y₁ x₂ y₂ b : ℝ) : 
  C₁ x₁ y₁ → C₁ x₂ y₂ → 
  Line x₁ y₁ b → Line x₂ y₂ b → 
  IsTangent b → 
  Perpendicular x₁ y₁ x₂ y₂ :=
sorry

end hyperbola_intersection_perpendicular_l4051_405167


namespace borrowing_schemes_l4051_405179

theorem borrowing_schemes (n : ℕ) (m : ℕ) :
  n = 5 →  -- number of students
  m = 4 →  -- number of novels
  (∃ (schemes : ℕ), schemes = 60) :=
by
  intros hn hm
  -- The proof goes here
  sorry

end borrowing_schemes_l4051_405179


namespace train_speed_problem_l4051_405101

theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 100 →
  faster_speed = 45 →
  passing_time = 9.599232061435085 →
  ∃ slower_speed : ℝ,
    slower_speed > 0 ∧
    slower_speed < faster_speed ∧
    (faster_speed + slower_speed) * (passing_time / 3600) = 2 * (train_length / 1000) ∧
    slower_speed = 30 :=
by sorry

end train_speed_problem_l4051_405101


namespace log_equation_solution_l4051_405153

-- Define the logarithm function for base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, x > 0 ∧ log5 x - 4 * log5 2 = -3 ∧ x = 16 / 125 := by
  sorry

end log_equation_solution_l4051_405153


namespace fractional_equation_solution_l4051_405143

theorem fractional_equation_solution :
  ∃ x : ℚ, (3 / x = 1 / (x - 1)) ∧ (x = 3 / 2) :=
by sorry

end fractional_equation_solution_l4051_405143


namespace parabola_circle_intersection_chord_length_l4051_405182

/-- Given a parabola and a circle, prove the length of the chord formed by their intersection -/
theorem parabola_circle_intersection_chord_length :
  ∀ (p : ℝ) (x y : ℝ → ℝ),
    p > 0 →
    (∀ t, y t ^ 2 = 2 * p * x t) →
    (∀ t, (x t - 1) ^ 2 + (y t + 2) ^ 2 = 9) →
    x 0 = 1 ∧ y 0 = -2 →
    ∃ (a b : ℝ), a ≠ b ∧
      x a = -1 ∧ x b = -1 ∧
      (x a - 1) ^ 2 + (y a + 2) ^ 2 = 9 ∧
      (x b - 1) ^ 2 + (y b + 2) ^ 2 = 9 ∧
      (y a - y b) ^ 2 = 20 :=
by sorry

end parabola_circle_intersection_chord_length_l4051_405182


namespace books_in_year_l4051_405190

/-- The number of books Jack can read in a day -/
def books_per_day : ℕ := 9

/-- The number of days in a year -/
def days_in_year : ℕ := 365

/-- Theorem: Jack can read 3285 books in a year -/
theorem books_in_year : books_per_day * days_in_year = 3285 := by
  sorry

end books_in_year_l4051_405190


namespace area_ratio_is_three_fourths_l4051_405198

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Points on the sides of the octagon -/
structure OctagonPoints (oct : RegularOctagon) where
  I : ℝ × ℝ
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  on_sides : sorry
  equally_spaced : sorry

/-- The ratio of areas of the inner octagon to the outer octagon -/
def area_ratio (oct : RegularOctagon) (pts : OctagonPoints oct) : ℝ := sorry

/-- The main theorem -/
theorem area_ratio_is_three_fourths (oct : RegularOctagon) (pts : OctagonPoints oct) :
  area_ratio oct pts = 3/4 := by sorry

end area_ratio_is_three_fourths_l4051_405198


namespace geometric_sequence_sum_l4051_405103

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n ∧ a n > 0

/-- The main theorem -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16 →
  a 3 + a 6 = 4 := by
  sorry


end geometric_sequence_sum_l4051_405103


namespace turtleneck_sweater_profit_profit_percentage_l4051_405133

theorem turtleneck_sweater_profit (C : ℝ) : 
  let first_markup := C * 1.20
  let second_markup := first_markup * 1.25
  let final_price := second_markup * 0.88
  final_price = C * 1.32 := by sorry

theorem profit_percentage (C : ℝ) : 
  let first_markup := C * 1.20
  let second_markup := first_markup * 1.25
  let final_price := second_markup * 0.88
  (final_price - C) / C = 0.32 := by sorry

end turtleneck_sweater_profit_profit_percentage_l4051_405133


namespace quadratic_roots_difference_l4051_405149

theorem quadratic_roots_difference (r₁ r₂ : ℝ) : 
  r₁^2 - 7*r₁ + 12 = 0 → r₂^2 - 7*r₂ + 12 = 0 → |r₁ - r₂| = 5 := by
  sorry

end quadratic_roots_difference_l4051_405149


namespace problem_solution_l4051_405100

theorem problem_solution : 
  ∃ x : ℝ, (0.4 * 2 = 0.25 * (0.3 * 15 + x)) ∧ (x = -1.3) := by
  sorry

end problem_solution_l4051_405100


namespace max_area_is_35_l4051_405147

/-- Represents the cost constraint for the rectangular frame -/
def cost_constraint (l w : ℕ) : Prop := 3 * l + 5 * w ≤ 50

/-- Represents the area of the rectangular frame -/
def area (l w : ℕ) : ℕ := l * w

/-- Theorem stating that the maximum area of the rectangular frame is 35 m² -/
theorem max_area_is_35 :
  ∃ (l w : ℕ), cost_constraint l w ∧ area l w = 35 ∧
  ∀ (l' w' : ℕ), cost_constraint l' w' → area l' w' ≤ 35 := by
  sorry

end max_area_is_35_l4051_405147


namespace consecutive_sum_2016_l4051_405181

def is_valid_n (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a : ℕ, n * (2 * a + n - 1) = 4032

theorem consecutive_sum_2016 :
  {n : ℕ | is_valid_n n} = {3, 7, 9, 21, 63} :=
by sorry

end consecutive_sum_2016_l4051_405181


namespace f_is_quadratic_l4051_405172

/-- Definition of a quadratic equation in x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation x^2 + 3x - 5 = 0 -/
def f (x : ℝ) : ℝ := x^2 + 3*x - 5

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l4051_405172


namespace partitioned_triangle_area_l4051_405139

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  area_quad : ℝ

/-- The theorem to be proved -/
theorem partitioned_triangle_area 
  (t : PartitionedTriangle) 
  (h1 : t.area1 = 4) 
  (h2 : t.area2 = 8) 
  (h3 : t.area3 = 12) : 
  t.area_quad = 16 := by
  sorry


end partitioned_triangle_area_l4051_405139


namespace equal_roots_right_triangle_equilateral_triangle_roots_l4051_405144

/-- A triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The quadratic equation associated with the triangle -/
def triangle_quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 + 2 * t.b * x + (t.a - t.c)

theorem equal_roots_right_triangle (t : Triangle) :
  (∃ x : ℝ, (∀ y : ℝ, triangle_quadratic t y = 0 ↔ y = x)) →
  t.a^2 = t.b^2 + t.c^2 :=
sorry

theorem equilateral_triangle_roots (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (∀ x : ℝ, triangle_quadratic t x = 0 ↔ x = 0 ∨ x = -1) :=
sorry

end equal_roots_right_triangle_equilateral_triangle_roots_l4051_405144


namespace min_value_of_sum_of_ratios_l4051_405113

theorem min_value_of_sum_of_ratios (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) 
  (h1 : 1 ≤ a₁) (h2 : a₁ ≤ a₂) (h3 : a₂ ≤ a₃) (h4 : a₃ ≤ a₄) 
  (h5 : a₄ ≤ a₅) (h6 : a₅ ≤ a₆) (h7 : a₆ ≤ 64) :
  (a₁ : ℚ) / a₂ + (a₃ : ℚ) / a₄ + (a₅ : ℚ) / a₆ ≥ 5 / 3 := by
  sorry

end min_value_of_sum_of_ratios_l4051_405113


namespace smallest_urn_satisfying_condition_l4051_405193

/-- An urn contains marbles of five colors: red, white, blue, green, and yellow. -/
structure Urn :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

/-- The total number of marbles in the urn -/
def Urn.total (u : Urn) : ℕ := u.red + u.white + u.blue + u.green + u.yellow

/-- The probability of drawing five red marbles -/
def Urn.prob_five_red (u : Urn) : ℚ :=
  (u.red.choose 5 : ℚ) / (u.total.choose 5)

/-- The probability of drawing one white, one blue, and three red marbles -/
def Urn.prob_one_white_one_blue_three_red (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.red.choose 3) : ℚ) / (u.total.choose 5)

/-- The probability of drawing one white, one blue, one green, and two red marbles -/
def Urn.prob_one_white_one_blue_one_green_two_red (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.green.choose 1) * (u.red.choose 2) : ℚ) / (u.total.choose 5)

/-- The probability of drawing one marble of each color except yellow -/
def Urn.prob_one_each_except_yellow (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.green.choose 1) * (u.red.choose 1) : ℚ) / (u.total.choose 5)

/-- The probability of drawing one marble of each color -/
def Urn.prob_one_each (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.green.choose 1) * (u.red.choose 1) * (u.yellow.choose 1) : ℚ) / (u.total.choose 5)

/-- The urn satisfies the equal probability condition -/
def Urn.satisfies_condition (u : Urn) : Prop :=
  u.prob_five_red = u.prob_one_white_one_blue_three_red ∧
  u.prob_five_red = u.prob_one_white_one_blue_one_green_two_red ∧
  u.prob_five_red = u.prob_one_each_except_yellow ∧
  u.prob_five_red = u.prob_one_each

theorem smallest_urn_satisfying_condition :
  ∃ (u : Urn), u.satisfies_condition ∧ u.total = 14 ∧ ∀ (v : Urn), v.satisfies_condition → u.total ≤ v.total :=
sorry

end smallest_urn_satisfying_condition_l4051_405193


namespace divisor_power_difference_l4051_405186

theorem divisor_power_difference (k : ℕ) : 
  (15 ^ k ∣ 759325) → 3 ^ k - k ^ 3 = 1 := by
  sorry

end divisor_power_difference_l4051_405186


namespace triangle_side_length_l4051_405152

theorem triangle_side_length (a : ℕ) : 
  (a % 2 = 1) → -- a is odd
  (2 + a > 3) ∧ (2 + 3 > a) ∧ (a + 3 > 2) → -- triangle inequality
  a = 3 := by
sorry

end triangle_side_length_l4051_405152


namespace simplify_and_rationalize_l4051_405170

theorem simplify_and_rationalize : 
  (Real.sqrt 7 / Real.sqrt 3) * (Real.sqrt 8 / Real.sqrt 5) * (Real.sqrt 9 / Real.sqrt 7) = 2 * Real.sqrt 30 / 5 := by
  sorry

end simplify_and_rationalize_l4051_405170


namespace smallest_number_with_given_remainders_l4051_405184

theorem smallest_number_with_given_remainders : ∃ (n : ℕ), 
  (n % 19 = 9 ∧ n % 23 = 7) ∧ 
  (∀ m : ℕ, m % 19 = 9 ∧ m % 23 = 7 → n ≤ m) ∧
  n = 161 := by
sorry

end smallest_number_with_given_remainders_l4051_405184


namespace complement_intersection_theorem_l4051_405185

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {2, 3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 4} := by sorry

end complement_intersection_theorem_l4051_405185


namespace common_root_and_parameter_l4051_405160

theorem common_root_and_parameter :
  ∃ (x p : ℚ), 
    x = -5 ∧ 
    p = 14/3 ∧ 
    p = -(x^2 - x - 2) / (x - 1) ∧ 
    p = -(x^2 + 2*x - 1) / (x + 2) := by
  sorry

end common_root_and_parameter_l4051_405160


namespace five_leaders_three_cities_l4051_405158

/-- The number of ways to allocate n leaders to k cities, with each city having at least one leader -/
def allocationSchemes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that allocating 5 leaders to 3 cities results in 240 schemes -/
theorem five_leaders_three_cities : allocationSchemes 5 3 = 240 := by sorry

end five_leaders_three_cities_l4051_405158


namespace tangent_line_to_parabola_l4051_405157

theorem tangent_line_to_parabola (d : ℝ) : 
  (∃ (x y : ℝ), y = 3*x + d ∧ y^2 = 12*x ∧ 
   ∀ (x' y' : ℝ), y' = 3*x' + d → y'^2 ≥ 12*x') → d = 1 := by
  sorry

end tangent_line_to_parabola_l4051_405157


namespace rectangle_side_lengths_l4051_405161

/-- Given a rectangle DRAK with area 44, rectangle DUPE with area 64,
    and polygon DUPLAK with area 92, this theorem proves that there are
    only three possible sets of integer side lengths for the polygon. -/
theorem rectangle_side_lengths :
  ∀ (dr de du dk pl la : ℕ),
    dr * de = 16 →
    dr * dk = 44 →
    du * de = 64 →
    dk - de = la →
    du - dr = pl →
    (dr, de, du, dk, pl, la) ∈ ({(1, 16, 4, 44, 3, 28), (2, 8, 8, 22, 6, 14), (4, 4, 16, 11, 12, 7)} : Set (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)) :=
by sorry

end rectangle_side_lengths_l4051_405161


namespace inequality_proof_l4051_405146

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 3 / 2) :
  x + 4 * y + 9 * z ≥ 10 := by
  sorry

end inequality_proof_l4051_405146


namespace smith_family_buffet_cost_l4051_405137

/-- Represents the cost calculation for a family at a seafood buffet. -/
def buffet_cost (adult_price : ℚ) (child_price : ℚ) (senior_discount : ℚ) 
  (num_adults num_seniors num_children : ℕ) : ℚ :=
  (num_adults * adult_price) + 
  (num_seniors * (adult_price * (1 - senior_discount))) + 
  (num_children * child_price)

/-- Theorem stating that the total cost for Mr. Smith's family at the seafood buffet is $159. -/
theorem smith_family_buffet_cost : 
  buffet_cost 30 15 (1/10) 2 2 3 = 159 := by
  sorry

end smith_family_buffet_cost_l4051_405137


namespace binary_table_theorem_l4051_405107

/-- Represents a table filled with 0s and 1s -/
def BinaryTable := List (List Bool)

/-- Checks if all rows in the table are unique -/
def allRowsUnique (table : BinaryTable) : Prop :=
  ∀ i j, i ≠ j → table.get! i ≠ table.get! j

/-- Checks if any 4×2 sub-table has two identical rows -/
def anySubTableHasTwoIdenticalRows (table : BinaryTable) : Prop :=
  ∀ c₁ c₂ r₁ r₂ r₃ r₄, 
    c₁ < table.head!.length → c₂ < table.head!.length → c₁ ≠ c₂ →
    r₁ < table.length → r₂ < table.length → r₃ < table.length → r₄ < table.length →
    r₁ ≠ r₂ → r₁ ≠ r₃ → r₁ ≠ r₄ → r₂ ≠ r₃ → r₂ ≠ r₄ → r₃ ≠ r₄ →
    ∃ i j, i ≠ j ∧ 
      (table.get! i).get! c₁ = (table.get! j).get! c₁ ∧
      (table.get! i).get! c₂ = (table.get! j).get! c₂

/-- Checks if a column has exactly one occurrence of a number -/
def columnHasExactlyOneOccurrence (table : BinaryTable) (col : Nat) : Prop :=
  (table.map (λ row => row.get! col)).count true = 1 ∨
  (table.map (λ row => row.get! col)).count false = 1

theorem binary_table_theorem (table : BinaryTable) 
  (h1 : allRowsUnique table)
  (h2 : anySubTableHasTwoIdenticalRows table) :
  ∃ col, columnHasExactlyOneOccurrence table col := by
  sorry


end binary_table_theorem_l4051_405107


namespace probability_one_black_one_white_l4051_405115

/-- The probability of selecting one black ball and one white ball from a jar containing 6 black balls and 2 white balls when picking two balls at the same time. -/
theorem probability_one_black_one_white (black_balls : ℕ) (white_balls : ℕ) 
  (h1 : black_balls = 6) (h2 : white_balls = 2) :
  (black_balls * white_balls : ℚ) / (Nat.choose (black_balls + white_balls) 2) = 3/7 := by
  sorry

end probability_one_black_one_white_l4051_405115


namespace law_firm_associates_tenure_l4051_405174

theorem law_firm_associates_tenure (total : ℝ) (first_year : ℝ) (second_year : ℝ) (more_than_two_years : ℝ)
  (h1 : second_year / total = 0.3)
  (h2 : (total - first_year) / total = 0.6) :
  more_than_two_years / total = 0.6 - 0.3 := by
sorry

end law_firm_associates_tenure_l4051_405174


namespace prime_factor_sum_l4051_405131

theorem prime_factor_sum (w x y z t : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 17^t = 107100 →
  2*w + 3*x + 5*y + 7*z + 11*t = 38 := by
sorry

end prime_factor_sum_l4051_405131


namespace quadratic_equation_solution_l4051_405169

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -4 ∧ x₂ = -4.5 ∧ 
  (∀ x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 7) ↔ x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_equation_solution_l4051_405169


namespace estimate_rabbit_population_l4051_405120

/-- Estimate the number of rabbits in a forest using the capture-recapture method. -/
theorem estimate_rabbit_population (initial_marked : ℕ) (second_capture : ℕ) (marked_in_second : ℕ) :
  initial_marked = 50 →
  second_capture = 42 →
  marked_in_second = 5 →
  (initial_marked * second_capture) / marked_in_second = 420 :=
by
  sorry

#check estimate_rabbit_population

end estimate_rabbit_population_l4051_405120


namespace adult_ticket_cost_l4051_405171

/-- Proves that the cost of each adult ticket is $4.50 -/
theorem adult_ticket_cost :
  let student_ticket_price : ℚ := 2
  let total_tickets : ℕ := 20
  let total_income : ℚ := 60
  let student_tickets_sold : ℕ := 12
  let adult_tickets_sold : ℕ := total_tickets - student_tickets_sold
  let adult_ticket_price : ℚ := (total_income - (student_ticket_price * student_tickets_sold)) / adult_tickets_sold
  adult_ticket_price = 4.5 := by
  sorry

end adult_ticket_cost_l4051_405171


namespace simple_interest_rate_l4051_405192

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time_years simple_interest : ℝ) :
  principal = 10000 →
  time_years = 1 →
  simple_interest = 400 →
  (simple_interest / (principal * time_years)) * 100 = 4 := by
sorry

end simple_interest_rate_l4051_405192


namespace candidate_B_votes_l4051_405177

/-- Represents a candidate in the election --/
inductive Candidate
  | A | B | C | D | E

/-- The total number of people in the class --/
def totalVotes : Nat := 46

/-- The number of votes received by candidate A --/
def votesForA : Nat := 25

/-- The number of votes received by candidate E --/
def votesForE : Nat := 4

/-- The voting results satisfy the given conditions --/
def validVotingResult (votes : Candidate → Nat) : Prop :=
  votes Candidate.A = votesForA ∧
  votes Candidate.E = votesForE ∧
  votes Candidate.B > votes Candidate.E ∧
  votes Candidate.B < votes Candidate.A ∧
  votes Candidate.C = votes Candidate.D ∧
  votes Candidate.A + votes Candidate.B + votes Candidate.C + votes Candidate.D + votes Candidate.E = totalVotes

theorem candidate_B_votes (votes : Candidate → Nat) 
  (h : validVotingResult votes) : votes Candidate.B = 7 := by
  sorry

end candidate_B_votes_l4051_405177


namespace gerald_initial_farthings_l4051_405119

/-- The number of farthings in a pfennig -/
def farthings_per_pfennig : ℕ := 6

/-- The cost of a meat pie in pfennigs -/
def pie_cost : ℕ := 2

/-- The number of pfennigs Gerald has left after buying the pie -/
def pfennigs_left : ℕ := 7

/-- The number of farthings Gerald has initially -/
def initial_farthings : ℕ := 54

theorem gerald_initial_farthings :
  initial_farthings = 
    pie_cost * farthings_per_pfennig + pfennigs_left * farthings_per_pfennig :=
by sorry

end gerald_initial_farthings_l4051_405119


namespace smallest_percentage_both_drinks_l4051_405104

/-- The percentage of adults who drink coffee -/
def coffee_drinkers : ℝ := 90

/-- The percentage of adults who drink tea -/
def tea_drinkers : ℝ := 85

/-- The smallest possible percentage of adults who drink both coffee and tea -/
def both_drinkers : ℝ := 75

theorem smallest_percentage_both_drinks (coffee_drinkers tea_drinkers both_drinkers : ℝ) 
  (h1 : coffee_drinkers = 90) 
  (h2 : tea_drinkers = 85) : 
  both_drinkers ≥ 75 ∧ ∃ (x : ℝ), x ≥ 75 ∧ 
  coffee_drinkers + tea_drinkers - x ≤ 100 :=
by sorry

end smallest_percentage_both_drinks_l4051_405104


namespace mean_home_runs_l4051_405194

def players_5 : ℕ := 4
def players_6 : ℕ := 3
def players_7 : ℕ := 2
def players_9 : ℕ := 1
def players_11 : ℕ := 1

def total_players : ℕ := players_5 + players_6 + players_7 + players_9 + players_11

def total_home_runs : ℕ := 5 * players_5 + 6 * players_6 + 7 * players_7 + 9 * players_9 + 11 * players_11

theorem mean_home_runs : 
  (total_home_runs : ℚ) / (total_players : ℚ) = 6.545454545 := by
  sorry

end mean_home_runs_l4051_405194


namespace decreasing_function_positive_range_l4051_405106

-- Define a decreasing function f on ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem decreasing_function_positive_range
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_derivative : ∀ x, HasDerivAt f (f' x) x)
  (h_inequality : ∀ x, f x / f' x + x < 1) :
  ∀ x, f x > 0 ↔ x > 1 := by
  sorry

end decreasing_function_positive_range_l4051_405106


namespace paint_cans_theorem_l4051_405148

/-- Represents the number of rooms that can be painted with the original amount of paint -/
def original_rooms : ℕ := 40

/-- Represents the number of rooms that can be painted after losing some paint -/
def remaining_rooms : ℕ := 31

/-- Represents the number of cans lost -/
def lost_cans : ℕ := 3

/-- Calculates the number of cans used to paint a given number of rooms -/
def cans_used (rooms : ℕ) : ℕ :=
  (rooms + 2) / 3

theorem paint_cans_theorem : cans_used remaining_rooms = 11 := by
  sorry

end paint_cans_theorem_l4051_405148


namespace treasure_day_l4051_405136

/-- Pongpong's starting amount -/
def pongpong_start : ℕ := 8000

/-- Longlong's starting amount -/
def longlong_start : ℕ := 5000

/-- Pongpong's daily increase -/
def pongpong_daily : ℕ := 300

/-- Longlong's daily increase -/
def longlong_daily : ℕ := 500

/-- The number of days until Pongpong and Longlong have the same amount -/
def days_until_equal : ℕ := 15

theorem treasure_day :
  pongpong_start + pongpong_daily * days_until_equal =
  longlong_start + longlong_daily * days_until_equal :=
by sorry

end treasure_day_l4051_405136


namespace digit_2003_is_4_l4051_405128

/-- Calculates the digit at a given position in the sequence of natural numbers written consecutively -/
def digitAtPosition (n : ℕ) : ℕ :=
  sorry

/-- The 2003rd digit in the sequence of natural numbers written consecutively is 4 -/
theorem digit_2003_is_4 : digitAtPosition 2003 = 4 := by
  sorry

end digit_2003_is_4_l4051_405128


namespace f_g_minus_g_f_l4051_405130

def f (x : ℝ) : ℝ := 2 * x - 1

def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem f_g_minus_g_f : f (g 3) - g (f 3) = -5 := by
  sorry

end f_g_minus_g_f_l4051_405130


namespace first_fun_friday_l4051_405105

/-- Represents a day of the week -/
inductive DayOfWeek
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- The company's year starts on Thursday, March 1st -/
def yearStart : MarchDate :=
  { day := 1, dayOfWeek := DayOfWeek.thursday }

/-- March has 31 days -/
def marchDays : Nat := 31

/-- Determines if a given date is a Friday -/
def isFriday (date : MarchDate) : Prop :=
  date.dayOfWeek = DayOfWeek.friday

/-- Counts the number of Fridays up to and including a given date in March -/
def fridayCount (date : MarchDate) : Nat :=
  sorry

/-- Determines if a given date is a Fun Friday -/
def isFunFriday (date : MarchDate) : Prop :=
  isFriday date ∧ fridayCount date = 5

/-- The theorem to be proved -/
theorem first_fun_friday : 
  ∃ (date : MarchDate), date.day = 30 ∧ isFunFriday date :=
sorry

end first_fun_friday_l4051_405105


namespace symmetry_wrt_y_axis_l4051_405117

/-- Given a point P in a 3D Cartesian coordinate system, 
    return its symmetric point P' with respect to the y-axis -/
def symmetric_point_y_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := P
  (-x, y, -z)

theorem symmetry_wrt_y_axis :
  let P : ℝ × ℝ × ℝ := (2, -4, 6)
  symmetric_point_y_axis P = (-2, -4, -6) := by
sorry

end symmetry_wrt_y_axis_l4051_405117


namespace vacation_tents_l4051_405110

/-- Represents the sleeping arrangements for a family vacation --/
structure SleepingArrangements where
  indoor_capacity : ℕ
  max_per_tent : ℕ
  total_people : ℕ
  teenagers : ℕ
  young_children : ℕ
  infant_families : ℕ
  single_adults : ℕ
  dogs : ℕ

/-- Calculates the number of tents needed given the sleeping arrangements --/
def calculate_tents (arrangements : SleepingArrangements) : ℕ :=
  let outdoor_people := arrangements.total_people - arrangements.indoor_capacity
  let teen_tents := (arrangements.teenagers + 1) / 2
  let child_tents := (arrangements.young_children + 1) / 2
  let adult_tents := (outdoor_people - arrangements.teenagers - arrangements.young_children - arrangements.infant_families + 1) / 2
  teen_tents + child_tents + adult_tents + arrangements.dogs

/-- Theorem stating that the given sleeping arrangements require 7 tents --/
theorem vacation_tents (arrangements : SleepingArrangements) 
  (h1 : arrangements.indoor_capacity = 6)
  (h2 : arrangements.max_per_tent = 2)
  (h3 : arrangements.total_people = 20)
  (h4 : arrangements.teenagers = 2)
  (h5 : arrangements.young_children = 5)
  (h6 : arrangements.infant_families = 3)
  (h7 : arrangements.single_adults = 1)
  (h8 : arrangements.dogs = 1) :
  calculate_tents arrangements = 7 := by
  sorry


end vacation_tents_l4051_405110


namespace total_paintable_area_l4051_405151

/-- Represents a rectangular surface with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents a wall with its dimensions and optional window or door -/
structure Wall where
  dimensions : Rectangle
  opening : Option Rectangle

/-- Calculates the paintable area of a wall -/
def Wall.paintableArea (w : Wall) : ℝ :=
  w.dimensions.area - (match w.opening with
    | some o => o.area
    | none => 0)

/-- The four walls of the room -/
def walls : List Wall := [
  { dimensions := { width := 4, height := 8 },
    opening := some { width := 2, height := 3 } },
  { dimensions := { width := 6, height := 8 },
    opening := some { width := 3, height := 6.5 } },
  { dimensions := { width := 4, height := 8 },
    opening := some { width := 3, height := 4 } },
  { dimensions := { width := 6, height := 8 },
    opening := none }
]

theorem total_paintable_area :
  (walls.map Wall.paintableArea).sum = 122.5 := by sorry

end total_paintable_area_l4051_405151


namespace jerrys_age_l4051_405116

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 22 → 
  mickey_age = 2 * jerry_age - 6 → 
  jerry_age = 14 := by
sorry

end jerrys_age_l4051_405116


namespace volleyball_team_starters_l4051_405125

def number_of_players : ℕ := 16
def number_of_triplets : ℕ := 3
def number_of_twins : ℕ := 2
def number_of_starters : ℕ := 7

def remaining_players : ℕ := number_of_players - number_of_triplets - number_of_twins

theorem volleyball_team_starters :
  (number_of_triplets * number_of_twins * (Nat.choose remaining_players (number_of_starters - 2))) = 2772 := by
  sorry

end volleyball_team_starters_l4051_405125


namespace wand_original_price_l4051_405197

/-- If a price is one-eighth of the original price and equals $12, then the original price is $96. -/
theorem wand_original_price (price : ℝ) (original : ℝ) : 
  price = original * (1/8) → price = 12 → original = 96 := by sorry

end wand_original_price_l4051_405197
