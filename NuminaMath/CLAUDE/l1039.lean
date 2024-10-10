import Mathlib

namespace min_value_x_plus_2y_l1039_103904

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y^2 = 4) :
  x + 2*y ≥ 3 * Real.rpow 4 (1/3) := by
sorry

end min_value_x_plus_2y_l1039_103904


namespace mixture_division_l1039_103964

/-- Converts pounds to ounces -/
def pounds_to_ounces (pounds : ℚ) : ℚ := pounds * 16

/-- Calculates the amount of mixture in each container -/
def mixture_per_container (total_weight : ℚ) (num_containers : ℕ) : ℚ :=
  (pounds_to_ounces total_weight) / num_containers

theorem mixture_division (total_weight : ℚ) (num_containers : ℕ) 
  (h1 : total_weight = 57 + 3/8) 
  (h2 : num_containers = 7) :
  ∃ (ε : ℚ), abs (mixture_per_container total_weight num_containers - 131.14) < ε ∧ ε > 0 :=
by sorry

end mixture_division_l1039_103964


namespace equation_solutions_l1039_103944

theorem equation_solutions : 
  ∀ x : ℝ, 2*x - 6 = 3*x*(x - 3) ↔ x = 3 ∨ x = 2/3 :=
by sorry

end equation_solutions_l1039_103944


namespace marks_radiator_cost_l1039_103941

/-- The total cost for Mark's car radiator replacement -/
def total_cost (labor_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  labor_hours * hourly_rate + part_cost

/-- Proof that Mark's total cost for car radiator replacement is $300 -/
theorem marks_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end marks_radiator_cost_l1039_103941


namespace min_sum_of_product_72_l1039_103915

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) :
  ∀ x y : ℤ, x * y = 72 → a + b ≤ x + y :=
by sorry

end min_sum_of_product_72_l1039_103915


namespace perpendicular_lines_a_values_l1039_103931

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 2) * x + 3 * y + a = 0
def l₂ (a x y : ℝ) : Prop := a * x + (a - 2) * y - 1 = 0

-- Define perpendicularity condition
def perpendicular (a : ℝ) : Prop := (a - 2) * a + 3 * (a - 2) = 0

-- Theorem statement
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular a → a = 2 ∨ a = -3 := by sorry

end perpendicular_lines_a_values_l1039_103931


namespace grade_distribution_l1039_103908

theorem grade_distribution (total_students : ℕ) 
  (below_b_percent : ℚ) (b_or_bplus_percent : ℚ) (a_or_aminus_percent : ℚ) (aplus_percent : ℚ) :
  total_students = 60 →
  below_b_percent = 40 / 100 →
  b_or_bplus_percent = 30 / 100 →
  a_or_aminus_percent = 20 / 100 →
  aplus_percent = 10 / 100 →
  below_b_percent + b_or_bplus_percent + a_or_aminus_percent + aplus_percent = 1 →
  (b_or_bplus_percent + a_or_aminus_percent) * total_students = 30 := by
  sorry

end grade_distribution_l1039_103908


namespace ticket_cost_difference_l1039_103966

def adult_count : ℕ := 9
def child_count : ℕ := 7
def adult_ticket_price : ℚ := 11
def child_ticket_price : ℚ := 7
def discount_rate : ℚ := 0.15
def discount_threshold : ℕ := 10

def total_tickets : ℕ := adult_count + child_count

def adult_total : ℚ := adult_count * adult_ticket_price
def child_total : ℚ := child_count * child_ticket_price
def total_cost : ℚ := adult_total + child_total

def discount_applies : Prop := total_tickets > discount_threshold

def discounted_cost : ℚ := total_cost * (1 - discount_rate)

def adult_proportion : ℚ := adult_total / total_cost
def child_proportion : ℚ := child_total / total_cost

def adult_discounted : ℚ := adult_total - (discount_rate * total_cost * adult_proportion)
def child_discounted : ℚ := child_total - (discount_rate * total_cost * child_proportion)

theorem ticket_cost_difference : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |adult_discounted - child_discounted - 42.52| < ε :=
sorry

end ticket_cost_difference_l1039_103966


namespace inequality_solution_l1039_103929

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 3) ↔ (-2 < x ∧ x ≤ 1) :=
by sorry

end inequality_solution_l1039_103929


namespace right_triangle_hypotenuse_l1039_103960

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 9 → b = 12 → c^2 = a^2 + b^2 → c = 15 :=
by
  sorry

end right_triangle_hypotenuse_l1039_103960


namespace a_has_winning_strategy_l1039_103983

/-- Represents the state of the game board -/
structure GameState where
  primes : List Nat
  product_mod_4 : Nat

/-- Represents a move in the game -/
inductive Move
  | erase_and_write (n : Nat) (erased : List Nat) (written : List Nat)

/-- The game between players A and B -/
def Game :=
  List Move

/-- Checks if a number is an odd prime -/
def is_odd_prime (n : Nat) : Prop :=
  Nat.Prime n ∧ n % 2 = 1

/-- The initial setup of the game -/
def initial_setup (primes : List Nat) : Prop :=
  primes.length = 1000 ∧ ∀ p ∈ primes, is_odd_prime p

/-- B's selection of primes -/
def b_selection (all_primes : List Nat) (selected : List Nat) : Prop :=
  selected.length = 500 ∧ ∀ p ∈ selected, p ∈ all_primes

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  sorry

/-- Checks if the game is over (board is empty) -/
def is_game_over (state : GameState) : Prop :=
  state.primes.isEmpty

/-- Player A's winning strategy -/
def a_winning_strategy (game : Game) : Prop :=
  sorry

/-- The main theorem stating that player A has a winning strategy -/
theorem a_has_winning_strategy 
  (initial_primes : List Nat)
  (h_initial : initial_setup initial_primes)
  (b_primes : List Nat)
  (h_b_selection : b_selection initial_primes b_primes) :
  ∃ (strategy : Game), a_winning_strategy strategy :=
sorry

end a_has_winning_strategy_l1039_103983


namespace unique_prime_with_remainder_l1039_103921

theorem unique_prime_with_remainder : ∃! n : ℕ,
  20 < n ∧ n < 30 ∧
  Prime n ∧
  n % 8 = 5 :=
by
  -- The proof would go here
  sorry

end unique_prime_with_remainder_l1039_103921


namespace trapezoid_area_l1039_103982

-- Define the lengths of the line segments
def a : ℝ := 1
def b : ℝ := 4
def c : ℝ := 4
def d : ℝ := 5

-- Define the possible areas
def area1 : ℝ := 6
def area2 : ℝ := 10

-- Statement of the theorem
theorem trapezoid_area :
  ∃ (S : ℝ), (S = area1 ∨ S = area2) ∧
  (∃ (h1 h2 base1 base2 : ℝ),
    (h1 = b ∧ h2 = c ∧ base1 = a ∧ base2 = d) ∨
    (h1 = b ∧ h2 = d ∧ base1 = a ∧ base2 = c) ∨
    (h1 = c ∧ h2 = d ∧ base1 = a ∧ base2 = b)) ∧
  S = (base1 + base2) * (h1 + h2) / 4 :=
sorry

end trapezoid_area_l1039_103982


namespace polynomial_roots_l1039_103919

def P (x : ℂ) : ℂ := x^5 - 5*x^4 + 11*x^3 - 13*x^2 + 9*x - 3

theorem polynomial_roots :
  let roots : List ℂ := [1, (3 + Complex.I * Real.sqrt 3) / 2, (1 - Complex.I * Real.sqrt 3) / 2,
                         (3 - Complex.I * Real.sqrt 3) / 2, (1 + Complex.I * Real.sqrt 3) / 2]
  ∀ x : ℂ, (P x = 0) ↔ (x ∈ roots) :=
by sorry

end polynomial_roots_l1039_103919


namespace initial_average_height_l1039_103988

/-- Given a class of students with an incorrect height measurement,
    prove that the initially calculated average height is 174 cm. -/
theorem initial_average_height
  (n : ℕ)  -- number of students
  (incorrect_height correct_height : ℝ)  -- heights of the misrecorded student
  (actual_average : ℝ)  -- actual average height after correction
  (h_n : n = 30)  -- there are 30 students
  (h_incorrect : incorrect_height = 151)  -- incorrectly recorded height
  (h_correct : correct_height = 136)  -- actual height of the misrecorded student
  (h_actual_avg : actual_average = 174.5)  -- actual average height
  : (n * actual_average - (incorrect_height - correct_height)) / n = 174 := by
  sorry

end initial_average_height_l1039_103988


namespace fraction_of_fraction_two_ninths_of_three_fourths_l1039_103992

theorem fraction_of_fraction (a b c d : ℚ) (h : b ≠ 0) (k : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem two_ninths_of_three_fourths :
  (2 / 9) / (3 / 4) = 8 / 27 :=
by sorry

end fraction_of_fraction_two_ninths_of_three_fourths_l1039_103992


namespace range_of_m2_plus_n2_l1039_103934

/-- An increasing function f with the property f(-x) + f(x) = 0 for all x -/
def IncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) + f x = 0)

theorem range_of_m2_plus_n2 
  (f : ℝ → ℝ) (m n : ℝ) 
  (h_f : IncreasingOddFunction f) 
  (h_ineq : f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0) :
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 := by
  sorry

end range_of_m2_plus_n2_l1039_103934


namespace race_time_calculation_l1039_103937

theorem race_time_calculation (race_length : ℝ) (distance_difference : ℝ) (time_difference : ℝ) :
  race_length = 1000 →
  distance_difference = 40 →
  time_difference = 8 →
  ∃ (time_A : ℝ),
    time_A > 0 ∧
    race_length / time_A = (race_length - distance_difference) / (time_A + time_difference) ∧
    time_A = 200 := by
  sorry

end race_time_calculation_l1039_103937


namespace triangle_height_relationship_l1039_103943

/-- Given two triangles A and B, proves the relationship between their heights
    when their bases and areas are related. -/
theorem triangle_height_relationship (b h : ℝ) (h_pos : 0 < h) (b_pos : 0 < b) :
  let base_A := 1.2 * b
  let area_B := (1 / 2) * b * h
  let area_A := 0.9975 * area_B
  let height_A := (2 * area_A) / base_A
  height_A / h = 0.83125 :=
by sorry

end triangle_height_relationship_l1039_103943


namespace sabrina_video_votes_l1039_103903

theorem sabrina_video_votes (total_votes : ℕ) (upvotes downvotes : ℕ) (score : ℤ) : 
  upvotes = (3 * total_votes) / 4 →
  downvotes = total_votes / 4 →
  score = 150 →
  (upvotes : ℤ) - (downvotes : ℤ) = score →
  total_votes = 300 := by
sorry

end sabrina_video_votes_l1039_103903


namespace sexagenary_cycle_3023_l1039_103976

/-- Represents a year in the sexagenary cycle -/
structure SexagenaryYear where
  heavenlyStem : Fin 10
  earthlyBranch : Fin 12

/-- The sexagenary cycle -/
def sexagenaryCycle : ℕ → SexagenaryYear := sorry

/-- Maps a natural number to its representation in the sexagenary cycle -/
def toSexagenaryYear (year : ℕ) : SexagenaryYear :=
  sexagenaryCycle (year % 60)

/-- Checks if a given SexagenaryYear corresponds to "Gui Mao" -/
def isGuiMao (year : SexagenaryYear) : Prop :=
  year.heavenlyStem = 9 ∧ year.earthlyBranch = 3

/-- Checks if a given SexagenaryYear corresponds to "Gui Wei" -/
def isGuiWei (year : SexagenaryYear) : Prop :=
  year.heavenlyStem = 9 ∧ year.earthlyBranch = 7

theorem sexagenary_cycle_3023 :
  isGuiMao (toSexagenaryYear 2023) →
  isGuiWei (toSexagenaryYear 3023) := by
  sorry

end sexagenary_cycle_3023_l1039_103976


namespace shopping_expenditure_l1039_103974

theorem shopping_expenditure (initial_amount : ℝ) : 
  initial_amount * (1 - 0.2) * (1 - 0.15) * (1 - 0.25) = 217 →
  initial_amount = 425.49 := by
sorry

end shopping_expenditure_l1039_103974


namespace equation_solution_l1039_103954

theorem equation_solution (x p : ℝ) : 
  (Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x) ↔ 
  (x = (4 - p) / Real.sqrt (8 * (2 - p)) ∧ 0 ≤ p ∧ p ≤ 4/3) :=
by sorry

end equation_solution_l1039_103954


namespace special_triangle_properties_l1039_103932

/-- Represents a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Conditions for our specific triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.A + t.C = 2 * Real.pi / 3 ∧
  t.b = 1 ∧
  0 < t.A ∧ t.A < Real.pi / 2 ∧
  0 < t.B ∧ t.B < Real.pi / 2 ∧
  0 < t.C ∧ t.C < Real.pi / 2

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 2 ∧
  ∃ (max_area : Real), max_area = Real.sqrt 3 / 4 ∧
    ∀ (area : Real), area = 1 / 2 * t.a * t.c * Real.sin t.B → area ≤ max_area :=
by sorry

end special_triangle_properties_l1039_103932


namespace lemonade_cups_calculation_l1039_103945

theorem lemonade_cups_calculation (sugar_cups : ℕ) (ratio : ℚ) : 
  sugar_cups = 28 → ratio = 1/2 → sugar_cups + (sugar_cups / ratio) = 84 := by
  sorry

end lemonade_cups_calculation_l1039_103945


namespace min_degree_for_connected_system_l1039_103938

/-- A graph representing a road system in a kingdom --/
structure RoadSystem where
  cities : Finset Nat
  roads : Finset (Nat × Nat)
  city_count : cities.card = 8
  road_symmetry : ∀ a b, (a, b) ∈ roads → (b, a) ∈ roads

/-- The maximum number of roads leading out from any city --/
def max_degree (g : RoadSystem) : Nat :=
  g.cities.sup (λ c => (g.roads.filter (λ r => r.1 = c)).card)

/-- A path between two cities with at most one intermediate city --/
def has_short_path (g : RoadSystem) (a b : Nat) : Prop :=
  (a, b) ∈ g.roads ∨ ∃ c, (a, c) ∈ g.roads ∧ (c, b) ∈ g.roads

/-- The property that any two cities are connected by a short path --/
def all_cities_connected (g : RoadSystem) : Prop :=
  ∀ a b, a ∈ g.cities → b ∈ g.cities → a ≠ b → has_short_path g a b

/-- The main theorem: the minimum degree for a connected road system is greater than 2 --/
theorem min_degree_for_connected_system (g : RoadSystem) (h : all_cities_connected g) :
  max_degree g > 2 := by
  sorry


end min_degree_for_connected_system_l1039_103938


namespace max_area_triangle_l1039_103981

/-- Given a triangle ABC where angle B equals angle C and 7a² + b² + c² = 4√3,
    the maximum possible area of the triangle is √5/5. -/
theorem max_area_triangle (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
    (h2 : 7 * a^2 + b^2 + c^2 = 4 * Real.sqrt 3)
    (h3 : b = c) : 
    ∃ (S : ℝ), S = Real.sqrt 5 / 5 ∧ 
    ∀ (A : ℝ), A = 1/2 * a * b * Real.sqrt (1 - (a / (2 * b))^2) → A ≤ S :=
by sorry

end max_area_triangle_l1039_103981


namespace triangle_equality_l1039_103962

theorem triangle_equality (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = c + a ∨ c = a + b := by
  sorry

end triangle_equality_l1039_103962


namespace arithmetic_sequence_terms_l1039_103922

/-- An arithmetic sequence with the given properties has 13 terms -/
theorem arithmetic_sequence_terms (a d : ℝ) (n : ℕ) 
  (h1 : 3 * a + 3 * d = 34)
  (h2 : 3 * a + 3 * (n - 1) * d = 146)
  (h3 : n * (2 * a + (n - 1) * d) / 2 = 390)
  : n = 13 := by
  sorry

end arithmetic_sequence_terms_l1039_103922


namespace factorial_ratio_l1039_103948

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 47 = 117600 := by
  sorry

end factorial_ratio_l1039_103948


namespace min_value_x_plus_4y_l1039_103956

theorem min_value_x_plus_4y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) : 
  x + 4*y ≥ 3/2 + Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1/x₀ + 1/(2*y₀) = 2 ∧ 
    x₀ + 4*y₀ = 3/2 + Real.sqrt 2 :=
sorry

end min_value_x_plus_4y_l1039_103956


namespace frank_saturday_bags_l1039_103933

def total_cans : ℕ := 40
def cans_per_bag : ℕ := 5
def bags_filled_sunday : ℕ := 3

def total_bags : ℕ := total_cans / cans_per_bag

def bags_filled_saturday : ℕ := total_bags - bags_filled_sunday

theorem frank_saturday_bags :
  bags_filled_saturday = 5 :=
by sorry

end frank_saturday_bags_l1039_103933


namespace first_few_terms_eighth_term_l1039_103951

/-- Definition of the sequence -/
def a (n : ℕ) : ℕ := n^2 + 2*n - 1

/-- The first few terms of the sequence -/
theorem first_few_terms :
  a 1 = 2 ∧ a 2 = 7 ∧ a 3 = 14 ∧ a 4 = 23 := by sorry

/-- The 8th term of the sequence is 79 -/
theorem eighth_term : a 8 = 79 := by sorry

end first_few_terms_eighth_term_l1039_103951


namespace max_value_expression_l1039_103955

theorem max_value_expression (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0) 
  (sum_constraint : x + y + z = 3) :
  (x^3 - x*y^2 + y^3) * (x^3 - x^2*z + z^3) * (y^3 - y^2*z + z^3) ≤ 1 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ z₀ ≥ 0 ∧ x₀ + y₀ + z₀ = 3 ∧
    (x₀^3 - x₀*y₀^2 + y₀^3) * (x₀^3 - x₀^2*z₀ + z₀^3) * (y₀^3 - y₀^2*z₀ + z₀^3) = 1 :=
by sorry

end max_value_expression_l1039_103955


namespace rectangle_perimeter_minus_4_l1039_103998

/-- The perimeter of a rectangle minus 4, given its width and length. -/
def perimeterMinus4 (width length : ℝ) : ℝ :=
  2 * width + 2 * length - 4

/-- Theorem: For a rectangle with width 4 cm and length 8 cm, 
    the perimeter minus 4 equals 20 cm. -/
theorem rectangle_perimeter_minus_4 :
  perimeterMinus4 4 8 = 20 := by
  sorry

end rectangle_perimeter_minus_4_l1039_103998


namespace line_problem_l1039_103902

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 2 * y - 1 = 0
def l₂ (x y : ℝ) : Prop := 5 * x + 2 * y + 1 = 0
def l₃ (a x y : ℝ) : Prop := (a^2 - 1) * x + a * y - 1 = 0

-- Define the intersection point A
def A : ℝ × ℝ := (-1, 2)

-- Define parallelism
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, f x y ↔ g (k * x) (k * y)

-- Define a line with equal intercepts
def equal_intercepts (m b : ℝ) : Prop :=
  b / m + b = 0

theorem line_problem :
  (∃ a : ℝ, parallel (l₃ a) l₁ ∧ a = -1/2) ∧
  (∃ m b : ℝ, (m = -1 ∧ b = 1) ∨ (m = -2 ∧ b = 0) ∧
    l₁ A.1 A.2 ∧ l₂ A.1 A.2 ∧ equal_intercepts m b) :=
sorry

end line_problem_l1039_103902


namespace day_301_is_sunday_l1039_103995

/-- Days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Function to determine the day of the week given a day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- Theorem: If the 35th day is a Sunday, then the 301st day is also a Sunday -/
theorem day_301_is_sunday (h : dayOfWeek 35 = DayOfWeek.Sunday) :
  dayOfWeek 301 = DayOfWeek.Sunday :=
by
  sorry


end day_301_is_sunday_l1039_103995


namespace total_days_1996_to_2000_l1039_103994

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

def totalDays (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).map (fun i => daysInYear (startYear + i))
    |> List.sum

theorem total_days_1996_to_2000 :
  totalDays 1996 2000 = 1827 := by sorry

end total_days_1996_to_2000_l1039_103994


namespace geometric_sequence_common_ratio_l1039_103969

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∃ (a₁ q : ℝ), ∀ n, a n = geometric_sequence a₁ q n)
  (h_a₁ : a 1 = 2)
  (h_a₄ : a 4 = 16) :
  ∃ q, ∀ n, a n = geometric_sequence 2 q n ∧ q = 2 :=
sorry

end geometric_sequence_common_ratio_l1039_103969


namespace polynomial_property_l1039_103991

-- Define the polynomial Q(x)
def Q (x : ℝ) (d e : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + 9

-- Define the conditions
theorem polynomial_property (d e : ℝ) :
  -- The mean of zeros equals the product of zeros
  (-(d / 3) / 3 = -3) →
  -- The product of zeros equals the sum of coefficients
  (-3 = 3 + d + e + 9) →
  -- The y-intercept is 9
  (Q 0 d e = 9) →
  -- Prove that e equals -42
  e = -42 := by sorry

end polynomial_property_l1039_103991


namespace x_plus_3y_equals_1_l1039_103980

theorem x_plus_3y_equals_1 (x y : ℝ) (h1 : x + y = 19) (h2 : x + 2*y = 10) : x + 3*y = 1 := by
  sorry

end x_plus_3y_equals_1_l1039_103980


namespace investment_growth_proof_l1039_103989

/-- The initial investment amount that results in $132 after two years with given growth rates and addition --/
def initial_investment : ℝ := 80

/-- The growth rate for the first year --/
def first_year_growth_rate : ℝ := 0.15

/-- The amount added after the first year --/
def added_amount : ℝ := 28

/-- The growth rate for the second year --/
def second_year_growth_rate : ℝ := 0.10

/-- The final portfolio value after two years --/
def final_value : ℝ := 132

theorem investment_growth_proof :
  ((1 + first_year_growth_rate) * initial_investment + added_amount) * 
  (1 + second_year_growth_rate) = final_value := by
  sorry

#eval initial_investment

end investment_growth_proof_l1039_103989


namespace converse_of_negative_square_positive_l1039_103971

theorem converse_of_negative_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, x^2 > 0 → x < 0) :=
sorry

end converse_of_negative_square_positive_l1039_103971


namespace polynomial_root_sum_l1039_103918

theorem polynomial_root_sum (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2500*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  p + q + r = 0 →
  p * q + q * r + r * p = -2500 →
  p * q * r = -m →
  |p| + |q| + |r| = 100 := by
  sorry

end polynomial_root_sum_l1039_103918


namespace benny_baseball_gear_expense_l1039_103900

/-- The amount Benny spent on baseball gear -/
def amount_spent (initial : ℕ) (left_over : ℕ) : ℕ :=
  initial - left_over

/-- Theorem stating that Benny spent 34 dollars on baseball gear -/
theorem benny_baseball_gear_expense :
  amount_spent 67 33 = 34 := by
  sorry

end benny_baseball_gear_expense_l1039_103900


namespace end_with_same_digits_l1039_103947

/-- A function that returns the last four digits of a number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- A function that returns the first three digits of a four-digit number -/
def firstThreeDigits (n : ℕ) : ℕ := n / 10

theorem end_with_same_digits (N : ℕ) (h1 : N > 0) 
  (h2 : lastFourDigits N = lastFourDigits (N^2)) 
  (h3 : lastFourDigits N ≥ 1000) : firstThreeDigits (lastFourDigits N) = 937 := by
  sorry

end end_with_same_digits_l1039_103947


namespace area_enclosed_circles_l1039_103973

/-- The area enclosed between the circumferences of four equal circles described about the corners of a square -/
theorem area_enclosed_circles (s : ℝ) (h : s = 14) :
  let r : ℝ := s / 2
  let square_area : ℝ := s ^ 2
  let circle_segment_area : ℝ := π * r ^ 2
  square_area - circle_segment_area = 196 - 49 * π :=
by sorry

end area_enclosed_circles_l1039_103973


namespace volleyball_betting_strategy_exists_l1039_103905

theorem volleyball_betting_strategy_exists : ∃ (x₁ x₂ x₃ x₄ : ℝ),
  x₁ + x₂ + x₃ + x₄ = 1 ∧
  x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧
  6 * x₁ ≥ 1 ∧ 2 * x₂ ≥ 1 ∧ 6 * x₃ ≥ 1 ∧ 7 * x₄ ≥ 1 := by
  sorry

end volleyball_betting_strategy_exists_l1039_103905


namespace g_at_negative_one_l1039_103958

def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

theorem g_at_negative_one : g (-1) = 5 / 3 := by
  sorry

end g_at_negative_one_l1039_103958


namespace xiaoming_relative_score_l1039_103940

def class_average : ℝ := 90
def xiaoming_score : ℝ := 85

theorem xiaoming_relative_score :
  xiaoming_score - class_average = -5 := by
sorry

end xiaoming_relative_score_l1039_103940


namespace probability_ratio_l1039_103979

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The probability of drawing all five slips with the same number -/
def p : ℚ := (distinct_numbers * 1) / Nat.choose total_slips drawn_slips

/-- The probability of drawing three slips with one number and two with another -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

theorem probability_ratio :
  q / p = 450 := by sorry

end probability_ratio_l1039_103979


namespace rotten_apples_percentage_l1039_103939

theorem rotten_apples_percentage (total : ℕ) (good : ℕ) 
  (h1 : total = 75) (h2 : good = 66) : 
  (((total - good : ℚ) / total) * 100 : ℚ) = 12 := by
  sorry

end rotten_apples_percentage_l1039_103939


namespace intersection_of_A_and_B_l1039_103935

def set_A : Set ℝ := {x | Real.sqrt (x + 1) < 2}
def set_B : Set ℝ := {x | 1 < x ∧ x < 4}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l1039_103935


namespace train_speed_problem_l1039_103999

theorem train_speed_problem (initial_distance : ℝ) (speed_train1 : ℝ) (distance_before_meet : ℝ) (time_before_meet : ℝ) :
  initial_distance = 120 →
  speed_train1 = 40 →
  distance_before_meet = 70 →
  time_before_meet = 1 →
  ∃ speed_train2 : ℝ,
    speed_train2 = 30 ∧
    initial_distance - (speed_train1 + speed_train2) * time_before_meet = distance_before_meet :=
by sorry

end train_speed_problem_l1039_103999


namespace equation_solution_l1039_103946

theorem equation_solution (x : ℝ) : x * (3 * x + 6) = 7 * (3 * x + 6) ↔ x = 7 ∨ x = -2 := by
  sorry

end equation_solution_l1039_103946


namespace washers_remaining_l1039_103906

/-- Calculates the number of washers remaining after a plumbing job. -/
theorem washers_remaining (pipe_length : ℕ) (feet_per_bolt : ℕ) (washers_per_bolt : ℕ) (initial_washers : ℕ) : 
  pipe_length = 40 ∧ 
  feet_per_bolt = 5 ∧ 
  washers_per_bolt = 2 ∧ 
  initial_washers = 20 → 
  initial_washers - (pipe_length / feet_per_bolt * washers_per_bolt) = 4 := by
sorry

end washers_remaining_l1039_103906


namespace number_1991_position_l1039_103942

/-- Represents a row in the number array -/
structure NumberArrayRow where
  startNumber : Nat
  length : Nat

/-- Defines the pattern of the number array -/
def numberArrayPattern (row : Nat) : NumberArrayRow :=
  { startNumber := row * 10,
    length := if row < 10 then row else 10 + (row - 10) * 10 }

/-- Checks if a number appears in a specific row and position -/
def appearsInRowAndPosition (n : Nat) (row : Nat) (position : Nat) : Prop :=
  let arrayRow := numberArrayPattern row
  n ≥ arrayRow.startNumber ∧ 
  n < arrayRow.startNumber + arrayRow.length ∧
  n = arrayRow.startNumber + position - 1

/-- Theorem stating that 1991 appears in the 199th row and 2nd position -/
theorem number_1991_position :
  appearsInRowAndPosition 1991 199 2 := by
  sorry


end number_1991_position_l1039_103942


namespace continued_fraction_value_l1039_103996

theorem continued_fraction_value : 
  ∃ y : ℝ, y = 3 + 5 / (4 + 5 / y) ∧ y = 5 := by
  sorry

end continued_fraction_value_l1039_103996


namespace correct_calculation_l1039_103952

theorem correct_calculation (a b : ℝ) : 2 * a^2 * b - 4 * a^2 * b = -2 * a^2 * b := by
  sorry

end correct_calculation_l1039_103952


namespace committee_probability_l1039_103967

def total_members : ℕ := 20
def boys : ℕ := 12
def girls : ℕ := 8
def committee_size : ℕ := 4

def probability_at_least_one_boy_and_girl : ℚ :=
  1 - (Nat.choose boys committee_size + Nat.choose girls committee_size : ℚ) / Nat.choose total_members committee_size

theorem committee_probability :
  probability_at_least_one_boy_and_girl = 4280 / 4845 :=
sorry

end committee_probability_l1039_103967


namespace initial_cats_l1039_103924

theorem initial_cats (initial_cats final_cats bought_cats : ℕ) : 
  final_cats = initial_cats + bought_cats ∧ 
  final_cats = 54 ∧ 
  bought_cats = 43 →
  initial_cats = 11 := by sorry

end initial_cats_l1039_103924


namespace min_value_expression_l1039_103907

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (c - a)^2) / b^2 ≥ 4/3 ∧
  ∃ (a' b' c' : ℝ), b' > c' ∧ c' > a' ∧ b' ≠ 0 ∧
    ((a' + b')^2 + (b' - c')^2 + (c' - a')^2) / b'^2 = 4/3 :=
by sorry

end min_value_expression_l1039_103907


namespace investment_change_l1039_103990

/-- Proves that an investment decreasing by 25% and then increasing by 40% results in a 5% overall increase -/
theorem investment_change (initial_value : ℝ) (initial_value_pos : initial_value > 0) :
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  let percent_change := (day2_value - initial_value) / initial_value * 100
  percent_change = 5 := by
  sorry

end investment_change_l1039_103990


namespace triangle_area_product_l1039_103949

theorem triangle_area_product (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + b * y = 12) →
  (1/2 * (12/a) * (12/b) = 12) →
  a * b = 6 := by sorry

end triangle_area_product_l1039_103949


namespace total_distance_driven_l1039_103963

/-- Proves that driving at 55 mph for 2 hours and then 3 hours results in a total distance of 275 miles -/
theorem total_distance_driven (speed : ℝ) (time_before_lunch : ℝ) (time_after_lunch : ℝ) 
  (h1 : speed = 55)
  (h2 : time_before_lunch = 2)
  (h3 : time_after_lunch = 3) :
  speed * time_before_lunch + speed * time_after_lunch = 275 := by
  sorry

#check total_distance_driven

end total_distance_driven_l1039_103963


namespace polar_coordinates_of_point_l1039_103928

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) :
  x = 1 ∧ y = -Real.sqrt 3 →
  ρ = 2 ∧ θ = 5 * Real.pi / 3 →
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end polar_coordinates_of_point_l1039_103928


namespace quadratic_radicals_theorem_l1039_103910

-- Define the condition that the radicals can be combined
def radicals_can_combine (a : ℝ) : Prop := 3 * a - 8 = 17 - 2 * a

-- Define the range of x that makes √(4a-2x) meaningful
def valid_x_range (a x : ℝ) : Prop := 4 * a - 2 * x ≥ 0

-- Theorem statement
theorem quadratic_radicals_theorem (a x : ℝ) :
  radicals_can_combine a → (∃ a, radicals_can_combine a ∧ a = 5) →
  (valid_x_range a x ↔ x ≤ 10) :=
sorry

end quadratic_radicals_theorem_l1039_103910


namespace modular_inverse_of_2_mod_191_l1039_103923

theorem modular_inverse_of_2_mod_191 : ∃ x : ℕ, x < 191 ∧ (2 * x) % 191 = 1 :=
  ⟨96, by
    constructor
    · simp
    · norm_num
  ⟩

#eval (2 * 96) % 191  -- This should output 1

end modular_inverse_of_2_mod_191_l1039_103923


namespace roots_independent_of_k_l1039_103984

/-- The polynomial function with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^4 - (k+3)*x^3 - (k-11)*x^2 + (k+3)*x + (k-12)

/-- Theorem stating that 1 and -1 are roots of the polynomial for all real k -/
theorem roots_independent_of_k :
  ∀ k : ℝ, f k 1 = 0 ∧ f k (-1) = 0 :=
by sorry

end roots_independent_of_k_l1039_103984


namespace total_packages_l1039_103914

theorem total_packages (num_trucks : ℕ) (packages_per_truck : ℕ) 
  (h1 : num_trucks = 7) 
  (h2 : packages_per_truck = 70) : 
  num_trucks * packages_per_truck = 490 := by
  sorry

end total_packages_l1039_103914


namespace translated_sine_function_l1039_103930

/-- Given a function f and its right-translated version g, prove that g has the expected form. -/
theorem translated_sine_function (f g : ℝ → ℝ) (h : ℝ → ℝ → Prop) : 
  (∀ x, f x = 2 * Real.sin (2 * x + 2 * Real.pi / 3)) →
  (∀ x, h x (g x) ↔ h (x - Real.pi / 6) (f x)) →
  (∀ x, g x = 2 * Real.sin (2 * x + Real.pi / 3)) := by
  sorry


end translated_sine_function_l1039_103930


namespace solve_for_y_l1039_103987

theorem solve_for_y (x y : ℝ) : 3 * x - 2 * y = 6 → y = (3 * x / 2) - 3 := by
  sorry

end solve_for_y_l1039_103987


namespace point_transformation_l1039_103950

def rotate_180 (x y : ℝ) : ℝ × ℝ :=
  (4 - x, 6 - y)

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  (reflect_y_eq_x (rotate_180 a b).1 (rotate_180 a b).2) = (2, -5) →
  a + b = 13 := by
sorry

end point_transformation_l1039_103950


namespace paper_clips_remaining_l1039_103961

theorem paper_clips_remaining (initial : ℕ) (used : ℕ) (remaining : ℕ) : 
  initial = 85 → used = 59 → remaining = initial - used → remaining = 26 := by
  sorry

end paper_clips_remaining_l1039_103961


namespace degree_of_x2y_l1039_103986

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (exponents : List ℕ) : ℕ :=
  exponents.sum

/-- The monomial x^2y has exponents [2, 1] -/
def monomial_x2y_exponents : List ℕ := [2, 1]

theorem degree_of_x2y :
  degree_of_monomial monomial_x2y_exponents = 3 := by
  sorry

end degree_of_x2y_l1039_103986


namespace principal_cup_problem_l1039_103977

/-- The probability of team A answering correctly -/
def P_A : ℚ := 3/4

/-- The probability of both teams A and C answering incorrectly -/
def P_AC_incorrect : ℚ := 1/12

/-- The probability of both teams B and C answering correctly -/
def P_BC_correct : ℚ := 1/4

/-- The probability of team B answering correctly -/
def P_B : ℚ := 3/8

/-- The probability of team C answering correctly -/
def P_C : ℚ := 2/3

/-- The probability of exactly two teams answering correctly -/
def P_two_correct : ℚ := 15/32

theorem principal_cup_problem (P_A P_AC_incorrect P_BC_correct P_B P_C P_two_correct : ℚ) :
  P_A = 3/4 →
  P_AC_incorrect = 1/12 →
  P_BC_correct = 1/4 →
  P_B = 3/8 ∧
  P_C = 2/3 ∧
  P_two_correct = 15/32 :=
by
  sorry

end principal_cup_problem_l1039_103977


namespace sin_graph_shift_l1039_103972

theorem sin_graph_shift (x : ℝ) :
  3 * Real.sin (2 * (x + π/8) - π/4) = 3 * Real.sin (2 * x) := by
  sorry

end sin_graph_shift_l1039_103972


namespace custom_product_of_A_and_B_l1039_103913

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the custom cartesian product operation
def custom_product (X Y : Set ℝ) : Set ℝ := {x : ℝ | x ∈ X ∪ Y ∧ x ∉ X ∩ Y}

-- Theorem statement
theorem custom_product_of_A_and_B :
  custom_product A B = {x : ℝ | x > 2} := by
  sorry

end custom_product_of_A_and_B_l1039_103913


namespace smallest_nonfactor_product_of_48_l1039_103917

theorem smallest_nonfactor_product_of_48 :
  ∃ (u v : ℕ), 
    u ≠ v ∧ 
    u > 0 ∧ 
    v > 0 ∧ 
    48 % u = 0 ∧ 
    48 % v = 0 ∧ 
    48 % (u * v) ≠ 0 ∧
    u * v = 18 ∧
    (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → 48 % x = 0 → 48 % y = 0 → 48 % (x * y) ≠ 0 → x * y ≥ 18) :=
by sorry

end smallest_nonfactor_product_of_48_l1039_103917


namespace class_size_l1039_103916

theorem class_size :
  ∀ (m d : ℕ),
  (m + d > 30) →
  (m + d < 40) →
  (3 * m = 5 * d) →
  (m + d = 32) :=
by
  sorry

end class_size_l1039_103916


namespace erased_value_determinable_l1039_103911

-- Define the type for our circle system
structure CircleSystem where
  -- The values in each circle (we'll use Option to represent the erased circle)
  circle_values : Fin 6 → Option ℝ
  -- The values on each segment
  segment_values : Fin 6 → ℝ

-- Define the property that circle values are sums of incoming segment values
def valid_circle_system (cs : CircleSystem) : Prop :=
  ∀ i : Fin 6, 
    cs.circle_values i = some (cs.segment_values i + cs.segment_values ((i + 5) % 6))

-- Define the property that exactly one circle value is erased (None)
def one_erased (cs : CircleSystem) : Prop :=
  ∃! i : Fin 6, cs.circle_values i = none

-- Theorem stating that the erased value can be determined
theorem erased_value_determinable (cs : CircleSystem) 
  (h1 : valid_circle_system cs) (h2 : one_erased cs) : 
  ∃ (x : ℝ), ∀ (cs' : CircleSystem), 
    valid_circle_system cs' → 
    (∀ i : Fin 6, cs.circle_values i ≠ none → cs'.circle_values i = cs.circle_values i) →
    (∀ i : Fin 6, cs'.segment_values i = cs.segment_values i) →
    (∃ i : Fin 6, cs'.circle_values i = some x ∧ cs.circle_values i = none) :=
sorry

end erased_value_determinable_l1039_103911


namespace jacket_purchase_price_l1039_103909

/-- The purchase price of a jacket given selling price and profit conditions -/
theorem jacket_purchase_price (S P : ℝ) (h1 : S = P + 0.25 * S) 
  (h2 : ∃ D : ℝ, D = 0.8 * S ∧ D - P = 4) : P = 60 := by
  sorry

end jacket_purchase_price_l1039_103909


namespace firstDigitOfPowerOfTwoNotPeriodic_l1039_103926

-- Define the sequence of first digits of powers of 2
def firstDigitOfPowerOfTwo (n : ℕ) : ℕ :=
  (2^n : ℕ).repr.front.toNat

-- Theorem statement
theorem firstDigitOfPowerOfTwoNotPeriodic :
  ¬ ∃ (d : ℕ), d > 0 ∧ ∀ (n : ℕ), firstDigitOfPowerOfTwo (n + d) = firstDigitOfPowerOfTwo n :=
sorry

end firstDigitOfPowerOfTwoNotPeriodic_l1039_103926


namespace problem_solution_l1039_103925

def f (x : ℝ) : ℝ := |x| - |2*x - 1|

def M : Set ℝ := {x | f x > -1}

theorem problem_solution :
  (M = {x : ℝ | 0 < x ∧ x < 2}) ∧
  (∀ a ∈ M,
    (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
    (a = 1 → a^2 - a + 1 = 1/a) ∧
    (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a)) :=
by sorry

end problem_solution_l1039_103925


namespace probability_of_specific_combination_l1039_103968

def total_marbles : ℕ := 12 + 8 + 5

def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def green_marbles : ℕ := 5

def marbles_drawn : ℕ := 4

def ways_to_draw_specific_combination : ℕ := (red_marbles.choose 2) * blue_marbles * green_marbles

def total_ways_to_draw : ℕ := total_marbles.choose marbles_drawn

theorem probability_of_specific_combination :
  (ways_to_draw_specific_combination : ℚ) / total_ways_to_draw = 264 / 1265 := by sorry

end probability_of_specific_combination_l1039_103968


namespace line_passes_through_point_l1039_103957

/-- Proves that k = 167/3 given that the line -1/3 - 3kx = 7y passes through the point (1/3, -8) -/
theorem line_passes_through_point (k : ℚ) : 
  (-1/3 : ℚ) - 3 * k * (1/3 : ℚ) = 7 * (-8 : ℚ) → k = 167/3 := by
  sorry

end line_passes_through_point_l1039_103957


namespace b_over_a_squared_is_seven_l1039_103936

theorem b_over_a_squared_is_seven (a : ℕ) (k : ℕ) (b : ℕ) :
  a > 1 →
  b = a * (10^k + 1) →
  k > 0 →
  a < 10^k →
  a^2 ∣ b →
  b / a^2 = 7 := by
sorry

end b_over_a_squared_is_seven_l1039_103936


namespace solution_set_of_inequality_l1039_103927

theorem solution_set_of_inequality (x : ℝ) :
  Set.Icc (-1/2 : ℝ) 3 \ {3} = {x | (2*x + 1) / (3 - x) ≥ 0 ∧ x ≠ 3} :=
sorry

end solution_set_of_inequality_l1039_103927


namespace perfect_square_condition_l1039_103997

theorem perfect_square_condition (n : ℤ) : 
  ∃ (k : ℤ), n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = k^2 ↔ n = 10 := by
  sorry

end perfect_square_condition_l1039_103997


namespace three_digit_divisible_by_11_l1039_103975

theorem three_digit_divisible_by_11 (x y z : ℕ) (A : ℕ) : 
  (100 ≤ A) ∧ (A < 1000) ∧ 
  (A = 100 * x + 10 * y + z) ∧ 
  (x + z = y) → 
  ∃ k : ℕ, A = 11 * k := by
sorry

end three_digit_divisible_by_11_l1039_103975


namespace choose_four_from_thirty_l1039_103920

theorem choose_four_from_thirty : Nat.choose 30 4 = 27405 := by
  sorry

end choose_four_from_thirty_l1039_103920


namespace line_through_point_parallel_to_polar_axis_l1039_103901

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on a line in polar coordinates -/
def pointOnLine (p : PolarPoint) (l : PolarLine) : Prop :=
  l.equation p.ρ p.θ

/-- Checks if a line is parallel to the polar axis -/
def parallelToPolarAxis (l : PolarLine) : Prop :=
  ∀ ρ θ, l.equation ρ θ ↔ ∃ k, ρ * Real.sin θ = k

theorem line_through_point_parallel_to_polar_axis 
  (p : PolarPoint) 
  (h_p : p.ρ = 2 ∧ p.θ = Real.pi / 6) :
  ∃ l : PolarLine, 
    pointOnLine p l ∧ 
    parallelToPolarAxis l ∧
    (∀ ρ θ, l.equation ρ θ ↔ ρ * Real.sin θ = 1) := by
  sorry

end line_through_point_parallel_to_polar_axis_l1039_103901


namespace pa_distance_bounds_l1039_103970

/-- Given a segment AB of length 2 and a point P satisfying |PA| + |PB| = 8,
    prove that the distance |PA| is bounded by 3 ≤ |PA| ≤ 5. -/
theorem pa_distance_bounds (A B P : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A B = 2)
  (h2 : dist P A + dist P B = 8) :
  3 ≤ dist P A ∧ dist P A ≤ 5 := by
  sorry

end pa_distance_bounds_l1039_103970


namespace balls_after_1729_steps_l1039_103978

/-- Represents the state of boxes in Lisa's ball-placing game -/
def BoxState := List Nat

/-- Converts a natural number to its septenary (base-7) representation -/
def toSeptenary (n : Nat) : List Nat :=
  sorry

/-- Calculates the sum of a list of natural numbers -/
def sum (l : List Nat) : Nat :=
  sorry

/-- Simulates Lisa's ball-placing process for a given number of steps -/
def simulateSteps (steps : Nat) : BoxState :=
  sorry

/-- Counts the total number of balls in a given box state -/
def countBalls (state : BoxState) : Nat :=
  sorry

/-- Theorem stating that the number of balls after 1729 steps
    is equal to the sum of digits in the septenary representation of 1729 -/
theorem balls_after_1729_steps :
  countBalls (simulateSteps 1729) = sum (toSeptenary 1729) :=
sorry

end balls_after_1729_steps_l1039_103978


namespace cone_base_circumference_l1039_103965

theorem cone_base_circumference (r : ℝ) (angle : ℝ) (h1 : r = 6) (h2 : angle = 120) :
  let original_circumference := 2 * π * r
  let sector_fraction := angle / 360
  let base_circumference := (1 - sector_fraction) * original_circumference
  base_circumference = 4 * π :=
by sorry

end cone_base_circumference_l1039_103965


namespace range_of_m_l1039_103912

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B (m : ℝ) : Set ℝ := {x | x > m}

-- State the theorem
theorem range_of_m (m : ℝ) : (Set.compl A) ⊆ B m → m < 1 := by
  sorry

end range_of_m_l1039_103912


namespace balloon_arrangements_l1039_103993

def balloon_permutations : ℕ := 1260

theorem balloon_arrangements :
  (7 * 6 * 5 * 4 * 3) / 2 = balloon_permutations := by
  sorry

end balloon_arrangements_l1039_103993


namespace root_transformation_l1039_103959

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 8 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 8 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 8 = 0) →
  ((3*r₁)^3 - 9*(3*r₁)^2 + 216 = 0) ∧
  ((3*r₂)^3 - 9*(3*r₂)^2 + 216 = 0) ∧
  ((3*r₃)^3 - 9*(3*r₃)^2 + 216 = 0) :=
by sorry

end root_transformation_l1039_103959


namespace existence_of_m_n_l1039_103953

theorem existence_of_m_n (d : ℤ) : ∃ m n : ℤ, d * (m^2 - n) = n - 2*m + 1 := by
  sorry

end existence_of_m_n_l1039_103953


namespace umbrella_cost_l1039_103985

theorem umbrella_cost (house_umbrellas car_umbrellas total_cost : ℕ) 
  (h1 : house_umbrellas = 2)
  (h2 : car_umbrellas = 1)
  (h3 : total_cost = 24) :
  total_cost / (house_umbrellas + car_umbrellas) = 8 := by
  sorry

end umbrella_cost_l1039_103985
