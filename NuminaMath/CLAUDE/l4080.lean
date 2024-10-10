import Mathlib

namespace hyperbola_equation_l4080_408043

/-- A hyperbola with center at the origin, one focus at (-√5, 0), and a point P such that
    the midpoint of PF₁ is at (0, 2) has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation (P : ℝ × ℝ) : 
  (∃ (x y : ℝ), P = (x, y) ∧ x^2 - y^2/4 = 1) ↔ 
  (∃ (x y : ℝ), P = (x, y) ∧ 
    -- P is on the hyperbola
    (x - (-Real.sqrt 5))^2 + y^2 = (x - Real.sqrt 5)^2 + y^2 ∧ 
    -- Midpoint of PF₁ is (0, 2)
    ((x + (-Real.sqrt 5))/2 = 0 ∧ (y + 0)/2 = 2)) :=
by sorry


end hyperbola_equation_l4080_408043


namespace three_lines_exist_l4080_408003

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : Option ℝ
  intercept : ℝ

/-- The hyperbola x^2 - y^2 = 2 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- A line passes through the point (√2, 0) -/
def passesThrough (l : Line) : Prop :=
  match l.slope with
  | none => l.intercept = Real.sqrt 2
  | some m => l.intercept = -m * Real.sqrt 2

/-- A line has exactly one common point with the hyperbola -/
def hasOneCommonPoint (l : Line) : Prop :=
  ∃! p : ℝ × ℝ, (match l.slope with
    | none => p.1 = l.intercept ∧ p.2 = 0
    | some m => p.2 = m * p.1 + l.intercept) ∧ hyperbola p.1 p.2

/-- The main theorem: there are exactly 3 lines satisfying the conditions -/
theorem three_lines_exist :
  ∃! (lines : Finset Line), lines.card = 3 ∧
    ∀ l ∈ lines, passesThrough l ∧ hasOneCommonPoint l :=
sorry

end three_lines_exist_l4080_408003


namespace wayne_shrimp_cocktail_l4080_408086

/-- Calculates the number of shrimp served per guest given the total spent, cost per pound, shrimp per pound, and number of guests. -/
def shrimp_per_guest (total_spent : ℚ) (cost_per_pound : ℚ) (shrimp_per_pound : ℕ) (num_guests : ℕ) : ℚ :=
  (total_spent / cost_per_pound * shrimp_per_pound) / num_guests

/-- Proves that Wayne plans to serve 5 shrimp per guest given the problem conditions. -/
theorem wayne_shrimp_cocktail :
  let total_spent : ℚ := 170
  let cost_per_pound : ℚ := 17
  let shrimp_per_pound : ℕ := 20
  let num_guests : ℕ := 40
  shrimp_per_guest total_spent cost_per_pound shrimp_per_pound num_guests = 5 := by
  sorry

end wayne_shrimp_cocktail_l4080_408086


namespace balloon_distribution_l4080_408020

/-- Given a total number of balloons and the ratios between different colors,
    calculate the number of balloons for each color. -/
theorem balloon_distribution (total : ℕ) (red_ratio blue_ratio black_ratio : ℕ) 
    (h_total : total = 180)
    (h_red : red_ratio = 3)
    (h_black : black_ratio = 2)
    (h_blue : blue_ratio = 1) :
    ∃ (red blue black : ℕ),
      red = 90 ∧ blue = 30 ∧ black = 60 ∧
      red = red_ratio * blue ∧
      black = black_ratio * blue ∧
      red + blue + black = total :=
by
  sorry

#check balloon_distribution

end balloon_distribution_l4080_408020


namespace train_length_l4080_408007

/-- The length of a train given its speed, a man's walking speed, and the time taken to pass the man. -/
theorem train_length (train_speed : Real) (man_speed : Real) (passing_time : Real) :
  train_speed = 63 →
  man_speed = 3 →
  passing_time = 44.99640028797696 →
  (train_speed - man_speed) * passing_time * (1000 / 3600) = 750 := by
  sorry

end train_length_l4080_408007


namespace tim_blue_marbles_l4080_408047

theorem tim_blue_marbles (fred_marbles : ℕ) (fred_tim_ratio : ℕ) (h1 : fred_marbles = 385) (h2 : fred_tim_ratio = 35) :
  fred_marbles / fred_tim_ratio = 11 := by
  sorry

end tim_blue_marbles_l4080_408047


namespace unique_solution_cubic_equation_l4080_408074

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x ≠ -1 ∧ (x^3 - x^2) / (x^2 + 2*x + 1) + 2*x = -4 :=
by
  use 4/3
  sorry

end unique_solution_cubic_equation_l4080_408074


namespace seating_arrangements_l4080_408095

-- Define the number of seats, adults, and children
def numSeats : ℕ := 6
def numAdults : ℕ := 3
def numChildren : ℕ := 3

-- Define a function to calculate permutations
def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

-- Theorem statement
theorem seating_arrangements :
  2 * (permutations numAdults numAdults) * (permutations numChildren numChildren) = 72 :=
by sorry

end seating_arrangements_l4080_408095


namespace estimate_fish_population_l4080_408024

/-- Estimates the total number of fish in a lake using the mark and recapture method. -/
theorem estimate_fish_population (m n k : ℕ) (h : k > 0) :
  let estimated_total := m * n / k
  ∃ x : ℚ, x = estimated_total ∧ x > 0 := by
  sorry

end estimate_fish_population_l4080_408024


namespace chessboard_tiling_l4080_408091

/-- Represents a chessboard configuration -/
inductive ChessboardConfig
  | OneCornerRemoved
  | TwoOppositeCorners
  | TwoNonOppositeCorners

/-- Represents whether a configuration is tileable or not -/
inductive Tileable
  | Yes
  | No

/-- Function to determine if a chessboard configuration is tileable with 2x1 dominoes -/
def isTileable (config : ChessboardConfig) : Tileable :=
  match config with
  | ChessboardConfig.OneCornerRemoved => Tileable.No
  | ChessboardConfig.TwoOppositeCorners => Tileable.No
  | ChessboardConfig.TwoNonOppositeCorners => Tileable.Yes

theorem chessboard_tiling :
  (isTileable ChessboardConfig.OneCornerRemoved = Tileable.No) ∧
  (isTileable ChessboardConfig.TwoOppositeCorners = Tileable.No) ∧
  (isTileable ChessboardConfig.TwoNonOppositeCorners = Tileable.Yes) := by
  sorry

end chessboard_tiling_l4080_408091


namespace original_price_calculation_l4080_408058

/-- 
Given an article sold at a 40% profit, where the profit amount is 700 (in some currency unit),
prove that the original price of the article is 1750 (in the same currency unit).
-/
theorem original_price_calculation (profit_percentage : ℝ) (profit_amount : ℝ) 
  (h1 : profit_percentage = 40) 
  (h2 : profit_amount = 700) : 
  ∃ (original_price : ℝ), 
    original_price * (1 + profit_percentage / 100) - original_price = profit_amount ∧ 
    original_price = 1750 := by
  sorry

end original_price_calculation_l4080_408058


namespace min_red_chips_l4080_408016

theorem min_red_chips (r w b : ℕ) : 
  b ≥ (w : ℚ) / 3 →
  (b : ℚ) ≤ r / 4 →
  w + b ≥ 70 →
  r ≥ 72 ∧ ∀ (r' : ℕ), (∃ (w' b' : ℕ), 
    b' ≥ (w' : ℚ) / 3 ∧
    (b' : ℚ) ≤ r' / 4 ∧
    w' + b' ≥ 70) → 
  r' ≥ 72 := by
sorry

end min_red_chips_l4080_408016


namespace given_program_has_syntax_error_l4080_408090

/-- Represents the structure of a DO-UNTIL loop -/
inductive DOUntilLoop
| correct : (body : String) → (condition : String) → DOUntilLoop
| incorrect : (body : String) → (untilKeyword : String) → (condition : String) → DOUntilLoop

/-- The given program structure -/
def givenProgram : DOUntilLoop :=
  DOUntilLoop.incorrect "x=x*x" "UNTIL" "x>10"

/-- Checks if a DO-UNTIL loop has correct syntax -/
def hasCorrectSyntax (loop : DOUntilLoop) : Prop :=
  match loop with
  | DOUntilLoop.correct _ _ => True
  | DOUntilLoop.incorrect _ _ _ => False

/-- Theorem stating that the given program has a syntax error -/
theorem given_program_has_syntax_error :
  ¬(hasCorrectSyntax givenProgram) := by
  sorry


end given_program_has_syntax_error_l4080_408090


namespace sin_fifth_power_coefficients_sum_of_squares_l4080_408005

theorem sin_fifth_power_coefficients_sum_of_squares :
  ∃ (b₁ b₂ b₃ b₄ b₅ : ℝ),
    (∀ θ : ℝ, (Real.sin θ)^5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 63 / 128 := by
  sorry

end sin_fifth_power_coefficients_sum_of_squares_l4080_408005


namespace line_hyperbola_intersection_l4080_408032

theorem line_hyperbola_intersection (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < -1 ∧ x₂ < -1 ∧
    (x₁^2 - (k*x₁ - 1)^2 = 1) ∧
    (x₂^2 - (k*x₂ - 1)^2 = 1)) ↔
  -Real.sqrt 2 < k ∧ k < -1 := by
sorry

end line_hyperbola_intersection_l4080_408032


namespace unchanged_100th_is_100_l4080_408021

def is_valid_sequence (s : List ℕ) : Prop :=
  s.length = 1982 ∧ s.toFinset = Finset.range 1983 \ {0}

def swap_adjacent (s : List ℕ) : List ℕ :=
  s.zipWith (λ a b => if a > b then b else a) (s.tail.append [0])

def left_to_right_pass (s : List ℕ) : List ℕ :=
  (s.length - 1).fold (λ _ s' => swap_adjacent s') s

def right_to_left_pass (s : List ℕ) : List ℕ :=
  (left_to_right_pass s.reverse).reverse

def double_pass (s : List ℕ) : List ℕ :=
  right_to_left_pass (left_to_right_pass s)

theorem unchanged_100th_is_100 (s : List ℕ) :
  is_valid_sequence s →
  (double_pass s).nthLe 99 (by sorry) = s.nthLe 99 (by sorry) →
  s.nthLe 99 (by sorry) = 100 := by sorry

end unchanged_100th_is_100_l4080_408021


namespace arithmetic_mean_problem_l4080_408017

theorem arithmetic_mean_problem : 
  let sequence1 := List.range 15 |> List.map (λ x => x - 6)
  let sequence2 := List.range 10 |> List.map (λ x => x + 1)
  let combined_sequence := sequence1 ++ sequence2
  let sum := combined_sequence.sum
  let count := combined_sequence.length
  (sum : ℚ) / count = 35 / 13 := by
  sorry

end arithmetic_mean_problem_l4080_408017


namespace oatmeal_raisin_percentage_l4080_408042

/-- Given a class of students and cookie distribution, calculate the percentage of students who want oatmeal raisin cookies. -/
theorem oatmeal_raisin_percentage 
  (total_students : ℕ) 
  (cookies_per_student : ℕ) 
  (oatmeal_raisin_cookies : ℕ) 
  (h1 : total_students = 40)
  (h2 : cookies_per_student = 2)
  (h3 : oatmeal_raisin_cookies = 8) : 
  (oatmeal_raisin_cookies / cookies_per_student) / total_students * 100 = 10 := by
  sorry

end oatmeal_raisin_percentage_l4080_408042


namespace parabola_axis_l4080_408002

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := x^2 = -8*y

/-- The equation of the axis of a parabola -/
def axis_equation (y : ℝ) : Prop := y = 2

/-- Theorem: The axis of the parabola x^2 = -8y is y = 2 -/
theorem parabola_axis :
  (∀ x y : ℝ, parabola_equation x y) →
  (∀ y : ℝ, axis_equation y) :=
sorry

end parabola_axis_l4080_408002


namespace scatter_plot_placement_l4080_408025

/-- Represents a variable in a scatter plot -/
inductive Variable
  | Explanatory
  | Forecast

/-- Represents an axis in a scatter plot -/
inductive Axis
  | X
  | Y

/-- Represents the correct placement of variables on axes in a scatter plot -/
def correct_placement (v : Variable) (a : Axis) : Prop :=
  match v, a with
  | Variable.Explanatory, Axis.X => True
  | Variable.Forecast, Axis.Y => True
  | _, _ => False

/-- Theorem stating the correct placement of variables in a scatter plot -/
theorem scatter_plot_placement :
  ∀ (v : Variable) (a : Axis),
    correct_placement v a ↔
      ((v = Variable.Explanatory ∧ a = Axis.X) ∨
       (v = Variable.Forecast ∧ a = Axis.Y)) :=
by sorry

end scatter_plot_placement_l4080_408025


namespace chess_points_theorem_l4080_408031

theorem chess_points_theorem :
  ∃! (s : Finset ℕ), s.card = 2 ∧
  (∀ x ∈ s, ∃ n : ℕ, 11 * n + x * (100 - n) = 800) ∧
  (3 ∈ s ∧ 4 ∈ s) := by
  sorry

end chess_points_theorem_l4080_408031


namespace cube_root_of_4x_plus_3y_is_3_l4080_408078

theorem cube_root_of_4x_plus_3y_is_3 (x y : ℝ) : 
  y = Real.sqrt (3 - x) + Real.sqrt (x - 3) + 5 → 
  (4 * x + 3 * y) ^ (1/3 : ℝ) = 3 := by
sorry

end cube_root_of_4x_plus_3y_is_3_l4080_408078


namespace sum_x_y_equals_negative_one_l4080_408052

theorem sum_x_y_equals_negative_one (x y : ℝ) 
  (eq1 : 3 * x - 4 * y = 18) 
  (eq2 : 5 * x + 3 * y = 1) : 
  x + y = -1 := by
  sorry

end sum_x_y_equals_negative_one_l4080_408052


namespace min_fraction_value_l4080_408084

/-- A function that checks if a natural number contains the digit string "11235" -/
def contains_11235 (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem min_fraction_value (N k : ℕ) (h1 : N > 0) (h2 : k > 0) (h3 : contains_11235 N) (h4 : 10^k > N) :
  (∀ N' k' : ℕ, N' > 0 → k' > 0 → contains_11235 N' → 10^k' > N' →
    (10^k' - 1) / Nat.gcd (10^k' - 1) N' ≥ 89) ∧
  (∃ N' k' : ℕ, N' > 0 ∧ k' > 0 ∧ contains_11235 N' ∧ 10^k' > N' ∧
    (10^k' - 1) / Nat.gcd (10^k' - 1) N' = 89) :=
by sorry

end min_fraction_value_l4080_408084


namespace rebus_unique_solution_l4080_408019

/-- Represents a four-digit number ABCD where A, B, C, D are distinct non-zero digits. -/
structure Rebus where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0
  h_digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

/-- The rebus equation ABCA = 182 * CD holds. -/
def rebusEquation (r : Rebus) : Prop :=
  1000 * r.a + 100 * r.b + 10 * r.c + r.a = 182 * (10 * r.c + r.d)

/-- The unique solution to the rebus is 2916. -/
theorem rebus_unique_solution :
  ∃! r : Rebus, rebusEquation r ∧ r.a = 2 ∧ r.b = 9 ∧ r.c = 1 ∧ r.d = 6 := by
  sorry

end rebus_unique_solution_l4080_408019


namespace distance_traveled_l4080_408001

/-- 
Given a skater's speed and time spent skating, calculate the total distance traveled.
-/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 10) (h2 : time = 8) :
  speed * time = 80 := by
  sorry

end distance_traveled_l4080_408001


namespace same_color_probability_l4080_408067

/-- The probability of drawing two balls of the same color from an urn -/
theorem same_color_probability (w b r : ℕ) (hw : w = 4) (hb : b = 6) (hr : r = 5) :
  let total := w + b + r
  let p_white := (w / total) * ((w - 1) / (total - 1))
  let p_black := (b / total) * ((b - 1) / (total - 1))
  let p_red := (r / total) * ((r - 1) / (total - 1))
  p_white + p_black + p_red = 31 / 105 := by
  sorry

#check same_color_probability

end same_color_probability_l4080_408067


namespace sin_510_degrees_l4080_408041

theorem sin_510_degrees : Real.sin (510 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_510_degrees_l4080_408041


namespace f_max_at_two_l4080_408069

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

-- State the theorem
theorem f_max_at_two :
  ∃ (max : ℝ), f 2 = max ∧ ∀ x, f x ≤ max :=
by
  sorry

end f_max_at_two_l4080_408069


namespace initial_cookies_l4080_408026

/-- Given that 2 cookies were eaten and 5 cookies remain, prove that the initial number of cookies was 7. -/
theorem initial_cookies (eaten : ℕ) (remaining : ℕ) (h1 : eaten = 2) (h2 : remaining = 5) :
  eaten + remaining = 7 := by
  sorry

end initial_cookies_l4080_408026


namespace percentage_problem_l4080_408099

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 640 = 0.20 * 650 + 190 → P = 50 := by
sorry

end percentage_problem_l4080_408099


namespace first_shipment_size_l4080_408023

/-- The size of the first shipment of tomatoes in kg -/
def first_shipment : ℕ := sorry

/-- The amount of tomatoes sold on Saturday in kg -/
def sold_saturday : ℕ := 300

/-- The amount of tomatoes that rotted on Sunday in kg -/
def rotted_sunday : ℕ := 200

/-- The total amount of tomatoes available on Tuesday in kg -/
def total_tuesday : ℕ := 2500

theorem first_shipment_size :
  first_shipment - sold_saturday - rotted_sunday + 2 * first_shipment = total_tuesday ∧
  first_shipment = 1000 := by sorry

end first_shipment_size_l4080_408023


namespace quadratic_equation_solution_l4080_408081

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 5 * x - 12 = 0 ↔ x = 3 ∨ x = -4/3) → k = 3 :=
by sorry

end quadratic_equation_solution_l4080_408081


namespace pair_probability_after_removal_l4080_408077

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset ℕ)
  (count : ℕ → ℕ)
  (total : ℕ)

/-- Initial deck configuration -/
def initial_deck : Deck :=
  { cards := Finset.range 12,
    count := λ n => if n ∈ Finset.range 12 then 4 else 0,
    total := 48 }

/-- Deck after removing two pairs -/
def deck_after_removal (d : Deck) : Deck :=
  { cards := d.cards,
    count := λ n => if n ∈ d.cards then d.count n - 2 else 0,
    total := d.total - 4 }

/-- Number of ways to choose 2 cards from n cards -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Number of ways to form pairs from remaining cards -/
def pair_ways (d : Deck) : ℕ :=
  (d.cards.filter (λ n => d.count n = 4)).card * 6 +
  (d.cards.filter (λ n => d.count n = 2)).card * 1

/-- Probability of selecting a pair -/
def pair_probability (d : Deck) : ℚ :=
  pair_ways d / choose_two d.total

theorem pair_probability_after_removal :
  pair_probability (deck_after_removal initial_deck) = 31 / 473 := by
  sorry

#eval pair_probability (deck_after_removal initial_deck)

end pair_probability_after_removal_l4080_408077


namespace solution_set_R_solution_set_m_lower_bound_l4080_408056

-- Define the inequality
def inequality (x m : ℝ) : Prop := x^2 - 2*(m+1)*x + 4*m ≥ 0

-- Statement 1
theorem solution_set_R (m : ℝ) : 
  (∀ x, inequality x m) ↔ m = 1 := by sorry

-- Statement 2
theorem solution_set (m : ℝ) :
  (m = 1 ∧ ∀ x, inequality x m) ∨
  (m > 1 ∧ ∀ x, inequality x m ↔ (x ≤ 2 ∨ x ≥ 2*m)) ∨
  (m < 1 ∧ ∀ x, inequality x m ↔ (x ≤ 2*m ∨ x ≥ 2)) := by sorry

-- Statement 3
theorem m_lower_bound :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → inequality x m) → m ≥ 1/2 := by sorry

end solution_set_R_solution_set_m_lower_bound_l4080_408056


namespace quadrilateral_sum_l4080_408013

/-- Given a quadrilateral PQRS with vertices P(a, b), Q(a, -b), R(-a, -b), and S(-a, b),
    where a and b are positive integers and a > b, if the area of PQRS is 32,
    then a + b = 5. -/
theorem quadrilateral_sum (a b : ℕ) (ha : a > b) (hb : b > 0)
  (harea : (2 * a) * (2 * b) = 32) : a + b = 5 := by
  sorry

#check quadrilateral_sum

end quadrilateral_sum_l4080_408013


namespace probability_properties_l4080_408066

theorem probability_properties (A B : Set Ω) (P : Set Ω → ℝ) 
  (hA : P A = 0.5) (hB : P B = 0.3) :
  (∀ h : A ∩ B = ∅, P (A ∪ B) = 0.8) ∧ 
  (∀ h : P (A ∩ B) = P A * P B, P (A ∪ B) = 0.65) ∧
  (∀ h : P (B ∩ A) / P A = 0.5, P (B ∩ Aᶜ) / P Aᶜ = 0.1) := by
  sorry

end probability_properties_l4080_408066


namespace remaining_income_percentage_l4080_408057

theorem remaining_income_percentage (total_income : ℝ) (food_percentage : ℝ) (education_percentage : ℝ) (rent_percentage : ℝ) :
  food_percentage = 35 →
  education_percentage = 25 →
  rent_percentage = 80 →
  total_income > 0 →
  let remaining_after_food_education := total_income * (1 - (food_percentage + education_percentage) / 100)
  let remaining_after_rent := remaining_after_food_education * (1 - rent_percentage / 100)
  remaining_after_rent / total_income = 0.08 := by
  sorry

end remaining_income_percentage_l4080_408057


namespace kira_morning_downloads_l4080_408059

/-- The number of songs Kira downloaded in the morning -/
def morning_songs : ℕ := sorry

/-- The number of songs Kira downloaded later in the day -/
def afternoon_songs : ℕ := 15

/-- The number of songs Kira downloaded at night -/
def night_songs : ℕ := 3

/-- The size of each song in MB -/
def song_size : ℕ := 5

/-- The total memory space occupied by all songs in MB -/
def total_memory : ℕ := 140

theorem kira_morning_downloads : 
  morning_songs = 10 ∧ 
  song_size * (morning_songs + afternoon_songs + night_songs) = total_memory := by
  sorry

end kira_morning_downloads_l4080_408059


namespace only_one_divides_power_minus_one_l4080_408088

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1 ↔ n = 1) := by sorry

end only_one_divides_power_minus_one_l4080_408088


namespace sum_and_opposites_l4080_408070

theorem sum_and_opposites : 
  let a := -5
  let b := -2
  let c := abs b
  let d := 0
  (a + b + c + d = -5) ∧ 
  (- a = 5) ∧ 
  (- b = 2) ∧ 
  (- c = -2) ∧ 
  (- d = 0) := by
  sorry

end sum_and_opposites_l4080_408070


namespace cantaloupes_total_l4080_408018

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 38

/-- The number of cantaloupes grown by Tim -/
def tim_cantaloupes : ℕ := 44

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := fred_cantaloupes + tim_cantaloupes

theorem cantaloupes_total : total_cantaloupes = 82 := by
  sorry

end cantaloupes_total_l4080_408018


namespace lathe_probabilities_l4080_408049

-- Define the yield rates and processing percentages
def yield_rate_1 : ℝ := 0.15
def yield_rate_2 : ℝ := 0.10
def process_percent_1 : ℝ := 0.60
def process_percent_2 : ℝ := 0.40

-- Define the theorem
theorem lathe_probabilities :
  -- Probability of both lathes producing excellent parts simultaneously
  yield_rate_1 * yield_rate_2 = 0.015 ∧
  -- Probability of randomly selecting an excellent part from mixed parts
  process_percent_1 * yield_rate_1 + process_percent_2 * yield_rate_2 = 0.13 :=
by sorry

end lathe_probabilities_l4080_408049


namespace excellent_students_probability_l4080_408037

/-- The probability of selecting exactly 4 excellent students when randomly choosing 7 students from a class of 10 students, where 6 are excellent, is equal to 0.5. -/
theorem excellent_students_probability :
  let total_students : ℕ := 10
  let excellent_students : ℕ := 6
  let selected_students : ℕ := 7
  let target_excellent : ℕ := 4
  (Nat.choose excellent_students target_excellent * Nat.choose (total_students - excellent_students) (selected_students - target_excellent)) / Nat.choose total_students selected_students = 1 / 2 :=
by sorry

end excellent_students_probability_l4080_408037


namespace dot_product_zero_l4080_408060

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![4, 3]

theorem dot_product_zero : 
  (Finset.sum Finset.univ (λ i => a i * (2 * a i - b i))) = 0 := by sorry

end dot_product_zero_l4080_408060


namespace professor_chair_selections_eq_24_l4080_408087

/-- Represents the number of chairs in a row -/
def total_chairs : ℕ := 11

/-- Represents the number of professors -/
def num_professors : ℕ := 3

/-- Represents the minimum number of chairs between professors -/
def min_separation : ℕ := 2

/-- Calculates the number of ways to select chairs for professors -/
def professor_chair_selections : ℕ := sorry

/-- Theorem stating that the number of ways to select chairs for professors is 24 -/
theorem professor_chair_selections_eq_24 :
  professor_chair_selections = 24 := by sorry

end professor_chair_selections_eq_24_l4080_408087


namespace line_intersection_with_axes_l4080_408027

/-- A line passing through two given points intersects the x-axis and y-axis at specific points -/
theorem line_intersection_with_axes (x₁ y₁ x₂ y₂ : ℝ) :
  let m : ℝ := (y₂ - y₁) / (x₂ - x₁)
  let b : ℝ := y₁ - m * x₁
  let line : ℝ → ℝ := λ x => m * x + b
  x₁ = 8 ∧ y₁ = 2 ∧ x₂ = 4 ∧ y₂ = 6 →
  (∃ x : ℝ, line x = 0 ∧ x = 10) ∧
  (∃ y : ℝ, line 0 = y ∧ y = 10) :=
by sorry

#check line_intersection_with_axes

end line_intersection_with_axes_l4080_408027


namespace f_composition_minus_one_l4080_408065

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_composition_minus_one : f (f (-1)) = 5 := by sorry

end f_composition_minus_one_l4080_408065


namespace characterization_of_brazilian_triples_l4080_408006

def is_brazilian (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∃ k : ℕ, b * c + 1 = k * a) ∧
  (∃ k : ℕ, a * c + 1 = k * b) ∧
  (∃ k : ℕ, a * b + 1 = k * c)

def brazilian_triples : Set (ℕ × ℕ × ℕ) :=
  {(3, 2, 1), (2, 3, 1), (1, 3, 2), (2, 1, 3), (1, 2, 3), (3, 1, 2),
   (7, 3, 2), (3, 7, 2), (2, 7, 3), (3, 2, 7), (2, 3, 7), (7, 2, 3),
   (2, 1, 1), (1, 2, 1), (1, 1, 2),
   (1, 1, 1)}

theorem characterization_of_brazilian_triples :
  ∀ a b c : ℕ, is_brazilian a b c ↔ (a, b, c) ∈ brazilian_triples :=
sorry

end characterization_of_brazilian_triples_l4080_408006


namespace sun_radius_scientific_notation_l4080_408000

theorem sun_radius_scientific_notation :
  696000 = 6.96 * (10 ^ 5) := by sorry

end sun_radius_scientific_notation_l4080_408000


namespace yarn_length_problem_l4080_408011

theorem yarn_length_problem (green_length red_length total_length : ℕ) : 
  green_length = 156 →
  red_length = 3 * green_length + 8 →
  total_length = green_length + red_length →
  total_length = 632 := by
  sorry

end yarn_length_problem_l4080_408011


namespace marble_selection_ways_l4080_408050

theorem marble_selection_ways (total_marbles : ℕ) (selected_marbles : ℕ) (blue_marble : ℕ) :
  total_marbles = 10 →
  selected_marbles = 4 →
  blue_marble = 1 →
  (total_marbles.choose (selected_marbles - blue_marble)) = 84 :=
by sorry

end marble_selection_ways_l4080_408050


namespace eighth_root_of_549755289601_l4080_408046

theorem eighth_root_of_549755289601 :
  let n : ℕ := 549755289601
  (n = 1 * 100^8 + 8 * 100^7 + 28 * 100^6 + 56 * 100^5 + 70 * 100^4 + 
       56 * 100^3 + 28 * 100^2 + 8 * 100 + 1) →
  (n : ℝ)^(1/8 : ℝ) = 101 := by
sorry

end eighth_root_of_549755289601_l4080_408046


namespace rhombus_perimeter_l4080_408051

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 48) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 100 := by
  sorry

end rhombus_perimeter_l4080_408051


namespace C₁_is_unit_circle_intersection_point_C₁_k4_equation_l4080_408045

-- Define the curves C₁ and C₂
def C₁ (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = Real.cos t ^ k ∧ p.2 = Real.sin t ^ k}

def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 - 16 * p.2 + 3 = 0}

-- Part 1: Prove that C₁ when k = 1 is a unit circle
theorem C₁_is_unit_circle :
  C₁ 1 = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} := by sorry

-- Part 2: Prove that (1/4, 1/4) is an intersection point of C₁ and C₂ when k = 4
theorem intersection_point :
  (1/4, 1/4) ∈ C₁ 4 ∧ (1/4, 1/4) ∈ C₂ := by sorry

-- Helper theorem: The equation of C₁ when k = 4 can be written as √x + √y = 1
theorem C₁_k4_equation (p : ℝ × ℝ) :
  p ∈ C₁ 4 ↔ Real.sqrt p.1 + Real.sqrt p.2 = 1 := by sorry

end C₁_is_unit_circle_intersection_point_C₁_k4_equation_l4080_408045


namespace f_properties_l4080_408092

/-- The function f(x) = x³ - 3mx + n --/
def f (m n x : ℝ) : ℝ := x^3 - 3*m*x + n

/-- Theorem stating the values of m and n, and the extrema in [0,3] --/
theorem f_properties (m n : ℝ) (hm : m > 0) 
  (hmax : ∃ x, ∀ y, f m n y ≤ f m n x)
  (hmin : ∃ x, ∀ y, f m n x ≤ f m n y)
  (hmax_val : ∃ x, f m n x = 6)
  (hmin_val : ∃ x, f m n x = 2) :
  m = 1 ∧ n = 4 ∧ 
  (∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, f 1 4 y ≤ f 1 4 x) ∧
  (∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, f 1 4 x ≤ f 1 4 y) ∧
  (∃ x ∈ Set.Icc 0 3, f 1 4 x = 2) ∧
  (∃ x ∈ Set.Icc 0 3, f 1 4 x = 22) := by
  sorry

end f_properties_l4080_408092


namespace flying_scotsman_norwich_difference_l4080_408040

/-- Proves that Flying Scotsman had 20 more carriages than Norwich -/
theorem flying_scotsman_norwich_difference :
  let euston : ℕ := 130
  let norwich : ℕ := 100
  let total : ℕ := 460
  let norfolk : ℕ := euston - 20
  let flying_scotsman : ℕ := total - (euston + norfolk + norwich)
  flying_scotsman - norwich = 20 := by
  sorry

end flying_scotsman_norwich_difference_l4080_408040


namespace min_value_of_function_l4080_408038

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  ∃ y_min : ℝ, y_min = 7 ∧ ∀ y : ℝ, y = 4*x + 1/(4*x - 5) → y ≥ y_min :=
by sorry

end min_value_of_function_l4080_408038


namespace trapezium_side_length_first_parallel_side_length_l4080_408008

theorem trapezium_side_length : ℝ → Prop :=
  fun x =>
    let area : ℝ := 247
    let other_side : ℝ := 18
    let height : ℝ := 13
    area = (1 / 2) * (x + other_side) * height →
    x = 20

/-- The length of the first parallel side of the trapezium is 20 cm. -/
theorem first_parallel_side_length : trapezium_side_length 20 := by
  sorry

end trapezium_side_length_first_parallel_side_length_l4080_408008


namespace opposite_signs_abs_sum_less_abs_diff_l4080_408071

theorem opposite_signs_abs_sum_less_abs_diff (a b : ℝ) (h : a * b < 0) :
  |a + b| < |a - b| := by sorry

end opposite_signs_abs_sum_less_abs_diff_l4080_408071


namespace min_questions_to_determine_order_l4080_408093

/-- Represents a question that reveals the relative order of 50 numbers -/
def Question := Fin 100 → Prop

/-- The set of all possible permutations of numbers from 1 to 100 -/
def Permutations := Fin 100 → Fin 100

/-- A function that determines if a given permutation is consistent with the answers to all questions -/
def IsConsistent (p : Permutations) (qs : List Question) : Prop := sorry

/-- The minimum number of questions needed to determine the order of 100 integers -/
def MinQuestions : ℕ := 5

theorem min_questions_to_determine_order :
  ∀ (qs : List Question),
    (∀ (p₁ p₂ : Permutations), IsConsistent p₁ qs ∧ IsConsistent p₂ qs → p₁ = p₂) →
    qs.length ≥ MinQuestions :=
sorry

end min_questions_to_determine_order_l4080_408093


namespace a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l4080_408053

theorem a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ∧ ¬(∀ a b : ℝ, a^2 > b^2 → a > b) :=
by sorry

end a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l4080_408053


namespace geometric_sequence_problem_l4080_408064

/-- Given a geometric sequence {a_n} where a₇ = 1/4 and a₃a₅ = 4(a₄ - 1), prove that a₂ = 8 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geometric : ∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m)
  (h_a7 : a 7 = 1 / 4)
  (h_a3a5 : a 3 * a 5 = 4 * (a 4 - 1)) :
  a 2 = 8 := by
  sorry

end geometric_sequence_problem_l4080_408064


namespace fair_rides_calculation_fair_rides_proof_l4080_408079

/-- Calculates the number of rides taken by each person at a fair given specific conditions. -/
theorem fair_rides_calculation (entrance_fee_under_18 : ℚ) (ride_cost : ℚ) 
  (total_spent : ℚ) (num_people : ℕ) : ℚ :=
  let entrance_fee_18_plus := entrance_fee_under_18 * (1 + 1/5)
  let total_entrance_fee := entrance_fee_18_plus + 2 * entrance_fee_under_18
  let rides_cost := total_spent - total_entrance_fee
  let total_rides := rides_cost / ride_cost
  total_rides / num_people

/-- Proves that under the given conditions, each person took 3 rides. -/
theorem fair_rides_proof :
  fair_rides_calculation 5 (1/2) (41/2) 3 = 3 := by
  sorry

end fair_rides_calculation_fair_rides_proof_l4080_408079


namespace max_happy_monkeys_theorem_l4080_408085

/-- Represents the number of each fruit type available --/
structure FruitCounts where
  pears : ℕ
  bananas : ℕ
  peaches : ℕ
  tangerines : ℕ

/-- Represents the criteria for a monkey to be happy --/
def happy_monkey (fruits : FruitCounts) : Prop :=
  ∃ (a b c : ℕ), a + b + c = 3 ∧ a + b + c ≤ fruits.pears + fruits.bananas + fruits.peaches + fruits.tangerines

/-- The maximum number of happy monkeys given the fruit counts --/
def max_happy_monkeys (fruits : FruitCounts) : ℕ :=
  Nat.min ((fruits.pears + fruits.bananas + fruits.peaches) / 2) fruits.tangerines

/-- Theorem stating the maximum number of happy monkeys for the given fruit counts --/
theorem max_happy_monkeys_theorem (fruits : FruitCounts) :
  fruits.pears = 20 →
  fruits.bananas = 30 →
  fruits.peaches = 40 →
  fruits.tangerines = 50 →
  max_happy_monkeys fruits = 45 :=
by
  sorry

#eval max_happy_monkeys ⟨20, 30, 40, 50⟩

end max_happy_monkeys_theorem_l4080_408085


namespace first_car_speed_l4080_408080

/-- 
Given two cars starting from opposite ends of a highway, this theorem proves
that the speed of the first car is 25 mph under the given conditions.
-/
theorem first_car_speed 
  (highway_length : ℝ) 
  (second_car_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : highway_length = 175) 
  (h2 : second_car_speed = 45) 
  (h3 : meeting_time = 2.5) :
  ∃ (first_car_speed : ℝ), 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    first_car_speed = 25 :=
by
  sorry

end first_car_speed_l4080_408080


namespace mn_solutions_l4080_408030

theorem mn_solutions (m n : ℤ) : 
  m * n ≥ 0 → m^3 + n^3 + 99*m*n = 33^3 → 
  ((m + n = 33 ∧ m ≥ 0 ∧ n ≥ 0) ∨ (m = -33 ∧ n = -33)) := by
  sorry

end mn_solutions_l4080_408030


namespace sum_of_two_integers_l4080_408094

theorem sum_of_two_integers (x y : ℤ) : x = 32 → y = 2 * x → x + y = 96 := by
  sorry

end sum_of_two_integers_l4080_408094


namespace feed_animals_count_l4080_408033

/-- Represents the number of pairs of animals in the zoo -/
def num_pairs : ℕ := 5

/-- Calculates the number of ways to feed all animals in the zoo -/
def feed_animals : ℕ :=
  (num_pairs) *  -- Choose from 5 females
  (num_pairs - 1) * (num_pairs - 1) *  -- Choose from 4 males, then 4 females
  (num_pairs - 2) * (num_pairs - 2) *  -- Choose from 3 males, then 3 females
  (num_pairs - 3) * (num_pairs - 3) *  -- Choose from 2 males, then 2 females
  (num_pairs - 4) * (num_pairs - 4)    -- Choose from 1 male, then 1 female

/-- Theorem stating the number of ways to feed all animals -/
theorem feed_animals_count : feed_animals = 2880 := by
  sorry

end feed_animals_count_l4080_408033


namespace simplify_expression_l4080_408036

theorem simplify_expression : (625 : ℝ) ^ (1/4 : ℝ) * (125 : ℝ) ^ (1/3 : ℝ) = 25 := by
  sorry

end simplify_expression_l4080_408036


namespace horner_method_correct_l4080_408076

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 0.3x + 2 -/
def f : ℝ → ℝ := fun x => x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

theorem horner_method_correct :
  let coeffs := [1, -5, 6, 0, 1, 0.3, 2]
  horner_eval coeffs (-2) = f (-2) ∧ f (-2) = 325.4 := by
  sorry

end horner_method_correct_l4080_408076


namespace bills_average_speed_day2_l4080_408015

/-- Represents the driving scenario of Bill's two-day journey --/
structure DrivingScenario where
  speed_day2 : ℝ  -- Average speed on the second day
  time_day2 : ℝ   -- Time spent driving on the second day
  total_distance : ℝ  -- Total distance driven over two days
  total_time : ℝ      -- Total time spent driving over two days

/-- Defines the conditions of Bill's journey --/
def journey_conditions (s : DrivingScenario) : Prop :=
  s.total_distance = 680 ∧
  s.total_time = 18 ∧
  s.total_distance = (s.speed_day2 + 5) * (s.time_day2 + 2) + s.speed_day2 * s.time_day2 ∧
  s.total_time = (s.time_day2 + 2) + s.time_day2

/-- Theorem stating that given the journey conditions, Bill's average speed on the second day was 35 mph --/
theorem bills_average_speed_day2 (s : DrivingScenario) : 
  journey_conditions s → s.speed_day2 = 35 := by
  sorry

end bills_average_speed_day2_l4080_408015


namespace chessboard_squares_l4080_408082

/-- The number of squares of a given size in an 8x8 chessboard -/
def squares_of_size (n : Nat) : Nat :=
  (9 - n) ^ 2

/-- The total number of squares in an 8x8 chessboard -/
def total_squares : Nat :=
  (Finset.range 8).sum squares_of_size

theorem chessboard_squares :
  total_squares = 204 := by
  sorry

end chessboard_squares_l4080_408082


namespace no_real_solutions_for_equation_l4080_408098

theorem no_real_solutions_for_equation : 
  ¬∃ y : ℝ, (10 - y)^2 = 4 * y^2 := by
  sorry

end no_real_solutions_for_equation_l4080_408098


namespace smallest_common_multiple_of_8_and_9_l4080_408039

theorem smallest_common_multiple_of_8_and_9 : ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 9 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 8 ∣ m ∧ 9 ∣ m) → n ≤ m :=
sorry

end smallest_common_multiple_of_8_and_9_l4080_408039


namespace initial_observations_count_l4080_408063

theorem initial_observations_count (n : ℕ) 
  (h1 : (n : ℝ) > 0)
  (h2 : ∃ S : ℝ, S / n = 11)
  (h3 : ∃ new_obs : ℝ, (S + new_obs) / (n + 1) = 10)
  (h4 : new_obs = 4) :
  n = 6 := by
sorry

end initial_observations_count_l4080_408063


namespace smallest_zucchini_count_l4080_408014

def is_divisible_by_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k * k * k

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_6 n ∧ is_perfect_square (n / 2) ∧ is_perfect_cube (n / 3)

theorem smallest_zucchini_count :
  satisfies_conditions 648 ∧ ∀ m : ℕ, m < 648 → ¬(satisfies_conditions m) :=
by sorry

end smallest_zucchini_count_l4080_408014


namespace value_of_a_l4080_408012

theorem value_of_a (a : ℝ) (h1 : a < 0) (h2 : |a| = 3) : a = -3 := by
  sorry

end value_of_a_l4080_408012


namespace smallest_integer_with_remainder_l4080_408054

theorem smallest_integer_with_remainder (n : ℕ) : n = 170 ↔ 
  (n > 1 ∧ 
   n % 3 = 2 ∧ 
   n % 7 = 2 ∧ 
   n % 8 = 2 ∧ 
   ∀ m : ℕ, m > 1 → m % 3 = 2 → m % 7 = 2 → m % 8 = 2 → n ≤ m) :=
by sorry

end smallest_integer_with_remainder_l4080_408054


namespace toothpick_grid_theorem_l4080_408022

/-- Represents a toothpick grid -/
structure ToothpickGrid where
  length : ℕ
  width : ℕ

/-- Calculates the total number of toothpicks in a grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  (grid.length + 1) * grid.width + (grid.width + 1) * grid.length

/-- Calculates the area enclosed by a grid -/
def enclosed_area (grid : ToothpickGrid) : ℕ :=
  grid.length * grid.width

theorem toothpick_grid_theorem (grid : ToothpickGrid) 
    (h1 : grid.length = 30) (h2 : grid.width = 50) : 
    total_toothpicks grid = 3080 ∧ enclosed_area grid = 1500 := by
  sorry

end toothpick_grid_theorem_l4080_408022


namespace f_max_value_l4080_408072

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := S n / ((n + 32) * S (n + 1))

theorem f_max_value :
  (∀ n : ℕ, n > 0 → f n ≤ 1/50) ∧ (∃ n : ℕ, n > 0 ∧ f n = 1/50) :=
sorry

end f_max_value_l4080_408072


namespace collection_for_44_members_l4080_408089

/-- Calculates the total collection amount in rupees for a group of students -/
def total_collection_rupees (num_members : ℕ) (paise_per_rupee : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / paise_per_rupee

/-- Proves that the total collection amount for 44 members is 19.36 rupees -/
theorem collection_for_44_members :
  total_collection_rupees 44 100 = 19.36 := by
  sorry

#eval total_collection_rupees 44 100

end collection_for_44_members_l4080_408089


namespace subset_family_bound_l4080_408035

theorem subset_family_bound (n k m : ℕ) (B : Fin m → Finset (Fin n)) :
  (∀ i, (B i).card = k) →
  (k ≥ 2) →
  (∀ i j, i < j → (B i ∩ B j).card ≤ 1) →
  m ≤ ⌊(n : ℝ) / k * ⌊(n - 1 : ℝ) / (k - 1)⌋⌋ :=
by sorry

end subset_family_bound_l4080_408035


namespace min_value_theorem_l4080_408075

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 2*a*b + b^2 + 3*c^2 ≥ 324 ∧ ∃ (a' b' c' : ℝ), a'^2 + 2*a'*b' + b'^2 + 3*c'^2 = 324 :=
by sorry

end min_value_theorem_l4080_408075


namespace hockey_league_games_l4080_408096

theorem hockey_league_games (n : ℕ) (m : ℕ) (total_games : ℕ) : 
  n = 25 → -- number of teams
  m = 15 → -- number of times each team faces every other team
  total_games = (n * (n - 1) / 2) * m →
  total_games = 4500 :=
by
  sorry

end hockey_league_games_l4080_408096


namespace conic_section_eccentricity_l4080_408068

/-- Given that real numbers 4, m, and 9 form a geometric sequence, 
    prove that the eccentricity of the conic section represented by 
    the equation x²/m + y² = 1 is either √(30)/6 or √7. -/
theorem conic_section_eccentricity (m : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ m = 4 * r ∧ 9 = m * r) →
  let e := if m > 0 
    then Real.sqrt (30) / 6 
    else Real.sqrt 7
  (∀ x y : ℝ, x^2 / m + y^2 = 1) →
  ∃ (a b c : ℝ), 
    (m > 0 → a^2 = m ∧ b^2 = 1 ∧ c^2 = a^2 - b^2 ∧ e = c / a) ∧
    (m < 0 → a^2 = 1 ∧ b^2 = -m ∧ c^2 = a^2 + b^2 ∧ e = c / a) :=
by sorry

end conic_section_eccentricity_l4080_408068


namespace inequality_solution_set_l4080_408004

theorem inequality_solution_set (x : ℝ) : (3 * x - 1) / (2 - x) ≥ 1 ↔ 3 / 4 ≤ x ∧ x < 2 := by
  sorry

end inequality_solution_set_l4080_408004


namespace negation_equivalence_l4080_408028

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 + 2 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 + 2 ≥ 0) := by
  sorry

end negation_equivalence_l4080_408028


namespace opposite_of_neg_nine_l4080_408048

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem stating that the opposite of -9 is 9
theorem opposite_of_neg_nine : opposite (-9) = 9 := by
  sorry

end opposite_of_neg_nine_l4080_408048


namespace bus_ride_difference_l4080_408010

theorem bus_ride_difference (vince_ride zachary_ride : ℝ) 
  (h1 : vince_ride = 0.62)
  (h2 : zachary_ride = 0.5) :
  vince_ride - zachary_ride = 0.12 := by
sorry

end bus_ride_difference_l4080_408010


namespace sammy_has_twenty_caps_l4080_408034

/-- Represents the number of bottle caps each person has -/
structure BottleCaps where
  sammy : ℕ
  janine : ℕ
  billie : ℕ
  tommy : ℕ

/-- The initial state of bottle caps -/
def initial_state (b : ℕ) : BottleCaps :=
  { sammy := 3 * b + 2
    janine := 3 * b
    billie := b
    tommy := 0 }

/-- The final state of bottle caps after Billie's gift -/
def final_state (b : ℕ) : BottleCaps :=
  { sammy := 3 * b + 2
    janine := 3 * b
    billie := b - 4
    tommy := 4 }

/-- The theorem stating Sammy has 20 bottle caps -/
theorem sammy_has_twenty_caps :
  ∃ b : ℕ,
    (final_state b).tommy = 2 * (final_state b).billie ∧
    (final_state b).sammy = 20 := by
  sorry

#check sammy_has_twenty_caps

end sammy_has_twenty_caps_l4080_408034


namespace sum_perfect_square_values_l4080_408073

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_perfect_square_values :
  ∀ K : ℕ, K > 0 → K < 150 →
    (∃ N : ℕ, N < 150 ∧ sum_first_n K = N * N) ↔ K = 8 ∨ K = 49 ∨ K = 59 := by
  sorry

end sum_perfect_square_values_l4080_408073


namespace sum_of_distances_less_than_diagonal_l4080_408055

-- Define the quadrilateral ABCD and point P
variable (A B C D P : ℝ × ℝ)

-- Define the conditions
variable (h1 : IsConvex A B C D)
variable (h2 : dist A B = dist C D)
variable (h3 : IsInside P A B C D)
variable (h4 : angle P B A + angle P C D = π)

-- State the theorem
theorem sum_of_distances_less_than_diagonal :
  dist P B + dist P C < dist A D :=
sorry

end sum_of_distances_less_than_diagonal_l4080_408055


namespace simultaneous_arrivals_l4080_408062

/-- The distance between points A and B in meters -/
def distance : ℕ := 2010

/-- The speed of the m-th messenger in meters per minute -/
def speed (m : ℕ) : ℕ := m

/-- The time taken by the m-th messenger to reach point B -/
def time (m : ℕ) : ℚ := distance / m

/-- The total number of messengers -/
def total_messengers : ℕ := distance

/-- Predicate for whether two messengers arrive simultaneously -/
def arrive_simultaneously (m n : ℕ) : Prop :=
  1 ≤ m ∧ m < n ∧ n ≤ total_messengers ∧ time m = time n

theorem simultaneous_arrivals :
  ∀ m n : ℕ, arrive_simultaneously m n ↔ m * n = distance ∧ 1 ≤ m ∧ m < n ∧ n ≤ total_messengers :=
by sorry

end simultaneous_arrivals_l4080_408062


namespace snowflake_weight_scientific_notation_l4080_408009

theorem snowflake_weight_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.00003 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof goes here
  sorry

end snowflake_weight_scientific_notation_l4080_408009


namespace books_difference_l4080_408061

theorem books_difference (bobby_books kristi_books : ℕ) 
  (h1 : bobby_books = 142) 
  (h2 : kristi_books = 78) : 
  bobby_books - kristi_books = 64 := by
sorry

end books_difference_l4080_408061


namespace two_tangent_circles_through_point_l4080_408097

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an angle formed by three points -/
structure Angle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Checks if a point is inside an angle -/
def isInsideAngle (M : Point) (angle : Angle) : Prop :=
  sorry

/-- Checks if a circle is tangent to both sides of an angle -/
def isTangentToAngle (circle : Circle) (angle : Angle) : Prop :=
  sorry

/-- Checks if a circle passes through a point -/
def passesThrough (circle : Circle) (point : Point) : Prop :=
  sorry

/-- Main theorem -/
theorem two_tangent_circles_through_point 
  (angle : Angle) (M : Point) (h : isInsideAngle M angle) :
  ∃ (c1 c2 : Circle), 
    c1 ≠ c2 ∧ 
    isTangentToAngle c1 angle ∧ 
    isTangentToAngle c2 angle ∧ 
    passesThrough c1 M ∧ 
    passesThrough c2 M ∧
    ∀ (c : Circle), 
      isTangentToAngle c angle → 
      passesThrough c M → 
      (c = c1 ∨ c = c2) :=
  sorry

end two_tangent_circles_through_point_l4080_408097


namespace latch_caught_14_necklaces_l4080_408029

/-- The number of necklaces caught by Boudreaux -/
def boudreaux_necklaces : ℕ := 12

/-- The number of necklaces caught by Rhonda -/
def rhonda_necklaces : ℕ := boudreaux_necklaces / 2

/-- The number of necklaces caught by Latch -/
def latch_necklaces : ℕ := 3 * rhonda_necklaces - 4

/-- Theorem stating that Latch caught 14 necklaces -/
theorem latch_caught_14_necklaces : latch_necklaces = 14 := by
  sorry

end latch_caught_14_necklaces_l4080_408029


namespace monkey_hop_distance_l4080_408044

/-- Represents the climbing problem of a monkey on a tree. -/
def monkey_climb (tree_height : ℝ) (total_hours : ℕ) (slip_distance : ℝ) (hop_distance : ℝ) : Prop :=
  let net_climb_per_hour := hop_distance - slip_distance
  (total_hours - 1 : ℝ) * net_climb_per_hour + hop_distance = tree_height

/-- Theorem stating that for the given conditions, the monkey must hop 3 feet each hour. -/
theorem monkey_hop_distance :
  monkey_climb 20 18 2 3 :=
sorry

end monkey_hop_distance_l4080_408044


namespace range_of_c_l4080_408083

/-- The statement of the theorem --/
theorem range_of_c (c : ℝ) : c > 0 ∧ c ≠ 1 →
  (((∀ x y : ℝ, x < y → c^x > c^y) ∨ (∀ x : ℝ, x + |x - 2*c| > 1)) ∧
   ¬((∀ x y : ℝ, x < y → c^x > c^y) ∧ (∀ x : ℝ, x + |x - 2*c| > 1))) ↔
  (c ∈ Set.Ioc 0 (1/2) ∪ Set.Ioi 1) :=
sorry

end range_of_c_l4080_408083
