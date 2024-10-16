import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_quadrant_l739_73996

theorem complex_number_quadrant (z : ℂ) (h : (1 - Complex.I)^2 / z = 1 + Complex.I) :
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l739_73996


namespace NUMINAMATH_CALUDE_correct_price_reduction_equation_l739_73961

/-- Represents the price reduction of a vehicle over two months -/
def price_reduction (initial_price final_price monthly_rate : ℝ) : Prop :=
  initial_price * (1 - monthly_rate)^2 = final_price

/-- Theorem stating the correct equation for the given scenario -/
theorem correct_price_reduction_equation :
  ∃ x : ℝ, price_reduction 23 18.63 x := by
  sorry

end NUMINAMATH_CALUDE_correct_price_reduction_equation_l739_73961


namespace NUMINAMATH_CALUDE_ellipse_equation_l739_73981

/-- The equation of an ellipse with specific properties -/
theorem ellipse_equation (a b c : ℝ) (h1 : a + b = 10) (h2 : 2 * c = 4 * Real.sqrt 5) 
  (h3 : a^2 = c^2 + b^2) (h4 : a > b) (h5 : b > 0) :
  ∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l739_73981


namespace NUMINAMATH_CALUDE_hospital_staff_count_l739_73986

theorem hospital_staff_count (doctors nurses : ℕ) (h1 : doctors * 9 = nurses * 5) (h2 : nurses = 180) :
  doctors + nurses = 280 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l739_73986


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l739_73969

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 506) : 
  x + (x + 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l739_73969


namespace NUMINAMATH_CALUDE_balloon_distribution_l739_73959

theorem balloon_distribution (yellow_balloons : ℕ) (blue_balloons : ℕ) (black_extra : ℕ) (schools : ℕ) :
  yellow_balloons = 3414 →
  blue_balloons = 5238 →
  black_extra = 1762 →
  schools = 15 →
  ((yellow_balloons + blue_balloons + (yellow_balloons + black_extra)) / schools : ℕ) = 921 :=
by sorry

end NUMINAMATH_CALUDE_balloon_distribution_l739_73959


namespace NUMINAMATH_CALUDE_consecutive_lcm_inequality_l739_73973

theorem consecutive_lcm_inequality : ∃ n : ℕ, 
  Nat.lcm (Nat.lcm n (n + 1)) (n + 2) > Nat.lcm (Nat.lcm (n + 3) (n + 4)) (n + 5) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_lcm_inequality_l739_73973


namespace NUMINAMATH_CALUDE_modulus_of_complex_l739_73910

theorem modulus_of_complex (z : ℂ) : (Complex.I * z = 3 + 4 * Complex.I) → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l739_73910


namespace NUMINAMATH_CALUDE_chess_tournament_schedules_l739_73900

/-- Represents a chess tournament between two schools --/
structure ChessTournament where
  /-- Number of players in each school --/
  players_per_school : Nat
  /-- Number of games each player plays against each opponent from the other school --/
  games_per_opponent : Nat
  /-- Number of games played simultaneously in each round --/
  games_per_round : Nat

/-- Calculates the total number of games in the tournament --/
def totalGames (t : ChessTournament) : Nat :=
  t.players_per_school * t.players_per_school * t.games_per_opponent

/-- Calculates the number of rounds in the tournament --/
def numberOfRounds (t : ChessTournament) : Nat :=
  totalGames t / t.games_per_round

/-- Theorem stating the number of ways to schedule the tournament --/
theorem chess_tournament_schedules (t : ChessTournament) 
  (h1 : t.players_per_school = 4)
  (h2 : t.games_per_opponent = 2)
  (h3 : t.games_per_round = 4) :
  Nat.factorial (numberOfRounds t) = 40320 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_schedules_l739_73900


namespace NUMINAMATH_CALUDE_pages_written_theorem_l739_73918

/-- Calculates the number of pages written in a year given the specified writing habits -/
def pages_written_per_year (pages_per_letter : ℕ) (num_friends : ℕ) (writing_frequency_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  pages_per_letter * num_friends * writing_frequency_per_week * weeks_per_year

/-- Proves that given the specified writing habits, the total number of pages written in a year is 624 -/
theorem pages_written_theorem :
  pages_written_per_year 3 2 2 52 = 624 := by
  sorry

end NUMINAMATH_CALUDE_pages_written_theorem_l739_73918


namespace NUMINAMATH_CALUDE_quadratic_increasing_negative_l739_73908

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2

-- Theorem statement
theorem quadratic_increasing_negative (x₁ x₂ : ℝ) :
  x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_negative_l739_73908


namespace NUMINAMATH_CALUDE_coloring_existence_and_impossibility_l739_73937

def is_monochromatic (color : ℕ → Bool) (x y z : ℕ) : Prop :=
  color x = color y ∧ color y = color z

theorem coloring_existence_and_impossibility :
  (∃ (color : ℕ → Bool),
    ∀ x y z, 1 ≤ x ∧ x ≤ 2017 ∧ 1 ≤ y ∧ y ≤ 2017 ∧ 1 ≤ z ∧ z ≤ 2017 →
      8 * (x + y) = z → ¬is_monochromatic color x y z) ∧
  (∀ n : ℕ, n ≥ 2056 →
    ¬∃ (color : ℕ → Bool),
      ∀ x y z, 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n ∧ 1 ≤ z ∧ z ≤ n →
        8 * (x + y) = z → ¬is_monochromatic color x y z) :=
by sorry

end NUMINAMATH_CALUDE_coloring_existence_and_impossibility_l739_73937


namespace NUMINAMATH_CALUDE_sally_recites_three_poems_l739_73929

/-- The number of poems Sally can still recite -/
def poems_recited (initial_poems : ℕ) (forgotten_poems : ℕ) : ℕ :=
  initial_poems - forgotten_poems

/-- Theorem: Sally can recite 3 poems -/
theorem sally_recites_three_poems :
  poems_recited 8 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sally_recites_three_poems_l739_73929


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l739_73950

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- The longer base of the trapezoid -/
  longerBase : ℝ
  /-- One of the base angles of the trapezoid -/
  baseAngle : ℝ
  /-- The height of the trapezoid -/
  height : ℝ

/-- The area of the isosceles trapezoid -/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  ∃ (t : IsoscelesTrapezoid),
    t.longerBase = 20 ∧
    t.baseAngle = Real.arcsin 0.6 ∧
    t.height = 9 ∧
    trapezoidArea t = 100 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l739_73950


namespace NUMINAMATH_CALUDE_car_distances_theorem_l739_73928

/-- Represents the distance traveled by a car -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the distances traveled by two cars under given conditions -/
theorem car_distances_theorem (distance_AB : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ) 
  (h1 : distance_AB = 70)
  (h2 : speed_car1 = 30)
  (h3 : speed_car2 = 40)
  (h4 : speed_car1 + speed_car2 > 0) -- Ensure division by zero is avoided
  : ∃ (time : ℝ), 
    distance speed_car1 time = 150 ∧ 
    distance speed_car2 time = 200 ∧
    time * (speed_car1 + speed_car2) = 5 * distance_AB :=
sorry

end NUMINAMATH_CALUDE_car_distances_theorem_l739_73928


namespace NUMINAMATH_CALUDE_new_sales_tax_percentage_l739_73966

/-- Proves that the new sales tax percentage is 3 1/3% given the conditions --/
theorem new_sales_tax_percentage
  (market_price : ℝ)
  (original_tax_rate : ℝ)
  (savings : ℝ)
  (h1 : market_price = 10800)
  (h2 : original_tax_rate = 3.5 / 100)
  (h3 : savings = 18) :
  let original_tax := market_price * original_tax_rate
  let new_tax := original_tax - savings
  let new_tax_rate := new_tax / market_price
  new_tax_rate = 10 / 3 / 100 := by sorry

end NUMINAMATH_CALUDE_new_sales_tax_percentage_l739_73966


namespace NUMINAMATH_CALUDE_room_area_l739_73912

theorem room_area (breadth length : ℝ) : 
  length = 3 * breadth →
  2 * (length + breadth) = 16 →
  length * breadth = 12 := by
sorry

end NUMINAMATH_CALUDE_room_area_l739_73912


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l739_73976

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * (a - 1) * x + 2

-- Define the property of f being decreasing on (-∞, 4]
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 4 → f x > f y

-- State the theorem
theorem f_decreasing_implies_a_range :
  ∀ a : ℝ, isDecreasingOn (f a) ↔ 0 ≤ a ∧ a ≤ 1/5 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l739_73976


namespace NUMINAMATH_CALUDE_y_order_on_quadratic_l739_73970

/-- A quadratic function of the form y = x² + 4x + k -/
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + k

/-- Theorem stating the order of y-coordinates for specific x-values on the quadratic function -/
theorem y_order_on_quadratic (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : quadratic_function k (-4) = y₁)
  (h₂ : quadratic_function k (-1) = y₂)
  (h₃ : quadratic_function k 1 = y₃) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry


end NUMINAMATH_CALUDE_y_order_on_quadratic_l739_73970


namespace NUMINAMATH_CALUDE_quadratic_equality_l739_73964

theorem quadratic_equality (a b c x : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∃ p q r : Fin 6, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (let f : Fin 6 → ℝ := λ i =>
      match i with
      | 0 => a*x^2 + b*x + c
      | 1 => a*x^2 + c*x + b
      | 2 => b*x^2 + c*x + a
      | 3 => b*x^2 + a*x + c
      | 4 => c*x^2 + a*x + b
      | 5 => c*x^2 + b*x + a
    f p = f q ∧ f q = f r)) →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equality_l739_73964


namespace NUMINAMATH_CALUDE_inequality_system_solution_l739_73902

theorem inequality_system_solution :
  let S := {x : ℝ | 3*x - 2 < 2*(x + 1) ∧ (x - 1)/2 > 1}
  S = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l739_73902


namespace NUMINAMATH_CALUDE_ball_distribution_after_199_students_l739_73948

/-- Represents the state of the boxes -/
structure BoxState :=
  (a b c d e : ℕ)

/-- Simulates one student's action -/
def moveOneBall (state : BoxState) : BoxState :=
  let minBox := min state.a (min state.b (min state.c (min state.d state.e)))
  { a := if state.a > minBox then state.a - 1 else state.a + 4,
    b := if state.b > minBox then state.b - 1 else state.b + 4,
    c := if state.c > minBox then state.c - 1 else state.c + 4,
    d := if state.d > minBox then state.d - 1 else state.d + 4,
    e := if state.e > minBox then state.e - 1 else state.e + 4 }

/-- Simulates n students' actions -/
def simulateNStudents (n : ℕ) (initialState : BoxState) : BoxState :=
  match n with
  | 0 => initialState
  | n + 1 => moveOneBall (simulateNStudents n initialState)

/-- The main theorem to prove -/
theorem ball_distribution_after_199_students :
  let initialState : BoxState := ⟨9, 5, 3, 2, 1⟩
  let finalState := simulateNStudents 199 initialState
  finalState = ⟨5, 6, 4, 3, 2⟩ := by
  sorry


end NUMINAMATH_CALUDE_ball_distribution_after_199_students_l739_73948


namespace NUMINAMATH_CALUDE_union_condition_implies_m_leq_4_l739_73949

/-- Given sets A and B, if their union equals A, then m ≤ 4 -/
theorem union_condition_implies_m_leq_4 (m : ℝ) : 
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
  let B := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  (A ∪ B = A) → m ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_union_condition_implies_m_leq_4_l739_73949


namespace NUMINAMATH_CALUDE_point_y_value_l739_73939

/-- An angle with vertex at the origin and initial side on the non-negative x-axis -/
structure AngleAtOrigin where
  α : ℝ
  initial_side_on_x_axis : 0 ≤ α ∧ α < 2 * Real.pi

/-- A point on the terminal side of an angle -/
structure PointOnTerminalSide (angle : AngleAtOrigin) where
  x : ℝ
  y : ℝ
  on_terminal_side : x = 6 ∧ y = 6 * Real.tan angle.α

/-- Theorem: For an angle α with sin α = -4/5, if P(6, y) is on its terminal side, then y = -8 -/
theorem point_y_value (angle : AngleAtOrigin) 
  (h_sin : Real.sin angle.α = -4/5) 
  (point : PointOnTerminalSide angle) : 
  point.y = -8 := by
  sorry

end NUMINAMATH_CALUDE_point_y_value_l739_73939


namespace NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l739_73990

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def coinTossProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

/-- Theorem: The probability of getting exactly 3 heads when tossing a fair coin 8 times is 7/32 -/
theorem three_heads_in_eight_tosses :
  coinTossProbability 8 3 = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l739_73990


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l739_73956

/-- Given a square of side length 2, divided into two congruent trapezoids and a pentagon,
    this theorem proves that the length of the longer parallel side of each trapezoid is 11/5
    under specific conditions. -/
theorem trapezoid_side_length (square_side : ℝ) (trapezoid_area pentagon_area : ℝ)
  (longer_side shorter_side : ℝ) :
  square_side = 2 →
  trapezoid_area = pentagon_area →
  2 * trapezoid_area = pentagon_area →
  trapezoid_area = (longer_side + shorter_side) / 2 →
  shorter_side = 1 →
  longer_side = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l739_73956


namespace NUMINAMATH_CALUDE_polygon_sides_l739_73993

theorem polygon_sides (n : ℕ) (h : n > 2) : 
  (360 : ℝ) / (180 * (n - 2)) = 2 / 9 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l739_73993


namespace NUMINAMATH_CALUDE_at_most_one_perfect_square_l739_73944

def sequence_a : ℕ → ℤ
  | 0 => 0  -- arbitrary initial value
  | n + 1 => (sequence_a n)^3 + 103

theorem at_most_one_perfect_square :
  ∃ (n : ℕ), (∃ (k : ℤ), sequence_a n = k^2) →
    ∀ (m : ℕ), m ≠ n → ¬∃ (l : ℤ), sequence_a m = l^2 :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_perfect_square_l739_73944


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l739_73920

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l739_73920


namespace NUMINAMATH_CALUDE_product_equals_nine_twentieths_l739_73903

theorem product_equals_nine_twentieths : 6 * 0.5 * (3/4) * 0.2 = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_nine_twentieths_l739_73903


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l739_73905

theorem subtraction_multiplication_equality : (3.625 - 1.047) * 4 = 10.312 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l739_73905


namespace NUMINAMATH_CALUDE_second_digit_of_n_l739_73957

theorem second_digit_of_n (n : ℕ) : 
  (10^99 ≤ 8*n) ∧ (8*n < 10^100) ∧ 
  (10^101 ≤ 81*n - 102) ∧ (81*n - 102 < 10^102) →
  (n / 10^97) % 10 = 2 :=
by sorry

end NUMINAMATH_CALUDE_second_digit_of_n_l739_73957


namespace NUMINAMATH_CALUDE_polynomial_factorization_l739_73926

theorem polynomial_factorization (x : ℝ) :
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l739_73926


namespace NUMINAMATH_CALUDE_percent_relation_l739_73984

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) :
  4 * b / a * 100 = 1000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l739_73984


namespace NUMINAMATH_CALUDE_lenny_video_game_spending_l739_73988

def video_game_expenditure (initial_amount grocery_spending remaining_amount : ℕ) : ℕ :=
  initial_amount - grocery_spending - remaining_amount

theorem lenny_video_game_spending :
  video_game_expenditure 84 21 39 = 24 :=
by sorry

end NUMINAMATH_CALUDE_lenny_video_game_spending_l739_73988


namespace NUMINAMATH_CALUDE_granger_bread_loaves_l739_73936

/-- Represents the grocery items and their prices --/
structure GroceryItems where
  spam_price : ℕ
  peanut_butter_price : ℕ
  bread_price : ℕ

/-- Represents the quantities of items bought --/
structure Quantities where
  spam_cans : ℕ
  peanut_butter_jars : ℕ

/-- Calculates the number of bread loaves bought given the total amount paid --/
def bread_loaves_bought (items : GroceryItems) (quantities : Quantities) (total_paid : ℕ) : ℕ :=
  (total_paid - (items.spam_price * quantities.spam_cans + items.peanut_butter_price * quantities.peanut_butter_jars)) / items.bread_price

/-- Theorem stating that Granger bought 4 loaves of bread --/
theorem granger_bread_loaves :
  let items := GroceryItems.mk 3 5 2
  let quantities := Quantities.mk 12 3
  let total_paid := 59
  bread_loaves_bought items quantities total_paid = 4 := by
  sorry


end NUMINAMATH_CALUDE_granger_bread_loaves_l739_73936


namespace NUMINAMATH_CALUDE_price_increase_and_discount_l739_73938

theorem price_increase_and_discount (original_price : ℝ) (increase_percentage : ℝ) :
  original_price * (1 + increase_percentage) * (1 - 0.2) = original_price →
  increase_percentage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_and_discount_l739_73938


namespace NUMINAMATH_CALUDE_proportional_segments_l739_73946

theorem proportional_segments (a b c d : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a / b = c / d) →
  a = 4 →
  b = 2 →
  c = 3 →
  d = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_proportional_segments_l739_73946


namespace NUMINAMATH_CALUDE_tomato_price_per_pound_l739_73980

/-- Calculates the price per pound of tomatoes in Scott's ratatouille recipe --/
theorem tomato_price_per_pound
  (eggplant_weight : ℝ) (eggplant_price : ℝ)
  (zucchini_weight : ℝ) (zucchini_price : ℝ)
  (tomato_weight : ℝ)
  (onion_weight : ℝ) (onion_price : ℝ)
  (basil_weight : ℝ) (basil_price : ℝ)
  (yield_quarts : ℝ) (price_per_quart : ℝ)
  (h1 : eggplant_weight = 5)
  (h2 : eggplant_price = 2)
  (h3 : zucchini_weight = 4)
  (h4 : zucchini_price = 2)
  (h5 : tomato_weight = 4)
  (h6 : onion_weight = 3)
  (h7 : onion_price = 1)
  (h8 : basil_weight = 1)
  (h9 : basil_price = 2.5 * 2)  -- $2.50 per half pound, so double for 1 pound
  (h10 : yield_quarts = 4)
  (h11 : price_per_quart = 10) :
  (yield_quarts * price_per_quart - 
   (eggplant_weight * eggplant_price + 
    zucchini_weight * zucchini_price + 
    onion_weight * onion_price + 
    basil_weight * basil_price)) / tomato_weight = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_tomato_price_per_pound_l739_73980


namespace NUMINAMATH_CALUDE_solve_brownies_problem_l739_73968

def brownies_problem (total : ℕ) (to_admin : ℕ) (to_simon : ℕ) (left : ℕ) : Prop :=
  let remaining_after_admin := total - to_admin
  let to_carl := remaining_after_admin - to_simon - left
  (to_carl : ℚ) / remaining_after_admin = 1 / 2

theorem solve_brownies_problem :
  brownies_problem 20 10 2 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_brownies_problem_l739_73968


namespace NUMINAMATH_CALUDE_problem_statement_l739_73987

theorem problem_statement (a b x y : ℝ) 
  (h1 : a + b = 2) 
  (h2 : x + y = 2) 
  (h3 : a * x + b * y = 5) : 
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l739_73987


namespace NUMINAMATH_CALUDE_line_through_three_points_l739_73947

/-- A line passes through three points: (2, 5), (-3, m), and (15, -1).
    This theorem proves that the value of m is 95/13. -/
theorem line_through_three_points (m : ℚ) : 
  (∃ (line : ℝ → ℝ), 
    line 2 = 5 ∧ 
    line (-3) = m ∧ 
    line 15 = -1) → 
  m = 95 / 13 :=
by sorry

end NUMINAMATH_CALUDE_line_through_three_points_l739_73947


namespace NUMINAMATH_CALUDE_power_of_81_l739_73978

theorem power_of_81 : (81 : ℝ) ^ (5/2) = 59049 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l739_73978


namespace NUMINAMATH_CALUDE_min_sum_given_product_l739_73941

theorem min_sum_given_product (a b c : ℕ+) : a * b * c = 2310 → (∀ x y z : ℕ+, x * y * z = 2310 → a + b + c ≤ x + y + z) → a + b + c = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l739_73941


namespace NUMINAMATH_CALUDE_horner_method_innermost_polynomial_l739_73934

def f (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

def horner_v1 (a : ℝ) : ℝ := 1 * a + 1

theorem horner_method_innermost_polynomial :
  horner_v1 3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_innermost_polynomial_l739_73934


namespace NUMINAMATH_CALUDE_stating_solutions_eq_partitions_l739_73991

/-- The number of solutions to the equation in positive integers -/
def numSolutions : ℕ := sorry

/-- The number of partitions of 7 -/
def numPartitions7 : ℕ := sorry

/-- 
Theorem stating that the number of solutions to the equation
a₁(b₁) + a₂(b₁+b₂) + ... + aₖ(b₁+b₂+...+bₖ) = 7
in positive integers (k; a₁, a₂, ..., aₖ; b₁, b₂, ..., bₖ)
is equal to the number of partitions of 7
-/
theorem solutions_eq_partitions : numSolutions = numPartitions7 := by sorry

end NUMINAMATH_CALUDE_stating_solutions_eq_partitions_l739_73991


namespace NUMINAMATH_CALUDE_original_number_proof_l739_73931

theorem original_number_proof (t : ℝ) : 
  t * (1 + 0.125) - t * (1 - 0.25) = 30 → t = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l739_73931


namespace NUMINAMATH_CALUDE_perimeter_PQR_l739_73922

/-- Represents a triangle with three points -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Theorem: Perimeter of PQR in the given configuration -/
theorem perimeter_PQR (ABC : Triangle)
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (R : ℝ × ℝ)
  (h_AB : distance ABC.A ABC.B = 13)
  (h_BC : distance ABC.B ABC.C = 14)
  (h_CA : distance ABC.C ABC.A = 15)
  (h_P_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • ABC.B + t • ABC.C)
  (h_Q_on_CA : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • ABC.C + t • ABC.A)
  (h_R_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • ABC.A + t • ABC.B)
  (h_equal_perimeters : perimeter ⟨ABC.A, Q, R⟩ = perimeter ⟨ABC.B, P, R⟩ ∧
                        perimeter ⟨ABC.A, Q, R⟩ = perimeter ⟨ABC.C, P, Q⟩)
  (h_ratio : perimeter ⟨ABC.A, Q, R⟩ = 4/5 * perimeter ⟨P, Q, R⟩) :
  perimeter ⟨P, Q, R⟩ = 30 := by sorry

end NUMINAMATH_CALUDE_perimeter_PQR_l739_73922


namespace NUMINAMATH_CALUDE_mel_age_when_katherine_is_two_dozen_l739_73909

/-- The age difference between Katherine and Mel -/
def age_difference : ℕ := 3

/-- Katherine's age when she is two dozen years old -/
def katherine_age : ℕ := 24

/-- Mel's age when Katherine is two dozen years old -/
def mel_age : ℕ := katherine_age - age_difference

theorem mel_age_when_katherine_is_two_dozen : mel_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_mel_age_when_katherine_is_two_dozen_l739_73909


namespace NUMINAMATH_CALUDE_product_increase_l739_73943

theorem product_increase (a b : ℕ) (h1 : a * b = 72) : a * (10 * b) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_l739_73943


namespace NUMINAMATH_CALUDE_stock_value_comparison_l739_73962

def initial_investment : ℝ := 200

def first_year_change_DD : ℝ := 1.10
def first_year_change_EE : ℝ := 0.85
def first_year_change_FF : ℝ := 1.05

def second_year_change_DD : ℝ := 1.05
def second_year_change_EE : ℝ := 1.15
def second_year_change_FF : ℝ := 0.90

def D : ℝ := initial_investment * first_year_change_DD * second_year_change_DD
def E : ℝ := initial_investment * first_year_change_EE * second_year_change_EE
def F : ℝ := initial_investment * first_year_change_FF * second_year_change_FF

theorem stock_value_comparison : F < E ∧ E < D := by
  sorry

end NUMINAMATH_CALUDE_stock_value_comparison_l739_73962


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l739_73917

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (V : ℝ) 
  (h : ℝ) 
  (l : ℝ) 
  (A : ℝ) :
  r = 3 →
  V = 12 * Real.pi →
  V = (1/3) * Real.pi * r^2 * h →
  l^2 = r^2 + h^2 →
  A = Real.pi * r * l →
  A = 15 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l739_73917


namespace NUMINAMATH_CALUDE_g_composition_of_six_l739_73999

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 2 * x + 1

theorem g_composition_of_six : g (g (g (g 6))) = 23 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_six_l739_73999


namespace NUMINAMATH_CALUDE_savings_ratio_l739_73979

/-- Proves that the ratio of Nora's savings to Lulu's savings is 5:1 given the problem conditions -/
theorem savings_ratio (debt : ℝ) (lulu_savings : ℝ) (remaining_per_person : ℝ)
  (h1 : debt = 40)
  (h2 : lulu_savings = 6)
  (h3 : remaining_per_person = 2)
  (h4 : ∃ (x : ℝ), x > 0 ∧ ∃ (tamara_savings : ℝ),
    x * lulu_savings = 3 * tamara_savings ∧
    debt + 3 * remaining_per_person = lulu_savings + x * lulu_savings + tamara_savings) :
  ∃ (nora_savings : ℝ), nora_savings / lulu_savings = 5 := by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l739_73979


namespace NUMINAMATH_CALUDE_password_recovery_l739_73904

def alphabet_size : Nat := 32

def encode (c : Char) : Nat := 
  sorry

def decode (n : Nat) : Char := 
  sorry

def generate_x (a b x : Nat) : Nat :=
  (a * x + b) % 10

def generate_c (x y : Nat) : Nat :=
  (x + y) % 10

def is_valid_sequence (s : List Nat) (password : String) (a b : Nat) : Prop :=
  sorry

theorem password_recovery (a b : Nat) : 
  ∃ (password : String),
    password.length = 4 ∧ 
    is_valid_sequence [2, 8, 5, 2, 8, 3, 1, 9, 8, 4, 1, 8, 4, 9, 7] (password ++ password) a b ∧
    password = "яхта" :=
  sorry

end NUMINAMATH_CALUDE_password_recovery_l739_73904


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l739_73967

theorem smallest_n_for_candy_purchase : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 → (20 * m) % 12 = 0 ∧ (20 * m) % 14 = 0 ∧ (20 * m) % 15 = 0 → m ≥ n) ∧
  (20 * n) % 12 = 0 ∧ (20 * n) % 14 = 0 ∧ (20 * n) % 15 = 0 ∧ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l739_73967


namespace NUMINAMATH_CALUDE_conference_session_duration_l739_73975

/-- Given a conference duration and break time, calculate the session time in minutes. -/
def conference_session_time (hours minutes break_time : ℕ) : ℕ :=
  hours * 60 + minutes - break_time

/-- Theorem: A conference lasting 8 hours and 45 minutes with a 30-minute break has a session time of 495 minutes. -/
theorem conference_session_duration :
  conference_session_time 8 45 30 = 495 :=
by sorry

end NUMINAMATH_CALUDE_conference_session_duration_l739_73975


namespace NUMINAMATH_CALUDE_three_primes_sum_l739_73911

theorem three_primes_sum (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p * q * r = 31 * (p + q + r) →
  p + q + r = 51 := by
sorry

end NUMINAMATH_CALUDE_three_primes_sum_l739_73911


namespace NUMINAMATH_CALUDE_decagon_diagonals_l739_73927

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals :
  diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l739_73927


namespace NUMINAMATH_CALUDE_total_money_is_75_l739_73906

/-- Represents the money distribution and orange selling scenario -/
structure MoneyDistribution where
  x : ℝ  -- The common factor in the money distribution
  cara_money : ℝ := 4 * x
  janet_money : ℝ := 5 * x
  jerry_money : ℝ := 6 * x
  total_money : ℝ := cara_money + janet_money + jerry_money
  combined_money : ℝ := cara_money + janet_money
  selling_price_ratio : ℝ := 0.8
  loss : ℝ := combined_money - (selling_price_ratio * combined_money)

/-- Theorem stating the total amount of money given the conditions -/
theorem total_money_is_75 (d : MoneyDistribution) 
  (h_loss : d.loss = 9) : d.total_money = 75 := by
  sorry


end NUMINAMATH_CALUDE_total_money_is_75_l739_73906


namespace NUMINAMATH_CALUDE_calculate_fraction_l739_73972

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

/-- The main theorem stating the result of the calculation -/
theorem calculate_fraction (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 ≠ 0) :
  (f 6 - f 3) / f 2 = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_calculate_fraction_l739_73972


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l739_73963

/-- Proves that the number of child tickets sold is 63 given the theater conditions --/
theorem theater_ticket_sales (total_seats : ℕ) (adult_price child_price : ℕ) (total_revenue : ℕ) 
  (h1 : total_seats = 80)
  (h2 : adult_price = 12)
  (h3 : child_price = 5)
  (h4 : total_revenue = 519) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_seats ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 63 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l739_73963


namespace NUMINAMATH_CALUDE_divide_fractions_example_l739_73901

theorem divide_fractions_example : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end NUMINAMATH_CALUDE_divide_fractions_example_l739_73901


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l739_73923

theorem ellipse_eccentricity (a b c : ℝ) (θ : ℝ) : 
  a > b ∧ b > 0 ∧  -- conditions for ellipse
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x-c)^2 + y^2 ≤ 1) ∧  -- circle inside ellipse
  (π/3 ≤ θ ∧ θ ≤ π/2) →  -- angle condition
  c/a = 3 - 2 * Real.sqrt 2 :=  -- eccentricity
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l739_73923


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l739_73913

/-- A circle with a given diameter -/
structure Circle where
  diameter : ℝ

/-- A line with a given distance from a point -/
structure Line where
  distanceFromPoint : ℝ

/-- Defines the relationship between a line and a circle -/
inductive Relationship
  | Intersecting
  | Tangent
  | Disjoint

theorem line_tangent_to_circle (c : Circle) (l : Line) :
  c.diameter = 13 →
  l.distanceFromPoint = 13 / 2 →
  Relationship.Tangent = (
    if l.distanceFromPoint = c.diameter / 2
    then Relationship.Tangent
    else if l.distanceFromPoint < c.diameter / 2
    then Relationship.Intersecting
    else Relationship.Disjoint
  ) := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l739_73913


namespace NUMINAMATH_CALUDE_intersection_point_property_l739_73960

theorem intersection_point_property (n : ℕ) (x₀ y₀ : ℝ) (hn : n ≥ 2) 
  (h1 : y₀^2 = n * x₀ - 1) (h2 : y₀ = x₀) :
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, k ≥ 2 ∧ (x₀^m)^2 = k * (x₀^m) - 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_property_l739_73960


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l739_73914

-- Define the system of equations
def equation1 (x : Fin 8 → ℝ) : Prop := x 0 + x 1 + x 2 = 6
def equation2 (x : Fin 8 → ℝ) : Prop := x 1 + x 2 + x 3 = 9
def equation3 (x : Fin 8 → ℝ) : Prop := x 2 + x 3 + x 4 = 3
def equation4 (x : Fin 8 → ℝ) : Prop := x 3 + x 4 + x 5 = -3
def equation5 (x : Fin 8 → ℝ) : Prop := x 4 + x 5 + x 6 = -9
def equation6 (x : Fin 8 → ℝ) : Prop := x 5 + x 6 + x 7 = -6
def equation7 (x : Fin 8 → ℝ) : Prop := x 6 + x 7 + x 0 = -2
def equation8 (x : Fin 8 → ℝ) : Prop := x 7 + x 0 + x 1 = 2

-- Define the solution
def solution : Fin 8 → ℝ
| 0 => 1
| 1 => 2
| 2 => 3
| 3 => 4
| 4 => -4
| 5 => -3
| 6 => -2
| 7 => -1

-- Theorem statement
theorem solution_satisfies_equations :
  equation1 solution ∧
  equation2 solution ∧
  equation3 solution ∧
  equation4 solution ∧
  equation5 solution ∧
  equation6 solution ∧
  equation7 solution ∧
  equation8 solution :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l739_73914


namespace NUMINAMATH_CALUDE_log_inequality_l739_73933

theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (Real.log b) / (a - 1) = (a + 1) / a) : 
  Real.log b / Real.log a > 2 := by
sorry

end NUMINAMATH_CALUDE_log_inequality_l739_73933


namespace NUMINAMATH_CALUDE_twenty_thousand_scientific_notation_l739_73958

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  prop : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_thousand_scientific_notation :
  toScientificNotation 20000 = ScientificNotation.mk 2 4 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_twenty_thousand_scientific_notation_l739_73958


namespace NUMINAMATH_CALUDE_fayes_rows_l739_73995

/-- Given that Faye has 210 crayons in total and places 30 crayons in each row,
    prove that she created 7 rows. -/
theorem fayes_rows (total_crayons : ℕ) (crayons_per_row : ℕ) (h1 : total_crayons = 210) (h2 : crayons_per_row = 30) :
  total_crayons / crayons_per_row = 7 := by
  sorry

end NUMINAMATH_CALUDE_fayes_rows_l739_73995


namespace NUMINAMATH_CALUDE_integral_equals_half_l739_73940

open Real MeasureTheory Interval

/-- The definite integral of 1 / (1 + sin x - cos x)^2 from 2 arctan(1/2) to π/2 equals 1/2 -/
theorem integral_equals_half :
  ∫ x in 2 * arctan (1/2)..π/2, 1 / (1 + sin x - cos x)^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_half_l739_73940


namespace NUMINAMATH_CALUDE_two_triangles_exist_l739_73915

/-- Represents a triangle with side lengths and heights -/
structure Triangle where
  a : ℝ
  m_b : ℝ
  m_c : ℝ

/-- Given conditions for the triangle construction problem -/
def givenConditions : Triangle where
  a := 6
  m_b := 1
  m_c := 2

/-- Predicate to check if a triangle satisfies the given conditions -/
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = givenConditions.a ∧ t.m_b = givenConditions.m_b ∧ t.m_c = givenConditions.m_c

/-- Theorem stating that exactly two distinct triangles satisfy the given conditions -/
theorem two_triangles_exist :
  ∃ (t1 t2 : Triangle), satisfiesConditions t1 ∧ satisfiesConditions t2 ∧ t1 ≠ t2 ∧
  ∀ (t : Triangle), satisfiesConditions t → (t = t1 ∨ t = t2) :=
sorry

end NUMINAMATH_CALUDE_two_triangles_exist_l739_73915


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l739_73954

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallestDivisibleBy1To10 : ℕ := 2520

/-- Checks if a number is divisible by all integers from 1 to 10 -/
def isDivisibleBy1To10 (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 10 → n % i = 0

theorem smallest_divisible_by_1_to_10 :
  isDivisibleBy1To10 smallestDivisibleBy1To10 ∧
  ∀ n : ℕ, 0 < n ∧ n < smallestDivisibleBy1To10 → ¬isDivisibleBy1To10 n := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l739_73954


namespace NUMINAMATH_CALUDE_plane_equation_correct_l739_73924

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The foot of the perpendicular from the origin to the plane -/
def footPoint : Point3D := ⟨10, -5, 2⟩

/-- The coefficients of the plane equation -/
def planeCoeffs : PlaneCoefficients := ⟨10, -5, 2, -129⟩

/-- Checks if a point satisfies the plane equation -/
def satisfiesPlaneEquation (p : Point3D) (c : PlaneCoefficients) : Prop :=
  c.A * p.x + c.B * p.y + c.C * p.z + c.D = 0

/-- Checks if a vector is perpendicular to another vector -/
def isPerpendicular (v1 v2 : Point3D) : Prop :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z = 0

/-- Theorem stating that the given plane equation is correct -/
theorem plane_equation_correct :
  satisfiesPlaneEquation footPoint planeCoeffs ∧
  isPerpendicular footPoint ⟨planeCoeffs.A, planeCoeffs.B, planeCoeffs.C⟩ ∧
  planeCoeffs.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs planeCoeffs.A) (Int.natAbs planeCoeffs.B))
          (Nat.gcd (Int.natAbs planeCoeffs.C) (Int.natAbs planeCoeffs.D)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l739_73924


namespace NUMINAMATH_CALUDE_original_triangle_area_l739_73925

-- Define the properties of the triangle
def is_oblique_projection (original : Real → Real → Real → Real) 
  (projected : Real → Real → Real → Real) : Prop := sorry

def is_equilateral (triangle : Real → Real → Real → Real) : Prop := sorry

def side_length (triangle : Real → Real → Real → Real) : Real := sorry

def area (triangle : Real → Real → Real → Real) : Real := sorry

-- Theorem statement
theorem original_triangle_area 
  (original projected : Real → Real → Real → Real) :
  is_oblique_projection original projected →
  is_equilateral projected →
  side_length projected = 1 →
  (area projected) / (area original) = Real.sqrt 2 / 4 →
  area original = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l739_73925


namespace NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_line_l739_73942

/-- The equation of a line passing through the center of a circle and parallel to another line -/
theorem line_through_circle_center_parallel_to_line :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 2*x + 2*y = 0
  let parallel_line_eq : ℝ → ℝ → Prop := λ x y => 2*x - y = 0
  let result_line_eq : ℝ → ℝ → Prop := λ x y => 2*x - y - 3 = 0
  ∃ (center_x center_y : ℝ),
    (∀ x y, circle_eq x y ↔ (x - center_x)^2 + (y - center_y)^2 = (center_x^2 + center_y^2)) →
    (∀ x y, result_line_eq x y ↔ y - center_y = 2 * (x - center_x)) →
    (∀ x₁ y₁ x₂ y₂, parallel_line_eq x₁ y₁ ∧ parallel_line_eq x₂ y₂ → y₂ - y₁ = 2 * (x₂ - x₁)) →
    result_line_eq center_x center_y :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_line_l739_73942


namespace NUMINAMATH_CALUDE_angle_y_is_90_l739_73977

-- Define the angles
def angle_ABC : ℝ := 120
def angle_ABE : ℝ := 30

-- Define the theorem
theorem angle_y_is_90 :
  ∀ (angle_y angle_ABD : ℝ),
  -- Condition 3
  angle_ABD + angle_ABC = 180 →
  -- Condition 4
  angle_ABE + angle_y = 180 →
  -- Condition 5 (using angle_y instead of explicitly stating the third angle)
  angle_y + angle_ABD + angle_ABE = 180 →
  -- Conclusion
  angle_y = 90 := by
sorry

end NUMINAMATH_CALUDE_angle_y_is_90_l739_73977


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l739_73921

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in radians -/
  face_angle : ℝ

/-- Volume of the tetrahedron in cm³ -/
def tetrahedron_volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  ∃ t : Tetrahedron,
    t.ab_length = 5 ∧
    t.abc_area = 20 ∧
    t.abd_area = 18 ∧
    t.face_angle = π / 4 ∧
    tetrahedron_volume t = 24 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l739_73921


namespace NUMINAMATH_CALUDE_grace_landscaping_pricing_l739_73930

/-- Grace's landscaping business pricing and earnings in September --/
theorem grace_landscaping_pricing (mowing_rate : ℝ) (weeding_rate : ℝ) 
  (mowing_hours : ℝ) (weeding_hours : ℝ) (mulching_hours : ℝ) (total_earnings : ℝ) 
  (h1 : mowing_rate = 6)
  (h2 : weeding_rate = 11)
  (h3 : mowing_hours = 63)
  (h4 : weeding_hours = 9)
  (h5 : mulching_hours = 10)
  (h6 : total_earnings = 567)
  (h7 : total_earnings = mowing_rate * mowing_hours + weeding_rate * weeding_hours + mulching_hours * mulching_rate) :
  mulching_rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_grace_landscaping_pricing_l739_73930


namespace NUMINAMATH_CALUDE_chocolate_division_l739_73955

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (num_friends : ℕ) :
  total_chocolate = 75 / 7 →
  num_piles = 5 →
  num_friends = 4 →
  (total_chocolate / num_piles) * (num_piles - 1) / num_friends = 15 / 7 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_division_l739_73955


namespace NUMINAMATH_CALUDE_zero_points_product_bound_l739_73982

open Real

theorem zero_points_product_bound (a : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h_zero₁ : Real.log x₁ = a * x₁)
  (h_zero₂ : Real.log x₂ = a * x₂) :
  x₁ * x₂ > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zero_points_product_bound_l739_73982


namespace NUMINAMATH_CALUDE_circle_condition_l739_73989

theorem circle_condition (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*k*x - 2*y + 5*k = 0 ∧ 
   ∀ (x' y' : ℝ), x'^2 + y'^2 + 4*k*x' - 2*y' + 5*k = 0 → 
   (x' - x)^2 + (y' - y)^2 = (x - x)^2 + (y - y)^2) ↔ 
  (k > 1 ∨ k < 1/4) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l739_73989


namespace NUMINAMATH_CALUDE_system_solution_l739_73919

theorem system_solution : 
  ∃ (x y u v : ℝ), 
    (5 * x^7 + 3 * y^2 + 5 * u + 4 * v^4 = -2) ∧
    (2 * x^7 + 8 * y^2 + 7 * u + 4 * v^4 = 6^5 / (3^4 * 4^2)) ∧
    (8 * x^7 + 2 * y^2 + 3 * u + 6 * v^4 = -6) ∧
    (5 * x^7 + 7 * y^2 + 7 * u + 8 * v^4 = 8^3 / (2^6 * 4)) ∧
    ((x = -1 ∧ (y = 1 ∨ y = -1) ∧ u = 0 ∧ v = 0) ∨
     (x = 1 ∧ (y = 1 ∨ y = -1) ∧ u = 0 ∧ v = 0)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l739_73919


namespace NUMINAMATH_CALUDE_connect_four_shapes_l739_73932

/-- The number of columns in the Connect Four board -/
def num_columns : ℕ := 7

/-- The number of rows in the Connect Four board -/
def num_rows : ℕ := 8

/-- The number of possible states for each column (0 to 8 checkers) -/
def states_per_column : ℕ := num_rows + 1

/-- The total number of shapes before accounting for symmetry -/
def total_shapes : ℕ := states_per_column ^ num_columns

/-- The number of symmetric shapes -/
def symmetric_shapes : ℕ := states_per_column ^ (num_columns / 2 + 1)

/-- The formula for the number of distinct shapes -/
def distinct_shapes (n : ℕ) : ℕ := 9 * (n * (n + 1) / 2)

/-- The theorem stating that the number of distinct shapes is equal to 9(1+2+...+729) -/
theorem connect_four_shapes :
  ∃ n : ℕ, distinct_shapes n = symmetric_shapes + (total_shapes - symmetric_shapes) / 2 ∧ n = 729 := by
  sorry

end NUMINAMATH_CALUDE_connect_four_shapes_l739_73932


namespace NUMINAMATH_CALUDE_illuminated_cube_surface_area_l739_73916

/-- The area of the illuminated part of a cube's surface when a cylindrical beam of light is directed along its main diagonal -/
theorem illuminated_cube_surface_area 
  (a : ℝ) 
  (ρ : ℝ) 
  (h_a : a = 1 / Real.sqrt 2) 
  (h_ρ : ρ = Real.sqrt (2 - Real.sqrt 3)) : 
  ∃ (area : ℝ), area = (Real.sqrt 3 - 3/2) * (Real.pi + 3) := by
  sorry

end NUMINAMATH_CALUDE_illuminated_cube_surface_area_l739_73916


namespace NUMINAMATH_CALUDE_two_year_growth_l739_73983

/-- Calculates the final value after compound growth --/
def compound_growth (initial_value : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ years

/-- Theorem: After two years of 1/8 annual growth, 32000 becomes 40500 --/
theorem two_year_growth :
  compound_growth 32000 (1/8) 2 = 40500 := by
  sorry

end NUMINAMATH_CALUDE_two_year_growth_l739_73983


namespace NUMINAMATH_CALUDE_pecan_weight_in_mixture_l739_73951

/-- A mixture of pecans and cashews -/
structure NutMixture where
  pecan_price : ℝ
  cashew_price : ℝ
  cashew_weight : ℝ
  total_weight : ℝ

/-- The amount of pecans in the mixture -/
def pecan_weight (m : NutMixture) : ℝ :=
  m.total_weight - m.cashew_weight

/-- Theorem stating the amount of pecans in the specific mixture -/
theorem pecan_weight_in_mixture (m : NutMixture) 
  (h1 : m.pecan_price = 5.60)
  (h2 : m.cashew_price = 3.50)
  (h3 : m.cashew_weight = 2)
  (h4 : m.total_weight = 3.33333333333) :
  pecan_weight m = 1.33333333333 := by
  sorry

#check pecan_weight_in_mixture

end NUMINAMATH_CALUDE_pecan_weight_in_mixture_l739_73951


namespace NUMINAMATH_CALUDE_total_sample_volume_l739_73952

/-- The total sample volume -/
def M : ℝ := 50

/-- The frequency of the first group -/
def freq1 : ℝ := 10

/-- The frequency of the second group -/
def freq2 : ℝ := 0.35

/-- The frequency of the third group -/
def freq3 : ℝ := 0.45

/-- Theorem stating that M is the correct total sample volume given the frequencies -/
theorem total_sample_volume : M = freq1 + freq2 * M + freq3 * M := by
  sorry

end NUMINAMATH_CALUDE_total_sample_volume_l739_73952


namespace NUMINAMATH_CALUDE_tangent_line_equation_l739_73965

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

theorem tangent_line_equation :
  (∃ L : ℝ → ℝ, (L 0 = 0) ∧ (∀ x : ℝ, L x = 2*x) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x| < δ → |f x - L x| < ε * |x|)) ∧
  (∀ x₀ : ℝ, x₀ ≠ 0 →
    (∃ L : ℝ → ℝ, (L x₀ = f x₀) ∧ (∀ x : ℝ, L x = f' x₀ * (x - x₀) + f x₀) ∧
      (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → |f x - L x| < ε * |x - x₀|)) →
    f' x₀ = -1/4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l739_73965


namespace NUMINAMATH_CALUDE_junior_score_l739_73985

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (class_avg : ℝ) (senior_avg : ℝ) (h1 : junior_ratio = 0.2) 
  (h2 : senior_ratio = 0.8) (h3 : junior_ratio + senior_ratio = 1) 
  (h4 : class_avg = 84) (h5 : senior_avg = 82) : 
  (class_avg * n - senior_avg * senior_ratio * n) / (junior_ratio * n) = 92 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l739_73985


namespace NUMINAMATH_CALUDE_factor_sum_l739_73992

theorem factor_sum (R S : ℝ) : 
  (∃ d e : ℝ, x^4 + R*x^2 + S = (x^2 - 3*x + 7) * (x^2 + d*x + e)) →
  R + S = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l739_73992


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l739_73945

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∃ q : ℝ → ℝ, f = λ x => (x - a) * q x + f a) :=
sorry

theorem polynomial_remainder (f : ℝ → ℝ) (h : f = λ x => x^8 + 3) :
  ∃ q : ℝ → ℝ, f = λ x => (x + 1) * q x + 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l739_73945


namespace NUMINAMATH_CALUDE_horner_method_evaluation_l739_73974

def horner_polynomial (x : ℝ) : ℝ := (((((3 * x - 4) * x + 6) * x - 2) * x - 5) * x - 2)

theorem horner_method_evaluation :
  horner_polynomial 5 = 7548 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_evaluation_l739_73974


namespace NUMINAMATH_CALUDE_triangle_side_length_l739_73971

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 9 → b = 2 * Real.sqrt 3 → C = 150 * π / 180 → c = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l739_73971


namespace NUMINAMATH_CALUDE_savings_to_earnings_ratio_l739_73907

/-- Proves that the ratio of combined savings to total earnings is 1:2 --/
theorem savings_to_earnings_ratio
  (kimmie_earnings : ℚ)
  (zahra_earnings : ℚ)
  (combined_savings : ℚ)
  (h1 : kimmie_earnings = 450)
  (h2 : zahra_earnings = kimmie_earnings - kimmie_earnings / 3)
  (h3 : combined_savings = 375) :
  combined_savings / (kimmie_earnings + zahra_earnings) = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_savings_to_earnings_ratio_l739_73907


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l739_73994

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, (x > -1 ∧ x < 11) ↔ (∀ y : ℝ, y > x → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l739_73994


namespace NUMINAMATH_CALUDE_intersection_condition_l739_73998

/-- The parabola equation: x = -3y^2 - 4y + 10 -/
def parabola (x y : ℝ) : Prop := x = -3 * y^2 - 4 * y + 10

/-- The line equation: x = k -/
def line (x k : ℝ) : Prop := x = k

/-- The condition for exactly one intersection point -/
def unique_intersection (k : ℝ) : Prop :=
  ∃! y, parabola k y

theorem intersection_condition (k : ℝ) :
  unique_intersection k ↔ k = 34 / 3 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l739_73998


namespace NUMINAMATH_CALUDE_second_class_size_l739_73935

theorem second_class_size (n : ℕ) (avg_all : ℚ) : 
  n > 0 ∧ 
  (30 : ℚ) * 40 + n * 60 = (30 + n) * avg_all ∧ 
  avg_all = (105 : ℚ) / 2 → 
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_second_class_size_l739_73935


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l739_73953

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {a_n} where a_3 + a_11 = 40, 
    the value of a_6 - a_7 + a_8 is equal to 20 -/
theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 11 = 40) : 
  a 6 - a 7 + a 8 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l739_73953


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l739_73997

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (2*a + b)⁻¹ + (b + 1)⁻¹ = 1) : 
  ∀ x y, x > 0 → y > 0 → (2*x + y)⁻¹ + (y + 1)⁻¹ = 1 → a + 2*b ≤ x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l739_73997
