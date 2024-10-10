import Mathlib

namespace composition_value_l1619_161954

/-- Given two functions g and f, prove that f(g(3)) = 29 -/
theorem composition_value (g f : ℝ → ℝ) 
  (hg : ∀ x, g x = x^2) 
  (hf : ∀ x, f x = 3*x + 2) : 
  f (g 3) = 29 := by
  sorry

end composition_value_l1619_161954


namespace basement_water_pump_time_l1619_161999

/-- Proves that it takes 450 minutes to pump out water from a basement given specific conditions -/
theorem basement_water_pump_time : 
  let basement_length : ℝ := 30
  let basement_width : ℝ := 40
  let water_depth_inches : ℝ := 24
  let num_pumps : ℕ := 4
  let pump_rate : ℝ := 10  -- gallons per minute
  let cubic_foot_to_gallon : ℝ := 7.5
  let inches_per_foot : ℝ := 12

  let water_depth_feet : ℝ := water_depth_inches / inches_per_foot
  let water_volume_cubic_feet : ℝ := basement_length * basement_width * water_depth_feet
  let water_volume_gallons : ℝ := water_volume_cubic_feet * cubic_foot_to_gallon
  let total_pump_rate : ℝ := pump_rate * num_pumps
  let pump_time_minutes : ℝ := water_volume_gallons / total_pump_rate

  pump_time_minutes = 450 := by sorry

end basement_water_pump_time_l1619_161999


namespace solve_equation_l1619_161989

theorem solve_equation (x : ℝ) : (x^3).sqrt = 9 * (81^(1/9)) → x = 9 := by
  sorry

end solve_equation_l1619_161989


namespace not_necessarily_true_inequality_l1619_161919

theorem not_necessarily_true_inequality (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ¬(∀ a b c, c < b ∧ b < a ∧ a * c < 0 → b^2 / c > a^2 / c) :=
by sorry

end not_necessarily_true_inequality_l1619_161919


namespace total_seashells_is_142_l1619_161922

/-- The number of seashells Joan found initially -/
def joans_initial_seashells : ℕ := 79

/-- The number of seashells Mike gave to Joan -/
def mikes_seashells : ℕ := 63

/-- The total number of seashells Joan has -/
def total_seashells : ℕ := joans_initial_seashells + mikes_seashells

/-- Theorem stating that the total number of seashells Joan has is 142 -/
theorem total_seashells_is_142 : total_seashells = 142 := by
  sorry

end total_seashells_is_142_l1619_161922


namespace prob_sum_greater_than_seven_l1619_161965

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of ways to roll a sum of 7 or less with two dice -/
def waysToRollSevenOrLess : ℕ := 21

/-- The probability of rolling a sum greater than 7 with two dice -/
def probSumGreaterThanSeven : ℚ := 5 / 12

/-- Theorem stating that the probability of rolling a sum greater than 7 with two fair six-sided dice is 5/12 -/
theorem prob_sum_greater_than_seven :
  probSumGreaterThanSeven = 1 - (waysToRollSevenOrLess : ℚ) / totalOutcomes := by
  sorry

end prob_sum_greater_than_seven_l1619_161965


namespace floor_sqrt_30_squared_l1619_161914

theorem floor_sqrt_30_squared : ⌊Real.sqrt 30⌋^2 = 25 := by
  sorry

end floor_sqrt_30_squared_l1619_161914


namespace triangle_ratio_l1619_161950

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths
  (area : ℝ)   -- Area

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.area = 8 ∧ t.a = 5 ∧ Real.tan t.B = -4/3

-- Define the theorem
theorem triangle_ratio (t : Triangle) (h : triangle_properties t) :
  (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 5 * Real.sqrt 65 / 4 :=
sorry

end triangle_ratio_l1619_161950


namespace problem_solution_l1619_161937

theorem problem_solution (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) (h4 : a < 1) :
  (0 < a - b) ∧ (a - b < 1) ∧ (a * b < a^2) ∧ (a - 1/b < b - 1/a) := by
  sorry

end problem_solution_l1619_161937


namespace angela_beth_ages_l1619_161905

/-- Angela and Beth's ages problem -/
theorem angela_beth_ages (angela beth : ℕ) 
  (h1 : angela = 4 * beth) 
  (h2 : angela + beth = 55) : 
  angela + 5 = 49 := by sorry

end angela_beth_ages_l1619_161905


namespace difference_of_numbers_l1619_161903

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 104) :
  |x - y| = 4 * Real.sqrt 10 := by
  sorry

end difference_of_numbers_l1619_161903


namespace smallest_valid_number_l1619_161976

def is_valid_divisor (d : ℕ) : Prop :=
  d > 0 ∧ 150 % d = 50 ∧ 55 % d = 5 ∧ 175 % d = 25

def is_greatest_divisor (d : ℕ) : Prop :=
  is_valid_divisor d ∧ ∀ k > d, ¬is_valid_divisor k

theorem smallest_valid_number : ∃ n : ℕ, n > 0 ∧ is_valid_divisor n ∧ 
  ∃ d : ℕ, is_greatest_divisor d ∧ n % d = 5 ∧ 
  ∀ m < n, ¬(is_valid_divisor m ∧ ∃ k : ℕ, is_greatest_divisor k ∧ m % k = 5) :=
sorry

end smallest_valid_number_l1619_161976


namespace mary_snake_observation_l1619_161963

/-- Given the number of breeding balls, snakes per ball, and total snakes observed,
    calculate the number of additional pairs of snakes. -/
def additional_snake_pairs (breeding_balls : ℕ) (snakes_per_ball : ℕ) (total_snakes : ℕ) : ℕ :=
  ((total_snakes - breeding_balls * snakes_per_ball) / 2)

/-- Theorem stating that given 3 breeding balls with 8 snakes each,
    and a total of 36 snakes observed, the number of additional pairs of snakes is 6. -/
theorem mary_snake_observation :
  additional_snake_pairs 3 8 36 = 6 := by
  sorry

end mary_snake_observation_l1619_161963


namespace school_distance_proof_l1619_161907

/-- Represents the time taken to drive to school -/
structure DriveTime where
  rushHour : ℝ  -- Time in hours during rush hour
  holiday : ℝ   -- Time in hours during holiday

/-- Represents the speed of driving to school -/
structure DriveSpeed where
  rushHour : ℝ  -- Speed in miles per hour during rush hour
  holiday : ℝ   -- Speed in miles per hour during holiday

/-- The distance to school in miles -/
def distanceToSchool : ℝ := 10

theorem school_distance_proof (t : DriveTime) (s : DriveSpeed) : distanceToSchool = 10 :=
  by
  have h1 : t.rushHour = 1/2 := by sorry
  have h2 : t.holiday = 1/4 := by sorry
  have h3 : s.holiday = s.rushHour + 20 := by sorry
  have h4 : distanceToSchool = s.rushHour * t.rushHour := by sorry
  have h5 : distanceToSchool = s.holiday * t.holiday := by sorry
  sorry

#check school_distance_proof

end school_distance_proof_l1619_161907


namespace max_distance_complex_l1619_161961

theorem max_distance_complex (z : ℂ) (h : Complex.abs (z + 1 - Complex.I) = 1) :
  ∃ (max_val : ℝ), max_val = 3 ∧ ∀ w, Complex.abs (w + 1 - Complex.I) = 1 →
    Complex.abs (w - 1 - Complex.I) ≤ max_val :=
by sorry

end max_distance_complex_l1619_161961


namespace smallest_with_six_odd_twelve_even_divisors_l1619_161924

/-- Count the number of positive odd integer divisors of a natural number -/
def countOddDivisors (n : ℕ) : ℕ := sorry

/-- Count the number of positive even integer divisors of a natural number -/
def countEvenDivisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number has exactly six positive odd integer divisors and twelve positive even integer divisors -/
def hasSixOddTwelveEvenDivisors (n : ℕ) : Prop :=
  countOddDivisors n = 6 ∧ countEvenDivisors n = 12

theorem smallest_with_six_odd_twelve_even_divisors :
  ∃ (n : ℕ), n > 0 ∧ hasSixOddTwelveEvenDivisors n ∧
  ∀ (m : ℕ), m > 0 → hasSixOddTwelveEvenDivisors m → n ≤ m :=
by
  use 180
  sorry

end smallest_with_six_odd_twelve_even_divisors_l1619_161924


namespace valid_positions_count_l1619_161910

/-- Represents a 6x6 chess board -/
def Board := Fin 6 → Fin 6 → Bool

/-- Represents a position of 4 chips on the board -/
def ChipPosition := Fin 4 → Fin 6 × Fin 6

/-- Checks if four points are collinear -/
def areCollinear (p₁ p₂ p₃ p₄ : Fin 6 × Fin 6) : Bool :=
  sorry

/-- Checks if a square is attacked by at least one chip -/
def isAttacked (board : Board) (pos : ChipPosition) (x y : Fin 6) : Bool :=
  sorry

/-- Checks if all squares are attacked by at least one chip -/
def allSquaresAttacked (board : Board) (pos : ChipPosition) : Bool :=
  sorry

/-- Checks if a chip position is valid (chips are collinear and all squares are attacked) -/
def isValidPosition (board : Board) (pos : ChipPosition) : Bool :=
  sorry

/-- Counts the number of valid chip positions, including rotations and reflections -/
def countValidPositions (board : Board) : Nat :=
  sorry

/-- The main theorem: there are exactly 48 valid chip positions -/
theorem valid_positions_count :
  ∀ (board : Board), countValidPositions board = 48 :=
sorry

end valid_positions_count_l1619_161910


namespace train_distance_l1619_161996

/-- Calculates the distance traveled by a train given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a train traveling at 7 m/s for 6 seconds covers 42 meters -/
theorem train_distance : distance_traveled 7 6 = 42 := by
  sorry

end train_distance_l1619_161996


namespace max_value_of_function_l1619_161953

theorem max_value_of_function :
  (∀ x : ℝ, x > 1 → (2*x^2 + 7*x - 1) / (x^2 + 3*x) ≤ 19/9) ∧
  (∃ x : ℝ, x > 1 ∧ (2*x^2 + 7*x - 1) / (x^2 + 3*x) = 19/9) :=
by sorry

end max_value_of_function_l1619_161953


namespace cos_three_pi_halves_l1619_161977

theorem cos_three_pi_halves : Real.cos (3 * π / 2) = 0 := by
  sorry

end cos_three_pi_halves_l1619_161977


namespace z_purely_imaginary_iff_a_eq_neg_three_l1619_161913

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 + 2*a - 3) (a - 1)

/-- Theorem stating that z is purely imaginary if and only if a = -3 -/
theorem z_purely_imaginary_iff_a_eq_neg_three (a : ℝ) :
  isPurelyImaginary (z a) ↔ a = -3 := by
  sorry

end z_purely_imaginary_iff_a_eq_neg_three_l1619_161913


namespace f_properties_l1619_161978

def f (x : ℝ) : ℝ := |2*x - 1| + 1

theorem f_properties :
  (∀ x, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∀ m, (∃ n, f n ≤ m - f (-n)) → m ≥ 4) := by
  sorry

end f_properties_l1619_161978


namespace parabola_focus_directrix_distance_l1619_161975

/-- For a parabola with equation y² = 8x, the distance from its focus to its directrix is 4. -/
theorem parabola_focus_directrix_distance (x y : ℝ) : 
  y^2 = 8*x → (distance_focus_to_directrix : ℝ) = 4 := by
  sorry

end parabola_focus_directrix_distance_l1619_161975


namespace no_natural_pairs_satisfying_divisibility_l1619_161952

theorem no_natural_pairs_satisfying_divisibility : 
  ¬∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (b^a ∣ a^b - 1) := by
  sorry

end no_natural_pairs_satisfying_divisibility_l1619_161952


namespace sum_of_opposite_sign_integers_l1619_161900

theorem sum_of_opposite_sign_integers (a b : ℤ) : 
  (abs a = 3) → (abs b = 5) → (a * b < 0) → (a + b = -2 ∨ a + b = 2) := by
  sorry

end sum_of_opposite_sign_integers_l1619_161900


namespace regular_polygon_135_degrees_has_8_sides_l1619_161984

/-- A regular polygon with interior angles of 135 degrees has 8 sides -/
theorem regular_polygon_135_degrees_has_8_sides :
  ∀ n : ℕ, 
  n > 2 →
  (180 * (n - 2) : ℝ) = 135 * n →
  n = 8 :=
by
  sorry


end regular_polygon_135_degrees_has_8_sides_l1619_161984


namespace opposite_boys_implies_total_l1619_161985

/-- Represents a circular arrangement of boys -/
structure CircularArrangement where
  num_boys : ℕ
  is_opposite : (a b : ℕ) → Prop

/-- The property that the 5th boy is opposite to the 20th boy -/
def fifth_opposite_twentieth (c : CircularArrangement) : Prop :=
  c.is_opposite 5 20

/-- Theorem stating that if the 5th boy is opposite to the 20th boy,
    then the total number of boys is 33 -/
theorem opposite_boys_implies_total (c : CircularArrangement) :
  fifth_opposite_twentieth c → c.num_boys = 33 := by
  sorry

end opposite_boys_implies_total_l1619_161985


namespace ceiling_of_negative_decimal_l1619_161942

theorem ceiling_of_negative_decimal : ⌈(-3.87 : ℝ)⌉ = -3 := by sorry

end ceiling_of_negative_decimal_l1619_161942


namespace fibonacci_period_correct_l1619_161951

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The period of the Fibonacci sequence modulo 127 -/
def fibonacci_period : ℕ := 256

theorem fibonacci_period_correct :
  fibonacci_period = 256 ∧
  (∀ m : ℕ, m > 0 → m < 256 → ¬(fib m % 127 = 0 ∧ fib (m + 1) % 127 = 1)) ∧
  fib 256 % 127 = 0 ∧
  fib 257 % 127 = 1 := by
  sorry

#check fibonacci_period_correct

end fibonacci_period_correct_l1619_161951


namespace james_coins_value_l1619_161941

/-- Represents the value of James's coins in cents -/
def coin_value : ℕ := 38

/-- Represents the total number of coins James has -/
def total_coins : ℕ := 15

/-- Represents the number of nickels James has -/
def num_nickels : ℕ := 6

/-- Represents the number of pennies James has -/
def num_pennies : ℕ := 9

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

theorem james_coins_value :
  (num_nickels * nickel_value + num_pennies * penny_value = coin_value) ∧
  (num_nickels + num_pennies = total_coins) ∧
  (num_pennies = num_nickels + 2) := by
  sorry

end james_coins_value_l1619_161941


namespace village_population_equality_l1619_161927

/-- The rate at which Village X's population is decreasing per year -/
def decrease_rate : ℕ := sorry

/-- The initial population of Village X -/
def village_x_initial : ℕ := 74000

/-- The initial population of Village Y -/
def village_y_initial : ℕ := 42000

/-- The rate at which Village Y's population is increasing per year -/
def village_y_increase : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 16

theorem village_population_equality :
  village_x_initial - years_until_equal * decrease_rate =
  village_y_initial + years_until_equal * village_y_increase →
  decrease_rate = 1200 := by
sorry

end village_population_equality_l1619_161927


namespace sum_square_diagonals_formula_l1619_161930

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  R : ℝ  -- radius of the circumscribed circle
  OP : ℝ  -- length of segment OP
  h_R_pos : R > 0  -- radius is positive
  h_OP_pos : OP > 0  -- OP is positive
  h_OP_le_2R : OP ≤ 2 * R  -- OP cannot be longer than the diameter

/-- The sum of squares of diagonals of an inscribed quadrilateral -/
def sumSquareDiagonals (q : InscribedQuadrilateral) : ℝ :=
  8 * q.R^2 - 4 * q.OP^2

/-- Theorem: The sum of squares of diagonals of an inscribed quadrilateral
    is equal to 8R^2 - 4OP^2 -/
theorem sum_square_diagonals_formula (q : InscribedQuadrilateral) :
  sumSquareDiagonals q = 8 * q.R^2 - 4 * q.OP^2 := by
  sorry

end sum_square_diagonals_formula_l1619_161930


namespace sum_of_square_areas_l1619_161946

theorem sum_of_square_areas (square1_side : ℝ) (square2_side : ℝ) 
  (h1 : square1_side = 8) (h2 : square2_side = 10) : 
  square1_side^2 + square2_side^2 = 164 := by
  sorry

end sum_of_square_areas_l1619_161946


namespace min_value_of_expression_l1619_161970

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, n - 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y ≥ 2*Real.sqrt 2 + 3) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2/x + 1/y = 2*Real.sqrt 2 + 3) :=
by sorry

end min_value_of_expression_l1619_161970


namespace sandy_total_earnings_l1619_161962

def monday_earnings : ℚ := 12 * 0.5 + 5 * 0.25 + 10 * 0.1
def tuesday_earnings : ℚ := 8 * 0.5 + 15 * 0.25 + 5 * 0.1
def wednesday_earnings : ℚ := 3 * 1 + 4 * 0.5 + 10 * 0.25 + 7 * 0.05
def thursday_earnings : ℚ := 5 * 1 + 6 * 0.5 + 8 * 0.25 + 5 * 0.1 + 12 * 0.05
def friday_earnings : ℚ := 2 * 1 + 7 * 0.5 + 20 * 0.05 + 25 * 0.1

theorem sandy_total_earnings :
  monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings + friday_earnings = 44.45 := by
  sorry

end sandy_total_earnings_l1619_161962


namespace composite_ratio_l1619_161935

def first_seven_composites : List Nat := [4, 6, 8, 9, 10, 12, 14]
def next_seven_composites : List Nat := [15, 16, 18, 20, 21, 22, 24]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem composite_ratio :
  (product_of_list first_seven_composites) / 
  (product_of_list next_seven_composites) = 1 / 264 := by
  sorry

end composite_ratio_l1619_161935


namespace missing_chess_pieces_l1619_161906

/-- The number of pieces in a standard chess set -/
def standard_chess_set_pieces : ℕ := 32

/-- The number of pieces present -/
def present_pieces : ℕ := 24

/-- The number of missing chess pieces -/
def missing_pieces : ℕ := standard_chess_set_pieces - present_pieces

theorem missing_chess_pieces :
  missing_pieces = 8 := by sorry

end missing_chess_pieces_l1619_161906


namespace arithmetic_sequence_eighth_term_l1619_161959

/-- Given an arithmetic sequence with first term 11 and common difference -3,
    prove that its 8th term is -10. -/
theorem arithmetic_sequence_eighth_term :
  let a : ℕ → ℤ := fun n => 11 - 3 * (n - 1)
  a 8 = -10 := by sorry

end arithmetic_sequence_eighth_term_l1619_161959


namespace min_distance_circle_line_l1619_161995

/-- The minimum distance between a point on the circle x² + y² = 4 
    and the line √3y + x + 4√3 = 0 is 2√3 - 2 -/
theorem min_distance_circle_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | Real.sqrt 3 * p.2 + p.1 + 4 * Real.sqrt 3 = 0}
  ∃ (d : ℝ), d = 2 * Real.sqrt 3 - 2 ∧ 
    (∀ p ∈ circle, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ p ∈ circle, ∃ q ∈ line, d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by
  sorry


end min_distance_circle_line_l1619_161995


namespace ratio_product_l1619_161940

theorem ratio_product (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  a * b * c / (d * e * f) = 1 / 2 :=
by sorry

end ratio_product_l1619_161940


namespace expression_evaluation_l1619_161957

theorem expression_evaluation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a = 2 / b) :
  (a - 2 / a) * (b + 2 / b) = a^2 - 4 / a^2 := by
  sorry

end expression_evaluation_l1619_161957


namespace special_function_at_zero_l1619_161997

/-- A function satisfying f(x + y) = f(x) + f(xy) for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f (x * y)

/-- Theorem: If f is a special function, then f(0) = 0 -/
theorem special_function_at_zero (f : ℝ → ℝ) (h : special_function f) : f 0 = 0 := by
  sorry

end special_function_at_zero_l1619_161997


namespace shoes_cost_theorem_l1619_161911

theorem shoes_cost_theorem (cost_first_pair : ℝ) (percentage_increase : ℝ) : 
  cost_first_pair = 22 →
  percentage_increase = 50 →
  let cost_second_pair := cost_first_pair * (1 + percentage_increase / 100)
  let total_cost := cost_first_pair + cost_second_pair
  total_cost = 55 := by
sorry

end shoes_cost_theorem_l1619_161911


namespace last_locker_opened_l1619_161972

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction of the student's movement -/
inductive Direction
| Forward
| Backward

/-- Defines the locker opening process -/
def openLockers (n : Nat) : Nat :=
  sorry

/-- Theorem stating that the last locker opened is number 86 -/
theorem last_locker_opened (n : Nat) (h : n = 512) : openLockers n = 86 := by
  sorry

end last_locker_opened_l1619_161972


namespace CD_length_theorem_l1619_161983

-- Define the line segment CD
def CD : Set (ℝ × ℝ × ℝ) := sorry

-- Define the region within 4 units of CD
def region (CD : Set (ℝ × ℝ × ℝ)) : Set (ℝ × ℝ × ℝ) := sorry

-- Define the volume of a set in 3D space
def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the length of a line segment
def length (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem CD_length_theorem (CD : Set (ℝ × ℝ × ℝ)) :
  volume (region CD) = 448 * Real.pi → length CD = 68 / 3 := by
  sorry

end CD_length_theorem_l1619_161983


namespace rowing_downstream_speed_l1619_161991

/-- The speed of a man rowing downstream, given his upstream speed and still water speed -/
theorem rowing_downstream_speed (upstream_speed still_water_speed : ℝ) :
  upstream_speed = 27 →
  still_water_speed = 31 →
  still_water_speed + (still_water_speed - upstream_speed) = 35 := by
sorry

end rowing_downstream_speed_l1619_161991


namespace abs_value_equivalence_l1619_161971

theorem abs_value_equivalence (x : ℝ) : -1 < x ∧ x < 1 ↔ |x| < 1 := by sorry

end abs_value_equivalence_l1619_161971


namespace quadratic_comparison_l1619_161928

/-- Given two quadratic functions A and B, prove that B can be expressed in terms of x
    and that A is always greater than B for all real x. -/
theorem quadratic_comparison (x : ℝ) : 
  let A := 3 * x^2 - 2 * x + 1
  let B := 2 * x^2 - x - 3
  (A + B = 5 * x^2 - 4 * x - 2) ∧ (A > B) := by sorry

end quadratic_comparison_l1619_161928


namespace book_selection_combinations_l1619_161925

theorem book_selection_combinations :
  let mystery_books : ℕ := 3
  let fantasy_books : ℕ := 4
  let biography_books : ℕ := 3
  let total_combinations := mystery_books * fantasy_books * biography_books
  total_combinations = 36 := by
  sorry

end book_selection_combinations_l1619_161925


namespace geometric_sequence_between_9_and_243_l1619_161912

theorem geometric_sequence_between_9_and_243 :
  ∃ (a b : ℝ), 9 < a ∧ a < b ∧ b < 243 ∧
  (9 / a = a / b) ∧ (a / b = b / 243) ∧
  a = 27 ∧ b = 81 := by
sorry

end geometric_sequence_between_9_and_243_l1619_161912


namespace max_value_theorem_l1619_161994

theorem max_value_theorem (x y : ℝ) (hx : |x - 1| ≤ 1) (hy : |y - 2| ≤ 1) : 
  |x - 2*y + 1| ≤ 5 := by
sorry

end max_value_theorem_l1619_161994


namespace power_calculation_l1619_161920

theorem power_calculation : (8^8 / 8^5) * 2^10 * 2^3 = 2^22 := by sorry

end power_calculation_l1619_161920


namespace cookie_price_calculation_l1619_161945

def cupcake_price : ℚ := 2
def cupcake_quantity : ℕ := 5
def doughnut_price : ℚ := 1
def doughnut_quantity : ℕ := 6
def pie_slice_price : ℚ := 2
def pie_slice_quantity : ℕ := 4
def cookie_quantity : ℕ := 15
def total_spent : ℚ := 33

theorem cookie_price_calculation :
  ∃ (cookie_price : ℚ),
    cookie_price * cookie_quantity +
    cupcake_price * cupcake_quantity +
    doughnut_price * doughnut_quantity +
    pie_slice_price * pie_slice_quantity = total_spent ∧
    cookie_price = 0.60 := by
  sorry

end cookie_price_calculation_l1619_161945


namespace complex_multiplication_simplification_l1619_161992

theorem complex_multiplication_simplification :
  ((-3 - 2 * Complex.I) - (1 + 4 * Complex.I)) * (2 - 3 * Complex.I) = 10 := by
  sorry

end complex_multiplication_simplification_l1619_161992


namespace exam_results_l1619_161981

theorem exam_results (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_hindi = 0.25 * total)
  (h2 : failed_both = 0.4 * total)
  (h3 : passed_both = 0.8 * total) :
  ∃ failed_english : ℝ, failed_english = 0.35 * total :=
by
  sorry

end exam_results_l1619_161981


namespace fraction_equation_sum_l1619_161982

theorem fraction_equation_sum (A B : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 5 →
    (B * x - 17) / (x^2 - 9*x + 20) = A / (x - 4) + 5 / (x - 5)) →
  A + B = 9/5 := by
sorry

end fraction_equation_sum_l1619_161982


namespace max_3k_value_l1619_161915

theorem max_3k_value (k : ℝ) : 
  (∃ x : ℝ, Real.sqrt (x^2 - k) + 2 * Real.sqrt (x^3 - 1) = x) →
  k ≥ 0 →
  k < 2 →
  ∃ m : ℝ, m = 4 ∧ ∀ k' : ℝ, 
    (∃ x : ℝ, Real.sqrt (x'^2 - k') + 2 * Real.sqrt (x'^3 - 1) = x') →
    k' ≥ 0 →
    k' < 2 →
    3 * k' ≤ m :=
by sorry

end max_3k_value_l1619_161915


namespace modular_inverse_11_mod_1105_l1619_161988

theorem modular_inverse_11_mod_1105 :
  let m : ℕ := 1105
  let a : ℕ := 11
  let b : ℕ := 201
  m = 5 * 13 * 17 →
  (a * b) % m = 1 :=
by
  sorry

end modular_inverse_11_mod_1105_l1619_161988


namespace max_trees_bucked_l1619_161917

/-- Represents the energy and tree-bucking strategy over time -/
structure BuckingStrategy where
  restTime : ℕ
  initialEnergy : ℕ
  timePeriod : ℕ

/-- Calculates the total number of trees bucked given a strategy -/
def totalTreesBucked (s : BuckingStrategy) : ℕ :=
  let buckingTime := s.timePeriod - s.restTime
  let finalEnergy := s.initialEnergy + s.restTime - buckingTime + 1
  (buckingTime * (s.initialEnergy + s.restTime + finalEnergy)) / 2

/-- The main theorem to prove -/
theorem max_trees_bucked :
  ∃ (s : BuckingStrategy),
    s.initialEnergy = 100 ∧
    s.timePeriod = 60 ∧
    (∀ (t : BuckingStrategy),
      t.initialEnergy = 100 ∧
      t.timePeriod = 60 →
      totalTreesBucked t ≤ totalTreesBucked s) ∧
    totalTreesBucked s = 4293 := by
  sorry


end max_trees_bucked_l1619_161917


namespace intersection_M_complement_N_l1619_161986

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 10}
def N : Set ℝ := {x | x < -4/3 ∨ x > 3}

-- State the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 0 3 := by sorry

end intersection_M_complement_N_l1619_161986


namespace minimum_race_distance_l1619_161947

/-- The minimum distance a runner must travel in a race with given conditions -/
theorem minimum_race_distance (wall_length : ℝ) (dist_A_to_wall : ℝ) (dist_wall_to_B : ℝ) :
  wall_length = 1600 →
  dist_A_to_wall = 600 →
  dist_wall_to_B = 800 →
  round (Real.sqrt ((wall_length ^ 2) + (dist_A_to_wall + dist_wall_to_B) ^ 2)) = 2127 :=
by sorry

end minimum_race_distance_l1619_161947


namespace exists_return_steps_power_of_two_case_power_of_two_plus_one_case_l1619_161939

/-- Represents the state of a lamp (ON or OFF) -/
inductive LampState
| ON
| OFF

/-- Represents the configuration of n lamps -/
def LampConfig (n : ℕ) := Fin n → LampState

/-- Performs a single step of the lamp changing process -/
def step (n : ℕ) (config : LampConfig n) : LampConfig n :=
  sorry

/-- Checks if all lamps in the configuration are ON -/
def allOn (n : ℕ) (config : LampConfig n) : Prop :=
  sorry

/-- The initial configuration with all lamps ON -/
def initialConfig (n : ℕ) : LampConfig n :=
  sorry

/-- Theorem stating the existence of M(n) for any n > 1 -/
theorem exists_return_steps (n : ℕ) (h : n > 1) :
  ∃ M : ℕ, M > 0 ∧ allOn n ((step n)^[M] (initialConfig n)) :=
  sorry

/-- Theorem for the case when n is a power of 2 -/
theorem power_of_two_case (k : ℕ) :
  let n := 2^k
  allOn n ((step n)^[n^2 - 1] (initialConfig n)) :=
  sorry

/-- Theorem for the case when n is one more than a power of 2 -/
theorem power_of_two_plus_one_case (k : ℕ) :
  let n := 2^k + 1
  allOn n ((step n)^[n^2 - n + 1] (initialConfig n)) :=
  sorry

end exists_return_steps_power_of_two_case_power_of_two_plus_one_case_l1619_161939


namespace inequality_range_l1619_161944

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x) ↔ 
  (-2 < m ∧ m ≤ 2) :=
by sorry

end inequality_range_l1619_161944


namespace hyperbola_y_relationship_l1619_161921

theorem hyperbola_y_relationship (k : ℝ) (y₁ y₂ : ℝ) (h_k_pos : k > 0) 
  (h_A : y₁ = k / 2) (h_B : y₂ = k / 3) : y₁ > y₂ := by
  sorry

end hyperbola_y_relationship_l1619_161921


namespace identify_tricksters_l1619_161943

/-- Represents an inhabitant of the village -/
inductive Inhabitant
| Knight
| Trickster

/-- The village with its inhabitants -/
structure Village where
  inhabitants : Fin 65 → Inhabitant
  trickster_count : Nat
  knight_count : Nat
  trickster_count_eq : trickster_count = 2
  knight_count_eq : knight_count = 63
  total_count_eq : trickster_count + knight_count = 65

/-- A question asked to an inhabitant about a group of inhabitants -/
def Question := List (Fin 65) → Bool

/-- The result of asking questions to identify tricksters -/
structure IdentificationResult where
  questions_asked : Nat
  tricksters_found : List (Fin 65)
  all_tricksters_found : tricksters_found.length = 2

/-- The main theorem stating that tricksters can be identified with no more than 30 questions -/
theorem identify_tricksters (v : Village) : 
  ∃ (strategy : List Question), 
    ∃ (result : IdentificationResult), 
      result.questions_asked ≤ 30 ∧ 
      (∀ i : Fin 65, v.inhabitants i = Inhabitant.Trickster ↔ i ∈ result.tricksters_found) :=
sorry

end identify_tricksters_l1619_161943


namespace zoo_bird_difference_l1619_161956

/-- Proves that in a zoo with 450 birds and where the number of birds is 5 times
    the number of all other animals, there are 360 more birds than non-bird animals. -/
theorem zoo_bird_difference (total_birds : ℕ) (bird_ratio : ℕ) 
    (h1 : total_birds = 450)
    (h2 : bird_ratio = 5)
    (h3 : total_birds = bird_ratio * (total_birds / bird_ratio)) :
  total_birds - (total_birds / bird_ratio) = 360 := by
  sorry

#eval 450 - (450 / 5)  -- This should evaluate to 360

end zoo_bird_difference_l1619_161956


namespace sqrt_product_equality_l1619_161990

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l1619_161990


namespace cubic_difference_l1619_161909

theorem cubic_difference (x y : ℝ) 
  (h1 : x + y - x * y = 155) 
  (h2 : x^2 + y^2 = 325) : 
  |x^3 - y^3| = 4375 := by
sorry

end cubic_difference_l1619_161909


namespace passengers_from_other_continents_l1619_161966

theorem passengers_from_other_continents 
  (total : ℕ) 
  (north_america : ℚ)
  (europe : ℚ)
  (africa : ℚ)
  (asia : ℚ)
  (h1 : total = 108)
  (h2 : north_america = 1 / 12)
  (h3 : europe = 1 / 4)
  (h4 : africa = 1 / 9)
  (h5 : asia = 1 / 6)
  : ℕ := by
  sorry

end passengers_from_other_continents_l1619_161966


namespace f_odd_implies_a_zero_necessary_not_sufficient_l1619_161960

noncomputable def f (a x : ℝ) : ℝ := 1 / (x - 1) + a / (x + a - 1) + 1 / (x + 1)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

theorem f_odd_implies_a_zero_necessary_not_sufficient :
  (∃ a : ℝ, is_odd_function (f a)) ∧
  (∀ a : ℝ, is_odd_function (f a) → a = 0 ∨ a = 1) ∧
  (∃ a : ℝ, a ≠ 0 ∧ is_odd_function (f a)) :=
sorry

end f_odd_implies_a_zero_necessary_not_sufficient_l1619_161960


namespace race_distance_difference_l1619_161916

theorem race_distance_difference (lingling_distance mingming_distance : ℝ) 
  (h1 : lingling_distance = 380.5)
  (h2 : mingming_distance = 405.9) : 
  mingming_distance - lingling_distance = 25.4 := by
  sorry

end race_distance_difference_l1619_161916


namespace remaining_distance_to_nyc_l1619_161974

/-- Richard's journey from Cincinnati to New York City -/
def richards_journey (total_distance first_day second_day third_day : ℕ) : Prop :=
  let distance_walked := first_day + second_day + third_day
  total_distance - distance_walked = 36

theorem remaining_distance_to_nyc :
  richards_journey 70 20 4 10 := by sorry

end remaining_distance_to_nyc_l1619_161974


namespace sum_of_x_and_y_on_circle_l1619_161931

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 44) : x + y = 2 := by
  sorry

end sum_of_x_and_y_on_circle_l1619_161931


namespace binomial_coefficient_condition_l1619_161936

theorem binomial_coefficient_condition (a : ℚ) : 
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * a^(7-k) * 1^k) = (a + 1)^7 ∧ 
  (Nat.choose 7 6) * a * 1^6 = 1 → 
  a = 1/7 := by
  sorry

end binomial_coefficient_condition_l1619_161936


namespace smallest_board_is_7x7_l1619_161904

/-- Represents a ship in the Battleship game -/
structure Ship :=
  (length : Nat)

/-- The complete set of ships for the Battleship game -/
def battleshipSet : List Ship := [
  ⟨4⟩,  -- One 1x4 ship
  ⟨3⟩, ⟨3⟩,  -- Two 1x3 ships
  ⟨2⟩, ⟨2⟩, ⟨2⟩,  -- Three 1x2 ships
  ⟨1⟩, ⟨1⟩, ⟨1⟩, ⟨1⟩  -- Four 1x1 ships
]

/-- Represents a square board -/
structure Board :=
  (size : Nat)

/-- Checks if a given board can fit all ships without touching -/
def canFitShips (board : Board) (ships : List Ship) : Prop :=
  sorry

/-- Theorem stating that 7x7 is the smallest square board that can fit all ships -/
theorem smallest_board_is_7x7 :
  (∀ b : Board, b.size < 7 → ¬(canFitShips b battleshipSet)) ∧
  (canFitShips ⟨7⟩ battleshipSet) :=
sorry

end smallest_board_is_7x7_l1619_161904


namespace inequality_proof_l1619_161964

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b ≤ 4) :
  1 / a + 1 / b ≥ 1 := by
sorry

end inequality_proof_l1619_161964


namespace line_segment_lattice_points_l1619_161932

/-- The number of lattice points on a line segment --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem stating that the number of lattice points on the given line segment is 3 --/
theorem line_segment_lattice_points :
  latticePointCount 5 23 47 297 = 3 := by sorry

end line_segment_lattice_points_l1619_161932


namespace rectangle_y_value_l1619_161949

/-- Given a rectangle with vertices (-2, y), (8, y), (-2, 3), and (8, 3),
    if the area is 90 square units and y is positive, then y = 12. -/
theorem rectangle_y_value (y : ℝ) : y > 0 → (8 - (-2)) * (y - 3) = 90 → y = 12 := by
  sorry

end rectangle_y_value_l1619_161949


namespace white_ball_from_first_urn_l1619_161987

/-- Represents an urn with black and white balls -/
structure Urn :=
  (black : ℕ)
  (white : ℕ)

/-- The probability of choosing an urn -/
def urn_prob : ℚ := 1/2

/-- Calculate the probability of drawing a white ball from an urn -/
def white_ball_prob (u : Urn) : ℚ :=
  u.white / (u.black + u.white)

/-- The theorem to prove -/
theorem white_ball_from_first_urn 
  (urn1 : Urn)
  (urn2 : Urn)
  (h1 : urn1 = ⟨3, 7⟩)
  (h2 : urn2 = ⟨4, 6⟩)
  : (urn_prob * white_ball_prob urn1) / 
    (urn_prob * white_ball_prob urn1 + urn_prob * white_ball_prob urn2) = 7/13 :=
sorry

end white_ball_from_first_urn_l1619_161987


namespace population_trend_l1619_161979

theorem population_trend (P₀ k : ℝ) (h₁ : P₀ > 0) (h₂ : -1 < k) (h₃ : k < 0) :
  ∀ n : ℕ, P₀ * (1 + k) ^ (n + 1) < P₀ * (1 + k) ^ n :=
by sorry

end population_trend_l1619_161979


namespace product_nonzero_l1619_161926

theorem product_nonzero (n : ℤ) : n ≠ 5 → n ≠ 17 → n ≠ 257 → (n - 5) * (n - 17) * (n - 257) ≠ 0 := by
  sorry

end product_nonzero_l1619_161926


namespace complex_modulus_sqrt2_over_2_l1619_161908

theorem complex_modulus_sqrt2_over_2 (z : ℂ) (h : z * Complex.I / (z - Complex.I) = 1) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_sqrt2_over_2_l1619_161908


namespace total_paint_is_47_l1619_161933

/-- Calculates the total amount of paint used for all canvases --/
def total_paint_used (extra_large_count : ℕ) (large_count : ℕ) (medium_count : ℕ) (small_count : ℕ) 
  (extra_large_paint : ℕ) (large_paint : ℕ) (medium_paint : ℕ) (small_paint : ℕ) : ℕ :=
  extra_large_count * extra_large_paint + 
  large_count * large_paint + 
  medium_count * medium_paint + 
  small_count * small_paint

/-- Theorem stating that the total paint used is 47 ounces --/
theorem total_paint_is_47 : 
  total_paint_used 3 5 6 8 4 3 2 1 = 47 := by
  sorry

end total_paint_is_47_l1619_161933


namespace product_of_odd_numbers_not_always_composite_l1619_161934

theorem product_of_odd_numbers_not_always_composite :
  ∃ (a b : ℕ), 
    (a % 2 = 1) ∧ 
    (b % 2 = 1) ∧ 
    ¬(∃ (x : ℕ), 1 < x ∧ x < a * b ∧ (a * b) % x = 0) :=
by sorry

end product_of_odd_numbers_not_always_composite_l1619_161934


namespace winnie_balloon_distribution_l1619_161973

theorem winnie_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 272) (h2 : num_friends = 5) :
  total_balloons % num_friends = 2 := by
  sorry

end winnie_balloon_distribution_l1619_161973


namespace pentagram_impossible_l1619_161967

/-- Represents a pentagram arrangement of numbers -/
structure PentagramArrangement :=
  (numbers : Fin 10 → ℕ)
  (is_permutation : Function.Injective numbers)
  (valid_range : ∀ i, numbers i ∈ Finset.range 11 \ {0})

/-- Represents a line in the pentagram -/
inductive PentagramLine
  | Line1 | Line2 | Line3 | Line4 | Line5

/-- Get the four positions on a given line -/
def line_positions (l : PentagramLine) : Fin 4 → Fin 10 :=
  sorry  -- Implementation details omitted

/-- The sum of numbers on a given line -/
def line_sum (arr : PentagramArrangement) (l : PentagramLine) : ℕ :=
  (Finset.range 4).sum (λ i => arr.numbers (line_positions l i))

/-- Statement: It's impossible to arrange numbers 1 to 10 in a pentagram
    such that all line sums are equal -/
theorem pentagram_impossible : ¬ ∃ (arr : PentagramArrangement),
  ∀ (l1 l2 : PentagramLine), line_sum arr l1 = line_sum arr l2 :=
sorry

end pentagram_impossible_l1619_161967


namespace base3_to_decimal_21201_l1619_161918

/-- Converts a list of digits in base 3 to a decimal number -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number -/
def base3Number : List Nat := [1, 0, 2, 1, 2]

/-- Theorem stating that the conversion of 21201 in base 3 to decimal is 208 -/
theorem base3_to_decimal_21201 :
  base3ToDecimal base3Number = 208 := by
  sorry

#eval base3ToDecimal base3Number

end base3_to_decimal_21201_l1619_161918


namespace min_valid_configuration_l1619_161929

/-- Represents a configuration of two piles of bricks -/
structure BrickPiles where
  first : ℕ
  second : ℕ

/-- Checks if moving 100 bricks from the first pile to the second makes the second pile twice as large as the first -/
def satisfiesFirstCondition (piles : BrickPiles) : Prop :=
  2 * (piles.first - 100) = piles.second + 100

/-- Checks if there exists a number of bricks that can be moved from the second pile to the first to make the first pile six times as large as the second -/
def satisfiesSecondCondition (piles : BrickPiles) : Prop :=
  ∃ z : ℕ, piles.first + z = 6 * (piles.second - z)

/-- Checks if a given configuration satisfies both conditions -/
def isValidConfiguration (piles : BrickPiles) : Prop :=
  satisfiesFirstCondition piles ∧ satisfiesSecondCondition piles

/-- The main theorem stating the minimum valid configuration -/
theorem min_valid_configuration :
  ∀ piles : BrickPiles, isValidConfiguration piles →
  piles.first ≥ 170 ∧
  (piles.first = 170 → piles.second = 40) :=
by sorry

#check min_valid_configuration

end min_valid_configuration_l1619_161929


namespace a_equals_one_l1619_161901

theorem a_equals_one (a : ℝ) : 
  ((a - Complex.I) ^ 2 * Complex.I).re > 0 → a = 1 := by
  sorry

end a_equals_one_l1619_161901


namespace quadratic_two_distinct_roots_l1619_161938

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + m = 0 ∧ y^2 + 2*y + m = 0) → m < 1 := by
  sorry

end quadratic_two_distinct_roots_l1619_161938


namespace second_month_sale_l1619_161980

/-- Represents the sales data for a grocer over six months -/
structure GrocerSales where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Theorem: Given the sales for five months and the average sale,
    prove that the sale in the second month was 7000 -/
theorem second_month_sale
  (sales : GrocerSales)
  (h1 : sales.month1 = 6400)
  (h3 : sales.month3 = 6800)
  (h4 : sales.month4 = 7200)
  (h5 : sales.month5 = 6500)
  (h6 : sales.month6 = 5100)
  (avg : (sales.month1 + sales.month2 + sales.month3 + sales.month4 + sales.month5 + sales.month6) / 6 = 6500) :
  sales.month2 = 7000 := by
  sorry

end second_month_sale_l1619_161980


namespace fraction_sum_theorem_l1619_161969

theorem fraction_sum_theorem (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_sum : a + b + c + d = 100)
  (h_frac_sum : a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c) = 95) :
  1 / (b + c + d) + 1 / (a + c + d) + 1 / (a + b + d) + 1 / (a + b + c) = 99 / 100 := by
  sorry

end fraction_sum_theorem_l1619_161969


namespace jakes_weight_l1619_161923

/-- Represents the weights of Jake, his sister, and Mark -/
structure SiblingWeights where
  jake : ℝ
  sister : ℝ
  mark : ℝ

/-- The conditions of the problem -/
def weightConditions (w : SiblingWeights) : Prop :=
  w.jake - 12 = 2 * (w.sister + 4) ∧
  w.mark = w.jake + w.sister + 50 ∧
  w.jake + w.sister + w.mark = 385

/-- The theorem stating Jake's current weight -/
theorem jakes_weight (w : SiblingWeights) :
  weightConditions w → w.jake = 118 := by
  sorry

#check jakes_weight

end jakes_weight_l1619_161923


namespace abc_perfect_cube_l1619_161955

theorem abc_perfect_cube (a b c : ℤ) (h : (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = 3) :
  ∃ (n : ℤ), a * b * c = n^3 := by
sorry

end abc_perfect_cube_l1619_161955


namespace custom_op_difference_l1619_161948

-- Define the custom operator @
def customOp (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem custom_op_difference : (customOp 7 2) - (customOp 2 7) = -20 := by
  sorry

end custom_op_difference_l1619_161948


namespace alice_exceeds_quota_by_655_l1619_161968

/-- Represents the sales information for a shoe brand -/
structure ShoeBrand where
  name : String
  cost : Nat
  maxSales : Nat
  actualSales : Nat

/-- Calculates the total sales for a given shoe brand -/
def calculateSales (brand : ShoeBrand) : Nat :=
  brand.cost * brand.actualSales

/-- Calculates the total sales across all shoe brands -/
def totalSales (brands : List ShoeBrand) : Nat :=
  brands.foldl (fun acc brand => acc + calculateSales brand) 0

/-- The main theorem stating that Alice exceeds her quota by $655 -/
theorem alice_exceeds_quota_by_655 (brands : List ShoeBrand) (quota : Nat) : 
  brands = [
    { name := "Adidas", cost := 45, maxSales := 15, actualSales := 10 },
    { name := "Nike", cost := 60, maxSales := 12, actualSales := 12 },
    { name := "Reeboks", cost := 35, maxSales := 20, actualSales := 15 },
    { name := "Puma", cost := 50, maxSales := 10, actualSales := 8 },
    { name := "Converse", cost := 40, maxSales := 18, actualSales := 14 }
  ] ∧ quota = 2000 →
  totalSales brands - quota = 655 := by
  sorry

end alice_exceeds_quota_by_655_l1619_161968


namespace sqrt_expression_sum_l1619_161998

theorem sqrt_expression_sum (a b c : ℤ) : 
  (64 + 24 * Real.sqrt 3 : ℝ) = (a + b * Real.sqrt c)^2 →
  c > 0 →
  (∀ n : ℕ, n > 1 → ¬(∃ m : ℕ, c = n^2 * m)) →
  a + b + c = 11 := by
  sorry

end sqrt_expression_sum_l1619_161998


namespace final_marble_difference_l1619_161993

/- Define the initial difference in marbles between Ed and Doug -/
def initial_difference : ℕ := 30

/- Define the number of marbles Ed lost -/
def marbles_lost : ℕ := 21

/- Define Ed's final number of marbles -/
def ed_final_marbles : ℕ := 91

/- Define Doug's number of marbles (which remains constant) -/
def doug_marbles : ℕ := ed_final_marbles + marbles_lost - initial_difference

/- Theorem stating the final difference in marbles -/
theorem final_marble_difference :
  ed_final_marbles - doug_marbles = 9 :=
by
  sorry

end final_marble_difference_l1619_161993


namespace quadratic_function_ordering_l1619_161958

theorem quadratic_function_ordering (m y₁ y₂ y₃ : ℝ) : 
  m < -2 →
  y₁ = (m - 1)^2 - 2*(m - 1) →
  y₂ = m^2 - 2*m →
  y₃ = (m + 1)^2 - 2*(m + 1) →
  y₃ < y₂ ∧ y₂ < y₁ :=
by sorry

end quadratic_function_ordering_l1619_161958


namespace cricket_run_rate_l1619_161902

/-- Calculates the required run rate for the remaining overs in a cricket game -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_scored := first_run_rate * first_overs
  let runs_needed := target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate : required_run_rate 50 10 (32/10) 282 = 25/4 := by
  sorry

#eval required_run_rate 50 10 (32/10) 282

end cricket_run_rate_l1619_161902
