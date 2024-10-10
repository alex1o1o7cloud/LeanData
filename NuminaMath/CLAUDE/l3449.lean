import Mathlib

namespace subset_condition_implies_a_values_l3449_344950

theorem subset_condition_implies_a_values (a : ℝ) : 
  let A : Set ℝ := {x | x^2 = 1}
  let B : Set ℝ := {x | a * x = 1}
  B ⊆ A → a ∈ ({-1, 0, 1} : Set ℝ) := by
sorry

end subset_condition_implies_a_values_l3449_344950


namespace average_book_cost_l3449_344961

/-- Given that Fred had $236 initially, bought 6 books, and had $14 left after the purchase,
    prove that the average cost of each book is $37. -/
theorem average_book_cost (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) :
  initial_amount = 236 →
  num_books = 6 →
  remaining_amount = 14 →
  (initial_amount - remaining_amount) / num_books = 37 :=
by sorry

end average_book_cost_l3449_344961


namespace max_consecutive_semiprimes_l3449_344945

def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def IsSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, IsPrime p ∧ IsPrime q ∧ p ≠ q ∧ n = p + q

def ConsecutiveSemiPrimes (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → IsSemiPrime (k + 1)

theorem max_consecutive_semiprimes :
  ∀ n : ℕ, ConsecutiveSemiPrimes n → n ≤ 5 :=
sorry

end max_consecutive_semiprimes_l3449_344945


namespace batsman_average_increase_l3449_344944

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average for a batsman -/
def averageIncrease (prevInnings : ℕ) (prevTotalRuns : ℕ) (newScore : ℕ) : ℚ :=
  let newAverage := (prevTotalRuns + newScore) / (prevInnings + 1)
  let prevAverage := prevTotalRuns / prevInnings
  newAverage - prevAverage

/-- Theorem: The batsman's average increased by 3 runs -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 12 →
    b.average = 47 →
    averageIncrease 11 (11 * (b.totalRuns / 11)) 80 = 3 := by
  sorry

end batsman_average_increase_l3449_344944


namespace concert_ticket_purchase_daria_concert_money_l3449_344958

/-- Calculates the additional money needed to purchase concert tickets --/
theorem concert_ticket_purchase (num_tickets : ℕ) (original_price : ℚ) 
  (discount_percent : ℚ) (gift_card : ℚ) (current_money : ℚ) : ℚ :=
  let discounted_price := original_price * (1 - discount_percent / 100)
  let total_cost := num_tickets * discounted_price
  let after_gift_card := total_cost - gift_card
  after_gift_card - current_money

/-- Proves that Daria needs to earn $85 more for the concert tickets --/
theorem daria_concert_money : 
  concert_ticket_purchase 4 90 10 50 189 = 85 := by
  sorry

end concert_ticket_purchase_daria_concert_money_l3449_344958


namespace percentage_of_percentage_l3449_344998

theorem percentage_of_percentage (amount : ℝ) : (5 / 100) * ((25 / 100) * amount) = 20 :=
by
  -- Proof goes here
  sorry

end percentage_of_percentage_l3449_344998


namespace order_of_values_l3449_344924

theorem order_of_values : 
  let a := Real.sin (80 * π / 180)
  let b := (1/2)⁻¹
  let c := Real.log 3 / Real.log (1/2)
  b > a ∧ a > c := by
  sorry

end order_of_values_l3449_344924


namespace magnitude_of_z_l3449_344966

open Complex

theorem magnitude_of_z (z : ℂ) (h : (1 + 2*I) / z = 2 - I) : abs z = 1 := by
  sorry

end magnitude_of_z_l3449_344966


namespace complex_magnitude_proof_l3449_344973

theorem complex_magnitude_proof (z : ℂ) :
  (Complex.arg z = Real.pi / 3) →
  (Complex.abs (z - 1) ^ 2 = Complex.abs z * Complex.abs (z - 2)) →
  (Complex.abs z = Real.sqrt 2 + 1 ∨ Complex.abs z = Real.sqrt 2 - 1) := by
  sorry

end complex_magnitude_proof_l3449_344973


namespace difference_of_squares_form_l3449_344937

theorem difference_of_squares_form (x y : ℝ) :
  ∃ a b : ℝ, (-x + y) * (x + y) = -(a + b) * (a - b) := by sorry

end difference_of_squares_form_l3449_344937


namespace key_west_turtle_race_time_l3449_344933

/-- Represents the race times of turtles in the Key West Turtle Race -/
structure TurtleRaceTimes where
  greta : Float
  george : Float
  gloria : Float
  gary : Float
  gwen : Float

/-- Calculates the total race time for all turtles -/
def total_race_time (times : TurtleRaceTimes) : Float :=
  times.greta + times.george + times.gloria + times.gary + times.gwen

/-- Theorem stating the total race time for the given conditions -/
theorem key_west_turtle_race_time : ∃ (times : TurtleRaceTimes),
  times.greta = 6.5 ∧
  times.george = times.greta - 1.5 ∧
  times.gloria = 2 * times.george ∧
  times.gary = times.george + times.gloria + 1.75 ∧
  times.gwen = (times.greta + times.george) * 0.6 ∧
  total_race_time times = 45.15 := by
  sorry

end key_west_turtle_race_time_l3449_344933


namespace fixed_point_of_exponential_function_l3449_344985

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 3
  f 2 = 4 := by sorry

end fixed_point_of_exponential_function_l3449_344985


namespace clock_adjustment_l3449_344959

/-- Represents the number of minutes lost per day by the clock -/
def minutes_lost_per_day : ℕ := 3

/-- Represents the number of days between March 15 1 P.M. and March 22 9 A.M. -/
def days_elapsed : ℕ := 7

/-- Represents the total number of minutes lost by the clock -/
def total_minutes_lost : ℕ := minutes_lost_per_day * days_elapsed

theorem clock_adjustment :
  total_minutes_lost = 21 := by sorry

end clock_adjustment_l3449_344959


namespace solutions_count_l3449_344990

def count_solutions (n : ℕ) : ℕ := 4 * n

theorem solutions_count (n : ℕ) : 
  (count_solutions 1 = 4) → 
  (count_solutions 2 = 8) → 
  (count_solutions 3 = 12) → 
  (count_solutions 20 = 80) :=
by sorry

end solutions_count_l3449_344990


namespace shaded_area_constant_l3449_344907

/-- The total area of two triangles formed by joining the ends of two 1 cm segments 
    on opposite sides of an 8 cm square is always 4 cm², regardless of the segments' positions. -/
theorem shaded_area_constant (h : ℝ) (h_range : 0 ≤ h ∧ h ≤ 8) : 
  (1/2 * 1 * h) + (1/2 * 1 * (8 - h)) = 4 := by sorry

end shaded_area_constant_l3449_344907


namespace girls_share_l3449_344982

theorem girls_share (total_amount : ℕ) (total_children : ℕ) (boys_share : ℕ) (num_boys : ℕ)
  (h1 : total_amount = 460)
  (h2 : total_children = 41)
  (h3 : boys_share = 12)
  (h4 : num_boys = 33) :
  (total_amount - num_boys * boys_share) / (total_children - num_boys) = 8 := by
  sorry

end girls_share_l3449_344982


namespace divide_algebraic_expression_l3449_344956

theorem divide_algebraic_expression (a b c : ℝ) (h : b ≠ 0) :
  4 * a^2 * b^2 * c / (-2 * a * b^2) = -2 * a * c := by
  sorry

end divide_algebraic_expression_l3449_344956


namespace taxi_distribution_eq_14_l3449_344962

/-- The number of ways to distribute 4 people into 2 taxis with at least one person in each taxi -/
def taxi_distribution : ℕ :=
  2^4 - 2

/-- Theorem stating that the number of ways to distribute 4 people into 2 taxis
    with at least one person in each taxi is equal to 14 -/
theorem taxi_distribution_eq_14 : taxi_distribution = 14 := by
  sorry

end taxi_distribution_eq_14_l3449_344962


namespace volunteer_distribution_theorem_l3449_344943

/-- The number of ways to distribute volunteers among exits -/
def distribute_volunteers (num_volunteers : ℕ) (num_exits : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements -/
theorem volunteer_distribution_theorem :
  distribute_volunteers 5 4 = 240 :=
sorry

end volunteer_distribution_theorem_l3449_344943


namespace count_special_four_digit_numbers_l3449_344941

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (ℕ × ℕ × ℕ × ℕ)

/-- Checks if a FourDigitNumber is valid (between 1000 and 9999) -/
def is_valid (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

/-- Converts a pair of digits to a two-digit number -/
def to_two_digit (a b : ℕ) : ℕ := 10 * a + b

/-- Checks if three two-digit numbers form an increasing arithmetic sequence -/
def is_increasing_arithmetic_seq (ab bc cd : ℕ) : Prop :=
  ab < bc ∧ bc < cd ∧ bc - ab = cd - bc

/-- The main theorem to be proved -/
theorem count_special_four_digit_numbers :
  (∃ (S : Finset FourDigitNumber),
    (∀ n ∈ S, is_valid n ∧ 
      let (a, b, c, d) := n
      is_increasing_arithmetic_seq (to_two_digit a b) (to_two_digit b c) (to_two_digit c d)) ∧
    S.card = 17 ∧
    (∀ n : FourDigitNumber, 
      is_valid n ∧ 
      let (a, b, c, d) := n
      is_increasing_arithmetic_seq (to_two_digit a b) (to_two_digit b c) (to_two_digit c d) 
      → n ∈ S)) := by
  sorry

end count_special_four_digit_numbers_l3449_344941


namespace bluejay_league_members_l3449_344976

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 8

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := sock_cost / 2

/-- The total cost for one member's equipment (home and away sets) -/
def member_cost : ℕ := 2 * (sock_cost + tshirt_cost + cap_cost)

/-- The total expenditure for all members -/
def total_expenditure : ℕ := 3876

/-- The number of members in the Bluejay Basketball League -/
def num_members : ℕ := total_expenditure / member_cost

theorem bluejay_league_members : num_members = 84 := by
  sorry


end bluejay_league_members_l3449_344976


namespace usual_time_calculation_l3449_344935

/-- Given a man who takes 24 minutes more to cover a distance when walking at 75% of his usual speed, 
    his usual time to cover this distance is 72 minutes. -/
theorem usual_time_calculation (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_time > 0) 
  (h2 : usual_speed > 0)
  (h3 : usual_speed * usual_time = 0.75 * usual_speed * (usual_time + 24)) : 
  usual_time = 72 := by
  sorry

end usual_time_calculation_l3449_344935


namespace min_value_expression_l3449_344977

theorem min_value_expression (a b : ℝ) (hb : b ≠ 0) :
  a^2 + b^2 + a/b + 1/b^2 ≥ Real.sqrt 3 := by
  sorry

end min_value_expression_l3449_344977


namespace cannot_form_flipped_shape_asymmetrical_shape_requires_flipping_asymmetrical_shape_cannot_be_formed_l3449_344913

/-- Represents a rhombus with two colored triangles -/
structure ColoredRhombus :=
  (orientation : ℕ)  -- Represents the rotation (0, 90, 180, 270 degrees)

/-- Represents a larger shape composed of multiple rhombuses -/
structure LargerShape :=
  (rhombuses : List ColoredRhombus)

/-- Represents whether a shape requires flipping to be formed -/
def requiresFlipping (shape : LargerShape) : Prop :=
  sorry  -- Definition of when a shape requires flipping

/-- Represents whether a shape can be formed by rotation only -/
def canFormByRotationOnly (shape : LargerShape) : Prop :=
  sorry  -- Definition of when a shape can be formed by rotation only

/-- Theorem: A shape that requires flipping cannot be formed by rotation only -/
theorem cannot_form_flipped_shape
  (shape : LargerShape) :
  requiresFlipping shape → ¬(canFormByRotationOnly shape) :=
by sorry

/-- The asymmetrical shape that cannot be formed -/
def asymmetricalShape : LargerShape :=
  sorry  -- Definition of the specific asymmetrical shape

/-- Theorem: The asymmetrical shape requires flipping -/
theorem asymmetrical_shape_requires_flipping :
  requiresFlipping asymmetricalShape :=
by sorry

/-- Main theorem: The asymmetrical shape cannot be formed by rotation only -/
theorem asymmetrical_shape_cannot_be_formed :
  ¬(canFormByRotationOnly asymmetricalShape) :=
by sorry

end cannot_form_flipped_shape_asymmetrical_shape_requires_flipping_asymmetrical_shape_cannot_be_formed_l3449_344913


namespace square_difference_theorem_l3449_344917

theorem square_difference_theorem (x y : ℚ) 
  (sum_eq : x + y = 8/15) 
  (diff_eq : x - y = 1/105) : 
  x^2 - y^2 = 8/1575 := by
  sorry

end square_difference_theorem_l3449_344917


namespace remaining_distance_to_hotel_l3449_344984

def total_distance : ℝ := 1200

def segment1_speed : ℝ := 60
def segment1_time : ℝ := 2

def segment2_speed : ℝ := 70
def segment2_time : ℝ := 3

def segment3_speed : ℝ := 50
def segment3_time : ℝ := 4

def segment4_speed : ℝ := 80
def segment4_time : ℝ := 5

def distance_traveled : ℝ :=
  segment1_speed * segment1_time +
  segment2_speed * segment2_time +
  segment3_speed * segment3_time +
  segment4_speed * segment4_time

theorem remaining_distance_to_hotel :
  total_distance - distance_traveled = 270 := by sorry

end remaining_distance_to_hotel_l3449_344984


namespace locus_of_R_l3449_344905

-- Define the square ABCD
structure Square :=
  (A B C D : ℝ × ℝ)

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the perimeter of a square
def perimeter (s : Square) : Set Point := sorry

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (P Q R : Point)

-- Define a rotation around a point
def rotate (center : Point) (angle : ℝ) (p : Point) : Point := sorry

-- Define the theorem
theorem locus_of_R (ABCD : Square) (Q : Point) :
  ∀ P ∈ perimeter ABCD,
  Q ∉ perimeter ABCD →
  ∃ (PQR : EquilateralTriangle),
  PQR.P = P ∧ PQR.Q = Q →
  ∃ (A₁B₁C₁D₁ A₂B₂C₂D₂ : Square),
  A₁B₁C₁D₁ = Square.mk (rotate Q (π/3) ABCD.A) (rotate Q (π/3) ABCD.B) (rotate Q (π/3) ABCD.C) (rotate Q (π/3) ABCD.D) ∧
  A₂B₂C₂D₂ = Square.mk (rotate Q (-π/3) ABCD.A) (rotate Q (-π/3) ABCD.B) (rotate Q (-π/3) ABCD.C) (rotate Q (-π/3) ABCD.D) ∧
  PQR.R ∈ perimeter A₁B₁C₁D₁ ∪ perimeter A₂B₂C₂D₂ :=
by sorry

end locus_of_R_l3449_344905


namespace chord_existence_l3449_344989

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 16
def line (x y : ℝ) : Prop := y = x + 1

-- Theorem statement
theorem chord_existence :
  ∃ (length : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    length = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end chord_existence_l3449_344989


namespace inscribed_circle_radius_right_triangle_l3449_344900

/-- The radius of an inscribed circle in a right triangle -/
theorem inscribed_circle_radius_right_triangle (a b c r : ℝ) 
  (h_right : a^2 + c^2 = b^2) -- Pythagorean theorem for right triangle
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) -- Positive side lengths
  : r = (a + c - b) / 2 ↔ 
    -- Definition of inscribed circle: 
    -- The circle touches all three sides of the triangle
    ∃ (x y : ℝ), 
      x > 0 ∧ y > 0 ∧
      x + y = b ∧
      x + r = c ∧
      y + r = a :=
by sorry

end inscribed_circle_radius_right_triangle_l3449_344900


namespace paiges_science_problems_l3449_344906

theorem paiges_science_problems 
  (math_problems : ℕ) 
  (total_problems : ℕ → ℕ → ℕ) 
  (finished_problems : ℕ) 
  (remaining_problems : ℕ) 
  (h1 : math_problems = 43)
  (h2 : ∀ m s, total_problems m s = m + s)
  (h3 : finished_problems = 44)
  (h4 : remaining_problems = 11)
  (h5 : ∀ s, remaining_problems = total_problems math_problems s - finished_problems) :
  ∃ s : ℕ, s = 12 ∧ total_problems math_problems s = finished_problems + remaining_problems :=
sorry

end paiges_science_problems_l3449_344906


namespace intersection_when_a_half_range_of_a_when_disjoint_l3449_344934

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_when_a_half :
  A (1/2) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

theorem range_of_a_when_disjoint (h1 : A a ≠ ∅) (h2 : A a ∩ B = ∅) :
  (-2 < a ∧ a ≤ 1/2) ∨ (a ≥ 2) := by sorry

end intersection_when_a_half_range_of_a_when_disjoint_l3449_344934


namespace min_distance_between_curves_l3449_344975

/-- The minimum distance between a point on y = (1/2)e^x and a point on y = ln(2x) -/
theorem min_distance_between_curves : ∃ (d : ℝ),
  (∀ (x₁ x₂ : ℝ), 
    let p := (x₁, (1/2) * Real.exp x₁)
    let q := (x₂, Real.log (2 * x₂))
    d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
  d = Real.sqrt 2 * (1 - Real.log 2) := by
  sorry

#check min_distance_between_curves

end min_distance_between_curves_l3449_344975


namespace last_tree_distance_l3449_344931

/-- The distance between the last pair of trees in a yard with a specific planting pattern -/
theorem last_tree_distance (yard_length : ℕ) (num_trees : ℕ) (first_distance : ℕ) (increment : ℕ) :
  yard_length = 1200 →
  num_trees = 117 →
  first_distance = 5 →
  increment = 2 →
  (num_trees - 1) * (2 * first_distance + (num_trees - 2) * increment) ≤ 2 * yard_length →
  first_distance + (num_trees - 2) * increment = 235 :=
by sorry

end last_tree_distance_l3449_344931


namespace library_reorganization_l3449_344955

theorem library_reorganization (total_books : Nat) (books_per_new_stack : Nat) 
    (h1 : total_books = 1450)
    (h2 : books_per_new_stack = 45) : 
  total_books % books_per_new_stack = 10 := by
  sorry

end library_reorganization_l3449_344955


namespace regular_washes_count_l3449_344932

/-- Represents the number of gallons of water used for different types of washes --/
structure WaterUsage where
  heavy : ℕ
  regular : ℕ
  light : ℕ

/-- Represents the number of different types of washes --/
structure Washes where
  heavy : ℕ
  regular : ℕ
  light : ℕ
  bleached : ℕ

/-- Calculates the total water usage for a given set of washes --/
def calculateWaterUsage (usage : WaterUsage) (washes : Washes) : ℕ :=
  usage.heavy * washes.heavy +
  usage.regular * washes.regular +
  usage.light * washes.light +
  usage.light * washes.bleached

/-- Theorem stating that there are 3 regular washes given the problem conditions --/
theorem regular_washes_count (usage : WaterUsage) (washes : Washes) :
  usage.heavy = 20 →
  usage.regular = 10 →
  usage.light = 2 →
  washes.heavy = 2 →
  washes.light = 1 →
  washes.bleached = 2 →
  calculateWaterUsage usage washes = 76 →
  washes.regular = 3 := by
  sorry

end regular_washes_count_l3449_344932


namespace range_of_a_l3449_344948

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : ℝ) : Prop := x^2 - (2*a + 1) * x + a * (a + 1) ≤ 0

-- Define the set A for proposition p
def A : Set ℝ := {x | p x}

-- Define the set B for proposition q
def B (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) → 0 ≤ a ∧ a ≤ 1/2 := by
  sorry

end range_of_a_l3449_344948


namespace original_ratio_l3449_344974

theorem original_ratio (x y : ℕ) (h1 : y = 72) (h2 : (x + 6) / y = 1 / 3) : y / x = 4 := by
  sorry

end original_ratio_l3449_344974


namespace petya_cannot_equalize_coins_l3449_344939

/-- Represents the state of Petya's coins -/
structure CoinState where
  two_kopeck : ℕ
  ten_kopeck : ℕ

/-- Represents a single transaction with the machine -/
inductive Transaction
  | insert_two
  | insert_ten

/-- Applies a single transaction to the current coin state -/
def apply_transaction (state : CoinState) (t : Transaction) : CoinState :=
  match t with
  | Transaction.insert_two => CoinState.mk (state.two_kopeck - 1) (state.ten_kopeck + 5)
  | Transaction.insert_ten => CoinState.mk (state.two_kopeck + 5) (state.ten_kopeck - 1)

/-- Applies a sequence of transactions to the initial state -/
def apply_transactions (initial : CoinState) (ts : List Transaction) : CoinState :=
  ts.foldl apply_transaction initial

/-- The theorem stating that Petya cannot end up with equal coins -/
theorem petya_cannot_equalize_coins :
  ∀ (ts : List Transaction),
    let final_state := apply_transactions (CoinState.mk 1 0) ts
    final_state.two_kopeck ≠ final_state.ten_kopeck :=
by sorry


end petya_cannot_equalize_coins_l3449_344939


namespace symmetric_line_wrt_y_axis_l3449_344929

/-- Given a line with equation y = 3x + 1, this theorem states that its symmetric line
    with respect to the y-axis has the equation y = -3x + 1 -/
theorem symmetric_line_wrt_y_axis :
  ∀ (x y : ℝ), (∃ (m n : ℝ), n = 3 * m + 1 ∧ x + m = 0 ∧ y = n) →
  y = -3 * x + 1 := by
  sorry

end symmetric_line_wrt_y_axis_l3449_344929


namespace jumble_words_count_l3449_344901

/-- The number of letters in the Jumble alphabet -/
def alphabet_size : ℕ := 21

/-- The maximum word length in the Jumble language -/
def max_word_length : ℕ := 5

/-- The number of words of length n in the Jumble language that contain at least one 'A' -/
def words_with_a (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else alphabet_size^n - (alphabet_size - 1)^n

/-- The total number of words in the Jumble language -/
def total_words : ℕ :=
  (List.range max_word_length).map (λ i => words_with_a (i + 1)) |>.sum

theorem jumble_words_count :
  total_words = 920885 := by sorry

end jumble_words_count_l3449_344901


namespace group_size_calculation_l3449_344983

theorem group_size_calculation (children women men : ℕ) : 
  children = 30 →
  women = 3 * children →
  men = 2 * women →
  children + women + men = 300 :=
by
  sorry

end group_size_calculation_l3449_344983


namespace parabola_tangent_properties_l3449_344965

/-- Given a parabola and a point, proves properties of its tangent lines -/
theorem parabola_tangent_properties (S : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) :
  S = (-3, 7) →
  (∀ x y, parabola x y ↔ y^2 = 5*x) →
  ∃ (t₁ t₂ : ℝ → ℝ) (P₁ P₂ : ℝ × ℝ) (α : ℝ),
    -- Tangent line equations
    (∀ x, t₁ x = x/6 + 15/2) ∧
    (∀ x, t₂ x = -5/2*x - 1/2) ∧
    -- Points of tangency
    P₁ = (45, 15) ∧
    P₂ = (1/5, -1) ∧
    -- Angle between tangents
    α = Real.arctan (32/7) ∧
    -- Tangent lines pass through S
    t₁ (S.1) = S.2 ∧
    t₂ (S.1) = S.2 ∧
    -- Points of tangency lie on the parabola
    parabola P₁.1 P₁.2 ∧
    parabola P₂.1 P₂.2 ∧
    -- Tangent lines touch the parabola at points of tangency
    t₁ P₁.1 = P₁.2 ∧
    t₂ P₂.1 = P₂.2 :=
by
  sorry


end parabola_tangent_properties_l3449_344965


namespace jim_car_efficiency_l3449_344930

/-- Calculates the fuel efficiency of a car given its tank capacity, remaining fuel ratio, and trip distance. -/
def fuel_efficiency (tank_capacity : ℚ) (remaining_ratio : ℚ) (trip_distance : ℚ) : ℚ :=
  trip_distance / (tank_capacity * (1 - remaining_ratio))

/-- Theorem stating that under the given conditions, the fuel efficiency is 5 miles per gallon. -/
theorem jim_car_efficiency :
  let tank_capacity : ℚ := 12
  let remaining_ratio : ℚ := 2/3
  let trip_distance : ℚ := 20
  fuel_efficiency tank_capacity remaining_ratio trip_distance = 5 := by
  sorry

end jim_car_efficiency_l3449_344930


namespace first_cube_weight_l3449_344991

/-- Given two cubical blocks of the same metal, where the second cube's sides are twice as long
    as the first cube's and weighs 48 pounds, prove that the first cube weighs 6 pounds. -/
theorem first_cube_weight (s : ℝ) (weight_first : ℝ) (weight_second : ℝ) :
  s > 0 →
  weight_second = 48 →
  weight_second / weight_first = (2 * s)^3 / s^3 →
  weight_first = 6 :=
by sorry

end first_cube_weight_l3449_344991


namespace monotone_increasing_implies_k_bound_l3449_344903

/-- A function f is monotonically increasing on an interval [a, b] if for any x, y in [a, b] with x ≤ y, we have f(x) ≤ f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

/-- The main theorem stating that if f(x) = kx - ln(x) is monotonically increasing on [2, 5], then k ≥ 1/2 -/
theorem monotone_increasing_implies_k_bound (k : ℝ) :
  MonotonicallyIncreasing (fun x => k * x - Real.log x) 2 5 → k ≥ 1/2 := by
  sorry


end monotone_increasing_implies_k_bound_l3449_344903


namespace min_sums_theorem_l3449_344922

def min_sums_for_unique_determination (n : ℕ) : ℕ :=
  Nat.choose (n - 1) 2 + 1

theorem min_sums_theorem (n : ℕ) (h : n ≥ 3) :
  ∀ (a : Fin n → ℝ),
  ∀ (k : ℕ),
  (k < min_sums_for_unique_determination n →
    ∃ (b₁ b₂ : Fin n → ℝ),
      b₁ ≠ b₂ ∧
      (∀ (i j : Fin n), i.val > j.val →
        (∃ (S : Finset (Fin n × Fin n)),
          S.card = k ∧
          (∀ (p : Fin n × Fin n), p ∈ S → p.1.val > p.2.val) ∧
          (∀ (p : Fin n × Fin n), p ∈ S → a (p.1) + a (p.2) = b₁ (p.1) + b₁ (p.2)) ∧
          (∀ (p : Fin n × Fin n), p ∈ S → a (p.1) + a (p.2) = b₂ (p.1) + b₂ (p.2))))) ∧
  (k ≥ min_sums_for_unique_determination n →
    ∀ (b : Fin n → ℝ),
    (∀ (S : Finset (Fin n × Fin n)),
      S.card = k →
      (∀ (p : Fin n × Fin n), p ∈ S → p.1.val > p.2.val) →
      (∃! (c : Fin n → ℝ), ∀ (p : Fin n × Fin n), p ∈ S → c (p.1) + c (p.2) = b (p.1) + b (p.2))))
  := by sorry

end min_sums_theorem_l3449_344922


namespace parallelogram_base_l3449_344928

/-- Given a parallelogram with area 416 cm² and height 16 cm, its base is 26 cm. -/
theorem parallelogram_base (area height : ℝ) (h_area : area = 416) (h_height : height = 16) :
  area / height = 26 := by
  sorry

end parallelogram_base_l3449_344928


namespace expression_equals_seven_l3449_344919

theorem expression_equals_seven :
  (-2023)^0 + Real.sqrt 4 - 2 * Real.sin (30 * π / 180) + abs (-5) = 7 := by
  sorry

end expression_equals_seven_l3449_344919


namespace compact_connected_preserving_implies_continuous_l3449_344936

/-- A function that maps compact sets to compact sets and connected sets to connected sets -/
def CompactConnectedPreserving (n m : ℕ) :=
  {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin m) |
    (∀ S : Set (EuclideanSpace ℝ (Fin n)), IsCompact S → IsCompact (f '' S)) ∧
    (∀ S : Set (EuclideanSpace ℝ (Fin n)), IsConnected S → IsConnected (f '' S))}

/-- Theorem: A function preserving compactness and connectedness is continuous -/
theorem compact_connected_preserving_implies_continuous
  {n m : ℕ} (f : CompactConnectedPreserving n m) :
  Continuous (f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin m)) :=
by sorry

end compact_connected_preserving_implies_continuous_l3449_344936


namespace min_distance_between_curves_l3449_344992

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (2 * x - 1)

theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), 
    (∀ (x₁ x₂ : ℝ), f x₁ = g x₂ → |x₂ - x₁| ≥ min_dist) ∧
    (∃ (x₁ x₂ : ℝ), f x₁ = g x₂ ∧ |x₂ - x₁| = min_dist) ∧
    min_dist = (5 + Real.log 2) / 4 := by
  sorry

end min_distance_between_curves_l3449_344992


namespace manuscript_cost_calculation_l3449_344909

/-- The total cost of typing a manuscript with given revision requirements -/
def manuscript_typing_cost (initial_cost : ℕ) (revision_cost : ℕ) (total_pages : ℕ) 
  (once_revised : ℕ) (twice_revised : ℕ) : ℕ :=
  (initial_cost * total_pages) + 
  (revision_cost * once_revised) + 
  (2 * revision_cost * twice_revised)

/-- Theorem stating the total cost of typing the manuscript -/
theorem manuscript_cost_calculation : 
  manuscript_typing_cost 6 4 100 35 15 = 860 := by
  sorry

end manuscript_cost_calculation_l3449_344909


namespace distribute_five_balls_to_three_children_l3449_344923

/-- The number of ways to distribute n identical balls to k children,
    with each child receiving at least one ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 6 ways to distribute 5 identical balls to 3 children,
    with each child receiving at least one ball -/
theorem distribute_five_balls_to_three_children :
  distribute_balls 5 3 = 6 := by
  sorry

end distribute_five_balls_to_three_children_l3449_344923


namespace race_finishing_orders_eq_twelve_l3449_344964

/-- Represents the number of possible finishing orders in a race with three participants,
    allowing for a tie only in the first place. -/
def race_finishing_orders : ℕ := 12

/-- Theorem stating that the number of possible finishing orders in a race with three participants,
    allowing for a tie only in the first place, is 12. -/
theorem race_finishing_orders_eq_twelve : race_finishing_orders = 12 := by
  sorry

end race_finishing_orders_eq_twelve_l3449_344964


namespace paper_crane_folding_time_l3449_344969

theorem paper_crane_folding_time (time_A time_B : ℝ) (h1 : time_A = 30) (h2 : time_B = 45) :
  (1 / time_A + 1 / time_B)⁻¹ = 18 := by sorry

end paper_crane_folding_time_l3449_344969


namespace two_numbers_with_given_means_l3449_344946

theorem two_numbers_with_given_means (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧ 
  (a + b) / 2 = 4 → 
  (a = 4 + Real.sqrt 11 ∧ b = 4 - Real.sqrt 11) ∨ 
  (a = 4 - Real.sqrt 11 ∧ b = 4 + Real.sqrt 11) :=
by sorry

end two_numbers_with_given_means_l3449_344946


namespace root_equation_n_value_l3449_344988

theorem root_equation_n_value : 
  ∀ n : ℝ, (1 : ℝ)^2 + 3*(1 : ℝ) + n = 0 → n = -4 := by
  sorry

end root_equation_n_value_l3449_344988


namespace min_value_sum_l3449_344938

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  x + y ≥ 18 := by
sorry

end min_value_sum_l3449_344938


namespace stone_145_is_2_l3449_344908

/-- The number of stones in the arrangement -/
def num_stones : ℕ := 14

/-- The period of the counting sequence -/
def period : ℕ := 26

/-- The target count we're looking for -/
def target_count : ℕ := 145

/-- Function to convert the new count to the original stone number -/
def count_to_original (n : ℕ) : ℕ :=
  if n % period ≤ num_stones then n % period
  else period - (n % period) + 1

theorem stone_145_is_2 :
  count_to_original target_count = 2 := by sorry

end stone_145_is_2_l3449_344908


namespace rational_equation_solution_l3449_344952

theorem rational_equation_solution :
  ∃ (x : ℝ), (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21) ∧ x = 4.5 := by
  sorry

end rational_equation_solution_l3449_344952


namespace projection_implies_y_coordinate_l3449_344902

/-- Given vectors a and b, if the projection of b in the direction of a is -√2, then the y-coordinate of b is 4. -/
theorem projection_implies_y_coordinate (a b : ℝ × ℝ) :
  a = (1, -1) →
  b.1 = 2 →
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt ((a.1 ^ 2 + a.2 ^ 2) : ℝ) = -Real.sqrt 2 →
  b.2 = 4 := by
  sorry

end projection_implies_y_coordinate_l3449_344902


namespace cubic_polynomial_sum_l3449_344979

/-- A cubic polynomial with coefficients in ℝ -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of a cubic polynomial at a point -/
def CubicPolynomial.eval (Q : CubicPolynomial) (x : ℝ) : ℝ :=
  Q.a * x^3 + Q.b * x^2 + Q.c * x + Q.d

theorem cubic_polynomial_sum (k : ℝ) (Q : CubicPolynomial) 
    (h0 : Q.eval 0 = k)
    (h1 : Q.eval 1 = 3*k)
    (h2 : Q.eval (-1) = 4*k) :
  Q.eval 2 + Q.eval (-2) = 22*k := by
  sorry

end cubic_polynomial_sum_l3449_344979


namespace bee_hatch_count_l3449_344911

/-- The number of bees hatching from the queen's eggs every day -/
def daily_hatch : ℕ := 3001

/-- The number of bees the queen loses every day -/
def daily_loss : ℕ := 900

/-- The number of days -/
def days : ℕ := 7

/-- The total number of bees in the hive after 7 days -/
def final_bees : ℕ := 27201

/-- The initial number of bees -/
def initial_bees : ℕ := 12500

/-- Theorem stating that the number of bees hatching daily is correct -/
theorem bee_hatch_count :
  initial_bees + days * (daily_hatch - daily_loss) = final_bees :=
by sorry

end bee_hatch_count_l3449_344911


namespace product_of_sum_and_cube_sum_l3449_344920

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 100) : 
  a * b = -3 := by
sorry

end product_of_sum_and_cube_sum_l3449_344920


namespace students_not_playing_sports_l3449_344997

theorem students_not_playing_sports (total : ℕ) (football : ℕ) (volleyball : ℕ) (one_sport : ℕ)
  (h_total : total = 40)
  (h_football : football = 20)
  (h_volleyball : volleyball = 19)
  (h_one_sport : one_sport = 15) :
  total - (football + volleyball - (football + volleyball - one_sport)) = 13 := by
  sorry

end students_not_playing_sports_l3449_344997


namespace total_savings_l3449_344904

def holiday_savings (sam victory alex : ℕ) : Prop :=
  victory = sam - 200 ∧ alex = 2 * victory ∧ sam = 1200

theorem total_savings (sam victory alex : ℕ) 
  (h : holiday_savings sam victory alex) : 
  sam + victory + alex = 4200 := by
  sorry

end total_savings_l3449_344904


namespace michael_goals_multiplier_l3449_344953

theorem michael_goals_multiplier (bruce_goals : ℕ) (total_goals : ℕ) : 
  bruce_goals = 4 → total_goals = 16 → 
  ∃ x : ℕ, x * bruce_goals = total_goals - bruce_goals ∧ x = 3 := by
sorry

end michael_goals_multiplier_l3449_344953


namespace line_direction_vector_l3449_344980

/-- Given a line y = (5x - 7) / 2 parameterized as (x, y) = v + t * d,
    where the distance between (x, y) and (4, 2) is t for x ≥ 4,
    prove that the direction vector d is (2/√29, 5/√29). -/
theorem line_direction_vector (v d : ℝ × ℝ) :
  (∀ x y t : ℝ, x ≥ 4 →
    y = (5 * x - 7) / 2 →
    (x, y) = v + t • d →
    ‖(x, y) - (4, 2)‖ = t) →
  d = (2 / Real.sqrt 29, 5 / Real.sqrt 29) :=
by sorry

end line_direction_vector_l3449_344980


namespace johnny_red_pencils_l3449_344947

/-- The number of red pencils Johnny bought given the conditions of the problem -/
def total_red_pencils (total_packs : ℕ) (regular_red_per_pack : ℕ) (extra_red_packs : ℕ) (extra_red_per_pack : ℕ) : ℕ :=
  total_packs * regular_red_per_pack + extra_red_packs * extra_red_per_pack

/-- Theorem stating that Johnny bought 21 red pencils -/
theorem johnny_red_pencils :
  total_red_pencils 15 1 3 2 = 21 := by
  sorry

end johnny_red_pencils_l3449_344947


namespace taehyung_walk_distance_l3449_344996

/-- Proves that given Taehyung's step length of 0.45 meters, and moving 90 steps 13 times, the total distance walked is 526.5 meters. -/
theorem taehyung_walk_distance :
  let step_length : ℝ := 0.45
  let steps_per_set : ℕ := 90
  let num_sets : ℕ := 13
  step_length * (steps_per_set * num_sets : ℝ) = 526.5 := by sorry

end taehyung_walk_distance_l3449_344996


namespace complex_equation_solution_l3449_344954

theorem complex_equation_solution (z : ℂ) :
  Complex.abs z + z = 2 + 4 * Complex.I → z = -3 + 4 * Complex.I := by
sorry

end complex_equation_solution_l3449_344954


namespace soccer_team_subjects_l3449_344999

theorem soccer_team_subjects (total : ℕ) (physics : ℕ) (both : ℕ) (math : ℕ) :
  total = 20 →
  physics = 12 →
  both = 6 →
  total = physics + math - both →
  math = 14 :=
by
  sorry

end soccer_team_subjects_l3449_344999


namespace carpenters_for_chairs_l3449_344916

/-- Represents the number of carpenters needed to make a certain number of chairs in a given number of days. -/
def carpenters_needed (initial_carpenters : ℕ) (initial_chairs : ℕ) (target_chairs : ℕ) : ℕ :=
  (initial_carpenters * target_chairs + initial_chairs - 1) / initial_chairs

/-- Proves that 12 carpenters are needed to make 75 chairs in 10 days, given that 8 carpenters can make 50 chairs in 10 days. -/
theorem carpenters_for_chairs : carpenters_needed 8 50 75 = 12 := by
  sorry

end carpenters_for_chairs_l3449_344916


namespace intersection_empty_iff_a_lt_neg_one_l3449_344915

/-- Define set A as {x | -1 ≤ x < 2} -/
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}

/-- Define set B as {x | x ≤ a} -/
def B (a : ℝ) : Set ℝ := {x | x ≤ a}

/-- Theorem: The intersection of A and B is empty if and only if a < -1 -/
theorem intersection_empty_iff_a_lt_neg_one (a : ℝ) :
  A ∩ B a = ∅ ↔ a < -1 := by
  sorry

end intersection_empty_iff_a_lt_neg_one_l3449_344915


namespace sum_of_digits_greatest_prime_divisor_16385_l3449_344970

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_greatest_prime_divisor_16385 :
  sum_of_digits (greatest_prime_divisor 16385) = 19 := by sorry

end sum_of_digits_greatest_prime_divisor_16385_l3449_344970


namespace coefficient_of_x_squared_l3449_344978

theorem coefficient_of_x_squared (z : ℂ) (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  z = 1 + I →
  (∀ x : ℂ, (x + z)^4 = a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = 12*I :=
by sorry

end coefficient_of_x_squared_l3449_344978


namespace floor_of_e_l3449_344927

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by sorry

end floor_of_e_l3449_344927


namespace total_spent_after_discount_and_tax_l3449_344963

def bracelet_price : ℝ := 4
def keychain_price : ℝ := 5
def coloring_book_price : ℝ := 3
def sticker_pack_price : ℝ := 1
def toy_car_price : ℝ := 6

def bracelet_discount_rate : ℝ := 0.1
def sales_tax_rate : ℝ := 0.05

def paula_bracelets : ℕ := 3
def paula_keychains : ℕ := 2
def paula_coloring_books : ℕ := 1
def paula_sticker_packs : ℕ := 4

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 2
def olive_toy_cars : ℕ := 1
def olive_sticker_packs : ℕ := 3

def nathan_toy_cars : ℕ := 4
def nathan_sticker_packs : ℕ := 5
def nathan_keychains : ℕ := 1

theorem total_spent_after_discount_and_tax : 
  let paula_total := paula_bracelets * bracelet_price + paula_keychains * keychain_price + 
                     paula_coloring_books * coloring_book_price + paula_sticker_packs * sticker_pack_price
  let olive_total := olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price + 
                     olive_toy_cars * toy_car_price + olive_sticker_packs * sticker_pack_price
  let nathan_total := nathan_toy_cars * toy_car_price + nathan_sticker_packs * sticker_pack_price + 
                      nathan_keychains * keychain_price
  let paula_discount := paula_bracelets * bracelet_price * bracelet_discount_rate
  let olive_discount := olive_bracelets * bracelet_price * bracelet_discount_rate
  let total_before_tax := paula_total - paula_discount + olive_total - olive_discount + nathan_total
  let total_after_tax := total_before_tax * (1 + sales_tax_rate)
  total_after_tax = 85.05 := by sorry

end total_spent_after_discount_and_tax_l3449_344963


namespace doubled_cost_percentage_new_cost_percentage_l3449_344960

-- Define the cost function
def cost (t b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percentage (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  cost t (2 * b) = 16 * cost t b :=
by sorry

-- Main theorem
theorem new_cost_percentage (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  (cost t (2 * b) / cost t b) * 100 = 1600 :=
by sorry

end doubled_cost_percentage_new_cost_percentage_l3449_344960


namespace perimeter_of_externally_touching_circles_l3449_344926

/-- Given two externally touching circles with radii in the ratio 3:1 and a common external tangent
    of length 6√3, the perimeter of the figure formed by the external tangents and the external
    parts of the circles is 14π + 12√3. -/
theorem perimeter_of_externally_touching_circles (r R : ℝ) (h1 : R = 3 * r) 
    (h2 : r > 0) (h3 : 6 * Real.sqrt 3 = 2 * r * Real.sqrt 3) : 
    2 * (6 * Real.sqrt 3) + 2 * π * r * (1/3) + 2 * π * R * (2/3) = 14 * π + 12 * Real.sqrt 3 :=
by sorry

end perimeter_of_externally_touching_circles_l3449_344926


namespace ellipse_k_value_l3449_344912

/-- The equation of an ellipse with parameter k -/
def ellipse_equation (k x y : ℝ) : Prop :=
  2 * k * x^2 + k * y^2 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (0, -4)

/-- Theorem stating that for an ellipse with the given equation and focus, k = 1/32 -/
theorem ellipse_k_value :
  ∃ (k : ℝ), k ≠ 0 ∧
  (∀ x y : ℝ, ellipse_equation k x y ↔ 2 * k * x^2 + k * y^2 = 1) ∧
  (∃ x y : ℝ, ellipse_equation k x y ∧ (x, y) = focus) ∧
  k = 1/32 := by
  sorry

end ellipse_k_value_l3449_344912


namespace two_digit_number_existence_l3449_344940

/-- Two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- First digit of a two-digit number -/
def firstDigit (n : ℕ) : ℕ := n / 10

/-- Second digit of a two-digit number -/
def secondDigit (n : ℕ) : ℕ := n % 10

/-- Sum of digits of a two-digit number -/
def digitSum (n : ℕ) : ℕ := firstDigit n + secondDigit n

/-- Absolute difference of digits of a two-digit number -/
def digitDiff (n : ℕ) : ℕ := Int.natAbs (firstDigit n - secondDigit n)

theorem two_digit_number_existence :
  ∃ (X Y : ℕ), 
    TwoDigitNumber X ∧ 
    TwoDigitNumber Y ∧ 
    X = 2 * Y ∧
    (firstDigit Y = digitSum X ∨ secondDigit Y = digitSum X) ∧
    (firstDigit Y = digitDiff X ∨ secondDigit Y = digitDiff X) ∧
    X = 34 ∧ 
    Y = 17 :=
by
  sorry

end two_digit_number_existence_l3449_344940


namespace alex_shirts_l3449_344972

theorem alex_shirts (alex joe ben : ℕ) 
  (h1 : joe = alex + 3)
  (h2 : ben = joe + 8)
  (h3 : ben = 15) : 
  alex = 4 := by
  sorry

end alex_shirts_l3449_344972


namespace complex_exp_form_l3449_344986

/-- For the complex number z = 1 + i√3, when expressed in the form re^(iθ), θ = π/3 -/
theorem complex_exp_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (π / 3)) :=
by sorry

end complex_exp_form_l3449_344986


namespace smallest_positive_period_of_f_triangle_area_l3449_344967

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sqrt 3 * Real.sin x * Real.cos x + 1/2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_positive_period_of_f :
  ∃ T > 0, is_periodic f T ∧ ∀ S, 0 < S ∧ S < T → ¬ is_periodic f S :=
sorry

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  f (B + C) = 3/2 →
  a = Real.sqrt 3 →
  b + c = 3 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
sorry

end smallest_positive_period_of_f_triangle_area_l3449_344967


namespace investment_time_period_l3449_344987

theorem investment_time_period (P : ℝ) (rate_diff : ℝ) (interest_diff : ℝ) :
  P = 8400 →
  rate_diff = 0.05 →
  interest_diff = 840 →
  (P * rate_diff * 2 = interest_diff) := by
  sorry

end investment_time_period_l3449_344987


namespace soccer_balls_with_holes_l3449_344994

theorem soccer_balls_with_holes (total_soccer : ℕ) (total_basketball : ℕ) (basketball_with_holes : ℕ) (total_without_holes : ℕ) :
  total_soccer = 40 →
  total_basketball = 15 →
  basketball_with_holes = 7 →
  total_without_holes = 18 →
  total_soccer - (total_without_holes - (total_basketball - basketball_with_holes)) = 30 := by
sorry

end soccer_balls_with_holes_l3449_344994


namespace set_operations_and_range_l3449_344949

def U := Set ℝ

def A : Set ℝ := {x | x ≥ 3}

def B : Set ℝ := {x | x^2 - 8*x + 7 ≤ 0}

def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem set_operations_and_range :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x : ℝ | x ≥ 1}) ∧
  (∀ a : ℝ, C a ∪ A = A → a ≥ 4) :=
sorry

end set_operations_and_range_l3449_344949


namespace arctan_equation_solution_l3449_344995

theorem arctan_equation_solution :
  ∀ x : ℝ, Real.arctan (1 / x) + Real.arctan (1 / x^2) = π / 4 → x = 2 := by
sorry

end arctan_equation_solution_l3449_344995


namespace rectangle_area_l3449_344918

/-- Given a rectangle with diagonal length 2a + b, its area is 2ab -/
theorem rectangle_area (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = (2*a + b)^2 ∧ x * y = 2*a*b :=
by sorry

end rectangle_area_l3449_344918


namespace sphere_surface_area_l3449_344981

theorem sphere_surface_area (v : ℝ) (r : ℝ) (h : v = 72 * Real.pi) :
  4 * Real.pi * r^2 = 36 * Real.pi * (2^(2/3)) :=
by
  sorry

end sphere_surface_area_l3449_344981


namespace sqrt_11_simplest_l3449_344914

def is_simplest_sqrt (n : ℕ) (others : List ℕ) : Prop :=
  ∀ m ∈ others, ¬ (∃ k : ℕ, k > 1 ∧ k * k ∣ n) ∧ (∃ k : ℕ, k > 1 ∧ k * k ∣ m)

theorem sqrt_11_simplest : is_simplest_sqrt 11 [8, 12, 36] := by
  sorry

end sqrt_11_simplest_l3449_344914


namespace largest_integer_satisfying_inequality_l3449_344957

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * x + 1 > 4 * x - 6) → x ≤ 6 ∧ (3 * 6 + 1 > 4 * 6 - 6) :=
by
  sorry

end largest_integer_satisfying_inequality_l3449_344957


namespace initial_sony_games_l3449_344971

/-- The number of Sony games Kelly gives away -/
def games_given_away : ℕ := 101

/-- The number of Sony games Kelly has left after giving away games -/
def games_left : ℕ := 31

/-- The initial number of Sony games Kelly has -/
def initial_games : ℕ := games_given_away + games_left

theorem initial_sony_games : initial_games = 132 := by sorry

end initial_sony_games_l3449_344971


namespace problem_solution_l3449_344951

theorem problem_solution (x : ℝ) (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 12) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 200/9 := by
  sorry

end problem_solution_l3449_344951


namespace counterexample_exists_l3449_344910

theorem counterexample_exists : ∃ n : ℕ, 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b) ∧ 
  (∃ k : ℕ, n = 3 * k) ∧ 
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ n - 2 = x * y) ∧ 
  n - 2 ≠ 2 :=
by sorry

end counterexample_exists_l3449_344910


namespace total_distance_traveled_l3449_344968

def speed : ℝ := 60
def driving_sessions : List ℝ := [4, 5, 3, 2]

theorem total_distance_traveled :
  (List.sum driving_sessions) * speed = 840 := by
  sorry

end total_distance_traveled_l3449_344968


namespace repeating_decimal_sum_l3449_344925

theorem repeating_decimal_sum (a b c : Nat) : 
  a < 10 ∧ b < 10 ∧ c < 10 →
  (10 * a + b) / 99 + (100 * a + 10 * b) / 9900 + (10 * b + c) / 99 = 25 / 99 →
  100 * a + 10 * b + c = 23 :=
by sorry

end repeating_decimal_sum_l3449_344925


namespace arithmetic_sequence_problem_l3449_344921

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define the sum of the first n terms of b_n
def T (n : ℕ) : ℚ := n / (2 * n + 1)

theorem arithmetic_sequence_problem :
  S 9 = 81 ∧ a 3 + a 5 = 14 →
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∀ n : ℕ, T n = n / (2 * n + 1)) :=
by sorry

end arithmetic_sequence_problem_l3449_344921


namespace increase_by_percentage_l3449_344993

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 80 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 200 := by
  sorry

end increase_by_percentage_l3449_344993


namespace part_I_part_II_l3449_344942

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 2 - m

-- Define the set A
def A (m : ℝ) : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ y = f m x}

-- Theorem for part (I)
theorem part_I (m : ℝ) : 
  (∀ x, f m x ≥ x - m*x) → m ∈ Set.Icc (-7 : ℝ) 1 := by sorry

-- Theorem for part (II)
theorem part_II : 
  (∃ m : ℝ, A m ⊆ Set.Ici 0 ∧ ∀ m' : ℝ, A m' ⊆ Set.Ici 0 → m' ≤ m) → 
  (∃ m : ℝ, m = 1 ∧ A m ⊆ Set.Ici 0 ∧ ∀ m' : ℝ, A m' ⊆ Set.Ici 0 → m' ≤ m) := by sorry

end part_I_part_II_l3449_344942
