import Mathlib

namespace solve_system_l3686_368619

theorem solve_system (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 :=
by sorry

end solve_system_l3686_368619


namespace polynomial_division_remainder_l3686_368651

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (X^5 - 1) * (X^3 - 1) = (X^3 + X^2 + 1) * q + (-2*X^2 + X + 1) := by sorry

end polynomial_division_remainder_l3686_368651


namespace complex_magnitude_l3686_368690

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (2 * z - w) = 25)
  (h2 : Complex.abs (z + 2 * w) = 5)
  (h3 : Complex.abs (z + w) = 2) : 
  Complex.abs z = 9 := by sorry

end complex_magnitude_l3686_368690


namespace min_value_at_3_l3686_368612

def S (n : ℕ) : ℤ := n^2 - 10*n

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1
  else S n - S (n-1)

def na (n : ℕ) : ℤ := n * (a n)

theorem min_value_at_3 :
  ∀ k : ℕ, k ≥ 1 → na 3 ≤ na k :=
by sorry

end min_value_at_3_l3686_368612


namespace mollys_present_age_l3686_368627

def mollys_age_equation (x : ℕ) : Prop :=
  x + 18 = 5 * (x - 6)

theorem mollys_present_age : 
  ∃ (x : ℕ), mollys_age_equation x ∧ x = 12 :=
by sorry

end mollys_present_age_l3686_368627


namespace increase_by_percentage_increase_500_by_30_percent_l3686_368624

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) : 
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) :=
by sorry

theorem increase_500_by_30_percent : 
  500 * (1 + 30 / 100) = 650 :=
by sorry

end increase_by_percentage_increase_500_by_30_percent_l3686_368624


namespace book_pages_l3686_368630

/-- The number of pages Frank reads per day -/
def pages_per_day : ℕ := 22

/-- The number of days it took Frank to finish the book -/
def days_to_finish : ℕ := 569

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_day * days_to_finish

theorem book_pages : total_pages = 12518 := by
  sorry

end book_pages_l3686_368630


namespace ivan_max_bars_ivan_min_bars_ivan_exact_bars_l3686_368643

/-- Represents the state of the game -/
structure GameState where
  ivan_bars : ℕ
  chest_bars : ℕ

/-- Represents a player's move -/
structure Move where
  bars : ℕ

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (move : Move) : Prop :=
  move.bars > 0 ∧ move.bars ≤ state.chest_bars

/-- Defines the game's rules and outcome -/
def game_outcome (initial_bars : ℕ) : ℕ :=
  sorry

/-- Theorem stating that Ivan can always take at most 13 bars -/
theorem ivan_max_bars (n : ℕ) (h : n ≥ 13) : 
  game_outcome n ≤ 13 :=
sorry

/-- Theorem stating that Ivan can always take at least 13 bars -/
theorem ivan_min_bars (n : ℕ) (h : n ≥ 13) : 
  game_outcome n ≥ 13 :=
sorry

/-- Main theorem proving that Ivan can always take exactly 13 bars -/
theorem ivan_exact_bars (n : ℕ) (h : n ≥ 13) : 
  game_outcome n = 13 :=
sorry

end ivan_max_bars_ivan_min_bars_ivan_exact_bars_l3686_368643


namespace chess_grandmaster_learning_time_l3686_368646

theorem chess_grandmaster_learning_time 
  (total_time : ℕ) 
  (proficiency_multiplier : ℕ) 
  (mastery_multiplier : ℕ) 
  (h1 : total_time = 10100)
  (h2 : proficiency_multiplier = 49)
  (h3 : mastery_multiplier = 100) : 
  ∃ (rule_learning_time : ℕ), 
    rule_learning_time = 2 ∧ 
    total_time = rule_learning_time + 
                 proficiency_multiplier * rule_learning_time + 
                 mastery_multiplier * (rule_learning_time + proficiency_multiplier * rule_learning_time) :=
by sorry

end chess_grandmaster_learning_time_l3686_368646


namespace transportation_optimization_l3686_368607

/-- Demand function for transportation --/
def demand (p : ℝ) : ℝ := 3000 - 20 * p

/-- Transportation cost function for the bus company --/
def transportCost (y : ℝ) : ℝ := y + 5

/-- Fixed train fare --/
def trainFare : ℝ := 10

/-- Maximum train capacity --/
def trainCapacity : ℝ := 1000

/-- Optimal bus fare when train is operating --/
def optimalBusFare : ℝ := 50.5

/-- Decrease in total passengers after train closure --/
def passengerDecrease : ℝ := 500

theorem transportation_optimization :
  let busDemand (p : ℝ) := max 0 (demand p - trainCapacity)
  let busProfit (p : ℝ) := p * busDemand p - transportCost (busDemand p)
  let totalDemandWithTrain := trainCapacity + busDemand optimalBusFare
  let totalDemandWithoutTrain := demand 75.5
  (∀ p, p > trainFare → busProfit p ≤ busProfit optimalBusFare) ∧
  (totalDemandWithTrain - totalDemandWithoutTrain = passengerDecrease) := by
  sorry

end transportation_optimization_l3686_368607


namespace distance_FM_l3686_368644

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

-- Define the length of AB
def length_AB : ℝ := 6

-- Define the perpendicular bisector of AB
def perp_bisector (k : ℝ) (x y : ℝ) : Prop :=
  y - k = -(1/k) * (x - 2)

-- Define the point M
def point_M (k : ℝ) : ℝ × ℝ :=
  (4, 0)

-- Theorem statement
theorem distance_FM (k : ℝ) :
  let F := focus
  let M := point_M k
  (M.1 - F.1)^2 + (M.2 - F.2)^2 = 3^2 :=
sorry

end distance_FM_l3686_368644


namespace point_b_position_l3686_368672

theorem point_b_position (a b : ℤ) : 
  a = -5 → (b = a + 4 ∨ b = a - 4) → (b = -1 ∨ b = -9) := by
  sorry

end point_b_position_l3686_368672


namespace jim_gave_away_900_cards_l3686_368613

/-- The number of cards Jim gave away -/
def cards_given_away (initial_cards : ℕ) (set_size : ℕ) (sets_to_brother sets_to_sister sets_to_friend sets_to_cousin sets_to_classmate : ℕ) : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend + sets_to_cousin + sets_to_classmate) * set_size

/-- Proof that Jim gave away 900 cards -/
theorem jim_gave_away_900_cards :
  cards_given_away 1500 25 15 8 4 6 3 = 900 := by
  sorry

end jim_gave_away_900_cards_l3686_368613


namespace rook_paths_on_chessboard_l3686_368680

def rook_paths (n m k : ℕ) : ℕ :=
  if n + m ≠ k then 0
  else Nat.choose (n + m) n

theorem rook_paths_on_chessboard :
  (rook_paths 7 7 14 = 3432) ∧
  (rook_paths 7 7 12 = 57024) ∧
  (rook_paths 7 7 5 = 2000) := by
  sorry

end rook_paths_on_chessboard_l3686_368680


namespace square_area_proof_l3686_368668

theorem square_area_proof (s : ℝ) (h1 : s = 4) : 
  (s^2 + s) - (4 * s) = 4 → s^2 = 16 := by
sorry

end square_area_proof_l3686_368668


namespace taxi_driver_probability_l3686_368676

-- Define the number of checkpoints
def num_checkpoints : ℕ := 6

-- Define the probability of encountering a red light at each checkpoint
def red_light_prob : ℚ := 1/3

-- Define the probability of passing exactly two checkpoints before encountering a red light
def pass_two_checkpoints_prob : ℚ := 4/27

-- State the theorem
theorem taxi_driver_probability :
  ∀ (n : ℕ) (p : ℚ),
  n = num_checkpoints →
  p = red_light_prob →
  pass_two_checkpoints_prob = (1 - p) * (1 - p) * p :=
by sorry

end taxi_driver_probability_l3686_368676


namespace expression_evaluation_l3686_368660

theorem expression_evaluation (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  let x := (4 * a * b) / (a + b)
  ((x + 2*b) / (x - 2*b) + (x + 2*a) / (x - 2*a)) / (x / 2) = (a + b) / (a * b) :=
by sorry

end expression_evaluation_l3686_368660


namespace D_largest_l3686_368686

def D : ℚ := 3006 / 3005 + 3006 / 3007
def E : ℚ := 3006 / 3007 + 3008 / 3007
def F : ℚ := 3007 / 3006 + 3007 / 3008

theorem D_largest : D > E ∧ D > F := by
  sorry

end D_largest_l3686_368686


namespace division_remainder_l3686_368614

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 149 →
  divisor = 16 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end division_remainder_l3686_368614


namespace trivia_team_absentees_l3686_368610

theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 7 →
  points_per_member = 4 →
  total_points = 20 →
  total_members - (total_points / points_per_member) = 2 :=
by sorry

end trivia_team_absentees_l3686_368610


namespace jesse_sam_earnings_l3686_368617

theorem jesse_sam_earnings (t : ℝ) : 
  t > 0 → 
  (t - 3) * (3 * t - 4) = 2 * (3 * t - 6) * (t - 3) → 
  t = 4 := by
sorry

end jesse_sam_earnings_l3686_368617


namespace number_of_big_boxes_l3686_368648

theorem number_of_big_boxes (soaps_per_package : ℕ) (packages_per_box : ℕ) (total_soaps : ℕ) : 
  soaps_per_package = 192 →
  packages_per_box = 6 →
  total_soaps = 2304 →
  total_soaps / (soaps_per_package * packages_per_box) = 2 :=
by
  sorry

#check number_of_big_boxes

end number_of_big_boxes_l3686_368648


namespace a_minus_b_is_perfect_square_l3686_368633

theorem a_minus_b_is_perfect_square (a b : ℕ+) (h : 2 * a ^ 2 + a = 3 * b ^ 2 + b) :
  ∃ k : ℕ, (a : ℤ) - (b : ℤ) = k ^ 2 := by
  sorry

end a_minus_b_is_perfect_square_l3686_368633


namespace trig_identity_l3686_368673

theorem trig_identity : 
  (Real.cos (70 * π / 180) + Real.cos (50 * π / 180)) * 
  (Real.cos (310 * π / 180) + Real.cos (290 * π / 180)) + 
  (Real.cos (40 * π / 180) + Real.cos (160 * π / 180)) * 
  (Real.cos (320 * π / 180) - Real.cos (380 * π / 180)) = 1 := by
  sorry

end trig_identity_l3686_368673


namespace exponential_inequality_l3686_368697

theorem exponential_inequality (x y : ℝ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : Real.log x - Real.log y < 1 / Real.log x - 1 / Real.log y) : 
  Real.exp (y - x) > 1 := by
  sorry

end exponential_inequality_l3686_368697


namespace quadratic_inequality_condition_l3686_368615

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5)*x - k + 9 > 1) ↔ k > -1 ∧ k < 7 := by
  sorry

end quadratic_inequality_condition_l3686_368615


namespace prob_B_not_lose_l3686_368625

/-- The probability of player A winning in a chess game -/
def prob_A_win : ℝ := 0.3

/-- The probability of a draw in a chess game -/
def prob_draw : ℝ := 0.5

/-- Theorem: The probability of player B not losing in a chess game -/
theorem prob_B_not_lose : prob_A_win + prob_draw + (1 - prob_A_win - prob_draw) = 0.7 := by
  sorry

end prob_B_not_lose_l3686_368625


namespace symmetric_point_coordinates_l3686_368688

/-- Given a point A(a, 4) in the second quadrant and a vertical line m with x = 2,
    the point symmetric to A with respect to m has coordinates (4-a, 4). -/
theorem symmetric_point_coordinates (a : ℝ) (h1 : a < 0) :
  let A : ℝ × ℝ := (a, 4)
  let m : Set (ℝ × ℝ) := {p | p.1 = 2}
  let symmetric_point := (4 - a, 4)
  symmetric_point.1 = 4 - a ∧ symmetric_point.2 = 4 := by
  sorry


end symmetric_point_coordinates_l3686_368688


namespace son_age_proof_l3686_368687

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end son_age_proof_l3686_368687


namespace mulch_cost_theorem_l3686_368621

/-- The cost of mulch in dollars per cubic foot -/
def mulch_cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of mulch in cubic yards -/
def mulch_volume_cubic_yards : ℝ := 7

/-- The cost of mulch for a given volume in cubic yards -/
def mulch_cost (volume_cubic_yards : ℝ) : ℝ :=
  volume_cubic_yards * cubic_yards_to_cubic_feet * mulch_cost_per_cubic_foot

theorem mulch_cost_theorem : mulch_cost mulch_volume_cubic_yards = 1512 := by
  sorry

end mulch_cost_theorem_l3686_368621


namespace inverse_function_point_correspondence_l3686_368636

theorem inverse_function_point_correspondence 
  (f : ℝ → ℝ) (hf : Function.Bijective f) :
  (1 - f 1 = 2) → (f⁻¹ (-1) - (-1) = 2) :=
by
  sorry

end inverse_function_point_correspondence_l3686_368636


namespace region_location_l3686_368696

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x - 2*y + 6 > 0

-- Define what it means to be on the lower right side of the line
def lower_right_side (x y : ℝ) : Prop := 
  x > -6 ∧ y < 3 ∧ region x y

-- Theorem statement
theorem region_location : 
  ∀ x y : ℝ, region x y → lower_right_side x y :=
sorry

end region_location_l3686_368696


namespace n2o5_molecular_weight_l3686_368605

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O5 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O5 -/
def oxygen_count : ℕ := 5

/-- The molecular weight of N2O5 in g/mol -/
def n2o5_weight : ℝ := nitrogen_count * nitrogen_weight + oxygen_count * oxygen_weight

theorem n2o5_molecular_weight : n2o5_weight = 108.02 := by
  sorry

end n2o5_molecular_weight_l3686_368605


namespace magnitude_squared_l3686_368678

theorem magnitude_squared (w : ℂ) (h : Complex.abs w = 11) : (2 * Complex.abs w)^2 = 484 := by
  sorry

end magnitude_squared_l3686_368678


namespace min_difference_of_product_l3686_368670

theorem min_difference_of_product (a b : ℤ) (h : a * b = 156) :
  ∀ x y : ℤ, x * y = 156 → a - b ≤ x - y :=
by sorry

end min_difference_of_product_l3686_368670


namespace sign_of_product_l3686_368692

theorem sign_of_product (h1 : 0 < 1 ∧ 1 < Real.pi / 2) 
  (h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.sin x < Real.sin y)
  (h3 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.cos y < Real.cos x) :
  (Real.cos (Real.cos 1) - Real.cos 1) * (Real.sin (Real.sin 1) - Real.sin 1) < 0 := by
  sorry

end sign_of_product_l3686_368692


namespace race_head_start_l3686_368656

theorem race_head_start (v_a v_b L H : ℝ) : 
  v_a = (32/27) * v_b →
  (L / v_a = (L - H) / v_b) →
  H = (5/32) * L :=
by sorry

end race_head_start_l3686_368656


namespace min_value_of_a_l3686_368682

theorem min_value_of_a (a : ℝ) (h1 : a > 0) : 
  (∀ x : ℝ, x > 1 → x + a / (x - 1) ≥ 5) → a ≥ 4 := by
  sorry

end min_value_of_a_l3686_368682


namespace vectors_are_parallel_l3686_368693

def a : ℝ × ℝ × ℝ := (1, 2, -2)
def b : ℝ × ℝ × ℝ := (-2, -4, 4)

theorem vectors_are_parallel : ∃ (k : ℝ), b = k • a := by sorry

end vectors_are_parallel_l3686_368693


namespace hash_composition_l3686_368637

-- Define the # operation
def hash (a b : ℝ) : ℝ := a * b - b + b^2

-- Theorem statement
theorem hash_composition (z : ℝ) : hash (hash 3 8) z = 79 * z + z^2 := by
  sorry

end hash_composition_l3686_368637


namespace new_person_weight_l3686_368654

/-- Given a group of 6 persons where replacing a 65 kg person with a new person
    increases the average weight by 1.5 kg, prove that the weight of the new person is 74 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_replaced : ℝ) (avg_increase : ℝ) :
  initial_count = 6 →
  weight_replaced = 65 →
  avg_increase = 1.5 →
  (initial_count : ℝ) * avg_increase + weight_replaced = 74 := by
  sorry

end new_person_weight_l3686_368654


namespace line_equation_l3686_368664

/-- A line passing through point A(1,4) with the sum of its intercepts on the two axes equal to zero -/
structure LineWithZeroSumIntercepts where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point A(1,4) -/
  passes_through_A : 4 = slope * 1 + y_intercept
  /-- The sum of intercepts on the two axes is zero -/
  zero_sum_intercepts : (-y_intercept / slope) + y_intercept = 0

/-- The equation of the line is either 4x - y = 0 or x - y + 3 = 0 -/
theorem line_equation (l : LineWithZeroSumIntercepts) :
  (l.slope = 4 ∧ l.y_intercept = 0) ∨ (l.slope = 1 ∧ l.y_intercept = 3) :=
sorry

end line_equation_l3686_368664


namespace discount_calculation_l3686_368652

-- Define the discount rate
def discount_rate : ℝ := 0.2

-- Define the original price (can be any positive real number)
variable (original_price : ℝ)
variable (original_price_positive : original_price > 0)

-- Define the purchase price
def purchase_price (original_price : ℝ) : ℝ := original_price * (1 - discount_rate)

-- Define the selling price
def selling_price (original_price : ℝ) : ℝ := original_price * 1.24

-- Theorem statement
theorem discount_calculation (original_price : ℝ) (original_price_positive : original_price > 0) :
  selling_price original_price = purchase_price original_price * 1.55 := by
  sorry

#check discount_calculation

end discount_calculation_l3686_368652


namespace ceremonial_team_arrangements_l3686_368650

def total_boys : ℕ := 48
def total_girls : ℕ := 32

def is_valid_arrangement (n : ℕ) : Prop :=
  n > 1 ∧
  total_boys % n = 0 ∧
  total_girls % n = 0 ∧
  (total_boys / n) = (total_girls / n) * 3 / 2

theorem ceremonial_team_arrangements :
  {n : ℕ | is_valid_arrangement n} = {2, 4, 8, 16} :=
by sorry

end ceremonial_team_arrangements_l3686_368650


namespace equidistant_chord_length_l3686_368631

theorem equidistant_chord_length 
  (d : ℝ) 
  (c1 c2 : ℝ) 
  (dist : ℝ) 
  (h1 : d = 20) 
  (h2 : c1 = 10) 
  (h3 : c2 = 14) 
  (h4 : dist = 6) :
  ∃ (x : ℝ), x^2 = 164 ∧ 
  (∃ (y : ℝ), y > 0 ∧ y < dist ∧
    (d/2)^2 = (c1/2)^2 + y^2 ∧
    (d/2)^2 = (c2/2)^2 + (dist - y)^2 ∧
    x^2/4 + (y + (dist - y)/2)^2 = (d/2)^2) :=
by sorry

end equidistant_chord_length_l3686_368631


namespace max_value_of_x_l3686_368640

theorem max_value_of_x (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (prod_sum_eq : x * y + x * z + y * z = 3) : 
  x ≤ 1 + Real.sqrt 2 := by
sorry

end max_value_of_x_l3686_368640


namespace student_committee_size_l3686_368669

theorem student_committee_size (ways_to_select : ℕ) (h : ways_to_select = 42) :
  ∃ n : ℕ, n > 1 ∧ n * (n - 1) = ways_to_select ∧ n = 7 := by
  sorry

end student_committee_size_l3686_368669


namespace parabola_chord_constant_l3686_368666

/-- Given a parabola y = 2x^2 and a point C(0, c), if t = 1/AC + 1/BC is constant
    for all chords AB passing through C, then t = -20/(7c) -/
theorem parabola_chord_constant (c : ℝ) :
  let parabola := fun (x : ℝ) => 2 * x^2
  let C := (0, c)
  let chord_length (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  ∃ t : ℝ, ∀ A B : ℝ × ℝ,
    A.2 = parabola A.1 →
    B.2 = parabola B.1 →
    (∃ m b : ℝ, ∀ x : ℝ, m * x + b = parabola x ↔ (x = A.1 ∨ x = B.1)) →
    C.2 = m * C.1 + b →
    t = 1 / chord_length A C + 1 / chord_length B C →
    t = -20 / (7 * c) :=
by
  sorry


end parabola_chord_constant_l3686_368666


namespace smallest_number_with_remainders_l3686_368657

theorem smallest_number_with_remainders : ∃! n : ℕ,
  (∀ k ∈ Finset.range 10, n % (k + 3) = k + 2) ∧
  (∀ m : ℕ, m < n → ∃ k ∈ Finset.range 10, m % (k + 3) ≠ k + 2) :=
by
  use 27719
  sorry

end smallest_number_with_remainders_l3686_368657


namespace root_difference_zero_l3686_368620

theorem root_difference_zero (x : ℝ) : 
  (x ^ 2 + 30 * x + 225 = 0) → (abs (x - x) = 0) := by
  sorry

end root_difference_zero_l3686_368620


namespace anya_lost_games_l3686_368635

/-- Represents a girl in the table tennis game --/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents a game of table tennis --/
structure Game where
  number : Nat
  players : Fin 2 → Girl
  loser : Girl

/-- The total number of games played --/
def total_games : Nat := 19

/-- The number of games each girl played --/
def games_played (g : Girl) : Nat :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- Predicate to check if a girl lost a specific game --/
def lost_game (g : Girl) (n : Nat) : Prop := ∃ game : Game, game.number = n ∧ game.loser = g

/-- The main theorem to prove --/
theorem anya_lost_games :
  (lost_game Girl.Anya 4) ∧
  (lost_game Girl.Anya 8) ∧
  (lost_game Girl.Anya 12) ∧
  (lost_game Girl.Anya 16) ∧
  (∀ n : Nat, n ≤ total_games → n ≠ 4 → n ≠ 8 → n ≠ 12 → n ≠ 16 → ¬(lost_game Girl.Anya n)) :=
sorry

end anya_lost_games_l3686_368635


namespace parabola_hyperbola_foci_coincide_l3686_368629

/-- The value of n for which the focus of the parabola y^2 = 8x coincides with 
    one of the foci of the hyperbola x^2/3 - y^2/n = 1 -/
theorem parabola_hyperbola_foci_coincide : ∃ n : ℝ,
  (∀ x y : ℝ, y^2 = 8*x → x^2/3 - y^2/n = 1) ∧ 
  (∃ x y : ℝ, y^2 = 8*x ∧ x^2/3 - y^2/n = 1 ∧ x = 2 ∧ y = 0) →
  n = 1 := by
  sorry

end parabola_hyperbola_foci_coincide_l3686_368629


namespace friday_to_monday_ratio_l3686_368632

def num_rabbits : ℕ := 16
def monday_toys : ℕ := 6
def wednesday_toys : ℕ := 2 * monday_toys
def saturday_toys : ℕ := wednesday_toys / 2
def toys_per_rabbit : ℕ := 3

def total_toys : ℕ := num_rabbits * toys_per_rabbit

def friday_toys : ℕ := total_toys - monday_toys - wednesday_toys - saturday_toys

theorem friday_to_monday_ratio :
  friday_toys / monday_toys = 4 ∧ friday_toys % monday_toys = 0 := by
  sorry

end friday_to_monday_ratio_l3686_368632


namespace income_ratio_l3686_368616

/-- Represents the financial data of a person --/
structure PersonFinance where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup --/
def problem_setup (p1 p2 : PersonFinance) : Prop :=
  p1.income = 4000 ∧
  p1.savings = 1600 ∧
  p2.savings = 1600 ∧
  3 * p2.expenditure = 2 * p1.expenditure ∧
  p1.savings = p1.income - p1.expenditure ∧
  p2.savings = p2.income - p2.expenditure

/-- The theorem to be proved --/
theorem income_ratio (p1 p2 : PersonFinance) :
  problem_setup p1 p2 → 5 * p2.income = 4 * p1.income :=
by
  sorry


end income_ratio_l3686_368616


namespace area_triangle_ADG_l3686_368679

/-- Regular octagon with side length 3 -/
structure RegularOctagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Triangle ADG in the regular octagon -/
def TriangleADG (octagon : RegularOctagon) : Set (Fin 3 → ℝ × ℝ) :=
  sorry

/-- Area of a triangle -/
def triangleArea (triangle : Set (Fin 3 → ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: Area of triangle ADG in a regular octagon with side length 3 -/
theorem area_triangle_ADG (octagon : RegularOctagon) :
  triangleArea (TriangleADG octagon) = (27 - 9 * Real.sqrt 2 + 9 * Real.sqrt (2 - 2 * Real.sqrt 2)) / (2 * Real.sqrt 2) :=
by sorry

end area_triangle_ADG_l3686_368679


namespace min_tangent_product_right_triangle_l3686_368618

theorem min_tangent_product_right_triangle (A B C : Real) : 
  0 < A → A < π / 2 →
  0 < B → B < π / 2 →
  C ≤ π / 2 →
  A + B + C = π →
  (∀ A' B' C', 0 < A' → A' < π / 2 → 0 < B' → B' < π / 2 → C' ≤ π / 2 → A' + B' + C' = π → 
    Real.tan A * Real.tan B ≤ Real.tan A' * Real.tan B') →
  C = π / 2 :=
by sorry

end min_tangent_product_right_triangle_l3686_368618


namespace cube_sum_from_sixth_power_sum_l3686_368622

theorem cube_sum_from_sixth_power_sum (x : ℝ) (h : 47 = x^6 + 1/x^6) : x^3 + 1/x^3 = 7 := by
  sorry

end cube_sum_from_sixth_power_sum_l3686_368622


namespace sqrt_sum_squared_l3686_368639

theorem sqrt_sum_squared (x y z : ℝ) : 
  (Real.sqrt 80 + 3 * Real.sqrt 5 + Real.sqrt 450 / 3)^2 = 295 := by
  sorry

end sqrt_sum_squared_l3686_368639


namespace problem_statement_l3686_368699

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y + x + y = 7) : 
  x^2*y + x*y^2 = 49 := by
sorry

end problem_statement_l3686_368699


namespace gain_percent_is_one_percent_l3686_368647

-- Define the gain and cost price
def gain : ℚ := 70 / 100  -- 70 paise = 0.70 Rs
def cost_price : ℚ := 70  -- 70 Rs

-- Define the gain percent formula
def gain_percent (g c : ℚ) : ℚ := (g / c) * 100

-- Theorem statement
theorem gain_percent_is_one_percent :
  gain_percent gain cost_price = 1 := by
  sorry

end gain_percent_is_one_percent_l3686_368647


namespace james_writing_time_l3686_368683

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  pages_per_day_per_person : ℕ
  people_per_day : ℕ

/-- Calculate the hours spent writing per week -/
def hours_per_week (s : WritingScenario) : ℕ :=
  let pages_per_day := s.pages_per_day_per_person * s.people_per_day
  let pages_per_week := pages_per_day * 7
  pages_per_week / s.pages_per_hour

/-- Theorem: James spends 7 hours a week writing -/
theorem james_writing_time :
  let james := WritingScenario.mk 10 5 2
  hours_per_week james = 7 := by
  sorry

end james_writing_time_l3686_368683


namespace max_k_value_l3686_368641

/-- The maximum value of k satisfying the given inequality -/
theorem max_k_value (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a + b + c = 1) :
  (∃ k : ℝ, ∀ a b c : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 1 →
    (a / (1 + 9*b*c + k*(b-c)^2)) + (b / (1 + 9*c*a + k*(c-a)^2)) + (c / (1 + 9*a*b + k*(a-b)^2)) ≥ 1/2) ∧
  (∀ k' : ℝ, k' > 4 →
    ∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 ∧
      (a / (1 + 9*b*c + k'*(b-c)^2)) + (b / (1 + 9*c*a + k'*(c-a)^2)) + (c / (1 + 9*a*b + k'*(a-b)^2)) < 1/2) :=
by sorry

end max_k_value_l3686_368641


namespace interest_difference_proof_l3686_368626

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem interest_difference_proof : 
  let principal : ℚ := 2500
  let rate : ℚ := 8
  let time : ℚ := 8
  let interest := simple_interest principal rate time
  principal - interest = 900 := by
  sorry

end interest_difference_proof_l3686_368626


namespace total_leaves_on_ferns_l3686_368628

theorem total_leaves_on_ferns : 
  let total_ferns : ℕ := 12
  let type_a_ferns : ℕ := 4
  let type_b_ferns : ℕ := 5
  let type_c_ferns : ℕ := 3
  let type_a_fronds : ℕ := 15
  let type_a_leaves_per_frond : ℕ := 45
  let type_b_fronds : ℕ := 20
  let type_b_leaves_per_frond : ℕ := 30
  let type_c_fronds : ℕ := 25
  let type_c_leaves_per_frond : ℕ := 40

  total_ferns = type_a_ferns + type_b_ferns + type_c_ferns →
  (type_a_ferns * type_a_fronds * type_a_leaves_per_frond +
   type_b_ferns * type_b_fronds * type_b_leaves_per_frond +
   type_c_ferns * type_c_fronds * type_c_leaves_per_frond) = 8700 :=
by
  sorry

#check total_leaves_on_ferns

end total_leaves_on_ferns_l3686_368628


namespace canada_avg_sqft_approx_l3686_368608

/-- The population of Canada in the year 2000 -/
def canada_population : ℕ := 30690000

/-- The total area of Canada in square miles -/
def canada_area : ℕ := 3855103

/-- The number of square feet in one square mile -/
def sqft_per_sqmile : ℕ := 5280 * 5280

/-- The average number of square feet per person in Canada -/
def avg_sqft_per_person : ℚ :=
  (canada_area * sqft_per_sqmile) / canada_population

/-- Theorem stating that the average square feet per person in Canada 
    is approximately 3,000,000 -/
theorem canada_avg_sqft_approx :
  ∃ ε > 0, |avg_sqft_per_person - 3000000| < ε :=
sorry

end canada_avg_sqft_approx_l3686_368608


namespace equation_solutions_l3686_368609

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 11 ∧ x₂ = 3 - Real.sqrt 11 ∧
    x₁^2 - 6*x₁ - 2 = 0 ∧ x₂^2 - 6*x₂ - 2 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = -2 ∧
    (2*y₁ + 1)^2 = -6*y₁ - 3 ∧ (2*y₂ + 1)^2 = -6*y₂ - 3) :=
by sorry

end equation_solutions_l3686_368609


namespace min_slide_time_l3686_368667

/-- A vertical circle fixed to a horizontal line -/
structure VerticalCircle where
  center : ℝ × ℝ
  radius : ℝ
  is_vertical : center.2 = radius

/-- A point outside and above the circle -/
structure OutsidePoint (C : VerticalCircle) where
  coords : ℝ × ℝ
  is_outside : (coords.1 - C.center.1)^2 + (coords.2 - C.center.2)^2 > C.radius^2
  is_above : coords.2 > C.center.2 + C.radius

/-- A point on the circle -/
def CirclePoint (C : VerticalCircle) := { p : ℝ × ℝ // (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2 }

/-- The time function for a particle to slide down from P to Q under gravity -/
noncomputable def slide_time (C : VerticalCircle) (P : OutsidePoint C) (Q : CirclePoint C) : ℝ := sorry

/-- The lowest point on the circle -/
def lowest_point (C : VerticalCircle) : CirclePoint C :=
  ⟨(C.center.1, C.center.2 - C.radius), sorry⟩

/-- Theorem: The point Q that minimizes the slide time is the lowest point on the circle -/
theorem min_slide_time (C : VerticalCircle) (P : OutsidePoint C) :
  ∀ Q : CirclePoint C, slide_time C P Q ≥ slide_time C P (lowest_point C) :=
sorry

end min_slide_time_l3686_368667


namespace gcd_1239_2829_times_15_l3686_368649

theorem gcd_1239_2829_times_15 : 15 * Int.gcd 1239 2829 = 315 := by
  sorry

end gcd_1239_2829_times_15_l3686_368649


namespace hemisphere_intersection_area_l3686_368634

/-- Given two hemispheres A and B, where A has a surface area of 50π, and B has twice the surface area of A,
    if B shares 1/4 of its surface area with A, then the surface area of the remainder of hemisphere B
    after the intersection is 75π. -/
theorem hemisphere_intersection_area (A B : ℝ) : 
  A = 50 * Real.pi →
  B = 2 * A →
  let shared := (1/4) * B
  B - shared = 75 * Real.pi := by sorry

end hemisphere_intersection_area_l3686_368634


namespace divisible_by_4_or_6_count_l3686_368684

def countDivisibleByEither (n : ℕ) (a b : ℕ) : ℕ :=
  (n / a) + (n / b) - (n / (Nat.lcm a b))

theorem divisible_by_4_or_6_count :
  countDivisibleByEither 80 4 6 = 27 := by
  sorry

end divisible_by_4_or_6_count_l3686_368684


namespace ternary_35_implies_k_2_l3686_368638

def ternary_to_decimal (k : ℕ+) : ℕ := 1 * 3^3 + k * 3^2 + 2

theorem ternary_35_implies_k_2 : 
  ∀ k : ℕ+, ternary_to_decimal k = 35 → k = 2 := by
  sorry

end ternary_35_implies_k_2_l3686_368638


namespace infinitely_many_skew_lines_l3686_368601

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Predicate to check if a line intersects a plane -/
def intersects (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is perpendicular to a plane -/
def perpendicular (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if two lines are skew -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if a line is within a plane -/
def within_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- The main theorem -/
theorem infinitely_many_skew_lines 
  (l : Line3D) (α : Plane3D) 
  (h1 : intersects l α) 
  (h2 : ¬perpendicular l α) :
  ∃ S : Set Line3D, (∀ l' ∈ S, within_plane l' α ∧ skew l l') ∧ Set.Infinite S :=
sorry

end infinitely_many_skew_lines_l3686_368601


namespace factor_divisor_statements_l3686_368675

theorem factor_divisor_statements : 
  (∃ n : ℤ, 24 = 4 * n) ∧ 
  (∃ n : ℤ, 209 = 19 * n) ∧ 
  ¬(∃ n : ℤ, 63 = 19 * n) ∧
  (∃ n : ℤ, 180 = 9 * n) := by
sorry

end factor_divisor_statements_l3686_368675


namespace cycle_selling_price_l3686_368698

/-- Given a cycle bought for a certain price with a specific gain percent,
    calculate the selling price. -/
def selling_price (cost_price : ℚ) (gain_percent : ℚ) : ℚ :=
  cost_price * (1 + gain_percent / 100)

/-- Theorem: The selling price of a cycle bought for Rs. 675 with a 60% gain is Rs. 1080 -/
theorem cycle_selling_price :
  selling_price 675 60 = 1080 := by
  sorry

end cycle_selling_price_l3686_368698


namespace dans_cards_l3686_368685

theorem dans_cards (initial_cards : ℕ) (bought_cards : ℕ) (total_cards : ℕ) : 
  initial_cards = 27 → bought_cards = 20 → total_cards = 88 → 
  total_cards - bought_cards - initial_cards = 41 := by
sorry

end dans_cards_l3686_368685


namespace eve_gift_cost_l3686_368665

def hand_mitts_cost : ℝ := 14
def apron_cost : ℝ := 16
def utensils_cost : ℝ := 10
def knife_cost : ℝ := 2 * utensils_cost
def discount_percentage : ℝ := 0.25
def num_nieces : ℕ := 3

def total_cost_per_niece : ℝ := hand_mitts_cost + apron_cost + utensils_cost + knife_cost

def total_cost_before_discount : ℝ := num_nieces * total_cost_per_niece

def discount_amount : ℝ := discount_percentage * total_cost_before_discount

theorem eve_gift_cost : total_cost_before_discount - discount_amount = 135 := by
  sorry

end eve_gift_cost_l3686_368665


namespace geometric_sequence_inequality_l3686_368671

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: For a geometric sequence, a₁ < a₃ if and only if a₅ < a₇ -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 1 < a 3 ↔ a 5 < a 7 := by
  sorry

end geometric_sequence_inequality_l3686_368671


namespace intersection_of_A_and_B_l3686_368658

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l3686_368658


namespace distance_calculation_l3686_368603

theorem distance_calculation (speed_to : ℝ) (speed_from : ℝ) (total_time : ℝ) 
  (h1 : speed_to = 50)
  (h2 : speed_from = 75)
  (h3 : total_time = 10) :
  (total_time * speed_to * speed_from) / (speed_to + speed_from) = 300 :=
by sorry

end distance_calculation_l3686_368603


namespace students_surveyed_students_surveyed_proof_l3686_368655

theorem students_surveyed : ℕ :=
  let total_students : ℕ := sorry
  let french_speakers : ℕ := sorry
  let french_english_speakers : ℕ := 10
  let french_only_speakers : ℕ := 40

  have h1 : french_speakers = french_english_speakers + french_only_speakers := by sorry
  have h2 : french_speakers = 50 := by sorry
  have h3 : french_speakers = total_students / 4 := by sorry

  200

theorem students_surveyed_proof : students_surveyed = 200 := by sorry

end students_surveyed_students_surveyed_proof_l3686_368655


namespace sum_of_fractions_l3686_368602

theorem sum_of_fractions : (6 : ℚ) / 5 + (1 : ℚ) / 10 = (13 : ℚ) / 10 := by
  sorry

end sum_of_fractions_l3686_368602


namespace library_books_percentage_l3686_368604

theorem library_books_percentage (total_books adult_books : ℕ) 
  (h1 : total_books = 160)
  (h2 : adult_books = 104) :
  (total_books - adult_books : ℚ) / total_books * 100 = 35 := by
sorry

end library_books_percentage_l3686_368604


namespace factorization_1_l3686_368662

theorem factorization_1 (m n : ℝ) :
  3 * m^2 * n - 12 * m * n + 12 * n = 3 * n * (m - 2)^2 := by sorry

end factorization_1_l3686_368662


namespace right_triangle_area_l3686_368674

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 72 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 216 :=
by sorry

end right_triangle_area_l3686_368674


namespace triangle_tangent_ratio_l3686_368611

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(B) - b * cos(A) = 3/5 * c, then tan(A) / tan(B) = 4 -/
theorem triangle_tangent_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π → B > 0 → B < π → C > 0 → C < π →
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = 3/5 * c →
  Real.tan A / Real.tan B = 4 := by
  sorry

end triangle_tangent_ratio_l3686_368611


namespace average_cost_28_apples_l3686_368695

/-- Represents the cost and quantity of apples in a bundle --/
structure AppleBundle where
  quantity : ℕ
  cost : ℕ

/-- Calculates the total number of apples received when purchasing a given amount --/
def totalApples (purchased : ℕ) : ℕ :=
  if purchased ≥ 20 then purchased + 5 else purchased

/-- Calculates the total cost of apples purchased --/
def totalCost (purchased : ℕ) : ℕ :=
  let bundle1 : AppleBundle := ⟨4, 15⟩
  let bundle2 : AppleBundle := ⟨7, 25⟩
  (purchased / bundle2.quantity) * bundle2.cost

/-- Theorem stating the average cost per apple when purchasing 28 apples --/
theorem average_cost_28_apples :
  (totalCost 28 : ℚ) / (totalApples 28 : ℚ) = 100 / 33 := by
  sorry

#check average_cost_28_apples

end average_cost_28_apples_l3686_368695


namespace mean_of_smallest_elements_l3686_368645

/-- F(n, r) represents the arithmetic mean of the smallest elements in all r-element subsets of {1, 2, ..., n} -/
def F (n r : ℕ) : ℚ :=
  sorry

/-- Theorem stating that F(n, r) = (n+1)/(r+1) for 1 ≤ r ≤ n -/
theorem mean_of_smallest_elements (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  F n r = (n + 1 : ℚ) / (r + 1 : ℚ) := by
  sorry

end mean_of_smallest_elements_l3686_368645


namespace greatest_value_b_l3686_368661

theorem greatest_value_b (b : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + 3 < -x + 6 → x ≤ b) ↔ b = (3 + Real.sqrt 21) / 2 :=
sorry

end greatest_value_b_l3686_368661


namespace paint_cube_cost_l3686_368623

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost 
  (paint_cost : ℝ)       -- Cost of paint per kg in Rs
  (paint_coverage : ℝ)   -- Area covered by 1 kg of paint in sq. ft
  (cube_side : ℝ)        -- Length of cube side in feet
  (h1 : paint_cost = 20) -- Paint costs 20 Rs per kg
  (h2 : paint_coverage = 15) -- 1 kg of paint covers 15 sq. ft
  (h3 : cube_side = 5)   -- Cube side is 5 feet
  : ℝ :=
by
  -- The cost to paint the cube is 200 Rs
  sorry

#check paint_cube_cost

end paint_cube_cost_l3686_368623


namespace angle_from_point_l3686_368689

theorem angle_from_point (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∃ (P : ℝ × ℝ), P.1 = Real.sin (3 * Real.pi / 4) ∧ 
                   P.2 = Real.cos (3 * Real.pi / 4) ∧ 
                   P.1 = Real.sin θ ∧ 
                   P.2 = Real.cos θ) →
  θ = 7 * Real.pi / 4 := by
sorry

end angle_from_point_l3686_368689


namespace gcd_105_490_l3686_368606

theorem gcd_105_490 : Nat.gcd 105 490 = 35 := by
  sorry

end gcd_105_490_l3686_368606


namespace area_between_parabola_and_line_l3686_368694

theorem area_between_parabola_and_line : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := x
  let area := ∫ x in (0:ℝ)..1, (g x - f x)
  area = 1/6 := by sorry

end area_between_parabola_and_line_l3686_368694


namespace carson_counted_six_clouds_l3686_368663

/-- The number of clouds Carson counted that look like funny animals -/
def carson_clouds : ℕ := sorry

/-- The number of clouds Carson's little brother counted that look like dragons -/
def brother_clouds : ℕ := sorry

/-- The total number of clouds counted -/
def total_clouds : ℕ := 24

theorem carson_counted_six_clouds :
  carson_clouds = 6 ∧
  brother_clouds = 3 * carson_clouds ∧
  carson_clouds + brother_clouds = total_clouds :=
sorry

end carson_counted_six_clouds_l3686_368663


namespace sticker_distribution_l3686_368677

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 8 identical stickers among 4 distinct sheets of paper -/
theorem sticker_distribution : distribute 8 4 = 165 := by sorry

end sticker_distribution_l3686_368677


namespace max_scores_is_45_l3686_368659

/-- Represents a test with multiple-choice questions. -/
structure Test where
  num_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  unanswered_points : ℤ

/-- Calculates the maximum number of different possible total scores for a given test. -/
def max_different_scores (t : Test) : ℕ :=
  sorry

/-- The specific test described in the problem. -/
def problem_test : Test :=
  { num_questions := 10
  , correct_points := 4
  , incorrect_points := -1
  , unanswered_points := 0 }

/-- Theorem stating that the maximum number of different possible total scores for the problem_test is 45. -/
theorem max_scores_is_45 : max_different_scores problem_test = 45 := by
  sorry

end max_scores_is_45_l3686_368659


namespace least_positive_integer_congruence_l3686_368642

theorem least_positive_integer_congruence :
  ∃! y : ℕ+, y.val + 3077 ≡ 1456 [ZMOD 15] ∧
  ∀ z : ℕ+, z.val + 3077 ≡ 1456 [ZMOD 15] → y ≤ z ∧ y.val = 14 :=
by sorry

end least_positive_integer_congruence_l3686_368642


namespace perpendicular_line_equation_l3686_368600

/-- The equation of a line perpendicular to x + y = 0 and passing through (-1, 0) -/
theorem perpendicular_line_equation :
  let l1 : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}  -- Line x + y = 0
  let point : ℝ × ℝ := (-1, 0)                -- Given point
  let l2 : Set (ℝ × ℝ) := {p | p.1 - p.2 + 1 = 0}  -- Claimed perpendicular line
  (∀ p ∈ l2, (p.1 - point.1) * (p.1 + p.2) = -(p.2 - point.2) * (p.1 + p.2)) ∧  -- Perpendicularity condition
  (point ∈ l2)  -- Point (-1, 0) lies on the line
  :=
by
  sorry

end perpendicular_line_equation_l3686_368600


namespace equation_solutions_l3686_368691

theorem equation_solutions :
  (∀ x : ℝ, (x - 4)^2 - 9 = 0 ↔ x = 7 ∨ x = 1) ∧
  (∀ x : ℝ, (x + 1)^3 = -27 ↔ x = -4) := by sorry

end equation_solutions_l3686_368691


namespace fractional_equation_solution_l3686_368653

theorem fractional_equation_solution : 
  ∃! x : ℝ, (x ≠ 0 ∧ x ≠ 2) ∧ (5 / x = 7 / (x - 2)) := by
  sorry

end fractional_equation_solution_l3686_368653


namespace parabola_sum_l3686_368681

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) :
  p.x_coord (-6) = 7 →  -- vertex condition
  p.x_coord 0 = 5 →     -- point condition
  p.a + p.b + p.c = -32/3 := by
sorry

end parabola_sum_l3686_368681
