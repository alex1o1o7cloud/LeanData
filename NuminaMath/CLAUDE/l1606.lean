import Mathlib

namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1606_160608

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1606_160608


namespace NUMINAMATH_CALUDE_cube_volume_from_face_area_l1606_160678

theorem cube_volume_from_face_area (face_area : ℝ) (volume : ℝ) :
  face_area = 16 →
  volume = face_area ^ (3/2) →
  volume = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_area_l1606_160678


namespace NUMINAMATH_CALUDE_product_difference_sum_l1606_160633

theorem product_difference_sum (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  A * B = 72 →
  C * D = 72 →
  A - B = C + D →
  A = 18 := by
sorry

end NUMINAMATH_CALUDE_product_difference_sum_l1606_160633


namespace NUMINAMATH_CALUDE_triangle_value_l1606_160684

theorem triangle_value (Δ q : ℤ) 
  (h1 : 3 * Δ * q = 63) 
  (h2 : 7 * (Δ + q) = 161) : 
  Δ = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l1606_160684


namespace NUMINAMATH_CALUDE_green_ball_probability_l1606_160623

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The probability of selecting a green ball given the problem conditions -/
theorem green_ball_probability :
  let containerX : Container := ⟨5, 5⟩
  let containerY : Container := ⟨8, 2⟩
  let containerZ : Container := ⟨3, 7⟩
  let totalContainers : ℕ := 3
  (1 : ℚ) / totalContainers * (greenProbability containerX +
                               greenProbability containerY +
                               greenProbability containerZ) = 7 / 15 := by
  sorry


end NUMINAMATH_CALUDE_green_ball_probability_l1606_160623


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1606_160660

/-- Proves that the overall average score for a cricketer who played 7 matches
    with given averages for the first 4 and last 3 matches is 56. -/
theorem cricketer_average_score 
  (total_matches : ℕ)
  (first_matches : ℕ)
  (last_matches : ℕ)
  (first_average : ℚ)
  (last_average : ℚ)
  (h1 : total_matches = 7)
  (h2 : first_matches = 4)
  (h3 : last_matches = 3)
  (h4 : first_matches + last_matches = total_matches)
  (h5 : first_average = 46)
  (h6 : last_average = 69333333333333 / 1000000000000) : 
  (first_average * first_matches + last_average * last_matches) / total_matches = 56 := by
sorry

#eval (46 * 4 + 69333333333333 / 1000000000000 * 3) / 7

end NUMINAMATH_CALUDE_cricketer_average_score_l1606_160660


namespace NUMINAMATH_CALUDE_triangle_area_l1606_160693

theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 36 → inradius = 2.5 → area = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1606_160693


namespace NUMINAMATH_CALUDE_ella_jasper_passing_count_l1606_160621

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the track in meters
  direction : ℝ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (duration : ℝ) : ℕ :=
  sorry

/-- Theorem: Ella and Jasper pass each other 93 times during their 40-minute jog -/
theorem ella_jasper_passing_count : 
  let ella : Runner := { speed := 300, radius := 40, direction := 1 }
  let jasper : Runner := { speed := 360, radius := 50, direction := -1 }
  passingCount ella jasper 40 = 93 := by
  sorry

end NUMINAMATH_CALUDE_ella_jasper_passing_count_l1606_160621


namespace NUMINAMATH_CALUDE_player_A_win_probability_l1606_160683

/-- The probability of winning a single game for either player -/
def win_prob : ℚ := 1/2

/-- The number of games player A needs to win to become the final winner -/
def games_needed_A : ℕ := 2

/-- The number of games player B needs to win to become the final winner -/
def games_needed_B : ℕ := 3

/-- The probability of player A becoming the final winner -/
def prob_A_wins : ℚ := 11/16

theorem player_A_win_probability :
  prob_A_wins = 11/16 := by sorry

end NUMINAMATH_CALUDE_player_A_win_probability_l1606_160683


namespace NUMINAMATH_CALUDE_soccer_league_games_l1606_160663

theorem soccer_league_games (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1606_160663


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l1606_160651

theorem chinese_remainder_theorem_example :
  ∃! x : ℕ, x < 504 ∧ 
    x % 7 = 1 ∧
    x % 8 = 1 ∧
    x % 9 = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l1606_160651


namespace NUMINAMATH_CALUDE_sector_area_l1606_160627

theorem sector_area (circumference : Real) (central_angle : Real) :
  circumference = 8 * π / 9 + 4 →
  central_angle = 80 * π / 180 →
  (1 / 2) * (circumference - 2 * (circumference / (2 * π + central_angle))) ^ 2 * central_angle / (2 * π) = 8 * π / 9 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1606_160627


namespace NUMINAMATH_CALUDE_divisibility_of_three_digit_numbers_l1606_160667

theorem divisibility_of_three_digit_numbers (n : ℕ) (h1 : n ≥ 100) (h2 : n ≤ 999) :
  (∃ (S : Finset ℕ), S.card = 19 ∧ 
   (∀ m ∈ S, 100 ≤ m ∧ m ≤ 999 ∧ (m % 78 = 0 ∨ m % n = 0))) →
  n = 78 := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_three_digit_numbers_l1606_160667


namespace NUMINAMATH_CALUDE_quadratic_no_roots_implies_line_not_in_third_quadrant_l1606_160611

theorem quadratic_no_roots_implies_line_not_in_third_quadrant 
  (m : ℝ) (h : ∀ x : ℝ, m * x^2 - 2*x - 1 ≠ 0) :
  ∀ x y : ℝ, y = m*x - m → ¬(x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_implies_line_not_in_third_quadrant_l1606_160611


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l1606_160668

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : Line

/-- The equation of a parabola in general form -/
def parabola_equation (p : Parabola) (x y : ℝ) : ℝ :=
  25 * x^2 + 40 * x * y + 16 * y^2 - 650 * x + 184 * y + 1009

theorem parabola_equation_correct (p : Parabola) :
  p.focus = Point.mk 5 (-2) →
  p.directrix = Line.mk 4 (-5) (-20) →
  ∀ x y : ℝ, (x - p.focus.x)^2 + (y - p.focus.y)^2 = 
    ((p.directrix.a * x + p.directrix.b * y + p.directrix.c)^2 / 
     (p.directrix.a^2 + p.directrix.b^2)) →
  parabola_equation p x y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l1606_160668


namespace NUMINAMATH_CALUDE_vacation_cost_problem_l1606_160697

/-- The vacation cost problem -/
theorem vacation_cost_problem (sarah_paid derek_paid rita_paid : ℚ)
  (h_sarah : sarah_paid = 150)
  (h_derek : derek_paid = 210)
  (h_rita : rita_paid = 240)
  (s d : ℚ) :
  let total_paid := sarah_paid + derek_paid + rita_paid
  let equal_share := total_paid / 3
  let sarah_owes := equal_share - sarah_paid
  let derek_owes := equal_share - derek_paid
  s = sarah_owes ∧ d = derek_owes →
  s - d = 60 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_problem_l1606_160697


namespace NUMINAMATH_CALUDE_ticket_price_calculation_l1606_160665

def commission_rate : ℝ := 0.12
def desired_net_amount : ℝ := 22

theorem ticket_price_calculation :
  ∃ (price : ℝ), price * (1 - commission_rate) = desired_net_amount ∧ price = 25 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_calculation_l1606_160665


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l1606_160686

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 2}
def N : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = {x : ℝ | x < 2 ∨ x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l1606_160686


namespace NUMINAMATH_CALUDE_wire_length_theorem_l1606_160625

/-- Represents the wire and pole configuration -/
structure WireConfig where
  initial_poles : ℕ
  initial_distance : ℝ
  new_distance_increase : ℝ
  total_length : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem wire_length_theorem (config : WireConfig) 
  (h1 : config.initial_poles = 26)
  (h2 : config.new_distance_increase = 5/3)
  (h3 : (config.initial_poles - 1) * (config.initial_distance + config.new_distance_increase) = config.initial_poles * config.initial_distance - config.initial_distance) :
  config.total_length = 1000 := by
  sorry


end NUMINAMATH_CALUDE_wire_length_theorem_l1606_160625


namespace NUMINAMATH_CALUDE_rectangle_height_calculation_l1606_160654

/-- Represents a rectangle with a base and height in centimeters -/
structure Rectangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.base * r.height

theorem rectangle_height_calculation (r : Rectangle) 
  (h_base : r.base = 9)
  (h_area : area r = 33.3) :
  r.height = 3.7 := by
sorry


end NUMINAMATH_CALUDE_rectangle_height_calculation_l1606_160654


namespace NUMINAMATH_CALUDE_clock_hand_overlaps_l1606_160647

/-- Represents the number of revolutions a clock hand makes in a day -/
structure ClockHand where
  revolutions : ℕ

/-- Calculates the number of overlaps between two clock hands in a day -/
def overlaps (hand1 hand2 : ClockHand) : ℕ :=
  hand2.revolutions - hand1.revolutions

theorem clock_hand_overlaps :
  let hour_hand : ClockHand := ⟨2⟩
  let minute_hand : ClockHand := ⟨24⟩
  let second_hand : ClockHand := ⟨1440⟩
  (overlaps hour_hand minute_hand = 22) ∧
  (overlaps minute_hand second_hand = 1416) :=
by sorry

end NUMINAMATH_CALUDE_clock_hand_overlaps_l1606_160647


namespace NUMINAMATH_CALUDE_roots_modulus_one_preserved_l1606_160644

theorem roots_modulus_one_preserved (a b c : ℂ) :
  (∀ z : ℂ, z^3 + a*z^2 + b*z + c = 0 → Complex.abs z = 1) →
  (∀ w : ℂ, w^3 + Complex.abs a * w^2 + Complex.abs b * w + Complex.abs c = 0 → Complex.abs w = 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_modulus_one_preserved_l1606_160644


namespace NUMINAMATH_CALUDE_last_i_becomes_w_l1606_160643

/-- Represents a letter in the alphabet --/
def Letter := Fin 26

/-- The encryption shift for the nth occurrence of a letter --/
def shift (n : Nat) : Nat := n^2

/-- The message to be encrypted --/
def message : String := "Mathematics is meticulous"

/-- Count occurrences of a character in a string --/
def countOccurrences (c : Char) (s : String) : Nat :=
  s.toList.filter (· = c) |>.length

/-- Apply the shift to a letter --/
def applyShift (l : Letter) (s : Nat) : Letter :=
  ⟨(l.val + s) % 26, by sorry⟩

/-- The theorem to be proved --/
theorem last_i_becomes_w :
  let iCount := countOccurrences 'i' message
  let totalShift := (List.range iCount).map shift |>.sum
  let iLetter : Letter := ⟨8, by sorry⟩  -- 'i' is the 9th letter (0-indexed)
  applyShift iLetter totalShift = ⟨22, by sorry⟩  -- 'w' is the 23rd letter (0-indexed)
  := by sorry

end NUMINAMATH_CALUDE_last_i_becomes_w_l1606_160643


namespace NUMINAMATH_CALUDE_complement_of_union_equals_four_l1606_160695

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_equals_four :
  (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_four_l1606_160695


namespace NUMINAMATH_CALUDE_routes_count_l1606_160614

/-- The number of horizontal moves required to reach point B from point A -/
def horizontal_moves : ℕ := 8

/-- The number of vertical moves required to reach point B from point A -/
def vertical_moves : ℕ := 5

/-- The total number of moves required to reach point B from point A -/
def total_moves : ℕ := horizontal_moves + vertical_moves

/-- The number of distinct routes from point A to point B -/
def num_routes : ℕ := Nat.choose total_moves vertical_moves

theorem routes_count : num_routes = 1287 := by
  sorry

end NUMINAMATH_CALUDE_routes_count_l1606_160614


namespace NUMINAMATH_CALUDE_students_at_start_l1606_160648

theorem students_at_start (initial_students final_students left_students new_students : ℕ) :
  final_students = 43 →
  left_students = 3 →
  new_students = 42 →
  initial_students + new_students - left_students = final_students →
  initial_students = 4 := by
sorry

end NUMINAMATH_CALUDE_students_at_start_l1606_160648


namespace NUMINAMATH_CALUDE_house_sale_profit_l1606_160675

/-- Calculates the net profit from a house sale and repurchase --/
def netProfit (initialValue : ℝ) (sellProfit : ℝ) (buyLoss : ℝ) : ℝ :=
  let sellPrice := initialValue * (1 + sellProfit)
  let buyPrice := sellPrice * (1 - buyLoss)
  sellPrice - buyPrice

/-- Theorem stating that the net profit is $1725 given the specified conditions --/
theorem house_sale_profit :
  netProfit 15000 0.15 0.10 = 1725 := by
  sorry

#eval netProfit 15000 0.15 0.10

end NUMINAMATH_CALUDE_house_sale_profit_l1606_160675


namespace NUMINAMATH_CALUDE_substitution_result_l1606_160673

theorem substitution_result (x y : ℝ) :
  (4 * x + 5 * y = 7) ∧ (y = 2 * x - 1) →
  (4 * x + 10 * x - 5 = 7) :=
by sorry

end NUMINAMATH_CALUDE_substitution_result_l1606_160673


namespace NUMINAMATH_CALUDE_terms_before_negative_twenty_l1606_160685

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem terms_before_negative_twenty :
  let a₁ := 100
  let d := -4
  let n := 31
  arithmetic_sequence a₁ d n = -20 ∧ n - 1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_terms_before_negative_twenty_l1606_160685


namespace NUMINAMATH_CALUDE_statue_cost_proof_l1606_160696

theorem statue_cost_proof (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) : 
  selling_price = 670 ∧ 
  profit_percentage = 0.25 ∧ 
  selling_price = original_cost * (1 + profit_percentage) →
  original_cost = 536 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_proof_l1606_160696


namespace NUMINAMATH_CALUDE_nut_boxes_problem_l1606_160612

theorem nut_boxes_problem (first second third : ℕ) : 
  (second = (11 * first) / 10) →
  (second = (13 * third) / 10) →
  (first = third + 80) →
  (first = 520 ∧ second = 572 ∧ third = 440) :=
by sorry

end NUMINAMATH_CALUDE_nut_boxes_problem_l1606_160612


namespace NUMINAMATH_CALUDE_tire_circumference_l1606_160634

/-- Given a tire rotating at 400 revolutions per minute on a car traveling at 144 km/h, 
    the circumference of the tire is 6 meters. -/
theorem tire_circumference (revolutions_per_minute : ℝ) (speed_km_per_hour : ℝ) 
  (h1 : revolutions_per_minute = 400) 
  (h2 : speed_km_per_hour = 144) : 
  let speed_m_per_minute : ℝ := speed_km_per_hour * 1000 / 60
  let circumference : ℝ := speed_m_per_minute / revolutions_per_minute
  circumference = 6 := by
sorry

end NUMINAMATH_CALUDE_tire_circumference_l1606_160634


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l1606_160671

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (marks_per_correct : ℕ) 
  (marks_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 55)
  (h3 : marks_per_correct = 3)
  (h4 : marks_per_incorrect = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums * marks_per_correct - (total_sums - correct_sums) * marks_per_incorrect = total_marks ∧ 
    correct_sums = 23 := by
sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l1606_160671


namespace NUMINAMATH_CALUDE_amusement_park_ticket_price_l1606_160642

/-- Given the following conditions for an amusement park admission:
  * The total cost for admission tickets is $720
  * The price of an adult ticket is $15
  * There are 15 children in the group
  * There are 25 more adults than children
  Prove that the price of a child ticket is $8 -/
theorem amusement_park_ticket_price 
  (total_cost : ℕ) 
  (adult_price : ℕ) 
  (num_children : ℕ) 
  (adult_child_diff : ℕ) 
  (h1 : total_cost = 720)
  (h2 : adult_price = 15)
  (h3 : num_children = 15)
  (h4 : adult_child_diff = 25) :
  ∃ (child_price : ℕ), 
    child_price = 8 ∧ 
    total_cost = adult_price * (num_children + adult_child_diff) + child_price * num_children :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_ticket_price_l1606_160642


namespace NUMINAMATH_CALUDE_probability_five_heads_ten_coins_l1606_160669

theorem probability_five_heads_ten_coins : 
  let n : ℕ := 10  -- total number of coins
  let k : ℕ := 5   -- number of heads we're looking for
  let p : ℚ := 1/2 -- probability of getting heads on a single coin flip
  Nat.choose n k * p^k * (1-p)^(n-k) = 63/256 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_heads_ten_coins_l1606_160669


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_m_l1606_160658

/-- Given a hyperbola with equation y² + x²/m = 1 and asymptote y = ±(√3/3)x, prove that m = -3 -/
theorem hyperbola_asymptote_m (m : ℝ) : 
  (∀ x y : ℝ, y^2 + x^2/m = 1 → (y = (Real.sqrt 3)/3 * x ∨ y = -(Real.sqrt 3)/3 * x)) → 
  m = -3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_m_l1606_160658


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1606_160617

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x + 3 = 0
def equation2 (y : ℝ) : Prop := 4*(2*y - 5)^2 = (3*y - 1)^2

-- Theorem for the first equation
theorem solutions_equation1 : 
  (∃ x : ℝ, equation1 x) ↔ (equation1 1 ∧ equation1 3) :=
sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  (∃ y : ℝ, equation2 y) ↔ (equation2 9 ∧ equation2 (11/7)) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1606_160617


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1606_160618

theorem sqrt_expression_equality : 
  Real.sqrt 25 - Real.sqrt 3 + |Real.sqrt 3 - 2| + ((-8) ^ (1/3 : ℝ)) = 5 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1606_160618


namespace NUMINAMATH_CALUDE_binary_three_is_three_l1606_160638

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of the number 3 -/
def binary_three : List Bool := [true, true]

theorem binary_three_is_three :
  binary_to_decimal binary_three = 3 := by
  sorry

end NUMINAMATH_CALUDE_binary_three_is_three_l1606_160638


namespace NUMINAMATH_CALUDE_max_ratio_theorem_l1606_160662

theorem max_ratio_theorem :
  ∃ (A B : ℝ), 
    (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x^3 + y^4 = x^2*y → x ≤ A ∧ y ≤ B) ∧
    (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^3 + y^4 = x^2*y ∧ x = A) ∧
    (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^3 + y^4 = x^2*y ∧ y = B) ∧
    A/B = 729/1024 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_theorem_l1606_160662


namespace NUMINAMATH_CALUDE_manicure_total_cost_l1606_160609

-- Define the cost of the manicure
def manicure_cost : ℝ := 30

-- Define the tip percentage
def tip_percentage : ℝ := 0.30

-- Define the function to calculate the total amount paid
def total_amount_paid (cost tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- Theorem to prove
theorem manicure_total_cost :
  total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end NUMINAMATH_CALUDE_manicure_total_cost_l1606_160609


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l1606_160666

/-- The function g(x) as defined in the problem -/
def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

/-- Theorem stating that √21/7 is the greatest root of g(x) -/
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 21 / 7 ∧ g r = 0 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l1606_160666


namespace NUMINAMATH_CALUDE_store_items_cost_price_l1606_160655

/-- The cost price of an item given its profit and loss prices -/
def costPrice (profitPrice lossPrice : ℚ) : ℚ := (profitPrice + lossPrice) / 2

/-- The combined cost price of three items -/
def combinedCostPrice (cpA cpB cpC : ℚ) : ℚ := cpA + cpB + cpC

theorem store_items_cost_price : 
  let cpA := costPrice 110 70
  let cpB := costPrice 90 30
  let cpC := costPrice 150 50
  combinedCostPrice cpA cpB cpC = 250 := by
sorry

#eval costPrice 110 70 -- Expected output: 90
#eval costPrice 90 30  -- Expected output: 60
#eval costPrice 150 50 -- Expected output: 100
#eval combinedCostPrice (costPrice 110 70) (costPrice 90 30) (costPrice 150 50) -- Expected output: 250

end NUMINAMATH_CALUDE_store_items_cost_price_l1606_160655


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l1606_160601

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem third_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 4) :
  a 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l1606_160601


namespace NUMINAMATH_CALUDE_circle_properties_l1606_160641

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Define point P
def P : ℝ × ℝ := (4, 5)

-- Theorem statement
theorem circle_properties :
  -- P is on circle C
  C P.1 P.2 ∧
  -- Distance PQ
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 10 ∧
  -- Slope of PQ
  (P.2 - Q.2) / (P.1 - Q.1) = 1/3 ∧
  -- Maximum and minimum distances from Q to any point on C
  (∀ M : ℝ × ℝ, C M.1 M.2 → 
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≤ 6 * Real.sqrt 2 ∧
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≥ 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l1606_160641


namespace NUMINAMATH_CALUDE_container_volume_increase_l1606_160605

/-- Given a container with an initial volume and a volume multiplier, 
    calculate the new volume after applying the multiplier. -/
def new_volume (initial_volume : ℝ) (volume_multiplier : ℝ) : ℝ :=
  initial_volume * volume_multiplier

/-- Theorem: If a container's volume is multiplied by 16, and its original volume was 5 gallons,
    then the new volume is 80 gallons. -/
theorem container_volume_increase :
  let initial_volume : ℝ := 5
  let volume_multiplier : ℝ := 16
  new_volume initial_volume volume_multiplier = 80 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_increase_l1606_160605


namespace NUMINAMATH_CALUDE_train_length_l1606_160636

/-- The length of a train passing a bridge -/
theorem train_length (v : ℝ) (t : ℝ) (b : ℝ) : v = 72 * 1000 / 3600 → t = 25 → b = 140 → v * t - b = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1606_160636


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1606_160622

theorem quadratic_factorization (c d : ℕ) (hc : c > d) :
  (∀ x, x^2 - 18*x + 72 = (x - c)*(x - d)) →
  4*d - c = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1606_160622


namespace NUMINAMATH_CALUDE_a_properties_l1606_160606

def a (n : ℕ+) : ℚ := (n - 1) / n

theorem a_properties :
  (∀ n : ℕ+, a n < 1) ∧
  (∀ n : ℕ+, a (n + 1) > a n) :=
by sorry

end NUMINAMATH_CALUDE_a_properties_l1606_160606


namespace NUMINAMATH_CALUDE_ore_without_alloy_percentage_l1606_160682

/-- Represents the composition of an ore -/
structure Ore where
  alloy_percentage : Real
  iron_in_alloy : Real
  total_ore : Real
  pure_iron : Real

/-- Theorem: The percentage of ore not containing the alloy with iron is 75% -/
theorem ore_without_alloy_percentage (ore : Ore)
  (h1 : ore.alloy_percentage = 0.25)
  (h2 : ore.iron_in_alloy = 0.90)
  (h3 : ore.total_ore = 266.6666666666667)
  (h4 : ore.pure_iron = 60) :
  1 - ore.alloy_percentage = 0.75 := by
  sorry

#check ore_without_alloy_percentage

end NUMINAMATH_CALUDE_ore_without_alloy_percentage_l1606_160682


namespace NUMINAMATH_CALUDE_grid_square_covers_at_least_four_l1606_160637

/-- A square on a grid -/
structure GridSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The area of the square is four times the unit area -/
  area_is_four : side^2 = 4

/-- The minimum number of grid points covered by a grid square -/
def min_covered_points (s : GridSquare) : ℕ := 4

/-- Theorem: A GridSquare covers at least 4 grid points -/
theorem grid_square_covers_at_least_four (s : GridSquare) :
  ∃ (n : ℕ), n ≥ 4 ∧ n = min_covered_points s :=
sorry

end NUMINAMATH_CALUDE_grid_square_covers_at_least_four_l1606_160637


namespace NUMINAMATH_CALUDE_probability_odd_divisor_25_factorial_l1606_160604

theorem probability_odd_divisor_25_factorial (n : ℕ) (h : n = 25) :
  let factorial := n.factorial
  let total_divisors := (factorial.divisors.filter (· > 0)).card
  let odd_divisors := (factorial.divisors.filter (λ d => d > 0 ∧ d % 2 = 1)).card
  (odd_divisors : ℚ) / total_divisors = 1 / 23 := by
  sorry

end NUMINAMATH_CALUDE_probability_odd_divisor_25_factorial_l1606_160604


namespace NUMINAMATH_CALUDE_emmy_and_gerry_apples_l1606_160628

/-- The number of apples Emmy and Gerry can buy together -/
def total_apples (apple_price : ℕ) (emmy_money : ℕ) (gerry_money : ℕ) : ℕ :=
  (emmy_money + gerry_money) / apple_price

/-- Theorem: Emmy and Gerry can buy 150 apples altogether -/
theorem emmy_and_gerry_apples :
  total_apples 2 200 100 = 150 := by
  sorry

#eval total_apples 2 200 100

end NUMINAMATH_CALUDE_emmy_and_gerry_apples_l1606_160628


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l1606_160680

theorem P_greater_than_Q : ∀ x : ℝ, (x^2 + 2) > (2*x) := by
  sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l1606_160680


namespace NUMINAMATH_CALUDE_min_square_area_is_121_l1606_160616

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The set of rectangles given in the problem -/
def problem_rectangles : List Rectangle := [
  { width := 2, height := 3 },
  { width := 3, height := 4 },
  { width := 1, height := 4 }
]

/-- 
  Given a list of rectangles, computes the smallest possible side length of a square 
  that can contain all rectangles without overlapping
-/
def min_square_side (rectangles : List Rectangle) : ℕ :=
  sorry

/-- 
  Theorem: The smallest possible area of a square containing the given rectangles 
  without overlapping is 121
-/
theorem min_square_area_is_121 : 
  (min_square_side problem_rectangles) ^ 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_min_square_area_is_121_l1606_160616


namespace NUMINAMATH_CALUDE_polynomial_efficient_evaluation_l1606_160610

/-- The polynomial 6x^5+5x^4+4x^3+3x^2+2x+2002 can be evaluated using 5 multiplications and 5 additions -/
theorem polynomial_efficient_evaluation :
  ∃ (f : ℝ → ℝ),
    (∀ x, f x = 6*x^5 + 5*x^4 + 4*x^3 + 3*x^2 + 2*x + 2002) ∧
    (∃ (g : ℝ → ℝ) (a b c d e : ℝ → ℝ),
      (∀ x, f x = g x + 2002) ∧
      (∀ x, g x = (((a x * x + b x) * x + c x) * x + d x) * x + e x) ∧
      (∀ x, a x = 6*x + 5) ∧
      (∀ x, b x = 4) ∧
      (∀ x, c x = 3) ∧
      (∀ x, d x = 2) ∧
      (∀ x, e x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_efficient_evaluation_l1606_160610


namespace NUMINAMATH_CALUDE_smallest_square_sum_of_consecutive_integers_l1606_160631

theorem smallest_square_sum_of_consecutive_integers :
  ∃ n : ℕ, 
    (n > 0) ∧ 
    (10 * (2 * n + 19) = 250) ∧ 
    (∀ m : ℕ, m > 0 → m < n → ¬∃ k : ℕ, 10 * (2 * m + 19) = k * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_sum_of_consecutive_integers_l1606_160631


namespace NUMINAMATH_CALUDE_original_number_proof_l1606_160681

theorem original_number_proof :
  ∃ N : ℕ, 
    (∃ k : ℤ, Odd (N * k) ∧ (N * k) % 9 = 0) ∧
    N * 4 = 108 ∧
    N = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1606_160681


namespace NUMINAMATH_CALUDE_wrapper_cap_difference_l1606_160679

/-- Represents Danny's collection of bottle caps and wrappers -/
structure Collection where
  caps : ℕ
  wrappers : ℕ

/-- The number of bottle caps and wrappers Danny found at the park -/
def park_find : Collection :=
  { caps := 15, wrappers := 18 }

/-- Danny's current collection -/
def current_collection : Collection :=
  { caps := 35, wrappers := 67 }

/-- The theorem stating the difference between wrappers and bottle caps in Danny's collection -/
theorem wrapper_cap_difference :
  current_collection.wrappers - current_collection.caps = 32 :=
by sorry

end NUMINAMATH_CALUDE_wrapper_cap_difference_l1606_160679


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l1606_160687

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio -/
def originalRatio : RecipeRatio :=
  { flour := 11, water := 8, sugar := 1 }

/-- The new recipe ratio -/
def newRatio : RecipeRatio :=
  { flour := 22, water := 8, sugar := 1 }

/-- The amount of water in the new recipe -/
def newWaterAmount : ℚ := 4

/-- Theorem stating that the amount of sugar in the new recipe is 0.5 cups -/
theorem sugar_amount_in_new_recipe :
  (newWaterAmount * newRatio.sugar) / newRatio.water = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l1606_160687


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l1606_160649

/-- A hyperbola with vertex and center at (1, 0) and eccentricity 2 -/
structure Hyperbola where
  vertex : ℝ × ℝ
  center : ℝ × ℝ
  eccentricity : ℝ
  vertex_eq_center : vertex = center
  vertex_x : vertex.1 = 1
  vertex_y : vertex.2 = 0
  eccentricity_val : eccentricity = 2

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 - y^2/3 = 1

/-- Theorem stating that the given hyperbola has the equation x² - y²/3 = 1 -/
theorem hyperbola_equation_theorem (h : Hyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 - y^2/3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l1606_160649


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1606_160613

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 9 → (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ (a - b)^2 + (b - c)^2 + (c - a)^2) →
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 27 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1606_160613


namespace NUMINAMATH_CALUDE_cannot_finish_third_l1606_160602

-- Define the set of runners
inductive Runner : Type
| A | B | C | D | E | F

-- Define the finish order relation
def finishes_before (x y : Runner) : Prop := sorry

-- Define the race conditions
axiom race_condition1 : finishes_before Runner.A Runner.B ∧ finishes_before Runner.A Runner.D
axiom race_condition2 : finishes_before Runner.B Runner.C ∧ finishes_before Runner.B Runner.F
axiom race_condition3 : finishes_before Runner.C Runner.D
axiom race_condition4 : finishes_before Runner.E Runner.F ∧ finishes_before Runner.A Runner.E

-- Define a function to represent the finishing position of a runner
def finishing_position (r : Runner) : ℕ := sorry

-- Define what it means to finish in third place
def finishes_third (r : Runner) : Prop := finishing_position r = 3

-- Theorem to prove
theorem cannot_finish_third : 
  ¬(finishes_third Runner.A) ∧ ¬(finishes_third Runner.F) := sorry

end NUMINAMATH_CALUDE_cannot_finish_third_l1606_160602


namespace NUMINAMATH_CALUDE_book_arrangements_eq_103680_l1606_160677

/-- The number of ways to arrange 11 books (3 Arabic, 2 German, 4 Spanish, and 2 French) on a shelf,
    keeping the Arabic books together and the Spanish books together. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let german_books : ℕ := 2
  let spanish_books : ℕ := 4
  let french_books : ℕ := 2
  let grouped_units : ℕ := 1 + 1 + german_books + french_books  -- Arabic and Spanish groups + individual German and French books
  (Nat.factorial grouped_units) * (Nat.factorial arabic_books) * (Nat.factorial spanish_books)

theorem book_arrangements_eq_103680 : book_arrangements = 103680 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_eq_103680_l1606_160677


namespace NUMINAMATH_CALUDE_library_book_return_percentage_l1606_160615

theorem library_book_return_percentage 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (loaned_books : ℕ) 
  (h1 : initial_books = 300) 
  (h2 : final_books = 244) 
  (h3 : loaned_books = 160) : 
  (((loaned_books - (initial_books - final_books)) / loaned_books) * 100 : ℚ) = 65 := by
  sorry

end NUMINAMATH_CALUDE_library_book_return_percentage_l1606_160615


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l1606_160694

/-- Given that 9 oranges weigh the same as 6 apples, prove that 54 oranges
    weigh the same as 36 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
    orange_weight > 0 →
    apple_weight > 0 →
    9 * orange_weight = 6 * apple_weight →
    54 * orange_weight = 36 * apple_weight := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l1606_160694


namespace NUMINAMATH_CALUDE_james_car_value_l1606_160699

/-- The value of James' old car -/
def old_car_value : ℝ := 20000

/-- The percentage of the old car's value James received when selling it -/
def old_car_sell_percentage : ℝ := 0.8

/-- The sticker price of the new car -/
def new_car_sticker_price : ℝ := 30000

/-- The percentage of the new car's sticker price James paid after haggling -/
def new_car_buy_percentage : ℝ := 0.9

/-- The out-of-pocket amount James paid -/
def out_of_pocket : ℝ := 11000

theorem james_car_value :
  new_car_buy_percentage * new_car_sticker_price - old_car_sell_percentage * old_car_value = out_of_pocket :=
by sorry

end NUMINAMATH_CALUDE_james_car_value_l1606_160699


namespace NUMINAMATH_CALUDE_f_derivative_sum_l1606_160688

noncomputable def f (x : ℝ) : ℝ := Real.log 9 * (Real.log x / Real.log 3)

theorem f_derivative_sum : 
  (deriv (λ _ : ℝ => f 2)) 0 + (deriv f) 2 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_sum_l1606_160688


namespace NUMINAMATH_CALUDE_unit_conversions_l1606_160640

-- Define conversion factors
def cm_to_dm : ℚ := 10
def cm_to_m : ℚ := 100
def kg_to_ton : ℚ := 1000
def g_to_kg : ℚ := 1000
def min_to_hour : ℚ := 60

-- Define the theorem
theorem unit_conversions :
  (4800 / cm_to_dm = 480 ∧ 4800 / cm_to_m = 48) ∧
  (5080 / kg_to_ton = 5 ∧ 5080 % kg_to_ton = 80) ∧
  (8 * g_to_kg + 60 = 8060) ∧
  (3 * min_to_hour + 20 = 200) := by
  sorry


end NUMINAMATH_CALUDE_unit_conversions_l1606_160640


namespace NUMINAMATH_CALUDE_no_zeros_of_g_l1606_160652

open Set
open Function
open Topology

theorem no_zeros_of_g (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ≠ 0, deriv f x + f x / x > 0) : 
  ∀ x ≠ 0, f x + 1 / x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_zeros_of_g_l1606_160652


namespace NUMINAMATH_CALUDE_tim_surprise_combinations_l1606_160626

/-- Represents the number of choices for each day of the week --/
structure WeekChoices where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total number of combinations for Tim's surprise arrangements --/
def totalCombinations (choices : WeekChoices) : Nat :=
  choices.monday * choices.tuesday * choices.wednesday * choices.thursday * choices.friday

/-- Tim's specific choices for each day of the week --/
def timChoices : WeekChoices :=
  { monday := 1
  , tuesday := 2
  , wednesday := 6
  , thursday := 5
  , friday := 2 }

theorem tim_surprise_combinations :
  totalCombinations timChoices = 120 := by
  sorry

end NUMINAMATH_CALUDE_tim_surprise_combinations_l1606_160626


namespace NUMINAMATH_CALUDE_distance_from_origin_l1606_160650

theorem distance_from_origin (x y : ℝ) (h1 : x > 2) (h2 : x = 15) 
  (h3 : (x - 2)^2 + (y - 7)^2 = 13^2) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt 274 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1606_160650


namespace NUMINAMATH_CALUDE_parabola_vertex_above_x_axis_l1606_160653

/-- A parabola with equation y = x^2 - 3x + k has its vertex above the x-axis if and only if k > 9/4 -/
theorem parabola_vertex_above_x_axis (k : ℝ) : 
  (∃ (x y : ℝ), y = x^2 - 3*x + k ∧ y > 0 ∧ ∀ (x' : ℝ), x'^2 - 3*x' + k ≤ y) ↔ k > 9/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_above_x_axis_l1606_160653


namespace NUMINAMATH_CALUDE_new_students_count_l1606_160630

theorem new_students_count : ∃! n : ℕ, n < 400 ∧ n % 17 = 16 ∧ n % 19 = 12 ∧ n = 288 := by
  sorry

end NUMINAMATH_CALUDE_new_students_count_l1606_160630


namespace NUMINAMATH_CALUDE_stratified_sampling_elderly_count_l1606_160692

theorem stratified_sampling_elderly_count 
  (total_employees : ℕ) 
  (elderly_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 550) 
  (h2 : elderly_employees = 100) 
  (h3 : sample_size = 33) :
  (sample_size : ℚ) / total_employees * elderly_employees = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_elderly_count_l1606_160692


namespace NUMINAMATH_CALUDE_third_month_relation_l1606_160664

def freelancer_earnings 
  (first_month : ℕ) 
  (second_month : ℕ) 
  (third_month : ℕ) 
  (total : ℕ) : Prop :=
  first_month = 350 ∧
  second_month = 2 * first_month + 50 ∧
  total = first_month + second_month + third_month ∧
  total = 5500

theorem third_month_relation 
  (first_month second_month third_month total : ℕ) :
  freelancer_earnings first_month second_month third_month total →
  third_month = 4 * (first_month + second_month) :=
by
  sorry

end NUMINAMATH_CALUDE_third_month_relation_l1606_160664


namespace NUMINAMATH_CALUDE_sqrt_three_times_sqrt_twelve_l1606_160689

theorem sqrt_three_times_sqrt_twelve : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_sqrt_twelve_l1606_160689


namespace NUMINAMATH_CALUDE_pure_imaginary_roots_of_f_l1606_160620

/-- The polynomial function we're analyzing -/
def f (x : ℂ) : ℂ := x^5 - 2*x^4 + 4*x^3 - 8*x^2 + 16*x - 32

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_roots_of_f :
  ∃! (z : ℂ), f z = 0 ∧ isPureImaginary z ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_roots_of_f_l1606_160620


namespace NUMINAMATH_CALUDE_max_y_value_l1606_160624

theorem max_y_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : x * y = (x - y) / (x + 3 * y)) : 
  y ≤ 1 / 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ = 1 / 3 ∧ x₀ * y₀ = (x₀ - y₀) / (x₀ + 3 * y₀) :=
by sorry

end NUMINAMATH_CALUDE_max_y_value_l1606_160624


namespace NUMINAMATH_CALUDE_train_length_proof_l1606_160619

def train_speed : ℝ := 62.99999999999999
def crossing_time : ℝ := 40

theorem train_length_proof :
  train_speed * crossing_time = 2520 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l1606_160619


namespace NUMINAMATH_CALUDE_max_value_expression_l1606_160698

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  3 * x * y * Real.sqrt 5 + 9 * y * z ≤ (3/2) * Real.sqrt 409 := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1606_160698


namespace NUMINAMATH_CALUDE_summer_degrees_l1606_160632

/-- Given two people where one has five more degrees than the other, 
    and their combined degrees total 295, prove that the person with 
    more degrees has 150 degrees. -/
theorem summer_degrees (s j : ℕ) 
    (h1 : s = j + 5)
    (h2 : s + j = 295) : 
  s = 150 := by
  sorry

end NUMINAMATH_CALUDE_summer_degrees_l1606_160632


namespace NUMINAMATH_CALUDE_probability_rgb_draw_specific_l1606_160690

/-- The probability of drawing a red shoe first, a green shoe second, and a blue shoe third
    from a closet containing red, green, and blue shoes. -/
def probability_rgb_draw (red green blue : ℕ) : ℚ :=
  (red : ℚ) / (red + green + blue) *
  (green : ℚ) / (red + green + blue - 1) *
  (blue : ℚ) / (red + green + blue - 2)

/-- Theorem stating that the probability of drawing a red shoe first, a green shoe second,
    and a blue shoe third from a closet containing 5 red shoes, 4 green shoes, and 3 blue shoes
    is equal to 1/22. -/
theorem probability_rgb_draw_specific : probability_rgb_draw 5 4 3 = 1 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_rgb_draw_specific_l1606_160690


namespace NUMINAMATH_CALUDE_divisibility_of_n_squared_plus_n_plus_two_l1606_160674

theorem divisibility_of_n_squared_plus_n_plus_two (n : ℕ) :
  (∃ k : ℕ, n^2 + n + 2 = 2 * k) ∧ (∃ m : ℕ, ¬(∃ l : ℕ, m^2 + m + 2 = 5 * l)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_n_squared_plus_n_plus_two_l1606_160674


namespace NUMINAMATH_CALUDE_yanni_paintings_l1606_160646

def painting_count : ℕ := 5

def square_feet_per_painting : List ℕ := [25, 25, 25, 80, 45]

theorem yanni_paintings :
  (painting_count = 5) ∧
  (square_feet_per_painting.length = painting_count) ∧
  (square_feet_per_painting.sum = 200) := by
  sorry

end NUMINAMATH_CALUDE_yanni_paintings_l1606_160646


namespace NUMINAMATH_CALUDE_coefficient_of_y_in_equation3_l1606_160639

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := 6*x - 5*y + 3*z = 22
def equation2 (x y z : ℝ) : Prop := 4*x + 8*y - 11*z = 7
def equation3 (x y z : ℝ) : Prop := 5*x - y + 2*z = 12/6

-- Define the sum condition
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coefficient_of_y_in_equation3 (x y z : ℝ) 
  (eq1 : equation1 x y z) 
  (eq2 : equation2 x y z) 
  (eq3 : equation3 x y z) 
  (sum : sum_condition x y z) : 
  ∃ (a b c : ℝ), equation3 x y z ↔ a*x + (-1)*y + c*z = b :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_in_equation3_l1606_160639


namespace NUMINAMATH_CALUDE_volleyball_game_employees_l1606_160659

/-- Calculates the number of employees participating in a volleyball game given the number of managers, teams, and people per team. -/
def employees_participating (managers : ℕ) (teams : ℕ) (people_per_team : ℕ) : ℕ :=
  teams * people_per_team - managers

/-- Theorem stating that with 23 managers, 6 teams, and 5 people per team, there are 7 employees participating. -/
theorem volleyball_game_employees :
  employees_participating 23 6 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_game_employees_l1606_160659


namespace NUMINAMATH_CALUDE_poles_for_given_plot_l1606_160600

/-- Calculates the number of poles needed for a side of a plot -/
def polesForSide (length : ℕ) (spacing : ℕ) : ℕ :=
  (length / spacing) + 1

/-- Represents a trapezoidal plot with given side lengths and pole spacings -/
structure TrapezoidalPlot where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  side4 : ℕ
  spacing1 : ℕ
  spacing2 : ℕ

/-- Calculates the total number of poles needed for a trapezoidal plot -/
def totalPoles (plot : TrapezoidalPlot) : ℕ :=
  polesForSide plot.side1 plot.spacing1 +
  polesForSide plot.side2 plot.spacing2 +
  polesForSide plot.side3 plot.spacing1 +
  polesForSide plot.side4 plot.spacing2

/-- The main theorem stating that the number of poles for the given plot is 40 -/
theorem poles_for_given_plot :
  let plot := TrapezoidalPlot.mk 60 30 50 40 5 4
  totalPoles plot = 40 := by
  sorry


end NUMINAMATH_CALUDE_poles_for_given_plot_l1606_160600


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l1606_160670

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions --/
theorem maxwell_walking_speed :
  ∀ (maxwell_speed : ℝ),
    maxwell_speed > 0 →
    (4 * maxwell_speed + 18 = 34) →
    maxwell_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_maxwell_walking_speed_l1606_160670


namespace NUMINAMATH_CALUDE_circle_center_sum_l1606_160656

theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) →  -- Circle equation
  (∀ a b : ℝ, (a - x)^2 + (b - y)^2 ≤ (x - a)^2 + (y - b)^2) →  -- Definition of center
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1606_160656


namespace NUMINAMATH_CALUDE_largest_circle_at_a_l1606_160603

/-- A pentagon with circles centered at each vertex -/
structure PentagonWithCircles where
  -- Lengths of the pentagon sides
  ab : ℝ
  bc : ℝ
  cd : ℝ
  de : ℝ
  ea : ℝ
  -- Radii of the circles
  r_a : ℝ
  r_b : ℝ
  r_c : ℝ
  r_d : ℝ
  r_e : ℝ
  -- Conditions for circles touching on sides
  h_ab : r_a + r_b = ab
  h_bc : r_b + r_c = bc
  h_cd : r_c + r_d = cd
  h_de : r_d + r_e = de
  h_ea : r_e + r_a = ea

/-- The circle centered at A has the largest radius -/
theorem largest_circle_at_a (p : PentagonWithCircles)
  (h_ab : p.ab = 16)
  (h_bc : p.bc = 14)
  (h_cd : p.cd = 17)
  (h_de : p.de = 13)
  (h_ea : p.ea = 14) :
  p.r_a = max p.r_a (max p.r_b (max p.r_c (max p.r_d p.r_e))) :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_at_a_l1606_160603


namespace NUMINAMATH_CALUDE_conference_handshakes_eq_360_l1606_160657

/-- Represents the number of handshakes in a conference with specific groupings -/
def conference_handshakes (total : ℕ) (group_a : ℕ) (group_b1 : ℕ) (group_b2 : ℕ) : ℕ :=
  let handshakes_a_b1 := group_b1 * (group_a - group_a / 2)
  let handshakes_a_b2 := group_b2 * group_a
  let handshakes_b2 := group_b2 * (group_b2 - 1) / 2
  handshakes_a_b1 + handshakes_a_b2 + handshakes_b2

/-- The theorem stating that the number of handshakes in the given conference scenario is 360 -/
theorem conference_handshakes_eq_360 :
  conference_handshakes 40 25 5 10 = 360 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_eq_360_l1606_160657


namespace NUMINAMATH_CALUDE_cost_of_eight_books_l1606_160635

theorem cost_of_eight_books (cost_of_two : ℝ) (h : cost_of_two = 34) :
  8 * (cost_of_two / 2) = 136 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_eight_books_l1606_160635


namespace NUMINAMATH_CALUDE_water_formation_l1606_160676

/-- Represents the balanced chemical equation for the reaction between NH4Cl and NaOH -/
structure ChemicalReaction where
  nh4cl : ℕ
  naoh : ℕ
  h2o : ℕ
  balanced : nh4cl = naoh ∧ nh4cl = h2o

/-- Calculates the moles of water produced in the reaction -/
def waterProduced (reaction : ChemicalReaction) (nh4cl_moles : ℕ) (naoh_moles : ℕ) : ℕ :=
  min nh4cl_moles naoh_moles

theorem water_formation (reaction : ChemicalReaction) 
  (h1 : reaction.nh4cl = 1 ∧ reaction.naoh = 1 ∧ reaction.h2o = 1) 
  (h2 : nh4cl_moles = 3) 
  (h3 : naoh_moles = 3) : 
  waterProduced reaction nh4cl_moles naoh_moles = 3 := by
  sorry

end NUMINAMATH_CALUDE_water_formation_l1606_160676


namespace NUMINAMATH_CALUDE_kitten_growth_l1606_160691

/-- The length of a kitten after doubling twice -/
def kittenLength (initialLength : ℕ) (doublings : ℕ) : ℕ :=
  initialLength * (2 ^ doublings)

/-- Theorem: A kitten with initial length 4 inches that doubles twice will be 16 inches long -/
theorem kitten_growth : kittenLength 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_l1606_160691


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1606_160607

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.cos x - Real.cos (3 * x)) / x else 0

theorem f_derivative_at_zero : 
  deriv f 0 = 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1606_160607


namespace NUMINAMATH_CALUDE_product_of_positive_real_solutions_l1606_160672

def solutions (x : ℂ) : Prop := x^8 = -256

def positive_real_part (z : ℂ) : Prop := z.re > 0

theorem product_of_positive_real_solutions :
  ∃ (S : Finset ℂ), 
    (∀ z ∈ S, solutions z ∧ positive_real_part z) ∧ 
    (∀ z, solutions z ∧ positive_real_part z → z ∈ S) ∧
    S.prod id = 8 :=
sorry

end NUMINAMATH_CALUDE_product_of_positive_real_solutions_l1606_160672


namespace NUMINAMATH_CALUDE_set_union_problem_l1606_160629

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 4} →
  B = {0, 1, a} →
  A ∪ B = {0, 1, 4} →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1606_160629


namespace NUMINAMATH_CALUDE_investment_principal_l1606_160645

/-- Proves that an investment with a monthly interest payment of $228 and a simple annual interest rate of 9% has a principal amount of $30,400. -/
theorem investment_principal (monthly_interest : ℝ) (annual_rate : ℝ) (principal : ℝ) : 
  monthly_interest = 228 →
  annual_rate = 0.09 →
  principal = (monthly_interest * 12) / annual_rate →
  principal = 30400 := by
  sorry


end NUMINAMATH_CALUDE_investment_principal_l1606_160645


namespace NUMINAMATH_CALUDE_negation_existence_gt_one_l1606_160661

theorem negation_existence_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_gt_one_l1606_160661
