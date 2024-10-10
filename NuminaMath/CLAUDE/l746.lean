import Mathlib

namespace quotient_calculation_l746_74664

theorem quotient_calculation (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 166)
  (h2 : divisor = 18)
  (h3 : remainder = 4)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 := by
  sorry

end quotient_calculation_l746_74664


namespace det_A_eq_neg94_l746_74610

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 4, -2; 3, -1, 5; -1, 3, 2]

theorem det_A_eq_neg94 : Matrix.det A = -94 := by
  sorry

end det_A_eq_neg94_l746_74610


namespace cycle_cut_orthogonality_l746_74660

-- Define a graph
structure Graph where
  V : Type
  E : Type
  incident : E → V → Prop

-- Define cycle space and cut space
def CycleSpace (G : Graph) : Type := sorry
def CutSpace (G : Graph) : Type := sorry

-- Define orthogonal complement
def OrthogonalComplement (S : Type) : Type := sorry

-- State the theorem
theorem cycle_cut_orthogonality (G : Graph) :
  (CycleSpace G = OrthogonalComplement (CutSpace G)) ∧
  (CutSpace G = OrthogonalComplement (CycleSpace G)) := by
  sorry

end cycle_cut_orthogonality_l746_74660


namespace coin_collection_problem_l746_74609

/-- Represents the state of a coin collection --/
structure CoinCollection where
  gold : ℕ
  silver : ℕ

/-- Calculates the ratio of gold to silver coins --/
def goldSilverRatio (c : CoinCollection) : ℚ :=
  c.gold / c.silver

/-- Represents the coin collection problem --/
theorem coin_collection_problem 
  (initial : CoinCollection)
  (final : CoinCollection)
  (added_gold : ℕ) :
  goldSilverRatio initial = 1 / 3 →
  goldSilverRatio final = 1 / 2 →
  final.gold + final.silver = 135 →
  final.gold = initial.gold + added_gold →
  final.silver = initial.silver →
  added_gold = 15 := by
  sorry


end coin_collection_problem_l746_74609


namespace logarithm_expression_equals_zero_l746_74633

theorem logarithm_expression_equals_zero :
  Real.log 14 - 2 * Real.log (7/3) + Real.log 7 - Real.log 18 = 0 := by
  sorry

end logarithm_expression_equals_zero_l746_74633


namespace point_movement_l746_74657

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point left by a given number of units -/
def moveLeft (p : Point) (units : ℝ) : Point :=
  ⟨p.x - units, p.y⟩

/-- Move a point up by a given number of units -/
def moveUp (p : Point) (units : ℝ) : Point :=
  ⟨p.x, p.y + units⟩

theorem point_movement :
  let A : Point := ⟨2, -1⟩
  let B : Point := moveUp (moveLeft A 3) 4
  B.x = -1 ∧ B.y = 3 := by
  sorry

end point_movement_l746_74657


namespace equivalent_statements_l746_74627

theorem equivalent_statements :
  (∀ x : ℝ, x ≥ 0 → x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) := by
  sorry

end equivalent_statements_l746_74627


namespace tank_filling_time_l746_74678

/-- Represents the time (in hours) it takes to fill the tank without the hole -/
def T : ℝ := 15

/-- Represents the time (in hours) it takes to fill the tank with the hole -/
def fill_time_with_hole : ℝ := 20

/-- Represents the time (in hours) it takes for the hole to empty the full tank -/
def empty_time : ℝ := 60

theorem tank_filling_time :
  (1 / T - 1 / empty_time = 1 / fill_time_with_hole) ∧
  (T > 0) ∧ (fill_time_with_hole > 0) ∧ (empty_time > 0) :=
sorry

end tank_filling_time_l746_74678


namespace radio_loss_percentage_l746_74651

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percentage for a radio with cost price 1500 and selling price 1305 is 13% -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 1500
  let selling_price : ℚ := 1305
  loss_percentage cost_price selling_price = 13 := by
  sorry

end radio_loss_percentage_l746_74651


namespace triangle_area_approx_l746_74614

-- Define the triangle DEF and point Q
structure Triangle :=
  (D E F Q : ℝ × ℝ)

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  let d_to_q := Real.sqrt ((t.D.1 - t.Q.1)^2 + (t.D.2 - t.Q.2)^2)
  let e_to_q := Real.sqrt ((t.E.1 - t.Q.1)^2 + (t.E.2 - t.Q.2)^2)
  let f_to_q := Real.sqrt ((t.F.1 - t.Q.1)^2 + (t.F.2 - t.Q.2)^2)
  d_to_q = 5 ∧ e_to_q = 13 ∧ f_to_q = 12

def is_equilateral (t : Triangle) : Prop :=
  let d_to_e := Real.sqrt ((t.D.1 - t.E.1)^2 + (t.D.2 - t.E.2)^2)
  let e_to_f := Real.sqrt ((t.E.1 - t.F.1)^2 + (t.E.2 - t.F.2)^2)
  let f_to_d := Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2)
  d_to_e = e_to_f ∧ e_to_f = f_to_d

-- Define the theorem
theorem triangle_area_approx (t : Triangle) 
  (h1 : is_valid_triangle t) (h2 : is_equilateral t) : 
  ∃ (area : ℝ), abs (area - 132) < 1 := by
  sorry

end triangle_area_approx_l746_74614


namespace mike_picked_12_pears_l746_74638

/-- The number of pears Mike picked -/
def mike_pears : ℕ := sorry

/-- The number of pears Keith picked initially -/
def keith_initial_pears : ℕ := 47

/-- The number of pears Keith gave away -/
def keith_gave_away : ℕ := 46

/-- The total number of pears Keith and Mike have left -/
def total_pears_left : ℕ := 13

theorem mike_picked_12_pears : mike_pears = 12 := by
  sorry

end mike_picked_12_pears_l746_74638


namespace sphere_unique_orientation_independent_projections_l746_74645

-- Define the type for 3D objects
inductive Object3D
  | Cube
  | RegularTetrahedron
  | RightTriangularPyramid
  | Sphere

-- Define a function to check if an object's projections are orientation-independent
def hasOrientationIndependentProjections (obj : Object3D) : Prop :=
  match obj with
  | Object3D.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_unique_orientation_independent_projections :
  ∀ (obj : Object3D), hasOrientationIndependentProjections obj ↔ obj = Object3D.Sphere :=
sorry

end sphere_unique_orientation_independent_projections_l746_74645


namespace pear_price_is_correct_l746_74613

/-- The price of a pear in won -/
def pear_price : ℕ := 6300

/-- The price of an apple in won -/
def apple_price : ℕ := pear_price + 2400

/-- The sum of the prices of an apple and a pear in won -/
def total_price : ℕ := 15000

theorem pear_price_is_correct : pear_price = 6300 := by
  have h1 : apple_price + pear_price = total_price := by sorry
  have h2 : apple_price = pear_price + 2400 := by sorry
  sorry

end pear_price_is_correct_l746_74613


namespace units_digit_of_product_l746_74630

theorem units_digit_of_product : ((30 * 31 * 32 * 33 * 34 * 35) / 1000) % 10 = 6 := by
  sorry

end units_digit_of_product_l746_74630


namespace difference_of_squares_ratio_l746_74623

theorem difference_of_squares_ratio : 
  (1732^2 - 1725^2) / (1739^2 - 1718^2) = 1/3 := by
sorry

end difference_of_squares_ratio_l746_74623


namespace fixed_point_of_exponential_function_l746_74649

/-- The function f(x) = a^(x-1) + 7 always passes through the point (1, 8) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x-1) + 7
  f 1 = 8 := by
  sorry

end fixed_point_of_exponential_function_l746_74649


namespace number_exceeding_percentage_l746_74640

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.16 * x + 126 ∧ x = 150 := by
  sorry

end number_exceeding_percentage_l746_74640


namespace books_loaned_out_l746_74685

/-- Proves that the number of books loaned out is 50 given the initial and final book counts and return rate -/
theorem books_loaned_out (initial_books : ℕ) (final_books : ℕ) (return_rate : ℚ) : 
  initial_books = 75 → final_books = 65 → return_rate = 4/5 → 
  (initial_books - final_books) / (1 - return_rate) = 50 := by
  sorry

end books_loaned_out_l746_74685


namespace or_false_necessary_not_sufficient_for_and_false_l746_74637

theorem or_false_necessary_not_sufficient_for_and_false (p q : Prop) :
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬(p ∨ q)) :=
by sorry

end or_false_necessary_not_sufficient_for_and_false_l746_74637


namespace investment_rate_proof_l746_74673

theorem investment_rate_proof (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (additional_rate : ℝ) :
  initial_investment = 8000 →
  initial_rate = 0.05 →
  additional_investment = 4000 →
  additional_rate = 0.08 →
  let total_interest := initial_investment * initial_rate + additional_investment * additional_rate
  let total_investment := initial_investment + additional_investment
  (total_interest / total_investment) = 0.06 := by
  sorry

end investment_rate_proof_l746_74673


namespace no_squares_in_sequence_l746_74615

def a : ℕ → ℤ
  | 0 => 91
  | n + 1 => 10 * a n + (-1) ^ n

theorem no_squares_in_sequence : ∀ n : ℕ, ¬∃ m : ℤ, a n = m ^ 2 := by
  sorry

end no_squares_in_sequence_l746_74615


namespace quadratic_root_implies_coefficients_l746_74674

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic (b c x : ℂ) : ℂ := x^2 + b*x + c

theorem quadratic_root_implies_coefficients :
  ∀ (b c : ℝ), quadratic b c (2 - i) = 0 → b = -4 ∧ c = 5 := by sorry

end quadratic_root_implies_coefficients_l746_74674


namespace tan_function_property_l746_74626

theorem tan_function_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * (x - c)) = a * Real.tan (b * (x - c) + π)) →  -- period is π/4
  (a * Real.tan (b * (π/3 - c)) = -4) →  -- passes through (π/3, -4)
  (b * (π/4 - c) = π/2) →  -- vertical asymptote at x = π/4
  4 * a * b = 64 * Real.sqrt 3 := by
sorry

end tan_function_property_l746_74626


namespace interest_rate_calculation_l746_74603

/-- Proves that the interest rate is 8% per annum given the conditions of the problem -/
theorem interest_rate_calculation (P : ℝ) (t : ℝ) (I : ℝ) (r : ℝ) :
  P = 2500 →
  t = 8 →
  I = P - 900 →
  I = P * r * t / 100 →
  r = 8 := by
  sorry

end interest_rate_calculation_l746_74603


namespace lcm_of_ratio_and_hcf_l746_74698

theorem lcm_of_ratio_and_hcf (a b : ℕ+) :
  (a : ℚ) / b = 14 / 21 →
  Nat.gcd a b = 28 →
  Nat.lcm a b = 1176 := by
sorry

end lcm_of_ratio_and_hcf_l746_74698


namespace car_meeting_problem_l746_74694

/-- Represents a car with a speed and initial position -/
structure Car where
  speed : ℝ
  initial_position : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  car_x : Car
  car_y : Car
  first_meeting_time : ℝ
  distance_between_meetings : ℝ

/-- The theorem statement -/
theorem car_meeting_problem (setup : ProblemSetup)
  (h1 : setup.car_x.speed = 50)
  (h2 : setup.first_meeting_time = 1)
  (h3 : setup.distance_between_meetings = 20)
  (h4 : setup.car_x.initial_position = 0)
  (h5 : setup.car_y.initial_position = setup.car_x.initial_position + 
        setup.car_x.speed * setup.first_meeting_time + 
        setup.car_y.speed * setup.first_meeting_time) :
  setup.car_y.initial_position - setup.car_x.initial_position = 110 ∧
  setup.car_y.speed = 60 := by
  sorry

end car_meeting_problem_l746_74694


namespace min_distance_ellipse_line_l746_74646

/-- The minimum distance between a point on the ellipse x²/3 + y² = 1 and 
    a point on the line x + y = 4, along with the coordinates of the point 
    on the ellipse at this minimum distance. -/
theorem min_distance_ellipse_line :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}
  let line := {q : ℝ × ℝ | q.1 + q.2 = 4}
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧ 
    (∀ (p' : ℝ × ℝ) (q : ℝ × ℝ), p' ∈ ellipse → q ∈ line → 
      Real.sqrt 2 ≤ Real.sqrt ((p'.1 - q.1)^2 + (p'.2 - q.2)^2)) ∧
    p = (3/2, 1/2) := by
  sorry

end min_distance_ellipse_line_l746_74646


namespace orange_count_l746_74612

/-- Represents the count of fruits in a box -/
structure FruitBox where
  apples : ℕ
  pears : ℕ
  oranges : ℕ

/-- The properties of the fruit box as described in the problem -/
def is_valid_fruit_box (box : FruitBox) : Prop :=
  box.apples + box.pears + box.oranges = 60 ∧
  box.apples = 3 * (box.pears + box.oranges) ∧
  box.pears * 5 = box.apples + box.oranges

/-- Theorem stating that a valid fruit box has 5 oranges -/
theorem orange_count (box : FruitBox) (h : is_valid_fruit_box box) : box.oranges = 5 := by
  sorry

end orange_count_l746_74612


namespace complex_calculation_l746_74635

theorem complex_calculation (z : ℂ) (h : z = 1 + I) : z^2 + 2/z = 1 + I := by
  sorry

end complex_calculation_l746_74635


namespace tom_and_michael_have_nine_robots_l746_74693

/-- The number of car robots Bob has -/
def bob_robots : ℕ := 81

/-- The factor by which Bob's robots outnumber Tom and Michael's combined -/
def factor : ℕ := 9

/-- The number of car robots Tom and Michael have combined -/
def tom_and_michael_robots : ℕ := bob_robots / factor

theorem tom_and_michael_have_nine_robots : tom_and_michael_robots = 9 := by
  sorry

end tom_and_michael_have_nine_robots_l746_74693


namespace cricketer_average_score_l746_74618

/-- Represents a cricketer's scoring statistics -/
structure CricketerStats where
  innings : ℕ
  lastInningScore : ℕ
  averageIncrease : ℕ

/-- Calculates the new average score after the last inning -/
def newAverageScore (stats : CricketerStats) : ℕ :=
  sorry

theorem cricketer_average_score 
  (stats : CricketerStats) 
  (h1 : stats.innings = 19) 
  (h2 : stats.lastInningScore = 98) 
  (h3 : stats.averageIncrease = 4) : 
  newAverageScore stats = 26 := by
  sorry

end cricketer_average_score_l746_74618


namespace projectile_trajectory_area_l746_74662

theorem projectile_trajectory_area 
  (u : ℝ) 
  (k : ℝ) 
  (φ : ℝ) 
  (h_φ_range : 30 * π / 180 ≤ φ ∧ φ ≤ 150 * π / 180) 
  (h_u_pos : u > 0) 
  (h_k_pos : k > 0) : 
  ∃ d : ℝ, d = π / 8 ∧ 
    (∀ x y : ℝ, (x^2 / (u^2 / (2 * k))^2 + (y - u^2 / (4 * k))^2 / (u^2 / (4 * k))^2 = 1) → 
      π * (u^2 / (2 * k)) * (u^2 / (4 * k)) = d * u^4 / k^2) := by
sorry

end projectile_trajectory_area_l746_74662


namespace polynomial_roots_l746_74644

theorem polynomial_roots (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 4*x^4 + a*x = b*x^2 + 4*c ↔ x = 2 ∨ x = -2) ↔ 
  (a = -16 ∧ b = 48 ∧ c = -32) :=
sorry

end polynomial_roots_l746_74644


namespace time_to_cross_signal_pole_l746_74632

-- Define the train and platform parameters
def train_length : ℝ := 300
def platform_length : ℝ := 400
def time_cross_platform : ℝ := 42

-- Define the theorem
theorem time_to_cross_signal_pole :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / time_cross_platform
  let time_cross_pole := train_length / train_speed
  time_cross_pole = 18 := by
  sorry


end time_to_cross_signal_pole_l746_74632


namespace cube_root_abs_sqrt_squared_sum_l746_74628

theorem cube_root_abs_sqrt_squared_sum (π : ℝ) : 
  ((-8 : ℝ) ^ (1/3 : ℝ)) + |1 - π| + (9 : ℝ).sqrt - (-1 : ℝ)^2 = π - 1 := by
  sorry

end cube_root_abs_sqrt_squared_sum_l746_74628


namespace complex_magnitude_proof_l746_74605

theorem complex_magnitude_proof : 
  Complex.abs ((1 + Complex.I) / (1 - Complex.I) + Complex.I) = 2 := by
  sorry

end complex_magnitude_proof_l746_74605


namespace min_c_over_d_l746_74689

theorem min_c_over_d (x C D : ℝ) (hx : x > 0) (hC : C > 0) (hD : D > 0)
  (eq1 : x^2 + 1/x^2 = C) (eq2 : x + 1/x = D) : 
  ∀ y : ℝ, y > 0 → y^2 + 1/y^2 = C → y + 1/y = D → C / D ≥ 1 :=
by sorry

end min_c_over_d_l746_74689


namespace wall_height_proof_l746_74606

/-- Proves that the height of each wall is 2 meters given the painting conditions --/
theorem wall_height_proof (num_walls : ℕ) (wall_width : ℝ) (paint_rate : ℝ) 
  (total_time : ℝ) (spare_time : ℝ) :
  num_walls = 5 →
  wall_width = 3 →
  paint_rate = 1 / 10 →
  total_time = 10 →
  spare_time = 5 →
  ∃ (wall_height : ℝ), 
    wall_height = 2 ∧ 
    (total_time - spare_time) * 60 * paint_rate = num_walls * wall_width * wall_height :=
by sorry

end wall_height_proof_l746_74606


namespace specificPolygonArea_l746_74648

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A polygon defined by a list of grid points -/
def Polygon := List GridPoint

/-- The polygon formed by connecting specific points on a 4x4 grid -/
def specificPolygon : Polygon :=
  [⟨0,0⟩, ⟨1,0⟩, ⟨1,1⟩, ⟨0,1⟩, ⟨1,2⟩, ⟨0,2⟩, ⟨1,3⟩, ⟨0,3⟩, 
   ⟨3,3⟩, ⟨2,3⟩, ⟨3,2⟩, ⟨2,2⟩, ⟨2,1⟩, ⟨3,1⟩, ⟨3,0⟩, ⟨2,0⟩]

/-- Function to calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℕ := sorry

/-- Theorem stating that the area of the specific polygon is 16 square units -/
theorem specificPolygonArea : calculateArea specificPolygon = 16 := by sorry

end specificPolygonArea_l746_74648


namespace jiajia_clover_problem_l746_74616

theorem jiajia_clover_problem :
  ∀ (n : ℕ),
    (3 * n + 4 = 40) →
    (n = 12) :=
by
  sorry

end jiajia_clover_problem_l746_74616


namespace least_factorial_divisible_by_7875_l746_74642

theorem least_factorial_divisible_by_7875 :
  ∃ (n : ℕ), n > 0 ∧ 7875 ∣ n.factorial ∧ ∀ (m : ℕ), m > 0 → 7875 ∣ m.factorial → n ≤ m :=
by
  -- The proof goes here
  sorry

end least_factorial_divisible_by_7875_l746_74642


namespace inequality_proof_l746_74669

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0)
  (h5 : x * y * z = 1) (h6 : y + z + t = 2) :
  x^2 + y^2 + z^2 + t^2 ≥ 3 := by
sorry

end inequality_proof_l746_74669


namespace probability_of_drawing_balls_l746_74671

theorem probability_of_drawing_balls (prob_A prob_B : ℝ) 
  (h_prob_A : prob_A = 1/3) (h_prob_B : prob_B = 1/2) :
  let prob_both_red := prob_A * prob_B
  let prob_exactly_one_red := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B
  let prob_both_not_red := (1 - prob_A) * (1 - prob_B)
  let prob_at_least_one_red := 1 - prob_both_not_red
  (prob_both_red = 1/6) ∧
  (prob_exactly_one_red = 1/2) ∧
  (prob_both_not_red = 5/6) ∧
  (prob_at_least_one_red = 2/3) :=
by sorry

end probability_of_drawing_balls_l746_74671


namespace shopping_lottery_results_l746_74668

/-- Represents the lottery event with 10 coupons -/
structure LotteryEvent where
  total_coupons : Nat
  first_prize_coupons : Nat
  second_prize_coupons : Nat
  non_prize_coupons : Nat
  first_prize_value : Nat
  second_prize_value : Nat
  drawn_coupons : Nat

/-- The specific lottery event described in the problem -/
def shopping_lottery : LotteryEvent :=
  { total_coupons := 10
  , first_prize_coupons := 1
  , second_prize_coupons := 3
  , non_prize_coupons := 6
  , first_prize_value := 50
  , second_prize_value := 10
  , drawn_coupons := 2
  }

/-- The probability of winning a prize in the shopping lottery -/
def win_probability (l : LotteryEvent) : Rat :=
  1 - (Nat.choose l.non_prize_coupons l.drawn_coupons) / (Nat.choose l.total_coupons l.drawn_coupons)

/-- The mathematical expectation of the total prize value in the shopping lottery -/
def prize_expectation (l : LotteryEvent) : Rat :=
  let p0 := (Nat.choose l.non_prize_coupons l.drawn_coupons) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p10 := (Nat.choose l.second_prize_coupons 1 * Nat.choose l.non_prize_coupons 1) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p20 := (Nat.choose l.second_prize_coupons 2) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p50 := (Nat.choose l.first_prize_coupons 1 * Nat.choose l.non_prize_coupons 1) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p60 := (Nat.choose l.first_prize_coupons 1 * Nat.choose l.second_prize_coupons 1) / (Nat.choose l.total_coupons l.drawn_coupons)
  0 * p0 + 10 * p10 + 20 * p20 + 50 * p50 + 60 * p60

theorem shopping_lottery_results :
  win_probability shopping_lottery = 2/3 ∧
  prize_expectation shopping_lottery = 16 := by
  sorry

end shopping_lottery_results_l746_74668


namespace circular_track_catchup_l746_74641

/-- The time (in minutes) for Person A to catch up with Person B on a circular track -/
def catchUpTime (trackCircumference : ℝ) (speedA speedB : ℝ) (restInterval : ℝ) (restDuration : ℝ) : ℝ :=
  sorry

theorem circular_track_catchup :
  let trackCircumference : ℝ := 400
  let speedA : ℝ := 52
  let speedB : ℝ := 46
  let restInterval : ℝ := 100
  let restDuration : ℝ := 1
  catchUpTime trackCircumference speedA speedB restInterval restDuration = 147 + 1/3 := by
  sorry

end circular_track_catchup_l746_74641


namespace abs_z_equals_one_l746_74697

theorem abs_z_equals_one (z : ℂ) (h : (1 - 2*I)^2 / z = 4 - 3*I) : Complex.abs z = 1 := by
  sorry

end abs_z_equals_one_l746_74697


namespace james_out_of_pocket_cost_l746_74684

/-- Calculates the out-of-pocket cost for a given service -/
def outOfPocketCost (cost : ℝ) (coveragePercent : ℝ) : ℝ :=
  cost - (cost * coveragePercent)

/-- Theorem: James's total out-of-pocket cost is $262.70 -/
theorem james_out_of_pocket_cost : 
  let consultation_cost : ℝ := 300
  let consultation_coverage : ℝ := 0.83
  let xray_cost : ℝ := 150
  let xray_coverage : ℝ := 0.74
  let medication_cost : ℝ := 75
  let medication_coverage : ℝ := 0.55
  let therapy_cost : ℝ := 120
  let therapy_coverage : ℝ := 0.62
  let equipment_cost : ℝ := 85
  let equipment_coverage : ℝ := 0.49
  let followup_cost : ℝ := 200
  let followup_coverage : ℝ := 0.75
  
  (outOfPocketCost consultation_cost consultation_coverage +
   outOfPocketCost xray_cost xray_coverage +
   outOfPocketCost medication_cost medication_coverage +
   outOfPocketCost therapy_cost therapy_coverage +
   outOfPocketCost equipment_cost equipment_coverage +
   outOfPocketCost followup_cost followup_coverage) = 262.70 := by
  sorry


end james_out_of_pocket_cost_l746_74684


namespace min_intersection_cardinality_l746_74653

-- Define the cardinality function
def card (S : Set α) : ℕ := sorry

-- Define the power set function
def powerset (S : Set α) : Set (Set α) := sorry

theorem min_intersection_cardinality 
  (A B C D : Set α) 
  (h1 : card A = 150) 
  (h2 : card B = 150) 
  (h3 : card D = 102) 
  (h4 : card (powerset A) + card (powerset B) + card (powerset C) + card (powerset D) = 
        card (powerset (A ∪ B ∪ C ∪ D)))
  (h5 : card (powerset A) + card (powerset B) + card (powerset C) + card (powerset D) = 2^152) :
  card (A ∩ B ∩ C ∩ D) ≥ 99 ∧ ∃ (A' B' C' D' : Set α), 
    card A' = 150 ∧ 
    card B' = 150 ∧ 
    card D' = 102 ∧ 
    card (powerset A') + card (powerset B') + card (powerset C') + card (powerset D') = 
      card (powerset (A' ∪ B' ∪ C' ∪ D')) ∧
    card (powerset A') + card (powerset B') + card (powerset C') + card (powerset D') = 2^152 ∧
    card (A' ∩ B' ∩ C' ∩ D') = 99 :=
by sorry

end min_intersection_cardinality_l746_74653


namespace symmetric_points_difference_l746_74639

/-- Given two points A(a, 3) and B(-4, b) that are symmetric with respect to the origin,
    prove that a - b = 7 -/
theorem symmetric_points_difference (a b : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (a, 3) ∧ B = (-4, b) ∧ A = (-B.1, -B.2)) →
  a - b = 7 := by
sorry

end symmetric_points_difference_l746_74639


namespace min_value_sum_l746_74600

theorem min_value_sum (a b : ℝ) (h : a^2 + 2*b^2 = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 = 6 → m ≤ x + y) ∧ (∃ (u v : ℝ), u^2 + 2*v^2 = 6 ∧ m = u + v) ∧ m = -3 := by
  sorry

end min_value_sum_l746_74600


namespace evaluate_expression_l746_74686

theorem evaluate_expression : (1 / ((5^2)^4)) * 5^11 * 2 = 250 := by sorry

end evaluate_expression_l746_74686


namespace min_value_expression_l746_74647

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) (hab : a + b = 1) :
  ((2 * a + b) / (a * b) - 3) * c + Real.sqrt 2 / (c - 1) ≥ 4 + 2 * Real.sqrt 2 := by
  sorry

end min_value_expression_l746_74647


namespace greatest_prime_factor_of_299_l746_74601

theorem greatest_prime_factor_of_299 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 299 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 299 → q ≤ p :=
  sorry

end greatest_prime_factor_of_299_l746_74601


namespace probability_two_pairs_one_odd_l746_74622

def total_socks : ℕ := 12
def socks_per_color : ℕ := 3
def num_colors : ℕ := 4
def drawn_socks : ℕ := 5

def favorable_outcomes : ℕ := 324
def total_outcomes : ℕ := Nat.choose total_socks drawn_socks

theorem probability_two_pairs_one_odd (h : total_outcomes = 792) :
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 22 := by
  sorry

end probability_two_pairs_one_odd_l746_74622


namespace dana_saturday_hours_l746_74625

theorem dana_saturday_hours
  (hourly_rate : ℕ)
  (friday_hours : ℕ)
  (sunday_hours : ℕ)
  (total_earnings : ℕ)
  (h1 : hourly_rate = 13)
  (h2 : friday_hours = 9)
  (h3 : sunday_hours = 3)
  (h4 : total_earnings = 286) :
  (total_earnings - (friday_hours + sunday_hours) * hourly_rate) / hourly_rate = 10 :=
by sorry

end dana_saturday_hours_l746_74625


namespace marble_selection_ways_l746_74699

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of marbles -/
def total_marbles : ℕ := 16

/-- The number of colored marbles -/
def colored_marbles : ℕ := 4

/-- The number of non-colored marbles -/
def non_colored_marbles : ℕ := total_marbles - colored_marbles

/-- The number of marbles to be chosen -/
def chosen_marbles : ℕ := 5

/-- The number of colored marbles to be chosen -/
def chosen_colored : ℕ := 2

/-- The number of non-colored marbles to be chosen -/
def chosen_non_colored : ℕ := chosen_marbles - chosen_colored

theorem marble_selection_ways :
  choose colored_marbles chosen_colored * choose non_colored_marbles chosen_non_colored = 1320 :=
sorry

end marble_selection_ways_l746_74699


namespace vertex_y_coordinate_l746_74617

/-- The y-coordinate of the vertex of the parabola y = 3x^2 - 6x + 2 is -1 -/
theorem vertex_y_coordinate (x y : ℝ) : 
  y = 3 * x^2 - 6 * x + 2 → 
  ∃ x₀, ∀ x', 3 * x'^2 - 6 * x' + 2 ≥ 3 * x₀^2 - 6 * x₀ + 2 ∧ 
            3 * x₀^2 - 6 * x₀ + 2 = -1 :=
by sorry

end vertex_y_coordinate_l746_74617


namespace amy_haircut_l746_74681

/-- Represents the hair growth problem --/
def hair_problem (initial_length : ℝ) (growth_rate : ℝ) (weeks : ℕ) (final_length : ℝ) : Prop :=
  let growth := growth_rate * weeks
  let before_cut := initial_length + growth
  before_cut - final_length = 6

/-- Theorem stating the solution to Amy's haircut problem --/
theorem amy_haircut : hair_problem 11 0.5 4 7 := by
  sorry

end amy_haircut_l746_74681


namespace geometric_progression_first_term_l746_74683

theorem geometric_progression_first_term
  (S : ℝ)
  (sum_first_two : ℝ)
  (h1 : S = 9)
  (h2 : sum_first_two = 7) :
  ∃ (a : ℝ), (a = 3 * (3 - Real.sqrt 2) ∨ a = 3 * (3 + Real.sqrt 2)) ∧
    (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end geometric_progression_first_term_l746_74683


namespace simplify_expression_l746_74654

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  |a - 2| - Real.sqrt ((a - 3)^2) = 2*a - 5 := by
  sorry

end simplify_expression_l746_74654


namespace football_field_length_proof_l746_74611

/-- The length of a football field in yards -/
def football_field_length : ℝ := 200

/-- The number of football fields a potato is launched across -/
def fields_crossed : ℕ := 6

/-- The speed of the dog in feet per minute -/
def dog_speed : ℝ := 400

/-- The time taken by the dog to fetch the potato in minutes -/
def fetch_time : ℝ := 9

/-- The number of feet in a yard -/
def feet_per_yard : ℝ := 3

theorem football_field_length_proof :
  football_field_length = 
    (dog_speed / feet_per_yard * fetch_time) / fields_crossed := by
  sorry

end football_field_length_proof_l746_74611


namespace inverse_variation_problem_l746_74680

/-- Given that y^4 varies inversely with ⁴√z, prove that z = 1/4096 when y = 6, given that y = 3 when z = 16 -/
theorem inverse_variation_problem (y z : ℝ) (h1 : ∃ k : ℝ, ∀ y z, y^4 * z^(1/4) = k) 
  (h2 : 3^4 * 16^(1/4) = 6^4 * z^(1/4)) : 
  y = 6 → z = 1/4096 := by
  sorry

end inverse_variation_problem_l746_74680


namespace tangency_values_l746_74636

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 5

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 1

/-- The tangency condition -/
def tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, parabola x y ∧ hyperbola m x y ∧
  ∀ x' y' : ℝ, parabola x' y' → hyperbola m x' y' → (x = x' ∧ y = y')

/-- The theorem stating the values of m for which the parabola and hyperbola are tangent -/
theorem tangency_values :
  ∀ m : ℝ, tangent m ↔ (m = 10 + 4 * Real.sqrt 6 ∨ m = 10 - 4 * Real.sqrt 6) :=
sorry

end tangency_values_l746_74636


namespace counterfeit_identification_possible_l746_74659

/-- Represents the result of weighing two coins on a balance scale -/
inductive WeighResult
  | Equal : WeighResult
  | LeftLighter : WeighResult
  | RightLighter : WeighResult

/-- Represents a coin, which can be either real or counterfeit -/
inductive Coin
  | Real : Coin
  | Counterfeit : Coin

/-- A function that simulates weighing two coins on a balance scale -/
def weigh (a b : Coin) : WeighResult :=
  match a, b with
  | Coin.Real, Coin.Real => WeighResult.Equal
  | Coin.Counterfeit, Coin.Real => WeighResult.LeftLighter
  | Coin.Real, Coin.Counterfeit => WeighResult.RightLighter
  | Coin.Counterfeit, Coin.Counterfeit => WeighResult.Equal

/-- A function that identifies the counterfeit coin based on one weighing -/
def identifyCounterfeit (coins : Fin 3 → Coin) : Fin 3 :=
  match weigh (coins 0) (coins 1) with
  | WeighResult.Equal => 2
  | WeighResult.LeftLighter => 0
  | WeighResult.RightLighter => 1

theorem counterfeit_identification_possible :
  ∀ (coins : Fin 3 → Coin),
  (∃! i, coins i = Coin.Counterfeit) →
  coins (identifyCounterfeit coins) = Coin.Counterfeit :=
sorry


end counterfeit_identification_possible_l746_74659


namespace drain_rate_calculation_l746_74692

/-- Represents the filling and draining system of a tank -/
structure TankSystem where
  capacity : ℝ
  fill_rate_A : ℝ
  fill_rate_B : ℝ
  drain_rate_C : ℝ
  cycle_time : ℝ
  total_time : ℝ

/-- Theorem stating the drain rate of pipe C given the system conditions -/
theorem drain_rate_calculation (s : TankSystem)
  (h1 : s.capacity = 950)
  (h2 : s.fill_rate_A = 40)
  (h3 : s.fill_rate_B = 30)
  (h4 : s.cycle_time = 3)
  (h5 : s.total_time = 57)
  (h6 : (s.total_time / s.cycle_time) * (s.fill_rate_A + s.fill_rate_B - s.drain_rate_C) = s.capacity) :
  s.drain_rate_C = 20 := by
  sorry

#check drain_rate_calculation

end drain_rate_calculation_l746_74692


namespace not_prime_3999991_l746_74604

theorem not_prime_3999991 : ¬ Nat.Prime 3999991 :=
  sorry

end not_prime_3999991_l746_74604


namespace largest_angle_measure_l746_74696

/-- A triangle XYZ is obtuse and isosceles with one of the equal angles measuring 30 degrees. -/
structure ObtuseIsoscelesTriangle where
  X : ℝ
  Y : ℝ
  Z : ℝ
  sum_180 : X + Y + Z = 180
  obtuse : Z > 90
  isosceles : X = Y
  x_measure : X = 30

/-- The largest interior angle of an obtuse isosceles triangle with one equal angle measuring 30 degrees is 120 degrees. -/
theorem largest_angle_measure (t : ObtuseIsoscelesTriangle) : t.Z = 120 := by
  sorry

end largest_angle_measure_l746_74696


namespace product_equivalence_l746_74619

theorem product_equivalence : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * 
  (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 
  5^128 - 4^128 := by sorry

end product_equivalence_l746_74619


namespace smallest_subtraction_for_divisibility_l746_74687

theorem smallest_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 100 ∧ (427751 - x) % 101 = 0 ∧ ∀ y : ℕ, y < x → (427751 - y) % 101 ≠ 0 :=
by sorry

end smallest_subtraction_for_divisibility_l746_74687


namespace combinations_count_l746_74682

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the total value we're aiming for in cents -/
def total_value : ℕ := 30

/-- 
  Counts the number of non-negative integer solutions (p, n, d) to the equation 
  p * penny_value + n * nickel_value + d * dime_value = total_value
-/
def count_combinations : ℕ := sorry

theorem combinations_count : count_combinations = 20 := by sorry

end combinations_count_l746_74682


namespace cost_price_calculation_l746_74658

theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  marked_price = 65 ∧ 
  discount_rate = 0.05 ∧ 
  profit_rate = 0.30 →
  ∃ (cost_price : ℝ),
    cost_price = 47.50 ∧
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) := by
  sorry

end cost_price_calculation_l746_74658


namespace meal_combinations_l746_74607

theorem meal_combinations (n : ℕ) (h : n = 15) : n * (n - 1) = 210 := by
  sorry

end meal_combinations_l746_74607


namespace prop_one_prop_two_l746_74666

-- Proposition 1
theorem prop_one (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a > b) :
  a - 1 / a > b - 1 / b :=
sorry

-- Proposition 2
theorem prop_two (a b : ℝ) (hb : b ≠ 0) :
  b * (b - a) ≤ 0 ↔ a / b ≥ 1 :=
sorry

end prop_one_prop_two_l746_74666


namespace possible_sets_for_B_l746_74631

def set_A : Set ℕ := {1, 2}
def set_B : Set ℕ := {1, 2, 3, 4}

theorem possible_sets_for_B (B : Set ℕ) 
  (h1 : set_A ⊆ B) (h2 : B ⊆ set_B) :
  B = set_A ∨ B = {1, 2, 3} ∨ B = {1, 2, 4} :=
sorry

end possible_sets_for_B_l746_74631


namespace g_of_3_l746_74661

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 - 7 * x + 3

theorem g_of_3 : g 3 = 126 := by
  sorry

end g_of_3_l746_74661


namespace charity_race_total_l746_74667

/-- Represents the total amount raised by students in a charity race -/
def total_raised (
  total_students : ℕ
  ) (
  group_a_students : ℕ
  ) (
  group_b_students : ℕ
  ) (
  group_c_students : ℕ
  ) (
  group_a_race_amount : ℕ
  ) (
  group_a_extra_amount : ℕ
  ) (
  group_b_race_amount : ℕ
  ) (
  group_b_extra_amount : ℕ
  ) (
  group_c_race_amount : ℕ
  ) (
  group_c_extra_total : ℕ
  ) : ℕ :=
  (group_a_students * (group_a_race_amount + group_a_extra_amount)) +
  (group_b_students * (group_b_race_amount + group_b_extra_amount)) +
  (group_c_students * group_c_race_amount + group_c_extra_total)

/-- Theorem stating that the total amount raised is $1080 -/
theorem charity_race_total :
  total_raised 30 10 12 8 20 5 30 10 25 150 = 1080 := by
  sorry

end charity_race_total_l746_74667


namespace set_intersection_range_l746_74621

theorem set_intersection_range (m : ℝ) : 
  let A : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  let B : Set ℝ := {x | x^2 - 7*x + 10 ≤ 0}
  A ∩ B = A → 2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end set_intersection_range_l746_74621


namespace right_triangle_perimeter_l746_74629

theorem right_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (1/2) * a * b = 150) (h_leg : a = 30) : 
  a + b + c = 40 + 10 * Real.sqrt 10 := by
  sorry

end right_triangle_perimeter_l746_74629


namespace right_triangle_acute_angles_l746_74670

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ -- Angles are positive
  a + b = 90 ∧ -- Sum of acute angles in a right triangle
  b = 4 * a -- Ratio of angles is 4:1
  → a = 18 ∧ b = 72 := by
  sorry

end right_triangle_acute_angles_l746_74670


namespace table_price_is_300_l746_74656

def table_selling_price (num_trees : ℕ) (planks_per_tree : ℕ) (planks_per_table : ℕ) 
                        (labor_cost : ℕ) (profit : ℕ) : ℕ :=
  let total_planks := num_trees * planks_per_tree
  let num_tables := total_planks / planks_per_table
  let total_revenue := labor_cost + profit
  total_revenue / num_tables

theorem table_price_is_300 :
  table_selling_price 30 25 15 3000 12000 = 300 := by
  sorry

end table_price_is_300_l746_74656


namespace total_pets_is_415_l746_74602

/-- The number of dogs at the farm -/
def num_dogs : ℕ := 43

/-- The number of fish at the farm -/
def num_fish : ℕ := 72

/-- The number of cats at the farm -/
def num_cats : ℕ := 34

/-- The number of chickens at the farm -/
def num_chickens : ℕ := 120

/-- The number of rabbits at the farm -/
def num_rabbits : ℕ := 57

/-- The number of parrots at the farm -/
def num_parrots : ℕ := 89

/-- The total number of pets at the farm -/
def total_pets : ℕ := num_dogs + num_fish + num_cats + num_chickens + num_rabbits + num_parrots

theorem total_pets_is_415 : total_pets = 415 := by
  sorry

end total_pets_is_415_l746_74602


namespace multiplication_simplification_l746_74672

theorem multiplication_simplification : 11 * (1 / 17) * 34 = 22 := by
  sorry

end multiplication_simplification_l746_74672


namespace equation_solution_l746_74676

theorem equation_solution (x : ℝ) : 4 / (1 + 3/x) = 1 → x = 1 := by
  sorry

end equation_solution_l746_74676


namespace gumball_calculation_l746_74695

/-- The number of gumballs originally in the dispenser -/
def original_gumballs : ℝ := 100

/-- The fraction of gumballs remaining after each day -/
def daily_remaining_fraction : ℝ := 0.7

/-- The number of days that have passed -/
def days : ℕ := 3

/-- The number of gumballs remaining after 3 days -/
def remaining_gumballs : ℝ := 34.3

/-- Theorem stating that the original number of gumballs is correct -/
theorem gumball_calculation :
  original_gumballs * daily_remaining_fraction ^ days = remaining_gumballs := by
  sorry

end gumball_calculation_l746_74695


namespace range_of_a_l746_74688

theorem range_of_a (a : ℝ) : 
  ((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∨ 
   (∃ x : ℝ, x^2 - x + a = 0)) ∧ 
  ¬((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
    (∃ x : ℝ, x^2 - x + a = 0)) ↔ 
  a < 0 ∨ (1/4 < a ∧ a < 4) :=
by sorry

end range_of_a_l746_74688


namespace limit_of_expression_l746_74677

theorem limit_of_expression (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    |((4 * (n : ℝ)^2 + 4 * n - 1) / (4 * (n : ℝ)^2 + 2 * n + 3))^(1 - 2 * n) - Real.exp (-1)| < ε :=
sorry

end limit_of_expression_l746_74677


namespace probability_theorem_l746_74691

/-- The probability of selecting exactly one high-quality item and one defective item
    from a set of 4 high-quality items and 1 defective item, when two items are randomly selected. -/
def probability_one_high_quality_one_defective : ℚ := 2 / 5

/-- The number of high-quality items -/
def num_high_quality : ℕ := 4

/-- The number of defective items -/
def num_defective : ℕ := 1

/-- The total number of items -/
def total_items : ℕ := num_high_quality + num_defective

/-- The number of items to be selected -/
def items_to_select : ℕ := 2

/-- Theorem stating that the probability of selecting exactly one high-quality item
    and one defective item is 2/5 -/
theorem probability_theorem :
  probability_one_high_quality_one_defective =
    (num_high_quality.choose 1 * num_defective.choose 1 : ℚ) /
    (total_items.choose items_to_select : ℚ) :=
by sorry

end probability_theorem_l746_74691


namespace rational_function_sum_l746_74675

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  h_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  h_horiz_asymp : ∀ ε > 0, ∃ M, ∀ x, |x| > M → |p x / q x| < ε
  h_vert_asymp : ContinuousAt q (-2) ∧ q (-2) = 0
  h_p3 : p 3 = 1
  h_q3 : q 3 = 4

/-- The main theorem -/
theorem rational_function_sum (f : RationalFunction) : 
  ∀ x, f.p x + f.q x = (4 * x^2 + 7 * x - 9) / 10 := by
  sorry

end rational_function_sum_l746_74675


namespace arithmetic_geometric_mean_problem_l746_74624

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 100) :
  x^2 + y^2 = 1400 := by
  sorry

end arithmetic_geometric_mean_problem_l746_74624


namespace bacon_percentage_of_total_l746_74650

def total_sandwich_calories : ℕ := 1250
def bacon_strips : ℕ := 2
def calories_per_bacon_strip : ℕ := 125

def bacon_calories : ℕ := bacon_strips * calories_per_bacon_strip

theorem bacon_percentage_of_total (h : bacon_calories = bacon_strips * calories_per_bacon_strip) :
  (bacon_calories : ℚ) / total_sandwich_calories * 100 = 20 := by
  sorry

end bacon_percentage_of_total_l746_74650


namespace twenty_is_eighty_percent_of_twentyfive_l746_74679

theorem twenty_is_eighty_percent_of_twentyfive : 
  ∃ x : ℝ, (20 : ℝ) / x = (80 : ℝ) / 100 ∧ x = 25 := by sorry

end twenty_is_eighty_percent_of_twentyfive_l746_74679


namespace average_speed_calculation_l746_74690

theorem average_speed_calculation (distance : ℝ) (time : ℝ) (average_speed : ℝ) :
  distance = 210 →
  time = 4.5 →
  average_speed = distance / time →
  average_speed = 140 / 3 := by
  sorry

end average_speed_calculation_l746_74690


namespace decagon_diagonals_l746_74663

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals :
  num_diagonals decagon_sides = 35 := by sorry

end decagon_diagonals_l746_74663


namespace ship_length_proof_l746_74608

/-- The length of the ship in meters -/
def ship_length : ℝ := 72

/-- The speed of the ship in meters per second -/
def ship_speed : ℝ := 4

/-- Emily's walking speed in meters per second -/
def emily_speed : ℝ := 6

/-- The number of steps Emily takes from back to front of the ship -/
def steps_back_to_front : ℕ := 300

/-- The number of steps Emily takes from front to back of the ship -/
def steps_front_to_back : ℕ := 60

/-- The length of each of Emily's steps in meters -/
def step_length : ℝ := 2

theorem ship_length_proof :
  let relative_speed_forward := emily_speed - ship_speed
  let relative_speed_backward := emily_speed + ship_speed
  let distance_forward := steps_back_to_front * step_length
  let distance_backward := steps_front_to_back * step_length
  let time_forward := distance_forward / relative_speed_forward
  let time_backward := distance_backward / relative_speed_backward
  ship_length = distance_forward - ship_speed * time_forward ∧
  ship_length = distance_backward + ship_speed * time_backward :=
by sorry

end ship_length_proof_l746_74608


namespace intersection_perpendicular_bisector_l746_74652

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem intersection_perpendicular_bisector :
  ∀ A B : ℝ × ℝ,
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B →
  ∀ x y : ℝ,
  perp_bisector x y ↔
  (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end intersection_perpendicular_bisector_l746_74652


namespace a_divides_iff_k_divides_l746_74643

/-- Definition of a_n as the integer consisting of n repetitions of the digit 1 in base 10 -/
def a (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem stating that a_k divides a_l if and only if k divides l -/
theorem a_divides_iff_k_divides (k l : ℕ) (h : k ≥ 1) :
  (a k ∣ a l) ↔ k ∣ l :=
by sorry

end a_divides_iff_k_divides_l746_74643


namespace line_equation_45_degree_slope_2_intercept_l746_74665

/-- The equation of a line with a slope angle of 45° and a y-intercept of 2 is y = x + 2 -/
theorem line_equation_45_degree_slope_2_intercept :
  let slope_angle : Real := 45 * (π / 180)  -- Convert 45° to radians
  let y_intercept : Real := 2
  let slope : Real := Real.tan slope_angle
  ∀ x y : Real, y = slope * x + y_intercept ↔ y = x + 2 := by
  sorry

end line_equation_45_degree_slope_2_intercept_l746_74665


namespace complex_number_pure_imaginary_l746_74620

/-- Given a complex number z = (m-1) + (m+1)i where m is a real number and z is pure imaginary, prove that m = 1 -/
theorem complex_number_pure_imaginary (m : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk (m - 1) (m + 1))
  (h2 : z.re = 0) : m = 1 := by
  sorry

end complex_number_pure_imaginary_l746_74620


namespace marilyns_bottle_caps_l746_74634

/-- The problem of Marilyn's bottle caps -/
theorem marilyns_bottle_caps :
  ∀ (initial : ℕ), 
    (initial - 36 = 15) → 
    initial = 51 := by
  sorry

end marilyns_bottle_caps_l746_74634


namespace steves_pencils_l746_74655

/-- Steve's pencil distribution problem -/
theorem steves_pencils (boxes : ℕ) (pencils_per_box : ℕ) (lauren_pencils : ℕ) (matt_extra : ℕ) :
  boxes = 2 →
  pencils_per_box = 12 →
  lauren_pencils = 6 →
  matt_extra = 3 →
  boxes * pencils_per_box - lauren_pencils - (lauren_pencils + matt_extra) = 9 :=
by sorry

end steves_pencils_l746_74655
