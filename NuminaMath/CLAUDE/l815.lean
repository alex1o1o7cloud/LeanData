import Mathlib

namespace point_60_coordinates_l815_81516

/-- Defines the x-coordinate of the nth point in the sequence -/
def x (n : ℕ) : ℕ := sorry

/-- Defines the y-coordinate of the nth point in the sequence -/
def y (n : ℕ) : ℕ := sorry

/-- The sum of x and y coordinates for the nth point -/
def sum (n : ℕ) : ℕ := x n + y n

theorem point_60_coordinates :
  x 60 = 5 ∧ y 60 = 7 := by sorry

end point_60_coordinates_l815_81516


namespace wire_cutting_l815_81520

theorem wire_cutting (total_length piece_length : ℝ) 
  (h1 : total_length = 27.9)
  (h2 : piece_length = 3.1) : 
  ⌊total_length / piece_length⌋ = 9 := by
  sorry

end wire_cutting_l815_81520


namespace largest_integer_K_l815_81500

theorem largest_integer_K : ∃ K : ℕ, (∀ n : ℕ, n^200 < 5^300 → n ≤ K) ∧ K^200 < 5^300 ∧ K = 11 := by
  sorry

end largest_integer_K_l815_81500


namespace exact_four_white_probability_l815_81562

-- Define the number of balls
def n : ℕ := 8

-- Define the probability of a ball being white (or black)
def p : ℚ := 1/2

-- Define the number of white balls we're interested in
def k : ℕ := 4

-- State the theorem
theorem exact_four_white_probability :
  (n.choose k) * p^k * (1 - p)^(n - k) = 35/128 := by
  sorry

end exact_four_white_probability_l815_81562


namespace average_age_decrease_l815_81595

def original_strength : ℕ := 8
def original_average_age : ℕ := 40
def new_students : ℕ := 8
def new_students_average_age : ℕ := 32

theorem average_age_decrease :
  let original_total_age := original_strength * original_average_age
  let new_total_age := original_total_age + new_students * new_students_average_age
  let new_total_strength := original_strength + new_students
  let new_average_age := new_total_age / new_total_strength
  original_average_age - new_average_age = 4 := by
sorry

end average_age_decrease_l815_81595


namespace range_of_complex_function_l815_81599

theorem range_of_complex_function (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (a b : ℝ), a = Real.sqrt 2 - 1 ∧ b = Real.sqrt 2 + 1 ∧
  ∀ θ : ℝ, a ≤ Complex.abs (z^2 + Complex.I * z^2 + 1) ∧
           Complex.abs (z^2 + Complex.I * z^2 + 1) ≤ b :=
by sorry

end range_of_complex_function_l815_81599


namespace lawn_mowing_problem_l815_81537

-- Define the rates and time
def mary_rate : ℚ := 1 / 3
def tom_rate : ℚ := 1 / 5
def work_time : ℚ := 3 / 2

-- Define the theorem
theorem lawn_mowing_problem :
  let combined_rate := mary_rate + tom_rate
  let mowed_fraction := work_time * combined_rate
  1 - mowed_fraction = 1 / 5 := by sorry

end lawn_mowing_problem_l815_81537


namespace five_or_king_probability_l815_81504

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the ranks in a standard deck -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents the suits in a standard deck -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- A card in the deck -/
structure Card :=
  (rank : Rank)
  (suit : Suit)

/-- The probability of drawing a specific card from the deck -/
def draw_probability (d : Deck) (c : Card) : ℚ :=
  1 / 52

/-- The probability of drawing a card with a specific rank -/
def draw_rank_probability (d : Deck) (r : Rank) : ℚ :=
  4 / 52

/-- Theorem: The probability of drawing either a 5 or a King from a standard 52-card deck is 2/13 -/
theorem five_or_king_probability (d : Deck) : 
  draw_rank_probability d Rank.Five + draw_rank_probability d Rank.King = 2 / 13 := by
  sorry


end five_or_king_probability_l815_81504


namespace on_y_axis_on_x_axis_abscissa_greater_than_ordinate_l815_81503

-- Define point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (2*m + 4, m - 1)

-- Theorem for condition (1)
theorem on_y_axis (m : ℝ) : 
  P m = (0, -3) ↔ P m = (0, (P m).2) :=
sorry

-- Theorem for condition (2)
theorem on_x_axis (m : ℝ) :
  P m = (6, 0) ↔ P m = ((P m).1, 0) :=
sorry

-- Theorem for condition (3)
theorem abscissa_greater_than_ordinate (m : ℝ) :
  P m = (-4, -5) ↔ (P m).1 = (P m).2 + 1 :=
sorry

end on_y_axis_on_x_axis_abscissa_greater_than_ordinate_l815_81503


namespace parabola_directrix_tangent_circle_l815_81536

/-- The value of p for a parabola y^2 = 2px (p > 0) with directrix tangent to the circle x^2 + y^2 + 2x = 0 -/
theorem parabola_directrix_tangent_circle (p : ℝ) : 
  p > 0 ∧ 
  (∃ x y : ℝ, y^2 = 2*p*x) ∧
  (∃ x y : ℝ, x^2 + y^2 + 2*x = 0) ∧
  (∃ x : ℝ, x = -p/2) ∧  -- directrix equation
  (∃ x y : ℝ, x^2 + y^2 + 2*x = 0 ∧ x = -p/2)  -- tangency condition
  → p = 4 := by
sorry

end parabola_directrix_tangent_circle_l815_81536


namespace three_eighths_count_l815_81544

theorem three_eighths_count : (8 + 5/3 - 3) / (3/8) = 160/9 := by sorry

end three_eighths_count_l815_81544


namespace second_rectangle_width_l815_81532

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: Given two rectangles with specified properties, the width of the second rectangle is 3 inches -/
theorem second_rectangle_width (r1 r2 : Rectangle) : 
  r1.width = 4 → 
  r1.height = 5 → 
  r2.height = 6 → 
  area r1 = area r2 + 2 → 
  r2.width = 3 := by
  sorry

end second_rectangle_width_l815_81532


namespace equal_sum_black_white_cells_l815_81522

/-- Represents a cell in the Pythagorean multiplication table frame -/
structure Cell where
  row : ℕ
  col : ℕ
  value : ℕ
  isBlack : Bool

/-- Represents a rectangular frame in the Pythagorean multiplication table -/
structure Frame where
  width : ℕ
  height : ℕ
  cells : List Cell

def isPythagoreanMultiplicationTable (frame : Frame) : Prop :=
  ∀ cell ∈ frame.cells, cell.value = cell.row * cell.col

def hasOddSidedFrame (frame : Frame) : Prop :=
  Odd frame.width ∧ Odd frame.height

def hasAlternatingColors (frame : Frame) : Prop :=
  ∀ i j, i + j ≡ 0 [MOD 2] → 
    (∃ cell ∈ frame.cells, cell.row = i ∧ cell.col = j ∧ cell.isBlack)

def hasBlackCorners (frame : Frame) : Prop :=
  ∀ cell ∈ frame.cells, (cell.row = 1 ∨ cell.row = frame.height) ∧ 
                        (cell.col = 1 ∨ cell.col = frame.width) → 
                        cell.isBlack

def sumOfBlackCells (frame : Frame) : ℕ :=
  (frame.cells.filter (·.isBlack)).map (·.value) |> List.sum

def sumOfWhiteCells (frame : Frame) : ℕ :=
  (frame.cells.filter (¬·.isBlack)).map (·.value) |> List.sum

theorem equal_sum_black_white_cells (frame : Frame) 
  (h1 : isPythagoreanMultiplicationTable frame)
  (h2 : hasOddSidedFrame frame)
  (h3 : hasAlternatingColors frame)
  (h4 : hasBlackCorners frame) :
  sumOfBlackCells frame = sumOfWhiteCells frame :=
sorry

end equal_sum_black_white_cells_l815_81522


namespace pizza_toppings_l815_81541

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 20)
  (h2 : pepperoni_slices = 12)
  (h3 : mushroom_slices = 14)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 6 :=
by sorry

end pizza_toppings_l815_81541


namespace cylinder_volume_ratio_l815_81555

/-- The ratio of volumes of cylinders formed by rolling a 6x9 rectangle -/
theorem cylinder_volume_ratio : 
  let length : ℝ := 6
  let width : ℝ := 9
  let volume1 := π * (length / (2 * π))^2 * width
  let volume2 := π * (width / (2 * π))^2 * length
  volume2 / volume1 = 3 / 2 := by
  sorry

end cylinder_volume_ratio_l815_81555


namespace expression_evaluation_l815_81549

theorem expression_evaluation (a b : ℤ) (h1 : a = 2) (h2 : b = -1) :
  (2 * a^2 * b - 4 * a * b^2) - 2 * (a * b^2 + a^2 * b) = -12 := by
  sorry

end expression_evaluation_l815_81549


namespace ellipse_eccentricity_range_l815_81583

/-- The eccentricity of an ellipse with a perpendicular bisector through a point on the ellipse --/
theorem ellipse_eccentricity_range (a b : ℝ) (h_pos : 0 < b ∧ b < a) :
  let e := Real.sqrt (1 - b^2 / a^2)
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    x^2 + y^2 = (a^2 - b^2)) → 
    Real.sqrt 2 / 2 ≤ e ∧ e < 1 := by
  sorry

end ellipse_eccentricity_range_l815_81583


namespace equilateral_triangle_perimeter_l815_81501

/-- An equilateral triangle with area twice the side length has perimeter 8√3 -/
theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l815_81501


namespace jerusha_earned_68_l815_81578

/-- Jerusha's earnings given Lottie's earnings and their total earnings -/
def jerushas_earnings (lotties_earnings : ℚ) (total_earnings : ℚ) : ℚ :=
  4 * lotties_earnings

theorem jerusha_earned_68 :
  ∃ (lotties_earnings : ℚ),
    jerushas_earnings lotties_earnings 85 = 68 ∧
    lotties_earnings + jerushas_earnings lotties_earnings 85 = 85 := by
  sorry

end jerusha_earned_68_l815_81578


namespace part1_part2_l815_81569

-- Define the polynomials A and B
def A (x : ℝ) : ℝ := x^2 - x
def B (m : ℝ) (x : ℝ) : ℝ := m * x + 1

-- Part 1: Prove that when ■ = 2, 2A - B = 2x^2 - 4x - 1
theorem part1 (x : ℝ) : 2 * A x - B 2 x = 2 * x^2 - 4 * x - 1 := by
  sorry

-- Part 2: Prove that when A - B does not contain x terms, ■ = -1
theorem part2 (x : ℝ) : (∀ m : ℝ, A x - B m x = (-1 : ℝ)) → m = -1 := by
  sorry

end part1_part2_l815_81569


namespace sqrt_equation_solution_l815_81509

theorem sqrt_equation_solution (x : ℝ) :
  x > 6 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3) ↔
  x ≥ 18 :=
by sorry

end sqrt_equation_solution_l815_81509


namespace inequality_proof_l815_81568

theorem inequality_proof (a b c A α : Real) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hA : a + b + c = A) (hA1 : A ≤ 1) (hα : α > 0) : 
  (1/a - a)^α + (1/b - b)^α + (1/c - c)^α ≥ 3*(3/A - A/3)^α := by
  sorry

#check inequality_proof

end inequality_proof_l815_81568


namespace train_journey_constant_speed_time_l815_81502

/-- Represents the journey of a train with uniform acceleration, constant speed, and uniform deceleration phases. -/
structure TrainJourney where
  totalDistance : ℝ  -- in km
  totalTime : ℝ      -- in hours
  constantSpeed : ℝ  -- in km/h

/-- Calculates the time spent at constant speed during the journey. -/
def timeAtConstantSpeed (journey : TrainJourney) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, the time at constant speed is 1/5 hours (12 minutes). -/
theorem train_journey_constant_speed_time 
  (journey : TrainJourney) 
  (h1 : journey.totalDistance = 21) 
  (h2 : journey.totalTime = 4/15)
  (h3 : journey.constantSpeed = 90) :
  timeAtConstantSpeed journey = 1/5 := by
  sorry

end train_journey_constant_speed_time_l815_81502


namespace min_value_theorem_min_value_achievable_l815_81573

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  2 * x + 1 / (x + 3) ≥ 2 * Real.sqrt 2 - 6 :=
by sorry

theorem min_value_achievable :
  ∃ x > -3, 2 * x + 1 / (x + 3) = 2 * Real.sqrt 2 - 6 :=
by sorry

end min_value_theorem_min_value_achievable_l815_81573


namespace factorial_sum_of_squares_solutions_l815_81564

theorem factorial_sum_of_squares_solutions :
  ∀ a b n : ℕ+,
    a ≤ b →
    n < 14 →
    a^2 + b^2 = n! →
    ((a = 1 ∧ b = 1 ∧ n = 2) ∨ (a = 12 ∧ b = 24 ∧ n = 6)) :=
by sorry

end factorial_sum_of_squares_solutions_l815_81564


namespace equation_solution_l815_81589

theorem equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x * (4 : ℝ)^x * (16 : ℝ)^(x + 1) = (1024 : ℝ)^2 ∧ x = 2 := by
  sorry

end equation_solution_l815_81589


namespace max_trig_product_bound_max_trig_product_achievable_l815_81535

theorem max_trig_product_bound (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 9 / 2 :=
sorry

theorem max_trig_product_achievable :
  ∃ x y z : ℝ,
    (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
    (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) = 9 / 2 :=
sorry

end max_trig_product_bound_max_trig_product_achievable_l815_81535


namespace expansion_properties_l815_81548

theorem expansion_properties (x : ℝ) (x_ne_zero : x ≠ 0) : 
  let expansion := (1/x - x)^6
  ∃ (coeffs : List ℤ), 
    -- The expansion can be represented as a list of integer coefficients
    (∀ i, 0 ≤ i ∧ i < 7 → coeffs.get! i = (Nat.choose 6 i) * (-1)^i) ∧
    -- The binomial coefficient of the 4th term is the largest
    (∀ i, 0 ≤ i ∧ i < 7 → coeffs.get! 3 ≥ coeffs.get! i) ∧
    -- The sum of all coefficients is 0
    (coeffs.sum = 0) := by
  sorry

end expansion_properties_l815_81548


namespace sum_of_quadratic_roots_l815_81559

theorem sum_of_quadratic_roots (x₁ x₂ : ℝ) : 
  (-48 : ℝ) * x₁^2 + 110 * x₁ + 165 = 0 ∧
  (-48 : ℝ) * x₂^2 + 110 * x₂ + 165 = 0 →
  x₁ + x₂ = 55 / 24 := by
sorry

end sum_of_quadratic_roots_l815_81559


namespace complex_expression_evaluation_l815_81534

theorem complex_expression_evaluation :
  - Real.sqrt 3 * Real.sqrt 6 + abs (1 - Real.sqrt 2) - (1/3)⁻¹ = -4 * Real.sqrt 2 - 4 := by
  sorry

end complex_expression_evaluation_l815_81534


namespace round_trip_completion_percentage_l815_81550

/-- Calculates the completion percentage of a round-trip given delays on the outbound journey -/
theorem round_trip_completion_percentage 
  (T : ℝ) -- Normal one-way travel time
  (h1 : T > 0) -- Assumption that travel time is positive
  (traffic_delay : ℝ := 0.15) -- 15% increase due to traffic
  (construction_delay : ℝ := 0.10) -- 10% increase due to construction
  (return_completion : ℝ := 0.20) -- 20% of return journey completed
  : (T * (1 + traffic_delay + construction_delay) + return_completion * T) / (2 * T) = 0.725 := by
sorry

end round_trip_completion_percentage_l815_81550


namespace journey_distance_is_correct_l815_81567

/-- Represents the cab fare structure and journey details -/
structure CabJourney where
  baseFare : ℝ
  peakRateFirst2Miles : ℝ
  peakRateAfter2Miles : ℝ
  toll1 : ℝ
  toll2 : ℝ
  tipPercentage : ℝ
  totalPaid : ℝ

/-- Calculates the distance of the journey based on the given fare structure and total paid -/
def calculateDistance (journey : CabJourney) : ℝ :=
  sorry

/-- Theorem stating that the calculated distance for the given journey is 6.58 miles -/
theorem journey_distance_is_correct (journey : CabJourney) 
  (h1 : journey.baseFare = 3)
  (h2 : journey.peakRateFirst2Miles = 5)
  (h3 : journey.peakRateAfter2Miles = 4)
  (h4 : journey.toll1 = 1.5)
  (h5 : journey.toll2 = 2.5)
  (h6 : journey.tipPercentage = 0.15)
  (h7 : journey.totalPaid = 39.57) :
  calculateDistance journey = 6.58 := by
  sorry

end journey_distance_is_correct_l815_81567


namespace f_positive_iff_l815_81577

-- Define the function
def f (x : ℝ) := x^2 + x - 12

-- State the theorem
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x < -4 ∨ x > 3 := by sorry

end f_positive_iff_l815_81577


namespace angle_in_second_quadrant_l815_81539

theorem angle_in_second_quadrant (θ : Real) : 
  (Real.tan θ * Real.sin θ < 0) → 
  (Real.tan θ * Real.cos θ > 0) → 
  (0 < θ) ∧ (θ < Real.pi) := by
sorry

end angle_in_second_quadrant_l815_81539


namespace ellipse_properties_l815_81579

-- Define the ellipses
def ellipse_N (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

def ellipse_M (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem ellipse_properties (a b : ℝ) :
  (∀ x y, ellipse_M a b x y ↔ ellipse_N x y) →  -- M and N share foci
  (ellipse_M a b 0 2) →  -- M passes through (0,2)
  (∃ A B : ℝ × ℝ, 
    ellipse_M a b A.1 A.2 ∧ 
    ellipse_M a b B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A.1 > B.1) →  -- A and B are intersections of M and y = x + 2, with A to the right of B
  (2 * a = 4 * Real.sqrt 2) ∧  -- Length of major axis is 4√2
  (∃ A B : ℝ × ℝ, 
    ellipse_M a b A.1 A.2 ∧ 
    ellipse_M a b B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A.1 > B.1 ∧ 
    A.1 * B.1 + A.2 * B.2 = -4/3) :=  -- Dot product of OA and OB is -4/3
by
  sorry

end ellipse_properties_l815_81579


namespace simplify_fraction_product_l815_81585

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end simplify_fraction_product_l815_81585


namespace perfect_square_trinomial_factor_difference_of_cubes_l815_81592

-- Theorem 1
theorem perfect_square_trinomial (m : ℝ) : m^2 - 10*m + 25 = (m - 5)^2 := by
  sorry

-- Theorem 2
theorem factor_difference_of_cubes (a b : ℝ) : a^3*b - a*b = a*b*(a + 1)*(a - 1) := by
  sorry

end perfect_square_trinomial_factor_difference_of_cubes_l815_81592


namespace rohan_entertainment_expenses_l815_81524

/-- Proves that Rohan's entertainment expenses are 10% of his salary -/
theorem rohan_entertainment_expenses :
  let salary : ℝ := 7500
  let food_percent : ℝ := 40
  let rent_percent : ℝ := 20
  let conveyance_percent : ℝ := 10
  let savings : ℝ := 1500
  let entertainment_percent : ℝ := 100 - (food_percent + rent_percent + conveyance_percent + (savings / salary * 100))
  entertainment_percent = 10 := by
  sorry

end rohan_entertainment_expenses_l815_81524


namespace race_distance_proof_l815_81563

/-- The distance in meters by which runner A beats runner B -/
def beat_distance : ℝ := 56

/-- The time in seconds by which runner A beats runner B -/
def beat_time : ℝ := 7

/-- Runner A's time to complete the race in seconds -/
def a_time : ℝ := 8

/-- The total distance of the race in meters -/
def race_distance : ℝ := 120

theorem race_distance_proof :
  (beat_distance / beat_time) * (a_time + beat_time) = race_distance :=
sorry

end race_distance_proof_l815_81563


namespace wade_team_score_l815_81533

/-- Calculates the total points scored by a basketball team after a given number of games -/
def team_total_points (wade_avg : ℕ) (teammates_avg : ℕ) (games : ℕ) : ℕ :=
  (wade_avg + teammates_avg) * games

/-- Proves that Wade's team scores 300 points in 5 games -/
theorem wade_team_score : team_total_points 20 40 5 = 300 := by
  sorry

end wade_team_score_l815_81533


namespace joe_fruit_probability_l815_81584

def num_fruit_types : ℕ := 4
def num_meals : ℕ := 3

def probability_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals * num_fruit_types

theorem joe_fruit_probability :
  1 - probability_same_fruit = 15/16 := by sorry

end joe_fruit_probability_l815_81584


namespace quadratic_equation_solution_l815_81545

/-- The quadratic equation (a-2)x^2 + x + a^2 - 4 = 0 has 0 as one of its roots -/
def has_zero_root (a : ℝ) : Prop :=
  ∃ x : ℝ, (a - 2) * x^2 + x + a^2 - 4 = 0 ∧ x = 0

/-- The value of a that satisfies the condition -/
def solution : ℝ := -2

theorem quadratic_equation_solution :
  ∀ a : ℝ, has_zero_root a → a = solution :=
sorry

end quadratic_equation_solution_l815_81545


namespace no_prime_roots_sum_65_l815_81510

theorem no_prime_roots_sum_65 : ¬∃ (p q k : ℕ), Prime p ∧ Prime q ∧ p + q = 65 ∧ p * q = k ∧ p^2 - 65*p + k = 0 ∧ q^2 - 65*q + k = 0 := by
  sorry

end no_prime_roots_sum_65_l815_81510


namespace train_average_speed_l815_81574

def train_distance_1 : ℝ := 240
def train_time_1 : ℝ := 3
def train_distance_2 : ℝ := 450
def train_time_2 : ℝ := 5

theorem train_average_speed :
  (train_distance_1 + train_distance_2) / (train_time_1 + train_time_2) = 86.25 := by
  sorry

end train_average_speed_l815_81574


namespace EDTA_Ca_complex_weight_l815_81530

-- Define the molecular weight of EDTA
def EDTA_weight : ℝ := 292.248

-- Define the atomic weight of calcium
def Ca_weight : ℝ := 40.08

-- Define the complex ratio
def complex_ratio : ℕ := 1

-- Theorem statement
theorem EDTA_Ca_complex_weight :
  complex_ratio * (EDTA_weight + Ca_weight) = 332.328 := by sorry

end EDTA_Ca_complex_weight_l815_81530


namespace tims_movie_marathon_l815_81575

/-- The duration of Tim's movie marathon --/
def movie_marathon_duration (first_movie : ℝ) (second_movie_percentage : ℝ) (third_movie_difference : ℝ) : ℝ :=
  let second_movie := first_movie * (1 + second_movie_percentage)
  let first_two := first_movie + second_movie
  let third_movie := first_two - third_movie_difference
  first_movie + second_movie + third_movie

/-- Theorem stating the duration of Tim's movie marathon --/
theorem tims_movie_marathon :
  movie_marathon_duration 2 0.5 1 = 9 := by
  sorry

end tims_movie_marathon_l815_81575


namespace optimal_sampling_methods_l815_81557

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents income levels --/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a community with families of different income levels --/
structure Community where
  totalFamilies : ℕ
  highIncomeFamilies : ℕ
  middleIncomeFamilies : ℕ
  lowIncomeFamilies : ℕ

/-- Represents a group of volleyball players --/
structure VolleyballTeam where
  totalPlayers : ℕ

/-- Determines the optimal sampling method for a given community and sample size --/
def optimalSamplingMethodForCommunity (c : Community) (sampleSize : ℕ) : SamplingMethod :=
  sorry

/-- Determines the optimal sampling method for a volleyball team and selection size --/
def optimalSamplingMethodForTeam (t : VolleyballTeam) (selectionSize : ℕ) : SamplingMethod :=
  sorry

/-- The main theorem stating the optimal sampling methods for the given scenarios --/
theorem optimal_sampling_methods 
  (community : Community)
  (team : VolleyballTeam)
  (h1 : community.totalFamilies = 400)
  (h2 : community.highIncomeFamilies = 120)
  (h3 : community.middleIncomeFamilies = 180)
  (h4 : community.lowIncomeFamilies = 100)
  (h5 : team.totalPlayers = 12) :
  (optimalSamplingMethodForCommunity community 100 = SamplingMethod.Stratified) ∧
  (optimalSamplingMethodForTeam team 3 = SamplingMethod.Random) :=
sorry

end optimal_sampling_methods_l815_81557


namespace franklin_valentines_l815_81528

/-- The number of Valentines Mrs. Franklin gave to her students -/
def valentines_given (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Mrs. Franklin gave 42 Valentines to her students -/
theorem franklin_valentines : valentines_given 58 16 = 42 := by
  sorry

end franklin_valentines_l815_81528


namespace train_speed_equation_l815_81597

/-- Represents the equation for two trains traveling the same distance at different speeds -/
theorem train_speed_equation (distance : ℝ) (speed_difference : ℝ) (time_difference : ℝ) 
  (h1 : distance = 236)
  (h2 : speed_difference = 40)
  (h3 : time_difference = 1/4) :
  ∀ x : ℝ, x > speed_difference → 
    (distance / (x - speed_difference) - distance / x = time_difference) :=
by
  sorry

end train_speed_equation_l815_81597


namespace one_solution_less_than_two_l815_81514

def f (x : ℝ) : ℝ := x^8 + 6*x^7 + 14*x^6 + 1429*x^5 - 1279*x^4

theorem one_solution_less_than_two :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end one_solution_less_than_two_l815_81514


namespace mode_of_data_set_l815_81552

def data_set : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data_set :
  mode data_set = 3 := by sorry

end mode_of_data_set_l815_81552


namespace french_fries_cooking_time_l815_81527

/-- Calculates the remaining cooking time in seconds -/
def remaining_cooking_time (recommended_minutes : ℕ) (actual_seconds : ℕ) : ℕ :=
  recommended_minutes * 60 - actual_seconds

/-- Theorem: Given the recommended cooking time of 5 minutes and an actual cooking time of 45 seconds, the remaining cooking time is 255 seconds -/
theorem french_fries_cooking_time : remaining_cooking_time 5 45 = 255 := by
  sorry

end french_fries_cooking_time_l815_81527


namespace modified_cube_edge_count_l815_81505

/-- Represents a modified cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ

/-- Calculates the number of edges in a modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that a cube of side length 4 with unit cubes removed from corners has 48 edges -/
theorem modified_cube_edge_count :
  let cube : ModifiedCube := ⟨4, 1⟩
  edgeCount cube = 48 := by
  sorry

end modified_cube_edge_count_l815_81505


namespace max_triangle_area_in_circle_l815_81531

/-- Given a circle with center C and radius r, and a chord AB that intersects
    the circle at points A and B, forming a triangle ABC. The central angle
    subtended by chord AB is α. -/
theorem max_triangle_area_in_circle (r : ℝ) (α : ℝ) (h : 0 < r) :
  let area := (1/2) * r^2 * Real.sin α
  (∀ θ, 0 ≤ θ ∧ θ ≤ π → area ≥ (1/2) * r^2 * Real.sin θ) ↔ α = π/2 ∧ 
  let chord_length := 2 * r * Real.sin (α/2)
  chord_length = r * Real.sqrt 2 := by
  sorry

end max_triangle_area_in_circle_l815_81531


namespace existence_of_infinite_set_l815_81512

def PositiveInt := { n : ℕ // n > 0 }

def SatisfiesCondition (f : PositiveInt → PositiveInt) : Prop :=
  ∀ x : PositiveInt, (f x).val + (f ⟨x.val + 2, sorry⟩).val ≤ 2 * (f ⟨x.val + 1, sorry⟩).val

theorem existence_of_infinite_set (f : PositiveInt → PositiveInt) (h : SatisfiesCondition f) :
  ∃ M : Set PositiveInt, Set.Infinite M ∧
    ∀ i j k : PositiveInt, i ∈ M → j ∈ M → k ∈ M →
      (i.val - j.val) * (f k).val + (j.val - k.val) * (f i).val + (k.val - i.val) * (f j).val = 0 := by
  sorry

end existence_of_infinite_set_l815_81512


namespace certain_number_is_six_l815_81551

theorem certain_number_is_six : ∃ x : ℝ, 7 * x - 6 = 4 * x + 12 ∧ x = 6 := by
  sorry

end certain_number_is_six_l815_81551


namespace range_of_a_l815_81580

theorem range_of_a (a : ℝ) : 
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |4^x₀ - a * 2^x₀ + 1| ≤ 2^(x₀ + 1)) ↔ 
  a ∈ Set.Icc 0 (9/2) :=
sorry

end range_of_a_l815_81580


namespace other_factor_l815_81572

def n : ℕ := 75

def expression (k : ℕ) : ℕ := k * (2^5) * (6^2) * (7^3)

theorem other_factor : 
  (∃ (m : ℕ), expression n = m * (3^3)) ∧ 
  (∀ (k : ℕ), k < n → ¬∃ (m : ℕ), expression k = m * (3^3)) →
  ∃ (p : ℕ), n = p * 25 ∧ p % 3 = 0 :=
sorry

end other_factor_l815_81572


namespace rhombus_other_diagonal_l815_81556

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ
  area : ℝ
  h_area_formula : area = (d1 * d2) / 2

/-- Theorem: In a rhombus with one diagonal of 12 cm and an area of 90 cm², the other diagonal is 15 cm -/
theorem rhombus_other_diagonal
  (r : Rhombus)
  (h1 : r.d1 = 12)
  (h2 : r.area = 90) :
  r.d2 = 15 := by
  sorry

end rhombus_other_diagonal_l815_81556


namespace batsman_average_after_12_innings_l815_81540

/-- Represents a batsman's performance over multiple innings -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman after a given number of innings -/
def averageScore (b : Batsman) : Rat :=
  b.totalRuns / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after 12 innings is 47 -/
theorem batsman_average_after_12_innings (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.lastInningsScore = 80)
  (h3 : b.averageIncrease = 3)
  : averageScore b = 47 := by
  sorry

end batsman_average_after_12_innings_l815_81540


namespace election_votes_l815_81515

/-- Represents the total number of votes in an election --/
def total_votes : ℕ := 5468

/-- Represents the number of valid votes for candidate B --/
def votes_B : ℕ := 1859

/-- Theorem stating that given the conditions of the election, the total number of votes is 5468 --/
theorem election_votes : 
  (0.8 * total_votes : ℝ) = (votes_B : ℝ) + (votes_B : ℝ) + 0.15 * (total_votes : ℝ) ∧ 
  (votes_B : ℝ) = 1859 ∧
  total_votes = 5468 := by
  sorry

#check election_votes

end election_votes_l815_81515


namespace square_not_always_positive_l815_81576

theorem square_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by sorry

end square_not_always_positive_l815_81576


namespace magnitude_of_AB_l815_81596

def vector_AB : ℝ × ℝ := (3, -4)

theorem magnitude_of_AB : Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2) = 5 := by
  sorry

end magnitude_of_AB_l815_81596


namespace medium_kite_area_l815_81518

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a kite on a 2D grid -/
structure Kite where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint
  v4 : GridPoint

/-- Calculates the area of a kite given its vertices on a grid with 2-inch spacing -/
def kiteArea (k : Kite) : Real :=
  sorry

/-- Theorem: The area of the specified kite is 288 square inches -/
theorem medium_kite_area : 
  let k : Kite := {
    v1 := { x := 0, y := 4 },
    v2 := { x := 4, y := 10 },
    v3 := { x := 12, y := 4 },
    v4 := { x := 4, y := 0 }
  }
  kiteArea k = 288 := by
  sorry

end medium_kite_area_l815_81518


namespace pentagon_perimeter_is_40_l815_81558

/-- An irregular pentagon with given side lengths -/
structure IrregularPentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ

/-- The perimeter of an irregular pentagon -/
def perimeter (p : IrregularPentagon) : ℝ :=
  p.side1 + p.side2 + p.side3 + p.side4 + p.side5

/-- Theorem: The perimeter of the given irregular pentagon is 40 -/
theorem pentagon_perimeter_is_40 :
  let p : IrregularPentagon := {
    side1 := 6,
    side2 := 7,
    side3 := 8,
    side4 := 9,
    side5 := 10
  }
  perimeter p = 40 := by
  sorry

end pentagon_perimeter_is_40_l815_81558


namespace total_profit_is_23200_l815_81586

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  b_investment : ℝ
  b_period : ℝ
  a_investment : ℝ := 3 * b_investment
  a_period : ℝ := 2 * b_period
  c_investment : ℝ := 2 * b_investment
  c_period : ℝ := 0.5 * b_period
  a_rate : ℝ := 0.10
  b_rate : ℝ := 0.15
  c_rate : ℝ := 0.12
  b_profit : ℝ := 4000

/-- Calculates the total profit for the business investment -/
def total_profit (bi : BusinessInvestment) : ℝ :=
  bi.a_investment * bi.a_period * bi.a_rate +
  bi.b_investment * bi.b_period * bi.b_rate +
  bi.c_investment * bi.c_period * bi.c_rate

/-- Theorem stating that the total profit is 23200 -/
theorem total_profit_is_23200 (bi : BusinessInvestment) :
  total_profit bi = 23200 := by
  sorry

end total_profit_is_23200_l815_81586


namespace min_participants_l815_81543

/-- Represents a participant in the race -/
structure Participant where
  name : String
  position : Nat

/-- Represents the race with its participants -/
structure Race where
  participants : List Participant

/-- Checks if the race satisfies the given conditions -/
def satisfiesConditions (race : Race) : Prop :=
  ∃ (andrei dima lenya : Participant),
    andrei ∈ race.participants ∧
    dima ∈ race.participants ∧
    lenya ∈ race.participants ∧
    (∀ p1 p2 : Participant, p1 ∈ race.participants → p2 ∈ race.participants → p1 ≠ p2 → p1.position ≠ p2.position) ∧
    (2 * (andrei.position - 1) = race.participants.length - andrei.position) ∧
    (3 * (dima.position - 1) = race.participants.length - dima.position) ∧
    (4 * (lenya.position - 1) = race.participants.length - lenya.position)

/-- The theorem stating the minimum number of participants -/
theorem min_participants : ∀ race : Race, satisfiesConditions race → race.participants.length ≥ 61 := by
  sorry

end min_participants_l815_81543


namespace simplify_radical_product_l815_81526

theorem simplify_radical_product (q : ℝ) : 
  Real.sqrt (80 * q) * Real.sqrt (45 * q^2) * Real.sqrt (20 * q^3) = 120 * q^3 * Real.sqrt 5 := by
  sorry

end simplify_radical_product_l815_81526


namespace businessmen_beverages_l815_81570

theorem businessmen_beverages (total : ℕ) (coffee tea juice : ℕ) 
  (coffee_tea coffee_juice tea_juice : ℕ) (all_three : ℕ) : 
  total = 30 → 
  coffee = 15 → 
  tea = 12 → 
  juice = 8 → 
  coffee_tea = 6 → 
  coffee_juice = 4 → 
  tea_juice = 2 → 
  all_three = 1 → 
  total - (coffee + tea + juice - coffee_tea - coffee_juice - tea_juice + all_three) = 6 := by
  sorry

end businessmen_beverages_l815_81570


namespace min_c_value_l815_81598

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_sum : a + b + c = 2010)
  (h_unique : ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a| + |x - 2*b| + |x - c|) :
  c ≥ 1014 :=
sorry

end min_c_value_l815_81598


namespace sara_peaches_total_l815_81508

/-- The total number of peaches Sara picked -/
def total_peaches (initial_peaches additional_peaches : Float) : Float :=
  initial_peaches + additional_peaches

/-- Theorem stating that Sara picked 85.0 peaches in total -/
theorem sara_peaches_total :
  let initial_peaches : Float := 61.0
  let additional_peaches : Float := 24.0
  total_peaches initial_peaches additional_peaches = 85.0 := by
  sorry

end sara_peaches_total_l815_81508


namespace other_endpoint_of_line_segment_l815_81517

/-- Given a line segment with midpoint (3, 0) and one endpoint at (7, -4),
    prove that the other endpoint is at (-1, 4). -/
theorem other_endpoint_of_line_segment
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (3, 0))
  (h_endpoint1 : endpoint1 = (7, -4)) :
  ∃ (endpoint2 : ℝ × ℝ),
    endpoint2 = (-1, 4) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end other_endpoint_of_line_segment_l815_81517


namespace sum_of_complex_equality_l815_81546

theorem sum_of_complex_equality (x y : ℝ) :
  (x - 2 : ℂ) + y * Complex.I = -1 + Complex.I →
  x + y = 2 := by sorry

end sum_of_complex_equality_l815_81546


namespace salary_changes_l815_81561

def initial_salary : ℝ := 1800

def may_raise : ℝ := 0.30
def june_cut : ℝ := 0.25
def july_increase : ℝ := 0.10

def final_salary : ℝ := initial_salary * (1 + july_increase)

theorem salary_changes :
  final_salary = 1980 := by sorry

end salary_changes_l815_81561


namespace air_conditioner_sales_l815_81529

theorem air_conditioner_sales (ac_ratio : ℕ) (ref_ratio : ℕ) (difference : ℕ) : 
  ac_ratio = 5 ∧ ref_ratio = 3 ∧ difference = 54 →
  ac_ratio * (difference / (ac_ratio - ref_ratio)) = 135 :=
by sorry

end air_conditioner_sales_l815_81529


namespace line_intersects_y_axis_l815_81591

/-- A line passing through two points (1, 7) and (3, 11) -/
def line (x : ℝ) : ℝ := 2 * x + 5

/-- The y-axis is defined as the set of points with x-coordinate equal to 0 -/
def y_axis (x : ℝ) : Prop := x = 0

theorem line_intersects_y_axis :
  ∃ y : ℝ, y_axis 0 ∧ line 0 = y ∧ y = 5 := by sorry

end line_intersects_y_axis_l815_81591


namespace select_from_m_gives_correct_probability_l815_81511

def set_m : Finset Int := {-6, -5, -4, -3, -2}
def set_t : Finset Int := {-3, -2, -1, 0, 1, 2, 3, 4, 5}

def probability_negative_product : ℚ := 5 / 9

theorem select_from_m_gives_correct_probability :
  (set_m.card : ℚ) * (set_t.filter (λ x => x > 0)).card / set_t.card = probability_negative_product :=
sorry

end select_from_m_gives_correct_probability_l815_81511


namespace sum_of_two_arithmetic_sequences_l815_81593

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem sum_of_two_arithmetic_sequences :
  let seq1 := arithmetic_sequence 1 10 5
  let seq2 := arithmetic_sequence 9 10 5
  List.sum seq1 + List.sum seq2 = 250 := by
  sorry

end sum_of_two_arithmetic_sequences_l815_81593


namespace water_bottles_duration_l815_81581

theorem water_bottles_duration (total_bottles : ℕ) (bottles_per_day : ℕ) (duration : ℕ) : 
  total_bottles = 153 → bottles_per_day = 9 → duration = 17 → 
  total_bottles = bottles_per_day * duration := by
sorry

end water_bottles_duration_l815_81581


namespace roots_product_plus_one_l815_81507

theorem roots_product_plus_one (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  (1+a)*(1+b)*(1+c) = 51 := by
  sorry

end roots_product_plus_one_l815_81507


namespace system_solution_l815_81521

theorem system_solution : 
  ∀ (x y : ℝ), (x^2 + y^3 = x + 1 ∧ x^3 + y^2 = y + 1) ↔ ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end system_solution_l815_81521


namespace overtime_hours_calculation_l815_81560

/-- Calculates the number of overtime hours worked by an employee given their gross pay and pay rates. -/
theorem overtime_hours_calculation (regular_rate overtime_rate gross_pay : ℚ) : 
  regular_rate = 11.25 →
  overtime_rate = 16 →
  gross_pay = 622 →
  ∃ (overtime_hours : ℕ), 
    overtime_hours = 11 ∧ 
    gross_pay = (40 * regular_rate) + (overtime_hours : ℚ) * overtime_rate :=
by sorry

end overtime_hours_calculation_l815_81560


namespace abc_and_fourth_power_sum_l815_81506

theorem abc_and_fourth_power_sum (a b c : ℝ) 
  (sum_1 : a + b + c = 1)
  (sum_2 : a^2 + b^2 + c^2 = 2)
  (sum_3 : a^3 + b^3 + c^3 = 3) :
  a * b * c = 1/6 ∧ a^4 + b^4 + c^4 = 25/6 := by
  sorry

end abc_and_fourth_power_sum_l815_81506


namespace g_1993_of_2_equals_65_53_l815_81519

-- Define the function g
def g (x : ℚ) : ℚ := (2 + x) / (1 - 4 * x^2)

-- Define the recursive function g_n
def g_n : ℕ → ℚ → ℚ
  | 0, x => x
  | 1, x => g (g x)
  | (n+2), x => g (g_n (n+1) x)

-- Theorem statement
theorem g_1993_of_2_equals_65_53 : g_n 1993 2 = 65 / 53 := by
  sorry

end g_1993_of_2_equals_65_53_l815_81519


namespace quadratic_solution_difference_l815_81582

/-- The positive difference between solutions of the quadratic equation x^2 - 5x + m = 13 + (x+5) -/
theorem quadratic_solution_difference (m : ℝ) (h : 27 - m > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁^2 - 5*x₁ + m = 13 + (x₁ + 5)) ∧
  (x₂^2 - 5*x₂ + m = 13 + (x₂ + 5)) ∧
  |x₁ - x₂| = 2 * Real.sqrt (27 - m) :=
by sorry

end quadratic_solution_difference_l815_81582


namespace triangle_abc_exists_l815_81525

/-- Triangle ABC with specific properties -/
structure TriangleABC where
  /-- Side length opposite to angle A -/
  a : ℝ
  /-- Side length opposite to angle B -/
  b : ℝ
  /-- Length of angle bisector from angle C -/
  l_c : ℝ
  /-- Measure of angle A in radians -/
  angle_A : ℝ
  /-- Height to side a -/
  h_a : ℝ
  /-- Perimeter of the triangle -/
  p : ℝ

/-- Theorem stating the existence of a triangle with given properties -/
theorem triangle_abc_exists (a b l_c angle_A h_a p : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : l_c > 0) 
  (h4 : 0 < angle_A ∧ angle_A < π) 
  (h5 : h_a > 0) (h6 : p > 0) :
  ∃ (t : TriangleABC), t.a = a ∧ t.b = b ∧ t.l_c = l_c ∧ 
    t.angle_A = angle_A ∧ t.h_a = h_a ∧ t.p = p :=
sorry

end triangle_abc_exists_l815_81525


namespace opposite_of_five_l815_81588

theorem opposite_of_five : 
  ∃ x : ℤ, (5 + x = 0) ∧ (x = -5) := by
sorry

end opposite_of_five_l815_81588


namespace cube_inequality_l815_81538

theorem cube_inequality (a b : ℝ) (ha : a > 0) (hb : b < 0) : a^3 > b^3 := by
  sorry

end cube_inequality_l815_81538


namespace clock_in_probability_l815_81587

/-- The probability of an employee clocking in on time given a total time window and valid clock-in time -/
theorem clock_in_probability (total_window : ℕ) (valid_time : ℕ) 
  (h1 : total_window = 40) 
  (h2 : valid_time = 15) : 
  (valid_time : ℚ) / total_window = 3/8 := by
  sorry

end clock_in_probability_l815_81587


namespace guppies_count_l815_81566

/-- The number of Goldfish -/
def num_goldfish : ℕ := 2

/-- The amount of food each Goldfish gets (in teaspoons) -/
def goldfish_food : ℚ := 1

/-- The number of Swordtails -/
def num_swordtails : ℕ := 3

/-- The amount of food each Swordtail gets (in teaspoons) -/
def swordtail_food : ℚ := 2

/-- The amount of food each Guppy gets (in teaspoons) -/
def guppy_food : ℚ := 1/2

/-- The total amount of food given to all fish (in teaspoons) -/
def total_food : ℚ := 12

/-- The number of Guppies Layla has -/
def num_guppies : ℕ := 8

theorem guppies_count :
  (num_goldfish : ℚ) * goldfish_food +
  (num_swordtails : ℚ) * swordtail_food +
  (num_guppies : ℚ) * guppy_food = total_food :=
by sorry

end guppies_count_l815_81566


namespace find_w_l815_81590

theorem find_w : ∃ w : ℝ, ((2^5 : ℝ) * (9^2)) / ((8^2) * w) = 0.16666666666666666 ∧ w = 243 := by
  sorry

end find_w_l815_81590


namespace tracy_candies_problem_l815_81542

theorem tracy_candies_problem (x : ℕ) : 
  x > 0 ∧ 
  x % 4 = 0 ∧ 
  (x * 3 / 4 * 2 / 3 - 20 - 3 = 7) → 
  x = 60 := by
sorry

end tracy_candies_problem_l815_81542


namespace van_distance_theorem_l815_81554

theorem van_distance_theorem (initial_time : ℝ) (speed : ℝ) :
  initial_time = 6 →
  speed = 30 →
  (3 / 2 : ℝ) * initial_time * speed = 270 :=
by
  sorry

end van_distance_theorem_l815_81554


namespace rose_cost_l815_81571

/-- Proves that the cost of each rose is $5 given the wedding decoration costs --/
theorem rose_cost (num_tables : ℕ) (tablecloth_cost place_setting_cost lily_cost total_cost : ℚ)
  (place_settings_per_table roses_per_table lilies_per_table : ℕ) :
  num_tables = 20 →
  tablecloth_cost = 25 →
  place_setting_cost = 10 →
  place_settings_per_table = 4 →
  roses_per_table = 10 →
  lilies_per_table = 15 →
  lily_cost = 4 →
  total_cost = 3500 →
  (total_cost - 
   (num_tables * tablecloth_cost + 
    num_tables * place_settings_per_table * place_setting_cost + 
    num_tables * lilies_per_table * lily_cost)) / (num_tables * roses_per_table) = 5 := by
  sorry


end rose_cost_l815_81571


namespace ryan_fundraising_goal_l815_81565

/-- The total amount Ryan wants to raise for his business -/
def total_amount (avg_funding : ℕ) (num_people : ℕ) (existing_funds : ℕ) : ℕ :=
  avg_funding * num_people + existing_funds

/-- Proof that Ryan wants to raise $1000 for his business -/
theorem ryan_fundraising_goal :
  let avg_funding : ℕ := 10
  let num_people : ℕ := 80
  let existing_funds : ℕ := 200
  total_amount avg_funding num_people existing_funds = 1000 := by
  sorry

end ryan_fundraising_goal_l815_81565


namespace intersection_of_lines_l815_81513

theorem intersection_of_lines (p q : ℝ) : 
  (∃ x y : ℝ, y = p * x + 4 ∧ p * y = q * x - 7 ∧ x = 3 ∧ y = 1) → q = 2 := by
  sorry

end intersection_of_lines_l815_81513


namespace derivative_f_at_zero_l815_81553

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 3^(x^2 * Real.sin (2/x)) - 1 + 2*x else 0

theorem derivative_f_at_zero : 
  deriv f 0 = -2 := by sorry

end derivative_f_at_zero_l815_81553


namespace sculpture_height_l815_81594

/-- Converts feet to inches -/
def feet_to_inches (feet : ℝ) : ℝ := feet * 12

theorem sculpture_height :
  let base_height : ℝ := 2
  let total_height_feet : ℝ := 3
  let total_height_inches : ℝ := feet_to_inches total_height_feet
  let sculpture_height : ℝ := total_height_inches - base_height
  sculpture_height = 34 := by
  sorry

end sculpture_height_l815_81594


namespace parabola_sum_l815_81547

theorem parabola_sum (a b c : ℝ) : 
  (∀ y : ℝ, 10 = a * (-6)^2 + b * (-6) + c) →
  (∀ y : ℝ, 8 = a * (-4)^2 + b * (-4) + c) →
  a + b + c = -39 := by
sorry

end parabola_sum_l815_81547


namespace unique_valid_code_l815_81523

def is_valid_code (n : ℕ) : Prop :=
  -- The code is an eight-digit number
  100000000 > n ∧ n ≥ 10000000 ∧
  -- The code is a multiple of both 3 and 25
  n % 3 = 0 ∧ n % 25 = 0 ∧
  -- The code is between 20,000,000 and 30,000,000
  30000000 > n ∧ n > 20000000 ∧
  -- The digits in the millions and hundred thousand places are the same
  (n / 1000000) % 10 = (n / 100000) % 10 ∧
  -- The digit in the hundreds place is 2 less than the digit in the ten thousands place
  (n / 100) % 10 + 2 = (n / 10000) % 10 ∧
  -- The three-digit number formed by the digits in the hundred thousands, ten thousands, and thousands places,
  -- divided by the two-digit number formed by the digits in the ten millions and millions places, gives a quotient of 25
  ((n / 100000) % 1000) / ((n / 1000000) % 100) = 25

theorem unique_valid_code : ∃! n : ℕ, is_valid_code n ∧ n = 26650350 :=
  sorry

#check unique_valid_code

end unique_valid_code_l815_81523
