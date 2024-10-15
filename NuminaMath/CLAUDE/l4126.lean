import Mathlib

namespace NUMINAMATH_CALUDE_max_triangle_area_in_circle_l4126_412683

/-- Given a circle with center C and radius r, and a chord AB that intersects
    the circle at points A and B, forming a triangle ABC. The central angle
    subtended by chord AB is α. -/
theorem max_triangle_area_in_circle (r : ℝ) (α : ℝ) (h : 0 < r) :
  let area := (1/2) * r^2 * Real.sin α
  (∀ θ, 0 ≤ θ ∧ θ ≤ π → area ≥ (1/2) * r^2 * Real.sin θ) ↔ α = π/2 ∧ 
  let chord_length := 2 * r * Real.sin (α/2)
  chord_length = r * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_in_circle_l4126_412683


namespace NUMINAMATH_CALUDE_triangle_side_length_l4126_412605

theorem triangle_side_length (a b c : ℝ) (area : ℝ) : 
  a = 1 → b = Real.sqrt 7 → area = Real.sqrt 3 / 2 → 
  c = 2 ∨ c = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4126_412605


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_100_l4126_412608

theorem greatest_common_divisor_under_100 : ∃ (n : ℕ), n = 90 ∧ 
  n ∣ 540 ∧ n < 100 ∧ n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 ∧ m < 100 ∧ m ∣ 180 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_100_l4126_412608


namespace NUMINAMATH_CALUDE_problem_2012_l4126_412611

theorem problem_2012 (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (eq1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2011)
  (eq2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2011) :
  (c*d)^2012 - (a*b)^2012 = 2011 := by
sorry

end NUMINAMATH_CALUDE_problem_2012_l4126_412611


namespace NUMINAMATH_CALUDE_opposite_of_five_l4126_412649

theorem opposite_of_five : 
  ∃ x : ℤ, (5 + x = 0) ∧ (x = -5) := by
sorry

end NUMINAMATH_CALUDE_opposite_of_five_l4126_412649


namespace NUMINAMATH_CALUDE_train_speed_equation_l4126_412690

/-- Represents the equation for two trains traveling the same distance at different speeds -/
theorem train_speed_equation (distance : ℝ) (speed_difference : ℝ) (time_difference : ℝ) 
  (h1 : distance = 236)
  (h2 : speed_difference = 40)
  (h3 : time_difference = 1/4) :
  ∀ x : ℝ, x > speed_difference → 
    (distance / (x - speed_difference) - distance / x = time_difference) :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_equation_l4126_412690


namespace NUMINAMATH_CALUDE_larger_number_proof_l4126_412617

theorem larger_number_proof (x y : ℝ) 
  (h1 : x - y = 1860)
  (h2 : 0.075 * x = 0.125 * y) :
  x = 4650 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4126_412617


namespace NUMINAMATH_CALUDE_round_trip_completion_percentage_l4126_412678

/-- Calculates the completion percentage of a round-trip given delays on the outbound journey -/
theorem round_trip_completion_percentage 
  (T : ℝ) -- Normal one-way travel time
  (h1 : T > 0) -- Assumption that travel time is positive
  (traffic_delay : ℝ := 0.15) -- 15% increase due to traffic
  (construction_delay : ℝ := 0.10) -- 10% increase due to construction
  (return_completion : ℝ := 0.20) -- 20% of return journey completed
  : (T * (1 + traffic_delay + construction_delay) + return_completion * T) / (2 * T) = 0.725 := by
sorry

end NUMINAMATH_CALUDE_round_trip_completion_percentage_l4126_412678


namespace NUMINAMATH_CALUDE_point_60_coordinates_l4126_412687

/-- Defines the x-coordinate of the nth point in the sequence -/
def x (n : ℕ) : ℕ := sorry

/-- Defines the y-coordinate of the nth point in the sequence -/
def y (n : ℕ) : ℕ := sorry

/-- The sum of x and y coordinates for the nth point -/
def sum (n : ℕ) : ℕ := x n + y n

theorem point_60_coordinates :
  x 60 = 5 ∧ y 60 = 7 := by sorry

end NUMINAMATH_CALUDE_point_60_coordinates_l4126_412687


namespace NUMINAMATH_CALUDE_wire_cutting_l4126_412663

theorem wire_cutting (total_length piece_length : ℝ) 
  (h1 : total_length = 27.9)
  (h2 : piece_length = 3.1) : 
  ⌊total_length / piece_length⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l4126_412663


namespace NUMINAMATH_CALUDE_expression_evaluation_l4126_412677

theorem expression_evaluation (a b : ℤ) (h1 : a = 2) (h2 : b = -1) :
  (2 * a^2 * b - 4 * a * b^2) - 2 * (a * b^2 + a^2 * b) = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4126_412677


namespace NUMINAMATH_CALUDE_expected_adjacent_pairs_l4126_412602

/-- The expected number of adjacent boy-girl pairs in a random permutation of boys and girls -/
theorem expected_adjacent_pairs (n_boys n_girls : ℕ) : 
  let n_total := n_boys + n_girls
  let n_pairs := n_total - 1
  let p_boy := n_boys / n_total
  let p_girl := n_girls / n_total
  let p_adjacent := p_boy * p_girl + p_girl * p_boy
  n_boys = 10 → n_girls = 15 → n_pairs * p_adjacent = 12 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_pairs_l4126_412602


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l4126_412639

/-- The positive difference between solutions of the quadratic equation x^2 - 5x + m = 13 + (x+5) -/
theorem quadratic_solution_difference (m : ℝ) (h : 27 - m > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁^2 - 5*x₁ + m = 13 + (x₁ + 5)) ∧
  (x₂^2 - 5*x₂ + m = 13 + (x₂ + 5)) ∧
  |x₁ - x₂| = 2 * Real.sqrt (27 - m) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l4126_412639


namespace NUMINAMATH_CALUDE_water_donation_difference_l4126_412666

/-- The number of food items donated by five food companies to a local food bank. -/
def food_bank_donation : ℕ := 375

/-- The number of dressed chickens donated by Foster Farms. -/
def foster_farms_chickens : ℕ := 45

/-- The number of bottles of water donated by American Summits. -/
def american_summits_water : ℕ := 2 * foster_farms_chickens

/-- The number of dressed chickens donated by Hormel. -/
def hormel_chickens : ℕ := 3 * foster_farms_chickens

/-- The number of dressed chickens donated by Boudin Butchers. -/
def boudin_butchers_chickens : ℕ := hormel_chickens / 3

/-- The number of bottles of water donated by Del Monte Foods. -/
def del_monte_water : ℕ := food_bank_donation - (foster_farms_chickens + american_summits_water + hormel_chickens + boudin_butchers_chickens)

/-- Theorem stating the difference in water bottles donated between American Summits and Del Monte Foods. -/
theorem water_donation_difference :
  american_summits_water - del_monte_water = 30 := by
  sorry

end NUMINAMATH_CALUDE_water_donation_difference_l4126_412666


namespace NUMINAMATH_CALUDE_cube_edge_length_l4126_412693

theorem cube_edge_length : ∃ s : ℝ, s > 0 ∧ s^3 = 6 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l4126_412693


namespace NUMINAMATH_CALUDE_power_function_value_l4126_412627

/-- A power function is a function of the form f(x) = x^α for some real α -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

theorem power_function_value (f : ℝ → ℝ) (h1 : IsPowerFunction f) (h2 : f 2 / f 4 = 1 / 2) :
  f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l4126_412627


namespace NUMINAMATH_CALUDE_arithmetic_progression_formula_geometric_progression_formula_l4126_412618

-- Arithmetic Progression
def arithmeticProgression (u₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := u₁ + (n - 1 : ℝ) * d

-- Geometric Progression
def geometricProgression (u₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := u₁ * q ^ (n - 1)

theorem arithmetic_progression_formula (u₁ d : ℝ) (n : ℕ) :
  ∀ k : ℕ, k ≤ n → arithmeticProgression u₁ d k = u₁ + (k - 1 : ℝ) * d :=
by sorry

theorem geometric_progression_formula (u₁ q : ℝ) (n : ℕ) :
  ∀ k : ℕ, k ≤ n → geometricProgression u₁ q k = u₁ * q ^ (k - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_formula_geometric_progression_formula_l4126_412618


namespace NUMINAMATH_CALUDE_teacups_left_result_l4126_412620

/-- Calculates the number of teacups left after arranging --/
def teacups_left (total_boxes : ℕ) (pan_boxes : ℕ) (rows_per_box : ℕ) (cups_per_row : ℕ) (broken_per_box : ℕ) : ℕ :=
  let remaining_boxes := total_boxes - pan_boxes
  let decoration_boxes := remaining_boxes / 2
  let teacup_boxes := remaining_boxes - decoration_boxes
  let cups_per_box := rows_per_box * cups_per_row
  let total_cups := teacup_boxes * cups_per_box
  let broken_cups := teacup_boxes * broken_per_box
  total_cups - broken_cups

/-- Theorem stating the number of teacups left after arranging --/
theorem teacups_left_result : teacups_left 26 6 5 4 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_teacups_left_result_l4126_412620


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l4126_412681

/-- The ratio of volumes of cylinders formed by rolling a 6x9 rectangle -/
theorem cylinder_volume_ratio : 
  let length : ℝ := 6
  let width : ℝ := 9
  let volume1 := π * (length / (2 * π))^2 * width
  let volume2 := π * (width / (2 * π))^2 * length
  volume2 / volume1 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l4126_412681


namespace NUMINAMATH_CALUDE_max_trig_product_bound_max_trig_product_achievable_l4126_412665

theorem max_trig_product_bound (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 9 / 2 :=
sorry

theorem max_trig_product_achievable :
  ∃ x y z : ℝ,
    (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
    (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) = 9 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_trig_product_bound_max_trig_product_achievable_l4126_412665


namespace NUMINAMATH_CALUDE_num_ways_to_place_pawns_l4126_412628

/-- Represents a chess board configuration -/
def ChessBoard := Fin 5 → Fin 5

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- Checks if a chess board configuration is valid (no more than one pawn per row and column) -/
def is_valid_configuration (board : ChessBoard) : Prop :=
  (∀ i j : Fin 5, i ≠ j → board i ≠ board j) ∧
  (∀ i : Fin 5, ∃ j : Fin 5, board j = i)

/-- The number of valid chess board configurations -/
def num_valid_configurations : ℕ := factorial 5

/-- The main theorem stating the number of ways to place five distinct pawns -/
theorem num_ways_to_place_pawns :
  (num_valid_configurations * factorial 5 : ℕ) = 14400 :=
sorry

end NUMINAMATH_CALUDE_num_ways_to_place_pawns_l4126_412628


namespace NUMINAMATH_CALUDE_batsman_average_after_12_innings_l4126_412675

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

end NUMINAMATH_CALUDE_batsman_average_after_12_innings_l4126_412675


namespace NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l4126_412612

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}

theorem complement_of_A_relative_to_U :
  {x ∈ U | x ∉ A} = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l4126_412612


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l4126_412606

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 7 y = 14 ∧ y = 42 :=
by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l4126_412606


namespace NUMINAMATH_CALUDE_expansion_properties_l4126_412656

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

end NUMINAMATH_CALUDE_expansion_properties_l4126_412656


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l4126_412658

theorem angle_in_second_quadrant (θ : Real) : 
  (Real.tan θ * Real.sin θ < 0) → 
  (Real.tan θ * Real.cos θ > 0) → 
  (0 < θ) ∧ (θ < Real.pi) := by
sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l4126_412658


namespace NUMINAMATH_CALUDE_M_equals_N_l4126_412625

def M : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def N : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l4126_412625


namespace NUMINAMATH_CALUDE_other_factor_l4126_412641

def n : ℕ := 75

def expression (k : ℕ) : ℕ := k * (2^5) * (6^2) * (7^3)

theorem other_factor : 
  (∃ (m : ℕ), expression n = m * (3^3)) ∧ 
  (∀ (k : ℕ), k < n → ¬∃ (m : ℕ), expression k = m * (3^3)) →
  ∃ (p : ℕ), n = p * 25 ∧ p % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_other_factor_l4126_412641


namespace NUMINAMATH_CALUDE_magnitude_of_AB_l4126_412689

def vector_AB : ℝ × ℝ := (3, -4)

theorem magnitude_of_AB : Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_AB_l4126_412689


namespace NUMINAMATH_CALUDE_businessmen_beverages_l4126_412633

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

end NUMINAMATH_CALUDE_businessmen_beverages_l4126_412633


namespace NUMINAMATH_CALUDE_square_not_always_positive_l4126_412670

theorem square_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l4126_412670


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4126_412624

open Set

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x ≤ 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4126_412624


namespace NUMINAMATH_CALUDE_election_votes_l4126_412686

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

end NUMINAMATH_CALUDE_election_votes_l4126_412686


namespace NUMINAMATH_CALUDE_remainder_theorem_l4126_412630

-- Define the polynomial
def f (x : ℝ) : ℝ := x^5 - 3*x^3 + x + 5

-- Define the divisor
def g (x : ℝ) : ℝ := (x - 3)^2

-- Theorem statement
theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, f x = g x * q x + 65 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4126_412630


namespace NUMINAMATH_CALUDE_medium_kite_area_l4126_412661

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

end NUMINAMATH_CALUDE_medium_kite_area_l4126_412661


namespace NUMINAMATH_CALUDE_parabola_sum_l4126_412655

theorem parabola_sum (a b c : ℝ) : 
  (∀ y : ℝ, 10 = a * (-6)^2 + b * (-6) + c) →
  (∀ y : ℝ, 8 = a * (-4)^2 + b * (-4) + c) →
  a + b + c = -39 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l4126_412655


namespace NUMINAMATH_CALUDE_sum_of_complex_equality_l4126_412654

theorem sum_of_complex_equality (x y : ℝ) :
  (x - 2 : ℂ) + y * Complex.I = -1 + Complex.I →
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_complex_equality_l4126_412654


namespace NUMINAMATH_CALUDE_equal_sum_black_white_cells_l4126_412696

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

end NUMINAMATH_CALUDE_equal_sum_black_white_cells_l4126_412696


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l4126_412660

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

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l4126_412660


namespace NUMINAMATH_CALUDE_rose_cost_l4126_412640

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


end NUMINAMATH_CALUDE_rose_cost_l4126_412640


namespace NUMINAMATH_CALUDE_f_positive_iff_l4126_412671

-- Define the function
def f (x : ℝ) := x^2 + x - 12

-- State the theorem
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x < -4 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_l4126_412671


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l4126_412621

/-- A rectangular solid with prime edge lengths and volume 273 has surface area 302 -/
theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a → Prime b → Prime c → a * b * c = 273 → 2 * (a * b + b * c + c * a) = 302 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l4126_412621


namespace NUMINAMATH_CALUDE_existence_of_infinite_set_l4126_412651

def PositiveInt := { n : ℕ // n > 0 }

def SatisfiesCondition (f : PositiveInt → PositiveInt) : Prop :=
  ∀ x : PositiveInt, (f x).val + (f ⟨x.val + 2, sorry⟩).val ≤ 2 * (f ⟨x.val + 1, sorry⟩).val

theorem existence_of_infinite_set (f : PositiveInt → PositiveInt) (h : SatisfiesCondition f) :
  ∃ M : Set PositiveInt, Set.Infinite M ∧
    ∀ i j k : PositiveInt, i ∈ M → j ∈ M → k ∈ M →
      (i.val - j.val) * (f k).val + (j.val - k.val) * (f i).val + (k.val - i.val) * (f j).val = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_infinite_set_l4126_412651


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l4126_412623

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/3 + 1/2, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/9, 1/3 + 1/10]
  (∀ x ∈ sums, x ≤ 1/3 + 1/2) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l4126_412623


namespace NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l4126_412629

/-- Represents the position of a particle -/
structure Position where
  x : ℕ
  y : ℕ

/-- Calculates the time taken to complete n squares -/
def time_for_squares (n : ℕ) : ℕ :=
  n^2 + 5*n

/-- Determines the position of the particle after a given time -/
def particle_position (time : ℕ) : Position :=
  sorry

/-- The theorem to be proved -/
theorem particle_position_after_2023_minutes :
  particle_position 2023 = Position.mk 43 43 := by
  sorry

end NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l4126_412629


namespace NUMINAMATH_CALUDE_cube_inequality_l4126_412657

theorem cube_inequality (a b : ℝ) (ha : a > 0) (hb : b < 0) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l4126_412657


namespace NUMINAMATH_CALUDE_equation_solution_l4126_412650

theorem equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x * (4 : ℝ)^x * (16 : ℝ)^(x + 1) = (1024 : ℝ)^2 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4126_412650


namespace NUMINAMATH_CALUDE_EDTA_Ca_complex_weight_l4126_412643

-- Define the molecular weight of EDTA
def EDTA_weight : ℝ := 292.248

-- Define the atomic weight of calcium
def Ca_weight : ℝ := 40.08

-- Define the complex ratio
def complex_ratio : ℕ := 1

-- Theorem statement
theorem EDTA_Ca_complex_weight :
  complex_ratio * (EDTA_weight + Ca_weight) = 332.328 := by sorry

end NUMINAMATH_CALUDE_EDTA_Ca_complex_weight_l4126_412643


namespace NUMINAMATH_CALUDE_percent_students_in_school_l4126_412622

/-- Given that 40% of students are learning from home and the remaining students are equally divided
    into two groups with only one group attending school on any day, prove that the percent of
    students present in school is 30%. -/
theorem percent_students_in_school :
  let total_percent : ℚ := 100
  let home_percent : ℚ := 40
  let remaining_percent : ℚ := total_percent - home_percent
  let in_school_percent : ℚ := remaining_percent / 2
  in_school_percent = 30 := by sorry

end NUMINAMATH_CALUDE_percent_students_in_school_l4126_412622


namespace NUMINAMATH_CALUDE_average_age_decrease_l4126_412680

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

end NUMINAMATH_CALUDE_average_age_decrease_l4126_412680


namespace NUMINAMATH_CALUDE_x_value_l4126_412615

theorem x_value (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l4126_412615


namespace NUMINAMATH_CALUDE_train_average_speed_l4126_412636

def train_distance_1 : ℝ := 240
def train_time_1 : ℝ := 3
def train_distance_2 : ℝ := 450
def train_time_2 : ℝ := 5

theorem train_average_speed :
  (train_distance_1 + train_distance_2) / (train_time_1 + train_time_2) = 86.25 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l4126_412636


namespace NUMINAMATH_CALUDE_joe_haircut_time_l4126_412667

/-- Represents the time taken for different types of haircuts and the number of each type performed --/
structure HaircutData where
  womenTime : ℕ  -- Time to cut a woman's hair
  menTime : ℕ    -- Time to cut a man's hair
  kidsTime : ℕ   -- Time to cut a kid's hair
  womenCount : ℕ -- Number of women's haircuts
  menCount : ℕ   -- Number of men's haircuts
  kidsCount : ℕ  -- Number of kids' haircuts

/-- Calculates the total time spent cutting hair --/
def totalHaircutTime (data : HaircutData) : ℕ :=
  data.womenTime * data.womenCount +
  data.menTime * data.menCount +
  data.kidsTime * data.kidsCount

/-- Theorem stating that Joe's total haircut time is 255 minutes --/
theorem joe_haircut_time :
  let data : HaircutData := {
    womenTime := 50,
    menTime := 15,
    kidsTime := 25,
    womenCount := 3,
    menCount := 2,
    kidsCount := 3
  }
  totalHaircutTime data = 255 := by
  sorry

end NUMINAMATH_CALUDE_joe_haircut_time_l4126_412667


namespace NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l4126_412601

/-- Calculates the total amount of ethanol in a fuel tank -/
def total_ethanol (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) : ℝ :=
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let ethanol_a := fuel_a_volume * fuel_a_ethanol_percent
  let ethanol_b := fuel_b_volume * fuel_b_ethanol_percent
  ethanol_a + ethanol_b

/-- The theorem states that the total amount of ethanol in the specified fuel mixture is 30 gallons -/
theorem ethanol_in_fuel_tank :
  total_ethanol 208 82 0.12 0.16 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l4126_412601


namespace NUMINAMATH_CALUDE_second_rectangle_width_l4126_412684

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

end NUMINAMATH_CALUDE_second_rectangle_width_l4126_412684


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l4126_412659

theorem sum_of_quadratic_roots (x₁ x₂ : ℝ) : 
  (-48 : ℝ) * x₁^2 + 110 * x₁ + 165 = 0 ∧
  (-48 : ℝ) * x₂^2 + 110 * x₂ + 165 = 0 →
  x₁ + x₂ = 55 / 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l4126_412659


namespace NUMINAMATH_CALUDE_dust_storm_coverage_l4126_412648

/-- Given a prairie and a dust storm, calculate the area covered by the storm -/
theorem dust_storm_coverage (total_prairie_area untouched_area : ℕ) 
  (h1 : total_prairie_area = 65057)
  (h2 : untouched_area = 522) :
  total_prairie_area - untouched_area = 64535 := by
  sorry

end NUMINAMATH_CALUDE_dust_storm_coverage_l4126_412648


namespace NUMINAMATH_CALUDE_hannahs_dogs_food_l4126_412634

/-- The amount of food eaten by Hannah's first dog -/
def first_dog_food : ℝ := 1.5

/-- The amount of food eaten by Hannah's second dog -/
def second_dog_food : ℝ := 2 * first_dog_food

/-- The amount of food eaten by Hannah's third dog -/
def third_dog_food : ℝ := second_dog_food + 2.5

/-- The total amount of food prepared by Hannah for her three dogs -/
def total_food : ℝ := 10

theorem hannahs_dogs_food :
  first_dog_food + second_dog_food + third_dog_food = total_food :=
by sorry

end NUMINAMATH_CALUDE_hannahs_dogs_food_l4126_412634


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l4126_412613

-- Define a regular pentagon
def RegularPentagon : Type := Unit

-- Define the function to calculate the sum of interior angles of a polygon
def sumInteriorAngles (n : ℕ) : ℝ := (n - 2) * 180

-- Theorem: The sum of interior angles of a regular pentagon is 540°
theorem sum_interior_angles_pentagon (p : RegularPentagon) :
  sumInteriorAngles 5 = 540 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l4126_412613


namespace NUMINAMATH_CALUDE_simplify_radical_product_l4126_412699

theorem simplify_radical_product (q : ℝ) : 
  Real.sqrt (80 * q) * Real.sqrt (45 * q^2) * Real.sqrt (20 * q^3) = 120 * q^3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l4126_412699


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l4126_412631

/-- Given a quadratic equation 3x^2 + 4x + 5 = 0 with roots r and s,
    if we construct a new quadratic equation x^2 + px + q = 0 with roots 2r and 2s,
    then p = 56/9 -/
theorem quadratic_root_relation (r s : ℝ) (p q : ℝ) : 
  (3 * r^2 + 4 * r + 5 = 0) →
  (3 * s^2 + 4 * s + 5 = 0) →
  ((2 * r)^2 + p * (2 * r) + q = 0) →
  ((2 * s)^2 + p * (2 * s) + q = 0) →
  p = 56 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l4126_412631


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l4126_412626

/-- Proves that the total number of students sampled is 135 given the conditions of the stratified sampling problem -/
theorem stratified_sampling_result (grade10 : ℕ) (grade11 : ℕ) (grade12 : ℕ) (sampled10 : ℕ) 
  (h1 : grade10 = 2000)
  (h2 : grade11 = 1500)
  (h3 : grade12 = 1000)
  (h4 : sampled10 = 60) :
  (grade10 + grade11 + grade12) * sampled10 / grade10 = 135 := by
  sorry

#check stratified_sampling_result

end NUMINAMATH_CALUDE_stratified_sampling_result_l4126_412626


namespace NUMINAMATH_CALUDE_tracy_candies_problem_l4126_412645

theorem tracy_candies_problem (x : ℕ) : 
  x > 0 ∧ 
  x % 4 = 0 ∧ 
  (x * 3 / 4 * 2 / 3 - 20 - 3 = 7) → 
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_tracy_candies_problem_l4126_412645


namespace NUMINAMATH_CALUDE_g_1993_of_2_equals_65_53_l4126_412662

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

end NUMINAMATH_CALUDE_g_1993_of_2_equals_65_53_l4126_412662


namespace NUMINAMATH_CALUDE_race_distance_proof_l4126_412672

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

end NUMINAMATH_CALUDE_race_distance_proof_l4126_412672


namespace NUMINAMATH_CALUDE_wade_team_score_l4126_412685

/-- Calculates the total points scored by a basketball team after a given number of games -/
def team_total_points (wade_avg : ℕ) (teammates_avg : ℕ) (games : ℕ) : ℕ :=
  (wade_avg + teammates_avg) * games

/-- Proves that Wade's team scores 300 points in 5 games -/
theorem wade_team_score : team_total_points 20 40 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_wade_team_score_l4126_412685


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l4126_412691

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l4126_412691


namespace NUMINAMATH_CALUDE_triangle_abc_exists_l4126_412698

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

end NUMINAMATH_CALUDE_triangle_abc_exists_l4126_412698


namespace NUMINAMATH_CALUDE_system_solution_l4126_412688

theorem system_solution : 
  ∀ (x y : ℝ), (x^2 + y^3 = x + 1 ∧ x^3 + y^2 = y + 1) ↔ ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4126_412688


namespace NUMINAMATH_CALUDE_three_eighths_count_l4126_412668

theorem three_eighths_count : (8 + 5/3 - 3) / (3/8) = 160/9 := by sorry

end NUMINAMATH_CALUDE_three_eighths_count_l4126_412668


namespace NUMINAMATH_CALUDE_water_bottles_duration_l4126_412638

theorem water_bottles_duration (total_bottles : ℕ) (bottles_per_day : ℕ) (duration : ℕ) : 
  total_bottles = 153 → bottles_per_day = 9 → duration = 17 → 
  total_bottles = bottles_per_day * duration := by
sorry

end NUMINAMATH_CALUDE_water_bottles_duration_l4126_412638


namespace NUMINAMATH_CALUDE_intersection_of_lines_l4126_412652

theorem intersection_of_lines (p q : ℝ) : 
  (∃ x y : ℝ, y = p * x + 4 ∧ p * y = q * x - 7 ∧ x = 3 ∧ y = 1) → q = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l4126_412652


namespace NUMINAMATH_CALUDE_adlai_total_animal_legs_l4126_412609

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of dogs Adlai has -/
def adlai_dogs : ℕ := 2

/-- The number of chickens Adlai has -/
def adlai_chickens : ℕ := 1

/-- Theorem stating the total number of animal legs Adlai has -/
theorem adlai_total_animal_legs : 
  adlai_dogs * dog_legs + adlai_chickens * chicken_legs = 10 := by
  sorry

end NUMINAMATH_CALUDE_adlai_total_animal_legs_l4126_412609


namespace NUMINAMATH_CALUDE_order_of_abc_l4126_412647

theorem order_of_abc (a b c : ℝ) 
  (h1 : 1.001 * Real.exp a = Real.exp 1.001)
  (h2 : b - Real.sqrt (1000 / 1001) = 1.001 - Real.sqrt 1.001)
  (h3 : c = 1.001) : 
  b < a ∧ a < c :=
sorry

end NUMINAMATH_CALUDE_order_of_abc_l4126_412647


namespace NUMINAMATH_CALUDE_range_of_a_l4126_412637

theorem range_of_a (a : ℝ) : 
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |4^x₀ - a * 2^x₀ + 1| ≤ 2^(x₀ + 1)) ↔ 
  a ∈ Set.Icc 0 (9/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4126_412637


namespace NUMINAMATH_CALUDE_female_students_count_l4126_412600

/-- Represents a school with male and female students. -/
structure School where
  total_students : ℕ
  sample_size : ℕ
  sample_boys_girls_diff : ℕ

/-- Calculates the number of female students in the school based on the given parameters. -/
def female_students (s : School) : ℕ :=
  let sampled_girls := (s.sample_size - s.sample_boys_girls_diff) / 2
  let ratio := s.total_students / s.sample_size
  sampled_girls * ratio

/-- Theorem stating that for the given school parameters, the number of female students is 760. -/
theorem female_students_count (s : School) 
  (h1 : s.total_students = 1600)
  (h2 : s.sample_size = 200)
  (h3 : s.sample_boys_girls_diff = 10) :
  female_students s = 760 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l4126_412600


namespace NUMINAMATH_CALUDE_unique_valid_code_l4126_412644

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

end NUMINAMATH_CALUDE_unique_valid_code_l4126_412644


namespace NUMINAMATH_CALUDE_ellipse_properties_l4126_412646

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

end NUMINAMATH_CALUDE_ellipse_properties_l4126_412646


namespace NUMINAMATH_CALUDE_pizza_toppings_l4126_412676

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 20)
  (h2 : pepperoni_slices = 12)
  (h3 : mushroom_slices = 14)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l4126_412676


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l4126_412603

theorem sum_remainder_mod_nine : 
  (9151 + 9152 + 9153 + 9154 + 9155 + 9156 + 9157) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l4126_412603


namespace NUMINAMATH_CALUDE_part1_part2_l4126_412632

-- Define the polynomials A and B
def A (x : ℝ) : ℝ := x^2 - x
def B (m : ℝ) (x : ℝ) : ℝ := m * x + 1

-- Part 1: Prove that when ■ = 2, 2A - B = 2x^2 - 4x - 1
theorem part1 (x : ℝ) : 2 * A x - B 2 x = 2 * x^2 - 4 * x - 1 := by
  sorry

-- Part 2: Prove that when A - B does not contain x terms, ■ = -1
theorem part2 (x : ℝ) : (∀ m : ℝ, A x - B m x = (-1 : ℝ)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_l4126_412632


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l4126_412682

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

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l4126_412682


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l4126_412642

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  2 * x + 1 / (x + 3) ≥ 2 * Real.sqrt 2 - 6 :=
by sorry

theorem min_value_achievable :
  ∃ x > -3, 2 * x + 1 / (x + 3) = 2 * Real.sqrt 2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l4126_412642


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4126_412669

/-- The quadratic equation (a-2)x^2 + x + a^2 - 4 = 0 has 0 as one of its roots -/
def has_zero_root (a : ℝ) : Prop :=
  ∃ x : ℝ, (a - 2) * x^2 + x + a^2 - 4 = 0 ∧ x = 0

/-- The value of a that satisfies the condition -/
def solution : ℝ := -2

theorem quadratic_equation_solution :
  ∀ a : ℝ, has_zero_root a → a = solution :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4126_412669


namespace NUMINAMATH_CALUDE_range_of_complex_function_l4126_412674

theorem range_of_complex_function (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (a b : ℝ), a = Real.sqrt 2 - 1 ∧ b = Real.sqrt 2 + 1 ∧
  ∀ θ : ℝ, a ≤ Complex.abs (z^2 + Complex.I * z^2 + 1) ∧
           Complex.abs (z^2 + Complex.I * z^2 + 1) ≤ b :=
by sorry

end NUMINAMATH_CALUDE_range_of_complex_function_l4126_412674


namespace NUMINAMATH_CALUDE_rohan_entertainment_expenses_l4126_412697

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

end NUMINAMATH_CALUDE_rohan_entertainment_expenses_l4126_412697


namespace NUMINAMATH_CALUDE_point_distance_on_line_l4126_412619

/-- Given two points on a line, prove that the horizontal distance between them is 3 -/
theorem point_distance_on_line (m n : ℝ) : 
  (m = n / 5 - 2 / 5) → 
  (m + 3 = (n + 15) / 5 - 2 / 5) := by
sorry

end NUMINAMATH_CALUDE_point_distance_on_line_l4126_412619


namespace NUMINAMATH_CALUDE_factorial_sum_of_squares_solutions_l4126_412673

theorem factorial_sum_of_squares_solutions :
  ∀ a b n : ℕ+,
    a ≤ b →
    n < 14 →
    a^2 + b^2 = n! →
    ((a = 1 ∧ b = 1 ∧ n = 2) ∨ (a = 12 ∧ b = 24 ∧ n = 6)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_of_squares_solutions_l4126_412673


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l4126_412664

/-- The value of p for a parabola y^2 = 2px (p > 0) with directrix tangent to the circle x^2 + y^2 + 2x = 0 -/
theorem parabola_directrix_tangent_circle (p : ℝ) : 
  p > 0 ∧ 
  (∃ x y : ℝ, y^2 = 2*p*x) ∧
  (∃ x y : ℝ, x^2 + y^2 + 2*x = 0) ∧
  (∃ x : ℝ, x = -p/2) ∧  -- directrix equation
  (∃ x y : ℝ, x^2 + y^2 + 2*x = 0 ∧ x = -p/2)  -- tangency condition
  → p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l4126_412664


namespace NUMINAMATH_CALUDE_tims_movie_marathon_l4126_412679

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

end NUMINAMATH_CALUDE_tims_movie_marathon_l4126_412679


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l4126_412614

theorem prime_square_mod_180 (p : Nat) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  ∃ (r₁ r₂ : Nat), r₁ ≠ r₂ ∧ 
  (∀ (r : Nat), p^2 % 180 = r → (r = r₁ ∨ r = r₂)) :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l4126_412614


namespace NUMINAMATH_CALUDE_basketball_weight_proof_l4126_412694

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 16

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 24

theorem basketball_weight_proof : 
  (6 * basketball_weight = 4 * kayak_weight) ∧ 
  (3 * kayak_weight = 72) → 
  basketball_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_basketball_weight_proof_l4126_412694


namespace NUMINAMATH_CALUDE_air_conditioner_sales_l4126_412695

theorem air_conditioner_sales (ac_ratio : ℕ) (ref_ratio : ℕ) (difference : ℕ) : 
  ac_ratio = 5 ∧ ref_ratio = 3 ∧ difference = 54 →
  ac_ratio * (difference / (ac_ratio - ref_ratio)) = 135 :=
by sorry

end NUMINAMATH_CALUDE_air_conditioner_sales_l4126_412695


namespace NUMINAMATH_CALUDE_vector_sum_equality_l4126_412604

/-- Given two 2D vectors a and b, prove that 2a + 3b equals the specified result. -/
theorem vector_sum_equality (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (-1, 2) →
  2 • a + 3 • b = (1, 8) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l4126_412604


namespace NUMINAMATH_CALUDE_cubic_inverse_exists_l4126_412610

noncomputable def k : ℝ := Real.sqrt 3

theorem cubic_inverse_exists (x y z : ℚ) (h : x + y * k + z * k^2 ≠ 0) :
  ∃ u v w : ℚ, (x + y * k + z * k^2) * (u + v * k + w * k^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inverse_exists_l4126_412610


namespace NUMINAMATH_CALUDE_select_from_m_gives_correct_probability_l4126_412692

def set_m : Finset Int := {-6, -5, -4, -3, -2}
def set_t : Finset Int := {-3, -2, -1, 0, 1, 2, 3, 4, 5}

def probability_negative_product : ℚ := 5 / 9

theorem select_from_m_gives_correct_probability :
  (set_m.card : ℚ) * (set_t.filter (λ x => x > 0)).card / set_t.card = probability_negative_product :=
sorry

end NUMINAMATH_CALUDE_select_from_m_gives_correct_probability_l4126_412692


namespace NUMINAMATH_CALUDE_class_size_l4126_412635

theorem class_size (boys girls : ℕ) : 
  boys = 3 * (boys / 3) ∧ 
  girls = 2 * (boys / 3) ∧ 
  boys = girls + 20 → 
  boys + girls = 100 := by
sorry

end NUMINAMATH_CALUDE_class_size_l4126_412635


namespace NUMINAMATH_CALUDE_max_distance_OM_l4126_412607

/-- The ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The circle O -/
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- The tangent line l -/
def tangent_line_l (x y m t : ℝ) : Prop :=
  x = m * y + t

theorem max_distance_OM (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (a^2 - b^2).sqrt / a = Real.sqrt 3 / 2) (h4 : 2 * b = 2)
  (A B M : ℝ × ℝ) (m t : ℝ)
  (hA : ellipse_C A.1 A.2 a b)
  (hB : ellipse_C B.1 B.2 a b)
  (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hl : tangent_line_l A.1 A.2 m t ∧ tangent_line_l B.1 B.2 m t)
  (htangent : t^2 = m^2 + 1) :
  (∀ P : ℝ × ℝ, ellipse_C P.1 P.2 a b → (P.1^2 + P.2^2).sqrt ≤ 5/4) ∧
  (∃ Q : ℝ × ℝ, ellipse_C Q.1 Q.2 a b ∧ (Q.1^2 + Q.2^2).sqrt = 5/4) :=
sorry

end NUMINAMATH_CALUDE_max_distance_OM_l4126_412607


namespace NUMINAMATH_CALUDE_mode_of_data_set_l4126_412653

def data_set : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data_set :
  mode data_set = 3 := by sorry

end NUMINAMATH_CALUDE_mode_of_data_set_l4126_412653


namespace NUMINAMATH_CALUDE_lucy_groceries_weight_l4126_412616

/-- The total weight of groceries Lucy bought -/
def total_weight (cookies_packs noodles_packs cookie_weight noodle_weight : ℕ) : ℕ :=
  cookies_packs * cookie_weight + noodles_packs * noodle_weight

/-- Theorem stating that the total weight of Lucy's groceries is 11000g -/
theorem lucy_groceries_weight :
  total_weight 12 16 250 500 = 11000 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_weight_l4126_412616
