import Mathlib

namespace NUMINAMATH_CALUDE_max_product_constraint_l2135_213552

theorem max_product_constraint (x y : ℝ) (h : x + y = 1) : x * y ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l2135_213552


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l2135_213537

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 15) :
  Nat.lcm a b * Nat.gcd a b = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l2135_213537


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_B_subset_complement_A_l2135_213504

-- Define the sets A and B
def A : Set ℝ := {x | (1 + x) / (2 - x) > 0}
def B (a : ℝ) : Set ℝ := {x | (a * x - 1) * (x + 2) ≥ 0}

-- Theorem 1: When a = 1, A ∩ B = {x | 1 ≤ x < 2}
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem 2: When B ⊆ ℝ\A, the range of a is 0 < a ≤ 1/2
theorem range_of_a_when_B_subset_complement_A :
  ∀ a : ℝ, (0 < a ∧ B a ⊆ (Set.univ \ A)) ↔ (0 < a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_B_subset_complement_A_l2135_213504


namespace NUMINAMATH_CALUDE_destination_distance_l2135_213543

theorem destination_distance (d : ℝ) : 
  (¬ (d ≥ 8)) →  -- Alice's statement is false
  (¬ (d ≤ 7)) →  -- Bob's statement is false
  (d ≠ 6) →      -- Charlie's statement is false
  7 < d ∧ d < 8 := by
sorry

end NUMINAMATH_CALUDE_destination_distance_l2135_213543


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l2135_213553

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) (hα : a ≠ 0) : 
  (∃ (x : ℚ), a * x^2 + b * x + c = 0) → 
  (Even a ∨ Even b ∨ Even c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l2135_213553


namespace NUMINAMATH_CALUDE_arthurs_wallet_l2135_213526

theorem arthurs_wallet (initial_amount : ℝ) : 
  (1 / 5 : ℝ) * initial_amount = 40 → initial_amount = 200 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_wallet_l2135_213526


namespace NUMINAMATH_CALUDE_sum_of_numbers_ge_threshold_l2135_213559

theorem sum_of_numbers_ge_threshold : 
  let numbers : List ℝ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let threshold : ℝ := 1.1
  (numbers.filter (λ x => x ≥ threshold)).sum = 3.9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_ge_threshold_l2135_213559


namespace NUMINAMATH_CALUDE_marie_cash_register_cost_l2135_213587

/-- A bakery's daily sales and expenses -/
structure BakeryFinances where
  bread_quantity : ℕ
  bread_price : ℝ
  cake_quantity : ℕ
  cake_price : ℝ
  rent : ℝ
  electricity : ℝ

/-- Calculate the cost of a cash register based on daily sales and expenses -/
def cash_register_cost (finances : BakeryFinances) (days : ℕ) : ℝ :=
  let daily_sales := finances.bread_quantity * finances.bread_price + 
                     finances.cake_quantity * finances.cake_price
  let daily_expenses := finances.rent + finances.electricity
  let daily_profit := daily_sales - daily_expenses
  days * daily_profit

/-- Marie's bakery finances -/
def marie_finances : BakeryFinances :=
  { bread_quantity := 40
  , bread_price := 2
  , cake_quantity := 6
  , cake_price := 12
  , rent := 20
  , electricity := 2 }

/-- Theorem: The cost of Marie's cash register is $1040 -/
theorem marie_cash_register_cost :
  cash_register_cost marie_finances 8 = 1040 := by
  sorry

end NUMINAMATH_CALUDE_marie_cash_register_cost_l2135_213587


namespace NUMINAMATH_CALUDE_right_triangle_area_l2135_213591

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_ratio : a / b = 7 / 24)
  (h_distance : (c / 2) * ((c / 2) - 2 * ((a + b - c) / 2)) = 1) :
  (1 / 2) * a * b = 336 / 325 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2135_213591


namespace NUMINAMATH_CALUDE_cloth_coloring_problem_l2135_213516

/-- The length of cloth that can be colored by a given number of men in a given number of days -/
def clothLength (men : ℕ) (days : ℚ) : ℚ :=
  sorry

theorem cloth_coloring_problem :
  let men₁ : ℕ := 6
  let days₁ : ℚ := 2
  let men₂ : ℕ := 2
  let days₂ : ℚ := 4.5
  let length₂ : ℚ := 36

  clothLength men₂ days₂ = length₂ →
  clothLength men₁ days₁ = 48 :=
by sorry

end NUMINAMATH_CALUDE_cloth_coloring_problem_l2135_213516


namespace NUMINAMATH_CALUDE_world_expo_ticket_sales_l2135_213555

theorem world_expo_ticket_sales :
  let regular_price : ℕ := 200
  let concession_price : ℕ := 120
  let total_tickets : ℕ := 1200
  let total_revenue : ℕ := 216000
  ∃ (regular_tickets concession_tickets : ℕ),
    regular_tickets + concession_tickets = total_tickets ∧
    regular_tickets * regular_price + concession_tickets * concession_price = total_revenue ∧
    regular_tickets = 900 ∧
    concession_tickets = 300 := by
sorry

end NUMINAMATH_CALUDE_world_expo_ticket_sales_l2135_213555


namespace NUMINAMATH_CALUDE_bandwidth_calculation_correct_l2135_213500

/-- Represents the parameters for an audio channel --/
structure AudioChannelParams where
  sessionDurationMinutes : ℕ
  samplingRate : ℕ
  samplingDepth : ℕ
  metadataBytes : ℕ
  metadataPerAudioKilobits : ℕ

/-- Calculates the required bandwidth for a stereo audio channel --/
def calculateBandwidth (params : AudioChannelParams) : ℚ :=
  let sessionDurationSeconds := params.sessionDurationMinutes * 60
  let dataVolume := params.samplingRate * params.samplingDepth * sessionDurationSeconds
  let metadataVolume := params.metadataBytes * 8 * dataVolume / (params.metadataPerAudioKilobits * 1024)
  let totalDataVolume := (dataVolume + metadataVolume) * 2
  totalDataVolume / (sessionDurationSeconds * 1024)

/-- Theorem stating that the calculated bandwidth matches the expected result --/
theorem bandwidth_calculation_correct (params : AudioChannelParams) 
  (h1 : params.sessionDurationMinutes = 51)
  (h2 : params.samplingRate = 63)
  (h3 : params.samplingDepth = 17)
  (h4 : params.metadataBytes = 47)
  (h5 : params.metadataPerAudioKilobits = 5) :
  calculateBandwidth params = 2.25 := by
  sorry

#eval calculateBandwidth {
  sessionDurationMinutes := 51,
  samplingRate := 63,
  samplingDepth := 17,
  metadataBytes := 47,
  metadataPerAudioKilobits := 5
}

end NUMINAMATH_CALUDE_bandwidth_calculation_correct_l2135_213500


namespace NUMINAMATH_CALUDE_intersecting_line_properties_l2135_213570

/-- A line that intersects both positive x-axis and positive y-axis -/
structure IntersectingLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line intersects the positive x-axis -/
  pos_x_intersect : ∃ x : ℝ, x > 0 ∧ m * x + b = 0
  /-- The line intersects the positive y-axis -/
  pos_y_intersect : b > 0

/-- Theorem: An intersecting line has negative slope and positive y-intercept -/
theorem intersecting_line_properties (l : IntersectingLine) : l.m < 0 ∧ l.b > 0 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_line_properties_l2135_213570


namespace NUMINAMATH_CALUDE_third_month_sale_l2135_213571

def average_sale : ℕ := 3500
def number_of_months : ℕ := 6
def sale_month1 : ℕ := 3435
def sale_month2 : ℕ := 3920
def sale_month4 : ℕ := 4230
def sale_month5 : ℕ := 3560
def sale_month6 : ℕ := 2000

theorem third_month_sale :
  let total_sales := average_sale * number_of_months
  let known_sales := sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 3855 := by
sorry

end NUMINAMATH_CALUDE_third_month_sale_l2135_213571


namespace NUMINAMATH_CALUDE_complement_of_120_degrees_l2135_213536

-- Define the angle in degrees
def given_angle : ℝ := 120

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem complement_of_120_degrees :
  complement given_angle = 60 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_120_degrees_l2135_213536


namespace NUMINAMATH_CALUDE_gcf_of_120_180_300_l2135_213598

theorem gcf_of_120_180_300 : Nat.gcd 120 (Nat.gcd 180 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_120_180_300_l2135_213598


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2135_213556

theorem product_sum_theorem (x y : ℤ) : 
  y = x + 2 → x * y = 20400 → x + y = 286 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2135_213556


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2135_213593

theorem square_plus_reciprocal_square (m : ℝ) (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 6 = 104 := by
sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2135_213593


namespace NUMINAMATH_CALUDE_youngest_child_age_l2135_213506

/-- Represents a family with its members and ages -/
structure Family where
  memberCount : ℕ
  totalAge : ℕ

/-- Calculates the average age of a family -/
def averageAge (f : Family) : ℚ :=
  f.totalAge / f.memberCount

theorem youngest_child_age (initialFamily : Family) 
  (finalFamily : Family) (yearsPassed : ℕ) :
  initialFamily.memberCount = 4 →
  averageAge initialFamily = 24 →
  yearsPassed = 10 →
  finalFamily.memberCount = initialFamily.memberCount + 2 →
  averageAge finalFamily = 24 →
  ∃ (youngestAge olderAge : ℕ), 
    olderAge = youngestAge + 2 ∧
    youngestAge + olderAge = finalFamily.totalAge - (initialFamily.totalAge + yearsPassed * initialFamily.memberCount) ∧
    youngestAge = 3 := by
  sorry


end NUMINAMATH_CALUDE_youngest_child_age_l2135_213506


namespace NUMINAMATH_CALUDE_min_operations_to_2187_l2135_213515

/-- Represents the possible operations on the calculator --/
inductive Operation
  | AddOne
  | TimesThree

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.TimesThree => n * 3

/-- Checks if a sequence of operations transforms 1 into the target --/
def isValidSequence (ops : List Operation) (target : ℕ) : Prop :=
  ops.foldl applyOperation 1 = target

/-- The main theorem to prove --/
theorem min_operations_to_2187 :
  ∃ (ops : List Operation), isValidSequence ops 2187 ∧ 
    ops.length = 7 ∧ 
    (∀ (other_ops : List Operation), isValidSequence other_ops 2187 → other_ops.length ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_2187_l2135_213515


namespace NUMINAMATH_CALUDE_correct_calculation_l2135_213524

theorem correct_calculation (a : ℝ) : 8 * a^2 - 5 * a^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2135_213524


namespace NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l2135_213542

theorem video_recorder_wholesale_cost :
  ∀ (wholesale_cost retail_price employee_price : ℝ),
    retail_price = 1.2 * wholesale_cost →
    employee_price = 0.7 * retail_price →
    employee_price = 168 →
    wholesale_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l2135_213542


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2135_213521

theorem sufficient_not_necessary_condition (x y : ℝ) : 
  (((x ≥ 2 ∧ y ≥ 2) → x^2 + y^2 ≥ 4) ∧ 
   (∃ a b : ℝ, a^2 + b^2 ≥ 4 ∧ ¬(a ≥ 2 ∧ b ≥ 2))) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2135_213521


namespace NUMINAMATH_CALUDE_max_sum_constrained_l2135_213599

theorem max_sum_constrained (x y : ℝ) (h : x^2 + y^2 + x*y = 1) :
  x + y ≤ 2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_sum_constrained_l2135_213599


namespace NUMINAMATH_CALUDE_count_without_one_between_1_and_2000_l2135_213517

/-- Count of numbers without digit 1 in a given range -/
def count_without_digit_one (lower : Nat) (upper : Nat) : Nat :=
  sorry

/-- The main theorem -/
theorem count_without_one_between_1_and_2000 :
  count_without_digit_one 1 2000 = 1457 := by sorry

end NUMINAMATH_CALUDE_count_without_one_between_1_and_2000_l2135_213517


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l2135_213588

/-- Given even integers a and b where b = a + 2, and c = ab, √(a^2 + b^2 + c^2) is always irrational. -/
theorem sqrt_D_irrational (a b c : ℤ) : 
  Even a → Even b → b = a + 2 → c = a * b → 
  Irrational (Real.sqrt ((a^2 : ℝ) + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l2135_213588


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l2135_213511

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An equilateral triangle with two vertices given -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  third_in_first_quadrant : Bool

/-- The y-coordinate of the third vertex of an equilateral triangle -/
def third_vertex_y_coord (t : EquilateralTriangle) : ℝ :=
  sorry

theorem equilateral_triangle_third_vertex_y_coord 
  (t : EquilateralTriangle) 
  (h1 : t.v1 = ⟨1, 3⟩) 
  (h2 : t.v2 = ⟨9, 3⟩) 
  (h3 : t.third_in_first_quadrant = true) : 
  third_vertex_y_coord t = 3 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l2135_213511


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2135_213503

-- Define complex numbers
variable (a b c d e f : ℝ)

-- Define the theorem
theorem complex_sum_problem :
  b = 4 →
  e = -2*a - c →
  (a + b*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = 5*Complex.I →
  d + 2*f = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2135_213503


namespace NUMINAMATH_CALUDE_rectangular_to_cubic_block_l2135_213590

/-- The edge length of a cube with the same volume as a rectangular block -/
def cube_edge_length (l w h : ℝ) : ℝ :=
  (l * w * h) ^ (1/3)

/-- Theorem stating that a 50cm x 8cm x 20cm rectangular block forged into a cube has an edge length of 20cm -/
theorem rectangular_to_cubic_block :
  cube_edge_length 50 8 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_cubic_block_l2135_213590


namespace NUMINAMATH_CALUDE_point_p_final_position_point_q_initial_position_l2135_213541

-- Define the movement of point P
def point_p_movement : ℝ := 2

-- Define the movement of point Q
def point_q_movement : ℝ := 3

-- Theorem for point P's final position
theorem point_p_final_position :
  point_p_movement = 2 → 0 + point_p_movement = 2 :=
by sorry

-- Theorem for point Q's initial position
theorem point_q_initial_position :
  point_q_movement = 3 →
  (0 + point_q_movement = 3 ∨ 0 - point_q_movement = -3) :=
by sorry

end NUMINAMATH_CALUDE_point_p_final_position_point_q_initial_position_l2135_213541


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odds_l2135_213583

theorem largest_divisor_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  ∃ (k : ℕ), k = 105 ∧ 
  (∀ (d : ℕ), d ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → d ≤ k) ∧
  k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odds_l2135_213583


namespace NUMINAMATH_CALUDE_alec_class_size_l2135_213505

theorem alec_class_size :
  ∀ S : ℕ,
  (3 * S / 4 : ℚ) = S / 2 + 5 + ((S / 2 - 5) / 5 : ℚ) + 5 →
  S = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_alec_class_size_l2135_213505


namespace NUMINAMATH_CALUDE_count_two_digit_S_equal_l2135_213581

def S (n : ℕ) : ℕ :=
  (n % 2) + (n % 3) + (n % 4) + (n % 5)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem count_two_digit_S_equal : 
  ∃ (l : List ℕ), (∀ n ∈ l, is_two_digit n ∧ S n = S (n + 1)) ∧ 
                  (∀ n, is_two_digit n → S n = S (n + 1) → n ∈ l) ∧
                  l.length = 6 :=
sorry

end NUMINAMATH_CALUDE_count_two_digit_S_equal_l2135_213581


namespace NUMINAMATH_CALUDE_area_triangle_PAB_l2135_213529

/-- Given points A(-1, 2), B(3, 4), and P on the x-axis such that |PA| = |PB|,
    the area of triangle PAB is 15/2. -/
theorem area_triangle_PAB :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (3, 4)
  ∀ P : ℝ × ℝ,
    P.2 = 0 →  -- P is on the x-axis
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 →  -- |PA| = |PB|
    abs ((B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1)) / 2 = 15/2 :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_PAB_l2135_213529


namespace NUMINAMATH_CALUDE_yellow_pencils_count_l2135_213560

/-- Represents a grid of colored pencils -/
structure PencilGrid :=
  (size : ℕ)
  (perimeter_color : String)
  (inside_color : String)

/-- Calculates the number of pencils of the inside color in the grid -/
def count_inside_pencils (grid : PencilGrid) : ℕ :=
  grid.size * grid.size - (4 * grid.size - 4)

/-- The theorem to be proved -/
theorem yellow_pencils_count (grid : PencilGrid) 
  (h1 : grid.size = 10)
  (h2 : grid.perimeter_color = "red")
  (h3 : grid.inside_color = "yellow") :
  count_inside_pencils grid = 64 := by
  sorry

end NUMINAMATH_CALUDE_yellow_pencils_count_l2135_213560


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2135_213569

theorem geometric_sequence_property (x : ℝ) (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence property
  a 1 = Real.sin x →
  a 2 = Real.cos x →
  a 3 = Real.tan x →
  a 8 = 1 + Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2135_213569


namespace NUMINAMATH_CALUDE_divides_two_pow_36_minus_1_l2135_213551

theorem divides_two_pow_36_minus_1 : 
  ∃! (n : ℕ), 40 ≤ n ∧ n ≤ 50 ∧ (2^36 - 1) % n = 0 ∧ n = 49 := by
  sorry

end NUMINAMATH_CALUDE_divides_two_pow_36_minus_1_l2135_213551


namespace NUMINAMATH_CALUDE_jeds_board_games_l2135_213534

/-- The number of board games Jed's family bought -/
def num_board_games : ℕ := 6

/-- The cost of each board game in dollars -/
def cost_per_game : ℕ := 15

/-- The amount Jed paid in dollars -/
def amount_paid : ℕ := 100

/-- The number of $5 bills Jed received as change -/
def num_change_bills : ℕ := 2

/-- The value of each change bill in dollars -/
def change_bill_value : ℕ := 5

theorem jeds_board_games :
  num_board_games = (amount_paid - (num_change_bills * change_bill_value)) / cost_per_game :=
by sorry

end NUMINAMATH_CALUDE_jeds_board_games_l2135_213534


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2135_213550

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x = 4 - Real.sqrt 2 ∧
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ (y : ℝ), (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2135_213550


namespace NUMINAMATH_CALUDE_sqrt_eight_simplification_l2135_213572

theorem sqrt_eight_simplification : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_simplification_l2135_213572


namespace NUMINAMATH_CALUDE_polynomial_condition_l2135_213546

/-- A polynomial P satisfying the given condition for all real a, b, c is of the form ax² + bx -/
theorem polynomial_condition (P : ℝ → ℝ) : 
  (∀ (a b c : ℝ), P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)) →
  ∃ (a b : ℝ), ∀ x, P x = a * x^2 + b * x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_condition_l2135_213546


namespace NUMINAMATH_CALUDE_work_completion_time_l2135_213585

/-- Given two workers a and b, where a is twice as fast as b, and b can complete a work in 24 days,
    prove that a and b together can complete the work in 8 days. -/
theorem work_completion_time (a b : ℝ) (h1 : a = 2 * b) (h2 : b * 24 = 1) :
  1 / (a + b) = 8 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2135_213585


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2135_213518

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (time : ℝ)
  (rate : ℝ)
  (h1 : interest = 2500)
  (h2 : time = 5)
  (h3 : rate = 10)
  : interest = (5000 * rate * time) / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2135_213518


namespace NUMINAMATH_CALUDE_t_greater_than_a_squared_l2135_213544

/-- An equilateral triangle with a point on one of its sides -/
structure EquilateralTriangleWithPoint where
  a : ℝ  -- Side length of the equilateral triangle
  x : ℝ  -- Distance from A to P on side AB
  h1 : 0 < a  -- Side length is positive
  h2 : 0 ≤ x ∧ x ≤ a  -- P is on side AB

/-- The expression t = AP^2 + PB^2 + CP^2 -/
def t (triangle : EquilateralTriangleWithPoint) : ℝ :=
  let a := triangle.a
  let x := triangle.x
  x^2 + (a - x)^2 + (a^2 - a*x + x^2)

/-- Theorem: t is always greater than a^2 -/
theorem t_greater_than_a_squared (triangle : EquilateralTriangleWithPoint) :
  t triangle > triangle.a^2 := by
  sorry

end NUMINAMATH_CALUDE_t_greater_than_a_squared_l2135_213544


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2135_213594

theorem smallest_n_congruence : ∃! n : ℕ+, 
  (∀ m : ℕ+, 13 * m ≡ 456 [ZMOD 5] → n ≤ m) ∧ 
  13 * n ≡ 456 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2135_213594


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_first_four_primes_l2135_213576

def first_four_primes : List ℕ := [2, 3, 5, 7]

theorem arithmetic_mean_of_reciprocals_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length : ℚ) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_first_four_primes_l2135_213576


namespace NUMINAMATH_CALUDE_artist_paintings_l2135_213538

/-- Calculates the number of paintings an artist can make in a given number of weeks -/
def paintings_in_weeks (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) : ℕ :=
  (hours_per_week / hours_per_painting) * num_weeks

/-- Proves that an artist who spends 30 hours painting per week and takes 3 hours to complete a painting can make 40 paintings in four weeks -/
theorem artist_paintings : paintings_in_weeks 30 3 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_artist_paintings_l2135_213538


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l2135_213563

theorem algebraic_expression_simplification (x : ℝ) :
  x = 2 * Real.cos (45 * π / 180) + 1 →
  (1 / (x - 1) - (x - 3) / (x^2 - 2*x + 1)) / (2 / (x - 1)) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l2135_213563


namespace NUMINAMATH_CALUDE_lassis_from_twelve_mangoes_l2135_213567

/-- The number of lassis Caroline can make from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  (11 * mangoes) / 2

/-- Theorem stating that Caroline can make 66 lassis from 12 mangoes -/
theorem lassis_from_twelve_mangoes :
  lassis_from_mangoes 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_lassis_from_twelve_mangoes_l2135_213567


namespace NUMINAMATH_CALUDE_friend_bike_speed_l2135_213535

/-- Proves that given Joann's speed and time, Fran's speed can be calculated for the same distance --/
theorem friend_bike_speed 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4) 
  (h3 : fran_time = 5) :
  joann_speed * joann_time / fran_time = 12 := by
  sorry

#check friend_bike_speed

end NUMINAMATH_CALUDE_friend_bike_speed_l2135_213535


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2135_213509

theorem arithmetic_sequence_length (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) (n : ℕ) 
  (h₁ : a₁ = 3.25)
  (h₂ : aₙ = 55.25)
  (h₃ : d = 4)
  (h₄ : aₙ = a₁ + (n - 1) * d) :
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2135_213509


namespace NUMINAMATH_CALUDE_reynald_soccer_balls_l2135_213564

/-- The number of soccer balls Reynald bought -/
def soccer_balls : ℕ := 20

/-- The total number of balls Reynald bought -/
def total_balls : ℕ := 145

/-- The number of volleyballs Reynald bought -/
def volleyballs : ℕ := 30

theorem reynald_soccer_balls :
  soccer_balls = 20 ∧
  soccer_balls + (soccer_balls + 5) + (2 * soccer_balls) + (soccer_balls + 10) + volleyballs = total_balls :=
by sorry

end NUMINAMATH_CALUDE_reynald_soccer_balls_l2135_213564


namespace NUMINAMATH_CALUDE_product_remainder_one_mod_three_l2135_213574

theorem product_remainder_one_mod_three (a b : ℕ) :
  a % 3 = 1 → b % 3 = 1 → (a * b) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_one_mod_three_l2135_213574


namespace NUMINAMATH_CALUDE_trajectory_intersection_properties_l2135_213568

-- Define the trajectory of point M
def trajectory (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) = |x| + 1

-- Define line l₁
def line_l1 (x y : ℝ) : Prop :=
  y = x + 1

-- Define line l₂
def line_l2 (x y : ℝ) : Prop :=
  y = Real.sqrt 3 / 3 * (x - 1)

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem trajectory_intersection_properties :
  ∃ (A B : ℝ × ℝ),
    (trajectory A.1 A.2 ∧ line_l2 A.1 A.2) ∧
    (trajectory B.1 B.2 ∧ line_l2 B.1 B.2) ∧
    (A ≠ B) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16) ∧
    (Real.sqrt ((A.1 - point_F.1)^2 + (A.2 - point_F.2)^2) *
     Real.sqrt ((B.1 - point_F.1)^2 + (B.2 - point_F.2)^2) = 16) :=
sorry

end NUMINAMATH_CALUDE_trajectory_intersection_properties_l2135_213568


namespace NUMINAMATH_CALUDE_faye_age_l2135_213548

/-- Represents the ages of the people in the problem -/
structure Ages where
  diana : ℕ
  eduardo : ℕ
  chad : ℕ
  faye : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.diana = ages.eduardo - 4 ∧
  ages.eduardo = ages.chad + 5 ∧
  ages.faye = ages.chad + 4 ∧
  ages.diana = 18

/-- The theorem stating that under the given conditions, Faye is 21 years old -/
theorem faye_age (ages : Ages) : problem_conditions ages → ages.faye = 21 := by
  sorry

end NUMINAMATH_CALUDE_faye_age_l2135_213548


namespace NUMINAMATH_CALUDE_problem_statement_l2135_213582

theorem problem_statement (x y : ℤ) (h1 : x = 7) (h2 : y = x + 5) :
  (x - y) * (x + y) = -95 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2135_213582


namespace NUMINAMATH_CALUDE_earnings_ratio_l2135_213580

theorem earnings_ratio (mork_rate mindy_rate combined_rate : ℝ) 
  (h1 : mork_rate = 0.30)
  (h2 : mindy_rate = 0.20)
  (h3 : combined_rate = 0.225) : 
  ∃ (m k : ℝ), m > 0 ∧ k > 0 ∧ 
    (mindy_rate * m + mork_rate * k) / (m + k) = combined_rate ∧ 
    m / k = 3 := by
  sorry

end NUMINAMATH_CALUDE_earnings_ratio_l2135_213580


namespace NUMINAMATH_CALUDE_geometric_sum_remainder_l2135_213595

theorem geometric_sum_remainder (n : ℕ) : 
  (((5^(n+1) - 1) / 4) % 500 = 31) ∧ (n = 1002) := by sorry

end NUMINAMATH_CALUDE_geometric_sum_remainder_l2135_213595


namespace NUMINAMATH_CALUDE_colberts_treehouse_l2135_213586

theorem colberts_treehouse (total : ℕ) (storage : ℕ) (parents : ℕ) (store : ℕ) (friends : ℕ) : 
  total = 200 →
  storage = total / 4 →
  parents = total / 2 →
  store = 30 →
  total = storage + parents + store + friends →
  friends = 20 := by
sorry

end NUMINAMATH_CALUDE_colberts_treehouse_l2135_213586


namespace NUMINAMATH_CALUDE_students_after_three_stops_l2135_213565

/-- Calculates the number of students on the bus after three stops --/
def studentsOnBusAfterThreeStops (initial : ℕ) 
  (firstOff firstOn : ℕ) 
  (secondOff secondOn : ℕ) 
  (thirdOff thirdOn : ℕ) : ℕ :=
  initial - firstOff + firstOn - secondOff + secondOn - thirdOff + thirdOn

/-- Theorem stating the number of students on the bus after three stops --/
theorem students_after_three_stops :
  studentsOnBusAfterThreeStops 10 3 4 2 5 6 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_students_after_three_stops_l2135_213565


namespace NUMINAMATH_CALUDE_eccentricity_range_l2135_213532

/-- An ellipse with center O and endpoint A of its major axis -/
structure Ellipse where
  center : ℝ × ℝ
  majorAxis : ℝ
  eccentricity : ℝ

/-- The condition that there is no point P on the ellipse such that ∠OPA = π/2 -/
def noRightAngle (e : Ellipse) : Prop :=
  ∀ p : ℝ × ℝ, p ≠ e.center → p ≠ (e.center.1 + e.majorAxis, e.center.2) →
    (p.1 - e.center.1)^2 + (p.2 - e.center.2)^2 = e.majorAxis^2 * (1 - e.eccentricity^2) →
    (p.1 - e.center.1) * (p.1 - (e.center.1 + e.majorAxis)) +
    (p.2 - e.center.2) * p.2 ≠ 0

/-- The theorem stating the range of eccentricity -/
theorem eccentricity_range (e : Ellipse) :
  0 < e.eccentricity ∧ e.eccentricity < 1 ∧ noRightAngle e →
  0 < e.eccentricity ∧ e.eccentricity ≤ Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_eccentricity_range_l2135_213532


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2135_213554

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_1 + a_7 = 10, then a_3 + a_5 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 1 + a 7 = 10) :
  a 3 + a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2135_213554


namespace NUMINAMATH_CALUDE_correct_order_of_operations_l2135_213522

-- Define the expression
def expression : List ℤ := [150, -50, 25, 5]

-- Define the operations
inductive Operation
| Addition
| Subtraction
| Multiplication

-- Define the order of operations
def orderOfOperations : List Operation := [Operation.Multiplication, Operation.Subtraction, Operation.Addition]

-- Function to evaluate the expression
def evaluate (expr : List ℤ) (ops : List Operation) : ℤ :=
  sorry

-- Theorem statement
theorem correct_order_of_operations :
  evaluate expression orderOfOperations = 225 :=
sorry

end NUMINAMATH_CALUDE_correct_order_of_operations_l2135_213522


namespace NUMINAMATH_CALUDE_impossibility_of_tiling_l2135_213519

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)
  (missing_corner : Bool)

/-- Represents a trimino -/
structure Trimino :=
  (length : ℕ)
  (width : ℕ)

/-- Determines if a checkerboard can be tiled with triminos -/
def can_tile (board : Checkerboard) (tile : Trimino) : Prop :=
  ∃ (tiling : ℕ), 
    (board.rows * board.cols - if board.missing_corner then 1 else 0) = 
    tiling * (tile.length * tile.width)

theorem impossibility_of_tiling (board : Checkerboard) (tile : Trimino) : 
  (board.rows = 8 ∧ board.cols = 8 ∧ tile.length = 3 ∧ tile.width = 1) →
  (¬ can_tile board tile) ∧ 
  (¬ can_tile {rows := board.rows, cols := board.cols, missing_corner := true} tile) :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_tiling_l2135_213519


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2135_213513

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (sum_eq_8 : p + q + r + s + t + u = 8) : 
  (1/p + 4/q + 9/r + 16/s + 25/t + 49/u) ≥ 60.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2135_213513


namespace NUMINAMATH_CALUDE_quarterback_no_throw_percentage_l2135_213508

/-- Given a quarterback's statistics in a game, calculate the percentage of time he doesn't throw a pass. -/
theorem quarterback_no_throw_percentage 
  (total_attempts : ℕ) 
  (sacks : ℕ) 
  (h1 : total_attempts = 80) 
  (h2 : sacks = 12) 
  (h3 : 2 * sacks = total_attempts - (total_attempts - 2 * sacks)) : 
  (2 * sacks : ℚ) / total_attempts = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_quarterback_no_throw_percentage_l2135_213508


namespace NUMINAMATH_CALUDE_high_school_math_club_payment_l2135_213549

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem high_school_math_club_payment :
  ∀ B : ℕ, 
    B < 10 →
    is_divisible_by (2000 + 100 * B + 40) 15 →
    B = 7 := by
  sorry

end NUMINAMATH_CALUDE_high_school_math_club_payment_l2135_213549


namespace NUMINAMATH_CALUDE_stating_days_worked_when_net_zero_l2135_213547

/-- Represents the number of days in the work period -/
def total_days : ℕ := 30

/-- Represents the daily wage in su -/
def daily_wage : ℕ := 24

/-- Represents the daily penalty for skipping work in su -/
def daily_penalty : ℕ := 6

/-- 
Theorem stating that if a worker's net earnings are zero after the work period,
given the specified daily wage and penalty, then the number of days worked is 6.
-/
theorem days_worked_when_net_zero : 
  ∀ (days_worked : ℕ), 
    days_worked ≤ total_days →
    (daily_wage * days_worked - daily_penalty * (total_days - days_worked) = 0) →
    days_worked = 6 := by
  sorry

end NUMINAMATH_CALUDE_stating_days_worked_when_net_zero_l2135_213547


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l2135_213578

/-- Given a pizza with 8 slices, if two people each eat 3/2 slices, then 5 slices remain. -/
theorem pizza_slices_remaining (total_slices : ℕ) (slices_per_person : ℚ) (people : ℕ) : 
  total_slices = 8 → slices_per_person = 3/2 → people = 2 → 
  total_slices - (↑people * slices_per_person).num = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l2135_213578


namespace NUMINAMATH_CALUDE_wide_flags_count_l2135_213527

/-- Represents the flag-making scenario with given parameters -/
structure FlagScenario where
  totalFabric : ℕ
  squareFlagSide : ℕ
  wideRectFlagWidth : ℕ
  wideRectFlagHeight : ℕ
  tallRectFlagWidth : ℕ
  tallRectFlagHeight : ℕ
  squareFlagsMade : ℕ
  tallFlagsMade : ℕ
  fabricLeft : ℕ

/-- Calculates the number of wide rectangular flags made -/
def wideFlagsMade (scenario : FlagScenario) : ℕ :=
  let squareFlagArea := scenario.squareFlagSide * scenario.squareFlagSide
  let wideFlagArea := scenario.wideRectFlagWidth * scenario.wideRectFlagHeight
  let tallFlagArea := scenario.tallRectFlagWidth * scenario.tallRectFlagHeight
  let usedFabric := scenario.totalFabric - scenario.fabricLeft
  let squareAndTallFlagsArea := scenario.squareFlagsMade * squareFlagArea + scenario.tallFlagsMade * tallFlagArea
  let wideFlagsArea := usedFabric - squareAndTallFlagsArea
  wideFlagsArea / wideFlagArea

/-- Theorem stating that the number of wide flags made is 20 -/
theorem wide_flags_count (scenario : FlagScenario) 
  (h1 : scenario.totalFabric = 1000)
  (h2 : scenario.squareFlagSide = 4)
  (h3 : scenario.wideRectFlagWidth = 5)
  (h4 : scenario.wideRectFlagHeight = 3)
  (h5 : scenario.tallRectFlagWidth = 3)
  (h6 : scenario.tallRectFlagHeight = 5)
  (h7 : scenario.squareFlagsMade = 16)
  (h8 : scenario.tallFlagsMade = 10)
  (h9 : scenario.fabricLeft = 294) :
  wideFlagsMade scenario = 20 := by
  sorry


end NUMINAMATH_CALUDE_wide_flags_count_l2135_213527


namespace NUMINAMATH_CALUDE_evaluate_expression_l2135_213533

theorem evaluate_expression : (30 - (3030 - 303)) * (3030 - (303 - 30)) = -7435969 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2135_213533


namespace NUMINAMATH_CALUDE_matthew_crackers_left_l2135_213558

/-- Calculates the number of crackers Matthew has left after distributing them to friends and the friends eating some. -/
def crackers_left (initial_crackers : ℕ) (num_friends : ℕ) (crackers_eaten_per_friend : ℕ) : ℕ :=
  let distributed_crackers := initial_crackers - 1
  let crackers_per_friend := distributed_crackers / num_friends
  let remaining_with_friends := (crackers_per_friend - crackers_eaten_per_friend) * num_friends
  1 + remaining_with_friends

/-- Proves that Matthew has 11 crackers left given the initial conditions. -/
theorem matthew_crackers_left :
  crackers_left 23 2 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_matthew_crackers_left_l2135_213558


namespace NUMINAMATH_CALUDE_smallest_divisor_of_4500_l2135_213557

theorem smallest_divisor_of_4500 : 
  ∀ n : ℕ, n > 0 ∧ n ∣ (4499 + 1) → n ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_4500_l2135_213557


namespace NUMINAMATH_CALUDE_scaling_transformation_result_l2135_213510

/-- The scaling transformation applied to a point (x, y) -/
def scaling (x y : ℝ) : ℝ × ℝ := (x, 3 * y)

/-- The original curve C: x^2 + 9y^2 = 9 -/
def original_curve (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

/-- The transformed curve -/
def transformed_curve (x' y' : ℝ) : Prop := x'^2 + y'^2 = 9

/-- Theorem stating that the scaling transformation of the original curve
    results in the transformed curve -/
theorem scaling_transformation_result :
  ∀ x y : ℝ, original_curve x y →
  let (x', y') := scaling x y
  transformed_curve x' y' := by
  sorry

end NUMINAMATH_CALUDE_scaling_transformation_result_l2135_213510


namespace NUMINAMATH_CALUDE_max_value_rational_function_l2135_213596

theorem max_value_rational_function : 
  ∃ (M : ℤ), M = 57 ∧ 
  (∀ (x : ℝ), (3 * x^2 + 9 * x + 21) / (3 * x^2 + 9 * x + 7) ≤ M) ∧
  (∃ (x : ℝ), (3 * x^2 + 9 * x + 21) / (3 * x^2 + 9 * x + 7) > M - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_rational_function_l2135_213596


namespace NUMINAMATH_CALUDE_tangent_line_equation_range_of_a_l2135_213530

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 10

-- Theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) :
  a = 1 →
  ∃ (m b : ℝ), m = 8 ∧ b = -2 ∧
  ∀ (x y : ℝ), y = f a x → (x = 2 → m*x - y + b = 0) :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ (a : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f a x < 0) →
  a > 9/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_range_of_a_l2135_213530


namespace NUMINAMATH_CALUDE_james_new_hourly_wage_l2135_213575

/-- Jame's hourly wage calculation --/
theorem james_new_hourly_wage :
  ∀ (new_hours_per_week old_hours_per_week old_hourly_wage : ℕ)
    (weeks_per_year : ℕ) (yearly_increase : ℕ),
  new_hours_per_week = 40 →
  old_hours_per_week = 25 →
  old_hourly_wage = 16 →
  weeks_per_year = 52 →
  yearly_increase = 20800 →
  ∃ (new_hourly_wage : ℕ),
    new_hourly_wage = 530 ∧
    new_hourly_wage * new_hours_per_week * weeks_per_year =
      old_hourly_wage * old_hours_per_week * weeks_per_year + yearly_increase :=
by
  sorry

end NUMINAMATH_CALUDE_james_new_hourly_wage_l2135_213575


namespace NUMINAMATH_CALUDE_store_profit_loss_l2135_213520

theorem store_profit_loss (price : ℝ) (profit_margin loss_margin : ℝ) : 
  price = 168 ∧ profit_margin = 0.2 ∧ loss_margin = 0.2 →
  (price - price / (1 + profit_margin)) + (price - price / (1 - loss_margin)) = -14 := by
  sorry

end NUMINAMATH_CALUDE_store_profit_loss_l2135_213520


namespace NUMINAMATH_CALUDE_picture_distance_l2135_213597

/-- Proves that for a wall of width 24 feet and a picture of width 4 feet hung in the center,
    the distance from the end of the wall to the nearest edge of the picture is 10 feet. -/
theorem picture_distance (wall_width picture_width : ℝ) (h1 : wall_width = 24) (h2 : picture_width = 4) :
  let distance := (wall_width - picture_width) / 2
  distance = 10 := by
sorry

end NUMINAMATH_CALUDE_picture_distance_l2135_213597


namespace NUMINAMATH_CALUDE_BA_equals_AB_l2135_213545

-- Define the matrices A and B
variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

-- Define the given conditions
def condition1 : Prop := A + B = A * B
def condition2 : Prop := A * B = !![12, -6; 9, -3]

-- State the theorem
theorem BA_equals_AB (h1 : condition1 A B) (h2 : condition2 A B) : 
  B * A = !![12, -6; 9, -3] := by sorry

end NUMINAMATH_CALUDE_BA_equals_AB_l2135_213545


namespace NUMINAMATH_CALUDE_total_age_proof_l2135_213523

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 8 years old
  Prove that the total of their ages is 22 years. -/
theorem total_age_proof (a b c : ℕ) : 
  b = 8 → a = b + 2 → b = 2 * c → a + b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_age_proof_l2135_213523


namespace NUMINAMATH_CALUDE_unique_solution_system_l2135_213589

theorem unique_solution_system : 
  ∃! (x y z : ℕ+), 
    (x : ℝ)^2 = 2 * ((y : ℝ) + (z : ℝ)) ∧ 
    (x : ℝ)^6 = (y : ℝ)^6 + (z : ℝ)^6 + 31 * ((y : ℝ)^2 + (z : ℝ)^2) ∧
    x = 2 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2135_213589


namespace NUMINAMATH_CALUDE_cubic_factor_identity_l2135_213562

theorem cubic_factor_identity (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_factor_identity_l2135_213562


namespace NUMINAMATH_CALUDE_cars_meeting_time_l2135_213566

/-- Two cars driving towards each other meet after a certain time -/
theorem cars_meeting_time (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) : 
  speed1 = 100 →
  speed1 = 1.25 * speed2 →
  distance = 720 →
  (distance / (speed1 + speed2)) = 4 := by
sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l2135_213566


namespace NUMINAMATH_CALUDE_rectangle_y_coordinate_l2135_213502

/-- Given a rectangle with vertices (-8, 1), (1, 1), (1, y), and (-8, y) in a rectangular coordinate system,
    if the area of the rectangle is 72, then y = 9 -/
theorem rectangle_y_coordinate (y : ℝ) : 
  let vertex1 : ℝ × ℝ := (-8, 1)
  let vertex2 : ℝ × ℝ := (1, 1)
  let vertex3 : ℝ × ℝ := (1, y)
  let vertex4 : ℝ × ℝ := (-8, y)
  let length : ℝ := vertex2.1 - vertex1.1
  let width : ℝ := vertex3.2 - vertex2.2
  let area : ℝ := length * width
  area = 72 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_coordinate_l2135_213502


namespace NUMINAMATH_CALUDE_tangent_line_minimum_two_roots_inequality_l2135_213540

noncomputable section

variables (m : ℝ) (x x₁ x₂ : ℝ) (a b n : ℝ)

def f (x : ℝ) : ℝ := Real.log x - m * x

theorem tangent_line_minimum (h : f e x = a * x + b) :
  ∃ (x₀ : ℝ), a + 2 * b = 1 / x₀ + 2 * Real.log x₀ - e - 2 ∧ 
  ∀ (x : ℝ), 1 / x + 2 * Real.log x - e - 2 ≥ 1 / x₀ + 2 * Real.log x₀ - e - 2 :=
sorry

theorem two_roots_inequality (h1 : f m x₁ = (2 - m) * x₁ + n) 
                             (h2 : f m x₂ = (2 - m) * x₂ + n) 
                             (h3 : x₁ < x₂) :
  2 * x₁ + x₂ > e / 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_two_roots_inequality_l2135_213540


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2135_213573

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem fourth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_3 : a 3 = 2) 
  (h_5 : a 5 = 16) : 
  a 4 = 4 * Real.sqrt 2 ∨ a 4 = -4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2135_213573


namespace NUMINAMATH_CALUDE_birthday_cookies_l2135_213561

theorem birthday_cookies (friends : ℕ) (packages : ℕ) (cookies_per_package : ℕ) :
  friends = 7 →
  packages = 5 →
  cookies_per_package = 36 →
  (packages * cookies_per_package) / (friends + 1) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_birthday_cookies_l2135_213561


namespace NUMINAMATH_CALUDE_first_chapter_pages_l2135_213531

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  chapter2_pages : ℕ

/-- The number of pages in the first chapter of a book -/
def pages_in_chapter1 (b : Book) : ℕ := b.total_pages - b.chapter2_pages

/-- Theorem stating that for a book with 93 total pages and 33 pages in the second chapter,
    the first chapter has 60 pages -/
theorem first_chapter_pages :
  ∀ (b : Book), b.total_pages = 93 → b.chapter2_pages = 33 → pages_in_chapter1 b = 60 := by
  sorry

end NUMINAMATH_CALUDE_first_chapter_pages_l2135_213531


namespace NUMINAMATH_CALUDE_sector_area_l2135_213501

/-- A sector with perimeter 12 cm and central angle 2 rad has an area of 9 cm² -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 12 → central_angle = 2 → area = 9 := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l2135_213501


namespace NUMINAMATH_CALUDE_quadratic_roots_l2135_213577

theorem quadratic_roots (p q : ℤ) (h1 : p + q = 198) :
  ∃ x₁ x₂ : ℤ, (x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) →
  ((x₁ = 2 ∧ x₂ = 200) ∨ (x₁ = 0 ∧ x₂ = -198)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2135_213577


namespace NUMINAMATH_CALUDE_train_crossing_time_l2135_213579

/-- The time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : Real) (train_speed_kmph : Real) (platform_length : Real) :
  train_length = 120 ∧ 
  train_speed_kmph = 72 ∧ 
  platform_length = 380.04 →
  (train_length + platform_length) / (train_speed_kmph * 1000 / 3600) = 25.002 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2135_213579


namespace NUMINAMATH_CALUDE_plane_relations_l2135_213584

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (in_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem plane_relations (a b : Plane) (h : a ≠ b) :
  (∀ (l : Line), in_plane l a → 
    (∀ (m : Line), in_plane m b → perpendicular l m) → 
    perpendicular_planes a b) ∧
  (∀ (l : Line), in_plane l a → 
    parallel_line_plane l b → 
    parallel_planes a b) ∧
  (parallel_planes a b → 
    ∀ (l : Line), in_plane l a → 
    parallel_line_plane l b) :=
by sorry

end NUMINAMATH_CALUDE_plane_relations_l2135_213584


namespace NUMINAMATH_CALUDE_definite_integral_tangent_fraction_l2135_213528

theorem definite_integral_tangent_fraction : 
  ∫ x in (0)..(π/4), (4 - 7 * Real.tan x) / (2 + 3 * Real.tan x) = Real.log (25/8) - π/4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_tangent_fraction_l2135_213528


namespace NUMINAMATH_CALUDE_luncheon_cost_is_105_l2135_213512

/-- The cost of a luncheon consisting of one sandwich, one cup of coffee, and one piece of pie -/
def luncheon_cost (s c p : ℚ) : ℚ := s + c + p

/-- The cost of the first luncheon combination -/
def first_combination (s c p : ℚ) : ℚ := 3 * s + 7 * c + p

/-- The cost of the second luncheon combination -/
def second_combination (s c p : ℚ) : ℚ := 4 * s + 10 * c + p

theorem luncheon_cost_is_105 
  (s c p : ℚ) 
  (h1 : first_combination s c p = 315/100) 
  (h2 : second_combination s c p = 420/100) : 
  luncheon_cost s c p = 105/100 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_cost_is_105_l2135_213512


namespace NUMINAMATH_CALUDE_green_eyed_students_l2135_213507

theorem green_eyed_students (total : ℕ) (both : ℕ) (neither : ℕ) :
  total = 50 →
  both = 10 →
  neither = 5 →
  ∃ (green : ℕ),
    green * 2 = (total - both - neither) - green ∧
    green = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_eyed_students_l2135_213507


namespace NUMINAMATH_CALUDE_percentage_sum_problem_l2135_213592

theorem percentage_sum_problem : (0.2 * 40) + (0.25 * 60) = 23 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_problem_l2135_213592


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l2135_213539

/-- The number of combinations when choosing one item from each of four categories -/
def total_combinations (wrapping_paper : ℕ) (ribbon : ℕ) (gift_cards : ℕ) (stickers : ℕ) : ℕ :=
  wrapping_paper * ribbon * gift_cards * stickers

/-- Theorem stating that the total number of combinations is 400 -/
theorem gift_wrapping_combinations :
  total_combinations 10 4 5 2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l2135_213539


namespace NUMINAMATH_CALUDE_brick_height_l2135_213514

/-- Proves that the height of a brick is 7.5 cm given the specified conditions -/
theorem brick_height (brick_length : ℝ) (brick_width : ℝ) 
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 20 →
  brick_width = 10 →
  wall_length = 2500 →
  wall_width = 200 →
  wall_height = 75 →
  num_bricks = 25000 →
  ∃ (brick_height : ℝ), 
    brick_height = 7.5 ∧ 
    brick_length * brick_width * brick_height * num_bricks = wall_length * wall_width * wall_height :=
by
  sorry

end NUMINAMATH_CALUDE_brick_height_l2135_213514


namespace NUMINAMATH_CALUDE_T_recursive_relation_l2135_213525

/-- The number of binary strings of length n such that any 4 adjacent digits sum to at least 1 -/
def T (n : ℕ) : ℕ :=
  if n < 4 then
    match n with
    | 0 => 1  -- Convention: empty string is valid
    | 1 => 2  -- "0" and "1" are valid
    | 2 => 3  -- "00", "01", "10", "11" are valid except "00"
    | 3 => 6  -- All combinations except "0000"
    | _ => 0  -- Should never reach here
  else
    T (n - 1) + T (n - 2) + T (n - 3) + T (n - 4)

/-- The main theorem stating the recursive relation for T(n) when n ≥ 4 -/
theorem T_recursive_relation (n : ℕ) (h : n ≥ 4) :
  T n = T (n - 1) + T (n - 2) + T (n - 3) + T (n - 4) := by sorry

end NUMINAMATH_CALUDE_T_recursive_relation_l2135_213525
