import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1544_154470

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - (a + 2) * x + 2 < 0}
  (a = 0 → solution_set = {x | x > 1}) ∧
  (0 < a ∧ a < 2 → solution_set = {x | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → solution_set = ∅) ∧
  (a > 2 → solution_set = {x | 2/a < x ∧ x < 1}) ∧
  (a < 0 → solution_set = {x | x < 2/a ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1544_154470


namespace NUMINAMATH_CALUDE_range_of_a_l1544_154441

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 4) → -3 ≤ a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1544_154441


namespace NUMINAMATH_CALUDE_unique_cube_root_l1544_154476

theorem unique_cube_root (M : ℕ+) : 18^3 * 50^3 = 30^3 * M^3 ↔ M = 30 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_root_l1544_154476


namespace NUMINAMATH_CALUDE_same_number_probability_l1544_154478

/-- The upper bound for the selected numbers -/
def upperBound : ℕ := 200

/-- Alice's number is a multiple of this value -/
def aliceMultiple : ℕ := 16

/-- Alan's number is a multiple of this value -/
def alanMultiple : ℕ := 28

/-- The probability of Alice and Alan selecting the same number -/
def sameProbability : ℚ := 1 / 84

theorem same_number_probability :
  (∃ (n : ℕ), n < upperBound ∧ n % aliceMultiple = 0 ∧ n % alanMultiple = 0) ∧
  (∀ (m : ℕ), m < upperBound → m % aliceMultiple = 0 → m % alanMultiple = 0 → m = lcm aliceMultiple alanMultiple) →
  sameProbability = (Nat.card {n : ℕ | n < upperBound ∧ n % aliceMultiple = 0 ∧ n % alanMultiple = 0}) /
    ((Nat.card {n : ℕ | n < upperBound ∧ n % aliceMultiple = 0}) * (Nat.card {n : ℕ | n < upperBound ∧ n % alanMultiple = 0})) :=
by sorry

end NUMINAMATH_CALUDE_same_number_probability_l1544_154478


namespace NUMINAMATH_CALUDE_equation_solution_l1544_154483

theorem equation_solution : ∃ (x₁ x₂ : ℚ),
  x₁ = -1/3 ∧ x₂ = -2 ∧
  (∀ x : ℚ, x ≠ 3 → x ≠ 1/2 → 
    ((2*x + 4) / (x - 3) = (x + 2) / (2*x - 1) ↔ x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1544_154483


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l1544_154481

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define a point on the x-axis
def on_x_axis (Q : ℝ × ℝ) : Prop := Q.2 = 0

-- Define the tangent property
def are_tangents (Q A B : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (P₁ P₂ : ℝ × ℝ) : ℝ := sorry

-- Define a line passing through a point
def line_passes_through (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop := l P

-- The main theorem
theorem circle_tangent_properties :
  ∀ (Q A B : ℝ × ℝ),
  circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
  on_x_axis Q ∧
  are_tangents Q A B →
  (distance A B = 4 * Real.sqrt 2 / 3 → distance (0, 2) Q = 3) ∧
  (∃ (l : ℝ × ℝ → Prop), ∀ (Q' : ℝ × ℝ), on_x_axis Q' ∧ are_tangents Q' A B → 
    line_passes_through (0, 3/2) l) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l1544_154481


namespace NUMINAMATH_CALUDE_rooms_with_two_windows_l1544_154406

/-- Represents a building with rooms and windows. -/
structure Building where
  total_windows : ℕ
  rooms_with_four : ℕ
  rooms_with_three : ℕ
  rooms_with_two : ℕ

/-- Conditions for the building. -/
def building_conditions (b : Building) : Prop :=
  b.total_windows = 122 ∧
  b.rooms_with_four = 5 ∧
  b.rooms_with_three = 8 ∧
  b.total_windows = 4 * b.rooms_with_four + 3 * b.rooms_with_three + 2 * b.rooms_with_two

/-- Theorem stating the number of rooms with two windows. -/
theorem rooms_with_two_windows (b : Building) :
  building_conditions b → b.rooms_with_two = 39 := by
  sorry

end NUMINAMATH_CALUDE_rooms_with_two_windows_l1544_154406


namespace NUMINAMATH_CALUDE_total_points_is_265_l1544_154494

/-- Given information about Paul's point assignment in the first quarter -/
structure PointAssignment where
  homework_points : ℕ
  quiz_points : ℕ
  test_points : ℕ
  hw_quiz_relation : quiz_points = homework_points + 5
  quiz_test_relation : test_points = 4 * quiz_points
  hw_given : homework_points = 40

/-- The total points assigned by Paul in the first quarter -/
def total_points (pa : PointAssignment) : ℕ :=
  pa.homework_points + pa.quiz_points + pa.test_points

/-- Theorem stating that the total points assigned is 265 -/
theorem total_points_is_265 (pa : PointAssignment) : total_points pa = 265 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_265_l1544_154494


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n_l1544_154496

/-- A positive integer greater than 1 is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k, 1 < k ∧ k < n ∧ n % k = 0

/-- For all composite integers n, 6 divides n^4 - n and is the largest such divisor. -/
theorem largest_divisor_of_n4_minus_n (n : ℕ) (h : IsComposite n) :
    (6 ∣ n^4 - n) ∧ ∀ m : ℕ, m > 6 → ¬(∀ k : ℕ, IsComposite k → (m ∣ k^4 - k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n_l1544_154496


namespace NUMINAMATH_CALUDE_percentage_relation_l1544_154433

theorem percentage_relation (x y : ℝ) (h : 0.15 * x = 0.2 * y) : y = 0.75 * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l1544_154433


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1544_154455

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  a 3 + a 6 + a 10 + a 13 = 32 →
  a m = 8 →
  m = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1544_154455


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l1544_154432

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (max : ℝ), max = Real.sqrt 15 ∧ x + 1/x ≤ max ∧ ∃ (y : ℝ), 13 = y^2 + 1/y^2 ∧ y + 1/y = max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l1544_154432


namespace NUMINAMATH_CALUDE_inverse_matrices_values_l1544_154418

theorem inverse_matrices_values (a b : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -9; a, 14]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![14, b; 5, 4]
  (A * B = 1 ∧ B * A = 1) → (a = -5 ∧ b = 9) :=
by sorry

end NUMINAMATH_CALUDE_inverse_matrices_values_l1544_154418


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_leq_one_l1544_154488

-- Define the property of being a meaningful square root
def is_meaningful_sqrt (x : ℝ) : Prop := 1 - x ≥ 0

-- State the theorem
theorem sqrt_meaningful_iff_leq_one :
  ∀ x : ℝ, is_meaningful_sqrt x ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_leq_one_l1544_154488


namespace NUMINAMATH_CALUDE_simplify_radicals_l1544_154461

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l1544_154461


namespace NUMINAMATH_CALUDE_dog_toy_cost_l1544_154493

/-- The cost of dog toys with a "buy one get one half off" deal -/
theorem dog_toy_cost (regular_price : ℝ) (num_toys : ℕ) : regular_price = 12 → num_toys = 4 →
  let discounted_price := regular_price / 2
  let pair_price := regular_price + discounted_price
  let total_cost := (num_toys / 2 : ℝ) * pair_price
  total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_dog_toy_cost_l1544_154493


namespace NUMINAMATH_CALUDE_terminal_side_angle_theorem_l1544_154439

theorem terminal_side_angle_theorem (α : Real) :
  (∃ (x y : Real), x = -2 ∧ y = 1 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  1 / Real.sin (2 * α) = -5/4 := by
sorry

end NUMINAMATH_CALUDE_terminal_side_angle_theorem_l1544_154439


namespace NUMINAMATH_CALUDE_meeting_participants_count_l1544_154428

theorem meeting_participants_count :
  ∀ (F M : ℕ),
  F > 0 →
  M > 0 →
  F / 2 = 125 →
  F / 2 + M / 4 = (F + M) / 3 →
  F + M = 750 :=
by sorry

end NUMINAMATH_CALUDE_meeting_participants_count_l1544_154428


namespace NUMINAMATH_CALUDE_three_squares_inequality_l1544_154412

/-- Given three equal squares arranged in a specific configuration, 
    this theorem proves that the length of the diagonal spanning two squares (AB) 
    is greater than the length of the diagonal spanning one square 
    and the side of another square (BC). -/
theorem three_squares_inequality (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  Real.sqrt (5 * x^2 + 4 * x * y + y^2) > Real.sqrt (5 * x^2 + 2 * x * y + y^2) := by
  sorry


end NUMINAMATH_CALUDE_three_squares_inequality_l1544_154412


namespace NUMINAMATH_CALUDE_blind_cave_scorpion_diet_l1544_154427

/-- The number of segments in the first millipede eaten by a blind cave scorpion -/
def first_millipede_segments : ℕ := 60

/-- The total number of segments the scorpion needs to eat daily -/
def total_required_segments : ℕ := 800

/-- The number of additional 50-segment millipedes the scorpion needs to eat -/
def additional_millipedes : ℕ := 10

/-- The number of segments in each additional millipede -/
def segments_per_additional_millipede : ℕ := 50

theorem blind_cave_scorpion_diet (x : ℕ) :
  x = first_millipede_segments ↔
    x + 2 * (2 * x) + additional_millipedes * segments_per_additional_millipede = total_required_segments :=
by
  sorry

end NUMINAMATH_CALUDE_blind_cave_scorpion_diet_l1544_154427


namespace NUMINAMATH_CALUDE_spring_sales_l1544_154469

/-- Represents the sales data for a fast food chain's hamburger sales across seasons --/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The total sales is the sum of sales from all seasons --/
def totalSales (s : SeasonalSales) : ℝ :=
  s.spring + s.summer + s.fall + s.winter

/-- Given the conditions of the problem --/
theorem spring_sales (s : SeasonalSales)
    (h1 : s.summer = 6)
    (h2 : s.fall = 4)
    (h3 : s.winter = 3)
    (h4 : s.winter = 0.2 * totalSales s) :
    s.spring = 2 := by
  sorry


end NUMINAMATH_CALUDE_spring_sales_l1544_154469


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1544_154460

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_of_M_in_U : 
  (U \ M) = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1544_154460


namespace NUMINAMATH_CALUDE_class_average_problem_l1544_154446

theorem class_average_problem (n : ℝ) (h1 : n > 0) :
  let total_average : ℝ := 80
  let quarter_average : ℝ := 92
  let quarter_sum : ℝ := quarter_average * (n / 4)
  let total_sum : ℝ := total_average * n
  let rest_sum : ℝ := total_sum - quarter_sum
  let rest_average : ℝ := rest_sum / (3 * n / 4)
  rest_average = 76 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l1544_154446


namespace NUMINAMATH_CALUDE_swimming_area_probability_l1544_154405

theorem swimming_area_probability (lake_radius swimming_area_radius : ℝ) 
  (lake_radius_pos : 0 < lake_radius)
  (swimming_area_radius_pos : 0 < swimming_area_radius)
  (swimming_area_in_lake : swimming_area_radius ≤ lake_radius) :
  lake_radius = 5 → swimming_area_radius = 3 →
  (π * swimming_area_radius^2) / (π * lake_radius^2) = 9 / 25 := by
sorry

end NUMINAMATH_CALUDE_swimming_area_probability_l1544_154405


namespace NUMINAMATH_CALUDE_machine_depletion_rate_l1544_154423

theorem machine_depletion_rate 
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (time : ℝ) 
  (h1 : initial_value = 400)
  (h2 : final_value = 225)
  (h3 : time = 2) :
  ∃ (rate : ℝ), 
    final_value = initial_value * (1 - rate) ^ time ∧ 
    rate = 0.25 := by
sorry

end NUMINAMATH_CALUDE_machine_depletion_rate_l1544_154423


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l1544_154419

theorem greatest_prime_factor_of_4_pow_17_minus_2_pow_29 :
  ∃ (p : ℕ), Prime p ∧ p ∣ (4^17 - 2^29) ∧ ∀ (q : ℕ), Prime q → q ∣ (4^17 - 2^29) → q ≤ p ∧ p = 31 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l1544_154419


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_at_min_mn_l1544_154466

/-- Given that 1/m + 2/n = 1 with m > 0 and n > 0, prove that the eccentricity of the ellipse
    x²/m² + y²/n² = 1 is √3/2 when mn takes its minimum value. -/
theorem ellipse_eccentricity_at_min_mn (m n : ℝ) 
  (h1 : m > 0) (h2 : n > 0) (h3 : 1/m + 2/n = 1) : 
  let e := Real.sqrt (1 - (min m n)^2 / (max m n)^2)
  ∃ (x : ℝ), (x = mn) ∧ (∀ y : ℝ, y = m*n → x ≤ y) → e = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_at_min_mn_l1544_154466


namespace NUMINAMATH_CALUDE_square_root_representation_l1544_154458

theorem square_root_representation (x : ℝ) (h : x = 0.25) :
  ∃ y : ℝ, y > 0 ∧ y^2 = x ∧ (∀ z : ℝ, z^2 = x → z = y ∨ z = -y) :=
by sorry

end NUMINAMATH_CALUDE_square_root_representation_l1544_154458


namespace NUMINAMATH_CALUDE_volume_of_region_l1544_154413

-- Define the region
def Region := {p : ℝ × ℝ × ℝ | 
  let (x, y, z) := p
  (|x - y + z| + |x - y - z| ≤ 12) ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0}

-- State the theorem
theorem volume_of_region : MeasureTheory.volume Region = 108 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_region_l1544_154413


namespace NUMINAMATH_CALUDE_solve_equation_l1544_154431

theorem solve_equation (a : ℚ) (h : 2 * a + 2 * a / 4 = 4) : a = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1544_154431


namespace NUMINAMATH_CALUDE_minimum_a_value_l1544_154445

def set_A : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 4/5}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p | |p.1 - 1| + 2*|p.2 - 2| ≤ a}

theorem minimum_a_value (a : ℝ) (h : set_A ⊆ set_B a) : a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_a_value_l1544_154445


namespace NUMINAMATH_CALUDE_chicken_nuggets_cost_l1544_154464

/-- Calculates the total cost of chicken nuggets including discount and tax -/
def total_cost (nuggets : ℕ) (box_size : ℕ) (box_price : ℚ) (discount_threshold : ℕ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let boxes := nuggets / box_size
  let initial_cost := boxes * box_price
  let discounted_cost := if nuggets ≥ discount_threshold then initial_cost * (1 - discount_rate) else initial_cost
  let total := discounted_cost * (1 + tax_rate)
  total

/-- The problem statement -/
theorem chicken_nuggets_cost :
  total_cost 100 20 4 80 (75/1000) (8/100) = 1998/100 :=
sorry

end NUMINAMATH_CALUDE_chicken_nuggets_cost_l1544_154464


namespace NUMINAMATH_CALUDE_lindys_speed_l1544_154474

/-- Proves that Lindy's speed is 9 feet per second given the problem conditions --/
theorem lindys_speed (initial_distance : ℝ) (jack_speed christina_speed : ℝ) 
  (lindy_distance : ℝ) : ℝ :=
by
  -- Define the given conditions
  have h1 : initial_distance = 240 := by sorry
  have h2 : jack_speed = 5 := by sorry
  have h3 : christina_speed = 3 := by sorry
  have h4 : lindy_distance = 270 := by sorry

  -- Calculate the time it takes for Jack and Christina to meet
  let total_speed := jack_speed + christina_speed
  let time_to_meet := initial_distance / total_speed

  -- Calculate Lindy's speed
  let lindy_speed := lindy_distance / time_to_meet

  -- Prove that Lindy's speed is 9 feet per second
  have h5 : lindy_speed = 9 := by sorry

  exact lindy_speed

end NUMINAMATH_CALUDE_lindys_speed_l1544_154474


namespace NUMINAMATH_CALUDE_standard_pairs_parity_l1544_154490

/-- Represents the color of a square on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard -/
def Chessboard (m n : ℕ) := Fin m → Fin n → Color

/-- Counts the number of standard pairs on the chessboard -/
def count_standard_pairs (board : Chessboard m n) : ℕ := sorry

/-- Counts the number of blue squares on the edges (excluding corners) -/
def count_blue_edges (board : Chessboard m n) : ℕ := sorry

/-- The main theorem: The parity of standard pairs is equivalent to the parity of blue edge squares -/
theorem standard_pairs_parity (m n : ℕ) (h_m : m ≥ 3) (h_n : n ≥ 3) (board : Chessboard m n) :
  Even (count_standard_pairs board) ↔ Even (count_blue_edges board) := by sorry

end NUMINAMATH_CALUDE_standard_pairs_parity_l1544_154490


namespace NUMINAMATH_CALUDE_partnership_investment_time_l1544_154486

/-- Represents the investment and profit scenario of two partners -/
structure PartnershipScenario where
  /-- Ratio of partner p's investment to partner q's investment -/
  investment_ratio_p_q : Rat
  /-- Ratio of partner p's profit to partner q's profit -/
  profit_ratio_p_q : Rat
  /-- Number of months partner p invested -/
  p_investment_time : ℕ
  /-- Number of months partner q invested -/
  q_investment_time : ℕ

/-- Theorem stating the relationship between investment ratios, profit ratios, and investment times -/
theorem partnership_investment_time 
  (scenario : PartnershipScenario) 
  (h1 : scenario.investment_ratio_p_q = 7 / 5)
  (h2 : scenario.profit_ratio_p_q = 7 / 10)
  (h3 : scenario.p_investment_time = 7) :
  scenario.q_investment_time = 14 := by
  sorry

#check partnership_investment_time

end NUMINAMATH_CALUDE_partnership_investment_time_l1544_154486


namespace NUMINAMATH_CALUDE_diagonals_bisect_angles_and_parallel_implies_parallelogram_diagonals_bisect_area_implies_parallelogram_l1544_154467

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define diagonals
def diagonal1 (q : Quadrilateral) : Line := sorry
def diagonal2 (q : Quadrilateral) : Line := sorry

-- Define the property of diagonals bisecting each other's interior angles
def diagonals_bisect_angles (q : Quadrilateral) : Prop := sorry

-- Define the property of diagonals being parallel
def diagonals_parallel (q : Quadrilateral) : Prop := sorry

-- Define the property of diagonals bisecting the area
def diagonals_bisect_area (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem diagonals_bisect_angles_and_parallel_implies_parallelogram 
  (q : Quadrilateral) (h1 : diagonals_bisect_angles q) (h2 : diagonals_parallel q) : 
  is_parallelogram q := sorry

-- Theorem 2
theorem diagonals_bisect_area_implies_parallelogram 
  (q : Quadrilateral) (h : diagonals_bisect_area q) : 
  is_parallelogram q := sorry

end NUMINAMATH_CALUDE_diagonals_bisect_angles_and_parallel_implies_parallelogram_diagonals_bisect_area_implies_parallelogram_l1544_154467


namespace NUMINAMATH_CALUDE_expression_simplification_l1544_154447

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 3 / y) :
  (3 * x - 3 / x) * (3 * y + 3 / y) = 9 * x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1544_154447


namespace NUMINAMATH_CALUDE_jerry_logs_count_l1544_154489

/-- The number of logs produced by a pine tree -/
def logsPerPine : ℕ := 80

/-- The number of logs produced by a maple tree -/
def logsPerMaple : ℕ := 60

/-- The number of logs produced by a walnut tree -/
def logsPerWalnut : ℕ := 100

/-- The number of pine trees Jerry cuts -/
def pineTreesCut : ℕ := 8

/-- The number of maple trees Jerry cuts -/
def mapleTreesCut : ℕ := 3

/-- The number of walnut trees Jerry cuts -/
def walnutTreesCut : ℕ := 4

/-- The total number of logs Jerry gets -/
def totalLogs : ℕ := logsPerPine * pineTreesCut + logsPerMaple * mapleTreesCut + logsPerWalnut * walnutTreesCut

theorem jerry_logs_count : totalLogs = 1220 := by sorry

end NUMINAMATH_CALUDE_jerry_logs_count_l1544_154489


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1544_154411

/-- Given a line and a circle with specific properties, prove the minimum value of 1/a + 1/b --/
theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, 2*a*x - b*y + 2 = 0 ∧ (x + 1)^2 + (y - 2)^2 = 4) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    2*a*x₁ - b*y₁ + 2 = 0 ∧ (x₁ + 1)^2 + (y₁ - 2)^2 = 4 ∧
    2*a*x₂ - b*y₂ + 2 = 0 ∧ (x₂ + 1)^2 + (y₂ - 2)^2 = 4 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  (1/a + 1/b) ≥ 2 :=
by sorry


end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1544_154411


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l1544_154416

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  ∃ m : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = m^2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l1544_154416


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1544_154471

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (2 * y + 3) = 1 / 4) : 
  x + 3 * y ≥ 2 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1544_154471


namespace NUMINAMATH_CALUDE_root_problems_l1544_154430

theorem root_problems :
  (∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ x = 2) ∧
  (∃ x y : ℝ, x^2 = 5 ∧ y^2 = 5 ∧ x = -y ∧ x ≠ 0) ∧
  (∃ x : ℝ, x^3 = -27 ∧ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_root_problems_l1544_154430


namespace NUMINAMATH_CALUDE_simplify_expression_l1544_154450

theorem simplify_expression (x : ℝ) : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1544_154450


namespace NUMINAMATH_CALUDE_f_max_min_l1544_154424

-- Define the function
def f (x : ℝ) : ℝ := |x^2 - x| + |x + 1|

-- State the theorem
theorem f_max_min :
  ∃ (max min : ℝ),
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x ≤ max) ∧
    (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = max) ∧
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → min ≤ f x) ∧
    (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = min) ∧
    max = 7 ∧ min = 1 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_l1544_154424


namespace NUMINAMATH_CALUDE_power_of_2016_expression_evaluation_l1544_154487

-- Part 1
theorem power_of_2016 (m n : ℕ) (h1 : 3^m = 4) (h2 : 3^(m+4*n) = 324) : 
  2016^n = 2016 := by sorry

-- Part 2
theorem expression_evaluation (a : ℝ) (h : a = 5) : 
  (a+2)*(a-2) + a*(1-a) = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_2016_expression_evaluation_l1544_154487


namespace NUMINAMATH_CALUDE_next_sales_amount_l1544_154477

theorem next_sales_amount (initial_sales : ℝ) (initial_royalties : ℝ) (next_royalties : ℝ) (decrease_ratio : ℝ) :
  initial_sales = 20000000 →
  initial_royalties = 8000000 →
  next_royalties = 9000000 →
  decrease_ratio = 0.7916666666666667 →
  ∃ (next_sales : ℝ),
    next_sales = 108000000 ∧
    (next_royalties / next_sales) = (initial_royalties / initial_sales) * (1 - decrease_ratio) :=
by sorry

end NUMINAMATH_CALUDE_next_sales_amount_l1544_154477


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l1544_154492

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x - 4

-- Theorem statement
theorem f_strictly_increasing :
  (∀ x y, x < y ∧ x < -2/3 → f x < f y) ∧
  (∀ x y, x < y ∧ 2 < x → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l1544_154492


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1544_154484

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ n : ℤ, (d * n^3 + c * n^2 + b * n + a) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1544_154484


namespace NUMINAMATH_CALUDE_max_bookshelves_l1544_154473

def room_space : ℕ := 400
def shelf_space : ℕ := 80
def reserved_space : ℕ := 160

theorem max_bookshelves : 
  (room_space - reserved_space) / shelf_space = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_bookshelves_l1544_154473


namespace NUMINAMATH_CALUDE_set_union_problem_l1544_154497

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem set_union_problem (x : ℝ) :
  (∃ y, A y ∩ B y = {9}) →
  (∃ z, A z ∪ B z = {-4, -7, -8, 4, 9}) :=
by sorry

end NUMINAMATH_CALUDE_set_union_problem_l1544_154497


namespace NUMINAMATH_CALUDE_problem_statement_l1544_154420

theorem problem_statement :
  (∃ (x₁ x₂ x₃ x₄ x₅ : ℚ), x₁ < 0 ∧ x₂ < 0 ∧ x₃ < 0 ∧ x₄ * x₅ > 0 ∧ x₁ * x₂ * x₃ * x₄ * x₅ < 0) ∧
  (∀ m : ℝ, abs m + m = 0 → m ≤ 0) ∧
  (∃ a b : ℝ, 1 / a < 1 / b ∧ (a < b ∨ a > b)) ∧
  (∀ a : ℝ, 5 - abs (a - 5) ≤ 5) ∧ (∃ a : ℝ, 5 - abs (a - 5) = 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1544_154420


namespace NUMINAMATH_CALUDE_sqrt_twelve_div_sqrt_two_eq_sqrt_six_l1544_154456

theorem sqrt_twelve_div_sqrt_two_eq_sqrt_six :
  Real.sqrt 12 / Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_div_sqrt_two_eq_sqrt_six_l1544_154456


namespace NUMINAMATH_CALUDE_tom_catches_sixteen_trout_l1544_154457

/-- The number of trout Melanie catches -/
def melanie_trout : ℕ := 8

/-- Tom catches twice as many trout as Melanie -/
def tom_multiplier : ℕ := 2

/-- The number of trout Tom catches -/
def tom_trout : ℕ := tom_multiplier * melanie_trout

theorem tom_catches_sixteen_trout : tom_trout = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_catches_sixteen_trout_l1544_154457


namespace NUMINAMATH_CALUDE_total_distance_not_unique_l1544_154407

/-- Represents a part of a journey with a specific speed -/
structure JourneyPart where
  speed : ℝ
  time : ℝ

/-- Represents a complete journey -/
structure Journey where
  parts : List JourneyPart
  totalTime : ℝ

/-- Calculates the distance of a journey part -/
def distanceOfPart (part : JourneyPart) : ℝ :=
  part.speed * part.time

/-- Calculates the total distance of a journey -/
def totalDistance (journey : Journey) : ℝ :=
  (journey.parts.map distanceOfPart).sum

/-- Theorem stating that the total distance cannot be uniquely determined -/
theorem total_distance_not_unique (totalTime : ℝ) (speeds : List ℝ) :
  ∃ (j1 j2 : Journey), 
    j1.totalTime = totalTime ∧ 
    j2.totalTime = totalTime ∧ 
    (j1.parts.map (·.speed)) = speeds ∧ 
    (j2.parts.map (·.speed)) = speeds ∧ 
    totalDistance j1 ≠ totalDistance j2 := by
  sorry

#check total_distance_not_unique

end NUMINAMATH_CALUDE_total_distance_not_unique_l1544_154407


namespace NUMINAMATH_CALUDE_range_of_m_l1544_154482

theorem range_of_m (x y z : ℝ) (h1 : 6 * x = 3 * y + 12) (h2 : 6 * x = 2 * z) 
  (h3 : y ≥ 0) (h4 : z ≤ 9) : 
  let m := 2 * x + y - 3 * z
  ∀ m', m = m' → -19 ≤ m' ∧ m' ≤ -14 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1544_154482


namespace NUMINAMATH_CALUDE_a_range_l1544_154498

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a^2 / (4 * x)

noncomputable def g (x : ℝ) : ℝ := x - log x

theorem a_range (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ (x₁ x₂ : ℝ), 1 ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧ 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → f a x₁ ≥ g x₂) : 
  a ≥ 2 * sqrt (Real.exp 1 - 2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l1544_154498


namespace NUMINAMATH_CALUDE_intersection_range_l1544_154417

-- Define the line equation
def line_equation (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for intersection
def intersects_hyperbola (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  hyperbola_equation x₁ (line_equation k x₁) ∧
  hyperbola_equation x₂ (line_equation k x₂) ∧
  x₁ * x₂ < 0  -- Ensures points are on different branches

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, intersects_hyperbola k ↔ -1 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l1544_154417


namespace NUMINAMATH_CALUDE_exists_monochromatic_isosceles_triangle_l1544_154400

-- Define a color type
inductive Color
  | Red
  | Green
  | Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define an isosceles triangle
def isIsoscelesTriangle (p q r : Point) : Prop := sorry

-- Theorem statement
theorem exists_monochromatic_isosceles_triangle :
  ∃ (p q r : Point), 
    isIsoscelesTriangle p q r ∧ 
    coloring p = coloring q ∧ 
    coloring q = coloring r := 
by sorry

end NUMINAMATH_CALUDE_exists_monochromatic_isosceles_triangle_l1544_154400


namespace NUMINAMATH_CALUDE_min_sum_squares_l1544_154465

theorem min_sum_squares (a b c d e f g h : ℤ) : 
  a ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  b ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  c ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  d ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  e ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  f ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  g ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  h ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g → a ≠ h →
  b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g → b ≠ h →
  c ≠ d → c ≠ e → c ≠ f → c ≠ g → c ≠ h →
  d ≠ e → d ≠ f → d ≠ g → d ≠ h →
  e ≠ f → e ≠ g → e ≠ h →
  f ≠ g → f ≠ h →
  g ≠ h →
  34 ≤ (a + b + c + d)^2 + (e + f + g + h)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1544_154465


namespace NUMINAMATH_CALUDE_special_trapezoid_angle_l1544_154414

/-- A trapezoid with special properties -/
structure SpecialTrapezoid where
  /-- The diagonals intersect at a right angle -/
  diagonals_right_angle : Bool
  /-- One diagonal is equal to the midsegment -/
  diagonal_equals_midsegment : Bool

/-- The angle formed by the special diagonal and the bases of the trapezoid -/
def diagonal_base_angle (t : SpecialTrapezoid) : Real :=
  sorry

/-- Theorem: In a special trapezoid, the angle between the special diagonal and the bases is 60° -/
theorem special_trapezoid_angle (t : SpecialTrapezoid) 
  (h1 : t.diagonals_right_angle = true) 
  (h2 : t.diagonal_equals_midsegment = true) : 
  diagonal_base_angle t = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_angle_l1544_154414


namespace NUMINAMATH_CALUDE_third_segment_less_than_quarter_l1544_154452

open Real

/-- Given a triangle ABC with angles A, B, C, and side lengths a, b, c, 
    where angle B is divided into four equal parts, prove that the third segment 
    on AC (counting from A) is less than |AC| / 4 -/
theorem third_segment_less_than_quarter (A B C : ℝ) (a b c : ℝ) : 
  A > 0 → B > 0 → C > 0 → 
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  3 * A - C < π →
  ∃ (K L M : ℝ), 0 < K ∧ K < L ∧ L < M ∧ M < b ∧
    (L - K = M - L) ∧ (M - L = b - M) ∧
    (L - K < b / 4) :=
by sorry

end NUMINAMATH_CALUDE_third_segment_less_than_quarter_l1544_154452


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1544_154435

theorem polynomial_factorization (x : ℤ) : 
  x^4 + 3*x^3 - 15*x^2 - 19*x + 30 = (x+2)*(x+5)*(x-1)*(x-3) :=
by
  sorry

#check polynomial_factorization

end NUMINAMATH_CALUDE_polynomial_factorization_l1544_154435


namespace NUMINAMATH_CALUDE_complex_product_real_l1544_154499

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := x - 2*I
  (z₁ * z₂).im = 0 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l1544_154499


namespace NUMINAMATH_CALUDE_max_wrong_questions_l1544_154485

theorem max_wrong_questions (total_questions : Nat) (success_percentage : Rat) 
  (h1 : total_questions = 50)
  (h2 : success_percentage = 75 / 100) :
  ∃ (max_wrong : Nat), 
    (max_wrong ≤ total_questions) ∧ 
    ((total_questions - max_wrong : Rat) / total_questions ≥ success_percentage) ∧
    (∀ (n : Nat), n > max_wrong → (total_questions - n : Rat) / total_questions < success_percentage) ∧
    max_wrong = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_wrong_questions_l1544_154485


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l1544_154463

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let R := Real.sqrt ((a^2 + b^2 + c^2) / 4)
  4 * Real.pi * R^2 = 50 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l1544_154463


namespace NUMINAMATH_CALUDE_number_calculation_l1544_154401

theorem number_calculation (x : ℝ) : ((x + 1.4) / 3 - 0.7) * 9 = 5.4 ↔ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l1544_154401


namespace NUMINAMATH_CALUDE_max_value_of_z_minus_i_l1544_154468

theorem max_value_of_z_minus_i (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - Complex.I) ≤ 2 ∧ ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_z_minus_i_l1544_154468


namespace NUMINAMATH_CALUDE_absolute_value_plus_power_minus_sqrt_l1544_154462

theorem absolute_value_plus_power_minus_sqrt : |-2| + 2023^0 - Real.sqrt 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_power_minus_sqrt_l1544_154462


namespace NUMINAMATH_CALUDE_candy_purchase_sum_l1544_154440

/-- A sequence of daily candy purchases where each day's purchase is one more than the previous day -/
def candy_sequence (first_day : ℕ) : ℕ → ℕ :=
  fun n => first_day + n - 1

theorem candy_purchase_sum (first_day : ℕ) 
  (h : candy_sequence first_day 0 + candy_sequence first_day 1 + candy_sequence first_day 2 = 504) :
  candy_sequence first_day 3 + candy_sequence first_day 4 + candy_sequence first_day 5 = 513 := by
  sorry

end NUMINAMATH_CALUDE_candy_purchase_sum_l1544_154440


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l1544_154425

theorem cubic_equation_sum (p q r : ℝ) : 
  (p^3 - 6*p^2 + 11*p = 14) → 
  (q^3 - 6*q^2 + 11*q = 14) → 
  (r^3 - 6*r^2 + 11*r = 14) → 
  (p*q/r + q*r/p + r*p/q = -47/14) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l1544_154425


namespace NUMINAMATH_CALUDE_curves_intersect_once_l1544_154491

/-- Two curves intersect at exactly one point -/
def intersect_once (f g : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = g x

/-- The first curve -/
def curve1 (b : ℝ) (x : ℝ) : ℝ := b * x^2 - 2 * x + 5

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := 3 * x + 4

/-- The theorem stating the condition for the curves to intersect at exactly one point -/
theorem curves_intersect_once :
  ∀ b : ℝ, intersect_once (curve1 b) curve2 ↔ b = 25/4 := by sorry

end NUMINAMATH_CALUDE_curves_intersect_once_l1544_154491


namespace NUMINAMATH_CALUDE_probability_three_green_apples_l1544_154449

theorem probability_three_green_apples (total_apples green_apples selected_apples : ℕ) :
  total_apples = 10 →
  green_apples = 4 →
  selected_apples = 3 →
  (Nat.choose green_apples selected_apples : ℚ) / (Nat.choose total_apples selected_apples) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_green_apples_l1544_154449


namespace NUMINAMATH_CALUDE_arithmetic_mean_square_difference_l1544_154403

theorem arithmetic_mean_square_difference (p u v : ℕ) : 
  Nat.Prime p → 
  u ≠ v → 
  u > 0 → 
  v > 0 → 
  p * p = (u * u + v * v) / 2 → 
  ∃ (x : ℕ), (2 * p - u - v = x * x) ∨ (2 * p - u - v = 2 * x * x) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_square_difference_l1544_154403


namespace NUMINAMATH_CALUDE_candy_bar_total_cost_l1544_154472

/-- The cost of a candy bar in dollars -/
def candy_bar_cost : ℕ := 3

/-- The number of candy bars bought -/
def number_of_candy_bars : ℕ := 2

/-- The total cost of candy bars -/
def total_cost : ℕ := candy_bar_cost * number_of_candy_bars

theorem candy_bar_total_cost : total_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_total_cost_l1544_154472


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_804_l1544_154437

/-- 
  Represents a pentagon formed by cutting a triangular corner from a rectangle.
  The sides of the pentagon have lengths 12, 15, 18, 30, and 34 in some order.
-/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {12, 15, 18, 30, 34}

/-- The area of the CornerCutPentagon is 804. -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ℕ :=
  804

/-- Proves that the area of the CornerCutPentagon is indeed 804. -/
theorem corner_cut_pentagon_area_is_804 (p : CornerCutPentagon) : 
  corner_cut_pentagon_area p = 804 := by
  sorry

#check corner_cut_pentagon_area_is_804

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_804_l1544_154437


namespace NUMINAMATH_CALUDE_area_enclosed_by_functions_l1544_154434

/-- The area enclosed by y = x and f(x) = 2 - x^2 -/
theorem area_enclosed_by_functions : ∃ (a : ℝ), a = (9 : ℝ) / 2 ∧ 
  a = ∫ x in (-2 : ℝ)..1, (2 - x^2 - x) := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_functions_l1544_154434


namespace NUMINAMATH_CALUDE_infinite_solutions_implies_c_equals_three_l1544_154454

theorem infinite_solutions_implies_c_equals_three :
  (∀ (c : ℝ), (∃ (S : Set ℝ), Set.Infinite S ∧ 
    ∀ (y : ℝ), y ∈ S → (3 * (5 + 2 * c * y) = 18 * y + 15))) →
  c = 3 :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_implies_c_equals_three_l1544_154454


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l1544_154480

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 11010₂ -/
def a : Nat := binary_to_nat [true, true, false, true, false]

/-- Represents the binary number 11100₂ -/
def b : Nat := binary_to_nat [true, true, true, false, false]

/-- Represents the binary number 100₂ -/
def c : Nat := binary_to_nat [true, false, false]

/-- Represents the binary number 10101101₂ -/
def result : Nat := binary_to_nat [true, false, true, false, true, true, false, true]

/-- Theorem stating that 11010₂ × 11100₂ ÷ 100₂ = 10101101₂ -/
theorem binary_multiplication_division :
  a * b / c = result := by sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l1544_154480


namespace NUMINAMATH_CALUDE_factor_72x3_minus_252x7_l1544_154475

theorem factor_72x3_minus_252x7 (x : ℝ) : 72 * x^3 - 252 * x^7 = 36 * x^3 * (2 - 7 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_72x3_minus_252x7_l1544_154475


namespace NUMINAMATH_CALUDE_apple_sale_percentage_l1544_154426

theorem apple_sale_percentage (total_apples : ℝ) (first_batch_percentage : ℝ) 
  (first_batch_profit : ℝ) (second_batch_profit : ℝ) (total_profit : ℝ) :
  first_batch_percentage > 0 ∧ first_batch_percentage < 100 →
  first_batch_profit = second_batch_profit →
  first_batch_profit = total_profit →
  (100 - first_batch_percentage) = (100 - first_batch_percentage) := by
sorry

end NUMINAMATH_CALUDE_apple_sale_percentage_l1544_154426


namespace NUMINAMATH_CALUDE_total_pets_is_108_l1544_154459

/-- The total number of pets owned by Teddy, Ben, and Dave -/
def totalPets : ℕ :=
  let teddy_initial_dogs : ℕ := 7
  let teddy_initial_cats : ℕ := 8
  let teddy_initial_rabbits : ℕ := 6
  let teddy_adopted_dogs : ℕ := 2
  let teddy_adopted_rabbits : ℕ := 4
  
  let teddy_final_dogs : ℕ := teddy_initial_dogs + teddy_adopted_dogs
  let teddy_final_cats : ℕ := teddy_initial_cats
  let teddy_final_rabbits : ℕ := teddy_initial_rabbits + teddy_adopted_rabbits
  
  let ben_dogs : ℕ := 3 * teddy_initial_dogs
  let ben_cats : ℕ := 2 * teddy_final_cats
  
  let dave_dogs : ℕ := teddy_final_dogs - 4
  let dave_cats : ℕ := teddy_final_cats + 13
  let dave_rabbits : ℕ := 3 * teddy_initial_rabbits
  
  let teddy_total : ℕ := teddy_final_dogs + teddy_final_cats + teddy_final_rabbits
  let ben_total : ℕ := ben_dogs + ben_cats
  let dave_total : ℕ := dave_dogs + dave_cats + dave_rabbits
  
  teddy_total + ben_total + dave_total

theorem total_pets_is_108 : totalPets = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_108_l1544_154459


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1544_154444

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem seventh_term_of_geometric_sequence 
  (a r : ℝ) 
  (h_positive : ∀ n, geometric_sequence a r n > 0)
  (h_fifth : geometric_sequence a r 5 = 16)
  (h_ninth : geometric_sequence a r 9 = 2) :
  geometric_sequence a r 7 = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1544_154444


namespace NUMINAMATH_CALUDE_max_tulips_is_15_l1544_154451

/-- Represents the cost of yellow and red tulips in rubles -/
structure TulipCosts where
  yellow : ℕ
  red : ℕ

/-- Represents the number of yellow and red tulips in the bouquet -/
structure Bouquet where
  yellow : ℕ
  red : ℕ

/-- Calculates the total cost of a bouquet given the costs of tulips -/
def totalCost (b : Bouquet) (c : TulipCosts) : ℕ :=
  b.yellow * c.yellow + b.red * c.red

/-- Checks if a bouquet satisfies the conditions -/
def isValidBouquet (b : Bouquet) : Prop :=
  (b.yellow + b.red) % 2 = 1 ∧ 
  (b.yellow = b.red + 1 ∨ b.red = b.yellow + 1)

/-- The maximum number of tulips in the bouquet -/
def maxTulips : ℕ := 15

/-- The theorem stating that 15 is the maximum number of tulips -/
theorem max_tulips_is_15 (c : TulipCosts) 
    (h1 : c.yellow = 50) 
    (h2 : c.red = 31) : 
    (∀ b : Bouquet, isValidBouquet b → totalCost b c ≤ 600 → b.yellow + b.red ≤ maxTulips) ∧
    (∃ b : Bouquet, isValidBouquet b ∧ totalCost b c ≤ 600 ∧ b.yellow + b.red = maxTulips) :=
  sorry

end NUMINAMATH_CALUDE_max_tulips_is_15_l1544_154451


namespace NUMINAMATH_CALUDE_square_difference_equality_l1544_154479

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1544_154479


namespace NUMINAMATH_CALUDE_largest_number_l1544_154443

theorem largest_number : 
  let numbers : List ℝ := [0.978, 0.9719, 0.9781, 0.917, 0.9189]
  ∀ x ∈ numbers, x ≤ 0.9781 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1544_154443


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1544_154438

theorem arithmetic_computation : 8 + 6 * (3 - 8)^2 = 158 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1544_154438


namespace NUMINAMATH_CALUDE_stratified_sampling_calculation_l1544_154402

/-- Stratified sampling calculation -/
theorem stratified_sampling_calculation 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (stratum_size : ℕ) 
  (h1 : total_population = 2000) 
  (h2 : sample_size = 200) 
  (h3 : stratum_size = 250) : 
  (stratum_size : ℚ) / total_population * sample_size = 25 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_calculation_l1544_154402


namespace NUMINAMATH_CALUDE_max_value_of_f_l1544_154409

theorem max_value_of_f (α : ℝ) :
  ∃ M : ℝ, M = (Real.sqrt 2 + 1) / 2 ∧
  (∀ x : ℝ, 1 - Real.sin (x + α)^2 + Real.cos (x + α) * Real.sin (x + α) ≤ M) ∧
  (∃ x : ℝ, 1 - Real.sin (x + α)^2 + Real.cos (x + α) * Real.sin (x + α) = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1544_154409


namespace NUMINAMATH_CALUDE_angle_sum_bounds_l1544_154453

theorem angle_sum_bounds (x y z : Real) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (hz : 0 < z ∧ z < π/2) 
  (h : Real.cos x ^ 2 + Real.cos y ^ 2 + Real.cos z ^ 2 = 1) : 
  3 * π / 4 < x + y + z ∧ x + y + z < π := by
sorry

end NUMINAMATH_CALUDE_angle_sum_bounds_l1544_154453


namespace NUMINAMATH_CALUDE_jane_usable_sheets_l1544_154448

/-- Represents the total number of sheets Jane has for each type and size --/
structure TotalSheets where
  brownA4 : ℕ
  yellowA4 : ℕ
  yellowA3 : ℕ

/-- Represents the number of damaged sheets (less than 70% intact) for each type and size --/
structure DamagedSheets where
  brownA4 : ℕ
  yellowA4 : ℕ
  yellowA3 : ℕ

/-- Calculates the number of usable sheets given the total and damaged sheets --/
def usableSheets (total : TotalSheets) (damaged : DamagedSheets) : ℕ :=
  (total.brownA4 - damaged.brownA4) + (total.yellowA4 - damaged.yellowA4) + (total.yellowA3 - damaged.yellowA3)

theorem jane_usable_sheets :
  let total := TotalSheets.mk 28 18 9
  let damaged := DamagedSheets.mk 3 5 2
  usableSheets total damaged = 45 := by
  sorry

end NUMINAMATH_CALUDE_jane_usable_sheets_l1544_154448


namespace NUMINAMATH_CALUDE_point_above_x_axis_l1544_154436

theorem point_above_x_axis (a : ℝ) : 
  (a > 0) → (a = Real.sqrt 3) → ∃ (x y : ℝ), x = -2 ∧ y = a ∧ y > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_above_x_axis_l1544_154436


namespace NUMINAMATH_CALUDE_abs_ratio_sum_l1544_154422

theorem abs_ratio_sum (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a| / a + |b| / b : ℚ) = 2 ∨ (|a| / a + |b| / b : ℚ) = -2 ∨ (|a| / a + |b| / b : ℚ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_abs_ratio_sum_l1544_154422


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_6_with_digit_sum_15_l1544_154429

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_6_with_digit_sum_15 :
  ∀ n : ℕ, is_three_digit n → n % 6 = 0 → digit_sum n = 15 → n ≤ 690 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_6_with_digit_sum_15_l1544_154429


namespace NUMINAMATH_CALUDE_function_composition_property_l1544_154415

def iteratedFunction (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, n => n
| (i + 1), n => f (iteratedFunction f i n)

theorem function_composition_property (k : ℕ) :
  (k ≥ 2) ↔
  (∃ (f g : ℕ → ℕ),
    (∀ (S : Set ℕ), (∃ n, g n ∉ S) → Set.Infinite S) ∧
    (∀ n, iteratedFunction f (g n) n = f n + k)) :=
sorry

end NUMINAMATH_CALUDE_function_composition_property_l1544_154415


namespace NUMINAMATH_CALUDE_soccer_tournament_equation_l1544_154421

theorem soccer_tournament_equation (x : ℕ) (h : x > 1) : 
  (x.choose 2 = 28) ↔ (x * (x - 1) / 2 = 28) := by sorry

end NUMINAMATH_CALUDE_soccer_tournament_equation_l1544_154421


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1544_154442

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ b : ℝ, (m^2 + Complex.I) * (1 + m * Complex.I) = Complex.I * b) → m = 0 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1544_154442


namespace NUMINAMATH_CALUDE_locus_of_centers_l1544_154408

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₃ with equation (x - 3)² + y² = 25 -/
def C₃ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₃ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₃ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

/-- The locus of centers (a,b) of circles externally tangent to C₁ and internally tangent to C₃ -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₃ a b r) → 
  12 * a^2 + 16 * b^2 - 36 * a - 81 = 0 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1544_154408


namespace NUMINAMATH_CALUDE_speed_conversion_l1544_154404

/-- Proves that a speed of 36.003 km/h is equivalent to 10.0008 meters per second. -/
theorem speed_conversion (speed_kmh : ℝ) (speed_ms : ℝ) : 
  speed_kmh = 36.003 ∧ speed_ms = 10.0008 → speed_kmh * (1000 / 3600) = speed_ms := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l1544_154404


namespace NUMINAMATH_CALUDE_polynomial_square_l1544_154410

theorem polynomial_square (a b : ℚ) : 
  (∃ p q : ℚ, ∀ x : ℚ, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + p*x + q)^2) → 
  b = 25/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_square_l1544_154410


namespace NUMINAMATH_CALUDE_weight_loss_days_l1544_154495

/-- The number of days it takes to lose a given amount of weight, given daily calorie intake, burn rate, and calories needed to lose one pound. -/
def days_to_lose_weight (calories_eaten : ℕ) (calories_burned : ℕ) (calories_per_pound : ℕ) (pounds_to_lose : ℕ) : ℕ :=
  let daily_deficit := calories_burned - calories_eaten
  let days_per_pound := calories_per_pound / daily_deficit
  days_per_pound * pounds_to_lose

/-- Theorem stating that it takes 80 days to lose 10 pounds under given conditions -/
theorem weight_loss_days : days_to_lose_weight 1800 2300 4000 10 = 80 := by
  sorry

#eval days_to_lose_weight 1800 2300 4000 10

end NUMINAMATH_CALUDE_weight_loss_days_l1544_154495
