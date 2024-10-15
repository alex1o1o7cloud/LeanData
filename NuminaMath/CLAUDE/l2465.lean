import Mathlib

namespace NUMINAMATH_CALUDE_average_difference_l2465_246534

def num_students : ℕ := 120
def num_teachers : ℕ := 6
def class_sizes : List ℕ := [40, 35, 25, 10, 5, 5]

def t : ℚ := (List.sum class_sizes) / num_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes)) / num_students

theorem average_difference : t - s = -10 := by sorry

end NUMINAMATH_CALUDE_average_difference_l2465_246534


namespace NUMINAMATH_CALUDE_taqeeshas_grade_l2465_246514

theorem taqeeshas_grade (total_students : ℕ) (students_present : ℕ) (initial_average : ℕ) (final_average : ℕ) :
  total_students = 17 →
  students_present = 16 →
  initial_average = 77 →
  final_average = 78 →
  (students_present * initial_average + (total_students - students_present) * 94) / total_students = final_average :=
by sorry

end NUMINAMATH_CALUDE_taqeeshas_grade_l2465_246514


namespace NUMINAMATH_CALUDE_vector_equation_l2465_246528

/-- Given vectors a, b, and c in ℝ², prove that if a = (5, 2), b = (-4, -3), 
    and 3a - 2b + c = 0, then c = (-23, -12). -/
theorem vector_equation (a b c : ℝ × ℝ) : 
  a = (5, 2) → 
  b = (-4, -3) → 
  3 • a - 2 • b + c = (0, 0) → 
  c = (-23, -12) := by sorry

end NUMINAMATH_CALUDE_vector_equation_l2465_246528


namespace NUMINAMATH_CALUDE_combined_teaching_years_l2465_246526

/-- The combined teaching years of Mr. Spencer and Mrs. Randall -/
theorem combined_teaching_years : 
  let spencer_fourth_grade : ℕ := 12
  let spencer_first_grade : ℕ := 5
  let randall_third_grade : ℕ := 18
  let randall_second_grade : ℕ := 8
  (spencer_fourth_grade + spencer_first_grade + randall_third_grade + randall_second_grade) = 43 := by
  sorry

end NUMINAMATH_CALUDE_combined_teaching_years_l2465_246526


namespace NUMINAMATH_CALUDE_average_of_first_20_multiples_of_17_l2465_246532

theorem average_of_first_20_multiples_of_17 : 
  let n : ℕ := 20
  let first_multiple : ℕ := 17
  let sum_of_multiples : ℕ := n * (first_multiple + n * first_multiple) / 2
  (sum_of_multiples : ℚ) / n = 178.5 := by sorry

end NUMINAMATH_CALUDE_average_of_first_20_multiples_of_17_l2465_246532


namespace NUMINAMATH_CALUDE_parabola_focus_lines_range_l2465_246520

/-- A parabola with equation y^2 = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A line passing through the focus of the parabola -/
structure FocusLine (para : Parabola) where
  k : ℝ  -- slope of the line

/-- Intersection points of a focus line with the parabola -/
def intersection_points (para : Parabola) (line : FocusLine para) : ℝ × ℝ := sorry

/-- Distance between intersection points -/
def distance (para : Parabola) (line : FocusLine para) : ℝ := sorry

/-- Number of focus lines with a specific intersection distance -/
def num_lines_with_distance (para : Parabola) (d : ℝ) : ℕ := sorry

theorem parabola_focus_lines_range (para : Parabola) :
  (num_lines_with_distance para 4 = 2) → (0 < para.p ∧ para.p < 2) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_lines_range_l2465_246520


namespace NUMINAMATH_CALUDE_container_capacity_l2465_246507

theorem container_capacity : 
  ∀ (capacity : ℝ), 
  (1/4 : ℝ) * capacity + 120 = (2/3 : ℝ) * capacity → 
  capacity = 288 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l2465_246507


namespace NUMINAMATH_CALUDE_smallest_angle_quadrilateral_l2465_246505

theorem smallest_angle_quadrilateral (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a + b + c + d = 360) →
  (b = 5/4 * a) → (c = 3/2 * a) → (d = 7/4 * a) →
  a = 720 / 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_quadrilateral_l2465_246505


namespace NUMINAMATH_CALUDE_different_types_of_players_l2465_246593

/-- Represents the types of players in the game. -/
inductive PlayerType
  | Cricket
  | Hockey
  | Football
  | Softball

/-- The number of players for each type. -/
def num_players (t : PlayerType) : ℕ :=
  match t with
  | .Cricket => 12
  | .Hockey => 17
  | .Football => 11
  | .Softball => 10

/-- The total number of players on the ground. -/
def total_players : ℕ := 50

/-- The list of all player types. -/
def all_player_types : List PlayerType :=
  [PlayerType.Cricket, PlayerType.Hockey, PlayerType.Football, PlayerType.Softball]

theorem different_types_of_players :
  (List.length all_player_types = 4) ∧
  (List.sum (List.map num_players all_player_types) = total_players) := by
  sorry

end NUMINAMATH_CALUDE_different_types_of_players_l2465_246593


namespace NUMINAMATH_CALUDE_apple_delivery_proof_l2465_246584

/-- Represents the number of apples delivered by the truck -/
def apples_delivered : ℕ → ℕ → ℕ → ℕ
  | initial_green, initial_red, final_green_excess =>
    final_green_excess + initial_red - initial_green

theorem apple_delivery_proof :
  let initial_green := 32
  let initial_red := initial_green + 200
  let final_green_excess := 140
  apples_delivered initial_green initial_red final_green_excess = 340 := by
sorry

#eval apples_delivered 32 232 140

end NUMINAMATH_CALUDE_apple_delivery_proof_l2465_246584


namespace NUMINAMATH_CALUDE_initial_bench_weight_l2465_246549

/-- Represents the weightlifting scenario for John --/
structure WeightliftingScenario where
  initialSquat : ℝ
  initialDeadlift : ℝ
  squatLossPercentage : ℝ
  deadliftLoss : ℝ
  newTotal : ℝ

/-- Calculates the initial bench weight given the weightlifting scenario --/
def calculateInitialBench (scenario : WeightliftingScenario) : ℝ :=
  scenario.newTotal - 
  (scenario.initialSquat * (1 - scenario.squatLossPercentage)) - 
  (scenario.initialDeadlift - scenario.deadliftLoss)

/-- Theorem stating that the initial bench weight is 400 pounds --/
theorem initial_bench_weight (scenario : WeightliftingScenario) 
  (h1 : scenario.initialSquat = 700)
  (h2 : scenario.initialDeadlift = 800)
  (h3 : scenario.squatLossPercentage = 0.3)
  (h4 : scenario.deadliftLoss = 200)
  (h5 : scenario.newTotal = 1490) :
  calculateInitialBench scenario = 400 := by
  sorry


end NUMINAMATH_CALUDE_initial_bench_weight_l2465_246549


namespace NUMINAMATH_CALUDE_tan_inequality_l2465_246515

-- Define the constants and their properties
axiom α : Real
axiom β : Real
axiom k : Int

-- Define the conditions
axiom sin_inequality : Real.sin α > Real.sin β
axiom α_not_right_angle : ∀ k, α ≠ k * Real.pi + Real.pi / 2
axiom β_not_right_angle : ∀ k, β ≠ k * Real.pi + Real.pi / 2
axiom fourth_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi ∧ 3 * Real.pi / 2 < β ∧ β < 2 * Real.pi

-- State the theorem
theorem tan_inequality : Real.tan α > Real.tan β := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l2465_246515


namespace NUMINAMATH_CALUDE_equation_roots_range_l2465_246511

theorem equation_roots_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
    2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
    |x₁ - 2*n| = k * Real.sqrt x₁ ∧
    |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_range_l2465_246511


namespace NUMINAMATH_CALUDE_square_diagonal_triangle_dimensions_l2465_246560

theorem square_diagonal_triangle_dimensions :
  ∀ (square_side : ℝ) (triangle_leg1 triangle_leg2 triangle_hypotenuse : ℝ),
    square_side = 10 →
    triangle_leg1 = square_side →
    triangle_leg2 = square_side →
    triangle_hypotenuse^2 = triangle_leg1^2 + triangle_leg2^2 →
    (triangle_leg1 = 10 ∧ triangle_leg2 = 10 ∧ triangle_hypotenuse = 10 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_triangle_dimensions_l2465_246560


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2465_246521

/-- Two-dimensional vector type -/
def Vector2D := ℝ × ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Perpendicularity of two 2D vectors -/
def perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors (k : ℝ) :
  let a : Vector2D := (2, 1)
  let b : Vector2D := (-1, k)
  perpendicular a b → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2465_246521


namespace NUMINAMATH_CALUDE_sum_x_coordinates_q3_l2465_246596

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Creates a new polygon from the midpoints of the sides of a given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Calculates the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

/-- The main theorem -/
theorem sum_x_coordinates_q3 (q1 : Polygon) 
  (h1 : q1.vertices.length = 45)
  (h2 : sumXCoordinates q1 = 135) :
  let q2 := midpointPolygon q1
  let q3 := midpointPolygon q2
  sumXCoordinates q3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_q3_l2465_246596


namespace NUMINAMATH_CALUDE_distribution_6_boxes_8_floors_l2465_246567

/-- The number of ways to distribute boxes among floors with at least two on the top floor -/
def distributionWays (numBoxes numFloors : ℕ) : ℕ :=
  numFloors^numBoxes - (numFloors - 1)^numBoxes - numBoxes * (numFloors - 1)^(numBoxes - 1)

/-- Theorem: For 6 boxes and 8 floors, the number of distributions with at least 2 boxes on the top floor -/
theorem distribution_6_boxes_8_floors :
  distributionWays 6 8 = 8^6 - 13 * 7^5 := by
  sorry

end NUMINAMATH_CALUDE_distribution_6_boxes_8_floors_l2465_246567


namespace NUMINAMATH_CALUDE_jim_savings_rate_l2465_246508

/-- 
Given:
- Sara has already saved 4100 dollars
- Sara saves 10 dollars per week
- Jim saves x dollars per week
- After 820 weeks, Sara and Jim have saved the same amount

Prove: x = 15
-/

theorem jim_savings_rate (x : ℚ) : 
  (4100 + 820 * 10 = 820 * x) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_jim_savings_rate_l2465_246508


namespace NUMINAMATH_CALUDE_remainder_theorem_l2465_246542

theorem remainder_theorem (m : ℤ) (k : ℤ) : 
  m = 40 * k - 1 → (m^2 + 3*m + 5) % 40 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2465_246542


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l2465_246553

/-- The curve y = x^3 + 2x has a tangent line at (1, 3) perpendicular to ax - y + 2019 = 0 -/
theorem tangent_perpendicular_to_line (a : ℝ) : 
  let f (x : ℝ) := x^3 + 2*x
  let f' (x : ℝ) := 3*x^2 + 2
  let tangent_slope := f' 1
  let perpendicular_slope := a
  (f 1 = 3) ∧ (tangent_slope * perpendicular_slope = -1) → a = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l2465_246553


namespace NUMINAMATH_CALUDE_divisibility_property_l2465_246516

theorem divisibility_property (a n p : ℕ) : 
  a ≥ 2 → 
  n ≥ 1 → 
  Nat.Prime p → 
  p ∣ (a^(2^n) + 1) → 
  2^(n+1) ∣ (p-1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2465_246516


namespace NUMINAMATH_CALUDE_g_continuous_c_plus_d_equals_negative_three_l2465_246555

-- Define the piecewise function g(x)
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then c * x + 2
  else if -3 ≤ x ∧ x ≤ 1 then 2 * x - 4
  else 3 * x - d

-- Theorem stating the continuity condition
theorem g_continuous (c d : ℝ) :
  (∀ x, ContinuousAt (g c d) x) ↔ c = -4 ∧ d = 1 := by
  sorry

-- Corollary for the sum of c and d
theorem c_plus_d_equals_negative_three (c d : ℝ) :
  (∀ x, ContinuousAt (g c d) x) → c + d = -3 := by
  sorry

end NUMINAMATH_CALUDE_g_continuous_c_plus_d_equals_negative_three_l2465_246555


namespace NUMINAMATH_CALUDE_gcd_g_x_l2465_246572

def g (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(12*x+7)*(3*x+10)

theorem gcd_g_x (x : ℤ) (h : 46800 ∣ x) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l2465_246572


namespace NUMINAMATH_CALUDE_bisection_interval_valid_l2465_246512

/-- The function f(x) = x^3 + 5 -/
def f (x : ℝ) : ℝ := x^3 + 5

/-- Theorem stating that [-2, 1] is a valid initial interval for the bisection method -/
theorem bisection_interval_valid :
  f (-2) * f 1 < 0 := by sorry

end NUMINAMATH_CALUDE_bisection_interval_valid_l2465_246512


namespace NUMINAMATH_CALUDE_limit_at_two_l2465_246590

/-- The limit of (3x^2 - 5x - 2) / (x - 2) as x approaches 2 is 7 -/
theorem limit_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 2 →
    0 < |x - 2| ∧ |x - 2| < δ →
    |((3 * x^2 - 5 * x - 2) / (x - 2)) - 7| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_at_two_l2465_246590


namespace NUMINAMATH_CALUDE_inequality_solution_l2465_246580

theorem inequality_solution :
  ∀ x : ℝ, (x / 2 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Ici (-6) ∩ Set.Iio (-3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2465_246580


namespace NUMINAMATH_CALUDE_dilation_image_l2465_246575

def dilation (center : ℂ) (scale : ℝ) (point : ℂ) : ℂ :=
  center + scale * (point - center)

theorem dilation_image : 
  let center : ℂ := -1 + 2*I
  let scale : ℝ := 2
  let point : ℂ := 3 + 4*I
  dilation center scale point = 7 + 6*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_image_l2465_246575


namespace NUMINAMATH_CALUDE_quadratic_decreasing_interval_l2465_246545

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_decreasing_interval 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : quadratic_function a b c (-5) = 0) 
  (h3 : quadratic_function a b c 3 = 0) :
  ∀ x ∈ Set.Iic (-1), 
    ∀ y ∈ Set.Iic (-1), 
      x < y → quadratic_function a b c x > quadratic_function a b c y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_interval_l2465_246545


namespace NUMINAMATH_CALUDE_complex_power_sum_l2465_246581

theorem complex_power_sum : 
  let i : ℂ := Complex.I
  ((1 + i) / 2) ^ 8 + ((1 - i) / 2) ^ 8 = (1 : ℂ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2465_246581


namespace NUMINAMATH_CALUDE_hens_count_l2465_246536

/-- Given a total number of heads and feet, and the number of feet for hens and cows,
    calculate the number of hens. -/
def count_hens (total_heads : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) : ℕ :=
  sorry

theorem hens_count :
  let total_heads := 44
  let total_feet := 140
  let hen_feet := 2
  let cow_feet := 4
  count_hens total_heads total_feet hen_feet cow_feet = 18 := by
  sorry

end NUMINAMATH_CALUDE_hens_count_l2465_246536


namespace NUMINAMATH_CALUDE_committee_meeting_attendance_l2465_246538

/-- Given a committee meeting with only associate and assistant professors, where:
    - Each associate professor brings 2 pencils and 1 chart
    - Each assistant professor brings 1 pencil and 2 charts
    - A total of 10 pencils and 11 charts are brought to the meeting
    Prove that the total number of people present is 7. -/
theorem committee_meeting_attendance :
  ∀ (associate_profs assistant_profs : ℕ),
    2 * associate_profs + assistant_profs = 10 →
    associate_profs + 2 * assistant_profs = 11 →
    associate_profs + assistant_profs = 7 :=
by sorry

end NUMINAMATH_CALUDE_committee_meeting_attendance_l2465_246538


namespace NUMINAMATH_CALUDE_max_choir_members_correct_l2465_246535

/-- The maximum number of choir members that satisfies the given conditions. -/
def max_choir_members : ℕ := 266

/-- Predicate to check if a number satisfies the square formation condition. -/
def is_square_formation (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k + 11

/-- Predicate to check if a number satisfies the rectangular formation condition. -/
def is_rectangular_formation (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n * (n + 5)

/-- Theorem stating that max_choir_members satisfies both formation conditions
    and is the maximum number that does so. -/
theorem max_choir_members_correct :
  is_square_formation max_choir_members ∧
  is_rectangular_formation max_choir_members ∧
  ∀ m : ℕ, m > max_choir_members →
    ¬(is_square_formation m ∧ is_rectangular_formation m) :=
by sorry

end NUMINAMATH_CALUDE_max_choir_members_correct_l2465_246535


namespace NUMINAMATH_CALUDE_optimal_sapling_positions_l2465_246566

/-- Represents the number of trees planted -/
def num_trees : ℕ := 20

/-- Represents the distance between adjacent trees in meters -/
def tree_spacing : ℕ := 10

/-- Calculates the total distance walked by students for given sapling positions -/
def total_distance (pos1 pos2 : ℕ) : ℕ := sorry

/-- Theorem stating that positions 10 and 11 minimize the total distance -/
theorem optimal_sapling_positions :
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ num_trees →
    total_distance 10 11 ≤ total_distance a b :=
by sorry

end NUMINAMATH_CALUDE_optimal_sapling_positions_l2465_246566


namespace NUMINAMATH_CALUDE_prob_A_leading_after_three_prob_B_wins_3_2_l2465_246569

-- Define the probability of Team A winning a single game
def p_A_win : ℝ := 0.60

-- Define the probability of Team B winning a single game
def p_B_win : ℝ := 1 - p_A_win

-- Define the number of games needed to win the match
def games_to_win : ℕ := 3

-- Define the total number of games in a full match
def total_games : ℕ := 5

-- Theorem for the probability of Team A leading after the first three games
theorem prob_A_leading_after_three : 
  (Finset.sum (Finset.range 2) (λ k => Nat.choose 3 (3 - k) * p_A_win ^ (3 - k) * p_B_win ^ k)) = 0.648 := by sorry

-- Theorem for the probability of Team B winning the match with a score of 3:2
theorem prob_B_wins_3_2 : 
  (Nat.choose 4 2 * p_A_win ^ 2 * p_B_win ^ 2 * p_B_win) = 0.138 := by sorry

end NUMINAMATH_CALUDE_prob_A_leading_after_three_prob_B_wins_3_2_l2465_246569


namespace NUMINAMATH_CALUDE_circle_area_after_folding_l2465_246501

theorem circle_area_after_folding (original_area : ℝ) (sector_area : ℝ) : 
  sector_area = 5 → original_area / 64 = sector_area → original_area = 320 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_after_folding_l2465_246501


namespace NUMINAMATH_CALUDE_interval_intersection_l2465_246527

theorem interval_intersection (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1/2 < x ∧ x < 0.6) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l2465_246527


namespace NUMINAMATH_CALUDE_ed_lost_marbles_ed_lost_eleven_marbles_l2465_246583

theorem ed_lost_marbles (doug : ℕ) : ℕ :=
  let ed_initial := doug + 19
  let ed_final := doug + 8
  ed_initial - ed_final

theorem ed_lost_eleven_marbles (doug : ℕ) :
  ed_lost_marbles doug = 11 := by
  sorry

end NUMINAMATH_CALUDE_ed_lost_marbles_ed_lost_eleven_marbles_l2465_246583


namespace NUMINAMATH_CALUDE_teapot_sale_cost_comparison_l2465_246576

/-- Represents the cost calculation for promotional methods in a teapot and teacup sale. -/
structure TeapotSale where
  teapot_price : ℝ
  teacup_price : ℝ
  discount_rate : ℝ
  teapots_bought : ℕ
  min_teacups : ℕ

/-- Calculates the cost under promotional method 1 (buy 1 teapot, get 1 teacup free) -/
def cost_method1 (sale : TeapotSale) (x : ℕ) : ℝ :=
  sale.teapot_price * sale.teapots_bought + sale.teacup_price * (x - sale.teapots_bought)

/-- Calculates the cost under promotional method 2 (9.2% discount on total price) -/
def cost_method2 (sale : TeapotSale) (x : ℕ) : ℝ :=
  (sale.teapot_price * sale.teapots_bought + sale.teacup_price * x) * (1 - sale.discount_rate)

/-- Theorem stating the relationship between costs of two promotional methods -/
theorem teapot_sale_cost_comparison (sale : TeapotSale)
    (h_teapot : sale.teapot_price = 20)
    (h_teacup : sale.teacup_price = 5)
    (h_discount : sale.discount_rate = 0.092)
    (h_teapots : sale.teapots_bought = 4)
    (h_min_teacups : sale.min_teacups = 4) :
    ∀ x : ℕ, x ≥ sale.min_teacups →
      (cost_method1 sale x < cost_method2 sale x ↔ x < 34) ∧
      (cost_method1 sale x = cost_method2 sale x ↔ x = 34) ∧
      (cost_method1 sale x > cost_method2 sale x ↔ x > 34) := by
  sorry


end NUMINAMATH_CALUDE_teapot_sale_cost_comparison_l2465_246576


namespace NUMINAMATH_CALUDE_fourth_root_sum_equals_expression_l2465_246541

theorem fourth_root_sum_equals_expression : 
  (1 + Real.sqrt 2 + Real.sqrt 3)^4 = 
    Real.sqrt 6400 + Real.sqrt 6144 + Real.sqrt 4800 + Real.sqrt 4608 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sum_equals_expression_l2465_246541


namespace NUMINAMATH_CALUDE_sqrt_sum_irrational_l2465_246503

theorem sqrt_sum_irrational (n : ℕ+) : Irrational (Real.sqrt (n + 1) + Real.sqrt n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_irrational_l2465_246503


namespace NUMINAMATH_CALUDE_yard_length_is_250_l2465_246504

/-- The length of a yard with trees planted at equal distances -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of the yard is 250 meters -/
theorem yard_length_is_250 :
  yard_length 51 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_is_250_l2465_246504


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2465_246574

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2465_246574


namespace NUMINAMATH_CALUDE_divisible_by_27_l2465_246518

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A natural number is three times the sum of its digits -/
def is_three_times_sum_of_digits (n : ℕ) : Prop :=
  n = 3 * sum_of_digits n

theorem divisible_by_27 (n : ℕ) (h : is_three_times_sum_of_digits n) : 
  27 ∣ n := by sorry

end NUMINAMATH_CALUDE_divisible_by_27_l2465_246518


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l2465_246561

-- Problem 1
theorem problem_one : (Real.sqrt 12 - Real.sqrt 6) / Real.sqrt 3 + 2 / Real.sqrt 2 = 2 := by sorry

-- Problem 2
theorem problem_two : (2 + Real.sqrt 3) * (2 - Real.sqrt 3) + (2 - Real.sqrt 3)^2 = 8 - 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l2465_246561


namespace NUMINAMATH_CALUDE_cone_distance_theorem_l2465_246550

/-- Represents a right circular cone -/
structure RightCircularCone where
  slantHeight : ℝ
  topRadius : ℝ

/-- The shortest distance between two points on a cone's surface -/
def shortestDistance (cone : RightCircularCone) (pointA pointB : ℝ × ℝ) : ℝ := sorry

theorem cone_distance_theorem (cone : RightCircularCone) 
  (h1 : cone.slantHeight = 21)
  (h2 : cone.topRadius = 14) :
  let midpoint : ℝ × ℝ := (cone.slantHeight / 2, 0)
  let oppositePoint : ℝ × ℝ := (cone.slantHeight / 2, cone.topRadius)
  Int.floor (shortestDistance cone midpoint oppositePoint) = 18 := by sorry

end NUMINAMATH_CALUDE_cone_distance_theorem_l2465_246550


namespace NUMINAMATH_CALUDE_omega_range_l2465_246582

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃ a b : ℝ, π ≤ a ∧ a < b ∧ b ≤ 2*π ∧ Real.sin (ω*a) + Real.sin (ω*b) = 2) →
  (9/4 ≤ ω ∧ ω < 5/2) ∨ (13/4 ≤ ω) :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l2465_246582


namespace NUMINAMATH_CALUDE_school_population_l2465_246573

theorem school_population (b g t a : ℕ) : 
  b = 4 * g ∧ 
  g = 8 * t ∧ 
  t = 2 * a → 
  b + g + t + a = 83 * a :=
by sorry

end NUMINAMATH_CALUDE_school_population_l2465_246573


namespace NUMINAMATH_CALUDE_a_greater_equal_two_l2465_246577

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

theorem a_greater_equal_two (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a (-1) ≤ f a x) ∧
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ f a 1) →
  a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_a_greater_equal_two_l2465_246577


namespace NUMINAMATH_CALUDE_greg_trousers_bought_l2465_246570

/-- The cost of a shirt -/
def shirt_cost : ℝ := sorry

/-- The cost of a trouser -/
def trouser_cost : ℝ := sorry

/-- The cost of a tie -/
def tie_cost : ℝ := sorry

/-- The number of trousers bought in the first scenario -/
def trousers_bought : ℕ := sorry

theorem greg_trousers_bought : 
  (3 * shirt_cost + trousers_bought * trouser_cost + 2 * tie_cost = 90) ∧
  (7 * shirt_cost + 2 * trouser_cost + 2 * tie_cost = 50) ∧
  (5 * shirt_cost + 3 * trouser_cost + 2 * tie_cost = 70) →
  trousers_bought = 4 := by
sorry

end NUMINAMATH_CALUDE_greg_trousers_bought_l2465_246570


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l2465_246598

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) (angle : ℝ) :
  cube_side = 5 →
  cylinder_radius = 1 →
  angle = Real.pi / 4 →
  ∃ (remaining_volume : ℝ),
    remaining_volume = cube_side^3 - cylinder_radius^2 * Real.pi * (cube_side * Real.sqrt 2) ∧
    remaining_volume = 125 - 5 * Real.sqrt 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l2465_246598


namespace NUMINAMATH_CALUDE_root_implies_m_values_l2465_246533

theorem root_implies_m_values (m : ℝ) : 
  ((m + 2) * 1^2 - 2 * 1 + m^2 - 2 * m - 6 = 0) → (m = -2 ∨ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_root_implies_m_values_l2465_246533


namespace NUMINAMATH_CALUDE_beams_per_panel_is_two_l2465_246517

/-- Represents the number of fence panels in the fence -/
def num_panels : ℕ := 10

/-- Represents the number of metal sheets in each fence panel -/
def sheets_per_panel : ℕ := 3

/-- Represents the number of metal rods in each sheet -/
def rods_per_sheet : ℕ := 10

/-- Represents the number of metal rods in each beam -/
def rods_per_beam : ℕ := 4

/-- Represents the total number of metal rods needed for the fence -/
def total_rods : ℕ := 380

/-- Calculates the number of metal beams in each fence panel -/
def beams_per_panel : ℕ := 
  let total_sheets := num_panels * sheets_per_panel
  let rods_for_sheets := total_sheets * rods_per_sheet
  let remaining_rods := total_rods - rods_for_sheets
  let total_beams := remaining_rods / rods_per_beam
  total_beams / num_panels

/-- Theorem stating that the number of metal beams in each fence panel is 2 -/
theorem beams_per_panel_is_two : beams_per_panel = 2 := by sorry

end NUMINAMATH_CALUDE_beams_per_panel_is_two_l2465_246517


namespace NUMINAMATH_CALUDE_rahim_average_price_per_book_l2465_246557

/-- The average price per book given two purchases -/
def average_price_per_book (books1 : ℕ) (cost1 : ℕ) (books2 : ℕ) (cost2 : ℕ) : ℚ :=
  (cost1 + cost2) / (books1 + books2)

/-- Theorem: The average price per book for Rahim's purchases is 85 -/
theorem rahim_average_price_per_book :
  average_price_per_book 65 6500 35 2000 = 85 := by
  sorry

end NUMINAMATH_CALUDE_rahim_average_price_per_book_l2465_246557


namespace NUMINAMATH_CALUDE_max_x_minus_y_l2465_246589

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l2465_246589


namespace NUMINAMATH_CALUDE_min_omega_for_max_values_l2465_246579

theorem min_omega_for_max_values (ω : ℝ) (h1 : ω > 0) :
  (∀ x ∈ Set.Icc 0 1, ∃ (n : ℕ), n ≥ 50 ∧ 
    (∀ y ∈ Set.Icc 0 1, Real.sin (ω * x) ≥ Real.sin (ω * y))) →
  ω ≥ 197 * Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_for_max_values_l2465_246579


namespace NUMINAMATH_CALUDE_cube_root_2450_l2465_246506

theorem cube_root_2450 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (2450 : ℝ)^(1/3) = a * b^(1/3) ∧ 
  (∀ (c d : ℕ), c > 0 → d > 0 → (2450 : ℝ)^(1/3) = c * d^(1/3) → d ≥ b) ∧
  a = 35 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_2450_l2465_246506


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2465_246565

theorem margin_in_terms_of_selling_price
  (C S M n : ℝ)
  (h1 : M = (2 / n) * C)
  (h2 : S - M = C)
  : M = 2 * S / (n + 2) := by
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2465_246565


namespace NUMINAMATH_CALUDE_dress_design_combinations_l2465_246546

theorem dress_design_combinations (colors patterns : ℕ) (h1 : colors = 5) (h2 : patterns = 6) : colors * patterns = 30 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_combinations_l2465_246546


namespace NUMINAMATH_CALUDE_expression_values_l2465_246540

theorem expression_values : 
  (0.64^(-1/2) - (-1/8)^0 + 8^(2/3) + (9/16)^(1/2) = 6) ∧ 
  (Real.log 2^2 + Real.log 2 * Real.log 5 + Real.log 5 = 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l2465_246540


namespace NUMINAMATH_CALUDE_julia_puppy_cost_l2465_246556

def adoption_fee : ℝ := 20
def dog_food : ℝ := 20
def treats_price : ℝ := 2.5
def treats_quantity : ℕ := 2
def toys : ℝ := 15
def crate : ℝ := 20
def bed : ℝ := 20
def collar_leash : ℝ := 15
def discount_rate : ℝ := 0.2

def total_cost : ℝ :=
  adoption_fee +
  (1 - discount_rate) * (dog_food + treats_price * treats_quantity + toys + crate + bed + collar_leash)

theorem julia_puppy_cost :
  total_cost = 96 := by sorry

end NUMINAMATH_CALUDE_julia_puppy_cost_l2465_246556


namespace NUMINAMATH_CALUDE_selling_price_example_l2465_246599

/-- Calculates the selling price of an article given the gain and gain percentage. -/
def selling_price (gain : ℚ) (gain_percentage : ℚ) : ℚ :=
  let cost_price := gain / (gain_percentage / 100)
  cost_price + gain

/-- Theorem stating that given a gain of $75 and a gain percentage of 50%, 
    the selling price of an article is $225. -/
theorem selling_price_example : selling_price 75 50 = 225 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_example_l2465_246599


namespace NUMINAMATH_CALUDE_line_intercept_ratio_l2465_246519

theorem line_intercept_ratio (b : ℝ) (s t : ℝ) 
  (h_b : b ≠ 0)
  (h_s : 0 = 10 * s + b)
  (h_t : 0 = 6 * t + b) :
  s / t = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_line_intercept_ratio_l2465_246519


namespace NUMINAMATH_CALUDE_mollys_age_l2465_246525

/-- Given the ratio of Sandy's age to Molly's age and Sandy's future age, 
    prove Molly's current age -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  sandy_age / molly_age = 4 / 3 →
  sandy_age + 6 = 38 →
  molly_age = 24 := by
  sorry

#check mollys_age

end NUMINAMATH_CALUDE_mollys_age_l2465_246525


namespace NUMINAMATH_CALUDE_function_behavior_l2465_246500

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_increasing : ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 7 → f x ≤ f y)
variable (h_f7 : f 7 = 6)

-- State the theorem
theorem function_behavior :
  (∀ x y, -7 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f y ≤ f x) ∧
  (∀ x, -7 ≤ x ∧ x ≤ 7 → f x ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_function_behavior_l2465_246500


namespace NUMINAMATH_CALUDE_min_colors_is_three_l2465_246530

/-- Represents a coloring of a 5x5 grid -/
def Coloring := Fin 5 → Fin 5 → ℕ

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Fin 5 × Fin 5) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x3 - x1) * (y2 - y1) = (x2 - x1) * (y3 - y1)

/-- Checks if a coloring is valid (no three same-colored points are collinear) -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ p1 p2 p3 : Fin 5 × Fin 5,
    collinear p1 p2 p3 →
    (c p1.1 p1.2 = c p2.1 p2.2 ∧ c p2.1 p2.2 = c p3.1 p3.2) →
    p1 = p2 ∨ p2 = p3 ∨ p3 = p1

/-- The main theorem: the minimum number of colors for a valid coloring is 3 -/
theorem min_colors_is_three :
  (∃ (c : Coloring), valid_coloring c ∧ (∀ i j, c i j < 3)) ∧
  (∀ (c : Coloring), valid_coloring c → ∃ i j, c i j ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_colors_is_three_l2465_246530


namespace NUMINAMATH_CALUDE_socks_difference_l2465_246523

/-- The number of pairs of socks Laticia knitted in the first week -/
def first_week : ℕ := 12

/-- The number of pairs of socks Laticia knitted in the second week -/
def second_week : ℕ := sorry

/-- The number of pairs of socks Laticia knitted in the third week -/
def third_week : ℕ := (first_week + second_week) / 2

/-- The number of pairs of socks Laticia knitted in the fourth week -/
def fourth_week : ℕ := third_week - 3

/-- The total number of pairs of socks Laticia knitted -/
def total_socks : ℕ := 57

theorem socks_difference : 
  first_week + second_week + third_week + fourth_week = total_socks ∧ 
  second_week - first_week = 1 := by sorry

end NUMINAMATH_CALUDE_socks_difference_l2465_246523


namespace NUMINAMATH_CALUDE_intersection_of_P_and_complement_of_M_l2465_246558

-- Define the universal set U as integers
def U : Set Int := Set.univ

-- Define set M
def M : Set Int := {1, 2}

-- Define set P
def P : Set Int := {x : Int | |x| ≤ 2}

-- State the theorem
theorem intersection_of_P_and_complement_of_M :
  P ∩ (U \ M) = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_complement_of_M_l2465_246558


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l2465_246586

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem sum_of_powers_of_i :
  i^1520 + i^1521 + i^1522 + i^1523 + i^1524 = (2 : ℂ) := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l2465_246586


namespace NUMINAMATH_CALUDE_monitor_pixels_l2465_246592

/-- Calculates the total number of pixels on a monitor given its dimensions and resolution. -/
def totalPixels (width : ℕ) (height : ℕ) (dotsPerInch : ℕ) : ℕ :=
  (width * dotsPerInch) * (height * dotsPerInch)

/-- Theorem stating that a 21x12 inch monitor with 100 dots per inch has 2,520,000 pixels. -/
theorem monitor_pixels :
  totalPixels 21 12 100 = 2520000 := by
  sorry

end NUMINAMATH_CALUDE_monitor_pixels_l2465_246592


namespace NUMINAMATH_CALUDE_parabola_focus_specific_parabola_focus_l2465_246529

/-- The focus of a parabola with equation y^2 = ax has coordinates (a/4, 0) -/
theorem parabola_focus (a : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = a * x}
  let focus := (a / 4, 0)
  focus ∈ parabola ∧ 
  ∀ (p : ℝ × ℝ), p ∈ parabola → dist p focus = dist p (0, -a/4) :=
sorry

/-- The focus of the parabola y^2 = 8x has coordinates (2, 0) -/
theorem specific_parabola_focus :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 8 * x}
  let focus := (2, 0)
  focus ∈ parabola ∧ 
  ∀ (p : ℝ × ℝ), p ∈ parabola → dist p focus = dist p (0, -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_specific_parabola_focus_l2465_246529


namespace NUMINAMATH_CALUDE_point_coordinates_l2465_246578

def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

theorem point_coordinates :
  ∀ (p : ℝ × ℝ),
    second_quadrant p →
    distance_to_x_axis p = 3 →
    distance_to_y_axis p = 1 →
    p = (-1, 3) :=
by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2465_246578


namespace NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l2465_246537

/-- Definition of a periodic function -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

/-- Statement: sin(x^2) is not periodic -/
theorem sin_x_squared_not_periodic : ¬ IsPeriodic (fun x ↦ Real.sin (x^2)) := by
  sorry


end NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l2465_246537


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2465_246543

theorem equal_roots_quadratic : ∃ (x : ℝ), x^2 - 2*x + 1 = 0 ∧ 
  ∀ (y : ℝ), y^2 - 2*y + 1 = 0 → y = x :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2465_246543


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_2017_l2465_246554

theorem tens_digit_of_3_to_2017 : ∃ n : ℕ, 3^2017 ≡ 87 + 100*n [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_2017_l2465_246554


namespace NUMINAMATH_CALUDE_peter_investment_duration_l2465_246502

/-- Calculates the final amount after simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + (principal * rate * time)

theorem peter_investment_duration :
  ∀ (rate : ℝ),
  rate > 0 →
  simple_interest 650 rate 3 = 815 →
  simple_interest 650 rate 4 = 870 →
  ∃ (t : ℝ), t = 3 ∧ simple_interest 650 rate t = 815 :=
by sorry

end NUMINAMATH_CALUDE_peter_investment_duration_l2465_246502


namespace NUMINAMATH_CALUDE_mixture_quantity_proof_l2465_246544

theorem mixture_quantity_proof (petrol kerosene diesel : ℝ) 
  (h1 : petrol / kerosene = 3 / 2)
  (h2 : petrol / diesel = 3 / 5)
  (h3 : (petrol - 6) / ((kerosene - 4) + 20) = 2 / 3)
  (h4 : (petrol - 6) / (diesel - 10) = 2 / 5)
  (h5 : petrol + kerosene + diesel > 0) :
  petrol + kerosene + diesel = 100 := by
sorry

end NUMINAMATH_CALUDE_mixture_quantity_proof_l2465_246544


namespace NUMINAMATH_CALUDE_symmetric_hexagon_relationship_l2465_246531

/-- A hexagon that is both inscribed and circumscribed, and symmetric about the perpendicular bisector of one of its sides. -/
structure SymmetricHexagon where
  R : ℝ  -- radius of circumscribed circle
  r : ℝ  -- radius of inscribed circle
  c : ℝ  -- distance between centers of circles
  R_pos : 0 < R
  r_pos : 0 < r
  c_pos : 0 < c
  inscribed : True  -- represents that the hexagon is inscribed
  circumscribed : True  -- represents that the hexagon is circumscribed
  symmetric : True  -- represents that the hexagon is symmetric about the perpendicular bisector of one of its sides

/-- The relationship between R, r, and c for a symmetric hexagon -/
theorem symmetric_hexagon_relationship (h : SymmetricHexagon) :
  3 * (h.R^2 - h.c^2)^4 - 4 * h.r^2 * (h.R^2 - h.c^2)^2 * (h.R^2 + h.c^2) - 16 * h.R^2 * h.c^2 * h.r^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_hexagon_relationship_l2465_246531


namespace NUMINAMATH_CALUDE_range_of_m_l2465_246597

def p (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ m - 2 > 0 ∧ 6 - m > 0 ∧
  ∀ x y : ℝ, x^2 / (m - 2) + y^2 / (6 - m) = 1 ↔ (x / a)^2 + (y / b)^2 = 1

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m > 0

theorem range_of_m :
  ∃ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (1 < m ∧ m ≤ 2) ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2465_246597


namespace NUMINAMATH_CALUDE_sin_75_cos_75_l2465_246551

theorem sin_75_cos_75 : Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_75_l2465_246551


namespace NUMINAMATH_CALUDE_parallel_tangents_sum_l2465_246587

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

theorem parallel_tangents_sum (a : ℝ) (h : a ≥ 3) :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  (deriv (f a)) x₁ = (deriv (f a)) x₂ ∧
  x₁ + x₂ > 6/5 := by sorry

end NUMINAMATH_CALUDE_parallel_tangents_sum_l2465_246587


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l2465_246510

def base_7_to_10 (n : Nat) : Nat :=
  5 * 7^5 + 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0

def base_10_to_4 (n : Nat) : List Nat :=
  let rec convert (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else convert (m / 4) ((m % 4) :: acc)
  convert n []

theorem base_conversion_theorem :
  (base_7_to_10 543210 = 94773) ∧
  (base_10_to_4 94773 = [1, 1, 3, 2, 3, 0, 1, 1]) := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l2465_246510


namespace NUMINAMATH_CALUDE_female_officers_count_l2465_246522

theorem female_officers_count (total_on_duty : ℕ) (female_ratio_on_duty : ℚ) (female_percentage : ℚ) :
  total_on_duty = 100 →
  female_ratio_on_duty = 1/2 →
  female_percentage = 1/5 →
  (female_ratio_on_duty * total_on_duty : ℚ) / female_percentage = 250 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2465_246522


namespace NUMINAMATH_CALUDE_sandwiches_bought_l2465_246595

theorem sandwiches_bought (sandwich_cost : ℝ) (soda_count : ℕ) (soda_cost : ℝ) (total_cost : ℝ)
  (h1 : sandwich_cost = 2.44)
  (h2 : soda_count = 4)
  (h3 : soda_cost = 0.87)
  (h4 : total_cost = 8.36)
  : ∃ (sandwich_count : ℕ), 
    sandwich_count * sandwich_cost + soda_count * soda_cost = total_cost ∧ 
    sandwich_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_bought_l2465_246595


namespace NUMINAMATH_CALUDE_yogurt_combinations_l2465_246588

theorem yogurt_combinations (num_flavors : Nat) (num_toppings : Nat) (num_sizes : Nat) :
  num_flavors = 6 →
  num_toppings = 8 →
  num_sizes = 2 →
  num_flavors * (num_toppings.choose 2) * num_sizes = 336 := by
  sorry

#eval Nat.choose 8 2

end NUMINAMATH_CALUDE_yogurt_combinations_l2465_246588


namespace NUMINAMATH_CALUDE_caitlin_sara_weight_l2465_246509

/-- Given the weights of three people (Annette, Caitlin, and Sara), proves that
    Caitlin and Sara weigh 87 pounds together. -/
theorem caitlin_sara_weight 
  (annette caitlin sara : ℝ) 
  (h1 : annette + caitlin = 95)   -- Annette and Caitlin weigh 95 pounds together
  (h2 : annette = sara + 8) :     -- Annette weighs 8 pounds more than Sara
  caitlin + sara = 87 := by sorry

end NUMINAMATH_CALUDE_caitlin_sara_weight_l2465_246509


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_l2465_246539

theorem cubic_expansion_coefficient (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_l2465_246539


namespace NUMINAMATH_CALUDE_gift_cost_theorem_l2465_246594

/-- Calculates the total cost of gifts for all workers in a company -/
def total_gift_cost (workers_per_block : ℕ) (num_blocks : ℕ) (gift_worth : ℕ) : ℕ :=
  workers_per_block * num_blocks * gift_worth

/-- The total cost of gifts for all workers in the company is $6000 -/
theorem gift_cost_theorem :
  total_gift_cost 200 15 2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_theorem_l2465_246594


namespace NUMINAMATH_CALUDE_water_fraction_after_replacements_l2465_246571

/-- Represents the fraction of water in the radiator mixture -/
def water_fraction (n : ℕ) : ℚ :=
  (3/4 : ℚ) ^ n

/-- The radiator capacity in quarts -/
def radiator_capacity : ℕ := 16

/-- The amount of mixture removed and replaced in each iteration -/
def replacement_amount : ℕ := 4

/-- The number of replacement iterations -/
def num_iterations : ℕ := 4

theorem water_fraction_after_replacements :
  water_fraction num_iterations = 81/256 := by
  sorry

end NUMINAMATH_CALUDE_water_fraction_after_replacements_l2465_246571


namespace NUMINAMATH_CALUDE_largest_abab_divisible_by_14_l2465_246524

/-- Represents a four-digit number of the form abab -/
def IsAbabForm (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * a + b

/-- Checks if a number is the product of a two-digit and a three-digit number -/
def IsProductOfTwoAndThreeDigit (n : ℕ) : Prop :=
  ∃ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 100 ≤ y ∧ y < 1000 ∧ n = x * y

/-- The main theorem stating the largest four-digit number of the form abab
    that is divisible by 14 and a product of two-digit and three-digit numbers -/
theorem largest_abab_divisible_by_14 :
  ∀ A : ℕ,
  IsAbabForm A →
  IsProductOfTwoAndThreeDigit A →
  A % 14 = 0 →
  A ≤ 9898 :=
by sorry

end NUMINAMATH_CALUDE_largest_abab_divisible_by_14_l2465_246524


namespace NUMINAMATH_CALUDE_old_edition_pages_l2465_246564

/-- The number of pages in the new edition of the Geometry book -/
def new_edition_pages : ℕ := 450

/-- The difference between twice the number of pages in the old edition and the new edition -/
def page_difference : ℕ := 230

/-- Theorem stating that the old edition of the Geometry book had 340 pages -/
theorem old_edition_pages : 
  ∃ (x : ℕ), 2 * x - page_difference = new_edition_pages ∧ x = 340 := by
  sorry

end NUMINAMATH_CALUDE_old_edition_pages_l2465_246564


namespace NUMINAMATH_CALUDE_edward_initial_amount_l2465_246591

def initial_amount (books_cost pens_cost remaining : ℕ) : ℕ :=
  books_cost + pens_cost + remaining

theorem edward_initial_amount :
  initial_amount 6 16 19 = 41 :=
by sorry

end NUMINAMATH_CALUDE_edward_initial_amount_l2465_246591


namespace NUMINAMATH_CALUDE_valid_number_difference_l2465_246563

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000000) ∧ (n < 10000000000) ∧ (n % 11 = 0) ∧
  (∀ d : ℕ, d < 10 → (∃! i : ℕ, i < 10 ∧ (n / 10^i) % 10 = d))

def largest_valid_number : ℕ := 9876524130

def smallest_valid_number : ℕ := 1024375869

theorem valid_number_difference :
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_valid_number) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_valid_number) ∧
  (largest_valid_number - smallest_valid_number = 8852148261) :=
sorry

end NUMINAMATH_CALUDE_valid_number_difference_l2465_246563


namespace NUMINAMATH_CALUDE_base_10_to_12_conversion_l2465_246585

/-- Represents a digit in base 12 -/
inductive Base12Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Converts a Base12Digit to its corresponding natural number -/
def Base12Digit.toNat : Base12Digit → Nat
| D0 => 0
| D1 => 1
| D2 => 2
| D3 => 3
| D4 => 4
| D5 => 5
| D6 => 6
| D7 => 7
| D8 => 8
| D9 => 9
| A => 10
| B => 11

/-- Represents a number in base 12 -/
def Base12Number := List Base12Digit

/-- Converts a Base12Number to its corresponding natural number -/
def Base12Number.toNat : Base12Number → Nat
| [] => 0
| d::ds => d.toNat * (12 ^ ds.length) + Base12Number.toNat ds

theorem base_10_to_12_conversion :
  Base12Number.toNat [Base12Digit.B, Base12Digit.D5] = 173 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_12_conversion_l2465_246585


namespace NUMINAMATH_CALUDE_xiaoming_mother_height_l2465_246547

/-- Given Xiaoming's height, stool height, and the difference between Xiaoming on the stool and his mother's height, prove the height of Xiaoming's mother. -/
theorem xiaoming_mother_height 
  (xiaoming_height : ℝ) 
  (stool_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : xiaoming_height = 1.30)
  (h2 : stool_height = 0.4)
  (h3 : height_difference = 0.08)
  (h4 : xiaoming_height + stool_height = height_difference + mother_height) :
  mother_height = 1.62 :=
by
  sorry

#check xiaoming_mother_height

end NUMINAMATH_CALUDE_xiaoming_mother_height_l2465_246547


namespace NUMINAMATH_CALUDE_parallelogram_max_area_l2465_246552

/-- Given a parallelogram with perimeter 60 units and one side three times the length of the other,
    the maximum possible area is 168.75 square units. -/
theorem parallelogram_max_area :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  a = 3 * b →
  2 * a + 2 * b = 60 →
  ∀ (θ : ℝ),
  0 < θ → θ < π →
  a * b * Real.sin θ ≤ 168.75 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_max_area_l2465_246552


namespace NUMINAMATH_CALUDE_ratio_of_divisors_sums_l2465_246548

def N : ℕ := 68 * 68 * 125 * 135

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry
def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_of_odd_divisors N) * 30 = sum_of_even_divisors N :=
sorry

end NUMINAMATH_CALUDE_ratio_of_divisors_sums_l2465_246548


namespace NUMINAMATH_CALUDE_smallest_y_in_triangle_l2465_246568

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_y_in_triangle (A B C x y : ℕ) : 
  A + B + C = 180 →
  isPrime A ∧ isPrime C →
  B ≤ A ∧ B ≤ C →
  2 * x + y = 180 →
  isPrime x →
  (∀ z : ℕ, z < y → ¬(isPrime z ∧ ∃ w : ℕ, isPrime w ∧ 2 * w + z = 180)) →
  y = 101 := by
sorry

end NUMINAMATH_CALUDE_smallest_y_in_triangle_l2465_246568


namespace NUMINAMATH_CALUDE_initial_mask_sets_l2465_246559

/-- The number of mask sets Alicia gave away -/
def given_away : ℕ := 51

/-- The number of mask sets Alicia had left -/
def left : ℕ := 39

/-- The initial number of mask sets in Alicia's collection -/
def initial : ℕ := given_away + left

/-- Theorem stating that the initial number of mask sets is 90 -/
theorem initial_mask_sets : initial = 90 := by
  sorry

end NUMINAMATH_CALUDE_initial_mask_sets_l2465_246559


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2465_246513

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_roots : a 4 > 0 ∧ a 8 > 0 ∧ a 4^2 - 4*a 4 + 3 = 0 ∧ a 8^2 - 4*a 8 + 3 = 0) :
  a 6 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2465_246513


namespace NUMINAMATH_CALUDE_cricketer_stats_l2465_246562

/-- Represents a cricketer's bowling statistics -/
structure BowlingStats where
  wickets : ℕ
  runs : ℕ
  balls : ℕ

/-- Calculates the bowling average (runs per wicket) -/
def bowlingAverage (stats : BowlingStats) : ℚ :=
  stats.runs / stats.wickets

/-- Calculates the strike rate (balls per wicket) -/
def strikeRate (stats : BowlingStats) : ℚ :=
  stats.balls / stats.wickets

/-- Theorem about the cricketer's statistics -/
theorem cricketer_stats 
  (initial : BowlingStats) 
  (current_match : BowlingStats) 
  (new_stats : BowlingStats) :
  initial.wickets ≥ 50 →
  bowlingAverage initial = 124/10 →
  strikeRate initial = 30 →
  current_match.wickets = 5 →
  current_match.runs = 26 →
  bowlingAverage new_stats = bowlingAverage initial - 4/10 →
  strikeRate new_stats = 28 →
  new_stats.wickets = initial.wickets + current_match.wickets →
  new_stats.runs = initial.runs + current_match.runs →
  initial.wickets = 85 ∧ initial.balls = 2550 := by
  sorry


end NUMINAMATH_CALUDE_cricketer_stats_l2465_246562
