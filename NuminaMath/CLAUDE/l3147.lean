import Mathlib

namespace NUMINAMATH_CALUDE_direct_variation_with_constant_value_of_y_at_negative_ten_l3147_314702

/-- A function representing direct variation with an additional constant. -/
def y (k c : ℝ) (x : ℝ) : ℝ := k * x + c

/-- Theorem: If y(5) = 15 and c = 3, then y(-10) = -21 -/
theorem direct_variation_with_constant 
  (k : ℝ) 
  (h1 : y k 3 5 = 15) 
  : y k 3 (-10) = -21 := by
  sorry

/-- Corollary: The value of y when x = -10 and c = 3 is -21 -/
theorem value_of_y_at_negative_ten 
  (k : ℝ) 
  (h1 : y k 3 5 = 15) 
  : ∃ (y_value : ℝ), y k 3 (-10) = y_value ∧ y_value = -21 := by
  sorry

end NUMINAMATH_CALUDE_direct_variation_with_constant_value_of_y_at_negative_ten_l3147_314702


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3147_314714

theorem cube_volume_problem (reference_cube_volume : ℝ) 
  (unknown_cube_surface_area : ℝ) (reference_cube_surface_area : ℝ) :
  reference_cube_volume = 8 →
  unknown_cube_surface_area = 3 * reference_cube_surface_area →
  reference_cube_surface_area = 6 * (reference_cube_volume ^ (1/3)) ^ 2 →
  let unknown_cube_side_length := (unknown_cube_surface_area / 6) ^ (1/2)
  unknown_cube_side_length ^ 3 = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3147_314714


namespace NUMINAMATH_CALUDE_average_salary_raj_roshan_l3147_314700

theorem average_salary_raj_roshan (raj_salary roshan_salary : ℕ) : 
  (raj_salary + roshan_salary + 7000) / 3 = 5000 →
  (raj_salary + roshan_salary) / 2 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_raj_roshan_l3147_314700


namespace NUMINAMATH_CALUDE_max_balls_in_cube_l3147_314798

theorem max_balls_in_cube (cube_side : ℝ) (ball_radius : ℝ) :
  cube_side = 9 →
  ball_radius = 3 →
  ⌊(cube_side^3) / ((4/3) * π * ball_radius^3)⌋ = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_balls_in_cube_l3147_314798


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l3147_314777

/-- Represents a systematic sampling sequence -/
def SystematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => start + i * (total / sampleSize))

/-- The problem statement -/
theorem systematic_sampling_proof (total : ℕ) (sampleSize : ℕ) (start : ℕ) :
  total = 60 →
  sampleSize = 6 →
  start = 3 →
  SystematicSample total sampleSize start = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l3147_314777


namespace NUMINAMATH_CALUDE_iterative_insertion_square_l3147_314712

theorem iterative_insertion_square (n : ℕ) : ∃ m : ℕ, 
  4 * (10^n - 1) / 9 * 10^(n-1) + 8 * (10^(n-1) - 1) / 9 + 9 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_iterative_insertion_square_l3147_314712


namespace NUMINAMATH_CALUDE_prob_A_wins_sixth_game_l3147_314796

/-- Represents a player in the coin tossing game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents the outcome of a single game -/
inductive GameOutcome : Type
| Win : Player → GameOutcome
| Lose : Player → GameOutcome

/-- Represents the state of the game after a certain number of rounds -/
structure GameState :=
  (round : ℕ)
  (last_loser : Player)

/-- The probability of winning a single coin toss -/
def coin_toss_prob : ℚ := 1/2

/-- The probability of a player winning a game given they start first -/
def win_prob_starting (p : Player) : ℚ := coin_toss_prob

/-- The probability of a player winning a game given they start second -/
def win_prob_second (p : Player) : ℚ := 1 - coin_toss_prob

/-- The probability of player A winning the nth game given the initial state -/
def prob_A_wins_nth_game (n : ℕ) (initial_state : GameState) : ℚ :=
  sorry

theorem prob_A_wins_sixth_game :
  prob_A_wins_nth_game 6 ⟨0, Player.B⟩ = 7/30 :=
sorry

end NUMINAMATH_CALUDE_prob_A_wins_sixth_game_l3147_314796


namespace NUMINAMATH_CALUDE_point_coordinates_l3147_314781

/-- A point in the second quadrant with a specific distance from the x-axis -/
def SecondQuadrantPoint (m : ℝ) : Prop :=
  m - 3 < 0 ∧ m + 2 > 0 ∧ |m + 2| = 4

/-- The theorem stating that a point with the given properties has coordinates (-1, 4) -/
theorem point_coordinates (m : ℝ) (h : SecondQuadrantPoint m) : 
  (m - 3 = -1) ∧ (m + 2 = 4) :=
sorry

end NUMINAMATH_CALUDE_point_coordinates_l3147_314781


namespace NUMINAMATH_CALUDE_two_digit_number_ending_with_zero_l3147_314764

/-- A two-digit number -/
structure TwoDigitNumber where
  value : ℕ
  is_two_digit : 10 ≤ value ∧ value ≤ 99

/-- Reverse the digits of a two-digit number -/
def reverse_digits (n : TwoDigitNumber) : ℕ :=
  (n.value % 10) * 10 + (n.value / 10)

/-- Check if a natural number is a perfect fourth power -/
def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^4

theorem two_digit_number_ending_with_zero (N : TwoDigitNumber) :
  (N.value - reverse_digits N > 0) →
  is_perfect_fourth_power (N.value - reverse_digits N) →
  N.value % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_ending_with_zero_l3147_314764


namespace NUMINAMATH_CALUDE_min_sum_squares_with_real_root_l3147_314704

theorem min_sum_squares_with_real_root (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → a^2 + b^2 ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_with_real_root_l3147_314704


namespace NUMINAMATH_CALUDE_pentagon_regularity_l3147_314799

/-- A pentagon is a closed polygon with five vertices and five edges. -/
structure Pentagon where
  vertices : Fin 5 → ℝ × ℝ

/-- A regular pentagon is a pentagon with all sides equal and all angles equal. -/
def IsRegularPentagon (p : Pentagon) : Prop := sorry

/-- Two line segments are parallel if they have the same slope or are both vertical. -/
def AreParallel (seg1 seg2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

/-- The diagonal of a pentagon connecting two non-adjacent vertices. -/
def Diagonal (p : Pentagon) (i j : Fin 5) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The side of a pentagon connecting two adjacent vertices. -/
def Side (p : Pentagon) (i : Fin 5) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Two line segments have equal length. -/
def HaveEqualLength (seg1 seg2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

theorem pentagon_regularity (p : Pentagon) 
  (h1 : HaveEqualLength (Side p 1) (Side p 2))
  (h2 : HaveEqualLength (Side p 2) (Side p 3))
  (h3 : ∀ (i j : Fin 5), i ≠ j → i.val + 1 ≠ j.val → 
    AreParallel (Diagonal p i j) (Side p ((i.val + j.val) % 5))) :
  IsRegularPentagon p := by
  sorry

end NUMINAMATH_CALUDE_pentagon_regularity_l3147_314799


namespace NUMINAMATH_CALUDE_tickets_problem_l3147_314739

/-- The total number of tickets Tate and Peyton have together -/
def total_tickets (tate_initial : ℕ) (tate_additional : ℕ) : ℕ :=
  let tate_total := tate_initial + tate_additional
  let peyton_tickets := tate_total / 2
  tate_total + peyton_tickets

/-- Theorem stating that given the initial conditions, Tate and Peyton have 51 tickets together -/
theorem tickets_problem (tate_initial : ℕ) (tate_additional : ℕ) 
    (h1 : tate_initial = 32) 
    (h2 : tate_additional = 2) : 
  total_tickets tate_initial tate_additional = 51 := by
  sorry

end NUMINAMATH_CALUDE_tickets_problem_l3147_314739


namespace NUMINAMATH_CALUDE_roses_ratio_l3147_314740

/-- Proves that the ratio of roses given to Susan's daughter to the total number of roses in the bouquet is 1:2 -/
theorem roses_ratio (total : ℕ) (vase : ℕ) (daughter : ℕ) : 
  total = 3 * 12 →
  total = vase + daughter →
  vase = 18 →
  12 = (2/3) * vase →
  daughter / total = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_roses_ratio_l3147_314740


namespace NUMINAMATH_CALUDE_total_pure_acid_in_mixture_l3147_314703

def solution1_concentration : ℝ := 0.20
def solution1_volume : ℝ := 8
def solution2_concentration : ℝ := 0.35
def solution2_volume : ℝ := 5

theorem total_pure_acid_in_mixture :
  let pure_acid1 := solution1_concentration * solution1_volume
  let pure_acid2 := solution2_concentration * solution2_volume
  pure_acid1 + pure_acid2 = 3.35 := by sorry

end NUMINAMATH_CALUDE_total_pure_acid_in_mixture_l3147_314703


namespace NUMINAMATH_CALUDE_cos_4theta_value_l3147_314708

/-- If e^(iθ) = (3 - i√2) / 4, then cos 4θ = 121/256 -/
theorem cos_4theta_value (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (3 - Complex.I * Real.sqrt 2) / 4) : 
  Real.cos (4 * θ) = 121 / 256 := by
  sorry

end NUMINAMATH_CALUDE_cos_4theta_value_l3147_314708


namespace NUMINAMATH_CALUDE_faucet_flow_rate_l3147_314717

/-- The flow rate of a faucet given the number of barrels, capacity per barrel, and time to fill --/
def flowRate (numBarrels : ℕ) (capacityPerBarrel : ℚ) (timeToFill : ℚ) : ℚ :=
  (numBarrels : ℚ) * capacityPerBarrel / timeToFill

/-- Theorem stating that the flow rate for the given conditions is 3.5 gallons per minute --/
theorem faucet_flow_rate :
  flowRate 4 7 8 = (7/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_faucet_flow_rate_l3147_314717


namespace NUMINAMATH_CALUDE_ragnar_wood_chopping_l3147_314787

/-- Represents the number of blocks of wood obtained from chopping trees over a period of time. -/
structure WoodChopping where
  trees_per_day : ℕ
  days : ℕ
  total_blocks : ℕ

/-- Calculates the number of blocks of wood obtained from one tree. -/
def blocks_per_tree (w : WoodChopping) : ℚ :=
  w.total_blocks / (w.trees_per_day * w.days)

/-- Theorem stating that given the specific conditions, the number of blocks per tree is 3. -/
theorem ragnar_wood_chopping :
  let w : WoodChopping := { trees_per_day := 2, days := 5, total_blocks := 30 }
  blocks_per_tree w = 3 := by sorry

end NUMINAMATH_CALUDE_ragnar_wood_chopping_l3147_314787


namespace NUMINAMATH_CALUDE_salary_distribution_l3147_314705

/-- Calculates the tax on a given salary --/
def calculate_tax (salary : ℚ) : ℚ :=
  let tax1 := max 0 (min (salary - 1000) 1000) * (1 / 10)
  let tax2 := max 0 (min (salary - 2000) 1000) * (2 / 10)
  let tax3 := max 0 (salary - 3000) * (3 / 10)
  tax1 + tax2 + tax3

/-- Represents the salary distribution problem --/
theorem salary_distribution (total_parts : ℕ) (a_parts b_parts c_parts d_parts : ℕ)
  (d_c_difference : ℚ) (min_wage : ℚ) :
  total_parts = a_parts + b_parts + c_parts + d_parts →
  a_parts = 2 →
  b_parts = 3 →
  c_parts = 4 →
  d_parts = 6 →
  d_c_difference = 700 →
  min_wage = 1000 →
  ∃ (part_value : ℚ),
    let a_salary := max (part_value * a_parts) min_wage
    let b_salary := max (part_value * b_parts) min_wage
    let c_salary := max (part_value * c_parts) min_wage
    let d_salary := max (part_value * d_parts) min_wage
    d_salary - c_salary = d_c_difference ∧
    a_salary ≤ 4000 ∧
    b_salary ≤ 3500 ∧
    c_salary ≤ 4500 ∧
    d_salary ≤ 6000 ∧
    b_salary - calculate_tax b_salary = 1450 :=
sorry

end NUMINAMATH_CALUDE_salary_distribution_l3147_314705


namespace NUMINAMATH_CALUDE_largest_view_angle_point_l3147_314797

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle -/
structure Angle where
  vertex : Point
  side1 : Point
  side2 : Point

/-- Checks if an angle is acute -/
def isAcute (α : Angle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a point is on one side of an angle -/
def isOnSide (p : Point) (α : Angle) : Prop := sorry

/-- Checks if a point is on the other side of an angle -/
def isOnOtherSide (p : Point) (α : Angle) : Prop := sorry

/-- Calculates the angle at which a segment is seen from a point -/
def viewAngle (p : Point) (a b : Point) : ℝ := sorry

/-- States that a point maximizes the view angle of a segment -/
def maximizesViewAngle (c : Point) (a b : Point) (α : Angle) : Prop :=
  ∀ p, isOnOtherSide p α → viewAngle c a b ≥ viewAngle p a b

theorem largest_view_angle_point (α : Angle) (a b c : Point) :
  isAcute α →
  isOnSide a α →
  isOnSide b α →
  isOnOtherSide c α →
  maximizesViewAngle c a b α →
  (distance α.vertex c)^2 = distance α.vertex a * distance α.vertex b := by
  sorry

end NUMINAMATH_CALUDE_largest_view_angle_point_l3147_314797


namespace NUMINAMATH_CALUDE_find_x_value_l3147_314728

theorem find_x_value (x y : ℝ) (h1 : (12 : ℝ)^3 * 6^2 / x = y) (h2 : y = 144) : x = 432 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l3147_314728


namespace NUMINAMATH_CALUDE_unit_digit_14_power_100_l3147_314721

theorem unit_digit_14_power_100 : (14^100) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_14_power_100_l3147_314721


namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l3147_314750

/-- For a quadratic equation x^2 + (2-p)x - p - 3 = 0, 
    the sum of the squares of its roots is minimized when p = 1 -/
theorem min_sum_squares_roots (p : ℝ) : 
  let f : ℝ → ℝ := λ p => p^2 - 2*p + 10
  ∀ q : ℝ, f p ≥ f 1 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l3147_314750


namespace NUMINAMATH_CALUDE_sprinkles_remaining_l3147_314767

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) : 
  initial_cans = 12 → 
  remaining_cans = initial_cans / 2 - 3 → 
  remaining_cans = 3 := by
sorry

end NUMINAMATH_CALUDE_sprinkles_remaining_l3147_314767


namespace NUMINAMATH_CALUDE_min_value_trig_function_l3147_314741

theorem min_value_trig_function (x : ℝ) : 
  Real.sin x ^ 4 + Real.cos x ^ 4 + (1 / Real.cos x) ^ 4 + (1 / Real.sin x) ^ 4 ≥ 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_function_l3147_314741


namespace NUMINAMATH_CALUDE_number_of_children_l3147_314726

theorem number_of_children (bottle_caps_per_child : ℕ) (total_bottle_caps : ℕ) : 
  bottle_caps_per_child = 5 → total_bottle_caps = 45 → 
  ∃ num_children : ℕ, num_children * bottle_caps_per_child = total_bottle_caps ∧ num_children = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l3147_314726


namespace NUMINAMATH_CALUDE_line_equation_theorem_l3147_314786

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the area of the triangle formed by a line and the coordinate axes -/
def triangleArea (l : Line) : ℝ := sorry

/-- Check if a line passes through a given point -/
def passesThrough (l : Line) (x y : ℝ) : Prop := 
  l.a * x + l.b * y + l.c = 0

/-- The main theorem -/
theorem line_equation_theorem (l : Line) :
  triangleArea l = 3 ∧ passesThrough l (-3) 4 →
  (l.a = 2 ∧ l.b = 3 ∧ l.c = -6) ∨ (l.a = 8 ∧ l.b = 3 ∧ l.c = 12) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_theorem_l3147_314786


namespace NUMINAMATH_CALUDE_triangle_angles_theorem_l3147_314795

theorem triangle_angles_theorem (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  A + C = 2 * B ∧ -- Given condition
  Real.tan A * Real.tan C = 2 + Real.sqrt 3 -- Given condition
  →
  ((A = π / 4 ∧ B = π / 3 ∧ C = 5 * π / 12) ∨
   (A = 5 * π / 12 ∧ B = π / 3 ∧ C = π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_theorem_l3147_314795


namespace NUMINAMATH_CALUDE_fraction_equality_l3147_314768

theorem fraction_equality : (7 + 21) / (14 + 42) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3147_314768


namespace NUMINAMATH_CALUDE_special_function_at_50_l3147_314773

/-- A function satisfying f(xy) = xf(y) for all real x and y, and f(1) = 10 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = x * f y) ∧ (f 1 = 10)

/-- Theorem: If f is a special function, then f(50) = 500 -/
theorem special_function_at_50 (f : ℝ → ℝ) (h : special_function f) : f 50 = 500 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_50_l3147_314773


namespace NUMINAMATH_CALUDE_initial_ratio_is_four_to_five_l3147_314766

-- Define the initial number of men and women
variable (M W : ℕ)

-- Define the final number of men and women
def final_men := M + 2
def final_women := 2 * (W - 3)

-- Theorem statement
theorem initial_ratio_is_four_to_five : 
  final_men = 14 ∧ final_women = 24 → M * 5 = W * 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_four_to_five_l3147_314766


namespace NUMINAMATH_CALUDE_harmonic_mean_inequality_l3147_314794

theorem harmonic_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b > 1 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_inequality_l3147_314794


namespace NUMINAMATH_CALUDE_renovation_project_dirt_calculation_l3147_314757

theorem renovation_project_dirt_calculation (total material sand cement : ℚ)
  (h1 : sand = 0.17)
  (h2 : cement = 0.17)
  (h3 : total = 0.67)
  (h4 : material = total - (sand + cement)) :
  material = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_renovation_project_dirt_calculation_l3147_314757


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l3147_314790

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l3147_314790


namespace NUMINAMATH_CALUDE_minjun_height_l3147_314731

/-- Calculates the current height given initial height and growth over two years -/
def current_height (initial : ℝ) (growth_last_year : ℝ) (growth_this_year : ℝ) : ℝ :=
  initial + growth_last_year + growth_this_year

/-- Theorem stating that Minjun's current height is 1.4 meters -/
theorem minjun_height :
  let initial_height : ℝ := 1.1
  let growth_last_year : ℝ := 0.2
  let growth_this_year : ℝ := 1/10
  current_height initial_height growth_last_year growth_this_year = 1.4 := by
  sorry

#eval current_height 1.1 0.2 0.1

end NUMINAMATH_CALUDE_minjun_height_l3147_314731


namespace NUMINAMATH_CALUDE_range_of_a_plus_3b_l3147_314776

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) 
  (h3 : 1 ≤ a - 2*b) (h4 : a - 2*b ≤ 3) : 
  ∃ (x : ℝ), x = a + 3*b ∧ -11/3 ≤ x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_plus_3b_l3147_314776


namespace NUMINAMATH_CALUDE_correct_division_l3147_314792

theorem correct_division (x : ℝ) (h : x / 1.5 = 3.8) : x / 6 = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l3147_314792


namespace NUMINAMATH_CALUDE_recommendation_plans_count_l3147_314746

/-- The number of universities --/
def num_universities : ℕ := 3

/-- The number of students to be recommended --/
def num_students : ℕ := 4

/-- The maximum number of students a university can accept --/
def max_students_per_university : ℕ := 2

/-- The function that calculates the number of recommendation plans --/
noncomputable def num_recommendation_plans : ℕ := sorry

/-- Theorem stating that the number of recommendation plans is 54 --/
theorem recommendation_plans_count : num_recommendation_plans = 54 := by sorry

end NUMINAMATH_CALUDE_recommendation_plans_count_l3147_314746


namespace NUMINAMATH_CALUDE_expression_evaluation_l3147_314779

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -2
  (3*x + 2*y) * (3*x - 2*y) - 5*x*(x - y) - (2*x - y)^2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3147_314779


namespace NUMINAMATH_CALUDE_player_A_wins_l3147_314793

/-- Represents a pile of matches -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Represents a player's move -/
structure Move :=
  (take : Nat)
  (split : Nat)
  (into : Nat × Nat)

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  move.take ∈ state.piles.map Pile.count ∧
  move.split ∈ state.piles.map Pile.count ∧
  move.split ≠ move.take ∧
  move.into.1 > 0 ∧ move.into.2 > 0 ∧
  move.into.1 + move.into.2 = move.split

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { piles := (state.piles.filter (λ p => p.count ≠ move.take ∧ p.count ≠ move.split)) ++
              [Pile.mk move.into.1, Pile.mk move.into.2] }

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Prop :=
  ∀ move, ¬isValidMove state move

/-- Represents the optimal strategy for a player -/
def OptimalStrategy := GameState → Option Move

/-- Theorem: Player A has a winning strategy -/
theorem player_A_wins (initialState : GameState)
  (h : initialState.piles = [Pile.mk 100, Pile.mk 200, Pile.mk 300]) :
  ∃ (strategyA : OptimalStrategy),
    ∀ (strategyB : OptimalStrategy),
      ∃ (finalState : GameState),
        isGameOver finalState ∧
        -- The last move was made by Player B (meaning A wins)
        (∃ (moves : List Move),
          moves.length % 2 = 1 ∧
          finalState = moves.foldl applyMove initialState) :=
sorry

end NUMINAMATH_CALUDE_player_A_wins_l3147_314793


namespace NUMINAMATH_CALUDE_rainfall_ratio_rainfall_ratio_is_three_to_two_l3147_314720

/-- Given the total rainfall over two weeks and the rainfall in the second week,
    calculate the ratio of rainfall in the second week to the first week. -/
theorem rainfall_ratio (total : ℝ) (second_week : ℝ) :
  total = 25 →
  second_week = 15 →
  second_week / (total - second_week) = 3 / 2 := by
  sorry

/-- The ratio of rainfall in the second week to the first week is 3:2. -/
theorem rainfall_ratio_is_three_to_two :
  ∃ (total : ℝ) (second_week : ℝ),
    total = 25 ∧
    second_week = 15 ∧
    second_week / (total - second_week) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_rainfall_ratio_is_three_to_two_l3147_314720


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3147_314785

theorem complex_fraction_sum (a b : ℝ) :
  (a + b * Complex.I : ℂ) = (3 + Complex.I) / (1 - Complex.I) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3147_314785


namespace NUMINAMATH_CALUDE_circle_origin_inside_l3147_314747

theorem circle_origin_inside (m : ℝ) : 
  (∀ x y : ℝ, (x - m)^2 + (y + m)^2 < 4 → x^2 + y^2 = 0) → 
  -Real.sqrt 2 < m ∧ m < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_origin_inside_l3147_314747


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3147_314748

theorem complex_absolute_value (z : ℂ) (h : z = 1 + Complex.I) : 
  Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3147_314748


namespace NUMINAMATH_CALUDE_matt_problem_time_l3147_314713

/-- The time it takes Matt to do a problem without a calculator -/
def time_without_calculator : ℝ := sorry

/-- The time it takes Matt to do a problem with a calculator -/
def time_with_calculator : ℝ := 2

/-- The number of problems in Matt's assignment -/
def number_of_problems : ℕ := 20

/-- The total time saved by using a calculator -/
def time_saved : ℝ := 60

theorem matt_problem_time :
  time_without_calculator = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_matt_problem_time_l3147_314713


namespace NUMINAMATH_CALUDE_inscribed_circle_probability_l3147_314778

/-- Given a right-angled triangle with legs of 5 and 12 steps, 
    the probability that a randomly selected point within the triangle 
    lies within its inscribed circle is 2π/15 -/
theorem inscribed_circle_probability (a b : ℝ) (h1 : a = 5) (h2 : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let r := (a + b - c) / 2
  let triangle_area := a * b / 2
  let circle_area := π * r^2
  circle_area / triangle_area = 2 * π / 15 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_probability_l3147_314778


namespace NUMINAMATH_CALUDE_number_of_mappings_l3147_314725

theorem number_of_mappings (n m : ℕ) : 
  (Finset.univ : Finset (Fin n → Fin m)).card = m ^ n := by
  sorry

end NUMINAMATH_CALUDE_number_of_mappings_l3147_314725


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3147_314769

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms
  h1 : a 1 < 0
  h2 : a 10 + a 15 = a 12
  h3 : ∀ n, a n = a 1 + (n - 1) * d
  h4 : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n m, n < m → seq.a n < seq.a m) ∧
  (∀ n, n ≠ 12 ∧ n ≠ 13 → seq.S 12 ≤ seq.S n ∧ seq.S 13 ≤ seq.S n) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3147_314769


namespace NUMINAMATH_CALUDE_dale_toast_count_l3147_314718

/-- The cost of breakfast for Dale and Andrew -/
def breakfast_cost (toast_price egg_price : ℕ) (dale_toast : ℕ) : Prop :=
  toast_price * dale_toast + 2 * egg_price + toast_price + 2 * egg_price = 15

/-- Theorem stating that Dale had 2 slices of toast -/
theorem dale_toast_count : breakfast_cost 1 3 2 := by sorry

end NUMINAMATH_CALUDE_dale_toast_count_l3147_314718


namespace NUMINAMATH_CALUDE_construction_cost_l3147_314707

/-- The cost of hiring builders to construct houses -/
theorem construction_cost
  (builders_per_floor : ℕ)
  (days_per_floor : ℕ)
  (pay_per_day : ℕ)
  (num_builders : ℕ)
  (num_houses : ℕ)
  (floors_per_house : ℕ)
  (h1 : builders_per_floor = 3)
  (h2 : days_per_floor = 30)
  (h3 : pay_per_day = 100)
  (h4 : num_builders = 6)
  (h5 : num_houses = 5)
  (h6 : floors_per_house = 6) :
  (num_houses * floors_per_house * days_per_floor * pay_per_day * num_builders) / builders_per_floor = 270000 :=
by sorry

end NUMINAMATH_CALUDE_construction_cost_l3147_314707


namespace NUMINAMATH_CALUDE_spherical_sector_volume_equals_cone_volume_l3147_314735

/-- The volume of a spherical sector is equal to the volume of specific cones -/
theorem spherical_sector_volume_equals_cone_volume (R h : ℝ) (h_pos : 0 < h) (R_pos : 0 < R) :
  let V := (2 * Real.pi * R^2 * h) / 3
  (V = (1/3) * Real.pi * R^2 * (2*h)) ∧ 
  (V = (1/3) * Real.pi * (R*Real.sqrt 2)^2 * h) :=
by
  sorry


end NUMINAMATH_CALUDE_spherical_sector_volume_equals_cone_volume_l3147_314735


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l3147_314761

/-- Given an elevator with 6 people and an average weight of 152 lbs, 
    prove that when a new person weighing 145 lbs enters, 
    the new average weight of all 7 people is 151 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℚ) 
  (new_person_weight : ℚ) (new_avg_weight : ℚ) :
  initial_people = 6 →
  initial_avg_weight = 152 →
  new_person_weight = 145 →
  new_avg_weight = (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) →
  new_avg_weight = 151 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l3147_314761


namespace NUMINAMATH_CALUDE_no_solution_system_l3147_314710

/-- Proves that the system of equations 3x - 4y = 10 and 6x - 8y = 12 has no solution -/
theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 10) ∧ (6 * x - 8 * y = 12) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l3147_314710


namespace NUMINAMATH_CALUDE_additional_driving_hours_l3147_314709

/-- The number of hours Carl drives per day before promotion -/
def hours_per_day : ℝ := 2

/-- The number of days in a week -/
def days_per_week : ℝ := 7

/-- The total number of hours Carl drives in two weeks after promotion -/
def total_hours_two_weeks : ℝ := 40

/-- The number of weeks in the given period -/
def num_weeks : ℝ := 2

theorem additional_driving_hours :
  let hours_before := hours_per_day * days_per_week
  let hours_after := total_hours_two_weeks / num_weeks
  hours_after - hours_before = 6 := by sorry

end NUMINAMATH_CALUDE_additional_driving_hours_l3147_314709


namespace NUMINAMATH_CALUDE_investment_growth_l3147_314782

theorem investment_growth (x : ℝ) : 
  (1 + x / 100) * (1 - 30 / 100) = 1 + 11.99999999999999 / 100 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l3147_314782


namespace NUMINAMATH_CALUDE_remainder_sum_l3147_314772

theorem remainder_sum (c d : ℤ) 
  (hc : c % 52 = 48) 
  (hd : d % 87 = 82) : 
  (c + d) % 29 = 22 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l3147_314772


namespace NUMINAMATH_CALUDE_quadratic_root_sum_inverse_cubes_l3147_314760

theorem quadratic_root_sum_inverse_cubes 
  (a b c r s : ℝ) 
  (h1 : a ≠ 0)
  (h2 : a * r^2 + b * r + c = 0)
  (h3 : a * s^2 + b * s + c = 0)
  (h4 : r ≠ s)
  (h5 : a + b + c = 0) :
  1 / r^3 + 1 / s^3 = -b * (b^2 + 3*a^2 + 3*a*b) / (a + b)^3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_inverse_cubes_l3147_314760


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l3147_314729

/-- Represents the number of book types in each category -/
structure BookCategories where
  chinese : Nat
  mathematics : Nat
  liberal_arts : Nat
  english : Nat

/-- Calculates the total number of book types -/
def total_types (bc : BookCategories) : Nat :=
  bc.chinese + bc.mathematics + bc.liberal_arts + bc.english

/-- Calculates the number of books to be sampled from a category -/
def sample_size (category_size : Nat) (total : Nat) (sample : Nat) : Nat :=
  (category_size * sample) / total

theorem stratified_sample_sum (bc : BookCategories) (sample : Nat) :
  let total := total_types bc
  let math_sample := sample_size bc.mathematics total sample
  let liberal_arts_sample := sample_size bc.liberal_arts total sample
  bc.chinese = 20 →
  bc.mathematics = 10 →
  bc.liberal_arts = 40 →
  bc.english = 30 →
  sample = 20 →
  math_sample + liberal_arts_sample = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sum_l3147_314729


namespace NUMINAMATH_CALUDE_gcd_98_63_l3147_314749

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l3147_314749


namespace NUMINAMATH_CALUDE_lakers_win_in_seven_games_l3147_314762

/-- The probability of the Knicks winning a single game -/
def p_knicks_win : ℚ := 3/4

/-- The probability of the Lakers winning a single game -/
def p_lakers_win : ℚ := 1 - p_knicks_win

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The maximum number of games in the series -/
def max_games : ℕ := 7

/-- The number of ways to choose 3 wins from 6 games -/
def ways_to_choose_3_from_6 : ℕ := 20

theorem lakers_win_in_seven_games :
  let p_lakers_win_series := (ways_to_choose_3_from_6 : ℚ) * p_lakers_win^3 * p_knicks_win^3 * p_lakers_win
  p_lakers_win_series = 540/16384 := by sorry

end NUMINAMATH_CALUDE_lakers_win_in_seven_games_l3147_314762


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3147_314715

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon -/
def sum_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

/-- Theorem: Each interior angle of a regular octagon measures 135 degrees -/
theorem regular_octagon_interior_angle :
  (sum_interior_angles octagon_sides) / octagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3147_314715


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l3147_314716

/-- Given the total selling price, profit per meter, and cost price per meter of cloth,
    calculate the number of meters sold. -/
theorem cloth_sale_calculation (total_selling_price profit_per_meter cost_price_per_meter : ℚ) :
  total_selling_price = 10000 ∧ 
  profit_per_meter = 7 ∧ 
  cost_price_per_meter = 118 →
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℚ) = 80 := by
  sorry

#eval (10000 : ℚ) / (118 + 7) -- This should evaluate to 80

end NUMINAMATH_CALUDE_cloth_sale_calculation_l3147_314716


namespace NUMINAMATH_CALUDE_ABCD_requires_16_bits_l3147_314755

/-- Represents a base-16 digit --/
def Hex : Type := Fin 16

/-- Represents a base-16 number with 4 digits --/
def HexNumber := Fin 4 → Hex

/-- Converts a HexNumber to its decimal (base-10) representation --/
def toDecimal (h : HexNumber) : ℕ :=
  (h 0).val * 16^3 + (h 1).val * 16^2 + (h 2).val * 16^1 + (h 3).val * 16^0

/-- The specific HexNumber ABCD --/
def ABCD : HexNumber :=
  fun i => match i with
    | 0 => ⟨10, by norm_num⟩
    | 1 => ⟨11, by norm_num⟩
    | 2 => ⟨12, by norm_num⟩
    | 3 => ⟨13, by norm_num⟩

/-- Number of bits required to represent a natural number --/
def bitsRequired (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem ABCD_requires_16_bits :
  bitsRequired (toDecimal ABCD) = 16 :=
sorry

end NUMINAMATH_CALUDE_ABCD_requires_16_bits_l3147_314755


namespace NUMINAMATH_CALUDE_magnitude_comparison_l3147_314784

theorem magnitude_comparison (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1/2) 
  (A : ℝ) (hA : A = 1 - a^2)
  (B : ℝ) (hB : B = 1 + a^2)
  (C : ℝ) (hC : C = 1 / (1 - a))
  (D : ℝ) (hD : D = 1 / (1 + a)) :
  (1 - a > a^2) ∧ (D < A ∧ A < B ∧ B < C) := by
sorry

end NUMINAMATH_CALUDE_magnitude_comparison_l3147_314784


namespace NUMINAMATH_CALUDE_fraction_equality_l3147_314763

theorem fraction_equality (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3147_314763


namespace NUMINAMATH_CALUDE_not_all_greater_than_one_l3147_314719

theorem not_all_greater_than_one (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 2) 
  (hc : 0 < c ∧ c < 2) : 
  ¬(a * (2 - b) > 1 ∧ b * (2 - c) > 1 ∧ c * (2 - a) > 1) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_one_l3147_314719


namespace NUMINAMATH_CALUDE_max_product_sum_l3147_314742

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l3147_314742


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3147_314775

theorem rectangle_dimension_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let new_length := 1.15 * L
  let new_area := 1.035 * (L * B)
  let new_breadth := new_area / new_length
  (new_breadth / B) = 0.9 := by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3147_314775


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l3147_314751

/-- Sum of arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- Sum of even integers from 2 to 120 -/
def a : ℕ := arithmeticSum 2 120 60

/-- Sum of odd integers from 1 to 119 -/
def b : ℕ := arithmeticSum 1 119 60

/-- The difference between the sum of even integers from 2 to 120 and
    the sum of odd integers from 1 to 119 is 60 -/
theorem even_odd_sum_difference : a - b = 60 := by sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l3147_314751


namespace NUMINAMATH_CALUDE_misread_signs_count_l3147_314711

def f (x : ℝ) : ℝ := 10*x^9 + 9*x^8 + 8*x^7 + 7*x^6 + 6*x^5 + 5*x^4 + 4*x^3 + 3*x^2 + 2*x + 1

theorem misread_signs_count :
  let correct_result := f (-1)
  let incorrect_result := 7
  let difference := incorrect_result - correct_result
  difference / 2 = 6 := by sorry

end NUMINAMATH_CALUDE_misread_signs_count_l3147_314711


namespace NUMINAMATH_CALUDE_total_rainfall_l3147_314756

def rainfall_problem (sunday monday tuesday : ℕ) : Prop :=
  (tuesday = 2 * monday) ∧
  (monday = sunday + 3) ∧
  (sunday = 4)

theorem total_rainfall : 
  ∀ sunday monday tuesday : ℕ, 
  rainfall_problem sunday monday tuesday → 
  sunday + monday + tuesday = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_l3147_314756


namespace NUMINAMATH_CALUDE_certain_number_problem_l3147_314754

theorem certain_number_problem (C : ℝ) : C - |(-10 + 6)| = 26 → C = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3147_314754


namespace NUMINAMATH_CALUDE_max_int_diff_l3147_314771

theorem max_int_diff (x y : ℤ) (hx : 6 < x ∧ x < 10) (hy : 10 < y ∧ y < 17) :
  (∀ a b : ℤ, 6 < a ∧ a < 10 ∧ 10 < b ∧ b < 17 → y - x ≥ b - a) ∧ y - x = 7 :=
sorry

end NUMINAMATH_CALUDE_max_int_diff_l3147_314771


namespace NUMINAMATH_CALUDE_min_value_theorem_l3147_314734

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  ∃ (x : ℝ), x = 2 * Real.sqrt 3 - 2 ∧ ∀ (y : ℝ), 2 * a + b + c ≥ y :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3147_314734


namespace NUMINAMATH_CALUDE_pizza_sector_chord_length_squared_l3147_314780

theorem pizza_sector_chord_length_squared (r : ℝ) (h : r = 8) :
  let chord_length_squared := 2 * r^2
  chord_length_squared = 128 := by sorry

end NUMINAMATH_CALUDE_pizza_sector_chord_length_squared_l3147_314780


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3147_314737

theorem quadratic_factorization (c d : ℤ) : 
  (∀ x, 25 * x^2 - 155 * x - 150 = (5 * x + c) * (5 * x + d)) → 
  c + 3 * d = -43 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3147_314737


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l3147_314753

theorem square_area_error_percentage (x : ℝ) (h : x > 0) : 
  let measured_side := 1.18 * x
  let actual_area := x ^ 2
  let calculated_area := measured_side ^ 2
  let area_error_percentage := (calculated_area - actual_area) / actual_area * 100
  area_error_percentage = 39.24 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l3147_314753


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l3147_314722

-- Define the custom multiplication operation
def custom_mult (a b : ℤ) : ℤ := a^2 - a*b + b^2

-- Theorem statement
theorem custom_mult_four_three : custom_mult 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l3147_314722


namespace NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_four_satisfies_inequality_four_is_smallest_l3147_314759

theorem smallest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 13*n + 36 ≤ 0 → n ≥ 4 :=
by sorry

theorem four_satisfies_inequality :
  4^2 - 13*4 + 36 ≤ 0 :=
by sorry

theorem four_is_smallest :
  ∀ n : ℤ, n < 4 → n^2 - 13*n + 36 > 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_four_satisfies_inequality_four_is_smallest_l3147_314759


namespace NUMINAMATH_CALUDE_repeating_decimal_47_equals_fraction_sum_of_numerator_and_denominator_l3147_314774

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

theorem repeating_decimal_47_equals_fraction :
  RepeatingDecimal 4 7 = 47 / 99 :=
sorry

theorem sum_of_numerator_and_denominator :
  (47 : ℕ) + 99 = 146 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_47_equals_fraction_sum_of_numerator_and_denominator_l3147_314774


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l3147_314758

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x^2 - 6*x + 8 ≠ 0) 
  (h2 : x^2 - 8*x + 15 ≠ 0) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 
  (x - 3) / (x^2 - 6*x + 8) :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_division_l3147_314758


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l3147_314783

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : ℕ
  intersections : ℕ

/-- Predicate to check if a configuration is valid -/
def IsValidConfiguration (config : LineConfiguration) : Prop :=
  config.lines = 100 ∧ (config.intersections = 100 ∨ config.intersections = 99)

theorem intersection_points_theorem :
  ∃ (config1 config2 : LineConfiguration),
    IsValidConfiguration config1 ∧
    IsValidConfiguration config2 ∧
    config1.intersections = 100 ∧
    config2.intersections = 99 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l3147_314783


namespace NUMINAMATH_CALUDE_frustum_views_l3147_314744

/-- A frustum is a portion of a solid (usually a cone or pyramid) lying between two parallel planes cutting the solid. -/
structure Frustum where
  -- Add necessary fields to define a frustum

/-- Represents a 2D view of a 3D object -/
inductive View
  | IsoscelesTrapezoid
  | ConcentricCircles

/-- Front view of a frustum -/
def front_view (f : Frustum) : View := sorry

/-- Side view of a frustum -/
def side_view (f : Frustum) : View := sorry

/-- Top view of a frustum -/
def top_view (f : Frustum) : View := sorry

/-- Two views are congruent -/
def congruent (v1 v2 : View) : Prop := sorry

theorem frustum_views (f : Frustum) : 
  front_view f = View.IsoscelesTrapezoid ∧ 
  side_view f = View.IsoscelesTrapezoid ∧
  congruent (front_view f) (side_view f) ∧
  top_view f = View.ConcentricCircles := by sorry

end NUMINAMATH_CALUDE_frustum_views_l3147_314744


namespace NUMINAMATH_CALUDE_shelter_dogs_l3147_314733

theorem shelter_dogs (C : ℕ) (h1 : C > 0) (h2 : (15 : ℚ) / C = 11 / (C + 8)) : 
  (15 : ℕ) * C = 15 * 15 :=
sorry

end NUMINAMATH_CALUDE_shelter_dogs_l3147_314733


namespace NUMINAMATH_CALUDE_emmas_average_speed_l3147_314743

theorem emmas_average_speed 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ)
  (h1 : distance1 = 420)
  (h2 : time1 = 7)
  (h3 : distance2 = 480)
  (h4 : time2 = 8) :
  (distance1 + distance2) / (time1 + time2) = 60 := by
sorry

end NUMINAMATH_CALUDE_emmas_average_speed_l3147_314743


namespace NUMINAMATH_CALUDE_exists_uncolored_diameter_l3147_314752

/-- Represents a circle with some arcs colored black -/
structure BlackArcCircle where
  /-- The total circumference of the circle -/
  circumference : ℝ
  /-- The total length of black arcs -/
  blackArcLength : ℝ
  /-- Assumption that the black arc length is less than half the circumference -/
  blackArcLengthLessThanHalf : blackArcLength < circumference / 2

/-- A point on the circle -/
structure CirclePoint where
  /-- The angle of the point relative to a fixed reference point -/
  angle : ℝ

/-- Represents a diameter of the circle -/
structure Diameter where
  /-- One endpoint of the diameter -/
  point1 : CirclePoint
  /-- The other endpoint of the diameter -/
  point2 : CirclePoint
  /-- Assumption that the points are opposite each other on the circle -/
  oppositePoints : point2.angle = point1.angle + π

/-- Function to determine if a point is on a black arc -/
def isOnBlackArc (c : BlackArcCircle) (p : CirclePoint) : Prop := sorry

/-- Theorem stating that there exists a diameter with both ends uncolored -/
theorem exists_uncolored_diameter (c : BlackArcCircle) : 
  ∃ d : Diameter, ¬isOnBlackArc c d.point1 ∧ ¬isOnBlackArc c d.point2 := by sorry

end NUMINAMATH_CALUDE_exists_uncolored_diameter_l3147_314752


namespace NUMINAMATH_CALUDE_spinner_probability_l3147_314791

theorem spinner_probability : ∀ (p_C p_D p_E : ℚ),
  (p_C = p_D) →
  (p_D = p_E) →
  (1/5 : ℚ) + (1/5 : ℚ) + p_C + p_D + p_E = 1 →
  p_C = (1/5 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3147_314791


namespace NUMINAMATH_CALUDE_proposition_evaluation_l3147_314765

theorem proposition_evaluation : 
  let p : Prop := (2 + 4 = 7)
  let q : Prop := (∀ x : ℝ, x = 1 → x^2 ≠ 1)
  ¬(p ∧ q) ∧ (p ∨ q) := by
sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l3147_314765


namespace NUMINAMATH_CALUDE_catering_budget_theorem_l3147_314724

def total_guests : ℕ := 80
def steak_cost : ℕ := 25
def chicken_cost : ℕ := 18

def catering_budget (chicken_guests : ℕ) : ℕ :=
  chicken_guests * chicken_cost + (3 * chicken_guests) * steak_cost

theorem catering_budget_theorem :
  ∃ (chicken_guests : ℕ),
    chicken_guests + 3 * chicken_guests = total_guests ∧
    catering_budget chicken_guests = 1860 :=
by
  sorry

end NUMINAMATH_CALUDE_catering_budget_theorem_l3147_314724


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l3147_314770

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit in one direction -/
def tilesInOneDimension (floorSize tileSize : ℕ) : ℕ :=
  floorSize / tileSize

/-- Calculates the total number of tiles for a given orientation -/
def totalTiles (floor tile : Dimensions) : ℕ :=
  (tilesInOneDimension floor.length tile.length) * (tilesInOneDimension floor.width tile.width)

/-- Theorem stating the maximum number of tiles that can be accommodated -/
theorem max_tiles_on_floor (floor : Dimensions) (tile : Dimensions) 
    (h_floor : floor = ⟨1000, 210⟩) (h_tile : tile = ⟨35, 30⟩) :
  max (totalTiles floor tile) (totalTiles floor ⟨tile.width, tile.length⟩) = 198 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l3147_314770


namespace NUMINAMATH_CALUDE_jelly_overlap_l3147_314732

/-- The number of jellies -/
def num_jellies : ℕ := 12

/-- The length of each jelly in centimeters -/
def jelly_length : ℝ := 18

/-- The circumference of the ring in centimeters -/
def ring_circumference : ℝ := 210

/-- The overlapping portion of each jelly in millimeters -/
def overlap_mm : ℝ := 5

theorem jelly_overlap :
  (num_jellies : ℝ) * jelly_length - ring_circumference = num_jellies * overlap_mm / 10 := by
  sorry

end NUMINAMATH_CALUDE_jelly_overlap_l3147_314732


namespace NUMINAMATH_CALUDE_selling_price_formula_l3147_314701

-- Define the relationship between quantity and selling price
def selling_price (x : ℕ+) : ℚ :=
  match x with
  | 1 => 8 + 0.3
  | 2 => 16 + 0.6
  | 3 => 24 + 0.9
  | 4 => 32 + 1.2
  | _ => 8.3 * x.val

-- Theorem statement
theorem selling_price_formula (x : ℕ+) :
  selling_price x = 8.3 * x.val := by sorry

end NUMINAMATH_CALUDE_selling_price_formula_l3147_314701


namespace NUMINAMATH_CALUDE_corn_acreage_l3147_314738

/-- Given a total of 1034 acres divided in the ratio 5 : 2 : 4 for beans, wheat, and corn respectively,
    the number of acres used for corn is 376. -/
theorem corn_acreage (total_acres : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
    (h1 : total_acres = 1034)
    (h2 : bean_ratio = 5)
    (h3 : wheat_ratio = 2)
    (h4 : corn_ratio = 4) : 
  (total_acres * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l3147_314738


namespace NUMINAMATH_CALUDE_square_area_from_points_l3147_314723

/-- The area of a square with adjacent points (4,3) and (5,7) is 17 -/
theorem square_area_from_points : 
  let p1 : ℝ × ℝ := (4, 3)
  let p2 : ℝ × ℝ := (5, 7)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 17 := by sorry

end NUMINAMATH_CALUDE_square_area_from_points_l3147_314723


namespace NUMINAMATH_CALUDE_initial_coloring_books_l3147_314736

theorem initial_coloring_books (books_removed : ℝ) (coupons_per_book : ℝ) (total_coupons : ℕ) :
  books_removed = 20 →
  coupons_per_book = 4 →
  total_coupons = 80 →
  ∃ (initial_books : ℕ), initial_books = 40 ∧ 
    (initial_books : ℝ) - books_removed = (total_coupons : ℝ) / coupons_per_book :=
by
  sorry

end NUMINAMATH_CALUDE_initial_coloring_books_l3147_314736


namespace NUMINAMATH_CALUDE_stratified_sampling_best_for_survey1_simple_random_sampling_best_for_survey2_l3147_314788

/-- Represents the income level of a family -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Structure representing the community for Survey 1 -/
structure Community where
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat
  sampleSize : Nat

/-- Structure representing the student group for Survey 2 -/
structure StudentGroup where
  totalStudents : Nat
  sampleSize : Nat

/-- Function to determine the most appropriate sampling method for Survey 1 -/
def bestSamplingMethodSurvey1 (community : Community) : SamplingMethod := sorry

/-- Function to determine the most appropriate sampling method for Survey 2 -/
def bestSamplingMethodSurvey2 (studentGroup : StudentGroup) : SamplingMethod := sorry

/-- Theorem stating that stratified sampling is most appropriate for Survey 1 -/
theorem stratified_sampling_best_for_survey1 (community : Community) 
  (h1 : community.highIncomeFamilies = 125)
  (h2 : community.middleIncomeFamilies = 280)
  (h3 : community.lowIncomeFamilies = 95)
  (h4 : community.sampleSize = 100) :
  bestSamplingMethodSurvey1 community = SamplingMethod.Stratified := sorry

/-- Theorem stating that simple random sampling is most appropriate for Survey 2 -/
theorem simple_random_sampling_best_for_survey2 (studentGroup : StudentGroup)
  (h1 : studentGroup.totalStudents = 15)
  (h2 : studentGroup.sampleSize = 3) :
  bestSamplingMethodSurvey2 studentGroup = SamplingMethod.SimpleRandom := sorry

end NUMINAMATH_CALUDE_stratified_sampling_best_for_survey1_simple_random_sampling_best_for_survey2_l3147_314788


namespace NUMINAMATH_CALUDE_counterexample_dot_product_equality_l3147_314730

theorem counterexample_dot_product_equality :
  ∃ (a b c : ℝ × ℝ), a ≠ (0, 0) ∧ b ≠ (0, 0) ∧ c ≠ (0, 0) ∧
  (a.1 * b.1 + a.2 * b.2 = a.1 * c.1 + a.2 * c.2) ∧ b ≠ c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_dot_product_equality_l3147_314730


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l3147_314789

def is_multiple_of_8_plus_2 (n : ℕ) : Prop := ∃ k, n = 8 * k + 2

def is_multiple_of_6_plus_4 (n : ℕ) : Prop := ∃ j, n = 6 * j + 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_number :
  (is_three_digit 986) ∧
  (is_multiple_of_8_plus_2 986) ∧
  (is_multiple_of_6_plus_4 986) ∧
  (∀ m : ℕ, 
    (is_three_digit m) ∧ 
    (is_multiple_of_8_plus_2 m) ∧ 
    (is_multiple_of_6_plus_4 m) → 
    m ≤ 986) :=
by sorry

#check greatest_three_digit_number

end NUMINAMATH_CALUDE_greatest_three_digit_number_l3147_314789


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equal_area_l3147_314745

theorem right_triangle_perimeter_equal_area (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive integers
  a^2 + b^2 = c^2 →        -- Right-angled triangle (Pythagorean theorem)
  a + b + c = (a * b) / 2  -- Perimeter equals area
  → (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 5 ∧ b = 12 ∧ c = 13) ∨ 
    (a = 8 ∧ b = 6 ∧ c = 10) ∨ (a = 12 ∧ b = 5 ∧ c = 13) :=
by sorry

#check right_triangle_perimeter_equal_area

end NUMINAMATH_CALUDE_right_triangle_perimeter_equal_area_l3147_314745


namespace NUMINAMATH_CALUDE_feeding_to_total_ratio_l3147_314706

/-- Represents the time Larry spends on his dog in minutes -/
structure DogTime where
  walking_playing : ℕ  -- Time spent walking and playing (in minutes)
  total : ℕ           -- Total time spent on the dog (in minutes)

/-- The ratio of feeding time to total time is 1:6 -/
theorem feeding_to_total_ratio (t : DogTime) 
  (h1 : t.walking_playing = 30 * 2)
  (h2 : t.total = 72) : 
  (t.total - t.walking_playing) * 6 = t.total :=
by sorry

end NUMINAMATH_CALUDE_feeding_to_total_ratio_l3147_314706


namespace NUMINAMATH_CALUDE_exists_integer_square_one_l3147_314727

theorem exists_integer_square_one : ∃ x : ℤ, x^2 = 1 := by sorry

end NUMINAMATH_CALUDE_exists_integer_square_one_l3147_314727
