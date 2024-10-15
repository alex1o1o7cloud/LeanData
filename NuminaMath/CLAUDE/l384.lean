import Mathlib

namespace NUMINAMATH_CALUDE_hospital_staff_count_l384_38461

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) (h1 : total = 456) (h2 : doctor_ratio = 8) (h3 : nurse_ratio = 11) : 
  (nurse_ratio * total) / (doctor_ratio + nurse_ratio) = 264 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l384_38461


namespace NUMINAMATH_CALUDE_james_max_lift_l384_38464

def farmers_walk_20m : ℝ := 300

def increase_20m : ℝ := 50

def short_distance_increase_percent : ℝ := 0.3

def strap_increase_percent : ℝ := 0.2

def calculate_max_weight (base_weight : ℝ) (short_distance_increase : ℝ) (strap_increase : ℝ) : ℝ :=
  base_weight * (1 + short_distance_increase) * (1 + strap_increase)

theorem james_max_lift :
  calculate_max_weight (farmers_walk_20m + increase_20m) short_distance_increase_percent strap_increase_percent = 546 := by
  sorry

end NUMINAMATH_CALUDE_james_max_lift_l384_38464


namespace NUMINAMATH_CALUDE_arrangement_count_is_24_l384_38498

/-- The number of ways to arrange 8 balls in a row, with 5 red balls and 3 white balls,
    such that exactly three red balls are consecutive. -/
def arrangement_count : ℕ := 24

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of consecutive red balls required -/
def consecutive_red : ℕ := 3

theorem arrangement_count_is_24 :
  arrangement_count = 24 ∧
  total_balls = 8 ∧
  red_balls = 5 ∧
  white_balls = 3 ∧
  consecutive_red = 3 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_24_l384_38498


namespace NUMINAMATH_CALUDE_vector_perpendicular_l384_38413

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![0, -2]

def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

theorem vector_perpendicular :
  perpendicular (λ i => a i + 2 * b i) ![3, 2] := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l384_38413


namespace NUMINAMATH_CALUDE_chord_length_l384_38421

/-- The length of the chord cut off by a circle on a line --/
theorem chord_length (x y : ℝ) : 
  let line := {(x, y) | x - y - 3 = 0}
  let circle := {(x, y) | (x - 2)^2 + y^2 = 4}
  let chord := line ∩ circle
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l384_38421


namespace NUMINAMATH_CALUDE_initial_maple_trees_count_l384_38408

/-- The number of maple trees to be planted -/
def trees_to_plant : ℕ := 9

/-- The final number of maple trees after planting -/
def final_maple_trees : ℕ := 11

/-- The initial number of maple trees in the park -/
def initial_maple_trees : ℕ := final_maple_trees - trees_to_plant

theorem initial_maple_trees_count : initial_maple_trees = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_maple_trees_count_l384_38408


namespace NUMINAMATH_CALUDE_apple_box_weight_l384_38414

theorem apple_box_weight : 
  ∀ (x : ℝ), 
  (x > 0) →  -- Ensure positive weight
  (3 * x - 3 * 4 = x) → 
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_box_weight_l384_38414


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l384_38400

/-- Two vectors are orthogonal if and only if their dot product is zero -/
def orthogonal (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The problem statement -/
theorem orthogonal_vectors (x : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-3, x)
  orthogonal a b ↔ x = -3/2 := by
sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l384_38400


namespace NUMINAMATH_CALUDE_problem_statement_l384_38483

theorem problem_statement (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + b)^2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l384_38483


namespace NUMINAMATH_CALUDE_sequence_position_l384_38471

/-- The general term of the sequence -/
def sequenceTerm (n : ℕ) : ℚ := (n + 3) / (n + 1)

/-- The position we want to prove -/
def position : ℕ := 14

/-- The fraction we're looking for -/
def targetFraction : ℚ := 17 / 15

theorem sequence_position :
  sequenceTerm position = targetFraction := by sorry

end NUMINAMATH_CALUDE_sequence_position_l384_38471


namespace NUMINAMATH_CALUDE_arc_length_calculation_l384_38412

/-- 
Given an arc with radius π cm and central angle 120°, 
prove that its arc length is (2/3)π² cm.
-/
theorem arc_length_calculation (r : ℝ) (θ_degrees : ℝ) (l : ℝ) : 
  r = π → θ_degrees = 120 → l = (2/3) * π^2 → 
  l = r * (θ_degrees * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l384_38412


namespace NUMINAMATH_CALUDE_fraction_irreducible_l384_38435

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l384_38435


namespace NUMINAMATH_CALUDE_triangle_theorem_l384_38459

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sin t.A ^ 2 + Real.sin t.A * Real.sin t.B - 6 * Real.sin t.B ^ 2 = 0) :
  (t.a / t.b = 2) ∧ 
  (Real.cos t.C = 3/4 → Real.sin t.B = Real.sqrt 14 / 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l384_38459


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l384_38416

theorem arithmetic_square_root_of_four : ∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ ∀ y : ℝ, y > 0 ∧ y^2 = 4 → y = x :=
sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l384_38416


namespace NUMINAMATH_CALUDE_arccos_one_half_l384_38454

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l384_38454


namespace NUMINAMATH_CALUDE_triangle_properties_l384_38477

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

def isArithmeticSequence (t : Triangle) : Prop :=
  t.a + t.c = 2 * t.b

def aEquals2c (t : Triangle) : Prop :=
  t.a = 2 * t.c

def areaIs3Sqrt15Over4 (t : Triangle) : Prop :=
  1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 15 / 4

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : isArithmeticSequence t)
  (h3 : aEquals2c t)
  (h4 : areaIs3Sqrt15Over4 t) :
  Real.cos t.A = -1/4 ∧ t.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l384_38477


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l384_38465

/-- The shortest distance between a point on the parabola y = x^2 - 4x and a point on the line y = 2x - 3 is 6√5/5 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2 - 4*p.1}
  let line := {p : ℝ × ℝ | p.2 = 2*p.1 - 3}
  ∀ A ∈ parabola, ∀ B ∈ line,
  ∃ C ∈ parabola, ∃ D ∈ line,
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ≤ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 6 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l384_38465


namespace NUMINAMATH_CALUDE_problem_solution_l384_38481

theorem problem_solution (a b c d : ℝ) : 
  8 = (4 / 100) * a →
  4 = (d / 100) * a →
  8 = (d / 100) * b →
  c = b / a →
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l384_38481


namespace NUMINAMATH_CALUDE_fraction_calculation_l384_38425

theorem fraction_calculation : (1/4 + 1/5) / (3/7 - 1/8) = 126/85 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l384_38425


namespace NUMINAMATH_CALUDE_carlo_thursday_practice_l384_38455

/-- Represents the practice schedule for Carlo's music recital --/
structure PracticeSchedule where
  thursday : ℕ  -- Minutes practiced on Thursday
  wednesday : ℕ := thursday + 5  -- Minutes practiced on Wednesday
  tuesday : ℕ := wednesday - 10  -- Minutes practiced on Tuesday
  monday : ℕ := 2 * tuesday  -- Minutes practiced on Monday
  friday : ℕ := 60  -- Minutes practiced on Friday

/-- Calculates the total practice time for the week --/
def totalPracticeTime (schedule : PracticeSchedule) : ℕ :=
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday

/-- Theorem stating that Carlo practiced for 50 minutes on Thursday --/
theorem carlo_thursday_practice :
  ∃ (schedule : PracticeSchedule), totalPracticeTime schedule = 300 ∧ schedule.thursday = 50 := by
  sorry

end NUMINAMATH_CALUDE_carlo_thursday_practice_l384_38455


namespace NUMINAMATH_CALUDE_revenue_growth_equation_l384_38466

def january_revenue : ℝ := 250
def quarter_target : ℝ := 900

theorem revenue_growth_equation (x : ℝ) :
  january_revenue + january_revenue * (1 + x) + january_revenue * (1 + x)^2 = quarter_target :=
by sorry

end NUMINAMATH_CALUDE_revenue_growth_equation_l384_38466


namespace NUMINAMATH_CALUDE_division_result_l384_38499

theorem division_result : (5 / 2) / 7 = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_division_result_l384_38499


namespace NUMINAMATH_CALUDE_partnership_investment_l384_38428

/-- Represents a partnership investment. -/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  c_investment : ℚ
  b_profit : ℚ
  a_profit : ℚ

/-- Theorem stating the relationship between investments and profits in a partnership. -/
theorem partnership_investment (p : Partnership) 
  (hb : p.b_investment = 11000)
  (hc : p.c_investment = 18000)
  (hbp : p.b_profit = 880)
  (hap : p.a_profit = 560) :
  p.a_investment = 7700 := by
  sorry


end NUMINAMATH_CALUDE_partnership_investment_l384_38428


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l384_38405

/-- Given a parallelogram with sides measuring 7, 9, 8y-1, and 2x+3 units consecutively,
    prove that x + y = 4 -/
theorem parallelogram_side_sum (x y : ℝ) : 
  (7 : ℝ) = 8*y - 1 → (9 : ℝ) = 2*x + 3 → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l384_38405


namespace NUMINAMATH_CALUDE_sum_segment_lengths_equals_78_l384_38485

/-- Triangle with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Sum of lengths of segments cut by horizontal integer lines -/
def sumSegmentLengths (t : Triangle) : ℝ :=
  sorry

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { A := (1, 3.5),
    B := (13.5, 3.5),
    C := (11, 16) }

theorem sum_segment_lengths_equals_78 :
  sumSegmentLengths problemTriangle = 78 :=
sorry

end NUMINAMATH_CALUDE_sum_segment_lengths_equals_78_l384_38485


namespace NUMINAMATH_CALUDE_zed_wye_value_l384_38458

-- Define the types of coins
structure Coin where
  value : ℚ

-- Define the coins
def Ex : Coin := ⟨1⟩
def Wye : Coin := ⟨1⟩
def Zed : Coin := ⟨1⟩

-- Define the given conditions
axiom ex_wye_relation : 2 * Ex.value = 29 * Wye.value
axiom zed_ex_relation : Zed.value = 16 * Ex.value

theorem zed_wye_value : Zed.value = 232 * Wye.value :=
by sorry

end NUMINAMATH_CALUDE_zed_wye_value_l384_38458


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l384_38495

theorem geometric_arithmetic_sequence_ratio 
  (x y z r : ℝ) 
  (h1 : y = r * x) 
  (h2 : z = r * y) 
  (h3 : x ≠ y) 
  (h4 : 2 * (2 * y) = x + 3 * z) : 
  r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l384_38495


namespace NUMINAMATH_CALUDE_time_in_terms_of_angle_and_angular_velocity_l384_38438

theorem time_in_terms_of_angle_and_angular_velocity 
  (α ω ω₀ θ t : ℝ) 
  (h1 : ω = α * t + ω₀) 
  (h2 : θ = (1/2) * α * t^2 + ω₀ * t) : 
  t = 2 * θ / (ω + ω₀) := by
sorry

end NUMINAMATH_CALUDE_time_in_terms_of_angle_and_angular_velocity_l384_38438


namespace NUMINAMATH_CALUDE_special_square_difference_l384_38403

theorem special_square_difference : 123456789^2 - 123456788 * 123456790 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_square_difference_l384_38403


namespace NUMINAMATH_CALUDE_marble_probability_l384_38453

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  hSum : total = black + white

/-- The probability of drawing a specific color from a box -/
def drawProbability (box : MarbleBox) (color : ℕ) : ℚ :=
  color / box.total

theorem marble_probability (box1 box2 : MarbleBox)
  (hTotal : box1.total + box2.total = 30)
  (hBlackProb : drawProbability box1 box1.black * drawProbability box2 box2.black = 1/2) :
  drawProbability box1 box1.white * drawProbability box2 box2.white = 0 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l384_38453


namespace NUMINAMATH_CALUDE_candies_added_l384_38430

theorem candies_added (initial_candies final_candies : ℕ) (h1 : initial_candies = 6) (h2 : final_candies = 10) :
  final_candies - initial_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_candies_added_l384_38430


namespace NUMINAMATH_CALUDE_silk_per_dress_is_five_l384_38446

/-- Calculates the amount of silk needed for each dress given the initial silk amount,
    number of friends, silk given to each friend, and number of dresses made. -/
def silk_per_dress (initial_silk : ℕ) (num_friends : ℕ) (silk_per_friend : ℕ) (num_dresses : ℕ) : ℕ :=
  (initial_silk - num_friends * silk_per_friend) / num_dresses

/-- Proves that given the specified conditions, the amount of silk needed for each dress is 5 meters. -/
theorem silk_per_dress_is_five :
  silk_per_dress 600 5 20 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_silk_per_dress_is_five_l384_38446


namespace NUMINAMATH_CALUDE_prime_triplets_equation_l384_38449

theorem prime_triplets_equation :
  ∀ p q r : ℕ,
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r →
    (p : ℚ) / q = 8 / (r - 1) + 1 →
    ((p = 3 ∧ q = 2 ∧ r = 17) ∨
     (p = 7 ∧ q = 3 ∧ r = 7) ∨
     (p = 5 ∧ q = 3 ∧ r = 13)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triplets_equation_l384_38449


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l384_38460

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - 2*x + a ≤ 0) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l384_38460


namespace NUMINAMATH_CALUDE_no_solutions_to_inequality_l384_38407

theorem no_solutions_to_inequality :
  ¬∃ x : ℝ, (6 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_to_inequality_l384_38407


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l384_38418

theorem inequality_system_solution_set :
  ∀ x : ℝ, (3 * x - 1 ≥ x + 1 ∧ x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l384_38418


namespace NUMINAMATH_CALUDE_constant_zero_sequence_l384_38447

def is_sum_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ k, S (k + 1) = S k + a (k + 1)

theorem constant_zero_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h : ∀ k, S (k + 1) + S k = a (k + 1)) :
  ∀ n, a n = 0 :=
by sorry

end NUMINAMATH_CALUDE_constant_zero_sequence_l384_38447


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l384_38442

theorem quadratic_roots_sum (x : ℝ) (h : x^2 - 9*x + 20 = 0) : 
  ∃ (y : ℝ), y ≠ x ∧ y^2 - 9*y + 20 = 0 ∧ x + y = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l384_38442


namespace NUMINAMATH_CALUDE_particle_position_1989_l384_38475

/-- Represents the position of a particle -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Calculates the position of the particle after a given number of minutes -/
def particlePosition (minutes : ℕ) : Position :=
  sorry

/-- The theorem stating the particle's position after 1989 minutes -/
theorem particle_position_1989 : particlePosition 1989 = Position.mk 44 35 := by
  sorry

end NUMINAMATH_CALUDE_particle_position_1989_l384_38475


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l384_38494

theorem quadratic_root_theorem (c : ℝ) : 
  (∀ x : ℝ, 2*x^2 + 8*x + c = 0 ↔ x = -2 + Real.sqrt 3 ∨ x = -2 - Real.sqrt 3) →
  c = 13/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l384_38494


namespace NUMINAMATH_CALUDE_no_solution_system_l384_38437

theorem no_solution_system :
  ¬∃ (x y : ℝ), 
    (x^3 + x + y + 1 = 0) ∧ 
    (y*x^2 + x + y = 0) ∧ 
    (y^2 + y - x^2 + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l384_38437


namespace NUMINAMATH_CALUDE_probability_theorem_l384_38493

/-- The probability that the straight-line distance between two randomly chosen points 
    on the sides of a square with side length 2 is at least 1 -/
def probability_distance_at_least_one (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- A square with side length 2 -/
def square_side_two : Set (ℝ × ℝ) :=
  sorry

theorem probability_theorem :
  let S := square_side_two
  probability_distance_at_least_one S = (22 - π) / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l384_38493


namespace NUMINAMATH_CALUDE_tournament_theorem_l384_38457

/-- Represents a team in the tournament -/
structure Team :=
  (city : Fin 16)
  (is_team_a : Bool)

/-- The number of matches played by a team -/
def matches_played (t : Team) : Fin 32 := sorry

/-- The statement that all teams except one have unique match counts -/
def all_but_one_unique (exception : Team) : Prop :=
  ∀ t1 t2 : Team, t1 ≠ exception → t2 ≠ exception → t1 ≠ t2 → matches_played t1 ≠ matches_played t2

theorem tournament_theorem :
  ∃ (exception : Team),
    (all_but_one_unique exception) →
    (matches_played exception = 15) :=
  sorry

end NUMINAMATH_CALUDE_tournament_theorem_l384_38457


namespace NUMINAMATH_CALUDE_sam_candy_bars_l384_38491

/-- Represents the number of candy bars Sam bought -/
def candy_bars : ℕ := sorry

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of dimes Sam initially had -/
def initial_dimes : ℕ := 19

/-- The number of quarters Sam initially had -/
def initial_quarters : ℕ := 6

/-- The cost of a candy bar in dimes -/
def candy_bar_cost_dimes : ℕ := 3

/-- The cost of a lollipop in quarters -/
def lollipop_cost_quarters : ℕ := 1

/-- The amount of money Sam has left after purchases, in cents -/
def remaining_cents : ℕ := 195

theorem sam_candy_bars : 
  candy_bars = 4 ∧
  initial_dimes * dime_value + initial_quarters * quarter_value = 
  remaining_cents + candy_bars * (candy_bar_cost_dimes * dime_value) + 
  lollipop_cost_quarters * quarter_value :=
sorry

end NUMINAMATH_CALUDE_sam_candy_bars_l384_38491


namespace NUMINAMATH_CALUDE_coincident_centers_of_inscribed_ngons_l384_38429

/-- A regular n-gon in a 2D plane. -/
structure RegularNGon where
  n : ℕ
  center : ℝ × ℝ
  radius : ℝ
  rotation : ℝ  -- Rotation angle of the first vertex

/-- The vertices of a regular n-gon. -/
def vertices (ngon : RegularNGon) : Finset (ℝ × ℝ) :=
  sorry

/-- Predicate to check if a point lies on the perimeter of an n-gon. -/
def on_perimeter (point : ℝ × ℝ) (ngon : RegularNGon) : Prop :=
  sorry

/-- Theorem: If the vertices of one regular n-gon lie on the perimeter of another,
    their centers coincide (for n ≥ 4). -/
theorem coincident_centers_of_inscribed_ngons
  (n : ℕ)
  (h_n : n ≥ 4)
  (ngon1 ngon2 : RegularNGon)
  (h_same_n : ngon1.n = n ∧ ngon2.n = n)
  (h_inscribed : ∀ v ∈ vertices ngon1, on_perimeter v ngon2) :
  ngon1.center = ngon2.center :=
sorry

end NUMINAMATH_CALUDE_coincident_centers_of_inscribed_ngons_l384_38429


namespace NUMINAMATH_CALUDE_stratified_sampling_sophomores_l384_38401

theorem stratified_sampling_sophomores (total_students : ℕ) (sophomores : ℕ) (selected : ℕ) 
  (h1 : total_students = 2800) 
  (h2 : sophomores = 930) 
  (h3 : selected = 280) :
  (sophomores * selected) / total_students = 93 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sophomores_l384_38401


namespace NUMINAMATH_CALUDE_FGH_supermarket_count_l384_38480

def FGH_supermarkets : Type := Unit

def location : FGH_supermarkets → Bool
  | _ => sorry

def in_US (s : FGH_supermarkets) : Prop := location s = true
def in_Canada (s : FGH_supermarkets) : Prop := location s = false

axiom all_in_US_or_Canada : ∀ s : FGH_supermarkets, in_US s ∨ in_Canada s

def count_US : Nat := 42
def count_Canada : Nat := count_US - 14

def total_count : Nat := count_US + count_Canada

theorem FGH_supermarket_count : total_count = 70 := by sorry

end NUMINAMATH_CALUDE_FGH_supermarket_count_l384_38480


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l384_38431

/-- An arithmetic sequence with first term 6 and the sum of the 3rd and 5th terms equal to 0 -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  a 1 = 6 ∧ 
  a 3 + a 5 = 0 ∧ 
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The general term formula for the arithmetic sequence -/
def GeneralTermFormula (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 8 - 2 * n

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ) (h : ArithmeticSequence a) : GeneralTermFormula a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l384_38431


namespace NUMINAMATH_CALUDE_inequality_proof_l384_38450

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c = 1)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x^2 + y^2 + z^2) * (a^3 / (x^2 + 2*y^2) + b^3 / (y^2 + 2*z^2) + c^3 / (z^2 + 2*x^2)) ≥ 1/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l384_38450


namespace NUMINAMATH_CALUDE_citizenship_test_study_time_l384_38468

/-- Calculates the total study time in hours for a citizenship test -/
theorem citizenship_test_study_time :
  let total_questions : ℕ := 60
  let multiple_choice_questions : ℕ := 30
  let fill_in_blank_questions : ℕ := 30
  let multiple_choice_time : ℕ := 15  -- minutes per question
  let fill_in_blank_time : ℕ := 25    -- minutes per question
  
  total_questions = multiple_choice_questions + fill_in_blank_questions →
  (multiple_choice_questions * multiple_choice_time + fill_in_blank_questions * fill_in_blank_time) / 60 = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_citizenship_test_study_time_l384_38468


namespace NUMINAMATH_CALUDE_positive_rationals_decomposition_l384_38434

-- Define the set of positive integers
def PositiveIntegers : Set ℚ := {x : ℚ | x > 0 ∧ x.den = 1}

-- Define the set of positive fractions
def PositiveFractions : Set ℚ := {x : ℚ | x > 0 ∧ x.den ≠ 1}

-- Define the set of positive rational numbers
def PositiveRationals : Set ℚ := {x : ℚ | x > 0}

-- Theorem statement
theorem positive_rationals_decomposition :
  PositiveRationals = PositiveIntegers ∪ PositiveFractions :=
by sorry

end NUMINAMATH_CALUDE_positive_rationals_decomposition_l384_38434


namespace NUMINAMATH_CALUDE_amount_transferred_l384_38424

def initial_balance : ℕ := 27004
def remaining_balance : ℕ := 26935

theorem amount_transferred : initial_balance - remaining_balance = 69 := by
  sorry

end NUMINAMATH_CALUDE_amount_transferred_l384_38424


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_2_5_11_l384_38419

theorem smallest_five_digit_divisible_by_2_5_11 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 2 ∣ n ∧ 5 ∣ n ∧ 11 ∣ n → 10010 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_2_5_11_l384_38419


namespace NUMINAMATH_CALUDE_divisibility_by_1989_l384_38470

theorem divisibility_by_1989 (n : ℕ) : ∃ k : ℤ, 
  13 * (-50)^n + 17 * 40^n - 30 = 1989 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1989_l384_38470


namespace NUMINAMATH_CALUDE_smallest_n_perfect_square_and_fifth_power_l384_38474

theorem smallest_n_perfect_square_and_fifth_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (a : ℕ), 4 * n = a^2) ∧
  (∃ (b : ℕ), 5 * n = b^5) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 4 * m = x^2) → 
    (∃ (y : ℕ), 5 * m = y^5) → 
    m ≥ n) ∧
  n = 3125 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_square_and_fifth_power_l384_38474


namespace NUMINAMATH_CALUDE_club_members_after_five_years_l384_38482

/-- Represents the number of people in the club after k years -/
def club_members (k : ℕ) : ℕ :=
  if k = 0 then 18
  else 3 * club_members (k - 1) - 10

/-- The number of people in the club after 5 years is 3164 -/
theorem club_members_after_five_years :
  club_members 5 = 3164 := by
  sorry

end NUMINAMATH_CALUDE_club_members_after_five_years_l384_38482


namespace NUMINAMATH_CALUDE_vasyas_numbers_l384_38492

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l384_38492


namespace NUMINAMATH_CALUDE_de_moivre_formula_l384_38486

theorem de_moivre_formula (x : ℝ) (n : ℕ) (h : x ∈ Set.Ioo 0 (π / 2)) :
  (Complex.exp (Complex.I * x)) ^ n = Complex.exp (Complex.I * (n : ℝ) * x) := by
  sorry

#check de_moivre_formula

end NUMINAMATH_CALUDE_de_moivre_formula_l384_38486


namespace NUMINAMATH_CALUDE_rationalize_denominator_l384_38489

theorem rationalize_denominator : 7 / Real.sqrt 63 = Real.sqrt 7 / 3 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l384_38489


namespace NUMINAMATH_CALUDE_rod_speed_l384_38422

/-- 
Given a rod moving freely between a horizontal floor and a slanted wall:
- v: speed of the end in contact with the floor
- θ: angle between the rod and the horizontal floor
- α: angle such that (α - θ) is the angle between the rod and the slanted wall

This theorem states that the speed of the end in contact with the wall 
is v * cos(θ) / cos(α - θ)
-/
theorem rod_speed (v θ α : ℝ) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_rod_speed_l384_38422


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l384_38444

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (3 - m) * x + 2 * m * y + 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := 2 * m * x + 2 * y + m = 0

-- Define parallel and perpendicular conditions
def parallel (m : ℝ) : Prop := ∀ x y, l₁ m x y ↔ l₂ m x y
def perpendicular (m : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, l₁ m x₁ y₁ → l₂ m x₂ y₂ → 
  ((3 - m) * (2 * m) + (2 * m) * 2 = 0)

-- Theorem for parallel lines
theorem parallel_lines : ∀ m : ℝ, parallel m ↔ m = -3/2 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines : ∀ m : ℝ, perpendicular m ↔ (m = 0 ∨ m = 5) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l384_38444


namespace NUMINAMATH_CALUDE_xy_reciprocal_l384_38496

theorem xy_reciprocal (x y : ℝ) 
  (h1 : x * y > 0) 
  (h2 : 1 / x + 1 / y = 15) 
  (h3 : (x + y) / 5 = 0.6) : 
  1 / (x * y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_reciprocal_l384_38496


namespace NUMINAMATH_CALUDE_remainder_252_power_252_mod_13_l384_38436

theorem remainder_252_power_252_mod_13 : 252^252 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_252_power_252_mod_13_l384_38436


namespace NUMINAMATH_CALUDE_least_number_divisible_l384_38411

theorem least_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((1076 + m) % 23 = 0 ∧ (1076 + m) % 29 = 0 ∧ (1076 + m) % 31 = 0)) ∧ 
  ((1076 + n) % 23 = 0 ∧ (1076 + n) % 29 = 0 ∧ (1076 + n) % 31 = 0) → 
  n = 19601 := by
sorry

end NUMINAMATH_CALUDE_least_number_divisible_l384_38411


namespace NUMINAMATH_CALUDE_tourist_catch_up_l384_38410

/-- The distance traveled by both tourists when the second catches up to the first -/
def catch_up_distance : ℝ := 56

theorem tourist_catch_up 
  (v_bicycle : ℝ) 
  (v_motorcycle : ℝ) 
  (initial_ride_time : ℝ) 
  (break_time : ℝ) 
  (delay_time : ℝ) :
  v_bicycle = 16 →
  v_motorcycle = 56 →
  initial_ride_time = 1.5 →
  break_time = 1.5 →
  delay_time = 4 →
  ∃ t : ℝ, 
    t > 0 ∧ 
    v_bicycle * (initial_ride_time + t) = 
    v_motorcycle * t + v_bicycle * initial_ride_time ∧
    v_bicycle * (initial_ride_time + t) = catch_up_distance :=
by sorry

end NUMINAMATH_CALUDE_tourist_catch_up_l384_38410


namespace NUMINAMATH_CALUDE_merill_marble_count_l384_38420

/-- The number of marbles each person has -/
structure MarbleCount where
  merill : ℕ
  elliot : ℕ
  selma : ℕ

/-- The conditions of the marble problem -/
def marbleProblemConditions (m : MarbleCount) : Prop :=
  m.merill = 2 * m.elliot ∧
  m.merill + m.elliot = m.selma - 5 ∧
  m.selma = 50

/-- Theorem stating that under the given conditions, Merill has 30 marbles -/
theorem merill_marble_count (m : MarbleCount) 
  (h : marbleProblemConditions m) : m.merill = 30 := by
  sorry


end NUMINAMATH_CALUDE_merill_marble_count_l384_38420


namespace NUMINAMATH_CALUDE_equation_solution_l384_38478

theorem equation_solution :
  ∃ x : ℝ, (x + 1 = 5) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l384_38478


namespace NUMINAMATH_CALUDE_range_of_a_satisfying_equation_l384_38409

open Real

theorem range_of_a_satisfying_equation :
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 3 * x + a * (2 * y - 4 * ℯ * x) * (log y - log x) = 0) ↔ 
  (a < 0 ∨ a ≥ 3 / (2 * ℯ)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_satisfying_equation_l384_38409


namespace NUMINAMATH_CALUDE_polygon_exterior_angle_pairs_l384_38467

def is_valid_pair (m n : ℕ) : Prop :=
  m ≥ 3 ∧ n ≥ 3 ∧ 360 / m = n ∧ 360 / n = m

theorem polygon_exterior_angle_pairs :
  ∃! (S : Finset (ℕ × ℕ)), S.card = 20 ∧ ∀ p : ℕ × ℕ, p ∈ S ↔ is_valid_pair p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angle_pairs_l384_38467


namespace NUMINAMATH_CALUDE_function_and_inequality_l384_38448

/-- Given a function f(x) = (ax+b)/(x-2) where f(x) - x + 12 = 0 has roots 3 and 4,
    prove the form of f(x) and the solution set of f(x) < k for k > 1 -/
theorem function_and_inequality (a b : ℝ) (h1 : ∀ x : ℝ, x ≠ 2 → (a * x + b) / (x - 2) - x + 12 = 0) 
    (h2 : (a * 3 + b) / (3 - 2) - 3 + 12 = 0) (h3 : (a * 4 + b) / (4 - 2) - 4 + 12 = 0) :
  (∀ x : ℝ, x ≠ 2 → (a * x + b) / (x - 2) = (-x + 2) / (x - 2)) ∧
  (∀ k : ℝ, k > 1 →
    (1 < k ∧ k < 2 → {x : ℝ | (-x + 2) / (x - 2) < k} = {x : ℝ | 1 < x ∧ x < k} ∪ {x : ℝ | x > 2}) ∧
    (k = 2 → {x : ℝ | (-x + 2) / (x - 2) < k} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | x > 2}) ∧
    (k > 2 → {x : ℝ | (-x + 2) / (x - 2) < k} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | x > k})) :=
by sorry

end NUMINAMATH_CALUDE_function_and_inequality_l384_38448


namespace NUMINAMATH_CALUDE_imaginary_part_of_fraction_l384_38462

theorem imaginary_part_of_fraction (i : ℂ) : i * i = -1 → Complex.im (5 * i / (2 - i)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_fraction_l384_38462


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l384_38479

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l384_38479


namespace NUMINAMATH_CALUDE_line_AB_slope_and_equation_l384_38404

/-- Given points A(0,-2) and B(√3,1), prove the slope of line AB is √3 and its equation is y = √3x - 2 -/
theorem line_AB_slope_and_equation :
  let A : ℝ × ℝ := (0, -2)
  let B : ℝ × ℝ := (Real.sqrt 3, 1)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let equation (x : ℝ) : ℝ := slope * x + (A.2 - slope * A.1)
  slope = Real.sqrt 3 ∧ ∀ x, equation x = Real.sqrt 3 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_line_AB_slope_and_equation_l384_38404


namespace NUMINAMATH_CALUDE_complementary_events_l384_38456

-- Define the sample space
def SampleSpace := Fin 4 × Fin 4

-- Define the events
def AtLeastOneBlack (outcome : SampleSpace) : Prop :=
  outcome.1 < 2 ∨ outcome.2 < 2

def BothRed (outcome : SampleSpace) : Prop :=
  outcome.1 ≥ 2 ∧ outcome.2 ≥ 2

-- Theorem statement
theorem complementary_events :
  ∀ (outcome : SampleSpace), AtLeastOneBlack outcome ↔ ¬BothRed outcome := by
  sorry

end NUMINAMATH_CALUDE_complementary_events_l384_38456


namespace NUMINAMATH_CALUDE_logical_equivalences_l384_38472

theorem logical_equivalences :
  (∀ A B C : Prop,
    (A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C)) ∧
    (¬((A ∨ (¬B)) ∨ (C ∧ (A ∨ (¬B)))) ↔ (¬A) ∧ B)) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalences_l384_38472


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l384_38426

theorem polynomial_evaluation : (3 : ℝ)^3 + (3 : ℝ)^2 + (3 : ℝ) + 1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l384_38426


namespace NUMINAMATH_CALUDE_fraction_simplification_l384_38427

theorem fraction_simplification : (5 * 8) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l384_38427


namespace NUMINAMATH_CALUDE_pictures_hung_vertically_l384_38443

/-- Given a total of 30 pictures, with half hung horizontally and 5 hung haphazardly,
    prove that 10 pictures are hung vertically. -/
theorem pictures_hung_vertically (total : ℕ) (horizontal : ℕ) (haphazard : ℕ) :
  total = 30 →
  horizontal = total / 2 →
  haphazard = 5 →
  total - horizontal - haphazard = 10 := by
sorry

end NUMINAMATH_CALUDE_pictures_hung_vertically_l384_38443


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_l384_38445

/-- A right triangle with sides 9, 12, and 15 units -/
structure RightTriangle where
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  is_right_triangle : side_a^2 + side_b^2 = side_c^2
  side_a_eq : side_a = 9
  side_b_eq : side_b = 12
  side_c_eq : side_c = 15

/-- A circle with radius 2 units -/
def circle_radius : ℝ := 2

/-- The inner triangle formed by the path of the circle's center -/
def inner_triangle (t : RightTriangle) : ℝ × ℝ × ℝ :=
  (t.side_a - 2 * circle_radius, t.side_b - 2 * circle_radius, t.side_c - 2 * circle_radius)

/-- Theorem: The perimeter of the inner triangle is 24 units -/
theorem inner_triangle_perimeter (t : RightTriangle) :
  let (a, b, c) := inner_triangle t
  a + b + c = 24 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_l384_38445


namespace NUMINAMATH_CALUDE_sin_difference_equals_four_l384_38415

theorem sin_difference_equals_four : 
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_equals_four_l384_38415


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l384_38473

theorem sqrt_fraction_simplification : 
  Real.sqrt ((25 : ℝ) / 36 - 4 / 9) = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l384_38473


namespace NUMINAMATH_CALUDE_ellipse_equivalence_l384_38488

/-- Given ellipse equation -/
def given_ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

/-- New ellipse equation -/
def new_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- Foci of an ellipse -/
def has_same_foci (e1 e2 : (ℝ → ℝ → Prop)) : Prop := sorry

theorem ellipse_equivalence :
  has_same_foci given_ellipse new_ellipse ∧ new_ellipse (-3) 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_equivalence_l384_38488


namespace NUMINAMATH_CALUDE_charity_event_total_is_1080_l384_38440

/-- Represents the total money raised from a charity event with raffle ticket sales and donations -/
def charity_event_total (a_price b_price c_price : ℚ) 
                        (a_sold b_sold c_sold : ℕ) 
                        (donations : List ℚ) : ℚ :=
  a_price * a_sold + b_price * b_sold + c_price * c_sold + donations.sum

/-- Theorem stating the total money raised from the charity event -/
theorem charity_event_total_is_1080 : 
  charity_event_total 3 5.5 10 100 50 25 [30, 30, 50, 45, 100] = 1080 := by
  sorry

end NUMINAMATH_CALUDE_charity_event_total_is_1080_l384_38440


namespace NUMINAMATH_CALUDE_tangent_line_determines_function_l384_38476

noncomputable def f (a b x : ℝ) : ℝ := a * x / (x^2 + b)

theorem tangent_line_determines_function (a b : ℝ) :
  (∃ x, f a b x = 2 ∧ (deriv (f a b)) x = 0) ∧ f a b 1 = 2 →
  ∀ x, f a b x = 4 * x / (x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_determines_function_l384_38476


namespace NUMINAMATH_CALUDE_jellybean_problem_l384_38497

theorem jellybean_problem :
  ∃ n : ℕ, n = 164 ∧ 
  (∀ m : ℕ, m ≥ 150 ∧ m % 15 = 14 → m ≥ n) ∧
  n ≥ 150 ∧ 
  n % 15 = 14 :=
sorry

end NUMINAMATH_CALUDE_jellybean_problem_l384_38497


namespace NUMINAMATH_CALUDE_square_root_expression_simplification_l384_38451

theorem square_root_expression_simplification :
  (2 + Real.sqrt 3)^2 - Real.sqrt 18 * Real.sqrt (2/3) = 7 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_simplification_l384_38451


namespace NUMINAMATH_CALUDE_total_edges_after_ten_cuts_l384_38433

/-- Represents the number of edges after a given number of cuts -/
def num_edges : ℕ → ℕ
| 0 => 4  -- Initial square has 4 edges
| n + 1 => num_edges n + 3  -- Each cut adds 3 edges

/-- The theorem stating that after 10 cuts, there are 34 edges in total -/
theorem total_edges_after_ten_cuts :
  num_edges 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_edges_after_ten_cuts_l384_38433


namespace NUMINAMATH_CALUDE_ellipse_from_hyperbola_vertices_l384_38487

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, 
    the equation of the ellipse whose foci are the vertices of the hyperbola 
    is x²/16 + y²/12 = 1 -/
theorem ellipse_from_hyperbola_vertices (x y : ℝ) :
  let hyperbola := (x^2 / 4 - y^2 / 12 = 1)
  let ellipse := (x^2 / 16 + y^2 / 12 = 1)
  let hyperbola_vertex := 2
  let hyperbola_focus := 4
  hyperbola → ellipse := by sorry

end NUMINAMATH_CALUDE_ellipse_from_hyperbola_vertices_l384_38487


namespace NUMINAMATH_CALUDE_cousins_distribution_l384_38406

-- Define the number of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 3

-- Function to calculate the number of ways to distribute cousins
def distribute_cousins (n : ℕ) (k : ℕ) : ℕ := sorry

-- Theorem stating the result
theorem cousins_distribution :
  distribute_cousins num_cousins num_rooms = 66 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l384_38406


namespace NUMINAMATH_CALUDE_complex_power_difference_l384_38432

theorem complex_power_difference (i : ℂ) : i * i = -1 → (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l384_38432


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l384_38423

theorem least_positive_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 2 = 0) ∧ 
  (a % 5 = 1) ∧ 
  (a % 4 = 2) ∧ 
  (∀ (b : ℕ), b > 0 ∧ b % 2 = 0 ∧ b % 5 = 1 ∧ b % 4 = 2 → a ≤ b) ∧
  (a = 6) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l384_38423


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l384_38402

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im (10 * i / (1 - 2 * i)) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l384_38402


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l384_38417

theorem inscribed_cube_surface_area (V : ℝ) (h : V = 256 * Real.pi / 3) :
  let R := (3 * V / (4 * Real.pi)) ^ (1/3)
  let a := 2 * R / Real.sqrt 3
  6 * a^2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l384_38417


namespace NUMINAMATH_CALUDE_bowl_game_points_l384_38463

/-- The total points scored by Noa and Phillip in a bowl game. -/
def total_points (noa_points phillip_points : ℕ) : ℕ := noa_points + phillip_points

/-- Theorem stating that given Noa's score and Phillip scoring twice as much,
    the total points scored by Noa and Phillip is 90. -/
theorem bowl_game_points :
  let noa_points : ℕ := 30
  let phillip_points : ℕ := 2 * noa_points
  total_points noa_points phillip_points = 90 := by
  sorry

end NUMINAMATH_CALUDE_bowl_game_points_l384_38463


namespace NUMINAMATH_CALUDE_backyard_length_is_20_l384_38469

-- Define the backyard and shed dimensions
def backyard_width : ℝ := 13
def shed_length : ℝ := 3
def shed_width : ℝ := 5
def sod_area : ℝ := 245

-- Theorem statement
theorem backyard_length_is_20 :
  ∃ (L : ℝ), L * backyard_width - shed_length * shed_width = sod_area ∧ L = 20 := by
  sorry

end NUMINAMATH_CALUDE_backyard_length_is_20_l384_38469


namespace NUMINAMATH_CALUDE_room_breadth_calculation_l384_38452

theorem room_breadth_calculation (room_length : ℝ) (carpet_width_cm : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  room_length = 18 →
  carpet_width_cm = 75 →
  cost_per_meter = 4.50 →
  total_cost = 810 →
  (total_cost / cost_per_meter) / room_length * (carpet_width_cm / 100) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_room_breadth_calculation_l384_38452


namespace NUMINAMATH_CALUDE_equality_of_positive_reals_l384_38441

theorem equality_of_positive_reals (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 - b*d) / (b + 2*c + d) + (b^2 - c*a) / (c + 2*d + a) + 
  (c^2 - d*b) / (d + 2*a + b) + (d^2 - a*c) / (a + 2*b + c) = 0 →
  a = b ∧ b = c ∧ c = d := by
sorry

end NUMINAMATH_CALUDE_equality_of_positive_reals_l384_38441


namespace NUMINAMATH_CALUDE_prob_four_blue_exact_l384_38439

-- Define the number of blue pens, red pens, and total draws
def blue_pens : ℕ := 5
def red_pens : ℕ := 4
def total_draws : ℕ := 7

-- Define the probability of picking a blue pen in a single draw
def prob_blue : ℚ := blue_pens / (blue_pens + red_pens)

-- Define the probability of picking a red pen in a single draw
def prob_red : ℚ := red_pens / (blue_pens + red_pens)

-- Define the number of ways to choose 4 blue pens out of 7 draws
def ways_to_choose : ℕ := Nat.choose total_draws 4

-- Define the probability of picking exactly 4 blue pens in 7 draws
def prob_four_blue : ℚ := ways_to_choose * (prob_blue ^ 4 * prob_red ^ 3)

-- Theorem statement
theorem prob_four_blue_exact :
  prob_four_blue = 35 * 40000 / 4782969 := by sorry

end NUMINAMATH_CALUDE_prob_four_blue_exact_l384_38439


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l384_38490

/-- Calculates the total bill for three families at Billy's Restaurant -/
theorem billys_restaurant_bill (adult_meal_cost child_meal_cost drink_cost : ℕ) 
  (family1_adults family1_children : ℕ)
  (family2_adults family2_children : ℕ)
  (family3_adults family3_children : ℕ) :
  adult_meal_cost = 8 →
  child_meal_cost = 5 →
  drink_cost = 2 →
  family1_adults = 2 →
  family1_children = 3 →
  family2_adults = 4 →
  family2_children = 2 →
  family3_adults = 3 →
  family3_children = 4 →
  (family1_adults * adult_meal_cost + family1_children * child_meal_cost + 
   (family1_adults + family1_children) * drink_cost) +
  (family2_adults * adult_meal_cost + family2_children * child_meal_cost + 
   (family2_adults + family2_children) * drink_cost) +
  (family3_adults * adult_meal_cost + family3_children * child_meal_cost + 
   (family3_adults + family3_children) * drink_cost) = 153 :=
by
  sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l384_38490


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l384_38484

theorem maintenance_check_increase (original_days : ℝ) (new_days : ℝ) 
  (h1 : original_days = 30) 
  (h2 : new_days = 45) : 
  ((new_days - original_days) / original_days) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l384_38484
