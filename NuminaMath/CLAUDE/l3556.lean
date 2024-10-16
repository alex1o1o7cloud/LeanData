import Mathlib

namespace NUMINAMATH_CALUDE_container_volume_ratio_l3556_355631

theorem container_volume_ratio (C D : ℚ) 
  (h : C > 0 ∧ D > 0) 
  (transfer : (3 / 4 : ℚ) * C = (2 / 3 : ℚ) * D) : 
  C / D = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3556_355631


namespace NUMINAMATH_CALUDE_logical_reasoning_classification_l3556_355612

-- Define the types of reasoning
inductive ReasoningType
  | Sphere
  | Triangle
  | Chair
  | Polygon

-- Define a predicate for logical reasoning
def is_logical (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Sphere => true   -- Analogy reasoning
  | ReasoningType.Triangle => true -- Inductive reasoning
  | ReasoningType.Chair => false   -- Not logical
  | ReasoningType.Polygon => true  -- Inductive reasoning

-- Theorem statement
theorem logical_reasoning_classification :
  (is_logical ReasoningType.Sphere) ∧
  (is_logical ReasoningType.Triangle) ∧
  (¬is_logical ReasoningType.Chair) ∧
  (is_logical ReasoningType.Polygon) :=
sorry

end NUMINAMATH_CALUDE_logical_reasoning_classification_l3556_355612


namespace NUMINAMATH_CALUDE_point_motion_l3556_355690

/-- Given two points A and B on a number line, prove properties about their motion and positions. -/
theorem point_motion (a b : ℝ) (h : |a + 20| + |b - 12| = 0) :
  -- 1. Initial positions
  (a = -20 ∧ b = 12) ∧ 
  -- 2. Time when A and B are equidistant from origin
  (∃ t : ℝ, t = 2 ∧ |a - 6*t| = |b - 2*t|) ∧
  -- 3. Times when A and B are 8 units apart
  (∃ t : ℝ, (t = 3 ∨ t = 5 ∨ t = 10) ∧ 
    |a - 6*t - (b - 2*t)| = 8) := by
  sorry

end NUMINAMATH_CALUDE_point_motion_l3556_355690


namespace NUMINAMATH_CALUDE_sum_abc_values_l3556_355679

theorem sum_abc_values (a b c : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1)
  (hab : b = a^2 / (2 - a^2))
  (hbc : c = b^2 / (2 - b^2))
  (hca : a = c^2 / (2 - c^2)) :
  (a + b + c = 6) ∨ (a + b + c = -4) ∨ (a + b + c = -6) := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_values_l3556_355679


namespace NUMINAMATH_CALUDE_cube_greater_than_l3556_355686

theorem cube_greater_than (a b : ℝ) : a > b → ¬(a^3 ≤ b^3) := by
  sorry

end NUMINAMATH_CALUDE_cube_greater_than_l3556_355686


namespace NUMINAMATH_CALUDE_unique_valid_number_l3556_355672

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ 
  n = (n / 10)^3 + (n % 10)^3 - 3

theorem unique_valid_number : 
  ∃! n : ℕ, is_valid_number n ∧ n = 32 := by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3556_355672


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3556_355699

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3556_355699


namespace NUMINAMATH_CALUDE_extreme_value_cubic_l3556_355629

/-- Given a cubic function f(x) = ax³ + 3x² + 3x + 3,
    if f has an extreme value at x = 1, then a = -3 -/
theorem extreme_value_cubic (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 + 3 * x + 3
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_cubic_l3556_355629


namespace NUMINAMATH_CALUDE_arithmetic_progression_cubes_l3556_355678

theorem arithmetic_progression_cubes (x y z : ℤ) : 
  x < y ∧ y < z ∧ y = (x + z) / 2 → ¬(y^3 = (x^3 + z^3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cubes_l3556_355678


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3556_355617

theorem right_triangle_sides : 
  (7^2 + 24^2 = 25^2) ∧ 
  (1.5^2 + 2^2 = 2.5^2) ∧ 
  (8^2 + 15^2 = 17^2) ∧ 
  (Real.sqrt 3)^2 + (Real.sqrt 4)^2 ≠ (Real.sqrt 5)^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3556_355617


namespace NUMINAMATH_CALUDE_swimmers_pass_count_l3556_355657

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  turnDelay : ℝ

/-- Calculates the number of times swimmers pass each other --/
def countPasses (poolLength : ℝ) (swimmer1 : Swimmer) (swimmer2 : Swimmer) (totalTime : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of passes for the given problem --/
theorem swimmers_pass_count :
  let poolLength : ℝ := 120
  let swimmer1 : Swimmer := ⟨4, 2⟩
  let swimmer2 : Swimmer := ⟨3, 0⟩
  let totalTime : ℝ := 900
  countPasses poolLength swimmer1 swimmer2 totalTime = 26 :=
by sorry

end NUMINAMATH_CALUDE_swimmers_pass_count_l3556_355657


namespace NUMINAMATH_CALUDE_average_of_numbers_divisible_by_4_l3556_355621

theorem average_of_numbers_divisible_by_4 :
  let numbers := (Finset.range 25).filter (fun n => 6 < n + 6 ∧ n + 6 ≤ 30 ∧ (n + 6) % 4 = 0)
  (numbers.sum id) / numbers.card = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_divisible_by_4_l3556_355621


namespace NUMINAMATH_CALUDE_root_intervals_l3556_355639

noncomputable def f (x : ℝ) : ℝ :=
  if x > -2 then Real.exp (x + 1) - 2
  else Real.exp (-x - 3) - 2

theorem root_intervals (e : ℝ) (h_e : e = Real.exp 1) :
  {k : ℤ | ∃ x : ℝ, f x = 0 ∧ k - 1 < x ∧ x < k} = {-4, 0} := by
  sorry

end NUMINAMATH_CALUDE_root_intervals_l3556_355639


namespace NUMINAMATH_CALUDE_bothMiss_mutually_exclusive_with_hitAtLeastOnce_bothMiss_complement_of_hitAtLeastOnce_l3556_355659

-- Define the sample space for two shots
inductive ShotOutcome
| Hit
| Miss

-- Define the type for a two-shot experiment
def TwoShots := (ShotOutcome × ShotOutcome)

-- Define the event "hitting the target at least once"
def hitAtLeastOnce (outcome : TwoShots) : Prop :=
  outcome.1 = ShotOutcome.Hit ∨ outcome.2 = ShotOutcome.Hit

-- Define the event "both shots miss"
def bothMiss (outcome : TwoShots) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

-- Theorem stating that "both shots miss" is mutually exclusive with "hitting at least once"
theorem bothMiss_mutually_exclusive_with_hitAtLeastOnce :
  ∀ (outcome : TwoShots), ¬(hitAtLeastOnce outcome ∧ bothMiss outcome) :=
sorry

-- Theorem stating that "both shots miss" is the complement of "hitting at least once"
theorem bothMiss_complement_of_hitAtLeastOnce :
  ∀ (outcome : TwoShots), hitAtLeastOnce outcome ↔ ¬(bothMiss outcome) :=
sorry

end NUMINAMATH_CALUDE_bothMiss_mutually_exclusive_with_hitAtLeastOnce_bothMiss_complement_of_hitAtLeastOnce_l3556_355659


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l3556_355667

theorem right_triangle_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (right_triangle : a^2 + b^2 = c^2) (leg_relation : a = 2*b) :
  (a + b) / c = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l3556_355667


namespace NUMINAMATH_CALUDE_inequality_proof_l3556_355636

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (7 * a^2 + b^2 + c^2)) + 
  (b / Real.sqrt (a^2 + 7 * b^2 + c^2)) + 
  (c / Real.sqrt (a^2 + b^2 + 7 * c^2)) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3556_355636


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3556_355658

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying 4a_3 + a_11 - 3a_5 = 10, prove that 1/5 * a_4 = 1 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h : 4 * seq.a 3 + seq.a 11 - 3 * seq.a 5 = 10) : 
  1/5 * seq.a 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3556_355658


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_value_l3556_355609

theorem unique_solution_implies_a_value (a : ℝ) : 
  (∃! x : ℝ, x - 1000 ≥ 1018 ∧ x + 1 ≤ a) → a = 2019 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_value_l3556_355609


namespace NUMINAMATH_CALUDE_function_value_range_l3556_355697

theorem function_value_range (x : ℝ) :
  x ∈ Set.Icc 1 4 →
  2 ≤ x^2 - 4*x + 6 ∧ x^2 - 4*x + 6 ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_function_value_range_l3556_355697


namespace NUMINAMATH_CALUDE_distinct_collections_l3556_355600

def word : String := "PHYSICS"

def num_magnets : ℕ := 7

def vowels_fallen : ℕ := 3

def consonants_fallen : ℕ := 3

def s_indistinguishable : Prop := True

theorem distinct_collections : ℕ := by
  sorry

end NUMINAMATH_CALUDE_distinct_collections_l3556_355600


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l3556_355683

/-- Given that N(4,7) is the midpoint of line segment CD and C(5,3) is one endpoint,
    prove that the product of the coordinates of point D is 33. -/
theorem midpoint_coordinate_product (D : ℝ × ℝ) : 
  let N : ℝ × ℝ := (4, 7)
  let C : ℝ × ℝ := (5, 3)
  (N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2) →
  D.1 * D.2 = 33 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l3556_355683


namespace NUMINAMATH_CALUDE_chef_potato_problem_l3556_355649

/-- The number of potatoes a chef needs to cook -/
def total_potatoes (already_cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) : ℕ :=
  already_cooked + (remaining_cooking_time / cooking_time_per_potato)

/-- Proof that the chef needs to cook 12 potatoes in total -/
theorem chef_potato_problem : 
  total_potatoes 6 6 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_chef_potato_problem_l3556_355649


namespace NUMINAMATH_CALUDE_janice_bottle_caps_l3556_355642

/-- The number of boxes available to store bottle caps -/
def num_boxes : ℕ := 79

/-- The number of bottle caps that must be in each box -/
def caps_per_box : ℕ := 4

/-- The total number of bottle caps Janice has -/
def total_caps : ℕ := num_boxes * caps_per_box

theorem janice_bottle_caps : total_caps = 316 := by
  sorry

end NUMINAMATH_CALUDE_janice_bottle_caps_l3556_355642


namespace NUMINAMATH_CALUDE_lloyd_work_hours_l3556_355698

/-- Calculates the total hours worked given regular hours, regular pay rate, overtime multiplier, and total earnings -/
def total_hours_worked (regular_hours : ℝ) (regular_rate : ℝ) (overtime_multiplier : ℝ) (total_earnings : ℝ) : ℝ :=
  sorry

/-- Theorem: Given Lloyd's work conditions, the total hours worked is 10.5 -/
theorem lloyd_work_hours :
  let regular_hours : ℝ := 7.5
  let regular_rate : ℝ := 5.5
  let overtime_multiplier : ℝ := 1.5
  let total_earnings : ℝ := 66
  total_hours_worked regular_hours regular_rate overtime_multiplier total_earnings = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_lloyd_work_hours_l3556_355698


namespace NUMINAMATH_CALUDE_toilet_paper_supply_duration_l3556_355633

/-- Calculates the number of days a toilet paper supply will last for a family -/
def toilet_paper_duration (bill_usage : ℕ) (wife_usage : ℕ) (kid_usage : ℕ) (num_kids : ℕ) (num_rolls : ℕ) (squares_per_roll : ℕ) : ℕ :=
  let total_squares := num_rolls * squares_per_roll
  let daily_usage := bill_usage + wife_usage + kid_usage * num_kids
  total_squares / daily_usage

theorem toilet_paper_supply_duration :
  toilet_paper_duration 15 32 30 2 1000 300 = 2803 := by
  sorry

end NUMINAMATH_CALUDE_toilet_paper_supply_duration_l3556_355633


namespace NUMINAMATH_CALUDE_initial_amount_A_l3556_355626

theorem initial_amount_A (a b c : ℝ) : 
  b = 28 → 
  c = 20 → 
  (a - b - c) + 2 * (a - b - c) + 4 * (a - b - c) = 24 →
  (b + b) - (2 * (a - b - c) + 2 * c) + 2 * ((b + b) - (2 * (a - b - c) + 2 * c)) = 24 →
  (c + c) - (4 * (a - b - c) + 2 * ((b + b) - (2 * (a - b - c) + 2 * c))) = 24 →
  a = 54 := by
  sorry

#check initial_amount_A

end NUMINAMATH_CALUDE_initial_amount_A_l3556_355626


namespace NUMINAMATH_CALUDE_book_magazine_cost_l3556_355618

theorem book_magazine_cost (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 18.40)
  (h2 : 2 * x + 3 * y = 17.60) :
  2 * x + y = 11.20 := by
  sorry

end NUMINAMATH_CALUDE_book_magazine_cost_l3556_355618


namespace NUMINAMATH_CALUDE_tournament_divisibility_l3556_355669

theorem tournament_divisibility (n : ℕ) : 
  let tournament_year := fun i => 1978 + i
  (tournament_year 43 = 2021) →
  (∃! k, k = 3 ∧ 
    (∀ i ∈ Finset.range k, 
      ∃ m > 43, tournament_year m % m = 0 ∧
      ∀ j ∈ Finset.Icc 44 (m - 1), tournament_year j % j ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_tournament_divisibility_l3556_355669


namespace NUMINAMATH_CALUDE_seven_digit_increasing_numbers_l3556_355630

theorem seven_digit_increasing_numbers (n : ℕ) (h : n = 7) :
  (Nat.choose (9 + n - 1) n) % 1000 = 435 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_increasing_numbers_l3556_355630


namespace NUMINAMATH_CALUDE_baseball_season_length_l3556_355684

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The total number of baseball games in a season -/
def games_in_season : ℕ := 14

/-- The number of months in a baseball season -/
def season_length : ℕ := games_in_season / games_per_month

theorem baseball_season_length :
  season_length = 2 := by sorry

end NUMINAMATH_CALUDE_baseball_season_length_l3556_355684


namespace NUMINAMATH_CALUDE_range_of_a_l3556_355674

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + a| < 3) → a ∈ Set.Ioo (-4) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3556_355674


namespace NUMINAMATH_CALUDE_accessories_total_cost_l3556_355687

def mouse_cost : ℕ := 20

def keyboard_cost (m : ℕ) : ℕ := 2 * m

def headphones_cost (m : ℕ) : ℕ := m + 15

def usb_hub_cost (m : ℕ) : ℕ := 36 - m

def total_cost (m : ℕ) : ℕ :=
  m + keyboard_cost m + headphones_cost m + usb_hub_cost m

theorem accessories_total_cost :
  total_cost mouse_cost = 111 := by sorry

end NUMINAMATH_CALUDE_accessories_total_cost_l3556_355687


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_given_pyramid_l3556_355677

/-- A right truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  height : ℝ
  volume : ℝ
  base_ratio : ℝ × ℝ

/-- The lateral surface area of a truncated pyramid -/
def lateral_surface_area (p : TruncatedPyramid) : ℝ :=
  sorry

/-- The given truncated pyramid -/
def given_pyramid : TruncatedPyramid :=
  { height := 3,
    volume := 38,
    base_ratio := (4, 9) }

/-- Theorem: The lateral surface area of the given truncated pyramid is 10√19 -/
theorem lateral_surface_area_of_given_pyramid :
  lateral_surface_area given_pyramid = 10 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_given_pyramid_l3556_355677


namespace NUMINAMATH_CALUDE_tan_alpha_for_point_on_terminal_side_l3556_355653

theorem tan_alpha_for_point_on_terminal_side (α : Real) :
  let P : ℝ × ℝ := (1, -2)
  (P.1 = 1 ∧ P.2 = -2) →  -- Point P(1, -2) lies on the terminal side of angle α
  Real.tan α = -2 :=
by sorry

end NUMINAMATH_CALUDE_tan_alpha_for_point_on_terminal_side_l3556_355653


namespace NUMINAMATH_CALUDE_area_ratio_of_concentric_circles_l3556_355605

/-- Two concentric circles with center Q -/
structure ConcentricCircles where
  center : Point
  smallerRadius : ℝ
  largerRadius : ℝ
  smallerRadius_pos : 0 < smallerRadius
  largerRadius_pos : 0 < largerRadius
  smallerRadius_lt_largerRadius : smallerRadius < largerRadius

/-- The arc length of a circle given its radius and central angle (in radians) -/
def arcLength (radius : ℝ) (angle : ℝ) : ℝ := radius * angle

theorem area_ratio_of_concentric_circles 
  (circles : ConcentricCircles) 
  (h : arcLength circles.smallerRadius (π/3) = arcLength circles.largerRadius (π/6)) : 
  (circles.smallerRadius^2) / (circles.largerRadius^2) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_of_concentric_circles_l3556_355605


namespace NUMINAMATH_CALUDE_smallest_quadratic_root_l3556_355651

theorem smallest_quadratic_root : 
  let f : ℝ → ℝ := λ y => 4 * y^2 - 7 * y + 3
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → y ≤ z ∧ y = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_quadratic_root_l3556_355651


namespace NUMINAMATH_CALUDE_math_books_arrangement_l3556_355627

theorem math_books_arrangement (num_math_books num_english_books : ℕ) : 
  num_math_books = 2 → num_english_books = 2 → 
  (num_math_books.factorial * (num_math_books + num_english_books).factorial) = 12 := by
  sorry

end NUMINAMATH_CALUDE_math_books_arrangement_l3556_355627


namespace NUMINAMATH_CALUDE_isosceles_when_neg_one_root_right_triangle_when_equal_roots_equilateral_roots_l3556_355645

/-- Triangle represented by side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- Quadratic equation associated with a triangle -/
def triangleQuadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.b) * x^2 + 2 * t.c * x + (t.b - t.a)

theorem isosceles_when_neg_one_root (t : Triangle) :
  triangleQuadratic t (-1) = 0 → t.b = t.c := by sorry

theorem right_triangle_when_equal_roots (t : Triangle) :
  (2 * t.c)^2 = 4 * (t.a + t.b) * (t.b - t.a) → t.a^2 + t.c^2 = t.b^2 := by sorry

theorem equilateral_roots (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (∃ x : ℝ, triangleQuadratic t x = 0) →
  (triangleQuadratic t 0 = 0 ∧ triangleQuadratic t (-1) = 0) := by sorry

end NUMINAMATH_CALUDE_isosceles_when_neg_one_root_right_triangle_when_equal_roots_equilateral_roots_l3556_355645


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3556_355665

/-- A circle with equation x^2 + y^2 = m is tangent to the line x - y = √m if and only if m = 0 -/
theorem circle_tangent_to_line (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x - y = Real.sqrt m) ↔ m = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3556_355665


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3556_355610

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → S n = (n : ℚ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) →
  (∀ n : ℕ, n > 0 → T n = (n : ℚ) / 2 * (2 * b 1 + (n - 1) * (b 2 - b 1))) →
  (∀ n : ℕ, n > 0 → S n / T n = (n : ℚ) / (2 * n + 1)) →
  (a 5 / b 5 = 9 / 19) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3556_355610


namespace NUMINAMATH_CALUDE_worker_assessment_correct_l3556_355640

/-- Worker's skill assessment model -/
structure WorkerAssessment where
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- Probability of ending assessment with 10 products -/
def prob_end_10 (w : WorkerAssessment) : ℝ :=
  w.p^9 * (10 - 9 * w.p)

/-- Expected value of total products produced and debugged -/
def expected_total (w : WorkerAssessment) : ℝ :=
  20 - 10 * w.p - 10 * w.p^9 + 10 * w.p^10

/-- Main theorem: Correctness of worker assessment model -/
theorem worker_assessment_correct (w : WorkerAssessment) :
  (prob_end_10 w = w.p^9 * (10 - 9 * w.p)) ∧
  (expected_total w = 20 - 10 * w.p - 10 * w.p^9 + 10 * w.p^10) := by
  sorry

end NUMINAMATH_CALUDE_worker_assessment_correct_l3556_355640


namespace NUMINAMATH_CALUDE_equation_solutions_l3556_355603

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, (2*x - 1)^2 = (3 - x)^2 ↔ x = x₁ ∨ x = x₂) ∧ x₁ = -2 ∧ x₂ = 4/3) ∧
  (∃ y₁ y₂ : ℝ, (∀ x : ℝ, x^2 - Real.sqrt 3 * x - 1/4 = 0 ↔ x = y₁ ∨ x = y₂) ∧ 
    y₁ = (Real.sqrt 3 + 2)/2 ∧ y₂ = (Real.sqrt 3 - 2)/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3556_355603


namespace NUMINAMATH_CALUDE_no_rational_roots_l3556_355694

def f (x : ℚ) : ℚ := 3 * x^4 - 2 * x^3 - 8 * x^2 + 3 * x + 1

theorem no_rational_roots : ∀ (x : ℚ), f x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l3556_355694


namespace NUMINAMATH_CALUDE_east_west_convention_l3556_355616

-- Define the direction type
inductive Direction
| West
| East

-- Define a function to convert distance and direction to a signed number
def signedDistance (dist : ℝ) (dir : Direction) : ℝ :=
  match dir with
  | Direction.West => dist
  | Direction.East => -dist

-- State the theorem
theorem east_west_convention (westDistance : ℝ) (eastDistance : ℝ) :
  westDistance > 0 →
  signedDistance westDistance Direction.West = westDistance →
  signedDistance eastDistance Direction.East = -eastDistance :=
by
  sorry

-- Example with the given values
example : signedDistance 3 Direction.East = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_east_west_convention_l3556_355616


namespace NUMINAMATH_CALUDE_least_three_digit_7_shifty_l3556_355625

def is_7_shifty (n : ℕ) : Prop := n % 7 > 2

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_shifty : 
  (∀ m : ℕ, is_three_digit m → is_7_shifty m → 101 ≤ m) ∧ 
  is_three_digit 101 ∧ 
  is_7_shifty 101 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_7_shifty_l3556_355625


namespace NUMINAMATH_CALUDE_shoe_discount_ratio_l3556_355648

theorem shoe_discount_ratio (price1 price2 final_price : ℚ) : 
  price1 = 40 →
  price2 = 60 →
  final_price = 60 →
  let total := price1 + price2
  let extra_discount := total / 4
  let discounted_total := total - extra_discount
  let cheaper_discount := discounted_total - final_price
  (cheaper_discount / price1) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_shoe_discount_ratio_l3556_355648


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l3556_355611

theorem triangle_abc_problem (A B C : ℝ) (a b c : ℝ) :
  b * Real.sin A = 3 * c * Real.sin B →
  a = 3 →
  Real.cos B = 2/3 →
  b = Real.sqrt 6 ∧ 
  Real.sin (2*B - π/3) = (4*Real.sqrt 5 + Real.sqrt 3) / 18 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l3556_355611


namespace NUMINAMATH_CALUDE_value_of_a_l3556_355644

theorem value_of_a (A B : Set ℕ) (a : ℕ) :
  A = {a, 2} →
  B = {1, 2} →
  A ∪ B = {1, 2, 3} →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3556_355644


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l3556_355628

theorem five_dice_not_same_probability :
  let n_faces : ℕ := 6
  let n_dice : ℕ := 5
  let total_outcomes : ℕ := n_faces ^ n_dice
  let same_number_outcomes : ℕ := n_faces
  (1 : ℚ) - (same_number_outcomes : ℚ) / total_outcomes = 1295 / 1296 :=
by sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l3556_355628


namespace NUMINAMATH_CALUDE_closest_fraction_is_one_fourth_l3556_355691

def total_medals : ℕ := 150
def won_medals : ℕ := 38

theorem closest_fraction_is_one_fourth :
  let fraction := won_medals / total_medals
  ∀ x ∈ ({1/3, 1/5, 1/6, 1/7} : Set ℚ),
    |fraction - (1/4 : ℚ)| ≤ |fraction - x| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_is_one_fourth_l3556_355691


namespace NUMINAMATH_CALUDE_prob_grad_degree_is_three_nineteenths_l3556_355671

/-- Represents the ratio of two quantities -/
structure Ratio :=
  (antecedent : ℕ)
  (consequent : ℕ)

/-- Represents the company's employee composition -/
structure Company :=
  (grad_ratio : Ratio)     -- Ratio of graduates with graduate degree to non-graduates
  (nongrad_ratio : Ratio)  -- Ratio of graduates without graduate degree to non-graduates

/-- Calculates the probability of a randomly picked college graduate having a graduate degree -/
def probability_grad_degree (c : Company) : ℚ :=
  let lcm := Nat.lcm c.grad_ratio.consequent c.nongrad_ratio.consequent
  let grad_scaled := c.grad_ratio.antecedent * (lcm / c.grad_ratio.consequent)
  let nongrad_scaled := c.nongrad_ratio.antecedent * (lcm / c.nongrad_ratio.consequent)
  grad_scaled / (grad_scaled + nongrad_scaled)

/-- The main theorem to be proved -/
theorem prob_grad_degree_is_three_nineteenths :
  ∀ c : Company,
    c.grad_ratio = ⟨1, 8⟩ →
    c.nongrad_ratio = ⟨2, 3⟩ →
    probability_grad_degree c = 3 / 19 :=
by
  sorry


end NUMINAMATH_CALUDE_prob_grad_degree_is_three_nineteenths_l3556_355671


namespace NUMINAMATH_CALUDE_greatest_second_term_arithmetic_sequence_l3556_355602

theorem greatest_second_term_arithmetic_sequence :
  ∀ (a d : ℕ),
    a > 0 →
    d > 0 →
    a + (a + d) + (a + 2*d) + (a + 3*d) = 80 →
    ∀ (b e : ℕ),
      b > 0 →
      e > 0 →
      b + (b + e) + (b + 2*e) + (b + 3*e) = 80 →
      a + d ≤ 19 :=
by sorry

end NUMINAMATH_CALUDE_greatest_second_term_arithmetic_sequence_l3556_355602


namespace NUMINAMATH_CALUDE_pencil_sharpening_ishas_pencil_l3556_355652

/-- The length sharpened off a pencil is equal to the difference between
    the original length and the new length after sharpening. -/
theorem pencil_sharpening (original_length new_length : ℝ) :
  original_length ≥ new_length →
  original_length - new_length = original_length - new_length :=
by
  sorry

/-- Isha's pencil problem -/
theorem ishas_pencil :
  let original_length : ℝ := 31
  let new_length : ℝ := 14
  original_length - new_length = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_ishas_pencil_l3556_355652


namespace NUMINAMATH_CALUDE_dina_machine_l3556_355693

def f (x : ℚ) : ℚ := 2 * x - 3

theorem dina_machine (x : ℚ) : f (f x) = -35 → x = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_dina_machine_l3556_355693


namespace NUMINAMATH_CALUDE_right_triangle_and_symmetric_circle_l3556_355650

/-- Given a right triangle OAB in a rectangular coordinate system where:
  - O is the origin (0, 0)
  - A is the right-angle vertex at (4, -3)
  - |AB| = 2|OA|
  - The y-coordinate of B is positive
This theorem proves the coordinates of B and the equation of a symmetric circle. -/
theorem right_triangle_and_symmetric_circle :
  ∃ (B : ℝ × ℝ),
    let O : ℝ × ℝ := (0, 0)
    let A : ℝ × ℝ := (4, -3)
    -- B is in the first quadrant
    B.1 > 0 ∧ B.2 > 0 ∧
    -- OA ⟂ AB (right angle at A)
    (B.1 - A.1) * A.1 + (B.2 - A.2) * A.2 = 0 ∧
    -- |AB| = 2|OA|
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 4 * (A.1^2 + A.2^2) ∧
    -- B has coordinates (10, 5)
    B = (10, 5) ∧
    -- The equation of the symmetric circle
    ∀ (x y : ℝ),
      (x^2 - 6*x + y^2 + 2*y = 0) ↔
      ((x - 1)^2 + (y - 3)^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_and_symmetric_circle_l3556_355650


namespace NUMINAMATH_CALUDE_cos_two_x_value_l3556_355654

theorem cos_two_x_value (x : ℝ) 
  (h1 : x ∈ Set.Ioo (-3 * π / 4) (π / 4))
  (h2 : Real.cos (π / 4 - x) = -3 / 5) : 
  Real.cos (2 * x) = -24 / 25 := by
sorry

end NUMINAMATH_CALUDE_cos_two_x_value_l3556_355654


namespace NUMINAMATH_CALUDE_least_integer_divisible_by_four_primes_l3556_355663

theorem least_integer_divisible_by_four_primes : 
  ∃ n : ℕ, (n > 0) ∧ 
  (∃ p q r s : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
   p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
   n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧ n % s = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ p q r s : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
     p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
     m % p = 0 ∧ m % q = 0 ∧ m % r = 0 ∧ m % s = 0) → 
    m ≥ 210) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_divisible_by_four_primes_l3556_355663


namespace NUMINAMATH_CALUDE_select_students_problem_l3556_355623

/-- The number of ways to select students for a meeting -/
def select_students (num_boys num_girls : ℕ) (total_selected : ℕ) (min_boys min_girls : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to select students for the given conditions -/
theorem select_students_problem : select_students 5 4 4 2 1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_select_students_problem_l3556_355623


namespace NUMINAMATH_CALUDE_cube_volume_doubling_l3556_355666

theorem cube_volume_doubling (original_volume : ℝ) (new_volume : ℝ) : 
  original_volume = 216 →
  new_volume = (2 * original_volume^(1/3))^3 →
  new_volume = 1728 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_doubling_l3556_355666


namespace NUMINAMATH_CALUDE_exists_m_eq_power_plus_n_l3556_355675

/-- n(m) denotes the number of factors of 2 in m! -/
def n (m : ℕ+) : ℕ := sorry

/-- Theorem: There exists a natural number m > 2006^2006 such that m = 3^2006 + n(m) -/
theorem exists_m_eq_power_plus_n : ∃ m : ℕ+, 
  (m : ℕ) > 2006^2006 ∧ (m : ℕ) = 3^2006 + n m := by
  sorry

end NUMINAMATH_CALUDE_exists_m_eq_power_plus_n_l3556_355675


namespace NUMINAMATH_CALUDE_cube_sum_greater_than_mixed_products_l3556_355692

theorem cube_sum_greater_than_mixed_products {a b : ℝ} (ha : a > 0) (hb : b > 0) (hnq : a ≠ b) :
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_greater_than_mixed_products_l3556_355692


namespace NUMINAMATH_CALUDE_jamie_ball_collection_l3556_355662

/-- Calculates the total number of balls Jamie has after all transactions --/
def total_balls (initial_red : ℕ) (blue_multiplier : ℕ) (lost_red : ℕ) (yellow_multiplier : ℕ) : ℕ :=
  let initial_blue := initial_red * blue_multiplier
  let remaining_red := initial_red - lost_red
  let bought_yellow := lost_red * yellow_multiplier
  remaining_red + initial_blue + bought_yellow

theorem jamie_ball_collection :
  total_balls 16 2 6 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_jamie_ball_collection_l3556_355662


namespace NUMINAMATH_CALUDE_point_A_not_in_square_l3556_355607

-- Define the points
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (0, -4)
def C : ℝ × ℝ := (-2, -1)
def D : ℝ × ℝ := (1, 1)
def E : ℝ × ℝ := (3, -2)

-- Define a function to calculate the squared distance between two points
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define what it means for four points to form a square
def is_square (p q r s : ℝ × ℝ) : Prop :=
  let sides := [squared_distance p q, squared_distance q r, squared_distance r s, squared_distance s p]
  let diagonals := [squared_distance p r, squared_distance q s]
  (sides.all (· = sides.head!)) ∧ (diagonals.all (· = 2 * sides.head!))

-- Theorem statement
theorem point_A_not_in_square :
  ¬(is_square A B C D ∨ is_square A B C E ∨ is_square A B D E ∨ is_square A C D E) ∧
  (is_square B C D E) := by sorry

end NUMINAMATH_CALUDE_point_A_not_in_square_l3556_355607


namespace NUMINAMATH_CALUDE_bryans_books_l3556_355685

/-- Given that Bryan has 9 bookshelves and each bookshelf contains 56 books,
    prove that the total number of books Bryan has is 504. -/
theorem bryans_books (num_shelves : ℕ) (books_per_shelf : ℕ) 
    (h1 : num_shelves = 9) (h2 : books_per_shelf = 56) : 
    num_shelves * books_per_shelf = 504 := by
  sorry

end NUMINAMATH_CALUDE_bryans_books_l3556_355685


namespace NUMINAMATH_CALUDE_jordan_fourth_period_blocks_l3556_355696

/-- Represents the number of shots blocked by a hockey goalie in each period of a game --/
structure GoalieBlocks where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the total number of shots blocked in a game --/
def totalBlocks (blocks : GoalieBlocks) : ℕ :=
  blocks.first + blocks.second + blocks.third + blocks.fourth

/-- Theorem: Given the conditions of Jordan's game, he blocked 4 shots in the fourth period --/
theorem jordan_fourth_period_blocks :
  ∀ (blocks : GoalieBlocks),
    blocks.first = 4 →
    blocks.second = 2 * blocks.first →
    blocks.third = blocks.second - 3 →
    totalBlocks blocks = 21 →
    blocks.fourth = 4 := by
  sorry


end NUMINAMATH_CALUDE_jordan_fourth_period_blocks_l3556_355696


namespace NUMINAMATH_CALUDE_watermelon_theorem_l3556_355614

def watermelon_problem (initial_watermelons : ℕ) (consumption_pattern : List ℕ) : Prop :=
  let total_consumption := consumption_pattern.sum
  let complete_cycles := initial_watermelons / total_consumption
  let remaining_watermelons := initial_watermelons % total_consumption
  complete_cycles * consumption_pattern.length = 3 ∧
  remaining_watermelons < consumption_pattern.head!

theorem watermelon_theorem :
  watermelon_problem 30 [7, 8, 9] :=
by sorry

end NUMINAMATH_CALUDE_watermelon_theorem_l3556_355614


namespace NUMINAMATH_CALUDE_hole_filling_problem_l3556_355680

/-- The amount of additional water needed to fill a hole -/
def additional_water_needed (total_water : ℕ) (initial_water : ℕ) : ℕ :=
  total_water - initial_water

/-- Theorem stating the additional water needed to fill the hole -/
theorem hole_filling_problem (total_water : ℕ) (initial_water : ℕ)
    (h1 : total_water = 823)
    (h2 : initial_water = 676) :
    additional_water_needed total_water initial_water = 147 := by
  sorry

end NUMINAMATH_CALUDE_hole_filling_problem_l3556_355680


namespace NUMINAMATH_CALUDE_line_and_circle_equations_l3556_355604

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4*x - 3*y - 5 = 0
def line3 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the intersection point of line1 and line2
def intersection_point : ℝ × ℝ := (2, 1)

-- Define line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Main theorem
theorem line_and_circle_equations :
  ∃ (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop),
    -- l passes through the intersection of line1 and line2
    (l (intersection_point.1) (intersection_point.2)) ∧
    -- l is perpendicular to line3
    (∀ x y, l x y → line3 x y → (x + 1 = y)) ∧
    -- C passes through (1,0)
    (C 1 0) ∧
    -- Center of C is on positive x-axis
    (∃ a > 0, ∀ x y, C x y ↔ (x - a)^2 + y^2 = a^2) ∧
    -- Chord intercepted by l on C has length 2√2
    (∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) ∧
    -- l is the line y = x - 1
    (∀ x y, l x y ↔ line_l x y) ∧
    -- C is the circle (x-3)^2 + y^2 = 4
    (∀ x y, C x y ↔ circle_C x y) :=
by sorry

end NUMINAMATH_CALUDE_line_and_circle_equations_l3556_355604


namespace NUMINAMATH_CALUDE_divisibility_property_l3556_355664

theorem divisibility_property (p m n : ℕ) : 
  Nat.Prime p → 
  p % 2 = 1 →
  m > 1 → 
  n > 0 → 
  Nat.Prime ((m^(p*n) - 1) / (m^n - 1)) → 
  (p * n) ∣ ((p - 1)^n + 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l3556_355664


namespace NUMINAMATH_CALUDE_nested_cube_root_l3556_355619

theorem nested_cube_root (M : ℝ) (h : M > 1) : 
  (M * (M * (M * M^(1/3))^(1/3))^(1/3))^(1/3) = M^(40/81) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_root_l3556_355619


namespace NUMINAMATH_CALUDE_circle_tangency_and_intersection_l3556_355655

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O₂ (x y r : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = r^2

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 1 - 2 * Real.sqrt 2 = 0

-- Define the theorem
theorem circle_tangency_and_intersection :
  (∀ x y : ℝ, circle_O₁ x y → ¬circle_O₂ x y (Real.sqrt (12 - 8 * Real.sqrt 2))) →
  (∀ x y : ℝ, circle_O₁ x y → circle_O₂ x y (Real.sqrt (12 - 8 * Real.sqrt 2)) → tangent_line x y) ∧
  (∃ A B : ℝ × ℝ, 
    circle_O₁ A.1 A.2 ∧ circle_O₁ B.1 B.2 ∧
    circle_O₂ A.1 A.2 2 ∧ circle_O₂ B.1 B.2 2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) →
  (∀ x y : ℝ, circle_O₂ x y 2 ∨ circle_O₂ x y (Real.sqrt 20)) :=
sorry

end NUMINAMATH_CALUDE_circle_tangency_and_intersection_l3556_355655


namespace NUMINAMATH_CALUDE_touchdown_points_l3556_355635

theorem touchdown_points (total_points : ℕ) (num_touchdowns : ℕ) (points_per_touchdown : ℕ) :
  total_points = 21 →
  num_touchdowns = 3 →
  total_points = num_touchdowns * points_per_touchdown →
  points_per_touchdown = 7 := by
sorry

end NUMINAMATH_CALUDE_touchdown_points_l3556_355635


namespace NUMINAMATH_CALUDE_urea_formation_proof_l3556_355676

-- Define the chemical species
inductive Species
  | NH3
  | CO2
  | H2O
  | NH4_2CO3
  | NH4OH
  | NH2CONH2

-- Define the reaction equations
inductive Reaction
  | ammonium_carbonate_formation
  | ammonium_carbonate_hydrolysis
  | urea_formation

-- Define the initial quantities
def initial_quantities : Species → ℚ
  | Species.NH3 => 2
  | Species.CO2 => 1
  | Species.H2O => 1
  | _ => 0

-- Define the stoichiometric coefficients for each reaction
def stoichiometry : Reaction → Species → ℚ
  | Reaction.ammonium_carbonate_formation, Species.NH3 => -2
  | Reaction.ammonium_carbonate_formation, Species.CO2 => -1
  | Reaction.ammonium_carbonate_formation, Species.NH4_2CO3 => 1
  | Reaction.ammonium_carbonate_hydrolysis, Species.NH4_2CO3 => -1
  | Reaction.ammonium_carbonate_hydrolysis, Species.H2O => -1
  | Reaction.ammonium_carbonate_hydrolysis, Species.NH4OH => 2
  | Reaction.ammonium_carbonate_hydrolysis, Species.CO2 => 1
  | Reaction.urea_formation, Species.NH4OH => -1
  | Reaction.urea_formation, Species.CO2 => -1
  | Reaction.urea_formation, Species.NH2CONH2 => 1
  | Reaction.urea_formation, Species.H2O => 1
  | _, _ => 0

-- Define the function to calculate the amount of Urea formed
def urea_formed (reactions : List Reaction) : ℚ :=
  sorry

-- Theorem statement
theorem urea_formation_proof :
  urea_formed [Reaction.ammonium_carbonate_formation,
               Reaction.ammonium_carbonate_hydrolysis,
               Reaction.urea_formation] = 1 :=
sorry

end NUMINAMATH_CALUDE_urea_formation_proof_l3556_355676


namespace NUMINAMATH_CALUDE_field_goal_missed_fraction_l3556_355632

theorem field_goal_missed_fraction 
  (total_attempts : ℕ) 
  (wide_right_percentage : ℚ) 
  (wide_right_count : ℕ) 
  (h1 : total_attempts = 60) 
  (h2 : wide_right_percentage = 1/5) 
  (h3 : wide_right_count = 3) : 
  (wide_right_count / wide_right_percentage) / total_attempts = 1/4 :=
sorry

end NUMINAMATH_CALUDE_field_goal_missed_fraction_l3556_355632


namespace NUMINAMATH_CALUDE_scientific_notation_2310000_l3556_355660

theorem scientific_notation_2310000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2310000 = a * (10 : ℝ) ^ n ∧ a = 2.31 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_2310000_l3556_355660


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3556_355647

/-- A regular polygon is a polygon with all sides of equal length. -/
structure RegularPolygon where
  side_length : ℝ
  num_sides : ℕ
  perimeter : ℝ
  perimeter_eq : perimeter = side_length * num_sides

/-- Theorem: A regular polygon with side length 16 cm and perimeter 80 cm has 5 sides. -/
theorem regular_polygon_sides (p : RegularPolygon) 
  (h1 : p.side_length = 16) 
  (h2 : p.perimeter = 80) : 
  p.num_sides = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3556_355647


namespace NUMINAMATH_CALUDE_inequality_proof_l3556_355689

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h4 : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3556_355689


namespace NUMINAMATH_CALUDE_pen_ratio_theorem_l3556_355638

/-- Represents the number of pens bought by each person -/
structure PenPurchase where
  julia : ℕ
  dorothy : ℕ
  robert : ℕ

/-- Represents the given conditions of the problem -/
def ProblemConditions (p : PenPurchase) : Prop :=
  p.dorothy = p.julia / 2 ∧
  p.robert = 4 ∧
  p.julia + p.dorothy + p.robert = 22

theorem pen_ratio_theorem (p : PenPurchase) :
  ProblemConditions p → p.julia / p.robert = 3 := by
  sorry

#check pen_ratio_theorem

end NUMINAMATH_CALUDE_pen_ratio_theorem_l3556_355638


namespace NUMINAMATH_CALUDE_circle_line_theorem_l3556_355643

/-- Given two circles C₁ and C₂ passing through (2, -1), prove that the line
    through (D₁, E₁) and (D₂, E₂) has equation 2x - y + 2 = 0 -/
theorem circle_line_theorem (D₁ E₁ D₂ E₂ : ℝ) : 
  (2^2 + (-1)^2 + 2*D₁ - E₁ - 3 = 0) →
  (2^2 + (-1)^2 + 2*D₂ - E₂ - 3 = 0) →
  ∃ (k : ℝ), 2*D₁ - E₁ + 2 = k ∧ 2*D₂ - E₂ + 2 = k :=
by sorry

end NUMINAMATH_CALUDE_circle_line_theorem_l3556_355643


namespace NUMINAMATH_CALUDE_nilpotent_matrix_cube_zero_l3556_355681

theorem nilpotent_matrix_cube_zero
  (A : Matrix (Fin 3) (Fin 3) ℝ)
  (h : A ^ 4 = 0) :
  A ^ 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_cube_zero_l3556_355681


namespace NUMINAMATH_CALUDE_abc_value_l3556_355695

theorem abc_value (a b c : ℂ) 
  (h1 : a * b + 5 * b = -20)
  (h2 : b * c + 5 * c = -20)
  (h3 : c * a + 5 * a = -20) : 
  a * b * c = -100 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l3556_355695


namespace NUMINAMATH_CALUDE_universiade_volunteer_count_l3556_355608

/-- Represents the result of a stratified sampling by gender -/
structure StratifiedSample where
  total_pool : ℕ
  selected_group : ℕ
  selected_male : ℕ
  selected_female : ℕ

/-- Calculates the number of female students in the pool based on stratified sampling -/
def femaleInPool (sample : StratifiedSample) : ℕ :=
  (sample.selected_female * sample.total_pool) / sample.selected_group

theorem universiade_volunteer_count :
  ∀ (sample : StratifiedSample),
    sample.total_pool = 200 →
    sample.selected_group = 30 →
    sample.selected_male = 12 →
    sample.selected_female = sample.selected_group - sample.selected_male →
    femaleInPool sample = 120 := by
  sorry

#eval femaleInPool { total_pool := 200, selected_group := 30, selected_male := 12, selected_female := 18 }

end NUMINAMATH_CALUDE_universiade_volunteer_count_l3556_355608


namespace NUMINAMATH_CALUDE_min_distance_sliding_ruler_l3556_355637

/-- The minimum distance between a point and the endpoint of a sliding ruler -/
theorem min_distance_sliding_ruler (h s : ℝ) (h_pos : h > 0) (s_pos : s > 0) (h_gt_s : h > s) :
  let min_distance := Real.sqrt (h^2 - s^2)
  ∀ (distance : ℝ), distance ≥ min_distance :=
sorry

end NUMINAMATH_CALUDE_min_distance_sliding_ruler_l3556_355637


namespace NUMINAMATH_CALUDE_sandwich_count_l3556_355646

/-- Represents the number of days in the workweek -/
def workweek_days : ℕ := 6

/-- Represents the cost of a donut in cents -/
def donut_cost : ℕ := 80

/-- Represents the cost of a sandwich in cents -/
def sandwich_cost : ℕ := 120

/-- Represents the condition that the total expenditure is an exact number of dollars -/
def is_exact_dollar_amount (sandwiches : ℕ) : Prop :=
  ∃ (dollars : ℕ), sandwich_cost * sandwiches + donut_cost * (workweek_days - sandwiches) = 100 * dollars

theorem sandwich_count : 
  ∃! (sandwiches : ℕ), sandwiches ≤ workweek_days ∧ is_exact_dollar_amount sandwiches ∧ sandwiches = 3 :=
sorry

end NUMINAMATH_CALUDE_sandwich_count_l3556_355646


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l3556_355624

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_of_digit_factorials (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.map factorial |> List.sum

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = sum_of_digit_factorials n :=
by
  use 145
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l3556_355624


namespace NUMINAMATH_CALUDE_sequence_closed_form_l3556_355668

theorem sequence_closed_form (a : ℕ → ℤ) :
  a 1 = 0 ∧
  (∀ n : ℕ, n ≥ 2 → a n - 2 * a (n - 1) = n^2 - 3) →
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n + 2) - n^2 - 4*n - 3 :=
by sorry

end NUMINAMATH_CALUDE_sequence_closed_form_l3556_355668


namespace NUMINAMATH_CALUDE_fractional_equation_range_l3556_355620

theorem fractional_equation_range (x m : ℝ) : 
  (x / (x - 1) = m / (2 * x - 2) + 3) →
  (x ≥ 0) →
  (m ≤ 6 ∧ m ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_range_l3556_355620


namespace NUMINAMATH_CALUDE_eq1_represents_parallel_lines_eq2_represents_four_lines_eq3_represents_specific_lines_eq4_represents_half_circle_l3556_355641

-- Define the equations
def eq1 (x y : ℝ) : Prop := (2*x - y)^2 = 1
def eq2 (x y : ℝ) : Prop := 16*x^4 - 8*x^2*y^2 + y^4 - 8*x^2 - 2*y^2 + 1 = 0
def eq3 (x y : ℝ) : Prop := x^2*(1 - abs y / y) + y^2 + y*(abs y) = 8
def eq4 (x y : ℝ) : Prop := x^2 + x*(abs x) + y^2 + (abs x)*y^2/x = 8

-- Define geometric shapes
def ParallelLines (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a b c d : ℝ, ∀ x y : ℝ, f x y ↔ (y = a*x + b ∨ y = c*x + d) ∧ a = c ∧ b ≠ d

def FourLines (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a₁ b₁ a₂ b₂ a₃ b₃ a₄ b₄ : ℝ, ∀ x y : ℝ, 
    f x y ↔ (y = a₁*x + b₁ ∨ y = a₂*x + b₂ ∨ y = a₃*x + b₃ ∨ y = a₄*x + b₄)

def SpecificLines (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a b c : ℝ, ∀ x y : ℝ, 
    f x y ↔ ((y > 0 ∧ y = a) ∨ (y < 0 ∧ (x = b ∨ x = c)))

def HalfCircle (f : ℝ → ℝ → Prop) : Prop := 
  ∃ r : ℝ, ∀ x y : ℝ, f x y ↔ x > 0 ∧ x^2 + y^2 = r^2

-- Theorem statements
theorem eq1_represents_parallel_lines : ParallelLines eq1 := sorry

theorem eq2_represents_four_lines : FourLines eq2 := sorry

theorem eq3_represents_specific_lines : SpecificLines eq3 := sorry

theorem eq4_represents_half_circle : HalfCircle eq4 := sorry

end NUMINAMATH_CALUDE_eq1_represents_parallel_lines_eq2_represents_four_lines_eq3_represents_specific_lines_eq4_represents_half_circle_l3556_355641


namespace NUMINAMATH_CALUDE_tank_capacity_theorem_l3556_355601

/-- Represents a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating the relationship between tank properties and its capacity. -/
theorem tank_capacity_theorem (t : Tank) 
  (h1 : t.leak_empty_time = 6)
  (h2 : t.inlet_rate = 2.5 * 60)
  (h3 : t.combined_empty_time = 8) :
  t.capacity = 3600 / 7 := by
  sorry

#check tank_capacity_theorem

end NUMINAMATH_CALUDE_tank_capacity_theorem_l3556_355601


namespace NUMINAMATH_CALUDE_february_first_is_monday_l3556_355622

/-- Represents the days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the month of February in a specific year -/
structure February where
  /-- The day of the week that February 1 falls on -/
  first_day : Weekday
  /-- The number of days in February -/
  num_days : Nat
  /-- The number of Mondays in February -/
  num_mondays : Nat
  /-- The number of Thursdays in February -/
  num_thursdays : Nat

/-- Theorem stating that if February has exactly four Mondays and four Thursdays,
    then February 1 must fall on a Monday -/
theorem february_first_is_monday (feb : February) 
  (h1 : feb.num_mondays = 4)
  (h2 : feb.num_thursdays = 4) : 
  feb.first_day = Weekday.Monday := by
  sorry


end NUMINAMATH_CALUDE_february_first_is_monday_l3556_355622


namespace NUMINAMATH_CALUDE_specific_polyhedron_volume_l3556_355682

/-- Represents a polygon in the figure -/
inductive Polygon
| IsoscelesRightTriangle
| Square
| EquilateralTriangle

/-- Represents the figure that can be folded into a polyhedron -/
structure Figure where
  polygons : List Polygon
  can_fold_to_polyhedron : Bool

/-- Calculates the volume of the polyhedron formed by folding the figure -/
def polyhedron_volume (fig : Figure) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume :
  ∃ (fig : Figure),
    fig.polygons = [Polygon.IsoscelesRightTriangle, Polygon.IsoscelesRightTriangle, Polygon.IsoscelesRightTriangle,
                    Polygon.Square, Polygon.Square, Polygon.Square,
                    Polygon.EquilateralTriangle] ∧
    fig.can_fold_to_polyhedron = true ∧
    polyhedron_volume fig = 8 - (2 * Real.sqrt 2) / 3 :=
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_volume_l3556_355682


namespace NUMINAMATH_CALUDE_max_wrong_questions_l3556_355634

theorem max_wrong_questions (total_questions : ℕ) (success_threshold : ℚ) : 
  total_questions = 50 → success_threshold = 85 / 100 → 
  ∃ max_wrong : ℕ, max_wrong = 7 ∧ 
  (↑(total_questions - max_wrong) / ↑total_questions ≥ success_threshold) ∧
  ∀ wrong : ℕ, wrong > max_wrong → 
  (↑(total_questions - wrong) / ↑total_questions < success_threshold) :=
by
  sorry

end NUMINAMATH_CALUDE_max_wrong_questions_l3556_355634


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3556_355661

theorem repeating_decimal_sum : 
  (234 : ℚ) / 999 - (567 : ℚ) / 999 + (891 : ℚ) / 999 = (186 : ℚ) / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3556_355661


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3556_355606

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ

/-- Calculates the shortest distance between two points on a cone's surface -/
def shortestDistanceOnCone (cone : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem shortest_distance_on_specific_cone :
  let cone : Cone := { baseRadius := 500, height := 300 * Real.sqrt 3 }
  let p1 : ConePoint := { distanceFromVertex := 150, angle := 0 }
  let p2 : ConePoint := { distanceFromVertex := 450 * Real.sqrt 2, angle := 5 * Real.pi / Real.sqrt 52 }
  shortestDistanceOnCone cone p1 p2 = 450 * Real.sqrt 2 - 150 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3556_355606


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3556_355656

theorem binomial_expansion_coefficient (x : ℝ) : 
  let expansion := (x - 2 / Real.sqrt x) ^ 7
  ∃ (a b c : ℝ), expansion = a*x + 560*x + b*x^2 + c :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3556_355656


namespace NUMINAMATH_CALUDE_carson_gold_stars_l3556_355673

/-- Represents the number of gold stars Carson earned today -/
def gold_stars_today (yesterday : ℕ) (total : ℕ) : ℕ :=
  total - yesterday

theorem carson_gold_stars : gold_stars_today 6 15 = 9 := by
  sorry

end NUMINAMATH_CALUDE_carson_gold_stars_l3556_355673


namespace NUMINAMATH_CALUDE_parabola_equation_l3556_355615

/-- Given a parabola y^2 = mx (m > 0) whose directrix is at a distance of 3 from the line x = 1,
    prove that the equation of the parabola is y^2 = 8x. -/
theorem parabola_equation (m : ℝ) (h_m_pos : m > 0) : 
  (∃ (k : ℝ), k = -m/4 ∧ |k - 1| = 3) → 
  (∀ (x y : ℝ), y^2 = m*x ↔ y^2 = 8*x) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3556_355615


namespace NUMINAMATH_CALUDE_crew_member_count_l3556_355613

/-- The number of crew members working on all islands in a country -/
def total_crew_members (num_islands : ℕ) (ships_per_island : ℕ) (crew_per_ship : ℕ) : ℕ :=
  num_islands * ships_per_island * crew_per_ship

/-- Theorem stating the total number of crew members in the given scenario -/
theorem crew_member_count :
  total_crew_members 3 12 24 = 864 := by
  sorry

end NUMINAMATH_CALUDE_crew_member_count_l3556_355613


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3556_355688

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3556_355688


namespace NUMINAMATH_CALUDE_total_bathing_suits_l3556_355670

def one_piece : ℕ := 8500
def two_piece : ℕ := 12750
def trunks : ℕ := 5900
def shorts : ℕ := 7250
def children : ℕ := 1100

theorem total_bathing_suits :
  one_piece + two_piece + trunks + shorts + children = 35500 := by
  sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l3556_355670
