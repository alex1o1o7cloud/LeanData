import Mathlib

namespace NUMINAMATH_CALUDE_egg_problem_l3764_376468

theorem egg_problem (x : ℕ) : x > 0 ∧ 
  x % 2 = 1 ∧ 
  x % 3 = 1 ∧ 
  x % 4 = 1 ∧ 
  x % 5 = 1 ∧ 
  x % 6 = 1 ∧ 
  x % 7 = 0 → 
  x ≥ 301 :=
by sorry

end NUMINAMATH_CALUDE_egg_problem_l3764_376468


namespace NUMINAMATH_CALUDE_sum_and_product_equality_l3764_376416

theorem sum_and_product_equality : 2357 + 3572 + 5723 + 7235 * 2 = 26122 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_equality_l3764_376416


namespace NUMINAMATH_CALUDE_range_of_a_l3764_376407

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 = 0) ∧ 
  (∀ x : ℝ, x^2 - 2*x + a > 0) →
  a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3764_376407


namespace NUMINAMATH_CALUDE_folding_punching_theorem_l3764_376423

/-- Represents a rectangular piece of paper --/
structure Paper where
  width : ℝ
  height : ℝ
  (width_pos : width > 0)
  (height_pos : height > 0)

/-- Represents a fold operation on the paper --/
inductive Fold
  | BottomToTop
  | RightToHalfLeft
  | DiagonalBottomLeftToTopRight

/-- Represents a hole punched in the paper --/
structure Hole where
  x : ℝ
  y : ℝ

/-- Applies a sequence of folds to a paper --/
def applyFolds (p : Paper) (folds : List Fold) : Paper :=
  sorry

/-- Punches a hole in the folded paper --/
def punchHole (p : Paper) : Hole :=
  sorry

/-- Unfolds the paper and calculates the resulting hole pattern --/
def unfoldAndGetHoles (p : Paper) (folds : List Fold) (h : Hole) : List Hole :=
  sorry

/-- Checks if a list of holes is symmetric around the center and along two diagonals --/
def isSymmetricPattern (holes : List Hole) : Prop :=
  sorry

/-- The main theorem stating that the folding and punching process results in 8 symmetric holes --/
theorem folding_punching_theorem (p : Paper) :
  let folds := [Fold.BottomToTop, Fold.RightToHalfLeft, Fold.DiagonalBottomLeftToTopRight]
  let foldedPaper := applyFolds p folds
  let hole := punchHole foldedPaper
  let holePattern := unfoldAndGetHoles p folds hole
  (holePattern.length = 8) ∧ isSymmetricPattern holePattern :=
by
  sorry

end NUMINAMATH_CALUDE_folding_punching_theorem_l3764_376423


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3764_376417

/-- Given a system of linear equations with a parameter m, prove that when the system has a nontrivial solution, the ratio of xz/y^2 is 20. -/
theorem system_solution_ratio (m : ℚ) (x y z : ℚ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + m*y + 4*z = 0 →
  4*x + m*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  (∃ m, x + m*y + 4*z = 0 ∧ 4*x + m*y - 3*z = 0 ∧ 3*x + 5*y - 4*z = 0) →
  x*z / (y^2) = 20 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3764_376417


namespace NUMINAMATH_CALUDE_brothers_age_proof_l3764_376403

def hannah_age : ℕ := 48
def num_brothers : ℕ := 3

theorem brothers_age_proof (brothers_age : ℕ) 
  (h1 : hannah_age = 2 * (num_brothers * brothers_age)) : 
  brothers_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_proof_l3764_376403


namespace NUMINAMATH_CALUDE_intersection_implies_z_equals_i_l3764_376404

theorem intersection_implies_z_equals_i : 
  let i : ℂ := Complex.I
  let P : Set ℂ := {1, -1}
  let Q : Set ℂ := {i, i^2}
  ∀ z : ℂ, (P ∩ Q = {z * i}) → z = i := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_z_equals_i_l3764_376404


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3764_376437

theorem polygon_diagonals (n : ℕ) (h : (n - 2) * 180 + 360 = 1800) : n - 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3764_376437


namespace NUMINAMATH_CALUDE_student_council_committees_l3764_376440

theorem student_council_committees (n : ℕ) : 
  (n.choose 2 = 15) → (n.choose 3 = 20) :=
by sorry

end NUMINAMATH_CALUDE_student_council_committees_l3764_376440


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3764_376477

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = 150 * π / 180 →
  a = Real.sqrt 3 * c →
  b = 2 * Real.sqrt 7 →
  Real.sin A + Real.sqrt 3 * Real.sin C = Real.sqrt 2 / 2 →
  (∃ (S : Real), S = a * b * Real.sin C / 2 ∧ S = Real.sqrt 3) ∧
  C = 15 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3764_376477


namespace NUMINAMATH_CALUDE_H2O_weight_not_72_l3764_376425

/-- The molecular weight of H2O in g/mol -/
def molecular_weight_H2O : ℝ := 18.016

/-- The given incorrect molecular weight in g/mol -/
def given_weight : ℝ := 72

/-- Theorem stating that the molecular weight of H2O is not equal to the given weight -/
theorem H2O_weight_not_72 : molecular_weight_H2O ≠ given_weight := by
  sorry

end NUMINAMATH_CALUDE_H2O_weight_not_72_l3764_376425


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3764_376443

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 1 + Real.sqrt 6 ∧ x₁^2 - 2*x₁ - 5 = 0) ∧
  (x₂ = 1 - Real.sqrt 6 ∧ x₂^2 - 2*x₂ - 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3764_376443


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3764_376485

/-- Given a geometric sequence {a_n} with common ratio q = 4 and S_3 = 21,
    prove that the general term formula is a_n = 4^(n-1) -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) -- The sequence
  (q : ℝ) -- Common ratio
  (S₃ : ℝ) -- Sum of first 3 terms
  (h1 : ∀ n, a (n + 1) = q * a n) -- Definition of geometric sequence
  (h2 : q = 4) -- Given common ratio
  (h3 : S₃ = 21) -- Given sum of first 3 terms
  (h4 : S₃ = a 1 + a 2 + a 3) -- Definition of S₃
  : ∀ n : ℕ, a n = 4^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3764_376485


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_point_seven_l3764_376449

theorem at_least_one_greater_than_point_seven (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x > 0.7 ∨ y > 0.7 ∨ (1 / (x + y)) > 0.7 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_point_seven_l3764_376449


namespace NUMINAMATH_CALUDE_fred_initial_cards_l3764_376480

/-- Given that Fred gave away 18 cards, found 40 new cards, and ended up with 48 cards,
    prove that he must have started with 26 cards. -/
theorem fred_initial_cards :
  ∀ (initial_cards given_away new_cards final_cards : ℕ),
    given_away = 18 →
    new_cards = 40 →
    final_cards = 48 →
    initial_cards - given_away + new_cards = final_cards →
    initial_cards = 26 := by
  sorry

end NUMINAMATH_CALUDE_fred_initial_cards_l3764_376480


namespace NUMINAMATH_CALUDE_negation_of_universal_nonnegative_l3764_376445

theorem negation_of_universal_nonnegative :
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_nonnegative_l3764_376445


namespace NUMINAMATH_CALUDE_stratified_sample_older_45_correct_l3764_376412

/-- Calculates the number of employees older than 45 to be drawn in a stratified sample -/
def stratifiedSampleOlder45 (totalEmployees : ℕ) (employeesOlder45 : ℕ) (sampleSize : ℕ) : ℕ :=
  (employeesOlder45 * sampleSize) / totalEmployees

/-- Proves that the stratified sample for employees older than 45 is correct -/
theorem stratified_sample_older_45_correct :
  stratifiedSampleOlder45 400 160 50 = 20 := by
  sorry

#eval stratifiedSampleOlder45 400 160 50

end NUMINAMATH_CALUDE_stratified_sample_older_45_correct_l3764_376412


namespace NUMINAMATH_CALUDE_openai_robot_weight_problem_l3764_376497

/-- The OpenAI robotics competition weight problem -/
theorem openai_robot_weight_problem :
  let standard_weight : ℕ := 100
  let max_weight : ℕ := 210
  let min_additional_weight : ℕ := max_weight - standard_weight - (max_weight - standard_weight) / 2
  (2 * (standard_weight + min_additional_weight) ≤ max_weight) ∧
  (∀ w : ℕ, w < min_additional_weight → 2 * (standard_weight + w) > max_weight) →
  min_additional_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_openai_robot_weight_problem_l3764_376497


namespace NUMINAMATH_CALUDE_hare_wins_l3764_376432

/-- Race parameters --/
def race_duration : ℕ := 60
def hare_speed : ℕ := 10
def hare_run_time : ℕ := 30
def hare_nap_time : ℕ := 30
def tortoise_delay : ℕ := 10
def tortoise_speed : ℕ := 4

/-- Calculate distance covered by the hare --/
def hare_distance : ℕ := hare_speed * hare_run_time

/-- Calculate distance covered by the tortoise --/
def tortoise_distance : ℕ := tortoise_speed * (race_duration - tortoise_delay)

/-- Theorem stating that the hare wins the race --/
theorem hare_wins : hare_distance > tortoise_distance := by
  sorry

end NUMINAMATH_CALUDE_hare_wins_l3764_376432


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l3764_376402

-- Define the circle L
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a point Q inside the circle
def Q (L : Circle) : ℝ × ℝ :=
  sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

-- State the theorem
theorem locus_of_midpoints (L : Circle) :
  -- Q is an interior point of L
  (distance (Q L) L.center < L.radius) →
  -- Q is not the center of L
  (Q L ≠ L.center) →
  -- The distance from Q to the center of L is one-third the radius of L
  (distance (Q L) L.center = L.radius / 3) →
  -- The locus of midpoints of all chords passing through Q is a complete circle
  ∃ (C : Circle),
    -- The center of the locus circle is Q
    C.center = Q L ∧
    -- The radius of the locus circle is r/6
    C.radius = L.radius / 6 :=
sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l3764_376402


namespace NUMINAMATH_CALUDE_hyperbola_distance_property_l3764_376448

/-- A point on a hyperbola with specific distance properties -/
structure HyperbolaPoint where
  P : ℝ × ℝ
  on_hyperbola : (P.1^2 / 4) - P.2^2 = 1
  distance_to_right_focus : Real.sqrt ((P.1 - Real.sqrt 5)^2 + P.2^2) = 5

/-- The theorem stating the distance property of the hyperbola point -/
theorem hyperbola_distance_property (hp : HyperbolaPoint) :
  let d := Real.sqrt ((hp.P.1 + Real.sqrt 5)^2 + hp.P.2^2)
  d = 1 ∨ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_distance_property_l3764_376448


namespace NUMINAMATH_CALUDE_box_counting_l3764_376455

theorem box_counting (initial_boxes : ℕ) (boxes_per_operation : ℕ) (final_nonempty_boxes : ℕ) :
  initial_boxes = 2013 →
  boxes_per_operation = 13 →
  final_nonempty_boxes = 2013 →
  initial_boxes + boxes_per_operation * final_nonempty_boxes = 28182 := by
  sorry

#check box_counting

end NUMINAMATH_CALUDE_box_counting_l3764_376455


namespace NUMINAMATH_CALUDE_distance_is_13_l3764_376433

/-- The distance between two villages Yolkino and Palkino. -/
def distance_between_villages : ℝ := 13

/-- A point on the highway between Yolkino and Palkino. -/
structure HighwayPoint where
  distance_to_yolkino : ℝ
  distance_to_palkino : ℝ
  sum_is_13 : distance_to_yolkino + distance_to_palkino = 13

/-- The theorem stating that the distance between Yolkino and Palkino is 13 km. -/
theorem distance_is_13 : 
  ∀ (p : HighwayPoint), distance_between_villages = p.distance_to_yolkino + p.distance_to_palkino :=
by
  sorry

end NUMINAMATH_CALUDE_distance_is_13_l3764_376433


namespace NUMINAMATH_CALUDE_locus_and_line_theorem_l3764_376430

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Define the condition for P being outside C
def outside_circle (x y : ℝ) : Prop := x^2 + (y - 2)^2 > 1

-- Define the tangent condition (implicitly used in the problem)
def is_tangent (x y : ℝ) : Prop := 
  ∃ (t : ℝ), circle_C (x + t) (y + t) ∧ ¬(∃ (s : ℝ), s ≠ t ∧ circle_C (x + s) (y + s))

-- Define the PQ = √2 * PA condition
def pq_pa_relation (x y : ℝ) : Prop :=
  (x^2 + (y - 2)^2 - 1) = 2 * ((x - 3)^2 + y^2)

-- Define the locus of P
def locus_P (x y : ℝ) : Prop := x^2 + y^2 - 12*x + 4*y + 15 = 0

-- Define the condition for line l
def line_l (x y : ℝ) : Prop := x = 3 ∨ 5*x - 12*y - 15 = 0

-- Main theorem
theorem locus_and_line_theorem :
  ∀ (x y : ℝ),
    outside_circle x y →
    is_tangent x y →
    pq_pa_relation x y →
    (locus_P x y ∧
     (∃ (m n : ℝ × ℝ),
       locus_P m.1 m.2 ∧
       locus_P n.1 n.2 ∧
       line_l m.1 m.2 ∧
       line_l n.1 n.2 ∧
       (m.1 - n.1)^2 + (m.2 - n.2)^2 = 64)) :=
by sorry

end NUMINAMATH_CALUDE_locus_and_line_theorem_l3764_376430


namespace NUMINAMATH_CALUDE_perpendicular_lines_to_plane_are_parallel_l3764_376452

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_to_plane_are_parallel
  (α β γ : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_distinct_lines : m ≠ n)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_α : perpendicular n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_to_plane_are_parallel_l3764_376452


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3764_376438

/-- The probability of picking two red balls from a bag with 4 red, 3 blue, and 2 green balls is 1/6 -/
theorem probability_two_red_balls (total_balls : Nat) (red_balls : Nat) (blue_balls : Nat) (green_balls : Nat)
  (h1 : total_balls = red_balls + blue_balls + green_balls)
  (h2 : red_balls = 4)
  (h3 : blue_balls = 3)
  (h4 : green_balls = 2) :
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3764_376438


namespace NUMINAMATH_CALUDE_second_investment_value_l3764_376492

theorem second_investment_value (x : ℝ) : 
  (0.07 * 500 + 0.15 * x = 0.13 * (500 + x)) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_value_l3764_376492


namespace NUMINAMATH_CALUDE_max_value_theorem_l3764_376472

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (max : ℝ), max = -9/2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → -1/(2*x) - 2/y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3764_376472


namespace NUMINAMATH_CALUDE_circle_properties_l3764_376424

-- Define the circle C
def circle_C (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the conditions
def center_on_line (a : ℝ) : ℝ × ℝ := (a, -2*a)
def point_A : ℝ × ℝ := (2, -1)
def tangent_line (p : ℝ × ℝ) : Prop := p.1 + p.2 = 1

-- Theorem statement
theorem circle_properties (a : ℝ) :
  let center := center_on_line a
  let C := circle_C center (|a - 2*a - 1| / Real.sqrt 2)
  point_A ∈ C ∧ (∃ p, p ∈ C ∧ tangent_line p) →
  (C = circle_C (1, -2) (Real.sqrt 2)) ∧
  (Set.Icc (-3 : ℝ) (-1) ⊆ {y | (0, y) ∈ C}) ∧
  (Set.Ioo (-3 : ℝ) (-1) ⊆ {y | (0, y) ∉ C}) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3764_376424


namespace NUMINAMATH_CALUDE_log_inequality_l3764_376410

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.log (1/a) / Real.log 0.3 > Real.log (1/b) / Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3764_376410


namespace NUMINAMATH_CALUDE_coefficient_equals_49_l3764_376489

/-- The coefficient of x^3y^5 in the expansion of (x+2y)(x-y)^7 -/
def coefficient : ℤ :=
  2 * (Nat.choose 7 4) - (Nat.choose 7 5)

/-- Theorem stating that the coefficient of x^3y^5 in the expansion of (x+2y)(x-y)^7 is 49 -/
theorem coefficient_equals_49 : coefficient = 49 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_equals_49_l3764_376489


namespace NUMINAMATH_CALUDE_square_difference_equality_l3764_376463

theorem square_difference_equality : 1004^2 - 998^2 - 1002^2 + 1000^2 = 8008 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3764_376463


namespace NUMINAMATH_CALUDE_three_integers_sum_l3764_376415

theorem three_integers_sum (a b c : ℕ) : 
  a > 1 → b > 1 → c > 1 →
  a * b * c = 216000 →
  Nat.gcd a b = 1 → Nat.gcd a c = 1 → Nat.gcd b c = 1 →
  a + b + c = 184 :=
by sorry

end NUMINAMATH_CALUDE_three_integers_sum_l3764_376415


namespace NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l3764_376470

theorem sum_of_quotient_dividend_divisor (N : ℕ) (h : N = 40) : 
  (N / 2) + N + 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l3764_376470


namespace NUMINAMATH_CALUDE_composite_ratio_theorem_l3764_376461

/-- The nth positive composite number -/
def nthComposite (n : ℕ) : ℕ := sorry

/-- The product of the first n positive composite numbers -/
def productFirstNComposites (n : ℕ) : ℕ := sorry

theorem composite_ratio_theorem : 
  (productFirstNComposites 7) / (productFirstNComposites 14 / productFirstNComposites 7) = 1 / 110 := by
  sorry

end NUMINAMATH_CALUDE_composite_ratio_theorem_l3764_376461


namespace NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l3764_376487

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence 
  (a : ℕ → ℤ) (h : arithmetic_sequence a) (h5 : a 5 = 3) (h6 : a 6 = -2) : 
  ∃ d : ℤ, d = -5 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l3764_376487


namespace NUMINAMATH_CALUDE_number_factorization_l3764_376471

theorem number_factorization (n : ℤ) : 
  (∃ x y : ℤ, n = x * y ∧ y - x = 6 ∧ x^4 + y^4 = 272) → n = -8 := by
  sorry

end NUMINAMATH_CALUDE_number_factorization_l3764_376471


namespace NUMINAMATH_CALUDE_cookies_remaining_l3764_376446

theorem cookies_remaining (initial : ℕ) (given : ℕ) (eaten : ℕ) : 
  initial = 36 → given = 14 → eaten = 10 → initial - (given + eaten) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_l3764_376446


namespace NUMINAMATH_CALUDE_intermediate_root_existence_l3764_376428

theorem intermediate_root_existence (a b c x₁ x₂ : ℝ) 
  (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0)
  (hx₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧ 
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) := by
  sorry

end NUMINAMATH_CALUDE_intermediate_root_existence_l3764_376428


namespace NUMINAMATH_CALUDE_joan_gemstones_l3764_376469

/-- Represents Joan's rock collection --/
structure RockCollection where
  minerals_yesterday : ℕ
  gemstones : ℕ
  minerals_today : ℕ

/-- Theorem about Joan's rock collection --/
theorem joan_gemstones (collection : RockCollection) 
  (h1 : collection.gemstones = collection.minerals_yesterday / 2)
  (h2 : collection.minerals_today = collection.minerals_yesterday + 6)
  (h3 : collection.minerals_today = 48) : 
  collection.gemstones = 21 := by
sorry

end NUMINAMATH_CALUDE_joan_gemstones_l3764_376469


namespace NUMINAMATH_CALUDE_factor_expression_l3764_376418

theorem factor_expression (x : ℝ) : 6 * x^3 - 54 * x = 6 * x * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3764_376418


namespace NUMINAMATH_CALUDE_parabola_vertex_l3764_376434

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-9, -3)

/-- Theorem: The vertex of the parabola y = 2(x+9)^2 - 3 is at the point (-9, -3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3764_376434


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l3764_376419

/-- The number of distinct permutations of the word "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l3764_376419


namespace NUMINAMATH_CALUDE_m_greater_than_n_l3764_376491

/-- A line with slope -1 and y-intercept b -/
def line (b : ℝ) := fun (x : ℝ) ↦ -x + b

/-- Point A lies on the line -/
def point_A_on_line (m b : ℝ) : Prop := line b (-5) = m

/-- Point B lies on the line -/
def point_B_on_line (n b : ℝ) : Prop := line b 4 = n

theorem m_greater_than_n (m n b : ℝ) :
  point_A_on_line m b → point_B_on_line n b → m > n := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l3764_376491


namespace NUMINAMATH_CALUDE_division_of_decimals_l3764_376454

theorem division_of_decimals : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l3764_376454


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_lcm_l3764_376476

theorem two_numbers_sum_and_lcm : ∃ (x y : ℕ), 
  x + y = 316 ∧ 
  Nat.lcm x y = 4560 ∧ 
  x = 199 ∧ 
  y = 117 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_lcm_l3764_376476


namespace NUMINAMATH_CALUDE_special_line_equation_l3764_376467

/-- A line passing through point (2, 3) with intercepts on the coordinate axes that are opposite numbers -/
structure SpecialLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2, 3)
  passes_through : m * 2 + b = 3
  -- The intercepts are opposite numbers
  opposite_intercepts : b = m * b

theorem special_line_equation (L : SpecialLine) :
  (L.m = 3/2 ∧ L.b = 0) ∨ (L.m = 1 ∧ L.b = -1) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l3764_376467


namespace NUMINAMATH_CALUDE_cake_recipe_flour_l3764_376460

theorem cake_recipe_flour (sugar cups_of_sugar : ℕ) (flour initial_flour : ℕ) :
  cups_of_sugar = 6 →
  initial_flour = 2 →
  flour = cups_of_sugar + 1 →
  flour = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_l3764_376460


namespace NUMINAMATH_CALUDE_non_collinear_implies_nonzero_l3764_376427

-- Define the vector type
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the collinearity relation
def collinear (a b : V) : Prop := ∃ (k : ℝ), a = k • b

-- State the theorem
theorem non_collinear_implies_nonzero (a b : V) : 
  ¬(collinear a b) → a ≠ 0 ∧ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_non_collinear_implies_nonzero_l3764_376427


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3764_376478

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3764_376478


namespace NUMINAMATH_CALUDE_basketball_game_students_l3764_376482

/-- The total number of students in a basketball game given the number of 5th graders and a ratio of 6th to 5th graders -/
def total_students (fifth_graders : ℕ) (ratio : ℕ) : ℕ :=
  fifth_graders + ratio * fifth_graders

/-- Theorem stating that given 12 5th graders and 6 times as many 6th graders, the total number of students is 84 -/
theorem basketball_game_students :
  total_students 12 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_students_l3764_376482


namespace NUMINAMATH_CALUDE_dartboard_angle_measure_l3764_376441

/-- The measure of the central angle of a region on a circular dartboard, given its probability -/
theorem dartboard_angle_measure (p : ℝ) (h : p = 1 / 8) : 
  p * 360 = 45 := by sorry

end NUMINAMATH_CALUDE_dartboard_angle_measure_l3764_376441


namespace NUMINAMATH_CALUDE_ad_time_theorem_l3764_376442

/-- Calculates the total advertisement time in a week given the ad duration and cycle time -/
def total_ad_time_per_week (ad_duration : ℚ) (cycle_time : ℚ) : ℚ :=
  let ads_per_hour : ℚ := 60 / cycle_time
  let ad_time_per_hour : ℚ := ads_per_hour * ad_duration
  let hours_per_week : ℚ := 24 * 7
  ad_time_per_hour * hours_per_week

/-- Converts minutes to hours and minutes -/
def minutes_to_hours_and_minutes (total_minutes : ℚ) : ℚ × ℚ :=
  let hours : ℚ := total_minutes / 60
  let minutes : ℚ := total_minutes % 60
  (hours.floor, minutes)

theorem ad_time_theorem :
  let ad_duration : ℚ := 3/2  -- 1.5 minutes
  let cycle_time : ℚ := 20    -- 20 minutes (including ad duration)
  let total_minutes : ℚ := total_ad_time_per_week ad_duration cycle_time
  let (hours, minutes) := minutes_to_hours_and_minutes total_minutes
  hours = 12 ∧ minutes = 36 := by
  sorry


end NUMINAMATH_CALUDE_ad_time_theorem_l3764_376442


namespace NUMINAMATH_CALUDE_position_selection_count_l3764_376408

/-- The number of people in the group --/
def group_size : ℕ := 6

/-- The number of positions to be filled --/
def num_positions : ℕ := 3

/-- Theorem: The number of ways to choose a President, Vice-President, and Secretary
    from a group of 6 people, where all positions must be held by different individuals,
    is equal to 120. --/
theorem position_selection_count :
  (group_size.factorial) / ((group_size - num_positions).factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_position_selection_count_l3764_376408


namespace NUMINAMATH_CALUDE_fundraiser_group_composition_l3764_376422

theorem fundraiser_group_composition (p : ℕ) : 
  p > 0 ∧ 
  (p / 2 : ℚ) = (p / 2 : ℕ) ∧ 
  ((p / 2 - 2 : ℚ) / p = 2 / 5) → 
  p / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_group_composition_l3764_376422


namespace NUMINAMATH_CALUDE_distance_set_exists_l3764_376498

/-- A set of points in the plane satisfying the distance condition -/
def DistanceSet (m : ℕ) (S : Set (ℝ × ℝ)) : Prop :=
  (∀ A ∈ S, ∃! (points : Finset (ℝ × ℝ)), 
    points.card = m ∧ 
    (∀ B ∈ points, B ∈ S ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1))

/-- The existence of a finite set satisfying the distance condition for any m ≥ 1 -/
theorem distance_set_exists (m : ℕ) (hm : m ≥ 1) : 
  ∃ S : Set (ℝ × ℝ), S.Finite ∧ DistanceSet m S :=
sorry

end NUMINAMATH_CALUDE_distance_set_exists_l3764_376498


namespace NUMINAMATH_CALUDE_product_difference_l3764_376444

theorem product_difference (a b : ℕ+) : 
  a * b = 323 → a = 17 → b - a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_product_difference_l3764_376444


namespace NUMINAMATH_CALUDE_z_share_per_x_rupee_l3764_376400

/-- Proof that z gets 0.50 rupees for each rupee x gets --/
theorem z_share_per_x_rupee
  (total : ℝ)
  (y_share : ℝ)
  (y_per_x : ℝ)
  (h_total : total = 156)
  (h_y_share : y_share = 36)
  (h_y_per_x : y_per_x = 0.45)
  : ∃ (z_per_x : ℝ), z_per_x = 0.50 ∧
    ∃ (units : ℝ), units * (1 + y_per_x + z_per_x) = total ∧
                   units * y_per_x = y_share :=
by
  sorry


end NUMINAMATH_CALUDE_z_share_per_x_rupee_l3764_376400


namespace NUMINAMATH_CALUDE_farm_horse_food_calculation_l3764_376464

/-- Calculates the total amount of horse food needed daily on a farm -/
theorem farm_horse_food_calculation (sheep_count : ℕ) (sheep_to_horse_ratio : ℚ) (food_per_horse : ℕ) : 
  sheep_count = 48 →
  sheep_to_horse_ratio = 6 / 7 →
  food_per_horse = 230 →
  (sheep_count / sheep_to_horse_ratio : ℚ).num * food_per_horse = 12880 := by
  sorry

end NUMINAMATH_CALUDE_farm_horse_food_calculation_l3764_376464


namespace NUMINAMATH_CALUDE_kanul_total_amount_l3764_376414

/-- The total amount Kanul had -/
def total_amount : ℝ := 137500

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 80000

/-- The amount spent on machinery -/
def machinery : ℝ := 30000

/-- The percentage of total amount spent as cash -/
def cash_percentage : ℝ := 0.20

theorem kanul_total_amount : 
  total_amount = raw_materials + machinery + cash_percentage * total_amount := by
  sorry

end NUMINAMATH_CALUDE_kanul_total_amount_l3764_376414


namespace NUMINAMATH_CALUDE_function_properties_l3764_376494

noncomputable def f (A ω φ B x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + B

theorem function_properties :
  ∀ (A ω φ B : ℝ),
  A > 0 → ω > 0 → 0 < φ → φ < π →
  f A ω φ B (π / 3) = 1 →
  f A ω φ B (π / 2 / ω - φ / ω) = 3 →
  ∃ (x : ℝ), ω * x + φ = 0 ∧
  ∃ (y : ℝ), ω * y + φ = π ∧
  ω * (7 * π / 12) + φ = 2 * π →
  A = 1 ∧ B = 2 ∧ ω = 2 ∧ φ = 5 * π / 6 ∧
  (∀ (x : ℝ), f A ω φ B x = f A ω φ B (-4 * π / 3 - x)) ∧
  (∀ (x : ℝ), f A ω φ B (x - 5 * π / 12) = 4 - f A ω φ B (x + 5 * π / 12)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3764_376494


namespace NUMINAMATH_CALUDE_twice_one_fifth_of_ten_times_fifteen_l3764_376413

theorem twice_one_fifth_of_ten_times_fifteen : 2 * ((1 / 5 : ℚ) * (10 * 15)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_twice_one_fifth_of_ten_times_fifteen_l3764_376413


namespace NUMINAMATH_CALUDE_barium_oxide_weight_l3764_376458

/-- The atomic weight of Barium in g/mol -/
def barium_weight : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (ba_count o_count : ℕ) : ℝ :=
  ba_count * barium_weight + o_count * oxygen_weight

/-- Theorem: The molecular weight of a compound with 1 Barium and 1 Oxygen atom is 153.33 g/mol -/
theorem barium_oxide_weight : molecular_weight 1 1 = 153.33 := by
  sorry

end NUMINAMATH_CALUDE_barium_oxide_weight_l3764_376458


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3764_376462

theorem intersection_of_lines :
  ∃! (x y : ℚ), (8 * x - 5 * y = 40) ∧ (6 * x + 2 * y = 14) ∧ 
  (x = 75 / 23) ∧ (y = 161 / 23) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3764_376462


namespace NUMINAMATH_CALUDE_exists_square_with_digit_sum_2002_l3764_376496

/-- Sum of digits of a natural number in base 10 -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a square number whose sum of digits in base 10 is 2002 -/
theorem exists_square_with_digit_sum_2002 : 
  ∃ n : ℕ, sum_of_digits (n^2) = 2002 := by sorry

end NUMINAMATH_CALUDE_exists_square_with_digit_sum_2002_l3764_376496


namespace NUMINAMATH_CALUDE_factorial_not_equal_even_factorial_l3764_376465

theorem factorial_not_equal_even_factorial (n m : ℕ) (hn : n > 1) (hm : m > 1) :
  n.factorial ≠ 2^m * m.factorial := by
  sorry

end NUMINAMATH_CALUDE_factorial_not_equal_even_factorial_l3764_376465


namespace NUMINAMATH_CALUDE_min_value_product_l3764_376421

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z * (x + y + z) = 1) : 
  (x + y) * (y + z) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l3764_376421


namespace NUMINAMATH_CALUDE_singer_arrangements_l3764_376401

/-- The number of ways to arrange n objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k objects out of n objects. -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem singer_arrangements : 
  let total_singers : ℕ := 5
  let arrangements_case1 := permutations 4 4  -- when the singer who can't be last is first
  let arrangements_case2 := permutations 3 1 * permutations 3 1 * permutations 3 3  -- other cases
  arrangements_case1 + arrangements_case2 = 78 := by
  sorry

end NUMINAMATH_CALUDE_singer_arrangements_l3764_376401


namespace NUMINAMATH_CALUDE_exactly_two_rigid_motions_l3764_376453

/-- Represents a point on a plane --/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line on a plane --/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents the pattern on the line --/
inductive Pattern
  | Triangle
  | Square

/-- Represents a rigid motion transformation --/
inductive RigidMotion
  | Rotation (center : Point) (angle : ℝ)
  | Translation (dx : ℝ) (dy : ℝ)
  | ReflectionLine (l : Line)
  | ReflectionPerp (p : Point)

/-- The line with the pattern --/
def patternLine : Line := sorry

/-- The sequence of shapes along the line --/
def patternSequence : ℕ → Pattern := sorry

/-- Checks if a rigid motion preserves the pattern --/
def preservesPattern (rm : RigidMotion) : Prop := sorry

/-- The theorem to be proved --/
theorem exactly_two_rigid_motions :
  ∃! (s : Finset RigidMotion),
    s.card = 2 ∧ 
    (∀ rm ∈ s, preservesPattern rm) ∧
    (∀ rm, preservesPattern rm → rm ∈ s ∨ rm = RigidMotion.Translation 0 0) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_rigid_motions_l3764_376453


namespace NUMINAMATH_CALUDE_remainder_4039_div_31_l3764_376475

theorem remainder_4039_div_31 : 4039 % 31 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4039_div_31_l3764_376475


namespace NUMINAMATH_CALUDE_correct_classification_l3764_376451

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the structure of a reasoning process
structure ReasoningProcess where
  description : String
  correct_type : ReasoningType

-- Define the three reasoning processes
def process1 : ReasoningProcess :=
  { description := "The probability of a coin landing heads up is determined to be 0.5 through numerous trials",
    correct_type := ReasoningType.Inductive }

def process2 : ReasoningProcess :=
  { description := "The function f(x) = x^2 - |x| is an even function",
    correct_type := ReasoningType.Deductive }

def process3 : ReasoningProcess :=
  { description := "Scientists invented the electronic eagle eye by studying the eyes of eagles",
    correct_type := ReasoningType.Analogical }

-- Theorem to prove
theorem correct_classification :
  (process1.correct_type = ReasoningType.Inductive) ∧
  (process2.correct_type = ReasoningType.Deductive) ∧
  (process3.correct_type = ReasoningType.Analogical) :=
by sorry

end NUMINAMATH_CALUDE_correct_classification_l3764_376451


namespace NUMINAMATH_CALUDE_chord_count_l3764_376426

/-- The number of points on the circumference of a circle -/
def n : ℕ := 7

/-- The number of points needed to form a chord -/
def r : ℕ := 2

/-- The number of different chords that can be drawn -/
def num_chords : ℕ := n.choose r

theorem chord_count : num_chords = 21 := by
  sorry

end NUMINAMATH_CALUDE_chord_count_l3764_376426


namespace NUMINAMATH_CALUDE_broadcast_end_date_prove_broadcast_end_date_l3764_376457

/-- Represents a date with year, month, and day. -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Represents a day of the week. -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the broadcasting schedule. -/
structure BroadcastSchedule where
  wednesday : Nat
  friday : Nat
  saturday : Nat
  sunday : Nat

/-- Calculates the end date of the broadcast. -/
def calculateEndDate (startDate : Date) (totalEpisodes : Nat) (schedule : BroadcastSchedule) : Date :=
  sorry

/-- Determines the day of the week for a given date. -/
def getDayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Main theorem to prove -/
theorem broadcast_end_date (startDate : Date) (totalEpisodes : Nat) (schedule : BroadcastSchedule) :
  let endDate := calculateEndDate startDate totalEpisodes schedule
  endDate.year = 2016 ∧ endDate.month = 5 ∧ endDate.day = 29 ∧
  getDayOfWeek endDate = DayOfWeek.Sunday :=
by
  sorry

/-- Initial conditions -/
def initialDate : Date := { year := 2015, month := 12, day := 26 }
def episodeCount : Nat := 135
def broadcastSchedule : BroadcastSchedule := { wednesday := 1, friday := 1, saturday := 2, sunday := 2 }

/-- Proof of the main theorem with initial conditions -/
theorem prove_broadcast_end_date :
  let endDate := calculateEndDate initialDate episodeCount broadcastSchedule
  endDate.year = 2016 ∧ endDate.month = 5 ∧ endDate.day = 29 ∧
  getDayOfWeek endDate = DayOfWeek.Sunday :=
by
  sorry

end NUMINAMATH_CALUDE_broadcast_end_date_prove_broadcast_end_date_l3764_376457


namespace NUMINAMATH_CALUDE_sequence_formulas_l3764_376436

/-- Given a sequence {a_n} with sum of first n terms S_n satisfying S_n = 2 - a_n,
    and sequence {b_n} satisfying b_1 = 1 and b_{n+1} = b_n + a_n,
    prove the general term formulas for both sequences. -/
theorem sequence_formulas (a b : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n = 2 - a n) →
  b 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = b n + a n) →
  (∀ n : ℕ, n ≥ 1 → a n = (1/2)^(n-1)) ∧
  (∀ n : ℕ, n ≥ 2 → b n = 3 - 1/(2^(n-2))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formulas_l3764_376436


namespace NUMINAMATH_CALUDE_volume_Q3_is_7_l3764_376490

/-- Represents the volume of the i-th polyhedron in the sequence -/
def volume (i : ℕ) : ℚ :=
  match i with
  | 0 => 1
  | n + 1 => volume n + 6 * (1 / 3) * (1 / 4^n)

/-- The theorem stating that the volume of Q₃ is 7 -/
theorem volume_Q3_is_7 : volume 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_volume_Q3_is_7_l3764_376490


namespace NUMINAMATH_CALUDE_intersection_M_N_l3764_376431

def M : Set ℝ := {x | x^2 - x ≥ 0}
def N : Set ℝ := {x | x < 2}

theorem intersection_M_N : 
  ∀ x : ℝ, x ∈ M ∩ N ↔ x ≤ 0 ∨ (1 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3764_376431


namespace NUMINAMATH_CALUDE_box_width_l3764_376405

/-- The width of a rectangular box given specific conditions -/
theorem box_width (num_cubes : ℕ) (cube_volume length height : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  length = 8 →
  height = 12 →
  (num_cubes : ℝ) * cube_volume / (length * height) = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_box_width_l3764_376405


namespace NUMINAMATH_CALUDE_emily_flower_spending_l3764_376488

def flower_price : ℝ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2
def discount_threshold : ℕ := 3
def discount_rate : ℝ := 0.2

def total_flowers : ℕ := roses_bought + daisies_bought

def apply_discount (price : ℝ) : ℝ :=
  if total_flowers > discount_threshold then
    price * (1 - discount_rate)
  else
    price

theorem emily_flower_spending :
  apply_discount (flower_price * (roses_bought + daisies_bought : ℝ)) = 9.60 := by
  sorry

end NUMINAMATH_CALUDE_emily_flower_spending_l3764_376488


namespace NUMINAMATH_CALUDE_min_value_problem_l3764_376411

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 2 * x * (x + 1 / y + 1 / z) = y * z) :
  (x + 1 / y) * (x + 1 / z) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3764_376411


namespace NUMINAMATH_CALUDE_jean_money_l3764_376409

theorem jean_money (jane : ℕ) (jean : ℕ) : 
  jean = 3 * jane → 
  jean + jane = 76 → 
  jean = 57 := by
sorry

end NUMINAMATH_CALUDE_jean_money_l3764_376409


namespace NUMINAMATH_CALUDE_boat_current_rate_l3764_376435

/-- Proves that the rate of current is 5 km/hr given the conditions of the boat problem -/
theorem boat_current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 5 →
  downstream_time = 1/5 →
  ∃ (current_rate : ℝ), 
    downstream_distance = (boat_speed + current_rate) * downstream_time ∧
    current_rate = 5 :=
by sorry

end NUMINAMATH_CALUDE_boat_current_rate_l3764_376435


namespace NUMINAMATH_CALUDE_revenue_change_l3764_376466

/-- Proves that given a 75% price decrease and a specific ratio between percent increase in units sold
    and percent decrease in price, the new revenue is 50% of the original revenue -/
theorem revenue_change (P Q : ℝ) (P' Q' : ℝ) (h1 : P' = 0.25 * P) 
    (h2 : (Q' / Q - 1) / 0.75 = 1.3333333333333333) : P' * Q' = 0.5 * P * Q := by
  sorry

end NUMINAMATH_CALUDE_revenue_change_l3764_376466


namespace NUMINAMATH_CALUDE_incorrect_y_value_l3764_376429

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a sequence of 7 x values with equal differences -/
structure XSequence where
  x : Fin 7 → ℝ
  increasing : ∀ i j, i < j → x i < x j
  equal_diff : ∀ i : Fin 6, x (i + 1) - x i = x 1 - x 0

/-- The given y values -/
def y_values : Fin 7 → ℝ := ![51, 107, 185, 285, 407, 549, 717]

/-- The theorem to prove -/
theorem incorrect_y_value (f : QuadraticFunction) (xs : XSequence) :
  (∀ i : Fin 7, i.val ≠ 5 → y_values i = f.a * (xs.x i)^2 + f.b * (xs.x i) + f.c) →
  y_values 5 ≠ f.a * (xs.x 5)^2 + f.b * (xs.x 5) + f.c ∧
  571 = f.a * (xs.x 5)^2 + f.b * (xs.x 5) + f.c := by
  sorry

end NUMINAMATH_CALUDE_incorrect_y_value_l3764_376429


namespace NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l3764_376479

theorem square_sum_from_sum_and_product (x y : ℝ) :
  x + y = 5 → x * y = 6 → x^2 + y^2 = 13 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l3764_376479


namespace NUMINAMATH_CALUDE_local_extremum_and_minimum_l3764_376420

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the derivative of f(x)
def f_prime (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_and_minimum (a b : ℝ) :
  (f a b 1 = 10) ∧ 
  (f_prime a b 1 = 0) →
  (a = 4 ∧ b = -11) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 3, f a b x ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_local_extremum_and_minimum_l3764_376420


namespace NUMINAMATH_CALUDE_quadratic_maximum_l3764_376483

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 24

-- State the theorem
theorem quadratic_maximum :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max) ∧
  (∃ (x_max : ℝ), f x_max = 30.75) ∧
  (∀ (x : ℝ), f x ≤ 30.75) ∧
  f (3/2) = 30.75 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l3764_376483


namespace NUMINAMATH_CALUDE_square_sum_implies_product_zero_l3764_376484

theorem square_sum_implies_product_zero (n : ℝ) : 
  (n - 2022)^2 + (2023 - n)^2 = 1 → (n - 2022) * (2023 - n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_zero_l3764_376484


namespace NUMINAMATH_CALUDE_binary_sum_is_eleven_l3764_376499

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 101₂ -/
def binary1 : List Bool := [true, false, true]

/-- The second binary number 110₂ -/
def binary2 : List Bool := [false, true, true]

/-- The sum of binary1 and binary2 in decimal form -/
def sum_decimal : ℕ := binary_to_decimal binary1 + binary_to_decimal binary2

theorem binary_sum_is_eleven : sum_decimal = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_is_eleven_l3764_376499


namespace NUMINAMATH_CALUDE_house_store_transaction_l3764_376459

theorem house_store_transaction (house_selling_price store_selling_price : ℝ)
  (house_loss_percent store_gain_percent : ℝ) :
  house_selling_price = 12000 →
  store_selling_price = 12000 →
  house_loss_percent = 25 →
  store_gain_percent = 25 →
  let house_cost := house_selling_price / (1 - house_loss_percent / 100)
  let store_cost := store_selling_price / (1 + store_gain_percent / 100)
  let total_cost := house_cost + store_cost
  let total_selling_price := house_selling_price + store_selling_price
  total_cost - total_selling_price = 1600 := by
sorry

end NUMINAMATH_CALUDE_house_store_transaction_l3764_376459


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l3764_376474

theorem triangle_max_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x < 17 →
  x + 2*x > 17 →
  x + 17 > 2*x →
  2*x + 17 > x →
  (∀ y : ℕ, y > 0 → y < 17 → y + 2*y > 17 → y + 17 > 2*y → 2*y + 17 > y →
    x + 2*x + 17 ≥ y + 2*y + 17) →
  x + 2*x + 17 = 65 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l3764_376474


namespace NUMINAMATH_CALUDE_sum_of_integers_l3764_376493

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 2)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3764_376493


namespace NUMINAMATH_CALUDE_triangle_perimeter_and_shape_l3764_376481

/-- Given a triangle ABC with side lengths a, b, and c satisfying certain conditions,
    prove that its perimeter is 17 and it is an isosceles triangle. -/
theorem triangle_perimeter_and_shape (a b c : ℝ) : 
  (b - 5)^2 + (c - 7)^2 = 0 →
  |a - 3| = 2 →
  a + b + c = 17 ∧ a = b := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_and_shape_l3764_376481


namespace NUMINAMATH_CALUDE_product_equality_l3764_376473

theorem product_equality (a : ℝ) (h : a ≠ 0 ∧ a ≠ 2 ∧ a ≠ -2) :
  (a^2 + 2*a + 4 + 8/a + 16/a^2 + 64/((a-2)*a^2)) *
  (a^2 - 2*a + 4 - 8/a + 16/a^2 - 64/((a+2)*a^2))
  =
  (a^2 + 2*a + 4 + 8/a + 16/a^2) *
  (a^2 - 2*a + 4 - 8/a + 16/a^2) :=
by sorry

end NUMINAMATH_CALUDE_product_equality_l3764_376473


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l3764_376406

/-- Proves that it takes 2 years for a man's age to be twice his son's age -/
theorem mans_age_twice_sons (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 33 →
  age_difference = 35 →
  ∃ (years : ℕ), years = 2 ∧ 
    (son_age + age_difference + years) = 2 * (son_age + years) := by
  sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l3764_376406


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3764_376447

-- Define the sets A and B
def A : Set ℝ := {x | 3 - x > 0 ∧ x + 2 > 0}
def B : Set ℝ := {m | 3 > 2 * m - 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3764_376447


namespace NUMINAMATH_CALUDE_fish_bucket_problem_l3764_376439

/-- Proves that the number of buckets is 9 given the conditions of the fish problem -/
theorem fish_bucket_problem (total_fish_per_bucket : ℕ) (mackerels_per_bucket : ℕ) (total_mackerels : ℕ)
  (h1 : total_fish_per_bucket = 9)
  (h2 : mackerels_per_bucket = 3)
  (h3 : total_mackerels = 27) :
  total_mackerels / mackerels_per_bucket = 9 := by
  sorry

end NUMINAMATH_CALUDE_fish_bucket_problem_l3764_376439


namespace NUMINAMATH_CALUDE_two_dogs_weekly_distance_l3764_376450

/-- The total distance walked by two dogs in a week, given their daily walking distances -/
def total_weekly_distance (dog1_daily : ℕ) (dog2_daily : ℕ) : ℕ :=
  (dog1_daily * 7) + (dog2_daily * 7)

/-- Theorem: The total distance walked by two dogs in a week is 70 miles -/
theorem two_dogs_weekly_distance :
  total_weekly_distance 2 8 = 70 := by
  sorry

end NUMINAMATH_CALUDE_two_dogs_weekly_distance_l3764_376450


namespace NUMINAMATH_CALUDE_sphere_center_sum_l3764_376456

-- Define the points and constants
variable (a b c p q r α β γ : ℝ)

-- Define the conditions
variable (h1 : p^3 = α)
variable (h2 : q^3 = β)
variable (h3 : r^3 = γ)

-- Define the plane equation
variable (h4 : a/α + b/β + c/γ = 1)

-- Define that (p,q,r) is the center of the sphere passing through O, A, B, C
variable (h5 : p^2 + q^2 + r^2 = (p - α)^2 + q^2 + r^2)
variable (h6 : p^2 + q^2 + r^2 = p^2 + (q - β)^2 + r^2)
variable (h7 : p^2 + q^2 + r^2 = p^2 + q^2 + (r - γ)^2)

-- Theorem statement
theorem sphere_center_sum :
  a/p^3 + b/q^3 + c/r^3 = 1 :=
sorry

end NUMINAMATH_CALUDE_sphere_center_sum_l3764_376456


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3764_376495

theorem algebraic_expression_equality (x y : ℝ) : 
  x + 2 * y + 1 = 3 → 2 * x + 4 * y + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3764_376495


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3764_376486

theorem complex_modulus_problem (z : ℂ) : z = (2 * Complex.I) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3764_376486
