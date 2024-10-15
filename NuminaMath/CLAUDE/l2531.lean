import Mathlib

namespace NUMINAMATH_CALUDE_gcd_5005_11011_l2531_253147

theorem gcd_5005_11011 : Nat.gcd 5005 11011 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5005_11011_l2531_253147


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l2531_253126

theorem arithmetic_to_geometric_sequence (a₁ a₂ a₃ a₄ d : ℝ) : 
  d ≠ 0 ∧ 
  a₂ = a₁ + d ∧ 
  a₃ = a₁ + 2*d ∧ 
  a₄ = a₁ + 3*d ∧
  ((a₁*a₃ = a₂^2) ∨ (a₁*a₄ = a₂^2) ∨ (a₁*a₄ = a₃^2) ∨ (a₂*a₄ = a₃^2)) →
  a₁/d = 1 ∨ a₁/d = -4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l2531_253126


namespace NUMINAMATH_CALUDE_valid_draws_eq_189_l2531_253138

def total_cards : ℕ := 12
def cards_per_color : ℕ := 3
def num_colors : ℕ := 4
def cards_to_draw : ℕ := 3

def valid_draws : ℕ := Nat.choose total_cards cards_to_draw - 
                        (num_colors * Nat.choose cards_per_color cards_to_draw) - 
                        (Nat.choose cards_per_color 2 * Nat.choose (total_cards - cards_per_color) 1)

theorem valid_draws_eq_189 : valid_draws = 189 := by sorry

end NUMINAMATH_CALUDE_valid_draws_eq_189_l2531_253138


namespace NUMINAMATH_CALUDE_linear_function_constraint_l2531_253134

/-- Given a linear function y = x - k, if for all x < 3, y < 2k, then k ≥ 1 -/
theorem linear_function_constraint (k : ℝ) : 
  (∀ x : ℝ, x < 3 → x - k < 2 * k) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_constraint_l2531_253134


namespace NUMINAMATH_CALUDE_sin_390_degrees_l2531_253153

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l2531_253153


namespace NUMINAMATH_CALUDE_cheerful_not_green_l2531_253182

structure Snake where
  isGreen : Bool
  isCheerful : Bool
  canMultiply : Bool
  canDivide : Bool

def TomCollection : Nat := 15

theorem cheerful_not_green (snakes : Finset Snake) 
  (h1 : snakes.card = TomCollection)
  (h2 : (snakes.filter (fun s => s.isGreen)).card = 5)
  (h3 : (snakes.filter (fun s => s.isCheerful)).card = 6)
  (h4 : ∀ s ∈ snakes, s.isCheerful → s.canMultiply)
  (h5 : ∀ s ∈ snakes, s.isGreen → ¬s.canDivide)
  (h6 : ∀ s ∈ snakes, ¬s.canDivide → ¬s.canMultiply) :
  ∀ s ∈ snakes, s.isCheerful → ¬s.isGreen :=
sorry

end NUMINAMATH_CALUDE_cheerful_not_green_l2531_253182


namespace NUMINAMATH_CALUDE_shortest_distance_correct_l2531_253124

/-- Represents the lengths of six lines meeting at a point -/
structure SixLines where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e ∧ e ≥ f
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0

/-- The shortest distance to draw all lines without lifting the pencil -/
def shortestDistance (lines : SixLines) : ℝ :=
  lines.a + 2 * (lines.b + lines.c + lines.d + lines.e + lines.f)

/-- Theorem stating that the shortest distance formula is correct -/
theorem shortest_distance_correct (lines : SixLines) :
  shortestDistance lines = lines.a + 2 * (lines.b + lines.c + lines.d + lines.e + lines.f) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_correct_l2531_253124


namespace NUMINAMATH_CALUDE_fourteen_sided_figure_area_l2531_253148

/-- A fourteen-sided figure on a 1 cm × 1 cm graph paper -/
structure FourteenSidedFigure where
  /-- The number of full unit squares in the figure -/
  full_squares : ℕ
  /-- The number of small right-angled triangles in the figure -/
  small_triangles : ℕ

/-- Calculate the area of a fourteen-sided figure -/
def calculate_area (figure : FourteenSidedFigure) : ℝ :=
  figure.full_squares + (figure.small_triangles * 0.5)

theorem fourteen_sided_figure_area :
  ∀ (figure : FourteenSidedFigure),
    figure.full_squares = 10 →
    figure.small_triangles = 8 →
    calculate_area figure = 14 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_sided_figure_area_l2531_253148


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_a_values_l2531_253133

theorem ellipse_eccentricity_a_values (a : ℝ) :
  a > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 6 = 1) →
  (let e := Real.sqrt 6 / 6
   ∃ b : ℝ, e^2 = 1 - (min a (Real.sqrt 6))^2 / (max a (Real.sqrt 6))^2) →
  a = 6 * Real.sqrt 5 / 5 ∨ a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_a_values_l2531_253133


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a9_l2531_253195

/-- An arithmetic sequence {aₙ} where a₂ = -3 and a₃ = -5 has a₉ = -17 -/
theorem arithmetic_sequence_a9 (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence condition
  a 2 = -3 →
  a 3 = -5 →
  a 9 = -17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a9_l2531_253195


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l2531_253122

/-- Calculates the number of matches in a round-robin tournament. -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-person round-robin tennis tournament has 45 matches. -/
theorem ten_player_tournament_matches :
  roundRobinMatches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l2531_253122


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2531_253119

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (a 1 + a n) * n / 2

-- Theorem statement
theorem arithmetic_sequence_ratio 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_a3 : a 3 = 9) 
  (h_a5 : a 5 = 5) : 
  S a 9 / S a 5 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2531_253119


namespace NUMINAMATH_CALUDE_birds_on_fence_proof_l2531_253145

/-- Given an initial number of birds and the number of birds remaining,
    calculate the number of birds that flew away. -/
def birds_flew_away (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

theorem birds_on_fence_proof :
  let initial_birds : ℝ := 12.0
  let remaining_birds : ℕ := 4
  birds_flew_away initial_birds remaining_birds = 8 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_proof_l2531_253145


namespace NUMINAMATH_CALUDE_train_passing_time_l2531_253177

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 500 → 
  train_speed_kmh = 72 → 
  passing_time = 25 → 
  passing_time = train_length / (train_speed_kmh * (5/18)) := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l2531_253177


namespace NUMINAMATH_CALUDE_poster_board_side_length_l2531_253105

/-- Prove that a square poster board that can fit 24 rectangular cards
    measuring 2 inches by 3 inches has a side length of 1 foot. -/
theorem poster_board_side_length :
  ∀ (side_length : ℝ),
  (side_length * side_length = 24 * 2 * 3) →
  (side_length / 12 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_poster_board_side_length_l2531_253105


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l2531_253170

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle time -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationWindow (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℕ :=
  3 * observationDuration  -- 3 color changes per cycle

/-- The probability of observing a color change -/
def probabilityOfChange (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℚ :=
  changeObservationWindow cycle observationDuration / cycleDuration cycle

theorem traffic_light_change_probability :
  let cycle := TrafficLightCycle.mk 45 5 40
  let observationDuration := 4
  probabilityOfChange cycle observationDuration = 2 / 15 := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_change_probability_l2531_253170


namespace NUMINAMATH_CALUDE_triangle_inconsistency_l2531_253196

theorem triangle_inconsistency : ¬ ∃ (a b c : ℝ),
  (a = 40 ∧ b = 50 ∧ c = 2 * (a + b) ∧ a + b + c = 160) ∧
  (a + b > c ∧ b + c > a ∧ a + c > b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inconsistency_l2531_253196


namespace NUMINAMATH_CALUDE_tickets_problem_l2531_253181

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

end NUMINAMATH_CALUDE_tickets_problem_l2531_253181


namespace NUMINAMATH_CALUDE_wheat_field_and_fertilizer_l2531_253102

theorem wheat_field_and_fertilizer 
  (field_size : ℕ) 
  (fertilizer_amount : ℕ) 
  (h1 : 6 * field_size = fertilizer_amount + 300)
  (h2 : 5 * field_size + 200 = fertilizer_amount) :
  field_size = 500 ∧ fertilizer_amount = 2700 := by
sorry

end NUMINAMATH_CALUDE_wheat_field_and_fertilizer_l2531_253102


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l2531_253174

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersection :
  ∀ (x y : ℝ), perpendicular_bisector x y ↔
  (∃ (t : ℝ), (1 - t) • A.1 + t • B.1 = x ∧ (1 - t) • A.2 + t • B.2 = y) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l2531_253174


namespace NUMINAMATH_CALUDE_problem_statement_l2531_253185

theorem problem_statement : (2002 - 1999)^2 / 169 = 9 / 169 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2531_253185


namespace NUMINAMATH_CALUDE_night_temperature_l2531_253120

def noon_temperature : Int := -2
def temperature_drop : Int := 4

theorem night_temperature : 
  noon_temperature - temperature_drop = -6 := by sorry

end NUMINAMATH_CALUDE_night_temperature_l2531_253120


namespace NUMINAMATH_CALUDE_inequality_solution_l2531_253188

theorem inequality_solution (x : ℝ) : 
  (x + 3) / (x + 4) > (4 * x + 5) / (3 * x + 10) ↔ 
  (x > -10/3 ∧ x < -4) ∨ 
  (x > (-1 - Real.sqrt 41) / 4 ∧ x < (-1 + Real.sqrt 41) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2531_253188


namespace NUMINAMATH_CALUDE_determinant_max_value_l2531_253104

open Real

theorem determinant_max_value :
  let det (θ : ℝ) := 
    let a11 := 1
    let a12 := 1
    let a13 := 1
    let a21 := 1
    let a22 := 1 + sin θ ^ 2
    let a23 := 1
    let a31 := 1 + cos θ ^ 2
    let a32 := 1
    let a33 := 1
    a11 * (a22 * a33 - a23 * a32) - 
    a12 * (a21 * a33 - a23 * a31) + 
    a13 * (a21 * a32 - a22 * a31)
  ∀ θ : ℝ, det θ ≤ 1 ∧ ∃ θ₀ : ℝ, det θ₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_determinant_max_value_l2531_253104


namespace NUMINAMATH_CALUDE_frame_interior_edges_sum_l2531_253180

/-- Represents a rectangular picture frame -/
structure Frame where
  outerLength : ℝ
  outerWidth : ℝ
  frameWidth : ℝ

/-- Calculates the area of the frame -/
def frameArea (f : Frame) : ℝ :=
  f.outerLength * f.outerWidth - (f.outerLength - 2 * f.frameWidth) * (f.outerWidth - 2 * f.frameWidth)

/-- Calculates the sum of the lengths of all four interior edges of the frame -/
def interiorEdgesSum (f : Frame) : ℝ :=
  2 * (f.outerLength - 2 * f.frameWidth) + 2 * (f.outerWidth - 2 * f.frameWidth)

/-- Theorem stating that for a frame with given dimensions, the sum of interior edges is 7 -/
theorem frame_interior_edges_sum :
  ∃ (f : Frame),
    f.outerLength = 7 ∧
    f.frameWidth = 2 ∧
    frameArea f = 30 ∧
    interiorEdgesSum f = 7 := by
  sorry

end NUMINAMATH_CALUDE_frame_interior_edges_sum_l2531_253180


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2531_253103

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

/-- Vector addition -/
def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

/-- Scalar multiplication of a vector -/
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

theorem parallel_vectors_sum (y : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, y)
  parallel a b → vec_add a (vec_scalar_mul 2 b) = (5, 10) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2531_253103


namespace NUMINAMATH_CALUDE_stating_cardboard_ratio_l2531_253137

/-- Represents the number of dogs on a Type 1 cardboard -/
def dogs_type1 : ℕ := 28

/-- Represents the number of cats on a Type 1 cardboard -/
def cats_type1 : ℕ := 28

/-- Represents the number of cats on a Type 2 cardboard -/
def cats_type2 : ℕ := 42

/-- Represents the required ratio of cats to dogs -/
def required_ratio : ℚ := 5 / 3

/-- 
Theorem stating that the ratio of Type 1 to Type 2 cardboard 
that satisfies the required cat to dog ratio is 9:4
-/
theorem cardboard_ratio : 
  ∀ (x y : ℚ), 
    x > 0 → y > 0 →
    (cats_type1 * x + cats_type2 * y) / (dogs_type1 * x) = required_ratio →
    x / y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_stating_cardboard_ratio_l2531_253137


namespace NUMINAMATH_CALUDE_new_student_weight_l2531_253146

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℚ) (new_avg : ℚ) :
  initial_count = 29 →
  initial_avg = 28 →
  new_avg = 27.5 →
  (initial_count * initial_avg + (initial_count + 1) * new_avg - initial_count * initial_avg) / (initial_count + 1) = 13 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l2531_253146


namespace NUMINAMATH_CALUDE_curve_translation_l2531_253142

-- Define a function representing the original curve
variable (f : ℝ → ℝ)

-- Define the translation
def translate (curve : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ :=
  fun x ↦ curve (x - h) + k

-- Theorem statement
theorem curve_translation (f : ℝ → ℝ) :
  ∃ h k : ℝ, 
    (translate f h k 2 = 3) ∧ 
    (translate f h k = fun x ↦ f (x - 1) + 2) ∧
    (h = 1) ∧ (k = 2) := by
  sorry


end NUMINAMATH_CALUDE_curve_translation_l2531_253142


namespace NUMINAMATH_CALUDE_gina_initial_amount_l2531_253110

def initial_amount (remaining : ℚ) (fraction_given : ℚ) : ℚ :=
  remaining / (1 - fraction_given)

theorem gina_initial_amount :
  let fraction_to_mom : ℚ := 1/4
  let fraction_for_clothes : ℚ := 1/8
  let fraction_to_charity : ℚ := 1/5
  let total_fraction_given := fraction_to_mom + fraction_for_clothes + fraction_to_charity
  let remaining_amount : ℚ := 170
  initial_amount remaining_amount total_fraction_given = 400 := by
  sorry

end NUMINAMATH_CALUDE_gina_initial_amount_l2531_253110


namespace NUMINAMATH_CALUDE_binomial_512_512_l2531_253125

theorem binomial_512_512 : Nat.choose 512 512 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_512_512_l2531_253125


namespace NUMINAMATH_CALUDE_yard_trees_l2531_253193

/-- The number of trees in a yard with given length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  yard_length / tree_spacing + 1

/-- Theorem: In a 400-meter yard with trees spaced 16 meters apart, there are 26 trees -/
theorem yard_trees : num_trees 400 16 = 26 := by
  sorry

end NUMINAMATH_CALUDE_yard_trees_l2531_253193


namespace NUMINAMATH_CALUDE_frustum_views_l2531_253167

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

end NUMINAMATH_CALUDE_frustum_views_l2531_253167


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2531_253118

theorem decimal_to_fraction : 
  (3.68 : ℚ) = 92 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2531_253118


namespace NUMINAMATH_CALUDE_sum_calculation_l2531_253156

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

def sum_squares_odd_integers (a b : ℕ) : ℕ :=
  List.sum (List.map (λ x => x * x) (List.filter (λ x => x % 2 = 1) (List.range (b - a + 1) |>.map (λ x => x + a))))

theorem sum_calculation :
  sum_integers 30 50 + count_even_integers 30 50 + sum_squares_odd_integers 30 50 = 17661 := by
  sorry

end NUMINAMATH_CALUDE_sum_calculation_l2531_253156


namespace NUMINAMATH_CALUDE_fraction_simplification_l2531_253194

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (x^2 - 1) / (x^2 - 2*x + 1) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2531_253194


namespace NUMINAMATH_CALUDE_total_homework_time_l2531_253155

-- Define the time left for each person
def jacob_time : ℕ := 18
def greg_time : ℕ := jacob_time - 6
def patrick_time : ℕ := 2 * greg_time - 4

-- Theorem to prove
theorem total_homework_time :
  jacob_time + greg_time + patrick_time = 50 :=
by sorry

end NUMINAMATH_CALUDE_total_homework_time_l2531_253155


namespace NUMINAMATH_CALUDE_point_not_in_second_quadrant_l2531_253123

theorem point_not_in_second_quadrant (m : ℝ) : 
  let P : ℝ × ℝ := (m + 1, m)
  ¬ (P.1 < 0 ∧ P.2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_second_quadrant_l2531_253123


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l2531_253169

theorem difference_of_squares_factorization (x y : ℝ) : 
  x^2 - 4*y^2 = (x + 2*y) * (x - 2*y) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l2531_253169


namespace NUMINAMATH_CALUDE_range_of_f_inverse_l2531_253184

/-- The function f(x) = 2 - log₂(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

/-- The inverse function of f -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem range_of_f_inverse :
  Set.range f = Set.Ioi 1 →
  Set.range f_inv = Set.Ioo 0 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_inverse_l2531_253184


namespace NUMINAMATH_CALUDE_triangle_side_length_l2531_253139

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  a = Real.sqrt 3 →
  b = 1 →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2531_253139


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l2531_253159

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_sum_zero 
  (a b c : ℝ) 
  (h1 : QuadraticFunction a b c 1 = 0)
  (h2 : QuadraticFunction a b c 5 = 0)
  (h_min : ∀ x, QuadraticFunction a b c x ≥ 36)
  (h_reaches_min : ∃ x, QuadraticFunction a b c x = 36) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l2531_253159


namespace NUMINAMATH_CALUDE_tomatoes_eaten_by_birds_l2531_253117

theorem tomatoes_eaten_by_birds 
  (initial_tomatoes : ℕ) 
  (remaining_tomatoes : ℕ) 
  (h1 : initial_tomatoes = 21)
  (h2 : remaining_tomatoes = 14) :
  initial_tomatoes - remaining_tomatoes = 7 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_eaten_by_birds_l2531_253117


namespace NUMINAMATH_CALUDE_team_formation_count_l2531_253128

/-- The number of ways to form a team of 4 students from 6 university students -/
def team_formation_ways : ℕ := 180

/-- The number of university students -/
def total_students : ℕ := 6

/-- The number of students in the team -/
def team_size : ℕ := 4

/-- The number of team leaders -/
def num_leaders : ℕ := 1

/-- The number of deputy team leaders -/
def num_deputies : ℕ := 1

/-- The number of ordinary members -/
def num_ordinary : ℕ := 2

theorem team_formation_count :
  team_formation_ways = 
    (total_students.choose num_leaders) * 
    ((total_students - num_leaders).choose num_deputies) * 
    ((total_students - num_leaders - num_deputies).choose num_ordinary) :=
sorry

end NUMINAMATH_CALUDE_team_formation_count_l2531_253128


namespace NUMINAMATH_CALUDE_expression_value_l2531_253131

theorem expression_value (x y : ℝ) (h : 2 * x + y = 1) :
  (y + 1)^2 - (y^2 - 4 * x + 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2531_253131


namespace NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l2531_253116

theorem diagonal_length_of_quadrilateral (offset1 offset2 area : ℝ) 
  (h1 : offset1 = 11)
  (h2 : offset2 = 9)
  (h3 : area = 400) :
  (2 * area) / (offset1 + offset2) = 40 :=
sorry

end NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l2531_253116


namespace NUMINAMATH_CALUDE_work_completion_rate_l2531_253127

theorem work_completion_rate (a_days : ℕ) (b_days : ℕ) : 
  a_days = 8 → b_days = a_days / 2 → (1 : ℚ) / a_days + (1 : ℚ) / b_days = 3 / 8 := by
  sorry

#check work_completion_rate

end NUMINAMATH_CALUDE_work_completion_rate_l2531_253127


namespace NUMINAMATH_CALUDE_three_planes_six_parts_l2531_253199

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane (this is a simplified representation)
  dummy : Unit

/-- The number of parts that a set of planes divides the space into -/
def num_parts (planes : List Plane3D) : Nat :=
  sorry

/-- Defines if three planes are collinear -/
def are_collinear (p1 p2 p3 : Plane3D) : Prop :=
  sorry

/-- Defines if two planes are parallel -/
def are_parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Defines if two planes intersect -/
def intersect (p1 p2 : Plane3D) : Prop :=
  sorry

theorem three_planes_six_parts 
  (p1 p2 p3 : Plane3D) 
  (h : num_parts [p1, p2, p3] = 6) :
  (are_collinear p1 p2 p3) ∨ 
  ((are_parallel p1 p2 ∧ intersect p1 p3 ∧ intersect p2 p3) ∨
   (are_parallel p1 p3 ∧ intersect p1 p2 ∧ intersect p3 p2) ∨
   (are_parallel p2 p3 ∧ intersect p2 p1 ∧ intersect p3 p1)) :=
by
  sorry

end NUMINAMATH_CALUDE_three_planes_six_parts_l2531_253199


namespace NUMINAMATH_CALUDE_emma_cookies_problem_l2531_253157

theorem emma_cookies_problem :
  ∃! (N : ℕ), N < 150 ∧ N % 13 = 7 ∧ N % 8 = 5 ∧ N = 85 := by
  sorry

end NUMINAMATH_CALUDE_emma_cookies_problem_l2531_253157


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_specific_l2531_253143

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- The bridge length is 195 meters given specific train parameters -/
theorem bridge_length_specific : bridge_length 180 45 30 = 195 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_specific_l2531_253143


namespace NUMINAMATH_CALUDE_skew_lines_iff_b_neq_two_sevenths_l2531_253191

def line1 (b t : ℝ) : ℝ × ℝ × ℝ := (2 + 3*t, 1 + 4*t, b + 2*t)
def line2 (u : ℝ) : ℝ × ℝ × ℝ := (5 + u, 3 - u, 2 + 2*u)

def are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem skew_lines_iff_b_neq_two_sevenths (b : ℝ) :
  are_skew b ↔ b ≠ 2/7 :=
sorry

end NUMINAMATH_CALUDE_skew_lines_iff_b_neq_two_sevenths_l2531_253191


namespace NUMINAMATH_CALUDE_number_problem_l2531_253160

theorem number_problem (x : ℝ) : 0.4 * x + 60 = x → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2531_253160


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l2531_253187

/-- The orthocenter of a triangle is the point where all three altitudes intersect. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (1/2, 8, 1/2) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l2531_253187


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2531_253171

theorem complex_equation_solution (a : ℝ) : 
  (a * Complex.I) / (2 - Complex.I) = 1 - 2 * Complex.I → a = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2531_253171


namespace NUMINAMATH_CALUDE_distinct_collections_l2531_253130

def word : String := "PHYSICS"

def num_magnets : ℕ := 7

def vowels_fallen : ℕ := 3

def consonants_fallen : ℕ := 3

def s_indistinguishable : Prop := True

theorem distinct_collections : ℕ := by
  sorry

end NUMINAMATH_CALUDE_distinct_collections_l2531_253130


namespace NUMINAMATH_CALUDE_baseball_team_members_l2531_253152

theorem baseball_team_members (
  pouches_per_pack : ℕ)
  (num_coaches : ℕ)
  (num_helpers : ℕ)
  (num_packs : ℕ)
  (h1 : pouches_per_pack = 6)
  (h2 : num_coaches = 3)
  (h3 : num_helpers = 2)
  (h4 : num_packs = 3)
  : ∃ (team_members : ℕ),
    team_members = num_packs * pouches_per_pack - num_coaches - num_helpers ∧
    team_members = 13 :=
by sorry

end NUMINAMATH_CALUDE_baseball_team_members_l2531_253152


namespace NUMINAMATH_CALUDE_largest_number_proof_l2531_253163

theorem largest_number_proof (x y z : ℝ) 
  (h_order : x < y ∧ y < z)
  (h_sum : x + y + z = 102)
  (h_diff1 : z - y = 10)
  (h_diff2 : y - x = 5) :
  z = 127 / 3 := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l2531_253163


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l2531_253132

theorem min_value_of_expression (x : ℝ) (h : x > 1) :
  2*x + 7/(x-1) ≥ 2*Real.sqrt 14 + 2 :=
sorry

theorem lower_bound_achievable :
  ∃ x > 1, 2*x + 7/(x-1) = 2*Real.sqrt 14 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l2531_253132


namespace NUMINAMATH_CALUDE_emmas_average_speed_l2531_253166

theorem emmas_average_speed 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ)
  (h1 : distance1 = 420)
  (h2 : time1 = 7)
  (h3 : distance2 = 480)
  (h4 : time2 = 8) :
  (distance1 + distance2) / (time1 + time2) = 60 := by
sorry

end NUMINAMATH_CALUDE_emmas_average_speed_l2531_253166


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_value_l2531_253135

/-- Triangle DEF with given side lengths -/
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (FD : ℝ)

/-- Lines parallel to the sides of the triangle -/
structure ParallelLines :=
  (ℓD : ℝ)  -- Length of intersection with triangle interior
  (ℓE : ℝ)
  (ℓF : ℝ)

/-- The perimeter of the inner triangle formed by parallel lines -/
def inner_triangle_perimeter (t : Triangle) (p : ParallelLines) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the inner triangle -/
theorem inner_triangle_perimeter_value :
  let t : Triangle := { DE := 150, EF := 250, FD := 200 }
  let p : ParallelLines := { ℓD := 65, ℓE := 55, ℓF := 25 }
  inner_triangle_perimeter t p = 990 :=
sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_value_l2531_253135


namespace NUMINAMATH_CALUDE_best_fit_highest_r_squared_model_1_best_fit_l2531_253154

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  name : String
  r_squared : Real

/-- Determines if a model has the best fit among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

/-- The model with the highest R² value has the best fit -/
theorem best_fit_highest_r_squared (models : List RegressionModel) (model : RegressionModel) 
    (h : model ∈ models) :
    has_best_fit model models ↔ ∀ m ∈ models, model.r_squared ≥ m.r_squared :=
  sorry

/-- Given four specific models, prove that Model ① has the best fit -/
theorem model_1_best_fit :
  let models : List RegressionModel := [
    ⟨"①", 0.976⟩,
    ⟨"②", 0.776⟩,
    ⟨"③", 0.076⟩,
    ⟨"④", 0.351⟩
  ]
  let model_1 : RegressionModel := ⟨"①", 0.976⟩
  has_best_fit model_1 models :=
  sorry

end NUMINAMATH_CALUDE_best_fit_highest_r_squared_model_1_best_fit_l2531_253154


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2531_253179

theorem cube_volume_problem (reference_cube_volume : ℝ) 
  (unknown_cube_surface_area : ℝ) (reference_cube_surface_area : ℝ) :
  reference_cube_volume = 8 →
  unknown_cube_surface_area = 3 * reference_cube_surface_area →
  reference_cube_surface_area = 6 * (reference_cube_volume ^ (1/3)) ^ 2 →
  let unknown_cube_side_length := (unknown_cube_surface_area / 6) ^ (1/2)
  unknown_cube_side_length ^ 3 = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2531_253179


namespace NUMINAMATH_CALUDE_stationery_difference_is_fifty_l2531_253198

/-- The number of stationery pieces Georgia has -/
def georgia_stationery : ℕ := 25

/-- The number of stationery pieces Lorene has -/
def lorene_stationery : ℕ := 3 * georgia_stationery

/-- The difference in stationery pieces between Lorene and Georgia -/
def stationery_difference : ℕ := lorene_stationery - georgia_stationery

theorem stationery_difference_is_fifty : stationery_difference = 50 := by
  sorry

end NUMINAMATH_CALUDE_stationery_difference_is_fifty_l2531_253198


namespace NUMINAMATH_CALUDE_marble_division_l2531_253192

theorem marble_division (x : ℝ) : 
  (5*x + 2) + (2*x - 1) + (x + 4) = 35 → 
  ∃ (a b c : ℕ), a + b + c = 35 ∧ 
    (a : ℝ) = 5*x + 2 ∧ 
    (b : ℝ) = 2*x - 1 ∧ 
    (c : ℝ) = x + 4 := by
  sorry

end NUMINAMATH_CALUDE_marble_division_l2531_253192


namespace NUMINAMATH_CALUDE_day_relationship_l2531_253109

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to determine the day of the week for a given day number -/
def dayOfWeek (dayNumber : ℕ) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- Theorem stating the relationship between days in different years -/
theorem day_relationship (N : ℕ) :
  dayOfWeek 290 = DayOfWeek.Wednesday →
  dayOfWeek 210 = DayOfWeek.Wednesday →
  dayOfWeek 110 = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_day_relationship_l2531_253109


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l2531_253140

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithmic_equation_solution :
  ∃! x : ℝ, x > 3 ∧ lg (x - 3) + lg x = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l2531_253140


namespace NUMINAMATH_CALUDE_min_value_fraction_l2531_253175

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 2) :
  (x + y + z) / (x * y * z) ≥ 27 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2531_253175


namespace NUMINAMATH_CALUDE_increasing_on_negative_reals_l2531_253162

theorem increasing_on_negative_reals (x₁ x₂ : ℝ) (h1 : x₁ < 0) (h2 : x₂ < 0) (h3 : x₁ < x₂) :
  2 * x₁ < 2 * x₂ := by
  sorry

end NUMINAMATH_CALUDE_increasing_on_negative_reals_l2531_253162


namespace NUMINAMATH_CALUDE_inverse_of_A_l2531_253108

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -2; 5, 3]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![3/22, 1/11; -5/22, 2/11]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2531_253108


namespace NUMINAMATH_CALUDE_max_product_sum_l2531_253165

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l2531_253165


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l2531_253173

/-- Represents the game board -/
def GameBoard := Fin 2020 → Fin 2020 → Option Bool

/-- Checks if there are k consecutive cells of the same color in a row or column -/
def has_k_consecutive (board : GameBoard) (k : ℕ) : Prop :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameBoard → Fin 2020 × Fin 2020

/-- Checks if a strategy is winning for the first player -/
def is_winning_strategy (strategy : Strategy) (k : ℕ) : Prop :=
  sorry

/-- The main theorem stating the condition for the first player's winning strategy -/
theorem first_player_winning_strategy :
  ∀ k : ℕ, (∃ strategy : Strategy, is_winning_strategy strategy k) ↔ k ≤ 1011 :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l2531_253173


namespace NUMINAMATH_CALUDE_linear_function_through_origin_l2531_253100

theorem linear_function_through_origin (k : ℝ) : 
  (∀ x y : ℝ, y = (k - 2) * x + (k^2 - 4)) →  -- Definition of the linear function
  ((0 : ℝ) = (k - 2) * (0 : ℝ) + (k^2 - 4)) →  -- The function passes through the origin
  (k - 2 ≠ 0) →  -- Ensure the function remains linear
  k = -2 := by sorry

end NUMINAMATH_CALUDE_linear_function_through_origin_l2531_253100


namespace NUMINAMATH_CALUDE_new_profit_percentage_after_doubling_price_l2531_253136

-- Define the initial profit percentage
def initial_profit_percentage : ℝ := 30

-- Define the price multiplier for the new selling price
def price_multiplier : ℝ := 2

-- Theorem to prove
theorem new_profit_percentage_after_doubling_price :
  let original_selling_price := 100 + initial_profit_percentage
  let new_selling_price := price_multiplier * original_selling_price
  let new_profit := new_selling_price - 100
  let new_profit_percentage := (new_profit / 100) * 100
  new_profit_percentage = 160 := by sorry

end NUMINAMATH_CALUDE_new_profit_percentage_after_doubling_price_l2531_253136


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2531_253183

/-- A hyperbola with asymptotes forming an acute angle of 60° and passing through (√2, √3) -/
structure Hyperbola where
  /-- The acute angle formed by the asymptotes -/
  angle : ℝ
  /-- The point through which the hyperbola passes -/
  point : ℝ × ℝ
  /-- The angle is 60° -/
  angle_is_60 : angle = 60 * π / 180
  /-- The point is (√2, √3) -/
  point_is_sqrt : point = (Real.sqrt 2, Real.sqrt 3)

/-- The standard equation of the hyperbola -/
def standard_equation (h : Hyperbola) : (ℝ → ℝ → Prop) → Prop :=
  λ eq ↦ (eq = λ x y ↦ x^2/1 - y^2/3 = 1) ∨ (eq = λ x y ↦ x^2/7 - y^2/(7/3) = 1)

/-- Theorem stating that the given hyperbola has one of the two standard equations -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  ∃ eq : ℝ → ℝ → Prop, standard_equation h eq :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2531_253183


namespace NUMINAMATH_CALUDE_problem_statement_l2531_253112

theorem problem_statement :
  (∀ x : ℝ, x ≠ 0 → x^2 + 1/x^2 ≥ 2) ∧
  (¬ ∃ x : ℝ, x^2 + 1/x^2 ≤ 2) ∧
  ((∃ x : ℝ, x^2 + 1/x^2 ≤ 2) ∨ (∀ x : ℝ, x ≠ 0 → x^2 + 1/x^2 > 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2531_253112


namespace NUMINAMATH_CALUDE_gathering_handshakes_l2531_253113

/-- Represents a group of people at a gathering --/
structure Gathering where
  total : ℕ
  groupA : ℕ
  groupB : ℕ
  knownInA : ℕ
  hTotal : total = groupA + groupB
  hKnownInA : knownInA ≤ groupA

/-- Calculates the number of handshakes in the gathering --/
def handshakes (g : Gathering) : ℕ :=
  let handshakesBetweenGroups := g.groupB * (g.groupA - g.knownInA)
  let handshakesWithinB := g.groupB * (g.groupB - 1) / 2 - g.groupB * g.knownInA
  handshakesBetweenGroups + handshakesWithinB

/-- Theorem stating the number of handshakes in the specific gathering --/
theorem gathering_handshakes :
  ∃ (g : Gathering),
    g.total = 40 ∧
    g.groupA = 25 ∧
    g.groupB = 15 ∧
    g.knownInA = 5 ∧
    handshakes g = 330 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l2531_253113


namespace NUMINAMATH_CALUDE_only_square_relationship_functional_l2531_253149

/-- Represents a relationship between two variables -/
structure Relationship where
  is_functional : Bool

/-- The relationship between the side length and the area of a square -/
def square_relationship : Relationship := sorry

/-- The relationship between rice yield and the amount of fertilizer applied -/
def rice_fertilizer_relationship : Relationship := sorry

/-- The relationship between snowfall and the rate of traffic accidents -/
def snowfall_accidents_relationship : Relationship := sorry

/-- The relationship between a person's height and weight -/
def height_weight_relationship : Relationship := sorry

/-- Theorem stating that only the square relationship is functional -/
theorem only_square_relationship_functional :
  square_relationship.is_functional ∧
  ¬rice_fertilizer_relationship.is_functional ∧
  ¬snowfall_accidents_relationship.is_functional ∧
  ¬height_weight_relationship.is_functional :=
by sorry

end NUMINAMATH_CALUDE_only_square_relationship_functional_l2531_253149


namespace NUMINAMATH_CALUDE_double_burger_cost_l2531_253189

/-- The cost of a double burger given the following conditions:
  - Total spent: $68.50
  - Total number of hamburgers: 50
  - Single burger cost: $1.00 each
  - Number of double burgers: 37
-/
theorem double_burger_cost :
  let total_spent : ℚ := 68.5
  let total_burgers : ℕ := 50
  let single_burger_cost : ℚ := 1
  let double_burgers : ℕ := 37
  let single_burgers : ℕ := total_burgers - double_burgers
  let double_burger_cost : ℚ := (total_spent - (single_burgers : ℚ) * single_burger_cost) / (double_burgers : ℚ)
  double_burger_cost = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_double_burger_cost_l2531_253189


namespace NUMINAMATH_CALUDE_polynomial_relationship_l2531_253111

def f (x : ℝ) : ℝ := x^2 + x

theorem polynomial_relationship : 
  (f 1 = 2) ∧ 
  (f 2 = 6) ∧ 
  (f 3 = 12) ∧ 
  (f 4 = 20) ∧ 
  (f 5 = 30) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_relationship_l2531_253111


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2531_253141

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 11) % 29 = 0 ∧
  (n + 11) % 53 = 0 ∧
  (n + 11) % 37 = 0 ∧
  (n + 11) % 41 = 0 ∧
  (n + 11) % 47 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 109871748 ∧
  ∀ m : ℕ, m < 109871748 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2531_253141


namespace NUMINAMATH_CALUDE_kims_average_round_answers_l2531_253107

/-- Represents the number of correct answers in each round of a math contest -/
structure ContestResults where
  easy : ℕ
  average : ℕ
  hard : ℕ

/-- Calculates the total points earned in the contest -/
def totalPoints (results : ContestResults) : ℕ :=
  2 * results.easy + 3 * results.average + 5 * results.hard

/-- Kim's contest results -/
def kimsResults : ContestResults := {
  easy := 6,
  average := 2,  -- This is what we want to prove
  hard := 4
}

theorem kims_average_round_answers :
  totalPoints kimsResults = 38 :=
by sorry

end NUMINAMATH_CALUDE_kims_average_round_answers_l2531_253107


namespace NUMINAMATH_CALUDE_expression_equals_one_l2531_253106

theorem expression_equals_one (x : ℝ) (h1 : x^3 ≠ -1) (h2 : x^3 ≠ 1) : 
  ((x+1)^3 * (x^2-x+1)^3 / (x^3+1)^3)^2 * ((x-1)^3 * (x^2+x+1)^3 / (x^3-1)^3)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2531_253106


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2531_253186

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2531_253186


namespace NUMINAMATH_CALUDE_correct_calculation_l2531_253144

theorem correct_calculation (a : ℝ) : -3*a - 2*a = -5*a := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2531_253144


namespace NUMINAMATH_CALUDE_circle_center_is_two_two_l2531_253114

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y - 10 = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 18

theorem circle_center_is_two_two :
  CircleCenter 2 2 := by sorry

end NUMINAMATH_CALUDE_circle_center_is_two_two_l2531_253114


namespace NUMINAMATH_CALUDE_last_three_digits_factorial_sum_15_l2531_253197

def last_three_digits (n : ℕ) : ℕ := n % 1000

def factorial_sum (n : ℕ) : ℕ :=
  (List.range n).map Nat.factorial |> List.sum

theorem last_three_digits_factorial_sum_15 :
  last_three_digits (factorial_sum 15) = 193 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_factorial_sum_15_l2531_253197


namespace NUMINAMATH_CALUDE_field_width_l2531_253168

/-- Proves that a rectangular field of length 60 m with a 2.5 m wide path around it,
    having a path area of 1200 sq m, has a width of 175 m. -/
theorem field_width (field_length : ℝ) (path_width : ℝ) (path_area : ℝ) :
  field_length = 60 →
  path_width = 2.5 →
  path_area = 1200 →
  ∃ field_width : ℝ,
    (field_length + 2 * path_width) * (field_width + 2 * path_width) -
    field_length * field_width = path_area ∧
    field_width = 175 := by
  sorry


end NUMINAMATH_CALUDE_field_width_l2531_253168


namespace NUMINAMATH_CALUDE_students_taking_neither_subject_l2531_253178

-- Define the total number of students in the drama club
def total_students : ℕ := 60

-- Define the number of students taking mathematics
def math_students : ℕ := 40

-- Define the number of students taking physics
def physics_students : ℕ := 35

-- Define the number of students taking both mathematics and physics
def both_subjects : ℕ := 25

-- Theorem to prove
theorem students_taking_neither_subject : 
  total_students - (math_students + physics_students - both_subjects) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_neither_subject_l2531_253178


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2531_253121

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given conditions
  (2 * a + b) * Real.cos C + c * Real.cos B = 0 →
  c = 2 * Real.sqrt 6 / 3 →
  Real.sin A * Real.cos B = (Real.sqrt 3 - 1) / 4 →
  -- Conclusions
  C = 2 * π / 3 ∧
  (1/2 * b * c * Real.sin A : Real) = (6 - 2 * Real.sqrt 3) / 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2531_253121


namespace NUMINAMATH_CALUDE_domain_f_real_iff_a_gt_one_range_f_real_iff_a_between_zero_and_one_decreasing_interval_g_l2531_253176

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) := Real.log (a * x^2 + 2 * x + 1)
noncomputable def g (x : ℝ) := Real.log (x^2 - 4 * x - 5) / Real.log (1/2)

-- Theorem 1: Domain of f is ℝ iff a > 1
theorem domain_f_real_iff_a_gt_one (a : ℝ) :
  (∀ x, ∃ y, f a x = y) ↔ a > 1 :=
sorry

-- Theorem 2: Range of f is ℝ iff 0 ≤ a ≤ 1
theorem range_f_real_iff_a_between_zero_and_one (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

-- Theorem 3: Decreasing interval of g is (5, +∞)
theorem decreasing_interval_g :
  ∀ x₁ x₂, x₁ > 5 → x₂ > 5 → x₁ < x₂ → g x₁ > g x₂ :=
sorry

end NUMINAMATH_CALUDE_domain_f_real_iff_a_gt_one_range_f_real_iff_a_between_zero_and_one_decreasing_interval_g_l2531_253176


namespace NUMINAMATH_CALUDE_divisor_problem_l2531_253172

theorem divisor_problem (d : ℕ) : 
  (∃ q₁ q₂ : ℕ, 100 = q₁ * d + 4 ∧ 90 = q₂ * d + 18) → d = 24 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l2531_253172


namespace NUMINAMATH_CALUDE_prob_compatible_donor_is_65_percent_l2531_253115

/-- Represents the blood types --/
inductive BloodType
  | O
  | A
  | B
  | AB

/-- Distribution of blood types in the population --/
def bloodTypeDistribution : BloodType → ℝ
  | BloodType.O  => 0.50
  | BloodType.A  => 0.15
  | BloodType.B  => 0.30
  | BloodType.AB => 0.05

/-- Predicate for blood types compatible with Type A --/
def compatibleWithA : BloodType → Prop
  | BloodType.O => True
  | BloodType.A => True
  | _ => False

/-- The probability of selecting a compatible donor for a Type A patient --/
def probCompatibleDonor : ℝ :=
  (bloodTypeDistribution BloodType.O) + (bloodTypeDistribution BloodType.A)

theorem prob_compatible_donor_is_65_percent :
  probCompatibleDonor = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_prob_compatible_donor_is_65_percent_l2531_253115


namespace NUMINAMATH_CALUDE_f_zero_and_no_extreme_value_l2531_253190

noncomputable section

/-- The function f(x) = (x+2)lnx + ax^2 - 4x + 7a -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 2) * Real.log x + a * x^2 - 4 * x + 7 * a

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x + (x + 2) / x + 2 * a * x - 4

theorem f_zero_and_no_extreme_value :
  (∀ x > 0, f (1/2) x = 0 ↔ x = 1) ∧
  (∀ a ≥ 1/2, ∀ x > 0, f_derivative a x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_zero_and_no_extreme_value_l2531_253190


namespace NUMINAMATH_CALUDE_jake_jill_difference_l2531_253164

/-- The number of peaches each person has -/
structure Peaches where
  jill : ℕ
  steven : ℕ
  jake : ℕ

/-- Given conditions about peach quantities -/
def peach_conditions (p : Peaches) : Prop :=
  p.jill = 87 ∧
  p.steven = p.jill + 18 ∧
  p.jake = p.steven - 5

/-- Theorem stating the difference between Jake's and Jill's peaches -/
theorem jake_jill_difference (p : Peaches) :
  peach_conditions p → p.jake - p.jill = 13 := by
  sorry

end NUMINAMATH_CALUDE_jake_jill_difference_l2531_253164


namespace NUMINAMATH_CALUDE_twenty_team_tournament_games_l2531_253151

/-- Calculates the number of games in a single-elimination tournament. -/
def tournament_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

/-- Theorem: A single-elimination tournament with 20 teams requires 19 games. -/
theorem twenty_team_tournament_games :
  tournament_games 20 = 19 := by
  sorry

#eval tournament_games 20

end NUMINAMATH_CALUDE_twenty_team_tournament_games_l2531_253151


namespace NUMINAMATH_CALUDE_max_leftover_pencils_l2531_253101

theorem max_leftover_pencils :
  ∀ (n : ℕ), 
  ∃ (q : ℕ), 
  n = 7 * q + (n % 7) ∧ 
  n % 7 ≤ 6 ∧
  ∀ (r : ℕ), r > n % 7 → r > 6 := by
  sorry

end NUMINAMATH_CALUDE_max_leftover_pencils_l2531_253101


namespace NUMINAMATH_CALUDE_parabola_directrix_l2531_253161

/-- A parabola is defined by its equation in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola is a line parallel to the x-axis -/
structure Directrix where
  y : ℝ

/-- Given a parabola y = (x^2 - 8x + 16) / 8, its directrix is y = -1/2 -/
theorem parabola_directrix (p : Parabola) (d : Directrix) :
  p.a = 1/8 ∧ p.b = -1 ∧ p.c = 2 → d.y = -1/2 := by
  sorry

#check parabola_directrix

end NUMINAMATH_CALUDE_parabola_directrix_l2531_253161


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2531_253129

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.I : ℂ) / (2 + Complex.I) = ⟨a, b⟩ := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2531_253129


namespace NUMINAMATH_CALUDE_coin_difference_is_six_l2531_253158

/-- Represents the available coin denominations in cents -/
def CoinDenominations : List ℕ := [5, 10, 25]

/-- The amount Paul needs to pay in cents -/
def AmountToPay : ℕ := 45

/-- Calculates the minimum number of coins needed to make the payment -/
def MinCoins (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

/-- Calculates the maximum number of coins needed to make the payment -/
def MaxCoins (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

/-- Theorem stating the difference between max and min coins is 6 -/
theorem coin_difference_is_six :
  MaxCoins AmountToPay CoinDenominations - MinCoins AmountToPay CoinDenominations = 6 :=
sorry

end NUMINAMATH_CALUDE_coin_difference_is_six_l2531_253158


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l2531_253150

/-- Converts a ternary (base-3) number to decimal --/
def ternary_to_decimal (t₂ t₁ t₀ : ℕ) : ℕ :=
  t₂ * 3^2 + t₁ * 3^1 + t₀ * 3^0

/-- The ternary number 121₃ is equal to the decimal number 16 --/
theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l2531_253150
