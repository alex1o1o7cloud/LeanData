import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_consequence_l255_25587

open Complex Matrix

theorem matrix_equation_consequence (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ) 
  (p q : ℂ) (h : p • (A * B) - q • (B * A) = 1) :
  (A * B - B * A) ^ n = 0 ∨ (q ≠ 0 ∧ ∃ k : ℕ, (p / q) ^ k = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_consequence_l255_25587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_count_l255_25573

def S : Finset ℕ := Finset.range 9

theorem function_count : 
  Fintype.card {f : {x // x ∈ S} → {x // x ∈ S} | 
    (∀ s, f (f (f s)) = s) ∧ 
    (∀ s, ¬(3 ∣ (f s).val - s.val))} = 288 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_count_l255_25573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plates_in_smaller_purchase_l255_25595

/-- The cost of a single paper plate -/
def P : ℝ := sorry

/-- The cost of a single paper cup -/
def C : ℝ := sorry

/-- The number of paper plates in the smaller purchase -/
def x : ℝ := sorry

/-- The total cost of 100 paper plates and 200 paper cups is $6.00 -/
axiom total_cost : 100 * P + 200 * C = 6

/-- The total cost of x paper plates and 40 paper cups is $1.20 -/
axiom smaller_cost : x * P + 40 * C = 1.2

theorem plates_in_smaller_purchase : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plates_in_smaller_purchase_l255_25595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_points_l255_25563

/-- The function h representing the distance between points A and B -/
noncomputable def h (a : ℝ) : ℝ := a^2 - Real.log a + 2

/-- The theorem stating the minimum value of h(a) for a > 0 -/
theorem min_distance_between_points (a : ℝ) (ha : a > 0) :
  h a ≥ (5 + Real.log 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_points_l255_25563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_sum_components_l255_25576

noncomputable def circle_radius : ℝ := 48
noncomputable def chord_length : ℝ := 90
noncomputable def intersection_distance : ℝ := 24

noncomputable def area_expression (p q r : ℕ) : ℝ := p * Real.pi - q * Real.sqrt (r : ℝ)

theorem area_sum_components (p q r : ℕ) : 
  circle_radius = 48 →
  chord_length = 90 →
  intersection_distance = 24 →
  (∃ (area : ℝ), area = area_expression p q r ∧ 
    area > 0 ∧ 
    (∀ (k : ℕ), k > 1 → ¬(k * k ∣ r))) →
  p + q + r = 2139 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_sum_components_l255_25576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equalization_l255_25582

/-- The amount LeRoy must give to equalize the costs -/
noncomputable def equalize_cost (A B C : ℝ) : ℝ := (B + C - 2 * A) / 3

/-- Theorem: Given the conditions, the amount LeRoy must give to equalize the costs is (B + C - 2A) / 3 dollars -/
theorem correct_equalization (A B C : ℝ) (hAB : A < B) (hAC : A < C) :
  equalize_cost A B C = (B + C - 2 * A) / 3 ∧
  equalize_cost A B C > 0 := by
  sorry

#check correct_equalization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equalization_l255_25582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_Q_l255_25594

def P : ℝ × ℝ := (1, 3)
def Q : ℝ × ℝ := (4, -1)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_P_Q : distance P Q = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_Q_l255_25594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_amplitude_and_period_l255_25521

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi/6 - 2*x) + Real.cos (2*x)

theorem f_amplitude_and_period :
  (∃ (A : ℝ), ∀ (x : ℝ), |f x| ≤ A ∧ (∃ (x₀ : ℝ), |f x₀| = A) ∧ A = Real.sqrt 3) ∧
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
   ∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T' ∧ T = Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_amplitude_and_period_l255_25521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_edges_for_monochromatic_triangle_l255_25558

/-- A type representing a point in a plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing an edge between two points. -/
structure Edge where
  p1 : Point
  p2 : Point

/-- A type representing a color (either red or blue). -/
inductive Color where
  | Red
  | Blue

/-- A function that checks if three points are collinear. -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- A function that checks if a triangle is monochromatic. -/
def monochromatic_triangle (e1 e2 e3 : Edge) (color : Edge → Color) : Prop :=
  color e1 = color e2 ∧ color e2 = color e3

theorem min_edges_for_monochromatic_triangle 
  (points : Finset Point) 
  (edges : Finset Edge) 
  (color : Edge → Color) :
  (points.card = 9) →
  (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬collinear p1 p2 p3) →
  (edges.card = 33) →
  (∀ e ∈ edges, e.p1 ∈ points ∧ e.p2 ∈ points) →
  (∃ e1 e2 e3, e1 ∈ edges ∧ e2 ∈ edges ∧ e3 ∈ edges ∧ monochromatic_triangle e1 e2 e3 color) ∧
  (∀ edges' : Finset Edge, edges'.card < 33 → 
    ∃ color' : Edge → Color, ∀ e1 e2 e3, e1 ∈ edges' → e2 ∈ edges' → e3 ∈ edges' → ¬monochromatic_triangle e1 e2 e3 color') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_edges_for_monochromatic_triangle_l255_25558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_10_2_bounds_l255_25551

-- Define the given conditions
axiom pow_10_3 : (10 : Real)^3 = 1000
axiom pow_10_5 : (10 : Real)^5 = 100000
axiom pow_2_15 : (2 : Real)^15 = 32768
axiom pow_2_16 : (2 : Real)^16 = 65536

-- Define the theorem to prove
theorem log_10_2_bounds : 1/5 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 5/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_10_2_bounds_l255_25551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_volume_l255_25536

-- Define the square pyramid
structure SquarePyramid where
  edgeLength : ℝ
  totalEdgeLength : ℝ

-- Define the conditions
def equalEdges (p : SquarePyramid) : Prop :=
  p.totalEdgeLength = 8 * p.edgeLength

def totalLength40 (p : SquarePyramid) : Prop :=
  p.totalEdgeLength = 40

-- Define the volume function
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.edgeLength ^ 2 * Real.sqrt (p.edgeLength ^ 2 - (p.edgeLength * Real.sqrt 2 / 2) ^ 2)

-- Theorem statement
theorem square_pyramid_volume 
  (p : SquarePyramid) 
  (h1 : equalEdges p) 
  (h2 : totalLength40 p) : 
  pyramidVolume p = 25 * Real.sqrt 12.5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_volume_l255_25536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l255_25508

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | n + 2 => (1 / 2) * sequence_a (n + 1) + 1

theorem sequence_a_closed_form (n : ℕ) :
  n ≥ 1 → sequence_a n = 2 - (1 / 2) ^ (n - 1) := by
  sorry

#eval sequence_a 5  -- Optional: to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l255_25508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perspective_difference_l255_25599

/-- The number of students in the school -/
def num_students : ℕ := 100

/-- The number of teachers in the school -/
def num_teachers : ℕ := 5

/-- The list of class enrollments -/
def class_enrollments : List ℕ := [50, 20, 20, 5, 5]

/-- The average number of students per class from a teacher's perspective -/
def t : ℚ := (class_enrollments.sum : ℚ) / num_teachers

/-- The average number of students per class from a student's perspective -/
def s : ℚ := (class_enrollments.map (fun x => x * x)).sum / num_students

/-- The difference between teacher's and student's perspectives -/
theorem perspective_difference : t - s = -27/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perspective_difference_l255_25599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_project_cost_theorem_l255_25502

/-- Calculates the total cost of a barbed wire project for a square field -/
noncomputable def barbed_wire_project_cost (field_area : ℝ) (wire_cost_per_meter : ℝ) 
  (additional_wire : ℝ) (num_gates : ℕ) (gate_width : ℝ) : ℝ :=
  let side_length := (field_area : ℝ).sqrt
  let perimeter := 4 * side_length
  let wire_length := perimeter - (num_gates : ℝ) * gate_width + additional_wire
  wire_length * wire_cost_per_meter

/-- The total cost of the barbed wire project is 1005.75 -/
theorem barbed_wire_project_cost_theorem : 
  barbed_wire_project_cost 12544 2.25 5 3 2 = 1005.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_project_cost_theorem_l255_25502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pipe_fill_time_l255_25509

/-- Given two pipes with different fill rates, calculate the time to fill a tank together -/
theorem two_pipe_fill_time (slow_pipe_time : ℝ) (fast_pipe_ratio : ℝ) : 
  slow_pipe_time > 0 → 
  fast_pipe_ratio > 1 →
  let fast_pipe_time := slow_pipe_time / fast_pipe_ratio
  let combined_rate := 1 / fast_pipe_time + 1 / slow_pipe_time
  let combined_time := 1 / combined_rate
  slow_pipe_time = 180 →
  fast_pipe_ratio = 4 →
  combined_time = 36 := by
  intros h1 h2 h3 h4
  -- The proof goes here
  sorry

#check two_pipe_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pipe_fill_time_l255_25509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_theorem_l255_25538

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Checks if two integers are relatively prime -/
def isRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- Checks if a natural number is not divisible by the square of any prime -/
def notDivisibleBySquareOfPrime (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → n % (p^2) ≠ 0

theorem sphere_triangle_distance_theorem
  (O P Q R : Point3D)
  (hRadius : distance O P = 25 ∧ distance O Q = 25 ∧ distance O R = 25)
  (hPQ : distance P Q = 20)
  (hQR : distance Q R = 21)
  (hRP : distance R P = 29)
  (x y z : ℕ)
  (hPositive : x > 0 ∧ y > 0 ∧ z > 0)
  (hRelativePrime : isRelativelyPrime x z)
  (hNotSquareDivisible : notDivisibleBySquareOfPrime y)
  (hDistance : distance O (Point3D.mk 0 0 0) = (x * Real.sqrt (y : ℝ)) / z) :
  x + y + z = 46 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_theorem_l255_25538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_games_for_prediction_l255_25546

/-- Represents the chess tournament setup -/
structure ChessTournament where
  white_rook_students : ℕ
  black_elephant_students : ℕ
  games_per_white_student : ℕ
  total_games : ℕ

/-- Defines the specific tournament from the problem -/
def tournament : ChessTournament :=
  { white_rook_students := 15
  , black_elephant_students := 20
  , games_per_white_student := 20
  , total_games := 300 }

/-- Predicate representing that a participant can be named for the next game after n games -/
def participant_in_next_game (n : ℕ) (participant : ℕ) : Prop :=
  sorry

/-- Theorem stating the minimum number of games after which Sasha can predict a participant -/
theorem min_games_for_prediction (t : ChessTournament) (n : ℕ) : 
  (t.white_rook_students = tournament.white_rook_students) →
  (t.black_elephant_students = tournament.black_elephant_students) →
  (t.games_per_white_student = tournament.games_per_white_student) →
  (t.total_games = tournament.total_games) →
  (n ≥ t.white_rook_students * t.games_per_white_student - t.games_per_white_student) →
  (∀ m : ℕ, m < n → ¬(∃ participant, participant_in_next_game m participant)) →
  (∃ participant, participant_in_next_game n participant) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_games_for_prediction_l255_25546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_exists_min_omega_l255_25566

/-- The function that reaches its maximum at x = 2 -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

/-- Theorem stating the minimum positive value of ω -/
theorem min_omega_value (ω : ℝ) (h_pos : ω > 0) (h_max : ∀ x, f ω x ≤ f ω 2) : ω ≥ Real.pi / 6 := by
  sorry

/-- Theorem stating the existence of ω that satisfies the conditions -/
theorem exists_min_omega : ∃ ω : ℝ, ω > 0 ∧ (∀ x, f ω x ≤ f ω 2) ∧ ω = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_exists_min_omega_l255_25566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l255_25549

noncomputable def f (x : ℝ) := x^2 * Real.sin x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → HasDerivAt f (2*x*Real.sin x + x^2*Real.cos x) x) ∧
  (∀ x, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → 2*x*Real.sin x + x^2*Real.cos x ≥ 0) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → x₂ ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → (x₁ + x₂) * (f x₁ + f x₂) ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l255_25549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_one_abs_l255_25550

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (1 - 3*i) / (1 + i)

theorem z_plus_one_abs : Complex.abs (z + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_one_abs_l255_25550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_correct_l255_25565

/-- The compound interest formula -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The initial investment amount in rupees -/
def initial_investment : ℝ := 8000

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.05

/-- The final amount after interest in rupees -/
def final_amount : ℝ := 8820

/-- The investment duration in years -/
def investment_duration : ℕ := 2

theorem investment_duration_correct :
  ⌊Real.log (final_amount / initial_investment) / Real.log (1 + interest_rate)⌋ = investment_duration := by
  sorry

#eval investment_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_correct_l255_25565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indeterminate_ratio_l255_25519

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem indeterminate_ratio (seq : ArithmeticSequence) 
  (h : 8 * seq.a 2 + seq.a 5 = 0) :
  (∃ k : ℚ, ∀ n : ℕ, S seq (n + 1) / S seq n = k) → False ∧ 
  (∃ k : ℚ, seq.a 5 / seq.a 3 = k) ∧
  (∃ k : ℚ, S seq 5 / S seq 3 = k) ∧
  (∃ k : ℚ, ∀ n : ℕ, seq.a (n + 1) / seq.a n = k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indeterminate_ratio_l255_25519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_count_theorem_l255_25510

def total_books : ℕ := 250
def history_percentage : ℚ := 30 / 100
def geography_percentage : ℚ := 20 / 100
def science_percentage : ℚ := 15 / 100
def literature_percentage : ℚ := 10 / 100

def math_percentage : ℚ := 1 - (history_percentage + geography_percentage + science_percentage + literature_percentage)

def history_books : ℕ := Int.toNat ((history_percentage * total_books).floor)
def geography_books : ℕ := Int.toNat ((geography_percentage * total_books).floor)
def science_books : ℕ := Int.toNat ((science_percentage * total_books).floor)
def math_books : ℕ := Int.toNat ((math_percentage * total_books).floor)

theorem book_count_theorem :
  history_books + geography_books + science_books + math_books = 224 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_count_theorem_l255_25510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angelinas_speed_l255_25533

-- Define the constants
def distance_home_to_grocery : ℝ := 960
def distance_grocery_to_gym : ℝ := 480
def time_difference : ℝ := 40

-- Define Angelina's speed from home to grocery
noncomputable def speed_home_to_grocery : ℝ → ℝ := λ v => v

-- Define Angelina's speed from grocery to gym
noncomputable def speed_grocery_to_gym : ℝ → ℝ := λ v => 2 * v

-- Define the time taken from home to grocery
noncomputable def time_home_to_grocery : ℝ → ℝ := λ v => distance_home_to_grocery / (speed_home_to_grocery v)

-- Define the time taken from grocery to gym
noncomputable def time_grocery_to_gym : ℝ → ℝ := λ v => distance_grocery_to_gym / (speed_grocery_to_gym v)

-- Theorem statement
theorem angelinas_speed : 
  ∃ v : ℝ, v > 0 ∧ 
  time_home_to_grocery v - time_grocery_to_gym v = time_difference ∧
  speed_grocery_to_gym v = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angelinas_speed_l255_25533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l255_25555

/-- Given an angle α in the Cartesian coordinate system with its vertex at the origin,
    its initial side coinciding with the non-negative half-axis of the x-axis,
    and passing through the point P(-5, -12), prove that cos α = -5/13 -/
theorem cos_alpha_value (α : ℝ) (P : ℝ × ℝ) : 
  P.1 = -5 → P.2 = -12 → Real.cos α = -5/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l255_25555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_max_path_length_l255_25511

/-- The maximum path length for a fly in a 1×1×2 box visiting all corners --/
theorem fly_max_path_length :
  let box_length : ℝ := 1
  let box_width : ℝ := 1
  let box_height : ℝ := 2
  let max_path_length : ℝ := Real.sqrt 6 + 2 * Real.sqrt 5 + Real.sqrt 2 + 1
  ∀ path : List (ℝ × ℝ × ℝ),
    (∀ corner, corner ∈ [(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,2), (1,0,2), (0,1,2), (1,1,2)] → 
      corner ∈ path) →
    path.head? = some (0,0,0) →
    path.getLast? = some (1,1,2) →
    path.length = 8 →
    (∀ i, i ∈ List.range (path.length - 1) → 
      let (x₁, y₁, z₁) := path[i]!
      let (x₂, y₂, z₂) := path[i+1]!
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2) ≤ max_path_length) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_max_path_length_l255_25511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_C_D_l255_25506

def C : ℕ := 2 * 3 + 4 * 5 + 6 * 7 + Finset.sum (Finset.range 19) (λ n ↦ (2*n + 2) * (2*n + 3)) + 40

def D : ℕ := 2 + 3 * 4 + 5 * 6 + Finset.sum (Finset.range 19) (λ n ↦ (2*n + 3) * (2*n + 4)) + 39 * 40

theorem positive_difference_C_D : |Int.ofNat C - Int.ofNat D| = 361 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_C_D_l255_25506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_solution_l255_25561

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts a list of digits to a natural number -/
def listToNat (digits : List Digit) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d.val) 0

/-- Checks if all elements in a list are distinct -/
def allDistinct (l : List Digit) : Prop :=
  ∀ i j, i < j → j < l.length → l.get ⟨i, by sorry⟩ ≠ l.get ⟨j, by sorry⟩

theorem rebus_solution :
  ∃! (a b c d : Digit),
    allDistinct [a, b, c, d] ∧
    a.val ≠ 0 ∧ b.val ≠ 0 ∧ c.val ≠ 0 ∧ d.val ≠ 0 ∧
    listToNat [a, b, c, a] = 182 * listToNat [c, d] ∧
    listToNat [a, b, c, d] = 2916 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_solution_l255_25561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l255_25556

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := 2 * n - 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2^(n + 1)

-- Define the sum of the first n terms of b_n
def sum_b (n : ℕ) : ℝ := 2^(n + 1) - 2

theorem arithmetic_and_geometric_sequences :
  (∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) - a n = d) →  -- a_n is arithmetic with positive difference
  a 3 * a 6 = 55 →                                   -- a_3 * a_6 = 55
  a 2 + a 7 = 16 →                                   -- a_2 + a_7 = 16
  (∀ n : ℕ+, (Finset.range n).sum (λ i => b i / 2^i) = a n) →  -- Relation between a_n and b_n
  (∀ n : ℕ, a n = 2 * n - 1) ∧                       -- General formula for a_n
  (∀ n : ℕ, sum_b n = 2^(n + 1) - 2)                 -- Sum of first n terms of b_n
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l255_25556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_positive_integer_l255_25530

theorem fraction_positive_integer (p : ℕ) :
  (p > 0) →
  (((5 * p + 36) / (2 * p - 9) : ℚ).isInt ∧ ((5 * p + 36) / (2 * p - 9) : ℚ) > 0) ↔
  p ∈ ({5, 6, 9, 18} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_positive_integer_l255_25530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l255_25512

def sequence_term (n : Nat) : Nat :=
  Nat.factorial n + n^2

def sequence_sum : Nat :=
  (List.range 9).map (fun i => sequence_term (i + 1)) |>.sum

theorem units_digit_of_sequence_sum :
  sequence_sum % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l255_25512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l255_25548

theorem right_triangle_area (a c : ℝ) (h1 : a = 6) (h2 : c = 10) : 
  (1/2) * a * Real.sqrt (c^2 - a^2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l255_25548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l255_25537

/-- Given that two workers A and B can complete a job together in a certain number of days,
    and worker A can complete the job alone in a certain number of days,
    this function calculates how long it would take worker B to complete the job alone. -/
noncomputable def time_for_b_alone (time_together time_a_alone : ℝ) : ℝ :=
  1 / (1 / time_together - 1 / time_a_alone)

/-- Theorem stating that if A and B together can complete a work in 8 days,
    and A alone can complete the same work in 12 days,
    then B alone can complete the work in 24 days. -/
theorem b_alone_time (time_together : ℝ) (time_a_alone : ℝ)
    (h1 : time_together = 8)
    (h2 : time_a_alone = 12) :
    time_for_b_alone time_together time_a_alone = 24 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l255_25537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_theorem_l255_25524

theorem triangle_function_theorem (a b c A B C : Real) (f : Real → Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C →
  (a - c) * (a + c) = b * (b - c) →
  (∀ x, f x = 2 * Real.sin x * Real.cos (π/2 - x) - Real.sqrt 3 * Real.sin (π + x) * Real.cos x + Real.sin (π/2 + x) * Real.cos x) →
  f B = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_theorem_l255_25524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_perpendicular_lines_a_value_l255_25560

/-- Two lines are perpendicular if the product of their slopes is -1 -/
theorem perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line -/
def m1 : ℚ := -3

/-- The slope of the second line in terms of a -/
def m2 (a : ℚ) : ℚ := a / 4

/-- The theorem stating that if the given lines are perpendicular, then a = 4/3 -/
theorem perpendicular_lines_a_value (a : ℚ) :
  perpendicular (m1 : ℝ) ((m2 a) : ℝ) → a = 4/3 := by
  intro h
  -- Proof steps would go here
  sorry

#check perpendicular_lines_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_perpendicular_lines_a_value_l255_25560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l255_25552

/-- In a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_property (a b c : ℝ) (A B C : ℝ) 
  (h : Triangle a b c A B C) 
  (h1 : (2*a - c) * Real.cos B = b * Real.cos C) : 
  B = Real.pi/3 ∧ 
  (∀ A, 0 < A → A < Real.pi → Real.sin A * (-1) + 1 ≥ 0) ∧
  (∃ A, 0 < A ∧ A < Real.pi ∧ Real.sin A * (-1) + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l255_25552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_l255_25516

theorem sine_transformation (ω φ : ℝ) (h1 : ω > 0) (h2 : -π/2 < φ) (h3 : φ < π/2) :
  (∀ x, Real.sin (ω * x + φ) = Real.sin (2 * (x - π/6))) → ω = 1/2 ∧ φ = π/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_l255_25516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_l255_25513

theorem cos_equation_solution (x : ℝ) :
  Real.cos (3 * x) * Real.cos (6 * x) = Real.cos (4 * x) * Real.cos (7 * x) →
  ∃ n : ℤ, x = n * Real.pi / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_l255_25513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l255_25504

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A

theorem triangle_properties (t : Triangle) (h : given_condition t) :
  t.A = π / 6 ∧
  Set.Icc (-(Real.sqrt 3 + 2) / 2) (Real.sqrt 3 - 1) =
    Set.range (fun B => Real.cos (5 * π / 2 - B) - 2 * (Real.sin (t.C / 2)) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l255_25504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l255_25559

/-- Given a function f such that f(x+1) = x^2 - 5 for all x,
    prove that f(x) = x^2 - 2x - 4 for all x. -/
theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 5) :
  ∀ x, f x = x^2 - 2*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l255_25559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_range_g_intersection_l255_25527

-- Define the functions f and g
noncomputable def f (x : ℝ) := 1 / Real.sqrt (1 - x)
noncomputable def g (x : ℝ) := Real.log x

-- Define the domain of f
def A : Set ℝ := {x | x < 1}

-- Define the range of g (which is all real numbers)
def B : Set ℝ := Set.univ

-- State the theorem
theorem domain_f_range_g_intersection :
  A ∩ B = Set.Iio 1 := by sorry

#check domain_f_range_g_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_range_g_intersection_l255_25527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_unmatched_teams_l255_25597

/-- Represents a football championship. -/
structure Championship where
  teams : ℕ
  rounds : ℕ
  all_paired : Bool
  no_repeats : Bool

/-- Represents a match between two teams in a specific round. -/
structure Match where
  team1 : ℕ
  team2 : ℕ
  round : ℕ

/-- Theorem stating that in a championship with 18 teams and 8 rounds, 
    where all teams are paired each round and no pairs repeat, 
    there exist three teams that have not played against each other. -/
theorem exist_three_unmatched_teams (c : Championship) 
  (h1 : c.teams = 18) 
  (h2 : c.rounds = 8) 
  (h3 : c.all_paired = true) 
  (h4 : c.no_repeats = true) : 
  ∃ (t1 t2 t3 : ℕ), t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ 
  t1 ≤ c.teams ∧ t2 ≤ c.teams ∧ t3 ≤ c.teams ∧ 
  (¬ ∃ (m : Match), m.round ≤ c.rounds ∧ 
    ((m.team1 = t1 ∧ m.team2 = t2) ∨ 
     (m.team1 = t1 ∧ m.team2 = t3) ∨ 
     (m.team1 = t2 ∧ m.team2 = t3) ∨
     (m.team1 = t2 ∧ m.team2 = t1) ∨ 
     (m.team1 = t3 ∧ m.team2 = t1) ∨ 
     (m.team1 = t3 ∧ m.team2 = t2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_unmatched_teams_l255_25597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l255_25571

noncomputable def sequence_a (n : ℕ) : ℝ := 2^n - 1

noncomputable def S (n : ℕ) : ℝ := 2 * sequence_a n - n

axiom S_def (n : ℕ) : S n = 2 * sequence_a n - n

noncomputable def b (n : ℕ) : ℝ := 1 / (sequence_a n + 1) + 1 / (sequence_a n * sequence_a (n + 1) + 1)

noncomputable def T (n : ℕ) : ℝ := 1 - 1 / (2^(n + 1) - 1)

theorem sequence_properties :
  (∀ n : ℕ, sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, sequence_a n = 2^n - 1) ∧
  (∀ n : ℕ, T n = 1 - 1 / (2^(n + 1) - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l255_25571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_running_field_length_calc_l255_25523

/-- The length of a running field in kilometers, given the number of steps for one lap,
    the number of laps in a session, the total steps in a session, and the average step length. -/
noncomputable def running_field_length (steps_per_lap : ℕ) (laps_per_session : ℚ) 
  (total_steps : ℕ) (avg_step_length : ℝ) : ℝ :=
  (steps_per_lap : ℝ) * avg_step_length / 1000

/-- Theorem stating that the length of the running field is approximately 4.0767 kilometers. -/
theorem running_field_length_calc :
  let steps_per_lap : ℕ := 5350
  let laps_per_session : ℚ := 5/2
  let total_steps : ℕ := 13375
  let avg_step_length : ℝ := 0.762
  abs (running_field_length steps_per_lap laps_per_session total_steps avg_step_length - 4.0767) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_running_field_length_calc_l255_25523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_value_36k_minus_5l_l255_25574

theorem smallest_absolute_value_36k_minus_5l :
  (∀ k l : ℕ, k > 0 → l > 0 → |((36 : ℤ) ^ k) - ((5 : ℤ) ^ l)| ≥ 11) ∧
  (∃ k l : ℕ, k > 0 ∧ l > 0 ∧ |((36 : ℤ) ^ k) - ((5 : ℤ) ^ l)| = 11) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_value_36k_minus_5l_l255_25574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l255_25589

/-- A trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- The area formula for a trapezium -/
noncomputable def trapezium_area (t : Trapezium) : ℝ := (t.side1 + t.side2) * t.height / 2

/-- Theorem: Given a trapezium with one side 16 cm, height 14 cm, and area 196 cm², the other side is 12 cm -/
theorem trapezium_side_length (t : Trapezium) 
    (h1 : t.side1 = 16)
    (h2 : t.height = 14)
    (h3 : t.area = 196)
    (h4 : t.area = trapezium_area t) : 
  t.side2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l255_25589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l255_25543

noncomputable def f (x : ℝ) : ℝ := x^2 / Real.exp x

noncomputable def f_deriv (x : ℝ) : ℝ := (x * (2 - x)) / Real.exp x

theorem tangent_line_and_extrema (x : ℝ) :
  (∃ (t : ℝ → ℝ), t x = x / Real.exp 1 ∧ 
    ∀ y, y = f x → (t y - t 0) = (f_deriv x) * (y - 0)) ∧
  (∀ y ∈ Set.Ici (-3), f y ≤ 9 / Real.exp 3) ∧
  (∃ z ∈ Set.Ici (-3), f z = 0) ∧
  (∀ y ∈ Set.Ici (-3), f y ≥ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l255_25543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_inequality_l255_25515

-- Helper definitions
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry
def is_inscribed_in_circle (s : Set (ℝ × ℝ)) (a : ℝ) : Prop := sorry
def is_circumscribed_around_circle (s : Set (ℝ × ℝ)) (c : ℝ) : Prop := sorry

theorem polygon_area_inequality (A B C : ℝ) 
  (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h4 : ∃ (polygon : Set (ℝ × ℝ)), 
    area polygon = B ∧ 
    is_inscribed_in_circle polygon A ∧ 
    is_circumscribed_around_circle polygon C) :
  2 * B ≤ A + C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_inequality_l255_25515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_interest_rate_is_ten_percent_l255_25529

/-- Calculates the final value of an investment after two consecutive 6-month periods with different interest rates -/
noncomputable def investment_value (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial_amount * (1 + rate1 / 2) * (1 + rate2 / 2)

/-- Theorem stating that given the initial investment and final value, the second interest rate is 10% -/
theorem second_interest_rate_is_ten_percent
  (initial_amount : ℝ)
  (final_amount : ℝ)
  (rate1 : ℝ)
  (h1 : initial_amount = 10000)
  (h2 : final_amount = 11130)
  (h3 : rate1 = 0.12)
  : ∃ (rate2 : ℝ), investment_value initial_amount rate1 rate2 = final_amount ∧ rate2 = 0.10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_interest_rate_is_ten_percent_l255_25529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l255_25569

noncomputable def f (x : ℝ) := x^2 + 2*x - 4

theorem functional_equation_solution :
  let solutions : Set ℝ := {(-1 + Real.sqrt 17)/2, (-1 - Real.sqrt 17)/2, (-3 + Real.sqrt 13)/2, (-3 - Real.sqrt 13)/2}
  ∀ x : ℝ, f (f x) = x ↔ x ∈ solutions :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l255_25569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_root_of_g_between_3_and_4_l255_25539

noncomputable def g (x : ℝ) := 2 * Real.sin x + 3 * Real.cos x + 4 * Real.tan x

theorem smallest_root_of_g_between_3_and_4 :
  ∃ s : ℝ, s > 0 ∧ g s = 0 ∧ (∀ x, x > 0 ∧ g x = 0 → s ≤ x) ∧ 3 ≤ s ∧ s < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_root_of_g_between_3_and_4_l255_25539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l255_25564

theorem diophantine_equation_solution : 
  {(m, n) : ℕ × ℕ | m > 0 ∧ n > 0 ∧ 3 * 2^m + 1 = n^2} = {(3, 5), (4, 7)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l255_25564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_sum_l255_25525

/-- The original function f(x) -/
noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

/-- The shifted function g(x) -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x)

/-- Theorem stating the minimum value of g(x) + g(x/2) -/
theorem min_value_of_g_sum :
  ∃ (m : ℝ), ∀ (x : ℝ), m ≤ g x + g (x/2) ∧ m = -9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_sum_l255_25525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_in_first_and_last_game_l255_25598

/-- Represents a chess tournament --/
structure Tournament (n : ℕ) where
  /-- The number of players in the tournament --/
  num_players : ℕ
  /-- The number of players is 2n+3 --/
  player_count : num_players = 2*n + 3
  /-- Each player plays exactly once with every other player --/
  games_per_player : ℕ
  games_count : games_per_player = 2*n + 2
  /-- Minimum rest period between games for each player --/
  min_rest : ℕ
  rest_period : min_rest = n

/-- Theorem: In a tournament satisfying the given conditions, 
    at least one player from the first game also plays in the last game --/
theorem player_in_first_and_last_game (n : ℕ) (t : Tournament n) :
  ∃ (p : ℕ), p ∈ Finset.range t.num_players ∧ 
    (p ∈ Finset.range 2 ∧ p ∈ Finset.range 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_in_first_and_last_game_l255_25598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_equality_condition_l255_25593

theorem ln_equality_condition (x y : ℝ) :
  (∀ (x y : ℝ), x > 0 → y > 0 → Real.log x = Real.log y → x = y) ∧
  (∃ (x y : ℝ), x = y ∧ ¬(Real.log x = Real.log y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_equality_condition_l255_25593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_N_with_digit_5_l255_25590

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sequence a_n -/
noncomputable def a (n : ℕ+) : ℤ :=
  floor (n : ℝ)

/-- A number contains the digit 5 in its decimal representation -/
def contains_digit_5 (z : ℤ) : Prop :=
  ∃ (k : ℕ), (z / (10 ^ k)) % 10 = 5

/-- The main theorem -/
theorem exists_N_with_digit_5 :
  ∃ (N : ℕ+), ∀ (k : ℕ+), ∃ (i : ℕ), i < N.val ∧ contains_digit_5 (a (k + ⟨i + 1, Nat.succ_pos i⟩)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_N_with_digit_5_l255_25590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l255_25534

-- Define the parabola
def Parabola (C : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ), p > 0 ∧ C = {xy : ℝ × ℝ | xy.2^2 = 4*xy.1}

-- Define the focus
def Focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def Directrix : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = -1}

-- Define a line passing through K(-1, 0)
def Line (m : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = m*xy.2 - 1}

-- Define the intersection points of the line and the parabola
def Intersection (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  C ∩ l

-- Define symmetry with respect to x-axis
def SymmetricX (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

-- Define the dot product of vectors
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Main theorem
theorem parabola_properties (C : Set (ℝ × ℝ)) (m : ℝ) :
  Parabola C →
  m ≠ 0 →
  let l := Line m
  let AB := Intersection C l
  ∃ (A B D : ℝ × ℝ),
    A ∈ AB ∧ B ∈ AB ∧
    SymmetricX A D →
    (Focus ∈ {xy : ℝ × ℝ | ∃ t, xy.1 = B.1 + t*(D.1 - B.1) ∧ xy.2 = B.2 + t*(D.2 - B.2)}) ∧
    (DotProduct (A.1 - 1, A.2) (B.1 - 1, B.2) = 8/9 →
      m = 4/3 ∨ m = -4/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l255_25534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concurrent_lines_l255_25540

-- Define the necessary structures and objects
structure Point where
  x : ℝ
  y : ℝ

def Circle (center : Point) (radius : ℝ) : Set Point :=
  {p : Point | (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2}

def Line (p q : Point) : Set Point :=
  {r : Point | (r.y - p.y) * (q.x - p.x) = (r.x - p.x) * (q.y - p.y)}

def collinear (p q r : Point) : Prop :=
  (r.y - p.y) * (q.x - p.x) = (r.x - p.x) * (q.y - p.y)

def concurrent (l₁ l₂ l₃ : Set Point) : Prop :=
  ∃ (p : Point), p ∈ l₁ ∧ p ∈ l₂ ∧ p ∈ l₃

-- State the theorem
theorem concurrent_lines
  (A B C D X Y O M N : Point)
  (Γ₁ Γ₂ : Set Point)
  (h₁ : collinear A B C)
  (h₂ : collinear B C D)
  (h₃ : Γ₁ = Circle ⟨(A.x + C.x) / 2, (A.y + C.y) / 2⟩ (((C.x - A.x)^2 + (C.y - A.y)^2) / 4))
  (h₄ : Γ₂ = Circle ⟨(B.x + D.x) / 2, (B.y + D.y) / 2⟩ (((D.x - B.x)^2 + (D.y - B.y)^2) / 4))
  (h₅ : X ∈ Γ₁ ∧ X ∈ Γ₂)
  (h₆ : Y ∈ Γ₁ ∧ Y ∈ Γ₂)
  (h₇ : O ∈ Line X Y)
  (h₈ : ¬collinear O A B)
  (h₉ : M ∈ Γ₁ ∧ M ∈ Line C O)
  (h₁₀ : N ∈ Γ₂ ∧ N ∈ Line B O) :
  concurrent (Line A M) (Line D N) (Line X Y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concurrent_lines_l255_25540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tea_cost_july_approx_l255_25547

/-- Represents the cost per pound of tea/coffee in June -/
noncomputable def june_cost : ℝ := sorry

/-- Represents the cost of the mixture in July -/
def mixture_cost : ℝ := 12.50

/-- Represents the weight of the mixture in pounds -/
def mixture_weight : ℝ := 4.5

/-- Represents the parts of green tea in the mixture -/
def green_tea_parts : ℕ := 3

/-- Represents the parts of coffee in the mixture -/
def coffee_parts : ℕ := 2

/-- Represents the parts of black tea in the mixture -/
def black_tea_parts : ℕ := 4

/-- Calculates the July cost of green tea per pound -/
noncomputable def green_tea_july_cost : ℝ := 0.2 * june_cost

/-- Calculates the July cost of coffee per pound -/
noncomputable def coffee_july_cost : ℝ := 2.2 * june_cost

/-- Calculates the July cost of black tea per pound -/
noncomputable def black_tea_july_cost : ℝ := 1.6 * june_cost

/-- Theorem stating that the cost of green tea in July is approximately $0.4388 per pound -/
theorem green_tea_cost_july_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |green_tea_july_cost - 0.4388| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tea_cost_july_approx_l255_25547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_condition_l255_25517

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*m*x)

-- State the theorem
theorem f_monotone_increasing_condition (m : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, StrictMono (f m)) ↔ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_condition_l255_25517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dawn_hourly_rate_l255_25583

/-- Represents Dawn's painting job -/
structure PaintingJob where
  time_per_painting : ℚ  -- Time to complete one painting in hours
  num_paintings : ℕ      -- Number of paintings commissioned
  total_pay : ℚ          -- Total payment for all paintings in dollars

/-- Calculates the hourly rate for a painting job -/
def hourly_rate (job : PaintingJob) : ℚ :=
  job.total_pay / (job.time_per_painting * job.num_paintings)

/-- Theorem: Dawn's hourly rate is $150.00 -/
theorem dawn_hourly_rate : 
  let dawn_job : PaintingJob := {
    time_per_painting := 2,
    num_paintings := 12,
    total_pay := 3600
  }
  hourly_rate dawn_job = 150 := by
  -- Proof goes here
  sorry

#eval hourly_rate { time_per_painting := 2, num_paintings := 12, total_pay := 3600 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dawn_hourly_rate_l255_25583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_plus_one_l255_25545

open MeasureTheory

theorem integral_reciprocal_plus_one : ∫ x in (Set.Ici 0 ∩ Set.Iic 1), (1 : ℝ) / (1 + x) = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_plus_one_l255_25545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_divisible_statements_exist_l255_25557

theorem half_divisible_statements_exist :
  ∃ x : ℕ, (Finset.filter (fun k ↦ (x + k) % (2020 - k) = 0) (Finset.range 2018)).card = 1009 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_divisible_statements_exist_l255_25557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_2i_in_quadrant_II_l255_25592

-- Define Euler's formula
noncomputable def euler_formula (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the quadrants of the complex plane
def quadrant_II (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem e_2i_in_quadrant_II : quadrant_II (euler_formula 2) := by
  -- Unfold the definition of euler_formula and quadrant_II
  unfold euler_formula quadrant_II
  -- Split the goal into two parts: real part < 0 and imaginary part > 0
  apply And.intro
  -- Prove that the real part (cosine) is negative
  · sorry
  -- Prove that the imaginary part (sine) is positive
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_2i_in_quadrant_II_l255_25592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l255_25581

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x) - 2 * Real.sqrt 3 * (Real.sin (ω * x / 2))^2 + Real.sqrt 3

-- State the theorem
theorem min_value_f (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x y, x < y ∧ f ω x = 0 ∧ f ω y = 0 ∧ (∀ z, x < z → z < y → f ω z ≠ 0) → y - x = Real.pi / 2) :
  ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f ω x₀ ≤ f ω x ∧ f ω x₀ = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l255_25581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l255_25580

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x) + Real.log (x + 1)

theorem domain_of_f :
  {x : ℝ | x ∈ Set.Ioo (-1 : ℝ) 1 ∪ Set.Ioi 1} = {x : ℝ | x ≠ 1 ∧ x > -1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l255_25580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5614_to_hundredth_l255_25588

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_5614_to_hundredth :
  round_to_hundredth 5.614 = 5.61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5614_to_hundredth_l255_25588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l255_25591

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α + π/3) = -1/2) 
  (h2 : α ∈ Set.Ioo (2*π/3) π) : 
  Real.sin α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l255_25591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l255_25572

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) / b n = q

noncomputable def sum_arithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem sequence_problem (a b : ℕ → ℝ) (s : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, s n = sum_arithmetic a n) →
  geometric_sequence b →
  b 1 = 2 →
  (∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = q * b n) →
  b 2 + b 3 = 12 →
  b 3 = a 4 - 2 * a 1 →
  s 11 = 11 * b 4 →
  (∀ n : ℕ, a n = 3 * n - 2) ∧
  (∀ n : ℕ, b n = 2^n) ∧
  (∀ n : ℕ, sum_arithmetic (λ k ↦ a (2*k) * b (2*k - 1)) n = (3*n - 2)/3 * 4^(n+1) + 8/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l255_25572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_abs_x_equiv_abs_x_minus_3_l255_25596

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := if x ≥ 0 then x - 3 else -x

-- Theorem statement
theorem g_abs_x_equiv_abs_x_minus_3 : ∀ x : ℝ, g (|x|) = |x| - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_abs_x_equiv_abs_x_minus_3_l255_25596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_growth_equation_l255_25554

/-- Represents the annual growth rate of per capita disposable income -/
def x : ℝ := sorry

/-- Initial per capita disposable income in 2020 (in ten thousand yuan) -/
def initial_income : ℝ := 2.36

/-- Final per capita disposable income in 2022 (in ten thousand yuan) -/
def final_income : ℝ := 2.7

/-- Number of years between 2020 and 2022 -/
def years : ℕ := 2

/-- Theorem stating that the equation correctly represents the growth of per capita disposable income -/
theorem income_growth_equation : initial_income * (1 + x)^years = final_income := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_growth_equation_l255_25554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_proof_l255_25518

/-- Represents the age difference between a man and his son -/
def AgeDifference (man_age son_age : ℕ) : ℤ := man_age - son_age

theorem age_difference_proof (man_age son_age : ℕ) : 
  son_age = 22 →
  man_age > son_age →
  man_age + 2 = 2 * (son_age + 2) →
  AgeDifference man_age son_age = 24 := by
  intro h1 h2 h3
  unfold AgeDifference
  -- The proof steps would go here
  sorry

#check age_difference_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_proof_l255_25518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l255_25522

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^5 + Real.sin x + Real.tan x^3 - 8

-- State the theorem
theorem f_property : f (-2) = 10 → f 2 = -26 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l255_25522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l255_25544

/-- A quadratic function with vertex (1, 2) passing through (0, 3) is x^2 - 2x + 3 -/
theorem quadratic_function_unique (f : ℝ → ℝ) (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c)  -- f is a quadratic function
  (h2 : f 1 = 2 ∧ (a + b/2) = -1)        -- vertex at (1, 2)
  (h3 : f 0 = 3)                         -- passes through (0, 3)
  : f = λ x ↦ x^2 - 2*x + 3 := by
  sorry

#check quadratic_function_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l255_25544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_ratios_l255_25567

/-- A tetrahedron with edge length, surface area, and volume -/
structure Tetrahedron where
  edge : ℝ
  surface : ℝ
  volume : ℝ

/-- Create a new tetrahedron by connecting midpoints of edges -/
noncomputable def newTetrahedron (t : Tetrahedron) : Tetrahedron :=
  { edge := t.edge / 2,
    surface := t.surface / 4,
    volume := t.volume / 8 }

/-- The common ratio for edge lengths in successive tetrahedrons -/
noncomputable def edgeRatio (t : Tetrahedron) : ℝ :=
  (newTetrahedron t).edge / t.edge

/-- The common ratio for surface areas in successive tetrahedrons -/
noncomputable def surfaceRatio (t : Tetrahedron) : ℝ :=
  (newTetrahedron t).surface / t.surface

/-- The common ratio for volumes in successive tetrahedrons -/
noncomputable def volumeRatio (t : Tetrahedron) : ℝ :=
  (newTetrahedron t).volume / t.volume

theorem tetrahedron_ratios (t : Tetrahedron) :
  edgeRatio t = 1/2 ∧ surfaceRatio t = 1/4 ∧ volumeRatio t = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_ratios_l255_25567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrangea_cost_l255_25562

/-- The cost of each hydrangea plant, given Lily's purchasing history from 1989 to 2021 --/
theorem hydrangea_cost (start_year end_year : ℕ) (total_spent : ℚ) 
  (h1 : end_year ≥ start_year)
  (h2 : total_spent > 0)
  (h3 : start_year = 1989)
  (h4 : end_year = 2021)
  (h5 : total_spent = 640)
  : (total_spent / (end_year - start_year + 1 : ℚ)) = 20 := by
  sorry

#check hydrangea_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrangea_cost_l255_25562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l255_25514

/-- The sum of an arithmetic sequence from a to b -/
noncomputable def arithmetic_sequence_sum (a b : ℕ) : ℕ :=
  let n := b - a + 1
  n * (a + b) / 2

/-- The difference of three arithmetic sequence sums -/
noncomputable def arithmetic_sequence_difference (a b c : ℕ) : ℕ :=
  a - b - c

/-- The sum of the arithmetic sequence from 2501 to 2600 -/
noncomputable def sum1 : ℕ := arithmetic_sequence_sum 2501 2600

/-- The sum of the arithmetic sequence from 401 to 500 -/
noncomputable def sum2 : ℕ := arithmetic_sequence_sum 401 500

/-- The sum of the arithmetic sequence from 401 to 450 -/
noncomputable def sum3 : ℕ := arithmetic_sequence_sum 401 450

/-- The final result of the calculation -/
noncomputable def final_result : ℕ := arithmetic_sequence_difference sum1 sum2 sum3

theorem calculation_proof : final_result = 188725 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l255_25514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_addition_theorem_l255_25541

/-- Represents a number in base 7 --/
structure Base7 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 7

/-- Addition of two Base7 numbers --/
def add_base7 (a b : Base7) : Base7 :=
  sorry

/-- Conversion from a natural number to Base7 --/
def nat_to_base7 (n : Nat) : Base7 :=
  sorry

theorem base7_addition_theorem :
  add_base7 (nat_to_base7 21) (nat_to_base7 254) = nat_to_base7 505 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_addition_theorem_l255_25541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_when_a_1_b_neg_2_range_of_a_for_two_distinct_zeros_l255_25503

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

-- Theorem for part (1)
theorem zeros_when_a_1_b_neg_2 :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f 1 (-2) x₁ = 0 ∧ f 1 (-2) x₂ = 0 ∧ (x₁ = 3 ∧ x₂ = -1 ∨ x₁ = -1 ∧ x₂ = 3) :=
sorry

-- Theorem for part (2)
theorem range_of_a_for_two_distinct_zeros :
  ∀ (a : ℝ), (∀ (b : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) ↔ 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_when_a_1_b_neg_2_range_of_a_for_two_distinct_zeros_l255_25503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_riddles_l255_25526

/-- The number of riddles shared in a group of friends. -/
structure RiddleSharing where
  dan_received : ℕ
  andy_shared : ℕ
  bella_got : ℕ
  emma_received : ℕ
  fiona_received : ℕ
  dan_condition : dan_received = 21
  andy_condition : andy_shared = dan_received + 12
  bella_condition : bella_got = andy_shared - 7
  emma_condition : emma_received = bella_got / 2
  fiona_condition : fiona_received = andy_shared + bella_got

/-- Theorem stating that Fiona received 59 riddles. -/
theorem fiona_riddles (rs : RiddleSharing) : rs.fiona_received = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_riddles_l255_25526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_theorem_l255_25584

/-- Represents a standard six-sided die --/
inductive Die : Type
  | one : Die
  | two : Die
  | three : Die
  | four : Die
  | five : Die
  | six : Die

/-- Determines if a die value is even --/
def isEven (d : Die) : Bool :=
  match d with
  | Die.two | Die.four | Die.six => true
  | _ => false

/-- Represents a roll of five dice --/
def DiceRoll : Type := (Die × Die × Die × Die × Die)

/-- Checks if the product of dice values in a roll is even --/
def productIsEven (roll : DiceRoll) : Bool :=
  let (d1, d2, d3, d4, d5) := roll
  isEven d1 || isEven d2 || isEven d3 || isEven d4 || isEven d5

/-- Calculates the sum of dice values in a roll --/
def diceSum (roll : DiceRoll) : Nat :=
  sorry -- Implementation omitted

/-- Checks if the sum of dice values in a roll is even --/
def sumIsEven (roll : DiceRoll) : Bool :=
  diceSum roll % 2 == 0

/-- Counts the number of rolls satisfying a given predicate --/
def countRolls (p : DiceRoll → Prop) : Nat :=
  sorry -- Implementation omitted

/-- The main theorem to prove --/
theorem dice_probability_theorem :
  let allRolls := {roll : DiceRoll | productIsEven roll}
  (countRolls (fun roll => roll ∈ allRolls ∧ sumIsEven roll)) / (countRolls (fun roll => roll ∈ allRolls)) = 1296 / 2511 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_theorem_l255_25584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l255_25528

theorem solve_exponential_equation (x : ℝ) : (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x = (64 : ℝ)^3 → x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l255_25528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_proof_l255_25553

-- Define the radius and area of the sector
noncomputable def radius : ℝ := 12
noncomputable def sector_area : ℝ := 51.54285714285714

-- Define the formula for the area of a sector
noncomputable def sector_area_formula (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * Real.pi * r^2

-- Define the central angle we want to prove
noncomputable def central_angle : ℝ := 41.01

-- Theorem statement
theorem sector_angle_proof :
  ∃ ε > 0, |sector_area_formula radius central_angle - sector_area| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_proof_l255_25553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l255_25532

-- Define the parametric equation of curve C1
noncomputable def C1 (t : ℝ) : ℝ × ℝ := (2 - t, Real.sqrt 3 * t)

-- Define point A
noncomputable def A : ℝ × ℝ := C1 (-1)

-- Define point B as symmetric to A with respect to the origin
noncomputable def B : ℝ × ℝ := (-A.1, -A.2)

-- Define the polar equation of curve C2
noncomputable def C2 (θ : ℝ) : ℝ := 6 / Real.sqrt (9 + 3 * Real.sin θ ^ 2)

-- Statement to prove
theorem max_sum_distances (P : ℝ × ℝ) (h : ∃ θ, P = (C2 θ * Real.cos θ, C2 θ * Real.sin θ)) :
  (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2 ≤ 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l255_25532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_approx_3000_l255_25535

/-- Calculates the length of a bridge given a walking speed and crossing time -/
noncomputable def bridge_length (speed_km_hr : ℝ) (time_minutes : ℝ) : ℝ :=
  speed_km_hr * (1000 / 60) * time_minutes

/-- Theorem: The length of a bridge is approximately 3000 meters -/
theorem bridge_length_approx_3000 (speed_km_hr : ℝ) (time_minutes : ℝ) 
  (h1 : speed_km_hr = 10) 
  (h2 : time_minutes = 18) : 
  ∃ (ε : ℝ), ε > 0 ∧ |bridge_length speed_km_hr time_minutes - 3000| < ε := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_approx_3000_l255_25535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l255_25585

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Ensure positive side lengths
  C = π / 3 →  -- Angle C is π/3
  a = 1 →  -- Side a is 1
  b = 2 →  -- Side b is 2
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →  -- Cosine rule
  c = Real.sqrt 3 := by  -- Conclusion: side c is √3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l255_25585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_pairs_count_l255_25577

def number_of_sock_pairs (n : ℕ) : Prop := True

def number_of_selections (n : ℕ) : ℕ := n * (n - 1) / 2

theorem sock_pairs_count : ∃ n : ℕ, number_of_sock_pairs n ∧ number_of_selections (2 * n) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_pairs_count_l255_25577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_on_circle_l255_25575

/-- The circle passing through point (1, 1) -/
def circle_m (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 + m*p.2 = 0}

/-- The tangent line to the circle at point (1, 1) -/
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2*p.2 + 1 = 0}

theorem tangent_line_at_point_on_circle :
  ∃ m : ℝ, (1, 1) ∈ circle_m m ∧ 
  (∀ p : ℝ × ℝ, p ∈ circle_m m → p ∈ tangent_line ∨ (∃ q : ℝ × ℝ, q ≠ p ∧ q ∈ circle_m m ∧ q ∈ tangent_line)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_on_circle_l255_25575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l255_25542

/-- The volume of a pyramid formed from a square cardboard --/
theorem pyramid_volume (side_length corner_distance cut_angle : ℝ) : 
  side_length = 120 →
  corner_distance = 10 →
  cut_angle = 45 →
  (1/3) * side_length^2 * (side_length * Real.sqrt 2 / 2 / Real.sqrt 2) = 288000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l255_25542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_atom_count_is_one_l255_25568

/-- Represents the number of atoms of an element in a compound -/
def AtomCount : Type := ℕ

/-- Represents the atomic weight of an element in atomic mass units (amu) -/
def AtomicWeight : Type := ℚ

/-- Represents the molecular weight of a compound in atomic mass units (amu) -/
def MolecularWeight : Type := ℚ

/-- Calculate the number of Carbon atoms in a compound -/
def carbonAtomCount (
  hydrogen_count : AtomCount) 
  (oxygen_count : AtomCount) 
  (molecular_weight : MolecularWeight) 
  (hydrogen_weight : AtomicWeight) 
  (carbon_weight : AtomicWeight) 
  (oxygen_weight : AtomicWeight) : AtomCount :=
  sorry

/-- Theorem: The number of Carbon atoms in the given compound is 1 -/
theorem carbon_atom_count_is_one :
  carbonAtomCount (2 : ℕ) (3 : ℕ) (62 : ℚ) (1 : ℚ) (12 : ℚ) (16 : ℚ) = (1 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_atom_count_is_one_l255_25568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_abc_l255_25505

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := (1/2)^(-(1/2 : ℝ))

theorem order_abc : a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_abc_l255_25505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l255_25500

/-- An ellipse with center at the origin, one focus at (-2,0), and
    major to minor axis ratio of 2:√3 has the equation x²/16 + y²/12 = 1 -/
theorem ellipse_equation (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / 16 + y^2 / 12 = 1) ↔
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    F = (-2, 0) ∧
    a^2 = b^2 + 4 ∧
    a / b = 2 / Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l255_25500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_sum_theorem_l255_25507

theorem three_sum_theorem (n : ℕ) (hn : n ≥ 1) (X : Finset ℤ) 
  (hX_card : X.card = n + 2) 
  (hX_bound : ∀ x ∈ X, |x| ≤ n) :
  ∃ a b c, a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_sum_theorem_l255_25507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_four_consecutive_composites_l255_25570

/-- Four consecutive natural numbers -/
def consecutive_numbers (n : ℕ) : Fin 4 → ℕ
| 0 => n
| 1 => n + 1
| 2 => n + 2
| 3 => n + 3

/-- Check if a number is composite -/
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬(Nat.Prime n)

/-- All four numbers are composite -/
def all_composite (n : ℕ) : Prop :=
  ∀ i : Fin 4, is_composite (consecutive_numbers n i)

/-- Sum of four consecutive natural numbers -/
def sum_of_four (n : ℕ) : ℕ :=
  (consecutive_numbers n 0) + (consecutive_numbers n 1) + (consecutive_numbers n 2) + (consecutive_numbers n 3)

/-- The smallest possible sum of four consecutive composite natural numbers is 102 -/
theorem smallest_sum_of_four_consecutive_composites :
  (∃ n : ℕ, all_composite n ∧ sum_of_four n = 102) ∧
  (∀ m : ℕ, all_composite m → sum_of_four m ≥ 102) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_four_consecutive_composites_l255_25570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fold_count_l255_25501

/-- Represents the three positions of the folded triangle card -/
inductive Position
| One
| Two
| Three

/-- Represents the three colors of the edges -/
inductive EdgeColor
| Red
| Yellow
| Blue

/-- The function that determines the next position after a fold -/
def nextPosition (p : Position) : Position :=
  match p with
  | Position.One => Position.Two
  | Position.Two => Position.Three
  | Position.Three => Position.One

/-- The function that determines the next edge color in the sequence -/
def nextColor (c : EdgeColor) : EdgeColor :=
  match c with
  | EdgeColor.Red => EdgeColor.Yellow
  | EdgeColor.Yellow => EdgeColor.Blue
  | EdgeColor.Blue => EdgeColor.Red

/-- The function that simulates the folding process -/
def folding : Position → ℕ → Position × EdgeColor
  | p, 0 => (p, EdgeColor.Red)
  | p, n+1 => let (newP, c) := folding p n
              (nextPosition newP, nextColor c)

/-- The main theorem stating that 6 is the smallest number of folds to return to the original position -/
theorem smallest_fold_count : 
  ∀ (start : Position),
  ∃ (n : ℕ), n = 6 ∧ 
  (∀ (m : ℕ), m < n → 
    (folding start m).1 ≠ start) ∧
  (folding start n).1 = start :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fold_count_l255_25501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_at_135_deg_equation_AB_when_bisected_l255_25578

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define point P
def P : ℝ × ℝ := (-1, 2)

-- Define chord AB passing through P
def chord_AB (x y : ℝ) (α : ℝ) : Prop :=
  ∃ (t : ℝ), x = -1 + t * Real.cos α ∧ y = 2 + t * Real.sin α

-- Theorem 1
theorem length_AB_at_135_deg :
  ∀ (x y : ℝ),
  circle_eq x y →
  chord_AB x y (135 * π / 180) →
  (x - (-1))^2 + (y - 2)^2 = 30 :=
sorry

-- Theorem 2
theorem equation_AB_when_bisected :
  ∀ (x y : ℝ),
  circle_eq x y →
  chord_AB x y (Real.arctan (1/2)) →
  x - 2*y + 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_at_135_deg_equation_AB_when_bisected_l255_25578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l255_25531

-- Define the repeating decimal 0.568̄ as a rational number
def repeating_decimal : ℚ := 568 / 999

-- Theorem statement
theorem repeating_decimal_equals_fraction : repeating_decimal = 568 / 999 := by
  -- The proof is trivial since we defined repeating_decimal as 568 / 999
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l255_25531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l255_25520

/-- Profit function given promotional expenses -/
noncomputable def profit (m : ℝ) : ℝ := 29 - (16 / (m + 1) + (m + 1))

/-- The promotional expense that maximizes profit -/
def optimal_expense : ℝ := 3

/-- The maximum profit -/
def max_profit : ℝ := 21

theorem profit_maximization :
  (∀ m : ℝ, m ≥ 0 → profit m ≤ max_profit) ∧
  profit optimal_expense = max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l255_25520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l255_25579

theorem trig_inequality (α β : ℝ) : 
  1 / (Real.cos α) ^ 2 + 1 / ((Real.sin α) ^ 2 * (Real.sin β) ^ 2 * (Real.cos β) ^ 2) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l255_25579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_score_is_even_l255_25586

/-- Represents a student's answers to the math competition questions -/
structure StudentAnswers where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  sum_constraint : correct + unanswered + incorrect = 6

/-- Calculates the score for a single student based on their answers -/
def calculateScore (answers : StudentAnswers) : Int :=
  5 * answers.correct + answers.unanswered - answers.incorrect

/-- Represents a group of students participating in the math competition -/
structure StudentGroup where
  students : List StudentAnswers

/-- Calculates the total score for a group of students -/
def calculateGroupScore (group : StudentGroup) : Int :=
  group.students.foldl (fun acc student => acc + calculateScore student) 0

/-- Theorem: The total score of any group is always even -/
theorem group_score_is_even (group : StudentGroup) : Even (calculateGroupScore group) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_score_is_even_l255_25586
