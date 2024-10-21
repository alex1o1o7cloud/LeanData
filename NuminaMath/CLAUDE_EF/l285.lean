import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_vote_percentage_l285_28573

theorem candidate_vote_percentage :
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_valid_votes : ℕ := 333200
  let total_valid_votes : ℚ := (total_votes : ℚ) * (1 - invalid_percentage)
  let candidate_percentage : ℚ := (candidate_valid_votes : ℚ) / total_valid_votes * 100
  candidate_percentage = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_vote_percentage_l285_28573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l285_28570

noncomputable section

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the circle E
def circle_E (x y : ℝ) : Prop := x^2 + y^2 = 3/4

-- Define the eccentricity of C
def eccentricity : ℝ := Real.sqrt 3 / 2

-- Define the slope of line l
noncomputable def slope_l : ℝ := Real.tan (30 * Real.pi / 180)

-- Define the general line that is tangent to E
def tangent_line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the area of triangle OAB
noncomputable def area_OAB (k m : ℝ) : ℝ := sorry

-- Define the area of triangle QAB
noncomputable def area_QAB (k m : ℝ) : ℝ := sorry

-- Main theorem
theorem ellipse_theorem :
  (∀ x y, ellipse_C x y → circle_E x y → ∃ c, y = slope_l * (x - c)) →
  (∃ k m, k ≠ 0 ∧ ∀ x y, tangent_line k m x y → circle_E x y) →
  (∀ k m, k ≠ 0 → (∃ x₁ y₁ x₂ y₂, ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧ 
                   tangent_line k m x₁ y₁ ∧ tangent_line k m x₂ y₂)) →
  (∃ x₃ y₃, ellipse_C x₃ y₃ ∧ y₃ = -1/k * x₃) →
  (∀ k m, area_OAB k m ≤ 1) ∧
  (∃ k m, area_OAB k m = 1 ∧ area_OAB k m / area_QAB k m = (4 * Real.sqrt 42 + 21) / 11) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l285_28570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_distance_l285_28595

/-- Represents the speed of a car in km/h -/
def speed : ℚ := 180

/-- Represents the time gap between two cars in seconds -/
def time_gap : ℚ := 1

/-- Converts km/h to m/s -/
def kmh_to_ms (v : ℚ) : ℚ := v * 1000 / 3600

/-- Calculates the distance between two cars given their speed and time gap -/
def distance_between_cars (v : ℚ) (t : ℚ) : ℚ := kmh_to_ms v * t

theorem cars_distance :
  distance_between_cars speed time_gap = 50 := by
  unfold distance_between_cars kmh_to_ms speed time_gap
  -- Simplify the expression
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- Evaluate the arithmetic
  norm_num

#eval distance_between_cars speed time_gap

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_distance_l285_28595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_fifteen_percent_l285_28582

/-- The interest rate that yields 720 more interest than 12% over 2 years for a principal of 12000 -/
noncomputable def higher_interest_rate (principal : ℝ) (time : ℝ) (difference : ℝ) : ℝ :=
  (principal * 0.12 * time + difference) / (principal * time)

theorem interest_rate_is_fifteen_percent :
  higher_interest_rate 12000 2 720 = 0.15 := by
  -- Unfold the definition of higher_interest_rate
  unfold higher_interest_rate
  -- Simplify the arithmetic
  simp [mul_assoc, add_mul, mul_comm, mul_add]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_fifteen_percent_l285_28582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_center_distance_center_to_line_is_zero_line_intersects_and_passes_through_center_l285_28518

/-- The line equation -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := 2 * x^2 + 2 * y^2 - 4 * x - 2 * y + 1 = 0

/-- The center of the circle -/
noncomputable def center : ℝ × ℝ := (1, 1/2)

/-- Theorem stating that the line passes through the center of the circle -/
theorem line_passes_through_center : line center.1 center.2 := by sorry

/-- Theorem stating that the distance from the center to the line is zero -/
theorem distance_center_to_line_is_zero : 
  let (x₀, y₀) := center
  (3 * x₀ + 4 * y₀ - 5)^2 / (3^2 + 4^2) = 0 := by sorry

/-- Main theorem: The line intersects the circle and passes through its center -/
theorem line_intersects_and_passes_through_center : 
  ∃ (x y : ℝ), line x y ∧ circle_eq x y ∧ (x, y) = center := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_center_distance_center_to_line_is_zero_line_intersects_and_passes_through_center_l285_28518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_in_special_triangle_l285_28586

theorem max_angle_in_special_triangle (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  ∃ (k : ℝ), k > 0 ∧ Real.sin A = 3 * k ∧ Real.sin B = 5 * k ∧ Real.sin C = 7 * k →
  max A (max B C) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_in_special_triangle_l285_28586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_ratio_l285_28579

-- Define the square ABCD
structure Square where
  side : ℝ

-- Define the triangle XYZ
structure Triangle where
  base : ℝ
  height : ℝ

-- Define the relationship between square and triangle
def square_triangle_relation (s : Square) (t : Triangle) : Prop :=
  s.side^2 = (7/32) * (t.base * t.height / 2)

-- Define the ratio XA:XY
noncomputable def ratio (xa : ℝ) (xy : ℝ) : ℝ := xa / xy

-- Theorem statement
theorem square_triangle_ratio 
  (s : Square) (t : Triangle) (xa xy : ℝ) 
  (h : square_triangle_relation s t) :
  ratio xa xy = 7/8 ∨ ratio xa xy = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_ratio_l285_28579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l285_28532

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- Theorem stating that a 135 m long train moving at 140 km/h takes approximately 3.47 seconds to cross an electric pole -/
theorem train_crossing_pole_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_crossing_time 135 140 - 3.47| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l285_28532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_of_p_l285_28502

/-- The rational function f(x) -/
noncomputable def f (p : ℝ → ℝ) : ℝ → ℝ := λ x ↦ (3*x^7 + 4*x^6 - 2*x^3 - 2) / (p x)

/-- A function has a horizontal asymptote if it converges to a finite value as x approaches infinity -/
def has_horizontal_asymptote (g : ℝ → ℝ) : Prop :=
  ∃ L : ℝ, ∀ ε > 0, ∃ N : ℝ, ∀ x > N, |g x - L| < ε

/-- The degree of a polynomial -/
def polynomial_degree (p : ℝ → ℝ) : ℕ := sorry

/-- The theorem stating the smallest possible degree of p(x) -/
theorem smallest_degree_of_p (p : ℝ → ℝ) :
  has_horizontal_asymptote (f p) → polynomial_degree p ≥ 7 := by
  sorry

#check smallest_degree_of_p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_of_p_l285_28502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_range_of_a_l285_28554

noncomputable section

open Real

-- Define the interval [1/e, e]
def I : Set ℝ := {x | 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1}

-- Define the functions g and h
def g (a : ℝ) (x : ℝ) : ℝ := a - x^3
def h (x : ℝ) : ℝ := 3 * log x

-- Define the symmetry condition
def symmetric_points (a : ℝ) : Prop :=
  ∃ x ∈ I, g a x = -h x

-- Theorem statement
theorem symmetry_range_of_a :
  {a : ℝ | symmetric_points a} = {a : ℝ | 1 ≤ a ∧ a ≤ (Real.exp 1)^3 - 3} :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_range_of_a_l285_28554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l285_28508

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - x ≥ 0}

-- State the theorem
theorem complement_of_M_in_U : 
  (U \ M) = {x : ℝ | 0 < x ∧ x < 1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l285_28508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l285_28506

noncomputable def simple_interest (principal rate time : ℝ) : ℝ := (principal * rate * time) / 100

theorem interest_rate_calculation (principal time interest : ℝ) 
  (h_principal : principal = 6178.846153846154)
  (h_time : time = 5)
  (h_interest : interest = 4016.25) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l285_28506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_4_range_of_expression_l285_28521

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < 3 * Real.pi / 4 ∧
  c * Real.sin A = a * Real.cos C

-- Theorem for part 1
theorem angle_C_is_pi_over_4 {A B C a b c : ℝ} (h : Triangle A B C a b c) : 
  C = Real.pi / 4 := by sorry

-- Theorem for part 2
theorem range_of_expression {A B C a b c : ℝ} (h : Triangle A B C a b c) :
  Real.sqrt 6 / 2 - Real.sqrt 2 / 2 < Real.sqrt 3 * Real.sin A - Real.cos (B + C) ∧
  Real.sqrt 3 * Real.sin A - Real.cos (B + C) ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_4_range_of_expression_l285_28521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l285_28514

/-- Represents the total number of votes in an election --/
def total_votes : ℕ := sorry

/-- Represents the number of votes for the winner --/
def winner_votes : ℕ := sorry

/-- Represents the number of votes for the loser --/
def loser_votes : ℕ := sorry

/-- The margin of victory in the original election --/
def margin : ℚ := 1/5

/-- The number of people who change their vote --/
def vote_change : ℕ := 3000

theorem election_votes :
  -- The total votes is the sum of winner and loser votes
  total_votes = winner_votes + loser_votes ∧
  -- The winner's margin is 20% of total votes
  (winner_votes : ℚ) - (loser_votes : ℚ) = margin * total_votes ∧
  -- If 3000 people change their vote, the loser wins by 20%
  ((loser_votes + vote_change : ℚ) - (winner_votes - vote_change : ℚ)) = margin * total_votes →
  -- Then the total number of votes is 15000
  total_votes = 15000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l285_28514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l285_28596

-- Define the function f on [-3, 3]
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / 9^x + (-1) / 4^x else 4^x - 9^x

-- State the theorem
theorem odd_function_properties :
  -- f is odd
  (∀ x ∈ Set.Icc (-3) 3, f (-x) = -f x) ∧
  -- f(x) = 4^x - 9^x for x ∈ (0, 3]
  (∀ x ∈ Set.Ioo 0 3, f x = 4^x - 9^x) ∧
  -- The minimum value of m such that f(x) ≤ m/3^x - 1/4^(x-1) for all x ∈ [-1, -1/2] is 7
  (∀ m : ℝ, (∀ x ∈ Set.Icc (-1) (-1/2), f x ≤ m / 3^x - 1 / 4^(x-1)) ↔ m ≥ 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l285_28596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l285_28531

/-- Given a point C with coordinates (-3, 2), prove that the distance between C
    and its reflection C' over the y-axis is 6. -/
theorem reflection_distance : 
  let C : ℝ × ℝ := (-3, 2)
  let C' : ℝ × ℝ := (3, 2)
  ‖C - C'‖ = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l285_28531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l285_28557

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- State the theorem
theorem f_properties :
  -- Part 1: f is odd
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f (-x) = -f x) ∧
  -- Part 2: if f(m) - f(-m) = 2, then m = (1-e)/(1+e)
  (∀ m : ℝ, f m - f (-m) = 2 → m = (1 - Real.exp 1) / (1 + Real.exp 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l285_28557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABC_l285_28535

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin ((x - Real.pi / 6) / 2)

theorem max_area_triangle_ABC (ω φ a b c : ℝ) :
  (0 < ω) → (ω < 1) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f ω x ≤ Real.sqrt 2) →
  (0 < φ) → (φ < Real.pi / 2) →
  (∀ x, g (x + 7 * Real.pi / 6) = g (7 * Real.pi / 6 - x)) →
  (g (Real.pi / 6) = 0) →
  (c = 4) →
  (∃ A B C : ℝ, a * b * Real.sin C / 2 ≤ 8 + 4 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABC_l285_28535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l285_28588

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + 1

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m) ∧
  (let p := Real.pi; let m := -1;
    (p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
      ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
    (∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l285_28588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_is_six_l285_28567

/-- A cube with side length 2 -/
structure Cube where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- A line segment with length 3 -/
structure Segment where
  length : ℝ
  length_eq : length = 3

/-- A path connecting two opposite vertices of a cube -/
structure CubePath (cube : Cube) where
  segments : List Segment
  connects_opposite_vertices : Bool
  vertices_on_cube : Bool

/-- The minimum number of segments in a valid path -/
def min_segments (cube : Cube) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of segments is 6 -/
theorem min_segments_is_six (cube : Cube) : min_segments cube = 6 := by
  sorry

#check min_segments_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_is_six_l285_28567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_theorem_l285_28548

/-- The ratio of speeds of two objects A and B -/
noncomputable def speed_ratio (a b : ℝ) : ℝ := (a + b) / (b - a)

/-- The time taken for A and B to meet when moving towards each other -/
def meet_time (a : ℝ) : ℝ := a

/-- The time taken for A to overtake B when moving in the same direction -/
def overtake_time (b : ℝ) : ℝ := b

/-- The theorem stating the ratio of speeds of A and B -/
theorem speed_ratio_theorem (a b : ℝ) (ha : a > 0) (hb : b > a) :
  let v_ratio := speed_ratio a b
  let t_meet := meet_time a
  let t_overtake := overtake_time b
  v_ratio = (a + b) / (b - a) :=
by
  -- Unfold the definitions
  unfold speed_ratio meet_time overtake_time
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_theorem_l285_28548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l285_28572

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Check if a point lies on a parabola -/
def pointOnParabola (p : Point) (para : Parabola) : Prop :=
  (p.y - para.k)^2 = 4 * para.a * (p.x - para.h)

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem -/
theorem parabola_line_intersection
  (C : Parabola)
  (F : Point)
  (L : Line)
  (A B : Point)
  (h1 : C.a = 1 ∧ C.h = 0 ∧ C.k = 0)
  (h2 : F.x = 1 ∧ F.y = 0)
  (h3 : pointOnLine F L)
  (h4 : pointOnParabola A C ∧ pointOnLine A L)
  (h5 : pointOnParabola B C ∧ pointOnLine B L)
  (h6 : distance A F = 3 * distance B F)
  : L.slope = Real.sqrt 3 ∨ L.slope = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l285_28572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_Q_l285_28556

def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4)

def is_valid_Q (Q : ℝ → ℝ) : Prop :=
  ∃ (R : ℝ → ℝ), (∀ x, P (Q x) = P x * R x) ∧ 
  (∃ a b c d e, ∀ x, R x = a*x^4 + b*x^3 + c*x^2 + d*x + e)

-- We need to specify the type of Q more precisely
def valid_Q_set : Set (ℝ → ℝ) := {Q | is_valid_Q Q}

-- We can't directly use Fintype on functions ℝ → ℝ, so we'll state the theorem differently
theorem count_valid_Q : ∃ n : ℕ, n = 254 ∧ ∃ f : Fin n → valid_Q_set, Function.Injective f := by
  sorry

#check count_valid_Q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_Q_l285_28556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l285_28590

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

theorem intersection_A_B : A ∩ B = Set.Ioc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l285_28590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_between_roots_l285_28599

theorem count_integers_between_roots : 
  (Finset.filter (fun n : ℕ => (↑n : ℝ) > Real.sqrt 50 ∧ (↑n : ℝ) < Real.sqrt 200) (Finset.range 15)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_between_roots_l285_28599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_ride_time_l285_28569

/-- The time James rode in hours -/
noncomputable def time : ℚ := 80 / 16

/-- The distance James rode in miles -/
def distance : ℚ := 80

/-- James' speed in miles per hour -/
def speed : ℚ := 16

theorem james_ride_time : time = 5 := by
  -- Unfold the definition of time
  unfold time
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_ride_time_l285_28569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_determinable_l285_28566

def left_multipliers : Set ℕ := {4, 10, 12, 26}
def right_multipliers : Set ℕ := {7, 13, 21, 35}

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem coin_game_determinable :
  ∀ (left_coin right_coin : ℕ) (left_mult right_mult : ℕ),
    left_coin ∈ ({10, 15} : Set ℕ) →
    right_coin ∈ ({10, 15} : Set ℕ) →
    left_coin ≠ right_coin →
    left_mult ∈ left_multipliers →
    right_mult ∈ right_multipliers →
    is_even left_mult →
    ¬(is_even right_mult) →
    ∃ (determine_function : ℕ → Bool),
      determine_function (left_coin * left_mult + right_coin * right_mult) = 
        (left_coin = 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_determinable_l285_28566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l285_28503

/-- Represents a sinusoidal function with frequency and phase shift -/
noncomputable def SinFunction (ω : ℝ) (φ : ℝ) : ℝ → ℝ := λ x ↦ Real.sin (ω * x + φ)

/-- Represents a horizontal shift of a function -/
def HorizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x ↦ f (x + shift)

theorem sin_shift_equivalence :
  let f := SinFunction 3 (π/4)
  let g := SinFunction 3 0
  let h := HorizontalShift g (π/12)
  ∀ x, f x = h x := by
  intro x
  simp [SinFunction, HorizontalShift]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l285_28503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_values_l285_28584

noncomputable def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

noncomputable def slope_l1 (m : ℝ) : ℝ := -m / (1 - m)

noncomputable def slope_l2 (m : ℝ) : ℝ := -(m - 1) / (2 * m + 3)

theorem perpendicular_lines_m_values :
  ∀ m : ℝ, perpendicular (slope_l1 m) (slope_l2 m) → m = 1 ∨ m = -3 := by
  sorry

#check perpendicular_lines_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_values_l285_28584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_given_phi_inequality_l285_28522

open Real Set

/-- The function f(x) = (x+1)e^(-x) -/
noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp (-x)

/-- The function φ(x) = xf(x) + tf'(x) + e^(-x) -/
noncomputable def φ (t : ℝ) (x : ℝ) : ℝ := x * f x + t * (deriv f x) + Real.exp (-x)

/-- Theorem stating the range of t given the conditions -/
theorem t_range_given_phi_inequality :
  ∃ t, ∃ (x₁ x₂ : ℝ), x₁ ∈ Icc 0 1 ∧ x₂ ∈ Icc 0 1 ∧ 2 * φ t x₁ < φ t x₂ ∧
  t ∈ Iic (3 - 2 * Real.exp 1) ∪ Ioi (3 - Real.exp 1 / 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_given_phi_inequality_l285_28522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l285_28513

def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

theorem function_properties :
  (∀ x b, f x b 0 = -f (-x) b 0) ∧
  (∀ c, c > 0 → (∃! r, f r 0 c = 0)) ∧
  (∀ x b c, f x b c - c = -(f (-x) b c - c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l285_28513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_identity_l285_28583

/-- The function g(x) defined in terms of a constant b -/
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x / (b * x + 1)

/-- Theorem stating that g(g(x)) = x for all x ≠ -1/b if and only if b = -1 -/
theorem g_composition_identity (b : ℝ) : 
  (∀ x : ℝ, x ≠ -1/b → g b (g b x) = x) ↔ b = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_identity_l285_28583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ketchup_purchase_l285_28515

noncomputable section

structure KetchupOption where
  volume : ℝ
  price : ℝ

noncomputable def cost_per_ounce (option : KetchupOption) : ℝ :=
  option.price / option.volume

def available_options : List KetchupOption := [
  ⟨10, 1⟩,
  ⟨16, 2⟩,
  ⟨25, 2.5⟩,
  ⟨50, 5⟩,
  ⟨200, 10⟩
]

def budget : ℝ := 10

theorem optimal_ketchup_purchase :
  ∃ (best_option : KetchupOption),
    best_option ∈ available_options ∧
    best_option.price = budget ∧
    ∀ (option : KetchupOption),
      option ∈ available_options →
      cost_per_ounce best_option ≤ cost_per_ounce option :=
by
  -- The proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ketchup_purchase_l285_28515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_quadrilateral_l285_28509

/-- A point in the projective plane -/
structure ProjectivePoint where
  x : ℝ
  y : ℝ
  z : ℝ
  nontrivial : (x, y, z) ≠ (0, 0, 0)

/-- A line in the projective plane -/
structure ProjectiveLine where
  a : ℝ
  b : ℝ
  c : ℝ
  nontrivial : (a, b, c) ≠ (0, 0, 0)

/-- The intersection of two projective lines -/
noncomputable def intersection (l1 l2 : ProjectiveLine) : ProjectivePoint :=
  sorry

/-- The line passing through two projective points -/
noncomputable def line_through (p1 p2 : ProjectivePoint) : ProjectiveLine :=
  sorry

/-- The cross-ratio of four collinear points -/
noncomputable def cross_ratio (p1 p2 p3 p4 : ProjectivePoint) : ℝ :=
  sorry

/-- Theorem of the complete quadrilateral -/
theorem complete_quadrilateral 
  (A B C D : ProjectivePoint) : 
  let AB := line_through A B
  let CD := line_through C D
  let AC := line_through A C
  let BD := line_through B D
  let AD := line_through A D
  let BC := line_through B C
  let P := intersection AB CD
  let Q := intersection AD BC
  let R := intersection AC BD
  let QR := line_through Q R
  let K := intersection QR AB
  let L := intersection QR CD
  cross_ratio Q R K L = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_quadrilateral_l285_28509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_dioxide_moles_problem_solution_l285_28505

-- Define the moles of each substance
variable (moles_MgO : ℝ)
variable (moles_CO2 : ℝ)
variable (moles_MgCO3 : ℝ)

-- Define the reaction ratio
def reaction_ratio : ℝ := 1

-- State the theorem
theorem carbon_dioxide_moles (h1 : moles_MgO = moles_MgCO3) (h2 : reaction_ratio = 1) :
  moles_CO2 = moles_MgCO3 :=
by
  sorry

-- Specific instance for the problem
theorem problem_solution :
  let moles_MgO := 3
  let moles_MgCO3 := 3
  moles_CO2 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_dioxide_moles_problem_solution_l285_28505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_poly_iff_comp_square_l285_28562

-- Define what it means for a polynomial to be even
def is_even_poly (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, P.eval (-x) = P.eval x

-- State the theorem
theorem even_poly_iff_comp_square :
  ∀ P : Polynomial ℝ, is_even_poly P ↔ ∃ Q : Polynomial ℝ, ∀ x : ℝ, P.eval x = Q.eval (x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_poly_iff_comp_square_l285_28562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_minimum_l285_28541

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptotes of the hyperbola
def asymptote (a b x y : ℝ) : Prop := y = b / a * x ∨ y = -b / a * x

-- Define the line x = a
def vertical_line (a x : ℝ) : Prop := x = a

-- Define the area of triangle ODE
def triangle_area (a b : ℝ) : ℝ := a * b

-- Define the focal length of the hyperbola
noncomputable def focal_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

theorem hyperbola_focal_length_minimum 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_area : triangle_area a b = 8) :
  8 ≤ focal_length a b ∧ 
  ∃ (a' b' : ℝ), hyperbola a' b' a' b' ∧ 
                 asymptote a' b' a' b' ∧ 
                 vertical_line a' a' ∧ 
                 triangle_area a' b' = 8 ∧ 
                 focal_length a' b' = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_minimum_l285_28541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_age_l285_28563

theorem rachel_age : ℕ := by
  let emily_current_age : ℕ := 20
  let rachel_emily_age_diff : ℕ := 8 - (8 / 2)
  let rachel_current_age : ℕ := emily_current_age + rachel_emily_age_diff
  
  have h1 : rachel_emily_age_diff = 4 := by
    rfl
  
  have h2 : rachel_current_age = 24 := by
    rfl
  
  exact rachel_current_age

#check rachel_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_age_l285_28563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_at_pos_infinity_limit_at_neg_infinity_l285_28511

/-- The function g(x) = -x^4 + 5x^3 + 7 -/
def g (x : ℝ) : ℝ := -x^4 + 5*x^3 + 7

/-- The limit of g(x) as x approaches positive infinity is negative infinity -/
theorem limit_at_pos_infinity :
  Filter.Tendsto g Filter.atTop (Filter.atBot) :=
sorry

/-- The limit of g(x) as x approaches negative infinity is negative infinity -/
theorem limit_at_neg_infinity :
  Filter.Tendsto g Filter.atBot (Filter.atBot) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_at_pos_infinity_limit_at_neg_infinity_l285_28511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_races_for_top_three_l285_28529

/-- Represents a horse -/
structure Horse where
  id : Nat

/-- Represents a race between horses -/
structure Race where
  participants : Finset Horse

/-- Represents the result of a race -/
inductive RaceResult
  | Winner (h : Horse) : RaceResult
  | RunnerUp (h : Horse) : RaceResult
  | ThirdPlace (h : Horse) : RaceResult
  | Other (h : Horse) : RaceResult

/-- A function that determines the result of a race -/
noncomputable def raceOutcome (r : Race) : List RaceResult := sorry

/-- The set of all horses -/
def allHorses : Finset Horse := sorry

/-- The maximum number of horses that can race together -/
def maxHorsesPerRace : Nat := 4

/-- The total number of horses -/
def totalHorses : Nat := 35

/-- A function that determines if a horse is in the top 3 fastest -/
def isTopThree (h : Horse) : Prop := sorry

/-- The main theorem: proving the minimum number of races required -/
theorem min_races_for_top_three :
  ∃ (races : List Race),
    (∀ r ∈ races, r.participants.card ≤ maxHorsesPerRace) ∧
    (∀ h ∈ allHorses, ∃ r ∈ races, h ∈ r.participants) ∧
    (∀ h, isTopThree h ↔ ∃ r ∈ races, (RaceResult.Winner h ∈ raceOutcome r ∨ RaceResult.RunnerUp h ∈ raceOutcome r ∨ RaceResult.ThirdPlace h ∈ raceOutcome r)) ∧
    races.length = 10 ∧
    ∀ races' : List Race,
      (∀ r ∈ races', r.participants.card ≤ maxHorsesPerRace) →
      (∀ h ∈ allHorses, ∃ r ∈ races', h ∈ r.participants) →
      (∀ h, isTopThree h ↔ ∃ r ∈ races', (RaceResult.Winner h ∈ raceOutcome r ∨ RaceResult.RunnerUp h ∈ raceOutcome r ∨ RaceResult.ThirdPlace h ∈ raceOutcome r)) →
      races'.length ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_races_for_top_three_l285_28529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_five_eq_neg_twenty_two_l285_28519

/-- Given a function g(x) = 4x - 2, prove that g(-5) = -22 -/
theorem g_neg_five_eq_neg_twenty_two :
  (fun x : ℝ => 4 * x - 2) (-5) = -22 := by
  -- Define g
  let g : ℝ → ℝ := fun x => 4 * x - 2
  -- Evaluate g(-5)
  calc g (-5)
    = 4 * (-5) - 2 := rfl
    _ = -20 - 2 := by ring
    _ = -22 := by ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_five_eq_neg_twenty_two_l285_28519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l285_28593

-- Define the points
def A : ℝ × ℝ := (5, 0)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (6, 0)
def E : ℝ × ℝ := (3, 6)
def O : ℝ × ℝ := (0, 0)

-- Define the slopes
def slope_line1 : ℝ := -3

-- Define the area function for a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem area_of_quadrilateral_OBEC : 
  triangle_area O C E - triangle_area O B E = 19.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l285_28593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_l285_28568

/-- Represents the total number of valid votes in an election -/
def total_valid_votes : ℕ := sorry

/-- Represents the percentage of votes received by candidate A -/
def candidate_A_percentage : ℚ := sorry

/-- Represents the vote difference between candidates A and B -/
def vote_difference_A_B : ℕ := sorry

/-- Represents the percentage difference between candidates B and C -/
def percentage_difference_B_C : ℚ := sorry

/-- Represents the ratio of votes between candidates D and C -/
def vote_ratio_D_C : ℚ := sorry

/-- Represents the number of spoiled or invalidated votes -/
def spoiled_votes : ℕ := sorry

theorem election_result :
  candidate_A_percentage = 45/100 →
  vote_difference_A_B = 250 →
  percentage_difference_B_C = 5/100 →
  vote_ratio_D_C = 1/2 →
  spoiled_votes = 200 →
  total_valid_votes = 1250 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_l285_28568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_meal_options_l285_28536

-- Define the menu items and their prices
structure MenuItem where
  name : String
  price : Rat

-- Define the categories
inductive Category
  | Savory
  | Drink
  | Sweet

-- Define Maria's coins
def mariaCoins : List (Nat × Rat) :=
  [(5, 1/2), (7, 1/4), (4, 1/10), (5, 1/20)]

-- Define the menu
def menu : List (Category × MenuItem) :=
  [(Category.Savory, ⟨"Empanada", 39/10⟩),
   (Category.Savory, ⟨"Sandwich", 11/5⟩),
   (Category.Savory, ⟨"Pastry", 2⟩),
   (Category.Drink, ⟨"Soft drink", 19/10⟩),
   (Category.Drink, ⟨"Juice", 6/5⟩),
   (Category.Drink, ⟨"Refreshment", 1⟩),
   (Category.Sweet, ⟨"Ice cream", 1⟩),
   (Category.Sweet, ⟨"Bonbon", 1/2⟩),
   (Category.Sweet, ⟨"Coconut sweet", 2/5⟩)]

-- Define bus fare
def busFare : Rat := 9/10

-- Function to calculate total money
def totalMoney (coins : List (Nat × Rat)) : Rat :=
  coins.foldr (fun (count, value) acc => acc + count * value) 0

-- Function to count affordable meal combinations
def countAffordableMeals (money : Rat) (menu : List (Category × MenuItem)) : Nat :=
  sorry -- Implementation not required for the statement

-- Main theorem
theorem maria_meal_options :
  totalMoney mariaCoins = 49/10 ∧
  countAffordableMeals (totalMoney mariaCoins - busFare) menu = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_meal_options_l285_28536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_headcount_proof_l285_28540

def spring_03_04_headcount : ℕ := 10500
def spring_04_05_headcount : ℕ := 10700

def average_headcount : ℚ :=
  (spring_03_04_headcount + spring_04_05_headcount) / 2

def rounded_average : ℕ := Int.toNat ((average_headcount + 1/2).floor)

theorem average_headcount_proof :
  rounded_average = 10600 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_headcount_proof_l285_28540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l285_28591

/-- An ellipse with equation y²/16 + x²/12 = 1 -/
structure Ellipse where
  equation : ℝ → ℝ → Prop
  eq_def : ∀ x y : ℝ, equation x y ↔ y^2/16 + x^2/12 = 1

/-- Foci of the ellipse -/
structure Foci (e : Ellipse) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Points where a line through F₂ intersects the ellipse -/
structure IntersectionPoints (e : Ellipse) (f : Foci e) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_ellipse : e.equation A.1 A.2 ∧ e.equation B.1 B.2
  through_F₂ : ∃ t : ℝ, A = f.F₂ + t • (B - f.F₂) ∨ B = f.F₂ + t • (A - f.F₂)

/-- The perimeter of triangle ABF₁ -/
def trianglePerimeter (e : Ellipse) (f : Foci e) (p : IntersectionPoints e f) : ℝ :=
  dist p.A f.F₁ + dist p.B f.F₁ + dist p.A p.B

theorem ellipse_triangle_perimeter (e : Ellipse) (f : Foci e) (p : IntersectionPoints e f) :
  trianglePerimeter e f p = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l285_28591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_3_plus_4root3_l285_28520

/-- A line passing through the origin -/
structure OriginLine where
  slope : ℝ

/-- The line x = 1 -/
def vertical_line : Set (ℝ × ℝ) := {p | p.1 = 1}

/-- The line y = 1 + 1/√3 * x -/
def sloped_line : Set (ℝ × ℝ) := {p | p.2 = 1 + 1/(3:ℝ).sqrt * p.1}

/-- Check if three points form an equilateral triangle -/
def is_equilateral (a b c : ℝ × ℝ) : Prop :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
  (b.1 - c.1)^2 + (b.2 - c.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2

/-- The perimeter of a triangle given its three vertices -/
noncomputable def triangle_perimeter (a b c : ℝ × ℝ) : ℝ :=
  ((a.1 - b.1)^2 + (a.2 - b.2)^2).sqrt +
  ((b.1 - c.1)^2 + (b.2 - c.2)^2).sqrt +
  ((c.1 - a.1)^2 + (c.2 - a.2)^2).sqrt

theorem triangle_perimeter_is_3_plus_4root3 
  (l : OriginLine) 
  (p1 p2 p3 : ℝ × ℝ) 
  (h1 : p1 ∈ vertical_line) 
  (h2 : p2 ∈ sloped_line) 
  (h3 : p3.2 = l.slope * p3.1) 
  (h4 : is_equilateral p1 p2 p3) :
  triangle_perimeter p1 p2 p3 = 3 + 4 * (3:ℝ).sqrt := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_3_plus_4root3_l285_28520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_R_6789_l285_28576

noncomputable def a : ℝ := 4 + Real.sqrt 15
noncomputable def b : ℝ := 4 - Real.sqrt 15

noncomputable def R (n : ℕ) : ℝ := (1 / 2) * (a ^ n + b ^ n)

theorem units_digit_R_6789 : (Int.floor (R 6789) % 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_R_6789_l285_28576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MKF_is_45_degrees_l285_28516

/-- Parabola represented by the equation y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Point on the parabola -/
def M : ℝ × ℝ := (1, 2)

/-- Intersection of directrix and x-axis -/
def K : ℝ × ℝ := (-1, 0)

theorem angle_MKF_is_45_degrees 
  (h1 : M ∈ Parabola)
  (h2 : ‖(M.1 - focus.1, M.2 - focus.2)‖ = 2)
  (h3 : K.1 = -1 ∧ K.2 = 0) :
  ∃ θ : ℝ, θ = π / 4 ∧ 
    (K.1 - M.1) * (K.1 - focus.1) + (K.2 - M.2) * (K.2 - focus.2) = 
    ‖(K.1 - M.1, K.2 - M.2)‖ * ‖(K.1 - focus.1, K.2 - focus.2)‖ * Real.cos θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MKF_is_45_degrees_l285_28516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_product_l285_28594

theorem complex_sum_product (z : ℂ) (h : z = (1 + Complex.I) / Real.sqrt 2) : 
  (z^(1 : ℂ) + z^(4 : ℂ) + z^(9 : ℂ)) * (z^(-1 : ℂ) + z^(-4 : ℂ) + z^(-9 : ℂ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_product_l285_28594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_correct_propositions_l285_28549

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos (2 * x + Real.pi / 3) + 1

def symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = f x

def symmetric_graphs (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 - x) = f (x - 1)

def negation_equivalence : Prop :=
  (¬ ∀ x > 0, x^2 + 2*x - 3 > 0) ↔ (∃ x ≤ 0, x^2 + 2*x - 3 ≤ 0)

def sine_inequality : Prop :=
  ∀ α β, 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ α > β → Real.sin α > Real.sin β

theorem three_correct_propositions :
  (symmetry_center f (-5*Real.pi/12, 0)) ∧
  (symmetric_graphs f) ∧
  negation_equivalence ∧
  ¬sine_inequality :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_correct_propositions_l285_28549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_b_value_l285_28507

/-- A quadratic function f(x) = (1/12)x^2 + ax + b with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  x_intercept1 : ℝ
  x_intercept2 : ℝ
  y_intercept : ℝ
  T : ℝ × ℝ := (3, 3)
  h_x_intercepts : x_intercept1 ≠ x_intercept2
  h_y_intercept : y_intercept = b
  h_T_equidistant : 
    (T.1 - x_intercept1)^2 + T.2^2 = 
    T.1^2 + (T.2 - y_intercept)^2 ∧
    T.1^2 + (T.2 - y_intercept)^2 = 
    (T.1 - x_intercept2)^2 + T.2^2

/-- The main theorem stating that b = -6 for the given quadratic function -/
theorem quadratic_b_value (f : QuadraticFunction) : f.b = -6 := by
  sorry

#check quadratic_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_b_value_l285_28507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_value_l285_28552

def c : ℕ → ℕ
  | 0 => 2  -- Added this case to handle Nat.zero
  | 1 => 2
  | 2 => 3
  | n + 3 => c (n + 2) * c (n + 1)

theorem c_15_value : c 15 = 6^377 := by
  sorry  -- Proof is skipped using sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_value_l285_28552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_four_sided_polygon_l285_28546

/-- The set T of points (x, y) satisfying the given conditions -/
def T (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    let x := p.1
    let y := p.2
    3*a/4 ≤ x ∧ x ≤ 5*a/2 ∧
    3*a/4 ≤ y ∧ y ≤ 5*a/2 ∧
    x + a ≥ y ∧
    y + a ≥ x}

/-- The theorem stating that T forms a polygon with 4 sides -/
theorem T_is_four_sided_polygon (a : ℝ) (ha : a > 0) : 
  ∃ (vertices : Finset (ℝ × ℝ)), vertices.card = 4 ∧ 
  ∀ p : ℝ × ℝ, p ∈ T a ↔ p ∈ (convexHull ℝ (↑vertices : Set (ℝ × ℝ))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_four_sided_polygon_l285_28546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_properties_l285_28560

-- Define the original triangle ABC
def A : ℝ × ℝ := (1, 4)
def B : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (4, 1)

-- Define the reflection function across y = x
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Define the reflected triangle A'B'C'
def A' : ℝ × ℝ := reflect A
def B' : ℝ × ℝ := reflect B
def C' : ℝ × ℝ := reflect C

-- Function to calculate the area of a triangle
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Function to calculate the slope of a line
noncomputable def slopeOfLine (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (y2 - y1) / (x2 - x1)

theorem triangle_reflection_properties :
  (triangleArea A' B' C' = 0) ∧ (slopeOfLine A C = slopeOfLine A' C') := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_properties_l285_28560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aryan_payment_percentage_is_60_percent_l285_28598

noncomputable def aryan_debt : ℝ := 1200
noncomputable def kyro_debt : ℝ := aryan_debt / 2
noncomputable def kyro_payment_percentage : ℝ := 0.8
noncomputable def initial_savings : ℝ := 300
noncomputable def final_savings : ℝ := 1500

noncomputable def aryan_payment_percentage : ℝ :=
  (final_savings - initial_savings - kyro_debt * kyro_payment_percentage) / aryan_debt

theorem aryan_payment_percentage_is_60_percent :
  aryan_payment_percentage = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aryan_payment_percentage_is_60_percent_l285_28598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l285_28587

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle (renamed to avoid conflict)
def circle_eq (a b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2 + b^2

-- Define the point P
noncomputable def P : ℝ × ℝ := (-1, Real.sqrt 2 / 2)

-- Define the foci
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (t : ℝ) (x y : ℝ) : Prop := x = t * y + 1

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Main theorem
theorem ellipse_area_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse a b P.1 P.2) 
  (h4 : ∃ M : ℝ × ℝ, M.1 = 0 ∧ dot_product (P.1 - M.1, P.2 - M.2) (F2.1 - M.1, F2.2 - M.2) = 0) :
  (ellipse 2 1 P.1 P.2) ∧ 
  (∀ t : ℝ, t ≠ 0 → 
    ∃ A B C D : ℝ × ℝ,
    (circle_eq a b A.1 A.2) ∧ 
    (circle_eq a b B.1 B.2) ∧
    (ellipse 2 1 C.1 C.2) ∧ 
    (ellipse 2 1 D.1 D.2) ∧
    (line_l t A.1 A.2) ∧ 
    (line_l t B.1 B.2) ∧
    (line_l t C.1 C.2) ∧ 
    (line_l t D.1 D.2) →
    (2/3 ≤ dot_product (A.1 - F1.1, A.2 - F1.2) (B.1 - F1.1, B.2 - F1.2) ∧ 
     dot_product (A.1 - F1.1, A.2 - F1.2) (B.1 - F1.1, B.2 - F1.2) ≤ 1) →
    (4 * Real.sqrt 3 / 5 ≤ Real.sqrt ((C.1 - F1.1)^2 + (C.2 - F1.2)^2) * Real.sqrt ((D.1 - F1.1)^2 + (D.2 - F1.2)^2) / 2 ∧
     Real.sqrt ((C.1 - F1.1)^2 + (C.2 - F1.2)^2) * Real.sqrt ((D.1 - F1.1)^2 + (D.2 - F1.2)^2) / 2 ≤ 4 * Real.sqrt 6 / 7)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l285_28587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_comparison_l285_28501

noncomputable section

open Real

theorem function_comparison (f : ℝ → ℝ) (θ : ℝ) 
  (h1 : Differentiable ℝ f)
  (h2 : ∀ x, (deriv f x) + f x < 0)
  (h3 : sin θ + cos θ < sqrt 2) :
  f (sin θ + cos θ) / exp (sqrt 2 - sin θ - cos θ) > f (sqrt 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_comparison_l285_28501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_days_after_friday_is_sunday_l285_28589

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to calculate the day of the week after a given number of days
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

-- Theorem statement
theorem hundred_days_after_friday_is_sunday :
  dayAfter DayOfWeek.Friday 100 = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_days_after_friday_is_sunday_l285_28589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_with_lcm_120_l285_28564

theorem greatest_x_with_lcm_120 (x : ℕ) :
  (Nat.lcm x (Nat.lcm 8 12) = 120) → x ≤ 120 ∧ ∃ y : ℕ, y = 120 ∧ Nat.lcm y (Nat.lcm 8 12) = 120 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_with_lcm_120_l285_28564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_M_is_correct_l285_28543

-- Define the matrix M
def M : Matrix (Fin 2) (Fin 2) ℝ := !![6, 2; 4, 4]

-- Define the eigenvector
def eigenvector : Fin 2 → ℝ := ![1, 1]

-- Define the point that is transformed
def point : Fin 2 → ℝ := ![-1, 2]

-- Define the transformed point
def transformed_point : Fin 2 → ℝ := ![-2, 4]

theorem matrix_properties :
  -- Condition 1: M * eigenvector = 8 * eigenvector
  M.mulVec eigenvector = (8 : ℝ) • eigenvector ∧
  -- Condition 2: M transforms (-1, 2) to (-2, 4)
  M.mulVec point = transformed_point ∧
  -- Condition 3: For any point (x, y) on line l, M * [x, y] = [x', y'] where x' - 2y' = 4
  ∀ x y x' y' : ℝ, M.mulVec ![x, y] = ![x', y'] → x' - 2*y' = 4 →
  -- Conclusion 1: The equation of line l is x + 3y + 2 = 0
  x + 3*y + 2 = 0 := by sorry

-- Verify that M is indeed [[6, 2], [4, 4]]
theorem M_is_correct : M = !![6, 2; 4, 4] := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_M_is_correct_l285_28543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_eq_298_l285_28512

/-- A sequence {aₙ} where a₁ = 100 and aₙ₊₁ = aₙ + 2 for all n ≥ 1 -/
def a : ℕ → ℕ
  | 0 => 100  -- Define for 0 to cover all natural numbers
  | n + 1 => a n + 2

/-- The 100th term of the sequence is 298 -/
theorem a_100_eq_298 : a 100 = 298 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_eq_298_l285_28512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_satisfies_conditions_interest_rate_approx_l285_28527

/-- The interest rate that satisfies the given conditions --/
noncomputable def interest_rate : ℝ :=
  Real.sqrt (3.0000000000002274 / 1875)

/-- The principal amount --/
def principal : ℝ := 1875

/-- The time period in years --/
def time : ℕ := 2

/-- The difference between compound and simple interest amounts --/
def interest_difference : ℝ := 3.0000000000002274

/-- Theorem stating that the calculated interest rate satisfies the given conditions --/
theorem interest_rate_satisfies_conditions :
  principal * (1 + interest_rate) ^ time - (principal + principal * interest_rate * ↑time) = interest_difference :=
by sorry

/-- Theorem stating that the interest rate is approximately 0.04 --/
theorem interest_rate_approx :
  |interest_rate - 0.04| < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_satisfies_conditions_interest_rate_approx_l285_28527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l285_28592

-- Define the hyperbola
structure Hyperbola where
  focus : ℝ × ℝ
  asymptote_slope : ℝ

-- Define the given hyperbola
noncomputable def given_hyperbola : Hyperbola :=
  { focus := (5, 0),
    asymptote_slope := 3/4 }

-- Define the standard form of a hyperbola
def standard_form (a b : ℝ) : (ℝ → ℝ → Prop) :=
  λ x y => x^2 / a^2 - y^2 / b^2 = 1

-- Theorem stating the standard equation of the given hyperbola
theorem hyperbola_standard_equation (h : Hyperbola) 
  (hf : h.focus = (5, 0)) (ha : h.asymptote_slope = 3/4) :
  standard_form 4 3 = λ x y => x^2 / 16 - y^2 / 9 = 1 :=
by
  sorry

#check hyperbola_standard_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l285_28592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_ten_equals_twenty_l285_28538

-- Define the function f
noncomputable def f (y : ℝ) : ℝ := 
  let x := (y - 1) / 3
  x^2 + 3*x + 2

-- State the theorem
theorem f_of_ten_equals_twenty : f 10 = 20 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the let expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_ten_equals_twenty_l285_28538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_sharing_profit_l285_28545

noncomputable def profit (x : ℝ) : ℝ := -1/2 * x^2 + 60 * x - 800

theorem bike_sharing_profit :
  (∀ x : ℝ, x > 0 → (profit x > 800 ↔ x > 40 ∧ x < 80)) ∧
  (∀ x : ℝ, x > 0 → profit x / x ≤ profit 40 / 40) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_sharing_profit_l285_28545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_path_length_l285_28537

/-- Represents a rectangular box with dimensions 1 × 1 × 2 meters -/
structure Box where
  length : ℝ := 1
  width : ℝ := 1
  height : ℝ := 2

/-- Represents a path in the box -/
structure BoxPath where
  start_corner : Fin 8
  end_corner : Fin 8
  face_diagonals : Fin 2
  space_diagonals : Fin 1
  visits_all_corners : Bool

/-- Calculates the length of a path in the box -/
noncomputable def path_length (b : Box) (p : BoxPath) : ℝ := sorry

/-- Theorem: The longest path in the box has length 2√2 + √6 + 6 -/
theorem longest_path_length (b : Box) : 
  ∃ (p : BoxPath), p.visits_all_corners ∧ p.start_corner = p.end_corner ∧ 
  path_length b p = 2 * Real.sqrt 2 + Real.sqrt 6 + 6 ∧
  ∀ (q : BoxPath), q.visits_all_corners → q.start_corner = q.end_corner → 
  path_length b q ≤ path_length b p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_path_length_l285_28537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_equal_l285_28533

-- Define the polynomials and their properties
variable (P Q R : ℝ → ℝ)
variable (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)

-- Axioms based on the given conditions
axiom positive_leading_coeff : 
  (∃ a > 0, ∃ b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) ∧ 
  (∃ a > 0, ∃ b c : ℝ, ∀ x, Q x = a * x^2 + b * x + c) ∧ 
  (∃ a > 0, ∃ b c : ℝ, ∀ x, R x = a * x^2 + b * x + c)

axiom distinct_roots : 
  a₁ ≠ a₂ ∧ b₁ ≠ b₂ ∧ c₁ ≠ c₂

axiom roots_of_P : P a₁ = 0 ∧ P a₂ = 0
axiom roots_of_Q : Q b₁ = 0 ∧ Q b₂ = 0
axiom roots_of_R : R c₁ = 0 ∧ R c₂ = 0

axiom equal_values_R : P c₁ + Q c₁ = P c₂ + Q c₂
axiom equal_values_P : Q a₁ + R a₁ = Q a₂ + R a₂
axiom equal_values_Q : P b₁ + R b₁ = P b₂ + R b₂

-- Theorem statement
theorem roots_sum_equal : a₁ + a₂ = b₁ + b₂ ∧ b₁ + b₂ = c₁ + c₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_equal_l285_28533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_plus_q_l285_28577

noncomputable section

-- Define the rational function
def f (p q : ℝ → ℝ) : ℝ → ℝ := λ x ↦ p x / q x

-- Define the properties of p and q
def horizontal_asymptote (p q : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f p q x| < ε

def vertical_asymptote (p q : ℝ → ℝ) : Prop :=
  ∀ M > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + 2| ∧ |x + 2| < δ → |f p q x| > M

def q_quadratic (q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c

def p_at_3 (p : ℝ → ℝ) : Prop := p 3 = 2
def q_at_3 (q : ℝ → ℝ) : Prop := q 3 = 5

-- Theorem to prove
theorem p_plus_q (p q : ℝ → ℝ) 
  (h1 : horizontal_asymptote p q)
  (h2 : vertical_asymptote p q)
  (h3 : q_quadratic q)
  (h4 : p_at_3 p)
  (h5 : q_at_3 q) :
  ∀ x, p x + q x = x^2 + 3*x - 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_plus_q_l285_28577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_one_third_a_share_correct_l285_28551

/-- Represents the investment and profit sharing scenario of three partners -/
structure Partnership where
  a_investment : ℚ
  annual_gain : ℚ

/-- Calculates A's share of the annual gain -/
def a_share (p : Partnership) : ℚ :=
  p.annual_gain / 3

/-- Theorem stating that A's share is one-third of the annual gain -/
theorem a_share_is_one_third (p : Partnership) : 
  a_share p = p.annual_gain / 3 := by
  rfl

/-- Theorem stating that A's share is correct given the investment conditions -/
theorem a_share_correct (p : Partnership) (h : p.annual_gain = 18000) : 
  a_share p = 6000 := by
  rw [a_share, h]
  norm_num

#eval a_share { a_investment := 1000, annual_gain := 18000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_one_third_a_share_correct_l285_28551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_eight_factors_after_two_division_l285_28530

def has_eight_factors (m : Nat) : Prop :=
  (Finset.filter (fun i => i ∣ m) (Finset.range (m + 1))).card = 8

def highest_power_of_two (n : Nat) : Nat :=
  if n = 0 then 0 else Nat.log 2 n

theorem smallest_integer_with_eight_factors_after_two_division :
  ∀ n : Nat, n > 0 → n < 2187 →
    ¬(has_eight_factors (n / 2^(highest_power_of_two n))) →
    has_eight_factors (2187 / 2^(highest_power_of_two 2187)) := by
  sorry

#check smallest_integer_with_eight_factors_after_two_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_eight_factors_after_two_division_l285_28530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_transformation_first_step_l285_28558

/-- Represents the components of a population growth pattern -/
inductive PopulationGrowthComponent
  | BirthRate
  | MortalityRate
  | NaturalGrowthRate

/-- Represents a population growth pattern -/
structure PopulationGrowthPattern where
  components : List PopulationGrowthComponent

/-- Represents the first step in the transformation of a population growth pattern -/
inductive TransformationFirstStep
  | MortalityRateDecline
  | BirthRateDecline
  | NaturalGrowthRateDecline
  | NaturalGrowthRateIncrease

/-- Function to determine the first step in the transformation of a population growth pattern -/
def transformationFirstStep (pattern : PopulationGrowthPattern) : TransformationFirstStep :=
  TransformationFirstStep.MortalityRateDecline

/-- Given a population growth pattern composed of birth rate, mortality rate, and natural growth rate,
    prove that the first step in its transformation is a decline in the mortality rate -/
theorem population_growth_transformation_first_step
  (pattern : PopulationGrowthPattern)
  (h : pattern.components = [PopulationGrowthComponent.BirthRate,
                             PopulationGrowthComponent.MortalityRate,
                             PopulationGrowthComponent.NaturalGrowthRate]) :
  TransformationFirstStep.MortalityRateDecline = transformationFirstStep pattern :=
by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_transformation_first_step_l285_28558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l285_28571

theorem equation_solution : ∃ x : ℝ, 81 = 3 * (27 : ℝ) ^ (x - 2) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l285_28571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_statement_S_l285_28580

-- Define the types of aristocrats
inductive AristocratType
  | Type1
  | Type2

-- Define the possible answers
inductive Answer
  | Bal
  | Da

-- Define the meaning of "бал"
inductive BalMeaning
  | Yes
  | No

-- Define a function to determine if an aristocrat tells the truth
def tellsTruth (t : AristocratType) (m : BalMeaning) : Prop :=
  match t, m with
  | AristocratType.Type1, BalMeaning.Yes => True
  | AristocratType.Type2, BalMeaning.No => True
  | _, _ => False

-- Define the statement S
def S (t : AristocratType) : Prop := t = AristocratType.Type1

-- Define a function to represent asking the question "Is S equivalent to X?"
noncomputable def askQuestion (t : AristocratType) (m : BalMeaning) (X : Prop) : Answer :=
  if (tellsTruth t m) = (S t ↔ X) then Answer.Bal else Answer.Da

-- The main theorem
theorem magic_statement_S :
  ∀ (t : AristocratType) (m : BalMeaning) (X : Prop),
    (askQuestion t m X = Answer.Bal) ↔ X :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_statement_S_l285_28580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l285_28510

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + (Real.sqrt 7 / 3) * Real.cos x * Real.sin x

theorem f_extrema :
  ∃ (max_val min_val : ℝ) (max_x1 max_x2 min_x1 min_x2 : ℝ),
    (∀ x, 0 ≤ x → x ≤ 2 * Real.pi → f x ≤ max_val) ∧
    (∀ x, 0 ≤ x → x ≤ 2 * Real.pi → f x ≥ min_val) ∧
    (f max_x1 = max_val) ∧ (f max_x2 = max_val) ∧
    (f min_x1 = min_val) ∧ (f min_x2 = min_val) ∧
    (0 ≤ max_x1 ∧ max_x1 ≤ 2 * Real.pi) ∧
    (0 ≤ max_x2 ∧ max_x2 ≤ 2 * Real.pi) ∧
    (0 ≤ min_x1 ∧ min_x1 ≤ 2 * Real.pi) ∧
    (0 ≤ min_x2 ∧ min_x2 ≤ 2 * Real.pi) ∧
    (max_val = 7/6) ∧
    (min_val = -1/6) ∧
    (|max_x1 - 1.21| < 0.01) ∧
    (|max_x2 - 4.35| < 0.01) ∧
    (|min_x1 - 2.78| < 0.01) ∧
    (|min_x2 - 5.92| < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l285_28510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_work_hours_l285_28574

/-- Calculates the number of hours Nancy needs to work to pay the remaining tuition --/
noncomputable def hours_to_work (tuition hourly_wage : ℚ) (parent_contribution_ratio : ℚ) 
  (scholarship : ℚ) (loan_multiplier : ℚ) : ℚ :=
  let parent_contribution := parent_contribution_ratio * tuition
  let loan := loan_multiplier * scholarship
  let total_aid := parent_contribution + scholarship + loan
  let remaining_tuition := tuition - total_aid
  remaining_tuition / hourly_wage

/-- Theorem stating that Nancy needs to work 200 hours --/
theorem nancy_work_hours : 
  hours_to_work 22000 10 (1/2) 3000 2 = 200 := by
  -- Unfold the definition of hours_to_work
  unfold hours_to_work
  -- Simplify the arithmetic expressions
  simp [Rat.mul_num_den, Rat.add_num_den]
  -- The proof is completed
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_work_hours_l285_28574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l285_28542

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6) - 1

theorem triangle_properties (a b c A B C : ℝ) :
  c = Real.sqrt 3 →
  f C = 0 →
  Real.sin B = 2 * Real.sin A →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a * Real.sin C = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin B →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  (a = 1 ∧ b = 2) ∧
  (∀ a' b' : ℝ, c^2 = a'^2 + b'^2 - 2*a'*b'*(Real.cos C) →
    a' * b' * Real.sin C / 2 ≤ 3 * Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l285_28542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_theorem_l285_28517

/-- Triangle ABC with vertices A(1, 10), B(3, 0), and C(9, 0) -/
def A : ℝ × ℝ := (1, 10)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (9, 0)

/-- Horizontal line y = t intersects AB at J and AC at K -/
noncomputable def intersectionPoint (t : ℝ) (p q : ℝ × ℝ) : ℝ × ℝ :=
  let slope := (q.2 - p.2) / (q.1 - p.1)
  let x := (t - p.2) / slope + p.1
  (x, t)

noncomputable def J (t : ℝ) : ℝ × ℝ := intersectionPoint t A B
noncomputable def K (t : ℝ) : ℝ × ℝ := intersectionPoint t A C

/-- Area of triangle AJK is 7.5 -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  0.5 * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

/-- Main theorem: If area of AJK is 7.5, then t = 5 -/
theorem triangle_intersection_theorem :
  ∃ (t : ℝ), triangleArea A (J t) (K t) = 7.5 ∧ 0 < t ∧ t < 10 → t = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_theorem_l285_28517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l285_28575

-- Define the binomial expressions
noncomputable def f (x : ℝ) := (1 + 2 * Real.sqrt x) ^ 3
noncomputable def g (x : ℝ) := (1 - x ^ (1/3)) ^ 5

-- Define the product of the binomial expressions
noncomputable def h (x : ℝ) := f x * g x

-- Theorem statement
theorem coefficient_of_x_in_expansion :
  (deriv h) 0 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l285_28575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scott_bought_five_pounds_of_eggplants_l285_28528

/-- Represents the ingredients and costs for Scott's ratatouille recipe --/
structure Ratatouille where
  eggplant_price : ℚ
  zucchini_amount : ℚ
  zucchini_price : ℚ
  tomato_amount : ℚ
  tomato_price : ℚ
  onion_amount : ℚ
  onion_price : ℚ
  basil_amount : ℚ
  basil_price : ℚ
  quart_yield : ℚ
  quart_price : ℚ

/-- Calculates the number of pounds of eggplants Scott bought --/
noncomputable def eggplant_amount (r : Ratatouille) : ℚ :=
  (r.quart_yield * r.quart_price -
   (r.zucchini_amount * r.zucchini_price +
    r.tomato_amount * r.tomato_price +
    r.onion_amount * r.onion_price +
    r.basil_amount * r.basil_price * 2)) / r.eggplant_price

/-- Theorem stating that Scott bought 5 pounds of eggplants --/
theorem scott_bought_five_pounds_of_eggplants :
  let r : Ratatouille := {
    eggplant_price := 2,
    zucchini_amount := 4,
    zucchini_price := 2,
    tomato_amount := 4,
    tomato_price := 7/2,
    onion_amount := 3,
    onion_price := 1,
    basil_amount := 1,
    basil_price := 5/2,
    quart_yield := 4,
    quart_price := 10
  }
  eggplant_amount r = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scott_bought_five_pounds_of_eggplants_l285_28528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_count_l285_28525

theorem x_intercepts_count : ∃! (n : ℕ), ∃ (S : Finset ℝ),
  (∀ x ∈ S, (x - 5) * (x^2 + 7*x + 12) * (x - 1) = 0) ∧
  (∀ x, (x - 5) * (x^2 + 7*x + 12) * (x - 1) = 0 → x ∈ S) ∧
  S.card = n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_count_l285_28525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_sum_of_digits_l285_28544

def is_single_digit_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 10

def is_product_of_three_distinct_primes (n d e : ℕ) : Prop :=
  n = d * e * (12 * d + e) ∧
  Nat.Prime d ∧ Nat.Prime e ∧ Nat.Prime (12 * d + e) ∧
  d ≠ e ∧ d ≠ (12 * d + e) ∧ e ≠ (12 * d + e)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_sum_of_digits :
  ∃ n d e : ℕ,
    is_single_digit_prime d ∧
    is_single_digit_prime e ∧
    is_product_of_three_distinct_primes n d e ∧
    (∀ m k l : ℕ, is_single_digit_prime k ∧ is_single_digit_prime l ∧ 
      is_product_of_three_distinct_primes m k l → m ≤ n) ∧
    sum_of_digits n = 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_sum_of_digits_l285_28544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_25_l285_28534

def sequence_sum (n : ℕ) : ℕ := n^2 + 2*n - 1

def sequence_term (n : ℕ) : ℕ :=
  if n = 1 then sequence_sum 1
  else sequence_sum n - sequence_sum (n-1)

def odd_sum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => sequence_term (2*i + 1))

theorem odd_sum_25 : odd_sum 13 = 350 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_25_l285_28534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_5_days_avg_is_33_l285_28553

/-- Represents the TV production scenario in a factory --/
structure TVProduction where
  first_25_days_avg : ℚ
  total_days : ℕ
  overall_avg : ℚ

/-- Calculates the average production for the last 5 days given the production data --/
def last_5_days_avg (prod : TVProduction) : ℚ :=
  ((prod.overall_avg * prod.total_days) - (prod.first_25_days_avg * 25)) / 5

/-- Theorem stating that given the specific production data, the average for the last 5 days is 33 --/
theorem last_5_days_avg_is_33 (prod : TVProduction) 
  (h1 : prod.first_25_days_avg = 63)
  (h2 : prod.total_days = 30)
  (h3 : prod.overall_avg = 58) :
  last_5_days_avg prod = 33 := by
  sorry

#eval last_5_days_avg { first_25_days_avg := 63, total_days := 30, overall_avg := 58 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_5_days_avg_is_33_l285_28553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_sin_2x_l285_28581

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- State the theorem
theorem area_under_sin_2x : 
  (∫ x in (0)..(π), f x) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_sin_2x_l285_28581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_max_area_l285_28585

theorem triangle_angle_and_max_area (a b c A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  (1/2 * b - Real.sin C) * Real.cos A = Real.sin A * Real.cos C →
  a = 2 →
  A = π/4 ∧ ∃ (S : ℝ), S = Real.sqrt 2 + 1 ∧ 
    ∀ (S' : ℝ), (∃ (b' c' : ℝ), S' = 1/2 * b' * c' * Real.sin A ∧ 
      0 < b' ∧ 0 < c' ∧ b'^2 + c'^2 = 2 * b' * c' * Real.cos A + 4) → 
    S' ≤ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_max_area_l285_28585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_sine_function_l285_28539

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetry_center_of_sine_function
  (ω φ : ℝ)
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_period : ∀ x, f ω φ (x + 4 * π) = f ω φ x)
  (h_value : f ω φ (π / 3) = 1)
  : ∃ k : ℤ, ∀ x, f ω φ (x + 2 * k * π - 2 * π / 3) = f ω φ (-x + 2 * k * π - 2 * π / 3) := by
  sorry

#check symmetry_center_of_sine_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_sine_function_l285_28539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l285_28559

/-- The cubic function f(x) = x³ - 3x + 2 -/
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

/-- The quadratic function g(x) = x² - 2x + 2 -/
def g (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The set of x-coordinates where f(x) = g(x) -/
def X : Set ℝ := {x | f x = g x}

theorem intersection_sum :
  ∃ x₁ x₂ x₃, x₁ ∈ X ∧ x₂ ∈ X ∧ x₃ ∈ X ∧
    (∀ x ∈ X, x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    x₁ + x₂ + x₃ = 0 ∧
    f x₁ + f x₂ + f x₃ = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l285_28559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_intersecting_line_l285_28561

/-- Given an ellipse M with eccentricity e and a point on it, prove its standard equation and the equation of a line intersecting it. -/
theorem ellipse_and_intersecting_line 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (e : ℝ) 
  (h_e : e = Real.sqrt 3 / 2) 
  (h_point : a^2 * (1/2)^2 + b^2 * (Real.sqrt 2)^2 = a^2 * b^2) :
  -- Standard equation of ellipse M
  (∃ (x y : ℝ), x^2 / 3 + y^2 / (3/4) = 1) ∧
  -- Equation of line l
  (∃ (m : ℝ), m = Real.sqrt 30 / 5 ∨ m = -Real.sqrt 30 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_intersecting_line_l285_28561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l285_28504

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2 / 2
  | n + 1 => Real.sqrt 2 / 2 * Real.sqrt (1 - Real.sqrt (1 - a n ^ 2))

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (1 + b n ^ 2) - 1) / b n

theorem sequence_inequality (n : ℕ) :
  2^(n+2) * a n < Real.pi ∧ Real.pi < 2^(n+2) * b n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l285_28504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_area_l285_28500

/-- Parabola type representing y² = 2px --/
structure Parabola where
  p : ℝ

/-- Line with inclination angle π/4 --/
structure Line where
  c : ℝ

/-- Point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given three points --/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

theorem parabola_line_intersection_area 
  (para : Parabola) 
  (l : Line) 
  (p : Point) 
  (h1 : p.y^2 = 2 * para.p * p.x)  -- P lies on the parabola
  (h2 : p.x = 3 ∧ p.y = 2)         -- P is (3, 2)
  (h3 : l.c = 1)                   -- Line equation: y = x - 1
  : 
  ∃ (q : Point), 
    q.y^2 = 2 * para.p * q.x ∧     -- Q lies on the parabola
    q.y = q.x - 1 ∧                -- Q lies on the line
    triangleArea ⟨0, 0⟩ p q = 10/9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_area_l285_28500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_x_neg_solution_set_l285_28597

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 3 then -(x-2)^2 + 1
  else if x = 0 then 0
  else if -3 < x ∧ x < 0 then (x+2)^2 - 1
  else 0  -- This case should never occur given the domain, but Lean requires a total function

-- State the theorem
theorem f_cos_x_neg_solution_set (x : ℝ) :
  (f x * Real.cos x < 0) ↔ 
  (x ∈ Set.Ioo (-Real.pi/2) (-1) ∪ Set.Ioo 0 1 ∪ Set.Ioo (Real.pi/2) 3) :=
by sorry

-- Additional properties that might be needed
axiom f_odd (x : ℝ) : -3 < x → x < 3 → f (-x) = -f x
axiom f_domain (x : ℝ) : f x ≠ 0 → -3 < x ∧ x < 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_x_neg_solution_set_l285_28597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l285_28523

/-- The number of days it takes B to complete the work alone -/
noncomputable def b_days : ℝ := 18

/-- The number of days it takes A and B together to complete the work -/
noncomputable def together_days : ℝ := 6

/-- The number of days it takes A to complete the work alone -/
noncomputable def a_days : ℝ := 9

/-- The rate at which A completes the work per day -/
noncomputable def a_rate : ℝ := 1 / a_days

/-- The rate at which B completes the work per day -/
noncomputable def b_rate : ℝ := 1 / b_days

/-- The combined rate at which A and B complete the work per day -/
noncomputable def combined_rate : ℝ := 1 / together_days

theorem work_completion_time :
  a_rate + b_rate = combined_rate := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l285_28523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_right_directrix_l285_28524

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The distance from the center to a focus -/
noncomputable def c : ℝ := 1

/-- The distance from a point to the left focus -/
noncomputable def distance_to_left_focus : ℝ := 5/2

/-- The distance from a point to the right focus -/
noncomputable def distance_to_right_focus : ℝ := 2*c - distance_to_left_focus

/-- The eccentricity of the ellipse -/
noncomputable def e : ℝ := c / 2

theorem distance_to_right_directrix (P : ℝ × ℝ) :
  ellipse_equation P.1 P.2 →
  distance_to_left_focus = 5/2 →
  ∃ d : ℝ, d = 3 ∧ distance_to_right_focus / d = e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_right_directrix_l285_28524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_face_diagonals_equal_a_l285_28578

/-- A polyhedron with specific properties -/
structure SpecialPolyhedron where
  a : ℝ  -- Side length of the square face
  b : ℝ  -- Length of other edges
  h : 0 < a ∧ 0 < b  -- Positive side lengths

/-- The diagonal length of a face in the special polyhedron -/
noncomputable def faceDiagonal (p : SpecialPolyhedron) : ℝ := Real.sqrt ((3 * p.a^2 + p.a * p.b + p.b^2) / 4)

/-- Theorem: Two faces of the special polyhedron have diagonals of length a -/
theorem two_face_diagonals_equal_a (p : SpecialPolyhedron) : faceDiagonal p = p.a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_face_diagonals_equal_a_l285_28578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_area_ratio_l285_28547

/-- A circle tangent to a side and diagonal of a unit square -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- A unit square with two tangent circles -/
structure UnitSquareWithCircles where
  square : Set (ℝ × ℝ)
  circle1 : TangentCircle
  circle2 : TangentCircle

/-- The area of a circle -/
noncomputable def circle_area (c : TangentCircle) : ℝ :=
  Real.pi * c.radius^2

/-- The theorem stating that the ratio of areas of the two circles is 1 -/
theorem tangent_circles_area_ratio (s : UnitSquareWithCircles) :
  circle_area s.circle2 / circle_area s.circle1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_area_ratio_l285_28547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_probability_l285_28555

/-- The probability of arranging 6 books (1 Chinese, 2 English, 3 Mathematics) in a row
    such that all books of the same subject are together is 1/10. -/
theorem book_arrangement_probability :
  let total_books : ℕ := 6
  let chinese_books : ℕ := 1
  let english_books : ℕ := 2
  let math_books : ℕ := 3
  let probability : ℚ := 1 / 10
  probability = (
    (Nat.factorial 3) *  -- Arrangement of subject groups
    (Nat.factorial english_books) *  -- Arrangement within English books
    (Nat.factorial math_books)  -- Arrangement within Mathematics books
  ) / (Nat.factorial total_books) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_probability_l285_28555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_l285_28526

/-- Given points A, B, and C in 3D space, prove that A is equidistant from B and C -/
theorem point_equidistant (A B C : ℝ × ℝ × ℝ) : 
  A = (0, 0, 1) → B = (3, 3, 1) → C = (4, 1, 2) →
  (A.1 - B.1)^2 + (A.2.1 - B.2.1)^2 + (A.2.2 - B.2.2)^2 = 
  (A.1 - C.1)^2 + (A.2.1 - C.2.1)^2 + (A.2.2 - C.2.2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_l285_28526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_magnitude_l285_28565

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_subtraction_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1) :
  Real.sqrt ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_magnitude_l285_28565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_variance_from_peak_l285_28550

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation
  σ_pos : σ > 0

/-- The probability density function of a normal distribution -/
noncomputable def normalPDF (X : NormalRandomVariable) (x : ℝ) : ℝ :=
  1 / (X.σ * Real.sqrt (2 * Real.pi)) * Real.exp (-(1/2) * ((x - X.μ) / X.σ)^2)

/-- The theorem stating that if the highest point of the normal curve 
    has coordinates (10, 1/2), then the variance is 2/π -/
theorem normal_variance_from_peak (X : NormalRandomVariable) 
  (h_peak : normalPDF X 10 = 1/2) : X.σ^2 = 2/Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_variance_from_peak_l285_28550
