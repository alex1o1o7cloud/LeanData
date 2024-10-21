import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_counts_three_times_more_apples_l419_41928

/-- Represents the number of apple trees and apples counted by a person -/
structure Count where
  trees : ℕ
  apples : ℕ

/-- Represents a segment of the journey around the lake -/
inductive Segment
| AB
| BC
| CD

/-- The total number of apple trees around the lake -/
def total_trees : ℕ := sorry

/-- The total number of apples on all trees around the lake -/
def total_apples : ℕ := sorry

/-- Counts for each person on each segment of the journey -/
def count (person : Bool) (segment : Segment) : Count := sorry

/-- The conditions of the problem -/
axiom circular_lake : True
axiom opposite_directions : True
axiom twice_trees_AB : (count false Segment.AB).trees = 2 * (count true Segment.AB).trees
axiom seven_times_apples_AB : (count false Segment.AB).apples = 7 * (count true Segment.AB).apples
axiom twice_trees_BC : (count false Segment.BC).trees = 2 * (count true Segment.BC).trees
axiom seven_times_apples_BC : (count false Segment.BC).apples = 7 * (count true Segment.BC).apples
axiom twice_trees_CD : (count false Segment.CD).trees = 2 * (count true Segment.CD).trees

axiom total_trees_sum : 
  total_trees = (count false Segment.AB).trees + (count true Segment.AB).trees

axiom total_apples_sum : 
  total_apples = (count false Segment.AB).apples + (count true Segment.AB).apples

/-- The theorem to be proved -/
theorem vasya_counts_three_times_more_apples :
  (count true Segment.CD).apples = 3 * (count false Segment.CD).apples := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_counts_three_times_more_apples_l419_41928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l419_41996

-- Define the projection function as noncomputable
noncomputable def proj (a : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * a.1 + v.2 * a.2) / (a.1^2 + a.2^2)
  (scalar * a.1, scalar * a.2)

-- State the theorem
theorem projection_line_equation :
  ∀ (v : ℝ × ℝ), 
    proj (5, 2) v = (-5/2, -1) →
    v.2 = -5/2 * v.1 - 29/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l419_41996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l419_41933

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line passing through a point at a given angle -/
structure Line where
  point : ℝ × ℝ
  angle : ℝ

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The length of a chord in a circle -/
noncomputable def chord_length (c : Circle) (l : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem ellipse_eccentricity (e : Ellipse) (l : Line) (c : Circle) :
  c.r = e.b →
  l.angle = Real.pi / 6 →
  chord_length c l = Real.sqrt 3 * e.b →
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry

#check ellipse_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l419_41933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l419_41988

/-- Represents a quadratic function f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_nonzero : a ≠ 0

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_x (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_y (f : QuadraticFunction) : ℝ := (4 * f.a * f.c - f.b^2) / (4 * f.a)

/-- The vertex coordinates of a quadratic function -/
noncomputable def vertex (f : QuadraticFunction) : ℝ × ℝ := (vertex_x f, vertex_y f)

theorem quadratic_vertex (f : QuadraticFunction) :
  vertex f = (-f.b / (2 * f.a), (4 * f.a * f.c - f.b^2) / (4 * f.a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l419_41988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_empty_subsets_count_l419_41912

/-- The set S is defined as {1, 2, 3} -/
def S : Finset Nat := {1, 2, 3}

/-- Theorem stating that the number of non-empty subsets of S is 7 -/
theorem non_empty_subsets_count : Finset.card (Finset.powerset S \ {∅}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_empty_subsets_count_l419_41912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_fraction_l419_41967

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then -x^2 + 4 else x * Real.exp x

theorem range_of_fraction (x₁ x₂ : ℝ) (h1 : f x₁ = f x₂) (h2 : x₁ < x₂) :
  ∃ y, y ∈ Set.Iic 0 ∧ y = f x₂ / x₁ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_fraction_l419_41967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b₁_value_l419_41966

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 8 + 32*x - 12*x^2 - 4*x^3 + x^4

-- Define the set of roots for f(x)
variable (x₁ x₂ x₃ x₄ : ℝ)

-- Define the conditions for the roots of f(x)
axiom distinct_roots : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄
axiom roots_of_f : f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0

-- Define the coefficients of g(x)
variable (b₀ b₁ b₂ b₃ : ℝ)

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := b₀ + b₁*x + b₂*x^2 + b₃*x^3 + x^4

-- Define the conditions for the roots of g(x)
axiom roots_of_g : g (x₁^2) = 0 ∧ g (x₂^2) = 0 ∧ g (x₃^2) = 0 ∧ g (x₄^2) = 0

-- State the theorem
theorem b₁_value : b₁ = -1216 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b₁_value_l419_41966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_exists_l419_41914

/-- Represents a square in the plane -/
structure Square where
  sideLength : ℕ
  center : ℝ × ℝ

/-- Predicate to check if two squares touch at their vertices -/
def touchesAtVertex (s1 s2 : Square) : Prop := sorry

/-- Predicate to check if two squares overlap -/
def overlaps (s1 s2 : Square) : Prop := sorry

/-- Predicate to check if a square touches exactly two other squares -/
def touchesExactlyTwo (s : Square) (arrangement : List Square) : Prop := sorry

/-- Theorem: For any n ≥ 5, there exists an arrangement of n squares satisfying the conditions -/
theorem square_arrangement_exists (n : ℕ) (h : n ≥ 5) :
  ∃ (arrangement : List Square),
    arrangement.length = n ∧
    (∀ i, i ∈ arrangement.map Square.sideLength ↔ i ∈ Finset.range n) ∧
    (∀ s1 s2, s1 ∈ arrangement → s2 ∈ arrangement → s1 ≠ s2 → ¬overlaps s1 s2 ∨ touchesAtVertex s1 s2) ∧
    (∀ s, s ∈ arrangement → touchesExactlyTwo s arrangement) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_exists_l419_41914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edges_l419_41935

/-- A cube has 12 edges. -/
theorem cube_edges : 12 = 12 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edges_l419_41935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Q_less_than_bound_l419_41900

/-- The probability of drawing n-1 white marbles followed by a red marble -/
def Q (n : ℕ) : ℚ :=
  (2^n * (Nat.factorial (n-1))^2 : ℚ) / (Nat.factorial (2*n-1) * (2*n+1))

/-- The smallest positive integer n such that Q(n) < 1/3000 -/
theorem smallest_n_for_Q_less_than_bound : 
  (∀ k < 12, Q k ≥ 1/3000) ∧ Q 12 < 1/3000 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Q_less_than_bound_l419_41900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_s4_a4_l419_41909

/-- Geometric sequence with common ratio 3 -/
def geometric_sequence (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => 3 * geometric_sequence a₁ n

/-- Sum of first n terms of the geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - 3^n) / (1 - 3)

/-- Theorem: The ratio of S₄ to a₄ is 40/27 -/
theorem geometric_ratio_s4_a4 (a₁ : ℝ) :
  geometric_sum a₁ 4 / geometric_sequence a₁ 4 = 40 / 27 := by
  sorry

#check geometric_ratio_s4_a4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_s4_a4_l419_41909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_distance_approx_l419_41985

/-- Represents a bicycle trip with two parts -/
structure BicycleTrip where
  distance1 : ℝ
  speed1 : ℝ
  speed2 : ℝ
  avgSpeed : ℝ

/-- Calculates the distance traveled in the second part of the trip -/
noncomputable def secondDistance (trip : BicycleTrip) : ℝ :=
  let time1 := trip.distance1 / trip.speed1
  let x := trip.avgSpeed * (time1 + trip.distance1 / trip.speed2) - trip.distance1
  x

/-- Theorem stating that given the conditions, the second distance is approximately 10.01 km -/
theorem second_distance_approx (trip : BicycleTrip)
  (h1 : trip.distance1 = 8)
  (h2 : trip.speed1 = 10)
  (h3 : trip.speed2 = 8)
  (h4 : trip.avgSpeed = 8.78) :
  abs (secondDistance trip - 10.01) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_distance_approx_l419_41985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l419_41973

/-- Given a circle C with polar equation ρ = 2 and a line l: kx + y + 3 = 0 tangent to C -/
theorem circle_and_tangent_line (k : ℝ) :
  /- 1. The Cartesian equation of circle C is x² + y² = 4 -/
  (∀ x y : ℝ, x^2 + y^2 = 4 ↔ (x^2 + y^2 : ℝ) = 2^2) ∧
  /- 2. The value of k for the tangent line is ±√5/2 -/
  (∀ x y : ℝ, k * x + y + 3 = 0 → x^2 + y^2 = 4 → k = Real.sqrt 5 / 2 ∨ k = -Real.sqrt 5 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l419_41973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_felicity_gas_usage_l419_41989

/-- Proves that Felicity used 23 gallons of gas given the conditions of the problem -/
theorem felicity_gas_usage : ∃ (adhira_usage : ℝ), 4 * adhira_usage - 5 = 23 := by
  use 7
  norm_num

#check felicity_gas_usage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_felicity_gas_usage_l419_41989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_l419_41990

/-- Represents the cost price of a single article -/
noncomputable def C : ℝ := sorry

/-- Represents the selling price of a single article -/
noncomputable def S : ℝ := sorry

/-- The condition that the cost price of 40 articles equals the selling price of 25 articles -/
axiom price_relation : 40 * C = 25 * S

/-- The definition of profit percentage -/
noncomputable def profit_percentage : ℝ := (S - C) / C * 100

/-- Theorem stating that under the given condition, the profit percentage is 60% -/
theorem merchant_profit : profit_percentage = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_l419_41990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_schedule_ways_l419_41987

/-- Represents a chess match between two schools -/
structure ChessMatch where
  school1_players : Nat
  school2_players : Nat
  games_per_pairing : Nat
  games_per_round : Nat

/-- Calculate the total number of games in the match -/
def total_games (chess_match : ChessMatch) : Nat :=
  chess_match.school1_players * chess_match.school2_players * chess_match.games_per_pairing

/-- Calculate the number of rounds in the match -/
def number_of_rounds (chess_match : ChessMatch) : Nat :=
  total_games chess_match / chess_match.games_per_round

/-- Calculate the number of ways to schedule the match -/
def schedule_ways (chess_match : ChessMatch) : Nat :=
  Nat.factorial (number_of_rounds chess_match)

/-- Theorem stating the number of ways to schedule the specific chess match -/
theorem chess_match_schedule_ways :
  ∃ (chess_match : ChessMatch),
    chess_match.school1_players = 4 ∧
    chess_match.school2_players = 4 ∧
    chess_match.games_per_pairing = 3 ∧
    chess_match.games_per_round = 4 ∧
    schedule_ways chess_match = 479001600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_schedule_ways_l419_41987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annies_journey_time_l419_41902

-- Define the segments of Annie's journey
structure Segment where
  distance : ℚ
  speed : ℚ

-- Define Annie's journey
def annies_journey : List Segment := [
  { distance := 5, speed := 2 },  -- Walk to bus stop
  { distance := 7, speed := 4 },  -- Bus ride to train station
  { distance := 10, speed := 5 }, -- Train ride to friend's house
  { distance := 4, speed := 2 },  -- Walk to coffee shop
  { distance := 4, speed := 2 }   -- Walk back to friend's house
]

-- Calculate time for a single segment
def segment_time (s : Segment) : ℚ := s.distance / s.speed

-- Calculate total time for the journey
def total_journey_time (journey : List Segment) : ℚ :=
  2 * (List.sum (List.map segment_time journey)) -- Double for round trip

-- Theorem statement
theorem annies_journey_time :
  total_journey_time annies_journey = 33/2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annies_journey_time_l419_41902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_less_than_two_negation_l419_41931

theorem sin_less_than_two_negation :
  (¬ ∀ x : ℝ, Real.sin x < 2) ↔ (∃ x : ℝ, Real.sin x ≥ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_less_than_two_negation_l419_41931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l419_41984

/-- Given an ellipse with foci F₁(-c,0) and F₂(c,0), equation x²/a² + y²/b² = 1 where a > b > 0,
    and a point P on the ellipse such that PF₁ · PF₂ = 2c², 
    prove that the eccentricity e of the ellipse satisfies 1/2 ≤ e ≤ √3/3 -/
theorem ellipse_eccentricity_range (a b c : ℝ) (P : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
  (P.1 + c)^2 + P.2^2 + (P.1 - c)^2 + P.2^2 = 2*c^2 →
  let e := Real.sqrt (1 - b^2 / a^2)
  1/2 ≤ e ∧ e ≤ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l419_41984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_de_moivre_z_seq_formula_L_formula_l419_41948

noncomputable section

-- Define complex number
def z (r : ℝ) (α : ℝ) : ℂ := r * (Complex.cos α + Complex.I * Complex.sin α)

-- Define the sequence z_n
def z_seq (α : ℝ) : ℕ → ℂ
| 0 => ((1 + Complex.I) / (1 - Complex.I)) ^ 20
| n + 1 => z_seq α n * (1/2 * (Complex.cos α + Complex.I * Complex.sin α))

-- Define the sum L
def L (α : ℝ) : ℝ := Real.sqrt (5 - 4 * Real.cos α)

-- Theorem statements
theorem de_moivre (r : ℝ) (α : ℝ) (n : ℕ) (h : r > 0) :
  (z r α) ^ n = z (r ^ n) (n * α) := by
  sorry

theorem z_seq_formula (n : ℕ) (α : ℝ) :
  z_seq α n = (1/2) ^ (n - 1) * (Complex.cos ((n - 1) * α) + Complex.I * Complex.sin ((n - 1) * α)) := by
  sorry

theorem L_formula (α : ℝ) :
  ∑' n, Complex.abs (z_seq α (n + 1) - z_seq α n) = L α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_de_moivre_z_seq_formula_L_formula_l419_41948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l419_41968

-- Define the line l: kx - y + 2k - 1 = 0
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y + 2 * k - 1 = 0

-- Define the circle x² + y² = 6
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 6

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem line_circle_intersection (k : ℝ) :
  (∃ A B : ℝ × ℝ, 
    line k A.1 A.2 ∧ 
    line k B.1 B.2 ∧ 
    circleEq A.1 A.2 ∧ 
    circleEq B.1 B.2 ∧ 
    distance A.1 A.2 B.1 B.2 = 2 * Real.sqrt 2) →
  k = -3/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l419_41968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_one_third_l419_41927

/-- The sum of an infinite geometric series with first term 1/4 and common ratio 1/4 --/
noncomputable def shaded_area_sum : ℝ := (1/4) / (1 - 1/4)

/-- The theorem stating that the shaded area sum equals 1/3 --/
theorem shaded_area_equals_one_third : shaded_area_sum = 1/3 := by
  unfold shaded_area_sum
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_one_third_l419_41927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_fifteen_degrees_equality_l419_41903

theorem cosine_fifteen_degrees_equality : 
  1/2 - Real.cos (15 * π / 180)^2 = -Real.sqrt 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_fifteen_degrees_equality_l419_41903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_championship_probability_l419_41980

/-- The probability of the Lakers winning a single game -/
noncomputable def p : ℝ := 2/3

/-- The probability of the Celtics winning a single game -/
noncomputable def q : ℝ := 1/3

/-- The number of games the Lakers need to win to secure the Championship -/
def n : ℕ := 5

/-- The probability of the Lakers winning the Championship -/
noncomputable def lakers_win_probability : ℝ :=
  (p^n) +
  (5 * p^n * q) +
  (15 * p^n * q^2) +
  (35 * p^n * q^3) +
  (70 * p^n * q^4)

theorem lakers_championship_probability :
  ∃ ε > 0, |lakers_win_probability - 0.8498| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_championship_probability_l419_41980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_sum_theorem_l419_41920

/-- Represents the simple interest calculation for a given principal, rate, and time. -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- 
Given a sum invested at simple interest for 5 years, 
if an interest rate 3% higher would yield an additional Rs. 1200, 
then the sum invested is Rs. 8000.
-/
theorem investment_sum_theorem (principal : ℝ) (rate : ℝ) :
  simpleInterest principal (rate + 3) 5 - simpleInterest principal rate 5 = 1200 →
  principal = 8000 := by
  sorry

#check investment_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_sum_theorem_l419_41920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l419_41921

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (25 - x^2)

-- Define the domain of f
def dom_f : Set ℝ := { x | 0 ≤ x ∧ x ≤ 4 }

-- Define the proposed inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := Real.sqrt (25 - x^2)

-- Define the domain of f_inv
def dom_f_inv : Set ℝ := { x | 3 ≤ x ∧ x ≤ 5 }

-- Theorem stating that f_inv is the inverse of f
theorem f_inverse_correct :
  (∀ x ∈ dom_f, f_inv (f x) = x) ∧
  (∀ y ∈ dom_f_inv, f (f_inv y) = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l419_41921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_m_value_l419_41963

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line l
def line_l (x y m : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the condition for the shortest chord
def shortest_chord (m : ℝ) : Prop :=
  ∀ x y, circle_C x y ∧ line_l x y m →
    ∀ m', m' ≠ m → ∃ x' y', circle_C x' y' ∧ line_l x' y' m' ∧
      (x' - x)^2 + (y' - y)^2 > 0

-- Theorem statement
theorem shortest_chord_m_value :
  ∀ m : ℝ, shortest_chord m → m = -3/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_m_value_l419_41963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_molecules_after_reaction_l419_41945

-- Define Avogadro's constant
noncomputable def N_A : ℝ := sorry

-- Define the number of moles of H₂ and I₂
def n_H2 : ℝ := 0.1
def n_I2 : ℝ := 0.1

-- Define the reaction in a closed container
def closed_container_reaction (n_H2 n_I2 : ℝ) : ℝ := 2 * (n_H2 + n_I2)

-- Theorem statement
theorem total_molecules_after_reaction :
  closed_container_reaction n_H2 n_I2 * N_A = 0.2 * N_A := by
  -- Proof steps would go here
  sorry

#check total_molecules_after_reaction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_molecules_after_reaction_l419_41945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l419_41977

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3*x - x^2 else Real.log (x + 1)

-- State the theorem
theorem f_inequality_range (a : ℝ) :
  (∀ x, |f x| ≥ a * x) ↔ a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l419_41977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l419_41901

-- Define the parametric equation of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, t + 1)

-- Define the polar equation of curve C
noncomputable def curve_C_polar (θ : ℝ) : ℝ := 4 * Real.cos θ / (Real.sin θ)^2

-- Define the Cartesian equation of curve C
def curve_C_cartesian (x y : ℝ) : Prop := y^2 = 4 * x

-- Theorem statement
theorem intersection_length :
  ∃ (A B : ℝ × ℝ) (t₁ t₂ : ℝ),
    (∀ x y, curve_C_cartesian x y ↔ ∃ θ, (x, y) = (curve_C_polar θ * Real.cos θ, curve_C_polar θ * Real.sin θ)) →
    A = line_l t₁ ∧ 
    B = line_l t₂ ∧
    curve_C_cartesian A.1 A.2 ∧
    curve_C_cartesian B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 15 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l419_41901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlyn_visible_area_l419_41965

open Real

/-- The area visible to Charlyn as she walks around a square -/
noncomputable def visible_area (square_side : ℝ) (visibility_radius : ℝ) : ℝ :=
  let inner_square_side := square_side - 2 * visibility_radius
  let inner_square_area := inner_square_side ^ 2
  let visible_inside_square := square_side ^ 2 - inner_square_area
  let visible_rectangles := 4 * (square_side * visibility_radius)
  let visible_quarter_circles := 4 * (π * visibility_radius ^ 2 / 4)
  visible_inside_square + visible_rectangles + visible_quarter_circles

/-- Theorem stating the area visible to Charlyn -/
theorem charlyn_visible_area :
  visible_area 10 2 = 64 + 80 + 4 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlyn_visible_area_l419_41965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_distance_100_l419_41964

/-- Represents the complex number e^(πi/4) -/
noncomputable def θ : ℂ := Complex.exp (Complex.I * Real.pi / 4)

/-- The position of the bug after n steps -/
noncomputable def bugPosition : ℕ → ℂ
  | 0 => 0
  | 1 => 2
  | n + 2 => bugPosition (n + 1) + (↑(n + 3) : ℂ) * θ^(n + 1)

/-- The theorem stating the distance from Q₀ to Q₁₀₀ -/
theorem bug_distance_100 : 
  Complex.abs (bugPosition 100) = (Real.sqrt 10205 * Real.sqrt (2 + Real.sqrt 2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_distance_100_l419_41964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_in_triangle_l419_41995

/-- Given a triangle ABC and a point P in its plane, if BC = 2CP, then AP = -1/2 AB + 3/2 AC -/
theorem vector_relation_in_triangle (A B C P : EuclideanSpace ℝ (Fin 2)) : 
  (C - B) = 2 • (P - C) → 
  (P - A) = (-1/2 : ℝ) • (B - A) + (3/2 : ℝ) • (C - A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_in_triangle_l419_41995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l419_41992

/-- Definition of line l₁ -/
def l₁ (m : ℝ) (x y : ℝ) : Prop :=
  (m + 3) * x + 4 * y + 3 * m - 5 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) (x y : ℝ) : Prop :=
  2 * x + (m + 5) * y - 8 = 0

/-- Slope of l₁ when it exists -/
noncomputable def slope_l₁ (m : ℝ) : ℝ := (m + 3) / (-4)

/-- Slope of l₂ when it exists -/
noncomputable def slope_l₂ (m : ℝ) : ℝ := -2 / (m + 5)

/-- Two lines are parallel if their slopes are equal -/
def parallel (m : ℝ) : Prop :=
  m ≠ -5 ∧ slope_l₁ m = slope_l₂ m

theorem parallel_lines_m_value :
  ∃ m : ℝ, parallel m ∧ m = -7 := by
  use -7
  apply And.intro
  · -- Prove that the lines are parallel when m = -7
    apply And.intro
    · -- Prove m ≠ -5
      norm_num
    · -- Prove slope_l₁ (-7) = slope_l₂ (-7)
      unfold slope_l₁ slope_l₂
      norm_num
  · -- Prove m = -7
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l419_41992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_l419_41979

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + time * rate)

theorem balance_difference : 
  let cynthia_balance := compound_interest 9000 0.05 25
  let david_balance := simple_interest 12000 0.04 25
  ⌊cynthia_balance - david_balance⌋ = 6477 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_l419_41979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_zero_l419_41976

noncomputable def points : List (ℝ × ℝ) := [(4, 12), (7, 23), (13, 38), (19, 43), (21, 55)]

noncomputable def above_line (p : ℝ × ℝ) : Bool :=
  p.2 > 3 * p.1 + 5

noncomputable def sum_x_above_line (pts : List (ℝ × ℝ)) : ℝ :=
  (pts.filter above_line).map (·.1) |>.sum

theorem sum_x_above_line_is_zero : sum_x_above_line points = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_zero_l419_41976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_relation_l419_41983

-- Define the function f
noncomputable def f (k : ℝ) (y : ℝ) : ℝ := k / (y ^ 2)

-- State the theorem
theorem inverse_square_relation (k : ℝ) :
  (∃ y : ℝ, f k y = 1) →
  (f k 4 = 0.5625) →
  (f k 3 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_relation_l419_41983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_18_is_even_and_less_than_20_l419_41930

def numbers : List ℕ := [15, 18, 29, 21]

theorem only_18_is_even_and_less_than_20 : 
  ∃! n, n ∈ numbers ∧ n % 2 = 0 ∧ n < 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_18_is_even_and_less_than_20_l419_41930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_stretch_l419_41956

noncomputable def f (x : ℝ) : ℝ := (2/3) * Real.sin (-x/2 + Real.pi/3)

noncomputable def g (x : ℝ) : ℝ := (2/3) * Real.sin (-x + Real.pi/3)

theorem horizontal_stretch :
  ∀ (x : ℝ), f x = g (x/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_stretch_l419_41956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_bond_rate_l419_41918

/-- Calculates the final value of an investment after two consecutive one-year bonds --/
def final_value (initial_investment rate1 rate2 : ℝ) : ℝ :=
  initial_investment * (1 + rate1) * (1 + rate2)

/-- Theorem stating the interest rate of the second bond --/
theorem second_bond_rate (initial_investment rate1 rate2 final_val : ℝ) :
  initial_investment = 15000 →
  rate1 = 0.08 →
  final_value initial_investment rate1 rate2 = 17160 →
  rate2 = 0.06 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_bond_rate_l419_41918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l419_41971

noncomputable def f (x : ℝ) := Real.sqrt (4 - x^2)

theorem f_properties :
  (∀ x, f x ≠ 0 → x ∈ Set.Icc (-2) 2) ∧
  (∀ x ∈ Set.Icc (-2) 2, f (-x) = f x) ∧
  (Set.range f = Set.Icc 0 1 →
    {x | f x ≠ 0} = Set.Icc (-2) (-Real.sqrt 3) ∪ Set.Icc (Real.sqrt 3) 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l419_41971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_binomial_expansion_l419_41919

def binomialCoeff (n k : ℕ) : ℕ := sorry

theorem coefficient_x_squared_in_binomial_expansion :
  let n : ℕ := 10
  let general_term (r : ℕ) (x : ℚ) := 
    (binomialCoeff n r : ℚ) * ((-1)^r : ℚ) * x^(5 - 3*r/2)
  let coefficient_x_squared := general_term 2 1
  coefficient_x_squared = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_binomial_expansion_l419_41919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l419_41946

/-- The probability of drawing two balls of different colors from a bag containing 
    2 red balls and 3 white balls when randomly selecting 2 balls at once. -/
theorem different_color_probability : 
  (Nat.choose 2 1 * Nat.choose 3 1 : ℚ) / Nat.choose 5 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l419_41946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_heads_l419_41993

/-- The probability of getting at least two heads when tossing five fair coins -/
theorem probability_at_least_two_heads : ℝ := by
  -- Define the number of coins
  let n : ℕ := 5
  
  -- Define the probability of getting heads on a single coin toss
  let p : ℝ := 1 / 2
  
  -- Define the probability of getting at least two heads
  let prob_at_least_two : ℝ := 1 - (Nat.choose n 0 * p^0 * (1-p)^n + Nat.choose n 1 * p^1 * (1-p)^(n-1))
  
  -- State the theorem
  have : prob_at_least_two = 13 / 16 := by sorry
  
  -- Return the result
  exact (13 : ℝ) / 16


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_heads_l419_41993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_white_vertex_triangles_l419_41949

theorem two_white_vertex_triangles 
  (total_points : ℕ)
  (red_points white_points green_points : ℕ)
  (different_color_lines same_color_lines : ℕ)
  (diff_color_triangles : ℕ)
  (two_red_triangles : ℕ)
  (h1 : total_points = red_points + white_points + green_points)
  (h2 : different_color_lines = red_points * white_points + red_points * green_points + white_points * green_points)
  (h3 : same_color_lines = (red_points * (red_points - 1) + white_points * (white_points - 1) + green_points * (green_points - 1)) / 2)
  (h4 : diff_color_triangles = red_points * white_points * green_points)
  (h5 : two_red_triangles = (red_points * (red_points - 1) / 2) * (white_points + green_points))
  (h6 : different_color_lines = 213)
  (h7 : same_color_lines = 112)
  (h8 : diff_color_triangles = 540)
  (h9 : two_red_triangles = 612)
  (h10 : ∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → True) :
  (white_points * (white_points - 1) / 2 * (red_points + green_points) = 210) ∨
  (white_points * (white_points - 1) / 2 * (red_points + green_points) = 924) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_white_vertex_triangles_l419_41949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l419_41974

/-- A function f(x) with specific properties -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

/-- The theorem stating the properties of f and its derivative -/
theorem f_derivative_at_2 (a b : ℝ) :
  f a b 1 = -2 →
  (deriv (f a b)) 1 = 0 →
  (deriv (f a b)) 2 = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l419_41974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hitting_ground_time_l419_41908

/-- The time when a ball hits the ground, given initial conditions and height equation --/
theorem ball_hitting_ground_time :
  let initial_height : ℝ := 200
  let initial_velocity : ℝ := -30
  let gravity : ℝ := -16
  let height (t : ℝ) := gravity * t^2 + initial_velocity * t + initial_height
  ∃ t : ℝ, t > 0 ∧ height t = 0 ∧ |t - 2.7181| < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hitting_ground_time_l419_41908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l419_41905

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / (x - 1)

-- State the theorem
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l419_41905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l419_41969

def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def my_line (a b x y : ℝ) : Prop := a*x + b*y = 1

def point_outside_circle (a b : ℝ) : Prop := a^2 + b^2 > 1

theorem line_intersects_circle (a b : ℝ) 
  (h : point_outside_circle a b) : 
  ∃ x y : ℝ, my_circle x y ∧ my_line a b x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l419_41969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_closed_interval_l419_41916

-- Define set A
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + 4*x - 5 ≤ 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_open_closed_interval :
  A_intersect_B = Set.Ioc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_closed_interval_l419_41916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_parts_of_z_l419_41982

theorem complex_parts_of_z : 
  let z : ℂ := -7 - 9 * Complex.I 
  (Complex.re z = -7) ∧ (Complex.im z = -9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_parts_of_z_l419_41982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_hull_of_regular_polygons_l419_41953

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- The convex hull of a set of points in ℝ² -/
noncomputable def ConvexHull (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- The number of vertices in a convex polygon -/
noncomputable def numVertices (P : Set (ℝ × ℝ)) : ℕ := sorry

theorem convex_hull_of_regular_polygons 
  (polygons : Fin 1000 → RegularPolygon 100) :
  numVertices (ConvexHull (⋃ i, Set.range (polygons i).vertices)) > 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_hull_of_regular_polygons_l419_41953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speeds_l419_41972

/-- The average speed of a car traveling from A to B with two equal time segments -/
noncomputable def avg_speed_AB (v1 v2 : ℝ) : ℝ := (v1 + v2) / 2

/-- The average speed of a car traveling from B to A with two equal distance segments -/
noncomputable def avg_speed_BA (v3 v4 : ℝ) : ℝ := 2 / (1/v3 + 1/v4)

theorem car_average_speeds 
  (v1 v2 v3 v4 : ℝ) 
  (h1 : v1 = 60) 
  (h2 : v2 = 40) 
  (h3 : v3 = 80) 
  (h4 : v4 = 45) : 
  avg_speed_AB v1 v2 = 50 ∧ avg_speed_BA v3 v4 = 57.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speeds_l419_41972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_digit_palindromes_count_l419_41917

/-- A list representing the available digits -/
def available_digits : List Nat := [1, 1, 4, 4, 6, 6, 6]

/-- A function that checks if a list of digits forms a palindrome -/
def is_palindrome (digits : List Nat) : Bool :=
  digits = digits.reverse

/-- A function that counts the number of 7-digit palindromes -/
def count_palindromes (digits : List Nat) : Nat :=
  (digits.permutations.filter (fun l => l.length = 7 && is_palindrome l)).length

/-- Theorem stating that the number of 7-digit palindromes is 6 -/
theorem seven_digit_palindromes_count :
  count_palindromes available_digits = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_digit_palindromes_count_l419_41917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrationality_of_sqrt_seven_rationality_of_sqrt_36_rationality_of_neg_six_rationality_of_one_seventh_l419_41999

theorem irrationality_of_sqrt_seven : 
  ∃ (x : ℝ), x^2 = 7 ∧ ∀ (a b : ℤ), (b ≠ 0 → x ≠ a / b) :=
sorry

theorem rationality_of_sqrt_36 : 
  ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 36 = a / b :=
sorry

theorem rationality_of_neg_six : 
  ∃ (a b : ℤ), b ≠ 0 ∧ -6 = a / b :=
sorry

theorem rationality_of_one_seventh : 
  ∃ (a b : ℤ), b ≠ 0 ∧ 1/7 = a / b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrationality_of_sqrt_seven_rationality_of_sqrt_36_rationality_of_neg_six_rationality_of_one_seventh_l419_41999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sampling_is_42_percent_l419_41929

/-- Represents the probability of getting caught sampling a candy type -/
def catch_prob : Fin 3 → ℚ
| 0 => 25/100  -- Candy A
| 1 => 32/100  -- Candy B
| 2 => 50/100  -- Candy C

/-- Represents the distribution of caught candy samplers -/
def caught_dist : Fin 3 → ℚ
| 0 => 12/100  -- Candy A
| 1 => 5/100   -- Candy B
| 2 => 9/100   -- Candy C

/-- Represents the additional percentage of uncaught samplers -/
def uncaught_percent : Fin 3 → ℚ
| 0 => 7/100   -- Candy A
| 1 => 6/100   -- Candy B
| 2 => 3/100   -- Candy C

/-- The total percent of customers who sample any type of candy -/
def total_sampling_percent : ℚ :=
  (List.range 3).map (λ i => caught_dist (Fin.ofNat i) + uncaught_percent (Fin.ofNat i)) |>.sum

theorem total_sampling_is_42_percent :
  total_sampling_percent = 42/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sampling_is_42_percent_l419_41929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l419_41934

/-- A quadratic function passing through a specific point -/
structure QuadraticFunction where
  b : ℝ
  passes_through : -4^2 + b*(-4) + 5 = 5

/-- The axis of symmetry for a quadratic function -/
noncomputable def axis_of_symmetry (f : QuadraticFunction) : ℝ := -f.b / (2*(-1))

/-- The x-coordinates of intersection points with x-axis -/
def x_intersections (f : QuadraticFunction) : Set ℝ :=
  {x | -x^2 + f.b*x + 5 = 0}

/-- Main theorem about the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.b = -4 ∧
  axis_of_symmetry f = -2 ∧
  x_intersections f = {-5, 1} := by
  sorry

#check quadratic_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l419_41934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_30_l419_41938

/-- The number of hours marked on the clock face. -/
def hoursOnClock : ℕ := 12

/-- The number of degrees in a full rotation of the clock face. -/
def clockFaceDegrees : ℕ := 360

/-- The hour component of the time. -/
def hour : ℕ := 8

/-- The minute component of the time. -/
def minute : ℕ := 30

/-- Calculates the angle of the hour hand from the 12 o'clock position. -/
noncomputable def hourHandAngle (h : ℕ) (m : ℕ) : ℝ :=
  (h % hoursOnClock : ℝ) * (clockFaceDegrees : ℝ) / (hoursOnClock : ℝ) +
  (m : ℝ) * (clockFaceDegrees : ℝ) / (hoursOnClock * 60 : ℝ)

/-- Calculates the angle of the minute hand from the 12 o'clock position. -/
noncomputable def minuteHandAngle (m : ℕ) : ℝ :=
  (m : ℝ) * (clockFaceDegrees : ℝ) / 60

/-- Calculates the smaller angle between the hour and minute hands. -/
noncomputable def smallerAngleBetweenHands (h : ℕ) (m : ℕ) : ℝ :=
  let diff := abs (hourHandAngle h m - minuteHandAngle m)
  min diff (clockFaceDegrees - diff)

/-- The theorem to be proved. -/
theorem clock_angle_at_8_30 :
  smallerAngleBetweenHands hour minute = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_30_l419_41938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curved_figure_max_area_l419_41994

/-- The area of the curved figure ACBADEA as a function of x -/
noncomputable def curvedFigureArea (x : ℝ) : ℝ :=
  (2 * Real.pi * x - (Real.pi + 4) * x^2) / (2 * Real.pi^2)

/-- The maximum area of the curved figure ACBADEA -/
noncomputable def maxCurvedFigureArea : ℝ :=
  1 / (2 * (Real.pi + 4))

theorem curved_figure_max_area :
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 1 → curvedFigureArea y ≤ curvedFigureArea x) ∧
  curvedFigureArea x = maxCurvedFigureArea := by
  sorry

#check curved_figure_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curved_figure_max_area_l419_41994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_condition_l419_41959

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a - 1 / Real.exp x)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a + (x - 1) * (-1 / Real.exp x)

/-- Theorem stating the relationship between the range of 'a' and the existence of two points
    where the tangent is perpendicular to the y-axis -/
theorem tangent_perpendicular_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_derivative a x₁ = 0 ∧ f_derivative a x₂ = 0) ↔
  a > -1 / (Real.exp 2) ∧ a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_condition_l419_41959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_hyperbola_l419_41924

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the point P
noncomputable def P : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 2*x - y - Real.sqrt 2 = 0

-- Theorem statement
theorem tangent_to_hyperbola :
  ∀ x y : ℝ, hyperbola x y → (x, y) = P → tangent_line x y := by
  sorry

#check tangent_to_hyperbola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_hyperbola_l419_41924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_alpha_plus_pi_fourth_l419_41937

theorem cos_squared_alpha_plus_pi_fourth (α : Real) 
  (h : Real.sin (2 * α) = 2 / 3) : 
  (Real.cos (α + Real.pi / 4))^2 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_alpha_plus_pi_fourth_l419_41937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l419_41991

/-- Given polynomials A and B, prove that if 2A - B is a quadratic trinomial in x and y,
    then a = -1 and the given expression evaluates to -22. -/
theorem polynomial_problem (a : ℝ) : 
  let A := fun (x y : ℝ) => a * x^2 + 3 * x * y + 2 * abs a * x
  let B := fun (x y : ℝ) => 2 * x^2 + 6 * x * y + 4 * x + y + 1
  (∃ (p q r : ℝ), ∀ (x y : ℝ), 
    2 * A x y - B x y = p * x^2 + q * y + r ∧ p ≠ 0) →
  a = -1 ∧ 3 * (-3 * a^2 - 2 * a) - (a^2 - 2 * (5 * a - 4 * a^2 + 1) - 2 * a) = -22 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l419_41991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_sum_series_equals_two_l419_41911

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n+2 => fibonacci n + fibonacci (n+1)

noncomputable def fibonacci_sum_series : ℝ := ∑' n, (fibonacci (n+1)) / (2^(n+1))

theorem fibonacci_sum_series_equals_two : fibonacci_sum_series = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_sum_series_equals_two_l419_41911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spotlight_orientation_exists_l419_41910

/-- A cube in 3D space -/
structure Cube where
  side_length : ℝ
  center : Fin 3 → ℝ

/-- A right-angled trihedral spotlight -/
structure Spotlight where
  position : Fin 3 → ℝ
  orientation : Quaternion ℝ

/-- Predicate to check if a point is illuminated by the spotlight -/
def is_illuminated (s : Spotlight) (p : Fin 3 → ℝ) : Prop := sorry

/-- The vertices of a cube -/
def cube_vertices (c : Cube) : Set (Fin 3 → ℝ) := sorry

/-- Theorem: There exists an orientation of the spotlight that doesn't illuminate any cube vertex -/
theorem spotlight_orientation_exists (c : Cube) :
  ∃ (s : Spotlight), s.position = c.center ∧ 
    ∀ v ∈ cube_vertices c, ¬ is_illuminated s v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spotlight_orientation_exists_l419_41910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_iff_w_in_range_l419_41970

/-- The function f(x) = sin(wx) - √3 * cos(wx) -/
noncomputable def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x) - Real.sqrt 3 * Real.cos (w * x)

/-- Theorem stating the relationship between w and the number of zero points of f(x) in (0, π) -/
theorem two_zeros_iff_w_in_range (w : ℝ) :
  (w > 0) →
  (∃! (z₁ z₂ : ℝ), 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < π ∧ f w z₁ = 0 ∧ f w z₂ = 0) ↔
  (4/3 < w ∧ w ≤ 7/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_iff_w_in_range_l419_41970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l419_41906

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_neg : b < 0

/-- The eccentricity of a hyperbola -/
noncomputable def Hyperbola.eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance between the foci of a hyperbola -/
noncomputable def Hyperbola.focal_distance (h : Hyperbola) : ℝ :=
  2 * Real.sqrt (h.a^2 + h.b^2)

/-- The distance from the center to a vertex of a hyperbola -/
def Hyperbola.center_to_vertex (h : Hyperbola) : ℝ := h.a

/-- Theorem: If there exists a point P on the hyperbola such that |PA₁|² = |F₁F₂| · |A₁F₂|,
    then the eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity_sqrt_two (h : Hyperbola) 
  (h_point_exists : ∃ P : ℝ × ℝ, 
    (P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1) ∧ 
    ((P.1 - h.center_to_vertex)^2 + P.2^2 = h.focal_distance * (h.focal_distance / 2 + h.center_to_vertex))) :
  h.eccentricity = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l419_41906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_intersection_m_value_l419_41904

-- Define the hyperbola parameters
variable (a b : ℝ)
variable (h : a > 0 ∧ b > 0)

-- Define the eccentricity
noncomputable def eccentricity := Real.sqrt 3

-- Define the distance from foci to asymptotes
def foci_asymptote_distance := 2

-- Define the hyperbola equation
def hyperbola_eq (a b x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

-- Define the intersection line
def intersection_line (x y m : ℝ) := x - y + m = 0

-- Define the circle equation
def circle_eq (x y : ℝ) := x^2 + y^2 = 5

-- Theorem 1: Prove the specific hyperbola equation
theorem hyperbola_equation :
  (eccentricity = Real.sqrt 3) →
  (foci_asymptote_distance = 2) →
  ∃ (x y : ℝ), hyperbola_eq (9/2) 9 x y :=
by sorry

-- Theorem 2: Prove the value of m
theorem intersection_m_value :
  (eccentricity = Real.sqrt 3) →
  (foci_asymptote_distance = 2) →
  ∀ (m : ℝ),
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      hyperbola_eq (9/2) 9 x₁ y₁ ∧
      hyperbola_eq (9/2) 9 x₂ y₂ ∧
      intersection_line x₁ y₁ m ∧
      intersection_line x₂ y₂ m ∧
      x₁ ≠ x₂ ∧
      circle_eq ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)) →
    m = 1 ∨ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_intersection_m_value_l419_41904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l419_41936

/-- A power function with integer exponent -/
noncomputable def f (α : ℤ) (x : ℝ) : ℝ := x ^ α

/-- The property given in the problem -/
def has_property (α : ℤ) : Prop :=
  (f α 1)^2 + (f α (-1))^2 = 2 * ((f α 1) + (f α (-1)) - 1)

/-- Definition of an even function -/
def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

theorem power_function_property (α : ℤ) :
  has_property α → is_even_function (f α) := by
  sorry

#check power_function_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l419_41936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_approx_l419_41943

/-- A cyclic quadrilateral with given side lengths and perpendicular diagonals -/
structure CyclicQuadrilateral where
  K : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  kl_length : dist K L = 4
  lm_length : dist L M = 10
  mn_length : dist M N = 12
  cyclic : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    dist center K = radius ∧ 
    dist center L = radius ∧ 
    dist center M = radius ∧ 
    dist center N = radius
  perpendicular_diagonals : (M.1 - K.1) * (N.1 - L.1) + (M.2 - K.2) * (N.2 - L.2) = 0

/-- The distance between the midpoint of KN and line LM -/
noncomputable def midpoint_to_line_distance (q : CyclicQuadrilateral) : ℝ :=
  let midpoint_KN := ((q.K.1 + q.N.1) / 2, (q.K.2 + q.N.2) / 2)
  |midpoint_KN.1 - q.L.1|

/-- The main theorem -/
theorem midpoint_distance_approx (q : CyclicQuadrilateral) : 
  |midpoint_to_line_distance q - 6.87| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_approx_l419_41943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_angle_l419_41942

open Real

noncomputable def f (x : ℝ) := x^3 / 3 - x^2 + 1

noncomputable def tangent_angle (x : ℝ) := arctan (deriv f x)

theorem min_tangent_angle :
  ∀ x ∈ Set.Ioo 0 2, tangent_angle x ≥ π / 4 * 3 ∧
  ∃ y ∈ Set.Ioo 0 2, tangent_angle y = π / 4 * 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_angle_l419_41942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_with_lcm_and_ratio_l419_41975

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ) : 
  Nat.lcm a b = 30 → a * 3 = b * 2 → a + b = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_with_lcm_and_ratio_l419_41975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_a_squared_b_same_type_as_a_squared_b_l419_41962

-- Define a structure for polynomials
structure MyPolynomial where
  terms : List (ℕ × ℕ)  -- List of (exponent of a, exponent of b) pairs
  degree : ℕ
  vars : List Char

-- Define a function to check if two polynomials are of the same type
def same_type (p q : MyPolynomial) : Prop :=
  p.degree = q.degree ∧ p.vars = q.vars

-- Define the polynomials
def a_squared_b : MyPolynomial :=
  { terms := [(2, 1)], degree := 3, vars := ['a', 'b'] }

def two_a_squared_b : MyPolynomial :=
  { terms := [(2, 1)], degree := 3, vars := ['a', 'b'] }

-- State the theorem
theorem two_a_squared_b_same_type_as_a_squared_b :
  same_type two_a_squared_b a_squared_b := by
  -- Unfold the definition of same_type
  unfold same_type
  -- Split the conjunction
  apply And.intro
  -- Prove equality of degrees
  · rfl
  -- Prove equality of variables
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_a_squared_b_same_type_as_a_squared_b_l419_41962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_percentage_approx_70_l419_41978

/-- Represents a tiling of a plane with squares and hexagons -/
structure SquareHexagonTiling where
  -- Side length of squares and hexagons
  side_length : ℝ
  -- Number of squares in each tile arrangement
  num_squares : ℕ
  -- Number of hexagons in each tile arrangement
  num_hexagons : ℕ
  -- Assumption that side lengths are equal and positive
  side_length_pos : side_length > 0
  -- Assumption about the number of squares and hexagons
  tile_arrangement : num_squares = 4 ∧ num_hexagons = 4
  -- Assumption that hexagons only share sides with squares
  hexagon_adjacent_squares : True  -- This is a simplification as we can't easily represent this geometrically

/-- Calculates the area of a square given its side length -/
def square_area (s : ℝ) : ℝ := s^2

/-- Calculates the area of a regular hexagon given its side length -/
noncomputable def hexagon_area (s : ℝ) : ℝ := 3 * Real.sqrt 3 / 2 * s^2

/-- Calculates the percentage of the plane enclosed by hexagons -/
noncomputable def hexagon_percentage (t : SquareHexagonTiling) : ℝ :=
  let total_square_area := t.num_squares * square_area t.side_length
  let total_hexagon_area := t.num_hexagons * hexagon_area t.side_length
  let total_area := total_square_area + total_hexagon_area
  100 * total_hexagon_area / total_area

/-- Theorem stating that the percentage of the plane enclosed by hexagons is approximately 70% -/
theorem hexagon_percentage_approx_70 (t : SquareHexagonTiling) :
  ∃ ε > 0, abs (hexagon_percentage t - 70) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_percentage_approx_70_l419_41978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l419_41954

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector structure -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Defines the left focus of a hyperbola -/
noncomputable def left_focus (h : Hyperbola) : Point :=
  { x := -Real.sqrt (h.a^2 + h.b^2), y := 0 }

/-- Checks if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Checks if a point is on a line -/
def on_line (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Checks if two vectors are collinear -/
def collinear (v1 v2 : Vec) : Prop :=
  ∃ k : ℝ, v1.x = k * v2.x ∧ v1.y = k * v2.y

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (l : Line) (A B : Point) :
  l.slope = 1 →
  on_line l (left_focus h) →
  on_hyperbola h A →
  on_hyperbola h B →
  on_line l A →
  on_line l B →
  collinear { x := A.x + B.x, y := A.y + B.y } { x := -3, y := -1 } →
  Real.sqrt (h.a^2 + h.b^2) / h.a = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l419_41954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l419_41955

/-- Parabola with directrix x = -1 -/
structure Parabola where
  directrix : ℝ
  focus : ℝ × ℝ

/-- Line with slope √3 passing through a point -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Check if a point (x, y) is on the parabola -/
def onParabola (C : Parabola) (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

/-- Check if a point (x, y) is on the line -/
def onLine (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 - l.point.2 = l.slope * (p.1 - l.point.1)

/-- Theorem stating the properties of the parabola and line intersection -/
theorem parabola_line_intersection 
  (C : Parabola) 
  (l : Line) 
  (h1 : C.directrix = -1)
  (h2 : l.slope = Real.sqrt 3)
  (h3 : l.point = C.focus) :
  (∃ (A B : ℝ × ℝ), 
    onParabola C A ∧ onParabola C B ∧ 
    onLine l A ∧ onLine l B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l419_41955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_special_triangle_l419_41961

-- Define the type for the numbers in the circles
def CircleNumber := Fin 6 → Nat

-- Define the property that the numbers are 8, 9, 10, 11, 12, 13
def ValidNumbers (n : CircleNumber) : Prop :=
  (∀ i, n i ∈ ({8, 9, 10, 11, 12, 13} : Set Nat)) ∧
  (∀ i j, i ≠ j → n i ≠ n j)

-- Define the property that the sum of each side is equal
def EqualSides (n : CircleNumber) (S : Nat) : Prop :=
  n 0 + n 1 + n 2 = S ∧
  n 2 + n 3 + n 4 = S ∧
  n 4 + n 5 + n 0 = S

-- Theorem statement
theorem max_sum_special_triangle :
  ∃ (n : CircleNumber) (S : Nat),
    ValidNumbers n ∧
    EqualSides n S ∧
    (∀ (m : CircleNumber) (T : Nat), ValidNumbers m → EqualSides m T → T ≤ S) ∧
    S = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_special_triangle_l419_41961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platinum_matrix_existence_l419_41940

/-- Definition of a platinum matrix -/
def is_platinum_matrix (n : ℕ) (M : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  (∀ i j, M i j ∈ Finset.range n) ∧
  (∀ i, Function.Injective (M i)) ∧
  (∀ j, Function.Injective (λ i => M i j)) ∧
  (Function.Injective (λ i => M i i)) ∧
  (∃ f : Fin n → Fin n, Function.Injective f ∧
    (∀ i, f i ≠ i) ∧
    (∀ i, M i (f i) = i.val + 1))

/-- Theorem stating the existence condition for platinum matrices -/
theorem platinum_matrix_existence (n : ℕ) :
  (∃ M : Matrix (Fin n) (Fin n) ℕ, is_platinum_matrix n M) ↔ (n ≠ 2 ∧ n ≠ 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platinum_matrix_existence_l419_41940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_pyramid_l419_41947

/-- The lateral surface area of a regular quadrilateral pyramid -/
noncomputable def lateral_surface_area (a : ℝ) : ℝ :=
  a^2 * Real.sqrt 2

/-- Theorem: The lateral surface area of a regular quadrilateral pyramid 
    with base side length a and lateral face angle of 45° with the base plane 
    is equal to a^2 * √2 -/
theorem lateral_surface_area_of_pyramid (a : ℝ) (h : a > 0) :
  lateral_surface_area a = a^2 * Real.sqrt 2 := by
  -- Unfold the definition of lateral_surface_area
  unfold lateral_surface_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_pyramid_l419_41947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_smallest_k_l419_41925

def sequenceU (u : ℕ → ℚ) : Prop :=
  u 0 = 1/4 ∧ ∀ k, u (k + 1) = 3 * u k - 3 * (u k)^2

def hasLimit (u : ℕ → ℚ) (L : ℚ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ k ≥ N, |u k - L| < ε

def smallestK (u : ℕ → ℚ) (L : ℚ) (k : ℕ) : Prop :=
  (∀ j < k, |u j - L| > 1/2^2000) ∧ |u k - L| ≤ 1/2^2000

theorem sequence_limit_smallest_k (u : ℕ → ℚ) (L : ℚ) :
  sequenceU u → hasLimit u L → smallestK u L 12 :=
by
  intro hu hL
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_smallest_k_l419_41925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_implies_inclination_angle_l419_41950

/-- Given a function f(x) = a*sin(x) - b*cos(x) with symmetry axis x = π/4,
    prove that the angle of inclination of the line ax - by + c = 0 is 3π/4 radians. -/
theorem symmetry_axis_implies_inclination_angle 
  (a b c : ℝ) 
  (h_symmetry : ∀ x, a * Real.sin (π/4 + x) - b * Real.cos (π/4 + x) = 
                     a * Real.sin (π/4 - x) - b * Real.cos (π/4 - x)) :
  Real.arctan (a / b) = 3 * π / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_implies_inclination_angle_l419_41950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_BD_skew_l419_41913

-- Define a 3D space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define points A, B, C, D
variable (A B C D : V)

-- Define the condition that AB and CD are skew
def are_skew (A B C D : V) : Prop := 
  ∀ (t s : ℝ), A + t • (B - A) ≠ C + s • (D - C)

-- Define the condition that AC intersects AB and CD
def AC_intersects (A B C D : V) : Prop :=
  ∃ (t s : ℝ), A + t • (C - A) = A + s • (B - A) ∧
               A + t • (C - A) = C + (1 - s) • (D - C)

-- Define the condition that BD intersects AB and CD
def BD_intersects (A B C D : V) : Prop :=
  ∃ (t s : ℝ), B + t • (D - B) = A + s • (B - A) ∧
               B + t • (D - B) = C + (1 - s) • (D - C)

-- State the theorem
theorem AC_BD_skew {A B C D : V} 
  (h1 : are_skew A B C D) 
  (h2 : AC_intersects A B C D) 
  (h3 : BD_intersects A B C D) : 
  are_skew A C B D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_BD_skew_l419_41913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l419_41998

theorem calculation_proof :
  ((2/3 - 1/4 - 5/6) * 12 = -5) ∧
  ((-3)^2 * 2 + 4 * (-3) - 28 / (7/4) = -10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l419_41998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_to_identical_digits_l419_41957

theorem sum_to_identical_digits (n : ℕ) : 
  (n > 0 ∧ ∃ k : ℕ, k ∈ Finset.range 9 ∧ n * (n + 1) / 2 = 111 * k) ↔ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_to_identical_digits_l419_41957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_l419_41944

/-- The time (in minutes) it takes for two people walking in opposite directions 
    on a circular track to meet for the first time. -/
noncomputable def meetingTime (trackCircumference : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  trackCircumference / (speed1 + speed2)

/-- Theorem stating that two people walking in opposite directions on a 640m track
    at speeds of 4.2 km/hr and 3.8 km/hr will meet after approximately 4.8 minutes. -/
theorem meeting_time_approx :
  let trackCircumference : ℝ := 640
  let speed1 : ℝ := 4.2 * 1000 / 60  -- Convert km/hr to m/min
  let speed2 : ℝ := 3.8 * 1000 / 60  -- Convert km/hr to m/min
  abs (meetingTime trackCircumference speed1 speed2 - 4.8) < 0.1 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_l419_41944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_not_smaller_pair_l419_41997

-- Define the type for a star's properties
def StarProperty := ℕ × ℕ

-- Define the set of all stars
variable (S : Set StarProperty)

-- Axiom: The set of stars is infinite
axiom infinite_stars : Set.Infinite S

-- Axiom: All stars have unique properties
axiom unique_stars : ∀ s t : StarProperty, s ∈ S → t ∈ S → s ≠ t → s.1 ≠ t.1 ∨ s.2 ≠ t.2

-- Theorem: There exist two stars where one is not smaller than the other in both parameters
theorem exists_not_smaller_pair :
  ∃ s t : StarProperty, s ∈ S ∧ t ∈ S ∧ s.1 ≤ t.1 ∧ s.2 ≤ t.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_not_smaller_pair_l419_41997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_reasoning3_is_analogical_l419_41986

/-- Definition of inductive reasoning -/
def is_inductive_reasoning (r : Prop) : Prop :=
  ∃ (specific general : Prop), r ↔ (specific → general)

/-- Definition of deductive reasoning -/
def is_deductive_reasoning (r : Prop) : Prop :=
  ∃ (major minor conclusion : Prop), r ↔ (major ∧ minor → conclusion)

/-- Definition of similarity between cases -/
def similar (case1 case2 : Prop) : Prop := sorry

/-- Definition of analogical reasoning -/
def is_analogical_reasoning (r : Prop) : Prop :=
  ∃ (case1 case2 : Prop), r ↔ (case1 → case2) ∧ similar case1 case2

/-- The three types of reasoning presented in the problem -/
def reasoning1 : Prop := sorry
def reasoning2 : Prop := sorry
def reasoning3 : Prop := sorry

/-- Theorem stating that only reasoning3 is an example of analogical reasoning -/
theorem only_reasoning3_is_analogical :
  is_analogical_reasoning reasoning3 ∧
  ¬is_analogical_reasoning reasoning1 ∧
  ¬is_analogical_reasoning reasoning2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_reasoning3_is_analogical_l419_41986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l419_41960

theorem triangle_area_range (A B C : Real) (a b c : Real) (S : Real) :
  -- Conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧  -- Acute triangle
  A = π/3 ∧  -- Given angle A
  a = 2 * Real.sqrt 3 ∧  -- Given side a
  S = (1/2) * b * c * Real.sin A ∧  -- Area formula
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  a / Real.sin A = c / Real.sin C  -- Law of sines
  →
  2 * Real.sqrt 3 < S ∧ S ≤ 3 * Real.sqrt 3  -- Conclusion
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l419_41960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_equation_l419_41952

/-- A line passing through point (1,2) with distance 1 from the origin -/
structure SpecialLine where
  -- The slope of the line (None if vertical)
  slope : Option ℝ
  -- Assertion that the line passes through (1,2)
  passes_through_point : 
    match slope with
    | none => true  -- For vertical line x = 1
    | some k => 2 = k * (1 - 1) + 2
  -- Assertion that the distance from origin to line is 1
  distance_from_origin :
    match slope with
    | none => 1 = 1  -- For vertical line x = 1, distance is 1
    | some k => 1 = (|(- k) + 2|) / Real.sqrt (k^2 + 1)

/-- Theorem stating that a SpecialLine must be either x = 1 or 3x - 4y + 5 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.slope = none) ∨ 
  (l.slope = some (3/4)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_equation_l419_41952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l419_41958

theorem certain_number_proof (m n : ℕ) : 
  m = 6 → ((-2 : ℤ)^n : ℤ) = (2^(18 - m) : ℕ) → n = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l419_41958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_depends_on_s_and_r_l419_41951

/-- The perimeter of a square -/
def square_perimeter (s : ℝ) : ℝ := 4 * s

/-- The circumference of a circle -/
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

/-- The ratio of square perimeter to circle circumference -/
noncomputable def perimeter_ratio (s r : ℝ) : ℝ := (square_perimeter s) / (circle_circumference r)

/-- Theorem: The perimeter ratio depends on s and r -/
theorem perimeter_ratio_depends_on_s_and_r :
  ∀ (s₁ s₂ r₁ r₂ : ℝ), s₁ ≠ s₂ ∨ r₁ ≠ r₂ → perimeter_ratio s₁ r₁ ≠ perimeter_ratio s₂ r₂ :=
by
  intros s₁ s₂ r₁ r₂ h
  sorry

#check perimeter_ratio_depends_on_s_and_r

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_depends_on_s_and_r_l419_41951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_steps_l419_41915

theorem staircase_steps : ∃! n : ℕ, n > 0 ∧ (n + 2) / 3 - (n + 5) / 6 = 10 ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_steps_l419_41915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l419_41981

/-- A game state represented by a list of digits and the current player's turn -/
structure GameState where
  digits : List Int
  currentPlayer : Nat
deriving Repr

/-- Check if a number represented by a list of digits is divisible by 11 -/
def isDivisibleBy11 (digits : List Int) : Bool :=
  let alternatingSum := digits.enum.foldl (fun sum (i, d) => sum + (if i % 2 == 0 then d else -d)) 0
  alternatingSum % 11 == 0

/-- Check if a game state is losing (i.e., contains a number divisible by 11) -/
def isLosingState (state : GameState) : Bool :=
  List.any state.digits.tails isDivisibleBy11

/-- Get all possible next states from a given state -/
def nextStates (state : GameState) : List GameState :=
  (List.range 10).map fun d =>
    { digits := state.digits ++ [d]
    , currentPlayer := if state.currentPlayer == 1 then 2 else 1 }

/-- A winning strategy for a player is a function that always chooses a non-losing next state -/
def WinningStrategy := GameState → Option GameState

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : WinningStrategy), ∀ (initialState : GameState),
    initialState.currentPlayer == 2 →
    ∃ (finalState : GameState),
      finalState.digits.length ≥ initialState.digits.length ∧
      isLosingState finalState ∧
      finalState.currentPlayer == 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l419_41981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_calculation_l419_41932

/-- Calculates the cost of gas for a trip -/
def gas_cost (initial_reading final_reading : ℕ) (fuel_efficiency : ℚ) (price_per_gallon : ℚ) : ℚ :=
  let distance := final_reading - initial_reading
  let gallons_used := (distance : ℚ) / fuel_efficiency
  gallons_used * price_per_gallon

/-- Rounds a rational number to the nearest cent -/
def round_to_cent (x : ℚ) : ℚ :=
  (x * 100).floor / 100

theorem gas_cost_calculation :
  round_to_cent (gas_cost 75200 75238 32 (405/100)) = 481/100 := by
  sorry

#eval round_to_cent (gas_cost 75200 75238 32 (405/100))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_calculation_l419_41932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l419_41939

noncomputable section

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  (0 < A ∧ A < π) ∧ (0 < B ∧ B < π) ∧ (0 < C ∧ C < π) ∧ 
  (A + B + C = π) ∧
  -- Sides are positive
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  -- Given conditions
  (Real.cos B / b + Real.cos C / c = Real.sin A / (Real.sqrt 3 * Real.sin C)) ∧
  (Real.cos B + Real.sqrt 3 * Real.sin B = 2) →
  -- Conclusions
  (b = Real.sqrt 3) ∧
  (∀ (area : ℝ), area ≤ 3 * Real.sqrt 3 / 4 → 
    ∃ (a' c' : ℝ), 
      a' > 0 ∧ c' > 0 ∧
      area = 1/2 * a' * c' * Real.sin B) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l419_41939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l419_41923

noncomputable def f (x : Real) : Real := Real.sin x * Real.cos (x + Real.pi/3)

theorem triangle_area_proof (A B C : Real) (a b c : Real) :
  0 < A → A < Real.pi/2 →  -- A is acute
  f A = -(Real.sqrt 3)/4 →
  a = Real.sqrt 5 →
  b + c = 3 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l419_41923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_in_middle_l419_41922

/-- Represents the friends who need to be seated --/
inductive Friend
| Adam
| Brian
| Carla
| Diane
| Ellie

/-- Represents a seat in the row --/
inductive Seat
| One
| Two
| Three
| Four
| Five
| Six
| Seven

/-- Represents a seating arrangement --/
def Arrangement := Seat → Option Friend

/-- Checks if a friend is seated in a given seat --/
def is_seated (arr : Arrangement) (f : Friend) (s : Seat) : Prop :=
  arr s = some f

/-- Checks if two seats are adjacent --/
def are_adjacent (s1 s2 : Seat) : Prop :=
  match s1, s2 with
  | Seat.One, Seat.Two => True
  | Seat.Two, Seat.Three => True
  | Seat.Three, Seat.Four => True
  | Seat.Four, Seat.Five => True
  | Seat.Five, Seat.Six => True
  | Seat.Six, Seat.Seven => True
  | _, _ => False

/-- Checks if there are at least two seats between two given seats --/
def at_least_two_between (s1 s2 : Seat) : Prop :=
  match s1, s2 with
  | Seat.One, Seat.Four => True
  | Seat.One, Seat.Five => True
  | Seat.One, Seat.Six => True
  | Seat.One, Seat.Seven => True
  | Seat.Two, Seat.Five => True
  | Seat.Two, Seat.Six => True
  | Seat.Two, Seat.Seven => True
  | Seat.Three, Seat.Six => True
  | Seat.Three, Seat.Seven => True
  | Seat.Four, Seat.Seven => True
  | _, _ => False

/-- Compares two seats --/
def seat_lt (s1 s2 : Seat) : Prop :=
  match s1, s2 with
  | Seat.One, Seat.Two => True
  | Seat.One, Seat.Three => True
  | Seat.One, Seat.Four => True
  | Seat.One, Seat.Five => True
  | Seat.One, Seat.Six => True
  | Seat.One, Seat.Seven => True
  | Seat.Two, Seat.Three => True
  | Seat.Two, Seat.Four => True
  | Seat.Two, Seat.Five => True
  | Seat.Two, Seat.Six => True
  | Seat.Two, Seat.Seven => True
  | Seat.Three, Seat.Four => True
  | Seat.Three, Seat.Five => True
  | Seat.Three, Seat.Six => True
  | Seat.Three, Seat.Seven => True
  | Seat.Four, Seat.Five => True
  | Seat.Four, Seat.Six => True
  | Seat.Four, Seat.Seven => True
  | Seat.Five, Seat.Six => True
  | Seat.Five, Seat.Seven => True
  | Seat.Six, Seat.Seven => True
  | _, _ => False

/-- Defines a valid seating arrangement based on the given conditions --/
def valid_arrangement (arr : Arrangement) : Prop :=
  (is_seated arr Friend.Diane Seat.Seven) ∧
  (∃ s1 s2, are_adjacent s1 s2 ∧ is_seated arr Friend.Carla s1 ∧ is_seated arr Friend.Adam s2) ∧
  (∃ s1 s2, is_seated arr Friend.Brian s1 ∧ is_seated arr Friend.Adam s2 ∧ seat_lt s1 s2) ∧
  (∃ s1 s2, is_seated arr Friend.Brian s1 ∧ is_seated arr Friend.Ellie s2 ∧ at_least_two_between s1 s2)

/-- Theorem: In any valid arrangement, Carla must be seated in the middle seat (Seat.Four) --/
theorem carla_in_middle (arr : Arrangement) (h : valid_arrangement arr) : 
  is_seated arr Friend.Carla Seat.Four := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_in_middle_l419_41922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_wins_probability_l419_41926

-- Define the probability of getting a six
def prob_six : ℚ := 1 / 6

-- Define the probability of not getting a six
def prob_not_six : ℚ := 1 - prob_six

-- Define the probability of Carol winning in one cycle
def prob_carol_wins_cycle : ℚ := prob_not_six * prob_not_six * prob_six

-- Define the probability of no one winning in one cycle
def prob_no_win_cycle : ℚ := prob_not_six * prob_not_six * prob_not_six

-- The main theorem
theorem carol_wins_probability :
  (prob_carol_wins_cycle / (1 - prob_no_win_cycle)) = 25 / 91 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_wins_probability_l419_41926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l419_41907

/-- Two perpendicular lines intersecting at (8,6) with y-intercepts summing to -2 form a triangle with area 70 -/
theorem triangle_area : ∀ (m₁ m₂ b₁ b₂ : ℝ),
  -- Two lines intersect at (8,6)
  (8 * m₁ + b₁ = 6) →
  (8 * m₂ + b₂ = 6) →
  -- Lines are perpendicular
  (m₁ * m₂ = -1) →
  -- Sum of y-intercepts is -2
  (b₁ + b₂ = -2) →
  -- Area of triangle is 70
  (1/2 * 8 * |b₁ - b₂| = 70) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l419_41907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_UVCW_l419_41941

-- Define the points
variable (A B C U V W : ℝ × ℝ)

-- Define the triangles
def triangle (P Q R : ℝ × ℝ) : Set (ℝ × ℝ) := {t | t = P ∨ t = Q ∨ t = R}

-- Define isosceles right-angled triangle
def isosceles_right_triangle (P Q R : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1 - R.1)^2 + (P.2 - R.2)^2 ∧
  (P.1 - Q.1) * (P.1 - R.1) + (P.2 - Q.2) * (P.2 - R.2) = 0

-- Define the condition of one triangle being inside another
def triangle_inside (T1 T2 : Set (ℝ × ℝ)) : Prop := T1 ⊆ T2

-- Define the condition of one triangle being outside another
def triangle_outside (T1 T2 : Set (ℝ × ℝ)) : Prop := T1 ∩ T2 ⊆ {P | ∃ Q R, P = Q ∨ P = R}

-- Define parallelogram
def is_parallelogram (P Q R S : ℝ × ℝ) : Prop :=
  (P.1 - Q.1, P.2 - Q.2) = (R.1 - S.1, R.2 - S.2) ∧
  (P.1 - S.1, P.2 - S.2) = (Q.1 - R.1, Q.2 - R.2)

theorem parallelogram_UVCW (h1 : isosceles_right_triangle A U B)
                           (h2 : isosceles_right_triangle C V B)
                           (h3 : isosceles_right_triangle A W C)
                           (h4 : triangle_inside (triangle A U B) (triangle A B C))
                           (h5 : triangle_outside (triangle C V B) (triangle A B C))
                           (h6 : triangle_outside (triangle A W C) (triangle A B C)) :
  is_parallelogram U V C W :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_UVCW_l419_41941
