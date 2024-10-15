import Mathlib

namespace NUMINAMATH_CALUDE_g_range_contains_pi_quarters_l1625_162551

open Real

noncomputable def g (x : ℝ) : ℝ := arctan x + arctan ((x - 1) / (x + 1)) + arctan (1 / x)

theorem g_range_contains_pi_quarters :
  ∃ (x : ℝ), g x = π / 4 ∨ g x = 5 * π / 4 :=
sorry

end NUMINAMATH_CALUDE_g_range_contains_pi_quarters_l1625_162551


namespace NUMINAMATH_CALUDE_society_of_beggars_voting_l1625_162585

/-- The Society of Beggars voting problem -/
theorem society_of_beggars_voting (initial_for : ℕ) (initial_against : ℕ) (no_chair : ℕ) : 
  initial_for = 115 → 
  initial_against = 92 → 
  no_chair = 12 → 
  initial_for + initial_against + no_chair = 207 := by
sorry

end NUMINAMATH_CALUDE_society_of_beggars_voting_l1625_162585


namespace NUMINAMATH_CALUDE_percent_relation_l1625_162561

theorem percent_relation (x y z w : ℝ) 
  (hx : x = 1.2 * y) 
  (hy : y = 0.7 * z) 
  (hw : w = 1.5 * z) : 
  x / w = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1625_162561


namespace NUMINAMATH_CALUDE_exists_polyhedron_with_hidden_vertices_l1625_162509

/-- A polyhedron in 3D space -/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  is_valid : True  -- Additional conditions for a valid polyhedron

/-- Checks if a point is outside a polyhedron -/
def is_outside (P : Polyhedron) (Q : Fin 3 → ℝ) : Prop :=
  Q ∉ P.vertices ∧ ∀ f ∈ P.faces, Q ∉ f

/-- Checks if a line segment intersects the interior of a polyhedron -/
def intersects_interior (P : Polyhedron) (A B : Fin 3 → ℝ) : Prop :=
  ∃ C : Fin 3 → ℝ, C ≠ A ∧ C ≠ B ∧ C ∈ P.vertices ∧ 
    ∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = λ i => (1 - t) * A i + t * B i

/-- The main theorem -/
theorem exists_polyhedron_with_hidden_vertices : 
  ∃ (P : Polyhedron) (Q : Fin 3 → ℝ), 
    is_outside P Q ∧ 
    ∀ V ∈ P.vertices, intersects_interior P Q V :=
  sorry

end NUMINAMATH_CALUDE_exists_polyhedron_with_hidden_vertices_l1625_162509


namespace NUMINAMATH_CALUDE_equal_spaced_roots_value_l1625_162596

theorem equal_spaced_roots_value (k : ℝ) : 
  (∃ a b c d : ℝ, 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (a^2 - 1) * (a^2 - 4) = k ∧
    (b^2 - 1) * (b^2 - 4) = k ∧
    (c^2 - 1) * (c^2 - 4) = k ∧
    (d^2 - 1) * (d^2 - 4) = k ∧
    b - a = c - b ∧ c - b = d - c) →
  k = 7/4 := by
sorry

end NUMINAMATH_CALUDE_equal_spaced_roots_value_l1625_162596


namespace NUMINAMATH_CALUDE_factorization_xy_plus_3x_l1625_162534

theorem factorization_xy_plus_3x (x y : ℝ) : x * y + 3 * x = x * (y + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_plus_3x_l1625_162534


namespace NUMINAMATH_CALUDE_pear_arrangement_l1625_162590

theorem pear_arrangement (n : ℕ) (weights : Fin (2*n+2) → ℝ) :
  ∃ (perm : Fin (2*n+2) ≃ Fin (2*n+2)),
    ∀ i : Fin (2*n+2), |weights (perm i) - weights (perm (i+1))| ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_pear_arrangement_l1625_162590


namespace NUMINAMATH_CALUDE_person_age_puzzle_l1625_162592

theorem person_age_puzzle : ∃ (age : ℕ), 
  (3 * (age + 3) - 3 * (age - 3) = age) ∧ (age = 18) := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l1625_162592


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1625_162506

/-- The magnitude of the complex number z = 1 / (2 + i) is equal to √3 / 3 -/
theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := 1 / (2 + i)
  Complex.abs z = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1625_162506


namespace NUMINAMATH_CALUDE_z_range_l1625_162529

theorem z_range (x y : ℝ) (hx : x ≥ 0) (hxy : y ≥ x) (hsum : 4 * x + 3 * y ≤ 12) :
  let z := (x + 2 * y + 3) / (x + 1)
  2 ≤ z ∧ z ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_z_range_l1625_162529


namespace NUMINAMATH_CALUDE_wizard_collection_value_l1625_162510

def base7ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem wizard_collection_value :
  let crystal_ball := [3, 4, 2, 6]
  let wand := [0, 5, 6, 1]
  let book := [2, 0, 2]
  base7ToBase10 crystal_ball + base7ToBase10 wand + base7ToBase10 book = 2959 := by
  sorry

end NUMINAMATH_CALUDE_wizard_collection_value_l1625_162510


namespace NUMINAMATH_CALUDE_infinitely_many_heinersch_triples_l1625_162505

/-- A positive integer is heinersch if it can be written as the sum of a positive square and positive cube. -/
def IsHeinersch (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^3 ∧ a > 0 ∧ b > 0

/-- The main theorem stating the existence of infinitely many heinersch numbers h such that h-1 and h+1 are also heinersch. -/
theorem infinitely_many_heinersch_triples :
  ∀ N : ℕ, ∃ t : ℕ, t > N ∧
    let h := ((9*t^4)^3 - 1) / 2
    IsHeinersch h ∧
    IsHeinersch (h-1) ∧
    IsHeinersch (h+1) := by
  sorry

/-- Helper lemma for the identity used in the proof -/
lemma cube_identity (t : ℕ) :
  (9*t^3 - 1)^3 + (9*t^4 - 3*t)^3 = (9*t^4)^3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_heinersch_triples_l1625_162505


namespace NUMINAMATH_CALUDE_bookstore_sales_amount_l1625_162545

theorem bookstore_sales_amount (total_calculators : ℕ) (price1 price2 : ℕ) (quantity1 : ℕ) :
  total_calculators = 85 →
  price1 = 15 →
  price2 = 67 →
  quantity1 = 35 →
  (quantity1 * price1 + (total_calculators - quantity1) * price2 : ℕ) = 3875 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_amount_l1625_162545


namespace NUMINAMATH_CALUDE_running_track_area_l1625_162566

theorem running_track_area (r : ℝ) (w : ℝ) (h1 : r = 50) (h2 : w = 3) :
  π * ((r + w)^2 - r^2) = 309 * π := by
  sorry

end NUMINAMATH_CALUDE_running_track_area_l1625_162566


namespace NUMINAMATH_CALUDE_line_equations_l1625_162570

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using slope-intercept form
structure Line2D where
  slope : ℝ
  intercept : ℝ

def isParallel (l1 l2 : Line2D) : Prop :=
  l1.slope = l2.slope

def isPerpendicular (l1 l2 : Line2D) : Prop :=
  l1.slope * l2.slope = -1

def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem line_equations (P : Point2D) (l : Line2D) (given_line : Line2D) :
  P.x = -2 ∧ P.y = 1 ∧ given_line.slope = 1/2 ∧ given_line.intercept = -1/2 →
  (isParallel l given_line → l.slope = 1/2 ∧ l.intercept = 3/2) ∧
  (isPerpendicular l given_line → l.slope = -2 ∧ l.intercept = -5/2) :=
sorry

end NUMINAMATH_CALUDE_line_equations_l1625_162570


namespace NUMINAMATH_CALUDE_gcd_459_357_l1625_162578

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1625_162578


namespace NUMINAMATH_CALUDE_solve_equation_l1625_162544

theorem solve_equation (y : ℝ) : (4 / 7) * (1 / 5) * y - 2 = 10 → y = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1625_162544


namespace NUMINAMATH_CALUDE_larger_integer_problem_l1625_162532

theorem larger_integer_problem (x y : ℕ) (h1 : y = 4 * x) (h2 : (x + 6) / y = 1 / 2) : y = 24 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l1625_162532


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1625_162574

theorem fraction_subtraction : 
  let a := 3 + 6 + 9 + 12
  let b := 2 + 5 + 8 + 11
  (a / b) - (b / a) = 56 / 195 := by
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1625_162574


namespace NUMINAMATH_CALUDE_square_root_of_81_l1625_162573

theorem square_root_of_81 : Real.sqrt 81 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_81_l1625_162573


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1625_162593

theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, x]
  let b : Fin 2 → ℝ := ![2, 2 - x]
  (∃ (k : ℝ), a = k • b) → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1625_162593


namespace NUMINAMATH_CALUDE_exam_thresholds_l1625_162520

theorem exam_thresholds (T : ℝ) 
  (hA : 0.25 * T + 30 = 130) 
  (hB : 0.35 * T - 10 = 130) 
  (hC : 0.40 * T = 160) : 
  (130 : ℝ) = 130 ∧ (160 : ℝ) = 160 := by
  sorry

end NUMINAMATH_CALUDE_exam_thresholds_l1625_162520


namespace NUMINAMATH_CALUDE_tan_equality_225_l1625_162514

theorem tan_equality_225 (m : ℤ) :
  -180 < m ∧ m < 180 →
  (Real.tan (m * π / 180) = Real.tan (225 * π / 180) ↔ m = 45 ∨ m = -135) := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_225_l1625_162514


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1625_162513

/-- Given a geometric sequence {a_n} with sum S_n = 3^(n-1) + t for all n ≥ 1,
    prove that t + a_3 = 17/3 -/
theorem geometric_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (t : ℚ) 
  (h1 : ∀ n : ℕ, n ≥ 1 → S n = 3^(n-1) + t)
  (h2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1))
  (h3 : ∀ n m : ℕ, n ≥ 1 → m ≥ 1 → a (n+1) / a n = a (m+1) / a m) :
  t + a 3 = 17/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1625_162513


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1625_162597

/-- An arithmetic sequence with given third and seventeenth terms -/
structure ArithmeticSequence where
  a₃ : ℚ
  a₁₇ : ℚ
  is_arithmetic : ∃ d, a₁₇ = a₃ + 14 * d

/-- The properties we want to prove about this arithmetic sequence -/
def ArithmeticSequenceProperties (seq : ArithmeticSequence) : Prop :=
  ∃ (a₁₀ : ℚ),
    (seq.a₃ = 11/15) ∧
    (seq.a₁₇ = 2/3) ∧
    (a₁₀ = 7/10) ∧
    (seq.a₃ + a₁₀ + seq.a₁₇ = 21/10)

/-- The main theorem stating that our arithmetic sequence has the desired properties -/
theorem arithmetic_sequence_theorem (seq : ArithmeticSequence) 
    (h₁ : seq.a₃ = 11/15) (h₂ : seq.a₁₇ = 2/3) : 
    ArithmeticSequenceProperties seq := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1625_162597


namespace NUMINAMATH_CALUDE_f_of_2_equals_4_l1625_162569

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Theorem statement
theorem f_of_2_equals_4 : f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_4_l1625_162569


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1625_162549

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = Set.Ioo (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1625_162549


namespace NUMINAMATH_CALUDE_division_remainder_l1625_162541

theorem division_remainder : ∃ q : ℕ, 1234567 = 321 * q + 264 ∧ 264 < 321 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1625_162541


namespace NUMINAMATH_CALUDE_basketball_players_l1625_162523

theorem basketball_players (cricket : ℕ) (both : ℕ) (total : ℕ)
  (h1 : cricket = 8)
  (h2 : both = 6)
  (h3 : total = 11)
  (h4 : total = cricket + basketball - both) :
  basketball = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_players_l1625_162523


namespace NUMINAMATH_CALUDE_garden_fence_area_l1625_162591

/-- Given an L-shaped fence and two straight fence sections of 13m and 14m,
    prove that it's possible to create a rectangular area of at least 200 m². -/
theorem garden_fence_area (length : ℝ) (width : ℝ) : 
  length = 13 → width = 17 → length * width ≥ 200 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_area_l1625_162591


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_l1625_162564

theorem minimum_value_of_expression (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_l1625_162564


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1625_162562

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 4) : y = 4 - 3 * x := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1625_162562


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1625_162555

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : 
  (a ≥ 2 → a ≥ 1) ∧ ¬(a ≥ 1 → a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1625_162555


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1625_162563

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  ((x^2 + a) / x^2 + (x^2 - a) / x^2) / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1625_162563


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l1625_162557

theorem floor_plus_self_eq_seventeen_fourths (x : ℝ) : 
  ⌊x⌋ + x = 17/4 → x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l1625_162557


namespace NUMINAMATH_CALUDE_nikanor_lost_second_match_l1625_162526

/-- Represents a player in the knock-out table tennis game -/
inductive Player : Type
| Nikanor : Player
| Philemon : Player
| Agathon : Player

/-- Represents the state of the game after each match -/
structure GameState :=
  (matches_played : Nat)
  (nikanor_matches : Nat)
  (philemon_matches : Nat)
  (agathon_matches : Nat)
  (last_loser : Player)

/-- The rules of the knock-out table tennis game -/
def game_rules (state : GameState) : Prop :=
  state.matches_played = (state.nikanor_matches + state.philemon_matches + state.agathon_matches) / 2 ∧
  state.nikanor_matches + state.philemon_matches + state.agathon_matches = state.matches_played * 2 ∧
  state.nikanor_matches ≤ state.matches_played ∧
  state.philemon_matches ≤ state.matches_played ∧
  state.agathon_matches ≤ state.matches_played

/-- The final state of the game -/
def final_state : GameState :=
  { matches_played := 21
  , nikanor_matches := 10
  , philemon_matches := 15
  , agathon_matches := 17
  , last_loser := Player.Nikanor }

/-- Theorem stating that Nikanor lost the second match -/
theorem nikanor_lost_second_match :
  game_rules final_state →
  final_state.last_loser = Player.Nikanor :=
by sorry

end NUMINAMATH_CALUDE_nikanor_lost_second_match_l1625_162526


namespace NUMINAMATH_CALUDE_fraction_division_l1625_162571

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l1625_162571


namespace NUMINAMATH_CALUDE_intersection_on_ellipse_l1625_162535

/-- Ellipse C with given properties -/
structure EllipseC where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The eccentricity of ellipse C is √3/2 -/
axiom eccentricity (C : EllipseC) : (Real.sqrt (C.a^2 - C.b^2)) / C.a = Real.sqrt 3 / 2

/-- A circle centered at the origin with diameter equal to the minor axis of C
    is tangent to the line x - y + 2 = 0 -/
axiom circle_tangent (C : EllipseC) : C.b = Real.sqrt 2

/-- Point on ellipse C -/
def on_ellipse (C : EllipseC) (x y : ℝ) : Prop :=
  x^2 / C.a^2 + y^2 / C.b^2 = 1

/-- Theorem: If M and N are symmetric points on C, and T is the intersection of PM and QN,
    then T lies on the ellipse C -/
theorem intersection_on_ellipse (C : EllipseC) (x₀ y₀ x y : ℝ) :
  on_ellipse C x₀ y₀ →
  on_ellipse C (-x₀) y₀ →
  x = 2 * x₀ * (y - 1) →
  y = (3 * y₀ - 4) / (2 * y₀ - 3) →
  on_ellipse C x y := by
  sorry

end NUMINAMATH_CALUDE_intersection_on_ellipse_l1625_162535


namespace NUMINAMATH_CALUDE_jack_evening_emails_l1625_162518

/-- The number of emails Jack received in a day -/
def total_emails : ℕ := 10

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := total_emails - (afternoon_emails + morning_emails)

theorem jack_evening_emails : evening_emails = 1 := by
  sorry

end NUMINAMATH_CALUDE_jack_evening_emails_l1625_162518


namespace NUMINAMATH_CALUDE_ship_passengers_l1625_162524

theorem ship_passengers (total : ℝ) (round_trip_with_car : ℝ) 
  (h1 : 0 < total) 
  (h2 : 0 ≤ round_trip_with_car) 
  (h3 : round_trip_with_car ≤ total) 
  (h4 : round_trip_with_car / total = 0.2 * (round_trip_with_car / 0.2) / total) : 
  round_trip_with_car / 0.2 = total := by
sorry

end NUMINAMATH_CALUDE_ship_passengers_l1625_162524


namespace NUMINAMATH_CALUDE_train_speed_problem_l1625_162552

/-- Proves that given two trains of equal length 37.5 meters, where the faster train travels
    at 46 km/hr and passes the slower train in 27 seconds, the speed of the slower train is 36 km/hr. -/
theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) 
    (h1 : train_length = 37.5)
    (h2 : faster_speed = 46)
    (h3 : passing_time = 27) :
  ∃ slower_speed : ℝ, 
    slower_speed > 0 ∧ 
    slower_speed < faster_speed ∧
    2 * train_length = (faster_speed - slower_speed) * (5/18) * passing_time ∧
    slower_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1625_162552


namespace NUMINAMATH_CALUDE_a_in_A_l1625_162542

def A : Set ℝ := {x | x ≥ 2 * Real.sqrt 2}

theorem a_in_A : 3 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_a_in_A_l1625_162542


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l1625_162556

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 100) : 
  x^2 + y^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l1625_162556


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_l1625_162577

def S : Finset Int := {-8, 2, -5, 17, -3}

theorem smallest_sum_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z → a + b + c ≤ x + y + z) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = -16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_l1625_162577


namespace NUMINAMATH_CALUDE_cos_beta_equals_cos_alpha_l1625_162504

-- Define the angles α and β
variable (α β : Real)

-- Define the conditions
axiom vertices_at_origin : True  -- This condition is implicit in the angle definitions
axiom initial_sides_on_x_axis : True  -- This condition is implicit in the angle definitions
axiom terminal_sides_symmetric : β = 2 * Real.pi - α
axiom cos_alpha : Real.cos α = 2/3

-- Theorem to prove
theorem cos_beta_equals_cos_alpha : Real.cos β = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_beta_equals_cos_alpha_l1625_162504


namespace NUMINAMATH_CALUDE_existence_of_a_and_b_l1625_162580

/-- The number of positive divisors of a natural number -/
noncomputable def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Main theorem -/
theorem existence_of_a_and_b (k l c : ℕ+) :
  ∃ (a b : ℕ+),
    (b - a = c * Nat.gcd a b) ∧
    (tau a * tau (b / Nat.gcd a b) * l = tau b * tau (a / Nat.gcd a b) * k) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_and_b_l1625_162580


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_f_minus_quadratic_range_of_m_l1625_162579

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 : 
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem for the maximum value of f(x) - x^2 + x
theorem max_value_f_minus_quadratic :
  ∃ (x : ℝ), ∀ (y : ℝ), f y - y^2 + y ≤ f x - x^2 + x ∧ f x - x^2 + x = 5/4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ (x : ℝ), f x ≥ x^2 - x + m) ↔ m ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_f_minus_quadratic_range_of_m_l1625_162579


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1625_162568

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1625_162568


namespace NUMINAMATH_CALUDE_distribute_5_3_l1625_162598

/-- The number of ways to distribute n indistinguishable objects into k distinct containers,
    with each container containing at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l1625_162598


namespace NUMINAMATH_CALUDE_mixed_grains_in_rice_l1625_162530

theorem mixed_grains_in_rice (total_stones : ℕ) (sample_size : ℕ) (mixed_in_sample : ℕ) :
  total_stones = 1536 →
  sample_size = 256 →
  mixed_in_sample = 18 →
  (total_stones * mixed_in_sample) / sample_size = 108 :=
by
  sorry

#check mixed_grains_in_rice

end NUMINAMATH_CALUDE_mixed_grains_in_rice_l1625_162530


namespace NUMINAMATH_CALUDE_ceiling_of_negative_three_point_seven_l1625_162586

theorem ceiling_of_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_three_point_seven_l1625_162586


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1625_162558

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (k + 2) * x^2 - 2 * (k - 1) * x + k + 1 = 0) ↔ (k = -1/5 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1625_162558


namespace NUMINAMATH_CALUDE_parallelogram_area_and_scaling_l1625_162531

theorem parallelogram_area_and_scaling :
  let base : ℝ := 6
  let height : ℝ := 20
  let area := base * height
  let scaled_base := 3 * base
  let scaled_height := 3 * height
  let scaled_area := scaled_base * scaled_height
  (area = 120) ∧ 
  (scaled_area = 9 * area) ∧ 
  (scaled_area = 1080) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_and_scaling_l1625_162531


namespace NUMINAMATH_CALUDE_blocks_in_prism_l1625_162500

/-- The number of unit blocks needed to fill a rectangular prism -/
def num_blocks (length width height : ℕ) : ℕ := length * width * height

/-- The dimensions of the rectangular prism -/
def prism_length : ℕ := 4
def prism_width : ℕ := 3
def prism_height : ℕ := 3

/-- Theorem: The number of 1 cm³ blocks needed to fill the given rectangular prism is 36 -/
theorem blocks_in_prism : 
  num_blocks prism_length prism_width prism_height = 36 := by
  sorry

end NUMINAMATH_CALUDE_blocks_in_prism_l1625_162500


namespace NUMINAMATH_CALUDE_prob_a_b_not_same_class_l1625_162553

/-- The number of students to be distributed -/
def num_students : ℕ := 4

/-- The number of classes -/
def num_classes : ℕ := 3

/-- The probability that students A and B are not in the same class -/
def prob_not_same_class : ℚ := 5/6

/-- The total number of ways to distribute students into classes -/
def total_distributions : ℕ := num_students.choose 2 * num_classes.factorial

/-- The number of distributions where A and B are in different classes -/
def favorable_distributions : ℕ := total_distributions - num_classes.factorial

theorem prob_a_b_not_same_class :
  (favorable_distributions : ℚ) / total_distributions = prob_not_same_class :=
sorry

end NUMINAMATH_CALUDE_prob_a_b_not_same_class_l1625_162553


namespace NUMINAMATH_CALUDE_shelves_used_l1625_162511

theorem shelves_used (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 130 → books_sold = 47 → books_per_shelf = 15 →
  (initial_stock - books_sold + books_per_shelf - 1) / books_per_shelf = 6 := by
  sorry

#eval (130 - 47 + 15 - 1) / 15

end NUMINAMATH_CALUDE_shelves_used_l1625_162511


namespace NUMINAMATH_CALUDE_hyperbola_decreasing_condition_l1625_162519

/-- For a hyperbola y = (1-m)/x, y decreases as x increases when x > 0 if and only if m < 1 -/
theorem hyperbola_decreasing_condition (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → (1-m)/x₁ > (1-m)/x₂) ↔ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_decreasing_condition_l1625_162519


namespace NUMINAMATH_CALUDE_multiply_add_theorem_l1625_162548

theorem multiply_add_theorem : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_theorem_l1625_162548


namespace NUMINAMATH_CALUDE_fraction_simplification_l1625_162501

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1625_162501


namespace NUMINAMATH_CALUDE_gratuity_calculation_l1625_162575

-- Define the given values
def total_bill : ℝ := 140
def tax_rate : ℝ := 0.10
def striploin_cost : ℝ := 80
def wine_cost : ℝ := 10

-- Define the theorem
theorem gratuity_calculation :
  let pre_tax_total := striploin_cost + wine_cost
  let tax_amount := pre_tax_total * tax_rate
  let bill_with_tax := pre_tax_total + tax_amount
  let gratuity := total_bill - bill_with_tax
  gratuity = 41 := by sorry

end NUMINAMATH_CALUDE_gratuity_calculation_l1625_162575


namespace NUMINAMATH_CALUDE_unique_root_in_interval_l1625_162599

open Complex

theorem unique_root_in_interval : ∃! x : ℝ, 0 ≤ x ∧ x < 2 * π ∧
  2 + exp (I * x) - 2 * exp (2 * I * x) + exp (3 * I * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_in_interval_l1625_162599


namespace NUMINAMATH_CALUDE_team_a_games_l1625_162543

theorem team_a_games (a : ℕ) : 
  (2 : ℚ) / 3 * a = (5 : ℚ) / 8 * (a + 14) - 7 → a = 42 := by
  sorry

end NUMINAMATH_CALUDE_team_a_games_l1625_162543


namespace NUMINAMATH_CALUDE_coefficient_sum_l1625_162587

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Define the coefficients a, b, c, d
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry
def d : ℝ := sorry

-- State the theorem
theorem coefficient_sum :
  (∀ x, f (x + 3) = 3 * x^2 + 7 * x + 4) →
  (∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  a + b + c + d = -7 := by sorry

end NUMINAMATH_CALUDE_coefficient_sum_l1625_162587


namespace NUMINAMATH_CALUDE_point_A_coordinates_l1625_162595

/-- A translation that moves any point (a,b) to (a+2,b-6) -/
def translation (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2, p.2 - 6)

/-- The point A₁ after translation -/
def A1 : ℝ × ℝ := (4, -3)

theorem point_A_coordinates :
  ∃ A : ℝ × ℝ, translation A = A1 ∧ A = (2, 3) := by sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l1625_162595


namespace NUMINAMATH_CALUDE_exponent_sum_l1625_162581

theorem exponent_sum (m n : ℕ) (h : 2^m * 2^n = 16) : m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l1625_162581


namespace NUMINAMATH_CALUDE_shoes_difference_l1625_162533

/-- Represents the number of shoes tried on at each store --/
structure ShoesTried where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The total number of shoes tried on across all stores --/
def totalShoesTried (s : ShoesTried) : ℕ :=
  s.first + s.second + s.third + s.fourth

/-- The conditions from the problem --/
def problemConditions (s : ShoesTried) : Prop :=
  s.first = 7 ∧
  s.third = 0 ∧
  s.fourth = 2 * (s.first + s.second + s.third) ∧
  totalShoesTried s = 48

/-- The theorem to prove --/
theorem shoes_difference (s : ShoesTried) 
  (h : problemConditions s) : s.second - s.first = 2 := by
  sorry


end NUMINAMATH_CALUDE_shoes_difference_l1625_162533


namespace NUMINAMATH_CALUDE_divisible_by_18_sqrt_between_30_and_30_5_l1625_162583

theorem divisible_by_18_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), n > 0 ∧ n % 18 = 0 ∧ 30 < Real.sqrt n ∧ Real.sqrt n < 30.5 ∧
  (n = 900 ∨ n = 918) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_18_sqrt_between_30_and_30_5_l1625_162583


namespace NUMINAMATH_CALUDE_soccer_stars_games_l1625_162538

theorem soccer_stars_games (wins losses draws : ℕ) 
  (h1 : wins = 14)
  (h2 : losses = 2)
  (h3 : 3 * wins + draws = 46) :
  wins + losses + draws = 20 := by
sorry

end NUMINAMATH_CALUDE_soccer_stars_games_l1625_162538


namespace NUMINAMATH_CALUDE_unique_increasing_function_theorem_l1625_162507

def IncreasingFunction (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, x ≤ y → f x ≤ f y

theorem unique_increasing_function_theorem (f : ℕ+ → ℕ+) 
  (h_increasing : IncreasingFunction f)
  (h_inequality : ∀ x : ℕ+, (f x) * (f (f x)) ≤ x^2) :
  ∀ x : ℕ+, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_increasing_function_theorem_l1625_162507


namespace NUMINAMATH_CALUDE_ship_cargo_theorem_l1625_162508

def initial_cargo : ℕ := 5973
def loaded_cargo : ℕ := 8723

theorem ship_cargo_theorem : 
  initial_cargo + loaded_cargo = 14696 := by sorry

end NUMINAMATH_CALUDE_ship_cargo_theorem_l1625_162508


namespace NUMINAMATH_CALUDE_fraction_difference_l1625_162525

theorem fraction_difference (n : ℝ) : 
  let simplified := (n * (n + 3)) / (n^2 + 3*n + 1)
  (n^2 + 3*n + 1) - (n^2 + 3*n) = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_l1625_162525


namespace NUMINAMATH_CALUDE_expression_evaluation_l1625_162572

theorem expression_evaluation : 2 * 0 + 1 - 9 = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1625_162572


namespace NUMINAMATH_CALUDE_jeremy_tylenol_duration_l1625_162528

/-- Calculates the duration in days for which Jeremy takes Tylenol -/
def tylenol_duration (dose_mg : ℕ) (dose_interval_hours : ℕ) (total_pills : ℕ) (mg_per_pill : ℕ) : ℕ :=
  let total_mg := total_pills * mg_per_pill
  let total_doses := total_mg / dose_mg
  let total_hours := total_doses * dose_interval_hours
  total_hours / 24

/-- Theorem stating that Jeremy takes Tylenol for 14 days -/
theorem jeremy_tylenol_duration :
  tylenol_duration 1000 6 112 500 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_tylenol_duration_l1625_162528


namespace NUMINAMATH_CALUDE_mean_of_solutions_l1625_162515

-- Define the polynomial
def f (x : ℝ) := x^3 + 5*x^2 - 14*x

-- Define the set of solutions
def solutions := {x : ℝ | f x = 0}

-- State the theorem
theorem mean_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solutions ∧ s.card = 3 ∧ (s.sum id) / s.card = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_solutions_l1625_162515


namespace NUMINAMATH_CALUDE_percent_equation_solution_l1625_162502

theorem percent_equation_solution :
  ∃ x : ℝ, (0.75 / 100) * x = 0.06 ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_percent_equation_solution_l1625_162502


namespace NUMINAMATH_CALUDE_laptop_price_proof_l1625_162522

/-- The original sticker price of the laptop -/
def original_price : ℝ := 500

/-- The price at Store A after discount and rebate -/
def store_a_price (x : ℝ) : ℝ := 0.82 * x - 100

/-- The price at Store B after discount -/
def store_b_price (x : ℝ) : ℝ := 0.7 * x

/-- Theorem stating that the original price satisfies the given conditions -/
theorem laptop_price_proof :
  store_a_price original_price = store_b_price original_price - 40 := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l1625_162522


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1625_162550

theorem consecutive_integers_average (c d : ℝ) : 
  (d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5)) / 6) →
  ((d-2) + (d-1) + d + (d+1) + (d+2) + (d+3)) / 6 = c + 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1625_162550


namespace NUMINAMATH_CALUDE_a_annual_income_l1625_162584

/-- Prove that A's annual income is Rs. 504000 -/
theorem a_annual_income (a_monthly b_monthly c_monthly : ℕ) : 
  (a_monthly : ℚ) / b_monthly = 5 / 2 →
  b_monthly = (112 * c_monthly) / 100 →
  c_monthly = 15000 →
  12 * a_monthly = 504000 := by
  sorry

end NUMINAMATH_CALUDE_a_annual_income_l1625_162584


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1625_162576

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x > 2}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1625_162576


namespace NUMINAMATH_CALUDE_remainder_of_sum_squares_plus_20_l1625_162517

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem remainder_of_sum_squares_plus_20 : 
  (sum_of_squares 15 + 20) % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_squares_plus_20_l1625_162517


namespace NUMINAMATH_CALUDE_min_value_of_z_l1625_162594

-- Define the variables and the objective function
variables (x y : ℝ)
def z (x y : ℝ) : ℝ := 2 * x + y

-- State the theorem
theorem min_value_of_z (hx : x ≥ 1) (hxy : x + y ≤ 3) (hxy2 : x - 2 * y - 3 ≤ 0) :
  ∃ (x₀ y₀ : ℝ), x₀ ≥ 1 ∧ x₀ + y₀ ≤ 3 ∧ x₀ - 2 * y₀ - 3 ≤ 0 ∧
  ∀ (x y : ℝ), x ≥ 1 → x + y ≤ 3 → x - 2 * y - 3 ≤ 0 → z x₀ y₀ ≤ z x y ∧ z x₀ y₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1625_162594


namespace NUMINAMATH_CALUDE_partners_capital_time_l1625_162536

/-- A proof that under given business conditions, A's capital was used for 15 months -/
theorem partners_capital_time (C P : ℝ) : 
  C > 0 → P > 0 →
  let a_capital := C / 4
  let b_capital := 3 * C / 4
  let b_time := 10
  let b_profit := 2 * P / 3
  let a_profit := P / 3
  ∃ (a_time : ℝ),
    a_time * a_capital / (b_time * b_capital) = a_profit / b_profit ∧
    a_time = 15 :=
by sorry

end NUMINAMATH_CALUDE_partners_capital_time_l1625_162536


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l1625_162521

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 1)^2 = 4
def equation2 (x : ℝ) : Prop := x^2 + 4*x - 5 = 0

-- Theorem for equation 1
theorem equation1_solutions :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ equation1 x1 ∧ equation1 x2 ∧ 
  (∀ x : ℝ, equation1 x → x = x1 ∨ x = x2) ∧
  x1 = 1 ∧ x2 = -3 :=
sorry

-- Theorem for equation 2
theorem equation2_solutions :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ equation2 x1 ∧ equation2 x2 ∧ 
  (∀ x : ℝ, equation2 x → x = x1 ∨ x = x2) ∧
  x1 = -5 ∧ x2 = 1 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l1625_162521


namespace NUMINAMATH_CALUDE_dilation_transforms_line_l1625_162559

-- Define the original line
def original_line (x y : ℝ) : Prop := x + y = 1

-- Define the transformed line
def transformed_line (x y : ℝ) : Prop := 2*x + 3*y = 6

-- Define the dilation transformation
def dilation (x y : ℝ) : ℝ × ℝ := (2*x, 3*y)

-- Theorem statement
theorem dilation_transforms_line :
  ∀ x y : ℝ, original_line x y → transformed_line (dilation x y).1 (dilation x y).2 := by
  sorry

end NUMINAMATH_CALUDE_dilation_transforms_line_l1625_162559


namespace NUMINAMATH_CALUDE_min_freight_cost_l1625_162540

/-- Represents the freight problem with given parameters -/
structure FreightProblem where
  totalOre : ℕ
  truckCapacity1 : ℕ
  truckCapacity2 : ℕ
  truckCost1 : ℕ
  truckCost2 : ℕ

/-- Calculates the total cost for a given number of trucks -/
def totalCost (p : FreightProblem) (trucks1 : ℕ) (trucks2 : ℕ) : ℕ :=
  trucks1 * p.truckCost1 + trucks2 * p.truckCost2

/-- Checks if a combination of trucks can transport the required amount of ore -/
def isValidCombination (p : FreightProblem) (trucks1 : ℕ) (trucks2 : ℕ) : Prop :=
  trucks1 * p.truckCapacity1 + trucks2 * p.truckCapacity2 ≥ p.totalOre

/-- The main theorem stating that 685 is the minimum freight cost -/
theorem min_freight_cost (p : FreightProblem) 
  (h1 : p.totalOre = 73)
  (h2 : p.truckCapacity1 = 7)
  (h3 : p.truckCapacity2 = 5)
  (h4 : p.truckCost1 = 65)
  (h5 : p.truckCost2 = 50) :
  (∀ trucks1 trucks2 : ℕ, isValidCombination p trucks1 trucks2 → totalCost p trucks1 trucks2 ≥ 685) ∧ 
  (∃ trucks1 trucks2 : ℕ, isValidCombination p trucks1 trucks2 ∧ totalCost p trucks1 trucks2 = 685) :=
sorry


end NUMINAMATH_CALUDE_min_freight_cost_l1625_162540


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1625_162582

def U : Set Nat := {0, 2, 4, 6, 8, 10}
def A : Set Nat := {2, 4, 6}
def B : Set Nat := {1}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 1, 8, 10} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1625_162582


namespace NUMINAMATH_CALUDE_greatest_multiple_24_unique_digits_remainder_l1625_162527

/-- 
M is the greatest integer multiple of 24 with no two digits being the same.
-/
def M : ℕ := sorry

/-- 
A function that checks if a natural number has all unique digits.
-/
def has_unique_digits (n : ℕ) : Prop := sorry

theorem greatest_multiple_24_unique_digits_remainder (h1 : M % 24 = 0) 
  (h2 : has_unique_digits M) 
  (h3 : ∀ k : ℕ, k > M → k % 24 = 0 → ¬(has_unique_digits k)) : 
  M % 1000 = 720 := by sorry

end NUMINAMATH_CALUDE_greatest_multiple_24_unique_digits_remainder_l1625_162527


namespace NUMINAMATH_CALUDE_probability_club_after_removal_l1625_162567

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)

/-- Represents the deck after removing spade cards -/
structure ModifiedDeck :=
  (remaining_cards : ℕ)
  (club_cards : ℕ)

/-- The probability of drawing a club card from the modified deck -/
def probability_club (d : ModifiedDeck) : ℚ :=
  d.club_cards / d.remaining_cards

theorem probability_club_after_removal (standard_deck : Deck) (modified_deck : ModifiedDeck) :
  standard_deck.total_cards = 52 →
  standard_deck.ranks = 13 →
  standard_deck.suits = 4 →
  modified_deck.remaining_cards = 48 →
  modified_deck.club_cards = 13 →
  probability_club modified_deck = 13 / 48 := by
  sorry

#eval (13 : ℚ) / 48

end NUMINAMATH_CALUDE_probability_club_after_removal_l1625_162567


namespace NUMINAMATH_CALUDE_solution_set_part1_a_range_part2_l1625_162546

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + |2*x - 4| + a

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | (f x (-3)) > x^2 + |x|} = {x : ℝ | x < 1/3 ∨ x > 7} := by sorry

-- Part 2
theorem a_range_part2 :
  (∀ x : ℝ, f x a ≥ 0) → a ≥ -3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_a_range_part2_l1625_162546


namespace NUMINAMATH_CALUDE_largest_non_representable_integer_l1625_162516

theorem largest_non_representable_integer
  (a b c : ℕ+) 
  (coprime_ab : Nat.Coprime a b)
  (coprime_bc : Nat.Coprime b c)
  (coprime_ca : Nat.Coprime c a) :
  ¬ ∃ (x y z : ℕ), 2 * a * b * c - a * b - b * c - c * a = x * b * c + y * c * a + z * a * b :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_integer_l1625_162516


namespace NUMINAMATH_CALUDE_not_prime_n_fourth_plus_four_to_n_l1625_162560

theorem not_prime_n_fourth_plus_four_to_n (n : ℕ) (h : n > 1) :
  ¬ Nat.Prime (n^4 + 4^n) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n_fourth_plus_four_to_n_l1625_162560


namespace NUMINAMATH_CALUDE_total_marbles_is_193_l1625_162512

/-- The number of marbles in the jar when Ben, Leo, and Tim combine their marbles. -/
def totalMarbles : ℕ :=
  let benMarbles : ℕ := 56
  let leoMarbles : ℕ := benMarbles + 20
  let timMarbles : ℕ := leoMarbles - 15
  benMarbles + leoMarbles + timMarbles

/-- Theorem stating that the total number of marbles in the jar is 193. -/
theorem total_marbles_is_193 : totalMarbles = 193 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_193_l1625_162512


namespace NUMINAMATH_CALUDE_jellybean_theorem_l1625_162537

/-- Calculates the final number of jellybeans in a jar after a series of actions. -/
def final_jellybean_count (initial : ℕ) (samantha_took : ℕ) (shelby_ate : ℕ) : ℕ :=
  let scarlett_took := 2 * shelby_ate
  let scarlett_returned := (scarlett_took * 2) / 5  -- 40% rounded down
  let shannon_refilled := (samantha_took + shelby_ate) / 2
  initial - samantha_took - shelby_ate + scarlett_returned + shannon_refilled

/-- Theorem stating that given the initial conditions, the final number of jellybeans is 81. -/
theorem jellybean_theorem : final_jellybean_count 90 24 12 = 81 := by
  sorry

#eval final_jellybean_count 90 24 12

end NUMINAMATH_CALUDE_jellybean_theorem_l1625_162537


namespace NUMINAMATH_CALUDE_student_number_problem_l1625_162547

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 112 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1625_162547


namespace NUMINAMATH_CALUDE_botanist_flower_distribution_l1625_162589

theorem botanist_flower_distribution (total_flowers : ℕ) (num_bouquets : ℕ) (additional_flowers : ℕ) : 
  total_flowers = 601 →
  num_bouquets = 8 →
  additional_flowers = 7 →
  (total_flowers + additional_flowers) % num_bouquets = 0 ∧
  (total_flowers + additional_flowers - 1) % num_bouquets ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_botanist_flower_distribution_l1625_162589


namespace NUMINAMATH_CALUDE_correct_percentage_calculation_l1625_162503

theorem correct_percentage_calculation (x : ℝ) (h : x > 0) :
  let total_problems := 7 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  (correct_problems / total_problems) * 100 = (5 / 7) * 100 := by
sorry

end NUMINAMATH_CALUDE_correct_percentage_calculation_l1625_162503


namespace NUMINAMATH_CALUDE_min_value_and_nonexistence_l1625_162588

theorem min_value_and_nonexistence (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = Real.sqrt (a*b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = Real.sqrt (x*y) → x^3 + y^3 ≥ 4 * Real.sqrt 2) ∧ 
  (¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = Real.sqrt (x*y) ∧ 2*x + 3*y = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_nonexistence_l1625_162588


namespace NUMINAMATH_CALUDE_probability_three_same_group_l1625_162565

/-- The number of students in the school -/
def total_students : ℕ := 600

/-- The number of lunch groups -/
def num_groups : ℕ := 3

/-- Assumption that the groups are of equal size -/
axiom groups_equal_size : total_students % num_groups = 0

/-- The probability of a student being assigned to a specific group -/
def prob_one_group : ℚ := 1 / num_groups

/-- The probability of three specific students being assigned to the same lunch group -/
def prob_three_same_group : ℚ := prob_one_group * prob_one_group

theorem probability_three_same_group :
  prob_three_same_group = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_three_same_group_l1625_162565


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1625_162554

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 25) →
  (a 2 + a 8 = 10) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1625_162554


namespace NUMINAMATH_CALUDE_first_movie_duration_l1625_162539

/-- Represents the duration of a movie marathon with three movies --/
structure MovieMarathon where
  first_movie : ℝ
  second_movie : ℝ
  third_movie : ℝ

/-- Defines the conditions of the movie marathon --/
def valid_marathon (m : MovieMarathon) : Prop :=
  m.second_movie = 1.5 * m.first_movie ∧
  m.third_movie = m.first_movie + m.second_movie - 1 ∧
  m.first_movie + m.second_movie + m.third_movie = 9

theorem first_movie_duration :
  ∀ m : MovieMarathon, valid_marathon m → m.first_movie = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_movie_duration_l1625_162539
