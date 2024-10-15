import Mathlib

namespace NUMINAMATH_CALUDE_five_integer_chords_l2316_231613

/-- A circle with a point P inside --/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- The number of chords with integer lengths passing through P --/
def numIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The theorem statement --/
theorem five_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 17) 
  (h2 : c.distanceFromCenter = 8) : 
  numIntegerChords c = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_integer_chords_l2316_231613


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2316_231641

theorem cubic_equation_solution :
  ∃ x : ℝ, x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2316_231641


namespace NUMINAMATH_CALUDE_expand_expression_l2316_231618

theorem expand_expression (x : ℝ) : 6 * (x - 3) * (x^2 + 4*x + 16) = 6*x^3 + 6*x^2 + 24*x - 288 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2316_231618


namespace NUMINAMATH_CALUDE_least_number_divisible_by_seven_with_remainder_one_l2316_231666

theorem least_number_divisible_by_seven_with_remainder_one : ∃ n : ℕ, 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 6 → n % k = 1) ∧ 
  n % 7 = 0 ∧
  (∀ m : ℕ, m < n → ¬(∀ k : ℕ, 2 ≤ k ∧ k ≤ 6 → m % k = 1) ∨ m % 7 ≠ 0) ∧
  n = 301 :=
by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_seven_with_remainder_one_l2316_231666


namespace NUMINAMATH_CALUDE_abc_inequality_l2316_231633

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2316_231633


namespace NUMINAMATH_CALUDE_unique_integer_square_less_than_double_l2316_231691

theorem unique_integer_square_less_than_double :
  ∃! x : ℤ, x^2 < 2*x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_square_less_than_double_l2316_231691


namespace NUMINAMATH_CALUDE_square_side_sum_l2316_231673

theorem square_side_sum (b d : ℕ) : 
  15^2 = b^2 + 10^2 + d^2 → (b + d = 13 ∨ b + d = 15) :=
by sorry

end NUMINAMATH_CALUDE_square_side_sum_l2316_231673


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_system_of_equations_solution_l2316_231664

-- Problem 1
theorem sqrt_sum_equality : |Real.sqrt 2 - Real.sqrt 5| + 2 * Real.sqrt 2 = Real.sqrt 5 + Real.sqrt 2 := by
  sorry

-- Problem 2
theorem system_of_equations_solution :
  ∃ (x y : ℝ), 4 * x + y = 15 ∧ 3 * x - 2 * y = 3 ∧ x = 3 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_system_of_equations_solution_l2316_231664


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2316_231610

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length 
  (train_speed : Real) 
  (bridge_crossing_time : Real) 
  (bridge_length : Real) : Real :=
  let total_distance := train_speed * (1000 / 3600) * bridge_crossing_time
  total_distance - bridge_length

/-- Proves that a train traveling at 45 km/hr that crosses a 250 m bridge in 30 seconds has a length of 125 m. -/
theorem train_length_proof :
  train_length 45 30 250 = 125 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2316_231610


namespace NUMINAMATH_CALUDE_student_count_theorem_l2316_231662

def valid_student_count (n : ℕ) : Prop :=
  n < 50 ∧ n % 6 = 5 ∧ n % 3 = 2

theorem student_count_theorem : 
  {n : ℕ | valid_student_count n} = {5, 11, 17, 23, 29, 35, 41, 47} :=
sorry

end NUMINAMATH_CALUDE_student_count_theorem_l2316_231662


namespace NUMINAMATH_CALUDE_function_monotonicity_implies_a_value_l2316_231632

/-- A function f(x) = x^2 - ax that is decreasing on (-∞, 2] and increasing on (2, +∞) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 - a*x

/-- The function f is decreasing on (-∞, 2] -/
def decreasing_on_left (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 2 → f a x > f a y

/-- The function f is increasing on (2, +∞) -/
def increasing_on_right (a : ℝ) : Prop :=
  ∀ x y, 2 < x → x < y → f a x < f a y

/-- If f(x) = x^2 - ax is decreasing on (-∞, 2] and increasing on (2, +∞), then a = 4 -/
theorem function_monotonicity_implies_a_value (a : ℝ) :
  decreasing_on_left a → increasing_on_right a → a = 4 := by sorry

end NUMINAMATH_CALUDE_function_monotonicity_implies_a_value_l2316_231632


namespace NUMINAMATH_CALUDE_pascals_cycling_trip_l2316_231645

theorem pascals_cycling_trip (current_speed : ℝ) (speed_reduction : ℝ) (time_difference : ℝ) :
  current_speed = 8 →
  speed_reduction = 4 →
  time_difference = 16 →
  let reduced_speed := current_speed - speed_reduction
  let increased_speed := current_speed * 1.5
  ∃ (distance : ℝ), distance = 96 ∧
    distance / reduced_speed = distance / increased_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_pascals_cycling_trip_l2316_231645


namespace NUMINAMATH_CALUDE_triangle_altitude_area_theorem_l2316_231677

/-- Definition of a triangle with altitudes and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  area : ℝ

/-- Theorem stating the existence and non-existence of triangles with specific properties -/
theorem triangle_altitude_area_theorem :
  (∃ t : Triangle, t.ha < 1 ∧ t.hb < 1 ∧ t.hc < 1 ∧ t.area > 2) ∧
  (¬ ∃ t : Triangle, t.ha > 2 ∧ t.hb > 2 ∧ t.hc > 2 ∧ t.area < 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_area_theorem_l2316_231677


namespace NUMINAMATH_CALUDE_x_range_and_max_y_over_x_l2316_231699

/-- Circle C with center (4,3) and radius 3 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + (p.2 - 3)^2 = 9}

/-- A point P on circle C -/
def P : ℝ × ℝ := sorry

/-- P is on circle C -/
axiom hP : P ∈ C

theorem x_range_and_max_y_over_x :
  (1 ≤ P.1 ∧ P.1 ≤ 7) ∧
  ∀ Q ∈ C, Q.2 / Q.1 ≤ 24 / 7 := by sorry

end NUMINAMATH_CALUDE_x_range_and_max_y_over_x_l2316_231699


namespace NUMINAMATH_CALUDE_salesman_visits_l2316_231689

def S : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | n + 3 => S (n + 2) + S (n + 1) + S n

theorem salesman_visits (n : ℕ) : S 12 = 927 := by
  sorry

end NUMINAMATH_CALUDE_salesman_visits_l2316_231689


namespace NUMINAMATH_CALUDE_eight_points_on_circle_theorem_l2316_231697

/-- A point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- The theorem statement -/
theorem eight_points_on_circle_theorem
  (p : ℕ) (n : ℕ) (points : Finset IntPoint) :
  Nat.Prime p →
  p % 2 = 1 →
  n > 0 →
  points.card = 8 →
  (∀ pt ∈ points, ∃ (x y : ℤ), pt = ⟨x, y⟩) →
  (∃ (center : IntPoint) (r : ℤ), r^2 = (p^n)^2 / 4 ∧
    ∀ pt ∈ points, (pt.x - center.x)^2 + (pt.y - center.y)^2 = r^2) →
  ∃ (a b c : IntPoint), a ∈ points ∧ b ∈ points ∧ c ∈ points ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∃ (ab bc ca : ℤ),
      ab = (a.x - b.x)^2 + (a.y - b.y)^2 ∧
      bc = (b.x - c.x)^2 + (b.y - c.y)^2 ∧
      ca = (c.x - a.x)^2 + (c.y - a.y)^2 ∧
      ab % p^(n+1) = 0 ∧ bc % p^(n+1) = 0 ∧ ca % p^(n+1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_eight_points_on_circle_theorem_l2316_231697


namespace NUMINAMATH_CALUDE_briana_investment_proof_l2316_231652

def emma_investment : ℝ := 300
def emma_yield_rate : ℝ := 0.15
def briana_yield_rate : ℝ := 0.10
def years : ℕ := 2
def return_difference : ℝ := 10

def briana_investment : ℝ := 400

theorem briana_investment_proof :
  (years : ℝ) * emma_yield_rate * emma_investment - 
  (years : ℝ) * briana_yield_rate * briana_investment = return_difference :=
by sorry

end NUMINAMATH_CALUDE_briana_investment_proof_l2316_231652


namespace NUMINAMATH_CALUDE_value_of_a_l2316_231654

theorem value_of_a (a b d : ℝ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2316_231654


namespace NUMINAMATH_CALUDE_largest_m_for_inequality_l2316_231693

theorem largest_m_for_inequality : ∃ m : ℕ+, 
  (m = 27) ∧ 
  (∀ n : ℕ+, n ≤ m → (2*n + 1)/(3*n + 8) < (Real.sqrt 5 - 1)/2 ∧ (Real.sqrt 5 - 1)/2 < (n + 7)/(2*n + 1)) ∧
  (∀ m' : ℕ+, m' > m → ∃ n : ℕ+, n ≤ m' ∧ ((2*n + 1)/(3*n + 8) ≥ (Real.sqrt 5 - 1)/2 ∨ (Real.sqrt 5 - 1)/2 ≥ (n + 7)/(2*n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_inequality_l2316_231693


namespace NUMINAMATH_CALUDE_expand_expression_l2316_231601

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y + 6) = 36 * x + 48 * y + 72 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2316_231601


namespace NUMINAMATH_CALUDE_sin_15_cos_15_double_l2316_231619

theorem sin_15_cos_15_double : 2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_double_l2316_231619


namespace NUMINAMATH_CALUDE_ace_spade_probability_l2316_231639

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Probability of drawing an Ace as the first card and a spade as the second card -/
def prob_ace_then_spade : ℚ :=
  (NumAces / StandardDeck) * (NumSpades / (StandardDeck - 1))

theorem ace_spade_probability :
  prob_ace_then_spade = 3 / 127 := by
  sorry

end NUMINAMATH_CALUDE_ace_spade_probability_l2316_231639


namespace NUMINAMATH_CALUDE_min_trees_chopped_l2316_231604

def trees_per_sharpening : ℕ := 13
def cost_per_sharpening : ℕ := 5
def total_sharpening_cost : ℕ := 35

theorem min_trees_chopped :
  ∃ (n : ℕ), n ≥ 91 ∧ n ≥ (total_sharpening_cost / cost_per_sharpening) * trees_per_sharpening :=
by sorry

end NUMINAMATH_CALUDE_min_trees_chopped_l2316_231604


namespace NUMINAMATH_CALUDE_smallest_with_eight_prime_power_divisors_l2316_231682

def is_prime_power (n : ℕ) : Prop := ∃ p k, Prime p ∧ n = p ^ k

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem smallest_with_eight_prime_power_divisors :
  (∀ m : ℕ, m < 24 →
    (divisors m).card ≠ 8 ∨
    ¬(∀ d ∈ divisors m, is_prime_power d)) ∧
  (divisors 24).card = 8 ∧
  (∀ d ∈ divisors 24, is_prime_power d) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_eight_prime_power_divisors_l2316_231682


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l2316_231635

theorem polar_coordinates_of_point (x y : ℝ) (r θ : ℝ) :
  x = -Real.sqrt 3 ∧ y = -1 →
  r = 2 ∧ θ = 7 * Real.pi / 6 →
  x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ r ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l2316_231635


namespace NUMINAMATH_CALUDE_square_covering_theorem_l2316_231634

theorem square_covering_theorem (l : ℕ) (h1 : l > 0) : 
  (∃ n : ℕ, n > 0 ∧ 2 * n^2 = 8 * l^2 / 9 ∧ l^2 < 2 * (n + 1)^2) ↔ 
  l ∈ ({3, 6, 9, 12, 15, 18, 21, 24} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_square_covering_theorem_l2316_231634


namespace NUMINAMATH_CALUDE_smallest_max_sum_l2316_231650

theorem smallest_max_sum (p q r s t : ℕ+) (h_sum : p + q + r + s + t = 4020) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  ∀ m : ℕ, (∀ a b c d e : ℕ+, a + b + c + d + e = 4020 →
    m ≥ max (a + b) (max (b + c) (max (c + d) (d + e)))) →
  m ≥ 1342 :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l2316_231650


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2316_231672

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2316_231672


namespace NUMINAMATH_CALUDE_puzzle_solution_l2316_231622

/-- Given positive integers A and B less than 10 satisfying the equation 21A104 × 11 = 2B8016 × 9, 
    prove that A = 1 and B = 5. -/
theorem puzzle_solution (A B : ℕ) 
  (h1 : 0 < A ∧ A < 10) 
  (h2 : 0 < B ∧ B < 10) 
  (h3 : 21 * 100000 + A * 10000 + 104 * 11 = 2 * 100000 + B * 10000 + 8016 * 9) : 
  A = 1 ∧ B = 5 := by
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2316_231622


namespace NUMINAMATH_CALUDE_simplify_expression_l2316_231678

theorem simplify_expression (x : ℝ) : 5*x + 9*x^2 + 8 - (6 - 5*x - 3*x^2) = 12*x^2 + 10*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2316_231678


namespace NUMINAMATH_CALUDE_parabola_p_value_l2316_231694

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h : y^2 = 2 * C.p * x

/-- Theorem: Value of p for a parabola given specific point conditions -/
theorem parabola_p_value (C : Parabola) (A : PointOnParabola C)
  (h1 : Real.sqrt ((A.x - C.p/2)^2 + A.y^2) = 12)  -- Distance from A to focus is 12
  (h2 : A.x = 9)  -- Distance from A to y-axis is 9
  : C.p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_p_value_l2316_231694


namespace NUMINAMATH_CALUDE_security_compromise_l2316_231676

/-- Represents the security level of a system -/
inductive SecurityLevel
  | High
  | Medium
  | Low

/-- Represents a file type -/
inductive FileType
  | Secure
  | Suspicious

/-- Represents a website -/
structure Website where
  trusted : Bool

/-- Represents a user action -/
inductive UserAction
  | ShareInfo
  | DownloadFile (fileType : FileType)

/-- Represents the state of a system after a user action -/
structure SystemState where
  securityLevel : SecurityLevel

/-- Defines how a user action affects the system state -/
def updateSystemState (website : Website) (action : UserAction) (initialState : SystemState) : SystemState :=
  match website.trusted, action with
  | true, _ => initialState
  | false, UserAction.ShareInfo => ⟨SecurityLevel.Low⟩
  | false, UserAction.DownloadFile FileType.Suspicious => ⟨SecurityLevel.Low⟩
  | false, UserAction.DownloadFile FileType.Secure => initialState

theorem security_compromise (website : Website) (action : UserAction) (initialState : SystemState) :
  ¬website.trusted →
  (action = UserAction.ShareInfo ∨ (∃ (ft : FileType), action = UserAction.DownloadFile ft ∧ ft = FileType.Suspicious)) →
  (updateSystemState website action initialState).securityLevel = SecurityLevel.Low :=
by sorry


end NUMINAMATH_CALUDE_security_compromise_l2316_231676


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l2316_231698

theorem triangle_abc_proof (c : ℝ) (A C : ℝ) 
  (h_c : c = 10)
  (h_A : A = 45 * π / 180)
  (h_C : C = 30 * π / 180) :
  ∃ (a b B : ℝ),
    a = 10 * Real.sqrt 2 ∧
    b = 5 * (Real.sqrt 2 + Real.sqrt 6) ∧
    B = 105 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l2316_231698


namespace NUMINAMATH_CALUDE_planter_pots_cost_l2316_231605

/-- Calculates the total cost of filling planter pots with plants, including sales tax. -/
def total_cost (num_pots : ℕ) (palm_fern_cost creeping_jenny_cost geranium_cost elephant_ear_cost purple_grass_cost : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let plant_cost_per_pot := palm_fern_cost + 4 * creeping_jenny_cost + 4 * geranium_cost + 2 * elephant_ear_cost + 3 * purple_grass_cost
  let total_plant_cost := num_pots * plant_cost_per_pot
  let sales_tax := sales_tax_rate * total_plant_cost
  total_plant_cost + sales_tax

/-- Theorem stating that the total cost to fill 6 planter pots with the given plants and 7% sales tax is $494.34. -/
theorem planter_pots_cost : total_cost 6 15 4 3.5 7 6 (7/100) = 494.34 := by
  sorry

end NUMINAMATH_CALUDE_planter_pots_cost_l2316_231605


namespace NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l2316_231684

theorem sqrt_t6_plus_t4 (t : ℝ) : Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l2316_231684


namespace NUMINAMATH_CALUDE_triangle_equilateral_l2316_231629

theorem triangle_equilateral (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_condition : a^2 + b^2 + c^2 - a*b - b*c - a*c = 0) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l2316_231629


namespace NUMINAMATH_CALUDE_race_distance_proof_l2316_231616

/-- The distance of the race where B beats C -/
def race_distance : ℝ := 800

theorem race_distance_proof :
  ∀ (v_a v_b v_c : ℝ),  -- speeds of A, B, and C
  v_a > 0 ∧ v_b > 0 ∧ v_c > 0 →  -- positive speeds
  (1000 / v_a = 900 / v_b) →  -- A beats B by 100m in 1000m race
  (race_distance / v_b = (race_distance - 100) / v_c) →  -- B beats C by 100m in race_distance
  (1000 / v_a = 787.5 / v_c) →  -- A beats C by 212.5m in 1000m race
  race_distance = 800 := by
sorry

end NUMINAMATH_CALUDE_race_distance_proof_l2316_231616


namespace NUMINAMATH_CALUDE_dust_storm_coverage_l2316_231631

/-- The dust storm problem -/
theorem dust_storm_coverage (total_prairie : ℕ) (untouched : ℕ) (covered : ℕ) : 
  total_prairie = 64013 → untouched = 522 → covered = total_prairie - untouched → covered = 63491 := by
  sorry

end NUMINAMATH_CALUDE_dust_storm_coverage_l2316_231631


namespace NUMINAMATH_CALUDE_triple_layer_area_is_six_l2316_231690

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the hall and the arrangement of carpets -/
structure CarpetArrangement where
  hallSize : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area covered by all three carpets in the given arrangement -/
def tripleLayerArea (arrangement : CarpetArrangement) : ℝ :=
  sorry

/-- Theorem stating that the area covered by all three carpets is 6 square meters -/
theorem triple_layer_area_is_six (arrangement : CarpetArrangement) 
  (h1 : arrangement.hallSize = 10)
  (h2 : arrangement.carpet1 = ⟨6, 8⟩)
  (h3 : arrangement.carpet2 = ⟨6, 6⟩)
  (h4 : arrangement.carpet3 = ⟨5, 7⟩) :
  tripleLayerArea arrangement = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_layer_area_is_six_l2316_231690


namespace NUMINAMATH_CALUDE_adult_panda_consumption_is_138_l2316_231609

/-- The daily bamboo consumption of an adult panda -/
def adult_panda_daily_consumption : ℕ := 138

/-- The daily bamboo consumption of a baby panda -/
def baby_panda_daily_consumption : ℕ := 50

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total bamboo consumption of both pandas in a week -/
def total_weekly_consumption : ℕ := 1316

/-- Theorem stating that the adult panda's daily bamboo consumption is 138 pounds -/
theorem adult_panda_consumption_is_138 :
  adult_panda_daily_consumption = 
    (total_weekly_consumption - baby_panda_daily_consumption * days_in_week) / days_in_week :=
by sorry

end NUMINAMATH_CALUDE_adult_panda_consumption_is_138_l2316_231609


namespace NUMINAMATH_CALUDE_inequality_proof_l2316_231647

theorem inequality_proof (x y : ℝ) : 
  ((x * y - y^2) / (x^2 + 4 * x + 5))^3 ≤ ((x^2 - x * y) / (x^2 + 4 * x + 5))^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2316_231647


namespace NUMINAMATH_CALUDE_greatest_common_factor_4050_12320_l2316_231626

theorem greatest_common_factor_4050_12320 : Nat.gcd 4050 12320 = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_4050_12320_l2316_231626


namespace NUMINAMATH_CALUDE_unit_circle_from_sin_cos_l2316_231667

-- Define the set of points (x,y) = (sin t, cos t) for all real t
def unitCirclePoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = Real.sin t ∧ p.2 = Real.cos t}

-- Theorem: The set of points forms a circle with radius 1 centered at the origin
theorem unit_circle_from_sin_cos :
  unitCirclePoints = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_unit_circle_from_sin_cos_l2316_231667


namespace NUMINAMATH_CALUDE_quadratic_root_factorization_l2316_231628

theorem quadratic_root_factorization 
  (a₀ a₁ a₂ x r s : ℝ) 
  (h₁ : a₂ ≠ 0) 
  (h₂ : a₀ ≠ 0) 
  (h₃ : a₀ + a₁ * r + a₂ * r^2 = 0) 
  (h₄ : a₀ + a₁ * s + a₂ * s^2 = 0) :
  a₀ + a₁ * x + a₂ * x^2 = a₀ * (1 - x / r) * (1 - x / s) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_factorization_l2316_231628


namespace NUMINAMATH_CALUDE_draw_specific_nondefective_draw_at_least_one_defective_l2316_231624

/-- Represents the total number of products -/
def total_products : ℕ := 10

/-- Represents the number of defective products -/
def defective_products : ℕ := 2

/-- Represents the number of products drawn for inspection -/
def drawn_products : ℕ := 3

/-- Theorem for the number of ways to draw a specific non-defective product -/
theorem draw_specific_nondefective :
  (total_products - defective_products).choose (drawn_products - 1) = 36 := by sorry

/-- Theorem for the number of ways to draw at least one defective product -/
theorem draw_at_least_one_defective :
  (defective_products.choose 1 * (total_products - defective_products).choose (drawn_products - 1)) +
  (defective_products.choose 2 * (total_products - defective_products).choose (drawn_products - 2)) = 64 := by sorry

end NUMINAMATH_CALUDE_draw_specific_nondefective_draw_at_least_one_defective_l2316_231624


namespace NUMINAMATH_CALUDE_cube_volume_l2316_231657

theorem cube_volume (cube_diagonal : ℝ) (h : cube_diagonal = 6 * Real.sqrt 2) :
  ∃ (volume : ℝ), volume = 216 ∧ volume = (cube_diagonal / Real.sqrt 2) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l2316_231657


namespace NUMINAMATH_CALUDE_sara_minus_lucas_sum_l2316_231659

def sara_list := List.range 50

def replace_three_with_two (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

def lucas_list := sara_list.map replace_three_with_two

theorem sara_minus_lucas_sum : 
  sara_list.sum - lucas_list.sum = 105 := by sorry

end NUMINAMATH_CALUDE_sara_minus_lucas_sum_l2316_231659


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2316_231608

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) :
  (∀ y, 0 < y ∧ y < 6 → (6 - y) * y ≤ (6 - x) * x) → (6 - x) * x = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2316_231608


namespace NUMINAMATH_CALUDE_max_value_of_f_l2316_231668

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), M = 3/2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2316_231668


namespace NUMINAMATH_CALUDE_solve_for_t_l2316_231688

theorem solve_for_t (s t : ℚ) 
  (eq1 : 12 * s + 7 * t = 165)
  (eq2 : s = t + 3) : 
  t = 129 / 19 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l2316_231688


namespace NUMINAMATH_CALUDE_exam_comparison_l2316_231651

/-- Given a 50-question exam where Sylvia has one-fifth of incorrect answers
    and Sergio has 4 incorrect answers, prove that Sergio has 6 more correct
    answers than Sylvia. -/
theorem exam_comparison (total_questions : ℕ) (sylvia_incorrect_ratio : ℚ)
    (sergio_incorrect : ℕ) (h1 : total_questions = 50)
    (h2 : sylvia_incorrect_ratio = 1 / 5)
    (h3 : sergio_incorrect = 4) :
    (total_questions - (sylvia_incorrect_ratio * total_questions).num) -
    (total_questions - sergio_incorrect) = 6 := by
  sorry

end NUMINAMATH_CALUDE_exam_comparison_l2316_231651


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1001_l2316_231669

theorem largest_gcd_of_sum_1001 :
  ∃ (a b : ℕ+), a + b = 1001 ∧
  ∀ (c d : ℕ+), c + d = 1001 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 143 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1001_l2316_231669


namespace NUMINAMATH_CALUDE_refrigerator_transport_cost_l2316_231643

/-- Calculates the transport cost given the purchase details of a refrigerator --/
theorem refrigerator_transport_cost 
  (purchase_price_after_discount : ℕ)
  (discount_rate : ℚ)
  (installation_cost : ℕ)
  (selling_price_for_profit : ℕ) :
  purchase_price_after_discount = 12500 →
  discount_rate = 1/5 →
  installation_cost = 250 →
  selling_price_for_profit = 18560 →
  (purchase_price_after_discount / (1 - discount_rate) * (1 + 4/25) : ℚ) = selling_price_for_profit →
  (selling_price_for_profit : ℚ) - purchase_price_after_discount - installation_cost = 5810 :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_transport_cost_l2316_231643


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_l2316_231612

def is_valid_number (abc : ℕ) : Prop :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  let cab := c * 100 + a * 10 + b
  let bca := b * 100 + c * 10 + a
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  abc ≥ 100 ∧ abc < 1000 ∧
  2 * b = a + c ∧
  (cab * abc : ℚ) = bca * bca

theorem valid_three_digit_numbers :
  ∀ abc : ℕ, is_valid_number abc → abc = 432 ∨ abc = 864 :=
sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_l2316_231612


namespace NUMINAMATH_CALUDE_range_of_m_l2316_231625

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 = 0

-- Define the proposition p
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- Define the proposition q
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

-- Theorem statement
theorem range_of_m : 
  ∀ m : ℝ, (¬p m ∧ q m) → (1 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2316_231625


namespace NUMINAMATH_CALUDE_rent_is_1000_l2316_231642

/-- Calculates the rent given salary, remaining amount, and the relationship between rent and other expenses. -/
def calculate_rent (salary : ℕ) (remaining : ℕ) : ℕ :=
  let total_expenses := salary - remaining
  total_expenses / 3

/-- Proves that the rent is $1000 given the conditions -/
theorem rent_is_1000 (salary : ℕ) (remaining : ℕ) 
  (h1 : salary = 5000)
  (h2 : remaining = 2000)
  (h3 : calculate_rent salary remaining = 1000) : 
  calculate_rent salary remaining = 1000 := by
  sorry

#eval calculate_rent 5000 2000

end NUMINAMATH_CALUDE_rent_is_1000_l2316_231642


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l2316_231617

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y = 1

-- Define the condition for point P
def P_on_C₂ (P : ℝ × ℝ) : Prop := C₂ P.1 P.2

-- Define the condition for point R
def R_on_C₁ (R : ℝ × ℝ) : Prop := C₁ R.1 R.2

-- Define the condition that R is on OP
def R_on_OP (O P R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 0 ∧ R.1 = t * P.1 ∧ R.2 = t * P.2

-- Define the condition for point Q
def Q_condition (O P Q R : ℝ × ℝ) : Prop :=
  (Q.1^2 + Q.2^2) * (P.1^2 + P.2^2) = (R.1^2 + R.2^2)^2

-- The main theorem
theorem trajectory_of_Q (O P Q R : ℝ × ℝ) :
  O = (0, 0) →
  P_on_C₂ P →
  R_on_C₁ R →
  R_on_OP O P R →
  Q_condition O P Q R →
  (Q.1 - 1/2)^2 + (Q.2 - 1/2)^2 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l2316_231617


namespace NUMINAMATH_CALUDE_field_planted_fraction_l2316_231640

theorem field_planted_fraction (a b d : ℝ) (ha : a > 0) (hb : b > 0) (hd : d > 0) :
  let c := (a^2 + b^2).sqrt
  let x := (a * b * d) / (a^2 + b^2)
  let triangle_area := a * b / 2
  let square_area := x^2
  let planted_area := triangle_area - square_area
  a = 5 → b = 12 → d = 3 →
  planted_area / triangle_area = 52761 / 857430 := by
  sorry

end NUMINAMATH_CALUDE_field_planted_fraction_l2316_231640


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l2316_231636

theorem consecutive_integers_cube_sum : 
  ∀ n : ℕ, 
  n > 0 → 
  (n - 1) * n * (n + 1) = 8 * (3 * n) → 
  (n - 1)^3 + n^3 + (n + 1)^3 = 405 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l2316_231636


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2316_231656

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + 2 * a 8 + a 15 = 96) →
  2 * a 9 - a 10 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2316_231656


namespace NUMINAMATH_CALUDE_lime_juice_per_lime_l2316_231620

-- Define the variables and constants
def tablespoons_per_mocktail : ℚ := 1
def days : ℕ := 30
def limes_per_dollar : ℚ := 3
def dollars_spent : ℚ := 5

-- Define the theorem
theorem lime_juice_per_lime :
  let total_tablespoons := tablespoons_per_mocktail * days
  let total_limes := limes_per_dollar * dollars_spent
  let juice_per_lime := total_tablespoons / total_limes
  juice_per_lime = 2 := by
sorry


end NUMINAMATH_CALUDE_lime_juice_per_lime_l2316_231620


namespace NUMINAMATH_CALUDE_tangent_line_at_A_l2316_231695

/-- The curve C defined by y = x^3 - x + 2 -/
def C (x : ℝ) : ℝ := x^3 - x + 2

/-- The point A on the curve C -/
def A : ℝ × ℝ := (1, 2)

/-- The tangent line equation at point A -/
def tangent_line (x y : ℝ) : Prop := 2*x - y = 0

theorem tangent_line_at_A :
  tangent_line A.1 A.2 ∧
  ∀ x : ℝ, (tangent_line x (C x) ↔ x = A.1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_A_l2316_231695


namespace NUMINAMATH_CALUDE_sum_not_prime_l2316_231661

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_prime_l2316_231661


namespace NUMINAMATH_CALUDE_solve_equation_l2316_231644

theorem solve_equation (x : ℚ) : 5 * (2 * x - 3) = 3 * (3 - 4 * x) + 15 → x = 39 / 22 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2316_231644


namespace NUMINAMATH_CALUDE_toothpick_15th_stage_l2316_231674

def toothpick_sequence (n : ℕ) : ℕ :=
  5 + 3 * (n - 1)

theorem toothpick_15th_stage :
  toothpick_sequence 15 = 47 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_15th_stage_l2316_231674


namespace NUMINAMATH_CALUDE_self_square_root_numbers_l2316_231627

theorem self_square_root_numbers : {x : ℝ | x ≥ 0 ∧ x = Real.sqrt x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_self_square_root_numbers_l2316_231627


namespace NUMINAMATH_CALUDE_solve_for_y_l2316_231648

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2316_231648


namespace NUMINAMATH_CALUDE_spherical_triangle_area_l2316_231614

/-- The area of a spherical triangle formed by the intersection of a sphere with a trihedral angle -/
theorem spherical_triangle_area 
  (R : ℝ) 
  (α β γ : ℝ) 
  (h_positive : R > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_vertex_center : True)  -- Represents the condition that the vertex coincides with the sphere's center
  : ∃ (S_Δ : ℝ), S_Δ = R^2 * (α + β + γ - Real.pi) :=
sorry

end NUMINAMATH_CALUDE_spherical_triangle_area_l2316_231614


namespace NUMINAMATH_CALUDE_number_subtraction_division_l2316_231623

theorem number_subtraction_division : ∃! x : ℝ, (x - 5) / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_division_l2316_231623


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l2316_231621

/-- Binomial distribution -/
def binomial_distribution (n : ℕ) (p : ℝ) : ℝ → ℝ := sorry

/-- Probability of a random variable being greater than or equal to a value -/
def prob_ge (X : ℝ → ℝ) (k : ℝ) : ℝ := sorry

theorem binomial_probability_problem (p : ℝ) :
  let ξ := binomial_distribution 2 p
  let η := binomial_distribution 4 p
  prob_ge ξ 1 = 5/9 →
  prob_ge η 2 = 11/27 := by sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l2316_231621


namespace NUMINAMATH_CALUDE_six_player_tournament_games_l2316_231658

/-- The number of games in a chess tournament where each player plays twice with every other player. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 6 players, where each player plays twice with every other player, the total number of games played is 60. -/
theorem six_player_tournament_games :
  tournament_games 6 = 30 ∧ 2 * tournament_games 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_player_tournament_games_l2316_231658


namespace NUMINAMATH_CALUDE_nonstudent_ticket_price_l2316_231660

theorem nonstudent_ticket_price
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (student_price : ℕ)
  (student_tickets : ℕ)
  (h1 : total_tickets = 821)
  (h2 : total_revenue = 1933)
  (h3 : student_price = 2)
  (h4 : student_tickets = 530)
  (h5 : student_tickets < total_tickets) :
  let nonstudent_tickets : ℕ := total_tickets - student_tickets
  let nonstudent_price : ℕ := (total_revenue - student_price * student_tickets) / nonstudent_tickets
  nonstudent_price = 3 := by
sorry

end NUMINAMATH_CALUDE_nonstudent_ticket_price_l2316_231660


namespace NUMINAMATH_CALUDE_shed_width_calculation_l2316_231671

theorem shed_width_calculation (backyard_length backyard_width shed_length sod_area : ℝ)
  (h1 : backyard_length = 20)
  (h2 : backyard_width = 13)
  (h3 : shed_length = 3)
  (h4 : sod_area = 245)
  (h5 : backyard_length * backyard_width - sod_area = shed_length * shed_width) :
  shed_width = 5 := by
  sorry

end NUMINAMATH_CALUDE_shed_width_calculation_l2316_231671


namespace NUMINAMATH_CALUDE_ones_digit_73_power_l2316_231611

theorem ones_digit_73_power (n : ℕ) : 
  (73^n % 10 = 7) ↔ (n % 4 = 3) := by
sorry

end NUMINAMATH_CALUDE_ones_digit_73_power_l2316_231611


namespace NUMINAMATH_CALUDE_cafe_order_combinations_l2316_231637

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- Theorem: The number of ways two people can each choose one item from a set of 15 items,
    where order matters and repetition is allowed, is equal to 225. -/
theorem cafe_order_combinations :
  menu_items ^ num_people = 225 := by sorry

end NUMINAMATH_CALUDE_cafe_order_combinations_l2316_231637


namespace NUMINAMATH_CALUDE_sin_product_identity_l2316_231663

theorem sin_product_identity (α β : ℝ) :
  Real.sin α * Real.sin β = (Real.sin ((α + β) / 2))^2 - (Real.sin ((α - β) / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_identity_l2316_231663


namespace NUMINAMATH_CALUDE_archer_weekly_expenditure_is_1056_l2316_231681

/-- The archer's weekly expenditure on arrows -/
def archer_weekly_expenditure (shots_per_day : ℕ) (days_per_week : ℕ) 
  (recovery_rate : ℚ) (arrow_cost : ℚ) (team_contribution_rate : ℚ) : ℚ :=
  let total_shots := shots_per_day * days_per_week
  let recovered_arrows := (total_shots : ℚ) * recovery_rate
  let arrows_used := (total_shots : ℚ) - recovered_arrows
  let total_cost := arrows_used * arrow_cost
  let team_contribution := total_cost * team_contribution_rate
  total_cost - team_contribution

/-- Theorem stating the archer's weekly expenditure on arrows -/
theorem archer_weekly_expenditure_is_1056 :
  archer_weekly_expenditure 200 4 (1/5) (11/2) (7/10) = 1056 := by
  sorry

end NUMINAMATH_CALUDE_archer_weekly_expenditure_is_1056_l2316_231681


namespace NUMINAMATH_CALUDE_min_tests_for_passing_probability_l2316_231683

theorem min_tests_for_passing_probability (p : ℝ) (threshold : ℝ) : 
  (p = 3/4) → (threshold = 0.99) → 
  (∀ k : ℕ, k < 4 → 1 - (1 - p)^k ≤ threshold) ∧ 
  (1 - (1 - p)^4 > threshold) := by
sorry

end NUMINAMATH_CALUDE_min_tests_for_passing_probability_l2316_231683


namespace NUMINAMATH_CALUDE_factor_expression_l2316_231692

theorem factor_expression (x : ℝ) : 35 * x^11 + 49 * x^22 = 7 * x^11 * (5 + 7 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2316_231692


namespace NUMINAMATH_CALUDE_total_cars_in_week_is_450_l2316_231649

/-- The number of cars passing through a toll booth in a week -/
def totalCarsInWeek (mondayCars : ℕ) : ℕ :=
  -- Monday and Tuesday
  2 * mondayCars +
  -- Wednesday and Thursday
  2 * (2 * mondayCars) +
  -- Friday, Saturday, and Sunday
  3 * mondayCars

/-- Theorem stating that the total number of cars in a week is 450 -/
theorem total_cars_in_week_is_450 : totalCarsInWeek 50 = 450 := by
  sorry

#eval totalCarsInWeek 50

end NUMINAMATH_CALUDE_total_cars_in_week_is_450_l2316_231649


namespace NUMINAMATH_CALUDE_coin_arrangement_count_l2316_231615

/-- Represents the number of ways to arrange 5 gold and 5 silver coins -/
def colorArrangements : ℕ := Nat.choose 10 5

/-- Represents the number of valid face orientations for 10 coins -/
def validOrientations : ℕ := 144

/-- The total number of distinguishable arrangements -/
def totalArrangements : ℕ := colorArrangements * validOrientations

/-- Theorem stating the number of distinguishable arrangements -/
theorem coin_arrangement_count :
  totalArrangements = 36288 :=
sorry

end NUMINAMATH_CALUDE_coin_arrangement_count_l2316_231615


namespace NUMINAMATH_CALUDE_F_propagation_l2316_231687

-- Define F as a proposition on natural numbers
variable (F : ℕ → Prop)

-- State the theorem
theorem F_propagation (h1 : ∀ k : ℕ, k > 0 → (F k → F (k + 1)))
                      (h2 : ¬ F 7) :
  ¬ F 6 ∧ ¬ F 5 := by
  sorry

end NUMINAMATH_CALUDE_F_propagation_l2316_231687


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_values_l2316_231675

theorem perpendicular_lines_k_values (k : ℝ) : 
  (∀ x y : ℝ, (k - 1) * x + (2 * k + 3) * y - 2 = 0 ∧ 
               k * x + (1 - k) * y - 3 = 0 → 
               ((k - 1) * k + (2 * k + 3) * (1 - k) = 0)) → 
  k = 1 ∨ k = -3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_values_l2316_231675


namespace NUMINAMATH_CALUDE_final_short_oak_count_l2316_231686

/-- The number of short oak trees in the park after planting -/
def short_oak_trees_after_planting (current : ℕ) (to_plant : ℕ) : ℕ :=
  current + to_plant

/-- Theorem stating the number of short oak trees after planting -/
theorem final_short_oak_count :
  short_oak_trees_after_planting 3 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_short_oak_count_l2316_231686


namespace NUMINAMATH_CALUDE_opposite_sign_power_l2316_231670

theorem opposite_sign_power (x y : ℝ) : 
  (|x + 3| + (y - 2)^2 = 0) → x^y = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_power_l2316_231670


namespace NUMINAMATH_CALUDE_power_function_through_point_value_l2316_231655

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the theorem
theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 8 →
  f 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_value_l2316_231655


namespace NUMINAMATH_CALUDE_unique_coin_distribution_l2316_231630

/-- A structure representing the coin distribution in the piggy bank -/
structure CoinDistribution where
  one_ruble : ℕ
  two_rubles : ℕ
  five_rubles : ℕ

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Theorem stating the unique solution to the coin distribution problem -/
theorem unique_coin_distribution : 
  ∃! (d : CoinDistribution), 
    d.one_ruble + d.two_rubles + d.five_rubles = 1000 ∧ 
    d.one_ruble + 2 * d.two_rubles + 5 * d.five_rubles = 2000 ∧
    is_prime d.one_ruble ∧
    d.one_ruble = 3 ∧ d.two_rubles = 996 ∧ d.five_rubles = 1 := by
  sorry


end NUMINAMATH_CALUDE_unique_coin_distribution_l2316_231630


namespace NUMINAMATH_CALUDE_total_visitors_proof_l2316_231603

/-- The total number of visitors over two days at a tourist attraction -/
def total_visitors (m n : ℕ) : ℕ :=
  2 * m + n + 1000

/-- Theorem: The total number of visitors over two days is 2m + n + 1000 -/
theorem total_visitors_proof (m n : ℕ) : 
  total_visitors m n = 2 * m + n + 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_visitors_proof_l2316_231603


namespace NUMINAMATH_CALUDE_game_ends_in_36_rounds_l2316_231679

/-- Represents the state of a player in the game -/
structure PlayerState :=
  (tokens : ℕ)

/-- Represents the state of the game -/
structure GameState :=
  (a : PlayerState)
  (b : PlayerState)
  (c : PlayerState)
  (round : ℕ)

/-- Updates the game state for a single round -/
def updateRound (state : GameState) : GameState :=
  sorry

/-- Updates the game state for the extra discard every 5 rounds -/
def extraDiscard (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- The main theorem stating that the game ends after exactly 36 rounds -/
theorem game_ends_in_36_rounds :
  let initialState : GameState := {
    a := { tokens := 17 },
    b := { tokens := 16 },
    c := { tokens := 15 },
    round := 0
  }
  ∃ (finalState : GameState),
    (finalState.round = 36) ∧
    (gameEnded finalState = true) ∧
    (∀ (intermediateState : GameState),
      intermediateState.round < 36 →
      gameEnded intermediateState = false) :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_36_rounds_l2316_231679


namespace NUMINAMATH_CALUDE_common_factor_polynomial_l2316_231665

theorem common_factor_polynomial (a b c : ℤ) :
  ∃ (k : ℤ), (12 * a * b^3 * c + 8 * a^3 * b) = k * (4 * a * b) ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_common_factor_polynomial_l2316_231665


namespace NUMINAMATH_CALUDE_statue_of_liberty_model_height_l2316_231653

/-- The scale ratio of the model to the actual size -/
def scale_ratio : ℚ := 1 / 30

/-- The actual height of the Statue of Liberty in feet -/
def actual_height : ℕ := 305

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem statue_of_liberty_model_height :
  round_to_nearest (actual_height / scale_ratio) = 10 := by
  sorry

end NUMINAMATH_CALUDE_statue_of_liberty_model_height_l2316_231653


namespace NUMINAMATH_CALUDE_pythagorean_triple_check_l2316_231680

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_check :
  ¬ is_pythagorean_triple 12 15 18 ∧
  is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 6 9 15 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_check_l2316_231680


namespace NUMINAMATH_CALUDE_second_cube_surface_area_l2316_231638

theorem second_cube_surface_area (v1 v2 : ℝ) (h1 : v1 = 16) (h2 : v2 = 4 * v1) :
  6 * (v2 ^ (1/3 : ℝ))^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_second_cube_surface_area_l2316_231638


namespace NUMINAMATH_CALUDE_sandwich_cost_l2316_231606

/-- Proves that the cost of each sandwich is $5 --/
theorem sandwich_cost (num_sandwiches : ℕ) (paid : ℕ) (change : ℕ) :
  num_sandwiches = 3 ∧ paid = 20 ∧ change = 5 →
  (paid - change) / num_sandwiches = 5 :=
by
  sorry

#check sandwich_cost

end NUMINAMATH_CALUDE_sandwich_cost_l2316_231606


namespace NUMINAMATH_CALUDE_binders_per_student_is_one_l2316_231600

/-- Calculates the number of binders per student given the class size, costs of supplies, and total spent -/
def bindersPerStudent (
  classSize : ℕ) 
  (penCost notebookCost binderCost highlighterCost : ℚ)
  (pensPerStudent notebooksPerStudent highlightersPerStudent : ℕ)
  (teacherDiscount totalSpent : ℚ) : ℚ :=
  let totalPenCost := classSize * pensPerStudent * penCost
  let totalNotebookCost := classSize * notebooksPerStudent * notebookCost
  let totalHighlighterCost := classSize * highlightersPerStudent * highlighterCost
  let effectiveAmount := totalSpent + teacherDiscount
  let binderSpend := effectiveAmount - (totalPenCost + totalNotebookCost + totalHighlighterCost)
  let totalBinders := binderSpend / binderCost
  totalBinders / classSize

theorem binders_per_student_is_one :
  bindersPerStudent 30 0.5 1.25 4.25 0.75 5 3 2 100 260 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binders_per_student_is_one_l2316_231600


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2316_231685

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x| = 2 * x + 1 :=
by
  -- The unique solution is x = -1/3
  use -1/3
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2316_231685


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l2316_231646

theorem dining_bill_calculation (people : ℕ) (tip_percentage : ℚ) (individual_share : ℚ) :
  people = 8 →
  tip_percentage = 1/10 →
  individual_share = 191125/10000 →
  ∃ (original_bill : ℚ), 
    (original_bill * (1 + tip_percentage)) / people = individual_share ∧
    original_bill = 139 :=
by sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l2316_231646


namespace NUMINAMATH_CALUDE_polygon_sides_with_120_degree_interior_angles_l2316_231607

theorem polygon_sides_with_120_degree_interior_angles :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
    interior_angle = 120 →
    exterior_angle = 180 - interior_angle →
    (n : ℝ) * exterior_angle = 360 →
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_with_120_degree_interior_angles_l2316_231607


namespace NUMINAMATH_CALUDE_pb_cookie_probability_l2316_231696

/-- Represents the number of peanut butter cookies Jenny brought -/
def jenny_pb : ℕ := 40

/-- Represents the number of chocolate chip cookies Jenny brought -/
def jenny_cc : ℕ := 50

/-- Represents the number of peanut butter cookies Marcus brought -/
def marcus_pb : ℕ := 30

/-- Represents the number of lemon cookies Marcus brought -/
def marcus_lemon : ℕ := 20

/-- Represents the total number of cookies -/
def total_cookies : ℕ := jenny_pb + jenny_cc + marcus_pb + marcus_lemon

/-- Represents the total number of peanut butter cookies -/
def total_pb : ℕ := jenny_pb + marcus_pb

/-- Theorem stating that the probability of selecting a peanut butter cookie is 50% -/
theorem pb_cookie_probability : 
  (total_pb : ℚ) / total_cookies * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_pb_cookie_probability_l2316_231696


namespace NUMINAMATH_CALUDE_sequence_ratio_l2316_231602

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that (a₂ - a₁) / b₂ = 1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (-2 : ℝ) - a₁ = a₁ - a₂ ∧ 
  a₂ - a₁ = a₁ - (-8 : ℝ) ∧
  (-2 : ℝ) * b₁ = b₁ * b₂ ∧
  b₁ * b₂ = b₂ * b₃ ∧
  b₂ * b₃ = b₃ * (-8 : ℝ) →
  (a₂ - a₁) / b₂ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l2316_231602
