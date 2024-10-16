import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_evaluation_l4094_409426

theorem polynomial_evaluation : let x : ℝ := 3
  x^6 - 4*x^2 + 3*x = 702 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l4094_409426


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4094_409473

theorem sum_of_coefficients (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^10 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                           a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 1023 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4094_409473


namespace NUMINAMATH_CALUDE_max_silver_tokens_l4094_409429

/-- Represents the number of tokens Alex has -/
structure Tokens where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules for the booths -/
structure ExchangeRule where
  redCost : ℕ
  blueCost : ℕ
  redGain : ℕ
  blueGain : ℕ
  silverGain : ℕ

/-- Defines if an exchange is possible given the current tokens and an exchange rule -/
def canExchange (t : Tokens) (r : ExchangeRule) : Prop :=
  t.red ≥ r.redCost ∧ t.blue ≥ r.blueCost

/-- Applies an exchange rule to the current tokens -/
def applyExchange (t : Tokens) (r : ExchangeRule) : Tokens :=
  { red := t.red - r.redCost + r.redGain,
    blue := t.blue - r.blueCost + r.blueGain,
    silver := t.silver + r.silverGain }

/-- Theorem: The maximum number of silver tokens Alex can obtain is 23 -/
theorem max_silver_tokens :
  ∀ (initial : Tokens)
    (rule1 rule2 : ExchangeRule),
  initial.red = 60 ∧ initial.blue = 90 ∧ initial.silver = 0 →
  rule1 = { redCost := 3, blueCost := 0, redGain := 0, blueGain := 2, silverGain := 1 } →
  rule2 = { redCost := 0, blueCost := 4, redGain := 1, blueGain := 0, silverGain := 1 } →
  ∃ (final : Tokens),
    (∀ t, (canExchange t rule1 ∨ canExchange t rule2) → t.silver ≤ final.silver) ∧
    final.silver = 23 :=
by sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l4094_409429


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4094_409411

/-- Given an arithmetic sequence {a_n}, prove that a₃ + a₆ + a₉ = 33,
    when a₁ + a₄ + a₇ = 45 and a₂ + a₅ + a₈ = 39 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 4 + a 7 = 45 →
  a 2 + a 5 + a 8 = 39 →
  a 3 + a 6 + a 9 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4094_409411


namespace NUMINAMATH_CALUDE_weaving_problem_l4094_409487

/-- Represents the daily increase in cloth production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days of weaving -/
def days : ℕ := 30

/-- Represents the amount woven on the first day -/
def first_day_production : ℚ := 5

/-- Represents the total amount of cloth woven -/
def total_production : ℚ := 390

theorem weaving_problem :
  first_day_production * days + (days * (days - 1) / 2) * daily_increase = total_production := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l4094_409487


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l4094_409459

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l4094_409459


namespace NUMINAMATH_CALUDE_cars_without_ac_l4094_409418

theorem cars_without_ac (total : ℕ) (min_racing : ℕ) (max_ac_no_racing : ℕ) 
  (h_total : total = 100)
  (h_min_racing : min_racing = 41)
  (h_max_ac_no_racing : max_ac_no_racing = 59) :
  total - (max_ac_no_racing + 0) = 41 := by
  sorry

end NUMINAMATH_CALUDE_cars_without_ac_l4094_409418


namespace NUMINAMATH_CALUDE_unique_digit_solution_l4094_409475

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem unique_digit_solution (A M C : ℕ) 
  (h_A : is_digit A) (h_M : is_digit M) (h_C : is_digit C)
  (h_eq : (100*A + 10*M + C) * (A + M + C) = 2005) : 
  A = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l4094_409475


namespace NUMINAMATH_CALUDE_area_of_specific_region_l4094_409493

/-- The area of a specific region in a circle with an inscribed regular hexagon -/
theorem area_of_specific_region (r : ℝ) (s : ℝ) (h_r : r = 3) (h_s : s = 2) :
  let circle_area := π * r^2
  let hexagon_side := s
  let sector_angle := 120
  let sector_area := (sector_angle / 360) * circle_area
  let triangle_area := (1/2) * r^2 * Real.sin (sector_angle * π / 180)
  sector_area - triangle_area = 3 * π - (9 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_region_l4094_409493


namespace NUMINAMATH_CALUDE_sams_weight_l4094_409494

/-- Given the weights of Tyler, Sam, Peter, and Alex, prove Sam's weight --/
theorem sams_weight (tyler sam peter alex : ℝ) : 
  tyler = sam + 25 →
  peter = tyler / 2 →
  alex = 2 * (sam + peter) →
  peter = 65 →
  sam = 105 := by
  sorry

end NUMINAMATH_CALUDE_sams_weight_l4094_409494


namespace NUMINAMATH_CALUDE_crayon_distribution_l4094_409454

theorem crayon_distribution (total benny fred jason sarah : ℕ) : 
  total = 96 →
  benny = 12 →
  fred = 2 * benny →
  jason = 3 * sarah →
  fred + benny + jason + sarah = total →
  (fred = 24 ∧ benny = 12 ∧ jason = 45 ∧ sarah = 15) :=
by sorry

end NUMINAMATH_CALUDE_crayon_distribution_l4094_409454


namespace NUMINAMATH_CALUDE_peter_walks_to_grocery_store_l4094_409486

/-- The total distance Peter walks to the grocery store -/
def total_distance (walking_speed : ℝ) (distance_walked : ℝ) (remaining_time : ℝ) : ℝ :=
  distance_walked + walking_speed * remaining_time

/-- Theorem: Peter walks 2.5 miles to the grocery store -/
theorem peter_walks_to_grocery_store :
  let walking_speed : ℝ := 1 / 20 -- 1 mile per 20 minutes
  let distance_walked : ℝ := 1 -- 1 mile already walked
  let remaining_time : ℝ := 30 -- 30 more minutes to walk
  total_distance walking_speed distance_walked remaining_time = 2.5 := by
sorry

end NUMINAMATH_CALUDE_peter_walks_to_grocery_store_l4094_409486


namespace NUMINAMATH_CALUDE_quagga_placements_l4094_409474

/-- Represents a chessboard --/
def Chessboard := Fin 8 × Fin 8

/-- Represents a quagga's move --/
def QuaggaMove := (Int × Int) × (Int × Int)

/-- Defines the valid moves for a quagga --/
def validQuaggaMoves : List QuaggaMove :=
  [(( 6,  0), ( 0,  5)), (( 6,  0), ( 0, -5)),
   ((-6,  0), ( 0,  5)), ((-6,  0), ( 0, -5)),
   (( 0,  6), ( 5,  0)), (( 0,  6), (-5,  0)),
   (( 0, -6), ( 5,  0)), (( 0, -6), (-5,  0))]

/-- Checks if a move is valid on the chessboard --/
def isValidMove (start : Chessboard) (move : QuaggaMove) : Bool :=
  let ((dx1, dy1), (dx2, dy2)) := move
  let (x, y) := start
  let x1 := x + dx1
  let y1 := y + dy1
  let x2 := x1 + dx2
  let y2 := y1 + dy2
  0 ≤ x2 ∧ x2 < 8 ∧ 0 ≤ y2 ∧ y2 < 8

/-- Represents a placement of quaggas on the chessboard --/
def QuaggaPlacement := List Chessboard

/-- Checks if a placement is valid (no quaggas attack each other) --/
def isValidPlacement (placement : QuaggaPlacement) : Bool :=
  sorry

/-- The main theorem to prove --/
theorem quagga_placements :
  (∃ (placements : List QuaggaPlacement),
    placements.length = 68 ∧
    ∀ p ∈ placements,
      p.length = 51 ∧
      isValidPlacement p) :=
sorry

end NUMINAMATH_CALUDE_quagga_placements_l4094_409474


namespace NUMINAMATH_CALUDE_tip_to_cost_ratio_l4094_409483

def pizza_order (boxes : ℕ) (cost_per_box : ℚ) (money_given : ℚ) (change_received : ℚ) : ℚ × ℚ :=
  let total_cost := boxes * cost_per_box
  let amount_paid := money_given - change_received
  let tip := amount_paid - total_cost
  (tip, total_cost)

theorem tip_to_cost_ratio : 
  let (tip, total_cost) := pizza_order 5 7 100 60
  (tip : ℚ) / total_cost = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_tip_to_cost_ratio_l4094_409483


namespace NUMINAMATH_CALUDE_car_speed_proof_l4094_409482

/-- Proves that a car traveling at 400 km/h takes 9 seconds to travel 1 kilometer,
    given that it takes 5 seconds longer than traveling 1 kilometer at 900 km/h. -/
theorem car_speed_proof (v : ℝ) (h1 : v > 0) :
  (1 / v) * 3600 = 9 ↔ v = 400 ∧ (1 / 900) * 3600 + 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l4094_409482


namespace NUMINAMATH_CALUDE_committee_formation_count_l4094_409421

def club_size : ℕ := 12
def committee_size : ℕ := 5
def president_count : ℕ := 1

theorem committee_formation_count :
  (club_size.choose committee_size) - ((club_size - president_count).choose committee_size) = 330 :=
by sorry

end NUMINAMATH_CALUDE_committee_formation_count_l4094_409421


namespace NUMINAMATH_CALUDE_is_valid_factorization_l4094_409451

/-- Proves that x^2 - 2x + 1 = (x - 1)^2 is a valid factorization -/
theorem is_valid_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_is_valid_factorization_l4094_409451


namespace NUMINAMATH_CALUDE_jason_seashells_l4094_409450

/-- The number of seashells Jason has now -/
def current_seashells : ℕ := 36

/-- The number of seashells Jason gave away -/
def given_away_seashells : ℕ := 13

/-- The initial number of seashells Jason found -/
def initial_seashells : ℕ := current_seashells + given_away_seashells

theorem jason_seashells : initial_seashells = 49 := by
  sorry

end NUMINAMATH_CALUDE_jason_seashells_l4094_409450


namespace NUMINAMATH_CALUDE_inequality_proof_l4094_409432

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4094_409432


namespace NUMINAMATH_CALUDE_restaurant_ratio_change_l4094_409469

/-- Given a restaurant with an initial ratio of cooks to waiters of 3:8,
    9 cooks, and 12 additional waiters hired, prove that the new ratio
    of cooks to waiters is 1:4. -/
theorem restaurant_ratio_change (initial_cooks : ℕ) (initial_waiters : ℕ) 
    (additional_waiters : ℕ) :
  initial_cooks = 9 →
  initial_waiters = (8 * initial_cooks) / 3 →
  additional_waiters = 12 →
  (initial_cooks : ℚ) / (initial_waiters + additional_waiters : ℚ) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_restaurant_ratio_change_l4094_409469


namespace NUMINAMATH_CALUDE_special_trapezoid_smaller_side_l4094_409424

/-- A trapezoid with specific angle and side length properties -/
structure SpecialTrapezoid where
  /-- The angle at one end of the larger base -/
  angle1 : ℝ
  /-- The angle at the other end of the larger base -/
  angle2 : ℝ
  /-- The length of the larger lateral side -/
  larger_side : ℝ
  /-- The length of the smaller lateral side -/
  smaller_side : ℝ
  /-- Constraint: angle1 is 60 degrees -/
  angle1_is_60 : angle1 = 60
  /-- Constraint: angle2 is 30 degrees -/
  angle2_is_30 : angle2 = 30
  /-- Constraint: larger_side is 6√3 -/
  larger_side_is_6root3 : larger_side = 6 * Real.sqrt 3

/-- Theorem: In a SpecialTrapezoid, the smaller lateral side has length 6 -/
theorem special_trapezoid_smaller_side (t : SpecialTrapezoid) : t.smaller_side = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_smaller_side_l4094_409424


namespace NUMINAMATH_CALUDE_gcf_of_150_225_300_l4094_409462

theorem gcf_of_150_225_300 : Nat.gcd 150 (Nat.gcd 225 300) = 75 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_150_225_300_l4094_409462


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_thirds_l4094_409420

theorem sin_thirteen_pi_thirds : Real.sin (13 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_thirds_l4094_409420


namespace NUMINAMATH_CALUDE_a_age_is_eleven_l4094_409442

/-- Represents a person in the problem -/
inductive Person
  | A
  | B
  | C

/-- Represents a statement made by a person -/
structure Statement where
  person : Person
  content : Nat → Nat → Nat → Prop

/-- The set of all statements made by the three people -/
def statements : List Statement := sorry

/-- Predicate to check if a set of ages is consistent with the true statements -/
def consistent (a b c : Nat) : Prop := sorry

/-- Theorem stating that A's age is 11 -/
theorem a_age_is_eleven :
  ∃ (a b c : Nat),
    consistent a b c ∧
    (∀ (x y z : Nat), consistent x y z → (x = a ∧ y = b ∧ z = c)) ∧
    a = 11 := by sorry

end NUMINAMATH_CALUDE_a_age_is_eleven_l4094_409442


namespace NUMINAMATH_CALUDE_range_of_m_l4094_409444

-- Define set A
def A : Set ℝ := {x : ℝ | (x + 1) * (x - 6) ≤ 0}

-- Define set B
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (A ∩ B m = B m) ↔ (m < -2 ∨ (0 ≤ m ∧ m ≤ 5/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4094_409444


namespace NUMINAMATH_CALUDE_roots_of_equation_l4094_409488

theorem roots_of_equation : 
  let f : ℝ → ℝ := fun y ↦ (2 * y + 1) * (2 * y - 3)
  ∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = 3/2 ∧ (∀ y : ℝ, f y = 0 ↔ y = y₁ ∨ y = y₂) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l4094_409488


namespace NUMINAMATH_CALUDE_not_cheap_necessary_not_sufficient_for_good_quality_l4094_409441

-- Define the universe of products
variable (Product : Type)

-- Define predicates for product qualities
variable (not_cheap : Product → Prop)
variable (good_quality : Product → Prop)

-- Define the saying "you get what you pay for" as an axiom
axiom you_get_what_you_pay_for : ∀ (p : Product), good_quality p → not_cheap p

-- Theorem to prove
theorem not_cheap_necessary_not_sufficient_for_good_quality :
  (∀ (p : Product), good_quality p → not_cheap p) ∧
  (∃ (p : Product), not_cheap p ∧ ¬good_quality p) :=
sorry

end NUMINAMATH_CALUDE_not_cheap_necessary_not_sufficient_for_good_quality_l4094_409441


namespace NUMINAMATH_CALUDE_line_through_points_l4094_409414

/-- Given a line y = ax + b passing through points (3,7) and (7,19), prove that a - b = 5 -/
theorem line_through_points (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x + b) → 
  (7 : ℝ) = a * 3 + b → 
  (19 : ℝ) = a * 7 + b → 
  a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l4094_409414


namespace NUMINAMATH_CALUDE_expression_simplification_l4094_409422

theorem expression_simplification (m : ℝ) (h : m^2 - 2*m - 1 = 0) :
  (m + 2) / (2*m^2 - 6*m) / (m + 3 + 5 / (m - 3)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4094_409422


namespace NUMINAMATH_CALUDE_carlo_practice_difference_l4094_409496

/-- Represents Carlo's practice schedule for a week --/
structure PracticeSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem about Carlo's practice schedule --/
theorem carlo_practice_difference (schedule : PracticeSchedule) :
  schedule.monday = 2 * schedule.tuesday ∧
  schedule.tuesday < schedule.wednesday ∧
  schedule.wednesday = schedule.thursday + 5 ∧
  schedule.thursday = 50 ∧
  schedule.friday = 60 ∧
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday = 300 →
  schedule.wednesday - schedule.tuesday = 10 := by
  sorry

end NUMINAMATH_CALUDE_carlo_practice_difference_l4094_409496


namespace NUMINAMATH_CALUDE_f_not_monotonic_range_l4094_409492

/-- The function f(x) = x³ - 12x -/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 12

/-- A function is not monotonic on an interval if its derivative has a zero in that interval -/
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f' x = 0

/-- The theorem stating the range of k for which f is not monotonic on (k, k+2) -/
theorem f_not_monotonic_range :
  ∀ k : ℝ, not_monotonic f k (k+2) ↔ (k > -4 ∧ k < -2) ∨ (k > 0 ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_range_l4094_409492


namespace NUMINAMATH_CALUDE_quadratic_equation_with_prime_roots_l4094_409499

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The theorem statement -/
theorem quadratic_equation_with_prime_roots (a b : ℤ) :
  (∃ x y : ℕ, x ≠ y ∧ isPrime x ∧ isPrime y ∧ 
    (a : ℚ) * x^2 + (b : ℚ) * x - 2008 = 0 ∧ 
    (a : ℚ) * y^2 + (b : ℚ) * y - 2008 = 0) →
  3 * a + b = 1000 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_prime_roots_l4094_409499


namespace NUMINAMATH_CALUDE_marbles_lost_vs_found_l4094_409458

theorem marbles_lost_vs_found (initial : ℕ) (lost : ℕ) (found : ℕ) : 
  initial = 4 → lost = 16 → found = 8 → lost - found = 8 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_vs_found_l4094_409458


namespace NUMINAMATH_CALUDE_convex_polygon_not_divisible_into_nonconvex_quadrilaterals_l4094_409416

-- Define a polygon as a set of points in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define convexity for a polygon
def IsConvex (P : Polygon) : Prop := sorry

-- Define a quadrilateral as a polygon with exactly 4 vertices
def Quadrilateral (Q : Polygon) : Prop := sorry

-- Define nonconvexity for a quadrilateral
def IsNonConvex (Q : Polygon) : Prop := Quadrilateral Q ∧ ¬IsConvex Q

-- Main theorem
theorem convex_polygon_not_divisible_into_nonconvex_quadrilaterals 
  (M : Polygon) (n : ℕ) (M_i : Fin n → Polygon) :
  IsConvex M →
  (∀ i, IsNonConvex (M_i i)) →
  M ≠ ⋃ i, M_i i :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_not_divisible_into_nonconvex_quadrilaterals_l4094_409416


namespace NUMINAMATH_CALUDE_max_value_of_sqrt_sum_max_value_achievable_l4094_409439

theorem max_value_of_sqrt_sum (x y z : ℝ) :
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 6 →
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 3 * Real.sqrt 20 :=
by sorry

theorem max_value_achievable (x y z : ℝ) :
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 6 →
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 6 ∧
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) = 3 * Real.sqrt 20 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sqrt_sum_max_value_achievable_l4094_409439


namespace NUMINAMATH_CALUDE_dog_catches_rabbit_l4094_409472

/-- Proves that a dog chasing a rabbit catches up in 4 minutes under given conditions -/
theorem dog_catches_rabbit (dog_speed rabbit_speed : ℝ) (head_start : ℝ) :
  dog_speed = 24 ∧ rabbit_speed = 15 ∧ head_start = 0.6 →
  (head_start / (dog_speed - rabbit_speed)) * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_rabbit_l4094_409472


namespace NUMINAMATH_CALUDE_tan_alpha_2_implications_l4094_409443

theorem tan_alpha_2_implications (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6/11 ∧
  (1/4) * (Real.sin α)^2 + (1/3) * Real.sin α * Real.cos α + (1/2) * (Real.cos α)^2 + 1 = 43/30 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implications_l4094_409443


namespace NUMINAMATH_CALUDE_students_in_both_sports_l4094_409467

theorem students_in_both_sports (total : ℕ) (baseball : ℕ) (hockey : ℕ) 
  (h1 : total = 36) (h2 : baseball = 25) (h3 : hockey = 19) :
  baseball + hockey - total = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_sports_l4094_409467


namespace NUMINAMATH_CALUDE_power_two_equality_l4094_409435

theorem power_two_equality (m : ℕ) : 2^m = 2 * 16^2 * 4^3 * 8 → m = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_two_equality_l4094_409435


namespace NUMINAMATH_CALUDE_factorization_x4_minus_5x2_plus_4_l4094_409478

theorem factorization_x4_minus_5x2_plus_4 (x : ℝ) :
  x^4 - 5*x^2 + 4 = (x + 1)*(x - 1)*(x + 2)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_5x2_plus_4_l4094_409478


namespace NUMINAMATH_CALUDE_smooth_flow_probability_l4094_409447

def cable_capacities : List Nat := [1, 1, 2, 2, 3, 4]

def total_combinations : Nat := Nat.choose 6 3

def smooth_flow_combinations : Nat := 5

theorem smooth_flow_probability :
  (smooth_flow_combinations : ℚ) / total_combinations = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_smooth_flow_probability_l4094_409447


namespace NUMINAMATH_CALUDE_one_fifths_in_one_fourth_l4094_409498

theorem one_fifths_in_one_fourth : (1 : ℚ) / 4 / ((1 : ℚ) / 5) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_one_fifths_in_one_fourth_l4094_409498


namespace NUMINAMATH_CALUDE_pauls_lost_crayons_l4094_409431

/-- Given Paul's crayon situation, prove the number of lost crayons --/
theorem pauls_lost_crayons
  (initial_crayons : ℕ)
  (given_to_friends : ℕ)
  (total_lost_or_given : ℕ)
  (h1 : initial_crayons = 65)
  (h2 : given_to_friends = 213)
  (h3 : total_lost_or_given = 229)
  : total_lost_or_given - given_to_friends = 16 := by
  sorry

end NUMINAMATH_CALUDE_pauls_lost_crayons_l4094_409431


namespace NUMINAMATH_CALUDE_min_bags_for_candy_distribution_l4094_409425

theorem min_bags_for_candy_distribution : ∃ (n : ℕ), n > 0 ∧ 
  77 % n = 0 ∧ (7 * n) % 77 = 0 ∧ (11 * n) % 77 = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → 
    77 % m ≠ 0 ∨ (7 * m) % 77 ≠ 0 ∨ (11 * m) % 77 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_bags_for_candy_distribution_l4094_409425


namespace NUMINAMATH_CALUDE_rope_length_ratio_l4094_409412

def joeys_rope_length : ℕ := 56
def chads_rope_length : ℕ := 21

theorem rope_length_ratio : 
  (joeys_rope_length : ℚ) / (chads_rope_length : ℚ) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_ratio_l4094_409412


namespace NUMINAMATH_CALUDE_leo_mira_sum_difference_l4094_409452

def leo_sum : ℕ := (List.range 50).map (· + 1) |>.sum

def digit_replace (n : ℕ) : ℕ :=
  let s := toString n
  let s' := s.replace "2" "1" |>.replace "3" "0"
  s'.toNat!

def mira_sum : ℕ := (List.range 50).map (· + 1 |> digit_replace) |>.sum

theorem leo_mira_sum_difference : leo_sum - mira_sum = 420 := by
  sorry

end NUMINAMATH_CALUDE_leo_mira_sum_difference_l4094_409452


namespace NUMINAMATH_CALUDE_don_earlier_rum_l4094_409456

/-- The amount of rum Don had earlier in the day -/
def rum_earlier (pancake_rum : ℝ) (max_multiplier : ℝ) (remaining_rum : ℝ) : ℝ :=
  max_multiplier * pancake_rum - (pancake_rum + remaining_rum)

/-- Theorem stating the amount of rum Don had earlier -/
theorem don_earlier_rum :
  let pancake_rum : ℝ := 10
  let max_multiplier : ℝ := 3
  let remaining_rum : ℝ := 8
  rum_earlier pancake_rum max_multiplier remaining_rum = 12 := by
  sorry

end NUMINAMATH_CALUDE_don_earlier_rum_l4094_409456


namespace NUMINAMATH_CALUDE_bus_passenger_count_l4094_409428

/-- Calculates the final number of passengers on a bus after several stops -/
def final_passengers (initial : ℕ) (first_stop : ℕ) (off_other_stops : ℕ) (on_other_stops : ℕ) : ℕ :=
  initial + first_stop - off_other_stops + on_other_stops

/-- Theorem stating that given the specific passenger changes, the final number is 49 -/
theorem bus_passenger_count : final_passengers 50 16 22 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_bus_passenger_count_l4094_409428


namespace NUMINAMATH_CALUDE_simplify_fraction_l4094_409485

theorem simplify_fraction : (45 : ℚ) * (8 / 15) * (1 / 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4094_409485


namespace NUMINAMATH_CALUDE_inequality_proof_l4094_409417

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^3 + b^3 + 3*a*b = 1) (h2 : c + d = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 + (d + 1/d)^3 ≥ 40 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4094_409417


namespace NUMINAMATH_CALUDE_survey_results_l4094_409406

/-- Represents the distribution of students in the survey --/
structure StudentDistribution :=
  (high_proactive : ℕ)
  (high_not_proactive : ℕ)
  (average_proactive : ℕ)
  (average_not_proactive : ℕ)

/-- Calculates the chi-square test statistic --/
def chi_square (d : StudentDistribution) : ℚ :=
  let n := d.high_proactive + d.high_not_proactive + d.average_proactive + d.average_not_proactive
  let a := d.high_proactive
  let b := d.high_not_proactive
  let c := d.average_proactive
  let d := d.average_not_proactive
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem about the survey results --/
theorem survey_results (d : StudentDistribution) 
  (h1 : d.high_proactive = 18)
  (h2 : d.high_not_proactive = 7)
  (h3 : d.average_proactive = 6)
  (h4 : d.average_not_proactive = 19) :
  ∃ (X : ℕ → ℚ),
    X 0 = 57/100 ∧ 
    X 1 = 19/50 ∧ 
    X 2 = 1/20 ∧
    (X 0 * 0 + X 1 * 1 + X 2 * 2 = 12/25) ∧
    chi_square d > 10828/1000 := by
  sorry

end NUMINAMATH_CALUDE_survey_results_l4094_409406


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l4094_409408

theorem quadratic_roots_property (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x, x^2 + a*x + 3 = 0) ∧  -- equation has roots
  (x₁ ≠ x₂) ∧  -- roots are distinct
  (x₁^2 + a*x₁ + 3 = 0) ∧  -- x₁ is a root
  (x₂^2 + a*x₂ + 3 = 0) ∧  -- x₂ is a root
  (x₁^3 - 39/x₂ = x₂^3 - 39/x₁)  -- given condition
  → 
  a = 4 ∨ a = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l4094_409408


namespace NUMINAMATH_CALUDE_max_stick_length_l4094_409461

theorem max_stick_length (a b c : ℕ) (ha : a = 24) (hb : b = 32) (hc : c = 44) :
  Nat.gcd a (Nat.gcd b c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_stick_length_l4094_409461


namespace NUMINAMATH_CALUDE_fraction_simplification_l4094_409495

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4094_409495


namespace NUMINAMATH_CALUDE_board_game_ratio_l4094_409446

theorem board_game_ratio (total_students : ℕ) (reading_students : ℕ) (homework_students : ℕ) :
  total_students = 24 →
  reading_students = total_students / 2 →
  homework_students = 4 →
  (total_students - (reading_students + homework_students)) * 3 = total_students :=
by
  sorry

end NUMINAMATH_CALUDE_board_game_ratio_l4094_409446


namespace NUMINAMATH_CALUDE_shorter_side_is_ten_l4094_409466

/-- A rectangular room with given perimeter and area -/
structure Room where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 30
  area_eq : length * width = 200

/-- The shorter side of the room is 10 feet -/
theorem shorter_side_is_ten (room : Room) : min room.length room.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_shorter_side_is_ten_l4094_409466


namespace NUMINAMATH_CALUDE_total_points_proof_l4094_409407

def sam_points : ℕ := 75
def friend_points : ℕ := 12

theorem total_points_proof :
  sam_points + friend_points = 87 := by sorry

end NUMINAMATH_CALUDE_total_points_proof_l4094_409407


namespace NUMINAMATH_CALUDE_fib_100_mod_5_l4094_409400

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_5_l4094_409400


namespace NUMINAMATH_CALUDE_ellipse_b_value_l4094_409489

/-- The value of b for an ellipse with given properties -/
theorem ellipse_b_value (b : ℝ) (h1 : 0 < b) (h2 : b < 3) :
  (∀ x y : ℝ, x^2 / 9 + y^2 / b^2 = 1 →
    ∃ F₁ F₂ : ℝ × ℝ, 
      (F₁.1 < 0 ∧ F₂.1 > 0) ∧ 
      (∀ A B : ℝ × ℝ, 
        (A.1^2 / 9 + A.2^2 / b^2 = 1) ∧ 
        (B.1^2 / 9 + B.2^2 / b^2 = 1) ∧
        (∃ k : ℝ, A.2 = k * (A.1 - F₁.1) ∧ B.2 = k * (B.1 - F₁.1)) →
        (dist A F₁ + dist A F₂ = 6) ∧ 
        (dist B F₁ + dist B F₂ = 6) ∧
        (dist B F₂ + dist A F₂ ≤ 10))) →
  b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_b_value_l4094_409489


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4094_409401

theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∀ c, c ≠ 0 → (a * c^2 > b * c^2 → a > b)) ∧
  (∃ c, a > b ∧ ¬(a * c^2 > b * c^2)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4094_409401


namespace NUMINAMATH_CALUDE_scientific_notation_361000000_l4094_409464

/-- Express 361000000 in scientific notation -/
theorem scientific_notation_361000000 : ∃ (a : ℝ) (n : ℤ), 
  361000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.61 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_361000000_l4094_409464


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4094_409410

theorem min_value_sum_reciprocals (n : ℕ) (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
  (1 / (1 + a^n)) + (1 / (1 + b^n)) ≥ 1 ∧ 
  ((1 / (1 + 1^n)) + (1 / (1 + 1^n)) = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4094_409410


namespace NUMINAMATH_CALUDE_alex_calculation_l4094_409463

theorem alex_calculation (x : ℝ) : x / 6 - 18 = 24 → x * 6 + 18 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_alex_calculation_l4094_409463


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l4094_409477

theorem largest_triangle_perimeter :
  ∀ (x : ℕ),
  (7 + 8 > x) →
  (7 + x > 8) →
  (8 + x > 7) →
  (∀ y : ℕ, (7 + 8 > y) → (7 + y > 8) → (8 + y > 7) → y ≤ x) →
  7 + 8 + x = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l4094_409477


namespace NUMINAMATH_CALUDE_action_figure_cost_l4094_409460

theorem action_figure_cost 
  (current : ℕ) 
  (total : ℕ) 
  (cost : ℚ) : 
  current = 3 → 
  total = 8 → 
  cost = 30 → 
  (cost / (total - current) : ℚ) = 6 := by
sorry

end NUMINAMATH_CALUDE_action_figure_cost_l4094_409460


namespace NUMINAMATH_CALUDE_negative_half_to_fourth_power_l4094_409449

theorem negative_half_to_fourth_power :
  (-1/2 : ℚ)^4 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_negative_half_to_fourth_power_l4094_409449


namespace NUMINAMATH_CALUDE_cricket_team_throwers_l4094_409433

/-- Represents a cricket team with throwers and non-throwers -/
structure CricketTeam where
  total_players : ℕ
  throwers : ℕ
  right_handed : ℕ
  left_handed : ℕ

/-- Conditions for the cricket team problem -/
def valid_cricket_team (team : CricketTeam) : Prop :=
  team.total_players = 67 ∧
  team.throwers + team.right_handed + team.left_handed = team.total_players ∧
  team.right_handed + team.throwers = 57 ∧
  3 * team.left_handed = 2 * team.right_handed

theorem cricket_team_throwers (team : CricketTeam) 
  (h : valid_cricket_team team) : team.throwers = 37 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_throwers_l4094_409433


namespace NUMINAMATH_CALUDE_stock_profit_percentage_l4094_409405

theorem stock_profit_percentage 
  (stock_worth : ℝ) 
  (profit_portion : ℝ) 
  (loss_portion : ℝ) 
  (loss_percentage : ℝ) 
  (overall_loss : ℝ) 
  (h1 : stock_worth = 12499.99)
  (h2 : profit_portion = 0.2)
  (h3 : loss_portion = 0.8)
  (h4 : loss_percentage = 0.1)
  (h5 : overall_loss = 500) :
  ∃ (P : ℝ), 
    (stock_worth * profit_portion * (1 + P / 100) + 
     stock_worth * loss_portion * (1 - loss_percentage) = 
     stock_worth - overall_loss) ∧ 
    (abs (P - 20.04) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_stock_profit_percentage_l4094_409405


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4094_409491

theorem polynomial_factorization (x : ℝ) :
  4 * (x + 3) * (x + 7) * (x + 8) * (x + 12) - 5 * x^2 =
  (2 * x^2 + (60 - Real.sqrt 5) * x + 80) * (2 * x^2 + (60 + Real.sqrt 5) * x + 80) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4094_409491


namespace NUMINAMATH_CALUDE_rectangular_garden_area_rectangular_garden_area_proof_l4094_409465

/-- A rectangular garden with length three times its width and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_garden_area : ℝ → Prop :=
  fun w : ℝ =>
    w > 0 →                   -- width is positive
    2 * (w + 3 * w) = 72 →    -- perimeter is 72 meters
    w * (3 * w) = 243         -- area is 243 square meters

/-- Proof of the rectangular_garden_area theorem -/
theorem rectangular_garden_area_proof : ∃ w : ℝ, rectangular_garden_area w :=
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_rectangular_garden_area_proof_l4094_409465


namespace NUMINAMATH_CALUDE_x_coordinate_difference_on_line_l4094_409480

/-- Prove that for any two points on the line x = 4y + 5, where the y-coordinates differ by 0.5,
    the difference between their x-coordinates is 2. -/
theorem x_coordinate_difference_on_line (m n : ℝ) : 
  (m = 4 * n + 5) → 
  (∃ x, x = 4 * (n + 0.5) + 5) → 
  (∃ x, x - m = 2) :=
by sorry

end NUMINAMATH_CALUDE_x_coordinate_difference_on_line_l4094_409480


namespace NUMINAMATH_CALUDE_g_inverse_of_f_l4094_409402

-- Define the original function f
def f (x : ℝ) : ℝ := 4 - 5 * x^2

-- Define the inverse function g
def g (x : ℝ) : Set ℝ := {y : ℝ | y^2 = (4 - x) / 5 ∧ y ≥ 0 ∨ y^2 = (4 - x) / 5 ∧ y < 0}

-- Theorem stating that g is the inverse of f
theorem g_inverse_of_f : 
  ∀ x ∈ Set.range f, ∀ y ∈ g x, f y = x ∧ y ∈ Set.range f :=
sorry

end NUMINAMATH_CALUDE_g_inverse_of_f_l4094_409402


namespace NUMINAMATH_CALUDE_necessary_condition_transitivity_l4094_409468

theorem necessary_condition_transitivity (A B C : Prop) :
  (B → C) → (A → B) → (A → C) := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_transitivity_l4094_409468


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4094_409434

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + m = 0) → m ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4094_409434


namespace NUMINAMATH_CALUDE_largest_product_l4094_409497

def digits : List Nat := [5, 6, 7, 8]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n / 10) ∈ digits ∧ (n % 10) ∈ digits

def valid_pair (a b : Nat) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧
  (a / 10 ≠ b / 10) ∧ (a / 10 ≠ b % 10) ∧ (a % 10 ≠ b / 10) ∧ (a % 10 ≠ b % 10)

theorem largest_product :
  ∀ a b : Nat, valid_pair a b → a * b ≤ 3886 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_l4094_409497


namespace NUMINAMATH_CALUDE_hexagon_painting_arrangements_l4094_409484

/-- The number of ways to paint a hexagonal arrangement of equilateral triangles -/
def paint_arrangements : ℕ := 3^6 * 2^6

/-- The hexagonal arrangement consists of 6 inner sticks -/
def inner_sticks : ℕ := 6

/-- The number of available colors -/
def colors : ℕ := 3

/-- The number of triangles in the hexagonal arrangement -/
def triangles : ℕ := 6

/-- The number of ways to paint the inner sticks -/
def inner_stick_arrangements : ℕ := colors^inner_sticks

/-- The number of ways to complete each triangle given the two-color constraint -/
def triangle_completions : ℕ := 2^triangles

theorem hexagon_painting_arrangements :
  paint_arrangements = inner_stick_arrangements * triangle_completions :=
by sorry

end NUMINAMATH_CALUDE_hexagon_painting_arrangements_l4094_409484


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l4094_409455

theorem tan_sum_reciprocal (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 1)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4)
  (h3 : Real.tan x * Real.tan y = 1/3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l4094_409455


namespace NUMINAMATH_CALUDE_bucket_size_calculation_l4094_409448

/-- Given a leak rate and maximum time away, calculate the required bucket size -/
theorem bucket_size_calculation (leak_rate : ℝ) (max_time : ℝ) 
  (h1 : leak_rate = 1.5)
  (h2 : max_time = 12)
  (h3 : leak_rate > 0)
  (h4 : max_time > 0) :
  2 * (leak_rate * max_time) = 36 :=
by sorry

end NUMINAMATH_CALUDE_bucket_size_calculation_l4094_409448


namespace NUMINAMATH_CALUDE_halfway_fraction_l4094_409440

theorem halfway_fraction (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) :
  (a + b) / 2 = 19/24 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l4094_409440


namespace NUMINAMATH_CALUDE_cyclist_catchup_time_l4094_409413

/-- Two cyclists A and B travel from station A to station B -/
structure Cyclist where
  speed : ℝ
  startTime : ℝ

/-- The problem setup -/
def cyclistProblem (A B : Cyclist) (distance : ℝ) : Prop :=
  A.speed * 30 = distance ∧  -- A takes 30 minutes to reach station B
  B.speed * 40 = distance ∧  -- B takes 40 minutes to reach station B
  B.startTime = A.startTime - 5  -- B starts 5 minutes earlier than A

/-- The theorem to prove -/
theorem cyclist_catchup_time (A B : Cyclist) (distance : ℝ) 
  (h : cyclistProblem A B distance) : 
  ∃ t : ℝ, t = 15 ∧ A.speed * t = B.speed * (t + 5) :=
sorry

end NUMINAMATH_CALUDE_cyclist_catchup_time_l4094_409413


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l4094_409481

/-- Given that -x^2 + bx - 4 < 0 only when x ∈ (-∞, 0) ∪ (4, ∞), prove that b = 4 -/
theorem quadratic_inequality_roots (b : ℝ) 
  (h : ∀ x : ℝ, (-x^2 + b*x - 4 < 0) ↔ (x < 0 ∨ x > 4)) : 
  b = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l4094_409481


namespace NUMINAMATH_CALUDE_square_root_statements_l4094_409476

theorem square_root_statements :
  (Real.sqrt 2 * Real.sqrt 5 = Real.sqrt 10) ∧
  (Real.sqrt 2 + Real.sqrt 5 ≠ Real.sqrt 7) ∧
  (Real.sqrt 18 / Real.sqrt 2 = 3) ∧
  (Real.sqrt 12 = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_statements_l4094_409476


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l4094_409430

theorem quadratic_equations_solutions :
  (∀ x, x^2 - 7*x - 18 = 0 ↔ x = 9 ∨ x = -2) ∧
  (∀ x, 4*x^2 + 1 = 4*x ↔ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l4094_409430


namespace NUMINAMATH_CALUDE_goat_price_calculation_l4094_409409

theorem goat_price_calculation (total_cost total_hens total_goats hen_price : ℕ) 
  (h1 : total_cost = 10000)
  (h2 : total_hens = 35)
  (h3 : total_goats = 15)
  (h4 : hen_price = 125) :
  (total_cost - total_hens * hen_price) / total_goats = 375 := by
  sorry

end NUMINAMATH_CALUDE_goat_price_calculation_l4094_409409


namespace NUMINAMATH_CALUDE_initial_mask_sets_l4094_409471

/-- The number of mask sets Alicia gave away -/
def given_away : ℕ := 51

/-- The number of mask sets Alicia had left -/
def left : ℕ := 39

/-- The initial number of mask sets in Alicia's collection -/
def initial : ℕ := given_away + left

/-- Theorem stating that the initial number of mask sets is 90 -/
theorem initial_mask_sets : initial = 90 := by
  sorry

end NUMINAMATH_CALUDE_initial_mask_sets_l4094_409471


namespace NUMINAMATH_CALUDE_x_y_not_congruent_l4094_409457

def x : ℕ → ℕ
  | 0 => 365
  | n + 1 => x n * (x n ^ 1986 + 1) + 1622

def y : ℕ → ℕ
  | 0 => 16
  | n + 1 => y n * (y n ^ 3 + 1) - 1952

theorem x_y_not_congruent (n k : ℕ) : x n % 1987 ≠ y k % 1987 := by
  sorry

end NUMINAMATH_CALUDE_x_y_not_congruent_l4094_409457


namespace NUMINAMATH_CALUDE_expression_evaluation_l4094_409453

theorem expression_evaluation (x : ℝ) (h1 : x^2 - 3*x + 2 = 0) (h2 : x ≠ 2) :
  (x^2 / (x - 2) - x - 2) / (4*x / (x^2 - 4)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4094_409453


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_solution_l4094_409438

theorem complex_magnitude_equation_solution :
  ∃ x : ℝ, x > 0 ∧ 
  Complex.abs (x + Complex.I * Real.sqrt 7) * Complex.abs (3 - 2 * Complex.I * Real.sqrt 5) = 45 ∧
  x = Real.sqrt (1822 / 29) :=
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_solution_l4094_409438


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4094_409419

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- Properties of the specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 6 > seq.S 7 ∧ seq.S 7 > seq.S 5

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  seq.d < 0 ∧ 
  seq.S 11 > 0 ∧ 
  seq.S 12 > 0 ∧ 
  seq.S 8 < seq.S 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4094_409419


namespace NUMINAMATH_CALUDE_wattage_increase_percentage_l4094_409490

theorem wattage_increase_percentage (original_wattage new_wattage : ℝ) 
  (h1 : original_wattage = 110)
  (h2 : new_wattage = 143) : 
  (new_wattage - original_wattage) / original_wattage * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_wattage_increase_percentage_l4094_409490


namespace NUMINAMATH_CALUDE_equal_probabilities_l4094_409404

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  green : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  red_box : Box
  green_box : Box

/-- Initial state of the boxes -/
def initial_state : BoxState :=
  { red_box := { red := 100, green := 0 },
    green_box := { red := 0, green := 100 } }

/-- State after the first transfer -/
def first_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red - 8, green := state.red_box.green },
    green_box := { red := state.green_box.red + 8, green := state.green_box.green } }

/-- State after the second transfer -/
def second_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red, green := state.red_box.green + 1 },
    green_box := { red := state.green_box.red - 1, green := state.green_box.green - 7 } }

/-- Calculate the probability of drawing a specific color from a box -/
def draw_probability (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => (box.red : ℚ) / (box.red + box.green : ℚ)
  | "green" => (box.green : ℚ) / (box.red + box.green : ℚ)
  | _ => 0

/-- The main theorem to prove -/
theorem equal_probabilities :
  let final_state := second_transfer (first_transfer initial_state)
  (draw_probability final_state.red_box "green") = (draw_probability final_state.green_box "red") := by
  sorry

end NUMINAMATH_CALUDE_equal_probabilities_l4094_409404


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l4094_409479

theorem positive_numbers_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a^2 + b^2 - a*b = c^2) : 
  (a - c) * (b - c) ≤ 0 := by sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l4094_409479


namespace NUMINAMATH_CALUDE_fourth_week_miles_l4094_409445

-- Define the number of weeks
def num_weeks : ℕ := 4

-- Define the number of days walked per week
def days_per_week : ℕ := 6

-- Define the miles walked per day for each week
def miles_per_day (week : ℕ) : ℕ :=
  if week < 4 then week else 0  -- The 4th week is unknown, so we set it to 0 initially

-- Define the total miles walked
def total_miles : ℕ := 60

-- Theorem to prove
theorem fourth_week_miles :
  ∃ (x : ℕ), 
    (miles_per_day 1 * days_per_week +
     miles_per_day 2 * days_per_week +
     miles_per_day 3 * days_per_week +
     x * days_per_week = total_miles) ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_week_miles_l4094_409445


namespace NUMINAMATH_CALUDE_john_average_change_l4094_409415

def john_scores : List ℝ := [84, 88, 95, 92]

theorem john_average_change : 
  let initial_average := (john_scores.take 3).sum / 3
  let new_average := john_scores.sum / 4
  new_average - initial_average = 0.75 := by sorry

end NUMINAMATH_CALUDE_john_average_change_l4094_409415


namespace NUMINAMATH_CALUDE_scaling_transformation_theorem_l4094_409403

/-- Scaling transformation -/
def scaling (x y : ℝ) : ℝ × ℝ :=
  (2 * x, 3 * y)

/-- Transformed curve equation -/
def transformed_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

/-- Original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  4 * x^2 + 9 * y^2 = 1

/-- Theorem: If the transformed curve satisfies the equation,
    then the original curve satisfies its corresponding equation -/
theorem scaling_transformation_theorem :
  ∀ x y : ℝ,
  let (x'', y'') := scaling x y
  transformed_curve x'' y'' → original_curve x y :=
by
  sorry

end NUMINAMATH_CALUDE_scaling_transformation_theorem_l4094_409403


namespace NUMINAMATH_CALUDE_intersection_of_P_and_complement_of_M_l4094_409470

-- Define the universal set U as integers
def U : Set Int := Set.univ

-- Define set M
def M : Set Int := {1, 2}

-- Define set P
def P : Set Int := {x : Int | |x| ≤ 2}

-- State the theorem
theorem intersection_of_P_and_complement_of_M :
  P ∩ (U \ M) = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_complement_of_M_l4094_409470


namespace NUMINAMATH_CALUDE_slope_intercept_product_specific_line_l4094_409437

/-- A line in a cartesian plane. -/
structure Line where
  /-- The slope of the line. -/
  slope : ℝ
  /-- The y-intercept of the line. -/
  y_intercept : ℝ

/-- The product of the slope and y-intercept of a line. -/
def slope_intercept_product (l : Line) : ℝ := l.slope * l.y_intercept

/-- Theorem: For a line with y-intercept -3 and slope 3, the product of its slope and y-intercept is -9. -/
theorem slope_intercept_product_specific_line :
  ∃ (l : Line), l.y_intercept = -3 ∧ l.slope = 3 ∧ slope_intercept_product l = -9 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_product_specific_line_l4094_409437


namespace NUMINAMATH_CALUDE_savings_calculation_l4094_409427

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
def calculate_savings (income : ℚ) (income_ratio : ℚ) (expenditure_ratio : ℚ) : ℚ :=
  income - (income * expenditure_ratio / income_ratio)

/-- Proves that for a given income and income-to-expenditure ratio, the savings are as calculated -/
theorem savings_calculation (income : ℚ) (income_ratio : ℚ) (expenditure_ratio : ℚ) :
  income = 20000 ∧ income_ratio = 4 ∧ expenditure_ratio = 3 →
  calculate_savings income income_ratio expenditure_ratio = 5000 :=
by sorry

end NUMINAMATH_CALUDE_savings_calculation_l4094_409427


namespace NUMINAMATH_CALUDE_snarks_are_twerks_and_quarks_l4094_409423

variable (U : Type) -- Universe of discourse

-- Define the predicates
variable (Snark Garble Twerk Quark : U → Prop)

-- State the given conditions
variable (h1 : ∀ x, Snark x → Garble x)
variable (h2 : ∀ x, Twerk x → Garble x)
variable (h3 : ∀ x, Snark x → Quark x)
variable (h4 : ∀ x, Quark x → Twerk x)

-- State the theorem to be proved
theorem snarks_are_twerks_and_quarks :
  ∀ x, Snark x → Twerk x ∧ Quark x := by sorry

end NUMINAMATH_CALUDE_snarks_are_twerks_and_quarks_l4094_409423


namespace NUMINAMATH_CALUDE_eighth_group_student_number_l4094_409436

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  students_per_group : Nat
  selected_student : Nat
  selected_group : Nat

/-- Calculates the number of the student selected from a given group -/
def student_number_in_group (s : SystematicSampling) (group : Nat) : Nat :=
  s.selected_student + (group - s.selected_group) * s.students_per_group

/-- Theorem stating the correct student number in the eighth group -/
theorem eighth_group_student_number 
  (s : SystematicSampling) 
  (h1 : s.total_students = 50)
  (h2 : s.num_groups = 10)
  (h3 : s.students_per_group = 5)
  (h4 : s.selected_student = 12)
  (h5 : s.selected_group = 3) :
  student_number_in_group s 8 = 37 := by
  sorry


end NUMINAMATH_CALUDE_eighth_group_student_number_l4094_409436
