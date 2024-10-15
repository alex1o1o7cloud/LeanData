import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l804_80428

theorem quadratic_equation_transformation (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -3 ∧ ∀ x : ℝ, x^2 + p*x + q = 0 ↔ (x = x₁ ∨ x = x₂)) →
  ∀ x : ℝ, x^2 + p*x + q = 0 ↔ (x - 2)*(x + 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l804_80428


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l804_80416

theorem similar_triangles_leg_sum (a₁ a₂ : ℝ) (s : ℝ) :
  a₁ > 0 → a₂ > 0 → s > 0 →
  a₁ = 8 → a₂ = 200 → -- areas of the triangles
  s = 2 → -- shorter leg of smaller triangle
  ∃ (l₁ l₂ : ℝ), 
    l₁ > 0 ∧ l₂ > 0 ∧
    a₁ = (1/2) * s * l₁ ∧ -- area of smaller triangle
    a₂ = (1/2) * (5*s) * (5*l₁) ∧ -- area of larger triangle
    l₁ + l₂ = 50 -- sum of legs of larger triangle
  := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l804_80416


namespace NUMINAMATH_CALUDE_spot_difference_l804_80418

theorem spot_difference (granger cisco rover : ℕ) : 
  granger = 5 * cisco →
  granger + cisco = 108 →
  rover = 46 →
  cisco < rover / 2 →
  rover / 2 - cisco = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_spot_difference_l804_80418


namespace NUMINAMATH_CALUDE_inverse_variation_with_increase_l804_80401

/-- Given two inversely varying quantities a and b, prove that when their product increases by 50% and a becomes 1600, b equals 0.375 -/
theorem inverse_variation_with_increase (a b a' b' : ℝ) : 
  (a * b = 800 * 0.5) →  -- Initial condition
  (a' * b' = 1.5 * (a * b)) →  -- 50% increase in product
  (a' = 1600) →  -- New value of a
  (b' = 0.375) :=  -- Theorem to prove
by sorry

end NUMINAMATH_CALUDE_inverse_variation_with_increase_l804_80401


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l804_80450

theorem geometric_progression_first_term (S : ℝ) (sum_first_two : ℝ) 
  (h1 : S = 8) (h2 : sum_first_two = 5) :
  ∃ (a : ℝ), (a = 8 * (1 - Real.sqrt 6 / 4) ∨ a = 8 * (1 + Real.sqrt 6 / 4)) ∧
    (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l804_80450


namespace NUMINAMATH_CALUDE_car_length_is_113_steps_l804_80414

/-- Represents the scenario of a person jogging alongside a moving car --/
structure CarJoggingScenario where
  /-- The length of the car in terms of the jogger's steps --/
  car_length : ℝ
  /-- The distance the car moves during one step of the jogger --/
  car_step : ℝ
  /-- The number of steps counted when jogging from rear to front --/
  steps_rear_to_front : ℕ
  /-- The number of steps counted when jogging from front to rear --/
  steps_front_to_rear : ℕ
  /-- The car is moving faster than the jogger --/
  car_faster : car_step > 0
  /-- The car has a positive length --/
  car_positive : car_length > 0
  /-- Equation for jogging from rear to front --/
  eq_rear_to_front : (steps_rear_to_front : ℝ) = car_length / car_step + steps_rear_to_front
  /-- Equation for jogging from front to rear --/
  eq_front_to_rear : (steps_front_to_rear : ℝ) = car_length / car_step - steps_front_to_rear

/-- The length of the car is 113 steps when jogging 150 steps rear to front and 30 steps front to rear --/
theorem car_length_is_113_steps (scenario : CarJoggingScenario) 
  (h1 : scenario.steps_rear_to_front = 150) 
  (h2 : scenario.steps_front_to_rear = 30) : 
  scenario.car_length = 113 := by
  sorry

end NUMINAMATH_CALUDE_car_length_is_113_steps_l804_80414


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l804_80476

theorem solution_set_equivalence (x : ℝ) :
  (1 - |x|) * (1 + x) > 0 ↔ x < 1 ∧ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l804_80476


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_23_l804_80452

theorem modular_inverse_17_mod_23 :
  (∃ x : ℤ, (11 * x) % 23 = 1) →
  (∃ y : ℤ, (17 * y) % 23 = 1 ∧ 0 ≤ y ∧ y ≤ 22) ∧
  (∀ z : ℤ, (17 * z) % 23 = 1 → z % 23 = 19) :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_23_l804_80452


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l804_80468

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 2 ∧ 
  n % 6 = 3 ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → m ≥ n) ∧
  n = 57 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l804_80468


namespace NUMINAMATH_CALUDE_jenny_stamps_last_page_l804_80410

/-- Represents the stamp collection system -/
structure StampCollection where
  initialBooks : ℕ
  pagesPerBook : ℕ
  initialStampsPerPage : ℕ
  newStampsPerPage : ℕ
  filledBooks : ℕ
  filledPagesInLastBook : ℕ

/-- Calculates the number of stamps on the last page after reorganization -/
def stampsOnLastPage (sc : StampCollection) : ℕ :=
  let totalStamps := sc.initialBooks * sc.pagesPerBook * sc.initialStampsPerPage
  let filledPages := sc.filledBooks * sc.pagesPerBook + sc.filledPagesInLastBook
  totalStamps - (filledPages * sc.newStampsPerPage)

/-- Theorem: Given Jenny's stamp collection details, the last page contains 8 stamps -/
theorem jenny_stamps_last_page :
  let sc : StampCollection := {
    initialBooks := 10,
    pagesPerBook := 50,
    initialStampsPerPage := 6,
    newStampsPerPage := 8,
    filledBooks := 6,
    filledPagesInLastBook := 45
  }
  stampsOnLastPage sc = 8 := by
  sorry

end NUMINAMATH_CALUDE_jenny_stamps_last_page_l804_80410


namespace NUMINAMATH_CALUDE_log_2_irrational_l804_80422

theorem log_2_irrational : Irrational (Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_2_irrational_l804_80422


namespace NUMINAMATH_CALUDE_triangle_problem_l804_80426

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0)
  (h2 : t.c = 2)
  (h3 : t.a + t.b = t.a * t.b) :
  t.C = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l804_80426


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l804_80472

-- Define the given speeds
def speed_with_current : ℝ := 15
def current_speed : ℝ := 2.8

-- Define the speed against the current
def speed_against_current : ℝ := speed_with_current - 2 * current_speed

-- Theorem statement
theorem mans_speed_against_current :
  speed_against_current = 9.4 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l804_80472


namespace NUMINAMATH_CALUDE_line_intersects_plane_implies_skew_line_exists_l804_80433

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate to check if a line intersects a plane -/
def intersects (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Predicate to check if a line is within a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Predicate to check if two lines are skew -/
def skew (l1 l2 : Line3D) : Prop := sorry

/-- Main theorem -/
theorem line_intersects_plane_implies_skew_line_exists (l : Line3D) (α : Plane3D) :
  intersects l α → ∃ m : Line3D, line_in_plane m α ∧ skew l m := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_plane_implies_skew_line_exists_l804_80433


namespace NUMINAMATH_CALUDE_abby_damon_weight_l804_80436

/-- The weights of four people satisfying certain conditions -/
structure Weights where
  a : ℝ  -- Abby's weight
  b : ℝ  -- Bart's weight
  c : ℝ  -- Cindy's weight
  d : ℝ  -- Damon's weight
  ab_sum : a + b = 300
  bc_sum : b + c = 280
  cd_sum : c + d = 290
  ac_bd_diff : a + c = b + d + 10

/-- Theorem stating that given the conditions, Abby and Damon's combined weight is 310 pounds -/
theorem abby_damon_weight (w : Weights) : w.a + w.d = 310 := by
  sorry

end NUMINAMATH_CALUDE_abby_damon_weight_l804_80436


namespace NUMINAMATH_CALUDE_line_parameterization_l804_80458

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 4 * x - 9

-- Define the parameterization
def parameterization (x y s p t : ℝ) : Prop :=
  x = s + 5 * t ∧ y = 3 + p * t

-- Theorem statement
theorem line_parameterization (s p : ℝ) :
  (∀ x y t : ℝ, line_equation x y ∧ parameterization x y s p t) →
  s = 3 ∧ p = 20 := by sorry

end NUMINAMATH_CALUDE_line_parameterization_l804_80458


namespace NUMINAMATH_CALUDE_rug_strip_width_l804_80424

/-- Given a rectangular floor and a rug, proves that the width of the uncovered strip is 2 meters -/
theorem rug_strip_width (floor_length floor_width rug_area : ℝ) 
  (h1 : floor_length = 10) 
  (h2 : floor_width = 8) 
  (h3 : rug_area = 24) : 
  ∃ w : ℝ, w > 0 ∧ w < floor_width / 2 ∧ 
  (floor_length - 2 * w) * (floor_width - 2 * w) = rug_area ∧ 
  w = 2 :=
sorry

end NUMINAMATH_CALUDE_rug_strip_width_l804_80424


namespace NUMINAMATH_CALUDE_total_time_conversion_l804_80494

/-- Given 3450 minutes and 7523 seconds, prove that the total time is 59 hours, 35 minutes, and 23 seconds. -/
theorem total_time_conversion (minutes : ℕ) (seconds : ℕ) : 
  minutes = 3450 ∧ seconds = 7523 → 
  ∃ (hours : ℕ) (remaining_minutes : ℕ) (remaining_seconds : ℕ),
    hours = 59 ∧ 
    remaining_minutes = 35 ∧ 
    remaining_seconds = 23 ∧
    minutes * 60 + seconds = hours * 3600 + remaining_minutes * 60 + remaining_seconds :=
by sorry

end NUMINAMATH_CALUDE_total_time_conversion_l804_80494


namespace NUMINAMATH_CALUDE_set_equality_l804_80431

theorem set_equality (x y : ℝ) : 
  (x^2 - y^2 = x / (x^2 + y^2) ∧ 2*x*y + y / (x^2 + y^2) = 3) ↔ 
  (x^3 - 3*x*y^2 + 3*y = 1 ∧ 3*x^2*y - 3*x - y^3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l804_80431


namespace NUMINAMATH_CALUDE_door_cost_ratio_l804_80493

theorem door_cost_ratio (bedroom_doors : ℕ) (outside_doors : ℕ) 
  (outside_door_cost : ℚ) (total_cost : ℚ) :
  bedroom_doors = 3 →
  outside_doors = 2 →
  outside_door_cost = 20 →
  total_cost = 70 →
  ∃ (bedroom_door_cost : ℚ),
    bedroom_doors * bedroom_door_cost + outside_doors * outside_door_cost = total_cost ∧
    bedroom_door_cost / outside_door_cost = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_door_cost_ratio_l804_80493


namespace NUMINAMATH_CALUDE_khali_snow_volume_l804_80454

/-- Calculates the total volume of snow on a rectangular sidewalk with two layers -/
def total_snow_volume (length width depth1 depth2 : ℝ) : ℝ :=
  length * width * (depth1 + depth2)

/-- Theorem: The total volume of snow on Khali's sidewalk is 90 cubic feet -/
theorem khali_snow_volume :
  let length : ℝ := 30
  let width : ℝ := 3
  let depth1 : ℝ := 0.6
  let depth2 : ℝ := 0.4
  total_snow_volume length width depth1 depth2 = 90 := by
  sorry

#eval total_snow_volume 30 3 0.6 0.4

end NUMINAMATH_CALUDE_khali_snow_volume_l804_80454


namespace NUMINAMATH_CALUDE_person_age_puzzle_l804_80499

theorem person_age_puzzle (x : ℝ) : 4 * (x + 3) - 4 * (x - 3) = x ↔ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l804_80499


namespace NUMINAMATH_CALUDE_two_digit_number_patterns_l804_80467

theorem two_digit_number_patterns 
  (a m n : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hm : 0 < m ∧ m < 10) 
  (hn : 0 < n ∧ n < 10) : 
  ((10 * a + 5) ^ 2 = 100 * a * (a + 1) + 25) ∧ 
  ((10 * m + n) * (10 * m + (10 - n)) = 100 * m * (m + 1) + n * (10 - n)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_patterns_l804_80467


namespace NUMINAMATH_CALUDE_sock_ratio_is_two_elevenths_l804_80486

/-- Represents the sock order problem -/
structure SockOrder where
  blackPairs : ℕ
  bluePairs : ℕ
  blackPrice : ℝ
  bluePrice : ℝ

/-- The original sock order -/
def originalOrder : SockOrder :=
  { blackPairs := 6,
    bluePairs := 0,  -- This will be determined
    blackPrice := 0, -- This will be determined
    bluePrice := 0   -- This will be determined
  }

/-- The interchanged sock order -/
def interchangedOrder (o : SockOrder) : SockOrder :=
  { blackPairs := o.bluePairs,
    bluePairs := o.blackPairs,
    blackPrice := o.blackPrice,
    bluePrice := o.bluePrice
  }

/-- Calculate the total cost of a sock order -/
def totalCost (o : SockOrder) : ℝ :=
  o.blackPairs * o.blackPrice + o.bluePairs * o.bluePrice

/-- The theorem stating the ratio of black to blue socks -/
theorem sock_ratio_is_two_elevenths :
  ∃ (o : SockOrder),
    o.blackPairs = 6 ∧
    o.blackPrice = 2 * o.bluePrice ∧
    totalCost (interchangedOrder o) = 1.6 * totalCost o ∧
    o.blackPairs / o.bluePairs = 2 / 11 :=
  sorry

end NUMINAMATH_CALUDE_sock_ratio_is_two_elevenths_l804_80486


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l804_80411

/-- Three vectors in ℝ³ -/
def a : Fin 3 → ℝ := ![3, 7, 2]
def b : Fin 3 → ℝ := ![-2, 0, -1]
def c : Fin 3 → ℝ := ![2, 2, 1]

/-- Scalar triple product of three vectors in ℝ³ -/
def scalarTripleProduct (u v w : Fin 3 → ℝ) : ℝ :=
  Matrix.det !![u 0, u 1, u 2; v 0, v 1, v 2; w 0, w 1, w 2]

/-- Theorem: The vectors a, b, and c are not coplanar -/
theorem vectors_not_coplanar : scalarTripleProduct a b c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l804_80411


namespace NUMINAMATH_CALUDE_boxes_with_neither_l804_80402

theorem boxes_with_neither (total_boxes : ℕ) (marker_boxes : ℕ) (crayon_boxes : ℕ) (both_boxes : ℕ) 
  (h1 : total_boxes = 15)
  (h2 : marker_boxes = 9)
  (h3 : crayon_boxes = 5)
  (h4 : both_boxes = 4) :
  total_boxes - (marker_boxes + crayon_boxes - both_boxes) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l804_80402


namespace NUMINAMATH_CALUDE_farmer_brown_additional_cost_l804_80405

/-- The additional cost for Farmer Brown's new hay requirements -/
theorem farmer_brown_additional_cost :
  let original_bales : ℕ := 10
  let original_cost_per_bale : ℕ := 15
  let new_bales : ℕ := 2 * original_bales
  let new_cost_per_bale : ℕ := 18
  (new_bales * new_cost_per_bale) - (original_bales * original_cost_per_bale) = 210 :=
by sorry

end NUMINAMATH_CALUDE_farmer_brown_additional_cost_l804_80405


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l804_80437

theorem smallest_k_no_real_roots : 
  let f (k : ℤ) (x : ℝ) := 3 * x * (k * x - 5) - x^2 + 4
  ∀ k : ℤ, (∀ x : ℝ, f k x ≠ 0) → k ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l804_80437


namespace NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l804_80465

/-- Represents a 12-hour digital clock with a display error -/
structure ErrorClock where
  /-- The clock displays '5' instead of '2' -/
  display_error : ℕ → ℕ
  display_error_def : ∀ n, display_error n = if n = 2 then 5 else n

/-- The fraction of the day when the clock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : ℚ :=
  5/8

theorem error_clock_correct_time_fraction (clock : ErrorClock) :
  correct_time_fraction clock = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l804_80465


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l804_80475

theorem coefficient_x3y5_in_expansion_of_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * (1 : ℕ)^(8 - k) * (1 : ℕ)^k) = 256 ∧
  (Nat.choose 8 5) = 56 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l804_80475


namespace NUMINAMATH_CALUDE_remainder_theorem_l804_80489

def polynomial (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 6*x^4 - x^3 + x^2 - 15

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ), 
    polynomial x = (divisor x) * q x + polynomial 3 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l804_80489


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l804_80407

theorem sqrt_eight_div_sqrt_two_equals_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l804_80407


namespace NUMINAMATH_CALUDE_division_remainder_proof_l804_80443

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 997)
  (h2 : divisor = 23)
  (h3 : quotient = 43)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 8 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l804_80443


namespace NUMINAMATH_CALUDE_cos_two_pi_seventh_inequality_l804_80496

theorem cos_two_pi_seventh_inequality (a : ℝ) :
  a = Real.cos ((2 * Real.pi) / 7) →
  0 < (1 : ℝ) / 2 ∧ (1 : ℝ) / 2 < a ∧ a < Real.sqrt 2 / 2 ∧ Real.sqrt 2 / 2 < 1 →
  2^(a - 1/2) < 2 * a := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_seventh_inequality_l804_80496


namespace NUMINAMATH_CALUDE_special_calculator_problem_l804_80429

-- Define a function to reverse digits of a number
def reverse_digits (n : ℕ) : ℕ := sorry

-- Define the calculator operation
def calculator_operation (x : ℕ) : ℕ := reverse_digits (2 * x) + 2

-- Theorem statement
theorem special_calculator_problem (x : ℕ) :
  x ≥ 10 ∧ x < 100 →  -- two-digit number condition
  calculator_operation x = 45 →
  x = 17 := by sorry

end NUMINAMATH_CALUDE_special_calculator_problem_l804_80429


namespace NUMINAMATH_CALUDE_simplify_sqrt_m3n2_l804_80459

theorem simplify_sqrt_m3n2 (m n : ℝ) (hm : m > 0) (hn : n < 0) :
  Real.sqrt (m^3 * n^2) = -m * n * Real.sqrt m := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_m3n2_l804_80459


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l804_80408

/-- The probability of exactly k successes in n trials of a Bernoulli experiment -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of exactly 7 tails in 10 flips of an unfair coin -/
theorem unfair_coin_probability : 
  binomialProbability 10 7 (2/3) = 512/6561 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l804_80408


namespace NUMINAMATH_CALUDE_relationship_abc_l804_80462

-- Define the constants
noncomputable def a : ℝ := Real.pi ^ (1/3)
noncomputable def b : ℝ := (Real.log 3) / (Real.log Real.pi)
noncomputable def c : ℝ := Real.log (Real.sqrt 3 - 1)

-- State the theorem
theorem relationship_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l804_80462


namespace NUMINAMATH_CALUDE_ball_probability_l804_80409

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ)
  (h_total : total = 60)
  (h_white : white = 22)
  (h_green : green = 10)
  (h_yellow : yellow = 7)
  (h_red : red = 15)
  (h_purple : purple = 6)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 13 / 20 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l804_80409


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l804_80415

theorem wrapping_paper_fraction (total_used : ℚ) (num_presents : ℕ) (h1 : total_used = 1/2) (h2 : num_presents = 5) :
  total_used / num_presents = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l804_80415


namespace NUMINAMATH_CALUDE_marias_gum_count_l804_80456

/-- 
Given:
- Maria initially had 25 pieces of gum
- Tommy gave her 16 more pieces
- Luis gave her 20 more pieces

Prove that Maria now has 61 pieces of gum
-/
theorem marias_gum_count (initial : ℕ) (tommy : ℕ) (luis : ℕ) 
  (h1 : initial = 25)
  (h2 : tommy = 16)
  (h3 : luis = 20) :
  initial + tommy + luis = 61 := by
  sorry

end NUMINAMATH_CALUDE_marias_gum_count_l804_80456


namespace NUMINAMATH_CALUDE_parallel_condition_l804_80463

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The condition ab = 1 -/
def condition (l1 l2 : Line) : Prop :=
  l1.a * l2.b = 1

theorem parallel_condition (l1 l2 : Line) :
  (parallel l1 l2 → condition l1 l2) ∧
  ¬(condition l1 l2 → parallel l1 l2) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l804_80463


namespace NUMINAMATH_CALUDE_perpendicular_vectors_y_value_l804_80432

theorem perpendicular_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) : 
  a = (2, 4) → 
  b = (-4, y) → 
  a.1 * b.1 + a.2 * b.2 = 0 → 
  y = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_y_value_l804_80432


namespace NUMINAMATH_CALUDE_amelia_win_probability_l804_80419

/-- Represents the outcome of a single coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents a player in the game -/
inductive Player
| Amelia
| Blaine

/-- The state of the game after each round -/
structure GameState :=
  (round : Nat)
  (currentPlayer : Player)

/-- The result of the game -/
inductive GameResult
| AmeliaWins
| BlaineWins
| Tie

/-- The probability of getting heads for each player -/
def headsProbability (player : Player) : ℚ :=
  match player with
  | Player.Amelia => 1/4
  | Player.Blaine => 1/3

/-- The probability of the game ending in a specific result -/
noncomputable def gameResultProbability (result : GameResult) : ℚ :=
  sorry

/-- The main theorem stating the probability of Amelia winning -/
theorem amelia_win_probability :
  gameResultProbability GameResult.AmeliaWins = 15/32 :=
sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l804_80419


namespace NUMINAMATH_CALUDE_simple_compound_interest_relation_l804_80446

/-- Calculates simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates compound interest (annually compounded) -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

theorem simple_compound_interest_relation :
  ∀ (y : ℝ),
    compound_interest 6000 (y / 100) 2 = 615 →
    simple_interest 6000 (y / 100) 2 = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_compound_interest_relation_l804_80446


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l804_80473

def first_ten_integers : Finset ℕ := Finset.range 10

theorem least_common_multiple_first_ten : ∃ (n : ℕ), n > 0 ∧ 
  (∀ i ∈ first_ten_integers, i.succ ∣ n) ∧ 
  (∀ m : ℕ, m > 0 → (∀ i ∈ first_ten_integers, i.succ ∣ m) → n ≤ m) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l804_80473


namespace NUMINAMATH_CALUDE_clothes_cost_calculation_l804_80457

def savings_june : ℕ := 21
def savings_july : ℕ := 46
def savings_august : ℕ := 45
def school_supplies_cost : ℕ := 12
def remaining_balance : ℕ := 46

def total_savings : ℕ := savings_june + savings_july + savings_august

def clothes_cost : ℕ := total_savings - school_supplies_cost - remaining_balance

theorem clothes_cost_calculation :
  clothes_cost = 54 :=
by sorry

end NUMINAMATH_CALUDE_clothes_cost_calculation_l804_80457


namespace NUMINAMATH_CALUDE_cars_produced_in_europe_l804_80488

theorem cars_produced_in_europe (total_cars : ℕ) (north_america_cars : ℕ) (europe_cars : ℕ) :
  total_cars = 6755 →
  north_america_cars = 3884 →
  total_cars = north_america_cars + europe_cars →
  europe_cars = 2871 :=
by sorry

end NUMINAMATH_CALUDE_cars_produced_in_europe_l804_80488


namespace NUMINAMATH_CALUDE_max_erased_dots_l804_80442

/-- Represents a domino tile with two halves -/
structure Domino :=
  (left : ℕ)
  (right : ℕ)

/-- The problem setup -/
def DominoArrangement :=
  { tiles : List Domino // tiles.length = 8 }

/-- The sum of dots on all visible tiles -/
def visibleDots (arr : DominoArrangement) : ℕ :=
  (arr.val.take 7).foldl (fun acc tile => acc + tile.left + tile.right) 0

/-- The total number of dots including the erased half -/
def totalDots (arr : DominoArrangement) (erased : ℕ) : ℕ :=
  visibleDots arr + erased

theorem max_erased_dots (arr : DominoArrangement) 
  (h1 : visibleDots arr = 37)
  (h2 : ∀ n : ℕ, totalDots arr n % 4 = 0 → n ≤ 3) :
  ∃ (n : ℕ), n ≤ 3 ∧ totalDots arr n % 4 = 0 ∧ 
    ∀ (m : ℕ), totalDots arr m % 4 = 0 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_max_erased_dots_l804_80442


namespace NUMINAMATH_CALUDE_least_positive_y_l804_80455

-- Define variables
variable (c d : ℝ)
variable (y : ℝ)

-- Define the conditions
def condition1 : Prop := Real.tan y = (2 * c) / (3 * d)
def condition2 : Prop := Real.tan (2 * y) = (3 * d) / (2 * c + 3 * d)

-- State the theorem
theorem least_positive_y (h1 : condition1 c d y) (h2 : condition2 c d y) :
  y = Real.arctan (1 / 3) ∧ ∀ z, 0 < z ∧ z < y → ¬(condition1 c d z ∧ condition2 c d z) :=
sorry

end NUMINAMATH_CALUDE_least_positive_y_l804_80455


namespace NUMINAMATH_CALUDE_length_of_QR_l804_80477

-- Define the right triangle PQR
def right_triangle_PQR (QR : ℝ) : Prop :=
  ∃ (P Q R : ℝ × ℝ),
    P = (0, 0) ∧  -- P is at the origin
    Q.1 = 12 ∧ Q.2 = 0 ∧  -- Q is on the horizontal axis, 12 units from P
    R.2 ≠ 0 ∧  -- R is not on the horizontal axis (to ensure a right triangle)
    (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = QR^2 ∧  -- Pythagorean theorem
    (R.1 - P.1)^2 + (R.2 - P.2)^2 = QR^2  -- Pythagorean theorem

-- State the theorem
theorem length_of_QR :
  ∀ QR : ℝ, right_triangle_PQR QR → Real.cos (Real.arccos 0.3) = 12 / QR → QR = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_length_of_QR_l804_80477


namespace NUMINAMATH_CALUDE_apartment_cost_ratio_l804_80481

/-- Proves that the ratio of room costs on the third floor to the first floor is 4/3 --/
theorem apartment_cost_ratio :
  ∀ (cost_floor1 cost_floor2 rooms_per_floor total_earnings : ℕ),
    cost_floor1 = 15 →
    cost_floor2 = 20 →
    rooms_per_floor = 3 →
    total_earnings = 165 →
    (total_earnings - (cost_floor1 * rooms_per_floor + cost_floor2 * rooms_per_floor)) / rooms_per_floor / cost_floor1 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_apartment_cost_ratio_l804_80481


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l804_80482

theorem sphere_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * r)^3) = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l804_80482


namespace NUMINAMATH_CALUDE_unique_solution_l804_80449

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y) = f (x * y^2) - 2 * x^2 * f y - f x - 1

theorem unique_solution (f : ℝ → ℝ) (h : functional_equation f) :
  ∀ y : ℝ, f y = y^2 - 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l804_80449


namespace NUMINAMATH_CALUDE_completing_square_transformation_l804_80466

theorem completing_square_transformation (x : ℝ) : 
  (x^2 + 6*x - 4 = 0) ↔ ((x + 3)^2 = 13) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l804_80466


namespace NUMINAMATH_CALUDE_debt_payment_problem_l804_80487

/-- Proves that the amount of each of the first 20 payments is $410 given the problem conditions. -/
theorem debt_payment_problem (total_payments : ℕ) (first_payments : ℕ) (payment_increase : ℕ) (average_payment : ℕ) :
  total_payments = 65 →
  first_payments = 20 →
  payment_increase = 65 →
  average_payment = 455 →
  ∃ (x : ℕ),
    x * first_payments + (x + payment_increase) * (total_payments - first_payments) = average_payment * total_payments ∧
    x = 410 :=
by sorry

end NUMINAMATH_CALUDE_debt_payment_problem_l804_80487


namespace NUMINAMATH_CALUDE_exists_n_with_factorial_property_and_digit_sum_l804_80412

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The main theorem -/
theorem exists_n_with_factorial_property_and_digit_sum :
  ∃ n : ℕ, n > 0 ∧ 
    (Nat.factorial (n + 1) + Nat.factorial (n + 2) = Nat.factorial n * 1001) ∧
    (sum_of_digits n = 3) := by
  sorry

end NUMINAMATH_CALUDE_exists_n_with_factorial_property_and_digit_sum_l804_80412


namespace NUMINAMATH_CALUDE_science_to_novel_ratio_l804_80421

/-- Given the page counts of different books, prove the ratio of science to novel pages --/
theorem science_to_novel_ratio :
  let history_pages : ℕ := 300
  let science_pages : ℕ := 600
  let novel_pages : ℕ := history_pages / 2
  science_pages / novel_pages = 4 := by
  sorry


end NUMINAMATH_CALUDE_science_to_novel_ratio_l804_80421


namespace NUMINAMATH_CALUDE_juice_price_ratio_l804_80451

theorem juice_price_ratio :
  ∀ (v_B p_B : ℝ), v_B > 0 → p_B > 0 →
  let v_A := 1.25 * v_B
  let p_A := 0.85 * p_B
  (p_A / v_A) / (p_B / v_B) = 17 / 25 := by
sorry

end NUMINAMATH_CALUDE_juice_price_ratio_l804_80451


namespace NUMINAMATH_CALUDE_g_16_value_l804_80439

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^4 - x^3 + x^2 - x + 1

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^4 + b*x^3 + c*x^2 + d*x + (-1)) ∧
  (g 0 = -1) ∧
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g (r^2) = 0)

-- Theorem statement
theorem g_16_value (g : ℝ → ℝ) (h : is_valid_g g) : g 16 = -69905 := by
  sorry

end NUMINAMATH_CALUDE_g_16_value_l804_80439


namespace NUMINAMATH_CALUDE_two_by_two_squares_count_l804_80495

theorem two_by_two_squares_count (grid_size : ℕ) (cuts : ℕ) (figures : ℕ) 
  (h1 : grid_size = 100)
  (h2 : cuts = 10000)
  (h3 : figures = 2500) : 
  ∃ (x : ℕ), x = 2300 ∧ 
  (8 * x + 10 * (figures - x) = 4 * grid_size + 2 * cuts) := by
  sorry

#check two_by_two_squares_count

end NUMINAMATH_CALUDE_two_by_two_squares_count_l804_80495


namespace NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_three_one_five_satisfies_least_multiple_with_digit_product_multiple_is_315_l804_80464

/-- Returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Returns true if n is a multiple of m -/
def isMultipleOf (n m : ℕ) : Prop := ∃ k, n = m * k

theorem least_multiple_with_digit_product_multiple : 
  ∀ n : ℕ, n > 0 → isMultipleOf n 15 → isMultipleOf (digitProduct n) 15 → n ≥ 315 := by sorry

theorem three_one_five_satisfies :
  isMultipleOf 315 15 ∧ isMultipleOf (digitProduct 315) 15 := by sorry

theorem least_multiple_with_digit_product_multiple_is_315 : 
  ∀ n : ℕ, n > 0 → isMultipleOf n 15 → isMultipleOf (digitProduct n) 15 → n = 315 ∨ n > 315 := by sorry

end NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_three_one_five_satisfies_least_multiple_with_digit_product_multiple_is_315_l804_80464


namespace NUMINAMATH_CALUDE_median_to_hypotenuse_length_l804_80485

theorem median_to_hypotenuse_length (a b c m : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → m = c / 2 → m = 2.5 := by
sorry

end NUMINAMATH_CALUDE_median_to_hypotenuse_length_l804_80485


namespace NUMINAMATH_CALUDE_parabola_transformation_l804_80425

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b
  , c := p.c + v }

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { a := 1
  , b := 0
  , c := 0 }

theorem parabola_transformation :
  let p1 := shift_horizontal original_parabola 3
  let p2 := shift_vertical p1 4
  p2 = { a := 1, b := -6, c := 13 } := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l804_80425


namespace NUMINAMATH_CALUDE_propane_tank_burner_cost_is_14_l804_80413

def propane_tank_burner_cost (total_money sheet_cost rope_cost helium_cost_per_oz flight_height_per_oz max_height : ℚ) : ℚ :=
  let remaining_money := total_money - sheet_cost - rope_cost
  let helium_oz_needed := max_height / flight_height_per_oz
  let helium_cost := helium_oz_needed * helium_cost_per_oz
  remaining_money - helium_cost

theorem propane_tank_burner_cost_is_14 :
  propane_tank_burner_cost 200 42 18 1.5 113 9492 = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_propane_tank_burner_cost_is_14_l804_80413


namespace NUMINAMATH_CALUDE_michelle_savings_l804_80478

/-- Represents the number of $100 bills Michelle has after exchanging her savings -/
def number_of_bills : ℕ := 8

/-- Represents the value of each bill in dollars -/
def bill_value : ℕ := 100

/-- Theorem stating that Michelle's total savings equal $800 -/
theorem michelle_savings : number_of_bills * bill_value = 800 := by
  sorry

end NUMINAMATH_CALUDE_michelle_savings_l804_80478


namespace NUMINAMATH_CALUDE_quadratic_minimum_l804_80445

theorem quadratic_minimum (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 12*x + 35
  ∀ y : ℝ, f x ≤ f y ↔ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l804_80445


namespace NUMINAMATH_CALUDE_boys_to_total_ratio_l804_80483

theorem boys_to_total_ratio (boys girls : ℕ) (h1 : boys > 0) (h2 : girls > 0) : 
  let total := boys + girls
  let prob_boy := boys / total
  let prob_girl := girls / total
  prob_boy = (1 / 4 : ℚ) * prob_girl →
  (boys : ℚ) / total = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_boys_to_total_ratio_l804_80483


namespace NUMINAMATH_CALUDE_closest_root_l804_80474

def options : List ℤ := [2, 3, 4, 5]

theorem closest_root (x : ℝ) (h : x^3 - 9 = 16) : 
  3 = (options.argmin (λ y => |y - x|)).get sorry :=
sorry

end NUMINAMATH_CALUDE_closest_root_l804_80474


namespace NUMINAMATH_CALUDE_books_remaining_l804_80403

theorem books_remaining (initial_books : ℕ) (donating_people : ℕ) (books_per_donation : ℕ) (borrowed_books : ℕ) :
  initial_books = 500 →
  donating_people = 10 →
  books_per_donation = 8 →
  borrowed_books = 220 →
  initial_books + donating_people * books_per_donation - borrowed_books = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_books_remaining_l804_80403


namespace NUMINAMATH_CALUDE_profit_increase_l804_80404

theorem profit_increase (cost_price selling_price : ℝ) (a : ℝ) 
  (h1 : selling_price - cost_price = cost_price * (a / 100))
  (h2 : selling_price - (cost_price * 0.95) = (cost_price * 0.95) * ((a + 15) / 100)) :
  a = 185 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_l804_80404


namespace NUMINAMATH_CALUDE_line_equation_l804_80427

/-- Given a line with an angle of inclination of 45° and a y-intercept of 2,
    its equation is x - y + 2 = 0 -/
theorem line_equation (angle : ℝ) (y_intercept : ℝ) :
  angle = 45 ∧ y_intercept = 2 →
  ∀ x y : ℝ, (y = x + y_intercept) ↔ (x - y + y_intercept = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l804_80427


namespace NUMINAMATH_CALUDE_lg_ratio_theorem_l804_80447

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_ratio_theorem (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  (lg 12) / (lg 15) = (2 * a + b) / (1 - a + b) := by
  sorry

end NUMINAMATH_CALUDE_lg_ratio_theorem_l804_80447


namespace NUMINAMATH_CALUDE_max_sum_abs_coords_ellipse_l804_80469

def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

theorem max_sum_abs_coords_ellipse :
  ∃ (M : ℝ), M = 5 ∧ 
  (∀ x y : ℝ, ellipse x y → |x| + |y| ≤ M) ∧
  (∃ x y : ℝ, ellipse x y ∧ |x| + |y| = M) :=
sorry

end NUMINAMATH_CALUDE_max_sum_abs_coords_ellipse_l804_80469


namespace NUMINAMATH_CALUDE_sum_of_digits_8_pow_2010_l804_80491

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2010 is 1. -/
theorem sum_of_digits_8_pow_2010 : ∃ n : ℕ, 8^2010 = 100 * n + 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_pow_2010_l804_80491


namespace NUMINAMATH_CALUDE_boys_to_total_ratio_l804_80438

/-- Represents a classroom with boys and girls -/
structure Classroom where
  total_students : ℕ
  num_boys : ℕ
  num_girls : ℕ
  boys_plus_girls : num_boys + num_girls = total_students

/-- The probability of choosing a student from a group -/
def prob_choose (group : ℕ) (total : ℕ) : ℚ :=
  group / total

theorem boys_to_total_ratio (c : Classroom) 
  (h1 : c.total_students > 0)
  (h2 : prob_choose c.num_boys c.total_students = 
        (3 / 4) * prob_choose c.num_girls c.total_students) :
  (c.num_boys : ℚ) / c.total_students = 3 / 7 := by
  sorry

#check boys_to_total_ratio

end NUMINAMATH_CALUDE_boys_to_total_ratio_l804_80438


namespace NUMINAMATH_CALUDE_first_equation_is_golden_second_equation_root_l804_80441

-- Definition of a golden equation
def is_golden_equation (a b c : ℝ) : Prop := a ≠ 0 ∧ a - b + c = 0

-- Theorem 1: 4x^2 + 11x + 7 = 0 is a golden equation
theorem first_equation_is_golden : is_golden_equation 4 11 7 := by sorry

-- Theorem 2: If 3x^2 - mx + n = 0 is a golden equation and m is a root, then m = -1 or m = 3/2
theorem second_equation_root (m n : ℝ) :
  is_golden_equation 3 (-m) n →
  (3 * m^2 - m * m + n = 0) →
  (m = -1 ∨ m = 3/2) := by sorry

end NUMINAMATH_CALUDE_first_equation_is_golden_second_equation_root_l804_80441


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l804_80453

def original_equation (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

def pair_A (x y : ℝ) : Prop := (y = x^2 - x) ∧ (y = 2*x - 2)
def pair_B (x y : ℝ) : Prop := (y = x^2 - 3*x + 2) ∧ (y = 0)
def pair_C (x y : ℝ) : Prop := (y = x - 1) ∧ (y = x + 1)
def pair_D (x y : ℝ) : Prop := (y = x^2 - 3*x + 3) ∧ (y = 1)

theorem intersection_points_theorem :
  (∃ x y : ℝ, pair_A x y ∧ original_equation x) ∧
  (∃ x y : ℝ, pair_B x y ∧ original_equation x) ∧
  (∃ x y : ℝ, pair_D x y ∧ original_equation x) ∧
  ¬(∃ x y : ℝ, pair_C x y ∧ original_equation x) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l804_80453


namespace NUMINAMATH_CALUDE_ember_nate_ages_l804_80434

/-- Given that Ember is initially half as old as Nate, and Nate is initially 14 years old,
    prove that when Ember's age becomes 14, Nate's age will be 21. -/
theorem ember_nate_ages (ember_initial : ℕ) (nate_initial : ℕ) (ember_final : ℕ) (nate_final : ℕ) :
  ember_initial = nate_initial / 2 →
  nate_initial = 14 →
  ember_final = 14 →
  nate_final = nate_initial + (ember_final - ember_initial) →
  nate_final = 21 := by
sorry

end NUMINAMATH_CALUDE_ember_nate_ages_l804_80434


namespace NUMINAMATH_CALUDE_log_simplification_l804_80400

theorem log_simplification (p q r s t z : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hz : z > 0) :
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * t / (s * z)) = Real.log (z / t) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l804_80400


namespace NUMINAMATH_CALUDE_inequalities_from_sum_of_reciprocal_squares_l804_80498

theorem inequalities_from_sum_of_reciprocal_squares
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_sum : 1 / a^2 + 1 / b^2 + 1 / c^2 = 1) :
  (1 / a + 1 / b + 1 / c ≤ Real.sqrt 3) ∧
  (a^2 / b^4 + b^2 / c^4 + c^2 / a^4 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_from_sum_of_reciprocal_squares_l804_80498


namespace NUMINAMATH_CALUDE_sin_translation_l804_80406

/-- Given a function f(x) = 3sin(2x), translating its graph π/6 units to the left
    results in the function g(x) = 3sin(2x + π/3) -/
theorem sin_translation (x : ℝ) :
  (fun x => 3 * Real.sin (2 * x + π / 3)) x =
  (fun x => 3 * Real.sin (2 * (x + π / 6))) x := by
sorry

end NUMINAMATH_CALUDE_sin_translation_l804_80406


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l804_80490

theorem computer_literate_female_employees 
  (total_employees : ℕ)
  (female_percentage : ℝ)
  (male_computer_literate_percentage : ℝ)
  (total_computer_literate_percentage : ℝ)
  (h_total : total_employees = 1200)
  (h_female : female_percentage = 0.6)
  (h_male_cl : male_computer_literate_percentage = 0.5)
  (h_total_cl : total_computer_literate_percentage = 0.62) :
  ⌊female_percentage * total_employees - 
   (1 - female_percentage) * male_computer_literate_percentage * total_employees⌋ = 504 :=
by sorry

end NUMINAMATH_CALUDE_computer_literate_female_employees_l804_80490


namespace NUMINAMATH_CALUDE_dogs_not_liking_any_food_l804_80448

theorem dogs_not_liking_any_food (total : ℕ) (watermelon salmon chicken : Finset ℕ) :
  total = 100 →
  watermelon.card = 20 →
  salmon.card = 70 →
  (watermelon ∩ salmon).card = 10 →
  chicken.card = 15 →
  (watermelon ∩ chicken).card = 5 →
  (salmon ∩ chicken).card = 8 →
  (watermelon ∩ salmon ∩ chicken).card = 3 →
  (total : ℤ) - (watermelon ∪ salmon ∪ chicken).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_liking_any_food_l804_80448


namespace NUMINAMATH_CALUDE_det_A_l804_80484

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, -6, 6; 0, 6, -2; 3, -1, 2]

theorem det_A : Matrix.det A = -52 := by
  sorry

end NUMINAMATH_CALUDE_det_A_l804_80484


namespace NUMINAMATH_CALUDE_polynomial_integer_coefficients_l804_80460

theorem polynomial_integer_coefficients (a b c : ℚ) : 
  (∀ x : ℤ, ∃ n : ℤ, a * x^2 + b * x + c = n) → 
  (∃ (a' b' c' : ℤ), a = a' ∧ b = b' ∧ c = c') := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integer_coefficients_l804_80460


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l804_80461

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | 0 < x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l804_80461


namespace NUMINAMATH_CALUDE_count_problems_requiring_selection_l804_80420

-- Define a structure to represent a problem
structure Problem where
  id : Nat
  requires_selection : Bool

-- Define our set of problems
def problems : List Problem := [
  { id := 1, requires_selection := true },  -- absolute value
  { id := 2, requires_selection := false }, -- square perimeter
  { id := 3, requires_selection := true },  -- maximum of three numbers
  { id := 4, requires_selection := true }   -- function value
]

-- Theorem statement
theorem count_problems_requiring_selection :
  (problems.filter Problem.requires_selection).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_problems_requiring_selection_l804_80420


namespace NUMINAMATH_CALUDE_toby_friends_percentage_l804_80492

def toby_boy_friends : ℕ := 33
def toby_girl_friends : ℕ := 27

theorem toby_friends_percentage :
  (toby_boy_friends : ℚ) / (toby_boy_friends + toby_girl_friends : ℚ) * 100 = 55 := by
  sorry

end NUMINAMATH_CALUDE_toby_friends_percentage_l804_80492


namespace NUMINAMATH_CALUDE_angle_tangent_relation_l804_80423

theorem angle_tangent_relation (θ : Real) :
  (-(π / 2) < θ ∧ θ < 0) →  -- θ is in the fourth quadrant
  (Real.sin (θ + π / 4) = 3 / 5) →
  (Real.tan (θ - π / 4) = -4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_angle_tangent_relation_l804_80423


namespace NUMINAMATH_CALUDE_correct_balloons_popped_l804_80440

/-- The number of blue balloons Sally popped -/
def balloons_popped (joan_initial : ℕ) (jessica : ℕ) (total_now : ℕ) : ℕ :=
  joan_initial - total_now

theorem correct_balloons_popped (joan_initial jessica total_now : ℕ) 
  (h1 : joan_initial = 9)
  (h2 : jessica = 2)
  (h3 : total_now = 6) :
  balloons_popped joan_initial jessica total_now = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_balloons_popped_l804_80440


namespace NUMINAMATH_CALUDE_total_students_l804_80497

-- Define the score groups
inductive ScoreGroup
| Low : ScoreGroup    -- [20, 40)
| Medium : ScoreGroup -- [40, 60)
| High : ScoreGroup   -- [60, 80)
| VeryHigh : ScoreGroup -- [80, 100]

-- Define the frequency distribution
def FrequencyDistribution := ScoreGroup → ℕ

-- Theorem statement
theorem total_students (freq : FrequencyDistribution) 
  (below_60 : freq ScoreGroup.Low + freq ScoreGroup.Medium = 15) :
  freq ScoreGroup.Low + freq ScoreGroup.Medium + 
  freq ScoreGroup.High + freq ScoreGroup.VeryHigh = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_total_students_l804_80497


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l804_80470

theorem earth_inhabitable_fraction :
  let earth_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (2 : ℚ) / 3
  inhabitable_land_fraction * land_fraction * earth_surface = (2 : ℚ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l804_80470


namespace NUMINAMATH_CALUDE_add_2405_minutes_to_midnight_l804_80435

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := (totalMinutes / 60) % 24, minutes := totalMinutes % 60 }

-- Theorem statement
theorem add_2405_minutes_to_midnight :
  addMinutes { hours := 0, minutes := 0 } 2405 = { hours := 16, minutes := 5 } := by
  sorry

end NUMINAMATH_CALUDE_add_2405_minutes_to_midnight_l804_80435


namespace NUMINAMATH_CALUDE_problem_solution_l804_80417

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : x * x^2 * (1/x) = 100/81) : x = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l804_80417


namespace NUMINAMATH_CALUDE_max_days_same_shift_l804_80479

/-- The number of nurses in the ward -/
def num_nurses : ℕ := 15

/-- The number of shifts per day -/
def shifts_per_day : ℕ := 3

/-- Calculates the number of possible nurse pair combinations -/
def nurse_pair_combinations (n : ℕ) : ℕ := n.choose 2

/-- Theorem: Maximum days for two specific nurses to work the same shift again -/
theorem max_days_same_shift : 
  nurse_pair_combinations num_nurses / shifts_per_day = 35 := by
  sorry

end NUMINAMATH_CALUDE_max_days_same_shift_l804_80479


namespace NUMINAMATH_CALUDE_other_amount_theorem_l804_80480

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem other_amount_theorem :
  let initial_principal : ℝ := 200
  let initial_rate : ℝ := 0.1
  let initial_time : ℝ := 12
  let other_rate : ℝ := 0.12
  let other_time : ℝ := 2
  let other_principal : ℝ := 1000
  simple_interest initial_principal initial_rate initial_time =
    simple_interest other_principal other_rate other_time := by
  sorry

end NUMINAMATH_CALUDE_other_amount_theorem_l804_80480


namespace NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l804_80471

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 9*x^2 + 24*x + 36

-- State the theorem
theorem intersection_point_of_f_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-3, -3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l804_80471


namespace NUMINAMATH_CALUDE_binomial_divisibility_l804_80430

theorem binomial_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℤ, (Nat.choose (2*p - 1) (p - 1) : ℤ) - 1 = k * p^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l804_80430


namespace NUMINAMATH_CALUDE_rose_bed_fraction_l804_80444

/-- Proof that the rose bed occupies 1/20 of the park's area given the conditions -/
theorem rose_bed_fraction (park_length park_width : ℝ) 
  (flower_bed_fraction : ℝ) (rose_bed_fraction : ℝ) :
  park_length = 15 →
  park_width = 20 →
  flower_bed_fraction = 1/5 →
  rose_bed_fraction = 1/4 →
  (flower_bed_fraction * rose_bed_fraction * park_length * park_width) / 
  (park_length * park_width) = 1/20 := by
  sorry

#check rose_bed_fraction

end NUMINAMATH_CALUDE_rose_bed_fraction_l804_80444
