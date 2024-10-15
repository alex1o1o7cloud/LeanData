import Mathlib

namespace NUMINAMATH_CALUDE_jerry_walk_time_approx_l712_71283

/-- Calculates the time it takes Jerry to walk each way to the sink and recycling bin -/
def jerryWalkTime (totalCans : ℕ) (cansPerTrip : ℕ) (drainTime : ℕ) (totalTime : ℕ) : ℚ :=
  let trips := totalCans / cansPerTrip
  let totalDrainTime := drainTime * trips
  let totalWalkTime := totalTime - totalDrainTime
  let walkTimePerTrip := totalWalkTime / trips
  walkTimePerTrip / 3

/-- Theorem stating that Jerry's walk time is approximately 6.67 seconds -/
theorem jerry_walk_time_approx :
  let walkTime := jerryWalkTime 28 4 30 350
  (walkTime > 6.66) ∧ (walkTime < 6.68) := by
  sorry

end NUMINAMATH_CALUDE_jerry_walk_time_approx_l712_71283


namespace NUMINAMATH_CALUDE_conjugate_sum_product_l712_71205

theorem conjugate_sum_product (c d : ℝ) :
  (c + Real.sqrt d + (c - Real.sqrt d) = -8) →
  ((c + Real.sqrt d) * (c - Real.sqrt d) = 4) →
  c + d = 8 := by
sorry

end NUMINAMATH_CALUDE_conjugate_sum_product_l712_71205


namespace NUMINAMATH_CALUDE_nail_trimming_customers_l712_71258

/-- The number of nails per customer -/
def nails_per_customer : ℕ := 20

/-- The total number of sounds produced by the nail cutter -/
def total_sounds : ℕ := 120

/-- The number of customers -/
def num_customers : ℕ := total_sounds / nails_per_customer

theorem nail_trimming_customers :
  num_customers = 6 :=
by sorry

end NUMINAMATH_CALUDE_nail_trimming_customers_l712_71258


namespace NUMINAMATH_CALUDE_election_expectation_l712_71232

/-- The number of voters and candidates in the election -/
def n : ℕ := 5

/-- The probability of a candidate receiving no votes -/
def p_no_votes : ℚ := (4/5)^n

/-- The probability of a candidate receiving at least one vote -/
def p_at_least_one_vote : ℚ := 1 - p_no_votes

/-- The expected number of candidates receiving at least one vote -/
def expected_candidates_with_votes : ℚ := n * p_at_least_one_vote

theorem election_expectation :
  expected_candidates_with_votes = 2101/625 :=
by sorry

end NUMINAMATH_CALUDE_election_expectation_l712_71232


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l712_71278

theorem pure_imaginary_complex_number (m : ℝ) : 
  (∃ z : ℂ, z = (m^2 - 4 : ℝ) + (m + 2 : ℝ) * I ∧ z.re = 0 ∧ m + 2 ≠ 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l712_71278


namespace NUMINAMATH_CALUDE_number_puzzle_l712_71289

theorem number_puzzle : ∃ x : ℝ, 22 * (x - 36) = 748 ∧ x = 70 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l712_71289


namespace NUMINAMATH_CALUDE_inequality_solution_l712_71219

theorem inequality_solution (x : ℝ) : (3 * x - 9) / ((x - 3)^2) < 0 ↔ x < 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l712_71219


namespace NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l712_71210

def arithmetic_sequence (a : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

theorem seventh_term_of_arithmetic_sequence 
  (a d : ℚ) 
  (h1 : (arithmetic_sequence a d 0) + 
        (arithmetic_sequence a d 1) + 
        (arithmetic_sequence a d 2) + 
        (arithmetic_sequence a d 3) + 
        (arithmetic_sequence a d 4) = 20)
  (h2 : arithmetic_sequence a d 5 = 8) :
  arithmetic_sequence a d 6 = 28 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l712_71210


namespace NUMINAMATH_CALUDE_point_on_line_l712_71213

/-- Given a line passing through points (2, 1) and (10, 5), 
    prove that the point (14, 7) lies on this line. -/
theorem point_on_line : ∀ (t : ℝ), 
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (10, 5)
  let p3 : ℝ × ℝ := (t, 7)
  -- Check if p3 is on the line through p1 and p2
  (p3.2 - p1.2) * (p2.1 - p1.1) = (p3.1 - p1.1) * (p2.2 - p1.2) →
  t = 14 :=
by
  sorry

#check point_on_line

end NUMINAMATH_CALUDE_point_on_line_l712_71213


namespace NUMINAMATH_CALUDE_index_card_area_l712_71293

theorem index_card_area (length width : ℝ) 
  (h1 : length = 5 ∧ width = 7)
  (h2 : ∃ (shortened_side : ℝ), 
    (shortened_side = length - 2 ∨ shortened_side = width - 2) ∧
    shortened_side * (if shortened_side = length - 2 then width else length) = 21) :
  length * (width - 1) = 30 :=
by sorry

end NUMINAMATH_CALUDE_index_card_area_l712_71293


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l712_71259

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 22 → 
  a = 220 → 
  b = 462 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l712_71259


namespace NUMINAMATH_CALUDE_max_value_cos_sin_expression_l712_71233

theorem max_value_cos_sin_expression (a b c : ℝ) :
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c ≤ Real.sqrt (a^2 + b^2) + c) ∧
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c = Real.sqrt (a^2 + b^2) + c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_expression_l712_71233


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l712_71243

/-- For a parabola with equation y = 4x^2, the distance from the focus to the directrix is 1/8 -/
theorem parabola_focus_directrix_distance :
  let parabola := fun (x : ℝ) => 4 * x^2
  ∃ (focus : ℝ × ℝ) (directrix : ℝ → ℝ),
    (∀ x, parabola x = (x - focus.1)^2 / (4 * (focus.2 - directrix 0))) ∧
    (focus.2 - directrix 0 = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l712_71243


namespace NUMINAMATH_CALUDE_movie_of_the_year_requirement_l712_71238

/-- The number of members in the cinematic academy -/
def academy_members : ℕ := 795

/-- The fraction of lists a film must appear on to be considered for "movie of the year" -/
def required_fraction : ℚ := 1 / 4

/-- The smallest number of lists a film can appear on to be considered for "movie of the year" -/
def min_lists : ℕ := 199

/-- Theorem stating the smallest number of lists a film must appear on -/
theorem movie_of_the_year_requirement :
  min_lists = ⌈(required_fraction * academy_members : ℚ)⌉ :=
sorry

end NUMINAMATH_CALUDE_movie_of_the_year_requirement_l712_71238


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l712_71281

theorem largest_prime_factor_of_1729 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 1729 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l712_71281


namespace NUMINAMATH_CALUDE_jake_bitcoin_theorem_l712_71207

def jake_bitcoin_problem (initial_fortune : ℕ) (first_donation : ℕ) (second_donation : ℕ) : ℕ :=
  let after_first_donation := initial_fortune - first_donation
  let after_giving_to_brother := after_first_donation / 2
  let after_tripling := after_giving_to_brother * 3
  after_tripling - second_donation

theorem jake_bitcoin_theorem :
  jake_bitcoin_problem 80 20 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jake_bitcoin_theorem_l712_71207


namespace NUMINAMATH_CALUDE_complex_square_pure_imaginary_l712_71299

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_square_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((1 + a * Complex.I) ^ 2) → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_pure_imaginary_l712_71299


namespace NUMINAMATH_CALUDE_sum_gcd_lcm_6_15_30_l712_71297

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem sum_gcd_lcm_6_15_30 :
  gcd_three 6 15 30 + lcm_three 6 15 30 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_gcd_lcm_6_15_30_l712_71297


namespace NUMINAMATH_CALUDE_supplement_statement_is_proposition_l712_71222

-- Define what a proposition is
def isPropositon (s : String) : Prop := ∃ (b : Bool), (b = true ∨ b = false)

-- Define the statement
def supplementStatement : String := "The supplements of the same angle are equal"

-- Theorem to prove
theorem supplement_statement_is_proposition : isPropositon supplementStatement := by
  sorry

end NUMINAMATH_CALUDE_supplement_statement_is_proposition_l712_71222


namespace NUMINAMATH_CALUDE_golden_ratio_expressions_l712_71274

theorem golden_ratio_expressions (θ : Real) (h : θ = 18 * π / 180) :
  let φ := (Real.sqrt 5 - 1) / 4
  φ = Real.sin θ ∧
  φ = Real.cos (10 * π / 180) * Real.cos (82 * π / 180) + Real.sin (10 * π / 180) * Real.sin (82 * π / 180) ∧
  φ = Real.sin (173 * π / 180) * Real.cos (11 * π / 180) - Real.sin (83 * π / 180) * Real.cos (101 * π / 180) ∧
  φ = Real.sqrt ((1 - Real.sin (54 * π / 180)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_golden_ratio_expressions_l712_71274


namespace NUMINAMATH_CALUDE_inequality_equivalence_l712_71287

theorem inequality_equivalence (x : ℝ) : 3 * x^2 + 2 * x - 3 > 12 - 2 * x ↔ x < -3 ∨ x > 5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l712_71287


namespace NUMINAMATH_CALUDE_reciprocal_of_one_twentieth_l712_71282

theorem reciprocal_of_one_twentieth (x : ℚ) : x = 1 / 20 → 1 / x = 20 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_one_twentieth_l712_71282


namespace NUMINAMATH_CALUDE_min_score_for_average_increase_l712_71294

/-- Given 4 tests with an average score of 68, prove that a score of at least 78 on the 5th test 
    is necessary to achieve an average score of more than 70 over all 5 tests. -/
theorem min_score_for_average_increase (current_tests : Nat) (current_average : ℝ) 
  (target_average : ℝ) (min_score : ℝ) : 
  current_tests = 4 → 
  current_average = 68 → 
  target_average > 70 → 
  min_score ≥ 78 → 
  (current_tests * current_average + min_score) / (current_tests + 1) > target_average :=
by sorry

end NUMINAMATH_CALUDE_min_score_for_average_increase_l712_71294


namespace NUMINAMATH_CALUDE_swimmers_passing_count_l712_71254

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  turnTime : ℝ

/-- Calculates the number of times two swimmers pass each other --/
def calculatePassings (poolLength : ℝ) (totalTime : ℝ) (swimmerA : Swimmer) (swimmerB : Swimmer) : ℕ :=
  sorry

/-- The main theorem stating the number of times the swimmers pass each other --/
theorem swimmers_passing_count :
  let poolLength : ℝ := 120
  let totalTime : ℝ := 15 * 60  -- 15 minutes in seconds
  let swimmerA : Swimmer := { speed := 4, turnTime := 0 }
  let swimmerB : Swimmer := { speed := 3, turnTime := 2 }
  calculatePassings poolLength totalTime swimmerA swimmerB = 51 :=
by sorry

end NUMINAMATH_CALUDE_swimmers_passing_count_l712_71254


namespace NUMINAMATH_CALUDE_ketchup_tomatoes_ratio_l712_71247

/-- Given that 3 liters of ketchup require 69 kg of tomatoes, 
    prove that 5 liters of ketchup require 115 kg of tomatoes. -/
theorem ketchup_tomatoes_ratio (tomatoes_for_three : ℝ) (ketchup_liters : ℝ) : 
  tomatoes_for_three = 69 → ketchup_liters = 5 → 
  (tomatoes_for_three / 3) * ketchup_liters = 115 := by
sorry

end NUMINAMATH_CALUDE_ketchup_tomatoes_ratio_l712_71247


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l712_71271

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l712_71271


namespace NUMINAMATH_CALUDE_binomial_expansion_x_squared_term_l712_71255

theorem binomial_expansion_x_squared_term (x : ℝ) (n : ℕ) :
  (∃ r : ℕ, r ≤ n ∧ (n.choose r) * x^((5*r)/2 - n) = x^2) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_x_squared_term_l712_71255


namespace NUMINAMATH_CALUDE_prob_at_least_seven_stay_value_l712_71206

def num_friends : ℕ := 8
def num_unsure : ℕ := 5
def num_certain : ℕ := 3
def prob_unsure_stay : ℚ := 3/7

def prob_at_least_seven_stay : ℚ :=
  Nat.choose num_unsure 3 * (prob_unsure_stay ^ 3) * ((1 - prob_unsure_stay) ^ 2) +
  prob_unsure_stay ^ num_unsure

theorem prob_at_least_seven_stay_value :
  prob_at_least_seven_stay = 4563/16807 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_seven_stay_value_l712_71206


namespace NUMINAMATH_CALUDE_prices_and_schemes_l712_71204

def soccer_ball_price : ℕ := 60
def basketball_price : ℕ := 80

def initial_purchase_cost : ℕ := 1600
def initial_soccer_balls : ℕ := 8
def initial_basketballs : ℕ := 14

def total_balls : ℕ := 50
def min_budget : ℕ := 3200
def max_budget : ℕ := 3240

theorem prices_and_schemes :
  (initial_soccer_balls * soccer_ball_price + initial_basketballs * basketball_price = initial_purchase_cost) ∧
  (basketball_price = soccer_ball_price + 20) ∧
  (∀ y : ℕ, y ≤ total_balls →
    (y * soccer_ball_price + (total_balls - y) * basketball_price ≥ min_budget ∧
     y * soccer_ball_price + (total_balls - y) * basketball_price ≤ max_budget)
    ↔ (y = 38 ∨ y = 39 ∨ y = 40)) :=
by sorry

end NUMINAMATH_CALUDE_prices_and_schemes_l712_71204


namespace NUMINAMATH_CALUDE_crowdfunding_highest_level_l712_71223

theorem crowdfunding_highest_level (x : ℝ) : 
  x > 0 ∧ 
  7298 * x = 200000 → 
  ⌊1296 * x⌋ = 35534 :=
by
  sorry

end NUMINAMATH_CALUDE_crowdfunding_highest_level_l712_71223


namespace NUMINAMATH_CALUDE_exists_region_with_min_area_l712_71212

/-- Represents a line segment in a unit square --/
structure Segment where
  length : ℝ
  parallel_to_side : Bool

/-- Represents a configuration of segments in a unit square --/
structure SquareConfiguration where
  segments : List Segment
  total_length : ℝ
  total_length_eq : total_length = (segments.map Segment.length).sum
  total_length_bound : total_length = 18
  segments_within_square : ∀ s ∈ segments, s.length ≤ 1

/-- Represents a region formed by the segments --/
structure Region where
  area : ℝ

/-- The theorem to be proved --/
theorem exists_region_with_min_area (config : SquareConfiguration) :
  ∃ (r : Region), r.area ≥ 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_region_with_min_area_l712_71212


namespace NUMINAMATH_CALUDE_inequality_solution_set_l712_71290

theorem inequality_solution_set (x : ℝ) (h : x ≠ 1) :
  (1 / (x - 1) ≤ 1) ↔ (x < 1 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l712_71290


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l712_71240

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (distinct : Line → Line → Prop)
variable (nonCoincident : Plane → Plane → Prop)
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (m : Line) (α β : Plane)
  (h1 : nonCoincident α β)
  (h2 : perpendicular m α)
  (h3 : parallel m β) :
  planePerp α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l712_71240


namespace NUMINAMATH_CALUDE_hypotenuse_increase_bound_l712_71269

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  let c := Real.sqrt (x^2 + y^2)
  let c' := Real.sqrt ((x+1)^2 + (y+1)^2)
  c' - c ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_increase_bound_l712_71269


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l712_71279

theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l712_71279


namespace NUMINAMATH_CALUDE_first_triangle_is_isosceles_l712_71260

-- Define a triangle structure
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_eq_180 : α + β + γ = 180

-- Define the theorem
theorem first_triangle_is_isosceles 
  (t1 t2 : Triangle) 
  (h1 : ∃ (θ : Real), t1.α + t1.β = θ ∧ θ ≤ 180) 
  (h2 : ∃ (φ : Real), t1.β + t1.γ = φ ∧ φ ≤ 180) : 
  t1.α = t1.γ ∨ t1.α = t1.β ∨ t1.β = t1.γ :=
sorry

end NUMINAMATH_CALUDE_first_triangle_is_isosceles_l712_71260


namespace NUMINAMATH_CALUDE_fifth_bounce_height_l712_71262

/-- Calculates the height of a bouncing ball after a given number of bounces. -/
def bounceHeight (initialHeight : ℝ) (initialEfficiency : ℝ) (efficiencyDecrease : ℝ) (airResistanceLoss : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The height of the ball after the fifth bounce is approximately 0.82 feet. -/
theorem fifth_bounce_height :
  let initialHeight : ℝ := 96
  let initialEfficiency : ℝ := 0.5
  let efficiencyDecrease : ℝ := 0.05
  let airResistanceLoss : ℝ := 0.02
  let bounces : ℕ := 5
  abs (bounceHeight initialHeight initialEfficiency efficiencyDecrease airResistanceLoss bounces - 0.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_fifth_bounce_height_l712_71262


namespace NUMINAMATH_CALUDE_price_restoration_l712_71225

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) (restored_price : ℝ) : 
  reduced_price = original_price * (1 - 0.2) →
  restored_price = reduced_price * (1 + 0.25) →
  restored_price = original_price :=
by sorry

end NUMINAMATH_CALUDE_price_restoration_l712_71225


namespace NUMINAMATH_CALUDE_tshirt_company_profit_l712_71249

/-- Calculates the daily profit of a t-shirt company given specific conditions -/
theorem tshirt_company_profit 
  (num_employees : ℕ) 
  (shirts_per_employee : ℕ) 
  (hours_per_shift : ℕ) 
  (hourly_wage : ℚ) 
  (per_shirt_bonus : ℚ) 
  (shirt_price : ℚ) 
  (nonemployee_expenses : ℚ) 
  (h1 : num_employees = 20)
  (h2 : shirts_per_employee = 20)
  (h3 : hours_per_shift = 8)
  (h4 : hourly_wage = 12)
  (h5 : per_shirt_bonus = 5)
  (h6 : shirt_price = 35)
  (h7 : nonemployee_expenses = 1000) :
  (num_employees * shirts_per_employee * shirt_price) - 
  (num_employees * (hours_per_shift * hourly_wage + shirts_per_employee * per_shirt_bonus) + nonemployee_expenses) = 9080 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_company_profit_l712_71249


namespace NUMINAMATH_CALUDE_power_of_three_equality_l712_71228

theorem power_of_three_equality : (3^5)^6 = 3^12 * 3^18 := by sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l712_71228


namespace NUMINAMATH_CALUDE_z_eighth_power_equals_one_l712_71275

theorem z_eighth_power_equals_one :
  let z : ℂ := (-Real.sqrt 3 - I) / 2
  z^8 = 1 := by sorry

end NUMINAMATH_CALUDE_z_eighth_power_equals_one_l712_71275


namespace NUMINAMATH_CALUDE_parabola_focus_property_l712_71266

/-- Parabola with equation y^2 = 16x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 16 * p.1}

/-- Focus of the parabola -/
def F : ℝ × ℝ := (4, 0)

/-- Point on y-axis with |OA| = |OF| -/
def A : ℝ × ℝ := (0, 4) -- We choose the positive y-coordinate

/-- Intersection of directrix and x-axis -/
def B : ℝ × ℝ := (-4, 0)

/-- Vector from F to A -/
def FA : ℝ × ℝ := (A.1 - F.1, A.2 - F.2)

/-- Vector from A to B -/
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem parabola_focus_property :
  F ∈ Parabola ∧
  A.1 = 0 ∧
  (A.1 - 0)^2 + (A.2 - 0)^2 = (F.1 - 0)^2 + (F.2 - 0)^2 ∧
  B.2 = 0 →
  dot_product FA AB = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_property_l712_71266


namespace NUMINAMATH_CALUDE_expression_eval_zero_l712_71246

theorem expression_eval_zero (a : ℚ) (h : a = 3/2) : 
  (5 * a^2 - 13 * a + 4) * (2 * a - 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_eval_zero_l712_71246


namespace NUMINAMATH_CALUDE_container_filling_l712_71250

theorem container_filling (capacity : ℝ) (initial_fraction : ℝ) (added_water : ℝ) :
  capacity = 80 →
  initial_fraction = 1/2 →
  added_water = 20 →
  (initial_fraction * capacity + added_water) / capacity = 3/4 := by
sorry

end NUMINAMATH_CALUDE_container_filling_l712_71250


namespace NUMINAMATH_CALUDE_A_inter_B_l712_71280

def A : Set ℝ := {-2, -1, 0, 1, 2}

def B : Set ℝ := {x | -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3}

theorem A_inter_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_A_inter_B_l712_71280


namespace NUMINAMATH_CALUDE_min_value_theorem_l712_71285

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l712_71285


namespace NUMINAMATH_CALUDE_x_power_y_value_l712_71229

theorem x_power_y_value (x y : ℝ) (h : |x + 1/2| + (y - 3)^2 = 0) : x^y = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_x_power_y_value_l712_71229


namespace NUMINAMATH_CALUDE_capital_ratio_specific_case_l712_71242

/-- Given a total loss and Pyarelal's loss, calculate the ratio of Ashok's capital to Pyarelal's capital -/
def capital_ratio (total_loss : ℕ) (pyarelal_loss : ℕ) : ℕ × ℕ :=
  let ashok_loss := total_loss - pyarelal_loss
  (ashok_loss, pyarelal_loss)

/-- Theorem stating that given the specific losses, the capital ratio is 67:603 -/
theorem capital_ratio_specific_case :
  capital_ratio 670 603 = (67, 603) := by
  sorry

end NUMINAMATH_CALUDE_capital_ratio_specific_case_l712_71242


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l712_71215

/-- Given two right triangles and inscribed squares, proves the ratio of their side lengths -/
theorem inscribed_squares_ratio : 
  ∀ (x y : ℝ),
  (∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 ∧ 
    x * (a + b - x) = a * b) →
  (∃ (p q r : ℝ), p = 5 ∧ q = 12 ∧ r = 13 ∧ p^2 + q^2 = r^2 ∧ 
    y * (r - y) = p * q) →
  x / y = 444 / 1183 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l712_71215


namespace NUMINAMATH_CALUDE_greatest_a_value_l712_71296

theorem greatest_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -12) → 
  (a > 0) → 
  (∀ b : ℤ, b > a → ¬(∃ y : ℤ, y^2 + b*y = -12)) → 
  a = 13 := by
  sorry

end NUMINAMATH_CALUDE_greatest_a_value_l712_71296


namespace NUMINAMATH_CALUDE_nabla_example_l712_71211

-- Define the ∇ operation
def nabla (a b c d : ℝ) : ℝ := a * c + b * d

-- Theorem statement
theorem nabla_example : nabla 3 1 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_nabla_example_l712_71211


namespace NUMINAMATH_CALUDE_distinct_roots_condition_zero_root_condition_other_root_when_m_is_one_other_root_when_m_is_negative_one_l712_71209

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + 2*(m-1)*x + m^2 - 1 = 0

-- Part 1: Distinct real roots condition
theorem distinct_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x m ∧ quadratic_equation y m) ↔ m < 1 :=
sorry

-- Part 2: Zero root condition and other root
theorem zero_root_condition (m : ℝ) :
  quadratic_equation 0 m → (m = 1 ∨ m = -1) :=
sorry

theorem other_root_when_m_is_one :
  quadratic_equation 0 1 → quadratic_equation 0 1 :=
sorry

theorem other_root_when_m_is_negative_one :
  quadratic_equation 0 (-1) → quadratic_equation 4 (-1) :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_condition_zero_root_condition_other_root_when_m_is_one_other_root_when_m_is_negative_one_l712_71209


namespace NUMINAMATH_CALUDE_remainder_problem_l712_71286

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 38) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l712_71286


namespace NUMINAMATH_CALUDE_root_transformation_l712_71277

/-- Given that a, b, and c are the roots of x^3 - 4x + 6 = 0,
    prove that a - 3, b - 3, and c - 3 are the roots of x^3 + 9x^2 + 23x + 21 = 0 -/
theorem root_transformation (a b c : ℂ) : 
  (a^3 - 4*a + 6 = 0) ∧ (b^3 - 4*b + 6 = 0) ∧ (c^3 - 4*c + 6 = 0) →
  ((a - 3)^3 + 9*(a - 3)^2 + 23*(a - 3) + 21 = 0) ∧
  ((b - 3)^3 + 9*(b - 3)^2 + 23*(b - 3) + 21 = 0) ∧
  ((c - 3)^3 + 9*(c - 3)^2 + 23*(c - 3) + 21 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l712_71277


namespace NUMINAMATH_CALUDE_biology_group_specimen_exchange_l712_71298

/-- Represents the number of specimens exchanged in a biology interest group --/
def specimens_exchanged (x : ℕ) : ℕ := x * (x - 1)

/-- Theorem stating that the equation x(x-1) = 110 correctly represents the situation --/
theorem biology_group_specimen_exchange (x : ℕ) :
  specimens_exchanged x = 110 ↔ x * (x - 1) = 110 := by
  sorry

end NUMINAMATH_CALUDE_biology_group_specimen_exchange_l712_71298


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l712_71295

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 4 + a 9 = 24) 
  (h_sixth : a 6 = 11) : 
  a 7 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l712_71295


namespace NUMINAMATH_CALUDE_water_remaining_after_14_pourings_fourteen_pourings_is_minimum_l712_71208

/-- Calculates the fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- Theorem: After 14 pourings, exactly 1/8 of the original water remains -/
theorem water_remaining_after_14_pourings :
  waterRemaining 14 = 1/8 := by
  sorry

/-- Theorem: 14 is the smallest number of pourings that leaves exactly 1/8 of the original water -/
theorem fourteen_pourings_is_minimum :
  ∀ k : ℕ, k < 14 → waterRemaining k > 1/8 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_after_14_pourings_fourteen_pourings_is_minimum_l712_71208


namespace NUMINAMATH_CALUDE_isosceles_triangle_a_values_l712_71227

/-- An isosceles triangle with sides 10-a, 7, and 6 -/
structure IsoscelesTriangle (a : ℝ) :=
  (side1 : ℝ := 10 - a)
  (side2 : ℝ := 7)
  (side3 : ℝ := 6)
  (isIsosceles : (side1 = side2 ∧ side3 ≠ side1) ∨ 
                 (side1 = side3 ∧ side2 ≠ side1) ∨ 
                 (side2 = side3 ∧ side1 ≠ side2))

/-- The theorem stating that a is either 3 or 4 for an isosceles triangle with sides 10-a, 7, and 6 -/
theorem isosceles_triangle_a_values :
  ∀ a : ℝ, IsoscelesTriangle a → a = 3 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_a_values_l712_71227


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l712_71244

/-- A type representing a point on a circle -/
def CirclePoint : Type := Nat

/-- The total number of points on the circle -/
def total_points : Nat := 2023

/-- A type representing a selection of 6 distinct points -/
def Sextuple : Type := Fin 6 → CirclePoint

/-- Predicate to check if two chords intersect -/
def chords_intersect (a b c d : CirclePoint) : Prop := sorry

/-- The probability of selecting a sextuple where AB intersects both CD and EF -/
def intersecting_chords_probability : ℚ := 1 / 72

theorem intersecting_chords_theorem (s : Sextuple) :
  (∀ i j : Fin 6, i ≠ j → s i ≠ s j) →  -- all points are distinct
  (∀ s : Sextuple, (∀ i j : Fin 6, i ≠ j → s i ≠ s j) → 
    chords_intersect (s 0) (s 1) (s 2) (s 3) ∧ 
    chords_intersect (s 2) (s 3) (s 4) (s 5)) →  -- definition of intersecting chords
  intersecting_chords_probability = 1 / 72 := by sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l712_71244


namespace NUMINAMATH_CALUDE_initial_value_proof_l712_71292

theorem initial_value_proof : 
  ∃! x : ℕ, x ≥ 0 ∧ (∀ y : ℕ, y ≥ 0 → (y + 37) % 3 = 0 ∧ (y + 37) % 5 = 0 ∧ (y + 37) % 7 = 0 ∧ (y + 37) % 8 = 0 → x ≤ y) ∧
  (x + 37) % 3 = 0 ∧ (x + 37) % 5 = 0 ∧ (x + 37) % 7 = 0 ∧ (x + 37) % 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_initial_value_proof_l712_71292


namespace NUMINAMATH_CALUDE_quadratic_root_expression_l712_71234

theorem quadratic_root_expression (x : ℝ) : 
  x > 0 ∧ x^2 - 10*x - 10 = 0 → 1/20 * x^4 - 6*x^2 - 45 = -50 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_expression_l712_71234


namespace NUMINAMATH_CALUDE_andrews_hotdogs_l712_71248

theorem andrews_hotdogs (total : ℕ) (cheese_pops : ℕ) (chicken_nuggets : ℕ) 
  (h1 : total = 90)
  (h2 : cheese_pops = 20)
  (h3 : chicken_nuggets = 40)
  (h4 : ∃ hotdogs : ℕ, total = hotdogs + cheese_pops + chicken_nuggets) :
  ∃ hotdogs : ℕ, hotdogs = 30 ∧ total = hotdogs + cheese_pops + chicken_nuggets :=
by
  sorry

end NUMINAMATH_CALUDE_andrews_hotdogs_l712_71248


namespace NUMINAMATH_CALUDE_card_probability_l712_71239

theorem card_probability : 
  let total_cards : ℕ := 52
  let cards_per_suit : ℕ := 13
  let top_cards : ℕ := 4
  let favorable_suits : ℕ := 2  -- spades and clubs

  let favorable_outcomes : ℕ := favorable_suits * (cards_per_suit.descFactorial top_cards)
  let total_outcomes : ℕ := total_cards.descFactorial top_cards

  (favorable_outcomes : ℚ) / total_outcomes = 286 / 54145 := by sorry

end NUMINAMATH_CALUDE_card_probability_l712_71239


namespace NUMINAMATH_CALUDE_raul_money_left_l712_71202

/-- Calculates the money left after buying comics -/
def money_left (initial_money : ℕ) (num_comics : ℕ) (cost_per_comic : ℕ) : ℕ :=
  initial_money - (num_comics * cost_per_comic)

/-- Proves that Raul's remaining money is correct -/
theorem raul_money_left :
  money_left 87 8 4 = 55 := by
  sorry

end NUMINAMATH_CALUDE_raul_money_left_l712_71202


namespace NUMINAMATH_CALUDE_cube_edge_sum_l712_71203

/-- The number of edges in a cube -/
def cube_edges : ℕ := 12

/-- The length of one edge of the cube in centimeters -/
def edge_length : ℝ := 15

/-- The sum of the lengths of all edges of the cube in centimeters -/
def sum_of_edges : ℝ := cube_edges * edge_length

theorem cube_edge_sum :
  sum_of_edges = 180 := by sorry

end NUMINAMATH_CALUDE_cube_edge_sum_l712_71203


namespace NUMINAMATH_CALUDE_inequality_region_l712_71288

theorem inequality_region (x y : ℝ) : 
  x + 3*y - 1 < 0 → (x < 1 - 3*y) ∧ (y < (1 - x)/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_region_l712_71288


namespace NUMINAMATH_CALUDE_walts_investment_rate_l712_71230

/-- Proves that given the conditions of Walt's investment, the unknown interest rate is 8% -/
theorem walts_investment_rate : ∀ (total_money : ℝ) (known_rate : ℝ) (unknown_amount : ℝ) (total_interest : ℝ),
  total_money = 9000 →
  known_rate = 0.09 →
  unknown_amount = 4000 →
  total_interest = 770 →
  ∃ (unknown_rate : ℝ),
    unknown_rate * unknown_amount + known_rate * (total_money - unknown_amount) = total_interest ∧
    unknown_rate = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_walts_investment_rate_l712_71230


namespace NUMINAMATH_CALUDE_unique_positive_solution_l712_71257

theorem unique_positive_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ > 0) (h2 : x₂ > 0) (h3 : x₃ > 0) (h4 : x₄ > 0) (h5 : x₅ > 0)
  (eq1 : x₁ + x₂ = x₃^2)
  (eq2 : x₂ + x₃ = x₄^2)
  (eq3 : x₃ + x₄ = x₅^2)
  (eq4 : x₄ + x₅ = x₁^2)
  (eq5 : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := by
  sorry

#check unique_positive_solution

end NUMINAMATH_CALUDE_unique_positive_solution_l712_71257


namespace NUMINAMATH_CALUDE_distance_ratio_is_two_thirds_l712_71231

/-- Yan's position between home and stadium -/
structure Position where
  distanceToHome : ℝ
  distanceToStadium : ℝ
  isBetween : 0 < distanceToHome ∧ 0 < distanceToStadium

/-- Yan's travel speeds -/
structure Speeds where
  walkingSpeed : ℝ
  bicycleSpeed : ℝ
  bicycleFaster : bicycleSpeed = 5 * walkingSpeed

/-- The theorem stating the ratio of distances -/
theorem distance_ratio_is_two_thirds 
  (pos : Position) (speeds : Speeds) 
  (equalTime : pos.distanceToStadium / speeds.walkingSpeed = 
               pos.distanceToHome / speeds.walkingSpeed + 
               (pos.distanceToHome + pos.distanceToStadium) / speeds.bicycleSpeed) :
  pos.distanceToHome / pos.distanceToStadium = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_ratio_is_two_thirds_l712_71231


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l712_71214

/-- Given a positive integer n, if the sum of the binomial coefficients of the first three terms
    in the expansion of (1/2 + 2x)^n equals 79, then n = 12 and the 11th term has the largest coefficient -/
theorem binomial_expansion_property (n : ℕ) (hn : n > 0) 
  (h_sum : Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2 = 79) : 
  n = 12 ∧ ∀ k, 0 ≤ k ∧ k ≤ 12 → 
    Nat.choose 12 10 * 4^10 ≥ Nat.choose 12 k * 4^k := by
  sorry


end NUMINAMATH_CALUDE_binomial_expansion_property_l712_71214


namespace NUMINAMATH_CALUDE_ratio_to_eight_l712_71201

theorem ratio_to_eight : ∃ x : ℚ, (5 : ℚ) / 1 = x / 8 ∧ x = 40 := by sorry

end NUMINAMATH_CALUDE_ratio_to_eight_l712_71201


namespace NUMINAMATH_CALUDE_subset_implies_a_less_than_neg_two_l712_71256

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem subset_implies_a_less_than_neg_two (a : ℝ) : A ⊆ B a → a < -2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_less_than_neg_two_l712_71256


namespace NUMINAMATH_CALUDE_zach_rental_cost_l712_71253

/-- Calculates the total cost of renting a car given the base cost, per-mile cost, and miles driven -/
def total_rental_cost (base_cost : ℝ) (per_mile_cost : ℝ) (miles_monday : ℝ) (miles_thursday : ℝ) : ℝ :=
  base_cost + per_mile_cost * (miles_monday + miles_thursday)

/-- Theorem: Given the rental conditions, Zach's total cost is $832 -/
theorem zach_rental_cost :
  total_rental_cost 150 0.5 620 744 = 832 := by
  sorry

#eval total_rental_cost 150 0.5 620 744

end NUMINAMATH_CALUDE_zach_rental_cost_l712_71253


namespace NUMINAMATH_CALUDE_problem_solution_l712_71216

theorem problem_solution : (-1)^2023 + |Real.sqrt 3 - 3| + Real.sqrt 9 - (-4) * (1/2) = 7 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l712_71216


namespace NUMINAMATH_CALUDE_triangle_properties_l712_71224

-- Define the triangle
def Triangle (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let DC := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  AB = 25 ∧
  (B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2) = 0 ∧ -- Right angle at D
  AD / AB = 4 / 5 ∧ -- sin A = 4/5
  BD / BC = 1 / 5   -- sin C = 1/5

-- Theorem statement
theorem triangle_properties (A B C D : ℝ × ℝ) (h : Triangle A B C D) :
  let DC := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  DC = 40 * Real.sqrt 6 ∧
  1/2 * AD * BD = 150 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l712_71224


namespace NUMINAMATH_CALUDE_nine_point_circle_chords_l712_71261

/-- The number of chords that can be drawn in a circle with n points on its circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords that can be drawn in a circle with 9 points on its circumference is 36 -/
theorem nine_point_circle_chords : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_circle_chords_l712_71261


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_l712_71263

theorem square_reciprocal_sum (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 4 = 102 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_l712_71263


namespace NUMINAMATH_CALUDE_road_renovation_rates_l712_71220

-- Define the daily renovation rates for Team A and Team B
def daily_rate_A (x : ℝ) : ℝ := x + 20
def daily_rate_B (x : ℝ) : ℝ := x

-- Define the condition that the time to renovate 200m for Team A equals the time to renovate 150m for Team B
def time_equality (x : ℝ) : Prop := 200 / (daily_rate_A x) = 150 / (daily_rate_B x)

-- Theorem stating the solution
theorem road_renovation_rates :
  ∃ x : ℝ, time_equality x ∧ daily_rate_A x = 80 ∧ daily_rate_B x = 60 := by
  sorry

end NUMINAMATH_CALUDE_road_renovation_rates_l712_71220


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l712_71272

open Set

universe u

theorem complement_intersection_theorem (U M N : Set ℕ) :
  U = {0, 1, 2, 3, 4} →
  M = {0, 1, 2} →
  N = {2, 3} →
  (U \ M) ∩ N = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l712_71272


namespace NUMINAMATH_CALUDE_tinas_money_left_l712_71265

/-- Calculates the amount of money Tina has left after saving and spending --/
theorem tinas_money_left (june_savings july_savings august_savings : ℕ) 
  (book_expense shoe_expense : ℕ) : 
  june_savings = 27 →
  july_savings = 14 →
  august_savings = 21 →
  book_expense = 5 →
  shoe_expense = 17 →
  (june_savings + july_savings + august_savings) - (book_expense + shoe_expense) = 40 := by
sorry


end NUMINAMATH_CALUDE_tinas_money_left_l712_71265


namespace NUMINAMATH_CALUDE_third_house_price_l712_71264

/-- Brian's commission rate as a decimal -/
def commission_rate : ℚ := 0.02

/-- Selling price of the first house -/
def house1_price : ℚ := 157000

/-- Selling price of the second house -/
def house2_price : ℚ := 499000

/-- Total commission Brian earned from all three sales -/
def total_commission : ℚ := 15620

/-- The selling price of the third house -/
def house3_price : ℚ := (total_commission - (house1_price * commission_rate + house2_price * commission_rate)) / commission_rate

theorem third_house_price :
  house3_price = 125000 :=
by sorry

end NUMINAMATH_CALUDE_third_house_price_l712_71264


namespace NUMINAMATH_CALUDE_jingyuetan_park_probability_l712_71276

theorem jingyuetan_park_probability (total_envelopes : ℕ) (jingyuetan_tickets : ℕ) 
  (changying_tickets : ℕ) (h1 : total_envelopes = 5) (h2 : jingyuetan_tickets = 3) 
  (h3 : changying_tickets = 2) (h4 : total_envelopes = jingyuetan_tickets + changying_tickets) :
  (jingyuetan_tickets : ℚ) / total_envelopes = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_jingyuetan_park_probability_l712_71276


namespace NUMINAMATH_CALUDE_two_distinct_roots_l712_71235

theorem two_distinct_roots
  (a b c d : ℝ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (f_no_roots : ∀ x : ℝ, x^2 + b*x + a ≠ 0)
  (g_condition1 : a^2 + c*a + d = b)
  (g_condition2 : b^2 + c*b + d = a) :
  ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l712_71235


namespace NUMINAMATH_CALUDE_cricketer_bowling_runs_cricketer_last_match_runs_l712_71236

theorem cricketer_bowling_runs (initial_average : ℝ) (initial_wickets : ℕ) 
  (last_match_wickets : ℕ) (average_decrease : ℝ) : ℝ :=
  let final_average := initial_average - average_decrease
  let total_wickets := initial_wickets + last_match_wickets
  let initial_runs := initial_average * initial_wickets
  let final_runs := final_average * total_wickets
  final_runs - initial_runs

theorem cricketer_last_match_runs : 
  cricketer_bowling_runs 12.4 85 5 0.4 = 26 := by sorry

end NUMINAMATH_CALUDE_cricketer_bowling_runs_cricketer_last_match_runs_l712_71236


namespace NUMINAMATH_CALUDE_square_root_problem_l712_71291

theorem square_root_problem (m n : ℝ) 
  (h1 : (5*m - 2)^(1/3) = -3) 
  (h2 : Real.sqrt (3*m + 2*n - 1) = 4) : 
  Real.sqrt (2*m + n + 10) = 4 ∨ Real.sqrt (2*m + n + 10) = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l712_71291


namespace NUMINAMATH_CALUDE_brocard_and_steiner_coordinates_l712_71273

/-- Given a triangle with side lengths a, b, and c, this theorem states the trilinear coordinates
    of vertex A1 of the Brocard triangle and the Steiner point. -/
theorem brocard_and_steiner_coordinates (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (k₁ k₂ : ℝ),
    k₁ > 0 ∧ k₂ > 0 ∧
    (k₁ * (a * b * c), k₁ * c^3, k₁ * b^3) = (1, 1, 1) ∧
    (k₂ / (a * (b^2 - c^2)), k₂ / (b * (c^2 - a^2)), k₂ / (c * (a^2 - b^2))) = (1, 1, 1) :=
by sorry

end NUMINAMATH_CALUDE_brocard_and_steiner_coordinates_l712_71273


namespace NUMINAMATH_CALUDE_max_product_sum_2024_l712_71251

theorem max_product_sum_2024 :
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144) ∧
  (∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2024_l712_71251


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l712_71218

/-- Prove that the given expression evaluates to 1/5 -/
theorem complex_fraction_evaluation :
  (⌈(19 / 6 : ℚ) - ⌈(34 / 21 : ℚ)⌉⌉ : ℚ) / (⌈(34 / 6 : ℚ) + ⌈(6 * 19 / 34 : ℚ)⌉⌉ : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l712_71218


namespace NUMINAMATH_CALUDE_total_passengers_four_trips_l712_71241

/-- Calculates the total number of passengers transported in multiple round trips -/
def total_passengers (passengers_one_way : ℕ) (passengers_return : ℕ) (num_round_trips : ℕ) : ℕ :=
  (passengers_one_way + passengers_return) * num_round_trips

/-- Theorem stating that the total number of passengers transported in 4 round trips is 640 -/
theorem total_passengers_four_trips :
  total_passengers 100 60 4 = 640 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_four_trips_l712_71241


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l712_71267

/-- For a quadratic equation x^2 + mx + k = 0 with real coefficients,
    this function determines if it has two distinct real roots. -/
def has_two_distinct_real_roots (m k : ℝ) : Prop :=
  k > 0 ∧ (m < -2 * Real.sqrt k ∨ m > 2 * Real.sqrt k)

/-- Theorem stating the conditions for a quadratic equation to have two distinct real roots. -/
theorem quadratic_two_distinct_roots (m k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + k = 0 ∧ y^2 + m*y + k = 0) ↔ has_two_distinct_real_roots m k :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l712_71267


namespace NUMINAMATH_CALUDE_min_value_expression_l712_71245

theorem min_value_expression (α β : ℝ) (h1 : α ≠ 0) (h2 : |β| = 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), x = |((β + α) / (1 + α * β))| → x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l712_71245


namespace NUMINAMATH_CALUDE_soccer_league_games_l712_71284

theorem soccer_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 10) (h2 : total_games = 45) :
  (n * (n - 1)) / 2 = total_games → ∃ k : ℕ, k = 1 ∧ k * (n * (n - 1)) / 2 = total_games :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_games_l712_71284


namespace NUMINAMATH_CALUDE_simplify_expression_l712_71200

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) (hx2 : x ≠ 2) :
  (x - 2) / (x^2) / (1 - 2/x) = 1/x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l712_71200


namespace NUMINAMATH_CALUDE_exponential_multiplication_specific_exponential_multiplication_l712_71221

theorem exponential_multiplication (n : ℕ) : (10 : ℝ) ^ n * (10 : ℝ) ^ n = (10 : ℝ) ^ (2 * n) := by
  sorry

-- The specific case for n = 1000
theorem specific_exponential_multiplication : (10 : ℝ) ^ 1000 * (10 : ℝ) ^ 1000 = (10 : ℝ) ^ 2000 := by
  sorry

end NUMINAMATH_CALUDE_exponential_multiplication_specific_exponential_multiplication_l712_71221


namespace NUMINAMATH_CALUDE_max_odd_integers_in_even_product_l712_71237

theorem max_odd_integers_in_even_product (integers : Finset ℕ) :
  integers.card = 6 ∧
  (∀ n ∈ integers, n > 0) ∧
  Even (integers.prod id) →
  (integers.filter Odd).card ≤ 5 ∧
  ∃ (subset : Finset ℕ),
    subset ⊆ integers ∧
    subset.card = 5 ∧
    ∀ n ∈ subset, Odd n :=
by sorry

end NUMINAMATH_CALUDE_max_odd_integers_in_even_product_l712_71237


namespace NUMINAMATH_CALUDE_glenn_total_spent_l712_71270

/-- The cost of a movie ticket on Monday -/
def monday_price : ℚ := 5

/-- The cost of a movie ticket on Wednesday -/
def wednesday_price : ℚ := 2 * monday_price

/-- The cost of a movie ticket on Saturday -/
def saturday_price : ℚ := 5 * monday_price

/-- The discount rate for Wednesday -/
def discount_rate : ℚ := 1 / 10

/-- The cost of popcorn and drink on Saturday -/
def popcorn_drink_cost : ℚ := 7

/-- The total amount Glenn spends -/
def total_spent : ℚ := wednesday_price * (1 - discount_rate) + saturday_price + popcorn_drink_cost

theorem glenn_total_spent : total_spent = 41 := by
  sorry

end NUMINAMATH_CALUDE_glenn_total_spent_l712_71270


namespace NUMINAMATH_CALUDE_projectile_max_height_l712_71252

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 116

/-- Theorem stating that the maximum height reached by the projectile is 116 meters -/
theorem projectile_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l712_71252


namespace NUMINAMATH_CALUDE_min_distance_ab_value_l712_71217

theorem min_distance_ab_value (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a^2 - a*b + 1 = 0) (hcd : c^2 + d^2 = 1) :
  let f := fun (x y : ℝ) => (a - x)^2 + (b - y)^2
  ∃ (m : ℝ), (∀ x y, c^2 + d^2 = 1 → f x y ≥ m) ∧ 
             (a * b = Real.sqrt 2 / 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ab_value_l712_71217


namespace NUMINAMATH_CALUDE_point_on_line_l712_71268

/-- Given two points (m, n) and (m + p, n + 18) on the line x = (y / 6) - (2 / 5), prove that p = 3 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 6 - 2 / 5) → 
  (m + p = (n + 18) / 6 - 2 / 5) → 
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l712_71268


namespace NUMINAMATH_CALUDE_rex_cards_left_l712_71226

def nicole_cards : ℕ := 500

def cindy_cards : ℕ := (2 * nicole_cards) + (2 * nicole_cards * 25 / 100)

def total_cards : ℕ := nicole_cards + cindy_cards

def rex_cards : ℕ := (2 * total_cards) / 3

def num_siblings : ℕ := 5

def cards_per_person : ℕ := rex_cards / (num_siblings + 1)

def cards_given_away : ℕ := cards_per_person * num_siblings

theorem rex_cards_left : rex_cards - cards_given_away = 196 := by
  sorry

end NUMINAMATH_CALUDE_rex_cards_left_l712_71226
