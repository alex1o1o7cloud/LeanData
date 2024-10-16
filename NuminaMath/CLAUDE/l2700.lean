import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2700_270018

/-- Arithmetic sequence sum and remainder theorem -/
theorem arithmetic_sequence_sum_remainder
  (a : ℕ) -- First term
  (d : ℕ) -- Common difference
  (l : ℕ) -- Last term
  (h1 : a = 2)
  (h2 : d = 5)
  (h3 : l = 142)
  : (((l - a) / d + 1) * (a + l) / 2) % 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2700_270018


namespace NUMINAMATH_CALUDE_inequality_proof_l2700_270029

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  Real.sqrt (3 * x^2 + x * y) + Real.sqrt (3 * y^2 + y * z) + Real.sqrt (3 * z^2 + z * x) ≤ 2 * (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2700_270029


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2700_270038

/-- The length of a train given specific crossing times -/
theorem train_length (t_man : ℝ) (t_platform : ℝ) (l_platform : ℝ) : ℝ :=
  let train_length := (t_platform * l_platform) / (t_platform - t_man)
  186

/-- The train passes a stationary point in 8 seconds -/
def time_passing_man : ℝ := 8

/-- The train crosses a platform in 20 seconds -/
def time_crossing_platform : ℝ := 20

/-- The length of the platform is 279 meters -/
def platform_length : ℝ := 279

theorem train_length_proof :
  train_length time_passing_man time_crossing_platform platform_length = 186 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2700_270038


namespace NUMINAMATH_CALUDE_kanul_total_amount_l2700_270073

theorem kanul_total_amount 
  (raw_materials : ℝ) 
  (machinery : ℝ) 
  (cash_percentage : ℝ) 
  (total : ℝ) :
  raw_materials = 80000 →
  machinery = 30000 →
  cash_percentage = 0.20 →
  raw_materials + machinery + cash_percentage * total = total →
  total = 137500 :=
by sorry

end NUMINAMATH_CALUDE_kanul_total_amount_l2700_270073


namespace NUMINAMATH_CALUDE_solid_color_marbles_l2700_270099

theorem solid_color_marbles (total_marbles : ℕ) (solid_color_percent : ℚ) (solid_yellow_percent : ℚ)
  (h1 : solid_color_percent = 90 / 100)
  (h2 : solid_yellow_percent = 5 / 100) :
  solid_color_percent - solid_yellow_percent = 85 / 100 := by
  sorry

end NUMINAMATH_CALUDE_solid_color_marbles_l2700_270099


namespace NUMINAMATH_CALUDE_douglas_county_y_percentage_l2700_270064

-- Define the ratio of voters in county X to county Y
def voter_ratio : ℚ := 2 / 1

-- Define the percentage of votes Douglas won in both counties combined
def total_vote_percentage : ℚ := 60 / 100

-- Define the percentage of votes Douglas won in county X
def county_x_percentage : ℚ := 72 / 100

-- Theorem to prove
theorem douglas_county_y_percentage :
  let total_voters := 3 -- represents the sum of parts in the ratio (2 + 1)
  let county_x_voters := 2 -- represents the larger part of the ratio
  let county_y_voters := 1 -- represents the smaller part of the ratio
  let total_douglas_votes := total_vote_percentage * total_voters
  let county_x_douglas_votes := county_x_percentage * county_x_voters
  let county_y_douglas_votes := total_douglas_votes - county_x_douglas_votes
  county_y_douglas_votes / county_y_voters = 36 / 100 :=
by sorry

end NUMINAMATH_CALUDE_douglas_county_y_percentage_l2700_270064


namespace NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l2700_270075

theorem fair_coin_three_heads_probability :
  let p_head : ℝ := 1/2  -- Probability of getting heads on a fair coin
  let n : ℕ := 3        -- Number of tosses
  let p_all_heads : ℝ := p_head ^ n
  p_all_heads = 1/8 := by
sorry

end NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l2700_270075


namespace NUMINAMATH_CALUDE_dans_marbles_l2700_270072

theorem dans_marbles (dans_marbles mary_marbles : ℕ) 
  (h1 : mary_marbles = 2 * dans_marbles)
  (h2 : mary_marbles = 10) : 
  dans_marbles = 5 := by
sorry

end NUMINAMATH_CALUDE_dans_marbles_l2700_270072


namespace NUMINAMATH_CALUDE_square_area_and_perimeter_l2700_270028

/-- Given a square with diagonal length 12√2 cm, prove its area and perimeter -/
theorem square_area_and_perimeter (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  (s ^ 2 = 144) ∧ (4 * s = 48) := by sorry

end NUMINAMATH_CALUDE_square_area_and_perimeter_l2700_270028


namespace NUMINAMATH_CALUDE_temperature_at_midnight_l2700_270010

/-- Given temperature changes throughout a day, calculate the temperature at midnight. -/
theorem temperature_at_midnight 
  (morning_temp : Int) 
  (noon_rise : Int) 
  (midnight_drop : Int) 
  (h1 : morning_temp = -2)
  (h2 : noon_rise = 13)
  (h3 : midnight_drop = 8) : 
  morning_temp + noon_rise - midnight_drop = 3 :=
by sorry

end NUMINAMATH_CALUDE_temperature_at_midnight_l2700_270010


namespace NUMINAMATH_CALUDE_cone_radius_from_slant_height_and_surface_area_l2700_270008

/-- Given a cone with slant height 10 cm and curved surface area 157.07963267948966 cm²,
    the radius of the base is 5 cm. -/
theorem cone_radius_from_slant_height_and_surface_area :
  let slant_height : ℝ := 10
  let curved_surface_area : ℝ := 157.07963267948966
  let radius : ℝ := curved_surface_area / (Real.pi * slant_height)
  radius = 5 := by sorry

end NUMINAMATH_CALUDE_cone_radius_from_slant_height_and_surface_area_l2700_270008


namespace NUMINAMATH_CALUDE_exists_valid_strategy_l2700_270026

/-- Represents the problem of the father and two sons visiting their grandmother --/
structure VisitProblem where
  distance : ℝ
  scooter_speed_alone : ℝ
  scooter_speed_with_passenger : ℝ
  walking_speed : ℝ

/-- Defines the specific problem instance --/
def problem : VisitProblem :=
  { distance := 33
  , scooter_speed_alone := 25
  , scooter_speed_with_passenger := 20
  , walking_speed := 5
  }

/-- Represents a solution strategy for the visit problem --/
structure Strategy where
  (p : VisitProblem)
  travel_time : ℝ

/-- Predicate to check if a strategy is valid --/
def is_valid_strategy (s : Strategy) : Prop :=
  s.travel_time ≤ 3 ∧
  ∃ (t1 t2 t3 : ℝ),
    t1 ≤ s.travel_time ∧
    t2 ≤ s.travel_time ∧
    t3 ≤ s.travel_time ∧
    s.p.distance / s.p.walking_speed ≤ t1 ∧
    s.p.distance / s.p.walking_speed ≤ t2 ∧
    s.p.distance / s.p.scooter_speed_alone ≤ t3

/-- Theorem stating that there exists a valid strategy for the given problem --/
theorem exists_valid_strategy :
  ∃ (s : Strategy), s.p = problem ∧ is_valid_strategy s :=
sorry


end NUMINAMATH_CALUDE_exists_valid_strategy_l2700_270026


namespace NUMINAMATH_CALUDE_seventh_person_age_l2700_270048

theorem seventh_person_age
  (n : ℕ)
  (initial_people : ℕ)
  (future_average : ℕ)
  (new_average : ℕ)
  (years_passed : ℕ)
  (h1 : initial_people = 6)
  (h2 : future_average = 43)
  (h3 : new_average = 45)
  (h4 : years_passed = 2)
  (h5 : n = initial_people + 1) :
  (n * new_average) - (initial_people * (future_average + years_passed)) = 69 := by
  sorry

end NUMINAMATH_CALUDE_seventh_person_age_l2700_270048


namespace NUMINAMATH_CALUDE_min_distance_line_curve_l2700_270071

/-- The minimum distance between a point on the line 2x - y + 6 = 0 
    and a point on the curve y = 2ln x + 2 -/
theorem min_distance_line_curve : ∃ d : ℝ, d = (6 * Real.sqrt 5) / 5 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (2 * x₁ - y₁ + 6 = 0) →
    (y₂ = 2 * Real.log x₂ + 2) →
    d ≤ Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_curve_l2700_270071


namespace NUMINAMATH_CALUDE_first_term_of_sequence_l2700_270032

theorem first_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2 + 1) : a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_sequence_l2700_270032


namespace NUMINAMATH_CALUDE_sqrt_a_plus_one_range_l2700_270050

theorem sqrt_a_plus_one_range :
  ∀ a : ℝ, (∃ x : ℝ, x^2 = a + 1) ↔ a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_one_range_l2700_270050


namespace NUMINAMATH_CALUDE_wrong_mark_value_l2700_270047

/-- Proves that the wrongly entered mark is 85 given the conditions of the problem -/
theorem wrong_mark_value (n : ℕ) (correct_mark : ℕ) (average_increase : ℚ) 
  (h1 : n = 104) 
  (h2 : correct_mark = 33) 
  (h3 : average_increase = 1/2) : 
  ∃ x : ℕ, x = 85 ∧ (x - correct_mark : ℚ) = average_increase * n := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_value_l2700_270047


namespace NUMINAMATH_CALUDE_circle_and_tangents_l2700_270090

-- Define the points
def A : ℝ × ℝ := (-1, -3)
def B : ℝ × ℝ := (5, 5)
def M : ℝ × ℝ := (-3, 2)

-- Define the circle O
def O : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 25}

-- Define the tangent lines
def tangent_lines : Set (Set (ℝ × ℝ)) := 
  {{p | p.1 = -3}, {p | 12 * p.1 - 5 * p.2 + 46 = 0}}

theorem circle_and_tangents :
  (∀ p ∈ O, (p.1 - 2)^2 + (p.2 - 1)^2 = 25) ∧
  (∀ l ∈ tangent_lines, ∃ p ∈ O, p ∈ l ∧ 
    (∀ q ∈ O, q ≠ p → q ∉ l)) ∧
  (∀ l ∈ tangent_lines, M ∈ l) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangents_l2700_270090


namespace NUMINAMATH_CALUDE_expression_evaluation_l2700_270004

theorem expression_evaluation : 
  let x : ℚ := -2
  let y : ℚ := 1/2
  ((x + 2*y)^2 - (x + y)*(3*x - y) - 5*y^2) / (2*x) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2700_270004


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2700_270020

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2700_270020


namespace NUMINAMATH_CALUDE_sin_210_degrees_l2700_270042

theorem sin_210_degrees : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l2700_270042


namespace NUMINAMATH_CALUDE_sin_sum_angles_l2700_270035

/-- Given a point A(1, 2) on the terminal side of angle α in the Cartesian plane,
    and angle β formed by rotating α's terminal side counterclockwise by π/2,
    prove that sin(α + β) = -3/5 -/
theorem sin_sum_angles (α β : Real) : 
  (∃ A : ℝ × ℝ, A = (1, 2) ∧ A.1 = Real.cos α * Real.sqrt (A.1^2 + A.2^2) ∧ 
                   A.2 = Real.sin α * Real.sqrt (A.1^2 + A.2^2)) →
  β = α + π/2 →
  Real.sin (α + β) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_angles_l2700_270035


namespace NUMINAMATH_CALUDE_last_number_proof_l2700_270060

theorem last_number_proof (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 6)
  (h2 : a + d = 13)
  (h3 : (b + c + d) / 3 = 3) : 
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l2700_270060


namespace NUMINAMATH_CALUDE_tan_two_implies_sin_2theta_over_cos_squared_minus_sin_squared_l2700_270034

theorem tan_two_implies_sin_2theta_over_cos_squared_minus_sin_squared (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (2 * θ)) / (Real.cos θ ^ 2 - Real.sin θ ^ 2) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_sin_2theta_over_cos_squared_minus_sin_squared_l2700_270034


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2700_270069

def k : ℕ := 2017^2 + 2^2017

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := k) : (k^2 + 2^k) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2700_270069


namespace NUMINAMATH_CALUDE_zachary_pushups_l2700_270002

/-- Given the information about David and Zachary's push-ups and crunches, 
    prove that Zachary did 28 push-ups. -/
theorem zachary_pushups (david_pushups zachary_pushups david_crunches zachary_crunches : ℕ) :
  david_pushups = zachary_pushups + 40 →
  david_crunches + 17 = zachary_crunches →
  david_crunches = 45 →
  zachary_crunches = 62 →
  zachary_pushups = 28 := by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_l2700_270002


namespace NUMINAMATH_CALUDE_article_profit_percentage_l2700_270097

theorem article_profit_percentage (cost : ℝ) (reduced_sell : ℝ) (new_profit_percent : ℝ) :
  cost = 40 →
  reduced_sell = 8.40 →
  new_profit_percent = 30 →
  let new_cost := cost * 0.80
  let new_sell := new_cost * (1 + new_profit_percent / 100)
  let orig_sell := new_sell + reduced_sell
  let profit := orig_sell - cost
  let profit_percent := (profit / cost) * 100
  profit_percent = 25 := by
  sorry

end NUMINAMATH_CALUDE_article_profit_percentage_l2700_270097


namespace NUMINAMATH_CALUDE_ruby_initial_apples_l2700_270005

/-- The number of apples Ruby has initially -/
def initial_apples : ℕ := sorry

/-- The number of apples Emily takes away -/
def apples_taken : ℕ := 55

/-- The number of apples Ruby has left -/
def apples_left : ℕ := 8

/-- Theorem stating that Ruby's initial number of apples is 63 -/
theorem ruby_initial_apples : initial_apples = 63 := by sorry

end NUMINAMATH_CALUDE_ruby_initial_apples_l2700_270005


namespace NUMINAMATH_CALUDE_initial_angelfish_count_l2700_270046

/-- The number of fish initially in the tank -/
def initial_fish (angelfish : ℕ) : ℕ := 94 + angelfish + 89 + 58

/-- The number of fish sold -/
def sold_fish (angelfish : ℕ) : ℕ := 30 + 48 + 17 + 24

/-- The number of fish remaining after the sale -/
def remaining_fish (angelfish : ℕ) : ℕ := initial_fish angelfish - sold_fish angelfish

theorem initial_angelfish_count :
  ∃ (angelfish : ℕ), initial_fish angelfish > 0 ∧ remaining_fish angelfish = 198 ∧ angelfish = 76 := by
  sorry

end NUMINAMATH_CALUDE_initial_angelfish_count_l2700_270046


namespace NUMINAMATH_CALUDE_infiniteSeriesSum_l2700_270045

/-- The sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

/-- Theorem: The sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ is equal to 3/4 -/
theorem infiniteSeriesSum : infiniteSeries = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_infiniteSeriesSum_l2700_270045


namespace NUMINAMATH_CALUDE_christmas_presents_l2700_270080

theorem christmas_presents (birthday_presents christmas_presents : ℕ) : 
  christmas_presents = 2 * birthday_presents →
  christmas_presents + birthday_presents = 90 →
  christmas_presents = 60 := by
sorry

end NUMINAMATH_CALUDE_christmas_presents_l2700_270080


namespace NUMINAMATH_CALUDE_max_sum_with_condition_l2700_270015

/-- Given positive integers a and b not exceeding 100 satisfying the condition,
    the maximum value of a + b is 78. -/
theorem max_sum_with_condition (a b : ℕ) : 
  0 < a ∧ 0 < b ∧ a ≤ 100 ∧ b ≤ 100 →
  a * b = (Nat.lcm a b / Nat.gcd a b) ^ 2 →
  ∀ (x y : ℕ), 0 < x ∧ 0 < y ∧ x ≤ 100 ∧ y ≤ 100 →
    x * y = (Nat.lcm x y / Nat.gcd x y) ^ 2 →
    a + b ≤ 78 ∧ (∃ (a' b' : ℕ), a' + b' = 78 ∧ 
      0 < a' ∧ 0 < b' ∧ a' ≤ 100 ∧ b' ≤ 100 ∧
      a' * b' = (Nat.lcm a' b' / Nat.gcd a' b') ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_condition_l2700_270015


namespace NUMINAMATH_CALUDE_nursery_school_count_nursery_school_count_proof_l2700_270022

theorem nursery_school_count : ℕ → Prop :=
  fun total_students =>
    let students_4_and_older := total_students / 10
    let students_under_3 := 20
    let students_not_between_3_and_4 := 50
    students_4_and_older = students_not_between_3_and_4 - students_under_3 ∧
    total_students = 300

-- The proof of the theorem
theorem nursery_school_count_proof : ∃ n : ℕ, nursery_school_count n :=
  sorry

end NUMINAMATH_CALUDE_nursery_school_count_nursery_school_count_proof_l2700_270022


namespace NUMINAMATH_CALUDE_min_red_edges_six_red_edges_possible_l2700_270041

/-- Represents the color of an edge -/
inductive Color
| Red
| Green

/-- Represents a cube with colored edges -/
structure Cube :=
  (edges : Fin 12 → Color)

/-- Checks if a face has at least one red edge -/
def faceHasRedEdge (c : Cube) (face : Fin 6) : Prop := sorry

/-- The condition that every face of the cube has at least one red edge -/
def everyFaceHasRedEdge (c : Cube) : Prop :=
  ∀ face : Fin 6, faceHasRedEdge c face

/-- Counts the number of red edges in a cube -/
def countRedEdges (c : Cube) : Nat := sorry

/-- Theorem stating that the minimum number of red edges is 6 -/
theorem min_red_edges (c : Cube) (h : everyFaceHasRedEdge c) : 
  countRedEdges c ≥ 6 := sorry

/-- Theorem stating that 6 red edges is achievable -/
theorem six_red_edges_possible : 
  ∃ c : Cube, everyFaceHasRedEdge c ∧ countRedEdges c = 6 := sorry

end NUMINAMATH_CALUDE_min_red_edges_six_red_edges_possible_l2700_270041


namespace NUMINAMATH_CALUDE_function_value_at_two_l2700_270006

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x ≠ 0, f x - 3 * f (1 / x) = 3^x

theorem function_value_at_two
  (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 2 = -(9 + 3 * Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l2700_270006


namespace NUMINAMATH_CALUDE_parabola_equation_l2700_270079

/-- A parabola with vertex at the origin, symmetric about the x-axis, 
    and a chord of length 8 passing through the focus and perpendicular 
    to the axis of symmetry has the equation y² = ±8x -/
theorem parabola_equation (p : Set (ℝ × ℝ)) 
  (vertex_at_origin : (0, 0) ∈ p)
  (symmetric_x_axis : ∀ (x y : ℝ), (x, y) ∈ p ↔ (x, -y) ∈ p)
  (focus_chord_length : ∃ (a : ℝ), a ≠ 0 ∧ 
    (Set.Icc (-a) a).image (λ y => (a/2, y)) ⊆ p ∧
    Set.Icc (-a) a = Set.Icc (-4) 4) :
  p = {(x, y) | y^2 = 8*x ∨ y^2 = -8*x} :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2700_270079


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2700_270055

theorem necessary_not_sufficient_condition (a : ℝ) (h : a ≠ 0) :
  (∀ a, a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2700_270055


namespace NUMINAMATH_CALUDE_students_allowance_l2700_270068

theorem students_allowance (allowance : ℚ) : 
  (allowance > 0) →
  (3/5 * allowance + 1/3 * (2/5 * allowance) + 2/5) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_students_allowance_l2700_270068


namespace NUMINAMATH_CALUDE_max_sign_changes_is_n_minus_one_sign_changes_bounded_l2700_270056

/-- The maximum number of sign changes for the first element in a sequence of n real numbers 
    under the described averaging process. -/
def max_sign_changes (n : ℕ) : ℕ :=
  n - 1

/-- The theorem stating that the maximum number of sign changes for the first element
    is n-1 for any positive integer n. -/
theorem max_sign_changes_is_n_minus_one (n : ℕ) (hn : n > 0) : 
  max_sign_changes n = n - 1 := by
  sorry

/-- A helper function to represent the averaging operation on a sequence of real numbers. -/
def average_operation (seq : List ℝ) (i : ℕ) : List ℝ :=
  sorry

/-- A predicate to check if a number has changed sign. -/
def sign_changed (a b : ℝ) : Prop :=
  (a ≥ 0 ∧ b < 0) ∨ (a < 0 ∧ b ≥ 0)

/-- A function to count the number of sign changes in a₁ after a sequence of operations. -/
def count_sign_changes (initial_seq : List ℝ) (operations : List ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for any initial sequence and any sequence of operations,
    the number of sign changes in a₁ is at most n-1. -/
theorem sign_changes_bounded (n : ℕ) (hn : n > 0) 
  (initial_seq : List ℝ) (h_seq : initial_seq.length = n)
  (operations : List ℕ) :
  count_sign_changes initial_seq operations ≤ max_sign_changes n := by
  sorry

end NUMINAMATH_CALUDE_max_sign_changes_is_n_minus_one_sign_changes_bounded_l2700_270056


namespace NUMINAMATH_CALUDE_book_profit_percentage_l2700_270009

/-- Given a book's cost price and additional information about its profit, 
    calculate the initial profit percentage. -/
theorem book_profit_percentage 
  (cost_price : ℝ) 
  (additional_profit : ℝ) 
  (new_profit_percentage : ℝ) :
  cost_price = 2400 →
  additional_profit = 120 →
  new_profit_percentage = 15 →
  ∃ (initial_profit_percentage : ℝ),
    initial_profit_percentage = 10 ∧
    cost_price * (1 + new_profit_percentage / 100) = 
      cost_price * (1 + initial_profit_percentage / 100) + additional_profit :=
by sorry

end NUMINAMATH_CALUDE_book_profit_percentage_l2700_270009


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l2700_270031

theorem equilateral_triangle_area_decrease :
  ∀ s : ℝ,
  s > 0 →
  (s^2 * Real.sqrt 3) / 4 = 81 * Real.sqrt 3 →
  let s' := s - 3
  let new_area := (s'^2 * Real.sqrt 3) / 4
  let area_decrease := 81 * Real.sqrt 3 - new_area
  area_decrease = 24.75 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l2700_270031


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_planes_parallel_perpendicular_implies_perpendicular_l2700_270043

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (parallelLines : Line → Line → Prop)

-- Theorem 1: If m ⟂ α and m ∥ β, then α ⟂ β
theorem perpendicular_parallel_implies_perpendicular_planes
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicularPlanes α β :=
sorry

-- Theorem 2: If m ∥ n and m ⟂ α, then α ⟂ n
theorem parallel_perpendicular_implies_perpendicular
  (m n : Line) (α : Plane) :
  parallelLines m n → perpendicular m α → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_planes_parallel_perpendicular_implies_perpendicular_l2700_270043


namespace NUMINAMATH_CALUDE_inscribed_rectangle_exists_l2700_270040

/-- Represents a right triangle with sides 3, 4, and 5 -/
structure EgyptianTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 3
  hb : b = 4
  hc : c = 5
  right_angle : a^2 + b^2 = c^2

/-- Represents a rectangle inscribed in the Egyptian triangle -/
structure InscribedRectangle (t : EgyptianTriangle) where
  width : ℝ
  height : ℝ
  ratio : width * 3 = height
  fits_in_triangle : width ≤ t.a ∧ width ≤ t.b ∧ height ≤ t.b ∧ height ≤ t.c

/-- The theorem stating the existence and dimensions of the inscribed rectangle -/
theorem inscribed_rectangle_exists (t : EgyptianTriangle) :
  ∃ (r : InscribedRectangle t), r.width = 20/29 ∧ r.height = 60/29 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_exists_l2700_270040


namespace NUMINAMATH_CALUDE_exponent_sum_theorem_l2700_270053

theorem exponent_sum_theorem : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_theorem_l2700_270053


namespace NUMINAMATH_CALUDE_students_transferred_theorem_l2700_270093

/-- Calculates the number of students transferred to fifth grade -/
def students_transferred_to_fifth (initial_students : ℝ) (students_left : ℝ) (final_students : ℝ) : ℝ :=
  initial_students - students_left - final_students

/-- Proves that the number of students transferred to fifth grade is 10.0 -/
theorem students_transferred_theorem (initial_students : ℝ) (students_left : ℝ) (final_students : ℝ)
  (h1 : initial_students = 42.0)
  (h2 : students_left = 4.0)
  (h3 : final_students = 28.0) :
  students_transferred_to_fifth initial_students students_left final_students = 10.0 := by
  sorry

#eval students_transferred_to_fifth 42.0 4.0 28.0

end NUMINAMATH_CALUDE_students_transferred_theorem_l2700_270093


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l2700_270089

theorem smallest_x_for_equation : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∃ (y : ℕ), y > 0 ∧ (3 : ℚ) / 4 = y / (210 + x)) ∧
  (∀ (x' : ℕ), x' > 0 → x' < x → 
    ¬∃ (y : ℕ), y > 0 ∧ (3 : ℚ) / 4 = y / (210 + x')) ∧
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l2700_270089


namespace NUMINAMATH_CALUDE_wendys_laundry_l2700_270085

theorem wendys_laundry (machine_capacity : ℕ) (num_sweaters : ℕ) (num_loads : ℕ) :
  machine_capacity = 8 →
  num_sweaters = 33 →
  num_loads = 9 →
  num_loads * machine_capacity - num_sweaters = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_wendys_laundry_l2700_270085


namespace NUMINAMATH_CALUDE_candy_bars_total_l2700_270012

theorem candy_bars_total (people : Float) (bars_per_person : Float) : 
  people = 3.0 → 
  bars_per_person = 1.66666666699999 → 
  people * bars_per_person = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_candy_bars_total_l2700_270012


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l2700_270001

-- Define the arithmetic sequence
def a (n : ℕ) : ℚ := 2 * n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 2^(n-1) * a n

-- Define the sum of the first n terms of b_n
def T (n : ℕ) : ℚ := (n - 1) * 2^(n + 1) + 2

-- State the theorem
theorem arithmetic_sequence_theorem (d : ℚ) (h_d : d ≠ 0) :
  (a 2 + 2 * a 4 = 20) ∧
  (∃ r : ℚ, a 3 = r * a 1 ∧ a 9 = r * a 3) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n) ∧
  (∀ n : ℕ, T n = (n - 1) * 2^(n + 1) + 2) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l2700_270001


namespace NUMINAMATH_CALUDE_book_pages_l2700_270027

/-- A book with a certain number of pages -/
structure Book where
  pages : ℕ

/-- Reading progress over four days -/
structure ReadingProgress where
  day1 : Rat
  day2 : Rat
  day3 : Rat
  day4 : ℕ

/-- Theorem stating the total number of pages in the book -/
theorem book_pages (b : Book) (rp : ReadingProgress) 
  (h1 : rp.day1 = 1/2)
  (h2 : rp.day2 = 1/4)
  (h3 : rp.day3 = 1/6)
  (h4 : rp.day4 = 20)
  (h5 : rp.day1 + rp.day2 + rp.day3 + (rp.day4 : Rat) / b.pages = 1) :
  b.pages = 240 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l2700_270027


namespace NUMINAMATH_CALUDE_approximation_place_l2700_270007

/-- A function that returns the number of decimal places in a given number -/
def decimal_places (x : ℚ) : ℕ := sorry

/-- A function that returns the name of the decimal place given its position -/
def place_name (n : ℕ) : String := sorry

theorem approximation_place (x : ℚ) (h : decimal_places x = 2) :
  place_name (decimal_places x) = "hundredths" := by sorry

end NUMINAMATH_CALUDE_approximation_place_l2700_270007


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2700_270095

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Calculate total sum after simple interest -/
def total_sum (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal + simple_interest principal rate time

theorem simple_interest_calculation (P : ℚ) :
  total_sum P (5 : ℚ) (5 : ℚ) = 16065 →
  simple_interest P (5 : ℚ) (5 : ℚ) = 3213 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l2700_270095


namespace NUMINAMATH_CALUDE_mileage_reimbursement_rate_calculation_l2700_270036

/-- Calculates the mileage reimbursement rate given daily mileages and total reimbursement -/
def mileage_reimbursement_rate (monday_miles tuesday_miles wednesday_miles thursday_miles friday_miles total_reimbursement : ℚ) : ℚ :=
  total_reimbursement / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles)

theorem mileage_reimbursement_rate_calculation 
  (monday_miles tuesday_miles wednesday_miles thursday_miles friday_miles total_reimbursement : ℚ) :
  mileage_reimbursement_rate monday_miles tuesday_miles wednesday_miles thursday_miles friday_miles total_reimbursement =
  total_reimbursement / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles) :=
by sorry

end NUMINAMATH_CALUDE_mileage_reimbursement_rate_calculation_l2700_270036


namespace NUMINAMATH_CALUDE_paulines_dress_cost_l2700_270096

theorem paulines_dress_cost (pauline ida jean patty : ℕ) 
  (h1 : patty = ida + 10)
  (h2 : ida = jean + 30)
  (h3 : jean = pauline - 10)
  (h4 : pauline + ida + jean + patty = 160) :
  pauline = 30 := by
  sorry

end NUMINAMATH_CALUDE_paulines_dress_cost_l2700_270096


namespace NUMINAMATH_CALUDE_expression_evaluation_l2700_270003

theorem expression_evaluation : 
  Real.sqrt ((-3)^2) + (π - 3)^0 - 8^(2/3) + ((-4)^(1/3))^3 = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2700_270003


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l2700_270058

/-- Proves that a train 150 meters long running at 90 km/hr takes 6 seconds to pass a pole. -/
theorem train_passing_pole_time (train_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 150 ∧ train_speed_kmh = 90 →
  (train_length / (train_speed_kmh * (1000 / 3600))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l2700_270058


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2700_270066

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2700_270066


namespace NUMINAMATH_CALUDE_range_of_c_l2700_270092

/-- The range of c for which y = c^x is a decreasing function and x^2 - √2x + c > 0 does not hold for all x ∈ ℝ -/
theorem range_of_c (c : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → c^x₁ > c^x₂) → -- y = c^x is a decreasing function
  (¬∀ x : ℝ, x^2 - Real.sqrt 2 * x + c > 0) → -- negation of q
  ((∀ x₁ x₂ : ℝ, x₁ < x₂ → c^x₁ > c^x₂) ∨ (∀ x : ℝ, x^2 - Real.sqrt 2 * x + c > 0)) → -- p or q
  0 < c ∧ c ≤ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_range_of_c_l2700_270092


namespace NUMINAMATH_CALUDE_key_chain_profit_percentage_l2700_270067

theorem key_chain_profit_percentage 
  (selling_price : ℝ)
  (old_cost new_cost : ℝ)
  (h1 : old_cost = 65)
  (h2 : new_cost = 50)
  (h3 : selling_price - new_cost = 0.5 * selling_price) :
  (selling_price - old_cost) / selling_price = 0.35 :=
by sorry

end NUMINAMATH_CALUDE_key_chain_profit_percentage_l2700_270067


namespace NUMINAMATH_CALUDE_modified_triangle_property_unbounded_l2700_270039

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- A function that checks if a set of 10 consecutive integers contains a right triangle -/
def has_right_triangle (start : ℕ) : Prop :=
  ∃ (a b c : ℕ), start ≤ a ∧ a < b ∧ b < c ∧ c < start + 10 ∧ is_right_triangle a b c

/-- The main theorem stating that for any k ≥ 10, the set {5, 6, ..., k} 
    satisfies the modified triangle property for all 10-element subsets -/
theorem modified_triangle_property_unbounded (k : ℕ) (h : k ≥ 10) :
  ∀ (n : ℕ), 5 ≤ n ∧ n ≤ k - 9 → has_right_triangle n :=
sorry

end NUMINAMATH_CALUDE_modified_triangle_property_unbounded_l2700_270039


namespace NUMINAMATH_CALUDE_no_real_solutions_l2700_270088

theorem no_real_solutions :
  ¬∃ (z : ℝ), (3*z - 9*z + 27)^2 + 4 = -2*(abs z) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2700_270088


namespace NUMINAMATH_CALUDE_problem_solution_l2700_270033

-- Define the condition from the problem
def condition (m : ℝ) : Prop :=
  ∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2

-- Define the theorem
theorem problem_solution :
  (∃ m : ℝ, condition m) →
  (∀ m : ℝ, condition m → 1 ≤ m ∧ m ≤ 5) ∧
  (∀ x y z : ℝ, 3*x + 4*y + 5*z = 5 → x^2 + y^2 + z^2 ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2700_270033


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l2700_270081

theorem real_part_of_complex_product : ∃ z : ℂ, z = (1 - Complex.I) * (2 + Complex.I) ∧ z.re = 3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l2700_270081


namespace NUMINAMATH_CALUDE_distinct_positive_solutions_l2700_270049

theorem distinct_positive_solutions (a b : ℝ) :
  (∃ (x y z : ℝ), x + y + z = a ∧ x^2 + y^2 + z^2 = b^2 ∧ x*y = z^2 ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔
  (abs b < a ∧ a < Real.sqrt 3 * abs b) :=
sorry

end NUMINAMATH_CALUDE_distinct_positive_solutions_l2700_270049


namespace NUMINAMATH_CALUDE_sum_of_twelve_terms_special_case_l2700_270062

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- The common difference
  h : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_of_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

theorem sum_of_twelve_terms_special_case (seq : ArithmeticSequence) 
  (h₁ : seq.a 5 = 1)
  (h₂ : seq.a 17 = 18) :
  sum_of_terms seq 12 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_twelve_terms_special_case_l2700_270062


namespace NUMINAMATH_CALUDE_limit_proof_l2700_270082

/-- The limit of (3x^2 + 5x - 2) / (x + 2) as x approaches -2 is -7 -/
theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -2 → |x + 2| < δ →
    |(3*x^2 + 5*x - 2) / (x + 2) + 7| < ε :=
by
  use ε/3
  sorry

end NUMINAMATH_CALUDE_limit_proof_l2700_270082


namespace NUMINAMATH_CALUDE_grid_toothpicks_count_l2700_270076

/-- Calculates the total number of toothpicks in a rectangular grid with partitions. -/
def total_toothpicks (height width : ℕ) (partition_interval : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  let horizontal_toothpicks := horizontal_lines * width
  let vertical_toothpicks := vertical_lines * height
  let num_partitions := (height - 1) / partition_interval
  let partition_toothpicks := num_partitions * width
  horizontal_toothpicks + vertical_toothpicks + partition_toothpicks

/-- Theorem stating that the total number of toothpicks in the specified grid is 850. -/
theorem grid_toothpicks_count :
  total_toothpicks 25 15 5 = 850 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_count_l2700_270076


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2700_270000

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2700_270000


namespace NUMINAMATH_CALUDE_donna_has_40_bananas_l2700_270044

/-- The number of bananas Donna has -/
def donnas_bananas (total : ℕ) (dawns_extra : ℕ) (lydias : ℕ) : ℕ :=
  total - (lydias + dawns_extra) - lydias

/-- Proof that Donna has 40 bananas -/
theorem donna_has_40_bananas :
  donnas_bananas 200 40 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_donna_has_40_bananas_l2700_270044


namespace NUMINAMATH_CALUDE_journey_length_is_70_l2700_270030

-- Define the journey
def Journey (length : ℝ) : Prop :=
  -- Time taken at 40 kmph
  let time_at_40 := length / 40
  -- Time taken at 35 kmph
  let time_at_35 := length / 35
  -- The difference in time is 0.25 hours (15 minutes)
  time_at_35 - time_at_40 = 0.25

-- Theorem stating that the journey length is 70 km
theorem journey_length_is_70 : 
  ∃ (length : ℝ), Journey length ∧ length = 70 :=
sorry

end NUMINAMATH_CALUDE_journey_length_is_70_l2700_270030


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2700_270024

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 31 → left = 5 → new = 11 → final = initial - left + new → final = 37 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2700_270024


namespace NUMINAMATH_CALUDE_perpendicular_bisectors_intersection_l2700_270052

-- Define a triangle as a structure with three points in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem perpendicular_bisectors_intersection (t : Triangle) :
  ∃! O : ℝ × ℝ, distance O t.A = distance O t.B ∧ distance O t.A = distance O t.C := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisectors_intersection_l2700_270052


namespace NUMINAMATH_CALUDE_collinearity_condition_l2700_270037

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Three points are collinear if the area of the triangle formed by them is zero -/
def collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) = (C.x - A.x) * (B.y - A.y)

theorem collinearity_condition (n : ℝ) : 
  let A : Point := ⟨1, 1⟩
  let B : Point := ⟨4, 0⟩
  let C : Point := ⟨0, n⟩
  collinear A B C ↔ n = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_collinearity_condition_l2700_270037


namespace NUMINAMATH_CALUDE_x_value_l2700_270086

theorem x_value (x : ℝ) : x = 150 * (1 + 0.75) → x = 262.5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2700_270086


namespace NUMINAMATH_CALUDE_butter_problem_l2700_270084

theorem butter_problem (B : ℝ) : 
  (B / 2 + B / 5 + (B - B / 2 - B / 5) / 3 + 2 = B) → B = 10 :=
by sorry

end NUMINAMATH_CALUDE_butter_problem_l2700_270084


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2700_270061

/-- Calculates the average speed of a car trip given odometer readings and time taken -/
theorem average_speed_calculation
  (initial_reading : ℝ)
  (lunch_reading : ℝ)
  (final_reading : ℝ)
  (total_time : ℝ)
  (h1 : initial_reading < lunch_reading)
  (h2 : lunch_reading < final_reading)
  (h3 : total_time > 0) :
  let total_distance := final_reading - initial_reading
  (total_distance / total_time) = (final_reading - initial_reading) / total_time :=
by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2700_270061


namespace NUMINAMATH_CALUDE_golf_ball_difference_l2700_270017

theorem golf_ball_difference (bin_f bin_g : ℕ) : 
  bin_f = (2 * bin_g) / 3 →
  bin_f + bin_g = 150 →
  bin_g - bin_f = 30 := by
sorry

end NUMINAMATH_CALUDE_golf_ball_difference_l2700_270017


namespace NUMINAMATH_CALUDE_tim_cantaloupes_count_l2700_270014

/-- The number of cantaloupes Fred grew -/
def fred_cantaloupes : ℕ := 38

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Tim grew -/
def tim_cantaloupes : ℕ := total_cantaloupes - fred_cantaloupes

theorem tim_cantaloupes_count : tim_cantaloupes = 44 := by
  sorry

end NUMINAMATH_CALUDE_tim_cantaloupes_count_l2700_270014


namespace NUMINAMATH_CALUDE_smallest_multiple_l2700_270011

theorem smallest_multiple (n : ℕ) : n = 255 ↔ 
  (∃ k : ℕ, n = 15 * k) ∧ 
  (∃ m : ℕ, n = 65 * m + 7) ∧ 
  (∃ p : ℕ, n = 5 * p) ∧ 
  (∀ x : ℕ, x < n → 
    (¬(∃ k : ℕ, x = 15 * k) ∨ 
     ¬(∃ m : ℕ, x = 65 * m + 7) ∨ 
     ¬(∃ p : ℕ, x = 5 * p))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2700_270011


namespace NUMINAMATH_CALUDE_marble_selection_ways_l2700_270021

def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def red_marbles : ℕ := 1

def choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem marble_selection_ways : 
  choose_marbles (total_marbles - red_marbles) (marbles_to_choose - red_marbles) = 56 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l2700_270021


namespace NUMINAMATH_CALUDE_candy_cost_l2700_270023

/-- 
Given that each piece of bulk candy costs 8 cents and 28 gumdrops can be bought,
prove that the total amount of cents is 224.
-/
theorem candy_cost (cost_per_piece : ℕ) (num_gumdrops : ℕ) (h1 : cost_per_piece = 8) (h2 : num_gumdrops = 28) :
  cost_per_piece * num_gumdrops = 224 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l2700_270023


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2700_270063

theorem chess_tournament_participants (n m : ℕ) : 
  9 < n → n < 25 →  -- Total participants between 9 and 25
  (n - 2*m)^2 = n →  -- Derived equation from the condition about scoring half points against grandmasters
  (n = 16 ∧ (m = 6 ∨ m = 10)) := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2700_270063


namespace NUMINAMATH_CALUDE_andys_calculation_l2700_270074

theorem andys_calculation (y : ℝ) : 4 * y + 5 = 57 → (y + 5) * 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_andys_calculation_l2700_270074


namespace NUMINAMATH_CALUDE_student_handshake_problem_l2700_270070

/-- 
Given an m x n array of students where m, n ≥ 3, if each student shakes hands 
with adjacent students (horizontally, vertically, or diagonally) and the total 
number of handshakes is 1020, then the total number of students N is 140.
-/
theorem student_handshake_problem (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) : 
  (8 * m * n - 6 * m - 6 * n + 4) / 2 = 1020 → m * n = 140 := by
  sorry

#check student_handshake_problem

end NUMINAMATH_CALUDE_student_handshake_problem_l2700_270070


namespace NUMINAMATH_CALUDE_a_not_in_A_iff_a_lt_neg_three_l2700_270057

-- Define the set A
def A : Set ℝ := {x : ℝ | x + 3 ≥ 0}

-- State the theorem
theorem a_not_in_A_iff_a_lt_neg_three (a : ℝ) : a ∉ A ↔ a < -3 := by
  sorry

end NUMINAMATH_CALUDE_a_not_in_A_iff_a_lt_neg_three_l2700_270057


namespace NUMINAMATH_CALUDE_missing_number_proof_l2700_270054

theorem missing_number_proof :
  ∃ x : ℝ, 0.72 * 0.43 + x * 0.34 = 0.3504 ∧ abs (x - 0.12) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2700_270054


namespace NUMINAMATH_CALUDE_car_distance_calculation_l2700_270013

/-- The distance covered by the car in kilometers -/
def car_distance_km : ℝ := 2.2

/-- The distance covered by Amar in meters -/
def amar_distance_m : ℝ := 880

/-- The ratio of Amar's speed to the car's speed -/
def speed_ratio : ℚ := 2 / 5

theorem car_distance_calculation :
  car_distance_km = (amar_distance_m / speed_ratio) / 1000 := by
  sorry

#check car_distance_calculation

end NUMINAMATH_CALUDE_car_distance_calculation_l2700_270013


namespace NUMINAMATH_CALUDE_max_victory_margin_l2700_270094

/-- Represents the vote count for a candidate in two time periods -/
structure VoteCount where
  first_period : ℕ
  second_period : ℕ

/-- The election scenario with given conditions -/
def ElectionScenario : Prop :=
  ∃ (petya vasya : VoteCount),
    -- Total votes condition
    petya.first_period + petya.second_period + vasya.first_period + vasya.second_period = 27 ∧
    -- First two hours condition
    petya.first_period = vasya.first_period + 9 ∧
    -- Last hour condition
    vasya.second_period = petya.second_period + 9 ∧
    -- Petya wins condition
    petya.first_period + petya.second_period > vasya.first_period + vasya.second_period

/-- The theorem stating the maximum possible margin of Petya's victory -/
theorem max_victory_margin (h : ElectionScenario) :
  ∃ (petya vasya : VoteCount),
    petya.first_period + petya.second_period - (vasya.first_period + vasya.second_period) ≤ 9 :=
  sorry

end NUMINAMATH_CALUDE_max_victory_margin_l2700_270094


namespace NUMINAMATH_CALUDE_roof_dimensions_difference_l2700_270019

/-- Represents the dimensions of a rectangular roof side -/
structure RoofSide where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular roof side -/
def area (side : RoofSide) : ℝ := side.width * side.length

theorem roof_dimensions_difference (roof : RoofSide) 
  (h1 : roof.length = 4 * roof.width)  -- Length is 3 times longer than width
  (h2 : 2 * area roof = 588)  -- Combined area of two sides is 588
  : roof.length - roof.width = 3 * Real.sqrt (588 / 8) := by
  sorry

end NUMINAMATH_CALUDE_roof_dimensions_difference_l2700_270019


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l2700_270065

theorem unique_two_digit_integer (u : ℕ) : 
  (10 ≤ u ∧ u < 100) ∧ (13 * u) % 100 = 52 ↔ u = 4 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l2700_270065


namespace NUMINAMATH_CALUDE_parabola_directrix_l2700_270087

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop :=
  y = -5/4

/-- Theorem stating that the given directrix equation is correct for the given parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_d : ℝ, directrix_equation y_d ∧ 
  y_d = -5/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2700_270087


namespace NUMINAMATH_CALUDE_square_roots_theorem_l2700_270077

theorem square_roots_theorem (n : ℝ) (h : n > 0) :
  (∃ a : ℝ, (2*a + 1)^2 = n ∧ (a + 5)^2 = n) → 
  (∃ a : ℝ, 2*a + 1 + a + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l2700_270077


namespace NUMINAMATH_CALUDE_other_number_is_nine_l2700_270091

theorem other_number_is_nine (x : ℝ) (h1 : (x + 5) / 2 = 7) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_other_number_is_nine_l2700_270091


namespace NUMINAMATH_CALUDE_root_relation_implies_coefficient_ratio_l2700_270025

/-- Given two quadratic equations with roots related by a factor of 3, prove the ratio of coefficients -/
theorem root_relation_implies_coefficient_ratio
  (m n p : ℝ)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hp : p ≠ 0)
  (h_root_relation : ∀ x, x^2 + p*x + m = 0 → (3*x)^2 + m*(3*x) + n = 0) :
  n / p = -27 := by
  sorry

end NUMINAMATH_CALUDE_root_relation_implies_coefficient_ratio_l2700_270025


namespace NUMINAMATH_CALUDE_total_relaxing_is_66_l2700_270016

/-- Calculates the number of people remaining in a row after some leave --/
def remainingInRow (initial : ℕ) (leaving : ℕ) : ℕ :=
  if initial ≥ leaving then initial - leaving else 0

/-- Represents the beach scenario with 5 rows of people --/
structure BeachScenario where
  row1_initial : ℕ
  row1_leaving : ℕ
  row2_initial : ℕ
  row2_leaving : ℕ
  row3_initial : ℕ
  row3_leaving : ℕ
  row4_initial : ℕ
  row4_leaving : ℕ
  row5_initial : ℕ
  row5_leaving : ℕ

/-- Calculates the total number of people still relaxing on the beach --/
def totalRelaxing (scenario : BeachScenario) : ℕ :=
  remainingInRow scenario.row1_initial scenario.row1_leaving +
  remainingInRow scenario.row2_initial scenario.row2_leaving +
  remainingInRow scenario.row3_initial scenario.row3_leaving +
  remainingInRow scenario.row4_initial scenario.row4_leaving +
  remainingInRow scenario.row5_initial scenario.row5_leaving

/-- The given beach scenario --/
def givenScenario : BeachScenario :=
  { row1_initial := 24, row1_leaving := 7
  , row2_initial := 20, row2_leaving := 7
  , row3_initial := 18, row3_leaving := 2
  , row4_initial := 16, row4_leaving := 11
  , row5_initial := 30, row5_leaving := 15 }

/-- Theorem stating that the total number of people still relaxing is 66 --/
theorem total_relaxing_is_66 : totalRelaxing givenScenario = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_relaxing_is_66_l2700_270016


namespace NUMINAMATH_CALUDE_bob_cycling_wins_l2700_270078

/-- The number of weeks Bob has already won -/
def initial_wins : ℕ := 2

/-- The cost of the puppy in dollars -/
def puppy_cost : ℕ := 1000

/-- The additional number of wins Bob needs to afford the puppy -/
def additional_wins_needed : ℕ := 8

/-- The prize money Bob wins each week -/
def weekly_prize : ℚ := puppy_cost / (initial_wins + additional_wins_needed)

theorem bob_cycling_wins :
  ∀ (weeks : ℕ),
    weekly_prize * (initial_wins + weeks) ≥ puppy_cost →
    weeks ≥ additional_wins_needed :=
by
  sorry

end NUMINAMATH_CALUDE_bob_cycling_wins_l2700_270078


namespace NUMINAMATH_CALUDE_square_roots_of_sqrt_256_is_correct_l2700_270059

-- Define the set of square roots of √256
def square_roots_of_sqrt_256 : Set ℝ :=
  {x : ℝ | x ^ 2 = Real.sqrt 256}

-- Theorem statement
theorem square_roots_of_sqrt_256_is_correct :
  square_roots_of_sqrt_256 = {-4, 4} := by
sorry

end NUMINAMATH_CALUDE_square_roots_of_sqrt_256_is_correct_l2700_270059


namespace NUMINAMATH_CALUDE_transformed_expr_at_one_l2700_270051

-- Define the original expression
def original_expr (x : ℚ) : ℚ := (x + 2) / (x - 3)

-- Define the transformed expression
def transformed_expr (x : ℚ) : ℚ := 
  (original_expr x + 2) / (original_expr x - 3)

-- Theorem statement
theorem transformed_expr_at_one :
  transformed_expr 1 = -1/9 := by sorry

end NUMINAMATH_CALUDE_transformed_expr_at_one_l2700_270051


namespace NUMINAMATH_CALUDE_lines_skew_iff_b_ne_18_l2700_270083

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determines if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- The first line -/
def line1 (b : ℝ) : Line3D :=
  { point := (3, 2, b),
    direction := (2, 3, 4) }

/-- The second line -/
def line2 : Line3D :=
  { point := (4, 1, 0),
    direction := (3, 4, 2) }

/-- Main theorem: The lines are skew if and only if b ≠ 18 -/
theorem lines_skew_iff_b_ne_18 (b : ℝ) :
  are_skew (line1 b) line2 ↔ b ≠ 18 := by sorry

end NUMINAMATH_CALUDE_lines_skew_iff_b_ne_18_l2700_270083


namespace NUMINAMATH_CALUDE_missing_sale_proof_l2700_270098

/-- Calculates the missing sale amount to achieve a desired average -/
def calculate_missing_sale (sales : List ℕ) (required_sale : ℕ) (desired_average : ℕ) : ℕ :=
  let total_sales := desired_average * (sales.length + 2)
  let known_sales := sales.sum + required_sale
  total_sales - known_sales

theorem missing_sale_proof (sales : List ℕ) (required_sale : ℕ) (desired_average : ℕ) 
  (h1 : sales = [6335, 6927, 6855, 7230])
  (h2 : required_sale = 5091)
  (h3 : desired_average = 6500) :
  calculate_missing_sale sales required_sale desired_average = 6562 := by
  sorry

#eval calculate_missing_sale [6335, 6927, 6855, 7230] 5091 6500

end NUMINAMATH_CALUDE_missing_sale_proof_l2700_270098
