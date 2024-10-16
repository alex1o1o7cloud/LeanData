import Mathlib

namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l3880_388086

theorem digit_sum_puzzle (c d : ℕ) : 
  c < 10 → d < 10 → 
  (40 + c) * (10 * d + 5) = 215 →
  (40 + c) * 5 = 20 →
  (40 + c) * d * 10 = 180 →
  c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l3880_388086


namespace NUMINAMATH_CALUDE_line_relationship_l3880_388080

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (are_skew : Line → Line → Prop)
variable (are_parallel : Line → Line → Prop)
variable (are_intersecting : Line → Line → Prop)

-- Theorem statement
theorem line_relationship (a b c : Line)
  (h1 : are_skew a b)
  (h2 : are_parallel a c) :
  are_skew b c ∨ are_intersecting b c :=
sorry

end NUMINAMATH_CALUDE_line_relationship_l3880_388080


namespace NUMINAMATH_CALUDE_all_cars_meet_time_prove_all_cars_meet_time_l3880_388049

/-- Represents a car on a circular track -/
structure Car where
  speed : ℝ
  direction : Bool -- true for clockwise, false for counterclockwise

/-- Represents the race scenario -/
structure RaceScenario where
  track_length : ℝ
  car_a : Car
  car_b : Car
  car_c : Car
  first_ac_meet : ℝ
  first_ab_meet : ℝ

/-- Theorem stating when all three cars meet for the first time -/
theorem all_cars_meet_time (race : RaceScenario) : ℝ :=
  let first_ac_meet := race.first_ac_meet
  let first_ab_meet := race.first_ab_meet
  371

#check all_cars_meet_time

/-- Main theorem proving the time when all three cars meet -/
theorem prove_all_cars_meet_time (race : RaceScenario) 
  (h1 : race.car_a.direction = true)
  (h2 : race.car_b.direction = true)
  (h3 : race.car_c.direction = false)
  (h4 : race.car_a.speed ≠ race.car_b.speed)
  (h5 : race.car_a.speed ≠ race.car_c.speed)
  (h6 : race.car_b.speed ≠ race.car_c.speed)
  (h7 : race.first_ac_meet = 7)
  (h8 : race.first_ab_meet = 53)
  : all_cars_meet_time race = 371 := by
  sorry

#check prove_all_cars_meet_time

end NUMINAMATH_CALUDE_all_cars_meet_time_prove_all_cars_meet_time_l3880_388049


namespace NUMINAMATH_CALUDE_log_range_theorem_l3880_388095

-- Define the set of valid 'a' values
def validA : Set ℝ := {a | a ∈ (Set.Ioo 2 3) ∪ (Set.Ioo 3 5)}

-- Define the conditions for a meaningful logarithmic expression
def isValidLog (a : ℝ) : Prop :=
  a - 2 > 0 ∧ 5 - a > 0 ∧ a - 2 ≠ 1

-- Theorem statement
theorem log_range_theorem :
  ∀ a : ℝ, isValidLog a ↔ a ∈ validA :=
by sorry

end NUMINAMATH_CALUDE_log_range_theorem_l3880_388095


namespace NUMINAMATH_CALUDE_stool_leg_lengths_l3880_388076

/-- Represents the lengths of the cut pieces of a stool leg -/
structure CutPieces where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Checks if all pieces have different lengths -/
def all_different (pieces : CutPieces) : Prop :=
  pieces.a ≠ pieces.b ∧ pieces.a ≠ pieces.c ∧ pieces.a ≠ pieces.d ∧
  pieces.b ≠ pieces.c ∧ pieces.b ≠ pieces.d ∧
  pieces.c ≠ pieces.d

/-- Checks if the cut pieces form a parallelogram -/
def forms_parallelogram (pieces : CutPieces) : Prop :=
  pieces.a + pieces.c = pieces.b + pieces.d

/-- The main theorem about the stool leg lengths -/
theorem stool_leg_lengths :
  ∀ (pieces : CutPieces),
    pieces.a = 8 ∧ pieces.b = 9 ∧ pieces.c = 10 →
    all_different pieces →
    forms_parallelogram pieces →
    pieces.d = 7 ∨ pieces.d = 11 := by
  sorry


end NUMINAMATH_CALUDE_stool_leg_lengths_l3880_388076


namespace NUMINAMATH_CALUDE_constant_terms_are_like_terms_l3880_388055

/-- Two algebraic terms are considered "like terms" if they have the same variables with the same exponents. -/
def like_terms (term1 term2 : String) : Prop := sorry

/-- A constant term is a number without variables. -/
def is_constant_term (term : String) : Prop := sorry

theorem constant_terms_are_like_terms (a b : String) :
  is_constant_term a ∧ is_constant_term b → like_terms a b := by sorry

end NUMINAMATH_CALUDE_constant_terms_are_like_terms_l3880_388055


namespace NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l3880_388011

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 240 →
  (train_speed * crossing_time) - bridge_length = 135 :=
by
  sorry

/-- Proves that a train traveling at 45 km/hr that crosses a 240-meter bridge in 30 seconds has a length of 135 meters. -/
theorem train_length_proof : 
  ∃ (train_speed crossing_time bridge_length : ℝ),
    train_speed = 45 * (1000 / 3600) ∧
    crossing_time = 30 ∧
    bridge_length = 240 ∧
    (train_speed * crossing_time) - bridge_length = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l3880_388011


namespace NUMINAMATH_CALUDE_integer_between_sqrt3_plus_1_and_sqrt11_l3880_388068

theorem integer_between_sqrt3_plus_1_and_sqrt11 :
  ∃! n : ℤ, (Real.sqrt 3 + 1 < n) ∧ (n < Real.sqrt 11) :=
by
  -- We assume the following inequalities as given:
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := sorry
  have h2 : 3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4 := sorry

  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt3_plus_1_and_sqrt11_l3880_388068


namespace NUMINAMATH_CALUDE_third_roll_probability_l3880_388017

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 3
def biased_die_prob : ℚ := 2 / 3

-- Define the probability of rolling sixes or fives twice for each die
def fair_die_two_rolls : ℚ := fair_die_prob ^ 2
def biased_die_two_rolls : ℚ := biased_die_prob ^ 2

-- Define the normalized probabilities of using each die given the first two rolls
def prob_fair_die : ℚ := fair_die_two_rolls / (fair_die_two_rolls + biased_die_two_rolls)
def prob_biased_die : ℚ := biased_die_two_rolls / (fair_die_two_rolls + biased_die_two_rolls)

-- Theorem: The probability of rolling a six or five on the third roll is 3/5
theorem third_roll_probability : 
  prob_fair_die * fair_die_prob + prob_biased_die * biased_die_prob = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_third_roll_probability_l3880_388017


namespace NUMINAMATH_CALUDE_natural_numbers_less_than_two_l3880_388078

theorem natural_numbers_less_than_two : 
  {n : ℕ | n < 2} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_natural_numbers_less_than_two_l3880_388078


namespace NUMINAMATH_CALUDE_derivative_constant_sine_l3880_388029

theorem derivative_constant_sine (y : ℝ → ℝ) (h : y = λ _ => Real.sin (π / 3)) :
  deriv y = λ _ => 0 := by sorry

end NUMINAMATH_CALUDE_derivative_constant_sine_l3880_388029


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3880_388035

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 4) :
  1/x + 1/y ≥ 1 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 4 ∧ 1/a + 1/b = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3880_388035


namespace NUMINAMATH_CALUDE_calculation_proof_l3880_388077

theorem calculation_proof :
  ((-7) * 5 - (-36) / 4 = -26) ∧
  (-1^4 - (1-0.4) * (1/3) * (2-3^2) = 0.4) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l3880_388077


namespace NUMINAMATH_CALUDE_hat_cost_calculation_l3880_388063

/-- The price of a wooden toy -/
def wooden_toy_price : ℕ := 20

/-- The number of wooden toys Kendra bought -/
def wooden_toys_bought : ℕ := 2

/-- The number of hats Kendra bought -/
def hats_bought : ℕ := 3

/-- The amount Kendra paid with -/
def amount_paid : ℕ := 100

/-- The change Kendra received -/
def change_received : ℕ := 30

/-- The cost of a hat -/
def hat_cost : ℕ := 10

theorem hat_cost_calculation :
  hat_cost = (amount_paid - change_received - wooden_toy_price * wooden_toys_bought) / hats_bought :=
by sorry

end NUMINAMATH_CALUDE_hat_cost_calculation_l3880_388063


namespace NUMINAMATH_CALUDE_probability_white_ball_l3880_388041

/-- The probability of drawing a white ball from a box with red and white balls -/
theorem probability_white_ball (red_balls white_balls : ℕ) :
  red_balls = 5 →
  white_balls = 4 →
  (white_balls : ℚ) / (red_balls + white_balls : ℚ) = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_probability_white_ball_l3880_388041


namespace NUMINAMATH_CALUDE_man_work_days_l3880_388048

/-- Proves that if a woman can do a piece of work in 40 days and a man is 25% more efficient,
    then the man can do the same piece of work in 32 days. -/
theorem man_work_days (woman_days : ℕ) (man_efficiency : ℚ) :
  woman_days = 40 →
  man_efficiency = 1.25 →
  (woman_days : ℚ) / man_efficiency = 32 :=
by sorry

end NUMINAMATH_CALUDE_man_work_days_l3880_388048


namespace NUMINAMATH_CALUDE_quantity_count_l3880_388071

theorem quantity_count (total_sum : ℝ) (total_count : ℕ) 
  (subset1_sum : ℝ) (subset1_count : ℕ) 
  (subset2_sum : ℝ) (subset2_count : ℕ) 
  (h1 : total_sum / total_count = 12)
  (h2 : subset1_sum / subset1_count = 4)
  (h3 : subset2_sum / subset2_count = 24)
  (h4 : subset1_count = 3)
  (h5 : subset2_count = 2)
  (h6 : total_sum = subset1_sum + subset2_sum)
  (h7 : total_count = subset1_count + subset2_count) : 
  total_count = 5 := by
sorry


end NUMINAMATH_CALUDE_quantity_count_l3880_388071


namespace NUMINAMATH_CALUDE_south_american_countries_visited_l3880_388081

/-- Proves the number of South American countries visited given the conditions --/
theorem south_american_countries_visited
  (total : ℕ)
  (europe : ℕ)
  (asia : ℕ)
  (h1 : total = 42)
  (h2 : europe = 20)
  (h3 : asia = 6)
  (h4 : 2 * asia = total - europe - asia) :
  total - europe - asia = 8 :=
by sorry

end NUMINAMATH_CALUDE_south_american_countries_visited_l3880_388081


namespace NUMINAMATH_CALUDE_equation_solution_range_l3880_388045

theorem equation_solution_range (m : ℝ) :
  (∃ x : ℝ, 1 - 2 * Real.sin x ^ 2 + 2 * Real.cos x - m = 0) ↔ -3/2 ≤ m ∧ m ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3880_388045


namespace NUMINAMATH_CALUDE_coin_division_theorem_l3880_388067

theorem coin_division_theorem :
  let sum_20 := (20 * 21) / 2
  let sum_20_plus_100 := sum_20 + 100
  (sum_20 % 3 = 0) ∧ (sum_20_plus_100 % 3 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_coin_division_theorem_l3880_388067


namespace NUMINAMATH_CALUDE_total_jog_time_two_weeks_l3880_388003

/-- The number of hours jogged daily -/
def daily_jog_hours : ℝ := 1.5

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- Theorem: Total jogging time in two weeks -/
theorem total_jog_time_two_weeks : 
  daily_jog_hours * (days_in_two_weeks : ℝ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_jog_time_two_weeks_l3880_388003


namespace NUMINAMATH_CALUDE_total_paper_clips_l3880_388037

/-- The number of boxes used to distribute paper clips -/
def num_boxes : ℕ := 9

/-- The number of paper clips in each box -/
def clips_per_box : ℕ := 9

/-- Theorem: The total number of paper clips collected is 81 -/
theorem total_paper_clips : num_boxes * clips_per_box = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_paper_clips_l3880_388037


namespace NUMINAMATH_CALUDE_triangle_construction_existence_and_uniqueness_l3880_388066

-- Define the triangle structure
structure Triangle where
  sideA : ℝ
  sideB : ℝ
  angleC : ℝ
  sideA_pos : 0 < sideA
  sideB_pos : 0 < sideB
  angle_valid : 0 < angleC ∧ angleC < π

-- Theorem statement
theorem triangle_construction_existence_and_uniqueness 
  (a b : ℝ) (γ : ℝ) (ha : 0 < a) (hb : 0 < b) (hγ : 0 < γ ∧ γ < π) :
  ∃! t : Triangle, t.sideA = a ∧ t.sideB = b ∧ t.angleC = γ :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_existence_and_uniqueness_l3880_388066


namespace NUMINAMATH_CALUDE_cos_sum_equality_l3880_388089

theorem cos_sum_equality (x : Real) (h : Real.sin (x + π / 3) = 1 / 3) :
  Real.cos x + Real.cos (π / 3 - x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_equality_l3880_388089


namespace NUMINAMATH_CALUDE_complex_magnitude_l3880_388058

theorem complex_magnitude (a b : ℝ) (i : ℂ) :
  (i * i = -1) →
  ((a + i) * i = b + a * i) →
  Complex.abs (a + b * i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3880_388058


namespace NUMINAMATH_CALUDE_angle_C_is_84_l3880_388094

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the ratio condition
def ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), t.A = 4*k ∧ t.B = 4*k ∧ t.C = 7*k

-- Theorem statement
theorem angle_C_is_84 (t : Triangle) (h : ratio_condition t) : t.C = 84 :=
  sorry

end NUMINAMATH_CALUDE_angle_C_is_84_l3880_388094


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_square_sides_l3880_388054

open Real

theorem circle_radius_tangent_to_square_sides (a : ℝ) :
  a = Real.sqrt (2 + Real.sqrt 2) →
  ∃ (R : ℝ),
    R = Real.sqrt 2 + Real.sqrt (2 - Real.sqrt 2) ∧
    (Real.sin (π / 8) = Real.sqrt (2 - Real.sqrt 2) / 2) ∧
    (∃ (O : ℝ × ℝ) (C : ℝ × ℝ),
      -- O is the center of the circle, C is a vertex of the square
      -- The distance between O and C is related to R and the sine of 22.5°
      Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) = 4 * R / Real.sqrt (2 - Real.sqrt 2) ∧
      -- The angle between the tangents from C is 45°
      Real.arctan (R / (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) - R)) = π / 8) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_square_sides_l3880_388054


namespace NUMINAMATH_CALUDE_train_length_problem_l3880_388061

/-- Given a bridge length, train speed, and time to pass over the bridge, 
    calculate the length of the train. -/
def train_length (bridge_length : ℝ) (train_speed : ℝ) (time_to_pass : ℝ) : ℝ :=
  train_speed * time_to_pass - bridge_length

/-- Theorem stating that under the given conditions, the train length is 400 meters. -/
theorem train_length_problem : 
  let bridge_length : ℝ := 2800
  let train_speed : ℝ := 800
  let time_to_pass : ℝ := 4
  train_length bridge_length train_speed time_to_pass = 400 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l3880_388061


namespace NUMINAMATH_CALUDE_fifteen_divides_Q_largest_divisor_fifteen_largest_divisor_l3880_388004

/-- The product of four consecutive positive odd integers -/
def Q (n : ℕ) : ℕ := (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3)

/-- 15 divides Q for all n -/
theorem fifteen_divides_Q (n : ℕ) : 15 ∣ Q n :=
sorry

/-- For any integer k > 15, there exists an n such that k does not divide Q n -/
theorem largest_divisor (k : ℕ) (h : k > 15) : ∃ n : ℕ, ¬(k ∣ Q n) :=
sorry

/-- 15 is the largest integer that divides Q for all n -/
theorem fifteen_largest_divisor : ∀ k : ℕ, (∀ n : ℕ, k ∣ Q n) → k ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_fifteen_divides_Q_largest_divisor_fifteen_largest_divisor_l3880_388004


namespace NUMINAMATH_CALUDE_chocolate_chips_per_family_member_l3880_388059

/-- Represents the number of chocolate chips per cookie for each type --/
structure ChocolateChipsPerCookie :=
  (chocolate_chip : ℕ)
  (double_chocolate_chip : ℕ)
  (white_chocolate_chip : ℕ)

/-- Represents the number of cookies per batch for each type --/
structure CookiesPerBatch :=
  (chocolate_chip : ℕ)
  (double_chocolate_chip : ℕ)
  (white_chocolate_chip : ℕ)

/-- Represents the number of batches for each type of cookie --/
structure Batches :=
  (chocolate_chip : ℕ)
  (double_chocolate_chip : ℕ)
  (white_chocolate_chip : ℕ)

def total_chocolate_chips (chips_per_cookie : ChocolateChipsPerCookie) 
                          (cookies_per_batch : CookiesPerBatch) 
                          (batches : Batches) : ℕ :=
  chips_per_cookie.chocolate_chip * cookies_per_batch.chocolate_chip * batches.chocolate_chip +
  chips_per_cookie.double_chocolate_chip * cookies_per_batch.double_chocolate_chip * batches.double_chocolate_chip +
  chips_per_cookie.white_chocolate_chip * cookies_per_batch.white_chocolate_chip * batches.white_chocolate_chip

theorem chocolate_chips_per_family_member 
  (chips_per_cookie : ChocolateChipsPerCookie)
  (cookies_per_batch : CookiesPerBatch)
  (batches : Batches)
  (family_members : ℕ)
  (h1 : chips_per_cookie = ⟨2, 4, 3⟩)
  (h2 : cookies_per_batch = ⟨12, 10, 15⟩)
  (h3 : batches = ⟨3, 2, 1⟩)
  (h4 : family_members = 4)
  : (total_chocolate_chips chips_per_cookie cookies_per_batch batches) / family_members = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_chips_per_family_member_l3880_388059


namespace NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_contradictory_l3880_388091

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The set of all possible outcomes when drawing two balls from the pocket -/
def allOutcomes : Finset DrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Black ∧ outcome.second = BallColor.Red) ∨
  (outcome.first = BallColor.Red ∧ outcome.second = BallColor.Black)

/-- The event of drawing exactly two black balls -/
def exactlyTwoBlack (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.Black ∧ outcome.second = BallColor.Black

/-- The theorem stating that "Exactly one black ball" and "Exactly two black balls" are mutually exclusive but not contradictory -/
theorem exactly_one_two_black_mutually_exclusive_not_contradictory :
  (∀ outcome : DrawOutcome, ¬(exactlyOneBlack outcome ∧ exactlyTwoBlack outcome)) ∧
  (∃ outcome : DrawOutcome, exactlyOneBlack outcome) ∧
  (∃ outcome : DrawOutcome, exactlyTwoBlack outcome) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_contradictory_l3880_388091


namespace NUMINAMATH_CALUDE_marks_vote_ratio_l3880_388024

theorem marks_vote_ratio (total_voters_first_area : ℕ) (win_percentage : ℚ) (total_votes : ℕ) : 
  total_voters_first_area = 100000 →
  win_percentage = 70 / 100 →
  total_votes = 210000 →
  (total_votes - (total_voters_first_area * win_percentage).floor) / 
  ((total_voters_first_area * win_percentage).floor) = 2 := by
  sorry

end NUMINAMATH_CALUDE_marks_vote_ratio_l3880_388024


namespace NUMINAMATH_CALUDE_net_population_increase_l3880_388016

/-- Calculates the net population increase given birth, immigration, emigration, death rate, and initial population. -/
theorem net_population_increase
  (births : ℕ)
  (immigrants : ℕ)
  (emigrants : ℕ)
  (death_rate : ℚ)
  (initial_population : ℕ)
  (h_births : births = 90171)
  (h_immigrants : immigrants = 16320)
  (h_emigrants : emigrants = 8212)
  (h_death_rate : death_rate = 8 / 10000)
  (h_initial_population : initial_population = 2876543) :
  (births + immigrants) - (emigrants + Int.floor (death_rate * initial_population)) = 96078 :=
by sorry

end NUMINAMATH_CALUDE_net_population_increase_l3880_388016


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l3880_388036

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1027 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l3880_388036


namespace NUMINAMATH_CALUDE_exterior_angles_sum_360_l3880_388023

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  /-- The number of sides in the polygon. -/
  sides : ℕ
  /-- Assumption that the polygon has at least 3 sides. -/
  sides_ge_three : sides ≥ 3

/-- The sum of interior angles of a polygon. -/
def sum_of_interior_angles (p : Polygon) : ℝ := sorry

/-- The sum of exterior angles of a polygon. -/
def sum_of_exterior_angles (p : Polygon) : ℝ := sorry

/-- Theorem: For any polygon, if the sum of its interior angles is 1440°, 
    then the sum of its exterior angles is 360°. -/
theorem exterior_angles_sum_360 (p : Polygon) :
  sum_of_interior_angles p = 1440 → sum_of_exterior_angles p = 360 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angles_sum_360_l3880_388023


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l3880_388098

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l3880_388098


namespace NUMINAMATH_CALUDE_product_65_35_l3880_388005

theorem product_65_35 : 65 * 35 = 2275 := by
  sorry

end NUMINAMATH_CALUDE_product_65_35_l3880_388005


namespace NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l3880_388047

theorem three_digit_integers_with_remainders : 
  let n : ℕ → Prop := λ x => 
    100 ≤ x ∧ x < 1000 ∧ 
    x % 7 = 3 ∧ 
    x % 8 = 4 ∧ 
    x % 10 = 6
  (∃! (l : List ℕ), l.length = 4 ∧ ∀ x ∈ l, n x) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l3880_388047


namespace NUMINAMATH_CALUDE_total_apples_collected_l3880_388096

def apples_per_day : ℕ := 4
def days : ℕ := 30
def remaining_apples : ℕ := 230

theorem total_apples_collected :
  apples_per_day * days + remaining_apples = 350 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_collected_l3880_388096


namespace NUMINAMATH_CALUDE_charity_duck_race_money_raised_l3880_388040

/-- The amount of money raised in a charity rubber duck race -/
theorem charity_duck_race_money_raised
  (regular_price : ℚ)
  (large_price : ℚ)
  (regular_sold : ℕ)
  (large_sold : ℕ)
  (h1 : regular_price = 3)
  (h2 : large_price = 5)
  (h3 : regular_sold = 221)
  (h4 : large_sold = 185) :
  regular_price * regular_sold + large_price * large_sold = 1588 :=
by sorry

end NUMINAMATH_CALUDE_charity_duck_race_money_raised_l3880_388040


namespace NUMINAMATH_CALUDE_swapped_divisible_by_37_l3880_388079

/-- Represents a nine-digit number split into two parts -/
structure SplitNumber where
  x : ℕ
  y : ℕ
  k : ℕ
  h1 : k > 0
  h2 : k < 10

/-- The original nine-digit number -/
def originalNumber (n : SplitNumber) : ℕ :=
  n.x * 10^(9 - n.k) + n.y

/-- The swapped nine-digit number -/
def swappedNumber (n : SplitNumber) : ℕ :=
  n.y * 10^n.k + n.x

/-- Theorem stating that if the original number is divisible by 37,
    then the swapped number is also divisible by 37 -/
theorem swapped_divisible_by_37 (n : SplitNumber) :
  37 ∣ originalNumber n → 37 ∣ swappedNumber n := by
  sorry


end NUMINAMATH_CALUDE_swapped_divisible_by_37_l3880_388079


namespace NUMINAMATH_CALUDE_price_per_game_l3880_388028

def playstation_cost : ℝ := 500
def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def games_to_sell : ℕ := 20

theorem price_per_game :
  (playstation_cost - (birthday_money + christmas_money)) / games_to_sell = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_price_per_game_l3880_388028


namespace NUMINAMATH_CALUDE_collinear_vectors_perpendicular_vectors_l3880_388083

-- Problem 1
def point_A : ℝ × ℝ := (5, 4)
def point_C : ℝ × ℝ := (12, -2)

def vector_AB (k : ℝ) : ℝ × ℝ := (k - 5, 6)
def vector_BC (k : ℝ) : ℝ × ℝ := (12 - k, -12)

def are_collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem collinear_vectors :
  are_collinear (vector_AB (-2)) (vector_BC (-2)) := by sorry

-- Problem 2
def vector_OA : ℝ × ℝ := (-7, 6)
def vector_OC : ℝ × ℝ := (5, 7)

def vector_AB' (k : ℝ) : ℝ × ℝ := (10, k - 6)
def vector_BC' (k : ℝ) : ℝ × ℝ := (2, 7 - k)

def are_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem perpendicular_vectors :
  (are_perpendicular (vector_AB' 2) (vector_BC' 2)) ∧
  (are_perpendicular (vector_AB' 11) (vector_BC' 11)) := by sorry

end NUMINAMATH_CALUDE_collinear_vectors_perpendicular_vectors_l3880_388083


namespace NUMINAMATH_CALUDE_fourth_root_16_times_fifth_root_32_l3880_388057

theorem fourth_root_16_times_fifth_root_32 : (16 : ℝ) ^ (1/4) * (32 : ℝ) ^ (1/5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_16_times_fifth_root_32_l3880_388057


namespace NUMINAMATH_CALUDE_polygon_side_length_bound_l3880_388019

theorem polygon_side_length_bound (n : ℕ) (h : n > 0) :
  ∃ (side_length : ℝ), side_length ≥ Real.sqrt ((1 - Real.cos (π / n)) / 2) ∧
  (∃ (vertices : Fin (2 * n) → ℝ × ℝ),
    (∀ i : Fin n, ∃ j : Fin (2 * n), vertices j = (Real.cos (2 * π * i / n), Real.sin (2 * π * i / n))) ∧
    (∀ i : Fin (2 * n), ∃ j : Fin (2 * n), 
      j ≠ i ∧ 
      Real.sqrt ((vertices i).1 ^ 2 + (vertices i).2 ^ 2) = 1 ∧
      Real.sqrt ((vertices j).1 ^ 2 + (vertices j).2 ^ 2) = 1 ∧
      side_length = Real.sqrt (((vertices i).1 - (vertices j).1) ^ 2 + ((vertices i).2 - (vertices j).2) ^ 2))) :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_side_length_bound_l3880_388019


namespace NUMINAMATH_CALUDE_complex_equality_theorem_l3880_388026

theorem complex_equality_theorem :
  ∃ (x : ℝ), 
    (Complex.mk (Real.sin x ^ 2) (Real.cos (2 * x)) = Complex.mk (Real.sin x ^ 2) (Real.cos x)) ∧ 
    ((Complex.mk (Real.sin x ^ 2) (Real.cos x) = Complex.I) ∨ 
     (Complex.mk (Real.sin x ^ 2) (Real.cos x) = Complex.mk (3/4) (-1/2))) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_theorem_l3880_388026


namespace NUMINAMATH_CALUDE_average_salary_l3880_388002

/-- The average salary of 5 people with given salaries is 9000 --/
theorem average_salary (a b c d e : ℕ) 
  (ha : a = 8000) (hb : b = 5000) (hc : c = 16000) (hd : d = 7000) (he : e = 9000) : 
  (a + b + c + d + e) / 5 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_l3880_388002


namespace NUMINAMATH_CALUDE_geometric_sequence_26th_term_l3880_388065

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_26th_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_14th : a 14 = 10)
  (h_20th : a 20 = 80) :
  a 26 = 640 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_26th_term_l3880_388065


namespace NUMINAMATH_CALUDE_parallel_lines_solution_l3880_388072

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a b c d e f : ℝ) : Prop :=
  a * e = b * d

/-- The first line equation: ax + 2y + 6 = 0 -/
def line1 (a x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

/-- The second line equation: x + (a-1)y + (a^2-1) = 0 -/
def line2 (a x y : ℝ) : Prop :=
  x + (a - 1) * y + (a^2 - 1) = 0

/-- The theorem stating that given the two parallel lines, a = -1 -/
theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, parallel_lines a 2 1 (a-1) 1 (a^2-1)) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_solution_l3880_388072


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3880_388027

/-- Calculates the simple interest given principal, time, and rate. -/
def simpleInterest (principal : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  principal * time * rate / 100

theorem interest_rate_calculation (loanB_principal loanC_principal totalInterest : ℚ)
  (loanB_time loanC_time : ℚ) :
  loanB_principal = 5000 →
  loanC_principal = 3000 →
  loanB_time = 2 →
  loanC_time = 4 →
  totalInterest = 3300 →
  ∃ rate : ℚ, 
    simpleInterest loanB_principal loanB_time rate +
    simpleInterest loanC_principal loanC_time rate = totalInterest ∧
    rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3880_388027


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3880_388038

theorem linear_equation_solution (m : ℝ) : 
  (3 : ℝ) - m * (1 : ℝ) = 1 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3880_388038


namespace NUMINAMATH_CALUDE_escalator_travel_time_l3880_388012

-- Define the escalator properties and person's walking speed
def escalator_speed : ℝ := 11
def escalator_length : ℝ := 126
def person_speed : ℝ := 3

-- Theorem statement
theorem escalator_travel_time :
  let combined_speed := escalator_speed + person_speed
  let time := escalator_length / combined_speed
  time = 9 := by sorry

end NUMINAMATH_CALUDE_escalator_travel_time_l3880_388012


namespace NUMINAMATH_CALUDE_sphere_radius_equals_seven_l3880_388018

-- Define the constants
def cylinder_height : ℝ := 14
def cylinder_diameter : ℝ := 14

-- Define the theorem
theorem sphere_radius_equals_seven :
  ∃ (r : ℝ), 
    r = 7 ∧ 
    (4 * Real.pi * r^2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_equals_seven_l3880_388018


namespace NUMINAMATH_CALUDE_laptop_price_proof_l3880_388031

theorem laptop_price_proof (sticker_price : ℝ) : 
  (0.9 * sticker_price - 100 = 0.8 * sticker_price - 20) → 
  sticker_price = 800 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l3880_388031


namespace NUMINAMATH_CALUDE_max_value_theorem_l3880_388046

/-- The maximum value of ab/(a+b) + ac/(a+c) + bc/(b+c) given the conditions -/
theorem max_value_theorem (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 3)
  (h_product : a * b * c = 1) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 3 / 2 ∧ 
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ a * b * c = 1 ∧
    a * b / (a + b) + a * c / (a + c) + b * c / (b + c) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3880_388046


namespace NUMINAMATH_CALUDE_mike_pens_l3880_388033

/-- The number of pens Mike gave -/
def M : ℕ := sorry

/-- The initial number of pens -/
def initial_pens : ℕ := 5

/-- The number of pens given away -/
def pens_given_away : ℕ := 19

/-- The final number of pens -/
def final_pens : ℕ := 31

theorem mike_pens : 
  2 * (initial_pens + M) - pens_given_away = final_pens ∧ M = 20 := by sorry

end NUMINAMATH_CALUDE_mike_pens_l3880_388033


namespace NUMINAMATH_CALUDE_segment_length_l3880_388042

theorem segment_length : 
  let endpoints := {x : ℝ | |x - (27 : ℝ)^(1/3)| = 5}
  ∃ a b : ℝ, a ∈ endpoints ∧ b ∈ endpoints ∧ |b - a| = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_segment_length_l3880_388042


namespace NUMINAMATH_CALUDE_fraction_problem_l3880_388008

theorem fraction_problem (F : ℚ) : 
  3 + F * (1/3) * (1/5) * 90 = (1/15) * 90 → F = 1/2 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3880_388008


namespace NUMINAMATH_CALUDE_nested_square_root_value_l3880_388039

/-- Given that x is a real number satisfying x = √(2 + x), prove that x = 2 -/
theorem nested_square_root_value (x : ℝ) (h : x = Real.sqrt (2 + x)) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l3880_388039


namespace NUMINAMATH_CALUDE_min_minutes_for_cheaper_plan_y_l3880_388052

/-- The cost in cents for Plan X given y minutes of usage -/
def costX (y : ℕ) : ℚ := 15 * y

/-- The cost in cents for Plan Y given y minutes of usage -/
def costY (y : ℕ) : ℚ := 2500 + 8 * y

/-- Theorem stating that 358 is the minimum whole number of minutes for Plan Y to be cheaper -/
theorem min_minutes_for_cheaper_plan_y : 
  (∀ y : ℕ, y < 358 → costY y ≥ costX y) ∧ 
  costY 358 < costX 358 := by
  sorry

end NUMINAMATH_CALUDE_min_minutes_for_cheaper_plan_y_l3880_388052


namespace NUMINAMATH_CALUDE_min_y_over_x_on_ellipse_l3880_388030

/-- The minimum value of y/x for points on the ellipse 4(x-2)^2 + y^2 = 4 -/
theorem min_y_over_x_on_ellipse :
  ∃ (min : ℝ), min = -2 * Real.sqrt 3 / 3 ∧
  ∀ (x y : ℝ), 4 * (x - 2)^2 + y^2 = 4 →
  y / x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_y_over_x_on_ellipse_l3880_388030


namespace NUMINAMATH_CALUDE_contrapositive_equality_l3880_388015

theorem contrapositive_equality (a b : ℝ) :
  (¬(a * b = 0) ↔ (a ≠ 0 ∧ b ≠ 0)) ↔
  ((a * b = 0) → (a = 0 ∨ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equality_l3880_388015


namespace NUMINAMATH_CALUDE_max_divisor_with_remainder_l3880_388020

theorem max_divisor_with_remainder (A B : ℕ) : 
  (24 = A * B + 4) → A ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_max_divisor_with_remainder_l3880_388020


namespace NUMINAMATH_CALUDE_calculation_proof_l3880_388001

theorem calculation_proof :
  ((-3/4 - 5/8 + 9/12) * (-24) = 15) ∧
  (-1^6 + |(-2)^3 - 10| - (-3) / (-1)^2023 = 14) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3880_388001


namespace NUMINAMATH_CALUDE_lines_intersection_l3880_388034

/-- Two lines intersect at a unique point (-2/7, 5/7) -/
theorem lines_intersection :
  ∃! (p : ℝ × ℝ), 
    (∃ s : ℝ, p = (2 + 3*s, 3 + 4*s)) ∧ 
    (∃ v : ℝ, p = (-1 + v, 2 - v)) ∧
    p = (-2/7, 5/7) := by
  sorry


end NUMINAMATH_CALUDE_lines_intersection_l3880_388034


namespace NUMINAMATH_CALUDE_wednesday_spending_ratio_is_three_to_eight_l3880_388000

/-- Represents Bob's spending pattern and final amount --/
structure BobsSpending where
  initial : ℚ
  monday_spent : ℚ
  tuesday_spent : ℚ
  final : ℚ

/-- Calculates the ratio of Wednesday's spending to Tuesday's remaining amount --/
def wednesdaySpendingRatio (s : BobsSpending) : ℚ × ℚ :=
  let monday_left := s.initial - s.monday_spent
  let tuesday_left := monday_left - s.tuesday_spent
  let wednesday_spent := tuesday_left - s.final
  (wednesday_spent, tuesday_left)

/-- Theorem stating the ratio of Wednesday's spending to Tuesday's remaining amount --/
theorem wednesday_spending_ratio_is_three_to_eight (s : BobsSpending) 
  (h1 : s.initial = 80)
  (h2 : s.monday_spent = s.initial / 2)
  (h3 : s.tuesday_spent = (s.initial - s.monday_spent) / 5)
  (h4 : s.final = 20) :
  wednesdaySpendingRatio s = (3, 8) := by
  sorry


end NUMINAMATH_CALUDE_wednesday_spending_ratio_is_three_to_eight_l3880_388000


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3880_388074

/-- An arithmetic sequence starting with 5, having a common difference of 3, and ending with 203 -/
def arithmeticSequence : List ℕ :=
  let a₁ : ℕ := 5  -- first term
  let d : ℕ := 3   -- common difference
  let aₙ : ℕ := 203 -- last term
  List.range ((aₙ - a₁) / d + 1) |>.map (fun i => a₁ + i * d)

/-- The number of terms in the arithmetic sequence -/
def sequenceLength : ℕ := arithmeticSequence.length

theorem arithmetic_sequence_length :
  sequenceLength = 67 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3880_388074


namespace NUMINAMATH_CALUDE_brenda_skittles_l3880_388006

theorem brenda_skittles (x : ℕ) : x + 8 = 15 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_brenda_skittles_l3880_388006


namespace NUMINAMATH_CALUDE_apples_taken_per_basket_l3880_388051

theorem apples_taken_per_basket (initial_apples : ℕ) (num_baskets : ℕ) (apples_per_basket : ℕ) :
  initial_apples = 64 →
  num_baskets = 4 →
  apples_per_basket = 13 →
  ∃ (taken_per_basket : ℕ),
    taken_per_basket * num_baskets = initial_apples - (apples_per_basket * num_baskets) ∧
    taken_per_basket = 3 :=
by sorry

end NUMINAMATH_CALUDE_apples_taken_per_basket_l3880_388051


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l3880_388092

def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}

theorem subset_implies_m_values (m : ℝ) : Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l3880_388092


namespace NUMINAMATH_CALUDE_usable_seats_in_section_C_l3880_388060

def x : ℝ := 60 + 3 * 80
def y : ℝ := 3 * x + 20
def z : ℝ := 2 * y - 30.5

theorem usable_seats_in_section_C : z = 1809.5 := by
  sorry

end NUMINAMATH_CALUDE_usable_seats_in_section_C_l3880_388060


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l3880_388032

theorem complex_arithmetic_equation : 
  10 - 1.05 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.93 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l3880_388032


namespace NUMINAMATH_CALUDE_cube_edge_is_60_l3880_388056

-- Define the volume of the rectangular cuboid-shaped cabinet
def cuboid_volume : ℝ := 420000

-- Define the volume difference between the cabinets
def volume_difference : ℝ := 204000

-- Define the volume of the cube-shaped cabinet
def cube_volume : ℝ := cuboid_volume - volume_difference

-- Define the function to calculate the edge length of a cube given its volume
def cube_edge_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Theorem statement
theorem cube_edge_is_60 :
  cube_edge_length cube_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_is_60_l3880_388056


namespace NUMINAMATH_CALUDE_fraction_simplification_l3880_388073

theorem fraction_simplification : (5 - 2) / (2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3880_388073


namespace NUMINAMATH_CALUDE_prob_select_copresidents_value_l3880_388099

/-- Represents a math club with a given number of students -/
structure MathClub where
  students : ℕ
  co_presidents : Fin 2
  vice_president : Fin 1

/-- The set of math clubs in the school district -/
def school_clubs : Finset MathClub := sorry

/-- The probability of selecting both co-presidents when randomly selecting 
    three members from a randomly selected club -/
def prob_select_copresidents (clubs : Finset MathClub) : ℚ := sorry

theorem prob_select_copresidents_value : 
  prob_select_copresidents school_clubs = 43 / 420 := by sorry

end NUMINAMATH_CALUDE_prob_select_copresidents_value_l3880_388099


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3880_388082

theorem complex_modulus_problem (w z : ℂ) :
  w * z = 24 - 10 * I ∧ Complex.abs w = Real.sqrt 29 →
  Complex.abs z = (26 * Real.sqrt 29) / 29 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3880_388082


namespace NUMINAMATH_CALUDE_sandy_average_book_price_l3880_388009

theorem sandy_average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) : 
  books1 = 65 → 
  books2 = 55 → 
  price1 = 1480 → 
  price2 = 920 → 
  (price1 + price2) / (books1 + books2 : ℚ) = 20 := by
sorry

end NUMINAMATH_CALUDE_sandy_average_book_price_l3880_388009


namespace NUMINAMATH_CALUDE_P_in_M_l3880_388014

def P : Set Nat := {0, 1}

def M : Set (Set Nat) := {x | x ⊆ P}

theorem P_in_M : P ∈ M := by sorry

end NUMINAMATH_CALUDE_P_in_M_l3880_388014


namespace NUMINAMATH_CALUDE_garbage_classification_repost_l3880_388087

theorem garbage_classification_repost (n : ℕ) : 
  (1 + n + n^2 = 111) ↔ (n = 10) :=
sorry

end NUMINAMATH_CALUDE_garbage_classification_repost_l3880_388087


namespace NUMINAMATH_CALUDE_triangle_area_is_twelve_l3880_388070

/-- The area of a triangle formed by the x-axis, y-axis, and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangle -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

theorem triangle_area_is_twelve :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ > 0 ∧ y₂ > 0 ∧
    lineEquation x₁ 0 ∧
    lineEquation 0 y₂ ∧
    (1/2 : ℝ) * x₁ * y₂ = triangleArea :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_twelve_l3880_388070


namespace NUMINAMATH_CALUDE_union_of_sets_l3880_388064

open Set

theorem union_of_sets (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5, 6} →
  (U \ A) = {1, 2, 4} →
  (U \ B) = {3, 4, 5} →
  A ∪ B = {1, 2, 3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3880_388064


namespace NUMINAMATH_CALUDE_symmetry_composition_iff_intersection_l3880_388085

-- Define a type for lines in a plane
structure Line where
  -- Add necessary properties for a line

-- Define a type for points in a plane
structure Point where
  -- Add necessary properties for a point

-- Define a symmetry operation
def symmetry (l : Line) : Point → Point := sorry

-- Define composition of symmetries
def compose_symmetries (a b c : Line) : Point → Point :=
  symmetry c ∘ symmetry b ∘ symmetry a

-- Define a predicate for three lines intersecting at a single point
def intersect_at_single_point (a b c : Line) : Prop := sorry

-- The main theorem
theorem symmetry_composition_iff_intersection (a b c : Line) :
  (∃ l : Line, compose_symmetries a b c = symmetry l) ↔ intersect_at_single_point a b c := by
  sorry

end NUMINAMATH_CALUDE_symmetry_composition_iff_intersection_l3880_388085


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3880_388093

theorem triangle_third_side_length (a b c : ℕ) : 
  a = 2 → b = 14 → c % 2 = 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (b - a < c ∧ c - a < b ∧ c - b < a) →
  c = 14 := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3880_388093


namespace NUMINAMATH_CALUDE_smallest_non_negative_solution_l3880_388007

theorem smallest_non_negative_solution (x : ℕ) : 
  (x + 7263 : ℤ) ≡ 3507 [ZMOD 15] ↔ x = 9 ∨ (x > 9 ∧ (x : ℤ) ≡ 9 [ZMOD 15]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_negative_solution_l3880_388007


namespace NUMINAMATH_CALUDE_cylinder_speed_squared_l3880_388097

/-- The acceleration due to gravity in m/s^2 -/
def g : ℝ := 9.8

/-- The height of the incline in meters -/
def h : ℝ := 3.0

/-- The speed of the cylinder at the bottom of the incline in m/s -/
def v : ℝ := sorry

theorem cylinder_speed_squared (m : ℝ) (m_pos : m > 0) :
  v^2 = 2 * g * h := by sorry

end NUMINAMATH_CALUDE_cylinder_speed_squared_l3880_388097


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3880_388062

theorem quadratic_factorization_sum (d e f : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + d) * (x + e)) →
  (∀ x, x^2 - x - 56 = (x + e) * (x - f)) →
  d + e + f = 19 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3880_388062


namespace NUMINAMATH_CALUDE_new_encoding_correct_l3880_388043

-- Define the encoding function
def encode (c : Char) : String :=
  match c with
  | 'A' => "21"
  | 'B' => "122"
  | 'C' => "1"
  | _ => ""

-- Define the decoding function (simplified for this problem)
def decode (s : String) : String :=
  if s = "011011010011" then "ABCBA" else ""

-- Theorem statement
theorem new_encoding_correct : 
  let original := "011011010011"
  let decoded := decode original
  String.join (List.map encode decoded.data) = "211221121" := by
  sorry


end NUMINAMATH_CALUDE_new_encoding_correct_l3880_388043


namespace NUMINAMATH_CALUDE_football_club_penalty_kicks_l3880_388025

/-- Calculates the total number of penalty kicks in a football club's shootout contest. -/
def total_penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  (total_players - goalies) * goalies

/-- Theorem stating that for a football club with 25 players including 4 goalies, 
    where each player takes a shot against each goalie, the total number of penalty kicks is 96. -/
theorem football_club_penalty_kicks :
  total_penalty_kicks 25 4 = 96 := by
  sorry

#eval total_penalty_kicks 25 4

end NUMINAMATH_CALUDE_football_club_penalty_kicks_l3880_388025


namespace NUMINAMATH_CALUDE_line_through_quadrants_l3880_388044

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point (x, y) is in the first quadrant -/
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Predicate to check if a point (x, y) is in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Predicate to check if a point (x, y) is in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Predicate to check if a line passes through a given quadrant -/
def passes_through_quadrant (l : Line) (quad : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, quad x y ∧ l.a * x + l.b * y + l.c = 0

/-- Main theorem: If a line passes through the first, second, and fourth quadrants,
    then ac > 0 and bc < 0 -/
theorem line_through_quadrants (l : Line) :
  passes_through_quadrant l in_first_quadrant ∧
  passes_through_quadrant l in_second_quadrant ∧
  passes_through_quadrant l in_fourth_quadrant →
  l.a * l.c > 0 ∧ l.b * l.c < 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_quadrants_l3880_388044


namespace NUMINAMATH_CALUDE_extremum_point_implies_k_range_l3880_388069

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x) / (x^2) - k * (2/x + Real.log x)

theorem extremum_point_implies_k_range :
  (∀ x : ℝ, x > 0 → (∀ y : ℝ, y > 0 → f x k = f y k → x = y ∨ x = 2)) →
  k ∈ Set.Iic (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_extremum_point_implies_k_range_l3880_388069


namespace NUMINAMATH_CALUDE_sphere_center_sum_l3880_388050

theorem sphere_center_sum (x y z : ℝ) :
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 → x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_center_sum_l3880_388050


namespace NUMINAMATH_CALUDE_second_term_of_arithmetic_sequence_l3880_388021

def arithmetic_sequence (a₁ a₂ a₃ : ℤ) : Prop :=
  a₂ - a₁ = a₃ - a₂

theorem second_term_of_arithmetic_sequence :
  ∀ y : ℤ, arithmetic_sequence (3^2) y (3^4) → y = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_of_arithmetic_sequence_l3880_388021


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3880_388075

theorem tangent_line_slope (f : ℝ → ℝ) (f' : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, f x = x^2 + a * f' 1)
  (h3 : f' 1 = -2) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3880_388075


namespace NUMINAMATH_CALUDE_winnie_keeps_six_l3880_388084

/-- The number of cherry lollipops Winnie has -/
def cherry : ℕ := 60

/-- The number of wintergreen lollipops Winnie has -/
def wintergreen : ℕ := 135

/-- The number of grape lollipops Winnie has -/
def grape : ℕ := 5

/-- The number of shrimp cocktail lollipops Winnie has -/
def shrimp : ℕ := 250

/-- The number of Winnie's friends -/
def friends : ℕ := 12

/-- The total number of lollipops Winnie has -/
def total : ℕ := cherry + wintergreen + grape + shrimp

/-- The number of lollipops Winnie keeps for herself -/
def kept : ℕ := total % friends

theorem winnie_keeps_six : kept = 6 := by
  sorry

end NUMINAMATH_CALUDE_winnie_keeps_six_l3880_388084


namespace NUMINAMATH_CALUDE_abs_neg_five_halves_l3880_388053

theorem abs_neg_five_halves : |(-5 : ℚ) / 2| = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_halves_l3880_388053


namespace NUMINAMATH_CALUDE_marjs_wallet_problem_l3880_388013

/-- Prove that given the conditions in Marj's wallet problem, the value of each of the two bills is $20. -/
theorem marjs_wallet_problem (bill_value : ℚ) : 
  (2 * bill_value + 3 * 5 + 4.5 = 42 + 17.5) → bill_value = 20 := by
  sorry

end NUMINAMATH_CALUDE_marjs_wallet_problem_l3880_388013


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_l3880_388022

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem parallel_vectors_solution (a : ℝ) :
  let m : Vector2D := ⟨a, -2⟩
  let n : Vector2D := ⟨1, 1 - a⟩
  parallel m n → a = 2 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_l3880_388022


namespace NUMINAMATH_CALUDE_number_puzzle_l3880_388010

theorem number_puzzle (x : ℤ) : x - 62 + 45 = 55 → 7 * x = 504 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3880_388010


namespace NUMINAMATH_CALUDE_team_x_games_l3880_388090

/-- Prove that Team X played 24 games given the conditions -/
theorem team_x_games (x : ℕ) 
  (h1 : (3 : ℚ) / 4 * x = x - (1 : ℚ) / 4 * x)  -- Team X wins 3/4 of its games
  (h2 : (2 : ℚ) / 3 * (x + 9) = (x + 9) - (1 : ℚ) / 3 * (x + 9))  -- Team Y wins 2/3 of its games
  (h3 : (2 : ℚ) / 3 * (x + 9) = (3 : ℚ) / 4 * x + 4)  -- Team Y won 4 more games than Team X
  : x = 24 := by
  sorry

end NUMINAMATH_CALUDE_team_x_games_l3880_388090


namespace NUMINAMATH_CALUDE_glass_volume_proof_l3880_388088

theorem glass_volume_proof (V : ℝ) 
  (h1 : 0.4 * V = V - 0.6 * V)  -- pessimist's glass is 60% empty (40% full)
  (h2 : 0.6 * V - 0.4 * V = 46) -- difference between optimist's and pessimist's water volumes
  : V = 230 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_proof_l3880_388088
