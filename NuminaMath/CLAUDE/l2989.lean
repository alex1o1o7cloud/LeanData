import Mathlib

namespace NUMINAMATH_CALUDE_square_circle_puzzle_l2989_298985

theorem square_circle_puzzle (x y : ℚ) 
  (eq1 : 5 * x + 2 * y = 39)
  (eq2 : 3 * x + 3 * y = 27) :
  x = 7 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_puzzle_l2989_298985


namespace NUMINAMATH_CALUDE_intersection_A_B_l2989_298905

def A : Set ℝ := {x | x * (x - 3) < 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2989_298905


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2989_298994

theorem sin_2alpha_value (α : Real) (h : Real.sin α + Real.cos α = 1/3) :
  Real.sin (2 * α) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2989_298994


namespace NUMINAMATH_CALUDE_square_side_factor_l2989_298926

theorem square_side_factor : ∃ f : ℝ, 
  (∀ s : ℝ, s > 0 → s^2 = 20 * (f*s)^2) ∧ f = Real.sqrt 5 / 10 := by
  sorry

end NUMINAMATH_CALUDE_square_side_factor_l2989_298926


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2989_298954

theorem min_value_expression (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * Real.cos (2 * x) ≥ 2017 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * Real.cos (2 * x) = 2017 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2989_298954


namespace NUMINAMATH_CALUDE_compare_exponential_expressions_l2989_298934

theorem compare_exponential_expressions :
  let a := 4^(1/4)
  let b := 5^(1/5)
  let c := 16^(1/16)
  let d := 25^(1/25)
  (a > b ∧ a > c ∧ a > d) ∧
  (b > c ∧ b > d) :=
by sorry

end NUMINAMATH_CALUDE_compare_exponential_expressions_l2989_298934


namespace NUMINAMATH_CALUDE_roots_of_equation_l2989_298912

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^3 - 6*x^2 + 11*x - 6)*(x - 2)
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2989_298912


namespace NUMINAMATH_CALUDE_sum_of_powers_of_three_and_negative_three_l2989_298948

theorem sum_of_powers_of_three_and_negative_three : 
  (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_three_and_negative_three_l2989_298948


namespace NUMINAMATH_CALUDE_negative_square_two_l2989_298916

theorem negative_square_two : -2^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_two_l2989_298916


namespace NUMINAMATH_CALUDE_semicircle_radius_is_24_over_5_l2989_298978

/-- A right triangle with a semicircle inscribed -/
structure RightTriangleWithSemicircle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side QR -/
  qr : ℝ
  /-- Radius of the inscribed semicircle -/
  r : ℝ
  /-- PQ is positive -/
  pq_pos : 0 < pq
  /-- QR is positive -/
  qr_pos : 0 < qr
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : pq^2 = qr^2 + (pq - qr)^2
  /-- The radius satisfies the relation with sides -/
  radius_relation : r = (pq * qr) / (pq + qr + (pq - qr))

/-- The main theorem: For a right triangle with PQ = 15 and QR = 8, 
    the radius of the inscribed semicircle is 24/5 -/
theorem semicircle_radius_is_24_over_5 :
  ∃ (t : RightTriangleWithSemicircle), t.pq = 15 ∧ t.qr = 8 ∧ t.r = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_is_24_over_5_l2989_298978


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l2989_298947

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x

-- State the theorem
theorem f_satisfies_conditions :
  -- Condition 1: The domain is ℝ (implicitly satisfied by the definition)
  -- Condition 2: For any a, b ∈ ℝ where a + b = 0, f(a) + f(b) = 0
  (∀ a b : ℝ, a + b = 0 → f a + f b = 0) ∧
  -- Condition 3: For any x ∈ ℝ, if m < 0, then f(x) > f(x + m)
  (∀ x m : ℝ, m < 0 → f x > f (x + m)) :=
by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l2989_298947


namespace NUMINAMATH_CALUDE_solve_for_q_l2989_298963

theorem solve_for_q (m n q : ℚ) : 
  (7/8 : ℚ) = m/96 ∧ 
  (7/8 : ℚ) = (n + m)/112 ∧ 
  (7/8 : ℚ) = (q - m)/144 → 
  q = 210 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l2989_298963


namespace NUMINAMATH_CALUDE_tax_rate_percentage_l2989_298960

/-- A tax rate of $65 per $100.00 is equivalent to 65% -/
theorem tax_rate_percentage : 
  let tax_amount : ℚ := 65
  let base_amount : ℚ := 100
  (tax_amount / base_amount) * 100 = 65 := by sorry

end NUMINAMATH_CALUDE_tax_rate_percentage_l2989_298960


namespace NUMINAMATH_CALUDE_sphere_chords_theorem_l2989_298902

/-- Represents a sphere with a point inside and three perpendicular chords -/
structure SphereWithChords where
  R : ℝ  -- radius of the sphere
  a : ℝ  -- distance of point A from the center
  h : 0 < R ∧ 0 ≤ a ∧ a < R  -- constraints on R and a

/-- The sum of squares of three mutually perpendicular chords through a point in a sphere -/
def sum_of_squares_chords (s : SphereWithChords) : ℝ := 12 * s.R^2 - 4 * s.a^2

/-- The sum of squares of the segments of three mutually perpendicular chords created by a point in a sphere -/
def sum_of_squares_segments (s : SphereWithChords) : ℝ := 6 * s.R^2 - 2 * s.a^2

/-- Theorem stating the properties of chords in a sphere -/
theorem sphere_chords_theorem (s : SphereWithChords) :
  (sum_of_squares_chords s = 12 * s.R^2 - 4 * s.a^2) ∧
  (sum_of_squares_segments s = 6 * s.R^2 - 2 * s.a^2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_chords_theorem_l2989_298902


namespace NUMINAMATH_CALUDE_triangle_cosine_difference_l2989_298918

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if 4b * sin A = √7 * a and a, b, c are in arithmetic progression
    with positive common difference, then cos A - cos C = √7/2 -/
theorem triangle_cosine_difference (a b c : ℝ) (A B C : ℝ) (h1 : 4 * b * Real.sin A = Real.sqrt 7 * a)
  (h2 : ∃ (d : ℝ), d > 0 ∧ b = a + d ∧ c = b + d) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_difference_l2989_298918


namespace NUMINAMATH_CALUDE_curve_points_difference_l2989_298920

theorem curve_points_difference (a b : ℝ) : 
  (a ≠ b) → 
  ((a : ℝ)^2 + (Real.sqrt π)^4 = 2 * (Real.sqrt π)^2 * a + 1) → 
  ((b : ℝ)^2 + (Real.sqrt π)^4 = 2 * (Real.sqrt π)^2 * b + 1) → 
  |a - b| = 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_points_difference_l2989_298920


namespace NUMINAMATH_CALUDE_prob_at_least_one_2_or_4_is_5_9_l2989_298923

/-- The probability of rolling a 2 or 4 on a single fair 6-sided die -/
def prob_2_or_4 : ℚ := 1/3

/-- The probability of not rolling a 2 or 4 on a single fair 6-sided die -/
def prob_not_2_or_4 : ℚ := 2/3

/-- The probability of at least one die showing 2 or 4 when rolling two fair 6-sided dice -/
def prob_at_least_one_2_or_4 : ℚ := 1 - (prob_not_2_or_4 * prob_not_2_or_4)

theorem prob_at_least_one_2_or_4_is_5_9 : 
  prob_at_least_one_2_or_4 = 5/9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_2_or_4_is_5_9_l2989_298923


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l2989_298975

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l2989_298975


namespace NUMINAMATH_CALUDE_count_triples_satisfying_equation_l2989_298900

theorem count_triples_satisfying_equation : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun t : ℕ × ℕ × ℕ => 
      let (x, y, z) := t
      (x^y)^z = 64 ∧ x > 0 ∧ y > 0 ∧ z > 0)
    (Finset.product (Finset.range 64) (Finset.product (Finset.range 64) (Finset.range 64)))).card
  ∧ n = 9 := by
sorry

end NUMINAMATH_CALUDE_count_triples_satisfying_equation_l2989_298900


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2989_298915

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.64 * MP
  let SP := 0.86 * MP
  let gain := SP - CP
  let gain_percent := (gain / CP) * 100
  gain_percent = 34.375 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2989_298915


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l2989_298909

/-- The intersection point of f(x) = log_a(x) and g(x) = (1-a)x is in the fourth quadrant when a > 1 -/
theorem intersection_in_fourth_quadrant (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ Real.log x / Real.log a = (1 - a) * x := by
  sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l2989_298909


namespace NUMINAMATH_CALUDE_inequality_chain_l2989_298969

theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) < Real.sqrt (a * b) ∧
  Real.sqrt (a * b) < (a + b) / 2 ∧
  (a + b) / 2 < Real.sqrt ((a^2 + b^2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_chain_l2989_298969


namespace NUMINAMATH_CALUDE_negation_of_forall_gt_sin_l2989_298950

theorem negation_of_forall_gt_sin (P : ℝ → Prop) : 
  (¬ ∀ x > 0, 2 * x > Real.sin x) ↔ (∃ x₀ > 0, 2 * x₀ ≤ Real.sin x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_gt_sin_l2989_298950


namespace NUMINAMATH_CALUDE_possible_AC_values_l2989_298919

/-- Three points on a line with given distances between them -/
structure ThreePointsOnLine where
  A : ℝ
  B : ℝ
  C : ℝ
  AB_eq : |A - B| = 3
  BC_eq : |B - C| = 5

/-- The possible values for AC given AB = 3 and BC = 5 -/
theorem possible_AC_values (p : ThreePointsOnLine) : 
  |p.A - p.C| = 2 ∨ |p.A - p.C| = 8 :=
by sorry

end NUMINAMATH_CALUDE_possible_AC_values_l2989_298919


namespace NUMINAMATH_CALUDE_set_equality_l2989_298941

def A : Set ℤ := {-2, -1, 0, 1, 2}

theorem set_equality : {y : ℤ | ∃ x ∈ A, y = |x + 1|} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2989_298941


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l2989_298990

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (multiple_position : ℕ)
  (left_to_right_ratio : ℚ)
  (h1 : total_players = 120)
  (h2 : throwers = 60)
  (h3 : multiple_position = 20)
  (h4 : left_to_right_ratio = 2 / 3)
  (h5 : throwers + multiple_position ≤ total_players) :
  throwers + multiple_position + ((total_players - (throwers + multiple_position)) / (1 + left_to_right_ratio⁻¹)) = 104 :=
by sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l2989_298990


namespace NUMINAMATH_CALUDE_odometer_problem_l2989_298917

theorem odometer_problem (a b c : ℕ) (ha : a ≥ 1) (hsum : a + b + c = 9)
  (hx : ∃ x : ℕ, x > 0 ∧ 60 * x = 100 * c + 10 * a + b - (100 * a + 10 * b + c)) :
  a^2 + b^2 + c^2 = 51 := by
sorry

end NUMINAMATH_CALUDE_odometer_problem_l2989_298917


namespace NUMINAMATH_CALUDE_michelle_sandwiches_l2989_298908

/-- The number of sandwiches Michelle gave to the first co-worker -/
def sandwiches_given : ℕ := sorry

/-- The total number of sandwiches Michelle originally made -/
def total_sandwiches : ℕ := 20

/-- The number of sandwiches left for other co-workers -/
def sandwiches_left : ℕ := 8

/-- The number of sandwiches Michelle kept for herself -/
def sandwiches_kept : ℕ := 2 * sandwiches_given

theorem michelle_sandwiches :
  sandwiches_given + sandwiches_kept + sandwiches_left = total_sandwiches ∧
  sandwiches_given = 4 := by
  sorry

end NUMINAMATH_CALUDE_michelle_sandwiches_l2989_298908


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l2989_298996

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2*p.2, 2*p.1 - p.2)

-- Theorem statement
theorem preimage_of_3_1 : f (1, 1) = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l2989_298996


namespace NUMINAMATH_CALUDE_max_slope_on_circle_l2989_298956

theorem max_slope_on_circle (x y : ℝ) :
  x^2 + y^2 - 2*x - 2 = 0 →
  (∀ a b : ℝ, a^2 + b^2 - 2*a - 2 = 0 → (y + 1) / (x + 1) ≤ (b + 1) / (a + 1)) →
  (y + 1) / (x + 1) = 2 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_slope_on_circle_l2989_298956


namespace NUMINAMATH_CALUDE_ab_equals_six_l2989_298930

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l2989_298930


namespace NUMINAMATH_CALUDE_problem_solution_l2989_298991

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 6) : 
  (x^5 + 3*y^3) / 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2989_298991


namespace NUMINAMATH_CALUDE_badminton_probabilities_l2989_298988

/-- Represents the state of the badminton game -/
inductive GameState
  | Playing (a b c : ℕ) -- number of consecutive losses for each player
  | Winner (player : Fin 3)

/-- Represents a single game outcome -/
inductive GameOutcome
  | Win
  | Lose

/-- The rules of the badminton game -/
def next_state (s : GameState) (outcome : GameOutcome) : GameState :=
  sorry

/-- The probability of a player winning a single game -/
def win_probability : ℚ := 1/2

/-- The probability of A winning four consecutive games -/
def prob_a_wins_four : ℚ := sorry

/-- The probability of needing a fifth game -/
def prob_fifth_game : ℚ := sorry

/-- The probability of C being the ultimate winner -/
def prob_c_wins : ℚ := sorry

theorem badminton_probabilities :
  prob_a_wins_four = 1/16 ∧
  prob_fifth_game = 3/4 ∧
  prob_c_wins = 7/16 :=
sorry

end NUMINAMATH_CALUDE_badminton_probabilities_l2989_298988


namespace NUMINAMATH_CALUDE_klinker_age_relation_l2989_298993

/-- Represents the ages of Mr. Klinker, Julie, and Tim -/
structure Ages where
  klinker : ℕ
  julie : ℕ
  tim : ℕ

/-- The current ages -/
def currentAges : Ages := { klinker := 48, julie := 12, tim := 8 }

/-- The number of years to pass -/
def yearsLater : ℕ := 12

/-- Calculates the ages after a given number of years -/
def agesAfter (initial : Ages) (years : ℕ) : Ages :=
  { klinker := initial.klinker + years
  , julie := initial.julie + years
  , tim := initial.tim + years }

/-- Theorem stating that after 12 years, Mr. Klinker will be twice as old as Julie and thrice as old as Tim -/
theorem klinker_age_relation :
  let futureAges := agesAfter currentAges yearsLater
  futureAges.klinker = 2 * futureAges.julie ∧ futureAges.klinker = 3 * futureAges.tim :=
by sorry

end NUMINAMATH_CALUDE_klinker_age_relation_l2989_298993


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2989_298901

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 100 ∧
  white = 50 ∧
  green = 30 ∧
  red = 7 ∧
  purple = 3 ∧
  prob = 9/10 ∧
  prob = (white + green + (total - white - green - red - purple)) / total →
  total - white - green - red - purple = 10 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2989_298901


namespace NUMINAMATH_CALUDE_complex_power_sum_l2989_298942

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = -Real.sqrt 2) : z^12 + z⁻¹^12 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2989_298942


namespace NUMINAMATH_CALUDE_picture_placement_l2989_298981

/-- Given a wall of width 27 feet and a centered picture of width 5 feet,
    the distance from the end of the wall to the nearest edge of the picture is 11 feet. -/
theorem picture_placement (wall_width picture_width : ℝ) (h1 : wall_width = 27) (h2 : picture_width = 5) :
  (wall_width - picture_width) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l2989_298981


namespace NUMINAMATH_CALUDE_x27x_divisible_by_36_l2989_298933

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def four_digit_number (x : ℕ) : ℕ := x * 1000 + 270 + x

theorem x27x_divisible_by_36 : 
  ∃! x : ℕ, is_single_digit x ∧ (four_digit_number x) % 36 = 0 :=
sorry

end NUMINAMATH_CALUDE_x27x_divisible_by_36_l2989_298933


namespace NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l2989_298904

/-- Given Elizabeth's shopping scenario, prove she can buy exactly 5 pencils. -/
theorem elizabeth_pencil_purchase (
  initial_money : ℚ)
  (pen_cost : ℚ)
  (pencil_cost : ℚ)
  (pens_to_buy : ℕ)
  (h1 : initial_money = 20)
  (h2 : pen_cost = 2)
  (h3 : pencil_cost = 1.6)
  (h4 : pens_to_buy = 6) :
  (initial_money - pens_to_buy * pen_cost) / pencil_cost = 5 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l2989_298904


namespace NUMINAMATH_CALUDE_miguel_book_pages_l2989_298929

/-- The number of pages Miguel read in his book over two weeks --/
def total_pages : ℕ :=
  let first_four_days := 4 * 48
  let next_five_days := 5 * 35
  let subsequent_four_days := 4 * 28
  let last_day := 19
  first_four_days + next_five_days + subsequent_four_days + last_day

/-- Theorem stating that the total number of pages in Miguel's book is 498 --/
theorem miguel_book_pages : total_pages = 498 := by
  sorry

end NUMINAMATH_CALUDE_miguel_book_pages_l2989_298929


namespace NUMINAMATH_CALUDE_speed_conversion_l2989_298911

theorem speed_conversion (speed_ms : ℝ) (speed_kmh : ℝ) : 
  speed_ms = 0.2790697674418605 ∧ speed_kmh = 1.0046511627906978 → 
  speed_ms = speed_kmh / 3.6 :=
by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l2989_298911


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2989_298943

theorem decimal_to_fraction : 
  (3.68 : ℚ) = 92 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2989_298943


namespace NUMINAMATH_CALUDE_service_fee_calculation_l2989_298953

/-- Calculate the service fee for ticket purchase --/
theorem service_fee_calculation (num_tickets : ℕ) (ticket_price total_paid : ℚ) :
  num_tickets = 3 →
  ticket_price = 44 →
  total_paid = 150 →
  total_paid - (num_tickets : ℚ) * ticket_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_service_fee_calculation_l2989_298953


namespace NUMINAMATH_CALUDE_m_value_range_l2989_298982

/-- The equation x^2 + 2√2x + m = 0 has two distinct real roots -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + 2 * Real.sqrt 2 * x + m = 0 ∧ y^2 + 2 * Real.sqrt 2 * y + m = 0

/-- The solution set of the inequality 4x^2 + 4(m-2)x + 1 > 0 is ℝ -/
def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 > 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := m ≤ 1 ∨ (2 ≤ m ∧ m < 3)

theorem m_value_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_value_range_l2989_298982


namespace NUMINAMATH_CALUDE_triangle_area_set_S_is_two_horizontal_lines_l2989_298977

/-- The set of points A(x, y) for which the area of triangle ABC is 2,
    where B(1, 0) and C(-1, 0) are fixed points -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; abs y = 2}

/-- The area of triangle ABC given point A(x, y) and fixed points B(1, 0) and C(-1, 0) -/
def triangleArea (x y : ℝ) : ℝ := abs y

theorem triangle_area_set :
  ∀ (x y : ℝ), (x, y) ∈ S ↔ triangleArea x y = 2 :=
by sorry

theorem S_is_two_horizontal_lines :
  S = {p : ℝ × ℝ | let (x, y) := p; y = 2 ∨ y = -2} :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_set_S_is_two_horizontal_lines_l2989_298977


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l2989_298970

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a|

-- State the theorem
theorem monotonic_increasing_interval (a : ℝ) :
  (∀ x ≥ 3, Monotone (fun x => f a x)) ↔ a = -6 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l2989_298970


namespace NUMINAMATH_CALUDE_frank_reading_time_l2989_298980

theorem frank_reading_time (pages_per_day : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 8) (h2 : total_pages = 576) :
  total_pages / pages_per_day = 72 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_time_l2989_298980


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l2989_298949

theorem nested_subtraction_simplification (y : ℝ) : 1 - (2 - (3 - (4 - (5 - y)))) = 3 - y := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l2989_298949


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l2989_298971

theorem tax_rate_calculation (tax_rate_percent : ℝ) (base_amount : ℝ) :
  tax_rate_percent = 82 ∧ base_amount = 100 →
  tax_rate_percent / 100 * base_amount = 82 := by
sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l2989_298971


namespace NUMINAMATH_CALUDE_quiz_goal_achievement_l2989_298945

theorem quiz_goal_achievement (total_quizzes : ℕ) (goal_percentage : ℚ)
  (completed_quizzes : ℕ) (as_scored : ℕ) (remaining_quizzes : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 85 / 100)
  (h3 : completed_quizzes = 40)
  (h4 : as_scored = 32)
  (h5 : remaining_quizzes = 20)
  (h6 : completed_quizzes + remaining_quizzes = total_quizzes) :
  (total_quizzes * goal_percentage).floor - as_scored = remaining_quizzes - 1 :=
sorry

end NUMINAMATH_CALUDE_quiz_goal_achievement_l2989_298945


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2989_298925

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m ≤ n) → n + (n + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2989_298925


namespace NUMINAMATH_CALUDE_operation_twice_equals_twenty_l2989_298924

theorem operation_twice_equals_twenty (v : ℝ) : 
  (v - v / 3) - ((v - v / 3) / 3) = 20 → v = 45 := by
sorry

end NUMINAMATH_CALUDE_operation_twice_equals_twenty_l2989_298924


namespace NUMINAMATH_CALUDE_probability_divisible_by_9_l2989_298989

def number_set : Set ℕ := {n | 8 ≤ n ∧ n ≤ 28}

def is_divisible_by_9 (a b c : ℕ) : Prop :=
  (a + b + c) % 9 = 0

def favorable_outcomes : ℕ := 150

def total_outcomes : ℕ := 1330

theorem probability_divisible_by_9 :
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 133 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisible_by_9_l2989_298989


namespace NUMINAMATH_CALUDE_decreasing_function_range_l2989_298961

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a - 5) * x - 2
  else x^2 - 2 * (a + 1) * x + 3 * a

-- Define the condition for the function to be decreasing
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

-- Theorem statement
theorem decreasing_function_range (a : ℝ) :
  is_decreasing (f a) ↔ a ∈ Set.Icc 1 4 :=
sorry

end NUMINAMATH_CALUDE_decreasing_function_range_l2989_298961


namespace NUMINAMATH_CALUDE_product_of_numbers_l2989_298973

theorem product_of_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2989_298973


namespace NUMINAMATH_CALUDE_square_of_nine_ones_l2989_298940

theorem square_of_nine_ones : (111111111 : ℕ)^2 = 12345678987654321 := by
  sorry

end NUMINAMATH_CALUDE_square_of_nine_ones_l2989_298940


namespace NUMINAMATH_CALUDE_students_with_dogs_l2989_298997

theorem students_with_dogs (total_students : ℕ) (girls_percentage : ℚ) (boys_percentage : ℚ)
  (girls_with_dogs_percentage : ℚ) (boys_with_dogs_percentage : ℚ)
  (h1 : total_students = 100)
  (h2 : girls_percentage = 1/2)
  (h3 : boys_percentage = 1/2)
  (h4 : girls_with_dogs_percentage = 1/5)
  (h5 : boys_with_dogs_percentage = 1/10) :
  (total_students : ℚ) * girls_percentage * girls_with_dogs_percentage +
  (total_students : ℚ) * boys_percentage * boys_with_dogs_percentage = 15 :=
by sorry

end NUMINAMATH_CALUDE_students_with_dogs_l2989_298997


namespace NUMINAMATH_CALUDE_function_identity_l2989_298906

def NatPos := {n : ℕ // n > 0}

theorem function_identity (f : NatPos → NatPos) 
  (h : ∀ m n : NatPos, (m.val ^ 2 + (f n).val) ∣ (m.val * (f m).val + n.val)) :
  ∀ n : NatPos, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2989_298906


namespace NUMINAMATH_CALUDE_birth_year_age_problem_l2989_298983

theorem birth_year_age_problem :
  ∀ Y : ℕ,
  1900 ≤ Y → Y ≤ 1988 →
  (1988 - Y = (Y % 100 / 10) * (Y % 10)) →
  (Y = 1964 ∧ 1988 - Y = 24) := by
sorry

end NUMINAMATH_CALUDE_birth_year_age_problem_l2989_298983


namespace NUMINAMATH_CALUDE_scooter_travel_time_l2989_298921

/-- The time it takes for a scooter to travel a given distance, given the following conditions:
  * The distance between two points A and B is 50 miles
  * A bicycle travels 1/2 mile per hour slower than the scooter
  * The bicycle takes 45 minutes (3/4 hour) more than the scooter to make the trip
  * x is the scooter's rate of speed in miles per hour
-/
theorem scooter_travel_time (x : ℝ) : 
  (∃ y : ℝ, y > 0 ∧ y = x - 1/2) →  -- Bicycle speed exists and is positive
  50 / (x - 1/2) - 50 / x = 3/4 →   -- Time difference equation
  50 / x = 50 / x :=                -- Conclusion (trivial here, but represents the result)
by sorry

end NUMINAMATH_CALUDE_scooter_travel_time_l2989_298921


namespace NUMINAMATH_CALUDE_john_popcorn_profit_l2989_298998

/-- Calculates the profit from selling popcorn bags -/
def popcorn_profit (cost_price selling_price number_of_bags : ℕ) : ℕ :=
  (selling_price - cost_price) * number_of_bags

/-- Theorem: John's profit from selling 30 bags of popcorn is $120 -/
theorem john_popcorn_profit :
  popcorn_profit 4 8 30 = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_popcorn_profit_l2989_298998


namespace NUMINAMATH_CALUDE_female_listeners_count_l2989_298958

/-- Represents the survey results from radio station KMAT -/
structure SurveyResults where
  total_listeners : Nat
  total_non_listeners : Nat
  male_listeners : Nat
  male_non_listeners : Nat
  female_non_listeners : Nat
  undeclared_listeners : Nat
  undeclared_non_listeners : Nat

/-- Calculates the number of female listeners based on the survey results -/
def female_listeners (results : SurveyResults) : Nat :=
  results.total_listeners - results.male_listeners - results.undeclared_listeners

/-- Theorem stating that the number of female listeners is 65 -/
theorem female_listeners_count (results : SurveyResults)
  (h1 : results.total_listeners = 160)
  (h2 : results.total_non_listeners = 235)
  (h3 : results.male_listeners = 75)
  (h4 : results.male_non_listeners = 85)
  (h5 : results.female_non_listeners = 135)
  (h6 : results.undeclared_listeners = 20)
  (h7 : results.undeclared_non_listeners = 15) :
  female_listeners results = 65 := by
  sorry

#check female_listeners_count

end NUMINAMATH_CALUDE_female_listeners_count_l2989_298958


namespace NUMINAMATH_CALUDE_distance_to_reflection_l2989_298907

/-- The distance between a point (2, -4) and its reflection over the y-axis is 4. -/
theorem distance_to_reflection : Real.sqrt ((2 - (-2))^2 + (-4 - (-4))^2) = 4 := by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_l2989_298907


namespace NUMINAMATH_CALUDE_probability_of_even_sum_l2989_298955

def number_of_balls : ℕ := 12

def is_even_sum (a b : ℕ) : Prop := Even (a + b)

theorem probability_of_even_sum :
  let total_outcomes := number_of_balls * (number_of_balls - 1)
  let favorable_outcomes := (number_of_balls / 2) * ((number_of_balls / 2) - 1) * 2
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_even_sum_l2989_298955


namespace NUMINAMATH_CALUDE_circuit_disconnection_possibilities_l2989_298944

theorem circuit_disconnection_possibilities :
  let n : ℕ := 7  -- number of resistors
  let total_possibilities : ℕ := 2^n - 1  -- total number of ways at least one resistor can be disconnected
  total_possibilities = 63 := by
  sorry

end NUMINAMATH_CALUDE_circuit_disconnection_possibilities_l2989_298944


namespace NUMINAMATH_CALUDE_larger_number_proof_l2989_298984

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2989_298984


namespace NUMINAMATH_CALUDE_gina_purse_value_l2989_298903

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime"

/-- Converts cents to percentage of a dollar -/
def cents_to_percentage (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem gina_purse_value :
  cents_to_percentage (total_value 2 3 2) = 37 / 100 := by
  sorry

end NUMINAMATH_CALUDE_gina_purse_value_l2989_298903


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2989_298952

/-- Given that the solution set of ax² + bx + c > 0 is {x | -4 < x < 7},
    prove that the solution set of cx² - bx + a > 0 is {x | x < -1/7 or x > 1/4} -/
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, (a * x^2 + b * x + c > 0) ↔ (-4 < x ∧ x < 7)) :
  ∀ x : ℝ, (c * x^2 - b * x + a > 0) ↔ (x < -1/7 ∨ x > 1/4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2989_298952


namespace NUMINAMATH_CALUDE_linear_function_k_value_l2989_298987

/-- Given a linear function y = kx + 3 passing through the point (2, 5), prove that k = 1 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3) → 
  (5 : ℝ) = k * 2 + 3 → 
  k = 1 := by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l2989_298987


namespace NUMINAMATH_CALUDE_gdp_equality_l2989_298979

/-- Represents the GDP value in trillion yuan -/
def gdp_trillion : ℝ := 33.5

/-- Represents the GDP value in scientific notation -/
def gdp_scientific : ℝ := 3.35 * (10 ^ 13)

/-- Theorem stating that the GDP value in trillion yuan is equal to its scientific notation -/
theorem gdp_equality : gdp_trillion * (10 ^ 12) = gdp_scientific := by sorry

end NUMINAMATH_CALUDE_gdp_equality_l2989_298979


namespace NUMINAMATH_CALUDE_prime_with_integer_roots_l2989_298966

theorem prime_with_integer_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 300*p = 0 ∧ y^2 + p*y - 300*p = 0) → 
  1 < p ∧ p ≤ 11 := by
sorry

end NUMINAMATH_CALUDE_prime_with_integer_roots_l2989_298966


namespace NUMINAMATH_CALUDE_percent_equality_l2989_298931

theorem percent_equality (x : ℝ) : 
  (75 / 100) * 600 = (50 / 100) * x → x = 900 := by sorry

end NUMINAMATH_CALUDE_percent_equality_l2989_298931


namespace NUMINAMATH_CALUDE_lottery_expected_months_l2989_298932

/-- Represents the lottery system for car permits -/
structure LotterySystem where
  initial_participants : ℕ
  permits_per_month : ℕ
  new_participants_per_month : ℕ

/-- Calculates the expected number of months to win a permit with constant probability -/
def expected_months_constant (system : LotterySystem) : ℝ :=
  10 -- The actual calculation is omitted

/-- Calculates the expected number of months to win a permit with quarterly variable probabilities -/
def expected_months_variable (system : LotterySystem) : ℝ :=
  10 -- The actual calculation is omitted

/-- The main theorem stating that both lottery systems result in an expected 10 months wait -/
theorem lottery_expected_months (system : LotterySystem) 
    (h1 : system.initial_participants = 300000)
    (h2 : system.permits_per_month = 30000)
    (h3 : system.new_participants_per_month = 30000) :
    expected_months_constant system = 10 ∧ expected_months_variable system = 10 := by
  sorry

#check lottery_expected_months

end NUMINAMATH_CALUDE_lottery_expected_months_l2989_298932


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2989_298965

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2989_298965


namespace NUMINAMATH_CALUDE_cube_surface_area_l2989_298976

/-- The surface area of a cube with edge length 2 is 24 -/
theorem cube_surface_area : 
  let edge_length : ℝ := 2
  let surface_area_formula (x : ℝ) := 6 * x * x
  surface_area_formula edge_length = 24 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2989_298976


namespace NUMINAMATH_CALUDE_total_vegetables_l2989_298913

def vegetable_garden_problem (potatoes cucumbers tomatoes peppers carrots : ℕ) : Prop :=
  potatoes = 560 ∧
  cucumbers = potatoes - 132 ∧
  tomatoes = 3 * cucumbers ∧
  peppers = tomatoes / 2 ∧
  carrots = cucumbers + tomatoes

theorem total_vegetables (potatoes cucumbers tomatoes peppers carrots : ℕ) :
  vegetable_garden_problem potatoes cucumbers tomatoes peppers carrots →
  potatoes + cucumbers + tomatoes + peppers + carrots = 4626 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetables_l2989_298913


namespace NUMINAMATH_CALUDE_symmetric_scanning_codes_count_l2989_298939

/-- Represents a color of a square in the grid -/
inductive Color
| Black
| White

/-- Represents a square in the 8x8 grid -/
structure Square where
  row : Fin 8
  col : Fin 8
  color : Color

/-- Represents the 8x8 grid -/
def Grid := Array (Array Square)

/-- Checks if a square has at least one adjacent square of each color -/
def hasAdjacentColors (grid : Grid) (square : Square) : Prop :=
  sorry

/-- Checks if the grid is symmetric under 90 degree rotation -/
def isSymmetricUnder90Rotation (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under 180 degree rotation -/
def isSymmetricUnder180Rotation (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under 270 degree rotation -/
def isSymmetricUnder270Rotation (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under reflection across midpoint lines -/
def isSymmetricUnderMidpointReflection (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under reflection across diagonals -/
def isSymmetricUnderDiagonalReflection (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid satisfies all symmetry conditions -/
def isSymmetric (grid : Grid) : Prop :=
  isSymmetricUnder90Rotation grid ∧
  isSymmetricUnder180Rotation grid ∧
  isSymmetricUnder270Rotation grid ∧
  isSymmetricUnderMidpointReflection grid ∧
  isSymmetricUnderDiagonalReflection grid

/-- Counts the number of symmetric scanning codes -/
def countSymmetricCodes : Nat :=
  sorry

/-- The main theorem stating that the number of symmetric scanning codes is 254 -/
theorem symmetric_scanning_codes_count :
  countSymmetricCodes = 254 :=
sorry

end NUMINAMATH_CALUDE_symmetric_scanning_codes_count_l2989_298939


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l2989_298962

theorem intersection_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 - 4*x = 0 ∧ x^2 + y^2 + 4*y = 0) → y = -x := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l2989_298962


namespace NUMINAMATH_CALUDE_ice_distribution_proof_l2989_298959

/-- Calculates the number of ice cubes per ice chest after melting --/
def ice_cubes_per_chest (initial_cubes : ℕ) (num_chests : ℕ) (melt_rate : ℕ) (hours : ℕ) : ℕ :=
  let remaining_cubes := initial_cubes - melt_rate * hours
  (remaining_cubes / num_chests : ℕ)

/-- Theorem: Given the initial conditions, each ice chest will contain 39 ice cubes --/
theorem ice_distribution_proof :
  ice_cubes_per_chest 294 7 5 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_ice_distribution_proof_l2989_298959


namespace NUMINAMATH_CALUDE_f_value_at_ln_one_third_l2989_298951

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x) / (2^x + 1) + a * x

theorem f_value_at_ln_one_third (a : ℝ) :
  (f a (Real.log 3) = 3) → (f a (Real.log (1/3)) = -2) := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_ln_one_third_l2989_298951


namespace NUMINAMATH_CALUDE_working_hours_growth_equation_l2989_298937

theorem working_hours_growth_equation 
  (initial_hours : ℝ) 
  (final_hours : ℝ) 
  (growth_rate : ℝ) 
  (h1 : initial_hours = 40) 
  (h2 : final_hours = 48.4) :
  initial_hours * (1 + growth_rate)^2 = final_hours := by
sorry

end NUMINAMATH_CALUDE_working_hours_growth_equation_l2989_298937


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2989_298964

theorem triangle_abc_properties (A B C : Real) (a b c : Real) (S : Real) :
  c = Real.sqrt 3 →
  b = 1 →
  C = 2 * π / 3 →  -- 120° in radians
  B = π / 6 ∧      -- 30° in radians
  S = Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2989_298964


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2989_298967

theorem inequality_system_solution (x : ℝ) :
  (4 * x - 6 < 2 * x ∧ (3 * x - 1) / 2 ≥ 2 * x - 1) ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2989_298967


namespace NUMINAMATH_CALUDE_no_polynomial_satisfies_conditions_l2989_298914

/-- A polynomial function over real numbers. -/
def PolynomialFunction := ℝ → ℝ

/-- The degree of a polynomial function. -/
noncomputable def degree (f : PolynomialFunction) : ℕ := sorry

/-- Predicate for a function satisfying the given conditions. -/
def satisfiesConditions (f : PolynomialFunction) : Prop :=
  ∀ x : ℝ, f (x + 1) = (f x)^2 ∧ (f x)^2 = f (f x)

theorem no_polynomial_satisfies_conditions :
  ¬ ∃ f : PolynomialFunction, degree f ≥ 1 ∧ satisfiesConditions f := by sorry

end NUMINAMATH_CALUDE_no_polynomial_satisfies_conditions_l2989_298914


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2989_298946

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 127)
  (h2 : divisor = 14)
  (h3 : quotient = 9)
  (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2989_298946


namespace NUMINAMATH_CALUDE_range_of_a_l2989_298936

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x + a ≥ 0) → a ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2989_298936


namespace NUMINAMATH_CALUDE_unique_solution_l2989_298992

-- Define the possible colors
inductive Color
| Red
| Blue

-- Define a structure for a child's outfit
structure Outfit :=
  (tshirt : Color)
  (shorts : Color)

-- Define the children
structure Children :=
  (Alyna : Outfit)
  (Bohdan : Outfit)
  (Vika : Outfit)
  (Grysha : Outfit)

-- Define the conditions
def satisfies_conditions (c : Children) : Prop :=
  (c.Alyna.tshirt = Color.Red) ∧
  (c.Bohdan.tshirt = Color.Red) ∧
  (c.Alyna.shorts ≠ c.Bohdan.shorts) ∧
  (c.Vika.tshirt ≠ c.Grysha.tshirt) ∧
  (c.Vika.shorts = Color.Blue) ∧
  (c.Grysha.shorts = Color.Blue) ∧
  (c.Alyna.tshirt ≠ c.Vika.tshirt) ∧
  (c.Alyna.shorts ≠ c.Vika.shorts)

-- Define the correct answer
def correct_answer : Children :=
  { Alyna := { tshirt := Color.Red, shorts := Color.Red },
    Bohdan := { tshirt := Color.Red, shorts := Color.Blue },
    Vika := { tshirt := Color.Blue, shorts := Color.Blue },
    Grysha := { tshirt := Color.Red, shorts := Color.Blue } }

-- The theorem to prove
theorem unique_solution :
  ∀ c : Children, satisfies_conditions c → c = correct_answer :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2989_298992


namespace NUMINAMATH_CALUDE_molly_current_age_l2989_298935

/-- Represents the ages of Sandy, Molly, and Danny -/
structure Ages where
  sandy : ℕ
  molly : ℕ
  danny : ℕ

/-- The ratio of ages is 4:3:5 -/
def age_ratio (a : Ages) : Prop :=
  ∃ (x : ℕ), a.sandy = 4 * x ∧ a.molly = 3 * x ∧ a.danny = 5 * x

/-- Sandy's age after 6 years is 30 -/
def sandy_future_age (a : Ages) : Prop :=
  a.sandy + 6 = 30

/-- Theorem stating that under the given conditions, Molly's current age is 18 -/
theorem molly_current_age (a : Ages) :
  age_ratio a → sandy_future_age a → a.molly = 18 := by
  sorry


end NUMINAMATH_CALUDE_molly_current_age_l2989_298935


namespace NUMINAMATH_CALUDE_max_b_no_lattice_points_l2989_298986

theorem max_b_no_lattice_points :
  let max_b : ℚ := 67 / 199
  ∀ m : ℚ, 1/3 < m → m < max_b →
    ∀ x : ℕ, 0 < x → x ≤ 200 →
      ∀ y : ℤ, y ≠ ⌊m * x + 3⌋ ∧
    ∀ b : ℚ, b > max_b →
      ∃ m : ℚ, 1/3 < m ∧ m < b ∧
        ∃ x : ℕ, 0 < x ∧ x ≤ 200 ∧
          ∃ y : ℤ, y = ⌊m * x + 3⌋ := by
  sorry

end NUMINAMATH_CALUDE_max_b_no_lattice_points_l2989_298986


namespace NUMINAMATH_CALUDE_product_nonpositive_implies_factor_nonpositive_l2989_298968

theorem product_nonpositive_implies_factor_nonpositive (a b : ℝ) : 
  a * b ≤ 0 → a ≤ 0 ∨ b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_product_nonpositive_implies_factor_nonpositive_l2989_298968


namespace NUMINAMATH_CALUDE_production_average_l2989_298972

theorem production_average (n : ℕ) : 
  (n * 50 + 110) / (n + 1) = 55 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_production_average_l2989_298972


namespace NUMINAMATH_CALUDE_set_inclusion_l2989_298974

-- Define the sets A, B, and C
def A : Set ℝ := {x | ∃ k : ℤ, x = k / 6 + 1}
def B : Set ℝ := {x | ∃ k : ℤ, x = k / 3 + 1 / 2}
def C : Set ℝ := {x | ∃ k : ℤ, x = 2 * k / 3 + 1 / 2}

-- State the theorem
theorem set_inclusion : C ⊆ B ∧ B ⊆ A := by sorry

end NUMINAMATH_CALUDE_set_inclusion_l2989_298974


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l2989_298928

-- Define the given conditions
def pills_per_supply : ℕ := 90
def pill_fraction : ℚ := 3/4
def days_between_doses : ℕ := 3
def days_per_month : ℕ := 30

-- Define the theorem
theorem medicine_supply_duration :
  (pills_per_supply * days_between_doses / pill_fraction) / days_per_month = 12 := by
  sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l2989_298928


namespace NUMINAMATH_CALUDE_gcd_n_cube_minus_27_and_n_plus_3_l2989_298910

theorem gcd_n_cube_minus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 - 27) (n + 3) = if (n + 3) % 9 = 0 then 9 else 1 := by sorry

end NUMINAMATH_CALUDE_gcd_n_cube_minus_27_and_n_plus_3_l2989_298910


namespace NUMINAMATH_CALUDE_rectangle_length_percentage_l2989_298995

theorem rectangle_length_percentage (area : ℝ) (breadth : ℝ) (length : ℝ) : 
  area = 460 →
  breadth = 20 →
  area = length * breadth →
  (length - breadth) / breadth * 100 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_percentage_l2989_298995


namespace NUMINAMATH_CALUDE_eugene_pencils_left_l2989_298957

/-- The number of pencils Eugene has left after giving some away -/
def pencils_left (initial : Real) (given_away : Real) : Real :=
  initial - given_away

/-- Theorem: Eugene has 199.0 pencils left after giving away 35.0 pencils from his initial 234.0 pencils -/
theorem eugene_pencils_left : pencils_left 234.0 35.0 = 199.0 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_left_l2989_298957


namespace NUMINAMATH_CALUDE_surveyor_distance_theorem_l2989_298999

/-- The distance traveled by the surveyor when he heard the blast -/
def surveyorDistance : ℝ := 122

/-- The time it takes for the fuse to burn (in seconds) -/
def fuseTime : ℝ := 20

/-- The speed of the surveyor (in yards per second) -/
def surveyorSpeed : ℝ := 6

/-- The speed of sound (in feet per second) -/
def soundSpeed : ℝ := 960

/-- Conversion factor from yards to feet -/
def yardsToFeet : ℝ := 3

theorem surveyor_distance_theorem :
  let t := (soundSpeed * fuseTime) / (soundSpeed - surveyorSpeed * yardsToFeet)
  surveyorDistance = surveyorSpeed * t := by sorry

end NUMINAMATH_CALUDE_surveyor_distance_theorem_l2989_298999


namespace NUMINAMATH_CALUDE_hexagon_to_square_area_equality_l2989_298922

/-- Proves that a square with side length s = √(3√3/2) * a has the same area as a regular hexagon with side length a -/
theorem hexagon_to_square_area_equality (a : ℝ) (h : a > 0) :
  let s := Real.sqrt (3 * Real.sqrt 3 / 2) * a
  s^2 = (3 * Real.sqrt 3 / 2) * a^2 := by
  sorry

#check hexagon_to_square_area_equality

end NUMINAMATH_CALUDE_hexagon_to_square_area_equality_l2989_298922


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2989_298927

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = -3) ∧ (4 * x - 5 * y = -21) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2989_298927


namespace NUMINAMATH_CALUDE_function_properties_l2989_298938

def IsAdditive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem function_properties
    (f : ℝ → ℝ)
    (h_additive : IsAdditive f)
    (h_neg : ∀ x : ℝ, x > 0 → f x < 0)
    (h_f_neg_one : f (-1) = 2) :
    (f 0 = 0 ∧ ∀ x : ℝ, f (-x) = -f x) ∧
    (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
    Set.range (fun x => f x) ∩ Set.Icc (-2 : ℝ) 4 = Set.Icc (-8 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2989_298938
