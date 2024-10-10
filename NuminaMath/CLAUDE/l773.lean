import Mathlib

namespace box_triples_count_l773_77359

/-- The number of ordered triples (a, b, c) satisfying the box conditions -/
def box_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    a ≤ b ∧ b ≤ c ∧ 2 * a * b * c = 2 * a * b + 2 * b * c + 2 * a * c)
    (Finset.product (Finset.range 100) (Finset.product (Finset.range 100) (Finset.range 100)))).card

/-- The main theorem stating that there are exactly 10 ordered triples satisfying the conditions -/
theorem box_triples_count : box_triples = 10 := by
  sorry

end box_triples_count_l773_77359


namespace tangent_line_t_range_l773_77319

/-- A line tangent to a circle and intersecting a parabola at two points -/
structure TangentLineIntersectingParabola where
  k : ℝ
  t : ℝ
  tangent_condition : k^2 = t^2 + 2*t
  distinct_intersections : 16*(t^2 + 2*t) + 16*t > 0

/-- The range of t values for a tangent line intersecting a parabola at two points -/
theorem tangent_line_t_range (l : TangentLineIntersectingParabola) :
  l.t > 0 ∨ l.t < -3 := by
  sorry

end tangent_line_t_range_l773_77319


namespace set_intersection_and_union_l773_77345

-- Define the sets A and B as functions of a
def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {-4, a+3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

-- State the theorem
theorem set_intersection_and_union :
  ∃ (a : ℝ), (A a ∩ B a = {2, 5}) ∧ (a = 2) ∧ (A a ∪ B a = {-4, 2, 4, 5, 25}) := by
  sorry

end set_intersection_and_union_l773_77345


namespace probability_calculation_l773_77320

structure ClassStats where
  total_students : ℕ
  female_percentage : ℚ
  brunette_percentage : ℚ
  short_brunette_percentage : ℚ
  club_participation_percentage : ℚ
  short_club_percentage : ℚ

def probability_short_brunette_club (stats : ClassStats) : ℚ :=
  stats.female_percentage *
  stats.brunette_percentage *
  stats.club_participation_percentage *
  stats.short_club_percentage

theorem probability_calculation (stats : ClassStats) 
  (h1 : stats.total_students = 200)
  (h2 : stats.female_percentage = 3/5)
  (h3 : stats.brunette_percentage = 1/2)
  (h4 : stats.short_brunette_percentage = 1/2)
  (h5 : stats.club_participation_percentage = 2/5)
  (h6 : stats.short_club_percentage = 3/4) :
  probability_short_brunette_club stats = 9/100 := by
  sorry

end probability_calculation_l773_77320


namespace quadratic_b_value_l773_77334

/-- A quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x-coordinate on a quadratic function -/
def QuadraticFunction.y (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_b_value (f : QuadraticFunction) (y₁ y₂ : ℝ) 
  (h₁ : f.y 2 = y₁)
  (h₂ : f.y (-2) = y₂)
  (h₃ : y₁ - y₂ = -12) :
  f.b = -3 := by
  sorry

end quadratic_b_value_l773_77334


namespace rectangle_circle_area_ratio_l773_77396

theorem rectangle_circle_area_ratio :
  ∀ (w l r : ℝ),
  w > 0 → l > 0 → r > 0 →
  l = 2 * w →
  2 * l + 2 * w = 2 * π * r →
  (l * w) / (π * r^2) = 2 * π / 9 := by
sorry

end rectangle_circle_area_ratio_l773_77396


namespace spinsters_to_cats_ratio_l773_77399

/-- Given the number of spinsters and cats, prove their ratio is 2:7 -/
theorem spinsters_to_cats_ratio :
  ∀ (spinsters cats : ℕ),
    spinsters = 14 →
    cats = spinsters + 35 →
    (spinsters : ℚ) / (cats : ℚ) = 2 / 7 := by
  sorry

end spinsters_to_cats_ratio_l773_77399


namespace binary_110011_equals_51_l773_77303

def binary_to_decimal (b₅ b₄ b₃ b₂ b₁ b₀ : ℕ) : ℕ :=
  b₀ + 2 * b₁ + 2^2 * b₂ + 2^3 * b₃ + 2^4 * b₄ + 2^5 * b₅

theorem binary_110011_equals_51 : binary_to_decimal 1 1 0 0 1 1 = 51 := by
  sorry

end binary_110011_equals_51_l773_77303


namespace randolph_sydney_age_difference_l773_77341

/-- The age difference between Randolph and Sydney -/
def ageDifference (randolphAge sydneyAge : ℕ) : ℕ := randolphAge - sydneyAge

/-- Theorem stating the age difference between Randolph and Sydney -/
theorem randolph_sydney_age_difference :
  ∀ (sherryAge : ℕ),
    sherryAge = 25 →
    ∀ (sydneyAge : ℕ),
      sydneyAge = 2 * sherryAge →
      ∀ (randolphAge : ℕ),
        randolphAge = 55 →
        ageDifference randolphAge sydneyAge = 5 := by
  sorry

end randolph_sydney_age_difference_l773_77341


namespace no_balloons_remain_intact_l773_77382

/-- Represents the state of balloons in a hot air balloon --/
structure BalloonState where
  total : ℕ
  intact : ℕ
  doubleDurable : ℕ

/-- Calculates the number of intact balloons after the first 30 minutes --/
def afterFirstHalfHour (initial : ℕ) : ℕ :=
  initial - (initial / 5)

/-- Calculates the number of intact balloons after the next hour --/
def afterNextHour (intact : ℕ) : ℕ :=
  intact - (intact * 3 / 10)

/-- Calculates the number of double durable balloons --/
def doubleDurableBalloons (intact : ℕ) : ℕ :=
  intact / 10

/-- Calculates the final number of intact balloons --/
def finalIntactBalloons (state : BalloonState) : ℕ :=
  let nonDurableBlownUp := state.total - state.intact
  let toBlowUp := min (2 * (nonDurableBlownUp - state.doubleDurable)) state.intact
  state.intact - toBlowUp

/-- Main theorem: After all events, no balloons remain intact --/
theorem no_balloons_remain_intact (initialBalloons : ℕ) 
    (h1 : initialBalloons = 200) : 
    finalIntactBalloons 
      { total := initialBalloons,
        intact := afterNextHour (afterFirstHalfHour initialBalloons),
        doubleDurable := doubleDurableBalloons (afterNextHour (afterFirstHalfHour initialBalloons)) } = 0 := by
  sorry

#eval finalIntactBalloons 
  { total := 200,
    intact := afterNextHour (afterFirstHalfHour 200),
    doubleDurable := doubleDurableBalloons (afterNextHour (afterFirstHalfHour 200)) }

end no_balloons_remain_intact_l773_77382


namespace divisible_sequence_eventually_periodic_l773_77330

/-- A sequence of positive integers satisfying the given divisibility property -/
def DivisibleSequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, (a (n + 2*m)) ∣ (a n + a (n + m))

/-- The property of eventual periodicity for a sequence -/
def EventuallyPeriodic (a : ℕ → ℕ) : Prop :=
  ∃ N d : ℕ, d > 0 ∧ ∀ n : ℕ, n > N → a n = a (n + d)

/-- The main theorem: A divisible sequence is eventually periodic -/
theorem divisible_sequence_eventually_periodic (a : ℕ → ℕ) 
  (h : DivisibleSequence a) : EventuallyPeriodic a := by
  sorry

end divisible_sequence_eventually_periodic_l773_77330


namespace sum_of_numbers_l773_77364

theorem sum_of_numbers (x y : ℤ) : y = 3 * x + 11 → x = 11 → x + y = 55 := by
  sorry

end sum_of_numbers_l773_77364


namespace cubic_polynomial_c_value_l773_77331

/-- A cubic polynomial function with integer coefficients -/
def f (a b c : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem stating that under given conditions, c must equal 16 -/
theorem cubic_polynomial_c_value (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  f a b c a = a^3 → f a b c b = b^3 → c = 16 := by
  sorry


end cubic_polynomial_c_value_l773_77331


namespace pythagorean_triple_odd_l773_77344

theorem pythagorean_triple_odd (a : ℕ) (h1 : a ≥ 3) (h2 : Odd a) :
  a^2 + ((a^2 - 1) / 2)^2 = ((a^2 + 1) / 2)^2 := by
  sorry

#check pythagorean_triple_odd

end pythagorean_triple_odd_l773_77344


namespace lcm_problem_l773_77385

theorem lcm_problem (n : ℕ) (h1 : n > 0) (h2 : Nat.lcm 30 n = 90) (h3 : Nat.lcm n 45 = 180) : n = 36 := by
  sorry

end lcm_problem_l773_77385


namespace jamies_coins_value_l773_77329

/-- Proves that given 30 coins of nickels and dimes, if swapping their values
    results in a 90-cent increase, then the total value is $1.80. -/
theorem jamies_coins_value :
  ∀ (n d : ℕ),
  n + d = 30 →
  (10 * n + 5 * d) - (5 * n + 10 * d) = 90 →
  5 * n + 10 * d = 180 := by
sorry

end jamies_coins_value_l773_77329


namespace min_questionnaires_correct_l773_77381

/-- The minimum number of questionnaires needed to achieve the desired responses -/
def min_questionnaires : ℕ := 513

/-- The number of desired responses -/
def desired_responses : ℕ := 750

/-- The initial response rate -/
def initial_rate : ℚ := 60 / 100

/-- The decline rate for follow-ups -/
def decline_rate : ℚ := 20 / 100

/-- Calculate the total responses given the number of questionnaires sent -/
def total_responses (n : ℕ) : ℚ :=
  n * initial_rate * (1 + (1 - decline_rate) + (1 - decline_rate)^2)

/-- Theorem stating that min_questionnaires is the minimum number needed -/
theorem min_questionnaires_correct :
  (total_responses min_questionnaires ≥ desired_responses) ∧
  (∀ m : ℕ, m < min_questionnaires → total_responses m < desired_responses) :=
by sorry


end min_questionnaires_correct_l773_77381


namespace relay_schemes_count_l773_77338

/-- The number of segments in the Olympic torch relay route -/
def num_segments : ℕ := 6

/-- The number of torchbearers -/
def num_torchbearers : ℕ := 6

/-- The set of possible first runners -/
inductive FirstRunner
| A
| B
| C

/-- The set of possible last runners -/
inductive LastRunner
| A
| B

/-- A function to calculate the number of relay schemes -/
def count_relay_schemes : ℕ := sorry

/-- Theorem stating that the number of relay schemes is 96 -/
theorem relay_schemes_count :
  count_relay_schemes = 96 := by sorry

end relay_schemes_count_l773_77338


namespace weight_difference_theorem_l773_77323

def weight_difference (robbie_weight patty_multiplier jim_multiplier mary_multiplier patty_loss jim_loss mary_gain : ℝ) : ℝ :=
  let patty_weight := patty_multiplier * robbie_weight - patty_loss
  let jim_weight := jim_multiplier * robbie_weight - jim_loss
  let mary_weight := mary_multiplier * robbie_weight + mary_gain
  patty_weight + jim_weight + mary_weight - robbie_weight

theorem weight_difference_theorem :
  weight_difference 100 4.5 3 2 235 180 45 = 480 := by
  sorry

end weight_difference_theorem_l773_77323


namespace total_letters_received_l773_77356

theorem total_letters_received (brother_letters : ℕ) 
  (h1 : brother_letters = 40) 
  (h2 : ∃ greta_letters : ℕ, greta_letters = brother_letters + 10) 
  (h3 : ∃ mother_letters : ℕ, mother_letters = 2 * (brother_letters + (brother_letters + 10))) :
  ∃ total_letters : ℕ, total_letters = brother_letters + (brother_letters + 10) + 2 * (brother_letters + (brother_letters + 10)) ∧ total_letters = 270 := by
sorry


end total_letters_received_l773_77356


namespace negative_2023_times_99_l773_77390

theorem negative_2023_times_99 (p : ℤ) (h : p = (-2023) * 100) : (-2023) * 99 = p + 2023 := by
  sorry

end negative_2023_times_99_l773_77390


namespace smallest_binary_multiple_of_15_l773_77363

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_15 :
  ∀ T : ℕ, 
    T > 0 → 
    is_binary_number T → 
    T % 15 = 0 → 
    ∀ X : ℕ, 
      X = T / 15 → 
      X ≥ 74 :=
sorry

end smallest_binary_multiple_of_15_l773_77363


namespace coopers_savings_l773_77394

/-- Calculates the total savings given a daily savings amount and number of days -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Theorem: Cooper's savings after one year -/
theorem coopers_savings :
  totalSavings 34 365 = 12410 := by
  sorry

end coopers_savings_l773_77394


namespace line_symmetry_l773_77337

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = x

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := 3 * x - 2 * y - 1 = 0

-- Theorem stating the symmetry relationship
theorem line_symmetry :
  ∀ (x y : ℝ), original_line x y ↔ symmetric_line y x :=
by sorry

end line_symmetry_l773_77337


namespace sqrt_real_range_l773_77350

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 3 + x) ↔ x ≥ -3 := by sorry

end sqrt_real_range_l773_77350


namespace george_sock_order_l773_77376

/-- The ratio of black to blue socks in George's original order -/
def sock_ratio : ℚ := 2 / 11

theorem george_sock_order :
  ∀ (black_price blue_price : ℝ) (blue_count : ℝ),
    black_price = 2 * blue_price →
    3 * black_price + blue_count * blue_price = 
      (blue_count * black_price + 3 * blue_price) * (1 - 0.6) →
    sock_ratio = 3 / blue_count :=
by
  sorry

end george_sock_order_l773_77376


namespace students_not_enrolled_in_french_or_german_l773_77371

theorem students_not_enrolled_in_french_or_german 
  (total_students : ℕ) 
  (french_students : ℕ) 
  (german_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 78)
  (h2 : french_students = 41)
  (h3 : german_students = 22)
  (h4 : both_students = 9) :
  total_students - (french_students + german_students - both_students) = 24 :=
by sorry


end students_not_enrolled_in_french_or_german_l773_77371


namespace parabola_coefficient_l773_77367

/-- A quadratic function with vertex (h, k) has the form f(x) = a(x-h)^2 + k -/
def quadratic_vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

theorem parabola_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = quadratic_vertex_form a 2 5 x) →  -- Condition 2: vertex at (2, 5)
  f 3 = 7 →  -- Condition 3: point (3, 7) lies on the graph
  a = 2 := by  -- Question: Find the value of a
sorry

end parabola_coefficient_l773_77367


namespace quadratic_minimum_l773_77332

theorem quadratic_minimum (a b c d : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≥ d) ∧ 
  (∃ x, a * x^2 + b * x + c = d) →
  c = d + b^2 / (4 * a) := by
sorry

end quadratic_minimum_l773_77332


namespace arithmetic_sequence_line_passes_through_point_l773_77328

/-- Given that A, B, and C form an arithmetic sequence,
    prove that the line Ax + By + C = 0 passes through the point (1, -2) -/
theorem arithmetic_sequence_line_passes_through_point
  (A B C : ℝ) (h : 2 * B = A + C) :
  A * 1 + B * (-2) + C = 0 := by
  sorry

end arithmetic_sequence_line_passes_through_point_l773_77328


namespace locus_of_N_l773_77398

/-- The circle on which point M moves -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y = 0

/-- Point N lies on the ray OM -/
def OnRay (x y : ℝ) : Prop := ∃ (t : ℝ), t > 0 ∧ x = t * (x/((x^2 + y^2)^(1/2))) ∧ y = t * (y/((x^2 + y^2)^(1/2)))

/-- The product of distances |OM| and |ON| is 150 -/
def DistanceProduct (x y : ℝ) : Prop := (x^2 + y^2)^(1/2) * ((x^2 + y^2)^(1/2) / (x^2 + y^2)) = 150

theorem locus_of_N (x y : ℝ) :
  (∃ (mx my : ℝ), Circle mx my ∧ OnRay x y ∧ DistanceProduct x y) →
  3*x + 4*y = 75 := by sorry

end locus_of_N_l773_77398


namespace order_of_numbers_l773_77340

def Ψ : ℤ := (1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + (-2012)) / 2

def Ω : ℤ := 1 - 2 + 3 - 4 + 2014

def Θ : ℤ := 1 - 3 + 5 - 7 + 2015

theorem order_of_numbers : Θ < Ω ∧ Ω < Ψ :=
  sorry

end order_of_numbers_l773_77340


namespace y_order_l773_77321

/-- The quadratic function f(x) = -2x² + 4 --/
def f (x : ℝ) : ℝ := -2 * x^2 + 4

/-- Point A on the graph of f --/
def A : ℝ × ℝ := (1, f 1)

/-- Point B on the graph of f --/
def B : ℝ × ℝ := (2, f 2)

/-- Point C on the graph of f --/
def C : ℝ × ℝ := (-3, f (-3))

theorem y_order : A.2 > B.2 ∧ B.2 > C.2 := by sorry

end y_order_l773_77321


namespace hulk_jump_theorem_l773_77365

def jump_distance (n : ℕ) : ℝ :=
  2 * (3 ^ (n - 1))

theorem hulk_jump_theorem :
  (∀ k < 8, jump_distance k ≤ 2000) ∧ jump_distance 8 > 2000 := by
  sorry

end hulk_jump_theorem_l773_77365


namespace dodecagon_triangle_count_l773_77360

/-- A regular dodecagon -/
structure RegularDodecagon where
  vertices : Finset ℕ
  regular : vertices.card = 12

/-- Count of triangles with specific properties in a regular dodecagon -/
def triangle_count (d : RegularDodecagon) : ℕ × ℕ :=
  let equilateral := 4  -- Number of equilateral triangles
  let scalene := 168    -- Number of scalene triangles
  (equilateral, scalene)

/-- Theorem stating the correct count of equilateral and scalene triangles in a regular dodecagon -/
theorem dodecagon_triangle_count (d : RegularDodecagon) :
  triangle_count d = (4, 168) := by
  sorry

end dodecagon_triangle_count_l773_77360


namespace quadratic_solution_l773_77317

/-- A quadratic function passing through specific points -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the solutions of the quadratic equation -/
theorem quadratic_solution (a b c : ℝ) :
  (f a b c (-1) = 8) →
  (f a b c 0 = 3) →
  (f a b c 1 = 0) →
  (f a b c 2 = -1) →
  (f a b c 3 = 0) →
  (∀ x : ℝ, f a b c x = 0 ↔ x = 1 ∨ x = 3) :=
sorry

end quadratic_solution_l773_77317


namespace parallel_vectors_trig_ratio_l773_77389

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (2, sin θ) and b = (1, cos θ), then sin²θ / (1 + cos²θ) = 2/3 -/
theorem parallel_vectors_trig_ratio 
  (θ : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (2, Real.sin θ)) 
  (hb : b = (1, Real.cos θ)) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  (Real.sin θ)^2 / (1 + (Real.cos θ)^2) = 2/3 := by
  sorry

end parallel_vectors_trig_ratio_l773_77389


namespace power_two_plus_two_gt_square_l773_77308

theorem power_two_plus_two_gt_square (n : ℕ) (h : n > 0) : 2^n + 2 > n^2 := by
  sorry

end power_two_plus_two_gt_square_l773_77308


namespace pool_filling_time_l773_77357

def tap1_time : ℝ := 3
def tap2_time : ℝ := 6
def tap3_time : ℝ := 12

theorem pool_filling_time :
  let combined_rate := 1 / tap1_time + 1 / tap2_time + 1 / tap3_time
  (1 / combined_rate) = 12 / 7 :=
by sorry

end pool_filling_time_l773_77357


namespace intersection_distance_l773_77386

theorem intersection_distance (a b : ℤ) (k : ℝ) : 
  k = a + Real.sqrt b →
  (k + 4) / k = Real.sqrt 5 →
  a + b = 6 := by sorry

end intersection_distance_l773_77386


namespace root_squared_plus_double_eq_three_l773_77307

theorem root_squared_plus_double_eq_three (m : ℝ) : 
  m^2 + 2*m - 3 = 0 → m^2 + 2*m = 3 := by
  sorry

end root_squared_plus_double_eq_three_l773_77307


namespace warehouse_temp_restoration_time_l773_77309

def initial_temp : ℝ := 43
def increase_rate : ℝ := 8
def outage_duration : ℝ := 3
def decrease_rate : ℝ := 4

theorem warehouse_temp_restoration_time :
  let total_increase : ℝ := increase_rate * outage_duration
  let restoration_time : ℝ := total_increase / decrease_rate
  restoration_time = 6 := by sorry

end warehouse_temp_restoration_time_l773_77309


namespace population_growth_l773_77306

theorem population_growth (p : ℕ) : 
  p > 0 →                           -- p is positive
  (p^2 + 121 = q^2 + 16) →          -- 2005 population condition
  (p^2 + 346 = r^2) →               -- 2015 population condition
  ∃ (growth : ℝ), 
    growth = ((p^2 + 346 - p^2) / p^2) * 100 ∧ 
    abs (growth - 111) < abs (growth - 100) ∧ 
    abs (growth - 111) < abs (growth - 105) ∧ 
    abs (growth - 111) < abs (growth - 110) ∧ 
    abs (growth - 111) < abs (growth - 115) :=
by sorry

end population_growth_l773_77306


namespace workshop_duration_is_450_l773_77397

/-- Calculates the duration of a workshop excluding breaks -/
def workshop_duration (total_hours : ℕ) (total_minutes : ℕ) (break_minutes : ℕ) : ℕ :=
  total_hours * 60 + total_minutes - break_minutes

/-- Theorem: The workshop duration excluding breaks is 450 minutes -/
theorem workshop_duration_is_450 :
  workshop_duration 8 20 50 = 450 := by
  sorry

end workshop_duration_is_450_l773_77397


namespace sum_of_specific_numbers_l773_77347

theorem sum_of_specific_numbers : 1357 + 3571 + 5713 + 7135 = 17776 := by
  sorry

end sum_of_specific_numbers_l773_77347


namespace vector_properties_l773_77361

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-2, 4)

theorem vector_properties :
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  let proj_b_on_a := (Real.sqrt (b.1^2 + b.2^2) * cos_theta)
  cos_theta = (4 * Real.sqrt 65) / 65 ∧
  proj_b_on_a = (8 * Real.sqrt 13) / 13 := by
  sorry

end vector_properties_l773_77361


namespace greatest_x_value_l773_77311

theorem greatest_x_value (x : ℤ) (h : (6.1 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 620) :
  x ≤ 2 ∧ ∃ y : ℤ, y > 2 → (6.1 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 620 :=
by sorry

end greatest_x_value_l773_77311


namespace real_part_of_complex_product_l773_77336

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (1 - Complex.I) * (2 + Complex.I) ∧ z.re = 3 := by
  sorry

end real_part_of_complex_product_l773_77336


namespace carbon_count_in_compound_l773_77362

/-- Represents the atomic weights of elements in atomic mass units (amu) -/
structure AtomicWeights where
  copper : ℝ
  carbon : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound given its composition -/
def molecularWeight (weights : AtomicWeights) (copperCount : ℕ) (carbonCount : ℕ) (oxygenCount : ℕ) : ℝ :=
  weights.copper * copperCount + weights.carbon * carbonCount + weights.oxygen * oxygenCount

/-- Theorem stating that a compound with 1 Copper, n Carbon, and 3 Oxygen atoms
    with a molecular weight of 124 amu has 1 Carbon atom -/
theorem carbon_count_in_compound (weights : AtomicWeights) 
    (h1 : weights.copper = 63.55)
    (h2 : weights.carbon = 12.01)
    (h3 : weights.oxygen = 16.00) :
  ∃ (n : ℕ), molecularWeight weights 1 n 3 = 124 ∧ n = 1 := by
  sorry


end carbon_count_in_compound_l773_77362


namespace sqrt_mixed_number_simplification_l773_77342

theorem sqrt_mixed_number_simplification :
  Real.sqrt (12 + 9/16) = Real.sqrt 201 / 4 := by
  sorry

end sqrt_mixed_number_simplification_l773_77342


namespace problem_statement_l773_77315

theorem problem_statement (M N : ℝ) 
  (h1 : (4 : ℝ) / 7 = M / 77)
  (h2 : (4 : ℝ) / 7 = 98 / (N^2)) : 
  M + N = 57.1 := by
  sorry

end problem_statement_l773_77315


namespace second_snake_length_l773_77333

/-- Proves that the length of the second snake is 16 inches -/
theorem second_snake_length (total_snakes : Nat) (first_snake_feet : Nat) (third_snake_inches : Nat) (total_length_inches : Nat) (inches_per_foot : Nat) :
  total_snakes = 3 →
  first_snake_feet = 2 →
  third_snake_inches = 10 →
  total_length_inches = 50 →
  inches_per_foot = 12 →
  total_length_inches - (first_snake_feet * inches_per_foot + third_snake_inches) = 16 := by
  sorry

end second_snake_length_l773_77333


namespace polynomial_degree_l773_77318

/-- The degree of the polynomial (3x^5 + 2x^4 - x + 5)(4x^11 - 2x^8 + 5x^5 - 9) - (x^2 - 3)^9 is 18 -/
theorem polynomial_degree : ℕ := by
  sorry

end polynomial_degree_l773_77318


namespace triangle_side_ratio_l773_77366

theorem triangle_side_ratio (a b c : ℝ) (A : ℝ) (h1 : A = 2 * Real.pi / 3) 
  (h2 : a^2 = 2*b*c + 3*c^2) : c/b = 1/2 := by
  sorry

end triangle_side_ratio_l773_77366


namespace ellipse_k_range_l773_77358

/-- An ellipse represented by the equation x^2 + ky^2 = 2 with foci on the y-axis -/
structure Ellipse where
  k : ℝ
  eq : ∀ x y : ℝ, x^2 + k * y^2 = 2
  foci_on_y : True  -- This is a placeholder for the condition that foci are on y-axis

/-- The range of k for the given ellipse -/
def k_range (e : Ellipse) : Set ℝ :=
  {k : ℝ | 0 < k ∧ k < 1}

/-- Theorem stating that for the given ellipse, k is in the range (0, 1) -/
theorem ellipse_k_range (e : Ellipse) : e.k ∈ k_range e := by
  sorry

end ellipse_k_range_l773_77358


namespace correct_seating_arrangement_l773_77372

/-- Represents whether a person is sitting or not -/
inductive Sitting : Type
| yes : Sitting
| no : Sitting

/-- The seating arrangement of individuals M, I, P, and A -/
structure SeatingArrangement :=
  (M : Sitting)
  (I : Sitting)
  (P : Sitting)
  (A : Sitting)

/-- The theorem stating the correct seating arrangement based on the given conditions -/
theorem correct_seating_arrangement :
  ∀ (arrangement : SeatingArrangement),
    arrangement.M = Sitting.no →
    (arrangement.M = Sitting.no → arrangement.I = Sitting.yes) →
    (arrangement.I = Sitting.yes → arrangement.P = Sitting.yes) →
    arrangement.A = Sitting.no →
    (arrangement.P = Sitting.yes ∧ 
     arrangement.I = Sitting.yes ∧ 
     arrangement.M = Sitting.no ∧ 
     arrangement.A = Sitting.no) :=
by sorry


end correct_seating_arrangement_l773_77372


namespace parametric_to_ordinary_equation_l773_77379

theorem parametric_to_ordinary_equation 
  (t : ℝ) (x y : ℝ) 
  (h1 : t ≥ 0) 
  (h2 : x = Real.sqrt t + 1) 
  (h3 : y = 2 * Real.sqrt t - 1) : 
  y = 2 * x - 3 ∧ x ≥ 1 := by
  sorry

end parametric_to_ordinary_equation_l773_77379


namespace rectangle_area_increase_l773_77383

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let original_area := L * W
  let new_length := 1.2 * L
  let new_width := 1.2 * W
  let new_area := new_length * new_width
  (new_area - original_area) / original_area * 100 = 44 := by
sorry

end rectangle_area_increase_l773_77383


namespace subtraction_result_l773_77380

theorem subtraction_result : 3.57 - 2.15 = 1.42 := by
  sorry

end subtraction_result_l773_77380


namespace f_properties_l773_77354

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

theorem f_properties (a : ℝ) :
  -- Part 1: Monotonically decreasing intervals
  (∀ x < -1, (f' x) < 0) ∧
  (∀ x > 3, (f' x) < 0) ∧
  -- Part 2: Maximum and minimum values
  (∃ x ∈ Set.Icc (-2) 2, f a x = 20) →
  (∃ y ∈ Set.Icc (-2) 2, f a y = -7 ∧ ∀ z ∈ Set.Icc (-2) 2, f a z ≥ f a y) :=
by sorry

end f_properties_l773_77354


namespace pentagon_interior_angle_mean_l773_77349

/-- The mean value of the measures of the interior angles of a pentagon is 108 degrees. -/
theorem pentagon_interior_angle_mean :
  let n : ℕ := 5  -- number of sides in a pentagon
  let sum_of_angles : ℝ := (n - 2) * 180  -- sum of interior angles
  let mean_angle : ℝ := sum_of_angles / n  -- mean value of interior angles
  mean_angle = 108 := by
  sorry

end pentagon_interior_angle_mean_l773_77349


namespace binomial_coefficient_sum_l773_77327

theorem binomial_coefficient_sum (x a : ℝ) (x_nonzero : x ≠ 0) (a_nonzero : a ≠ 0) :
  (Finset.range 7).sum (λ k => Nat.choose 6 k) = 64 := by
  sorry

end binomial_coefficient_sum_l773_77327


namespace sunglasses_and_hat_probability_l773_77324

/-- The probability that a randomly selected person wearing sunglasses is also wearing a hat -/
theorem sunglasses_and_hat_probability
  (total_sunglasses : ℕ)
  (total_hats : ℕ)
  (prob_sunglasses_given_hat : ℚ)
  (h1 : total_sunglasses = 60)
  (h2 : total_hats = 45)
  (h3 : prob_sunglasses_given_hat = 3 / 5) :
  (total_hats : ℚ) * prob_sunglasses_given_hat / total_sunglasses = 9 / 20 := by
  sorry

end sunglasses_and_hat_probability_l773_77324


namespace sum_of_cubic_and_quartic_terms_l773_77375

theorem sum_of_cubic_and_quartic_terms (π : ℝ) : 3 * (3 - π)^3 + 4 * (2 - π)^4 = 1 := by
  sorry

end sum_of_cubic_and_quartic_terms_l773_77375


namespace parsley_sprig_count_l773_77377

/-- The number of parsley sprigs Carmen started with -/
def initial_sprigs : ℕ := 25

/-- The number of plates decorated with whole sprigs -/
def whole_sprig_plates : ℕ := 8

/-- The number of plates decorated with half sprigs -/
def half_sprig_plates : ℕ := 12

/-- The number of sprigs left after decorating -/
def remaining_sprigs : ℕ := 11

theorem parsley_sprig_count : 
  initial_sprigs = whole_sprig_plates + (half_sprig_plates / 2) + remaining_sprigs :=
by sorry

end parsley_sprig_count_l773_77377


namespace courtyard_length_l773_77300

/-- Proves that a courtyard with given width and number of bricks of specific dimensions has a certain length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) :
  width = 14 →
  brick_length = 0.25 →
  brick_width = 0.15 →
  num_bricks = 8960 →
  (width * (num_bricks * brick_length * brick_width / width)) = 24 := by
  sorry

#check courtyard_length

end courtyard_length_l773_77300


namespace rohan_salary_l773_77387

def food_expense : ℚ := 30 / 100
def rent_expense : ℚ := 20 / 100
def entertainment_expense : ℚ := 10 / 100
def conveyance_expense : ℚ := 5 / 100
def education_expense : ℚ := 10 / 100
def utilities_expense : ℚ := 10 / 100
def miscellaneous_expense : ℚ := 5 / 100
def savings_amount : ℕ := 2500

def total_expenses : ℚ :=
  food_expense + rent_expense + entertainment_expense + conveyance_expense +
  education_expense + utilities_expense + miscellaneous_expense

def savings_percentage : ℚ := 1 - total_expenses

theorem rohan_salary :
  ∃ (salary : ℕ), (↑savings_amount : ℚ) / (↑salary : ℚ) = savings_percentage ∧ salary = 25000 :=
by sorry

end rohan_salary_l773_77387


namespace fraction_operations_l773_77310

/-- Define the † operation for fractions -/
def dagger (a b c d : ℚ) : ℚ := a * c * (d / b)

/-- Define the * operation for fractions -/
def star (a b c d : ℚ) : ℚ := a * c * (b / d)

/-- Theorem stating that (5/6)†(7/9)*(2/3) = 140 -/
theorem fraction_operations : 
  star (dagger (5/6) (7/9)) (2/3) = 140 := by sorry

end fraction_operations_l773_77310


namespace cubic_factorization_l773_77322

theorem cubic_factorization (x : ℝ) : 2*x^3 - 4*x^2 + 2*x = 2*x*(x-1)^2 := by
  sorry

end cubic_factorization_l773_77322


namespace arithmetic_sequence_max_ratio_l773_77302

theorem arithmetic_sequence_max_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n * (a 1 + a n) / 2) →  -- Definition of S_n for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 17 > 0 →
  S 18 < 0 →
  (∀ k ∈ Finset.range 15, S (k + 1) / a (k + 1) ≤ S 9 / a 9) :=
by sorry

end arithmetic_sequence_max_ratio_l773_77302


namespace floor_of_2_99_l773_77301

-- Define the floor function
def floor (x : ℝ) : ℤ := sorry

-- State the properties of the floor function
axiom floor_le (x : ℝ) : (floor x : ℝ) ≤ x
axiom floor_lt (x : ℝ) : x < (floor x : ℝ) + 1

-- Theorem statement
theorem floor_of_2_99 : floor 2.99 = 2 := by sorry

end floor_of_2_99_l773_77301


namespace campers_in_two_classes_l773_77370

/-- Represents the number of campers in a single class -/
def class_size : ℕ := 20

/-- Represents the number of campers in all three classes -/
def in_all_classes : ℕ := 4

/-- Represents the number of campers in exactly one class -/
def in_one_class : ℕ := 24

/-- Represents the total number of campers -/
def total_campers : ℕ := class_size * 3 - 2 * in_all_classes

theorem campers_in_two_classes : 
  ∃ (x : ℕ), x = total_campers - in_one_class - in_all_classes ∧ x = 12 := by
  sorry

end campers_in_two_classes_l773_77370


namespace triangle_sine_cosine_equality_l773_77326

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Add conditions for a valid triangle
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : α + β + γ = π
  -- Add triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_sine_cosine_equality (t : Triangle) :
  t.b * Real.sin t.β + t.a * Real.cos t.β * Real.sin t.γ =
  t.c * Real.sin t.γ + t.a * Real.cos t.γ * Real.sin t.β := by
  sorry

end triangle_sine_cosine_equality_l773_77326


namespace xyz_inequalities_l773_77388

theorem xyz_inequalities (x y z : ℝ) 
  (h1 : x < y) (h2 : y < z) 
  (h3 : x + y + z = 6) 
  (h4 : x*y + y*z + z*x = 9) : 
  0 < x ∧ x < 1 ∧ 1 < y ∧ y < 3 ∧ 3 < z ∧ z < 4 := by
  sorry

end xyz_inequalities_l773_77388


namespace not_square_for_any_base_l773_77313

-- Define the representation of a number in base b
def base_b_representation (b : ℕ) : ℕ := b^2 + 3*b + 3

-- Theorem statement
theorem not_square_for_any_base :
  ∀ b : ℕ, b ≥ 2 → ¬ ∃ n : ℕ, base_b_representation b = n^2 :=
by sorry

end not_square_for_any_base_l773_77313


namespace strawberry_division_l773_77353

theorem strawberry_division (brother_baskets : ℕ) (strawberries_per_basket : ℕ) 
  (kimberly_multiplier : ℕ) (parents_difference : ℕ) (family_members : ℕ) :
  brother_baskets = 3 →
  strawberries_per_basket = 15 →
  kimberly_multiplier = 8 →
  parents_difference = 93 →
  family_members = 4 →
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let kimberly_strawberries := kimberly_multiplier * brother_strawberries
  let parents_strawberries := kimberly_strawberries - parents_difference
  let total_strawberries := brother_strawberries + kimberly_strawberries + parents_strawberries
  (total_strawberries / family_members : ℕ) = 168 :=
by
  sorry

end strawberry_division_l773_77353


namespace nested_fourth_root_l773_77374

theorem nested_fourth_root (M : ℝ) (h : M > 1) :
  (M * (M^(1/4) * M^(1/16)))^(1/4) = M^(21/64) :=
sorry

end nested_fourth_root_l773_77374


namespace largest_b_for_divisibility_by_three_l773_77346

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_b_for_divisibility_by_three :
  ∀ b : ℕ, b ≤ 9 →
    (is_divisible_by_three (500000 + 100000 * b + 6584) ↔ is_divisible_by_three (b + 28)) ∧
    (∀ k : ℕ, k ≤ 9 ∧ k > b → ¬is_divisible_by_three (500000 + 100000 * k + 6584)) →
    b = 8 :=
by sorry

end largest_b_for_divisibility_by_three_l773_77346


namespace white_square_area_l773_77392

/-- Given a cube with edge length 10 feet and 300 square feet of paint used for borders,
    the area of the white square on each face is 50 square feet. -/
theorem white_square_area (cube_edge : ℝ) (paint_area : ℝ) : 
  cube_edge = 10 →
  paint_area = 300 →
  (6 * cube_edge^2 - paint_area) / 6 = 50 := by
  sorry

end white_square_area_l773_77392


namespace remainder_eleven_pow_thousand_mod_five_hundred_l773_77348

theorem remainder_eleven_pow_thousand_mod_five_hundred :
  11^1000 % 500 = 1 := by
  sorry

end remainder_eleven_pow_thousand_mod_five_hundred_l773_77348


namespace f_neither_even_nor_odd_l773_77314

-- Define the function f on the given domain
def f : {x : ℝ | -1 < x ∧ x ≤ 1} → ℝ := fun x => x.val ^ 2

-- State the theorem
theorem f_neither_even_nor_odd :
  ¬(∀ x : {x : ℝ | -1 < x ∧ x ≤ 1}, f ⟨-x.val, by sorry⟩ = f x) ∧
  ¬(∀ x : {x : ℝ | -1 < x ∧ x ≤ 1}, f ⟨-x.val, by sorry⟩ = -f x) :=
by sorry

end f_neither_even_nor_odd_l773_77314


namespace fraction_equality_l773_77384

theorem fraction_equality (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3/7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4/3 := by
  sorry

end fraction_equality_l773_77384


namespace candy_distribution_l773_77343

/-- 
Given the initial number of candies, the number of friends, and the additional candies bought,
prove that the number of candies each friend will receive is equal to the total number of candies
divided by the number of friends.
-/
theorem candy_distribution (initial_candies : ℕ) (friends : ℕ) (additional_candies : ℕ)
  (h1 : initial_candies = 35)
  (h2 : friends = 10)
  (h3 : additional_candies = 15)
  (h4 : friends > 0) :
  (initial_candies + additional_candies) / friends = 5 := by
  sorry

end candy_distribution_l773_77343


namespace polygon_with_equal_angle_sums_l773_77352

/-- The number of sides of a polygon where the sum of interior angles equals the sum of exterior angles -/
theorem polygon_with_equal_angle_sums (n : ℕ) : n > 2 →
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end polygon_with_equal_angle_sums_l773_77352


namespace max_value_quadratic_l773_77351

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) :
  (6 - x) * x ≤ 9 ∧ ∃ y, 0 < y ∧ y < 6 ∧ (6 - y) * y = 9 := by
  sorry

end max_value_quadratic_l773_77351


namespace stating_last_seat_probability_is_reciprocal_seven_seats_probability_l773_77393

/-- 
Represents the probability that the last passenger sits in their own seat 
in a seating arrangement problem with n seats and n passengers.
-/
def last_seat_probability (n : ℕ) : ℚ :=
  if n = 0 then 0
  else 1 / n

/-- 
Theorem stating that the probability of the last passenger sitting in their own seat
is 1/n for any number of seats n > 0.
-/
theorem last_seat_probability_is_reciprocal (n : ℕ) (h : n > 0) : 
  last_seat_probability n = 1 / n := by
  sorry

/-- 
Corollary for the specific case of 7 seats, as in the original problem.
-/
theorem seven_seats_probability : 
  last_seat_probability 7 = 1 / 7 := by
  sorry

end stating_last_seat_probability_is_reciprocal_seven_seats_probability_l773_77393


namespace geometric_sequence_first_term_l773_77316

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℚ)
  (h_geometric : GeometricSequence a)
  (h_third_term : a 3 = 36)
  (h_fourth_term : a 4 = 54) :
  a 1 = 16 := by
  sorry

#check geometric_sequence_first_term

end geometric_sequence_first_term_l773_77316


namespace product_of_numbers_l773_77312

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 153) : x * y = 36 := by
  sorry

end product_of_numbers_l773_77312


namespace intersection_range_l773_77395

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_O₂ (x y r : ℝ) : Prop := (x - 7)^2 + y^2 = r^2

-- Define the condition for r
def r_positive (r : ℝ) : Prop := r > 0

-- Define the intersection condition
def circles_intersect (r : ℝ) : Prop :=
  ∃ x y, circle_O₁ x y ∧ circle_O₂ x y r

-- Main theorem
theorem intersection_range :
  ∀ r, r_positive r → (circles_intersect r ↔ 2 < r ∧ r < 12) :=
sorry

end intersection_range_l773_77395


namespace inequality_condition_l773_77355

theorem inequality_condition (a : ℝ) : 
  (∀ x, -2 < x ∧ x < -1 → (x + a) * (x + 1) < 0) ∧ 
  (∃ x, (x + a) * (x + 1) < 0 ∧ (x ≤ -2 ∨ x ≥ -1)) ↔ 
  a > 2 := by sorry

end inequality_condition_l773_77355


namespace probability_six_consecutive_heads_l773_77369

/-- A sequence of coin flips represented as a list of booleans, where true represents heads and false represents tails. -/
def CoinFlips := List Bool

/-- The number of coin flips. -/
def numFlips : Nat := 10

/-- A function that checks if a list of coin flips contains at least 6 consecutive heads. -/
def hasAtLeastSixConsecutiveHeads (flips : CoinFlips) : Bool :=
  sorry

/-- The total number of possible outcomes for 10 coin flips. -/
def totalOutcomes : Nat := 2^numFlips

/-- The number of favorable outcomes (sequences with at least 6 consecutive heads). -/
def favorableOutcomes : Nat := 129

/-- The probability of getting at least 6 consecutive heads in 10 flips of a fair coin. -/
theorem probability_six_consecutive_heads :
  (favorableOutcomes : ℚ) / totalOutcomes = 129 / 1024 :=
sorry

end probability_six_consecutive_heads_l773_77369


namespace num_factors_of_m_l773_77305

/-- The number of natural-number factors of m = 2^3 * 3^3 * 5^4 * 6^5 -/
def num_factors (m : ℕ) : ℕ := sorry

/-- m is defined as 2^3 * 3^3 * 5^4 * 6^5 -/
def m : ℕ := 2^3 * 3^3 * 5^4 * 6^5

theorem num_factors_of_m :
  num_factors m = 405 := by sorry

end num_factors_of_m_l773_77305


namespace halloween_costume_payment_l773_77368

theorem halloween_costume_payment (last_year_cost : ℝ) (price_increase_percent : ℝ) (deposit_percent : ℝ) : 
  last_year_cost = 250 →
  price_increase_percent = 40 →
  deposit_percent = 10 →
  let this_year_cost := last_year_cost * (1 + price_increase_percent / 100)
  let deposit := this_year_cost * (deposit_percent / 100)
  let remaining_payment := this_year_cost - deposit
  remaining_payment = 315 :=
by
  sorry

end halloween_costume_payment_l773_77368


namespace expression_equality_l773_77325

theorem expression_equality (x y z : ℝ) : 
  (2 * x - (3 * y - 4 * z)) - ((2 * x - 3 * y) - 5 * z) = 9 * z := by
  sorry

end expression_equality_l773_77325


namespace area_of_specific_isosceles_triangle_l773_77335

/-- An isosceles triangle with given heights -/
structure IsoscelesTriangle where
  /-- Height dropped to the base -/
  baseHeight : ℝ
  /-- Height dropped to the lateral side -/
  lateralHeight : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True

/-- Calculate the area of an isosceles triangle given its heights -/
def areaOfIsoscelesTriangle (triangle : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles triangle is 75 -/
theorem area_of_specific_isosceles_triangle :
  let triangle : IsoscelesTriangle := {
    baseHeight := 10,
    lateralHeight := 12,
    isIsosceles := True.intro
  }
  areaOfIsoscelesTriangle triangle = 75 := by
  sorry

end area_of_specific_isosceles_triangle_l773_77335


namespace infinite_pairs_exist_infinitely_many_pairs_l773_77391

/-- Recursive definition of the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 4
  | 1 => 11
  | (n + 2) => 3 * a (n + 1) - a n

/-- Theorem stating the existence of infinitely many pairs satisfying the given properties -/
theorem infinite_pairs_exist : ∀ n : ℕ, n ≥ 1 → 
  (a n < a (n + 1)) ∧ 
  (Nat.gcd (a n) (a (n + 1)) = 1) ∧
  (a n ∣ a (n + 1)^2 - 5) ∧
  (a (n + 1) ∣ a n^2 - 5) := by
  sorry

/-- Corollary: There exist infinitely many pairs of positive integers satisfying the properties -/
theorem infinitely_many_pairs : 
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ, 
    let (a, b) := f n
    (a > b) ∧
    (Nat.gcd a b = 1) ∧
    (a ∣ b^2 - 5) ∧
    (b ∣ a^2 - 5) := by
  sorry

end infinite_pairs_exist_infinitely_many_pairs_l773_77391


namespace beach_volleyball_max_players_l773_77373

theorem beach_volleyball_max_players : ∃ (n : ℕ), n > 0 ∧ n ≤ 13 ∧ 
  (∀ (m : ℕ), m > 13 → ¬(
    (∃ (games : Finset (Finset ℕ)), 
      games.card = m ∧ 
      (∀ g ∈ games, g.card = 4) ∧
      (∀ i j, i < m ∧ j < m ∧ i ≠ j → ∃ g ∈ games, i ∈ g ∧ j ∈ g)
    )
  )) := by sorry

end beach_volleyball_max_players_l773_77373


namespace unique_factorization_1870_l773_77378

/-- A function that returns true if a number is composed of only prime factors -/
def isPrimeComposite (n : Nat) : Bool :=
  sorry

/-- A function that returns true if a number is composed of a prime factor multiplied by a one-digit non-prime number -/
def isPrimeTimesNonPrime (n : Nat) : Bool :=
  sorry

/-- A function that counts the number of valid factorizations of n according to the given conditions -/
def countValidFactorizations (n : Nat) : Nat :=
  sorry

theorem unique_factorization_1870 :
  countValidFactorizations 1870 = 1 := by
  sorry

end unique_factorization_1870_l773_77378


namespace greatest_y_l773_77304

theorem greatest_y (y : ℕ) (h1 : y > 0) (h2 : ∃ k : ℕ, y = 4 * k) (h3 : y^3 < 8000) :
  y ≤ 16 ∧ ∃ y' : ℕ, y' > 0 ∧ (∃ k : ℕ, y' = 4 * k) ∧ y'^3 < 8000 ∧ y' = 16 :=
sorry

end greatest_y_l773_77304


namespace lewis_weekly_rent_l773_77339

/-- Calculates the weekly rent given the total rent and number of weeks -/
def weekly_rent (total_rent : ℕ) (num_weeks : ℕ) : ℚ :=
  (total_rent : ℚ) / (num_weeks : ℚ)

/-- Theorem: The weekly rent for Lewis during harvest season -/
theorem lewis_weekly_rent :
  weekly_rent 527292 1359 = 388 := by
  sorry

end lewis_weekly_rent_l773_77339
