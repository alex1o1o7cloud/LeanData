import Mathlib

namespace NUMINAMATH_CALUDE_conversion_equivalence_l1319_131953

/-- Conversion rates between different units --/
structure ConversionRates where
  knicks_to_knacks : ℚ  -- 5 knicks = 3 knacks
  knacks_to_knocks : ℚ  -- 2 knacks = 5 knocks
  knocks_to_kracks : ℚ  -- 4 knocks = 1 krack

/-- Calculate the equivalent number of knicks for a given number of knocks --/
def knocks_to_knicks (rates : ConversionRates) (knocks : ℚ) : ℚ :=
  knocks * rates.knacks_to_knocks * rates.knicks_to_knacks

/-- Calculate the equivalent number of kracks for a given number of knocks --/
def knocks_to_kracks (rates : ConversionRates) (knocks : ℚ) : ℚ :=
  knocks * rates.knocks_to_kracks

theorem conversion_equivalence (rates : ConversionRates) 
  (h1 : rates.knicks_to_knacks = 3 / 5)
  (h2 : rates.knacks_to_knocks = 5 / 2)
  (h3 : rates.knocks_to_kracks = 1 / 4) :
  knocks_to_knicks rates 50 = 100 / 3 ∧ knocks_to_kracks rates 50 = 25 / 3 := by
  sorry

#check conversion_equivalence

end NUMINAMATH_CALUDE_conversion_equivalence_l1319_131953


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l1319_131920

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop :=
  3 * x^2 - 36 * y^2 - 18 * x + 27 = 0

/-- The two lines represented by the equation -/
def line1 (x y : ℝ) : Prop :=
  x = 3 + 2 * Real.sqrt 3 * y

def line2 (x y : ℝ) : Prop :=
  x = 3 - 2 * Real.sqrt 3 * y

/-- Theorem stating that the equation represents two lines -/
theorem equation_represents_two_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l1319_131920


namespace NUMINAMATH_CALUDE_jessica_seashells_count_l1319_131938

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 18

/-- The total number of seashells Mary and Jessica found together -/
def total_seashells : ℕ := 59

/-- The number of seashells Jessica found -/
def jessica_seashells : ℕ := total_seashells - mary_seashells

theorem jessica_seashells_count : jessica_seashells = 41 := by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_count_l1319_131938


namespace NUMINAMATH_CALUDE_integral_tan_ln_cos_l1319_131939

theorem integral_tan_ln_cos (x : ℝ) :
  HasDerivAt (fun x => -1/2 * (Real.log (Real.cos x))^2) (Real.tan x * Real.log (Real.cos x)) x :=
by sorry

end NUMINAMATH_CALUDE_integral_tan_ln_cos_l1319_131939


namespace NUMINAMATH_CALUDE_tournament_27_teams_26_games_l1319_131990

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- Number of games needed to determine a winner in a single-elimination tournament -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: A single-elimination tournament with 27 teams requires 26 games to determine a winner -/
theorem tournament_27_teams_26_games :
  ∀ (t : Tournament), t.num_teams = 27 → t.no_ties = true → games_to_winner t = 26 := by
  sorry

end NUMINAMATH_CALUDE_tournament_27_teams_26_games_l1319_131990


namespace NUMINAMATH_CALUDE_vanessa_video_files_l1319_131902

theorem vanessa_video_files :
  ∀ (initial_music_files initial_video_files deleted_files remaining_files : ℕ),
    initial_music_files = 16 →
    deleted_files = 30 →
    remaining_files = 34 →
    initial_music_files + initial_video_files = deleted_files + remaining_files →
    initial_video_files = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_vanessa_video_files_l1319_131902


namespace NUMINAMATH_CALUDE_circle_to_octagon_area_ratio_l1319_131906

/-- The ratio of the area of a circle inscribed in a regular octagon
    (where the circle's radius equals the octagon's apothem)
    to the area of the octagon itself. -/
theorem circle_to_octagon_area_ratio : ∃ (a b : ℕ), (a : ℝ).sqrt / b * π = (π / (4 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_circle_to_octagon_area_ratio_l1319_131906


namespace NUMINAMATH_CALUDE_hand_mitt_cost_is_14_l1319_131900

/-- The cost of cooking gear for Eve's nieces --/
def cooking_gear_cost (hand_mitt_cost : ℝ) : Prop :=
  let apron_cost : ℝ := 16
  let utensils_cost : ℝ := 10
  let knife_cost : ℝ := 2 * utensils_cost
  let total_cost_per_niece : ℝ := hand_mitt_cost + apron_cost + utensils_cost + knife_cost
  let discount_rate : ℝ := 0.75
  let number_of_nieces : ℕ := 3
  let total_spent : ℝ := 135
  discount_rate * (number_of_nieces : ℝ) * total_cost_per_niece = total_spent

theorem hand_mitt_cost_is_14 :
  ∃ (hand_mitt_cost : ℝ), cooking_gear_cost hand_mitt_cost ∧ hand_mitt_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_hand_mitt_cost_is_14_l1319_131900


namespace NUMINAMATH_CALUDE_expression_equality_l1319_131946

theorem expression_equality : 6 * 1000 + 5 * 100 + 6 * 1 = 6506 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1319_131946


namespace NUMINAMATH_CALUDE_female_fox_terriers_count_l1319_131918

theorem female_fox_terriers_count 
  (total_dogs : ℕ) 
  (total_females : ℕ) 
  (total_fox_terriers : ℕ) 
  (male_shih_tzus : ℕ) 
  (h1 : total_dogs = 2012)
  (h2 : total_females = 1110)
  (h3 : total_fox_terriers = 1506)
  (h4 : male_shih_tzus = 202) :
  total_fox_terriers - (total_dogs - total_females - male_shih_tzus) = 806 :=
by
  sorry

end NUMINAMATH_CALUDE_female_fox_terriers_count_l1319_131918


namespace NUMINAMATH_CALUDE_garden_area_l1319_131931

/-- Represents a rectangular garden with given properties. -/
structure Garden where
  length : ℝ
  width : ℝ
  length_walk : ℝ
  perimeter_walk : ℝ
  length_condition : length * 30 = length_walk
  perimeter_condition : (2 * length + 2 * width) * 12 = perimeter_walk
  walk_equality : length_walk = perimeter_walk
  length_walk_value : length_walk = 1500

/-- The area of the garden with the given conditions is 625 square meters. -/
theorem garden_area (g : Garden) : g.length * g.width = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l1319_131931


namespace NUMINAMATH_CALUDE_max_a_value_l1319_131908

def is_lattice_point (x y : ℤ) : Prop := True

def line_passes_through_lattice_point (m : ℚ) : Prop :=
  ∃ x y : ℤ, 0 < x ∧ x ≤ 50 ∧ is_lattice_point x y ∧ y = m * x + 5

theorem max_a_value :
  ∀ a : ℚ, (∀ m : ℚ, 2/3 < m → m < a → ¬line_passes_through_lattice_point m) →
    a ≤ 35/51 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1319_131908


namespace NUMINAMATH_CALUDE_union_equality_implies_m_values_l1319_131904

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equality_implies_m_values (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_values_l1319_131904


namespace NUMINAMATH_CALUDE_parallelepiped_to_cube_l1319_131926

/-- Represents a rectangular parallelepiped with side lengths (a, b, c) -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a cube with side length s -/
structure Cube where
  s : ℝ

/-- Predicate to check if a parallelepiped can be divided into four parts
    that can be reassembled to form a cube -/
def can_form_cube (p : Parallelepiped) : Prop :=
  ∃ (cube : Cube), 
    cube.s ^ 3 = p.a * p.b * p.c ∧ 
    (∃ (x : ℝ), p.a = 8*x ∧ p.b = 8*x ∧ p.c = 27*x ∧ cube.s = 12*x)

/-- Theorem stating that a rectangular parallelepiped with side ratio 8:8:27
    can be divided into four parts that can be reassembled to form a cube -/
theorem parallelepiped_to_cube : 
  ∀ (p : Parallelepiped), p.a / p.b = 1 ∧ p.b / p.c = 8 / 27 → can_form_cube p :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_to_cube_l1319_131926


namespace NUMINAMATH_CALUDE_initial_girls_count_l1319_131949

theorem initial_girls_count (p : ℕ) : 
  p > 0 →  -- Ensure p is positive
  (p : ℚ) / 2 - 3 = ((p : ℚ) * 2) / 5 → 
  (p : ℚ) / 2 = 15 :=
by
  sorry

#check initial_girls_count

end NUMINAMATH_CALUDE_initial_girls_count_l1319_131949


namespace NUMINAMATH_CALUDE_second_batch_average_l1319_131981

theorem second_batch_average (n1 n2 n3 : ℕ) (a1 a2 a3 overall_avg : ℝ) :
  n1 = 40 →
  n2 = 50 →
  n3 = 60 →
  a1 = 45 →
  a3 = 65 →
  overall_avg = 56.333333333333336 →
  (n1 * a1 + n2 * a2 + n3 * a3) / (n1 + n2 + n3) = overall_avg →
  a2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_second_batch_average_l1319_131981


namespace NUMINAMATH_CALUDE_running_speed_calculation_l1319_131976

/-- Proves that given the conditions, the running speed must be 8 km/hr -/
theorem running_speed_calculation (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) : 
  walking_speed = 4 →
  total_distance = 16 →
  total_time = 3 →
  (total_distance / 2) / walking_speed + (total_distance / 2) / 8 = total_time :=
by
  sorry

#check running_speed_calculation

end NUMINAMATH_CALUDE_running_speed_calculation_l1319_131976


namespace NUMINAMATH_CALUDE_eh_length_l1319_131956

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = 7
  (ex - fx)^2 + (ey - fy)^2 = 7^2 ∧
  -- FG = 21
  (fx - gx)^2 + (fy - gy)^2 = 21^2 ∧
  -- GH = 7
  (gx - hx)^2 + (gy - hy)^2 = 7^2 ∧
  -- HE = 13
  (hx - ex)^2 + (hy - ey)^2 = 13^2 ∧
  -- Angle at H is a right angle
  (ex - hx) * (gx - hx) + (ey - hy) * (gy - hy) = 0

-- Theorem statement
theorem eh_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  let (ex, ey) := q.E
  let (hx, hy) := q.H
  (ex - hx)^2 + (ey - hy)^2 = 24^2 :=
sorry

end NUMINAMATH_CALUDE_eh_length_l1319_131956


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l1319_131952

/-- The number of days a ring toss game earned money, given total earnings and daily earnings. -/
def days_earned (total_earnings daily_earnings : ℕ) : ℕ :=
  total_earnings / daily_earnings

/-- Theorem stating that the ring toss game earned money for 5 days. -/
theorem ring_toss_earnings : days_earned 165 33 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l1319_131952


namespace NUMINAMATH_CALUDE_multiply_add_equality_l1319_131973

theorem multiply_add_equality : 45 * 27 + 18 * 45 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_equality_l1319_131973


namespace NUMINAMATH_CALUDE_not_monomial_two_over_a_l1319_131967

/-- Definition of a monomial -/
def is_monomial (e : ℤ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ x, e x = c * x^n

/-- The expression 2/a is not a monomial -/
theorem not_monomial_two_over_a : ¬ is_monomial (λ a => 2 / a) := by
  sorry

end NUMINAMATH_CALUDE_not_monomial_two_over_a_l1319_131967


namespace NUMINAMATH_CALUDE_farm_equation_correct_l1319_131935

/-- Represents the farm problem with chickens and pigs --/
structure FarmProblem where
  total_heads : ℕ
  total_legs : ℕ
  chicken_count : ℕ
  pig_count : ℕ

/-- The equation correctly represents the farm problem --/
theorem farm_equation_correct (farm : FarmProblem)
  (head_sum : farm.chicken_count + farm.pig_count = farm.total_heads)
  (head_count : farm.total_heads = 70)
  (leg_count : farm.total_legs = 196) :
  2 * farm.chicken_count + 4 * (70 - farm.chicken_count) = 196 := by
  sorry

#check farm_equation_correct

end NUMINAMATH_CALUDE_farm_equation_correct_l1319_131935


namespace NUMINAMATH_CALUDE_probability_mathematics_in_machine_l1319_131942

def mathematics_letters : Finset Char := {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S'}
def machine_letters : Finset Char := {'M', 'A', 'C', 'H', 'I', 'N', 'E'}

theorem probability_mathematics_in_machine :
  (mathematics_letters.filter (λ c => c ∈ machine_letters)).card / mathematics_letters.card = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_mathematics_in_machine_l1319_131942


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l1319_131923

/-- 
Theorem: The largest value of n such that 3x^2 + nx + 72 can be factored 
as the product of two linear factors with integer coefficients is 217.
-/
theorem largest_n_for_factorization : 
  ∃ (n : ℤ), n = 217 ∧ 
  (∀ m : ℤ, m > n → 
    ¬∃ (a b c d : ℤ), 3 * X^2 + m * X + 72 = (a * X + b) * (c * X + d)) ∧
  (∃ (a b c d : ℤ), 3 * X^2 + n * X + 72 = (a * X + b) * (c * X + d)) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l1319_131923


namespace NUMINAMATH_CALUDE_correct_oranges_to_put_back_l1319_131905

/-- Represents the fruit selection problem --/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back --/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to put back --/
theorem correct_oranges_to_put_back (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 15)
  (h4 : fs.initial_avg_price = 48/100)
  (h5 : fs.desired_avg_price = 45/100) :
  oranges_to_put_back fs = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_to_put_back_l1319_131905


namespace NUMINAMATH_CALUDE_triangle_side_length_l1319_131958

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  a = 2 → c = Real.sqrt 2 → Real.cos A = -(Real.sqrt 2) / 4 → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1319_131958


namespace NUMINAMATH_CALUDE_klinker_double_age_in_15_years_l1319_131969

/-- The number of years it will take for Mr. Klinker to be twice as old as his daughter -/
def years_until_double_age (klinker_age : ℕ) (daughter_age : ℕ) : ℕ :=
  (klinker_age - 2 * daughter_age)

/-- Proof that it will take 15 years for Mr. Klinker to be twice as old as his daughter -/
theorem klinker_double_age_in_15_years :
  years_until_double_age 35 10 = 15 := by
  sorry

#eval years_until_double_age 35 10

end NUMINAMATH_CALUDE_klinker_double_age_in_15_years_l1319_131969


namespace NUMINAMATH_CALUDE_share_division_l1319_131936

theorem share_division (total : ℚ) (a b c : ℚ) 
  (h1 : total = 595)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) :
  c = 420 := by sorry

end NUMINAMATH_CALUDE_share_division_l1319_131936


namespace NUMINAMATH_CALUDE_bus_schedule_hours_l1319_131992

/-- The number of hours per day that buses leave the station -/
def hours_per_day (total_buses : ℕ) (days : ℕ) (buses_per_hour : ℕ) : ℚ :=
  (total_buses : ℚ) / (days : ℚ) / (buses_per_hour : ℚ)

/-- Theorem stating that under given conditions, buses leave the station for 12 hours per day -/
theorem bus_schedule_hours (total_buses : ℕ) (days : ℕ) (buses_per_hour : ℕ)
    (h1 : total_buses = 120)
    (h2 : days = 5)
    (h3 : buses_per_hour = 2) :
    hours_per_day total_buses days buses_per_hour = 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_schedule_hours_l1319_131992


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l1319_131913

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (c d : ℝ), x^2 + 6*x - 4 = 0 ↔ (x + c)^2 = d ∧ d = 13 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l1319_131913


namespace NUMINAMATH_CALUDE_no_solution_exists_l1319_131984

theorem no_solution_exists :
  ¬∃ (B C : ℕ+), 
    (Nat.lcm 360 (Nat.lcm B C) = 55440) ∧ 
    (Nat.gcd 360 (Nat.gcd B C) = 15) ∧ 
    (B * C = 2316) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1319_131984


namespace NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l1319_131987

theorem min_value_a_squared_plus_b_squared :
  ∀ a b : ℝ,
  ((-2)^2 + a*(-2) + 2*b = 0) →
  ∀ c d : ℝ,
  (c^2 + d^2 ≥ a^2 + b^2) →
  (a^2 + b^2 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l1319_131987


namespace NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l1319_131971

/-- The largest prime with 2011 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2011 digits -/
axiom q_digits : ∃ (n : ℕ), 10^2010 ≤ q ∧ q < 10^2011

/-- q is the largest prime with 2011 digits -/
axiom q_largest : ∀ (p : ℕ), Nat.Prime p → (∃ (n : ℕ), 10^2010 ≤ p ∧ p < 10^2011) → p ≤ q

theorem q_squared_minus_one_div_fifteen : 15 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l1319_131971


namespace NUMINAMATH_CALUDE_sweet_potato_problem_l1319_131930

theorem sweet_potato_problem (total : ℕ) (sold_to_adams : ℕ) (sold_to_lenon : ℕ) 
  (h1 : total = 80) 
  (h2 : sold_to_adams = 20) 
  (h3 : sold_to_lenon = 15) : 
  total - (sold_to_adams + sold_to_lenon) = 45 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potato_problem_l1319_131930


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1319_131917

theorem unique_triple_solution : 
  ∃! (x y z : ℝ), x + y = 2 ∧ x * y - z^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1319_131917


namespace NUMINAMATH_CALUDE_peach_to_apricot_ratio_l1319_131994

/-- Given a total number of trees and a number of apricot trees, 
    calculate the ratio of peach trees to apricot trees. -/
def tree_ratio (total : ℕ) (apricot : ℕ) : ℚ × ℚ :=
  let peach := total - apricot
  (peach, apricot)

/-- The theorem states that for 232 total trees and 58 apricot trees,
    the ratio of peach trees to apricot trees is 3:1. -/
theorem peach_to_apricot_ratio :
  tree_ratio 232 58 = (3, 1) := by sorry

end NUMINAMATH_CALUDE_peach_to_apricot_ratio_l1319_131994


namespace NUMINAMATH_CALUDE_solution_to_equation_l1319_131947

theorem solution_to_equation : ∃ x : ℕ, 
  (x = 10^2023 - 1) ∧ 
  (567 * x^3 + 171 * x^2 + 15 * x - (3 * x + 5 * x * 10^2023 + 7 * x * 10^(2*2023)) = 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1319_131947


namespace NUMINAMATH_CALUDE_emily_sixth_quiz_score_l1319_131970

def emily_scores : List ℕ := [85, 90, 88, 92, 98]
def desired_mean : ℕ := 92
def num_quizzes : ℕ := 6

theorem emily_sixth_quiz_score :
  ∃ (sixth_score : ℕ),
    (emily_scores.sum + sixth_score) / num_quizzes = desired_mean ∧
    sixth_score = 99 := by
  sorry

end NUMINAMATH_CALUDE_emily_sixth_quiz_score_l1319_131970


namespace NUMINAMATH_CALUDE_alternating_sum_equals_three_to_seven_l1319_131945

theorem alternating_sum_equals_three_to_seven (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ - a₇ = 3^7 := by
sorry

end NUMINAMATH_CALUDE_alternating_sum_equals_three_to_seven_l1319_131945


namespace NUMINAMATH_CALUDE_rabbit_fraction_l1319_131924

theorem rabbit_fraction (initial_cage : ℕ) (added : ℕ) (park : ℕ) : 
  initial_cage = 13 → added = 7 → park = 60 → 
  (initial_cage + added : ℚ) / park = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_fraction_l1319_131924


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_4014_l1319_131980

/-- The area of a quadrilateral with vertices at (1, 1), (1, 5), (3, 5), and (2006, 2003) -/
def quadrilateralArea : ℝ :=
  let A := (1, 1)
  let B := (1, 5)
  let C := (3, 5)
  let D := (2006, 2003)
  -- Area calculation goes here
  0 -- Placeholder

/-- Theorem stating that the area of the quadrilateral is 4014 square units -/
theorem quadrilateral_area_is_4014 : quadrilateralArea = 4014 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_4014_l1319_131980


namespace NUMINAMATH_CALUDE_coffee_stock_solution_l1319_131911

/-- Represents the coffee stock problem -/
def coffee_stock_problem (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (second_batch_decaf_percent : ℝ) (final_decaf_percent : ℝ) : Prop :=
  ∃ (second_batch : ℝ),
    second_batch > 0 ∧
    (initial_stock * initial_decaf_percent + second_batch * second_batch_decaf_percent) / 
    (initial_stock + second_batch) = final_decaf_percent

/-- The solution to the coffee stock problem -/
theorem coffee_stock_solution :
  coffee_stock_problem 400 0.30 0.60 0.36 → 
  ∃ (second_batch : ℝ), second_batch = 100 := by
  sorry

end NUMINAMATH_CALUDE_coffee_stock_solution_l1319_131911


namespace NUMINAMATH_CALUDE_lineup_selection_theorem_l1319_131901

/-- The number of ways to select a lineup of 6 players from a team of 15 players -/
def lineup_selection_ways : ℕ := 3603600

/-- The size of the basketball team -/
def team_size : ℕ := 15

/-- The number of positions to be filled -/
def positions_to_fill : ℕ := 6

theorem lineup_selection_theorem :
  (Finset.range team_size).card.factorial / 
  ((team_size - positions_to_fill).factorial) = lineup_selection_ways :=
sorry

end NUMINAMATH_CALUDE_lineup_selection_theorem_l1319_131901


namespace NUMINAMATH_CALUDE_line_no_intersection_slope_range_l1319_131959

/-- Given points A(-2,3) and B(3,2), and a line l: y = kx - 2, 
    if l has no intersection with line segment AB, 
    then the slope k of line l is in the range (-5/2, 4/3). -/
theorem line_no_intersection_slope_range (k : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (3, 2)
  let l (x : ℝ) := k * x - 2
  (∀ x y, (x, y) ∈ Set.Icc A B → y ≠ l x) →
  k ∈ Set.Ioo (-5/2 : ℝ) (4/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_line_no_intersection_slope_range_l1319_131959


namespace NUMINAMATH_CALUDE_four_Z_three_equals_127_l1319_131978

-- Define the Z operation
def Z (a b : ℝ) : ℝ := a^3 - 3*a*b^2 + 3*a^2*b + b^3

-- Theorem statement
theorem four_Z_three_equals_127 : Z 4 3 = 127 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_equals_127_l1319_131978


namespace NUMINAMATH_CALUDE_problem_statement_l1319_131995

theorem problem_statement (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + 2 = e + Real.sqrt (a + b + c + d - e)) :
  e = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1319_131995


namespace NUMINAMATH_CALUDE_larry_cards_remaining_l1319_131909

/-- Given that Larry initially has 67 cards and Dennis takes 9 cards away,
    prove that Larry now has 58 cards. -/
theorem larry_cards_remaining (initial_cards : ℕ) (cards_taken : ℕ) : 
  initial_cards = 67 → cards_taken = 9 → initial_cards - cards_taken = 58 := by
  sorry

end NUMINAMATH_CALUDE_larry_cards_remaining_l1319_131909


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l1319_131934

def total_group_size : ℕ := 15
def men_count : ℕ := 9
def women_count : ℕ := 6
def selection_size : ℕ := 4

theorem probability_at_least_one_woman :
  let total_combinations := Nat.choose total_group_size selection_size
  let all_men_combinations := Nat.choose men_count selection_size
  (total_combinations - all_men_combinations : ℚ) / total_combinations = 137 / 151 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l1319_131934


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1319_131937

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

/-- The theorem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (n : ℕ) 
  (h_seq : is_positive_geometric_sequence a)
  (h_1 : a 1 * a 2 * a 3 = 4)
  (h_2 : a 4 * a 5 * a 6 = 12)
  (h_3 : a (n-1) * a n * a (n+1) = 324) :
  n = 14 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1319_131937


namespace NUMINAMATH_CALUDE_rope_cutting_l1319_131972

theorem rope_cutting (total_length : ℕ) (equal_pieces : ℕ) (equal_piece_length : ℕ) (remaining_piece_length : ℕ) : 
  total_length = 1165 ∧ 
  equal_pieces = 150 ∧ 
  equal_piece_length = 75 ∧ 
  remaining_piece_length = 100 → 
  (total_length * 10 - equal_pieces * equal_piece_length) / remaining_piece_length + equal_pieces = 154 :=
by sorry

end NUMINAMATH_CALUDE_rope_cutting_l1319_131972


namespace NUMINAMATH_CALUDE_sin_range_theorem_l1319_131940

theorem sin_range_theorem (x : ℝ) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  Real.sin x ≥ Real.sqrt 2 / 2 → 
  x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_sin_range_theorem_l1319_131940


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1319_131948

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1319_131948


namespace NUMINAMATH_CALUDE_f_composite_value_l1319_131921

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else a^x + b

theorem f_composite_value (a b : ℝ) :
  f 0 a b = 2 →
  f (-1) a b = 3 →
  f (f (-3) a b) a b = 2 := by
sorry

end NUMINAMATH_CALUDE_f_composite_value_l1319_131921


namespace NUMINAMATH_CALUDE_spatial_geometry_l1319_131910

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line)
variable (contains : Plane → Line → Prop)
variable (linePerpendicular : Line → Line → Prop)
variable (lineParallel : Line → Line → Prop)
variable (planePerpendicular : Line → Plane → Prop)

-- State the theorem
theorem spatial_geometry 
  (α β : Plane) (l m n : Line) 
  (h1 : perpendicular α β)
  (h2 : intersect α β = l)
  (h3 : contains α m)
  (h4 : contains β n)
  (h5 : linePerpendicular m n) :
  (lineParallel n l → planePerpendicular m β) ∧
  (planePerpendicular m β ∨ planePerpendicular n α) :=
sorry

end NUMINAMATH_CALUDE_spatial_geometry_l1319_131910


namespace NUMINAMATH_CALUDE_inequality_implication_l1319_131928

theorem inequality_implication (a b : ℝ) (h : a - b > 0) : a + 1 > b + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1319_131928


namespace NUMINAMATH_CALUDE_horizontal_distance_on_line_l1319_131975

/-- Given two points on a line, prove that the horizontal distance between them is 3 -/
theorem horizontal_distance_on_line (m n p : ℝ) : 
  (m = n / 7 - 2 / 5) → 
  (m + p = (n + 21) / 7 - 2 / 5) → 
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_horizontal_distance_on_line_l1319_131975


namespace NUMINAMATH_CALUDE_smallest_positive_integer_2016m_43200n_l1319_131965

theorem smallest_positive_integer_2016m_43200n :
  ∃ (k : ℕ+), (∀ (a : ℕ+), (∃ (m n : ℤ), a = 2016 * m + 43200 * n) → k ≤ a) ∧
  (∃ (m n : ℤ), (k : ℕ) = 2016 * m + 43200 * n) ∧
  k = 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_2016m_43200n_l1319_131965


namespace NUMINAMATH_CALUDE_max_qed_value_l1319_131955

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

theorem max_qed_value 
  (D E L M Q : Digit) 
  (h_distinct : D ≠ E ∧ D ≠ L ∧ D ≠ M ∧ D ≠ Q ∧ 
                E ≠ L ∧ E ≠ M ∧ E ≠ Q ∧ 
                L ≠ M ∧ L ≠ Q ∧ 
                M ≠ Q)
  (h_equation : 91 * E.val + 10 * L.val + 101 * M.val = 100 * Q.val + D.val) :
  (∀ (D' E' Q' : Digit), 
    D' ≠ E' ∧ D' ≠ Q' ∧ E' ≠ Q' → 
    100 * Q'.val + 10 * E'.val + D'.val ≤ 893) :=
sorry

end NUMINAMATH_CALUDE_max_qed_value_l1319_131955


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_to_yaxis_l1319_131929

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- A circle with center c and radius r -/
structure Circle where
  center : Point
  radius : ℝ

/-- The y-axis -/
def yAxis := {p : Point | p.x = 0}

/-- Predicate to check if a circle is tangent to the y-axis -/
def isTangentToYAxis (c : Circle) : Prop :=
  c.center.x = c.radius

/-- Theorem: For any point P on the parabola y² = 4x, 
    the circle with diameter PF (where F is the focus) 
    is tangent to the y-axis -/
theorem parabola_circle_tangent_to_yaxis 
  (P : Point) (h : P ∈ Parabola) : 
  ∃ (c : Circle), c.center = ⟨(P.x + focus.x) / 2, P.y / 2⟩ ∧ 
                  c.radius = (P.x + focus.x) / 2 ∧
                  isTangentToYAxis c :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_to_yaxis_l1319_131929


namespace NUMINAMATH_CALUDE_root_in_interval_l1319_131966

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  ∃ r ∈ Set.Ioo 2 2.5, f r = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1319_131966


namespace NUMINAMATH_CALUDE_pyramid_volume_theorem_l1319_131985

/-- Represents a pyramid with a square base and a vertex -/
structure Pyramid where
  base_area : ℝ
  triangle_abe_area : ℝ
  triangle_cde_area : ℝ

/-- Calculate the volume of a pyramid -/
def pyramid_volume (p : Pyramid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem pyramid_volume_theorem (p : Pyramid) 
  (h1 : p.base_area = 256)
  (h2 : p.triangle_abe_area = 120)
  (h3 : p.triangle_cde_area = 110) :
  pyramid_volume p = 1152 :=
sorry

end NUMINAMATH_CALUDE_pyramid_volume_theorem_l1319_131985


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l1319_131919

theorem modular_arithmetic_problem :
  ∃ (a b : ℕ), 
    (7 * a) % 60 = 1 ∧ 
    (13 * b) % 60 = 1 ∧ 
    ((3 * a + 6 * b) % 60) = 51 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l1319_131919


namespace NUMINAMATH_CALUDE_seven_times_coefficient_polynomials_l1319_131954

theorem seven_times_coefficient_polynomials (m n : ℤ) : 
  (∃ k : ℤ, 4 * m - n = 7 * k) → (∃ l : ℤ, 2 * m + 3 * n = 7 * l) := by
  sorry

end NUMINAMATH_CALUDE_seven_times_coefficient_polynomials_l1319_131954


namespace NUMINAMATH_CALUDE_friendly_pairs_complete_l1319_131996

def FriendlyPair (a b c d : ℕ+) : Prop :=
  2 * (a.val + b.val) = c.val * d.val ∧ 2 * (c.val + d.val) = a.val * b.val

def AllFriendlyPairs : Set (ℕ+ × ℕ+ × ℕ+ × ℕ+) :=
  {⟨22, 5, 54, 1⟩, ⟨13, 6, 38, 1⟩, ⟨10, 7, 34, 1⟩, ⟨10, 3, 13, 2⟩,
   ⟨6, 4, 10, 2⟩, ⟨6, 3, 6, 3⟩, ⟨4, 4, 4, 4⟩}

theorem friendly_pairs_complete :
  ∀ a b c d : ℕ+, FriendlyPair a b c d ↔ (a, b, c, d) ∈ AllFriendlyPairs :=
sorry

end NUMINAMATH_CALUDE_friendly_pairs_complete_l1319_131996


namespace NUMINAMATH_CALUDE_xiaoliang_draw_probability_l1319_131982

/-- Represents the labels of balls in the box -/
inductive Label : Type
| one : Label
| two : Label
| three : Label
| four : Label

/-- The state of the box after Xiaoming's draw -/
structure BoxState :=
  (remaining_two : Nat)
  (remaining_three : Nat)
  (remaining_four : Nat)

/-- The initial state of the box -/
def initial_box : BoxState :=
  { remaining_two := 2
  , remaining_three := 1
  , remaining_four := 2 }

/-- The total number of balls remaining in the box -/
def total_remaining (box : BoxState) : Nat :=
  box.remaining_two + box.remaining_three + box.remaining_four

/-- The probability of drawing a ball with a specific label -/
def prob_draw (box : BoxState) (label : Label) : Rat :=
  match label with
  | Label.one => 0  -- No balls labeled 1 remaining
  | Label.two => box.remaining_two / (total_remaining box)
  | Label.three => box.remaining_three / (total_remaining box)
  | Label.four => box.remaining_four / (total_remaining box)

/-- The probability of drawing a ball matching Xiaoming's drawn balls -/
def prob_match_xiaoming (box : BoxState) : Rat :=
  prob_draw box Label.three

theorem xiaoliang_draw_probability :
  prob_match_xiaoming initial_box = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_xiaoliang_draw_probability_l1319_131982


namespace NUMINAMATH_CALUDE_kim_sweaters_theorem_l1319_131974

/-- The number of sweaters Kim knit on Monday -/
def monday_sweaters : ℕ := 8

/-- The number of sweaters Kim knit on Tuesday -/
def tuesday_sweaters : ℕ := monday_sweaters + 2

/-- The number of sweaters Kim knit on Wednesday -/
def wednesday_sweaters : ℕ := tuesday_sweaters - 4

/-- The number of sweaters Kim knit on Thursday -/
def thursday_sweaters : ℕ := tuesday_sweaters - 4

/-- The number of sweaters Kim knit on Friday -/
def friday_sweaters : ℕ := monday_sweaters / 2

/-- The total number of sweaters Kim knit in the week -/
def total_sweaters : ℕ := monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters

theorem kim_sweaters_theorem : total_sweaters = 34 := by
  sorry

end NUMINAMATH_CALUDE_kim_sweaters_theorem_l1319_131974


namespace NUMINAMATH_CALUDE_sons_age_l1319_131925

/-- Proves that given the conditions, the son's present age is 25 years. -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 27 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l1319_131925


namespace NUMINAMATH_CALUDE_sonita_stamp_purchase_l1319_131951

theorem sonita_stamp_purchase (two_q_stamps : ℕ) 
  (h1 : two_q_stamps > 0)
  (h2 : two_q_stamps < 9)
  (h3 : two_q_stamps % 5 = 0) :
  2 * two_q_stamps + 10 * two_q_stamps + (100 - 12 * two_q_stamps) / 5 = 63 := by
  sorry

#check sonita_stamp_purchase

end NUMINAMATH_CALUDE_sonita_stamp_purchase_l1319_131951


namespace NUMINAMATH_CALUDE_magpie_call_not_correlation_l1319_131944

/-- Represents a statement that may or may not indicate a correlation. -/
inductive Statement
| A : Statement  -- A timely snow promises a good harvest
| B : Statement  -- A good teacher produces outstanding students
| C : Statement  -- Smoking is harmful to health
| D : Statement  -- The magpie's call is a sign of happiness

/-- Predicate to determine if a statement represents a correlation. -/
def is_correlation (s : Statement) : Prop :=
  match s with
  | Statement.A => True
  | Statement.B => True
  | Statement.C => True
  | Statement.D => False

/-- Theorem stating that Statement D does not represent a correlation. -/
theorem magpie_call_not_correlation :
  ¬ (is_correlation Statement.D) :=
sorry

end NUMINAMATH_CALUDE_magpie_call_not_correlation_l1319_131944


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l1319_131999

theorem mod_congruence_unique_solution : 
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 9 ∧ n ≡ -245 [ZMOD 10] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l1319_131999


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l1319_131915

theorem ceiling_product_equation : ∃ x : ℝ, (⌈x⌉ : ℝ) * x = 204 ∧ x = 13.6 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l1319_131915


namespace NUMINAMATH_CALUDE_max_gcd_triangular_number_l1319_131927

def triangular_number (n : ℕ+) : ℕ := (n : ℕ) * (n + 1) / 2

theorem max_gcd_triangular_number :
  ∃ (n : ℕ+), Nat.gcd (6 * triangular_number n) (n + 2) = 6 ∧
  ∀ (m : ℕ+), Nat.gcd (6 * triangular_number m) (m + 2) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_triangular_number_l1319_131927


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1319_131979

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * ((x + 5) + (y + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1319_131979


namespace NUMINAMATH_CALUDE_container_volume_l1319_131941

theorem container_volume (x y z : ℝ) 
  (h_order : x < y ∧ y < z)
  (h_xy : 5 * x * y = 120)
  (h_xz : 3 * x * z = 120)
  (h_yz : 2 * y * z = 120) :
  x * y * z = 240 := by
sorry

end NUMINAMATH_CALUDE_container_volume_l1319_131941


namespace NUMINAMATH_CALUDE_xyz_value_l1319_131993

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1319_131993


namespace NUMINAMATH_CALUDE_candle_count_l1319_131986

/-- The number of candles Alex used -/
def used_candles : ℕ := 32

/-- The number of candles Alex has left -/
def leftover_candles : ℕ := 12

/-- The total number of candles Alex had initially -/
def initial_candles : ℕ := used_candles + leftover_candles

theorem candle_count : initial_candles = 44 := by
  sorry

end NUMINAMATH_CALUDE_candle_count_l1319_131986


namespace NUMINAMATH_CALUDE_A_inverse_correct_l1319_131963

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![3, -1, 3],
    ![2, -1, 4],
    ![1,  2, -3]]

def A_inv : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![ 1/2, -3/10,  1/10],
    ![-1,    6/5,   3/5],
    ![-1/2,  7/10,  1/10]]

theorem A_inverse_correct : A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_A_inverse_correct_l1319_131963


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1319_131998

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1319_131998


namespace NUMINAMATH_CALUDE_third_level_lamps_l1319_131977

/-- Represents a pagoda with a given number of stories and lamps -/
structure Pagoda where
  stories : ℕ
  total_lamps : ℕ

/-- Calculates the number of lamps on a specific level of the pagoda -/
def lamps_on_level (p : Pagoda) (level : ℕ) : ℕ :=
  let first_level := p.total_lamps * (1 - 1 / 2^p.stories) / (2^p.stories - 1)
  first_level / 2^(level - 1)

theorem third_level_lamps (p : Pagoda) (h1 : p.stories = 7) (h2 : p.total_lamps = 381) :
  lamps_on_level p 5 = 12 := by
  sorry

#eval lamps_on_level ⟨7, 381⟩ 5

end NUMINAMATH_CALUDE_third_level_lamps_l1319_131977


namespace NUMINAMATH_CALUDE_sequence_squared_l1319_131991

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ m n, m ≥ n → a (m + n) + a (m - n) = (a (2 * m) + a (2 * n)) / 2

theorem sequence_squared (a : ℕ → ℝ) (h : sequence_property a) (h1 : a 1 = 1) :
  ∀ n : ℕ, a n = n^2 := by
  sorry

#check sequence_squared

end NUMINAMATH_CALUDE_sequence_squared_l1319_131991


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_x_plus_x_cubed_l1319_131989

theorem integral_sqrt_plus_x_plus_x_cubed (f : ℝ → ℝ) :
  (∫ x in (0)..(1), (Real.sqrt (1 - x^2) + x + x^3)) = (π + 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_x_plus_x_cubed_l1319_131989


namespace NUMINAMATH_CALUDE_john_earnings_increase_l1319_131943

/-- Calculates the percentage increase in earnings -/
def percentage_increase (initial_earnings final_earnings : ℚ) : ℚ :=
  (final_earnings - initial_earnings) / initial_earnings * 100

/-- Represents John's weekly earnings from two jobs -/
structure WeeklyEarnings where
  job_a_initial : ℚ
  job_a_final : ℚ
  job_b_initial : ℚ
  job_b_final : ℚ

theorem john_earnings_increase (john : WeeklyEarnings)
  (h1 : john.job_a_initial = 60)
  (h2 : john.job_a_final = 78)
  (h3 : john.job_b_initial = 100)
  (h4 : john.job_b_final = 120) :
  percentage_increase (john.job_a_initial + john.job_b_initial)
                      (john.job_a_final + john.job_b_final) = 23.75 := by
  sorry

end NUMINAMATH_CALUDE_john_earnings_increase_l1319_131943


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_four_equals_sqrt_two_l1319_131916

theorem sqrt_of_sqrt_four_equals_sqrt_two : Real.sqrt (Real.sqrt 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_four_equals_sqrt_two_l1319_131916


namespace NUMINAMATH_CALUDE_max_q_minus_r_for_1073_l1319_131988

theorem max_q_minus_r_for_1073 :
  ∃ (q r : ℕ+), 1073 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1073 = 23 * q' + r' → q - r ≥ q' - r' :=
by
  sorry

end NUMINAMATH_CALUDE_max_q_minus_r_for_1073_l1319_131988


namespace NUMINAMATH_CALUDE_percentage_problem_l1319_131933

theorem percentage_problem (P : ℝ) (h : (P / 4) * 2 = 0.02) : P = 4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1319_131933


namespace NUMINAMATH_CALUDE_birthday_crayons_l1319_131964

/-- The number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 291

/-- The number of crayons Paul lost or gave away -/
def crayons_lost_or_given : ℕ := 315

/-- The total number of crayons Paul got for his birthday -/
def total_crayons : ℕ := crayons_left + crayons_lost_or_given

theorem birthday_crayons : total_crayons = 606 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_l1319_131964


namespace NUMINAMATH_CALUDE_equation_solution_l1319_131983

theorem equation_solution : 
  ∃! x : ℚ, (x^2 - 4*x + 3)/(x^2 - 6*x + 5) = (x^2 - 3*x - 10)/(x^2 - 2*x - 15) ∧ x = -19/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1319_131983


namespace NUMINAMATH_CALUDE_circle_ratio_l1319_131912

theorem circle_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (h_area : π * r₂^2 - π * r₁^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l1319_131912


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l1319_131997

def x : ℕ := 5 * 15 * 35

theorem smallest_y_for_perfect_fourth_power (y : ℕ) : 
  y = 46485 ↔ 
  (∀ z : ℕ, z < y → ¬∃ (n : ℕ), x * z = n^4) ∧
  ∃ (n : ℕ), x * y = n^4 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l1319_131997


namespace NUMINAMATH_CALUDE_perfect_square_sum_l1319_131922

theorem perfect_square_sum (a b : ℤ) 
  (h : ∀ (m n : ℕ), ∃ (k : ℕ), a * m^2 + b * n^2 = k^2) : 
  a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l1319_131922


namespace NUMINAMATH_CALUDE_shirt_cost_l1319_131950

theorem shirt_cost (J S : ℝ) 
  (eq1 : 3 * J + 2 * S = 69) 
  (eq2 : 2 * J + 3 * S = 76) : 
  S = 18 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l1319_131950


namespace NUMINAMATH_CALUDE_fourth_game_shots_correct_l1319_131932

-- Define the initial conditions
def initial_shots : ℕ := 45
def initial_made : ℕ := 18
def initial_average : ℚ := 40 / 100
def fourth_game_shots : ℕ := 15
def new_average : ℚ := 55 / 100

-- Define the function to calculate the number of shots made in the fourth game
def fourth_game_made : ℕ := 15

-- Theorem statement
theorem fourth_game_shots_correct :
  (initial_made + fourth_game_made : ℚ) / (initial_shots + fourth_game_shots) = new_average :=
by sorry

end NUMINAMATH_CALUDE_fourth_game_shots_correct_l1319_131932


namespace NUMINAMATH_CALUDE_path_area_calculation_l1319_131907

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Proves that the area of the path around the given field is 675 sq m -/
theorem path_area_calculation (field_length field_width path_width : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5) :
  path_area field_length field_width path_width = 675 := by
  sorry

#eval path_area 75 55 2.5

end NUMINAMATH_CALUDE_path_area_calculation_l1319_131907


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l1319_131957

def complex_mul (a b c d : ℝ) : ℂ := Complex.mk (a * c - b * d) (a * d + b * c)

theorem imaginary_part_of_product :
  (complex_mul 2 1 1 (-3)).im = -5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l1319_131957


namespace NUMINAMATH_CALUDE_devin_initial_height_l1319_131961

/-- The chances of making the basketball team for a given height. -/
def chance_of_making_team (height : ℝ) : ℝ :=
  0.1 + (height - 66) * 0.1

/-- Devin's initial height before growth. -/
def initial_height : ℝ := 68

/-- The amount Devin grew in inches. -/
def growth : ℝ := 3

/-- Devin's final chance of making the team after growth. -/
def final_chance : ℝ := 0.3

theorem devin_initial_height :
  chance_of_making_team (initial_height + growth) = final_chance :=
by sorry

end NUMINAMATH_CALUDE_devin_initial_height_l1319_131961


namespace NUMINAMATH_CALUDE_minutes_in_year_scientific_notation_l1319_131903

/-- The number of days in a year -/
def days_in_year : ℕ := 360

/-- The number of hours in a day -/
def hours_in_day : ℕ := 24

/-- The number of minutes in an hour -/
def minutes_in_hour : ℕ := 60

/-- Converts a natural number to a real number -/
def to_real (n : ℕ) : ℝ := n

/-- Rounds a real number to three significant figures -/
noncomputable def round_to_three_sig_figs (x : ℝ) : ℝ := 
  sorry

/-- The main theorem stating that the number of minutes in a year,
    when expressed in scientific notation with three significant figures,
    is equal to 5.18 × 10^5 -/
theorem minutes_in_year_scientific_notation :
  round_to_three_sig_figs (to_real (days_in_year * hours_in_day * minutes_in_hour)) = 5.18 * 10^5 := by
  sorry

end NUMINAMATH_CALUDE_minutes_in_year_scientific_notation_l1319_131903


namespace NUMINAMATH_CALUDE_remainder_82460_div_8_l1319_131962

theorem remainder_82460_div_8 : 82460 % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_82460_div_8_l1319_131962


namespace NUMINAMATH_CALUDE_robot_models_properties_l1319_131968

/-- Represents the cost and quantity information for robot models --/
structure RobotModels where
  cost_A : ℕ  -- Cost of model A in yuan
  cost_B : ℕ  -- Cost of model B in yuan
  total_A : ℕ  -- Total spent on model A in yuan
  total_B : ℕ  -- Total spent on model B in yuan
  total_units : ℕ  -- Total units to be purchased

/-- Calculates the maximum number of model A units that can be purchased --/
def max_model_A (r : RobotModels) : ℕ :=
  min ((2 * r.total_units) / 3) r.total_units

/-- Theorem stating the properties of the robot models --/
theorem robot_models_properties (r : RobotModels) 
  (h1 : r.cost_B = 2 * r.cost_A - 400)
  (h2 : r.total_A = 96000)
  (h3 : r.total_B = 168000)
  (h4 : r.total_units = 100) :
  r.cost_A = 1600 ∧ r.cost_B = 2800 ∧ max_model_A r = 66 := by
  sorry

#eval max_model_A ⟨1600, 2800, 96000, 168000, 100⟩

end NUMINAMATH_CALUDE_robot_models_properties_l1319_131968


namespace NUMINAMATH_CALUDE_figure_tiling_iff_multiple_of_three_l1319_131960

/-- Represents a figure Φ consisting of three n×n squares. -/
structure Figure (n : ℕ) where
  squares : Fin 3 → Fin n → Fin n → Bool

/-- Represents a 1×3 or 3×1 tile. -/
inductive Tile
  | horizontal : Tile
  | vertical : Tile

/-- A tiling of the figure Φ using 1×3 and 3×1 tiles. -/
def Tiling (n : ℕ) := Set (ℕ × ℕ × Tile)

/-- Predicate to check if a tiling is valid for the given figure. -/
def isValidTiling (n : ℕ) (φ : Figure n) (t : Tiling n) : Prop := sorry

/-- The main theorem stating that a valid tiling exists if and only if n is a multiple of 3. -/
theorem figure_tiling_iff_multiple_of_three (n : ℕ) (φ : Figure n) :
  (n > 1) → (∃ t : Tiling n, isValidTiling n φ t) ↔ ∃ k : ℕ, n = 3 * k :=
sorry

end NUMINAMATH_CALUDE_figure_tiling_iff_multiple_of_three_l1319_131960


namespace NUMINAMATH_CALUDE_money_distribution_l1319_131914

theorem money_distribution (a b c : ℕ) (total : ℕ) : 
  a + b + c = 9 → 
  b = 3 → 
  900 * b = 2700 → 
  900 * (a + b + c) = 2700 * 3 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l1319_131914
