import Mathlib

namespace NUMINAMATH_CALUDE_sixth_term_equals_two_l1418_141836

/-- A geometric sequence with common ratio 2 and positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem sixth_term_equals_two
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_product : a 4 * a 10 = 16) :
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_equals_two_l1418_141836


namespace NUMINAMATH_CALUDE_x_value_l1418_141885

theorem x_value (x : ℝ) : x = 70 * (1 + 11 / 100) → x = 77.7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1418_141885


namespace NUMINAMATH_CALUDE_rice_containers_l1418_141843

theorem rice_containers (total_weight : ℚ) (container_capacity : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 25 / 4 →
  container_capacity = 25 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce / container_capacity : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rice_containers_l1418_141843


namespace NUMINAMATH_CALUDE_boys_playing_neither_l1418_141867

/-- Given a group of boys with information about their sports participation,
    calculate the number of boys who play neither basketball nor football. -/
theorem boys_playing_neither (total : ℕ) (basketball : ℕ) (football : ℕ) (both : ℕ)
    (h_total : total = 22)
    (h_basketball : basketball = 13)
    (h_football : football = 15)
    (h_both : both = 18) :
    total - (basketball + football - both) = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_playing_neither_l1418_141867


namespace NUMINAMATH_CALUDE_longest_segment_squared_in_quarter_circle_l1418_141826

-- Define the diameter of the circle
def circle_diameter : ℝ := 16

-- Define the number of equal sectors
def num_sectors : ℕ := 4

-- Define the longest line segment in a sector
def longest_segment (d : ℝ) (n : ℕ) : ℝ := d

-- Theorem statement
theorem longest_segment_squared_in_quarter_circle :
  (longest_segment circle_diameter num_sectors)^2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_squared_in_quarter_circle_l1418_141826


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1418_141824

/-- A geometric sequence is defined by its first term and common ratio. -/
def GeometricSequence (a : ℚ) (r : ℚ) : ℕ → ℚ := fun n => a * r^(n - 1)

/-- The common ratio of a geometric sequence. -/
def CommonRatio (seq : ℕ → ℚ) : ℚ := seq 2 / seq 1

theorem geometric_sequence_common_ratio :
  let seq := GeometricSequence 16 (-3/2)
  (seq 1 = 16) ∧ (seq 2 = -24) ∧ (seq 3 = 36) ∧ (seq 4 = -54) →
  CommonRatio seq = -3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1418_141824


namespace NUMINAMATH_CALUDE_billy_has_ten_fish_l1418_141890

/-- The number of fish each person has -/
structure FishCounts where
  billy : ℕ
  tony : ℕ
  sarah : ℕ
  bobby : ℕ

/-- The conditions of the fish distribution -/
def validFishCounts (fc : FishCounts) : Prop :=
  fc.tony = 3 * fc.billy ∧
  fc.sarah = fc.tony + 5 ∧
  fc.bobby = 2 * fc.sarah ∧
  fc.billy + fc.tony + fc.sarah + fc.bobby = 145

theorem billy_has_ten_fish :
  ∃ (fc : FishCounts), validFishCounts fc ∧ fc.billy = 10 := by
  sorry

end NUMINAMATH_CALUDE_billy_has_ten_fish_l1418_141890


namespace NUMINAMATH_CALUDE_telescope_visual_range_l1418_141882

/-- Given a telescope that increases the visual range by 66.67% to reach 150 kilometers,
    prove that the initial visual range without the telescope is 90 kilometers. -/
theorem telescope_visual_range (initial_range : ℝ) : 
  (initial_range + initial_range * (2/3) = 150) → initial_range = 90 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l1418_141882


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1418_141837

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (pos_p : p > 0) (pos_q : q > 0) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (p_neq_q : p ≠ q)
  (geom_seq : a^2 = p * q)
  (arith_seq : b + c = p + q) :
  ∀ x : ℝ, b * x^2 - 2 * a * x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1418_141837


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1418_141862

theorem root_sum_theorem (m n : ℝ) : 
  (∀ x, x^2 - (m+n)*x + m*n = 0 ↔ x = m ∨ x = n) → 
  m = 2*n → 
  m + n = 3*n :=
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1418_141862


namespace NUMINAMATH_CALUDE_sample_size_is_100_l1418_141831

/-- Represents a city with its total sales and number of cars selected for investigation. -/
structure City where
  name : String
  totalSales : Nat
  selected : Nat

/-- Represents the sampling data for the car manufacturer's investigation. -/
def samplingData : List City :=
  [{ name := "A", totalSales := 420, selected := 30 },
   { name := "B", totalSales := 280, selected := 20 },
   { name := "C", totalSales := 700, selected := 50 }]

/-- Checks if the sampling is proportional to the total sales. -/
def isProportionalSampling (data : List City) : Prop :=
  ∀ i j, i ∈ data → j ∈ data → 
    i.totalSales * j.selected = j.totalSales * i.selected

/-- The total sample size is the sum of all selected cars. -/
def totalSampleSize (data : List City) : Nat :=
  (data.map (·.selected)).sum

/-- Theorem stating that the total sample size is 100 given the conditions. -/
theorem sample_size_is_100 (h : isProportionalSampling samplingData) :
  totalSampleSize samplingData = 100 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_100_l1418_141831


namespace NUMINAMATH_CALUDE_probability_red_second_given_white_first_l1418_141876

-- Define the total number of balls and the number of each color
def total_balls : ℕ := 5
def red_balls : ℕ := 3
def white_balls : ℕ := 2

-- Define the probability of drawing a red ball second, given the first is white
def prob_red_second_given_white_first : ℚ := 3 / 4

-- Theorem statement
theorem probability_red_second_given_white_first :
  (red_balls : ℚ) / (total_balls - 1) = prob_red_second_given_white_first :=
by sorry

end NUMINAMATH_CALUDE_probability_red_second_given_white_first_l1418_141876


namespace NUMINAMATH_CALUDE_arithmetic_sequence_l1418_141841

theorem arithmetic_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  a 1 + a 3 + a 5 = 105 →
  a 2 + a 4 + a 6 = 99 →
  a 20 = 1 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_l1418_141841


namespace NUMINAMATH_CALUDE_least_number_of_cubes_l1418_141880

def block_length : ℕ := 15
def block_width : ℕ := 30
def block_height : ℕ := 75

theorem least_number_of_cubes :
  let gcd := Nat.gcd (Nat.gcd block_length block_width) block_height
  let cube_side := gcd
  let num_cubes := (block_length * block_width * block_height) / (cube_side * cube_side * cube_side)
  num_cubes = 10 := by sorry

end NUMINAMATH_CALUDE_least_number_of_cubes_l1418_141880


namespace NUMINAMATH_CALUDE_intersection_equality_l1418_141846

theorem intersection_equality (m : ℝ) : 
  let A : Set ℝ := {0, 1, 2}
  let B : Set ℝ := {1, m}
  A ∩ B = B → m = 0 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_l1418_141846


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1418_141858

theorem equal_roots_quadratic : ∃ (x : ℝ), x^2 - 2*x + 1 = 0 ∧ 
  ∀ (y : ℝ), y^2 - 2*y + 1 = 0 → y = x :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1418_141858


namespace NUMINAMATH_CALUDE_kangaroo_jumps_l1418_141877

theorem kangaroo_jumps (time_for_4_jumps : ℝ) (jumps_to_calculate : ℕ) : 
  time_for_4_jumps = 6 → jumps_to_calculate = 30 → 
  (time_for_4_jumps / 4) * jumps_to_calculate = 45 := by
sorry

end NUMINAMATH_CALUDE_kangaroo_jumps_l1418_141877


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1418_141810

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + 2*x - 3 > 0 ↔ x < -3 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1418_141810


namespace NUMINAMATH_CALUDE_brendans_weekly_yards_l1418_141828

/-- Represents the terrain type --/
inductive Terrain
  | Flat
  | Uneven

/-- Represents the weather condition --/
inductive Weather
  | Normal
  | Rain
  | ExtremeHeat

/-- Represents a day's conditions --/
structure DayCondition where
  terrain : Terrain
  weather : Weather

/-- Calculates the yards cut in a day given the base speed and conditions --/
def yardsCutInDay (baseSpeed : ℝ) (condition : DayCondition) : ℝ :=
  let flatSpeed := baseSpeed * 1.5
  let speed := match condition.terrain with
    | Terrain.Flat => flatSpeed
    | Terrain.Uneven => flatSpeed * 0.65
  match condition.weather with
    | Weather.Normal => speed
    | Weather.Rain => speed * 0.8
    | Weather.ExtremeHeat => speed * 0.9

/-- Calculates the total yards cut in a week --/
def totalYardsInWeek (baseSpeed : ℝ) (weekConditions : List DayCondition) : ℝ :=
  weekConditions.map (yardsCutInDay baseSpeed) |>.sum

/-- The week's conditions --/
def weekConditions : List DayCondition := [
  ⟨Terrain.Flat, Weather.Normal⟩,
  ⟨Terrain.Flat, Weather.Rain⟩,
  ⟨Terrain.Uneven, Weather.Normal⟩,
  ⟨Terrain.Flat, Weather.ExtremeHeat⟩,
  ⟨Terrain.Uneven, Weather.Rain⟩,
  ⟨Terrain.Flat, Weather.Normal⟩,
  ⟨Terrain.Uneven, Weather.ExtremeHeat⟩
]

/-- Theorem stating that Brendan's total yards cut in a week equals 65.46 yards --/
theorem brendans_weekly_yards :
  totalYardsInWeek 8 weekConditions = 65.46 := by
  sorry


end NUMINAMATH_CALUDE_brendans_weekly_yards_l1418_141828


namespace NUMINAMATH_CALUDE_determinant_zero_l1418_141891

theorem determinant_zero (θ φ : ℝ) : 
  Matrix.det !![0, Real.cos θ, Real.sin θ; 
                -Real.cos θ, 0, Real.cos φ; 
                -Real.sin θ, -Real.cos φ, 0] = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_zero_l1418_141891


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l1418_141879

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the total amount of money Sasha has in dollars -/
def total_money : ℚ := 4.80

/-- 
Given that Sasha has $4.80 in U.S. coins and three times as many nickels as quarters,
prove that the maximum number of quarters she could have is 12.
-/
theorem max_quarters_sasha : 
  ∃ (q : ℕ), q ≤ 12 ∧ 
  q * quarter_value + 3 * q * nickel_value = total_money ∧
  ∀ (n : ℕ), n * quarter_value + 3 * n * nickel_value = total_money → n ≤ q :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l1418_141879


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1418_141844

theorem inverse_proportion_ratio (c₁ c₂ d₁ d₂ : ℝ) : 
  c₁ ≠ 0 → c₂ ≠ 0 → d₁ ≠ 0 → d₂ ≠ 0 →
  (∃ k : ℝ, ∀ c d, c * d = k) →
  c₁ * d₁ = c₂ * d₂ →
  c₁ / c₂ = 3 / 4 →
  d₁ / d₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1418_141844


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l1418_141894

def polynomial_terms (n : ℕ) : ℕ := Nat.choose (n + 4 - 1) (4 - 1)

theorem simplified_expression_terms :
  polynomial_terms 5 = 56 := by sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l1418_141894


namespace NUMINAMATH_CALUDE_wilsons_theorem_l1418_141888

theorem wilsons_theorem (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ (Nat.factorial (n - 1) % n = n - 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l1418_141888


namespace NUMINAMATH_CALUDE_tangent_line_properties_l1418_141801

open Real

theorem tangent_line_properties (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₁ ≠ 1) :
  (∃ (k b : ℝ), 
    (∀ x, k * x + b = (1 / x₁) * x - 1 + log x₁) ∧
    (∀ x, k * x + b = exp x₂ * x + exp x₂ * (1 - x₂))) →
  (x₁ * exp x₂ = 1 ∧ (x₁ + 1) / (x₁ - 1) + x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l1418_141801


namespace NUMINAMATH_CALUDE_village_population_equality_l1418_141834

/-- The initial population of Village 1 -/
def initial_population_village1 : ℕ := 68000

/-- The yearly decrease in population of Village 1 -/
def yearly_decrease_village1 : ℕ := 1200

/-- The initial population of Village 2 -/
def initial_population_village2 : ℕ := 42000

/-- The yearly increase in population of Village 2 -/
def yearly_increase_village2 : ℕ := 800

/-- The number of years after which the populations are equal -/
def years_until_equal : ℕ := 13

theorem village_population_equality :
  initial_population_village1 - yearly_decrease_village1 * years_until_equal =
  initial_population_village2 + yearly_increase_village2 * years_until_equal :=
by sorry

end NUMINAMATH_CALUDE_village_population_equality_l1418_141834


namespace NUMINAMATH_CALUDE_a_2k_minus_1_has_three_prime_factors_l1418_141807

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 6
  | (n + 2) => 4 * a (n + 1) - a n + 2

theorem a_2k_minus_1_has_three_prime_factors (k : ℕ) (h : k > 3) :
  ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  (p * q * r : ℤ) ∣ a (2^k - 1) :=
sorry

end NUMINAMATH_CALUDE_a_2k_minus_1_has_three_prime_factors_l1418_141807


namespace NUMINAMATH_CALUDE_chelsea_needs_52_bullseyes_l1418_141889

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : Nat
  chelsea_lead : Nat
  bullseye_score : Nat
  chelsea_min_score : Nat
  opponent_max_score : Nat

/-- Calculates the minimum number of consecutive bullseyes needed for Chelsea to win -/
def min_bullseyes_needed (comp : ArcheryCompetition) : Nat :=
  let remaining_shots := comp.total_shots / 2
  let chelsea_score := remaining_shots * comp.chelsea_min_score + comp.chelsea_lead
  let opponent_max := remaining_shots * comp.opponent_max_score
  let score_diff := opponent_max - chelsea_score
  (score_diff + comp.bullseye_score - comp.chelsea_min_score - 1) / (comp.bullseye_score - comp.chelsea_min_score) + 1

/-- The main theorem stating that 52 consecutive bullseyes are needed for Chelsea to win -/
theorem chelsea_needs_52_bullseyes :
  let comp := ArcheryCompetition.mk 120 60 10 3 10
  min_bullseyes_needed comp = 52 := by
  sorry

end NUMINAMATH_CALUDE_chelsea_needs_52_bullseyes_l1418_141889


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_example_l1418_141823

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ aₙ d : ℤ) : ℤ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence with first term 2, last term 2017, 
    and common difference 5 has 404 terms -/
theorem arithmetic_sequence_length_example : 
  arithmeticSequenceLength 2 2017 5 = 404 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_example_l1418_141823


namespace NUMINAMATH_CALUDE_clock_shows_ten_to_five_l1418_141816

/-- Represents a clock hand --/
inductive ClockHand
  | A
  | B
  | C

/-- Represents the position of a clock hand --/
inductive HandPosition
  | ExactHourMark
  | SlightlyOffHourMark

/-- Represents a clock with three hands --/
structure Clock :=
  (hands : Fin 3 → ClockHand)
  (positions : ClockHand → HandPosition)

/-- The time shown on the clock --/
structure Time :=
  (hours : Nat)
  (minutes : Nat)

/-- Checks if the given clock configuration is valid --/
def isValidClock (c : Clock) : Prop :=
  ∃ (h1 h2 : ClockHand), h1 ≠ h2 ∧ 
    c.positions h1 = HandPosition.ExactHourMark ∧
    c.positions h2 = HandPosition.ExactHourMark ∧
    (∀ h, h ≠ h1 → h ≠ h2 → c.positions h = HandPosition.SlightlyOffHourMark)

/-- The main theorem --/
theorem clock_shows_ten_to_five (c : Clock) : 
  isValidClock c → ∃ (t : Time), t.hours = 4 ∧ t.minutes = 50 :=
sorry

end NUMINAMATH_CALUDE_clock_shows_ten_to_five_l1418_141816


namespace NUMINAMATH_CALUDE_peter_investment_duration_l1418_141813

/-- Calculates the final amount after simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + (principal * rate * time)

theorem peter_investment_duration :
  ∀ (rate : ℝ),
  rate > 0 →
  simple_interest 650 rate 3 = 815 →
  simple_interest 650 rate 4 = 870 →
  ∃ (t : ℝ), t = 3 ∧ simple_interest 650 rate t = 815 :=
by sorry

end NUMINAMATH_CALUDE_peter_investment_duration_l1418_141813


namespace NUMINAMATH_CALUDE_percentage_of_female_brunettes_l1418_141820

theorem percentage_of_female_brunettes 
  (total_students : ℕ) 
  (female_percentage : ℚ)
  (short_brunette_percentage : ℚ)
  (short_brunette_count : ℕ) :
  total_students = 200 →
  female_percentage = 3/5 →
  short_brunette_percentage = 1/2 →
  short_brunette_count = 30 →
  (short_brunette_count : ℚ) / (short_brunette_percentage * (female_percentage * total_students)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_female_brunettes_l1418_141820


namespace NUMINAMATH_CALUDE_camel_cost_calculation_l1418_141865

/-- The cost of a camel in rupees -/
def camel_cost : ℝ := 4184.62

/-- The cost of a horse in rupees -/
def horse_cost : ℝ := 1743.59

/-- The cost of an ox in rupees -/
def ox_cost : ℝ := 11333.33

/-- The cost of an elephant in rupees -/
def elephant_cost : ℝ := 17000

theorem camel_cost_calculation :
  (10 * camel_cost = 24 * horse_cost) ∧
  (26 * horse_cost = 4 * ox_cost) ∧
  (6 * ox_cost = 4 * elephant_cost) ∧
  (10 * elephant_cost = 170000) →
  camel_cost = 4184.62 := by
sorry

#eval camel_cost

end NUMINAMATH_CALUDE_camel_cost_calculation_l1418_141865


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1418_141849

/-- Two-dimensional vector type -/
def Vector2D := ℝ × ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Perpendicularity of two 2D vectors -/
def perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors (k : ℝ) :
  let a : Vector2D := (2, 1)
  let b : Vector2D := (-1, k)
  perpendicular a b → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1418_141849


namespace NUMINAMATH_CALUDE_flour_calculation_l1418_141861

/-- Given a recipe for cookies, calculate the amount of each type of flour needed when doubling the recipe and using two types of flour. -/
theorem flour_calculation (original_cookies : ℕ) (original_flour : ℚ) (new_cookies : ℕ) :
  original_cookies > 0 →
  original_flour > 0 →
  new_cookies = 2 * original_cookies →
  ∃ (flour_each : ℚ),
    flour_each = original_flour ∧
    flour_each * 2 = new_cookies / original_cookies * original_flour :=
by sorry

end NUMINAMATH_CALUDE_flour_calculation_l1418_141861


namespace NUMINAMATH_CALUDE_sum_reciprocals_inequality_l1418_141852

theorem sum_reciprocals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_inequality_l1418_141852


namespace NUMINAMATH_CALUDE_larger_number_proof_l1418_141829

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1345)
  (h2 : L = 6 * S + 15) : 
  L = 1611 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1418_141829


namespace NUMINAMATH_CALUDE_angle_E_is_180_l1418_141854

/-- A quadrilateral with specific angle relationships -/
structure SpecialQuadrilateral where
  E : ℝ  -- Angle E in degrees
  F : ℝ  -- Angle F in degrees
  G : ℝ  -- Angle G in degrees
  H : ℝ  -- Angle H in degrees
  angle_sum : E + F + G + H = 360  -- Sum of angles in a quadrilateral
  E_F_relation : E = 3 * F  -- Relationship between E and F
  E_G_relation : E = 2 * G  -- Relationship between E and G
  E_H_relation : E = 6 * H  -- Relationship between E and H

/-- The measure of angle E in the special quadrilateral is 180 degrees -/
theorem angle_E_is_180 (q : SpecialQuadrilateral) : q.E = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_E_is_180_l1418_141854


namespace NUMINAMATH_CALUDE_a_range_l1418_141840

-- Define propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 1 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → a^2 - 3*a - x + 1 ≤ 0

-- Define the range of a
def range_of_a : Set ℝ := Set.Icc 1 2 ∩ Set.Ioi 1

-- Theorem statement
theorem a_range (a : ℝ) : 
  (¬(p a ∧ q a)) ∧ (¬¬(q a)) → a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_a_range_l1418_141840


namespace NUMINAMATH_CALUDE_unique_correct_expression_l1418_141881

theorem unique_correct_expression :
  ((-3 - 1 = -2) = False) ∧
  ((-2 * (-1/2) = 1) = True) ∧
  ((16 / (-4/3) = 12) = False) ∧
  ((-3^2 / 4 = 9/4) = False) := by
  sorry

end NUMINAMATH_CALUDE_unique_correct_expression_l1418_141881


namespace NUMINAMATH_CALUDE_grandfather_gift_problem_l1418_141835

theorem grandfather_gift_problem (x y : ℕ) : 
  x + y = 30 → 
  5 * x * (x + 1) + 5 * y * (y + 1) = 2410 → 
  (x = 16 ∧ y = 14) ∨ (x = 14 ∧ y = 16) := by
sorry

end NUMINAMATH_CALUDE_grandfather_gift_problem_l1418_141835


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l1418_141818

/-- In an isosceles triangle DEF where angle D is congruent to angle E, 
    and the measure of angle E is three times the measure of angle F, 
    the measure of angle D is 540/7 degrees. -/
theorem isosceles_triangle_angle_measure (D E F : ℝ) : 
  D = E →                         -- Angle D is congruent to angle E
  E = 3 * F →                     -- Measure of angle E is three times the measure of angle F
  D + E + F = 180 →               -- Sum of angles in a triangle is 180 degrees
  D = 540 / 7 := by sorry         -- Measure of angle D is 540/7 degrees

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l1418_141818


namespace NUMINAMATH_CALUDE_exists_ring_configuration_l1418_141821

/-- A structure representing a configuration of connected rings -/
structure RingConfiguration (n : ℕ) where
  rings : Fin n → Bool
  connected : Bool
  
/-- A function that simulates cutting a ring from the configuration -/
def cut_ring (config : RingConfiguration n) (i : Fin n) : RingConfiguration n :=
  { rings := λ j => if j = i then false else config.rings j,
    connected := false }

/-- The property that a ring configuration satisfies the problem conditions -/
def satisfies_conditions (config : RingConfiguration n) : Prop :=
  (n ≥ 3) ∧
  config.connected ∧
  (∀ i : Fin n, ¬(cut_ring config i).connected)

/-- The main theorem stating that for any number of rings ≥ 3, 
    there exists a configuration satisfying the problem conditions -/
theorem exists_ring_configuration (n : ℕ) (h : n ≥ 3) :
  ∃ (config : RingConfiguration n), satisfies_conditions config :=
sorry

end NUMINAMATH_CALUDE_exists_ring_configuration_l1418_141821


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1418_141875

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, (a + 2 * Complex.I) / (2 - Complex.I) = b * Complex.I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1418_141875


namespace NUMINAMATH_CALUDE_distinct_numbers_probability_l1418_141897

def num_sides : ℕ := 5
def num_dice : ℕ := 5

theorem distinct_numbers_probability :
  (Nat.factorial num_dice : ℚ) / (num_sides ^ num_dice : ℚ) = 120 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_distinct_numbers_probability_l1418_141897


namespace NUMINAMATH_CALUDE_total_new_games_is_92_l1418_141887

/-- The number of new games Katie has -/
def katie_new_games : ℕ := 84

/-- The number of new games Katie's friends have -/
def friends_new_games : ℕ := 8

/-- The total number of new games Katie and her friends have together -/
def total_new_games : ℕ := katie_new_games + friends_new_games

/-- Theorem stating that the total number of new games is 92 -/
theorem total_new_games_is_92 : total_new_games = 92 := by sorry

end NUMINAMATH_CALUDE_total_new_games_is_92_l1418_141887


namespace NUMINAMATH_CALUDE_series_sum_implies_k_l1418_141809

theorem series_sum_implies_k (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 2) / k^n = 17/2) : k = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_implies_k_l1418_141809


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1418_141805

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℚ
  d : ℚ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1 : ℚ) * seq.d

/-- The sum of the first n terms of an arithmetic sequence -/
def ArithmeticSequence.sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * seq.a₁ + (n - 1 : ℚ) * seq.d)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  seq.nthTerm 5 = 1 → seq.nthTerm 17 = 18 → seq.sumFirstN 12 = 75/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1418_141805


namespace NUMINAMATH_CALUDE_set_equivalence_l1418_141848

theorem set_equivalence : {x : ℕ | x < 5} = {0, 1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_equivalence_l1418_141848


namespace NUMINAMATH_CALUDE_add_2023_minutes_to_midnight_l1418_141851

/-- Represents a time with day, hour, and minute components -/
structure DateTime where
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime and returns the resulting DateTime -/
def addMinutes (start : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting DateTime (midnight on December 31, 2020) -/
def startTime : DateTime :=
  { day := 0, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 2023

/-- The expected result DateTime (January 1 at 9:43 AM) -/
def expectedResult : DateTime :=
  { day := 1, hour := 9, minute := 43 }

/-- Theorem stating that adding 2023 minutes to midnight on December 31, 2020,
    results in January 1 at 9:43 AM -/
theorem add_2023_minutes_to_midnight :
  addMinutes startTime minutesToAdd = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_add_2023_minutes_to_midnight_l1418_141851


namespace NUMINAMATH_CALUDE_bookshelf_problem_l1418_141892

theorem bookshelf_problem (x : ℕ) 
  (h1 : (4 * x : ℚ) / (5 * x + 35 + 6 * x + 4 * x) = 22 / 100) : 
  4 * x = 44 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_problem_l1418_141892


namespace NUMINAMATH_CALUDE_remaining_reading_time_l1418_141884

/-- Calculates the remaining reading time for Sunday given the total assigned time and time spent reading on Friday and Saturday. -/
theorem remaining_reading_time 
  (total_assigned : ℕ) 
  (friday_reading : ℕ) 
  (saturday_reading : ℕ) 
  (h1 : total_assigned = 60)  -- 1 hour = 60 minutes
  (h2 : friday_reading = 16)
  (h3 : saturday_reading = 28) :
  total_assigned - (friday_reading + saturday_reading) = 16 :=
by sorry

#check remaining_reading_time

end NUMINAMATH_CALUDE_remaining_reading_time_l1418_141884


namespace NUMINAMATH_CALUDE_remainder_theorem_l1418_141857

theorem remainder_theorem (m : ℤ) (k : ℤ) : 
  m = 40 * k - 1 → (m^2 + 3*m + 5) % 40 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1418_141857


namespace NUMINAMATH_CALUDE_bobby_shoes_count_l1418_141832

theorem bobby_shoes_count (bonny_shoes becky_shoes bobby_shoes : ℕ) : 
  bonny_shoes = 13 →
  bonny_shoes = 2 * becky_shoes - 5 →
  bobby_shoes = 3 * becky_shoes →
  bobby_shoes = 27 := by
sorry

end NUMINAMATH_CALUDE_bobby_shoes_count_l1418_141832


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_parameters_l1418_141850

/-- Represents a hyperbola with center at the origin and foci on the x-axis -/
structure Hyperbola where
  focal_width : ℝ
  eccentricity : ℝ

/-- The equation of a hyperbola given its parameters -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- Theorem stating that a hyperbola with given parameters has the specified equation -/
theorem hyperbola_equation_from_parameters (h : Hyperbola) 
  (hw : h.focal_width = 8) 
  (he : h.eccentricity = 2) : 
  ∀ x y : ℝ, hyperbola_equation h x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_parameters_l1418_141850


namespace NUMINAMATH_CALUDE_f_properties_l1418_141855

noncomputable def f (x : ℝ) := Real.log (|x| + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (f 0 = 0 ∧ ∀ x : ℝ, f x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1418_141855


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l1418_141883

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 6000 → 
  candidate_percentage = 35/100 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 1800 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l1418_141883


namespace NUMINAMATH_CALUDE_number_of_parents_attending_l1418_141830

/-- The number of parents attending a school meeting -/
theorem number_of_parents_attending (S R B N : ℕ) : 
  S = 25 →  -- number of parents volunteering to supervise
  B = 11 →  -- number of parents volunteering for both supervising and bringing refreshments
  R = 42 →  -- number of parents volunteering to bring refreshments
  R = (3 * N) / 2 →  -- R is 1.5 times N
  S + R - B + N = 95 :=  -- total number of parents
by sorry

end NUMINAMATH_CALUDE_number_of_parents_attending_l1418_141830


namespace NUMINAMATH_CALUDE_lcm_18_20_l1418_141864

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_20_l1418_141864


namespace NUMINAMATH_CALUDE_football_banquet_min_guests_l1418_141859

/-- The minimum number of guests at a banquet given the total food consumed and maximum food per guest -/
def min_guests (total_food : ℕ) (max_food_per_guest : ℕ) : ℕ :=
  (total_food + max_food_per_guest - 1) / max_food_per_guest

/-- Theorem stating the minimum number of guests at the football banquet -/
theorem football_banquet_min_guests :
  min_guests 319 2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_football_banquet_min_guests_l1418_141859


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l1418_141845

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_condition
  (a : ℕ → ℝ) (m p q : ℕ) (h : arithmetic_sequence a) :
  (∀ m p q : ℕ, p + q = 2 * m → a p + a q = 2 * a m) ∧
  (∃ m p q : ℕ, a p + a q = 2 * a m ∧ p + q ≠ 2 * m) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l1418_141845


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l1418_141819

def A : Nat := 123456
def B : Nat := 162738
def M : Nat := 1000000
def N : Nat := 503339

theorem multiplicative_inverse_modulo :
  (A * B * N) % M = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l1418_141819


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l1418_141838

theorem consecutive_integers_product (n : ℤ) : 
  n * (n + 1) * (n + 2) * (n + 3) = 2520 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l1418_141838


namespace NUMINAMATH_CALUDE_first_digit_base7_528_l1418_141822

/-- The first digit of the base 7 representation of a natural number -/
def firstDigitBase7 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := (Nat.log n 7).succ
    (n / 7^(k-1)) % 7

theorem first_digit_base7_528 :
  firstDigitBase7 528 = 1 := by sorry

end NUMINAMATH_CALUDE_first_digit_base7_528_l1418_141822


namespace NUMINAMATH_CALUDE_flower_cost_minimization_l1418_141863

/-- The cost of flowers given the number of carnations -/
def cost (x : ℕ) : ℕ := 55 - x

/-- The problem statement -/
theorem flower_cost_minimization :
  let total_flowers : ℕ := 11
  let min_lilies : ℕ := 2
  let carnation_cost : ℕ := 4
  let lily_cost : ℕ := 5
  (2 * lily_cost + carnation_cost = 14) →
  (3 * carnation_cost = 2 * lily_cost + 2) →
  (∀ x : ℕ, x ≤ total_flowers - min_lilies → cost x = 55 - x) →
  (∃ x : ℕ, x ≤ total_flowers - min_lilies ∧ cost x = 46 ∧ 
    ∀ y : ℕ, y ≤ total_flowers - min_lilies → cost y ≥ cost x) := by
  sorry

end NUMINAMATH_CALUDE_flower_cost_minimization_l1418_141863


namespace NUMINAMATH_CALUDE_mikes_remaining_nickels_l1418_141811

/-- Given Mike's initial number of nickels and the number borrowed by his dad,
    proves that the number of nickels Mike has now is the difference between
    the initial number and the borrowed number. -/
theorem mikes_remaining_nickels
  (initial_nickels : ℕ)
  (borrowed_nickels : ℕ)
  (h1 : initial_nickels = 87)
  (h2 : borrowed_nickels = 75)
  : initial_nickels - borrowed_nickels = 12 := by
  sorry

end NUMINAMATH_CALUDE_mikes_remaining_nickels_l1418_141811


namespace NUMINAMATH_CALUDE_spilled_bag_candies_l1418_141878

theorem spilled_bag_candies (bags : ℕ) (average : ℕ) (known_bags : List ℕ) : 
  bags = 8 → 
  average = 22 → 
  known_bags = [12, 14, 18, 22, 24, 26, 29] → 
  (List.sum known_bags + (bags - known_bags.length) * average - List.sum known_bags) = 31 := by
  sorry

end NUMINAMATH_CALUDE_spilled_bag_candies_l1418_141878


namespace NUMINAMATH_CALUDE_nickel_difference_l1418_141817

/-- Given that Alice has 3p + 2 nickels and Bob has 2p + 6 nickels,
    the difference in their money in pennies is 5p - 20 --/
theorem nickel_difference (p : ℤ) : 
  let alice_nickels : ℤ := 3 * p + 2
  let bob_nickels : ℤ := 2 * p + 6
  let nickel_value : ℤ := 5  -- value of a nickel in pennies
  5 * p - 20 = nickel_value * (alice_nickels - bob_nickels) :=
by sorry

end NUMINAMATH_CALUDE_nickel_difference_l1418_141817


namespace NUMINAMATH_CALUDE_f_extrema_max_k_bound_l1418_141839

noncomputable section

def f (x : ℝ) : ℝ := x + x * Real.log x

theorem f_extrema :
  (∃ (x_min : ℝ), x_min = Real.exp (-2) ∧
    (∀ x > 0, f x ≥ f x_min) ∧
    f x_min = -Real.exp (-2)) ∧
  (∀ M : ℝ, ∃ x > 0, f x > M) :=
sorry

theorem max_k_bound :
  (∀ k : ℤ, (∀ x > 1, f x > k * (x - 1)) → k ≤ 3) ∧
  (∃ x > 1, f x > 3 * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_max_k_bound_l1418_141839


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l1418_141898

theorem polynomial_division_degree (f q d r : Polynomial ℝ) : 
  Polynomial.degree f = 17 →
  Polynomial.degree q = 10 →
  r = 5 * X^4 - 3 * X^3 + 2 * X^2 - X + 15 →
  f = d * q + r →
  Polynomial.degree d = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l1418_141898


namespace NUMINAMATH_CALUDE_dividend_calculation_l1418_141866

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 65)
  (h_divisor : divisor = 24)
  (h_remainder : remainder = 5) :
  (divisor * quotient) + remainder = 1565 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1418_141866


namespace NUMINAMATH_CALUDE_intersection_sum_l1418_141872

/-- Given two lines y = 2x + c and y = 4x + d intersecting at (3, 12), prove that c + d = 6 -/
theorem intersection_sum (c d : ℝ) : 
  (2 * 3 + c = 12) → (4 * 3 + d = 12) → c + d = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1418_141872


namespace NUMINAMATH_CALUDE_sqrt_sum_irrational_l1418_141814

theorem sqrt_sum_irrational (n : ℕ+) : Irrational (Real.sqrt (n + 1) + Real.sqrt n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_irrational_l1418_141814


namespace NUMINAMATH_CALUDE_mcgillicuddy_kindergarten_count_l1418_141827

/-- Calculates the total number of students present in two kindergarten sessions -/
def total_students (morning_registered : ℕ) (morning_absent : ℕ) 
                   (afternoon_registered : ℕ) (afternoon_absent : ℕ) : ℕ :=
  (morning_registered - morning_absent) + (afternoon_registered - afternoon_absent)

/-- Theorem stating that the total number of students is 42 given the specified conditions -/
theorem mcgillicuddy_kindergarten_count : 
  total_students 25 3 24 4 = 42 := by
  sorry

#eval total_students 25 3 24 4

end NUMINAMATH_CALUDE_mcgillicuddy_kindergarten_count_l1418_141827


namespace NUMINAMATH_CALUDE_calculation_proof_l1418_141870

theorem calculation_proof : 3^2 + Real.sqrt 25 - (64 : ℝ)^(1/3) + abs (-9) = 19 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1418_141870


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_a_minus_b_l1418_141833

theorem quadratic_solution_implies_a_minus_b (a b : ℝ) : 
  (4^2 + 4*a - 4*b = 0) → (a - b = -4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_a_minus_b_l1418_141833


namespace NUMINAMATH_CALUDE_perpendicular_line_theorem_l1418_141853

structure Plane where
  -- Define a plane structure

structure Point where
  -- Define a point structure

structure Line where
  -- Define a line structure

-- Define perpendicularity between planes
def perpendicular (α β : Plane) : Prop := sorry

-- Define a point lying in a plane
def lies_in (A : Point) (α : Plane) : Prop := sorry

-- Define a line passing through a point
def passes_through (l : Line) (A : Point) : Prop := sorry

-- Define a line perpendicular to a plane
def perpendicular_to_plane (l : Line) (β : Plane) : Prop := sorry

-- Define a line lying in a plane
def line_in_plane (l : Line) (α : Plane) : Prop := sorry

theorem perpendicular_line_theorem (α β : Plane) (A : Point) :
  perpendicular α β →
  lies_in A α →
  ∃! l : Line, passes_through l A ∧ perpendicular_to_plane l β ∧ line_in_plane l α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_theorem_l1418_141853


namespace NUMINAMATH_CALUDE_students_facing_teacher_l1418_141873

theorem students_facing_teacher (n : ℕ) (h : n = 50) : 
  n - (n / 3 + n / 7 - n / 21) = 31 :=
sorry

end NUMINAMATH_CALUDE_students_facing_teacher_l1418_141873


namespace NUMINAMATH_CALUDE_marble_weight_calculation_l1418_141825

/-- Given two pieces of marble of equal weight and a third piece,
    if the total weight is 0.75 tons and the third piece weighs 0.08333333333333333 ton,
    then the weight of each of the first two pieces is 0.33333333333333335 ton. -/
theorem marble_weight_calculation (w : ℝ) : 
  2 * w + 0.08333333333333333 = 0.75 → w = 0.33333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_calculation_l1418_141825


namespace NUMINAMATH_CALUDE_spencer_total_distance_l1418_141895

/-- The total distance Spencer walked throughout the day -/
def total_distance (d1 d2 d3 d4 d5 d6 d7 : ℝ) : ℝ :=
  d1 + d2 + d3 + d4 + d5 + d6 + d7

/-- Theorem: Given Spencer's walking distances, the total distance is 8.6 miles -/
theorem spencer_total_distance :
  total_distance 1.2 0.6 0.9 1.7 2.1 1.3 0.8 = 8.6 := by
  sorry

end NUMINAMATH_CALUDE_spencer_total_distance_l1418_141895


namespace NUMINAMATH_CALUDE_yard_length_is_250_l1418_141815

/-- The length of a yard with trees planted at equal distances -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of the yard is 250 meters -/
theorem yard_length_is_250 :
  yard_length 51 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_is_250_l1418_141815


namespace NUMINAMATH_CALUDE_fourth_root_sum_equals_expression_l1418_141856

theorem fourth_root_sum_equals_expression : 
  (1 + Real.sqrt 2 + Real.sqrt 3)^4 = 
    Real.sqrt 6400 + Real.sqrt 6144 + Real.sqrt 4800 + Real.sqrt 4608 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sum_equals_expression_l1418_141856


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1418_141871

theorem circle_equation_proof (x y : ℝ) : 
  let center := (1, -2)
  let radius := Real.sqrt 2
  let circle_eq := (x - 1)^2 + (y + 2)^2 = 2
  let center_line_eq := -2 * center.1 = center.2
  let tangent_line_eq := x + y = 1
  let tangent_point := (2, -1)
  (
    center_line_eq ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ circle_eq ∧
    (center.1 - tangent_point.1)^2 + (center.2 - tangent_point.2)^2 = radius^2 ∧
    (tangent_point.1 + tangent_point.2 = 1)
  ) := by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1418_141871


namespace NUMINAMATH_CALUDE_vasya_can_win_l1418_141874

/-- Represents the state of the water pots -/
structure PotState :=
  (pot3 : Nat)
  (pot5 : Nat)
  (pot7 : Nat)

/-- Represents a move by Vasya -/
inductive VasyaMove
  | FillPot3
  | FillPot5
  | FillPot7
  | TransferPot3ToPot5
  | TransferPot3ToPot7
  | TransferPot5ToPot3
  | TransferPot5ToPot7
  | TransferPot7ToPot3
  | TransferPot7ToPot5

/-- Represents a move by Dima -/
inductive DimaMove
  | EmptyPot3
  | EmptyPot5
  | EmptyPot7

/-- Applies Vasya's move to the current state -/
def applyVasyaMove (state : PotState) (move : VasyaMove) : PotState :=
  sorry

/-- Applies Dima's move to the current state -/
def applyDimaMove (state : PotState) (move : DimaMove) : PotState :=
  sorry

/-- Checks if the game is won (1 liter in any pot) -/
def isGameWon (state : PotState) : Bool :=
  state.pot3 = 1 || state.pot5 = 1 || state.pot7 = 1

/-- Theorem: Vasya can win the game -/
theorem vasya_can_win :
  ∃ (moves : List (VasyaMove × VasyaMove)),
    ∀ (dimaMoves : List DimaMove),
      let finalState := (moves.zip dimaMoves).foldl
        (fun state (vasyaMoves, dimaMove) =>
          let s1 := applyVasyaMove state vasyaMoves.1
          let s2 := applyVasyaMove s1 vasyaMoves.2
          applyDimaMove s2 dimaMove)
        { pot3 := 0, pot5 := 0, pot7 := 0 }
      isGameWon finalState :=
by
  sorry


end NUMINAMATH_CALUDE_vasya_can_win_l1418_141874


namespace NUMINAMATH_CALUDE_clare_remaining_money_l1418_141804

/-- Calculates the remaining money after Clare's purchases -/
def remaining_money (initial_amount bread_price milk_price cereal_price apple_price : ℕ) : ℕ :=
  let bread_cost := 4 * bread_price
  let milk_cost := 2 * milk_price
  let cereal_cost := 3 * cereal_price
  let apple_cost := apple_price
  let total_cost := bread_cost + milk_cost + cereal_cost + apple_cost
  initial_amount - total_cost

/-- Proves that Clare has $22 left after her purchases -/
theorem clare_remaining_money :
  remaining_money 47 2 2 3 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_clare_remaining_money_l1418_141804


namespace NUMINAMATH_CALUDE_second_project_length_l1418_141886

/-- Represents a digging project with depth, length, breadth, and duration -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ
  duration : ℝ

/-- Calculates the volume of earth dug in a project -/
def volume (p : DiggingProject) : ℝ := p.depth * p.length * p.breadth

/-- The first digging project -/
def project1 : DiggingProject := {
  depth := 100,
  length := 25,
  breadth := 30,
  duration := 12
}

/-- The second digging project with unknown length -/
def project2 (l : ℝ) : DiggingProject := {
  depth := 75,
  length := l,
  breadth := 50,
  duration := 12
}

/-- Main theorem: The length of the second digging project is 20 meters -/
theorem second_project_length :
  ∃ l : ℝ, l = 20 ∧ volume project1 = volume (project2 l) := by
  sorry

#check second_project_length

end NUMINAMATH_CALUDE_second_project_length_l1418_141886


namespace NUMINAMATH_CALUDE_total_nails_needed_l1418_141803

def nails_per_plank : ℕ := 2
def number_of_planks : ℕ := 16

theorem total_nails_needed : nails_per_plank * number_of_planks = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_needed_l1418_141803


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1418_141812

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem compound_interest_problem :
  let principal : ℝ := 3600
  let rate : ℝ := 0.05
  let time : ℕ := 2
  let final_amount : ℝ := 3969
  compound_interest principal rate time = final_amount := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1418_141812


namespace NUMINAMATH_CALUDE_unique_number_l1418_141847

theorem unique_number : ∃! x : ℝ, x / 4 = x - 6 := by sorry

end NUMINAMATH_CALUDE_unique_number_l1418_141847


namespace NUMINAMATH_CALUDE_quadratic_no_solution_b_range_l1418_141806

theorem quadratic_no_solution_b_range (b : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + 1 > 0) → -2 < b ∧ b < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_b_range_l1418_141806


namespace NUMINAMATH_CALUDE_smallest_x_value_l1418_141808

theorem smallest_x_value (x y : ℕ+) (h : (4 : ℚ) / 5 = y / (200 + x)) : 
  ∀ z : ℕ+, (4 : ℚ) / 5 = (y : ℚ) / (200 + z) → x ≤ z :=
by sorry

#check smallest_x_value

end NUMINAMATH_CALUDE_smallest_x_value_l1418_141808


namespace NUMINAMATH_CALUDE_product_of_x_and_y_l1418_141896

theorem product_of_x_and_y (x y : ℝ) : 
  3 * x + 4 * y = 60 → 6 * x - 4 * y = 12 → x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_x_and_y_l1418_141896


namespace NUMINAMATH_CALUDE_student_number_proof_l1418_141802

theorem student_number_proof : 
  ∃ x : ℝ, (2 * x - 138 = 102) ∧ (x = 120) := by
  sorry

end NUMINAMATH_CALUDE_student_number_proof_l1418_141802


namespace NUMINAMATH_CALUDE_popsicle_bottle_cost_l1418_141800

/-- Represents the cost of popsicle supplies and production -/
structure PopsicleSupplies where
  total_budget : ℚ
  mold_cost : ℚ
  stick_pack_cost : ℚ
  sticks_per_pack : ℕ
  popsicles_per_bottle : ℕ
  remaining_sticks : ℕ

/-- Calculates the cost of each bottle of juice -/
def bottle_cost (supplies : PopsicleSupplies) : ℚ :=
  let money_for_juice := supplies.total_budget - supplies.mold_cost - supplies.stick_pack_cost
  let used_sticks := supplies.sticks_per_pack - supplies.remaining_sticks
  let bottles_used := used_sticks / supplies.popsicles_per_bottle
  money_for_juice / bottles_used

/-- Theorem stating that given the conditions, the cost of each bottle is $2 -/
theorem popsicle_bottle_cost :
  let supplies := PopsicleSupplies.mk 10 3 1 100 20 40
  bottle_cost supplies = 2 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_bottle_cost_l1418_141800


namespace NUMINAMATH_CALUDE_hyperbola_trajectory_l1418_141868

/-- The trajectory of point P satisfying |PF₂| - |PF₁| = 4, where F₁(-4,0) and F₂(4,0) -/
theorem hyperbola_trajectory (x y : ℝ) : 
  let f₁ : ℝ × ℝ := (-4, 0)
  let f₂ : ℝ × ℝ := (4, 0)
  let p : ℝ × ℝ := (x, y)
  Real.sqrt ((x - 4)^2 + y^2) - Real.sqrt ((x + 4)^2 + y^2) = 4 →
  x^2 / 4 - y^2 / 12 = 1 ∧ x ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_trajectory_l1418_141868


namespace NUMINAMATH_CALUDE_pizza_lovers_count_l1418_141860

theorem pizza_lovers_count (total pupils_like_burgers pupils_like_both : ℕ) 
  (h1 : total = 200)
  (h2 : pupils_like_burgers = 115)
  (h3 : pupils_like_both = 40)
  : ∃ pupils_like_pizza : ℕ, 
    pupils_like_pizza + pupils_like_burgers - pupils_like_both = total ∧ 
    pupils_like_pizza = 125 :=
by sorry

end NUMINAMATH_CALUDE_pizza_lovers_count_l1418_141860


namespace NUMINAMATH_CALUDE_product_of_numbers_l1418_141869

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1418_141869


namespace NUMINAMATH_CALUDE_product_one_minus_reciprocals_l1418_141893

theorem product_one_minus_reciprocals : (1 - 1/3) * (1 - 1/4) * (1 - 1/5) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_product_one_minus_reciprocals_l1418_141893


namespace NUMINAMATH_CALUDE_min_value_7x_5y_min_value_achieved_min_value_is_7_plus_2sqrt6_l1418_141842

theorem min_value_7x_5y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (2 * x + y) + 4 / (x + y) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (2 * a + b) + 4 / (a + b) = 2 → 7 * x + 5 * y ≤ 7 * a + 5 * b :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (2 * x + y) + 4 / (x + y) = 2) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (2 * a + b) + 4 / (a + b) = 2 ∧ 7 * a + 5 * b = 7 + 2 * Real.sqrt 6 :=
by sorry

theorem min_value_is_7_plus_2sqrt6 :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / (2 * x + y) + 4 / (x + y) = 2 ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / (2 * a + b) + 4 / (a + b) = 2 → 7 * x + 5 * y ≤ 7 * a + 5 * b) ∧
  7 * x + 5 * y = 7 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_7x_5y_min_value_achieved_min_value_is_7_plus_2sqrt6_l1418_141842


namespace NUMINAMATH_CALUDE_inequality_proof_l1418_141899

theorem inequality_proof (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1418_141899
