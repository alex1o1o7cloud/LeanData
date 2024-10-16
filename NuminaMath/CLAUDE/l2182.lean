import Mathlib

namespace NUMINAMATH_CALUDE_sock_order_ratio_l2182_218264

/-- Represents the number of pairs of socks --/
structure SockOrder where
  green : ℕ
  red : ℕ

/-- Represents the price of socks --/
structure SockPrice where
  red : ℝ
  green : ℝ

/-- Calculates the total cost of a sock order --/
def totalCost (order : SockOrder) (price : SockPrice) : ℝ :=
  order.green * price.green + order.red * price.red

theorem sock_order_ratio (original : SockOrder) (price : SockPrice) :
  original.green = 6 →
  price.green = 3 * price.red →
  let interchanged : SockOrder := ⟨original.red, original.green⟩
  totalCost interchanged price = 1.2 * totalCost original price →
  2 * original.red = 3 * original.green := by
  sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l2182_218264


namespace NUMINAMATH_CALUDE_alternate_seating_count_l2182_218215

/-- The number of ways to seat 4 boys and 1 girl alternately in a row -/
def alternateSeating : ℕ :=
  let numBoys : ℕ := 4
  let numGirls : ℕ := 1
  let numPositionsForGirl : ℕ := numBoys + 1
  let numArrangementsForBoys : ℕ := Nat.factorial numBoys
  numPositionsForGirl * numArrangementsForBoys

theorem alternate_seating_count : alternateSeating = 120 := by
  sorry

end NUMINAMATH_CALUDE_alternate_seating_count_l2182_218215


namespace NUMINAMATH_CALUDE_data_set_average_l2182_218299

theorem data_set_average (x : ℝ) : 
  (2 + 1 + 4 + x + 6) / 5 = 4 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_data_set_average_l2182_218299


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2182_218216

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Definition of a geometric sequence -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  b ^ 2 = a * c

theorem arithmetic_sequence_general_term 
  (seq : ArithmeticSequence)
  (h3 : seq.a 4 = 10)
  (h4 : IsGeometricSequence (seq.a 3) (seq.a 6) (seq.a 10)) :
  ∃ k : ℝ, ∀ n : ℕ, seq.a n = n + k ∧ k = 6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2182_218216


namespace NUMINAMATH_CALUDE_james_flowers_l2182_218222

/-- The number of flowers planted by James and his friends --/
def flower_planting 
  (james friend_a friend_b friend_c friend_d friend_e friend_f friend_g : ℝ) : Prop :=
  james = friend_a * 1.2
  ∧ friend_a = friend_b * 1.15
  ∧ friend_b = friend_c * 0.7
  ∧ friend_c = friend_d * 1.1
  ∧ friend_d = friend_e * 1.25
  ∧ friend_e = friend_f
  ∧ friend_g = friend_f * 0.7
  ∧ friend_b = 12

/-- The theorem stating James plants 16.56 flowers per day --/
theorem james_flowers 
  (james friend_a friend_b friend_c friend_d friend_e friend_f friend_g : ℝ) 
  (h : flower_planting james friend_a friend_b friend_c friend_d friend_e friend_f friend_g) : 
  james = 16.56 := by
  sorry

end NUMINAMATH_CALUDE_james_flowers_l2182_218222


namespace NUMINAMATH_CALUDE_largest_root_bound_l2182_218224

theorem largest_root_bound (b₀ b₁ b₂ b₃ : ℝ) (h₀ : |b₀| ≤ 1) (h₁ : |b₁| ≤ 1) (h₂ : |b₂| ≤ 1) (h₃ : |b₃| ≤ 1) :
  ∃ r : ℝ, (5/2 < r ∧ r < 3) ∧
    (∀ x : ℝ, x > 0 → x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀ = 0 → x ≤ r) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_bound_l2182_218224


namespace NUMINAMATH_CALUDE_fred_red_marbles_l2182_218210

/-- Represents the number of marbles of each color in Fred's collection. -/
structure MarbleCount where
  total : ℕ
  darkBlue : ℕ
  green : ℕ
  red : ℕ

/-- Conditions for Fred's marble collection. -/
def fredMarbles : MarbleCount where
  total := 63
  darkBlue := 21  -- At least one-third of 63
  green := 4
  red := 38

/-- Theorem stating the number of red marbles in Fred's collection. -/
theorem fred_red_marbles :
  fredMarbles.red = 38 ∧
  fredMarbles.total = 63 ∧
  fredMarbles.darkBlue ≥ fredMarbles.total / 3 ∧
  fredMarbles.green = 4 ∧
  fredMarbles.red = fredMarbles.total - fredMarbles.darkBlue - fredMarbles.green :=
by
  sorry


end NUMINAMATH_CALUDE_fred_red_marbles_l2182_218210


namespace NUMINAMATH_CALUDE_tim_and_donna_dating_years_l2182_218292

/-- Represents the timeline of Tim and Donna's relationship -/
structure Relationship where
  meetYear : ℕ
  weddingYear : ℕ
  anniversaryYear : ℕ
  yearsBetweenMeetingAndDating : ℕ

/-- Calculate the number of years Tim and Donna dated before marriage -/
def yearsDatingBeforeMarriage (r : Relationship) : ℕ :=
  r.weddingYear - r.meetYear - r.yearsBetweenMeetingAndDating

/-- The main theorem stating that Tim and Donna dated for 3 years before marriage -/
theorem tim_and_donna_dating_years (r : Relationship) 
  (h1 : r.meetYear = 2000)
  (h2 : r.anniversaryYear = 2025)
  (h3 : r.anniversaryYear - r.weddingYear = 20)
  (h4 : r.yearsBetweenMeetingAndDating = 2) : 
  yearsDatingBeforeMarriage r = 3 := by
  sorry


end NUMINAMATH_CALUDE_tim_and_donna_dating_years_l2182_218292


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l2182_218241

theorem weight_loss_challenge (initial_loss : ℝ) (measured_loss : ℝ) (clothes_addition : ℝ) :
  initial_loss = 0.14 →
  measured_loss = 0.1228 →
  (1 - measured_loss) * (1 - initial_loss) = 1 + clothes_addition →
  clothes_addition = 0.02 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l2182_218241


namespace NUMINAMATH_CALUDE_tourists_scientific_correct_l2182_218297

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- The number of tourists per year -/
def tourists_per_year : ℕ := 876000

/-- The scientific notation representation of the number of tourists -/
def tourists_scientific : ScientificNotation where
  coefficient := 8.76
  exponent := 5
  one_le_coeff_lt_ten := by sorry

/-- Theorem stating that the scientific notation representation is correct -/
theorem tourists_scientific_correct : 
  (tourists_scientific.coefficient * (10 : ℝ) ^ tourists_scientific.exponent) = tourists_per_year := by sorry

end NUMINAMATH_CALUDE_tourists_scientific_correct_l2182_218297


namespace NUMINAMATH_CALUDE_intersection_complement_A_and_B_l2182_218251

def A : Set ℝ := {x : ℝ | |x| > 1}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem intersection_complement_A_and_B :
  (Aᶜ ∩ B) = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_and_B_l2182_218251


namespace NUMINAMATH_CALUDE_optimal_speed_yihuang_expressway_l2182_218226

/-- The optimal speed problem for the Yihuang Expressway -/
theorem optimal_speed_yihuang_expressway 
  (total_length : ℝ) 
  (min_speed max_speed : ℝ) 
  (fixed_cost : ℝ) 
  (k : ℝ) 
  (max_total_cost : ℝ) :
  total_length = 350 →
  min_speed = 60 →
  max_speed = 120 →
  fixed_cost = 200 →
  k * max_speed^2 + fixed_cost = max_total_cost →
  max_total_cost = 488 →
  ∃ (optimal_speed : ℝ), 
    optimal_speed = 100 ∧
    ∀ (v : ℝ), min_speed ≤ v ∧ v ≤ max_speed →
      total_length * (fixed_cost / v + k * v) ≥ 
      total_length * (fixed_cost / optimal_speed + k * optimal_speed) :=
by sorry

end NUMINAMATH_CALUDE_optimal_speed_yihuang_expressway_l2182_218226


namespace NUMINAMATH_CALUDE_curve_properties_l2182_218263

-- Define the curve
def curve (x y : ℝ) : Prop := x * y = 6

-- Define the property of the tangent being bisected
def tangent_bisected (x y : ℝ) : Prop :=
  ∀ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 →
    (curve x y) →
    (a * y = x * b) →
    ((x - 0) ^ 2 + (y - 0) ^ 2 = (a - x) ^ 2 + (0 - y) ^ 2) ∧
    ((x - 0) ^ 2 + (y - 0) ^ 2 = (0 - x) ^ 2 + (b - y) ^ 2)

theorem curve_properties :
  (curve 2 3) ∧
  (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → curve x y → tangent_bisected x y) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l2182_218263


namespace NUMINAMATH_CALUDE_diophantine_equation_implication_l2182_218234

-- Define the property of not being a perfect square
def NotPerfectSquare (n : ℤ) : Prop := ∀ m : ℤ, n ≠ m^2

-- Define a nontrivial integer solution
def HasNontrivialSolution (f : ℤ → ℤ → ℤ → ℤ) : Prop :=
  ∃ x y z : ℤ, f x y z = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)

-- Define a nontrivial integer solution for 4 variables
def HasNontrivialSolution4 (f : ℤ → ℤ → ℤ → ℤ → ℤ) : Prop :=
  ∃ x y z w : ℤ, f x y z w = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ w ≠ 0)

theorem diophantine_equation_implication (a b : ℤ) 
  (ha : NotPerfectSquare a) (hb : NotPerfectSquare b)
  (h : HasNontrivialSolution4 (fun x y z w => x^2 - a*y^2 - b*z^2 + a*b*w^2)) :
  HasNontrivialSolution (fun x y z => x^2 - a*y^2 - b*z^2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_implication_l2182_218234


namespace NUMINAMATH_CALUDE_greatest_common_divisor_480_90_under_60_l2182_218236

theorem greatest_common_divisor_480_90_under_60 : 
  ∃ n : ℕ, n > 0 ∧ 
    n ∣ 480 ∧ 
    n < 60 ∧ 
    n ∣ 90 ∧ 
    ∀ m : ℕ, m > 0 → m ∣ 480 → m < 60 → m ∣ 90 → m ≤ n :=
by
  use 30
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_480_90_under_60_l2182_218236


namespace NUMINAMATH_CALUDE_magic_act_disappearance_ratio_l2182_218289

theorem magic_act_disappearance_ratio :
  ∀ (total_performances : ℕ) 
    (total_reappearances : ℕ) 
    (double_reappearance_prob : ℚ),
  total_performances = 100 →
  total_reappearances = 110 →
  double_reappearance_prob = 1/5 →
  (total_performances - 
   (total_reappearances - total_performances * double_reappearance_prob)) / 
   total_performances = 1/10 := by
sorry

end NUMINAMATH_CALUDE_magic_act_disappearance_ratio_l2182_218289


namespace NUMINAMATH_CALUDE_unique_equidistant_cell_l2182_218202

-- Define the distance function for cells on an infinite chessboard
def distance (a b : ℤ × ℤ) : ℕ :=
  max (Int.natAbs (a.1 - b.1)) (Int.natAbs (a.2 - b.2))

-- Define the theorem
theorem unique_equidistant_cell
  (A B C : ℤ × ℤ)
  (hab : distance A B = 100)
  (hac : distance A C = 100)
  (hbc : distance B C = 100) :
  ∃! X : ℤ × ℤ, distance X A = 50 ∧ distance X B = 50 ∧ distance X C = 50 :=
sorry

end NUMINAMATH_CALUDE_unique_equidistant_cell_l2182_218202


namespace NUMINAMATH_CALUDE_gcf_32_48_l2182_218258

theorem gcf_32_48 : Nat.gcd 32 48 = 16 := by
  sorry

end NUMINAMATH_CALUDE_gcf_32_48_l2182_218258


namespace NUMINAMATH_CALUDE_valerie_light_bulb_purchase_l2182_218259

/-- Calculates the money left over after buying light bulbs --/
def money_left_over (small_bulbs : ℕ) (large_bulbs : ℕ) (small_cost : ℕ) (large_cost : ℕ) (total_money : ℕ) : ℕ :=
  total_money - (small_bulbs * small_cost + large_bulbs * large_cost)

/-- Theorem: Valerie will have $24 left over after buying light bulbs --/
theorem valerie_light_bulb_purchase :
  money_left_over 3 1 8 12 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_valerie_light_bulb_purchase_l2182_218259


namespace NUMINAMATH_CALUDE_expression_value_l2182_218284

theorem expression_value : (2^1001 + 5^1002)^2 - (2^1001 - 5^1002)^2 = 40 * 10^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2182_218284


namespace NUMINAMATH_CALUDE_square_is_quadratic_l2182_218237

/-- A function f: ℝ → ℝ is quadratic if there exist constants a, b, c with a ≠ 0 such that
    f(x) = a * x^2 + b * x + c for all x -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 is quadratic -/
theorem square_is_quadratic : IsQuadratic (fun x ↦ x^2) := by
  sorry

end NUMINAMATH_CALUDE_square_is_quadratic_l2182_218237


namespace NUMINAMATH_CALUDE_frustum_volume_l2182_218221

/-- Represents a frustum of a cone -/
structure Frustum where
  upper_base_area : ℝ
  lower_base_area : ℝ
  lateral_area : ℝ

/-- Calculate the volume of a frustum -/
def volume (f : Frustum) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem frustum_volume (f : Frustum) 
  (h1 : f.upper_base_area = π)
  (h2 : f.lower_base_area = 4 * π)
  (h3 : f.lateral_area = 6 * π) : 
  volume f = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_l2182_218221


namespace NUMINAMATH_CALUDE_sum_of_possible_e_values_l2182_218232

theorem sum_of_possible_e_values : 
  ∃ (e₁ e₂ : ℝ), (2 * |2 - e₁| = 5) ∧ (2 * |2 - e₂| = 5) ∧ (e₁ + e₂ = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_e_values_l2182_218232


namespace NUMINAMATH_CALUDE_november_rainfall_is_180_inches_l2182_218240

/-- Calculates the total rainfall in November given the conditions -/
def november_rainfall (days_in_november : ℕ) (first_half_days : ℕ) (first_half_daily_rainfall : ℝ) : ℝ :=
  let second_half_days := days_in_november - first_half_days
  let second_half_daily_rainfall := 2 * first_half_daily_rainfall
  let first_half_total := first_half_daily_rainfall * first_half_days
  let second_half_total := second_half_daily_rainfall * second_half_days
  first_half_total + second_half_total

/-- Theorem stating that the total rainfall in November is 180 inches -/
theorem november_rainfall_is_180_inches :
  november_rainfall 30 15 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_november_rainfall_is_180_inches_l2182_218240


namespace NUMINAMATH_CALUDE_cube_sum_zero_l2182_218277

theorem cube_sum_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum_zero : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_zero_l2182_218277


namespace NUMINAMATH_CALUDE_robert_ate_more_chocolates_l2182_218293

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 13

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 4

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_chocolates : chocolate_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_more_chocolates_l2182_218293


namespace NUMINAMATH_CALUDE_moral_education_story_time_l2182_218242

/-- Proves that telling a 7-minute "Moral Education Story" every week for 20 weeks equals 2 hours and 20 minutes -/
theorem moral_education_story_time :
  let story_duration : ℕ := 7  -- Duration of one story in minutes
  let weeks : ℕ := 20  -- Number of weeks
  let total_minutes : ℕ := story_duration * weeks
  let hours : ℕ := total_minutes / 60
  let remaining_minutes : ℕ := total_minutes % 60
  (hours = 2 ∧ remaining_minutes = 20) := by
  sorry


end NUMINAMATH_CALUDE_moral_education_story_time_l2182_218242


namespace NUMINAMATH_CALUDE_school_field_trip_buses_l2182_218214

/-- The number of buses needed for a school field trip --/
def buses_needed (fifth_graders sixth_graders seventh_graders : ℕ)
  (teachers_per_grade parents_per_grade : ℕ)
  (bus_capacity : ℕ) : ℕ :=
  let total_students := fifth_graders + sixth_graders + seventh_graders
  let total_chaperones := (teachers_per_grade + parents_per_grade) * 3
  let total_people := total_students + total_chaperones
  (total_people + bus_capacity - 1) / bus_capacity

theorem school_field_trip_buses :
  buses_needed 109 115 118 4 2 72 = 5 := by
  sorry

end NUMINAMATH_CALUDE_school_field_trip_buses_l2182_218214


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l2182_218290

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y : ℤ), x^2 - 7*y = 10 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l2182_218290


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2182_218261

theorem sum_of_reciprocals (x y : ℚ) :
  (1 / x + 1 / y = 4) → (1 / x - 1 / y = -6) → (x + y = -4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2182_218261


namespace NUMINAMATH_CALUDE_fountain_length_is_105_l2182_218275

/-- Represents the water fountain construction scenario -/
structure FountainConstruction where
  initialMen : ℕ := 20
  initialLength : ℕ := 56
  initialDays : ℕ := 7
  wallDays : ℕ := 3
  newMen : ℕ := 35
  totalDays : ℕ := 9
  wallEfficiencyFactor : ℚ := 1/2

/-- Calculates the length of the fountain that can be built given the construction parameters -/
def calculateFountainLength (fc : FountainConstruction) : ℚ :=
  let workRatePerMan : ℚ := fc.initialLength / (fc.initialMen * fc.initialDays)
  let newWallDays : ℚ := fc.wallDays * fc.wallEfficiencyFactor
  let availableDaysForFountain : ℚ := fc.totalDays - newWallDays
  workRatePerMan * fc.newMen * availableDaysForFountain

theorem fountain_length_is_105 (fc : FountainConstruction) :
  calculateFountainLength fc = 105 := by
  sorry

end NUMINAMATH_CALUDE_fountain_length_is_105_l2182_218275


namespace NUMINAMATH_CALUDE_lowest_possible_score_l2182_218273

/-- Represents a set of test scores -/
structure TestScores where
  scores : List ℕ
  deriving Repr

/-- Calculates the average of a list of numbers -/
def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

/-- Checks if a number is within a given range -/
def inRange (n : ℕ) (lower upper : ℕ) : Prop :=
  lower ≤ n ∧ n ≤ upper

theorem lowest_possible_score 
  (first_three : TestScores)
  (h1 : first_three.scores = [82, 90, 88])
  (h2 : first_three.scores.length = 3)
  (total_tests : ℕ)
  (h3 : total_tests = 6)
  (desired_average : ℚ)
  (h4 : desired_average = 85)
  (range_lower range_upper : ℕ)
  (h5 : range_lower = 70 ∧ range_upper = 85)
  (max_score : ℕ)
  (h6 : max_score = 100) :
  ∃ (remaining : TestScores),
    remaining.scores.length = 3 ∧
    (∃ (score : ℕ), score ∈ remaining.scores ∧ inRange score range_lower range_upper) ∧
    (∃ (lowest : ℕ), lowest ∈ remaining.scores ∧ lowest = 65) ∧
    average (first_three.scores ++ remaining.scores) = desired_average ∧
    (∀ (s : ℕ), s ∈ (first_three.scores ++ remaining.scores) → s ≤ max_score) :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l2182_218273


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l2182_218276

theorem real_roots_of_polynomial (x₁ x₂ : ℝ) :
  x₁^5 - 55*x₁ + 21 = 0 →
  x₂^5 - 55*x₂ + 21 = 0 →
  x₁ * x₂ = 1 →
  ((x₁ = (3 + Real.sqrt 5) / 2 ∧ x₂ = (3 - Real.sqrt 5) / 2) ∨
   (x₁ = (3 - Real.sqrt 5) / 2 ∧ x₂ = (3 + Real.sqrt 5) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l2182_218276


namespace NUMINAMATH_CALUDE_chord_length_midway_l2182_218235

theorem chord_length_midway (r : ℝ) (x y : ℝ) : 
  (24 : ℝ) ^ 2 / 4 + x^2 = r^2 →
  (32 : ℝ) ^ 2 / 4 + y^2 = r^2 →
  x + y = 14 →
  let d := (x - y) / 2
  2 * Real.sqrt (r^2 - d^2) = 2 * Real.sqrt 249 := by sorry

end NUMINAMATH_CALUDE_chord_length_midway_l2182_218235


namespace NUMINAMATH_CALUDE_manuscript_pages_l2182_218283

/-- Represents the typing service cost structure and manuscript details -/
structure ManuscriptTyping where
  first_time_cost : ℕ
  revision_cost : ℕ
  pages_revised_once : ℕ
  pages_revised_twice : ℕ
  total_cost : ℕ

/-- Calculates the total number of pages in the manuscript -/
def total_pages (mt : ManuscriptTyping) : ℕ :=
  sorry

/-- Theorem stating that the total number of pages is 100 -/
theorem manuscript_pages (mt : ManuscriptTyping) 
  (h1 : mt.first_time_cost = 5)
  (h2 : mt.revision_cost = 3)
  (h3 : mt.pages_revised_once = 30)
  (h4 : mt.pages_revised_twice = 20)
  (h5 : mt.total_cost = 710) :
  total_pages mt = 100 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_pages_l2182_218283


namespace NUMINAMATH_CALUDE_wendys_candy_boxes_l2182_218245

/-- Proves that Wendy had 2 boxes of candy given the problem conditions -/
theorem wendys_candy_boxes :
  ∀ (brother_candy : ℕ) (pieces_per_box : ℕ) (total_candy : ℕ) (wendys_boxes : ℕ),
    brother_candy = 6 →
    pieces_per_box = 3 →
    total_candy = 12 →
    total_candy = brother_candy + (wendys_boxes * pieces_per_box) →
    wendys_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_wendys_candy_boxes_l2182_218245


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2182_218281

-- Define the inequality
def inequality (x : ℝ) : Prop := x / (2 * x - 1) > 1

-- Define the solution set
def solution_set : Set ℝ := {x | 1/2 < x ∧ x < 1}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2182_218281


namespace NUMINAMATH_CALUDE_victor_trays_capacity_l2182_218225

/-- The number of trays Victor picked up from the first table -/
def trays_table1 : ℕ := 23

/-- The number of trays Victor picked up from the second table -/
def trays_table2 : ℕ := 5

/-- The total number of trips Victor made -/
def total_trips : ℕ := 4

/-- The number of trays Victor could carry at a time -/
def trays_per_trip : ℕ := (trays_table1 + trays_table2) / total_trips

theorem victor_trays_capacity : trays_per_trip = 7 := by
  sorry

end NUMINAMATH_CALUDE_victor_trays_capacity_l2182_218225


namespace NUMINAMATH_CALUDE_max_sum_ab_l2182_218200

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Given four distinct digits A, B, C, D, where (A+B)/(C+D) is an integer
    and C+D > 1, the maximum value of A+B is 15 -/
theorem max_sum_ab (A B C D : Digit) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_integer : ∃ k : ℕ+, k * (C.val + D.val) = A.val + B.val)
  (h_cd_gt_one : C.val + D.val > 1) :
  A.val + B.val ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_sum_ab_l2182_218200


namespace NUMINAMATH_CALUDE_intersection_is_empty_l2182_218227

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l2182_218227


namespace NUMINAMATH_CALUDE_employee_selection_probability_l2182_218255

/-- Represents the survey results of employees -/
structure EmployeeSurvey where
  total : ℕ
  uninsured : ℕ
  partTime : ℕ
  uninsuredPartTime : ℕ
  multipleJobs : ℕ
  alternativeInsurance : ℕ

/-- Calculates the probability of selecting an employee with specific characteristics -/
def calculateProbability (survey : EmployeeSurvey) : ℚ :=
  let neitherUninsuredNorPartTime := survey.total - (survey.uninsured + survey.partTime - survey.uninsuredPartTime)
  let targetEmployees := neitherUninsuredNorPartTime - survey.multipleJobs - survey.alternativeInsurance
  targetEmployees / survey.total

/-- The main theorem stating the probability of selecting an employee with specific characteristics -/
theorem employee_selection_probability :
  let survey := EmployeeSurvey.mk 500 140 80 6 35 125
  calculateProbability survey = 63 / 250 := by sorry

end NUMINAMATH_CALUDE_employee_selection_probability_l2182_218255


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_G_powers_of_three_l2182_218239

def G : ℕ → ℚ
  | 0 => 1
  | 1 => 4/3
  | (n + 2) => 3 * G (n + 1) - 2 * G n

theorem sum_of_reciprocal_G_powers_of_three : ∑' n, 1 / G (3^n) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_G_powers_of_three_l2182_218239


namespace NUMINAMATH_CALUDE_geometry_propositions_l2182_218228

-- Define the basic types
variable (α β : Plane) (l m : Line)

-- Define the relationships
def perpendicular_to_plane (line : Line) (plane : Plane) : Prop := sorry
def contained_in_plane (line : Line) (plane : Plane) : Prop := sorry
def parallel_planes (plane1 plane2 : Plane) : Prop := sorry
def perpendicular_planes (plane1 plane2 : Plane) : Prop := sorry
def perpendicular_lines (line1 line2 : Line) : Prop := sorry
def parallel_lines (line1 line2 : Line) : Prop := sorry

-- State the theorem
theorem geometry_propositions 
  (h1 : perpendicular_to_plane l α) 
  (h2 : contained_in_plane m β) :
  (parallel_planes α β → perpendicular_lines l m) ∧ 
  ¬(perpendicular_planes α β → parallel_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) := by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l2182_218228


namespace NUMINAMATH_CALUDE_crickets_collected_l2182_218295

theorem crickets_collected (total : ℕ) (more_needed : ℕ) (h : total = 11 ∧ more_needed = 4) : 
  total - more_needed = 7 := by
  sorry

end NUMINAMATH_CALUDE_crickets_collected_l2182_218295


namespace NUMINAMATH_CALUDE_x_values_l2182_218262

theorem x_values (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 144)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 :=
by sorry

end NUMINAMATH_CALUDE_x_values_l2182_218262


namespace NUMINAMATH_CALUDE_lowest_true_statement_l2182_218246

def statement201 (s203 : Bool) : Bool := s203
def statement202 (s201 : Bool) : Bool := s201
def statement203 (s206 : Bool) : Bool := ¬s206
def statement204 (s202 : Bool) : Bool := ¬s202
def statement205 (s201 s202 s203 s204 : Bool) : Bool := ¬(s201 ∨ s202 ∨ s203 ∨ s204)
def statement206 : Bool := 1 + 1 = 2

theorem lowest_true_statement :
  let s206 := statement206
  let s203 := statement203 s206
  let s201 := statement201 s203
  let s202 := statement202 s201
  let s204 := statement204 s202
  let s205 := statement205 s201 s202 s203 s204
  (¬s201 ∧ ¬s202 ∧ ¬s203 ∧ s204 ∧ ¬s205 ∧ s206) ∧
  (∀ n : Nat, n < 204 → ¬(n = 201 ∧ s201 ∨ n = 202 ∧ s202 ∨ n = 203 ∧ s203)) :=
by sorry

end NUMINAMATH_CALUDE_lowest_true_statement_l2182_218246


namespace NUMINAMATH_CALUDE_sequence_property_initial_condition_main_theorem_l2182_218220

def sequence_a (n : ℕ) : ℝ :=
  sorry

theorem sequence_property (n : ℕ) :
  (2 * n + 3 : ℝ) * sequence_a (n + 1) - (2 * n + 5 : ℝ) * sequence_a n =
  (2 * n + 3 : ℝ) * (2 * n + 5 : ℝ) * Real.log (1 + 1 / (n : ℝ)) :=
  sorry

theorem initial_condition : sequence_a 1 = 5 :=
  sorry

theorem main_theorem (n : ℕ) (hn : n > 0) :
  sequence_a n / (2 * n + 3 : ℝ) = 1 + Real.log n :=
  sorry

end NUMINAMATH_CALUDE_sequence_property_initial_condition_main_theorem_l2182_218220


namespace NUMINAMATH_CALUDE_days_without_calls_l2182_218205

/-- Represents the number of days in the year -/
def total_days : ℕ := 366

/-- Represents the frequency of calls for each niece -/
def call_frequencies : List ℕ := [2, 3, 4]

/-- Calculates the number of days with at least one call -/
def days_with_calls (frequencies : List ℕ) (total : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of days without calls -/
theorem days_without_calls :
  total_days - days_with_calls call_frequencies total_days = 122 :=
sorry

end NUMINAMATH_CALUDE_days_without_calls_l2182_218205


namespace NUMINAMATH_CALUDE_sound_propagation_at_10C_l2182_218288

-- Define the relationship between temperature and speed of sound
def speed_of_sound (temp : Int) : Int :=
  match temp with
  | -20 => 318
  | -10 => 324
  | 0 => 330
  | 10 => 336
  | 20 => 342
  | 30 => 348
  | _ => 0  -- For temperatures not in the data set

-- Theorem statement
theorem sound_propagation_at_10C :
  speed_of_sound 10 * 4 = 1344 := by
  sorry


end NUMINAMATH_CALUDE_sound_propagation_at_10C_l2182_218288


namespace NUMINAMATH_CALUDE_units_digit_G_500_l2182_218256

/-- The Modified Fermat number for a given n -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_500 : units_digit (G 500) = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_500_l2182_218256


namespace NUMINAMATH_CALUDE_min_value_of_z_l2182_218233

variable (a b x : ℝ)
variable (h : a ≠ b)

def z (x : ℝ) : ℝ := (x - a)^3 + (x - b)^3

theorem min_value_of_z :
  ∃ (x : ℝ), ∀ (y : ℝ), z a b x ≤ z a b y ↔ x = (a + b) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l2182_218233


namespace NUMINAMATH_CALUDE_walnut_trees_before_planting_l2182_218212

theorem walnut_trees_before_planting 
  (total_after : ℕ) 
  (planted : ℕ) 
  (h1 : total_after = 55) 
  (h2 : planted = 33) :
  total_after - planted = 22 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_before_planting_l2182_218212


namespace NUMINAMATH_CALUDE_train_speed_train_speed_is_24_l2182_218231

theorem train_speed (person_speed : ℝ) (overtake_time : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed := train_length / overtake_time * 3600 / 1000
  relative_speed + person_speed

#check train_speed 4 9 49.999999999999986 = 24

theorem train_speed_is_24 :
  train_speed 4 9 49.999999999999986 = 24 := by sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_is_24_l2182_218231


namespace NUMINAMATH_CALUDE_workshop_groups_l2182_218206

theorem workshop_groups (total_participants : ℕ) (max_group_size : ℕ) (h1 : total_participants = 36) (h2 : max_group_size = 12) :
  ∃ (num_groups : ℕ), num_groups * max_group_size ≥ total_participants ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_participants → k ≥ num_groups :=
by sorry

end NUMINAMATH_CALUDE_workshop_groups_l2182_218206


namespace NUMINAMATH_CALUDE_triangle_area_bound_l2182_218271

theorem triangle_area_bound (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) / 4 ≤ Real.sqrt 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_bound_l2182_218271


namespace NUMINAMATH_CALUDE_tangent_ratio_theorem_l2182_218286

theorem tangent_ratio_theorem (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (Real.cos θ ^ 2 + Real.sin θ * Real.cos θ) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_theorem_l2182_218286


namespace NUMINAMATH_CALUDE_beatrice_tv_shopping_l2182_218229

theorem beatrice_tv_shopping (first_store : ℕ) (online_store : ℕ) (auction_site : ℕ) :
  first_store = 8 →
  online_store = 3 * first_store →
  first_store + online_store + auction_site = 42 →
  auction_site = 10 := by
sorry

end NUMINAMATH_CALUDE_beatrice_tv_shopping_l2182_218229


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l2182_218238

/-- The height of a ball thrown upwards is given by y = -20t^2 + 32t + 60,
    where y is the height in feet and t is the time in seconds.
    This theorem proves that the time when the ball hits the ground (y = 0)
    is (4 + √91) / 5 seconds. -/
theorem ball_hitting_ground_time :
  let y (t : ℝ) := -20 * t^2 + 32 * t + 60
  ∃ t : ℝ, y t = 0 ∧ t = (4 + Real.sqrt 91) / 5 :=
by sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l2182_218238


namespace NUMINAMATH_CALUDE_stationery_store_bundles_l2182_218230

/-- Given the number of red and blue sheets of paper and the number of sheets per bundle,
    calculates the maximum number of complete bundles that can be made. -/
def max_bundles (red_sheets blue_sheets sheets_per_bundle : ℕ) : ℕ :=
  (red_sheets + blue_sheets) / sheets_per_bundle

/-- Proves that with 210 red sheets, 473 blue sheets, and 100 sheets per bundle,
    the maximum number of complete bundles is 6. -/
theorem stationery_store_bundles :
  max_bundles 210 473 100 = 6 := by
  sorry

#eval max_bundles 210 473 100

end NUMINAMATH_CALUDE_stationery_store_bundles_l2182_218230


namespace NUMINAMATH_CALUDE_barry_sotter_magic_l2182_218217

def length_increase (n : ℕ) : ℚ :=
  (n + 3 : ℚ) / 3

theorem barry_sotter_magic (n : ℕ) : length_increase n = 50 → n = 147 := by
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l2182_218217


namespace NUMINAMATH_CALUDE_ellipse_point_distance_l2182_218266

/-- The ellipse with equation x²/9 + y²/6 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) + (p.2^2 / 6) = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_point_distance :
  P ∈ Ellipse →
  angle F₁ P F₂ = Real.arccos (3/5) →
  distance P O = Real.sqrt 30 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_point_distance_l2182_218266


namespace NUMINAMATH_CALUDE_perpendicular_vector_with_sum_condition_l2182_218250

/-- Given two parallel lines l and m with direction vector (4, 3),
    prove that (-6, 8) is perpendicular to their direction vector
    and its components sum to 2. -/
theorem perpendicular_vector_with_sum_condition :
  let direction_vector : ℝ × ℝ := (4, 3)
  let perpendicular_vector : ℝ × ℝ := (-6, 8)
  (direction_vector.1 * perpendicular_vector.1 + direction_vector.2 * perpendicular_vector.2 = 0) ∧
  (perpendicular_vector.1 + perpendicular_vector.2 = 2) := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_vector_with_sum_condition_l2182_218250


namespace NUMINAMATH_CALUDE_derivative_at_point_not_equivalent_to_derivative_of_constant_l2182_218207

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Assume f is differentiable
variable (hf : Differentiable ℝ f)

-- Define a point x₀
variable (x₀ : ℝ)

-- Statement: f'(x₀) is not equivalent to [f(x₀)]'
theorem derivative_at_point_not_equivalent_to_derivative_of_constant :
  ¬(∀ x₀ : ℝ, (deriv f) x₀ = deriv (λ _ : ℝ => f x₀) x₀) :=
sorry

end NUMINAMATH_CALUDE_derivative_at_point_not_equivalent_to_derivative_of_constant_l2182_218207


namespace NUMINAMATH_CALUDE_bus_meeting_problem_l2182_218254

theorem bus_meeting_problem (n k : ℕ) (h1 : n > 3) 
  (h2 : n * (n - 1) * (2 * k - 1) = 600) : n * k = 52 ∨ n * k = 40 := by
  sorry

end NUMINAMATH_CALUDE_bus_meeting_problem_l2182_218254


namespace NUMINAMATH_CALUDE_symmetric_point_reciprocal_function_l2182_218252

theorem symmetric_point_reciprocal_function (k : ℝ) : 
  let B : ℝ × ℝ := (Real.cos (π / 3), -Real.sqrt 3)
  let A : ℝ × ℝ := (B.1, -B.2)
  (A.2 = k / A.1) → k = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_reciprocal_function_l2182_218252


namespace NUMINAMATH_CALUDE_files_remaining_l2182_218219

theorem files_remaining (initial_files initial_apps final_apps files_deleted : ℕ) :
  initial_files = 24 →
  initial_apps = 13 →
  final_apps = 17 →
  files_deleted = 3 →
  initial_files - (final_apps - initial_apps) - files_deleted = 17 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l2182_218219


namespace NUMINAMATH_CALUDE_ball_painting_probability_l2182_218201

def num_balls : ℕ := 8
def num_red : ℕ := 4
def num_blue : ℕ := 4

def prob_red : ℚ := 1 / 2
def prob_blue : ℚ := 1 / 2

theorem ball_painting_probability :
  (prob_red ^ num_red) * (prob_blue ^ num_blue) = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_ball_painting_probability_l2182_218201


namespace NUMINAMATH_CALUDE_smallest_a_l2182_218278

/-- A parabola with vertex at (1/3, -25/27) described by y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a > 0 → b = -2*a/3
  vertex_y : a > 0 → c = a/9 - 25/27
  integer_sum : ∃ k : ℤ, 3*a + 2*b + 4*c = k

/-- The smallest possible value of a for the given parabola conditions -/
theorem smallest_a (p : Parabola) : 
  (∀ q : Parabola, q.a > 0 → p.a ≤ q.a) → p.a = 300/19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_l2182_218278


namespace NUMINAMATH_CALUDE_alice_savings_difference_l2182_218285

def type_a_sales : ℝ := 1800
def type_b_sales : ℝ := 800
def type_c_sales : ℝ := 500
def basic_salary : ℝ := 500
def type_a_commission_rate : ℝ := 0.04
def type_b_commission_rate : ℝ := 0.06
def type_c_commission_rate : ℝ := 0.10
def monthly_expenses : ℝ := 600
def saving_goal : ℝ := 450
def usual_saving_rate : ℝ := 0.15

def total_commission : ℝ := 
  type_a_sales * type_a_commission_rate + 
  type_b_sales * type_b_commission_rate + 
  type_c_sales * type_c_commission_rate

def total_earnings : ℝ := basic_salary + total_commission

def net_earnings : ℝ := total_earnings - monthly_expenses

def actual_savings : ℝ := net_earnings * usual_saving_rate

theorem alice_savings_difference : saving_goal - actual_savings = 439.50 := by
  sorry

end NUMINAMATH_CALUDE_alice_savings_difference_l2182_218285


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2182_218270

/-- The function f(x) = x(x-1)(x-2) -/
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2)

/-- The theorem stating that the derivative of f at x=0 is 2 -/
theorem derivative_f_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2182_218270


namespace NUMINAMATH_CALUDE_levels_beaten_l2182_218268

theorem levels_beaten (total_levels : ℕ) (ratio : ℚ) : total_levels = 32 ∧ ratio = 3 / 1 → 
  ∃ (beaten : ℕ), beaten = 24 ∧ beaten * (1 + 1 / ratio) = total_levels := by
sorry

end NUMINAMATH_CALUDE_levels_beaten_l2182_218268


namespace NUMINAMATH_CALUDE_cube_root_8000_l2182_218279

theorem cube_root_8000 : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c : ℝ) * (d : ℝ)^(1/3) = 8000^(1/3) ∧
  (∀ (c' d' : ℕ), c' > 0 → d' > 0 → (c' : ℝ) * (d' : ℝ)^(1/3) = 8000^(1/3) → d ≤ d') ∧
  c = 20 ∧ d = 1 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_8000_l2182_218279


namespace NUMINAMATH_CALUDE_stock_value_change_l2182_218280

theorem stock_value_change (initial_value : ℝ) (day1_decrease : ℝ) (day2_increase : ℝ) :
  day1_decrease = 0.2 →
  day2_increase = 0.3 →
  (1 - day1_decrease) * (1 + day2_increase) = 1.04 := by
  sorry

end NUMINAMATH_CALUDE_stock_value_change_l2182_218280


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l2182_218269

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 67)
  (h2 : throwers = 37)
  (h3 : (total_players - throwers) % 3 = 0)
  (h4 : throwers ≤ total_players) :
  throwers + ((total_players - throwers) * 2 / 3) = 57 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l2182_218269


namespace NUMINAMATH_CALUDE_divisibility_by_203_l2182_218243

theorem divisibility_by_203 (n : ℕ+) : 
  (2013^n.val - 1803^n.val - 1781^n.val + 1774^n.val) % 203 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_203_l2182_218243


namespace NUMINAMATH_CALUDE_difference_is_10q_minus_10_l2182_218265

/-- The difference in dimes between two people's money, given their quarter amounts -/
def difference_in_dimes (charles_quarters richard_quarters : ℤ) : ℚ :=
  2.5 * (charles_quarters - richard_quarters)

/-- Proof that the difference in dimes between Charles and Richard's money is 10(q - 1) -/
theorem difference_is_10q_minus_10 (q : ℤ) :
  difference_in_dimes (5 * q + 1) (q + 5) = 10 * (q - 1) := by
  sorry

#check difference_is_10q_minus_10

end NUMINAMATH_CALUDE_difference_is_10q_minus_10_l2182_218265


namespace NUMINAMATH_CALUDE_bianca_winning_strategy_l2182_218248

/-- Represents a game state with two piles of marbles. -/
structure GameState where
  a : ℕ
  b : ℕ
  sum_eq_100 : a + b = 100

/-- Predicate to check if a move is valid. -/
def valid_move (s : GameState) (pile : ℕ) (remove : ℕ) : Prop :=
  (pile = s.a ∨ pile = s.b) ∧ 0 < remove ∧ remove ≤ pile / 2

/-- Predicate to check if a game state is a winning position for Bianca. -/
def is_winning_for_bianca (s : GameState) : Prop :=
  (s.a = 50 ∧ s.b = 50) ∨
  (s.a = 67 ∧ s.b = 33) ∨
  (s.a = 33 ∧ s.b = 67) ∨
  (s.a = 95 ∧ s.b = 5) ∨
  (s.a = 5 ∧ s.b = 95)

/-- Theorem stating that Bianca has a winning strategy if and only if
    the game state is one of the specified winning positions. -/
theorem bianca_winning_strategy (s : GameState) :
  (∀ (pile remove : ℕ), valid_move s pile remove →
    ∃ (new_s : GameState), ¬is_winning_for_bianca new_s) ↔
  is_winning_for_bianca s :=
sorry

end NUMINAMATH_CALUDE_bianca_winning_strategy_l2182_218248


namespace NUMINAMATH_CALUDE_total_books_l2182_218291

/-- The total number of books Tim, Sam, and Emma have together is 133. -/
theorem total_books (tim_books sam_books emma_books : ℕ) 
  (h1 : tim_books = 44)
  (h2 : sam_books = 52)
  (h3 : emma_books = 37) : 
  tim_books + sam_books + emma_books = 133 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l2182_218291


namespace NUMINAMATH_CALUDE_product_remainder_by_10_l2182_218282

theorem product_remainder_by_10 : (2456 * 7294 * 91803) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_10_l2182_218282


namespace NUMINAMATH_CALUDE_exp_sum_equals_ten_l2182_218204

theorem exp_sum_equals_ten (a b : ℝ) (h1 : Real.log 3 = a) (h2 : Real.log 7 = b) :
  Real.exp a + Real.exp b = 10 := by
  sorry

end NUMINAMATH_CALUDE_exp_sum_equals_ten_l2182_218204


namespace NUMINAMATH_CALUDE_star_example_l2182_218211

-- Define the star operation
def star (m n p q : ℚ) : ℚ := (m + p) * (m + q) * (q / n)

-- Theorem statement
theorem star_example : star (6/11) (6/11) (5/2) (5/2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l2182_218211


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l2182_218247

theorem percentage_equation_solution :
  ∃ x : ℝ, (65 / 100) * x = (20 / 100) * 682.50 ∧ x = 210 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l2182_218247


namespace NUMINAMATH_CALUDE_population_change_proof_l2182_218209

-- Define the initial population
def initial_population : ℕ := 4518

-- Define the sequence of population changes
def population_after_bombardment (p : ℕ) : ℕ := (p * 95) / 100
def population_after_migration (p : ℕ) : ℕ := (p * 80) / 100
def population_after_return (p : ℕ) : ℕ := (p * 115) / 100
def population_after_flood (p : ℕ) : ℕ := (p * 90) / 100

-- Define the final population
def final_population : ℕ := 3553

-- Theorem statement
theorem population_change_proof :
  population_after_flood
    (population_after_return
      (population_after_migration
        (population_after_bombardment initial_population)))
  = final_population := by sorry

end NUMINAMATH_CALUDE_population_change_proof_l2182_218209


namespace NUMINAMATH_CALUDE_ned_games_before_l2182_218257

/-- The number of games Ned had before giving away some -/
def games_before : ℕ := sorry

/-- The number of games Ned gave away -/
def games_given_away : ℕ := 13

/-- The number of games Ned has now -/
def games_now : ℕ := 6

/-- Theorem stating the number of games Ned had before -/
theorem ned_games_before :
  games_before = games_given_away + games_now := by sorry

end NUMINAMATH_CALUDE_ned_games_before_l2182_218257


namespace NUMINAMATH_CALUDE_contrapositive_geometric_sequence_l2182_218260

/-- A sequence (a, b, c) is geometric if there exists a common ratio r such that b = ar and c = br -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The main theorem: The contrapositive of "If (a,b,c) is geometric, then b^2 = ac" 
    is equivalent to "If b^2 ≠ ac, then (a,b,c) is not geometric" -/
theorem contrapositive_geometric_sequence (a b c : ℝ) :
  (¬(b^2 = a*c) → ¬(IsGeometricSequence a b c)) ↔
  (IsGeometricSequence a b c → b^2 = a*c) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_geometric_sequence_l2182_218260


namespace NUMINAMATH_CALUDE_sum_of_specific_sequence_l2182_218267

def arithmetic_sequence_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem sum_of_specific_sequence :
  arithmetic_sequence_sum 102 492 10 = 11880 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_sequence_l2182_218267


namespace NUMINAMATH_CALUDE_inscribed_equilateral_triangle_in_five_moves_l2182_218298

/-- Represents a point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle in the plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents the game state -/
structure GameState :=
  (knownPoints : Set Point)
  (lines : Set Line)
  (circles : Set Circle)

/-- Represents a move in the game -/
inductive Move
  | DrawLine (p1 p2 : Point)
  | DrawCircle (center : Point) (throughPoint : Point)

/-- Checks if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  let d12 := ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
  let d23 := ((p2.x - p3.x)^2 + (p2.y - p3.y)^2)
  let d31 := ((p3.x - p1.x)^2 + (p3.y - p1.y)^2)
  d12 = d23 ∧ d23 = d31

/-- The main theorem -/
theorem inscribed_equilateral_triangle_in_five_moves 
  (initialCircle : Circle) (initialPoint : Point) 
  (h : isOnCircle initialPoint initialCircle) :
  ∃ (moves : List Move) (p1 p2 p3 : Point),
    moves.length = 5 ∧
    isEquilateralTriangle p1 p2 p3 ∧
    isOnCircle p1 initialCircle ∧
    isOnCircle p2 initialCircle ∧
    isOnCircle p3 initialCircle :=
  sorry

end NUMINAMATH_CALUDE_inscribed_equilateral_triangle_in_five_moves_l2182_218298


namespace NUMINAMATH_CALUDE_fish_estimation_result_l2182_218218

/-- Represents the catch-release-recatch method for estimating fish population -/
structure FishEstimation where
  initial_catch : ℕ
  initial_marked : ℕ
  second_catch : ℕ
  second_marked : ℕ

/-- Calculates the estimated number of fish in the pond -/
def estimate_fish_population (fe : FishEstimation) : ℕ :=
  (fe.initial_marked * fe.second_catch) / fe.second_marked

/-- Theorem stating that the estimated number of fish in the pond is 2500 -/
theorem fish_estimation_result :
  let fe : FishEstimation := {
    initial_catch := 100,
    initial_marked := 100,
    second_catch := 200,
    second_marked := 8
  }
  estimate_fish_population fe = 2500 := by
  sorry


end NUMINAMATH_CALUDE_fish_estimation_result_l2182_218218


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l2182_218296

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (1/a + 2/b) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l2182_218296


namespace NUMINAMATH_CALUDE_difference_max_min_both_l2182_218213

def total_students : ℕ := 1500

def spanish_min : ℕ := 1050
def spanish_max : ℕ := 1125

def french_min : ℕ := 525
def french_max : ℕ := 675

def min_both : ℕ := spanish_min + french_min - total_students
def max_both : ℕ := spanish_max + french_max - total_students

theorem difference_max_min_both : max_both - min_both = 225 := by
  sorry

end NUMINAMATH_CALUDE_difference_max_min_both_l2182_218213


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2182_218244

def M : ℕ := 36 * 36 * 65 * 280

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M) * 254 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2182_218244


namespace NUMINAMATH_CALUDE_count_ordered_quadruples_l2182_218294

theorem count_ordered_quadruples (n : ℕ+) :
  (Finset.filter (fun (quad : ℕ × ℕ × ℕ × ℕ) =>
    let (a, b, c, d) := quad
    0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ n)
    (Finset.product (Finset.range (n + 1))
      (Finset.product (Finset.range (n + 1))
        (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))))).card
  = Nat.choose (n + 4) 4 :=
by sorry

end NUMINAMATH_CALUDE_count_ordered_quadruples_l2182_218294


namespace NUMINAMATH_CALUDE_problem_statement_l2182_218272

theorem problem_statement : (-1)^49 + 2^(3^3 + 5^2 - 48^2) = -1 + 1 / 2^2252 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2182_218272


namespace NUMINAMATH_CALUDE_subtraction_multiplication_theorem_l2182_218223

theorem subtraction_multiplication_theorem : ((3.65 - 1.27) * 2) = 4.76 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_theorem_l2182_218223


namespace NUMINAMATH_CALUDE_max_smoothie_servings_l2182_218203

/-- Represents the recipe for 8 servings -/
structure Recipe where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ
  milk : ℕ

/-- Represents Sarah's available ingredients -/
structure Ingredients where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ
  milk : ℕ

/-- Calculates the maximum number of servings possible for a given ingredient -/
def max_servings (recipe_amount : ℕ) (available_amount : ℕ) : ℚ :=
  (available_amount : ℚ) / (recipe_amount : ℚ) * 8

/-- Theorem stating the maximum number of servings Sarah can make -/
theorem max_smoothie_servings (recipe : Recipe) (sarah_ingredients : Ingredients) :
  recipe.bananas = 3 ∧ 
  recipe.strawberries = 2 ∧ 
  recipe.yogurt = 1 ∧ 
  recipe.milk = 4 ∧
  sarah_ingredients.bananas = 10 ∧
  sarah_ingredients.strawberries = 5 ∧
  sarah_ingredients.yogurt = 3 ∧
  sarah_ingredients.milk = 10 →
  ⌊min 
    (min (max_servings recipe.bananas sarah_ingredients.bananas) (max_servings recipe.strawberries sarah_ingredients.strawberries))
    (min (max_servings recipe.yogurt sarah_ingredients.yogurt) (max_servings recipe.milk sarah_ingredients.milk))
  ⌋ = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_smoothie_servings_l2182_218203


namespace NUMINAMATH_CALUDE_inscribed_isosceles_triangle_circle_properties_l2182_218208

/-- An isosceles triangle inscribed in a circle -/
structure InscribedIsoscelesTriangle where
  /-- Length of the two equal sides of the isosceles triangle -/
  side_length : ℝ
  /-- Length of the base of the isosceles triangle -/
  base_length : ℝ
  /-- Radius of the circumscribed circle -/
  circle_radius : ℝ

/-- Properties of the inscribed isosceles triangle -/
def triangle_properties (t : InscribedIsoscelesTriangle) : Prop :=
  t.side_length = 4 ∧ t.base_length = 3

theorem inscribed_isosceles_triangle_circle_properties
  (t : InscribedIsoscelesTriangle)
  (h : triangle_properties t) :
  t.circle_radius = 3.5 ∧ π * t.circle_radius ^ 2 = 12.25 * π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_isosceles_triangle_circle_properties_l2182_218208


namespace NUMINAMATH_CALUDE_quadrangular_pyramid_faces_l2182_218253

/-- A quadrangular pyramid is a geometric shape with triangular lateral faces and a quadrilateral base. -/
structure QuadrangularPyramid where
  lateral_faces : Nat
  base_face : Nat
  lateral_faces_are_triangles : lateral_faces = 4
  base_is_quadrilateral : base_face = 1

/-- The total number of faces in a quadrangular pyramid is 5. -/
theorem quadrangular_pyramid_faces (p : QuadrangularPyramid) : 
  p.lateral_faces + p.base_face = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadrangular_pyramid_faces_l2182_218253


namespace NUMINAMATH_CALUDE_elberta_amount_l2182_218287

def granny_smith : ℕ := 63

def anjou : ℕ := granny_smith / 3

def elberta : ℕ := anjou + 2

theorem elberta_amount : elberta = 23 := by
  sorry

end NUMINAMATH_CALUDE_elberta_amount_l2182_218287


namespace NUMINAMATH_CALUDE_sum_of_three_square_roots_inequality_l2182_218274

theorem sum_of_three_square_roots_inequality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) : 
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_square_roots_inequality_l2182_218274


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l2182_218249

theorem lcm_factor_problem (A B : ℕ+) (h : Nat.gcd A B = 25) (hA : A = 350) 
  (hlcm : ∃ x : ℕ+, Nat.lcm A B = 25 * 13 * x) : 
  ∃ x : ℕ+, Nat.lcm A B = 25 * 13 * x ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l2182_218249
