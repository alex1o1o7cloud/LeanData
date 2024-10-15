import Mathlib

namespace NUMINAMATH_CALUDE_function_max_min_l3554_355416

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + a^2 - 1

theorem function_max_min (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f a x ≤ 24) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f a x = 24) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f a x ≥ 3) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f a x = 3) →
  a = 2 ∨ a = -5 := by sorry

end NUMINAMATH_CALUDE_function_max_min_l3554_355416


namespace NUMINAMATH_CALUDE_fraction_equality_l3554_355428

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a - b) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3554_355428


namespace NUMINAMATH_CALUDE_chord_length_is_six_l3554_355447

/-- A circle with equation x^2 + y^2 + 8x - 10y + 41 = r^2 that is tangent to the x-axis --/
structure TangentCircle where
  r : ℝ
  tangent_to_x_axis : r = 5

/-- The length of the chord intercepted by the circle on the y-axis --/
def chord_length (c : TangentCircle) : ℝ :=
  let y₁ := 2
  let y₂ := 8
  |y₁ - y₂|

/-- Theorem stating that the length of the chord intercepted by the circle on the y-axis is 6 --/
theorem chord_length_is_six (c : TangentCircle) : chord_length c = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_is_six_l3554_355447


namespace NUMINAMATH_CALUDE_sqrt_expression_one_sqrt_expression_two_sqrt_expression_three_l3554_355439

-- Problem 1
theorem sqrt_expression_one : 
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem sqrt_expression_two : 
  1 / Real.sqrt 24 + |Real.sqrt 6 - 3| + (1/2)⁻¹ - 2016^0 = 4 - 11 * Real.sqrt 6 / 12 := by sorry

-- Problem 3
theorem sqrt_expression_three : 
  (Real.sqrt 3 + Real.sqrt 2)^2 - (Real.sqrt 3 - Real.sqrt 2)^2 = 4 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_one_sqrt_expression_two_sqrt_expression_three_l3554_355439


namespace NUMINAMATH_CALUDE_expand_expression_l3554_355441

theorem expand_expression (x : ℝ) : (17 * x + 18) * (3 * x + 4) = 51 * x^2 + 122 * x + 72 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3554_355441


namespace NUMINAMATH_CALUDE_max_odd_group_length_l3554_355470

/-- A sequence of consecutive natural numbers where each number
    is a product of prime factors with odd exponents -/
def OddGroup (start : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → 
    ∀ p : ℕ, Nat.Prime p → 
      Odd ((start + k).factorization p)

/-- The maximum length of an OddGroup -/
theorem max_odd_group_length : 
  (∃ (max : ℕ), ∀ (start n : ℕ), OddGroup start n → n ≤ max) ∧ 
  (∃ (start : ℕ), OddGroup start 7) :=
sorry

end NUMINAMATH_CALUDE_max_odd_group_length_l3554_355470


namespace NUMINAMATH_CALUDE_fraction_equality_l3554_355493

theorem fraction_equality (x y : ℚ) (h : y ≠ 0) (h1 : x / y = 7 / 2) :
  (x - 2 * y) / y = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3554_355493


namespace NUMINAMATH_CALUDE_bookstore_max_revenue_l3554_355498

/-- The revenue function for the bookstore -/
def revenue (p : ℝ) : ℝ := p * (150 - 6 * p)

/-- The maximum price allowed -/
def max_price : ℝ := 30

theorem bookstore_max_revenue :
  ∃ (p : ℝ), p ≤ max_price ∧
    ∀ (q : ℝ), q ≤ max_price → revenue q ≤ revenue p ∧
    p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_max_revenue_l3554_355498


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l3554_355406

/-- Represents the number of rooms that can be painted with the initial amount of paint -/
def initial_rooms : ℕ := 50

/-- Represents the number of rooms that can be painted after losing two cans -/
def remaining_rooms : ℕ := 42

/-- Represents the number of cans lost -/
def lost_cans : ℕ := 2

/-- Calculates the number of cans used to paint the remaining rooms -/
def cans_used : ℕ := 
  let rooms_per_can := (initial_rooms - remaining_rooms) / lost_cans
  (remaining_rooms + rooms_per_can - 1) / rooms_per_can

theorem paint_cans_theorem : cans_used = 11 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l3554_355406


namespace NUMINAMATH_CALUDE_average_equation_solution_l3554_355474

theorem average_equation_solution (x : ℚ) : 
  (1 / 3 : ℚ) * ((3 * x + 8) + (7 * x - 3) + (4 * x + 5)) = 5 * x - 6 → x = -28 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3554_355474


namespace NUMINAMATH_CALUDE_age_problem_l3554_355490

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3554_355490


namespace NUMINAMATH_CALUDE_no_extremum_l3554_355488

open Real

/-- A function satisfying the given differential equation and initial condition -/
def SolutionFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → x * (deriv f x) + f x = exp x / x) ∧ f 1 = exp 1

/-- The main theorem stating that the function has no maximum or minimum -/
theorem no_extremum (f : ℝ → ℝ) (hf : SolutionFunction f) :
    (∀ x, x > 0 → ¬ IsLocalMax f x) ∧ (∀ x, x > 0 → ¬ IsLocalMin f x) := by
  sorry


end NUMINAMATH_CALUDE_no_extremum_l3554_355488


namespace NUMINAMATH_CALUDE_discount_order_difference_l3554_355486

/-- Calculates the difference in final price when applying discounts in different orders -/
theorem discount_order_difference : 
  let original_price : ℚ := 30
  let fixed_discount : ℚ := 5
  let percentage_discount : ℚ := 0.25
  let scenario1 := (original_price - fixed_discount) * (1 - percentage_discount)
  let scenario2 := (original_price * (1 - percentage_discount)) - fixed_discount
  (scenario2 - scenario1) * 100 = 125 := by sorry

end NUMINAMATH_CALUDE_discount_order_difference_l3554_355486


namespace NUMINAMATH_CALUDE_day_care_ratio_l3554_355459

/-- Proves that the initial ratio of toddlers to infants is 7:3 given the conditions of the day care problem. -/
theorem day_care_ratio (toddlers initial_infants : ℕ) : 
  toddlers = 42 →
  (toddlers : ℚ) / (initial_infants + 12 : ℚ) = 7 / 5 →
  (toddlers : ℚ) / (initial_infants : ℚ) = 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_day_care_ratio_l3554_355459


namespace NUMINAMATH_CALUDE_leap_year_1996_l3554_355481

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

theorem leap_year_1996 : is_leap_year 1996 := by
  sorry

end NUMINAMATH_CALUDE_leap_year_1996_l3554_355481


namespace NUMINAMATH_CALUDE_distinct_combinations_count_l3554_355480

def word : String := "BIOLOGY"

def num_vowels : Nat := 3
def num_consonants : Nat := 3

def is_vowel (c : Char) : Bool :=
  c = 'I' || c = 'O'

def is_consonant (c : Char) : Bool :=
  c = 'B' || c = 'L' || c = 'G'

def indistinguishable (c : Char) : Bool :=
  c = 'I' || c = 'G'

theorem distinct_combinations_count :
  (∃ (vowel_combs consonant_combs : Nat),
    vowel_combs * consonant_combs = 12 ∧
    vowel_combs = (word.toList.filter is_vowel).length.choose num_vowels ∧
    consonant_combs = (word.toList.filter is_consonant).length.choose num_consonants) :=
by sorry

end NUMINAMATH_CALUDE_distinct_combinations_count_l3554_355480


namespace NUMINAMATH_CALUDE_batsman_average_after_20th_innings_l3554_355403

/-- Represents a batsman's innings record -/
structure BatsmanRecord where
  innings : ℕ
  totalScore : ℕ
  avgIncrease : ℚ
  lastScore : ℕ

/-- Calculates the average score of a batsman -/
def calculateAverage (record : BatsmanRecord) : ℚ :=
  record.totalScore / record.innings

theorem batsman_average_after_20th_innings 
  (record : BatsmanRecord)
  (h1 : record.innings = 20)
  (h2 : record.lastScore = 90)
  (h3 : record.avgIncrease = 2)
  : calculateAverage record = 52 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_20th_innings_l3554_355403


namespace NUMINAMATH_CALUDE_sqrt_14_between_3_and_4_l3554_355456

theorem sqrt_14_between_3_and_4 : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_between_3_and_4_l3554_355456


namespace NUMINAMATH_CALUDE_right_triangle_special_case_l3554_355445

theorem right_triangle_special_case (a b c : ℝ) :
  a > 0 →  -- AB is positive
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  c + b = 2*a →  -- Given condition
  b = 3/4 * a ∧ c = 5/4 * a := by sorry

end NUMINAMATH_CALUDE_right_triangle_special_case_l3554_355445


namespace NUMINAMATH_CALUDE_f_extrema_l3554_355415

-- Define the function f
def f (x y : ℝ) : ℝ := 2 * x^2 - 2 * y^2

-- Define the disk
def disk (x y : ℝ) : Prop := x^2 + y^2 ≤ 9

-- Theorem statement
theorem f_extrema :
  (∃ x y : ℝ, disk x y ∧ f x y = 18) ∧
  (∃ x y : ℝ, disk x y ∧ f x y = -18) ∧
  (∀ x y : ℝ, disk x y → f x y ≤ 18) ∧
  (∀ x y : ℝ, disk x y → f x y ≥ -18) := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l3554_355415


namespace NUMINAMATH_CALUDE_base15_divisible_by_9_l3554_355426

/-- Converts a base-15 integer to decimal --/
def base15ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (15 ^ i)) 0

/-- The base-15 representation of 2643₁₅ --/
def base15Number : List Nat := [3, 4, 6, 2]

/-- Theorem stating that 2643₁₅ divided by 9 has a remainder of 0 --/
theorem base15_divisible_by_9 :
  (base15ToDecimal base15Number) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_base15_divisible_by_9_l3554_355426


namespace NUMINAMATH_CALUDE_f_zero_equals_two_l3554_355471

def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ, f (x₁ + x₂ + x₃ + x₄ + x₅) = f x₁ + f x₂ + f x₃ + f x₄ + f x₅ - 8

theorem f_zero_equals_two (f : ℝ → ℝ) (h : f_property f) : f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_equals_two_l3554_355471


namespace NUMINAMATH_CALUDE_right_triangle_probability_l3554_355467

/-- A 3x3 grid with 16 vertices -/
structure Grid :=
  (vertices : Finset (ℕ × ℕ))
  (is_3x3 : vertices.card = 16)

/-- Three vertices from the grid -/
structure TripleOfVertices (g : Grid) :=
  (v₁ v₂ v₃ : ℕ × ℕ)
  (v₁_in : v₁ ∈ g.vertices)
  (v₂_in : v₂ ∈ g.vertices)
  (v₃_in : v₃ ∈ g.vertices)
  (distinct : v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃)

/-- Predicate to check if three vertices form a right triangle -/
def is_right_triangle (t : TripleOfVertices g) : Prop :=
  sorry

/-- The probability of forming a right triangle -/
def probability_right_triangle (g : Grid) : ℚ :=
  sorry

/-- The main theorem -/
theorem right_triangle_probability (g : Grid) :
  probability_right_triangle g = 9 / 35 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_probability_l3554_355467


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3554_355443

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 210 * r = b ∧ b * r = 135 / 56) →
  b = 22.5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3554_355443


namespace NUMINAMATH_CALUDE_original_egg_count_l3554_355484

/-- Given a jar of eggs, prove that the original number of eggs is 27 
    when 7 eggs are removed and 20 eggs remain. -/
theorem original_egg_count (removed : ℕ) (remaining : ℕ) : removed = 7 → remaining = 20 → removed + remaining = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_egg_count_l3554_355484


namespace NUMINAMATH_CALUDE_current_batting_average_l3554_355482

/-- Represents a cricket player's batting statistics -/
structure BattingStats where
  matches_played : ℕ
  total_runs : ℕ

/-- Calculates the batting average -/
def batting_average (stats : BattingStats) : ℚ :=
  stats.total_runs / stats.matches_played

/-- The theorem statement -/
theorem current_batting_average 
  (current_stats : BattingStats)
  (next_match_runs : ℕ)
  (new_average : ℚ)
  (h1 : current_stats.matches_played = 6)
  (h2 : batting_average 
    ⟨current_stats.matches_played + 1, current_stats.total_runs + next_match_runs⟩ = new_average)
  (h3 : next_match_runs = 78)
  (h4 : new_average = 54)
  : batting_average current_stats = 50 := by
  sorry

end NUMINAMATH_CALUDE_current_batting_average_l3554_355482


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l3554_355499

theorem geometric_mean_minimum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^x * 4^y)) :
  x^2 + 2*y^2 ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l3554_355499


namespace NUMINAMATH_CALUDE_food_lasts_14_days_l3554_355412

/-- Represents the amount of food each dog consumes per meal in grams -/
def dog_food_per_meal : List ℕ := [250, 350, 450, 550, 300, 400]

/-- Number of meals per day -/
def meals_per_day : ℕ := 3

/-- Weight of each sack in kilograms -/
def sack_weight_kg : ℕ := 50

/-- Number of sacks -/
def num_sacks : ℕ := 2

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ := 1000

theorem food_lasts_14_days :
  let total_food_per_meal := dog_food_per_meal.sum
  let daily_consumption := total_food_per_meal * meals_per_day
  let total_food := num_sacks * sack_weight_kg * kg_to_g
  (total_food / daily_consumption : ℕ) = 14 := by sorry

end NUMINAMATH_CALUDE_food_lasts_14_days_l3554_355412


namespace NUMINAMATH_CALUDE_delivery_driver_stops_l3554_355404

theorem delivery_driver_stops (initial_stops additional_stops : ℕ) 
  (h1 : initial_stops = 3) 
  (h2 : additional_stops = 4) : 
  initial_stops + additional_stops = 7 := by
  sorry

end NUMINAMATH_CALUDE_delivery_driver_stops_l3554_355404


namespace NUMINAMATH_CALUDE_minimum_balloons_l3554_355478

theorem minimum_balloons (red blue burst_red burst_blue : ℕ) : 
  red = 7 * blue →
  burst_red * 3 = burst_blue →
  burst_red ≥ 1 →
  burst_blue ≥ 1 →
  red + blue ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_minimum_balloons_l3554_355478


namespace NUMINAMATH_CALUDE_remainder_of_double_division_l3554_355468

theorem remainder_of_double_division (x : ℝ) : 
  let q₃ := (x^10 - 1) / (x - 1)
  let r₃ := x^10 - (x - 1) * q₃
  let q₄ := (q₃ - r₃) / (x - 1)
  let r₄ := q₃ - (x - 1) * q₄
  r₄ = 10 := by sorry

end NUMINAMATH_CALUDE_remainder_of_double_division_l3554_355468


namespace NUMINAMATH_CALUDE_min_S_proof_l3554_355431

/-- The number of dice rolled -/
def n : ℕ := 333

/-- The target sum -/
def target_sum : ℕ := 1994

/-- The minimum value of S -/
def min_S : ℕ := 334

/-- The probability of obtaining a sum of k when rolling n standard dice -/
noncomputable def prob_sum (k : ℕ) : ℝ := sorry

theorem min_S_proof :
  (prob_sum target_sum > 0) ∧
  (prob_sum target_sum = prob_sum min_S) ∧
  (∀ S : ℕ, S < min_S → prob_sum target_sum ≠ prob_sum S) :=
sorry

end NUMINAMATH_CALUDE_min_S_proof_l3554_355431


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3554_355477

/-- The distance between the foci of the ellipse x^2/36 + y^2/9 = 5 is 6√15 -/
theorem ellipse_foci_distance :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/36 + y^2/9 = 5}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    ∀ p ∈ ellipse, ‖p - f₁‖ + ‖p - f₂‖ = 2 * Real.sqrt (180 : ℝ) ∧
    ‖f₁ - f₂‖ = 6 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3554_355477


namespace NUMINAMATH_CALUDE_joseph_running_distance_l3554_355432

/-- Calculates the total distance run over a number of days with a given initial distance and daily increase. -/
def totalDistance (initialDistance : ℕ) (dailyIncrease : ℕ) (days : ℕ) : ℕ :=
  (days * (2 * initialDistance + (days - 1) * dailyIncrease)) / 2

/-- Proves that given an initial distance of 900 meters, a daily increase of 200 meters,
    and running for 3 days, the total distance run is 3300 meters. -/
theorem joseph_running_distance :
  totalDistance 900 200 3 = 3300 := by
  sorry

end NUMINAMATH_CALUDE_joseph_running_distance_l3554_355432


namespace NUMINAMATH_CALUDE_digital_signal_probability_l3554_355479

theorem digital_signal_probability (p_receive_0_given_send_0 : ℝ) 
                                   (p_receive_1_given_send_0 : ℝ)
                                   (p_receive_1_given_send_1 : ℝ)
                                   (p_receive_0_given_send_1 : ℝ)
                                   (p_send_0 : ℝ)
                                   (p_send_1 : ℝ)
                                   (h1 : p_receive_0_given_send_0 = 0.9)
                                   (h2 : p_receive_1_given_send_0 = 0.1)
                                   (h3 : p_receive_1_given_send_1 = 0.95)
                                   (h4 : p_receive_0_given_send_1 = 0.05)
                                   (h5 : p_send_0 = 0.5)
                                   (h6 : p_send_1 = 0.5) :
  p_send_0 * p_receive_1_given_send_0 + p_send_1 * p_receive_1_given_send_1 = 0.525 :=
by sorry

end NUMINAMATH_CALUDE_digital_signal_probability_l3554_355479


namespace NUMINAMATH_CALUDE_positive_y_solution_l3554_355489

theorem positive_y_solution (x y z : ℝ) 
  (eq1 : x * y = 8 - 3 * x - 2 * y)
  (eq2 : y * z = 15 - 5 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 4 * z)
  (y_pos : y > 0) :
  y = 4 := by
sorry

end NUMINAMATH_CALUDE_positive_y_solution_l3554_355489


namespace NUMINAMATH_CALUDE_card_value_decrease_l3554_355408

/-- Proves that if a value decreases by x% in the first year and 10% in the second year, 
    and the total decrease over two years is 37%, then x = 30. -/
theorem card_value_decrease (x : ℝ) : 
  (1 - x / 100) * (1 - 0.1) = 1 - 0.37 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_card_value_decrease_l3554_355408


namespace NUMINAMATH_CALUDE_zeros_properties_l3554_355444

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

-- State the theorem
theorem zeros_properties (k α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < 2)
  (h4 : f k α = 0) (h5 : f k β = 0) :
  (-7/2 < k ∧ k < -1) ∧ (1/α + 1/β < 4) := by
  sorry

end NUMINAMATH_CALUDE_zeros_properties_l3554_355444


namespace NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l3554_355427

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_isosceles_triangle (a b c : ℝ) (h1 : a = 4) (h2 : b = 4) (h3 : c = 3) :
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = (64 / 13.75) * π :=
sorry

end NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l3554_355427


namespace NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l3554_355483

theorem cubic_equation_sum_of_cubes (a b c : ℝ) : 
  (a - Real.rpow 20 (1/3 : ℝ)) * (a - Real.rpow 70 (1/3 : ℝ)) * (a - Real.rpow 170 (1/3 : ℝ)) = 1/2 →
  (b - Real.rpow 20 (1/3 : ℝ)) * (b - Real.rpow 70 (1/3 : ℝ)) * (b - Real.rpow 170 (1/3 : ℝ)) = 1/2 →
  (c - Real.rpow 20 (1/3 : ℝ)) * (c - Real.rpow 70 (1/3 : ℝ)) * (c - Real.rpow 170 (1/3 : ℝ)) = 1/2 →
  a ≠ b → b ≠ c → a ≠ c →
  a^3 + b^3 + c^3 = 260.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l3554_355483


namespace NUMINAMATH_CALUDE_price_of_zinc_l3554_355414

/-- Given the price of copper, the total weight of brass, the selling price of brass,
    and the amount of copper used, calculate the price of zinc per pound. -/
theorem price_of_zinc 
  (price_copper : ℚ)
  (total_weight : ℚ)
  (selling_price : ℚ)
  (copper_used : ℚ)
  (h1 : price_copper = 65/100)
  (h2 : total_weight = 70)
  (h3 : selling_price = 45/100)
  (h4 : copper_used = 30)
  : ∃ (price_zinc : ℚ), price_zinc = 30/100 := by
  sorry

#check price_of_zinc

end NUMINAMATH_CALUDE_price_of_zinc_l3554_355414


namespace NUMINAMATH_CALUDE_original_number_proof_l3554_355411

theorem original_number_proof (x : ℝ) : x * 1.4 = 1680 ↔ x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3554_355411


namespace NUMINAMATH_CALUDE_transformations_result_l3554_355452

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotates a point 180° around the x-axis -/
def rotateX (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

/-- Reflects a point through the xy-plane -/
def reflectXY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Reflects a point through the yz-plane -/
def reflectYZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Applies the sequence of transformations to a point -/
def applyTransformations (p : Point3D) : Point3D :=
  reflectYZ (rotateX (reflectYZ (reflectXY (rotateX p))))

theorem transformations_result :
  applyTransformations { x := 1, y := 1, z := 1 } = { x := 1, y := 1, z := -1 } := by
  sorry


end NUMINAMATH_CALUDE_transformations_result_l3554_355452


namespace NUMINAMATH_CALUDE_ellipse_condition_l3554_355469

def represents_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  ∀ (x y : ℝ), x^2 / (5 - m) + y^2 / (m + 3) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_condition (m : ℝ) : 
  represents_ellipse m → m > -3 ∧ m < 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3554_355469


namespace NUMINAMATH_CALUDE_expected_ones_three_dice_l3554_355466

/-- A standard die with 6 sides -/
def StandardDie : Type := Fin 6

/-- The probability of rolling a 1 on a standard die -/
def probOne : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def probNotOne : ℚ := 5 / 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expectedOnes : ℚ := 1 / 2

/-- Theorem stating that the expected number of 1's when rolling three standard dice is 1/2 -/
theorem expected_ones_three_dice :
  (numDice : ℚ) * probOne = expectedOnes :=
sorry

end NUMINAMATH_CALUDE_expected_ones_three_dice_l3554_355466


namespace NUMINAMATH_CALUDE_expression_value_l3554_355462

theorem expression_value (x y : ℝ) (h : x + y = 3) : 2*x + 2*y - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3554_355462


namespace NUMINAMATH_CALUDE_max_correct_answers_l3554_355413

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) : 
  total_questions = 25 →
  correct_points = 6 →
  incorrect_points = -3 →
  total_score = 60 →
  (∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_points + incorrect * incorrect_points = total_score) →
  (∀ (correct : ℕ),
    (∃ (incorrect unanswered : ℕ),
      correct + incorrect + unanswered = total_questions ∧
      correct * correct_points + incorrect * incorrect_points = total_score) →
    correct ≤ 15) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l3554_355413


namespace NUMINAMATH_CALUDE_complement_I_M_l3554_355421

def M : Set ℕ := {0, 1}
def I : Set ℕ := {0, 1, 2, 3, 4, 5}

theorem complement_I_M : (I \ M) = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_I_M_l3554_355421


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3554_355437

theorem container_volume_ratio (A B : ℝ) (h1 : A > 0) (h2 : B > 0) : 
  (2/3 * A = 1/2 * B) → (A / B = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3554_355437


namespace NUMINAMATH_CALUDE_sum_of_digits_theorem_l3554_355497

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- State the theorem
theorem sum_of_digits_theorem (n : ℕ) :
  sum_of_digits n = 351 → sum_of_digits (n + 1) = 352 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_theorem_l3554_355497


namespace NUMINAMATH_CALUDE_janes_age_l3554_355418

theorem janes_age (joe_age jane_age : ℕ) 
  (sum_of_ages : joe_age + jane_age = 54)
  (age_difference : joe_age - jane_age = 22) : 
  jane_age = 16 := by
sorry

end NUMINAMATH_CALUDE_janes_age_l3554_355418


namespace NUMINAMATH_CALUDE_no_real_solutions_for_inequality_l3554_355460

theorem no_real_solutions_for_inequality :
  ¬∃ (x : ℝ), x^2 - 4*x + 4 < 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_inequality_l3554_355460


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3554_355433

theorem cow_chicken_problem (num_cows num_chickens : ℕ) : 
  (4 * num_cows + 2 * num_chickens = 2 * (num_cows + num_chickens) + 14) → 
  num_cows = 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3554_355433


namespace NUMINAMATH_CALUDE_derivative_sin_cos_product_l3554_355424

open Real

theorem derivative_sin_cos_product (x : ℝ) : 
  deriv (λ x => sin x * cos x) x = cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_product_l3554_355424


namespace NUMINAMATH_CALUDE_product_xy_equals_one_l3554_355463

theorem product_xy_equals_one (x y : ℝ) (h_distinct : x ≠ y) 
    (h_eq : (1 / (1 + x^2)) + (1 / (1 + y^2)) = 2 / (1 + x*y)) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_one_l3554_355463


namespace NUMINAMATH_CALUDE_cost_calculation_l3554_355402

/-- The cost of buying pens and notebooks -/
def total_cost (pen_price notebook_price : ℝ) : ℝ :=
  5 * pen_price + 8 * notebook_price

/-- Theorem: The total cost of 5 pens at 'a' yuan each and 8 notebooks at 'b' yuan each is 5a + 8b yuan -/
theorem cost_calculation (a b : ℝ) : total_cost a b = 5 * a + 8 * b := by
  sorry

end NUMINAMATH_CALUDE_cost_calculation_l3554_355402


namespace NUMINAMATH_CALUDE_inequality_proof_l3554_355430

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let H := 3 / (1/a + 1/b + 1/c)
  let G := (a * b * c) ^ (1/3)
  let A := (a + b + c) / 3
  let Q := Real.sqrt ((a^2 + b^2 + c^2) / 3)
  (A * G) / (Q * H) ≥ (27/32) ^ (1/6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3554_355430


namespace NUMINAMATH_CALUDE_min_abs_sum_l3554_355419

theorem min_abs_sum (x : ℝ) : 
  |x + 2| + |x + 4| + |x + 5| ≥ 3 ∧ ∃ y : ℝ, |y + 2| + |y + 4| + |y + 5| = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_sum_l3554_355419


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l3554_355458

def A : ℕ := 123456
def B : ℕ := 171717
def M : ℕ := 1000003
def N : ℕ := 538447

theorem multiplicative_inverse_modulo :
  (A * B * N) % M = 1 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l3554_355458


namespace NUMINAMATH_CALUDE_garden_transformation_l3554_355496

/-- Represents a rectangular garden --/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden --/
structure SquareGarden where
  side : ℝ

/-- Calculates the perimeter of a rectangular garden --/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.width)

/-- Calculates the area of a rectangular garden --/
def areaRectangular (garden : RectangularGarden) : ℝ :=
  garden.length * garden.width

/-- Calculates the area of a square garden --/
def areaSquare (garden : SquareGarden) : ℝ :=
  garden.side * garden.side

/-- Theorem: Changing a 60x20 rectangular garden to a square with the same perimeter
    results in a 40x40 square garden and increases the area by 400 square feet --/
theorem garden_transformation (original : RectangularGarden) 
    (h1 : original.length = 60)
    (h2 : original.width = 20) :
    ∃ (new : SquareGarden),
      perimeter original = 4 * new.side ∧
      new.side = 40 ∧
      areaSquare new - areaRectangular original = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_transformation_l3554_355496


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3554_355425

/-- An arithmetic sequence of integers -/
def arithmeticSequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) :
  arithmeticSequence b d →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3554_355425


namespace NUMINAMATH_CALUDE_sine_equality_l3554_355495

theorem sine_equality (α β γ τ : ℝ) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0 ∧ τ > 0) 
  (h_eq : ∀ x, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (τ * x)) : 
  α = γ ∨ α = τ := by
sorry

end NUMINAMATH_CALUDE_sine_equality_l3554_355495


namespace NUMINAMATH_CALUDE_two_digit_number_concatenation_l3554_355453

/-- A two-digit number is an integer between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem two_digit_number_concatenation (x y : ℕ) 
  (hx : TwoDigitNumber x) (hy : TwoDigitNumber y) :
  ∃ (n : ℕ), n = 100 * x + y ∧ 1000 ≤ n ∧ n ≤ 9999 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_concatenation_l3554_355453


namespace NUMINAMATH_CALUDE_division_remainder_l3554_355450

theorem division_remainder (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 31 = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3554_355450


namespace NUMINAMATH_CALUDE_people_joined_line_l3554_355410

theorem people_joined_line (initial : ℕ) (left : ℕ) (final : ℕ) : 
  initial = 30 → left = 10 → final = 25 → final - (initial - left) = 5 := by
  sorry

end NUMINAMATH_CALUDE_people_joined_line_l3554_355410


namespace NUMINAMATH_CALUDE_smallest_square_ending_644_l3554_355454

theorem smallest_square_ending_644 :
  ∀ n : ℕ+, n.val < 194 → (n.val ^ 2) % 1000 ≠ 644 ∧ (194 ^ 2) % 1000 = 644 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_ending_644_l3554_355454


namespace NUMINAMATH_CALUDE_sheep_purchase_l3554_355487

/-- Calculates the number of sheep Mary needs to buy to have 69 fewer sheep than Bob -/
theorem sheep_purchase (mary_initial : ℕ) (bob_multiplier : ℕ) (bob_additional : ℕ) (target_difference : ℕ) : 
  mary_initial = 300 →
  bob_multiplier = 2 →
  bob_additional = 35 →
  target_difference = 69 →
  (mary_initial + (bob_multiplier * mary_initial + bob_additional - target_difference - mary_initial)) = 566 :=
by sorry

end NUMINAMATH_CALUDE_sheep_purchase_l3554_355487


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l3554_355448

theorem multiplication_puzzle :
  ∀ (A B C D : ℕ),
    A < 10 → B < 10 → C < 10 → D < 10 →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    C ≠ 0 → D ≠ 0 →
    100 * A + 10 * B + 1 = (10 * C + D) * (100 * C + D) →
    A + B = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l3554_355448


namespace NUMINAMATH_CALUDE_total_income_is_53_l3554_355440

def tshirt_price : ℕ := 5
def pants_price : ℕ := 4
def skirt_price : ℕ := 6
def refurbished_tshirt_price : ℕ := tshirt_price / 2

def tshirts_sold : ℕ := 2
def pants_sold : ℕ := 1
def skirts_sold : ℕ := 4
def refurbished_tshirts_sold : ℕ := 6

def total_income : ℕ := 
  tshirts_sold * tshirt_price + 
  pants_sold * pants_price + 
  skirts_sold * skirt_price + 
  refurbished_tshirts_sold * refurbished_tshirt_price

theorem total_income_is_53 : total_income = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_income_is_53_l3554_355440


namespace NUMINAMATH_CALUDE_second_train_speed_l3554_355420

/-- Given two trains traveling towards each other, prove that the speed of the second train is 16 km/hr -/
theorem second_train_speed
  (speed_train1 : ℝ)
  (total_distance : ℝ)
  (distance_difference : ℝ)
  (h1 : speed_train1 = 20)
  (h2 : total_distance = 630)
  (h3 : distance_difference = 70)
  : ∃ (speed_train2 : ℝ), speed_train2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l3554_355420


namespace NUMINAMATH_CALUDE_tangent_line_at_one_tangent_lines_through_one_l3554_355449

noncomputable section

-- Define the function f(x) = x^3 + a*ln(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * Real.log x

-- Part I
theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  ∃ (m b : ℝ), m * 1 - f a 1 + b = 0 ∧
    ∀ x, m * x - (f a x) + b = 0 ↔ 4 * x - (f a x) - 3 = 0 :=
sorry

-- Part II
theorem tangent_lines_through_one (a : ℝ) (h : a = 0) :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (m₁ * 1 - f a 1 + b₁ = 0 ∧ ∀ x, m₁ * x - (f a x) + b₁ = 0 ↔ 3 * x - (f a x) - 2 = 0) ∧
    (m₂ * 1 - f a 1 + b₂ = 0 ∧ ∀ x, m₂ * x - (f a x) + b₂ = 0 ↔ 3 * x - 4 * (f a x) + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_tangent_lines_through_one_l3554_355449


namespace NUMINAMATH_CALUDE_track_circumference_l3554_355446

/-- Represents a circular track with two runners -/
structure CircularTrack where
  /-- The circumference of the track in yards -/
  circumference : ℝ
  /-- The distance B travels before the first meeting in yards -/
  first_meeting_distance : ℝ
  /-- The distance A has left to complete a lap at the second meeting in yards -/
  second_meeting_remaining : ℝ

/-- The theorem stating the circumference of the track given the conditions -/
theorem track_circumference (track : CircularTrack)
  (h1 : track.first_meeting_distance = 150)
  (h2 : track.second_meeting_remaining = 90)
  (h3 : track.first_meeting_distance < track.circumference / 2)
  (h4 : track.second_meeting_remaining < track.circumference) :
  track.circumference = 720 := by
  sorry

#check track_circumference

end NUMINAMATH_CALUDE_track_circumference_l3554_355446


namespace NUMINAMATH_CALUDE_intersection_M_N_l3554_355442

-- Define set M
def M : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ 7}

-- Define set N
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | (3 < x ∧ x ≤ 7) ∨ (-4 ≤ x ∧ x < -2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3554_355442


namespace NUMINAMATH_CALUDE_squarable_numbers_l3554_355434

def isSquarable (n : ℕ) : Prop :=
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ (i : Fin n), ∃ (k : ℕ), (p i).val + i.val + 1 = k^2

theorem squarable_numbers : 
  (¬ isSquarable 7) ∧ 
  (isSquarable 9) ∧ 
  (¬ isSquarable 11) ∧ 
  (isSquarable 15) := by sorry

end NUMINAMATH_CALUDE_squarable_numbers_l3554_355434


namespace NUMINAMATH_CALUDE_average_departure_time_l3554_355429

def minutes_after_noon (hour : ℕ) (minute : ℕ) : ℕ :=
  (hour - 12) * 60 + minute

def passing_time : ℕ := minutes_after_noon 15 11
def alice_arrival : ℕ := minutes_after_noon 15 19
def bob_arrival : ℕ := minutes_after_noon 15 29

theorem average_departure_time :
  let alice_departure := alice_arrival - (alice_arrival - passing_time)
  let bob_departure := bob_arrival - (bob_arrival - passing_time)
  (alice_departure + bob_departure) / 2 = 179 := by
sorry

end NUMINAMATH_CALUDE_average_departure_time_l3554_355429


namespace NUMINAMATH_CALUDE_yoga_time_calculation_l3554_355475

/-- Calculates the yoga time given exercise ratios and bicycle riding time -/
theorem yoga_time_calculation (bicycle_time : ℚ) : 
  bicycle_time = 12 → (40 : ℚ) / 3 = 
    2 * (2 * bicycle_time / 3 + bicycle_time) / 3 := by
  sorry

#eval (40 : ℚ) / 3

end NUMINAMATH_CALUDE_yoga_time_calculation_l3554_355475


namespace NUMINAMATH_CALUDE_inequality_range_l3554_355435

theorem inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 - Real.log x / Real.log m < 0) ↔ 
  m ∈ Set.Icc (1/16) 1 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l3554_355435


namespace NUMINAMATH_CALUDE_cube_sum_is_26_l3554_355494

/-- Properties of a cube -/
structure Cube where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- Definition of a standard cube -/
def standardCube : Cube :=
  { faces := 6
  , edges := 12
  , vertices := 8 }

/-- Theorem: The sum of faces, edges, and vertices of a cube is 26 -/
theorem cube_sum_is_26 (c : Cube) (h : c = standardCube) : 
  c.faces + c.edges + c.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_is_26_l3554_355494


namespace NUMINAMATH_CALUDE_area_ratio_of_inner_triangle_l3554_355492

/-- Given a triangle with area T, if we divide each side of the triangle in the ratio of 1:2
    (starting from each vertex) and form a new triangle by connecting these points,
    the area of the new triangle S is related to the area of the original triangle T
    by the equation: S / T = 1 / 9 -/
theorem area_ratio_of_inner_triangle (T : ℝ) (S : ℝ) (h : T > 0) :
  (∀ (side : ℝ), ∃ (new_side : ℝ), new_side = side / 3) →
  S / T = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_inner_triangle_l3554_355492


namespace NUMINAMATH_CALUDE_intersection_A_B_l3554_355438

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2^x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ici 1 ∩ Set.Iio 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3554_355438


namespace NUMINAMATH_CALUDE_pearl_division_l3554_355457

theorem pearl_division (n : ℕ) : 
  (n > 0) →
  (n % 8 = 6) → 
  (n % 7 = 5) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 9 = 1) := by
sorry

end NUMINAMATH_CALUDE_pearl_division_l3554_355457


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l3554_355461

-- Define the types for planes and lines
variable (Plane Line : Type*)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_transitive
  (α β : Plane) (l : Line)
  (h1 : perpendicular l α)
  (h2 : parallel α β) :
  perpendicular l β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l3554_355461


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l3554_355422

/-- Proves that the number of meters of cloth sold is 45 given the specified conditions -/
theorem cloth_sale_problem (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  total_selling_price = 4500 →
  profit_per_meter = 14 →
  cost_price_per_meter = 86 →
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 45 := by
  sorry

#check cloth_sale_problem

end NUMINAMATH_CALUDE_cloth_sale_problem_l3554_355422


namespace NUMINAMATH_CALUDE_sample_size_from_model_a_l3554_355405

/-- Represents the ratio of quantities for models A, B, and C -/
structure ProductRatio :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Represents a stratified sample -/
structure StratifiedSample :=
  (size : ℕ) (model_a_count : ℕ)

/-- Theorem: Given the product ratio and model A count in a stratified sample, 
    prove the total sample size -/
theorem sample_size_from_model_a
  (ratio : ProductRatio)
  (sample : StratifiedSample)
  (h_ratio : ratio = ⟨3, 4, 7⟩)
  (h_model_a : sample.model_a_count = 15) :
  sample.size = 70 :=
sorry

end NUMINAMATH_CALUDE_sample_size_from_model_a_l3554_355405


namespace NUMINAMATH_CALUDE_class_attendance_multiple_l3554_355476

/-- Proves that the largest whole number multiple of students present yesterday
    that is less than or equal to 90% of students attending today is 0. -/
theorem class_attendance_multiple (total_registered : ℕ) (present_yesterday : ℕ) (absent_today : ℕ)
  (h1 : total_registered = 156)
  (h2 : present_yesterday = 70)
  (h3 : absent_today = 30)
  (h4 : present_yesterday > absent_today) :
  (∀ n : ℕ, n * present_yesterday ≤ (present_yesterday - absent_today) * 9 / 10 → n = 0) :=
by sorry

end NUMINAMATH_CALUDE_class_attendance_multiple_l3554_355476


namespace NUMINAMATH_CALUDE_chen_recorded_steps_l3554_355436

/-- The standard number of steps for the walking activity -/
def standard : ℕ := 5000

/-- The function to calculate the recorded steps -/
def recorded_steps (actual_steps : ℕ) : ℤ :=
  (actual_steps : ℤ) - standard

/-- Theorem stating that 4800 actual steps should be recorded as -200 -/
theorem chen_recorded_steps :
  recorded_steps 4800 = -200 := by sorry

end NUMINAMATH_CALUDE_chen_recorded_steps_l3554_355436


namespace NUMINAMATH_CALUDE_magic_polynomial_bound_l3554_355491

open Polynomial
open Nat

theorem magic_polynomial_bound (n : ℕ) (P : Polynomial ℚ) 
  (h_deg : degree P = n) (h_irr : Irreducible P) :
  ∃ (s : Finset (Polynomial ℚ)), 
    (∀ Q ∈ s, degree Q < n ∧ (P ∣ (P.comp Q))) ∧ 
    (∀ Q : Polynomial ℚ, degree Q < n → (P ∣ (P.comp Q)) → Q ∈ s) ∧
    s.card ≤ n := by
  sorry

end NUMINAMATH_CALUDE_magic_polynomial_bound_l3554_355491


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3554_355401

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + a*x₁ + 5 = 0 ∧ 
    x₂^2 + a*x₂ + 5 = 0 ∧ 
    x₁^2 + 250/(19*x₂^3) = x₂^2 + 250/(19*x₁^3)) → 
  a = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3554_355401


namespace NUMINAMATH_CALUDE_cabbage_production_increase_l3554_355473

theorem cabbage_production_increase (garden_size : ℕ) (this_year_production : ℕ) : 
  garden_size * garden_size = this_year_production →
  this_year_production = 9409 →
  this_year_production - (garden_size - 1) * (garden_size - 1) = 193 := by
sorry

end NUMINAMATH_CALUDE_cabbage_production_increase_l3554_355473


namespace NUMINAMATH_CALUDE_sum_product_inequality_l3554_355451

theorem sum_product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l3554_355451


namespace NUMINAMATH_CALUDE_conference_center_occupancy_l3554_355455

theorem conference_center_occupancy (rooms : Nat) (capacity : Nat) (occupancy_ratio : Rat) : 
  rooms = 12 →
  capacity = 150 →
  occupancy_ratio = 5/7 →
  (rooms * capacity * occupancy_ratio).floor = 1285 := by
  sorry

end NUMINAMATH_CALUDE_conference_center_occupancy_l3554_355455


namespace NUMINAMATH_CALUDE_stations_between_cities_l3554_355409

theorem stations_between_cities (n : ℕ) : 
  (((n + 2) * (n + 1)) / 2 = 132) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_stations_between_cities_l3554_355409


namespace NUMINAMATH_CALUDE_number_of_nickels_l3554_355400

def quarter_value : Rat := 25 / 100
def dime_value : Rat := 10 / 100
def nickel_value : Rat := 5 / 100
def penny_value : Rat := 1 / 100

def num_quarters : Nat := 10
def num_dimes : Nat := 3
def num_pennies : Nat := 200
def total_amount : Rat := 5

theorem number_of_nickels : 
  ∃ (num_nickels : Nat), 
    (num_quarters : Nat) * quarter_value + 
    (num_dimes : Nat) * dime_value + 
    (num_nickels : Nat) * nickel_value + 
    (num_pennies : Nat) * penny_value = total_amount ∧ 
    num_nickels = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_nickels_l3554_355400


namespace NUMINAMATH_CALUDE_box_triangle_area_theorem_l3554_355464

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℚ

/-- Calculates the area of the triangle formed by the center points of three faces meeting at a corner -/
def triangleArea (d : BoxDimensions) : ℝ :=
  sorry

/-- Checks if two integers are relatively prime -/
def relativelyPrime (m n : ℕ) : Prop :=
  sorry

theorem box_triangle_area_theorem 
  (d : BoxDimensions)
  (m n : ℕ)
  (h1 : d.width = 15)
  (h2 : d.length = 20)
  (h3 : d.height = m / n)
  (h4 : relativelyPrime m n)
  (h5 : triangleArea d = 40) :
  m + n = 69 :=
sorry

end NUMINAMATH_CALUDE_box_triangle_area_theorem_l3554_355464


namespace NUMINAMATH_CALUDE_multiply_82519_9999_l3554_355485

theorem multiply_82519_9999 : 82519 * 9999 = 825107481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_82519_9999_l3554_355485


namespace NUMINAMATH_CALUDE_eccentricity_of_conic_l3554_355407

/-- The conic section defined by the equation 6x^2 + 4xy + 9y^2 = 20 -/
def conic_section (x y : ℝ) : Prop :=
  6 * x^2 + 4 * x * y + 9 * y^2 = 20

/-- The eccentricity of a conic section -/
def eccentricity (c : (ℝ → ℝ → Prop)) : ℝ := sorry

/-- Theorem: The eccentricity of the given conic section is √2/2 -/
theorem eccentricity_of_conic : eccentricity conic_section = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_of_conic_l3554_355407


namespace NUMINAMATH_CALUDE_least_positive_angle_solution_l3554_355423

theorem least_positive_angle_solution (θ : Real) : 
  (θ > 0 ∧ θ < 360 ∧ Real.cos (10 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) + Real.sin (θ * Real.pi / 180)) →
  θ = 80 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_solution_l3554_355423


namespace NUMINAMATH_CALUDE_babies_age_sum_l3554_355417

def lioness_age : ℕ := 12

theorem babies_age_sum (hyena_age : ℕ) (lioness_baby_age : ℕ) (hyena_baby_age : ℕ)
  (h1 : lioness_age = 2 * hyena_age)
  (h2 : lioness_baby_age = lioness_age / 2)
  (h3 : hyena_baby_age = hyena_age / 2) :
  lioness_baby_age + 5 + hyena_baby_age + 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_babies_age_sum_l3554_355417


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3554_355465

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → 
  ∃ (m b : ℝ), (y = m * x + b) ∧ (m = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3554_355465


namespace NUMINAMATH_CALUDE_not_right_triangle_l3554_355472

theorem not_right_triangle (a b c : ℝ) (h : a = 3 ∧ b = 5 ∧ c = 7) : 
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l3554_355472
