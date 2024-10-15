import Mathlib

namespace NUMINAMATH_CALUDE_premium_probability_option2_higher_price_probability_relationship_l1704_170475

-- Define the grades of oranges
inductive Grade : Type
| Premium : Grade
| Special : Grade
| Superior : Grade
| FirstGrade : Grade

-- Define the distribution of boxes
def total_boxes : ℕ := 100
def premium_boxes : ℕ := 40
def special_boxes : ℕ := 30
def superior_boxes : ℕ := 10
def first_grade_boxes : ℕ := 20

-- Define the pricing options
def option1_price : ℚ := 27
def premium_price : ℚ := 36
def special_price : ℚ := 30
def superior_price : ℚ := 24
def first_grade_price : ℚ := 18

-- Theorem 1: Probability of selecting a premium grade box
theorem premium_probability : 
  (premium_boxes : ℚ) / total_boxes = 2 / 5 := by sorry

-- Theorem 2: Average price of Option 2 is higher than Option 1
theorem option2_higher_price :
  (premium_price * premium_boxes + special_price * special_boxes + 
   superior_price * superior_boxes + first_grade_price * first_grade_boxes) / 
  total_boxes > option1_price := by sorry

-- Define probabilities for selecting 3 boxes with different grades
def p₁ : ℚ := 1465 / 1617  -- from 100 boxes
def p₂ : ℚ := 53 / 57      -- from 20 boxes in stratified sampling

-- Theorem 3: Relationship between p₁ and p₂
theorem probability_relationship : p₁ < p₂ := by sorry

end NUMINAMATH_CALUDE_premium_probability_option2_higher_price_probability_relationship_l1704_170475


namespace NUMINAMATH_CALUDE_parallel_condition_l1704_170465

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Main theorem
theorem parallel_condition 
  (α β : Plane) 
  (m : Line) 
  (h_distinct : α ≠ β) 
  (h_m_in_α : line_in_plane m α) :
  (∀ α β : Plane, plane_parallel α β → line_parallel_plane m β) ∧ 
  (∃ α β : Plane, line_parallel_plane m β ∧ ¬plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l1704_170465


namespace NUMINAMATH_CALUDE_compute_fraction_square_l1704_170403

theorem compute_fraction_square : 6 * (3 / 7)^2 = 54 / 49 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_square_l1704_170403


namespace NUMINAMATH_CALUDE_retail_profit_calculation_l1704_170411

/-- Represents the pricing and profit calculations for a retail scenario -/
def RetailScenario (costPrice : ℝ) : Prop :=
  let markupPercentage : ℝ := 65
  let discountPercentage : ℝ := 25
  let actualProfitPercentage : ℝ := 23.75
  let markedPrice : ℝ := costPrice * (1 + markupPercentage / 100)
  let sellingPrice : ℝ := markedPrice * (1 - discountPercentage / 100)
  let actualProfit : ℝ := sellingPrice - costPrice
  let intendedProfit : ℝ := markedPrice - costPrice
  (actualProfit / costPrice * 100 = actualProfitPercentage) ∧
  (intendedProfit / costPrice * 100 = markupPercentage)

/-- Theorem stating that under the given retail scenario, the initially expected profit percentage is 65% -/
theorem retail_profit_calculation (costPrice : ℝ) (h : costPrice > 0) :
  RetailScenario costPrice → 65 = (65 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_retail_profit_calculation_l1704_170411


namespace NUMINAMATH_CALUDE_ship_length_observation_l1704_170474

/-- The length of a ship observed from shore --/
theorem ship_length_observation (same_direction : ℝ) (opposite_direction : ℝ) :
  same_direction = 200 →
  opposite_direction = 40 →
  (∃ ship_length : ℝ, (ship_length = 100 ∨ ship_length = 200 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ship_length_observation_l1704_170474


namespace NUMINAMATH_CALUDE_test_results_problem_l1704_170419

/-- Represents the number of questions a person got wrong on a test. -/
structure TestResult where
  wrong : Nat

/-- Represents the test results for Emily, Felix, Grace, and Henry. -/
structure GroupTestResults where
  emily : TestResult
  felix : TestResult
  grace : TestResult
  henry : TestResult

/-- The theorem statement for the test results problem. -/
theorem test_results_problem (results : GroupTestResults) : 
  (results.emily.wrong + results.felix.wrong + 4 = results.grace.wrong + results.henry.wrong) →
  (results.emily.wrong + results.henry.wrong = results.felix.wrong + results.grace.wrong + 8) →
  (results.grace.wrong = 6) →
  (results.emily.wrong = 8) := by
  sorry

#check test_results_problem

end NUMINAMATH_CALUDE_test_results_problem_l1704_170419


namespace NUMINAMATH_CALUDE_unique_triples_l1704_170460

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_triples : 
  ∀ a b c : ℕ, 
    (is_prime (a^2 - 23)) → 
    (is_prime (b^2 - 23)) → 
    ((a^2 - 23) * (b^2 - 23) = c^2 - 23) → 
    ((a = 5 ∧ b = 6 ∧ c = 7) ∨ (a = 6 ∧ b = 5 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_unique_triples_l1704_170460


namespace NUMINAMATH_CALUDE_final_amount_correct_l1704_170479

def total_income : ℝ := 1000000

def children_share : ℝ := 0.2
def num_children : ℕ := 3
def wife_share : ℝ := 0.3
def orphan_donation_rate : ℝ := 0.05

def amount_left : ℝ := 
  total_income * (1 - children_share * num_children - wife_share) * (1 - orphan_donation_rate)

theorem final_amount_correct : amount_left = 95000 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_correct_l1704_170479


namespace NUMINAMATH_CALUDE_gcd_problem_l1704_170472

theorem gcd_problem (a : ℕ+) : (Nat.gcd (Nat.gcd a 16) (Nat.gcd 18 a) = 2) → (a = 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_problem_l1704_170472


namespace NUMINAMATH_CALUDE_specific_room_surface_area_l1704_170449

/-- Calculates the interior surface area of a cubic room with a central cubical hole -/
def interior_surface_area (room_edge : ℝ) (hole_edge : ℝ) : ℝ :=
  6 * room_edge^2 - 3 * hole_edge^2

/-- Theorem stating the interior surface area of a specific cubic room with a hole -/
theorem specific_room_surface_area :
  interior_surface_area 10 2 = 588 := by
  sorry

#check specific_room_surface_area

end NUMINAMATH_CALUDE_specific_room_surface_area_l1704_170449


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1704_170407

/-- Represents the number of athletes in a sample -/
structure Sample where
  male : ℕ
  female : ℕ

/-- Represents the total population of athletes -/
structure Population where
  male : ℕ
  female : ℕ

/-- Checks if a sample is stratified with respect to a population -/
def isStratifiedSample (pop : Population) (samp : Sample) : Prop :=
  samp.male * pop.female = samp.female * pop.male

/-- The main theorem to prove -/
theorem stratified_sample_size 
  (pop : Population) 
  (samp : Sample) 
  (h1 : pop.male = 42)
  (h2 : pop.female = 30)
  (h3 : samp.female = 5)
  (h4 : isStratifiedSample pop samp) :
  samp.male + samp.female = 12 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_size_l1704_170407


namespace NUMINAMATH_CALUDE_amoeba_count_10_days_l1704_170438

/-- The number of amoebas in the petri dish after n days -/
def amoeba_count (n : ℕ) : ℕ := 3^n

/-- Theorem stating that the number of amoebas after 10 days is 59049 -/
theorem amoeba_count_10_days : amoeba_count 10 = 59049 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_10_days_l1704_170438


namespace NUMINAMATH_CALUDE_berrys_friday_temperature_l1704_170444

/-- Given Berry's temperatures for 6 days and the average for a week, 
    prove that his temperature on Friday was 99 degrees. -/
theorem berrys_friday_temperature 
  (temps : List ℝ) 
  (h_temps : temps = [99.1, 98.2, 98.7, 99.3, 99.8, 98.9]) 
  (h_avg : (temps.sum + x) / 7 = 99) : x = 99 := by
  sorry

end NUMINAMATH_CALUDE_berrys_friday_temperature_l1704_170444


namespace NUMINAMATH_CALUDE_prob_multiple_13_eq_l1704_170476

/-- Represents a standard deck of 54 cards with 4 suits (1-13) and 2 jokers -/
def Deck : Type := Fin 54

/-- Represents the rank of a card (1-13 for regular cards, 0 for jokers) -/
def rank (card : Deck) : ℕ :=
  if card.val < 52 then
    (card.val % 13) + 1
  else
    0

/-- Shuffles the deck uniformly randomly -/
def shuffle (deck : Deck → α) : Deck → α :=
  sorry

/-- Calculates the score based on the shuffled deck -/
def score (shuffled_deck : Deck → Deck) : ℕ :=
  sorry

/-- Probability that the score is a multiple of 13 -/
def prob_multiple_13 : ℚ :=
  sorry

/-- Main theorem: The probability of the score being a multiple of 13 is 77/689 -/
theorem prob_multiple_13_eq : prob_multiple_13 = 77 / 689 :=
  sorry

end NUMINAMATH_CALUDE_prob_multiple_13_eq_l1704_170476


namespace NUMINAMATH_CALUDE_smallest_sum_of_identical_numbers_l1704_170400

theorem smallest_sum_of_identical_numbers : ∃ (a b c : ℕ), 
  (6036 = 2010 * a) ∧ 
  (6036 = 2012 * b) ∧ 
  (6036 = 2013 * c) ∧ 
  (∀ (n : ℕ) (x y z : ℕ), 
    n > 0 ∧ n < 6036 → 
    ¬(n = 2010 * x ∧ n = 2012 * y ∧ n = 2013 * z)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_identical_numbers_l1704_170400


namespace NUMINAMATH_CALUDE_point_inside_circle_l1704_170495

-- Define the ellipse parameters
variable (a b c : ℝ)
variable (x₁ x₂ : ℝ)

-- Define the conditions
theorem point_inside_circle
  (h_positive : a > 0 ∧ b > 0)
  (h_eccentricity : c / a = 1 / 2)
  (h_ellipse : b^2 = a^2 - c^2)
  (h_roots : x₁ + x₂ = -b / a ∧ x₁ * x₂ = -c / a) :
  x₁^2 + x₂^2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l1704_170495


namespace NUMINAMATH_CALUDE_max_min_x_squared_l1704_170453

def f (x : ℝ) : ℝ := x^2

theorem max_min_x_squared :
  ∃ (max min : ℝ), 
    (∀ x, -3 ≤ x ∧ x ≤ 1 → f x ≤ max) ∧
    (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = max) ∧
    (∀ x, -3 ≤ x ∧ x ≤ 1 → min ≤ f x) ∧
    (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = min) ∧
    max = 9 ∧ min = 0 := by
  sorry

end NUMINAMATH_CALUDE_max_min_x_squared_l1704_170453


namespace NUMINAMATH_CALUDE_cube_sum_ge_triple_product_l1704_170416

theorem cube_sum_ge_triple_product (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^3 + b^3 + c^3 ≥ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ge_triple_product_l1704_170416


namespace NUMINAMATH_CALUDE_kims_candy_bars_l1704_170417

/-- The number of candy bars Kim's dad buys her each week -/
def candy_bars_per_week : ℕ := 2

/-- The number of weeks in the problem -/
def total_weeks : ℕ := 16

/-- The number of candy bars Kim eats in the given period -/
def candy_bars_eaten : ℕ := total_weeks / 4

/-- The number of candy bars Kim has saved after the given period -/
def candy_bars_saved : ℕ := 28

theorem kims_candy_bars : 
  candy_bars_per_week * total_weeks - candy_bars_eaten = candy_bars_saved :=
by sorry

end NUMINAMATH_CALUDE_kims_candy_bars_l1704_170417


namespace NUMINAMATH_CALUDE_zero_in_set_A_l1704_170431

theorem zero_in_set_A : 
  let A : Set ℕ := {0, 1, 2}
  0 ∈ A := by
sorry

end NUMINAMATH_CALUDE_zero_in_set_A_l1704_170431


namespace NUMINAMATH_CALUDE_integral_reciprocal_one_plus_x_squared_l1704_170470

theorem integral_reciprocal_one_plus_x_squared : 
  ∫ (x : ℝ) in (0)..(Real.sqrt 3), 1 / (1 + x^2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_one_plus_x_squared_l1704_170470


namespace NUMINAMATH_CALUDE_video_votes_l1704_170429

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 120 ∧ 
  like_percentage = 72 / 100 →
  ∃ (total_votes : ℕ), 
    (like_percentage * total_votes : ℚ) - ((1 - like_percentage) * total_votes : ℚ) = score ∧
    total_votes = 273 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l1704_170429


namespace NUMINAMATH_CALUDE_coordinate_problem_l1704_170454

theorem coordinate_problem (x₁ y₁ x₂ y₂ : ℕ) : 
  (x₁ > 0) → (y₁ > 0) → (x₂ > 0) → (y₂ > 0) →  -- Positive integer coordinates
  (y₁ > x₁) →  -- Angle OA > 45°
  (x₂ > y₂) →  -- Angle OB < 45°
  (x₂ * y₂ = x₁ * y₁ + 67) →  -- Area difference condition
  (x₁ = 1 ∧ y₁ = 5 ∧ x₂ = 9 ∧ y₂ = 8) := by
sorry

end NUMINAMATH_CALUDE_coordinate_problem_l1704_170454


namespace NUMINAMATH_CALUDE_coin_toss_experiment_l1704_170413

theorem coin_toss_experiment (total_tosses : ℕ) (heads_frequency : ℚ) 
  (h1 : total_tosses = 100)
  (h2 : heads_frequency = 49/100) :
  total_tosses - (total_tosses * heads_frequency).num = 51 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_experiment_l1704_170413


namespace NUMINAMATH_CALUDE_min_soldiers_to_add_l1704_170457

theorem min_soldiers_to_add (N : ℕ) : 
  N % 7 = 2 → N % 12 = 2 → (84 - N % 84) = 82 := by
  sorry

end NUMINAMATH_CALUDE_min_soldiers_to_add_l1704_170457


namespace NUMINAMATH_CALUDE_chords_from_nine_points_l1704_170458

/-- The number of different chords that can be drawn by connecting two points 
    out of n points on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem stating that the number of chords from 9 points is 36 -/
theorem chords_from_nine_points : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_chords_from_nine_points_l1704_170458


namespace NUMINAMATH_CALUDE_largest_prime_factor_l1704_170466

def expression : ℤ := 17^4 + 3 * 17^2 + 2 - 16^4

theorem largest_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression.natAbs ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ expression.natAbs → q ≤ p ∧
  p = 34087 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l1704_170466


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1704_170492

def f (x : ℝ) := 2 * x^2 + 4 * x - 1

theorem max_value_of_f_on_interval : 
  ∃ (c : ℝ), c ∈ Set.Icc (-2) 2 ∧ 
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ f c) ∧
  f c = 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1704_170492


namespace NUMINAMATH_CALUDE_data_instances_eq_720_l1704_170446

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The interval between recordings in seconds -/
def recording_interval : ℕ := 5

/-- The number of data instances recorded in one hour by a device that records every 5 seconds -/
def data_instances : ℕ := 
  (seconds_per_minute * minutes_per_hour) / recording_interval

/-- Theorem: The number of data instances recorded in one hour is 720 -/
theorem data_instances_eq_720 : data_instances = 720 := by
  sorry

end NUMINAMATH_CALUDE_data_instances_eq_720_l1704_170446


namespace NUMINAMATH_CALUDE_exists_integer_point_with_distance_l1704_170496

theorem exists_integer_point_with_distance : ∃ (x y : ℤ),
  (x : ℝ)^2 + (y : ℝ)^2 = 2 * 2017^2 + 2 * 2018^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_point_with_distance_l1704_170496


namespace NUMINAMATH_CALUDE_opposite_numbers_equation_l1704_170424

theorem opposite_numbers_equation (x : ℝ) : 2 * (x - 3) = -(4 * (1 - x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_equation_l1704_170424


namespace NUMINAMATH_CALUDE_magazine_circulation_ratio_l1704_170414

/-- The circulation ratio problem for magazine P -/
theorem magazine_circulation_ratio 
  (avg_circulation : ℝ) -- Average yearly circulation for 1962-1970
  (h : avg_circulation > 0) -- Assumption that circulation is positive
  : (4 * avg_circulation) / (4 * avg_circulation + 9 * avg_circulation) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_magazine_circulation_ratio_l1704_170414


namespace NUMINAMATH_CALUDE_sprint_distance_l1704_170468

/-- Given a constant speed and a duration, calculates the distance traveled. -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that sprinting at 6 miles per hour for 4 hours results in a distance of 24 miles. -/
theorem sprint_distance : distance_traveled 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sprint_distance_l1704_170468


namespace NUMINAMATH_CALUDE_square_area_error_l1704_170401

theorem square_area_error (a : ℝ) (h : a > 0) : 
  let measured_side := a * 1.05
  let actual_area := a ^ 2
  let calculated_area := measured_side ^ 2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.1025 := by sorry

end NUMINAMATH_CALUDE_square_area_error_l1704_170401


namespace NUMINAMATH_CALUDE_unique_divisible_by_792_l1704_170423

/-- Represents a 7-digit number in the form 13xy45z -/
def number (x y z : Nat) : Nat :=
  1300000 + x * 10000 + y * 1000 + 450 + z

/-- Checks if a number is of the form 13xy45z where x, y, z are single digits -/
def isValidForm (n : Nat) : Prop :=
  ∃ x y z, x < 10 ∧ y < 10 ∧ z < 10 ∧ n = number x y z

theorem unique_divisible_by_792 :
  ∃! n, isValidForm n ∧ n % 792 = 0 ∧ n = 1380456 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_792_l1704_170423


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l1704_170440

/-- 
Proves that the number of years until a man's age is twice his son's age is 2,
given that the man is 24 years older than his son and the son's present age is 22 years.
-/
theorem mans_age_twice_sons (
  son_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  son_age = 22 →
  man_age = son_age + 24 →
  man_age + years = 2 * (son_age + years) →
  years = 2 :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l1704_170440


namespace NUMINAMATH_CALUDE_unique_sum_value_l1704_170409

theorem unique_sum_value (n m : ℤ) 
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 := by
sorry

end NUMINAMATH_CALUDE_unique_sum_value_l1704_170409


namespace NUMINAMATH_CALUDE_corrected_mean_l1704_170432

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n = 40 ∧ original_mean = 36 ∧ incorrect_value = 20 ∧ correct_value = 34 →
  (n : ℝ) * original_mean - incorrect_value + correct_value = n * 36.35 :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l1704_170432


namespace NUMINAMATH_CALUDE_factorization_equality_l1704_170488

theorem factorization_equality (a b : ℝ) : a^2 * b - 6 * a * b + 9 * b = b * (a - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1704_170488


namespace NUMINAMATH_CALUDE_part_one_part_two_l1704_170425

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 4 → f a x ≤ 2) : a = 2 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h : 0 ≤ a ∧ a ≤ 3) :
  ∀ x : ℝ, f a (x + a) + f a (x - a) ≥ f a (a * x) - a * f a x := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1704_170425


namespace NUMINAMATH_CALUDE_arithmetic_geometric_means_l1704_170462

theorem arithmetic_geometric_means (a b c x y : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- a, b, c are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- a, b, c are distinct
  2 * b = a + c ∧          -- a, b, c form an arithmetic sequence
  x^2 = a * b ∧            -- x is the geometric mean of a and b
  y^2 = b * c →            -- y is the geometric mean of b and c
  (2 * b^2 = x^2 + y^2) ∧  -- x^2, b^2, y^2 form an arithmetic sequence
  (b^4 ≠ x^2 * y^2)        -- x^2, b^2, y^2 do not form a geometric sequence
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_means_l1704_170462


namespace NUMINAMATH_CALUDE_population_growth_l1704_170427

theorem population_growth (p q : ℕ) (h1 : p^2 + 180 = q^2 + 16) 
  (h2 : ∃ r : ℕ, p^2 + 360 = r^2) : 
  abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 21) < 
  min (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 18))
      (min (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 24))
           (min (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 27))
                (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 30)))) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_l1704_170427


namespace NUMINAMATH_CALUDE_subset_relation_l1704_170410

universe u

theorem subset_relation (A B : Set α) :
  (∃ x, x ∈ B) →
  (∀ y, y ∈ A → y ∈ B) →
  B ⊆ A :=
by sorry

end NUMINAMATH_CALUDE_subset_relation_l1704_170410


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_2023_l1704_170499

-- Define the cycle length of the last two digits of powers of 3
def cycleLengthPowersOf3 : ℕ := 20

-- Define the function that gives the last two digits of 3^n
def lastTwoDigits (n : ℕ) : ℕ := 3^n % 100

-- Define the function that gives the tens digit of a number
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_3_to_2023 :
  tensDigit (lastTwoDigits 2023) = 2 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_2023_l1704_170499


namespace NUMINAMATH_CALUDE_square_perimeter_l1704_170405

theorem square_perimeter (rectangle_perimeter : ℝ) (square_side : ℝ) : 
  (rectangle_perimeter + 4 * square_side) - rectangle_perimeter = 17 →
  4 * square_side = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1704_170405


namespace NUMINAMATH_CALUDE_expression_evaluation_l1704_170493

theorem expression_evaluation (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h : y - z / x ≠ 0) : 
  (x - z / y) / (y - z / x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1704_170493


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_three_composite_reciprocals_l1704_170428

/-- The arithmetic mean of the reciprocals of the first three composite numbers is 13/72. -/
theorem arithmetic_mean_of_first_three_composite_reciprocals :
  (1 / 4 + 1 / 6 + 1 / 8) / 3 = 13 / 72 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_three_composite_reciprocals_l1704_170428


namespace NUMINAMATH_CALUDE_g_behavior_l1704_170489

def g (x : ℝ) := -3 * x^4 + 5 * x^3 - 2

theorem g_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M) :=
sorry

end NUMINAMATH_CALUDE_g_behavior_l1704_170489


namespace NUMINAMATH_CALUDE_last_score_is_70_l1704_170497

def scores : List ℤ := [65, 70, 85, 90]

def is_divisible (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def valid_sequence (seq : List ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → n ≤ seq.length → is_divisible (seq.take n).sum n

theorem last_score_is_70 :
  ∃ (seq : List ℤ), seq.toFinset = scores.toFinset ∧
                    valid_sequence seq ∧
                    seq.getLast? = some 70 :=
sorry

end NUMINAMATH_CALUDE_last_score_is_70_l1704_170497


namespace NUMINAMATH_CALUDE_total_books_calculation_l1704_170450

theorem total_books_calculation (joan_books tom_books lisa_books steve_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : lisa_books = 27)
  (h4 : steve_books = 45) :
  joan_books + tom_books + lisa_books + steve_books = 120 := by
sorry

end NUMINAMATH_CALUDE_total_books_calculation_l1704_170450


namespace NUMINAMATH_CALUDE_units_digit_problem_l1704_170408

theorem units_digit_problem : ∃ n : ℕ, (6 * 16 * 1986 - 6^4) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1704_170408


namespace NUMINAMATH_CALUDE_kim_average_increase_l1704_170498

/-- Given Kim's exam scores, prove that her average increases by 1 after the fourth exam. -/
theorem kim_average_increase (score1 score2 score3 score4 : ℕ) 
  (h1 : score1 = 87)
  (h2 : score2 = 83)
  (h3 : score3 = 88)
  (h4 : score4 = 90) :
  (score1 + score2 + score3 + score4) / 4 - (score1 + score2 + score3) / 3 = 1 := by
  sorry

#eval (87 + 83 + 88 + 90) / 4 - (87 + 83 + 88) / 3

end NUMINAMATH_CALUDE_kim_average_increase_l1704_170498


namespace NUMINAMATH_CALUDE_bee_speed_is_11_5_l1704_170404

/-- Represents the bee's journey with given conditions -/
structure BeeJourney where
  v : ℝ  -- Bee's constant actual speed
  t_dr : ℝ := 10  -- Time from daisy to rose
  t_rp : ℝ := 6   -- Time from rose to poppy
  t_pt : ℝ := 8   -- Time from poppy to tulip
  slow : ℝ := 2   -- Speed reduction due to crosswind
  boost : ℝ := 3  -- Speed increase due to crosswind

  d_dr : ℝ := t_dr * (v - slow)  -- Distance from daisy to rose
  d_rp : ℝ := t_rp * (v + boost) -- Distance from rose to poppy
  d_pt : ℝ := t_pt * (v - slow)  -- Distance from poppy to tulip

  h_distance_diff : d_dr = d_rp + 8  -- Distance condition
  h_distance_equal : d_pt = d_dr     -- Distance equality condition

/-- Theorem stating that the bee's speed is 11.5 m/s given the conditions -/
theorem bee_speed_is_11_5 (j : BeeJourney) : j.v = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_bee_speed_is_11_5_l1704_170404


namespace NUMINAMATH_CALUDE_integral_tan_cos_equality_l1704_170448

open Real MeasureTheory Interval

theorem integral_tan_cos_equality : 
  ∫ x in (-1 : ℝ)..1, (tan x)^11 + (cos x)^21 = 2 * ∫ x in (0 : ℝ)..1, (cos x)^21 := by
  sorry

end NUMINAMATH_CALUDE_integral_tan_cos_equality_l1704_170448


namespace NUMINAMATH_CALUDE_june_upload_ratio_l1704_170467

/-- Represents the video upload scenario for a YouTuber in June --/
structure VideoUpload where
  totalDays : Nat
  halfMonth : Nat
  firstHalfDailyHours : Nat
  totalHours : Nat

/-- Calculates the ratio of daily video hours in the second half to the first half of the month --/
def uploadRatio (v : VideoUpload) : Rat :=
  let firstHalfTotal := v.firstHalfDailyHours * v.halfMonth
  let secondHalfTotal := v.totalHours - firstHalfTotal
  let secondHalfDaily := secondHalfTotal / v.halfMonth
  secondHalfDaily / v.firstHalfDailyHours

/-- The main theorem stating the upload ratio for the given scenario --/
theorem june_upload_ratio (v : VideoUpload) 
    (h1 : v.totalDays = 30)
    (h2 : v.halfMonth = 15)
    (h3 : v.firstHalfDailyHours = 10)
    (h4 : v.totalHours = 450) :
  uploadRatio v = 2 := by
  sorry

#eval uploadRatio { totalDays := 30, halfMonth := 15, firstHalfDailyHours := 10, totalHours := 450 }

end NUMINAMATH_CALUDE_june_upload_ratio_l1704_170467


namespace NUMINAMATH_CALUDE_max_2012_gons_less_than_1006_l1704_170473

/-- The number of sides in each polygon -/
def n : ℕ := 2012

/-- The maximum number of different n-gons that can be drawn with all vertices shared 
    and no sides shared between any two polygons -/
def max_polygons (n : ℕ) : ℕ := (n - 1) / 2

/-- Theorem: The maximum number of different 2012-gons that can be drawn with all vertices shared 
    and no sides shared between any two polygons is less than 1006 -/
theorem max_2012_gons_less_than_1006 : max_polygons n < 1006 := by
  sorry

end NUMINAMATH_CALUDE_max_2012_gons_less_than_1006_l1704_170473


namespace NUMINAMATH_CALUDE_john_gave_one_third_l1704_170442

/-- The fraction of burritos John gave to his friend -/
def fraction_given_away (boxes : ℕ) (burritos_per_box : ℕ) (days : ℕ) (burritos_per_day : ℕ) (burritos_left : ℕ) : ℚ :=
  let total_bought := boxes * burritos_per_box
  let total_eaten := days * burritos_per_day
  let total_before_eating := total_eaten + burritos_left
  let given_away := total_bought - total_before_eating
  given_away / total_bought

/-- Theorem stating that John gave away 1/3 of the burritos -/
theorem john_gave_one_third :
  fraction_given_away 3 20 10 3 10 = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_john_gave_one_third_l1704_170442


namespace NUMINAMATH_CALUDE_unique_solution_in_interval_l1704_170402

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 1

-- State the theorem
theorem unique_solution_in_interval :
  ∃! a : ℝ, 0 < a ∧ a < 3 ∧ f a = 7 ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_in_interval_l1704_170402


namespace NUMINAMATH_CALUDE_age_difference_proof_l1704_170477

/-- Proves the number of years ago when the elder person was twice as old as the younger person -/
theorem age_difference_proof (younger_age elder_age years_ago : ℕ) : 
  younger_age = 35 →
  elder_age - younger_age = 20 →
  elder_age - years_ago = 2 * (younger_age - years_ago) →
  years_ago = 15 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1704_170477


namespace NUMINAMATH_CALUDE_factorization_of_36x_squared_minus_4_l1704_170469

theorem factorization_of_36x_squared_minus_4 (x : ℝ) :
  36 * x^2 - 4 = 4 * (3*x + 1) * (3*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_36x_squared_minus_4_l1704_170469


namespace NUMINAMATH_CALUDE_lottery_probabilities_l1704_170464

def total_numbers : ℕ := 10
def numbers_per_ticket : ℕ := 5
def numbers_drawn : ℕ := 4

def probability_four_match : ℚ := 1 / 21
def probability_two_match : ℚ := 10 / 21

theorem lottery_probabilities :
  (total_numbers = 10) →
  (numbers_per_ticket = 5) →
  (numbers_drawn = 4) →
  (probability_four_match = 1 / 21) ∧
  (probability_two_match = 10 / 21) := by
  sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l1704_170464


namespace NUMINAMATH_CALUDE_prism_surface_area_l1704_170443

/-- A rectangular prism formed by unit cubes -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℕ :=
  p.length * p.width * p.height

/-- The surface area of a rectangular prism -/
def surfaceArea (p : RectangularPrism) : ℕ :=
  2 * (p.length * p.width + p.width * p.height + p.height * p.length)

/-- The number of unpainted cubes in a prism -/
def unpaintedCubes (p : RectangularPrism) : ℕ :=
  (p.length - 2) * (p.width - 2) * (p.height - 2)

theorem prism_surface_area :
  ∃ (p : RectangularPrism),
    volume p = 120 ∧
    unpaintedCubes p = 24 ∧
    surfaceArea p = 148 := by
  sorry

end NUMINAMATH_CALUDE_prism_surface_area_l1704_170443


namespace NUMINAMATH_CALUDE_function_value_at_negative_a_l1704_170459

/-- Given a function f(x) = ax² + bx, if f(a) = 8, then f(-a) = 8 - 2ab -/
theorem function_value_at_negative_a (a b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + b * x
  f a = 8 → f (-a) = 8 - 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_a_l1704_170459


namespace NUMINAMATH_CALUDE_positive_expression_l1704_170406

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -2 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  x + y^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l1704_170406


namespace NUMINAMATH_CALUDE_work_completion_time_l1704_170455

/-- The time taken for A, B, and C to complete the work together -/
def time_together (time_A time_B time_C : ℚ) : ℚ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem stating that A, B, and C can complete the work together in 2 days -/
theorem work_completion_time :
  time_together 4 6 12 = 2 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1704_170455


namespace NUMINAMATH_CALUDE_first_player_can_ensure_non_trivial_solution_l1704_170418

-- Define the system of equations
structure LinearSystem :=
  (eq1 eq2 eq3 : ℝ → ℝ → ℝ → ℝ)

-- Define the game state
structure GameState :=
  (system : LinearSystem)
  (player_turn : Bool)

-- Define a strategy for the first player
def FirstPlayerStrategy : GameState → GameState := sorry

-- Define a strategy for the second player
def SecondPlayerStrategy : GameState → GameState := sorry

-- Theorem statement
theorem first_player_can_ensure_non_trivial_solution :
  ∀ (initial_state : GameState),
  ∃ (x y z : ℝ), 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    (initial_state.system.eq1 x y z = 0) ∧
    (initial_state.system.eq2 x y z = 0) ∧
    (initial_state.system.eq3 x y z = 0) :=
sorry

end NUMINAMATH_CALUDE_first_player_can_ensure_non_trivial_solution_l1704_170418


namespace NUMINAMATH_CALUDE_banana_mush_proof_l1704_170487

theorem banana_mush_proof (flour_ratio : ℝ) (total_bananas : ℝ) (total_flour : ℝ)
  (h1 : flour_ratio = 3)
  (h2 : total_bananas = 20)
  (h3 : total_flour = 15) :
  (total_bananas * flour_ratio) / total_flour = 4 := by
  sorry

end NUMINAMATH_CALUDE_banana_mush_proof_l1704_170487


namespace NUMINAMATH_CALUDE_smallest_term_is_fifth_l1704_170435

def a (n : ℕ) : ℤ := 3 * n^2 - 28 * n

theorem smallest_term_is_fifth : 
  ∀ k : ℕ, k ≠ 0 → a 5 ≤ a k :=
sorry

end NUMINAMATH_CALUDE_smallest_term_is_fifth_l1704_170435


namespace NUMINAMATH_CALUDE_no_real_roots_l1704_170433

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 2) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l1704_170433


namespace NUMINAMATH_CALUDE_garden_pool_perimeter_l1704_170486

/-- Represents a rectangular garden with square plots and a pool -/
structure Garden where
  plot_area : ℝ
  garden_length : ℝ
  num_plots : ℕ

/-- Calculates the perimeter of the pool in the garden -/
def pool_perimeter (g : Garden) : ℝ :=
  2 * g.garden_length

/-- Theorem stating the perimeter of the pool in the given garden configuration -/
theorem garden_pool_perimeter (g : Garden) 
  (h1 : g.plot_area = 20)
  (h2 : g.garden_length = 9)
  (h3 : g.num_plots = 4) : 
  pool_perimeter g = 18 := by
  sorry

#check garden_pool_perimeter

end NUMINAMATH_CALUDE_garden_pool_perimeter_l1704_170486


namespace NUMINAMATH_CALUDE_intersecting_circles_angle_equality_l1704_170471

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define the property of a point being on a circle
variable (on_circle : Point → Circle → Prop)

-- Define the property of two circles intersecting
variable (intersect : Circle → Circle → Prop)

-- Define the property of points being collinear
variable (collinear : Point → Point → Point → Prop)

-- Define the angle between three points
variable (angle : Point → Point → Point → ℝ)

-- State the theorem
theorem intersecting_circles_angle_equality
  (C1 C2 : Circle) (O1 O2 P Q U V : Point) :
  center C1 = O1 →
  center C2 = O2 →
  intersect C1 C2 →
  on_circle P C1 →
  on_circle P C2 →
  on_circle Q C1 →
  on_circle Q C2 →
  on_circle U C1 →
  on_circle V C2 →
  collinear U P V →
  angle U Q V = angle O1 Q O2 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_angle_equality_l1704_170471


namespace NUMINAMATH_CALUDE_sum_side_lengths_eq_66_l1704_170439

/-- Represents a convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  -- Angle A
  angle_a : ℝ
  -- Parallel sides condition
  parallel_ab_cd : Prop
  -- Arithmetic progression condition
  arithmetic_progression : Prop
  -- AB is maximum length
  ab_max : Prop

/-- The sum of all possible values for a side length other than AB -/
def sum_possible_side_lengths (q : ConvexQuadrilateral) : ℝ := sorry

/-- Main theorem statement -/
theorem sum_side_lengths_eq_66 (q : ConvexQuadrilateral) 
  (h1 : q.ab = 18)
  (h2 : q.angle_a = 60 * π / 180)
  (h3 : q.parallel_ab_cd)
  (h4 : q.arithmetic_progression)
  (h5 : q.ab_max) :
  sum_possible_side_lengths q = 66 := by sorry

end NUMINAMATH_CALUDE_sum_side_lengths_eq_66_l1704_170439


namespace NUMINAMATH_CALUDE_problem_statement_l1704_170415

theorem problem_statement (a b x y : ℝ) 
  (h1 : a*x + b*y = 2)
  (h2 : a*x^2 + b*y^2 = 5)
  (h3 : a*x^3 + b*y^3 = 10)
  (h4 : a*x^4 + b*y^4 = 30) :
  a*x^5 + b*y^5 = 40 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1704_170415


namespace NUMINAMATH_CALUDE_f_composition_quarter_l1704_170482

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 4
  else 2^x

theorem f_composition_quarter : f (f (1/4)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_quarter_l1704_170482


namespace NUMINAMATH_CALUDE_two_true_statements_l1704_170441

theorem two_true_statements 
  (x y a b : ℝ) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ a ≠ 0 ∧ b ≠ 0) 
  (h_x_lt_a : x < a) 
  (h_y_lt_b : y < b) 
  (h_positive : x > 0 ∧ y > 0 ∧ a > 0 ∧ b > 0) : 
  ∃! n : ℕ, n = 2 ∧ n = (
    (if x + y < a + b then 1 else 0) +
    (if x - y < a - b then 1 else 0) +
    (if x * y < a * b then 1 else 0) +
    (if (x / y < a / b → x / y < a / b) then 1 else 0)
  ) := by sorry

end NUMINAMATH_CALUDE_two_true_statements_l1704_170441


namespace NUMINAMATH_CALUDE_inequality_proof_l1704_170434

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1704_170434


namespace NUMINAMATH_CALUDE_coffee_shop_run_time_l1704_170494

/-- Represents the time in minutes to run a given distance at a constant pace -/
def runTime (distance : ℝ) (pace : ℝ) : ℝ := distance * pace

theorem coffee_shop_run_time :
  let parkDistance : ℝ := 5
  let parkTime : ℝ := 30
  let coffeeShopDistance : ℝ := 2
  let pace : ℝ := parkTime / parkDistance
  runTime coffeeShopDistance pace = 12 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_run_time_l1704_170494


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1704_170485

theorem right_triangle_perimeter (a b c : ℝ) : 
  a = 10 ∧ b = 24 ∧ c = 26 →
  a + b > c ∧ a + c > b ∧ b + c > a →
  a^2 + b^2 = c^2 →
  a + b + c = 60 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1704_170485


namespace NUMINAMATH_CALUDE_addition_problem_base_5_l1704_170484

def base_5_to_10 (n : ℕ) : ℕ := sorry

def base_10_to_5 (n : ℕ) : ℕ := sorry

theorem addition_problem_base_5 (X Y : ℕ) : 
  base_10_to_5 (3 * 25 + X * 5 + Y) + base_10_to_5 (3 * 5 + 2) = 
  base_10_to_5 (4 * 25 + 2 * 5 + X) →
  X + Y = 6 := by sorry

end NUMINAMATH_CALUDE_addition_problem_base_5_l1704_170484


namespace NUMINAMATH_CALUDE_train_length_calculation_l1704_170426

-- Define the given values
def train_speed : ℝ := 60  -- km/hr
def man_speed : ℝ := 6     -- km/hr
def time_to_pass : ℝ := 17.998560115190788  -- seconds

-- Define the theorem
theorem train_length_calculation :
  let relative_speed : ℝ := (train_speed + man_speed) * (5 / 18)  -- Convert to m/s
  let train_length : ℝ := relative_speed * time_to_pass
  train_length = 330 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1704_170426


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l1704_170461

/-- The distance from a point P(2-a, -5) to the y-axis is |2-a| -/
theorem distance_to_y_axis (a : ℝ) : 
  let P : ℝ × ℝ := (2 - a, -5)
  abs (P.1) = abs (2 - a) := by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l1704_170461


namespace NUMINAMATH_CALUDE_johns_money_proof_l1704_170412

/-- Calculates John's initial amount of money given his purchases and remaining money -/
def johns_initial_money (roast_cost vegetables_cost remaining_money : ℕ) : ℕ :=
  roast_cost + vegetables_cost + remaining_money

theorem johns_money_proof (roast_cost vegetables_cost remaining_money : ℕ) 
  (h1 : roast_cost = 17)
  (h2 : vegetables_cost = 11)
  (h3 : remaining_money = 72) :
  johns_initial_money roast_cost vegetables_cost remaining_money = 100 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_proof_l1704_170412


namespace NUMINAMATH_CALUDE_equal_tape_length_l1704_170436

def minyoung_tape : ℕ := 1748
def yoojung_tape : ℕ := 850
def tape_to_give : ℕ := 449

theorem equal_tape_length : 
  minyoung_tape - tape_to_give = yoojung_tape + tape_to_give :=
by sorry

end NUMINAMATH_CALUDE_equal_tape_length_l1704_170436


namespace NUMINAMATH_CALUDE_system_solution_and_simplification_l1704_170452

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  x + y = m + 2 ∧ 4 * x + 5 * y = 6 * m + 3

-- Define the positivity condition for x and y
def positive_solution (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Theorem statement
theorem system_solution_and_simplification (m : ℝ) :
  (∃ x y, system x y m ∧ positive_solution x y) →
  (5/2 < m ∧ m < 7) ∧
  (|2*m - 5| - |m - 7| = 3*m - 12) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_and_simplification_l1704_170452


namespace NUMINAMATH_CALUDE_smallest_angle_satisfying_trig_equation_l1704_170445

theorem smallest_angle_satisfying_trig_equation :
  ∃ y : ℝ, y > 0 ∧ y < (π / 180) * 360 ∧
  (∀ θ : ℝ, 0 < θ ∧ θ < y → ¬(Real.sin (4 * θ) * Real.sin (5 * θ) = Real.cos (4 * θ) * Real.cos (5 * θ))) ∧
  Real.sin (4 * y) * Real.sin (5 * y) = Real.cos (4 * y) * Real.cos (5 * y) ∧
  y = (π / 180) * 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfying_trig_equation_l1704_170445


namespace NUMINAMATH_CALUDE_nearest_multiple_21_l1704_170483

theorem nearest_multiple_21 (x : ℤ) : 
  (∀ y : ℤ, y % 21 = 0 → |x - 2319| ≤ |x - y|) → x = 2318 :=
sorry

end NUMINAMATH_CALUDE_nearest_multiple_21_l1704_170483


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1704_170463

theorem sum_of_a_and_b (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 - b^2 = -12) : a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1704_170463


namespace NUMINAMATH_CALUDE_factorial_80_mod_7_l1704_170422

def last_three_nonzero_digits (n : ℕ) : ℕ := sorry

theorem factorial_80_mod_7 : 
  last_three_nonzero_digits (Nat.factorial 80) % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_factorial_80_mod_7_l1704_170422


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l1704_170451

theorem sum_of_two_squares (P : ℤ) (a b : ℤ) (h : P = a^2 + b^2) :
  ∃ x y : ℤ, 2*P = x^2 + y^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l1704_170451


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_y_coord_l1704_170480

/-- The y-coordinate of the intersection point of perpendicular tangents to y = 4x^2 -/
theorem perpendicular_tangents_intersection_y_coord (c d : ℝ) : 
  (c ≠ d) →                                  -- Ensure C and D are distinct points
  (4 * c^2 = (4 : ℝ) * c^2) →                -- C is on the parabola y = 4x^2
  (4 * d^2 = (4 : ℝ) * d^2) →                -- D is on the parabola y = 4x^2
  ((8 : ℝ) * c * (8 * d) = -1) →             -- Tangent lines are perpendicular
  (4 : ℝ) * c * d = -(1/16) :=               -- y-coordinate of intersection point Q is -1/16
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_y_coord_l1704_170480


namespace NUMINAMATH_CALUDE_fifty_eighth_digit_of_one_seventeenth_l1704_170430

def decimal_representation (n : ℕ) : List ℕ := sorry

def is_periodic (l : List ℕ) : Prop := sorry

def nth_digit (l : List ℕ) (n : ℕ) : ℕ := sorry

theorem fifty_eighth_digit_of_one_seventeenth (h : is_periodic (decimal_representation 17)) :
  nth_digit (decimal_representation 17) 58 = 4 := by sorry

end NUMINAMATH_CALUDE_fifty_eighth_digit_of_one_seventeenth_l1704_170430


namespace NUMINAMATH_CALUDE_python_to_boa_ratio_l1704_170478

/-- The ratio of pythons to boa constrictors in a park -/
theorem python_to_boa_ratio :
  let total_snakes : ℕ := 200
  let boa_constrictors : ℕ := 40
  let rattlesnakes : ℕ := 40
  let pythons : ℕ := total_snakes - (boa_constrictors + rattlesnakes)
  (pythons : ℚ) / boa_constrictors = 3 := by
  sorry

end NUMINAMATH_CALUDE_python_to_boa_ratio_l1704_170478


namespace NUMINAMATH_CALUDE_triangle_cosine_problem_l1704_170490

theorem triangle_cosine_problem (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = 3 →
  C = 2 * A →
  0 < A →
  A < π →
  0 < B →
  B < π →
  0 < C →
  C < π →
  a = 2 * Real.sin B * Real.sin (C / 2) →
  b = 2 * Real.sin A * Real.sin (C / 2) →
  c = 2 * Real.sin A * Real.sin B →
  Real.cos C = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_problem_l1704_170490


namespace NUMINAMATH_CALUDE_consecutive_divisibility_l1704_170421

theorem consecutive_divisibility (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ (start : ℕ), ∃ (x y z : ℕ), 
    (x ∈ Finset.range (2 * c) ∧ y ∈ Finset.range (2 * c) ∧ z ∈ Finset.range (2 * c)) ∧
    (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    ((a * b * c) ∣ (x * y * z)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_divisibility_l1704_170421


namespace NUMINAMATH_CALUDE_ball_max_height_l1704_170491

-- Define the function representing the ball's height
def f (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

-- Theorem stating that the maximum height is 40 feet
theorem ball_max_height :
  ∃ (max : ℝ), max = 40 ∧ ∀ (t : ℝ), f t ≤ max :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l1704_170491


namespace NUMINAMATH_CALUDE_ariel_fencing_start_year_l1704_170456

def birth_year : ℕ := 1992
def current_age : ℕ := 30
def fencing_years : ℕ := 16

theorem ariel_fencing_start_year :
  birth_year + current_age - fencing_years = 2006 :=
by sorry

end NUMINAMATH_CALUDE_ariel_fencing_start_year_l1704_170456


namespace NUMINAMATH_CALUDE_solve_scooter_problem_l1704_170447

def scooter_problem (C : ℝ) (repair_percentage : ℝ) (profit_percentage : ℝ) (profit : ℝ) : Prop :=
  let repair_cost := repair_percentage * C
  let selling_price := (1 + profit_percentage) * C
  selling_price - C = profit ∧ 
  repair_cost = 550

theorem solve_scooter_problem :
  ∃ C : ℝ, scooter_problem C 0.1 0.2 1100 :=
sorry

end NUMINAMATH_CALUDE_solve_scooter_problem_l1704_170447


namespace NUMINAMATH_CALUDE_shaded_area_is_36_l1704_170437

/-- Given a rectangle and a right triangle with the following properties:
    - Rectangle: width 12, height 12, lower right vertex at (12, 0)
    - Triangle: base 12, height 12, lower left vertex at (12, 0)
    - Line passing through (0, 12) and (24, 0)
    Prove that the area of the triangle formed by this line, the vertical line x = 12,
    and the x-axis is 36 square units. -/
theorem shaded_area_is_36 (rectangle_width rectangle_height triangle_base triangle_height : ℝ)
  (h_rect_width : rectangle_width = 12)
  (h_rect_height : rectangle_height = 12)
  (h_tri_base : triangle_base = 12)
  (h_tri_height : triangle_height = 12) :
  let line := fun x => -1/2 * x + 12
  let intersection_x := 12
  let intersection_y := line intersection_x
  let shaded_area := 1/2 * intersection_y * triangle_base
  shaded_area = 36 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_36_l1704_170437


namespace NUMINAMATH_CALUDE_rectangular_sheet_area_l1704_170481

theorem rectangular_sheet_area (area1 area2 : ℝ) : 
  area1 = 4 * area2 →  -- First part is four times larger than the second
  area1 - area2 = 2208 →  -- First part is 2208 cm² larger than the second
  area1 + area2 = 3680 :=  -- Total area of the sheet
by sorry

end NUMINAMATH_CALUDE_rectangular_sheet_area_l1704_170481


namespace NUMINAMATH_CALUDE_original_number_l1704_170420

theorem original_number (x : ℝ) (h : 5 * x - 9 = 51) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1704_170420
