import Mathlib

namespace NUMINAMATH_CALUDE_sequence_properties_l2629_262900

def sequence_a (n : ℕ) : ℝ := 6 * 2^(n-1) - 3

def sum_S (n : ℕ) : ℝ := 6 * 2^n - 3 * n - 6

theorem sequence_properties :
  let a := sequence_a
  let S := sum_S
  (∀ n : ℕ, n ≥ 1 → S n = 2 * a n - 3 * n) →
  (a 1 = 3 ∧
   ∀ n : ℕ, a (n + 1) = 2 * a n + 3 ∧
   ∀ n : ℕ, n ≥ 1 → a n = 6 * 2^(n-1) - 3 ∧
   ∀ n : ℕ, n ≥ 1 → S n = 6 * 2^n - 3 * n - 6) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2629_262900


namespace NUMINAMATH_CALUDE_min_value_expression_l2629_262967

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 4 / y) ≥ 9 ∧
  ((x + y) * (1 / x + 4 / y) = 9 ↔ y = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2629_262967


namespace NUMINAMATH_CALUDE_remaining_money_l2629_262916

def initial_amount : ℕ := 760
def ticket_price : ℕ := 300
def hotel_price : ℕ := ticket_price / 2

theorem remaining_money :
  initial_amount - (ticket_price + hotel_price) = 310 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l2629_262916


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2629_262974

/-- Given two vectors a and b in ℝ², where a = (2, 3) and b = (x, -9),
    if a is parallel to b, then x = -6. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![x, -9]
  (∃ (k : ℝ), b = k • a) →
  x = -6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2629_262974


namespace NUMINAMATH_CALUDE_negation_equivalence_l2629_262945

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x + 1 < 0 ∨ x^2 - x > 0) ↔ (∀ x : ℝ, x + 1 ≥ 0 ∧ x^2 - x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2629_262945


namespace NUMINAMATH_CALUDE_rope_folded_six_times_l2629_262983

/-- The number of segments a rope is cut into after being folded n times and cut along the middle -/
def rope_segments (n : ℕ) : ℕ := 2^n + 1

/-- Theorem: A rope folded in half 6 times and cut along the middle will result in 65 segments -/
theorem rope_folded_six_times : rope_segments 6 = 65 := by
  sorry

end NUMINAMATH_CALUDE_rope_folded_six_times_l2629_262983


namespace NUMINAMATH_CALUDE_cookies_taken_in_seven_days_l2629_262997

/-- Represents the number of cookies Jessica takes each day -/
def jessica_daily_cookies : ℝ := 1.5

/-- Represents the number of cookies Sarah takes each day -/
def sarah_daily_cookies : ℝ := 3 * jessica_daily_cookies

/-- Represents the number of cookies Paul takes each day -/
def paul_daily_cookies : ℝ := 2 * sarah_daily_cookies

/-- Represents the total number of cookies in the jar initially -/
def initial_cookies : ℕ := 200

/-- Represents the number of cookies left after 10 days -/
def cookies_left : ℕ := 50

/-- Represents the number of days they took cookies -/
def total_days : ℕ := 10

/-- Represents the number of days we want to calculate for -/
def target_days : ℕ := 7

theorem cookies_taken_in_seven_days :
  (jessica_daily_cookies + sarah_daily_cookies + paul_daily_cookies) * target_days = 105 :=
by sorry

end NUMINAMATH_CALUDE_cookies_taken_in_seven_days_l2629_262997


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2629_262979

theorem simplify_fraction_product : (225 : ℚ) / 10125 * 45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2629_262979


namespace NUMINAMATH_CALUDE_gift_wrapping_expenses_l2629_262985

def total_spent : ℝ := 700
def gift_cost : ℝ := 561

theorem gift_wrapping_expenses : total_spent - gift_cost = 139 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_expenses_l2629_262985


namespace NUMINAMATH_CALUDE_number_division_problem_l2629_262951

theorem number_division_problem (x y : ℝ) 
  (h1 : (x - 5) / y = 7)
  (h2 : (x - 4) / 10 = 5) : 
  y = 7 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l2629_262951


namespace NUMINAMATH_CALUDE_probability_both_red_probability_different_colors_l2629_262913

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents a ball with a label -/
structure Ball where
  color : Color
  label : String

/-- The set of all balls in the bag -/
def bag : Finset Ball := sorry

/-- The number of balls in the bag -/
def total_balls : Nat := 6

/-- The number of red balls -/
def red_balls : Nat := 4

/-- The number of black balls -/
def black_balls : Nat := 2

/-- The set of all possible combinations when drawing 2 balls -/
def all_combinations : Finset (Ball × Ball) := sorry

/-- The set of combinations where both balls are red -/
def both_red : Finset (Ball × Ball) := sorry

/-- The set of combinations where the balls have different colors -/
def different_colors : Finset (Ball × Ball) := sorry

theorem probability_both_red :
  (Finset.card both_red : ℚ) / (Finset.card all_combinations : ℚ) = 2 / 5 := by sorry

theorem probability_different_colors :
  (Finset.card different_colors : ℚ) / (Finset.card all_combinations : ℚ) = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_both_red_probability_different_colors_l2629_262913


namespace NUMINAMATH_CALUDE_round_trip_ticket_holders_l2629_262982

/-- The percentage of ship passengers holding round-trip tickets -/
def round_trip_percentage : ℝ := 62.5

theorem round_trip_ticket_holders (total_passengers : ℝ) (round_trip_with_car : ℝ) (round_trip_without_car : ℝ)
  (h1 : round_trip_with_car = 0.25 * total_passengers)
  (h2 : round_trip_without_car = 0.6 * (round_trip_with_car + round_trip_without_car)) :
  (round_trip_with_car + round_trip_without_car) / total_passengers * 100 = round_trip_percentage := by
  sorry

end NUMINAMATH_CALUDE_round_trip_ticket_holders_l2629_262982


namespace NUMINAMATH_CALUDE_instrument_probability_l2629_262922

theorem instrument_probability (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = 3/5 →
  two_or_more = 96 →
  (((at_least_one * total) - two_or_more) / total : ℚ) = 12/25 := by
sorry

end NUMINAMATH_CALUDE_instrument_probability_l2629_262922


namespace NUMINAMATH_CALUDE_circle_center_sum_l2629_262980

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, 
    the sum of the x and y coordinates of its center is -1. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 4*x - 6*y + 9 → 
  ∃ (h k : ℝ), (∀ (a b : ℝ), (a - h)^2 + (b - k)^2 = (x - h)^2 + (y - k)^2) ∧ h + k = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2629_262980


namespace NUMINAMATH_CALUDE_intersection_condition_l2629_262931

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line2D) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- The condition for two lines to intersect -/
def intersect (l₁ l₂ : Line2D) : Prop :=
  ¬ parallel l₁ l₂

/-- Definition of the two lines in the problem -/
def l₁ (a : ℝ) : Line2D := ⟨1, -a, 3⟩
def l₂ (a : ℝ) : Line2D := ⟨a, -4, 5⟩

/-- The main theorem to prove -/
theorem intersection_condition :
  (∀ a : ℝ, intersect (l₁ a) (l₂ a) → a ≠ 2) ∧
  ¬(∀ a : ℝ, a ≠ 2 → intersect (l₁ a) (l₂ a)) := by
  sorry


end NUMINAMATH_CALUDE_intersection_condition_l2629_262931


namespace NUMINAMATH_CALUDE_ceiling_sum_of_roots_l2629_262961

theorem ceiling_sum_of_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 27⌉ + ⌈Real.sqrt 243⌉ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_roots_l2629_262961


namespace NUMINAMATH_CALUDE_video_dislikes_calculation_l2629_262911

/-- Calculates the final number of dislikes for a video given initial likes, 
    initial dislikes formula, and additional dislikes. -/
def final_dislikes (initial_likes : ℕ) (additional_dislikes : ℕ) : ℕ :=
  (initial_likes / 2 + 100) + additional_dislikes

/-- Theorem stating that for a video with 3000 initial likes and 1000 additional dislikes,
    the final number of dislikes is 2600. -/
theorem video_dislikes_calculation :
  final_dislikes 3000 1000 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_video_dislikes_calculation_l2629_262911


namespace NUMINAMATH_CALUDE_sampling_methods_are_appropriate_l2629_262987

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region with sales points -/
structure Region where
  name : String
  salesPoints : Nat

/-- Represents a company with multiple regions -/
structure Company where
  regions : List Region
  totalSalesPoints : Nat

/-- Represents an investigation -/
structure Investigation where
  sampleSize : Nat
  samplingMethod : SamplingMethod

/-- The company in the problem -/
def problemCompany : Company :=
  { regions := [
      { name := "A", salesPoints := 150 },
      { name := "B", salesPoints := 120 },
      { name := "C", salesPoints := 180 },
      { name := "D", salesPoints := 150 }
    ],
    totalSalesPoints := 600
  }

/-- The first investigation in the problem -/
def investigation1 : Investigation :=
  { sampleSize := 100,
    samplingMethod := SamplingMethod.StratifiedSampling
  }

/-- The second investigation in the problem -/
def investigation2 : Investigation :=
  { sampleSize := 7,
    samplingMethod := SamplingMethod.SimpleRandomSampling
  }

/-- Checks if stratified sampling is appropriate for the given company and investigation -/
def isStratifiedSamplingAppropriate (company : Company) (investigation : Investigation) : Prop :=
  investigation.samplingMethod = SamplingMethod.StratifiedSampling ∧
  company.regions.length > 1 ∧
  investigation.sampleSize < company.totalSalesPoints

/-- Checks if simple random sampling is appropriate for the given sample size and population -/
def isSimpleRandomSamplingAppropriate (sampleSize : Nat) (populationSize : Nat) : Prop :=
  sampleSize < populationSize

/-- Theorem stating that the sampling methods are appropriate for the given investigations -/
theorem sampling_methods_are_appropriate :
  isStratifiedSamplingAppropriate problemCompany investigation1 ∧
  isSimpleRandomSamplingAppropriate investigation2.sampleSize 20 :=
  sorry

end NUMINAMATH_CALUDE_sampling_methods_are_appropriate_l2629_262987


namespace NUMINAMATH_CALUDE_distance_between_squares_l2629_262999

/-- Given two squares where:
  * The smaller square has a perimeter of 8 cm
  * The larger square has an area of 64 cm²
  * The bottom left corner of the larger square is 2 cm to the right of the top right corner of the smaller square
  Prove that the distance between the top right corner of the larger square (A) and 
  the top left corner of the smaller square (B) is √136 cm -/
theorem distance_between_squares (small_perimeter : ℝ) (large_area : ℝ) (horizontal_shift : ℝ) :
  small_perimeter = 8 →
  large_area = 64 →
  horizontal_shift = 2 →
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let horizontal_distance := horizontal_shift + large_side
  let vertical_distance := large_side - small_side
  Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2) = Real.sqrt 136 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_squares_l2629_262999


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2629_262958

theorem neither_sufficient_nor_necessary (a b : ℝ) :
  (∃ x y : ℝ, x - y > 0 ∧ x^2 - y^2 ≤ 0) ∧
  (∃ x y : ℝ, x - y ≤ 0 ∧ x^2 - y^2 > 0) :=
sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2629_262958


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2629_262996

theorem absolute_value_expression : |-2| * (|-25| - |5|) = -40 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2629_262996


namespace NUMINAMATH_CALUDE_unique_solution_l2629_262988

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The equation that the four-digit number must satisfy. -/
def SatisfiesEquation (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a * (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) * (a^6 + 2*b^6 + 3*c^6 + 4*d^6) = n

/-- The main theorem stating that 2010 is the only four-digit number satisfying the equation. -/
theorem unique_solution :
  ∀ n : ℕ, FourDigitNumber n → SatisfiesEquation n → n = 2010 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2629_262988


namespace NUMINAMATH_CALUDE_amp_eight_five_plus_ten_l2629_262964

def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem amp_eight_five_plus_ten : (amp 8 5) + 10 = 49 := by
  sorry

end NUMINAMATH_CALUDE_amp_eight_five_plus_ten_l2629_262964


namespace NUMINAMATH_CALUDE_total_time_is_80_minutes_l2629_262993

/-- The total time students spend outside of class -/
def total_time_outside_class (recess1 recess2 lunch recess3 : ℕ) : ℕ :=
  recess1 + recess2 + lunch + recess3

/-- Theorem stating that the total time outside class is 80 minutes -/
theorem total_time_is_80_minutes :
  total_time_outside_class 15 15 30 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_80_minutes_l2629_262993


namespace NUMINAMATH_CALUDE_q_divided_by_p_equals_44_l2629_262921

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 12

/-- The number of cards with each number -/
def cards_per_number : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability that all drawn cards bear the same number -/
noncomputable def p : ℚ := 12 / Nat.choose total_cards cards_drawn

/-- The probability that four cards bear one number and the fifth bears a different number -/
noncomputable def q : ℚ := 528 / Nat.choose total_cards cards_drawn

/-- Theorem stating that the ratio of q to p is 44 -/
theorem q_divided_by_p_equals_44 : q / p = 44 := by sorry

end NUMINAMATH_CALUDE_q_divided_by_p_equals_44_l2629_262921


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2629_262950

theorem arithmetic_mean_problem (p q r : ℝ) : 
  (p + q) / 2 = 10 → 
  (q + r) / 2 = 22 → 
  r - p = 24 → 
  (q + r) / 2 = 22 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2629_262950


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l2629_262914

/-- Calculates the average daily income of a cab driver over 5 days --/
theorem cab_driver_average_income :
  let day1_earnings := 250
  let day1_commission_rate := 0.1
  let day2_earnings := 400
  let day2_expense := 50
  let day3_earnings := 750
  let day3_commission_rate := 0.15
  let day4_earnings := 400
  let day4_expense := 40
  let day5_earnings := 500
  let day5_commission_rate := 0.2
  let total_days := 5
  let total_net_income := 
    (day1_earnings * (1 - day1_commission_rate)) +
    (day2_earnings - day2_expense) +
    (day3_earnings * (1 - day3_commission_rate)) +
    (day4_earnings - day4_expense) +
    (day5_earnings * (1 - day5_commission_rate))
  let average_daily_income := total_net_income / total_days
  average_daily_income = 394.50 := by
sorry


end NUMINAMATH_CALUDE_cab_driver_average_income_l2629_262914


namespace NUMINAMATH_CALUDE_colonization_combinations_eq_77056_l2629_262981

/-- The number of Earth-like planets -/
def earth_like_planets : ℕ := 8

/-- The number of Mars-like planets -/
def mars_like_planets : ℕ := 12

/-- The resource cost to colonize an Earth-like planet -/
def earth_cost : ℕ := 3

/-- The resource cost to colonize a Mars-like planet -/
def mars_cost : ℕ := 1

/-- The total available resources -/
def total_resources : ℕ := 18

/-- The function to calculate the number of combinations -/
def colonization_combinations : ℕ :=
  (Nat.choose earth_like_planets 2 * Nat.choose mars_like_planets 12) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 6) +
  (Nat.choose earth_like_planets 5 * Nat.choose mars_like_planets 3) +
  (Nat.choose earth_like_planets 6 * Nat.choose mars_like_planets 0)

/-- The theorem stating that the number of colonization combinations is 77056 -/
theorem colonization_combinations_eq_77056 : colonization_combinations = 77056 := by
  sorry

end NUMINAMATH_CALUDE_colonization_combinations_eq_77056_l2629_262981


namespace NUMINAMATH_CALUDE_color_cartridge_cost_l2629_262990

/-- The cost of each color cartridge given the total cost, number of cartridges, and cost of black-and-white cartridge. -/
theorem color_cartridge_cost (total_cost : ℕ) (bw_cost : ℕ) (num_color : ℕ) : 
  total_cost = bw_cost + num_color * 32 → 32 = (total_cost - bw_cost) / num_color :=
by sorry

end NUMINAMATH_CALUDE_color_cartridge_cost_l2629_262990


namespace NUMINAMATH_CALUDE_alternating_hexagon_area_l2629_262942

/-- A hexagon with alternating side lengths and specified corner triangles -/
structure AlternatingHexagon where
  short_side : ℝ
  long_side : ℝ
  corner_triangle_base : ℝ
  corner_triangle_altitude : ℝ

/-- The area of an alternating hexagon -/
def area (h : AlternatingHexagon) : ℝ := sorry

/-- Theorem: The area of the specified hexagon is 36 square units -/
theorem alternating_hexagon_area :
  let h : AlternatingHexagon := {
    short_side := 2,
    long_side := 4,
    corner_triangle_base := 2,
    corner_triangle_altitude := 3
  }
  area h = 36 := by sorry

end NUMINAMATH_CALUDE_alternating_hexagon_area_l2629_262942


namespace NUMINAMATH_CALUDE_output_for_15_l2629_262960

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 20 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 38 := by
  sorry

end NUMINAMATH_CALUDE_output_for_15_l2629_262960


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2629_262947

/-- A parabola in the Cartesian plane -/
structure Parabola where
  /-- The parameter p of the parabola -/
  p : ℝ
  /-- The vertex is at the origin -/
  vertex_at_origin : True
  /-- The focus is on the x-axis -/
  focus_on_x_axis : True
  /-- The parabola passes through the point (1, 2) -/
  passes_through_point : p = 2

/-- The distance from the focus to the directrix of a parabola -/
def focus_directrix_distance (c : Parabola) : ℝ := c.p

theorem parabola_focus_directrix_distance :
  ∀ c : Parabola, focus_directrix_distance c = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2629_262947


namespace NUMINAMATH_CALUDE_real_estate_transaction_result_verify_transaction_result_l2629_262953

/-- Calculates the overall gain or loss from a real estate transaction involving four properties. -/
theorem real_estate_transaction_result (property_cost : ℝ) (gain1 gain2 gain3 gain4 : ℝ) :
  property_cost = 512456 →
  gain1 = 0.25 →
  gain2 = -0.30 →
  gain3 = 0.35 →
  gain4 = -0.40 →
  (4 * property_cost) - (property_cost * (1 + gain1) + 
                         property_cost * (1 + gain2) + 
                         property_cost * (1 + gain3) + 
                         property_cost * (1 + gain4)) = 61245.6 := by
  sorry

/-- The actual financial result of the transaction. -/
def transaction_result : ℝ := 61245.6

/-- Proves that the calculated result matches the expected result. -/
theorem verify_transaction_result :
  (4 * 512456) - (512456 * (1 + 0.25) + 
                  512456 * (1 - 0.30) + 
                  512456 * (1 + 0.35) + 
                  512456 * (1 - 0.40)) = transaction_result := by
  sorry

end NUMINAMATH_CALUDE_real_estate_transaction_result_verify_transaction_result_l2629_262953


namespace NUMINAMATH_CALUDE_fraction_married_women_is_three_fourths_l2629_262919

/-- Represents the employee composition of a company -/
structure Company where
  total : ℕ
  women : ℕ
  married : ℕ
  single_men : ℕ

/-- The conditions of the company as described in the problem -/
def problem_company : Company :=
  { total := 100,  -- We assume 100 employees for simplicity
    women := 64,   -- 64% of 100
    married := 60, -- 60% of 100
    single_men := 24 } -- 2/3 of men (36) are single

/-- The fraction of women who are married in the company -/
def fraction_married_women (c : Company) : ℚ :=
  (c.married - (c.total - c.women - c.single_men)) / c.women

/-- Theorem stating that the fraction of married women is 3/4 -/
theorem fraction_married_women_is_three_fourths :
  fraction_married_women problem_company = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_fraction_married_women_is_three_fourths_l2629_262919


namespace NUMINAMATH_CALUDE_first_problem_number_l2629_262955

/-- Given a sequence of 48 consecutive integers ending with 125, 
    the first number in the sequence is 78. -/
theorem first_problem_number (last_number : ℕ) (total_problems : ℕ) :
  last_number = 125 → total_problems = 48 → 
  (last_number - total_problems + 1 : ℕ) = 78 := by
  sorry

end NUMINAMATH_CALUDE_first_problem_number_l2629_262955


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2629_262970

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (x + 1)}

-- State the theorem
theorem intersection_complement_theorem : A ∩ (U \ B) = {x | -2 ≤ x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2629_262970


namespace NUMINAMATH_CALUDE_equation_solution_l2629_262943

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2629_262943


namespace NUMINAMATH_CALUDE_michael_has_270_eggs_l2629_262906

/-- The number of eggs Michael has after buying and giving away crates -/
def michael_eggs (initial_crates : ℕ) (given_away : ℕ) (bought_later : ℕ) (eggs_per_crate : ℕ) : ℕ :=
  ((initial_crates - given_away) + bought_later) * eggs_per_crate

/-- Theorem stating that Michael has 270 eggs given the problem conditions -/
theorem michael_has_270_eggs :
  michael_eggs 6 2 5 30 = 270 := by
  sorry

end NUMINAMATH_CALUDE_michael_has_270_eggs_l2629_262906


namespace NUMINAMATH_CALUDE_tarun_departure_time_l2629_262933

theorem tarun_departure_time 
  (total_work : ℝ) 
  (combined_rate : ℝ) 
  (arun_rate : ℝ) 
  (remaining_days : ℝ) :
  combined_rate = total_work / 10 →
  arun_rate = total_work / 30 →
  remaining_days = 18 →
  ∃ (x : ℝ), x * combined_rate + remaining_days * arun_rate = total_work ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_tarun_departure_time_l2629_262933


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2629_262989

theorem profit_percentage_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 400)
  (h2 : selling_price = 560) :
  (selling_price - cost_price) / cost_price * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2629_262989


namespace NUMINAMATH_CALUDE_cone_sin_theta_l2629_262966

/-- Theorem: For a cone with base radius 5 and lateral area 65π, 
    if θ is the angle between the slant height and the height of the cone, 
    then sinθ = 5/13 -/
theorem cone_sin_theta (r : ℝ) (lat_area : ℝ) (θ : ℝ) 
    (h1 : r = 5) 
    (h2 : lat_area = 65 * Real.pi) 
    (h3 : θ = Real.arcsin (r / (lat_area / (2 * Real.pi * r)))) : 
  Real.sin θ = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cone_sin_theta_l2629_262966


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2629_262971

theorem cube_plus_reciprocal_cube (m : ℝ) (h : m + 1/m = 10) :
  m^3 + 1/m^3 + 6 = 976 := by sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2629_262971


namespace NUMINAMATH_CALUDE_work_hours_first_scenario_l2629_262965

/-- Represents the work rate of a person -/
structure WorkRate where
  rate : ℝ

/-- Represents a work scenario -/
structure WorkScenario where
  men : ℕ
  women : ℕ
  hours : ℝ
  days : ℝ

/-- The theorem to prove -/
theorem work_hours_first_scenario 
  (man_rate : WorkRate)
  (woman_rate : WorkRate)
  (scenario1 : WorkScenario)
  (scenario2 : WorkScenario)
  (scenario3 : WorkScenario)
  (h1 : scenario1.men = 2 ∧ scenario1.women = 3 ∧ scenario1.days = 5)
  (h2 : scenario2.men = 4 ∧ scenario2.women = 4 ∧ scenario2.hours = 3 ∧ scenario2.days = 7)
  (h3 : scenario3.men = 7 ∧ scenario3.hours = 4 ∧ scenario3.days = 5.000000000000001)
  (h4 : (scenario1.men : ℝ) * man_rate.rate * scenario1.hours * scenario1.days + 
        (scenario1.women : ℝ) * woman_rate.rate * scenario1.hours * scenario1.days = 1)
  (h5 : (scenario2.men : ℝ) * man_rate.rate * scenario2.hours * scenario2.days + 
        (scenario2.women : ℝ) * woman_rate.rate * scenario2.hours * scenario2.days = 1)
  (h6 : (scenario3.men : ℝ) * man_rate.rate * scenario3.hours * scenario3.days = 1) :
  scenario1.hours = 7 := by
  sorry


end NUMINAMATH_CALUDE_work_hours_first_scenario_l2629_262965


namespace NUMINAMATH_CALUDE_calculate_expression_l2629_262937

theorem calculate_expression : (1/3)⁻¹ + Real.sqrt 12 - |Real.sqrt 3 - 2| - (π - 2023)^0 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2629_262937


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_y_is_six_l2629_262927

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y < -8) ↔ (y ≥ 6) := by sorry

theorem smallest_integer_y_is_six : ∃ (y : ℤ), (7 - 3 * y < -8) ∧ (∀ (z : ℤ), (7 - 3 * z < -8) → z ≥ y) ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_y_is_six_l2629_262927


namespace NUMINAMATH_CALUDE_bob_start_time_l2629_262969

/-- Proves that Bob started walking 1 hour after Yolanda, given the conditions of the problem. -/
theorem bob_start_time (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) (bob_distance : ℝ) :
  total_distance = 10 →
  yolanda_rate = 3 →
  bob_rate = 4 →
  bob_distance = 4 →
  ∃ (bob_start_time : ℝ),
    bob_start_time = 1 ∧
    bob_start_time * bob_rate + yolanda_rate * (bob_start_time + bob_distance / bob_rate) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_bob_start_time_l2629_262969


namespace NUMINAMATH_CALUDE_min_additional_coins_for_alex_l2629_262994

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_additional_coins_for_alex : 
  min_additional_coins 15 63 = 57 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_coins_for_alex_l2629_262994


namespace NUMINAMATH_CALUDE_salary_growth_rate_l2629_262972

/-- Proves that the given annual compound interest rate satisfies the salary growth equation -/
theorem salary_growth_rate (initial_salary final_salary total_increase : ℝ) 
  (years : ℕ) (rate : ℝ) 
  (h1 : initial_salary = final_salary - total_increase)
  (h2 : final_salary = 90000)
  (h3 : total_increase = 25000)
  (h4 : years = 3) :
  final_salary = initial_salary * (1 + rate)^years := by
  sorry

end NUMINAMATH_CALUDE_salary_growth_rate_l2629_262972


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_length_l2629_262934

/-- Given a hyperbola and a circle with specific properties, prove the length of the chord formed by their intersection. -/
theorem hyperbola_circle_intersection_length :
  ∀ (a b : ℝ) (A B : ℝ × ℝ),
  a > 0 →
  b > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (∃ (t : ℝ), y = 2 * x * t ∨ y = -2 * x * t)) →  -- Asymptotes condition
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → Real.sqrt (1 + b^2 / a^2) = Real.sqrt 5) →  -- Eccentricity condition
  (∃ (t : ℝ), (A.1 - 2)^2 + (A.2 - 3)^2 = 1 ∧ 
              (B.1 - 2)^2 + (B.2 - 3)^2 = 1 ∧ 
              (A.2 = 2 * A.1 * t ∨ A.2 = -2 * A.1 * t) ∧ 
              (B.2 = 2 * B.1 * t ∨ B.2 = -2 * B.1 * t)) →  -- Intersection condition
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_circle_intersection_length_l2629_262934


namespace NUMINAMATH_CALUDE_rope_cutting_game_winner_l2629_262907

/-- Represents a player in the rope-cutting game -/
inductive Player : Type
| A : Player
| B : Player

/-- Determines if a number is a power of 3 -/
def isPowerOfThree (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3^k

/-- Represents the rope-cutting game -/
def RopeCuttingGame (a b : ℕ) : Prop :=
  a > 1 ∧ b > 1

/-- Determines if a player has a winning strategy -/
def hasWinningStrategy (p : Player) (a b : ℕ) : Prop :=
  RopeCuttingGame a b →
    (p = Player.B ↔ (a = 2 ∧ b = 3) ∨ isPowerOfThree a)

/-- Main theorem: Player B has a winning strategy iff a = 2 and b = 3, or a is a power of 3 -/
theorem rope_cutting_game_winner (a b : ℕ) :
  RopeCuttingGame a b →
    hasWinningStrategy Player.B a b := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_game_winner_l2629_262907


namespace NUMINAMATH_CALUDE_range_of_m_l2629_262977

theorem range_of_m (x m : ℝ) : 
  (∀ x, (x ≥ -2 ∧ x ≤ 10) → (x + m - 1) * (x - m - 1) ≤ 0) →
  m > 0 →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2629_262977


namespace NUMINAMATH_CALUDE_incenter_inside_l2629_262946

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The angle bisector of an angle in a triangle -/
def AngleBisector (t : Triangle) (vertex : Fin 3) : Set (ℝ × ℝ) :=
  sorry

/-- The intersection point of the three angle bisectors of a triangle -/
def Incenter (t : Triangle) : ℝ × ℝ :=
  sorry

/-- A point is inside a triangle if it's a convex combination of the triangle's vertices -/
def IsInside (p : ℝ × ℝ) (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = (a * t.A.1 + b * t.B.1 + c * t.C.1, a * t.A.2 + b * t.B.2 + c * t.C.2)

/-- Theorem: The incenter of a triangle lies inside the triangle -/
theorem incenter_inside (t : Triangle) : IsInside (Incenter t) t :=
  sorry

end NUMINAMATH_CALUDE_incenter_inside_l2629_262946


namespace NUMINAMATH_CALUDE_reward_fund_calculation_l2629_262944

theorem reward_fund_calculation (initial_bonus : ℕ) (actual_bonus : ℕ) (shortage : ℕ) (remaining : ℕ) :
  initial_bonus = 60 →
  actual_bonus = 55 →
  shortage = 10 →
  remaining = 85 →
  ∃ (members : ℕ), 
    members * actual_bonus + remaining = members * initial_bonus - shortage ∧
    members * initial_bonus - shortage = 1130 := by
  sorry

end NUMINAMATH_CALUDE_reward_fund_calculation_l2629_262944


namespace NUMINAMATH_CALUDE_parabola_opens_downwards_l2629_262959

/-- A parabola y = (a-1)x^2 + 2x opens downwards if and only if a < 1 -/
theorem parabola_opens_downwards (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + 2*x ≤ (a - 1) * 0^2 + 2*0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_opens_downwards_l2629_262959


namespace NUMINAMATH_CALUDE_decimal_33_is_quaternary_201_l2629_262986

-- Define a function to convert decimal to quaternary
def decimalToQuaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec convert (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else convert (m / 4) ((m % 4) :: acc)
    convert n []

-- Theorem statement
theorem decimal_33_is_quaternary_201 :
  decimalToQuaternary 33 = [2, 0, 1] := by
  sorry


end NUMINAMATH_CALUDE_decimal_33_is_quaternary_201_l2629_262986


namespace NUMINAMATH_CALUDE_nine_crosses_fit_on_chessboard_l2629_262910

/-- Represents a cross pentomino -/
structure CrossPentomino :=
  (size : ℕ := 5)

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- The area of a cross pentomino -/
def cross_pentomino_area (c : CrossPentomino) : ℕ := c.size

/-- The area of a chessboard -/
def chessboard_area (b : Chessboard) : ℕ := b.rows * b.cols

/-- Theorem: Nine cross pentominoes can fit on an 8x8 chessboard -/
theorem nine_crosses_fit_on_chessboard :
  ∃ (c : CrossPentomino) (b : Chessboard),
    b.rows = 8 ∧ b.cols = 8 ∧
    9 * (cross_pentomino_area c) ≤ chessboard_area b :=
by sorry

end NUMINAMATH_CALUDE_nine_crosses_fit_on_chessboard_l2629_262910


namespace NUMINAMATH_CALUDE_intersection_and_sufficient_condition_l2629_262920

def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

theorem intersection_and_sufficient_condition :
  (A (-2) ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B) ∧ (∃ x : ℝ, x ∈ B ∧ x ∉ A a) ↔ a ≤ -3 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_sufficient_condition_l2629_262920


namespace NUMINAMATH_CALUDE_negative_thirty_two_to_five_thirds_l2629_262991

theorem negative_thirty_two_to_five_thirds :
  (-32 : ℝ) ^ (5/3) = -256 * (2 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_thirty_two_to_five_thirds_l2629_262991


namespace NUMINAMATH_CALUDE_factorial_ratio_l2629_262917

theorem factorial_ratio : (50 : ℕ).factorial / (48 : ℕ).factorial = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2629_262917


namespace NUMINAMATH_CALUDE_nut_distribution_properties_l2629_262992

/-- Represents the state of nut distribution among three people -/
structure NutState where
  anya : ℕ
  borya : ℕ
  vitya : ℕ

/-- The nut distribution process -/
def distributeNuts (state : NutState) : NutState :=
  sorry

/-- Predicate to check if at least one nut is eaten during the entire process -/
def atLeastOneNutEaten (initialState : NutState) : Prop :=
  sorry

/-- Predicate to check if not all nuts are eaten during the entire process -/
def notAllNutsEaten (initialState : NutState) : Prop :=
  sorry

/-- Main theorem stating the properties of the nut distribution process -/
theorem nut_distribution_properties {n : ℕ} (h : n > 3) :
  let initialState : NutState := ⟨n, 0, 0⟩
  atLeastOneNutEaten initialState ∧ notAllNutsEaten initialState :=
by
  sorry

end NUMINAMATH_CALUDE_nut_distribution_properties_l2629_262992


namespace NUMINAMATH_CALUDE_stratified_sample_intermediate_count_l2629_262954

/-- Represents the composition of teachers in a school -/
structure TeacherPopulation where
  total : Nat
  intermediate : Nat
  
/-- Represents a stratified sample of teachers -/
structure StratifiedSample where
  sampleSize : Nat
  intermediateSample : Nat

/-- Calculates the expected number of teachers with intermediate titles in a stratified sample -/
def expectedIntermediateSample (pop : TeacherPopulation) (sample : StratifiedSample) : Rat :=
  (pop.intermediate : Rat) * sample.sampleSize / pop.total

/-- Theorem stating that the number of teachers with intermediate titles in the sample is 7 -/
theorem stratified_sample_intermediate_count 
  (pop : TeacherPopulation) 
  (sample : StratifiedSample) : 
  pop.total = 160 → 
  pop.intermediate = 56 → 
  sample.sampleSize = 20 → 
  expectedIntermediateSample pop sample = 7 := by
  sorry

#check stratified_sample_intermediate_count

end NUMINAMATH_CALUDE_stratified_sample_intermediate_count_l2629_262954


namespace NUMINAMATH_CALUDE_girls_in_class_l2629_262901

/-- Given a class with a 3:4 ratio of girls to boys and 35 total students,
    prove that the number of girls is 15. -/
theorem girls_in_class (g b : ℕ) : 
  g + b = 35 →  -- Total number of students
  4 * g = 3 * b →  -- Ratio of girls to boys is 3:4
  g = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l2629_262901


namespace NUMINAMATH_CALUDE_exponential_function_problem_l2629_262935

theorem exponential_function_problem (a : ℝ) (f : ℝ → ℝ) :
  a > 0 ∧ a ≠ 1 ∧ (∀ x, f x = a^x) ∧ f 3 = 8 →
  f (-1) = (1/2) := by
sorry

end NUMINAMATH_CALUDE_exponential_function_problem_l2629_262935


namespace NUMINAMATH_CALUDE_fraction_simplification_l2629_262905

theorem fraction_simplification :
  (2 * Real.sqrt 3) / (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5) = 
  (Real.sqrt 6 + 3 - Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2629_262905


namespace NUMINAMATH_CALUDE_inequality_solution_l2629_262936

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2629_262936


namespace NUMINAMATH_CALUDE_sixth_day_work_time_l2629_262957

def work_time (n : ℕ) : ℕ := 15 * 2^(n - 1)

theorem sixth_day_work_time :
  work_time 6 = 8 * 60 := by
  sorry

end NUMINAMATH_CALUDE_sixth_day_work_time_l2629_262957


namespace NUMINAMATH_CALUDE_ac_price_l2629_262948

/-- Given a car and an AC with prices in the ratio 3:2, where the car costs $500 more than the AC,
    prove that the price of the AC is $1000. -/
theorem ac_price (car_price ac_price : ℕ) : 
  car_price = 3 * (car_price / 5) ∧ 
  ac_price = 2 * (car_price / 5) ∧ 
  car_price = ac_price + 500 → 
  ac_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_ac_price_l2629_262948


namespace NUMINAMATH_CALUDE_triangle_side_sum_l2629_262928

theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) :
  B = 2 * A →
  Real.cos A = 4 / 5 →
  (1 / 2) * a * b * Real.sin C = 468 / 25 →
  a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l2629_262928


namespace NUMINAMATH_CALUDE_congruence_solution_l2629_262976

theorem congruence_solution (p q : Nat) (n : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Odd p ∧ Odd q ∧ n > 0 ∧
  (q^(n+2) : Nat) % (p^n) = (3^(n+2) : Nat) % (p^n) ∧
  (p^(n+2) : Nat) % (q^n) = (3^(n+2) : Nat) % (q^n) →
  p = 3 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_congruence_solution_l2629_262976


namespace NUMINAMATH_CALUDE_square_area_given_circle_l2629_262968

-- Define the circle's area
def circle_area : ℝ := 39424

-- Define the relationship between square perimeter and circle radius
def square_perimeter_equals_circle_radius (square_side : ℝ) (circle_radius : ℝ) : Prop :=
  4 * square_side = circle_radius

-- Theorem statement
theorem square_area_given_circle (square_side : ℝ) (circle_radius : ℝ) :
  circle_area = Real.pi * circle_radius^2 →
  square_perimeter_equals_circle_radius square_side circle_radius →
  square_side^2 = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_given_circle_l2629_262968


namespace NUMINAMATH_CALUDE_isosceles_max_perimeter_l2629_262918

/-- A circular arc with a fixed base and vertex angle -/
structure CircularArc where
  base : ℝ
  vertex_angle : ℝ

/-- A triangle inscribed in a circular arc -/
structure InscribedTriangle (arc : CircularArc) where
  apex : ℝ × ℝ  -- Coordinates of the apex point on the arc

/-- The perimeter of an inscribed triangle -/
def perimeter (arc : CircularArc) (triangle : InscribedTriangle arc) : ℝ :=
  sorry

/-- The isosceles triangle formed by connecting the midpoint of the arc to the base endpoints -/
def isosceles_triangle (arc : CircularArc) : InscribedTriangle arc :=
  sorry

theorem isosceles_max_perimeter (arc : CircularArc) :
  ∀ (triangle : InscribedTriangle arc),
    perimeter arc triangle ≤ perimeter arc (isosceles_triangle arc) :=
  sorry

end NUMINAMATH_CALUDE_isosceles_max_perimeter_l2629_262918


namespace NUMINAMATH_CALUDE_r_profit_share_is_one_third_of_total_l2629_262904

/-- Represents the capital and investment duration of an investor -/
structure Investor where
  capital : ℝ
  duration : ℝ

/-- Calculates the profit share of an investor -/
def profitShare (i : Investor) : ℝ := i.capital * i.duration

/-- Theorem: Given the conditions, r's share of the total profit is one-third of the total profit -/
theorem r_profit_share_is_one_third_of_total
  (p q r : Investor)
  (h1 : 4 * p.capital = 6 * q.capital)
  (h2 : 6 * q.capital = 10 * r.capital)
  (h3 : p.duration = 2)
  (h4 : q.duration = 3)
  (h5 : r.duration = 5)
  (total_profit : ℝ)
  : profitShare r = total_profit / 3 := by
  sorry

#check r_profit_share_is_one_third_of_total

end NUMINAMATH_CALUDE_r_profit_share_is_one_third_of_total_l2629_262904


namespace NUMINAMATH_CALUDE_max_queens_2017_l2629_262903

/-- Represents a chessboard of size n x n -/
def Chessboard (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a queen at position (x, y) attacks at most one other queen -/
def attacks_at_most_one (board : Chessboard 2017) (x y : Fin 2017) : Prop :=
  ∃! (x' y' : Fin 2017), x' ≠ x ∨ y' ≠ y ∧ board x' y' = true ∧
    (x' = x ∨ y' = y ∨ (x' : ℤ) - (x : ℤ) = (y' : ℤ) - (y : ℤ) ∨ 
     (x' : ℤ) - (x : ℤ) = (y : ℤ) - (y' : ℤ))

/-- The property that each queen on the board attacks at most one other queen -/
def valid_placement (board : Chessboard 2017) : Prop :=
  ∀ x y, board x y = true → attacks_at_most_one board x y

/-- Counts the number of queens on the board -/
def count_queens (board : Chessboard 2017) : ℕ :=
  (Finset.univ.filter (λ x : Fin 2017 × Fin 2017 => board x.1 x.2 = true)).card

/-- The main theorem: there exists a valid placement with 673359 queens -/
theorem max_queens_2017 : 
  ∃ (board : Chessboard 2017), valid_placement board ∧ count_queens board = 673359 :=
sorry

end NUMINAMATH_CALUDE_max_queens_2017_l2629_262903


namespace NUMINAMATH_CALUDE_sequence_sum_equals_29_l2629_262952

def sequence_term (n : ℕ) : ℤ :=
  if n % 2 = 0 then 2 + 3 * (n - 1) else -(5 + 3 * (n - 2))

def sequence_length : ℕ := 19

theorem sequence_sum_equals_29 :
  (Finset.range sequence_length).sum (λ i => sequence_term i) = 29 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_29_l2629_262952


namespace NUMINAMATH_CALUDE_factorization_of_2a2_minus_8b2_l2629_262939

theorem factorization_of_2a2_minus_8b2 (a b : ℝ) :
  2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a2_minus_8b2_l2629_262939


namespace NUMINAMATH_CALUDE_tangent_equations_not_equivalent_l2629_262978

open Real

theorem tangent_equations_not_equivalent :
  ¬(∀ x : ℝ, (tan (2 * x) - (1 / tan x) = 0) ↔ ((2 * tan x) / (1 - tan x ^ 2) - 1 / tan x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_equations_not_equivalent_l2629_262978


namespace NUMINAMATH_CALUDE_divisors_of_12m_squared_l2629_262975

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem divisors_of_12m_squared (m : ℕ) 
  (h_even : is_even m) 
  (h_divisors : count_divisors m = 7) : 
  count_divisors (12 * m^2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_12m_squared_l2629_262975


namespace NUMINAMATH_CALUDE_base5_product_132_23_l2629_262963

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base 10 number to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Multiplies two base 5 numbers -/
def multiplyBase5 (a b : List Nat) : List Nat :=
  base10ToBase5 (base5ToBase10 a * base5ToBase10 b)

theorem base5_product_132_23 :
  multiplyBase5 [2, 3, 1] [3, 2] = [1, 4, 1, 4] :=
sorry

end NUMINAMATH_CALUDE_base5_product_132_23_l2629_262963


namespace NUMINAMATH_CALUDE_count_squares_with_six_or_more_black_l2629_262949

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  position : Nat × Nat

/-- The checkerboard -/
def checkerboard : Nat := 8

/-- Function to count black squares in a given square -/
def countBlackSquares (s : Square) : Nat :=
  sorry

/-- Function to check if a square is valid (fits on the board) -/
def isValidSquare (s : Square) : Bool :=
  s.size > 0 && s.size ≤ checkerboard &&
  s.position.1 + s.size ≤ checkerboard &&
  s.position.2 + s.size ≤ checkerboard

/-- Function to generate all valid squares on the board -/
def allValidSquares : List Square :=
  sorry

/-- Main theorem -/
theorem count_squares_with_six_or_more_black : 
  (allValidSquares.filter (fun s => isValidSquare s && countBlackSquares s ≥ 6)).length = 55 :=
  sorry

end NUMINAMATH_CALUDE_count_squares_with_six_or_more_black_l2629_262949


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2629_262973

theorem complex_equation_solution (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + i) * i = b + i →
  a = 1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2629_262973


namespace NUMINAMATH_CALUDE_personal_trainer_cost_l2629_262902

-- Define the given conditions
def old_hourly_wage : ℚ := 40
def raise_percentage : ℚ := 5 / 100
def hours_per_day : ℚ := 8
def days_per_week : ℚ := 5
def old_bills : ℚ := 600
def leftover : ℚ := 980

-- Define the theorem
theorem personal_trainer_cost :
  let new_hourly_wage := old_hourly_wage * (1 + raise_percentage)
  let weekly_earnings := new_hourly_wage * hours_per_day * days_per_week
  let total_expenses := weekly_earnings - leftover
  total_expenses - old_bills = 100 := by sorry

end NUMINAMATH_CALUDE_personal_trainer_cost_l2629_262902


namespace NUMINAMATH_CALUDE_binomial_probability_l2629_262930

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV) : ℝ := X.n * X.p

/-- The probability mass function of a binomial random variable -/
def pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

/-- Theorem: For a binomial random variable X with p = 1/3 and E(X) = 2, P(X=2) = 80/243 -/
theorem binomial_probability (X : BinomialRV) 
  (h_p : X.p = 1/3) 
  (h_exp : expectedValue X = 2) : 
  pmf X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_l2629_262930


namespace NUMINAMATH_CALUDE_quadratic_positive_function_m_range_l2629_262995

/-- A function is positive on a domain if there exists a subinterval where the function maps the interval to itself -/
def PositiveFunction (f : ℝ → ℝ) (D : Set ℝ) :=
  ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧ Set.Icc a b = f '' Set.Icc a b

/-- The quadratic function g(x) = x^2 - m -/
def g (m : ℝ) : ℝ → ℝ := fun x ↦ x^2 - m

theorem quadratic_positive_function_m_range :
  (∃ m, PositiveFunction (g m) (Set.Iio 0)) → ∃ m, m ∈ Set.Ioo (3/4) 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_positive_function_m_range_l2629_262995


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2629_262962

theorem polynomial_remainder (x : ℝ) : 
  (x^15 + 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2629_262962


namespace NUMINAMATH_CALUDE_ball_cost_l2629_262915

/-- Given that Kyoko paid $4.62 for 3 balls, prove that each ball costs $1.54. -/
theorem ball_cost (total_paid : ℝ) (num_balls : ℕ) (h1 : total_paid = 4.62) (h2 : num_balls = 3) :
  total_paid / num_balls = 1.54 := by
sorry

end NUMINAMATH_CALUDE_ball_cost_l2629_262915


namespace NUMINAMATH_CALUDE_smallest_odd_island_has_nine_counties_l2629_262998

/-- A rectangular county (graphstum) -/
structure County where
  width : ℕ
  height : ℕ

/-- A rectangular island composed of counties -/
structure Island where
  counties : List County
  isRectangular : Bool
  hasDiagonalRoads : Bool
  hasClosedPath : Bool

/-- The property of having an odd number of counties -/
def hasOddCounties (i : Island) : Prop :=
  Odd (List.length i.counties)

/-- The property of being a valid island configuration -/
def isValidIsland (i : Island) : Prop :=
  i.isRectangular ∧ i.hasDiagonalRoads ∧ i.hasClosedPath ∧ hasOddCounties i

/-- The theorem stating that the smallest valid odd-county island has 9 counties -/
theorem smallest_odd_island_has_nine_counties :
  ∀ i : Island, isValidIsland i → List.length i.counties ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_island_has_nine_counties_l2629_262998


namespace NUMINAMATH_CALUDE_bruce_payment_l2629_262940

/-- The total amount Bruce paid to the shopkeeper -/
def total_amount (grape_quantity mangoe_quantity grape_rate mangoe_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mangoe_quantity * mangoe_rate

/-- Theorem stating that Bruce paid 1000 to the shopkeeper -/
theorem bruce_payment :
  total_amount 8 8 70 55 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l2629_262940


namespace NUMINAMATH_CALUDE_f_symmetric_about_pi_third_l2629_262941

/-- A function is symmetric about a point (a, 0) if f(a + x) = -f(a - x) for all x in the domain of f -/
def SymmetricAboutPoint (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = -f (a - x)

/-- The tangent function -/
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

/-- The given function f(x) = tan(x + π/6) -/
noncomputable def f (x : ℝ) : ℝ := tan (x + Real.pi / 6)

/-- Theorem stating that f(x) = tan(x + π/6) is symmetric about the point (π/3, 0) -/
theorem f_symmetric_about_pi_third : SymmetricAboutPoint f (Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_f_symmetric_about_pi_third_l2629_262941


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2629_262932

/-- A hyperbola centered at the origin -/
structure Hyperbola where
  center : ℝ × ℝ
  asymptotes : Set (ℝ → ℝ)

/-- A circle with equation (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of eccentricity for a hyperbola -/
def eccentricity (h : Hyperbola) : Set ℝ := sorry

/-- Definition of a line being tangent to a circle -/
def is_tangent (l : ℝ → ℝ) (c : Circle) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (c : Circle) :
  h.center = (0, 0) →
  c.center = (2, 0) →
  c.radius = Real.sqrt 3 →
  (∀ a ∈ h.asymptotes, is_tangent a c) →
  eccentricity h = {2, 2 * Real.sqrt 3 / 3} := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2629_262932


namespace NUMINAMATH_CALUDE_february_warmer_than_january_l2629_262956

/-- The average temperature in January 2023 in Taiyuan City (in °C) -/
def jan_temp : ℝ := -12

/-- The average temperature in February 2023 in Taiyuan City (in °C) -/
def feb_temp : ℝ := -6

/-- The difference in average temperature between February and January 2023 in Taiyuan City -/
def temp_difference : ℝ := feb_temp - jan_temp

theorem february_warmer_than_january : temp_difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_february_warmer_than_january_l2629_262956


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l2629_262912

theorem smallest_x_absolute_value_equation :
  ∃ x : ℚ, (∀ y : ℚ, |5 * y - 3| = 45 → x ≤ y) ∧ |5 * x - 3| = 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l2629_262912


namespace NUMINAMATH_CALUDE_complement_union_A_B_l2629_262909

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_union_A_B : 
  (A ∪ B)ᶜ = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l2629_262909


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l2629_262908

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -3457 [ZMOD 13] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l2629_262908


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_minus_b_zero_l2629_262926

-- Define the curve
def curve (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem tangent_line_implies_a_minus_b_zero (a b : ℝ) :
  (∃ x y, tangent_line x y ∧ y = curve a b x) →
  (tangent_line 0 b) →
  a - b = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_minus_b_zero_l2629_262926


namespace NUMINAMATH_CALUDE_valid_window_exists_l2629_262938

/-- A region in the window --/
structure Region where
  area : ℝ
  sides_equal : Bool

/-- A window configuration --/
structure Window where
  side_length : ℝ
  regions : List Region

/-- Checks if a window configuration is valid --/
def is_valid_window (w : Window) : Prop :=
  w.side_length = 1 ∧
  w.regions.length = 8 ∧
  w.regions.all (fun r => r.area = 1 / 8) ∧
  w.regions.all (fun r => r.sides_equal)

/-- Theorem: There exists a valid window configuration --/
theorem valid_window_exists : ∃ w : Window, is_valid_window w := by
  sorry


end NUMINAMATH_CALUDE_valid_window_exists_l2629_262938


namespace NUMINAMATH_CALUDE_complex_product_real_imag_equal_l2629_262925

theorem complex_product_real_imag_equal (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_equal_l2629_262925


namespace NUMINAMATH_CALUDE_rotation_to_second_quadrant_l2629_262924

/-- Given a complex number z = (-1+3i)/i, prove that rotating the point A 
    corresponding to z counterclockwise by 2π/3 radians results in a point B 
    in the second quadrant. -/
theorem rotation_to_second_quadrant (z : ℂ) : 
  z = (-1 + 3*Complex.I) / Complex.I → 
  let A := z
  let θ := 2 * Real.pi / 3
  let B := Complex.exp (Complex.I * θ) * A
  (B.re < 0 ∧ B.im > 0) := by sorry

end NUMINAMATH_CALUDE_rotation_to_second_quadrant_l2629_262924


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l2629_262929

/-- A geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℝ) (r : ℝ) : ℕ → ℝ :=
  λ n => a * r^(n - 1)

theorem tenth_term_of_specific_geometric_sequence :
  let a := 10
  let second_term := -30
  let r := second_term / a
  let seq := geometric_sequence a r
  seq 10 = -196830 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l2629_262929


namespace NUMINAMATH_CALUDE_companion_point_on_hyperbola_companion_point_on_line_companion_point_line_equation_l2629_262984

/-- Definition of companion point -/
def is_companion_point (P Q : ℝ × ℝ) : Prop :=
  Q.1 = P.1 + 2 ∧ Q.2 = P.2 - 4

/-- Theorem 1: The companion point of P(2,-1) lies on y = -20/x -/
theorem companion_point_on_hyperbola :
  ∀ Q : ℝ × ℝ, is_companion_point (2, -1) Q → Q.2 = -20 / Q.1 :=
sorry

/-- Theorem 2: If P(a,b) lies on y = x+5 and (-1,-2) is its companion point, then P = (-3,2) -/
theorem companion_point_on_line :
  ∀ P : ℝ × ℝ, P.2 = P.1 + 5 → is_companion_point P (-1, -2) → P = (-3, 2) :=
sorry

/-- Theorem 3: If P(a,b) lies on y = 2x+3, then its companion point Q lies on y = 2x-5 -/
theorem companion_point_line_equation :
  ∀ P Q : ℝ × ℝ, P.2 = 2 * P.1 + 3 → is_companion_point P Q → Q.2 = 2 * Q.1 - 5 :=
sorry

end NUMINAMATH_CALUDE_companion_point_on_hyperbola_companion_point_on_line_companion_point_line_equation_l2629_262984


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2629_262923

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficientInRange : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def originalNumber : ℕ := 44300000

/-- The scientific notation representation of the original number -/
def scientificForm : ScientificNotation := {
  coefficient := 4.43
  exponent := 7
  coefficientInRange := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (originalNumber : ℝ) = scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2629_262923
