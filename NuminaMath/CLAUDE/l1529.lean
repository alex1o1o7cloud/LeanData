import Mathlib

namespace even_product_implies_even_factor_l1529_152941

theorem even_product_implies_even_factor (a b : ℕ) : 
  a > 0 → b > 0 → Even (a * b) → Even a ∨ Even b :=
by sorry

end even_product_implies_even_factor_l1529_152941


namespace divides_cubic_minus_one_l1529_152921

theorem divides_cubic_minus_one (a : ℤ) : 
  35 ∣ (a^3 - 1) ↔ a % 35 = 1 ∨ a % 35 = 11 ∨ a % 35 = 16 := by
  sorry

end divides_cubic_minus_one_l1529_152921


namespace set_equality_l1529_152916

theorem set_equality : Finset.toSet {1, 2, 3, 4, 5} = Finset.toSet {5, 4, 3, 2, 1} := by
  sorry

end set_equality_l1529_152916


namespace triangle_isosceles_if_cosine_condition_l1529_152960

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- State the theorem
theorem triangle_isosceles_if_cosine_condition (t : Triangle) 
  (h : t.a * Real.cos t.B = t.b * Real.cos t.A) : 
  isIsosceles t := by
  sorry

end triangle_isosceles_if_cosine_condition_l1529_152960


namespace train_length_l1529_152975

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 80 * (5/18) → time = 9 → speed * time = 200 := by
  sorry

end train_length_l1529_152975


namespace square_sum_ge_double_product_l1529_152926

theorem square_sum_ge_double_product :
  (∀ x y : ℝ, x^2 + y^2 ≥ 2*x*y) ↔ (x^2 + y^2 ≥ 2*x*y) := by sorry

end square_sum_ge_double_product_l1529_152926


namespace initial_jar_state_l1529_152912

/-- Represents the initial state of the jar of balls -/
structure JarState where
  totalBalls : ℕ
  blueBalls : ℕ
  nonBlueBalls : ℕ
  hTotalSum : totalBalls = blueBalls + nonBlueBalls

/-- Represents the state of the jar after removing some blue balls -/
structure UpdatedJarState where
  initialState : JarState
  removedBlueBalls : ℕ
  newBlueBalls : ℕ
  hNewBlue : newBlueBalls = initialState.blueBalls - removedBlueBalls
  newProbability : ℚ
  hProbability : newProbability = newBlueBalls / (initialState.totalBalls - removedBlueBalls)

/-- The main theorem stating the initial number of balls in the jar -/
theorem initial_jar_state 
  (updatedState : UpdatedJarState)
  (hInitialBlue : updatedState.initialState.blueBalls = 9)
  (hRemoved : updatedState.removedBlueBalls = 5)
  (hNewProb : updatedState.newProbability = 1/5) :
  updatedState.initialState.totalBalls = 25 := by
  sorry

end initial_jar_state_l1529_152912


namespace right_triangle_check_l1529_152922

theorem right_triangle_check (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  a^2 + b^2 = c^2 :=
by sorry

end right_triangle_check_l1529_152922


namespace average_marks_first_five_subjects_l1529_152964

theorem average_marks_first_five_subjects 
  (total_subjects : Nat) 
  (average_six_subjects : ℝ) 
  (marks_sixth_subject : ℝ) 
  (h1 : total_subjects = 6) 
  (h2 : average_six_subjects = 76) 
  (h3 : marks_sixth_subject = 86) : 
  (total_subjects - 1 : ℝ)⁻¹ * (total_subjects * average_six_subjects - marks_sixth_subject) = 74 := by
  sorry

end average_marks_first_five_subjects_l1529_152964


namespace correct_change_l1529_152945

/-- The price of a red candy in won -/
def red_candy_price : ℕ := 350

/-- The price of a blue candy in won -/
def blue_candy_price : ℕ := 180

/-- The number of red candies bought -/
def red_candy_count : ℕ := 3

/-- The number of blue candies bought -/
def blue_candy_count : ℕ := 2

/-- The amount Eunseo pays in won -/
def amount_paid : ℕ := 2000

/-- The change Eunseo should receive -/
def change : ℕ := amount_paid - (red_candy_price * red_candy_count + blue_candy_price * blue_candy_count)

theorem correct_change : change = 590 := by
  sorry

end correct_change_l1529_152945


namespace candle_duration_first_scenario_l1529_152981

/-- The number of nights a candle lasts when burned for a given number of hours per night. -/
def candle_duration (hours_per_night : ℕ) : ℕ :=
  sorry

/-- The number of candles used over a given number of nights when burned for a given number of hours per night. -/
def candles_used (nights : ℕ) (hours_per_night : ℕ) : ℕ :=
  sorry

theorem candle_duration_first_scenario :
  let first_scenario_hours := 1
  let second_scenario_hours := 2
  let second_scenario_nights := 24
  let second_scenario_candles := 6
  candle_duration first_scenario_hours = 8 ∧
  candle_duration second_scenario_hours * second_scenario_candles = second_scenario_nights :=
by sorry

end candle_duration_first_scenario_l1529_152981


namespace height_comparison_l1529_152909

theorem height_comparison (a b : ℝ) (h : a = 0.6 * b) :
  (b - a) / a * 100 = 200 / 3 :=
sorry

end height_comparison_l1529_152909


namespace fraction_addition_l1529_152931

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l1529_152931


namespace band_photo_arrangement_min_band_members_l1529_152982

theorem band_photo_arrangement (n : ℕ) : n > 0 ∧ 9 ∣ n ∧ 10 ∣ n ∧ 11 ∣ n → n ≥ 990 := by
  sorry

theorem min_band_members : ∃ n : ℕ, n > 0 ∧ 9 ∣ n ∧ 10 ∣ n ∧ 11 ∣ n ∧ n = 990 := by
  sorry

end band_photo_arrangement_min_band_members_l1529_152982


namespace two_pairs_probability_l1529_152958

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The probability of rolling exactly two pairs of dice showing the same value,
    with the other two dice each showing different numbers that don't match the paired numbers,
    when rolling six standard six-sided dice once -/
def probabilityTwoPairs : ℚ :=
  25 / 72

theorem two_pairs_probability :
  probabilityTwoPairs = (
    (numFaces.choose 2) *
    (numDice.choose 2) *
    ((numDice - 2).choose 2) *
    (numFaces - 2) *
    (numFaces - 3)
  ) / (numFaces ^ numDice) :=
sorry

end two_pairs_probability_l1529_152958


namespace absent_children_absent_children_solution_l1529_152973

theorem absent_children (total_children : ℕ) (initial_bananas_per_child : ℕ) (extra_bananas : ℕ) : ℕ :=
  let total_bananas := total_children * initial_bananas_per_child
  let final_bananas_per_child := initial_bananas_per_child + extra_bananas
  let absent_children := total_children - (total_bananas / final_bananas_per_child)
  absent_children

theorem absent_children_solution :
  absent_children 320 2 2 = 160 := by
  sorry

end absent_children_absent_children_solution_l1529_152973


namespace upstream_speed_l1529_152910

/-- The speed of a man rowing upstream, given his speed in still water and the speed of the stream -/
theorem upstream_speed (downstream_speed still_water_speed stream_speed : ℝ) :
  downstream_speed = still_water_speed + stream_speed →
  still_water_speed > 0 →
  stream_speed > 0 →
  still_water_speed > stream_speed →
  (still_water_speed - stream_speed : ℝ) = 6 :=
by sorry

end upstream_speed_l1529_152910


namespace shirt_discount_calculation_l1529_152937

/-- Given an original price and a discounted price, calculate the percentage discount -/
def calculate_discount (original_price discounted_price : ℚ) : ℚ :=
  (original_price - discounted_price) / original_price * 100

/-- Theorem: The percentage discount for a shirt with original price 933.33 and discounted price 560 is 40% -/
theorem shirt_discount_calculation :
  let original_price : ℚ := 933.33
  let discounted_price : ℚ := 560
  calculate_discount original_price discounted_price = 40 :=
by
  sorry

#eval calculate_discount 933.33 560

end shirt_discount_calculation_l1529_152937


namespace mixed_number_sum_l1529_152949

theorem mixed_number_sum : (2 + 1/10) + (3 + 11/100) = 5.21 := by
  sorry

end mixed_number_sum_l1529_152949


namespace olympic_high_school_quiz_l1529_152920

theorem olympic_high_school_quiz (f s : ℚ) 
  (h1 : f > 0) 
  (h2 : s > 0) 
  (h3 : (3/7) * f = (5/7) * s) : 
  f = (5/3) * s :=
sorry

end olympic_high_school_quiz_l1529_152920


namespace stephanies_internet_bill_l1529_152996

/-- Stephanie's household budget problem -/
theorem stephanies_internet_bill :
  let electricity_bill : ℕ := 60
  let gas_bill : ℕ := 40
  let water_bill : ℕ := 40
  let gas_paid : ℚ := 3/4 * gas_bill + 5
  let water_paid : ℚ := 1/2 * water_bill
  let internet_payments : ℕ := 4
  let internet_payment_amount : ℕ := 5
  let total_remaining : ℕ := 30
  
  ∃ (internet_bill : ℕ),
    internet_bill = internet_payments * internet_payment_amount + 
      (total_remaining - (gas_bill - gas_paid) - (water_bill - water_paid)) :=
by
  sorry


end stephanies_internet_bill_l1529_152996


namespace partial_fraction_decomposition_l1529_152904

theorem partial_fraction_decomposition (M₁ M₂ : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (50 * x - 42) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) → 
  M₁ * M₂ = -6264 := by
sorry

end partial_fraction_decomposition_l1529_152904


namespace total_hotdogs_is_480_l1529_152918

/-- The number of hotdogs Helen's mother brought -/
def helen_hotdogs : ℕ := 101

/-- The number of hotdogs Dylan's mother brought -/
def dylan_hotdogs : ℕ := 379

/-- The total number of hotdogs -/
def total_hotdogs : ℕ := helen_hotdogs + dylan_hotdogs

/-- Theorem stating that the total number of hotdogs is 480 -/
theorem total_hotdogs_is_480 : total_hotdogs = 480 := by
  sorry

end total_hotdogs_is_480_l1529_152918


namespace paint_mixture_ratio_l1529_152983

/-- Given a paint mixture with a ratio of red:yellow:white as 5:3:7,
    if 21 quarts of white paint are used, then 15 quarts of red paint should be used. -/
theorem paint_mixture_ratio (red yellow white : ℚ) :
  red / white = 5 / 7 →
  yellow / white = 3 / 7 →
  white = 21 →
  red = 15 := by
  sorry

end paint_mixture_ratio_l1529_152983


namespace base4_calculation_l1529_152930

/-- Converts a base 4 number to base 10 --/
def base4_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10_to_base4 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 4 numbers --/
def mul_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a * base4_to_base10 b)

/-- Divides a base 4 number by another base 4 number --/
def div_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a / base4_to_base10 b)

theorem base4_calculation : 
  div_base4 (mul_base4 231 24) 3 = 1130 := by sorry

end base4_calculation_l1529_152930


namespace jessica_found_41_seashells_l1529_152927

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 18

/-- The total number of seashells Mary and Jessica found together -/
def total_seashells : ℕ := 59

/-- The number of seashells Jessica found -/
def jessica_seashells : ℕ := total_seashells - mary_seashells

theorem jessica_found_41_seashells : jessica_seashells = 41 := by
  sorry

end jessica_found_41_seashells_l1529_152927


namespace right_triangle_angle_calculation_l1529_152955

theorem right_triangle_angle_calculation (A B C : Real) 
  (h1 : A + B + C = 180) -- Sum of angles in a triangle is 180°
  (h2 : C = 90) -- Angle C is 90°
  (h3 : A = 35.5) -- Angle A is 35.5°
  : B = 54.5 := by
  sorry

end right_triangle_angle_calculation_l1529_152955


namespace projection_equals_three_l1529_152901

/-- Given vectors a and b in ℝ², with a specific angle between them, 
    prove that the projection of b onto a is 3. -/
theorem projection_equals_three (a b : ℝ × ℝ) (angle : ℝ) : 
  a = (1, Real.sqrt 3) → 
  b = (3, Real.sqrt 3) → 
  angle = π / 6 → 
  (b.1 * a.1 + b.2 * a.2) / Real.sqrt (a.1^2 + a.2^2) = 3 := by
  sorry

end projection_equals_three_l1529_152901


namespace quadratic_complete_square_l1529_152979

theorem quadratic_complete_square (a b c : ℝ) (h : a = 4 ∧ b = -16 ∧ c = -200) :
  ∃ q t : ℝ, (∀ x, a * x^2 + b * x + c = 0 ↔ (x + q)^2 = t) ∧ t = 54 :=
sorry

end quadratic_complete_square_l1529_152979


namespace interest_rate_middle_period_l1529_152938

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_middle_period 
  (principal : ℝ) 
  (rate1 : ℝ) 
  (rate3 : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (time3 : ℝ) 
  (total_interest : ℝ) :
  principal = 8000 →
  rate1 = 0.08 →
  rate3 = 0.12 →
  time1 = 4 →
  time2 = 6 →
  time3 = 5 →
  total_interest = 12160 →
  ∃ (rate2 : ℝ), 
    rate2 = 0.1 ∧
    total_interest = simple_interest principal rate1 time1 + 
                     simple_interest principal rate2 time2 + 
                     simple_interest principal rate3 time3 :=
by sorry

end interest_rate_middle_period_l1529_152938


namespace cost_price_calculation_l1529_152924

/-- Given an item with a cost price, this theorem proves that if the item is priced at 1.5 times
    its cost price and sold with a 40% profit after a 20 yuan discount, then the cost price
    of the item is 200 yuan. -/
theorem cost_price_calculation (cost_price : ℝ) : 
  (1.5 * cost_price - 20 - cost_price = 0.4 * cost_price) → cost_price = 200 := by
  sorry

end cost_price_calculation_l1529_152924


namespace rectangles_in_5x5_grid_l1529_152932

/-- The number of ways to choose 2 items from 5 items -/
def choose_2_from_5 : ℕ := 10

/-- The number of rectangles in a 5x5 grid -/
def num_rectangles : ℕ := choose_2_from_5 * choose_2_from_5

theorem rectangles_in_5x5_grid :
  num_rectangles = 100 := by sorry

end rectangles_in_5x5_grid_l1529_152932


namespace floor_difference_inequality_l1529_152935

theorem floor_difference_inequality (x y : ℝ) : 
  ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ := by sorry

end floor_difference_inequality_l1529_152935


namespace base_video_card_cost_l1529_152952

/-- Proves the cost of the base video card given the costs of other components --/
theorem base_video_card_cost 
  (computer_cost : ℝ)
  (peripheral_cost : ℝ)
  (upgraded_card_cost : ℝ → ℝ)
  (total_cost : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : peripheral_cost = computer_cost / 5)
  (h3 : ∀ x, upgraded_card_cost x = 2 * x)
  (h4 : total_cost = 2100)
  (h5 : ∃ x, computer_cost + peripheral_cost + upgraded_card_cost x = total_cost) :
  ∃ x, x = 150 ∧ computer_cost + peripheral_cost + upgraded_card_cost x = total_cost :=
by sorry

end base_video_card_cost_l1529_152952


namespace karen_has_128_crayons_l1529_152911

/-- The number of crayons in Judah's box -/
def judah_crayons : ℕ := 8

/-- The number of crayons in Gilbert's box -/
def gilbert_crayons : ℕ := 4 * judah_crayons

/-- The number of crayons in Beatrice's box -/
def beatrice_crayons : ℕ := 2 * gilbert_crayons

/-- The number of crayons in Karen's box -/
def karen_crayons : ℕ := 2 * beatrice_crayons

/-- Theorem stating that Karen's box contains 128 crayons -/
theorem karen_has_128_crayons : karen_crayons = 128 := by
  sorry

end karen_has_128_crayons_l1529_152911


namespace compare_negative_fractions_l1529_152969

theorem compare_negative_fractions : -3/4 > -4/5 := by
  sorry

end compare_negative_fractions_l1529_152969


namespace cricket_average_l1529_152967

/-- Calculates the average score for the last 4 matches of a cricket series -/
theorem cricket_average (total_matches : ℕ) (first_matches : ℕ) (total_average : ℚ) (first_average : ℚ) :
  total_matches = 10 →
  first_matches = 6 →
  total_average = 389/10 →
  first_average = 42 →
  (total_matches * total_average - first_matches * first_average) / (total_matches - first_matches) = 137/4 := by
  sorry

end cricket_average_l1529_152967


namespace inequality_solutions_l1529_152908

-- Define the solution sets
def solution_set1 : Set ℝ := {x | -3/2 < x ∧ x < 3}
def solution_set2 : Set ℝ := {x | -5 < x ∧ x ≤ 3/2}
def solution_set3 : Set ℝ := ∅
def solution_set4 : Set ℝ := Set.univ

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := -2*x^2 + 3*x + 9 > 0
def inequality2 (x : ℝ) : Prop := (8 - x) / (5 + x) > 1
def inequality3 (x : ℝ) : Prop := -x^2 + 2*x - 3 > 0
def inequality4 (x : ℝ) : Prop := x^2 - 14*x + 50 > 0

-- Theorem statements
theorem inequality_solutions :
  (∀ x, x ∈ solution_set1 ↔ inequality1 x) ∧
  (∀ x, x ∈ solution_set2 ↔ inequality2 x) ∧
  (∀ x, x ∈ solution_set3 ↔ inequality3 x) ∧
  (∀ x, x ∈ solution_set4 ↔ inequality4 x) :=
by sorry

end inequality_solutions_l1529_152908


namespace inequality_proof_l1529_152913

theorem inequality_proof (a A b B : ℝ) 
  (h1 : |A - 3*a| ≤ 1 - a)
  (h2 : |B - 3*b| ≤ 1 - b)
  (ha : a > 0)
  (hb : b > 0) :
  |A*B/3 - 3*a*b| - 3*a*b ≤ 1 - a*b :=
by sorry

end inequality_proof_l1529_152913


namespace smallest_k_no_real_roots_l1529_152954

theorem smallest_k_no_real_roots : 
  ∃ k : ℕ, k = 4 ∧ 
  (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 7 ≠ 0) ∧
  (∀ m : ℕ, m < k → ∃ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 7 = 0) := by
  sorry

end smallest_k_no_real_roots_l1529_152954


namespace point_coordinates_l1529_152951

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem stating the coordinates of point P given the conditions -/
theorem point_coordinates (P : Point) 
  (h1 : SecondQuadrant P) 
  (h2 : DistanceToXAxis P = 4) 
  (h3 : DistanceToYAxis P = 3) : 
  P.x = -3 ∧ P.y = 4 := by
  sorry


end point_coordinates_l1529_152951


namespace randys_trip_length_l1529_152972

theorem randys_trip_length :
  ∀ (total_length : ℝ),
  (total_length / 2 : ℝ) + 30 + (total_length / 4 : ℝ) = total_length →
  total_length = 120 := by
sorry

end randys_trip_length_l1529_152972


namespace perfect_line_fit_l1529_152966

/-- A sample point in a 2D plane -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : SamplePoint) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Calculate the sum of squared residuals -/
def sumSquaredResiduals (points : List SamplePoint) (l : Line) : ℝ :=
  (points.map (fun p => (p.y - (l.slope * p.x + l.intercept))^2)).sum

/-- Calculate the correlation coefficient -/
def correlationCoefficient (points : List SamplePoint) : ℝ :=
  sorry  -- Actual calculation of correlation coefficient

/-- Theorem: If all sample points fall on a straight line, 
    then the sum of squared residuals is 0 and 
    the absolute value of the correlation coefficient is 1 -/
theorem perfect_line_fit (points : List SamplePoint) (l : Line) :
  (∀ p ∈ points, pointOnLine p l) →
  sumSquaredResiduals points l = 0 ∧ |correlationCoefficient points| = 1 := by
  sorry

end perfect_line_fit_l1529_152966


namespace expression_simplification_and_evaluation_l1529_152968

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2022
  let y : ℝ := -Real.sqrt 2
  4 * x * y + (2 * x - y) * (2 * x + y) - (2 * x + y)^2 = -4 := by
  sorry

end expression_simplification_and_evaluation_l1529_152968


namespace min_value_inequality_l1529_152917

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≥ 2) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 ≥ 1/4 := by
  sorry

end min_value_inequality_l1529_152917


namespace quadratic_equation_solution_l1529_152914

theorem quadratic_equation_solution (x : ℝ) (h1 : x^2 - 6*x = 0) (h2 : x ≠ 0) : x = 6 := by
  sorry

end quadratic_equation_solution_l1529_152914


namespace divisibility_by_101_l1529_152947

theorem divisibility_by_101 (a b : ℕ) : 
  a < 10 → b < 10 → 
  (12 * 10^10 + a * 10^9 + b * 10^8 + 9876543) % 101 = 0 → 
  10 * a + b = 58 := by
  sorry

end divisibility_by_101_l1529_152947


namespace intersection_M_N_l1529_152902

def M : Set ℝ := {x | x^2 ≤ 1}
def N : Set ℝ := {-2, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l1529_152902


namespace six_people_arrangement_l1529_152900

/-- The number of ways to arrange n people in a line with two specific people always next to each other -/
def arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- Theorem: For 6 people with two specific people always next to each other, there are 240 possible arrangements -/
theorem six_people_arrangement : arrangements 6 = 240 := by
  sorry

end six_people_arrangement_l1529_152900


namespace min_tests_required_l1529_152965

/-- Represents a set of batteries -/
def Battery := Fin 8

/-- Represents a pair of batteries -/
def BatteryPair := Battery × Battery

/-- Represents the state of a battery -/
inductive BatteryState
| Good
| Bad

/-- The total number of batteries -/
def totalBatteries : Nat := 8

/-- The number of good batteries -/
def goodBatteries : Nat := 4

/-- The number of bad batteries -/
def badBatteries : Nat := 4

/-- A function that determines if a pair of batteries works -/
def works (pair : BatteryPair) (state : Battery → BatteryState) : Prop :=
  state pair.1 = BatteryState.Good ∧ state pair.2 = BatteryState.Good

/-- The main theorem stating the minimum number of tests required -/
theorem min_tests_required :
  ∀ (state : Battery → BatteryState),
  (∃ (good : Finset Battery), good.card = goodBatteries ∧ ∀ b ∈ good, state b = BatteryState.Good) →
  ∃ (tests : Finset BatteryPair),
    tests.card = 7 ∧
    (∀ (pair : BatteryPair), works pair state → pair ∈ tests) ∧
    ∀ (tests' : Finset BatteryPair),
      tests'.card < 7 →
      ∃ (pair : BatteryPair), works pair state ∧ pair ∉ tests' :=
sorry

end min_tests_required_l1529_152965


namespace distinct_triangles_in_grid_l1529_152999

/-- The number of points in each row of the grid -/
def rows : ℕ := 3

/-- The number of points in each column of the grid -/
def columns : ℕ := 4

/-- The total number of points in the grid -/
def total_points : ℕ := rows * columns

/-- The number of degenerate cases (collinear points) -/
def degenerate_cases : ℕ := rows + columns + 2

theorem distinct_triangles_in_grid : 
  (total_points.choose 3) - degenerate_cases = 76 := by sorry

end distinct_triangles_in_grid_l1529_152999


namespace fraction_division_problem_l1529_152977

theorem fraction_division_problem : (4 + 2 / 3) / (9 / 7) = 98 / 27 := by
  sorry

end fraction_division_problem_l1529_152977


namespace quadratic_completing_square_l1529_152989

theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 - 4*x - 5 = 0 ↔ (x - 2)^2 = 9 :=
by sorry

end quadratic_completing_square_l1529_152989


namespace race_head_start_l1529_152929

theorem race_head_start (L : ℝ) (vA vB : ℝ) (h : vA = (17/15) * vB) :
  let d := (2/17) * L
  L / vA = (L - d) / vB := by sorry

end race_head_start_l1529_152929


namespace fifteen_mangoes_make_120_lassis_l1529_152925

/-- Given that 3 mangoes can make 24 lassis, this function calculates
    the number of lassis that can be made from a given number of mangoes. -/
def lassisFromMangoes (mangoes : ℕ) : ℕ :=
  (24 * mangoes) / 3

/-- Theorem stating that 15 mangoes will produce 120 lassis -/
theorem fifteen_mangoes_make_120_lassis :
  lassisFromMangoes 15 = 120 := by
  sorry

#eval lassisFromMangoes 15

end fifteen_mangoes_make_120_lassis_l1529_152925


namespace trigonometric_equation_proof_l1529_152915

theorem trigonometric_equation_proof : 
  4.74 * (Real.cos (64 * π / 180) * Real.cos (4 * π / 180) - 
          Real.cos (86 * π / 180) * Real.cos (26 * π / 180)) / 
         (Real.cos (71 * π / 180) * Real.cos (41 * π / 180) - 
          Real.cos (49 * π / 180) * Real.cos (19 * π / 180)) = -1 := by
  sorry

end trigonometric_equation_proof_l1529_152915


namespace skittles_theorem_l1529_152978

def skittles_problem (brandon_initial bonnie_initial brandon_loss : ℕ) : Prop :=
  let brandon_after_loss := brandon_initial - brandon_loss
  let combined := brandon_after_loss + bonnie_initial
  let each_share := combined / 4
  let chloe_initial := each_share
  let dylan_initial := each_share
  let chloe_to_dylan := chloe_initial / 2
  let dylan_after_chloe := dylan_initial + chloe_to_dylan
  let dylan_to_bonnie := dylan_after_chloe / 3
  let dylan_final := dylan_after_chloe - dylan_to_bonnie
  dylan_final = 22

theorem skittles_theorem : skittles_problem 96 4 9 := by sorry

end skittles_theorem_l1529_152978


namespace spaceDivisions_correct_l1529_152991

/-- The number of parts that n planes can divide space into, given that
    each group of three planes intersects at one point and no group of
    four planes has a common point. -/
def spaceDivisions (n : ℕ) : ℚ :=
  (n^3 + 5*n + 6) / 6

/-- Theorem stating that spaceDivisions correctly calculates the number
    of parts that n planes can divide space into. -/
theorem spaceDivisions_correct (n : ℕ) :
  spaceDivisions n = (n^3 + 5*n + 6) / 6 :=
by sorry

end spaceDivisions_correct_l1529_152991


namespace odd_function_value_l1529_152943

def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem odd_function_value (a b c : ℝ) :
  (∀ x, f a b c (-x) = -(f a b c x)) →
  (∀ x ∈ Set.Icc (2*b - 5) (2*b - 3), f a b c x ∈ Set.range (f a b c)) →
  f a b c (1/2) = 9/8 := by
sorry

end odd_function_value_l1529_152943


namespace car_trip_speed_l1529_152994

/-- Proves that given the conditions of the car trip, the speed for the remaining part is 20 mph -/
theorem car_trip_speed (D : ℝ) (h_D_pos : D > 0) : 
  let first_part := 0.8 * D
  let second_part := 0.2 * D
  let first_speed := 80
  let total_avg_speed := 50
  let v := (first_speed * total_avg_speed * second_part) / 
           (first_speed * D - total_avg_speed * first_part)
  v = 20 := by
  sorry

end car_trip_speed_l1529_152994


namespace sum_norms_gt_sum_pairwise_norms_l1529_152995

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

/-- Given four pairwise non-parallel vectors whose sum is zero, 
    the sum of their norms is greater than the sum of the norms of their pairwise sums with the first vector -/
theorem sum_norms_gt_sum_pairwise_norms (a b c d : V) 
    (h_sum : a + b + c + d = 0)
    (h_ab : ¬ ∃ (k : ℝ), b = k • a)
    (h_ac : ¬ ∃ (k : ℝ), c = k • a)
    (h_ad : ¬ ∃ (k : ℝ), d = k • a)
    (h_bc : ¬ ∃ (k : ℝ), c = k • b)
    (h_bd : ¬ ∃ (k : ℝ), d = k • b)
    (h_cd : ¬ ∃ (k : ℝ), d = k • c) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ > ‖a + b‖ + ‖a + c‖ + ‖a + d‖ := by
  sorry

end sum_norms_gt_sum_pairwise_norms_l1529_152995


namespace abs_equation_sufficient_not_necessary_l1529_152919

/-- The distance from a point to the x-axis --/
def dist_to_x_axis (y : ℝ) : ℝ := |y|

/-- The distance from a point to the y-axis --/
def dist_to_y_axis (x : ℝ) : ℝ := |x|

/-- The condition that distances to both axes are equal --/
def equal_dist_to_axes (x y : ℝ) : Prop :=
  dist_to_x_axis y = dist_to_y_axis x

/-- The equation y = |x| --/
def abs_equation (x y : ℝ) : Prop := y = |x|

/-- Theorem stating that y = |x| is a sufficient but not necessary condition --/
theorem abs_equation_sufficient_not_necessary :
  (∀ x y : ℝ, abs_equation x y → equal_dist_to_axes x y) ∧
  ¬(∀ x y : ℝ, equal_dist_to_axes x y → abs_equation x y) :=
sorry

end abs_equation_sufficient_not_necessary_l1529_152919


namespace original_savings_calculation_l1529_152987

theorem original_savings_calculation (savings : ℝ) : 
  (4 / 5 : ℝ) * savings + 100 = savings → savings = 500 := by
  sorry

end original_savings_calculation_l1529_152987


namespace at_least_one_negative_l1529_152980

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  (a < 0) ∨ (b < 0) ∨ (c < 0) ∨ (d < 0) := by
  sorry

end at_least_one_negative_l1529_152980


namespace fixed_point_of_f_l1529_152963

/-- The logarithmic function with base a > 0 and a ≠ 1 -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = 1 + log_a(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + log a (x - 1)

/-- Theorem: For any base a > 0 and a ≠ 1, f(x) passes through the point (2,1) -/
theorem fixed_point_of_f (a : ℝ) (ha : a > 0 ∧ a ≠ 1) : f a 2 = 1 := by
  sorry

end fixed_point_of_f_l1529_152963


namespace find_a_l1529_152940

theorem find_a (A B : Set ℝ) (a : ℝ) :
  A = {-1, 1, 3} →
  B = {2, 2^a - 1} →
  A ∩ B = {1} →
  a = 1 := by
  sorry

end find_a_l1529_152940


namespace equality_of_exponents_l1529_152944

-- Define variables
variable (a b c d : ℝ)
variable (x y q z : ℝ)

-- State the theorem
theorem equality_of_exponents 
  (h1 : a^x = c^q) 
  (h2 : a^x = b) 
  (h3 : c^y = a^z) 
  (h4 : c^y = d) 
  : x * y = q * z := by
  sorry

end equality_of_exponents_l1529_152944


namespace student_arrangement_equality_l1529_152992

/-- The number of ways to arrange k items out of n items -/
def arrange (n k : ℕ) : ℕ := sorry

theorem student_arrangement_equality (n : ℕ) :
  arrange (2*n) (2*n) = arrange (2*n) n * arrange n n := by sorry

end student_arrangement_equality_l1529_152992


namespace more_cars_difference_l1529_152993

/-- The number of cars Tommy has -/
def tommy_cars : ℕ := 3

/-- The number of cars Jessie has -/
def jessie_cars : ℕ := 3

/-- The total number of cars all three have -/
def total_cars : ℕ := 17

/-- The number of cars Jessie's older brother has -/
def brother_cars : ℕ := total_cars - (tommy_cars + jessie_cars)

theorem more_cars_difference : brother_cars - (tommy_cars + jessie_cars) = 5 := by
  sorry

end more_cars_difference_l1529_152993


namespace geometric_sequence_a5_l1529_152928

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → q = 2 → a 3 = 3 → a 5 = 12 := by
  sorry

end geometric_sequence_a5_l1529_152928


namespace prob_sum_ge_12_is_zero_l1529_152956

-- Define a uniform random variable on (0,1)
def uniform_01 : Type := {x : ℝ // 0 < x ∧ x < 1}

-- Define the sum of 5 such variables
def sum_5_uniform (X₁ X₂ X₃ X₄ X₅ : uniform_01) : ℝ :=
  X₁.val + X₂.val + X₃.val + X₄.val + X₅.val

-- State the theorem
theorem prob_sum_ge_12_is_zero :
  ∀ X₁ X₂ X₃ X₄ X₅ : uniform_01,
  sum_5_uniform X₁ X₂ X₃ X₄ X₅ < 12 :=
by sorry

end prob_sum_ge_12_is_zero_l1529_152956


namespace circle_center_from_diameter_endpoints_l1529_152998

/-- The center of a circle given the endpoints of its diameter -/
theorem circle_center_from_diameter_endpoints (x₁ y₁ x₂ y₂ : ℝ) :
  let endpoint1 : ℝ × ℝ := (x₁, y₁)
  let endpoint2 : ℝ × ℝ := (x₂, y₂)
  let center : ℝ × ℝ := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  endpoint1 = (2, -3) → endpoint2 = (8, 9) → center = (5, 3) :=
by sorry

end circle_center_from_diameter_endpoints_l1529_152998


namespace decreasing_f_implies_a_range_l1529_152934

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  1/7 < a ∧ a < 1/3 :=
by
  sorry

end decreasing_f_implies_a_range_l1529_152934


namespace flower_shop_problem_l1529_152939

/-- The number of flowers brought at dawn -/
def flowers_at_dawn : ℕ := 300

/-- The fraction of flowers sold in the morning -/
def morning_sale_fraction : ℚ := 3/5

/-- The total number of flowers sold in the afternoon -/
def afternoon_sales : ℕ := 180

theorem flower_shop_problem :
  (flowers_at_dawn : ℚ) * morning_sale_fraction = afternoon_sales ∧
  (flowers_at_dawn : ℚ) * morning_sale_fraction = (flowers_at_dawn : ℚ) * (1 - morning_sale_fraction) + (afternoon_sales - (flowers_at_dawn : ℚ) * (1 - morning_sale_fraction)) :=
by sorry

end flower_shop_problem_l1529_152939


namespace picnic_attendance_theorem_l1529_152986

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Theorem: Given the conditions of the picnic, the total number of attendees is 240 -/
theorem picnic_attendance_theorem (p : PicnicAttendance) 
  (h1 : p.men = p.women + 80)
  (h2 : p.adults = p.children + 80)
  (h3 : p.men = 120)
  : p.adults + p.children = 240 := by
  sorry

#check picnic_attendance_theorem

end picnic_attendance_theorem_l1529_152986


namespace least_integer_greater_than_sqrt_500_l1529_152933

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
by sorry

end least_integer_greater_than_sqrt_500_l1529_152933


namespace point_C_representation_l1529_152974

def point_A : ℝ := -2

def point_B : ℝ := point_A - 2

def distance_BC : ℝ := 5

theorem point_C_representation :
  ∃ (point_C : ℝ), (point_C = point_B - distance_BC ∨ point_C = point_B + distance_BC) ∧
                    (point_C = -9 ∨ point_C = 1) := by
  sorry

end point_C_representation_l1529_152974


namespace regular_octagon_interior_angle_l1529_152984

/-- The measure of each interior angle in a regular octagon -/
theorem regular_octagon_interior_angle : ℝ := by
  -- Define the number of sides in an octagon
  let n : ℕ := 8

  -- Define the formula for the sum of interior angles of an n-sided polygon
  let sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

  -- Define the measure of each interior angle in a regular n-gon
  let interior_angle (n : ℕ) : ℝ := sum_interior_angles n / n

  -- Prove that the measure of each interior angle in a regular octagon is 135°
  have h : interior_angle n = 135 := by sorry

  -- Return the result
  exact 135


end regular_octagon_interior_angle_l1529_152984


namespace number_problem_l1529_152957

theorem number_problem : 
  ∃ x : ℚ, (x / 5 = 3 * (x / 6) - 40) ∧ (x = 400 / 3) := by
  sorry

end number_problem_l1529_152957


namespace marcus_percentage_of_team_points_l1529_152903

def marcus_three_pointers : ℕ := 5
def marcus_two_pointers : ℕ := 10
def team_total_points : ℕ := 70

def marcus_points : ℕ := marcus_three_pointers * 3 + marcus_two_pointers * 2

theorem marcus_percentage_of_team_points :
  (marcus_points : ℚ) / team_total_points * 100 = 50 := by
  sorry

end marcus_percentage_of_team_points_l1529_152903


namespace percent_change_equality_l1529_152946

theorem percent_change_equality (x y : ℝ) (p : ℝ) 
  (h1 : x ≠ 0)
  (h2 : y = x * (1 + 0.15) * (1 - p / 100))
  (h3 : y = x) : 
  p = 15 := by
sorry

end percent_change_equality_l1529_152946


namespace fourth_root_of_2560000_l1529_152906

theorem fourth_root_of_2560000 : (2560000 : ℝ) ^ (1/4 : ℝ) = 40 := by sorry

end fourth_root_of_2560000_l1529_152906


namespace products_from_equipment_B_l1529_152936

/-- Given a total number of products and a stratified sample, 
    calculate the number of products produced by equipment B -/
theorem products_from_equipment_B 
  (total : ℕ) 
  (sample_size : ℕ) 
  (sample_A : ℕ) 
  (h1 : total = 4800)
  (h2 : sample_size = 80)
  (h3 : sample_A = 50) : 
  total - (total * sample_A / sample_size) = 1800 :=
by sorry

end products_from_equipment_B_l1529_152936


namespace basketball_lineup_count_l1529_152997

def total_players : ℕ := 16
def twins : ℕ := 2
def seniors : ℕ := 5
def lineup_size : ℕ := 7

/-- The number of ways to choose a lineup of 7 players from a team of 16 players,
    including a set of twins and 5 seniors, where exactly one twin must be in the lineup
    and at least two seniors must be selected. -/
theorem basketball_lineup_count : 
  (Nat.choose twins 1) *
  (Nat.choose seniors 2 * Nat.choose (total_players - twins - seniors) 4 +
   Nat.choose seniors 3 * Nat.choose (total_players - twins - seniors) 3) = 4200 := by
  sorry

end basketball_lineup_count_l1529_152997


namespace root_product_sum_l1529_152907

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x, Real.sqrt 2023 * x^3 - 4047 * x^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 3 := by
sorry

end root_product_sum_l1529_152907


namespace min_value_theorem_l1529_152953

theorem min_value_theorem (a : ℝ) (ha : a > 0) :
  (∃ (x_min : ℝ), x_min > 0 ∧
    ∀ (x : ℝ), x > 0 → (a^2 + x^2) / x ≥ (a^2 + x_min^2) / x_min) ∧
  (∀ (x : ℝ), x > 0 → (a^2 + x^2) / x ≥ 2*a) := by
  sorry

end min_value_theorem_l1529_152953


namespace fraction_inequality_l1529_152976

theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / (a - c) > e / (b - d) := by
sorry

end fraction_inequality_l1529_152976


namespace min_value_of_function_l1529_152985

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  x + 2 / (2 * x + 1) - 1 ≥ 1 / 2 ∧ 
  ∃ y > 0, y + 2 / (2 * y + 1) - 1 = 1 / 2 := by
sorry

end min_value_of_function_l1529_152985


namespace typeC_migration_time_l1529_152905

/-- Represents the lakes in the migration sequence -/
inductive Lake : Type
| Jim : Lake
| Disney : Lake
| London : Lake
| Everest : Lake

/-- Defines the distance between two lakes -/
def distance (a b : Lake) : ℝ :=
  match a, b with
  | Lake.Jim, Lake.Disney => 42
  | Lake.Disney, Lake.London => 57
  | Lake.London, Lake.Everest => 65
  | Lake.Everest, Lake.Jim => 70
  | _, _ => 0  -- For all other combinations

/-- Calculates the total distance of one complete sequence -/
def totalDistance : ℝ :=
  distance Lake.Jim Lake.Disney +
  distance Lake.Disney Lake.London +
  distance Lake.London Lake.Everest +
  distance Lake.Everest Lake.Jim

/-- The average speed of Type C birds in miles per hour -/
def typeCSpeed : ℝ := 12

/-- Theorem: Type C birds take 39 hours to complete two full sequences -/
theorem typeC_migration_time :
  2 * (totalDistance / typeCSpeed) = 39 := by sorry


end typeC_migration_time_l1529_152905


namespace brothers_combined_age_l1529_152988

theorem brothers_combined_age : 
  ∀ (x y : ℕ), (x - 6 + y - 6 = 100) → (x + y = 112) :=
by
  sorry

end brothers_combined_age_l1529_152988


namespace kyle_lifts_320_l1529_152942

/-- Kyle's lifting capacity over the years -/
structure KylesLift where
  two_years_ago : ℝ
  last_year : ℝ
  this_year : ℝ

/-- Given information about Kyle's lifting capacity -/
def kyle_info (k : KylesLift) : Prop :=
  k.this_year = 1.6 * k.last_year ∧
  0.6 * k.last_year = 3 * k.two_years_ago ∧
  k.two_years_ago = 40

/-- Theorem: Kyle can lift 320 pounds this year -/
theorem kyle_lifts_320 (k : KylesLift) (h : kyle_info k) : k.this_year = 320 := by
  sorry


end kyle_lifts_320_l1529_152942


namespace cubic_complex_equation_l1529_152971

theorem cubic_complex_equation (a b c : ℕ+) :
  c = Complex.I.re * ((a + Complex.I * b) ^ 3 - 107 * Complex.I) →
  c = 198 := by
sorry

end cubic_complex_equation_l1529_152971


namespace set_intersection_condition_l1529_152948

theorem set_intersection_condition (m : ℝ) : 
  let A := {x : ℝ | x^2 - 3*x + 2 = 0}
  let C := {x : ℝ | x^2 - m*x + 2 = 0}
  (A ∩ C = C) ↔ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
sorry

end set_intersection_condition_l1529_152948


namespace line_equation_equivalence_l1529_152962

/-- Given a line described by the dot product equation (-1, 2) · ((x, y) - (3, -4)) = 0,
    prove that it is equivalent to the line y = (1/2)x - 11/2 -/
theorem line_equation_equivalence (x y : ℝ) :
  (-1 : ℝ) * (x - 3) + 2 * (y - (-4)) = 0 ↔ y = (1/2 : ℝ) * x - 11/2 := by
  sorry

end line_equation_equivalence_l1529_152962


namespace initial_tickets_count_l1529_152959

/-- The number of tickets sold in the first week -/
def first_week_sales : ℕ := 38

/-- The number of tickets sold in the second week -/
def second_week_sales : ℕ := 17

/-- The number of tickets left to sell -/
def remaining_tickets : ℕ := 35

/-- The initial number of tickets -/
def initial_tickets : ℕ := first_week_sales + second_week_sales + remaining_tickets

theorem initial_tickets_count : initial_tickets = 90 := by
  sorry

end initial_tickets_count_l1529_152959


namespace postman_pete_mileage_l1529_152950

def pedometer_max : ℕ := 99999
def flips_in_year : ℕ := 50
def last_day_steps : ℕ := 25000
def steps_per_mile : ℕ := 1500

def total_steps : ℕ := flips_in_year * (pedometer_max + 1) + last_day_steps

def miles_walked : ℚ := total_steps / steps_per_mile

theorem postman_pete_mileage :
  ∃ (m : ℕ), m ≥ 3000 ∧ m ≤ 4000 ∧ 
  ∀ (n : ℕ), (n ≥ 3000 ∧ n ≤ 4000) → |miles_walked - m| ≤ |miles_walked - n| :=
sorry

end postman_pete_mileage_l1529_152950


namespace circumradius_special_triangle_l1529_152961

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex A
def angle_at_A (t : Triangle) : ℝ := sorry

-- Define the radius of the circumscribed circle
def circumradius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem circumradius_special_triangle (t : Triangle) :
  angle_at_A t = π / 3 ∧
  distance t.B (incenter t) = 3 ∧
  distance t.C (incenter t) = 4 →
  circumradius t = Real.sqrt (37 / 3) := by
  sorry

end circumradius_special_triangle_l1529_152961


namespace expression_evaluation_l1529_152923

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  ((x - 2*y)*x - (x - 2*y)*(x + 2*y)) / y = -8 := by
sorry

end expression_evaluation_l1529_152923


namespace bird_families_remaining_l1529_152970

theorem bird_families_remaining (initial : ℕ) (flew_away : ℕ) (remaining : ℕ) : 
  initial = 41 → flew_away = 27 → remaining = initial - flew_away → remaining = 14 := by
  sorry

end bird_families_remaining_l1529_152970


namespace even_periodic_function_property_l1529_152990

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem even_periodic_function_property
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_period : has_period f 2)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end even_periodic_function_property_l1529_152990
