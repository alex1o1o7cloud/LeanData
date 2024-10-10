import Mathlib

namespace first_day_exceeding_threshold_l2395_239582

def initial_bacteria : ℕ := 5
def growth_rate : ℕ := 3
def threshold : ℕ := 200

def bacteria_count (n : ℕ) : ℕ := initial_bacteria * growth_rate ^ n

theorem first_day_exceeding_threshold :
  ∃ n : ℕ, bacteria_count n > threshold ∧ ∀ m : ℕ, m < n → bacteria_count m ≤ threshold :=
by sorry

end first_day_exceeding_threshold_l2395_239582


namespace cricket_team_age_difference_l2395_239548

theorem cricket_team_age_difference :
  let team_size : ℕ := 11
  let captain_age : ℕ := 25
  let team_average_age : ℕ := 22
  let remaining_average_age : ℕ := team_average_age - 1
  let wicket_keeper_age := captain_age + x

  (team_size * team_average_age = 
   (team_size - 2) * remaining_average_age + captain_age + wicket_keeper_age) →
  x = 3 := by
sorry

end cricket_team_age_difference_l2395_239548


namespace stratified_sample_size_l2395_239570

/-- Represents the number of items of each product type in a sample -/
structure SampleCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculates the total sample size -/
def totalSampleSize (s : SampleCounts) : ℕ :=
  s.typeA + s.typeB + s.typeC

/-- Represents the production ratio of the three product types -/
def productionRatio : Fin 3 → ℕ
| 0 => 1  -- Type A
| 1 => 3  -- Type B
| 2 => 5  -- Type C

theorem stratified_sample_size 
  (s : SampleCounts) 
  (h_ratio : s.typeA * productionRatio 1 = s.typeB * productionRatio 0 ∧ 
             s.typeB * productionRatio 2 = s.typeC * productionRatio 1) 
  (h_typeB : s.typeB = 12) : 
  totalSampleSize s = 36 := by
sorry


end stratified_sample_size_l2395_239570


namespace boat_speed_in_still_water_l2395_239590

/-- Given a boat that travels 11 km/hr along a stream and 7 km/hr against the same stream,
    its speed in still water is 9 km/hr. -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = 11 →
    boat_speed - stream_speed = 7 →
    boat_speed = 9 := by
  sorry

end boat_speed_in_still_water_l2395_239590


namespace arithmetic_sequence_fifth_term_l2395_239572

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  sum_property : a 1 + a 3 = 8
  geometric_mean : (a 4) ^ 2 = (a 2) * (a 9)
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (seq : ArithmeticSequence) : seq.a 5 = 13 := by sorry

end arithmetic_sequence_fifth_term_l2395_239572


namespace stratified_sampling_senior_managers_l2395_239575

theorem stratified_sampling_senior_managers 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (senior_managers : ℕ) 
  (h1 : total_population = 200) 
  (h2 : sample_size = 40) 
  (h3 : senior_managers = 10) :
  (sample_size : ℚ) / total_population * senior_managers = 2 := by
  sorry

end stratified_sampling_senior_managers_l2395_239575


namespace intersection_equals_A_l2395_239561

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | |x| < 2}

-- Theorem statement
theorem intersection_equals_A : A ∩ B = A := by sorry

end intersection_equals_A_l2395_239561


namespace curve_intersection_l2395_239503

theorem curve_intersection :
  ∃ (θ t : ℝ),
    0 ≤ θ ∧ θ ≤ π ∧
    Real.sqrt 5 * Real.cos θ = 5/6 ∧
    Real.sin θ = 2/3 ∧
    (5/4) * t = 5/6 ∧
    t = 2/3 := by
  sorry

end curve_intersection_l2395_239503


namespace smallest_positive_resolvable_debt_is_40_l2395_239552

/-- The value of a pig in dollars -/
def pig_value : ℕ := 280

/-- The value of a goat in dollars -/
def goat_value : ℕ := 200

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℤ) : Prop :=
  ∃ (p g : ℤ), debt = p * pig_value + g * goat_value

/-- The smallest positive resolvable debt -/
def smallest_positive_resolvable_debt : ℕ := 40

theorem smallest_positive_resolvable_debt_is_40 :
  (∀ d : ℕ, d < smallest_positive_resolvable_debt → ¬is_resolvable d) ∧
  is_resolvable smallest_positive_resolvable_debt :=
sorry

end smallest_positive_resolvable_debt_is_40_l2395_239552


namespace frank_allowance_proof_l2395_239525

def frank_allowance (initial_amount spent_amount final_amount : ℕ) : ℕ :=
  final_amount - (initial_amount - spent_amount)

theorem frank_allowance_proof :
  frank_allowance 11 3 22 = 14 := by
  sorry

end frank_allowance_proof_l2395_239525


namespace average_monthly_salary_l2395_239538

/-- Calculates the average monthly salary of five employees given their base salaries and bonus/deduction percentages. -/
theorem average_monthly_salary
  (base_A base_B base_C base_D base_E : ℕ)
  (bonus_A bonus_B1 bonus_D bonus_E : ℚ)
  (deduct_B deduct_D deduct_E : ℚ)
  (h_base_A : base_A = 8000)
  (h_base_B : base_B = 5000)
  (h_base_C : base_C = 16000)
  (h_base_D : base_D = 7000)
  (h_base_E : base_E = 9000)
  (h_bonus_A : bonus_A = 5 / 100)
  (h_bonus_B1 : bonus_B1 = 10 / 100)
  (h_deduct_B : deduct_B = 2 / 100)
  (h_bonus_D : bonus_D = 8 / 100)
  (h_deduct_D : deduct_D = 3 / 100)
  (h_bonus_E : bonus_E = 12 / 100)
  (h_deduct_E : deduct_E = 5 / 100) :
  (base_A * (1 + bonus_A) +
   base_B * (1 + bonus_B1 - deduct_B) +
   base_C +
   base_D * (1 + bonus_D - deduct_D) +
   base_E * (1 + bonus_E - deduct_E)) / 5 = 8756 :=
by sorry

end average_monthly_salary_l2395_239538


namespace sufficient_not_necessary_l2395_239527

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x + y ≥ 4) ∧
  (∃ x y : ℝ, x + y ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) := by
  sorry

end sufficient_not_necessary_l2395_239527


namespace quadratic_roots_condition_l2395_239541

theorem quadratic_roots_condition (p q r : ℝ) : 
  (p^4 * (q - r)^2 + 2 * p^2 * (q + r) + 1 = p^4) ↔ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    (x₁^2 + p*x₁ + q = 0) ∧ 
    (x₂^2 + p*x₂ + q = 0) ∧ 
    (y₁^2 - p*y₁ + r = 0) ∧ 
    (y₂^2 - p*y₂ + r = 0) ∧ 
    (x₁*y₁ - x₂*y₂ = 1)) :=
by sorry

end quadratic_roots_condition_l2395_239541


namespace parallelogram_side_length_l2395_239524

/-- A parallelogram with vertices A, B, C, and D in a real inner product space. -/
structure Parallelogram (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)
  (is_parallelogram : A - B = D - C)

/-- The theorem stating that if BD = 2 and 2(AD • AB) = |BC|^2 in a parallelogram ABCD,
    then |AB| = 2. -/
theorem parallelogram_side_length
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (para : Parallelogram V)
  (h1 : ‖para.B - para.D‖ = 2)
  (h2 : 2 * inner (para.A - para.D) (para.A - para.B) = ‖para.B - para.C‖^2) :
  ‖para.A - para.B‖ = 2 :=
sorry

end parallelogram_side_length_l2395_239524


namespace postcard_probability_l2395_239521

/-- The probability of arranging n unique items in a line, such that k specific items are consecutive. -/
def consecutive_probability (n k : ℕ) : ℚ :=
  if k ≤ n ∧ k > 0 then
    (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n
  else
    0

/-- Theorem: The probability of arranging 12 unique postcards in a line, 
    such that 4 specific postcards are consecutive, is 1/55. -/
theorem postcard_probability : consecutive_probability 12 4 = 1 / 55 := by
  sorry

end postcard_probability_l2395_239521


namespace bus_capacity_is_90_l2395_239580

/-- The number of people that can sit in a bus with given seat arrangements -/
def bus_capacity (left_seats : ℕ) (right_seat_difference : ℕ) (people_per_seat : ℕ) (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seat_difference
  let total_regular_seats := left_seats + right_seats
  let total_regular_capacity := total_regular_seats * people_per_seat
  total_regular_capacity + back_seat_capacity

/-- Theorem stating that the bus capacity is 90 given the specific conditions -/
theorem bus_capacity_is_90 : 
  bus_capacity 15 3 3 9 = 90 := by
  sorry

end bus_capacity_is_90_l2395_239580


namespace intersection_of_A_and_B_l2395_239510

-- Define sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x ≤ 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 5} := by sorry

end intersection_of_A_and_B_l2395_239510


namespace log_ratio_squared_equals_one_l2395_239500

theorem log_ratio_squared_equals_one (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1)
  (h_log : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
  (h_sum : x + y = 36) : 
  (Real.log (x / y) / Real.log 3)^2 = 1 := by
  sorry

end log_ratio_squared_equals_one_l2395_239500


namespace bahs_equal_to_500_yahs_l2395_239579

-- Define the conversion rates
def bah_to_rah : ℚ := 30 / 20
def rah_to_yah : ℚ := 25 / 10

-- Define the target number of yahs
def target_yahs : ℕ := 500

-- Theorem statement
theorem bahs_equal_to_500_yahs :
  ⌊(target_yahs : ℚ) / (rah_to_yah * bah_to_rah)⌋ = 133 := by
  sorry

end bahs_equal_to_500_yahs_l2395_239579


namespace conic_is_hyperbola_l2395_239588

/-- A conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines the type of conic section from its equation -/
def determineConicType (f : ℝ → ℝ → ℝ) : ConicType :=
  sorry

/-- The equation of the conic section -/
def conicEquation (x y : ℝ) : ℝ :=
  (x - 3)^2 - 2*(y + 1)^2 - 50

theorem conic_is_hyperbola :
  determineConicType conicEquation = ConicType.Hyperbola :=
sorry

end conic_is_hyperbola_l2395_239588


namespace norris_savings_l2395_239564

/-- The amount of money Norris saved in November -/
def november_savings : ℤ := sorry

/-- The amount of money Norris saved in September -/
def september_savings : ℤ := 29

/-- The amount of money Norris saved in October -/
def october_savings : ℤ := 25

/-- The amount of money Norris spent on an online game -/
def online_game_cost : ℤ := 75

/-- The amount of money Norris has left -/
def money_left : ℤ := 10

theorem norris_savings : november_savings = 31 := by
  sorry

end norris_savings_l2395_239564


namespace carlson_problem_max_candies_l2395_239593

/-- The maximum number of candies that can be eaten in the Carlson problem -/
def max_candies (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating the maximum number of candies for 32 initial ones -/
theorem carlson_problem_max_candies :
  max_candies 32 = 496 := by
  sorry

#eval max_candies 32  -- Should output 496

end carlson_problem_max_candies_l2395_239593


namespace sqrt_sum_equals_sqrt_ten_l2395_239567

-- Define the variables
variable (a b c : ℝ)

-- State the theorem
theorem sqrt_sum_equals_sqrt_ten
  (h1 : (2*a + 2)^(1/3) = 2)
  (h2 : b^(1/2) = 2)
  (h3 : ⌊Real.sqrt 15⌋ = c)
  : Real.sqrt (a + b + c) = Real.sqrt 10 :=
by sorry

end sqrt_sum_equals_sqrt_ten_l2395_239567


namespace ac_average_usage_time_l2395_239563

/-- Calculates the average usage time for each air conditioner -/
def averageUsageTime (totalAC : ℕ) (maxSimultaneous : ℕ) (hoursPerDay : ℕ) : ℚ :=
  (maxSimultaneous * hoursPerDay : ℚ) / totalAC

/-- Proves that the average usage time for each air conditioner is 20 hours -/
theorem ac_average_usage_time :
  let totalAC : ℕ := 6
  let maxSimultaneous : ℕ := 5
  let hoursPerDay : ℕ := 24
  averageUsageTime totalAC maxSimultaneous hoursPerDay = 20 := by
sorry

end ac_average_usage_time_l2395_239563


namespace correct_payments_l2395_239531

/-- Represents the payment information for the gardeners' plot plowing problem. -/
structure PlowingPayment where
  totalPayment : ℕ
  rectangularPlotArea : ℕ
  rectangularPlotSide : ℕ
  squarePlot1Side : ℕ
  squarePlot2Side : ℕ

/-- Calculates the payments for each gardener based on the given information. -/
def calculatePayments (info : PlowingPayment) : (ℕ × ℕ × ℕ) :=
  let rectangularPlotWidth := info.rectangularPlotArea / info.rectangularPlotSide
  let squarePlot1Area := info.squarePlot1Side * info.squarePlot1Side
  let squarePlot2Area := info.squarePlot2Side * info.squarePlot2Side
  let totalArea := info.rectangularPlotArea + squarePlot1Area + squarePlot2Area
  let pricePerArea := info.totalPayment / totalArea
  let payment1 := info.rectangularPlotArea * pricePerArea
  let payment2 := squarePlot1Area * pricePerArea
  let payment3 := squarePlot2Area * pricePerArea
  (payment1, payment2, payment3)

/-- Theorem stating that the calculated payments match the expected values. -/
theorem correct_payments (info : PlowingPayment) 
  (h1 : info.totalPayment = 570)
  (h2 : info.rectangularPlotArea = 600)
  (h3 : info.rectangularPlotSide = 20)
  (h4 : info.squarePlot1Side = info.rectangularPlotSide)
  (h5 : info.squarePlot2Side = info.rectangularPlotArea / info.rectangularPlotSide) :
  calculatePayments info = (180, 120, 270) := by
  sorry

end correct_payments_l2395_239531


namespace mary_tim_income_difference_l2395_239532

theorem mary_tim_income_difference (juan tim mary : ℝ) 
  (h1 : tim = 0.5 * juan)
  (h2 : mary = 0.8 * juan) :
  (mary - tim) / tim * 100 = 60 := by
sorry

end mary_tim_income_difference_l2395_239532


namespace f_properties_l2395_239505

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def monotonic_intervals (a : ℝ) : Prop :=
  (a ≤ 0 → ∀ x y, 0 < x ∧ x < y → f a x < f a y) ∧
  (a > 0 → (∀ x y, 0 < x ∧ x < y ∧ y < 1/a → f a x < f a y) ∧
           (∀ x y, 1/a < x ∧ x < y → f a y < f a x))

def minimum_value (a : ℝ) : ℝ :=
  if a ≥ 1 then f a 2
  else if 0 < a ∧ a < 1/2 then f a 1
  else min (f a 1) (f a 2)

theorem f_properties (a : ℝ) :
  monotonic_intervals a ∧
  (a > 0 → ∀ x, x ∈ Set.Icc 1 2 → f a x ≥ minimum_value a) :=
sorry

end

end f_properties_l2395_239505


namespace smallest_n_value_l2395_239504

/-- The number of ordered triplets (a, b, c) satisfying the conditions -/
def num_triplets : ℕ := 27000

/-- The greatest common divisor of a, b, and c -/
def gcd_value : ℕ := 91

/-- A function that counts the number of valid triplets for a given n -/
noncomputable def count_triplets (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating the smallest possible value of n -/
theorem smallest_n_value :
  ∃ (n : ℕ), n = 17836000 ∧
  count_triplets n = num_triplets ∧
  (∀ m : ℕ, m < n → count_triplets m ≠ num_triplets) :=
sorry

end smallest_n_value_l2395_239504


namespace letter_R_in_13th_space_l2395_239545

/-- The space number where the letter R should be placed on a sign -/
def letter_R_position (total_spaces : ℕ) (word_length : ℕ) : ℕ :=
  (total_spaces - word_length) / 2 + 1

/-- Theorem stating that the letter R should be in the 13th space -/
theorem letter_R_in_13th_space :
  letter_R_position 31 7 = 13 := by
  sorry

end letter_R_in_13th_space_l2395_239545


namespace photos_per_album_l2395_239528

theorem photos_per_album 
  (total_photos : ℕ) 
  (num_albums : ℕ) 
  (h1 : total_photos = 2560) 
  (h2 : num_albums = 32) 
  (h3 : total_photos % num_albums = 0) :
  total_photos / num_albums = 80 := by
sorry

end photos_per_album_l2395_239528


namespace geometric_sequence_first_term_l2395_239543

theorem geometric_sequence_first_term (a r : ℝ) : 
  a * r = 5 → a * r^3 = 45 → a = 5/3 := by
  sorry

end geometric_sequence_first_term_l2395_239543


namespace double_scientific_notation_doubling_2_4_times_10_to_8_l2395_239517

theorem double_scientific_notation (x : Real) (n : Nat) :
  2 * (x * (10 ^ n)) = (2 * x) * (10 ^ n) := by sorry

theorem doubling_2_4_times_10_to_8 :
  2 * (2.4 * (10 ^ 8)) = 4.8 * (10 ^ 8) := by sorry

end double_scientific_notation_doubling_2_4_times_10_to_8_l2395_239517


namespace pizza_fraction_proof_l2395_239584

theorem pizza_fraction_proof (total_slices : ℕ) (whole_slices_eaten : ℕ) (shared_slice_fraction : ℚ) :
  total_slices = 16 →
  whole_slices_eaten = 2 →
  shared_slice_fraction = 1/3 →
  (whole_slices_eaten : ℚ) / total_slices + shared_slice_fraction / total_slices = 7/48 := by
  sorry

end pizza_fraction_proof_l2395_239584


namespace base_of_exponent_l2395_239573

theorem base_of_exponent (a : ℝ) (x : ℝ) (h1 : a^(2*x + 2) = 16^(3*x - 1)) (h2 : x = 1) : a = 4 := by
  sorry

end base_of_exponent_l2395_239573


namespace price_of_short_is_13_50_l2395_239595

/-- The price of a single short, given the conditions of Jimmy and Irene's shopping trip -/
def price_of_short (num_shorts : ℕ) (num_shirts : ℕ) (shirt_price : ℚ) 
  (discount_rate : ℚ) (total_paid : ℚ) : ℚ :=
  let shirt_total := num_shirts * shirt_price
  let discounted_shirt_total := shirt_total * (1 - discount_rate)
  let shorts_total := total_paid - discounted_shirt_total
  shorts_total / num_shorts

/-- Theorem stating that the price of each short is $13.50 under the given conditions -/
theorem price_of_short_is_13_50 :
  price_of_short 3 5 17 (1/10) 117 = 27/2 := by sorry

end price_of_short_is_13_50_l2395_239595


namespace book_cost_problem_l2395_239578

theorem book_cost_problem (cost_loss : ℝ) (sell_price : ℝ) :
  cost_loss = 262.5 →
  sell_price = cost_loss * 0.85 →
  sell_price = (sell_price / 1.19) * 1.19 →
  cost_loss + (sell_price / 1.19) = 450 :=
by
  sorry

end book_cost_problem_l2395_239578


namespace existence_of_special_function_l2395_239539

theorem existence_of_special_function :
  ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f (f n) = 1993 * n^1945 :=
sorry

end existence_of_special_function_l2395_239539


namespace max_distinct_squares_sum_l2395_239587

/-- The sum of squares of the first n positive integers -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- A function that checks if there exists a set of n distinct positive integers
    whose squares sum to 2531 -/
def exists_distinct_squares_sum (n : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card = n ∧ (∀ x ∈ s, x > 0) ∧ (s.sum (λ x => x^2) = 2531)

theorem max_distinct_squares_sum :
  (∃ n : ℕ, exists_distinct_squares_sum n ∧
    ∀ m : ℕ, m > n → ¬exists_distinct_squares_sum m) ∧
  (∃ n : ℕ, exists_distinct_squares_sum n ∧ n = 18) := by
  sorry

end max_distinct_squares_sum_l2395_239587


namespace quadratic_equation_root_l2395_239571

theorem quadratic_equation_root (k l m : ℝ) :
  (2 * (k - l) * 2^2 + 3 * (l - m) * 2 + 4 * (m - k) = 0) →
  (∃ x : ℝ, 2 * (k - l) * x^2 + 3 * (l - m) * x + 4 * (m - k) = 0 ∧ x ≠ 2) →
  (∃ x : ℝ, 2 * (k - l) * x^2 + 3 * (l - m) * x + 4 * (m - k) = 0 ∧ x = (m - k) / (k - l)) :=
by sorry

end quadratic_equation_root_l2395_239571


namespace class_size_proof_l2395_239594

theorem class_size_proof (total : ℕ) 
  (h1 : 20 < total ∧ total < 30)
  (h2 : ∃ male : ℕ, total = 3 * male)
  (h3 : ∃ registered unregistered : ℕ, 
    registered + unregistered = total ∧ 
    registered = 3 * unregistered - 1) :
  total = 27 := by
  sorry

end class_size_proof_l2395_239594


namespace estimated_red_balls_l2395_239526

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 10

/-- Represents the number of draws -/
def total_draws : ℕ := 100

/-- Represents the number of white balls drawn -/
def white_draws : ℕ := 40

/-- Theorem: Given the conditions, the estimated number of red balls is 6 -/
theorem estimated_red_balls :
  total_balls * (total_draws - white_draws) / total_draws = 6 := by
  sorry

end estimated_red_balls_l2395_239526


namespace true_proposition_l2395_239581

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Proposition p: There exists a φ ∈ ℝ such that f(x) = sin(x + φ) is an even function -/
def p : Prop := ∃ φ : ℝ, IsEven (fun x ↦ Real.sin (x + φ))

/-- Proposition q: For all x ∈ ℝ, cos(2x) + 4sin(x) - 3 < 0 -/
def q : Prop := ∀ x : ℝ, Real.cos (2 * x) + 4 * Real.sin x - 3 < 0

/-- The true proposition is p ∨ (¬q) -/
theorem true_proposition : p ∨ ¬q := by
  sorry

end true_proposition_l2395_239581


namespace words_with_consonant_count_l2395_239540

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := alphabet \ consonants

def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def vowel_only_words : Nat := vowels.card ^ word_length

theorem words_with_consonant_count :
  total_words - vowel_only_words = 7744 :=
sorry

end words_with_consonant_count_l2395_239540


namespace simplify_expression_l2395_239515

theorem simplify_expression (a b : ℝ) : 3*a + 2*b - 2*(a - b) = a + 4*b := by
  sorry

end simplify_expression_l2395_239515


namespace fraction_simplification_l2395_239598

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : (a - b) / (b - a) = -1 := by
  sorry

end fraction_simplification_l2395_239598


namespace decimal_to_binary_119_l2395_239597

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 119

/-- The expected binary representation -/
def expectedBinary : List Bool := [true, true, true, false, true, true, true]

/-- Theorem stating that the binary representation of 119 is [1,1,1,0,1,1,1] -/
theorem decimal_to_binary_119 : toBinary decimalNumber = expectedBinary := by
  sorry

end decimal_to_binary_119_l2395_239597


namespace saree_stripe_theorem_l2395_239566

/-- Represents the stripes on a Saree --/
structure SareeStripes where
  brown : ℕ
  gold : ℕ
  blue : ℕ

/-- Represents the properties of the Saree's stripe pattern --/
def SareeProperties (s : SareeStripes) : Prop :=
  s.gold = 3 * s.brown ∧
  s.blue = 5 * s.gold ∧
  s.brown = 4 ∧
  s.brown + s.gold + s.blue = 100

/-- Calculates the number of complete patterns on the Saree --/
def patternCount (s : SareeStripes) : ℕ :=
  (s.brown + s.gold + s.blue) / 3

theorem saree_stripe_theorem (s : SareeStripes) 
  (h : SareeProperties s) : s.blue = 84 ∧ patternCount s = 33 := by
  sorry

#check saree_stripe_theorem

end saree_stripe_theorem_l2395_239566


namespace milk_quantity_proof_l2395_239547

/-- The initial quantity of milk in container A --/
def initial_quantity : ℝ := 1216

/-- The fraction of milk in container B compared to A --/
def b_fraction : ℝ := 0.375

/-- The amount transferred between containers --/
def transfer_amount : ℝ := 152

/-- Theorem stating the initial quantity of milk in container A --/
theorem milk_quantity_proof :
  ∃ (a b c : ℝ),
    a = initial_quantity ∧
    b = b_fraction * a ∧
    c = a - b ∧
    b + transfer_amount = c - transfer_amount :=
by sorry

end milk_quantity_proof_l2395_239547


namespace coin_order_correct_l2395_239513

/-- Represents the set of coins --/
inductive Coin : Type
  | A | B | C | D | E | F

/-- Defines the covering relation between coins --/
def covers (x y : Coin) : Prop := sorry

/-- The correct order of coins from top to bottom --/
def correct_order : List Coin := [Coin.F, Coin.D, Coin.A, Coin.E, Coin.B, Coin.C]

/-- Theorem stating that the given order is correct based on the covering relations --/
theorem coin_order_correct :
  (∀ x : Coin, ¬covers x Coin.F) ∧
  (covers Coin.F Coin.D) ∧
  (covers Coin.D Coin.B) ∧ (covers Coin.D Coin.C) ∧ (covers Coin.D Coin.E) ∧
  (covers Coin.D Coin.A) ∧ (covers Coin.A Coin.B) ∧ (covers Coin.A Coin.C) ∧
  (covers Coin.D Coin.E) ∧ (covers Coin.E Coin.C) ∧
  (covers Coin.D Coin.B) ∧ (covers Coin.A Coin.B) ∧ (covers Coin.E Coin.B) ∧ (covers Coin.B Coin.C) ∧
  (∀ x : Coin, x ≠ Coin.C → covers x Coin.C) →
  correct_order = [Coin.F, Coin.D, Coin.A, Coin.E, Coin.B, Coin.C] :=
by sorry

end coin_order_correct_l2395_239513


namespace jones_elementary_population_l2395_239530

theorem jones_elementary_population :
  let total_students : ℕ := 225
  let boys_percentage : ℚ := 40 / 100
  let boys_count : ℕ := 90
  (boys_count : ℚ) / (total_students * boys_percentage) = 1 :=
by sorry

end jones_elementary_population_l2395_239530


namespace point_in_fourth_quadrant_l2395_239535

/-- Definition of a point in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

/-- The point (3, -2) -/
def point : ℝ × ℝ := (3, -2)

/-- Theorem: The point (3, -2) is in the fourth quadrant -/
theorem point_in_fourth_quadrant : 
  in_fourth_quadrant point.1 point.2 := by sorry

end point_in_fourth_quadrant_l2395_239535


namespace non_adjacent_permutations_five_l2395_239583

def number_of_permutations (n : ℕ) : ℕ := Nat.factorial n

def adjacent_permutations (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

theorem non_adjacent_permutations_five :
  number_of_permutations 5 - adjacent_permutations 5 = 72 := by sorry

end non_adjacent_permutations_five_l2395_239583


namespace marble_distribution_l2395_239562

theorem marble_distribution (T : ℝ) (C B O : ℝ) : 
  T > 0 →
  C = 0.40 * T →
  O = (2/5) * T →
  C + B + O = T →
  B = 0.20 * T :=
by
  sorry

end marble_distribution_l2395_239562


namespace unique_solution_condition_l2395_239507

theorem unique_solution_condition (c k : ℝ) (h_c : c ≠ 0) : 
  (∃! b : ℝ, b > 0 ∧ 
    (∃! x : ℝ, x^2 + (b + 1/b) * x + c = 0) ∧ 
    b^4 + (2 - 4*c) * b^2 + k = 0) ↔ 
  c = 1 := by sorry

end unique_solution_condition_l2395_239507


namespace polynomial_simplification_l2395_239546

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + x^2 + 15) - (x^6 + x^5 - 2 * x^4 + 3 * x^2 + 20) =
  x^6 + 2 * x^5 + 3 * x^4 - 2 * x^2 - 5 := by
  sorry

end polynomial_simplification_l2395_239546


namespace one_third_to_fifth_power_l2395_239534

theorem one_third_to_fifth_power :
  (1 / 3 : ℚ) ^ 5 = 1 / 243 := by sorry

end one_third_to_fifth_power_l2395_239534


namespace solve_work_problem_l2395_239569

def work_problem (a_days b_days : ℕ) (b_share : ℚ) : Prop :=
  a_days > 0 ∧ b_days > 0 ∧ b_share > 0 →
  let a_rate : ℚ := 1 / a_days
  let b_rate : ℚ := 1 / b_days
  let total_rate : ℚ := a_rate + b_rate
  let a_proportion : ℚ := a_rate / total_rate
  let b_proportion : ℚ := b_rate / total_rate
  let total_amount : ℚ := b_share / b_proportion
  total_amount = 1000

theorem solve_work_problem :
  work_problem 30 20 600 := by
  sorry

end solve_work_problem_l2395_239569


namespace first_subject_grade_l2395_239589

/-- 
Given a student's grades in three subjects, prove that if the second subject is 60%,
the third subject is 70%, and the overall average is 60%, then the first subject's grade must be 50%.
-/
theorem first_subject_grade (grade1 : ℝ) (grade2 grade3 overall : ℝ) : 
  grade2 = 60 → grade3 = 70 → overall = 60 → (grade1 + grade2 + grade3) / 3 = overall → grade1 = 50 := by
  sorry

end first_subject_grade_l2395_239589


namespace sean_has_more_whistles_l2395_239554

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := 13

/-- The difference in whistle count between Sean and Charles -/
def whistle_difference : ℕ := sean_whistles - charles_whistles

theorem sean_has_more_whistles : whistle_difference = 32 := by
  sorry

end sean_has_more_whistles_l2395_239554


namespace fraction_sum_equals_cube_sum_l2395_239511

theorem fraction_sum_equals_cube_sum (x : ℝ) : 
  ((x - 1) * (x + 1)) / (x * (x - 1) + 1) + (2 * (0.5 - x)) / (x * (1 - x) - 1) = 
  ((x - 1) * (x + 1) / (x * (x - 1) + 1))^3 + (2 * (0.5 - x) / (x * (1 - x) - 1))^3 :=
by sorry

end fraction_sum_equals_cube_sum_l2395_239511


namespace students_without_A_l2395_239568

theorem students_without_A (total : ℕ) (lit_A : ℕ) (sci_A : ℕ) (both_A : ℕ) : 
  total = 35 → lit_A = 10 → sci_A = 15 → both_A = 5 → 
  total - (lit_A + sci_A - both_A) = 15 := by
  sorry

end students_without_A_l2395_239568


namespace kelly_apples_l2395_239599

/-- The number of apples Kelly has altogether, given her initial apples and the apples she picked. -/
def total_apples (initial : Float) (picked : Float) : Float :=
  initial + picked

/-- Theorem stating that Kelly has 161.0 apples altogether. -/
theorem kelly_apples : total_apples 56.0 105.0 = 161.0 := by
  sorry

end kelly_apples_l2395_239599


namespace fifth_term_of_sequence_l2395_239519

/-- Given a sequence {aₙ} where Sₙ (the sum of the first n terms) is defined as Sₙ = n² + 1,
    prove that the 5th term of the sequence (a₅) is equal to 9. -/
theorem fifth_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = n^2 + 1) : a 5 = 9 := by
  sorry

end fifth_term_of_sequence_l2395_239519


namespace average_of_series_l2395_239501

/-- The average value of the series 0², (2z)², (4z)², (8z)² is 21z² -/
theorem average_of_series (z : ℝ) : 
  (0^2 + (2*z)^2 + (4*z)^2 + (8*z)^2) / 4 = 21 * z^2 := by
  sorry

end average_of_series_l2395_239501


namespace cash_me_problem_l2395_239565

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def to_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem cash_me_problem :
  ¬∃ (C A S H M E O I D : ℕ),
    is_digit C ∧ is_digit A ∧ is_digit S ∧ is_digit H ∧
    is_digit M ∧ is_digit E ∧ is_digit O ∧ is_digit I ∧ is_digit D ∧
    C ≠ 0 ∧ M ≠ 0 ∧ O ≠ 0 ∧
    to_number C A S H + to_number M E 0 0 = to_number O S I D ∧
    to_number O S I D ≥ 1000 ∧ to_number O S I D < 10000 :=
by sorry

end cash_me_problem_l2395_239565


namespace decimal_addition_l2395_239574

theorem decimal_addition : (5.467 : ℝ) + 3.92 = 9.387 := by
  sorry

end decimal_addition_l2395_239574


namespace haley_magazines_l2395_239508

theorem haley_magazines 
  (num_boxes : ℕ) 
  (magazines_per_box : ℕ) 
  (h1 : num_boxes = 7)
  (h2 : magazines_per_box = 9) :
  num_boxes * magazines_per_box = 63 := by
  sorry

end haley_magazines_l2395_239508


namespace exists_number_satisfying_equation_l2395_239537

theorem exists_number_satisfying_equation : ∃ x : ℝ, (3.241 * x) / 100 = 0.045374000000000005 := by
  sorry

end exists_number_satisfying_equation_l2395_239537


namespace smallest_multiple_five_satisfies_five_is_smallest_l2395_239556

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 625 ∣ (x * 500) → x ≥ 5 := by
  sorry

theorem five_satisfies : 625 ∣ (5 * 500) := by
  sorry

theorem five_is_smallest : ∀ (x : ℕ), x > 0 ∧ 625 ∣ (x * 500) → x ≥ 5 := by
  sorry

end smallest_multiple_five_satisfies_five_is_smallest_l2395_239556


namespace samia_walk_distance_l2395_239544

def average_bike_speed : ℝ := 20
def bike_distance : ℝ := 2
def walk_speed : ℝ := 4
def total_time_minutes : ℝ := 78

theorem samia_walk_distance :
  let total_time_hours : ℝ := total_time_minutes / 60
  let bike_time : ℝ := bike_distance / average_bike_speed
  let walk_time : ℝ := total_time_hours - bike_time
  let walk_distance : ℝ := walk_time * walk_speed
  walk_distance = 4.8 := by sorry

end samia_walk_distance_l2395_239544


namespace particles_tend_to_unit_circle_l2395_239577

/-- Velocity field of the fluid -/
def velocity_field (x y : ℝ) : ℝ × ℝ :=
  (y + 2*x - 2*x^3 - 2*x*y^2, -x)

/-- The rate of change of r^2 with respect to t -/
def r_squared_derivative (x y : ℝ) : ℝ :=
  2*x*(y + 2*x - 2*x^3 - 2*x*y^2) + 2*y*(-x)

/-- Theorem stating that particles tend towards the unit circle as t → ∞ -/
theorem particles_tend_to_unit_circle :
  ∀ (x y : ℝ), x ≠ 0 →
  (r_squared_derivative x y > 0 ↔ x^2 + y^2 < 1) ∧
  (r_squared_derivative x y < 0 ↔ x^2 + y^2 > 1) :=
sorry

end particles_tend_to_unit_circle_l2395_239577


namespace friends_bread_slices_l2395_239596

/-- Calculates the number of slices each friend eats given the number of friends and the slices in each loaf -/
def slices_per_friend (n : ℕ) (loaf1 loaf2 loaf3 loaf4 : ℕ) : ℕ :=
  (loaf1 + loaf2 + loaf3 + loaf4)

/-- Theorem stating that each friend eats 78 slices of bread -/
theorem friends_bread_slices (n : ℕ) (h : n > 0) :
  slices_per_friend n 15 18 20 25 = 78 := by
  sorry

#check friends_bread_slices

end friends_bread_slices_l2395_239596


namespace flagstaff_shadow_length_l2395_239549

/-- Given a flagstaff and a building casting shadows under similar conditions,
    this theorem proves the length of the flagstaff's shadow. -/
theorem flagstaff_shadow_length
  (h_flagstaff : ℝ)
  (h_building : ℝ)
  (s_building : ℝ)
  (h_flagstaff_pos : h_flagstaff > 0)
  (h_building_pos : h_building > 0)
  (s_building_pos : s_building > 0)
  (h_flagstaff_val : h_flagstaff = 17.5)
  (h_building_val : h_building = 12.5)
  (s_building_val : s_building = 28.75) :
  ∃ s_flagstaff : ℝ, s_flagstaff = 40.15 ∧ h_flagstaff / s_flagstaff = h_building / s_building :=
by
  sorry


end flagstaff_shadow_length_l2395_239549


namespace eighteen_cubed_times_nine_cubed_l2395_239585

theorem eighteen_cubed_times_nine_cubed (L M : ℕ) : 18^3 * 9^3 = 2^3 * 3^12 := by
  sorry

end eighteen_cubed_times_nine_cubed_l2395_239585


namespace dvd_rental_cost_l2395_239520

theorem dvd_rental_cost (num_dvds : ℕ) (cost_per_dvd : ℚ) (total_cost : ℚ) :
  num_dvds = 4 →
  cost_per_dvd = 6/5 →
  total_cost = num_dvds * cost_per_dvd →
  total_cost = 24/5 := by
  sorry

end dvd_rental_cost_l2395_239520


namespace unique_square_difference_l2395_239560

theorem unique_square_difference (n : ℕ) : 
  (∃ k m : ℕ, n + 30 = k^2 ∧ n - 17 = m^2) ↔ n = 546 := by
sorry

end unique_square_difference_l2395_239560


namespace parabola_focus_l2395_239529

/-- A parabola is defined by the equation x = -1/16 * y^2 + 2 -/
def parabola (x y : ℝ) : Prop := x = -1/16 * y^2 + 2

/-- The focus of a parabola is a point -/
def focus : ℝ × ℝ := (-2, 0)

/-- Theorem: The focus of the parabola defined by x = -1/16 * y^2 + 2 is at (-2, 0) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola x y → focus = (-2, 0) := by
  sorry

end parabola_focus_l2395_239529


namespace positive_polynomial_fraction_representation_l2395_239542

/-- A polynomial with real coefficients. -/
def RealPolynomial := Polynomial ℝ

/-- Proposition that a polynomial is positive for all positive real numbers. -/
def IsPositiveForPositive (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, x > 0 → p.eval x > 0

/-- Proposition that a polynomial has nonnegative coefficients. -/
def HasNonnegativeCoeffs (p : RealPolynomial) : Prop :=
  ∀ n : ℕ, p.coeff n ≥ 0

/-- Main theorem: For any polynomial P that is positive for all positive real numbers,
    there exist polynomials Q and R with nonnegative coefficients such that
    P(x) = Q(x)/R(x) for all positive real numbers x. -/
theorem positive_polynomial_fraction_representation
  (P : RealPolynomial) (h : IsPositiveForPositive P) :
  ∃ (Q R : RealPolynomial), HasNonnegativeCoeffs Q ∧ HasNonnegativeCoeffs R ∧
    ∀ x : ℝ, x > 0 → P.eval x = (Q.eval x) / (R.eval x) := by
  sorry

end positive_polynomial_fraction_representation_l2395_239542


namespace spaceship_age_conversion_l2395_239586

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- The octal representation of the spaceship's age --/
def spaceship_age_octal : ℕ := 367

theorem spaceship_age_conversion :
  octal_to_decimal spaceship_age_octal = 247 := by sorry

end spaceship_age_conversion_l2395_239586


namespace x_value_l2395_239555

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 :=
by sorry

end x_value_l2395_239555


namespace max_d_value_l2395_239523

def a (n : ℕ) : ℕ := 99 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (M : ℕ), M = 401 ∧ ∀ (n : ℕ), n > 0 → d n ≤ M ∧ ∃ (k : ℕ), k > 0 ∧ d k = M :=
sorry

end max_d_value_l2395_239523


namespace euler_line_parallel_iff_condition_l2395_239512

/-- Triangle ABC with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The Euler line of a triangle -/
def EulerLine (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- A line parallel to the side BC of the triangle -/
def ParallelToBC (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The condition for Euler line parallelism -/
def EulerLineParallelCondition (t : Triangle) : Prop :=
  2 * t.a^4 = (t.b^2 - t.c^2)^2 + (t.b^2 + t.c^2) * t.a^2

/-- Theorem: The Euler line is parallel to side BC if and only if the condition holds -/
theorem euler_line_parallel_iff_condition (t : Triangle) :
  EulerLine t = ParallelToBC t ↔ EulerLineParallelCondition t := by sorry

end euler_line_parallel_iff_condition_l2395_239512


namespace symmetric_even_function_value_l2395_239509

/-- A function that is symmetric about x=2 -/
def SymmetricAbout2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (4 - x) = f x

/-- An even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem symmetric_even_function_value (f : ℝ → ℝ) 
  (h_sym : SymmetricAbout2 f) (h_even : EvenFunction f) (h_val : f 3 = 3) : 
  f (-1) = 3 := by
  sorry

end symmetric_even_function_value_l2395_239509


namespace complex_number_in_first_quadrant_l2395_239558

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (Complex.I - 1) / Complex.I
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_first_quadrant_l2395_239558


namespace triangle_dot_product_l2395_239551

/-- Given a triangle ABC with area √3 and angle A = π/3, 
    the dot product of vectors AB and AC is equal to 2. -/
theorem triangle_dot_product (A B C : ℝ × ℝ) : 
  let S := Real.sqrt 3
  let angleA := π / 3
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let area := Real.sqrt 3
  area = Real.sqrt 3 ∧ 
  angleA = π / 3 →
  AB.1 * AC.1 + AB.2 * AC.2 = 2 := by sorry

end triangle_dot_product_l2395_239551


namespace expression_simplification_l2395_239516

theorem expression_simplification (x : ℝ) : 
  (3*x^2 + 4*x - 5)*(x - 2) + (x - 2)*(2*x^2 - 3*x + 9) - (4*x - 7)*(x - 2)*(x - 3) = 
  x^3 + x^2 + 12*x - 36 := by
sorry

end expression_simplification_l2395_239516


namespace M_intersect_N_eq_unit_interval_l2395_239576

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x ^ 2 - Real.sin x ^ 2}

-- Define the set N
def N : Set ℝ := {x : ℝ | Complex.abs (2 * x / (1 - Complex.I * Real.sqrt 3)) < 1}

-- State the theorem
theorem M_intersect_N_eq_unit_interval : M ∩ N = Set.Icc 0 1 \ {1} := by sorry

end M_intersect_N_eq_unit_interval_l2395_239576


namespace polynomial_divisibility_l2395_239533

/-- The polynomial P(x) -/
def P (a : ℝ) (x : ℝ) : ℝ := x^1000 + a*x^2 + 9

/-- Theorem: P(x) is divisible by (x + 1) iff a = -10 -/
theorem polynomial_divisibility (a : ℝ) : 
  (∃ q : ℝ → ℝ, ∀ x, P a x = (x + 1) * q x) ↔ a = -10 := by
  sorry

end polynomial_divisibility_l2395_239533


namespace intersection_of_A_and_B_l2395_239553

def A : Set Int := {-1, 0, 1, 3, 5}
def B : Set Int := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by
  sorry

end intersection_of_A_and_B_l2395_239553


namespace warehouse_cleaning_time_l2395_239592

def lara_rate : ℚ := 1/4
def chris_rate : ℚ := 1/6
def break_time : ℚ := 2

theorem warehouse_cleaning_time (t : ℚ) : 
  (lara_rate + chris_rate) * (t - break_time) = 1 ↔ 
  t = (1 / (lara_rate + chris_rate)) + break_time :=
by sorry

end warehouse_cleaning_time_l2395_239592


namespace periodic_function_from_T_property_l2395_239514

-- Define the "T property" for a function
def has_T_property (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x : ℝ, (deriv f) (x + T) = (deriv f) x

-- Main theorem
theorem periodic_function_from_T_property (f : ℝ → ℝ) (T M : ℝ) 
  (hf : Continuous f) 
  (hT : has_T_property f T) 
  (hM : ∀ x : ℝ, |f x| < M) :
  ∀ x : ℝ, f (x + T) = f x :=
sorry

end periodic_function_from_T_property_l2395_239514


namespace locus_is_circle_l2395_239550

/-- The locus of points satisfying the given condition is a circle -/
theorem locus_is_circle (k : ℝ) (h : k > 0) :
  ∀ (x y : ℝ), (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ x / a + y / b = 1 ∧ 1 / a^2 + 1 / b^2 = 1 / k^2) 
  ↔ x^2 + y^2 = k^2 :=
sorry

end locus_is_circle_l2395_239550


namespace bowling_ball_weight_proof_l2395_239506

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18.75

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 30

theorem bowling_ball_weight_proof :
  (∀ (b k : ℝ), 8 * b = 5 * k ∧ 4 * k = 120 → b = bowling_ball_weight) :=
by sorry

end bowling_ball_weight_proof_l2395_239506


namespace trigonometric_identity_l2395_239518

theorem trigonometric_identity 
  (α β γ : Real) 
  (a b c : Real) 
  (h1 : 0 < α) (h2 : α < π)
  (h3 : 0 < β) (h4 : β < π)
  (h5 : 0 < γ) (h6 : γ < π)
  (h7 : a > 0) (h8 : b > 0) (h9 : c > 0)
  (h10 : b = c * (Real.cos α + Real.cos β * Real.cos γ) / (Real.sin γ)^2)
  (h11 : a = c * (Real.cos β + Real.cos α * Real.cos γ) / (Real.sin γ)^2) :
  1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0 := by
  sorry

end trigonometric_identity_l2395_239518


namespace joans_missed_games_l2395_239502

/-- Given that Joan's high school played 864 baseball games and she attended 395 games,
    prove that she missed 469 games. -/
theorem joans_missed_games (total_games : ℕ) (attended_games : ℕ)
  (h1 : total_games = 864)
  (h2 : attended_games = 395) :
  total_games - attended_games = 469 := by
sorry

end joans_missed_games_l2395_239502


namespace simplify_expression_l2395_239591

theorem simplify_expression (x : ℝ) : x^3 * x^2 * x + (x^3)^2 + (-2*x^2)^3 = -6*x^6 := by
  sorry

end simplify_expression_l2395_239591


namespace sin_2alpha_value_l2395_239536

theorem sin_2alpha_value (α : Real) (h : Real.cos (π/4 - α) = 3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end sin_2alpha_value_l2395_239536


namespace sugar_needed_is_six_l2395_239557

/-- Represents the ratios in a recipe --/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Calculates the amount of sugar needed in the new recipe --/
def sugar_needed (original : RecipeRatio) (water_new : ℚ) : ℚ :=
  let flour_water_ratio_new := 2 * (original.flour / original.water)
  let flour_sugar_ratio_new := (original.flour / original.sugar) / 2
  let flour_new := flour_water_ratio_new * water_new
  flour_new / flour_sugar_ratio_new

/-- Theorem stating that the amount of sugar needed is 6 cups --/
theorem sugar_needed_is_six :
  let original := RecipeRatio.mk 8 4 3
  let water_new := 2
  sugar_needed original water_new = 6 := by
  sorry

#eval sugar_needed (RecipeRatio.mk 8 4 3) 2

end sugar_needed_is_six_l2395_239557


namespace least_distinct_values_in_list_l2395_239559

theorem least_distinct_values_in_list (list : List ℕ) : 
  list.length = 2030 →
  ∃! m, m ∈ list ∧ (list.count m = 11) ∧ (∀ n ∈ list, n ≠ m → list.count n < 11) →
  (∃ x : ℕ, x = list.toFinset.card ∧ x ≥ 203 ∧ ∀ y : ℕ, y = list.toFinset.card → y ≥ x) :=
by sorry

end least_distinct_values_in_list_l2395_239559


namespace function_inequality_l2395_239522

/-- A function satisfying the given conditions -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, (x * (deriv f x) - f x) ≤ 0

theorem function_inequality
  (f : ℝ → ℝ)
  (h_nonneg : ∀ x > 0, f x ≥ 0)
  (h_diff : Differentiable ℝ f)
  (h_cond : SatisfiesCondition f)
  (m n : ℝ)
  (h_pos_m : m > 0)
  (h_pos_n : n > 0)
  (h_lt : m < n) :
  m * f n ≤ n * f m :=
sorry

end function_inequality_l2395_239522
