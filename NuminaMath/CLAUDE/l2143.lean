import Mathlib

namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2143_214368

/-- Proves that adding 300 mL of solution Y to 100 mL of solution X
    creates a solution that is 25% alcohol by volume. -/
theorem alcohol_mixture_proof 
  (x_conc : Real) -- Concentration of alcohol in solution X
  (y_conc : Real) -- Concentration of alcohol in solution Y
  (x_vol : Real)  -- Volume of solution X
  (y_vol : Real)  -- Volume of solution Y to be added
  (h1 : x_conc = 0.10) -- Solution X is 10% alcohol
  (h2 : y_conc = 0.30) -- Solution Y is 30% alcohol
  (h3 : x_vol = 100)   -- We start with 100 mL of solution X
  (h4 : y_vol = 300)   -- We add 300 mL of solution Y
  : (x_conc * x_vol + y_conc * y_vol) / (x_vol + y_vol) = 0.25 := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2143_214368


namespace NUMINAMATH_CALUDE_equation_solutions_l2143_214384

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => (14*x - x^2)/(x + 2) * (x + (14 - x)/(x + 2))
  ∃ (a b c : ℝ), 
    (f a = 48 ∧ f b = 48 ∧ f c = 48) ∧
    (a = 4 ∧ b = (1 + Real.sqrt 193)/2 ∧ c = (1 - Real.sqrt 193)/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2143_214384


namespace NUMINAMATH_CALUDE_simplify_expression_l2143_214363

theorem simplify_expression (w x : ℝ) : 
  3*w + 5*w + 7*w + 9*w + 11*w + 13*x + 15 = 35*w + 13*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2143_214363


namespace NUMINAMATH_CALUDE_class_reading_total_l2143_214360

/-- Calculates the total number of books read by a class per week given the following conditions:
  * There are 12 girls and 10 boys in the class.
  * 5/6 of the girls and 4/5 of the boys are reading.
  * Girls read at an average rate of 3 books per week.
  * Boys read at an average rate of 2 books per week.
  * 20% of reading girls read at a faster rate of 5 books per week.
  * 10% of reading boys read at a slower rate of 1 book per week.
-/
theorem class_reading_total (girls : ℕ) (boys : ℕ) 
  (girls_reading_ratio : ℚ) (boys_reading_ratio : ℚ)
  (girls_avg_rate : ℕ) (boys_avg_rate : ℕ)
  (girls_faster_ratio : ℚ) (boys_slower_ratio : ℚ)
  (girls_faster_rate : ℕ) (boys_slower_rate : ℕ) :
  girls = 12 →
  boys = 10 →
  girls_reading_ratio = 5/6 →
  boys_reading_ratio = 4/5 →
  girls_avg_rate = 3 →
  boys_avg_rate = 2 →
  girls_faster_ratio = 1/5 →
  boys_slower_ratio = 1/10 →
  girls_faster_rate = 5 →
  boys_slower_rate = 1 →
  (girls_reading_ratio * girls * girls_avg_rate +
   boys_reading_ratio * boys * boys_avg_rate +
   girls_reading_ratio * girls * girls_faster_ratio * (girls_faster_rate - girls_avg_rate) +
   boys_reading_ratio * boys * boys_slower_ratio * (boys_slower_rate - boys_avg_rate)) = 49 := by
  sorry


end NUMINAMATH_CALUDE_class_reading_total_l2143_214360


namespace NUMINAMATH_CALUDE_distance_difference_l2143_214380

def house_to_bank : ℕ := 800
def bank_to_pharmacy : ℕ := 1300
def pharmacy_to_school : ℕ := 1700

theorem distance_difference : 
  (house_to_bank + bank_to_pharmacy) - pharmacy_to_school = 400 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2143_214380


namespace NUMINAMATH_CALUDE_coffee_purchase_problem_l2143_214369

/-- Given a gift card balance, coffee price per pound, and remaining balance,
    calculate the number of pounds of coffee purchased. -/
def coffee_pounds_purchased (gift_card_balance : ℚ) (coffee_price_per_pound : ℚ) (remaining_balance : ℚ) : ℚ :=
  (gift_card_balance - remaining_balance) / coffee_price_per_pound

theorem coffee_purchase_problem :
  let gift_card_balance : ℚ := 70
  let coffee_price_per_pound : ℚ := 8.58
  let remaining_balance : ℚ := 35.68
  coffee_pounds_purchased gift_card_balance coffee_price_per_pound remaining_balance = 4 := by
  sorry

end NUMINAMATH_CALUDE_coffee_purchase_problem_l2143_214369


namespace NUMINAMATH_CALUDE_factorization_equality_l2143_214313

theorem factorization_equality (x : ℝ) : 84 * x^5 - 210 * x^9 = -42 * x^5 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2143_214313


namespace NUMINAMATH_CALUDE_boat_distance_downstream_l2143_214389

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_distance_downstream 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : time = 4) : 
  boat_speed + stream_speed * time = 112 := by
  sorry


end NUMINAMATH_CALUDE_boat_distance_downstream_l2143_214389


namespace NUMINAMATH_CALUDE_geometric_sequence_max_product_l2143_214398

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

def product_of_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a₁^n) * (q^((n * (n - 1)) / 2))

theorem geometric_sequence_max_product :
  ∃ (q : ℝ) (n : ℕ),
    geometric_sequence (-6) q 4 = (-3/4) ∧
    q = (1/2) ∧
    n = 4 ∧
    ∀ (m : ℕ), m ≠ 0 → product_of_terms (-6) q m ≤ product_of_terms (-6) q n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_product_l2143_214398


namespace NUMINAMATH_CALUDE_trailingZeroes_15_factorial_base12_l2143_214305

/-- The number of trailing zeroes in the base 12 representation of 15! -/
def trailingZeroesBase12Factorial15 : ℕ :=
  min (Nat.factorial 15 / 12^5) 1

theorem trailingZeroes_15_factorial_base12 :
  trailingZeroesBase12Factorial15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_trailingZeroes_15_factorial_base12_l2143_214305


namespace NUMINAMATH_CALUDE_remaining_plums_l2143_214383

def gyuris_plums (initial : ℝ) (given_to_sungmin : ℝ) (given_to_dongju : ℝ) : ℝ :=
  initial - given_to_sungmin - given_to_dongju

theorem remaining_plums :
  gyuris_plums 1.6 0.8 0.3 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_plums_l2143_214383


namespace NUMINAMATH_CALUDE_greg_age_l2143_214376

/-- Given the ages of five people with certain relationships, prove Greg's age --/
theorem greg_age (C D E F G : ℕ) : 
  D = E - 5 →
  E = 2 * C →
  F = C - 1 →
  G = 2 * F →
  D = 15 →
  G = 18 := by
  sorry

#check greg_age

end NUMINAMATH_CALUDE_greg_age_l2143_214376


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2143_214342

/-- Given arithmetic sequences a and b with sums S and T, prove the ratio of a_6 to b_8 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ) :
  (∀ n, S n = (n + 2) * (T n / (n + 1))) →  -- Condition: S_n / T_n = (n + 2) / (n + 1)
  (∀ n, S (n + 1) - S n = a (n + 1)) →      -- Definition of S as sum of a
  (∀ n, T (n + 1) - T n = b (n + 1)) →      -- Definition of T as sum of b
  (∀ n, a (n + 1) - a n = a 2 - a 1) →      -- a is arithmetic sequence
  (∀ n, b (n + 1) - b n = b 2 - b 1) →      -- b is arithmetic sequence
  a 6 / b 8 = 13 / 16 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2143_214342


namespace NUMINAMATH_CALUDE_interest_tax_rate_proof_l2143_214306

/-- The tax rate for interest tax on savings deposits in China --/
def interest_tax_rate : ℝ := 0.20

theorem interest_tax_rate_proof (initial_deposit : ℝ) (interest_rate : ℝ) (total_received : ℝ)
  (h1 : initial_deposit = 10000)
  (h2 : interest_rate = 0.0225)
  (h3 : total_received = 10180) :
  initial_deposit + initial_deposit * interest_rate * (1 - interest_tax_rate) = total_received :=
by sorry

end NUMINAMATH_CALUDE_interest_tax_rate_proof_l2143_214306


namespace NUMINAMATH_CALUDE_window_width_theorem_l2143_214341

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ

/-- Represents the dimensions and properties of a window -/
structure Window where
  rows : ℕ
  columns : ℕ
  pane : Pane
  border_width : ℝ

/-- Calculates the total width of the window -/
def total_width (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- Theorem stating the total width of the window with given conditions -/
theorem window_width_theorem (x : ℝ) : 
  let w : Window := {
    rows := 3,
    columns := 4,
    pane := { width := 4 * x, height := 3 * x },
    border_width := 3
  }
  total_width w = 16 * x + 15 := by sorry

end NUMINAMATH_CALUDE_window_width_theorem_l2143_214341


namespace NUMINAMATH_CALUDE_marks_songs_per_gig_l2143_214387

/-- Represents the number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- Represents the number of gigs Mark does in two weeks -/
def number_of_gigs : ℕ := days_in_two_weeks / 2

/-- Represents the duration of a short song in minutes -/
def short_song_duration : ℕ := 5

/-- Represents the duration of a long song in minutes -/
def long_song_duration : ℕ := 2 * short_song_duration

/-- Represents the number of short songs per gig -/
def short_songs_per_gig : ℕ := 2

/-- Represents the number of long songs per gig -/
def long_songs_per_gig : ℕ := 1

/-- Represents the total playing time for all gigs in minutes -/
def total_playing_time : ℕ := 280

/-- Theorem: Given the conditions, Mark plays 7 songs at each gig -/
theorem marks_songs_per_gig :
  ∃ (songs_per_gig : ℕ),
    songs_per_gig = short_songs_per_gig + long_songs_per_gig +
      ((total_playing_time / number_of_gigs) -
       (short_songs_per_gig * short_song_duration + long_songs_per_gig * long_song_duration)) /
      short_song_duration ∧
    songs_per_gig = 7 :=
by sorry

end NUMINAMATH_CALUDE_marks_songs_per_gig_l2143_214387


namespace NUMINAMATH_CALUDE_logging_time_is_ten_months_l2143_214392

/-- Represents the forest and logging scenario --/
structure LoggingScenario where
  forestLength : ℕ
  forestWidth : ℕ
  treesPerSquareMile : ℕ
  loggersCount : ℕ
  treesPerLoggerPerDay : ℕ
  daysPerMonth : ℕ

/-- Calculates the number of months required to cut down all trees --/
def monthsToLogForest (scenario : LoggingScenario) : ℚ :=
  let totalArea := scenario.forestLength * scenario.forestWidth
  let totalTrees := totalArea * scenario.treesPerSquareMile
  let treesPerDay := scenario.loggersCount * scenario.treesPerLoggerPerDay
  (totalTrees : ℚ) / (treesPerDay * scenario.daysPerMonth)

/-- Theorem stating that it takes 10 months to log the forest under given conditions --/
theorem logging_time_is_ten_months :
  let scenario : LoggingScenario := {
    forestLength := 4,
    forestWidth := 6,
    treesPerSquareMile := 600,
    loggersCount := 8,
    treesPerLoggerPerDay := 6,
    daysPerMonth := 30
  }
  monthsToLogForest scenario = 10 := by sorry

end NUMINAMATH_CALUDE_logging_time_is_ten_months_l2143_214392


namespace NUMINAMATH_CALUDE_cartesian_to_polar_l2143_214373

theorem cartesian_to_polar :
  let x : ℝ := -2
  let y : ℝ := 2 * Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 2 * π / 3
  ρ = 4 ∧ Real.cos θ = x / ρ ∧ Real.sin θ = y / ρ := by sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_l2143_214373


namespace NUMINAMATH_CALUDE_correct_termination_condition_l2143_214375

/-- Represents the state of the program at each iteration --/
structure ProgramState :=
  (i : ℕ)
  (S : ℕ)

/-- Simulates one iteration of the loop --/
def iterate (state : ProgramState) : ProgramState :=
  { i := state.i - 1, S := state.S * state.i }

/-- Checks if the given condition terminates the loop correctly --/
def is_correct_termination (condition : ℕ → Bool) : Prop :=
  let final_state := iterate (iterate (iterate (iterate { i := 12, S := 1 })))
  final_state.S = 11880 ∧ condition final_state.i = true ∧ 
  ∀ n, n < 4 → condition ((iterate^[n] { i := 12, S := 1 }).i) = false

theorem correct_termination_condition :
  is_correct_termination (λ i => i = 8) := by sorry

end NUMINAMATH_CALUDE_correct_termination_condition_l2143_214375


namespace NUMINAMATH_CALUDE_star_example_l2143_214331

-- Define the * operation
def star (a b c d : ℚ) : ℚ := a * c * (d / (b + 1))

-- Theorem statement
theorem star_example : star 5 11 9 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l2143_214331


namespace NUMINAMATH_CALUDE_product_congruence_l2143_214354

theorem product_congruence : 198 * 963 ≡ 24 [ZMOD 50] := by sorry

end NUMINAMATH_CALUDE_product_congruence_l2143_214354


namespace NUMINAMATH_CALUDE_largest_digit_sum_l2143_214345

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, and c are digits
  (10 ≤ y ∧ y ≤ 20) →  -- 10 ≤ y ≤ 20
  ((a * 100 + b * 10 + c) : ℚ) / 1000 = 1 / y →  -- 0.abc = 1/y
  a + b + c ≤ 5 :=  -- The sum is at most 5
by sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l2143_214345


namespace NUMINAMATH_CALUDE_weeks_to_buy_iphone_l2143_214333

def iphone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_buy_iphone :
  (iphone_cost - trade_in_value) / weekly_earnings = 7 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_buy_iphone_l2143_214333


namespace NUMINAMATH_CALUDE_det_transformation_l2143_214328

/-- Given a 2x2 matrix with determinant 6, prove that a specific transformation of this matrix results in a determinant of 24. -/
theorem det_transformation (p q r s : ℝ) (h : Matrix.det !![p, q; r, s] = 6) :
  Matrix.det !![p, 8*p + 4*q; r, 8*r + 4*s] = 24 := by
  sorry

end NUMINAMATH_CALUDE_det_transformation_l2143_214328


namespace NUMINAMATH_CALUDE_letters_in_small_envelopes_l2143_214358

/-- Given the total number of letters, the number of large envelopes, and the number of letters
    per large envelope, calculate the number of letters in small envelopes. -/
theorem letters_in_small_envelopes 
  (total_letters : ℕ) 
  (large_envelopes : ℕ) 
  (letters_per_large_envelope : ℕ) 
  (h1 : total_letters = 80)
  (h2 : large_envelopes = 30)
  (h3 : letters_per_large_envelope = 2) : 
  total_letters - large_envelopes * letters_per_large_envelope = 20 :=
by sorry

end NUMINAMATH_CALUDE_letters_in_small_envelopes_l2143_214358


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l2143_214357

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area is 2√3, angle B = π/3, and a² + c² = 3ac, then b = 4 -/
theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) :
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 3 →
  B = π / 3 →
  a^2 + c^2 = 3 * a * c →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l2143_214357


namespace NUMINAMATH_CALUDE_intersection_two_elements_l2143_214334

/-- The set M represents lines passing through (1,1) with slope k -/
def M (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - 1) + 1}

/-- The set N represents a circle with center (0,1) and radius 1 -/
def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.2 = 0}

/-- The intersection of M and N contains exactly two elements -/
theorem intersection_two_elements (k : ℝ) : ∃ (p q : ℝ × ℝ), p ≠ q ∧
  M k ∩ N = {p, q} :=
sorry

end NUMINAMATH_CALUDE_intersection_two_elements_l2143_214334


namespace NUMINAMATH_CALUDE_four_digit_greater_than_three_digit_l2143_214394

theorem four_digit_greater_than_three_digit :
  ∀ (a b : ℕ), (1000 ≤ a ∧ a < 10000) → (100 ≤ b ∧ b < 1000) → a > b :=
by
  sorry

end NUMINAMATH_CALUDE_four_digit_greater_than_three_digit_l2143_214394


namespace NUMINAMATH_CALUDE_participant_count_2019_l2143_214348

/-- The number of participants in the Science Quiz Bowl for different years --/
structure ParticipantCount where
  y2018 : ℕ
  y2019 : ℕ
  y2020 : ℕ

/-- The conditions given in the problem --/
def satisfiesConditions (p : ParticipantCount) : Prop :=
  p.y2018 = 150 ∧
  p.y2020 = p.y2019 / 2 - 40 ∧
  p.y2019 = p.y2020 + 200

/-- The theorem to be proved --/
theorem participant_count_2019 (p : ParticipantCount) 
  (h : satisfiesConditions p) : 
  p.y2019 = 320 ∧ p.y2019 - p.y2018 = 170 :=
by
  sorry

end NUMINAMATH_CALUDE_participant_count_2019_l2143_214348


namespace NUMINAMATH_CALUDE_twelfth_term_value_l2143_214378

/-- An arithmetic sequence with a₂ = -8 and common difference d = 2 -/
def arithmetic_seq (n : ℕ) : ℤ :=
  let a₁ : ℤ := -10  -- Derived from a₂ = -8 and d = 2
  a₁ + (n - 1) * 2

theorem twelfth_term_value : arithmetic_seq 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_value_l2143_214378


namespace NUMINAMATH_CALUDE_mary_regular_hours_l2143_214390

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  regularHours : ℝ
  overtimeHours : ℝ
  regularRate : ℝ
  overtimeRate : ℝ
  maxHours : ℝ
  maxEarnings : ℝ

/-- Calculates total earnings based on work schedule --/
def totalEarnings (w : WorkSchedule) : ℝ :=
  w.regularHours * w.regularRate + w.overtimeHours * w.overtimeRate

/-- Theorem stating that Mary works 20 hours at her regular rate --/
theorem mary_regular_hours (w : WorkSchedule) 
  (h1 : w.maxHours = 80)
  (h2 : w.regularRate = 8)
  (h3 : w.overtimeRate = w.regularRate * 1.25)
  (h4 : w.maxEarnings = 760)
  (h5 : w.regularHours + w.overtimeHours = w.maxHours)
  (h6 : totalEarnings w = w.maxEarnings) :
  w.regularHours = 20 := by
  sorry

#check mary_regular_hours

end NUMINAMATH_CALUDE_mary_regular_hours_l2143_214390


namespace NUMINAMATH_CALUDE_first_person_work_days_l2143_214351

-- Define the work rates
def work_rate_prakash : ℚ := 1 / 40
def work_rate_together : ℚ := 1 / 15

-- Define the theorem
theorem first_person_work_days :
  ∃ (x : ℚ), 
    x > 0 ∧ 
    (1 / x) + work_rate_prakash = work_rate_together ∧ 
    x = 24 := by
  sorry

end NUMINAMATH_CALUDE_first_person_work_days_l2143_214351


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l2143_214315

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 1, b = 2, and C = 60°, then c = √3 and the area is √3/2 -/
theorem triangle_side_and_area 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 1) 
  (h2 : b = 2) 
  (h3 : C = Real.pi / 3) -- 60° in radians
  (h4 : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) -- Law of cosines
  (h5 : (a*b*(Real.sin C))/2 = area_triangle) : 
  c = Real.sqrt 3 ∧ area_triangle = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_and_area_l2143_214315


namespace NUMINAMATH_CALUDE_f_value_at_3_l2143_214370

/-- Given a function f(x) = x^7 + ax^5 + bx - 5 where f(-3) = 5, prove that f(3) = -15 -/
theorem f_value_at_3 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^7 + a*x^5 + b*x - 5)
    (h2 : f (-3) = 5) : 
  f 3 = -15 := by sorry

end NUMINAMATH_CALUDE_f_value_at_3_l2143_214370


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2143_214362

theorem sum_of_coefficients (d : ℝ) (h : d ≠ 0) : ∃ a b c : ℝ, 
  (16 * d + 17 + 18 * d^3) + (4 * d + 2) = a * d^3 + b * d + c ∧ a + b + c = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2143_214362


namespace NUMINAMATH_CALUDE_smallest_n_not_divisible_by_10_l2143_214344

theorem smallest_n_not_divisible_by_10 :
  ∃ (n : ℕ), n = 2020 ∧ n > 2016 ∧
  ¬(10 ∣ (1^n + 2^n + 3^n + 4^n)) ∧
  ∀ (m : ℕ), 2016 < m ∧ m < n → (10 ∣ (1^m + 2^m + 3^m + 4^m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_not_divisible_by_10_l2143_214344


namespace NUMINAMATH_CALUDE_crushing_load_calculation_l2143_214314

theorem crushing_load_calculation (T H K : ℚ) (L : ℚ) 
  (h1 : T = 5)
  (h2 : H = 10)
  (h3 : K = 2)
  (h4 : L = (30 * T^3 * K) / H^3) :
  L = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_crushing_load_calculation_l2143_214314


namespace NUMINAMATH_CALUDE_square_odd_implies_odd_l2143_214393

theorem square_odd_implies_odd (n : ℤ) : Odd (n^2) → Odd n := by
  sorry

end NUMINAMATH_CALUDE_square_odd_implies_odd_l2143_214393


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l2143_214320

def batting_average (innings : ℕ) (total_runs : ℕ) : ℚ :=
  total_runs / innings

def revised_average (innings : ℕ) (total_runs : ℕ) (not_out : ℕ) : ℚ :=
  total_runs / (innings - not_out)

theorem batsman_average_theorem (total_runs_11 : ℕ) (innings : ℕ) (last_score : ℕ) (not_out : ℕ) :
  innings = 12 →
  last_score = 92 →
  not_out = 3 →
  batting_average innings (total_runs_11 + last_score) - batting_average (innings - 1) total_runs_11 = 2 →
  batting_average innings (total_runs_11 + last_score) = 70 ∧
  revised_average innings (total_runs_11 + last_score) not_out = 93.33 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l2143_214320


namespace NUMINAMATH_CALUDE_intersection_theorem_subset_theorem_l2143_214323

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | -x^2 + 2*m*x + 4 - m^2 ≥ 0}
def B : Set ℝ := {x | 2*x^2 - 5*x - 7 < 0}

-- Define the intersection of A and B
def A_intersect_B (m : ℝ) : Set ℝ := A m ∩ B

-- Define the complement of A in ℝ
def complement_A (m : ℝ) : Set ℝ := {x | x ∉ A m}

-- Theorem for part (1)
theorem intersection_theorem (m : ℝ) :
  A_intersect_B m = {x | 0 ≤ x ∧ x < 7/2} ↔ m = 2 :=
sorry

-- Theorem for part (2)
theorem subset_theorem (m : ℝ) :
  B ⊆ complement_A m ↔ m ≤ -3 ∨ m ≥ 11/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_theorem_subset_theorem_l2143_214323


namespace NUMINAMATH_CALUDE_expression_is_negative_l2143_214347

theorem expression_is_negative : 
  Real.sqrt (25 * Real.sqrt 7 - 27 * Real.sqrt 6) - Real.sqrt (17 * Real.sqrt 5 - 38) < 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_is_negative_l2143_214347


namespace NUMINAMATH_CALUDE_calculate_annual_interest_rate_l2143_214388

/-- Given an initial charge and the amount owed after one year with simple annual interest,
    calculate the annual interest rate. -/
theorem calculate_annual_interest_rate
  (initial_charge : ℝ)
  (amount_owed_after_year : ℝ)
  (h1 : initial_charge = 35)
  (h2 : amount_owed_after_year = 37.1)
  (h3 : amount_owed_after_year = initial_charge * (1 + interest_rate))
  : interest_rate = 0.06 :=
sorry

end NUMINAMATH_CALUDE_calculate_annual_interest_rate_l2143_214388


namespace NUMINAMATH_CALUDE_ratio_difference_l2143_214302

theorem ratio_difference (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_l2143_214302


namespace NUMINAMATH_CALUDE_store_distribution_problem_l2143_214332

/-- Represents the number of ways to distribute stores among cities -/
def distributionCount (totalStores : ℕ) (totalCities : ℕ) (maxStoresPerCity : ℕ) : ℕ :=
  sorry

/-- The specific problem conditions -/
theorem store_distribution_problem :
  distributionCount 4 5 2 = 45 := by sorry

end NUMINAMATH_CALUDE_store_distribution_problem_l2143_214332


namespace NUMINAMATH_CALUDE_football_result_unique_solution_l2143_214352

/-- Represents the result of a football team's performance -/
structure FootballResult where
  total_matches : ℕ
  lost_matches : ℕ
  total_points : ℕ
  wins : ℕ
  draws : ℕ

/-- Checks if a FootballResult is valid according to the given rules -/
def is_valid_result (r : FootballResult) : Prop :=
  r.total_matches = r.wins + r.draws + r.lost_matches ∧
  r.total_points = 3 * r.wins + r.draws

/-- Theorem stating the unique solution for the given problem -/
theorem football_result_unique_solution :
  ∃! (r : FootballResult),
    r.total_matches = 15 ∧
    r.lost_matches = 4 ∧
    r.total_points = 29 ∧
    is_valid_result r ∧
    r.wins = 9 ∧
    r.draws = 2 := by
  sorry

end NUMINAMATH_CALUDE_football_result_unique_solution_l2143_214352


namespace NUMINAMATH_CALUDE_most_efficient_numbering_system_l2143_214399

/-- Represents a numbering system for a population --/
inductive NumberingSystem
  | OneToN
  | ZeroToNMinusOne
  | TwoDigitZeroToNMinusOne
  | ThreeDigitZeroToNMinusOne

/-- Determines if a numbering system is most efficient for random number table sampling --/
def is_most_efficient (n : NumberingSystem) (population_size : ℕ) (sample_size : ℕ) : Prop :=
  n = NumberingSystem.ThreeDigitZeroToNMinusOne ∧ 
  population_size = 106 ∧ 
  sample_size = 10

/-- Theorem stating the most efficient numbering system for the given conditions --/
theorem most_efficient_numbering_system :
  ∃ (n : NumberingSystem), is_most_efficient n 106 10 :=
sorry

end NUMINAMATH_CALUDE_most_efficient_numbering_system_l2143_214399


namespace NUMINAMATH_CALUDE_inequality_multiplication_l2143_214327

theorem inequality_multiplication (x y : ℝ) (h : x < y) : 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l2143_214327


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2143_214396

theorem diophantine_equation_solutions (x y : ℤ) :
  x^6 - y^2 = 648 ↔ (x = 3 ∧ y = 9) ∨ (x = -3 ∧ y = 9) ∨ (x = 3 ∧ y = -9) ∨ (x = -3 ∧ y = -9) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2143_214396


namespace NUMINAMATH_CALUDE_total_oil_volume_l2143_214324

-- Define the volume of each bottle in mL
def bottle_volume : ℕ := 200

-- Define the number of bottles
def num_bottles : ℕ := 20

-- Define the conversion factor from mL to L
def ml_per_liter : ℕ := 1000

-- Theorem to prove
theorem total_oil_volume (bottle_volume : ℕ) (num_bottles : ℕ) (ml_per_liter : ℕ) :
  bottle_volume = 200 → num_bottles = 20 → ml_per_liter = 1000 →
  (bottle_volume * num_bottles) / ml_per_liter = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_volume_l2143_214324


namespace NUMINAMATH_CALUDE_difference_of_squares_of_odd_numbers_divisible_by_eight_l2143_214318

theorem difference_of_squares_of_odd_numbers_divisible_by_eight (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 2 * k + 1) (hb : ∃ k : ℤ, b = 2 * k + 1) : 
  ∃ m : ℤ, a^2 - b^2 = 8 * m :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_of_odd_numbers_divisible_by_eight_l2143_214318


namespace NUMINAMATH_CALUDE_equation_solution_l2143_214303

def solution_set : Set ℝ := {0, -6}

theorem equation_solution :
  ∀ x : ℝ, (2 * |x + 3| - 4 = 2) ↔ x ∈ solution_set := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2143_214303


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2143_214350

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) :
  (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2143_214350


namespace NUMINAMATH_CALUDE_mean_temperature_l2143_214316

def temperatures : List ℚ := [75, 80, 78, 82, 85, 90, 87, 84, 88, 93]

theorem mean_temperature : (temperatures.sum / temperatures.length : ℚ) = 421/5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l2143_214316


namespace NUMINAMATH_CALUDE_polynomial_four_positive_roots_l2143_214397

/-- A polynomial with four positive real roots -/
def P (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 - 8*a * x^3 + b * x^2 - 32*c * x + 16*c

/-- The theorem stating the conditions for the polynomial to have four positive real roots -/
theorem polynomial_four_positive_roots :
  ∀ (a : ℝ), a ≠ 0 →
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
    ∀ (x : ℝ), P a (16*a) a x = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
  (∀ (b c : ℝ), 
    (∃ (x₁ x₂ x₃ x₄ : ℝ), 
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
      ∀ (x : ℝ), P a b c x = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
    b = 16*a ∧ c = a) := by
  sorry


end NUMINAMATH_CALUDE_polynomial_four_positive_roots_l2143_214397


namespace NUMINAMATH_CALUDE_computation_problem_value_l2143_214374

theorem computation_problem_value (total_problems : Nat) (word_problem_value : Nat) 
  (total_points : Nat) (computation_problems : Nat) :
  total_problems = 30 →
  word_problem_value = 5 →
  total_points = 110 →
  computation_problems = 20 →
  ∃ (computation_value : Nat),
    computation_value = 3 ∧
    total_points = computation_problems * computation_value + 
      (total_problems - computation_problems) * word_problem_value :=
by sorry

end NUMINAMATH_CALUDE_computation_problem_value_l2143_214374


namespace NUMINAMATH_CALUDE_ball_diameter_proof_l2143_214379

theorem ball_diameter_proof (h s d : ℝ) (h_pos : h > 0) (s_pos : s > 0) (d_pos : d > 0) :
  h / s = (h / s) / (1 + d / (h / s)) → h / s = 1.25 → s = 1 → d = 0.23 → h / s = 0.23 :=
by sorry

end NUMINAMATH_CALUDE_ball_diameter_proof_l2143_214379


namespace NUMINAMATH_CALUDE_mittens_per_box_l2143_214329

theorem mittens_per_box (num_boxes : ℕ) (scarves_per_box : ℕ) (total_items : ℕ) 
  (h1 : num_boxes = 4)
  (h2 : scarves_per_box = 2)
  (h3 : total_items = 32) :
  (total_items - num_boxes * scarves_per_box) / num_boxes = 6 :=
by sorry

end NUMINAMATH_CALUDE_mittens_per_box_l2143_214329


namespace NUMINAMATH_CALUDE_grid_transformation_impossible_l2143_214382

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℤ

/-- The initial grid configuration -/
def initial_grid : Grid :=
  fun i j => 
    match i, j with
    | 0, 0 => 1 | 0, 1 => 2 | 0, 2 => 3
    | 1, 0 => 4 | 1, 1 => 5 | 1, 2 => 6
    | 2, 0 => 7 | 2, 1 => 8 | 2, 2 => 9

/-- The target grid configuration -/
def target_grid : Grid :=
  fun i j => 
    match i, j with
    | 0, 0 => 7 | 0, 1 => 9 | 0, 2 => 2
    | 1, 0 => 3 | 1, 1 => 5 | 1, 2 => 6
    | 2, 0 => 1 | 2, 1 => 4 | 2, 2 => 8

/-- Calculates the invariant of a grid -/
def grid_invariant (g : Grid) : ℤ :=
  (g 0 0 + g 0 2 + g 1 1 + g 2 0 + g 2 2) - (g 0 1 + g 1 0 + g 1 2 + g 2 1)

/-- Theorem stating the impossibility of transforming the initial grid to the target grid -/
theorem grid_transformation_impossible : 
  ¬∃ (f : Grid → Grid), (f initial_grid = target_grid ∧ 
    ∀ g : Grid, grid_invariant g = grid_invariant (f g)) :=
by
  sorry


end NUMINAMATH_CALUDE_grid_transformation_impossible_l2143_214382


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2143_214395

theorem smallest_number_divisible (n : ℕ) : n ≥ 1012 ∧ 
  (∀ m : ℕ, m < 1012 → 
    ¬(((m - 4) % 12 = 0) ∧ 
      ((m - 4) % 16 = 0) ∧ 
      ((m - 4) % 18 = 0) ∧ 
      ((m - 4) % 21 = 0) ∧ 
      ((m - 4) % 28 = 0))) →
  ((n - 4) % 12 = 0) ∧ 
  ((n - 4) % 16 = 0) ∧ 
  ((n - 4) % 18 = 0) ∧ 
  ((n - 4) % 21 = 0) ∧ 
  ((n - 4) % 28 = 0) :=
by sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l2143_214395


namespace NUMINAMATH_CALUDE_a_2017_equals_2_l2143_214364

def sequence_a : ℕ → ℚ
  | 0 => 2
  | n + 1 => (sequence_a n - 1) / (sequence_a n + 1)

theorem a_2017_equals_2 : sequence_a 2016 = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_2017_equals_2_l2143_214364


namespace NUMINAMATH_CALUDE_divisible_by_six_l2143_214372

theorem divisible_by_six (n : ℕ) : ∃ k : ℤ, (n - 1) * n * (n^3 + 1) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l2143_214372


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_l2143_214367

theorem negation_of_forall_inequality :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_l2143_214367


namespace NUMINAMATH_CALUDE_resulting_polygon_sides_l2143_214359

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ
  is_regular : sides ≥ 3

/-- Represents the sequence of polygons in the construction. -/
def polygon_sequence : List RegularPolygon :=
  [⟨3, by norm_num⟩, ⟨4, by norm_num⟩, ⟨5, by norm_num⟩, 
   ⟨6, by norm_num⟩, ⟨7, by norm_num⟩, ⟨8, by norm_num⟩, ⟨9, by norm_num⟩]

/-- Calculates the number of exposed sides in the resulting polygon. -/
def exposed_sides (seq : List RegularPolygon) : ℕ :=
  (seq.map (·.sides)).sum - 2 * (seq.length - 1)

/-- Theorem stating that the resulting polygon has 30 sides. -/
theorem resulting_polygon_sides : exposed_sides polygon_sequence = 30 := by
  sorry


end NUMINAMATH_CALUDE_resulting_polygon_sides_l2143_214359


namespace NUMINAMATH_CALUDE_chord_length_line_circle_l2143_214356

/-- The chord length cut by a line from a circle -/
theorem chord_length_line_circle 
  (line : (ℝ × ℝ) → Prop) 
  (circle : (ℝ × ℝ) → Prop) : 
  line = λ (x, y) ↦ 3*x - 4*y - 4 = 0 →
  circle = λ (x, y) ↦ (x - 3)^2 + y^2 = 9 →
  ∃ (a b : ℝ), 
    (a, b) ∈ {p | line p ∧ circle p} ∧
    (∀ (c d : ℝ), (c, d) ∈ {p | line p ∧ circle p} → (a - c)^2 + (b - d)^2 ≤ 32) ∧
    (∃ (e f : ℝ), (e, f) ∈ {p | line p ∧ circle p} ∧ (a - e)^2 + (b - f)^2 = 32) :=
by
  sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_l2143_214356


namespace NUMINAMATH_CALUDE_angle_C_indeterminate_l2143_214337

/-- Represents a quadrilateral with angles A, B, C, and D -/
structure Quadrilateral where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angleD : ℝ
  sum_360 : angleA + angleB + angleC + angleD = 360

/-- Theorem stating that ∠C cannot be determined in a quadrilateral ABCD 
    where ∠A = 80° and ∠B = 100° without information about ∠D -/
theorem angle_C_indeterminate (q : Quadrilateral) 
    (hA : q.angleA = 80) (hB : q.angleB = 100) :
  ∀ (x : ℝ), 0 < x ∧ x < 180 → 
  ∃ (q' : Quadrilateral), q'.angleA = q.angleA ∧ q'.angleB = q.angleB ∧ q'.angleC = x :=
sorry

end NUMINAMATH_CALUDE_angle_C_indeterminate_l2143_214337


namespace NUMINAMATH_CALUDE_xyz_sum_reciprocal_l2143_214307

theorem xyz_sum_reciprocal (x y z : ℝ) 
  (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_sum_x : x + 1 / z = 7)
  (h_sum_y : y + 1 / x = 31) :
  z + 1 / y = 5 / 27 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_reciprocal_l2143_214307


namespace NUMINAMATH_CALUDE_average_other_color_marbles_l2143_214349

/-- Given a collection of marbles where 40% are clear, 20% are black, and the remainder are other colors,
    prove that when taking 5 marbles, the average number of marbles of other colors is 2. -/
theorem average_other_color_marbles
  (total : ℕ) -- Total number of marbles
  (clear : ℕ) -- Number of clear marbles
  (black : ℕ) -- Number of black marbles
  (other : ℕ) -- Number of other color marbles
  (h1 : clear = (40 * total) / 100) -- 40% are clear
  (h2 : black = (20 * total) / 100) -- 20% are black
  (h3 : other = total - clear - black) -- Remainder are other colors
  : (40 : ℚ) / 100 * 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_other_color_marbles_l2143_214349


namespace NUMINAMATH_CALUDE_existence_of_points_with_derivative_sum_zero_l2143_214309

theorem existence_of_points_with_derivative_sum_zero
  {f : ℝ → ℝ} {a b : ℝ} (h_diff : DifferentiableOn ℝ f (Set.Icc a b))
  (h_eq : f a = f b) (h_lt : a < b) :
  ∃ x y, x ∈ Set.Icc a b ∧ y ∈ Set.Icc a b ∧ x ≠ y ∧
    (deriv f x) + 5 * (deriv f y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_points_with_derivative_sum_zero_l2143_214309


namespace NUMINAMATH_CALUDE_milk_water_ratio_l2143_214312

theorem milk_water_ratio (initial_milk : ℝ) (initial_water : ℝ) : 
  initial_milk > 0 ∧ initial_water > 0 →
  initial_milk + initial_water + 8 = 72 →
  (initial_milk + 8) / initial_water = 2 →
  initial_milk / initial_water = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l2143_214312


namespace NUMINAMATH_CALUDE_expression_evaluation_l2143_214340

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1
  let expr := ((2 * x - 1/2 * y)^2 - (-y + 2*x) * (2*x + y) + y * (x^2 * y - 5/4 * y)) / x
  expr = -4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2143_214340


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2143_214353

theorem solution_set_equivalence (x : ℝ) : (x + 3) * (x - 5) < 0 ↔ -3 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2143_214353


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2143_214322

theorem sum_of_three_numbers (second : ℕ) (h1 : second = 30) : ∃ (first third : ℕ),
  first = 2 * second ∧ 
  third = first / 3 ∧ 
  first + second + third = 110 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2143_214322


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2143_214310

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2143_214310


namespace NUMINAMATH_CALUDE_ellipse_left_right_vertices_l2143_214317

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2/16 + y^2/7 = 1

-- Define the left and right vertices
def left_right_vertices : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Theorem statement
theorem ellipse_left_right_vertices :
  ∀ (p : ℝ × ℝ), p ∈ left_right_vertices ↔ 
    (ellipse_equation p.1 p.2 ∧ 
     ∀ (q : ℝ × ℝ), ellipse_equation q.1 q.2 → abs q.1 ≤ abs p.1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_left_right_vertices_l2143_214317


namespace NUMINAMATH_CALUDE_fraction_simplification_l2143_214365

-- Define the statement
theorem fraction_simplification : (36 ^ 40) / (72 ^ 20) = 18 ^ 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2143_214365


namespace NUMINAMATH_CALUDE_nabla_calculation_l2143_214311

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l2143_214311


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2143_214304

/-- A regular dodecahedron -/
structure Dodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  valid_edge : ∀ e ∈ edges, e.1 ≠ e.2
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of randomly selecting two vertices that are endpoints of an edge in a regular dodecahedron -/
def edge_probability (d : Dodecahedron) : ℚ :=
  d.edges.card / Nat.choose 20 2

/-- Theorem: The probability of randomly selecting two vertices that are endpoints of an edge
    in a regular dodecahedron with 20 vertices is 3/19 -/
theorem dodecahedron_edge_probability (d : Dodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry


end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2143_214304


namespace NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_36_seconds_l2143_214381

/-- Time for a train to pass a jogger --/
theorem train_passing_jogger (jogger_speed : Real) (train_speed : Real) 
  (train_length : Real) (initial_distance : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The train passes the jogger in 36 seconds --/
theorem train_passes_jogger_in_36_seconds : 
  train_passing_jogger 9 45 120 240 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_36_seconds_l2143_214381


namespace NUMINAMATH_CALUDE_one_third_of_seven_point_two_l2143_214301

theorem one_third_of_seven_point_two :
  (7.2 : ℚ) / 3 = 2 + 2 / 5 := by sorry

end NUMINAMATH_CALUDE_one_third_of_seven_point_two_l2143_214301


namespace NUMINAMATH_CALUDE_range_of_negative_power_function_l2143_214391

open Set
open Function
open Real

theorem range_of_negative_power_function {m : ℝ} (hm : m < 0) :
  let g : ℝ → ℝ := fun x ↦ x ^ m
  range (g ∘ (fun x ↦ x) : Set.Ioo 0 1 → ℝ) = Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_power_function_l2143_214391


namespace NUMINAMATH_CALUDE_weight_of_A_l2143_214325

def avg_weight_ABC : ℝ := 60
def avg_weight_ABCD : ℝ := 65
def avg_weight_BCDE : ℝ := 64
def weight_difference_E_D : ℝ := 3

theorem weight_of_A (weight_A weight_B weight_C weight_D weight_E : ℝ) : 
  (weight_A + weight_B + weight_C) / 3 = avg_weight_ABC ∧
  (weight_A + weight_B + weight_C + weight_D) / 4 = avg_weight_ABCD ∧
  weight_E = weight_D + weight_difference_E_D ∧
  (weight_B + weight_C + weight_D + weight_E) / 4 = avg_weight_BCDE →
  weight_A = 87 := by
sorry

end NUMINAMATH_CALUDE_weight_of_A_l2143_214325


namespace NUMINAMATH_CALUDE_sqrt_inequality_not_arithmetic_sequence_l2143_214319

-- Statement 1
theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by sorry

-- Statement 2
theorem not_arithmetic_sequence : 
  ¬ ∃ (d k : ℝ), (k = 1 ∧ k + d = Real.sqrt 2 ∧ k + 2*d = 3) := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_not_arithmetic_sequence_l2143_214319


namespace NUMINAMATH_CALUDE_line_perp_to_parallel_planes_l2143_214377

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_to_parallel_planes
  (m : Line) (α β : Plane)
  (h1 : perpendicular m β)
  (h2 : parallel α β) :
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_parallel_planes_l2143_214377


namespace NUMINAMATH_CALUDE_center_octahedron_volume_ratio_l2143_214326

/-- A regular octahedron -/
structure RegularOctahedron where
  -- We don't need to define the structure fully, just declare it exists
  mk :: (dummy : Unit)

/-- The octahedron formed by the centers of faces of a regular octahedron -/
def center_octahedron (o : RegularOctahedron) : RegularOctahedron :=
  RegularOctahedron.mk ()

/-- The volume of an octahedron -/
def volume (o : RegularOctahedron) : ℝ :=
  sorry

/-- The theorem stating the volume ratio of the center octahedron to the original octahedron -/
theorem center_octahedron_volume_ratio (o : RegularOctahedron) :
  volume (center_octahedron o) / volume o = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_center_octahedron_volume_ratio_l2143_214326


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2143_214386

theorem negation_of_universal_proposition (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (¬ ∀ x : ℝ, a^x > 0) ↔ (∃ x₀ : ℝ, a^x₀ ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2143_214386


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2143_214366

theorem arithmetic_sequence_first_term
  (a d : ℚ)  -- First term and common difference
  (sum_60 : ℚ → ℚ → ℕ → ℚ)  -- Function to calculate sum of n terms
  (h1 : sum_60 a d 60 = 240)  -- Sum of first 60 terms
  (h2 : sum_60 (a + 60 * d) d 60 = 3240)  -- Sum of next 60 terms
  : a = -247 / 12 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2143_214366


namespace NUMINAMATH_CALUDE_jacob_guarantee_sheep_l2143_214385

/-- The maximum square number in the list -/
def max_square : Nat := 2021^2

/-- The list of square numbers from 1^2 to 2021^2 -/
def square_list : List Nat := List.range 2021 |>.map (λ x => (x + 1)^2)

/-- The game state, including the current sum on the whiteboard and the remaining numbers -/
structure GameState where
  sum : Nat
  remaining : List Nat

/-- A player's strategy for choosing a number from the list -/
def Strategy := GameState → Nat

/-- The result of playing the game, counting the number of times the sum is divisible by 4 after Jacob's turn -/
def play_game (jacob_strategy : Strategy) (laban_strategy : Strategy) : Nat :=
  sorry

/-- The theorem stating that Jacob can guarantee at least 506 sheep -/
theorem jacob_guarantee_sheep :
  ∃ (jacob_strategy : Strategy),
    ∀ (laban_strategy : Strategy),
      play_game jacob_strategy laban_strategy ≥ 506 := by
  sorry

end NUMINAMATH_CALUDE_jacob_guarantee_sheep_l2143_214385


namespace NUMINAMATH_CALUDE_election_winner_votes_l2143_214371

theorem election_winner_votes (total_votes : ℕ) : 
  (total_votes : ℝ) * 0.62 - (total_votes : ℝ) * 0.38 = 348 →
  (total_votes : ℝ) * 0.62 = 899 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l2143_214371


namespace NUMINAMATH_CALUDE_cone_base_diameter_l2143_214308

theorem cone_base_diameter (r : ℝ) (h1 : r > 0) : 
  (π * r * (2 * r) + π * r^2 = 3 * π) → (2 * r = 2) := by
  sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l2143_214308


namespace NUMINAMATH_CALUDE_weight_changes_result_l2143_214355

/-- Calculates the final weight after a series of weight changes. -/
def finalWeight (initialWeight : ℕ) (initialLoss : ℕ) : ℕ :=
  let weightAfterFirstLoss := initialWeight - initialLoss
  let weightAfterSecondGain := weightAfterFirstLoss + 2 * initialLoss
  let weightAfterThirdLoss := weightAfterSecondGain - 3 * initialLoss
  weightAfterThirdLoss + 6 / 2

/-- Theorem stating that given the specific weight changes, the final weight is 78 pounds. -/
theorem weight_changes_result :
  finalWeight 99 12 = 78 := by sorry

end NUMINAMATH_CALUDE_weight_changes_result_l2143_214355


namespace NUMINAMATH_CALUDE_thabo_book_difference_l2143_214330

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- The conditions of Thabo's book collection -/
def thabosBooks (b : BookCollection) : Prop :=
  b.paperbackFiction + b.paperbackNonfiction + b.hardcoverNonfiction = 200 ∧
  b.paperbackNonfiction > b.hardcoverNonfiction ∧
  b.paperbackFiction = 2 * b.paperbackNonfiction ∧
  b.hardcoverNonfiction = 35

theorem thabo_book_difference (b : BookCollection) 
  (h : thabosBooks b) : 
  b.paperbackNonfiction - b.hardcoverNonfiction = 20 := by
  sorry


end NUMINAMATH_CALUDE_thabo_book_difference_l2143_214330


namespace NUMINAMATH_CALUDE_janet_stuffies_l2143_214338

theorem janet_stuffies (x : ℚ) : 
  let total := x
  let kept := (3 / 7) * total
  let distributed := total - kept
  let ratio_sum := 3 + 4 + 2 + 1 + 5
  let janet_part := 1
  (janet_part / ratio_sum) * distributed = (4 * x) / 105 := by
sorry

end NUMINAMATH_CALUDE_janet_stuffies_l2143_214338


namespace NUMINAMATH_CALUDE_sin_negative_1290_degrees_l2143_214321

theorem sin_negative_1290_degrees (θ : ℝ) :
  (∀ k : ℤ, Real.sin (θ + k * (2 * π)) = Real.sin θ) →
  (∀ θ : ℝ, Real.sin (π - θ) = Real.sin θ) →
  Real.sin (π / 6) = 1 / 2 →
  Real.sin (-1290 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1290_degrees_l2143_214321


namespace NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l2143_214343

theorem half_plus_five_equals_fifteen (n : ℕ) (value : ℕ) : n = 20 → n / 2 + 5 = value → value = 15 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l2143_214343


namespace NUMINAMATH_CALUDE_sum_of_square_roots_lower_bound_l2143_214346

theorem sum_of_square_roots_lower_bound
  (a b c d e : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) +
  Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) +
  Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_lower_bound_l2143_214346


namespace NUMINAMATH_CALUDE_cube_root_and_square_roots_l2143_214361

theorem cube_root_and_square_roots (a b m : ℝ) : 
  (3 * a - 5)^(1/3) = -2 ∧ 
  m^2 = b ∧ 
  (1 - 5*m)^2 = b →
  a = -1 ∧ b = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_and_square_roots_l2143_214361


namespace NUMINAMATH_CALUDE_f_properties_l2143_214335

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x - 2

theorem f_properties :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 3 ∧ y > 3)) → f x > f y) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≥ -7) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = -7) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2143_214335


namespace NUMINAMATH_CALUDE_unique_intersection_and_geometric_progression_l2143_214339

noncomputable section

def f (x : ℝ) : ℝ := x / Real.exp x
def g (x : ℝ) : ℝ := Real.log x / x

theorem unique_intersection_and_geometric_progression :
  (∃! x : ℝ, f x = g x) ∧
  (∀ a : ℝ, 0 < a → a < Real.exp (-1) →
    (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
      f x₁ = a ∧ g x₂ = a ∧ f x₃ = a →
      ∃ r : ℝ, x₂ = x₁ * r ∧ x₃ = x₂ * r)) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_and_geometric_progression_l2143_214339


namespace NUMINAMATH_CALUDE_solve_equation_l2143_214300

theorem solve_equation : 42 / (7 - 3/7) = 147/23 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2143_214300


namespace NUMINAMATH_CALUDE_box_product_digits_l2143_214336

def box_product (n : ℕ) : ℕ := n * 100 + 28 * 4

theorem box_product_digits :
  (∀ n : ℕ, n ≤ 2 → box_product n < 1000) ∧
  (∀ n : ℕ, n ≥ 3 → box_product n ≥ 1000) :=
by sorry

end NUMINAMATH_CALUDE_box_product_digits_l2143_214336
