import Mathlib

namespace exists_three_numbers_sum_geq_54_l2234_223470

theorem exists_three_numbers_sum_geq_54 
  (S : Finset ℕ) 
  (distinct : S.card = 10) 
  (sum_gt_144 : S.sum id > 144) : 
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c ≥ 54 :=
by sorry

end exists_three_numbers_sum_geq_54_l2234_223470


namespace additional_volunteers_needed_l2234_223411

def volunteers_needed : ℕ := 50
def math_classes : ℕ := 6
def students_per_class : ℕ := 5
def teachers_volunteered : ℕ := 13

theorem additional_volunteers_needed :
  volunteers_needed - (math_classes * students_per_class + teachers_volunteered) = 7 := by
  sorry

end additional_volunteers_needed_l2234_223411


namespace hiking_rate_up_l2234_223475

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  rate_up : ℝ
  days_up : ℝ
  route_down_length : ℝ
  rate_down_multiplier : ℝ

/-- The hiking scenario satisfies the given conditions -/
def satisfies_conditions (h : HikingScenario) : Prop :=
  h.days_up = 2 ∧ 
  h.route_down_length = 15 ∧
  h.rate_down_multiplier = 1.5 ∧
  h.rate_up * h.days_up = h.route_down_length / h.rate_down_multiplier

/-- Theorem stating that the rate up the mountain is 5 miles per day -/
theorem hiking_rate_up (h : HikingScenario) 
  (hc : satisfies_conditions h) : h.rate_up = 5 := by
  sorry

#check hiking_rate_up

end hiking_rate_up_l2234_223475


namespace coin_flip_expected_earnings_l2234_223441

/-- Represents the possible outcomes of the coin flip -/
inductive CoinOutcome
| A
| B
| C
| Disappear

/-- The probability of each outcome -/
def probability (outcome : CoinOutcome) : ℚ :=
  match outcome with
  | CoinOutcome.A => 1/4
  | CoinOutcome.B => 1/4
  | CoinOutcome.C => 1/3
  | CoinOutcome.Disappear => 1/6

/-- The payout for each outcome -/
def payout (outcome : CoinOutcome) : ℤ :=
  match outcome with
  | CoinOutcome.A => 2
  | CoinOutcome.B => -1
  | CoinOutcome.C => 4
  | CoinOutcome.Disappear => -3

/-- The expected earnings from flipping the coin -/
def expected_earnings : ℚ :=
  (probability CoinOutcome.A * payout CoinOutcome.A) +
  (probability CoinOutcome.B * payout CoinOutcome.B) +
  (probability CoinOutcome.C * payout CoinOutcome.C) +
  (probability CoinOutcome.Disappear * payout CoinOutcome.Disappear)

theorem coin_flip_expected_earnings :
  expected_earnings = 13/12 := by
  sorry

end coin_flip_expected_earnings_l2234_223441


namespace least_common_multiple_first_ten_l2234_223426

def first_ten_integers : Finset ℕ := Finset.range 10

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ i ∈ first_ten_integers, i ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ i ∈ first_ten_integers, i ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end least_common_multiple_first_ten_l2234_223426


namespace solution_set_inequality_l2234_223401

theorem solution_set_inequality (x : ℝ) :
  x * (2 - x) ≤ 0 ↔ x ≤ 0 ∨ x ≥ 2 := by sorry

end solution_set_inequality_l2234_223401


namespace cos_alpha_plus_20_eq_neg_alpha_l2234_223495

theorem cos_alpha_plus_20_eq_neg_alpha (α : ℝ) (h : Real.sin (α - 70 * Real.pi / 180) = α) :
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end cos_alpha_plus_20_eq_neg_alpha_l2234_223495


namespace function_range_l2234_223486

/-- Given a function f(x) = x + 4a/x - a, where a < 0, 
    if f(x) < 0 for all x in (0, 1], then a ≤ -1/3 -/
theorem function_range (a : ℝ) (h1 : a < 0) :
  (∀ x ∈ Set.Ioo 0 1, x + 4 * a / x - a < 0) →
  a ≤ -1/3 := by
  sorry

end function_range_l2234_223486


namespace average_of_six_numbers_l2234_223456

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.85)
  (h3 : (e + f) / 2 = 4.45) :
  (a + b + c + d + e + f) / 6 = 3.9 := by
  sorry

end average_of_six_numbers_l2234_223456


namespace no_positive_integer_solution_l2234_223466

theorem no_positive_integer_solution :
  ¬ ∃ (a b : ℕ+), 4 * (a^2 + a) = b^2 + b := by
  sorry

end no_positive_integer_solution_l2234_223466


namespace sum_to_60_l2234_223496

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of integers from 1 to 60 is equal to 1830 -/
theorem sum_to_60 : sum_to_n 60 = 1830 := by
  sorry

end sum_to_60_l2234_223496


namespace peter_class_size_l2234_223436

/-- The number of students in Peter's class -/
def students_in_class : ℕ := 11

/-- The number of hands in the class, excluding Peter's -/
def hands_without_peter : ℕ := 20

/-- The number of hands each student has -/
def hands_per_student : ℕ := 2

/-- Theorem: The number of students in Peter's class is 11 -/
theorem peter_class_size :
  students_in_class = hands_without_peter / hands_per_student + 1 :=
by sorry

end peter_class_size_l2234_223436


namespace bella_stamps_l2234_223467

theorem bella_stamps (snowflake : ℕ) (truck : ℕ) (rose : ℕ) (butterfly : ℕ) 
  (h1 : snowflake = 15)
  (h2 : truck = snowflake + 11)
  (h3 : rose = truck - 17)
  (h4 : butterfly = 2 * rose) :
  snowflake + truck + rose + butterfly = 68 := by
  sorry

end bella_stamps_l2234_223467


namespace inequality_proof_l2234_223481

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (a + 2*b + c)^2 / (2*b^2 + (c + a)^2) +
  (a + b + 2*c)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
sorry

end inequality_proof_l2234_223481


namespace friday_five_times_in_june_l2234_223427

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Represents a month with its dates -/
structure Month :=
  (dates : List Date)
  (numDays : Nat)

def May : Month := sorry
def June : Month := sorry

/-- Counts the occurrences of a specific day of the week in a month -/
def countDayInMonth (d : DayOfWeek) (m : Month) : Nat := sorry

/-- Checks if a month has exactly five occurrences of a specific day of the week -/
def hasFiveOccurrences (d : DayOfWeek) (m : Month) : Prop :=
  countDayInMonth d m = 5

theorem friday_five_times_in_june 
  (h1 : hasFiveOccurrences DayOfWeek.Tuesday May)
  (h2 : May.numDays = 31)
  (h3 : June.numDays = 31) :
  hasFiveOccurrences DayOfWeek.Friday June := by
  sorry

end friday_five_times_in_june_l2234_223427


namespace john_jury_duty_days_l2234_223452

def jury_duty_days (jury_selection_days : ℕ) 
                   (trial_duration_multiplier : ℕ) 
                   (trial_extra_hours_per_day : ℕ) 
                   (deliberation_equivalent_full_days : ℕ) 
                   (deliberation_hours_per_day : ℕ) : ℕ :=
  let trial_days := jury_selection_days * trial_duration_multiplier
  let trial_extra_days := (trial_days * trial_extra_hours_per_day) / 24
  let deliberation_days := 
    (deliberation_equivalent_full_days * 24 + deliberation_hours_per_day - 1) / deliberation_hours_per_day
  jury_selection_days + trial_days + trial_extra_days + deliberation_days

theorem john_jury_duty_days : 
  jury_duty_days 2 4 3 6 14 = 22 := by sorry

end john_jury_duty_days_l2234_223452


namespace line_parameterization_l2234_223420

/-- Given a line y = 5x - 7 parameterized by [x, y] = [p, 3] + t[3, q], 
    prove that p = 2 and q = 15 -/
theorem line_parameterization (x y p q t : ℝ) : 
  (y = 5*x - 7) ∧ 
  (∃ t, x = p + 3*t ∧ y = 3 + q*t) →
  p = 2 ∧ q = 15 := by
sorry

end line_parameterization_l2234_223420


namespace percentage_ratio_proof_l2234_223469

theorem percentage_ratio_proof (P Q M N R : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.3 * P)
  (hN : N = 0.6 * P)
  (hR : R = 0.2 * P)
  (hP : P ≠ 0) : (M + R) / N = 8 / 15 := by
  sorry

end percentage_ratio_proof_l2234_223469


namespace divisibility_of_f_l2234_223450

def f (x : ℕ) : ℕ := x^3 + 17

theorem divisibility_of_f (n : ℕ) (hn : n ≥ 2) :
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬(3^(n+1) ∣ f x) := by
  sorry

end divisibility_of_f_l2234_223450


namespace roots_magnitude_l2234_223473

theorem roots_magnitude (q : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + q*r₁ - 10 = 0 → 
  r₂^2 + q*r₂ - 10 = 0 → 
  (|r₁| > 4 ∨ |r₂| > 4) :=
by
  sorry

end roots_magnitude_l2234_223473


namespace susan_board_game_l2234_223488

theorem susan_board_game (total_spaces : ℕ) (first_move : ℕ) (second_move : ℕ) (third_move : ℕ) (spaces_to_win : ℕ) :
  total_spaces = 48 →
  first_move = 8 →
  second_move = 2 →
  third_move = 6 →
  spaces_to_win = 37 →
  ∃ (spaces_moved_back : ℕ),
    first_move + second_move + third_move - spaces_moved_back = total_spaces - spaces_to_win ∧
    spaces_moved_back = 6 :=
by sorry

end susan_board_game_l2234_223488


namespace solve_for_a_l2234_223459

theorem solve_for_a (x a : ℝ) (h : 2 * x - a = -5) (hx : x = 5) : a = 15 := by
  sorry

end solve_for_a_l2234_223459


namespace zack_group_size_l2234_223461

/-- Proves that Zack tutors students in groups of 10, given the problem conditions -/
theorem zack_group_size :
  ∀ (x : ℕ),
  (∃ (n : ℕ), x * n = 70) →  -- Zack tutors 70 students in total
  (∃ (m : ℕ), 10 * m = 70) →  -- Karen tutors 70 students in total
  x = 10 := by sorry

end zack_group_size_l2234_223461


namespace sqrt_sum_eq_sqrt_prime_l2234_223440

theorem sqrt_sum_eq_sqrt_prime (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ, Real.sqrt x + Real.sqrt y = Real.sqrt p ↔ (x = 0 ∧ y = p) ∨ (x = p ∧ y = 0) :=
by sorry

end sqrt_sum_eq_sqrt_prime_l2234_223440


namespace unique_solution_exists_l2234_223410

theorem unique_solution_exists (m n : ℕ) : 
  (∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + m * b = n ∧ a + b = m * c) ↔ 
  (m > 1 ∧ (n - 1) % (m - 1) = 0 ∧ ¬∃k, n = m ^ k) := by
sorry

end unique_solution_exists_l2234_223410


namespace sweater_shirt_price_difference_l2234_223435

theorem sweater_shirt_price_difference : 
  let shirt_total : ℕ := 360
  let shirt_count : ℕ := 20
  let sweater_total : ℕ := 900
  let sweater_count : ℕ := 45
  let shirt_avg : ℚ := shirt_total / shirt_count
  let sweater_avg : ℚ := sweater_total / sweater_count
  sweater_avg - shirt_avg = 2 := by
sorry

end sweater_shirt_price_difference_l2234_223435


namespace no_periodic_sum_with_periods_one_and_pi_l2234_223437

/-- A function is periodic if it takes at least two different values and there exists a positive real number p such that f(x + p) = f(x) for all x. -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ ∃ p > 0, ∀ x, f (x + p) = f x

/-- p is a period of f if f(x + p) = f(x) for all x. -/
def IsPeriodOf (p : ℝ) (f : ℝ → ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

theorem no_periodic_sum_with_periods_one_and_pi :
  ¬ ∃ (g h : ℝ → ℝ),
    IsPeriodic g ∧ IsPeriodic h ∧
    IsPeriodOf 1 g ∧ IsPeriodOf Real.pi h ∧
    IsPeriodic (g + h) :=
sorry

end no_periodic_sum_with_periods_one_and_pi_l2234_223437


namespace quadratic_real_roots_condition_l2234_223433

/-- If the quadratic equation x^2 - 3x + 2m = 0 has real roots, then m ≤ 9/8 -/
theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9/8 := by
  sorry

end quadratic_real_roots_condition_l2234_223433


namespace arithmetic_sequence_property_l2234_223414

/-- An arithmetic sequence with the given properties has the general term a_n = n -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : 
  d ≠ 0 ∧  -- non-zero common difference
  (∀ n, a (n + 1) = a n + d) ∧  -- arithmetic sequence property
  a 1 = 1 ∧  -- first term is 1
  (a 3)^2 = a 1 * a 9  -- geometric sequence property for a_1, a_3, a_9
  →
  ∀ n, a n = n := by
sorry

end arithmetic_sequence_property_l2234_223414


namespace cube_sum_l2234_223421

/-- The number of faces in a cube -/
def cube_faces : ℕ := 6

/-- The number of edges in a cube -/
def cube_edges : ℕ := 12

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The sum of faces, edges, and vertices in a cube is 26 -/
theorem cube_sum : cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end cube_sum_l2234_223421


namespace complex_power_difference_l2234_223438

theorem complex_power_difference (x : ℂ) (h : x - 1/x = 2*I) : x^8 - 1/x^8 = 2 := by
  sorry

end complex_power_difference_l2234_223438


namespace only_j_has_inverse_l2234_223463

-- Define the types for our functions
def Function : Type := ℝ → ℝ

-- Define properties for each function
def is_parabola_upward (f : Function) : Prop := sorry

def is_discontinuous_two_segments (f : Function) : Prop := sorry

def is_horizontal_line (f : Function) : Prop := sorry

def is_sine_function (f : Function) : Prop := sorry

def is_linear_positive_slope (f : Function) : Prop := sorry

-- Define what it means for a function to have an inverse
def has_inverse (f : Function) : Prop := sorry

-- State the theorem
theorem only_j_has_inverse 
  (F G H I J : Function)
  (hF : is_parabola_upward F)
  (hG : is_discontinuous_two_segments G)
  (hH : is_horizontal_line H)
  (hI : is_sine_function I)
  (hJ : is_linear_positive_slope J) :
  (¬ has_inverse F) ∧ 
  (¬ has_inverse G) ∧ 
  (¬ has_inverse H) ∧ 
  (¬ has_inverse I) ∧ 
  has_inverse J :=
sorry

end only_j_has_inverse_l2234_223463


namespace fraction_value_l2234_223498

theorem fraction_value (p q : ℚ) (x : ℚ) 
  (h1 : p / q = 4 / 5)
  (h2 : x + (2 * q - p) / (2 * q + p) = 4) :
  x = 25 / 7 := by
sorry

end fraction_value_l2234_223498


namespace acme_vowel_soup_words_l2234_223429

/-- Represents the count of each vowel in the alphabet soup -/
structure VowelCount where
  a : Nat
  e : Nat
  i : Nat
  o : Nat
  u : Nat

/-- The modified Acme alphabet soup recipe -/
def acmeVowelSoup : VowelCount :=
  { a := 4, e := 6, i := 5, o := 3, u := 2 }

/-- The length of words to be formed -/
def wordLength : Nat := 5

/-- Calculates the number of five-letter words that can be formed from the given vowel counts -/
def countWords (vc : VowelCount) (len : Nat) : Nat :=
  sorry

/-- Theorem stating that the number of five-letter words from Acme Vowel Soup is 1125 -/
theorem acme_vowel_soup_words :
  countWords acmeVowelSoup wordLength = 1125 := by
  sorry

end acme_vowel_soup_words_l2234_223429


namespace martha_improvement_l2234_223484

/-- Represents Martha's running performance at a given time --/
structure Performance where
  laps : ℕ
  time : ℕ
  
/-- Calculates the lap time in seconds given a Performance --/
def lapTime (p : Performance) : ℚ :=
  (p.time * 60) / p.laps

/-- Martha's initial performance --/
def initialPerformance : Performance := ⟨15, 30⟩

/-- Martha's performance after two months --/
def finalPerformance : Performance := ⟨20, 27⟩

/-- Theorem stating the improvement in Martha's lap time --/
theorem martha_improvement :
  lapTime initialPerformance - lapTime finalPerformance = 39 := by
  sorry

end martha_improvement_l2234_223484


namespace wedge_volume_l2234_223477

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (angle : ℝ) : 
  d = 16 →
  angle = 60 →
  (π * (d / 2)^2 * d) / 2 = 512 * π :=
by sorry

end wedge_volume_l2234_223477


namespace polygon_sides_count_l2234_223409

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ

/-- Represents the number of triangles formed by connecting a point on a side to all vertices. -/
def triangles_formed (p : Polygon) : ℕ := p.sides - 1

/-- The polygon in our problem. -/
def our_polygon : Polygon :=
  { sides := 2024 }

/-- The theorem stating our problem. -/
theorem polygon_sides_count : triangles_formed our_polygon = 2023 := by
  sorry

end polygon_sides_count_l2234_223409


namespace bobs_head_start_l2234_223497

/-- Proves that Bob's head-start is 1 mile given the conditions -/
theorem bobs_head_start (bob_speed jim_speed : ℝ) (catch_time : ℝ) (head_start : ℝ) : 
  bob_speed = 6 → 
  jim_speed = 9 → 
  catch_time = 20 / 60 →
  head_start + bob_speed * catch_time = jim_speed * catch_time →
  head_start = 1 := by
sorry

end bobs_head_start_l2234_223497


namespace retirement_ratio_is_one_to_one_l2234_223416

def monthly_income : ℚ := 2500
def rent : ℚ := 700
def car_payment : ℚ := 300
def groceries : ℚ := 50
def remaining_after_retirement : ℚ := 650

def total_expenses : ℚ := rent + car_payment + (car_payment / 2) + groceries

def money_after_expenses : ℚ := monthly_income - total_expenses

def retirement_contribution : ℚ := money_after_expenses - remaining_after_retirement

theorem retirement_ratio_is_one_to_one :
  retirement_contribution = remaining_after_retirement :=
by sorry

end retirement_ratio_is_one_to_one_l2234_223416


namespace max_acute_angles_eq_three_l2234_223487

/-- A convex polygon with n sides, where n ≥ 3 --/
structure ConvexPolygon where
  n : ℕ
  n_ge_three : n ≥ 3

/-- The maximum number of acute angles in a convex polygon --/
def max_acute_angles (p : ConvexPolygon) : ℕ := 3

/-- Theorem: The maximum number of acute angles in a convex polygon is 3 --/
theorem max_acute_angles_eq_three (p : ConvexPolygon) :
  max_acute_angles p = 3 := by sorry

end max_acute_angles_eq_three_l2234_223487


namespace absolute_value_inequality_solution_set_l2234_223465

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 1} = Set.Ioo 0 2 := by sorry

end absolute_value_inequality_solution_set_l2234_223465


namespace sqrt_expression_equals_one_fifth_l2234_223402

theorem sqrt_expression_equals_one_fifth :
  (Real.sqrt 3 + Real.sqrt 2) ^ (2 * (Real.log (Real.sqrt 5) / Real.log (Real.sqrt 3 - Real.sqrt 2))) = 1 / 5 := by
  sorry

end sqrt_expression_equals_one_fifth_l2234_223402


namespace expected_ones_is_half_l2234_223447

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ :=
  0 * (prob_not_one ^ num_dice) +
  1 * (num_dice * prob_one * prob_not_one ^ (num_dice - 1)) +
  2 * (num_dice * (num_dice - 1) / 2 * prob_one ^ 2 * prob_not_one) +
  3 * (prob_one ^ num_dice)

theorem expected_ones_is_half : expected_ones = 1 / 2 := by
  sorry

end expected_ones_is_half_l2234_223447


namespace train_length_l2234_223418

/-- The length of a train passing a bridge -/
theorem train_length (v : ℝ) (t : ℝ) (b : ℝ) : v = 72 * 1000 / 3600 → t = 25 → b = 140 → v * t - b = 360 := by
  sorry

end train_length_l2234_223418


namespace roots_quadratic_equation_l2234_223492

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + 3*a - 4 = 0) → (b^2 + 3*b - 4 = 0) → (a^2 + 4*a + b - 3 = -2) := by
  sorry

end roots_quadratic_equation_l2234_223492


namespace m_range_l2234_223445

def p (m : ℝ) : Prop := ∀ x, |x - m| + |x - 1| > 1

def q (m : ℝ) : Prop := ∀ x > 0, (fun x => Real.log x / Real.log (3 + m)) x > 0

theorem m_range : 
  (∃ m : ℝ, (¬(p m ∧ q m)) ∧ (p m ∨ q m)) ↔ 
  (∃ m : ℝ, (-3 < m ∧ m < -2) ∨ (0 ≤ m ∧ m ≤ 2)) :=
sorry

end m_range_l2234_223445


namespace three_fourths_of_hundred_l2234_223464

theorem three_fourths_of_hundred : (3 / 4 : ℚ) * 100 = 75 := by sorry

end three_fourths_of_hundred_l2234_223464


namespace julias_change_julias_change_is_eight_l2234_223443

/-- Calculates Julia's change after purchasing Snickers and M&M's -/
theorem julias_change (snickers_price : ℝ) (snickers_quantity : ℕ) (mms_quantity : ℕ) 
  (payment : ℝ) : ℝ :=
  let mms_price := 2 * snickers_price
  let total_cost := snickers_price * snickers_quantity + mms_price * mms_quantity
  payment - total_cost

/-- Proves that Julia's change is $8 given the specific conditions -/
theorem julias_change_is_eight :
  julias_change 1.5 2 3 20 = 8 := by
  sorry

end julias_change_julias_change_is_eight_l2234_223443


namespace ellipse_sum_l2234_223478

theorem ellipse_sum (h k a b : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) →
  (h = 3 ∧ k = -5) →
  (a = 7 ∧ b = 4) →
  h + k + a + b = 9 := by
  sorry

end ellipse_sum_l2234_223478


namespace book_ratio_problem_l2234_223432

theorem book_ratio_problem (darla_books katie_books gary_books : ℕ) : 
  darla_books = 6 →
  gary_books = 5 * (darla_books + katie_books) →
  darla_books + katie_books + gary_books = 54 →
  katie_books = darla_books / 2 := by
  sorry

end book_ratio_problem_l2234_223432


namespace grid_square_covers_at_least_four_l2234_223419

/-- A square on a grid -/
structure GridSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The area of the square is four times the unit area -/
  area_is_four : side^2 = 4

/-- The minimum number of grid points covered by a grid square -/
def min_covered_points (s : GridSquare) : ℕ := 4

/-- Theorem: A GridSquare covers at least 4 grid points -/
theorem grid_square_covers_at_least_four (s : GridSquare) :
  ∃ (n : ℕ), n ≥ 4 ∧ n = min_covered_points s :=
sorry

end grid_square_covers_at_least_four_l2234_223419


namespace total_subjects_l2234_223444

theorem total_subjects (avg_all : ℝ) (avg_first_five : ℝ) (last_subject : ℝ) 
  (h1 : avg_all = 78)
  (h2 : avg_first_five = 74)
  (h3 : last_subject = 98) :
  ∃ n : ℕ, n = 6 ∧ 
    n * avg_all = (n - 1) * avg_first_five + last_subject :=
by sorry

end total_subjects_l2234_223444


namespace ratio_problem_l2234_223494

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end ratio_problem_l2234_223494


namespace parabolas_coincide_l2234_223434

/-- Represents a parabola with leading coefficient 1 -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  b : ℝ

/-- Returns the length of the segment intercepted by a line on a parabola -/
noncomputable def interceptLength (para : Parabola) (l : Line) : ℝ :=
  Real.sqrt ((para.p - l.k)^2 - 4*(para.q - l.b))

/-- Two lines are non-parallel if their slopes are different -/
def nonParallel (l₁ l₂ : Line) : Prop :=
  l₁.k ≠ l₂.k

theorem parabolas_coincide
  (Γ₁ Γ₂ : Parabola)
  (l₁ l₂ : Line)
  (h_nonparallel : nonParallel l₁ l₂)
  (h_equal_length₁ : interceptLength Γ₁ l₁ = interceptLength Γ₂ l₁)
  (h_equal_length₂ : interceptLength Γ₁ l₂ = interceptLength Γ₂ l₂) :
  Γ₁ = Γ₂ := by
  sorry

end parabolas_coincide_l2234_223434


namespace women_count_at_gathering_l2234_223400

/-- Represents a social gathering where men and women dance. -/
structure SocialGathering where
  men : ℕ
  women : ℕ
  manDances : ℕ
  womanDances : ℕ

/-- The number of women at the gathering is correct if the total number of dances
    from men's perspective equals the total number of dances from women's perspective. -/
def isCorrectWomenCount (g : SocialGathering) : Prop :=
  g.men * g.manDances = g.women * g.womanDances

/-- Theorem stating that in a gathering with 15 men, where each man dances with 4 women
    and each woman dances with 3 men, there are 20 women. -/
theorem women_count_at_gathering :
  ∀ g : SocialGathering,
    g.men = 15 →
    g.manDances = 4 →
    g.womanDances = 3 →
    isCorrectWomenCount g →
    g.women = 20 := by
  sorry

end women_count_at_gathering_l2234_223400


namespace B_wins_4_probability_C_wins_3_probability_l2234_223480

-- Define the players
inductive Player : Type
| A : Player
| B : Player
| C : Player

-- Define the win probabilities
def winProb (winner loser : Player) : ℝ :=
  match winner, loser with
  | Player.A, Player.B => 0.4
  | Player.B, Player.C => 0.5
  | Player.C, Player.A => 0.6
  | _, _ => 0 -- For other combinations, set probability to 0

-- Define the probability of B winning 4 consecutive matches
def prob_B_wins_4 : ℝ :=
  (1 - winProb Player.A Player.B) * 
  (winProb Player.B Player.C) * 
  (1 - winProb Player.A Player.B) * 
  (winProb Player.B Player.C)

-- Define the probability of C winning 3 consecutive matches
def prob_C_wins_3 : ℝ :=
  ((1 - winProb Player.A Player.B) * (1 - winProb Player.B Player.C) * 
   (winProb Player.C Player.A) * (1 - winProb Player.B Player.C)) +
  ((winProb Player.A Player.B) * (winProb Player.C Player.A) * 
   (1 - winProb Player.B Player.C) * (winProb Player.C Player.A))

-- Theorem statements
theorem B_wins_4_probability : prob_B_wins_4 = 0.09 := by sorry

theorem C_wins_3_probability : prob_C_wins_3 = 0.162 := by sorry

end B_wins_4_probability_C_wins_3_probability_l2234_223480


namespace probability_rgb_draw_specific_l2234_223404

/-- The probability of drawing a red shoe first, a green shoe second, and a blue shoe third
    from a closet containing red, green, and blue shoes. -/
def probability_rgb_draw (red green blue : ℕ) : ℚ :=
  (red : ℚ) / (red + green + blue) *
  (green : ℚ) / (red + green + blue - 1) *
  (blue : ℚ) / (red + green + blue - 2)

/-- Theorem stating that the probability of drawing a red shoe first, a green shoe second,
    and a blue shoe third from a closet containing 5 red shoes, 4 green shoes, and 3 blue shoes
    is equal to 1/22. -/
theorem probability_rgb_draw_specific : probability_rgb_draw 5 4 3 = 1 / 22 := by
  sorry

end probability_rgb_draw_specific_l2234_223404


namespace union_covers_reals_l2234_223428

open Set Real

theorem union_covers_reals (a : ℝ) : 
  let S : Set ℝ := {x | |x - 2| > 3}
  let T : Set ℝ := {x | a < x ∧ x < a + 8}
  (S ∪ T = univ) → (-3 < a ∧ a < -1) :=
by
  sorry

end union_covers_reals_l2234_223428


namespace b_contribution_l2234_223451

def a_investment : ℕ := 3500
def a_months : ℕ := 12
def b_months : ℕ := 7
def a_share : ℕ := 2
def b_share : ℕ := 3

theorem b_contribution (x : ℕ) : 
  (a_investment * a_months) / (x * b_months) = a_share / b_share → 
  x = 9000 := by
  sorry

end b_contribution_l2234_223451


namespace distinct_shapes_count_is_31_l2234_223468

/-- Represents a convex-shaped paper made of four 1×1 squares -/
structure ConvexPaper :=
  (squares : Fin 4 → (Fin 1 × Fin 1))

/-- Represents a 5×6 grid paper -/
structure GridPaper :=
  (grid : Fin 5 → Fin 6 → Bool)

/-- Represents a placement of the convex paper on the grid paper -/
structure Placement :=
  (position : Fin 5 × Fin 6)
  (orientation : Fin 4)

/-- Checks if a placement is valid (all squares of convex paper overlap with grid squares) -/
def isValidPlacement (cp : ConvexPaper) (gp : GridPaper) (p : Placement) : Prop :=
  sorry

/-- Checks if two placements are rotationally equivalent -/
def areRotationallyEquivalent (p1 p2 : Placement) : Prop :=
  sorry

/-- The number of distinct shapes that can be formed -/
def distinctShapesCount (cp : ConvexPaper) (gp : GridPaper) : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct shapes is 31 -/
theorem distinct_shapes_count_is_31 (cp : ConvexPaper) (gp : GridPaper) :
  distinctShapesCount cp gp = 31 :=
  sorry

end distinct_shapes_count_is_31_l2234_223468


namespace quadratic_inequality_solution_sets_l2234_223408

theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
  sorry

end quadratic_inequality_solution_sets_l2234_223408


namespace fencing_cost_calculation_l2234_223442

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculates the total cost of fencing for a rectangular plot. -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot. -/
theorem fencing_cost_calculation (plot : RectangularPlot)
  (h1 : plot.length = 61)
  (h2 : plot.breadth = plot.length - 22)
  (h3 : plot.fencing_cost_per_meter = 26.50) :
  total_fencing_cost plot = 5300 := by
  sorry

#eval total_fencing_cost { length := 61, breadth := 39, fencing_cost_per_meter := 26.50 }

end fencing_cost_calculation_l2234_223442


namespace total_days_2005_to_2010_l2234_223407

def is_leap_year (year : ℕ) : Bool := year = 2008

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def year_range : List ℕ := [2005, 2006, 2007, 2008, 2009, 2010]

theorem total_days_2005_to_2010 :
  (year_range.map days_in_year).sum = 2191 := by
  sorry

end total_days_2005_to_2010_l2234_223407


namespace bird_migration_distance_l2234_223454

/-- Calculates the total distance traveled by migrating birds -/
theorem bird_migration_distance 
  (num_birds : ℕ) 
  (distance_jim_disney : ℝ) 
  (distance_disney_london : ℝ) : 
  num_birds = 20 → 
  distance_jim_disney = 50 → 
  distance_disney_london = 60 → 
  (num_birds : ℝ) * (distance_jim_disney + distance_disney_london) = 2200 := by
  sorry

end bird_migration_distance_l2234_223454


namespace min_value_theorem_l2234_223413

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * x^2 + 1 / x^2 ≥ 2 * Real.sqrt 3 ∧ ∃ y > 0, 3 * y^2 + 1 / y^2 = 2 * Real.sqrt 3 := by
  sorry

end min_value_theorem_l2234_223413


namespace helen_cookies_l2234_223462

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 435

/-- The number of cookies Helen baked this morning -/
def cookies_today : ℕ := 139

/-- The total number of cookies Helen baked -/
def total_cookies : ℕ := cookies_yesterday + cookies_today

/-- Theorem stating that the total number of cookies Helen baked is 574 -/
theorem helen_cookies : total_cookies = 574 := by sorry

end helen_cookies_l2234_223462


namespace simplify_expression_l2234_223483

theorem simplify_expression (x y : ℝ) : (x - y)^3 / (x - y)^2 * (y - x) = -(x - y)^2 := by
  sorry

end simplify_expression_l2234_223483


namespace arctan_sum_two_five_l2234_223423

theorem arctan_sum_two_five : Real.arctan (2/5) + Real.arctan (5/2) = π/2 := by
  sorry

end arctan_sum_two_five_l2234_223423


namespace min_value_f_over_x_range_of_a_l2234_223455

-- Part 1
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem min_value_f_over_x (x : ℝ) (hx : x > 0) :
  ∃ (y : ℝ), y = (f 2 x) / x ∧ ∀ (z : ℝ), z > 0 → (f 2 z) / z ≥ y ∧ y = -2 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ a) ↔ a ∈ Set.Ici (3/4) :=
sorry

end min_value_f_over_x_range_of_a_l2234_223455


namespace hilt_trip_distance_l2234_223491

/-- Calculates the distance traveled given initial and final odometer readings -/
def distance_traveled (initial_reading final_reading : ℝ) : ℝ :=
  final_reading - initial_reading

theorem hilt_trip_distance :
  let initial_reading : ℝ := 212.3
  let final_reading : ℝ := 372
  distance_traveled initial_reading final_reading = 159.7 := by
  sorry

end hilt_trip_distance_l2234_223491


namespace test_points_calculation_l2234_223476

theorem test_points_calculation (total_problems : ℕ) (computation_problems : ℕ) 
  (computation_points : ℕ) (word_points : ℕ) :
  total_problems = 30 →
  computation_problems = 20 →
  computation_points = 3 →
  word_points = 5 →
  (computation_problems * computation_points) + 
  ((total_problems - computation_problems) * word_points) = 110 := by
sorry

end test_points_calculation_l2234_223476


namespace inequality_proof_l2234_223490

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hab : a ≤ b) (hbc : b ≤ c) : 
  (a*x + b*y + c*z) * (x/a + y/b + z/c) ≤ (x+y+z)^2 * (a+c)^2 / (4*a*c) := by
  sorry

end inequality_proof_l2234_223490


namespace simplify_expression_l2234_223457

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 81) - Real.sqrt (8 + 1/2))^2 = (35 - 6 * Real.sqrt 34) / 2 := by
  sorry

end simplify_expression_l2234_223457


namespace greatest_root_of_g_l2234_223489

/-- The function g(x) as defined in the problem -/
def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

/-- Theorem stating that √21/7 is the greatest root of g(x) -/
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 21 / 7 ∧ g r = 0 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end greatest_root_of_g_l2234_223489


namespace set_operations_l2234_223471

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem set_operations :
  (Set.univ \ (A ∩ B) = {x | x < 3 ∨ x ≥ 7}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 7) ∨ x ≥ 10}) := by
  sorry

end set_operations_l2234_223471


namespace oplus_one_three_l2234_223479

def oplus (x y : ℤ) : ℤ := -3 * x + 4 * y

theorem oplus_one_three : oplus 1 3 = 9 := by sorry

end oplus_one_three_l2234_223479


namespace f_is_quadratic_l2234_223482

/-- Definition of a one-variable quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 2 * (x - x^2) - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l2234_223482


namespace unique_solution_for_equation_l2234_223499

theorem unique_solution_for_equation :
  ∃! (n k : ℕ), n > 0 ∧ k > 0 ∧ (n + 1)^n = 2 * n^k + 3 * n + 1 :=
by
  -- The proof goes here
  sorry

end unique_solution_for_equation_l2234_223499


namespace tree_distance_l2234_223430

/-- Given two buildings 220 meters apart with 10 trees planted at equal intervals,
    the distance between the 1st tree and the 6th tree is 100 meters. -/
theorem tree_distance (building_distance : ℝ) (num_trees : ℕ) 
  (h1 : building_distance = 220)
  (h2 : num_trees = 10) : 
  let interval := building_distance / (num_trees + 1)
  (6 - 1) * interval = 100 := by
  sorry

end tree_distance_l2234_223430


namespace grace_has_30_pastries_l2234_223415

/-- The number of pastries each person has -/
structure Pastries where
  frank : ℕ
  calvin : ℕ
  phoebe : ℕ
  grace : ℕ

/-- The conditions of the pastry problem -/
def pastry_conditions (p : Pastries) : Prop :=
  p.calvin = p.frank + 8 ∧
  p.phoebe = p.frank + 8 ∧
  p.grace = p.calvin + 5 ∧
  p.frank + p.calvin + p.phoebe + p.grace = 97

/-- The theorem stating that Grace has 30 pastries -/
theorem grace_has_30_pastries (p : Pastries) 
  (h : pastry_conditions p) : p.grace = 30 := by
  sorry


end grace_has_30_pastries_l2234_223415


namespace area_increase_bound_l2234_223453

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  -- Add necessary fields here
  isConvex : Bool

/-- The perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- The area of a polygon -/
def area (p : ConvexPolygon) : ℝ := sorry

/-- The polygon resulting from moving all sides of p outward by distance h -/
def expandedPolygon (p : ConvexPolygon) (h : ℝ) : ConvexPolygon := sorry

theorem area_increase_bound (p : ConvexPolygon) (h : ℝ) (h_pos : h > 0) :
  area (expandedPolygon p h) - area p > perimeter p * h + π * h^2 := by
  sorry

end area_increase_bound_l2234_223453


namespace a_range_for_region_above_l2234_223425

/-- The inequality represents the region above the line -/
def represents_region_above (a : ℝ) : Prop :=
  ∀ x y : ℝ, 3 * a * x + (a^2 - 3 * a + 2) * y - 9 < 0 ↔ 
    y > (9 - 3 * a * x) / (a^2 - 3 * a + 2)

/-- The theorem stating the range of a -/
theorem a_range_for_region_above : 
  ∀ a : ℝ, represents_region_above a ↔ 1 < a ∧ a < 2 := by sorry

end a_range_for_region_above_l2234_223425


namespace train_speed_l2234_223412

theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 160)
  (h2 : bridge_length = 215)
  (h3 : crossing_time = 30) : 
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_l2234_223412


namespace roots_sum_of_squares_l2234_223460

theorem roots_sum_of_squares (r s : ℝ) : 
  r^2 - 5*r + 3 = 0 → s^2 - 5*s + 3 = 0 → r^2 + s^2 = 19 := by
  sorry

end roots_sum_of_squares_l2234_223460


namespace cost_of_eight_books_l2234_223417

theorem cost_of_eight_books (cost_of_two : ℝ) (h : cost_of_two = 34) :
  8 * (cost_of_two / 2) = 136 := by
  sorry

end cost_of_eight_books_l2234_223417


namespace kitten_growth_l2234_223405

/-- The length of a kitten after doubling twice -/
def kittenLength (initialLength : ℕ) (doublings : ℕ) : ℕ :=
  initialLength * (2 ^ doublings)

/-- Theorem: A kitten with initial length 4 inches that doubles twice will be 16 inches long -/
theorem kitten_growth : kittenLength 4 2 = 16 := by
  sorry

end kitten_growth_l2234_223405


namespace min_value_T_l2234_223458

theorem min_value_T (p : ℝ) (h1 : 0 < p) (h2 : p < 15) :
  ∃ (min_T : ℝ), min_T = 15 ∧
  ∀ x : ℝ, p ≤ x → x ≤ 15 →
    |x - p| + |x - 15| + |x - (15 + p)| ≥ min_T :=
by
  sorry

end min_value_T_l2234_223458


namespace arccos_of_one_eq_zero_l2234_223472

theorem arccos_of_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_of_one_eq_zero_l2234_223472


namespace cs_consecutive_probability_l2234_223424

/-- The number of people sitting at the table -/
def total_people : ℕ := 12

/-- The number of computer scientists -/
def num_cs : ℕ := 5

/-- The number of chemistry majors -/
def num_chem : ℕ := 4

/-- The number of history majors -/
def num_hist : ℕ := 3

/-- The probability of all computer scientists sitting consecutively -/
def prob_cs_consecutive : ℚ := 1 / 66

theorem cs_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let consecutive_arrangements := Nat.factorial (num_cs) * Nat.factorial (total_people - num_cs)
  (consecutive_arrangements : ℚ) / total_arrangements = prob_cs_consecutive :=
sorry

end cs_consecutive_probability_l2234_223424


namespace baker_pastries_l2234_223446

theorem baker_pastries (cakes : ℕ) (pastry_difference : ℕ) : 
  cakes = 19 → pastry_difference = 112 → cakes + pastry_difference = 131 := by
  sorry

end baker_pastries_l2234_223446


namespace sin_cos_sum_10_20_l2234_223448

theorem sin_cos_sum_10_20 : 
  Real.sin (10 * π / 180) * Real.cos (20 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (20 * π / 180) = 1 / 2 := by
sorry

end sin_cos_sum_10_20_l2234_223448


namespace solution_characterization_l2234_223422

theorem solution_characterization (x y : ℤ) :
  x^2 - y^4 = 2009 ↔ (x = 45 ∧ y = 2) ∨ (x = 45 ∧ y = -2) ∨ (x = -45 ∧ y = 2) ∨ (x = -45 ∧ y = -2) :=
by sorry

end solution_characterization_l2234_223422


namespace intersection_M_N_l2234_223485

def M : Set ℤ := {-2, -1, 0, 1}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l2234_223485


namespace cubic_equation_roots_l2234_223431

theorem cubic_equation_roots (a b : ℝ) :
  (∃ x y z : ℕ+, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    (∀ t : ℝ, t^3 - 8*t^2 + a*t - b = 0 ↔ (t = x ∨ t = y ∨ t = z))) →
  a + b = 31 := by
sorry

end cubic_equation_roots_l2234_223431


namespace largest_number_comparison_l2234_223474

theorem largest_number_comparison :
  (1/2 : ℝ) > (37.5/100 : ℝ) ∧ (1/2 : ℝ) > (7/22 : ℝ) ∧ (1/2 : ℝ) > (π/10 : ℝ) := by
  sorry

end largest_number_comparison_l2234_223474


namespace desired_outcome_probability_l2234_223406

/-- Represents a die with a fixed number of sides --/
structure Die :=
  (sides : Nat)
  (values : Fin sides → Nat)

/-- Carla's die always shows 7 --/
def carla_die : Die :=
  { sides := 6,
    values := λ _ => 7 }

/-- Derek's die has numbers from 2 to 7 --/
def derek_die : Die :=
  { sides := 6,
    values := λ i => i.val + 2 }

/-- Emily's die has four faces showing 3 and two faces showing 8 --/
def emily_die : Die :=
  { sides := 6,
    values := λ i => if i.val < 4 then 3 else 8 }

/-- The probability of the desired outcome --/
def probability : Rat :=
  8 / 27

/-- Theorem stating the probability of the desired outcome --/
theorem desired_outcome_probability :
  (∀ (c : Fin carla_die.sides) (d : Fin derek_die.sides) (e : Fin emily_die.sides),
    (carla_die.values c > derek_die.values d ∧
     carla_die.values c > emily_die.values e ∧
     derek_die.values d + emily_die.values e < 10) →
    probability = 8 / 27) :=
by
  sorry

end desired_outcome_probability_l2234_223406


namespace arcsin_sufficient_not_necessary_l2234_223403

theorem arcsin_sufficient_not_necessary :
  (∃ α : ℝ, α = Real.arcsin (1/3) ∧ Real.sin α = 1/3) ∧
  (∃ β : ℝ, Real.sin β = 1/3 ∧ β ≠ Real.arcsin (1/3)) := by
  sorry

end arcsin_sufficient_not_necessary_l2234_223403


namespace average_of_c_and_d_l2234_223449

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 8 + c + d) / 5 = 18 → (c + d) / 2 = 36 := by
  sorry

end average_of_c_and_d_l2234_223449


namespace square_field_area_proof_l2234_223439

/-- The time taken to cross the square field diagonally in hours -/
def crossing_time : ℝ := 6.0008333333333335

/-- The speed of the person crossing the field in km/hr -/
def crossing_speed : ℝ := 1.2

/-- The area of the square field in square meters -/
def field_area : ℝ := 25939744.8

/-- Theorem stating that the area of the square field is approximately 25939744.8 square meters -/
theorem square_field_area_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |field_area - (crossing_speed * 1000 * crossing_time)^2 / 2| < ε :=
sorry

end square_field_area_proof_l2234_223439


namespace daily_harvest_l2234_223493

/-- The number of sections in the orchard -/
def sections : ℕ := 8

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := sections * sacks_per_section

theorem daily_harvest : total_sacks = 360 := by
  sorry

end daily_harvest_l2234_223493
