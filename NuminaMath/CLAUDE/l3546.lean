import Mathlib

namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l3546_354659

/-- A complex number is purely imaginary if its real part is zero -/
def PurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The condition that (z+2)/(1-i) + z is a real number -/
def IsRealCondition (z : ℂ) : Prop := ((z + 2) / (1 - i) + z).im = 0

theorem purely_imaginary_complex : 
  ∀ z : ℂ, PurelyImaginary z → IsRealCondition z → z = -2/3 * i :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l3546_354659


namespace NUMINAMATH_CALUDE_sampling_survey_correct_l3546_354645

/-- Represents a statement about quality testing methods -/
inductive QualityTestingMethod
| SamplingSurvey
| Other

/-- Represents the correctness of a statement -/
inductive Correctness
| Correct
| Incorrect

/-- The correct method for testing the quality of a batch of light bulbs -/
def lightBulbQualityTestingMethod : QualityTestingMethod := QualityTestingMethod.SamplingSurvey

/-- Theorem stating that sampling survey is the correct method for testing light bulb quality -/
theorem sampling_survey_correct :
  Correctness.Correct = match lightBulbQualityTestingMethod with
    | QualityTestingMethod.SamplingSurvey => Correctness.Correct
    | QualityTestingMethod.Other => Correctness.Incorrect :=
by sorry

end NUMINAMATH_CALUDE_sampling_survey_correct_l3546_354645


namespace NUMINAMATH_CALUDE_cards_nell_has_left_nell_remaining_cards_l3546_354649

/-- Calculates the number of cards Nell has left after giving some to Jeff -/
theorem cards_nell_has_left (nell_initial : ℕ) (jeff_initial : ℕ) (jeff_final : ℕ) : ℕ :=
  let cards_transferred := jeff_final - jeff_initial
  nell_initial - cards_transferred

/-- Proves that Nell has 252 cards left after giving some to Jeff -/
theorem nell_remaining_cards : 
  cards_nell_has_left 528 11 287 = 252 := by
  sorry


end NUMINAMATH_CALUDE_cards_nell_has_left_nell_remaining_cards_l3546_354649


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3546_354614

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3546_354614


namespace NUMINAMATH_CALUDE_profit_equation_store_profit_equation_l3546_354690

/-- Represents the profit equation for a store selling goods -/
theorem profit_equation (initial_price initial_cost initial_volume : ℕ) 
                        (price_increase : ℕ) (volume_decrease_rate : ℕ) 
                        (profit_increase : ℕ) : Prop :=
  let new_price := initial_price + price_increase
  let new_volume := initial_volume - volume_decrease_rate * price_increase
  let profit_per_unit := new_price - initial_cost
  profit_per_unit * new_volume = initial_volume * (initial_price - initial_cost) + profit_increase

/-- The specific profit equation for the given problem -/
theorem store_profit_equation (x : ℕ) : 
  profit_equation 36 20 200 x 5 1200 ↔ (x + 16) * (200 - 5 * x) = 1200 :=
sorry

end NUMINAMATH_CALUDE_profit_equation_store_profit_equation_l3546_354690


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3546_354604

theorem complex_equation_solution (z : ℂ) :
  z * (1 + Complex.I) = -2 * Complex.I → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3546_354604


namespace NUMINAMATH_CALUDE_jack_emails_l3546_354610

theorem jack_emails (morning_emails : ℕ) (afternoon_emails : ℕ) 
  (h1 : morning_emails = 6)
  (h2 : afternoon_emails = morning_emails + 2) : 
  afternoon_emails = 8 := by
sorry

end NUMINAMATH_CALUDE_jack_emails_l3546_354610


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3546_354637

theorem system_of_equations_solution (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 20) 
  (eq2 : 4 * x + y = 25) : 
  (x + y)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3546_354637


namespace NUMINAMATH_CALUDE_combined_wave_amplitude_l3546_354681

noncomputable def y₁ (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y₂ (t : ℝ) : ℝ := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y₁ t + y₂ t

theorem combined_wave_amplitude :
  ∃ (A : ℝ) (φ : ℝ), ∀ t, y t = A * Real.sin (100 * Real.pi * t + φ) ∧ A = 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_combined_wave_amplitude_l3546_354681


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3546_354672

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (a b : ℝ), a ∈ Set.Icc 0 2 ∧ b ∈ Set.Icc 0 2 ∧
  (∀ x, x ∈ Set.Icc 0 2 → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc 0 2 → f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 :=
sorry


end NUMINAMATH_CALUDE_f_max_min_on_interval_l3546_354672


namespace NUMINAMATH_CALUDE_minimum_race_distance_l3546_354695

/-- The minimum distance problem for the race setup -/
theorem minimum_race_distance (wall_length : ℝ) (a_to_wall : ℝ) (wall_to_b : ℝ) :
  wall_length = 800 →
  a_to_wall = 200 →
  wall_to_b = 400 →
  ∃ (min_distance : ℝ),
    min_distance = 1000 ∧
    ∀ (path : ℝ),
      (∃ (x : ℝ), 0 ≤ x ∧ x ≤ wall_length ∧
        path = Real.sqrt ((x ^ 2) + (a_to_wall ^ 2)) +
               Real.sqrt (((wall_length - x) ^ 2) + (wall_to_b ^ 2))) →
      min_distance ≤ path :=
by sorry

end NUMINAMATH_CALUDE_minimum_race_distance_l3546_354695


namespace NUMINAMATH_CALUDE_fraction_simplification_l3546_354623

theorem fraction_simplification (a : ℝ) (h : a^2 ≠ 9) :
  3 / (a^2 - 9) - a / (9 - a^2) = 1 / (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3546_354623


namespace NUMINAMATH_CALUDE_dandelion_puffs_distribution_l3546_354624

theorem dandelion_puffs_distribution (total : ℕ) (given_away : ℕ) (friends : ℕ) 
  (h_total : total = 40)
  (h_given_away : given_away = 3 + 3 + 5 + 2)
  (h_friends : friends = 3)
  (h_positive : friends > 0) :
  (total - given_away) / friends = 9 :=
sorry

end NUMINAMATH_CALUDE_dandelion_puffs_distribution_l3546_354624


namespace NUMINAMATH_CALUDE_research_paper_word_count_l3546_354664

/-- Calculates the number of words typed given a typing speed and duration. -/
def words_typed (typing_speed : ℕ) (duration_hours : ℕ) : ℕ :=
  typing_speed * (duration_hours * 60)

/-- Proves that given a typing speed of 38 words per minute and a duration of 2 hours,
    the total number of words typed is 4560. -/
theorem research_paper_word_count :
  words_typed 38 2 = 4560 := by
  sorry

end NUMINAMATH_CALUDE_research_paper_word_count_l3546_354664


namespace NUMINAMATH_CALUDE_selection_theorem_l3546_354621

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of people to choose from -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def select_count : ℕ := 4

/-- The number of ways to select 4 people from 4 boys and 3 girls,
    such that the selection includes at least one boy and one girl -/
def selection_methods : ℕ := 34

theorem selection_theorem :
  (Nat.choose total_people select_count) - (Nat.choose num_boys select_count) = selection_methods :=
sorry

end NUMINAMATH_CALUDE_selection_theorem_l3546_354621


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3546_354697

theorem square_sum_geq_product_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3546_354697


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l3546_354668

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- There are 5 indistinguishable balls -/
def num_balls : ℕ := 5

/-- There are 3 distinguishable boxes -/
def num_boxes : ℕ := 3

/-- The theorem states that there are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : 
  distribute_balls num_balls num_boxes = 21 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l3546_354668


namespace NUMINAMATH_CALUDE_taras_birthday_money_l3546_354619

theorem taras_birthday_money (P : ℝ) : P * 1.1 = 99 → P = 90 := by
  sorry

end NUMINAMATH_CALUDE_taras_birthday_money_l3546_354619


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l3546_354609

/-- Represents the points on the circle -/
inductive Point : Type
| one | two | three | four | five | six | seven

/-- The next point function, implementing the jumping rules -/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.five
  | Point.three => Point.six
  | Point.four => Point.seven
  | Point.five => Point.seven
  | Point.six => Point.two
  | Point.seven => Point.two

/-- Performs n jumps starting from a given point -/
def jumpN (start : Point) (n : ℕ) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpN start n)

/-- The main theorem stating that after 2023 jumps from point 7, the bug lands on point 2 -/
theorem bug_position_after_2023_jumps :
  jumpN Point.seven 2023 = Point.two := by sorry

end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l3546_354609


namespace NUMINAMATH_CALUDE_F_4_f_5_equals_69_l3546_354606

-- Define the functions f and F
def f (a : ℝ) : ℝ := 2 * a - 2

def F (a b : ℝ) : ℝ := b^2 + a + 1

-- State the theorem
theorem F_4_f_5_equals_69 : F 4 (f 5) = 69 := by
  sorry

end NUMINAMATH_CALUDE_F_4_f_5_equals_69_l3546_354606


namespace NUMINAMATH_CALUDE_arithmetic_seq_problem_l3546_354643

/-- Define an arithmetic sequence {aₙ/n} with common difference d -/
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) :=
  ∀ n m : ℕ, a m / m - a n / n = d * (m - n)

theorem arithmetic_seq_problem (a : ℕ → ℚ) (d : ℚ) 
  (h_seq : arithmetic_seq a d)
  (h_a3 : a 3 = 2)
  (h_a9 : a 9 = 12) :
  d = 1/9 ∧ a 12 = 20 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_seq_problem_l3546_354643


namespace NUMINAMATH_CALUDE_purely_imaginary_value_l3546_354652

-- Define a complex number z as a function of real number m
def z (m : ℝ) : ℂ := m + 2 + (m - 1) * Complex.I

-- State the theorem
theorem purely_imaginary_value (m : ℝ) : 
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_value_l3546_354652


namespace NUMINAMATH_CALUDE_dividend_calculation_l3546_354658

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 36 → quotient = 19 → remainder = 6 → 
  divisor * quotient + remainder = 690 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3546_354658


namespace NUMINAMATH_CALUDE_blocks_lost_l3546_354647

/-- Given Carol's initial and final block counts, prove the number of blocks lost. -/
theorem blocks_lost (initial : ℕ) (final : ℕ) (h1 : initial = 42) (h2 : final = 17) :
  initial - final = 25 := by
  sorry

end NUMINAMATH_CALUDE_blocks_lost_l3546_354647


namespace NUMINAMATH_CALUDE_time_until_800_l3546_354660

def minutes_since_730 : ℕ := 16

def current_time : ℕ := 7 * 60 + 30 + minutes_since_730

def target_time : ℕ := 8 * 60

theorem time_until_800 : target_time - current_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_time_until_800_l3546_354660


namespace NUMINAMATH_CALUDE_value_of_M_l3546_354646

theorem value_of_M : ∃ M : ℝ, (0.3 * M = 0.6 * 500) ∧ (M = 1000) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l3546_354646


namespace NUMINAMATH_CALUDE_friend_team_assignments_l3546_354688

/-- The number of ways to assign n distinguishable objects to k distinct categories -/
def assignments (n k : ℕ) : ℕ := k^n

/-- The number of friends -/
def num_friends : ℕ := 8

/-- The number of teams -/
def num_teams : ℕ := 4

/-- Theorem: The number of ways to assign 8 friends to 4 teams is 65536 -/
theorem friend_team_assignments : assignments num_friends num_teams = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignments_l3546_354688


namespace NUMINAMATH_CALUDE_ultramen_defeat_monster_l3546_354663

theorem ultramen_defeat_monster (monster_health : ℕ) (ultraman1_rate : ℕ) (ultraman2_rate : ℕ) :
  monster_health = 100 →
  ultraman1_rate = 12 →
  ultraman2_rate = 8 →
  (monster_health : ℚ) / (ultraman1_rate + ultraman2_rate : ℚ) = 5 :=
by sorry

end NUMINAMATH_CALUDE_ultramen_defeat_monster_l3546_354663


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3546_354600

/-- The perimeter of a semicircle with radius 6.7 cm is equal to π * 6.7 + 13.4 cm. -/
theorem semicircle_perimeter : 
  let r : ℝ := 6.7
  ∀ π : ℝ, π * r + 2 * r = π * r + 13.4 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3546_354600


namespace NUMINAMATH_CALUDE_kelly_head_start_l3546_354642

/-- The length of the race in meters -/
def race_length : ℝ := 100

/-- The distance by which Abel lost to Kelly in meters -/
def losing_distance : ℝ := 0.5

/-- The additional distance Abel needs to run to overtake Kelly in meters -/
def overtake_distance : ℝ := 19.9

/-- Kelly's head start in meters -/
def head_start : ℝ := race_length - (race_length - losing_distance - overtake_distance)

theorem kelly_head_start :
  head_start = 20.4 :=
by sorry

end NUMINAMATH_CALUDE_kelly_head_start_l3546_354642


namespace NUMINAMATH_CALUDE_zeros_of_composite_function_l3546_354699

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then k * x + 1 else Real.log x

-- Define the composite function g
def g (k : ℝ) (x : ℝ) : ℝ := f k (f k x + 1)

-- Theorem statement
theorem zeros_of_composite_function (k : ℝ) :
  (k > 0 → (∃ x₁ x₂ x₃ x₄ : ℝ, g k x₁ = 0 ∧ g k x₂ = 0 ∧ g k x₃ = 0 ∧ g k x₄ = 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)) ∧
  (k < 0 → (∃! x : ℝ, g k x = 0)) :=
sorry

end

end NUMINAMATH_CALUDE_zeros_of_composite_function_l3546_354699


namespace NUMINAMATH_CALUDE_sum_of_acute_angles_not_always_obtuse_l3546_354689

-- Define what an acute angle is
def is_acute_angle (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define what an obtuse angle is
def is_obtuse_angle (α : Real) : Prop := Real.pi / 2 < α ∧ α < Real.pi

-- Theorem stating that the sum of two acute angles is not always obtuse
theorem sum_of_acute_angles_not_always_obtuse :
  ∃ (α β : Real), is_acute_angle α ∧ is_acute_angle β ∧ ¬is_obtuse_angle (α + β) :=
sorry

end NUMINAMATH_CALUDE_sum_of_acute_angles_not_always_obtuse_l3546_354689


namespace NUMINAMATH_CALUDE_profit_calculation_l3546_354669

/-- The number of pens John buys for $8 -/
def pens_bought : ℕ := 5

/-- The price John pays for pens_bought pens -/
def buy_price : ℚ := 8

/-- The number of pens John sells for $10 -/
def pens_sold : ℕ := 4

/-- The price John receives for pens_sold pens -/
def sell_price : ℚ := 10

/-- The desired profit -/
def target_profit : ℚ := 120

/-- The minimum number of pens John needs to sell to make the target profit -/
def min_pens_to_sell : ℕ := 134

theorem profit_calculation :
  ↑min_pens_to_sell * (sell_price / pens_sold - buy_price / pens_bought) ≥ target_profit ∧
  ∀ n : ℕ, n < min_pens_to_sell → ↑n * (sell_price / pens_sold - buy_price / pens_bought) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_calculation_l3546_354669


namespace NUMINAMATH_CALUDE_sum_of_factors_l3546_354670

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = -120 →
  a + b + c + d + e = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l3546_354670


namespace NUMINAMATH_CALUDE_divisibility_proof_l3546_354636

theorem divisibility_proof :
  ∃ (n : ℕ), (425897 + n) % 456 = 0 ∧ n = 47 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l3546_354636


namespace NUMINAMATH_CALUDE_candle_ratio_l3546_354612

/-- Proves that the ratio of candles in Kalani's bedroom to candles in the living room is 2:1 -/
theorem candle_ratio :
  ∀ (bedroom_candles living_room_candles donovan_candles total_candles : ℕ),
    bedroom_candles = 20 →
    donovan_candles = 20 →
    total_candles = 50 →
    bedroom_candles + living_room_candles + donovan_candles = total_candles →
    (bedroom_candles : ℚ) / living_room_candles = 2 := by
  sorry

end NUMINAMATH_CALUDE_candle_ratio_l3546_354612


namespace NUMINAMATH_CALUDE_equation_solution_l3546_354653

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (20 / (x^2 - 9)) + (5 / (x - 3))
  ∀ x : ℝ, f x = 2 ↔ x = (5 + Real.sqrt 449) / 4 ∨ x = (5 - Real.sqrt 449) / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3546_354653


namespace NUMINAMATH_CALUDE_two_digit_sqrt_prob_l3546_354638

theorem two_digit_sqrt_prob : 
  let two_digit_numbers := Finset.Icc 10 99
  let satisfying_numbers := two_digit_numbers.filter (λ n => n.sqrt < 8)
  (satisfying_numbers.card : ℚ) / two_digit_numbers.card = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sqrt_prob_l3546_354638


namespace NUMINAMATH_CALUDE_gcd_459_357_l3546_354613

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l3546_354613


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3546_354674

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 2) 
  (h2 : a * b = 3) : 
  (a + b)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3546_354674


namespace NUMINAMATH_CALUDE_mens_wages_are_fifty_l3546_354654

/-- Represents the wages of a group given the number of individuals and their equality relationships -/
def group_wages (men women boys : ℕ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (men + women + boys : ℚ) * men

/-- Theorem stating that under given conditions, the men's wages are 50 -/
theorem mens_wages_are_fifty
  (men : ℕ) (women : ℕ) (boys : ℕ) (total_earnings : ℚ)
  (h1 : men = 5)
  (h2 : boys = 8)
  (h3 : men = women)  -- 5 men are equal to W women
  (h4 : women = boys) -- W women are equal to 8 boys
  (h5 : total_earnings = 150) :
  group_wages men women boys total_earnings = 50 := by
  sorry

#eval group_wages 5 5 8 150

end NUMINAMATH_CALUDE_mens_wages_are_fifty_l3546_354654


namespace NUMINAMATH_CALUDE_complex_fraction_real_l3546_354686

theorem complex_fraction_real (a : ℝ) : 
  (((a : ℂ) + Complex.I) / (1 + Complex.I)).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l3546_354686


namespace NUMINAMATH_CALUDE_power_sum_equality_l3546_354671

theorem power_sum_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 ∧ 
  ∃ a b c d : ℝ, (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3546_354671


namespace NUMINAMATH_CALUDE_prime_condition_equivalence_l3546_354685

/-- For a prime number p, this function returns true if for each integer a 
    such that 1 < a < p/2, there exists an integer b such that p/2 < b < p 
    and p divides ab - 1 -/
def satisfies_condition (p : ℕ) : Prop :=
  ∀ a : ℕ, 1 < a → a < p / 2 → ∃ b : ℕ, p / 2 < b ∧ b < p ∧ p ∣ (a * b - 1)

theorem prime_condition_equivalence (p : ℕ) (hp : Nat.Prime p) : 
  satisfies_condition p ↔ p ∈ ({5, 7, 13} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_prime_condition_equivalence_l3546_354685


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3546_354655

theorem min_value_expression (x y : ℝ) (h1 : y > 0) (h2 : y = -1/x + 1) :
  2*x + 1/y ≥ 2*Real.sqrt 2 + 3 :=
by
  sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), y > 0 ∧ y = -1/x + 1 ∧ 2*x + 1/y = 2*Real.sqrt 2 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3546_354655


namespace NUMINAMATH_CALUDE_highest_power_divisibility_l3546_354687

theorem highest_power_divisibility (n : ℕ) : 
  (∃ k : ℕ, (1991 : ℕ)^k ∣ 1990^(1991^1002) + 1992^(1501^1901)) ∧ 
  (∀ m : ℕ, m > 1001 → ¬((1991 : ℕ)^m ∣ 1990^(1991^1002) + 1992^(1501^1901))) :=
by sorry

end NUMINAMATH_CALUDE_highest_power_divisibility_l3546_354687


namespace NUMINAMATH_CALUDE_rat_value_l3546_354680

/-- Represents the value of a letter based on its position in the alphabet -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0

/-- Calculates the number value of a word -/
def word_value (w : String) : ℕ :=
  (w.toList.map letter_value).sum * w.length

/-- Theorem: The number value of the word "rat" is 117 -/
theorem rat_value : word_value "rat" = 117 := by
  sorry

end NUMINAMATH_CALUDE_rat_value_l3546_354680


namespace NUMINAMATH_CALUDE_total_cost_is_543_l3546_354684

/-- Calculates the total amount John has to pay for earbuds and a smartwatch, including tax and discount. -/
def totalCost (earbudsCost smartwatchCost : ℝ) (earbudsTaxRate smartwatchTaxRate earbusDiscountRate : ℝ) : ℝ :=
  let discountedEarbudsCost := earbudsCost * (1 - earbusDiscountRate)
  let earbudsTax := discountedEarbudsCost * earbudsTaxRate
  let smartwatchTax := smartwatchCost * smartwatchTaxRate
  discountedEarbudsCost + earbudsTax + smartwatchCost + smartwatchTax

/-- Theorem stating that given the specific costs, tax rates, and discount, the total cost is $543. -/
theorem total_cost_is_543 :
  totalCost 200 300 0.15 0.12 0.10 = 543 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_543_l3546_354684


namespace NUMINAMATH_CALUDE_least_number_with_remainder_least_number_is_205517_least_number_unique_l3546_354693

theorem least_number_with_remainder (n : ℕ) : 
  (n % 45 = 2 ∧ n % 59 = 2 ∧ n % 77 = 2) → n ≥ 205517 :=
by
  sorry

theorem least_number_is_205517 : 
  205517 % 45 = 2 ∧ 205517 % 59 = 2 ∧ 205517 % 77 = 2 :=
by
  sorry

theorem least_number_unique : 
  ∃! n : ℕ, (n % 45 = 2 ∧ n % 59 = 2 ∧ n % 77 = 2) ∧ 
  ∀ m : ℕ, (m % 45 = 2 ∧ m % 59 = 2 ∧ m % 77 = 2) → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_least_number_is_205517_least_number_unique_l3546_354693


namespace NUMINAMATH_CALUDE_abs_neg_two_thirds_eq_two_thirds_l3546_354620

theorem abs_neg_two_thirds_eq_two_thirds : |(-2 : ℚ) / 3| = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_thirds_eq_two_thirds_l3546_354620


namespace NUMINAMATH_CALUDE_inequality_proof_l3546_354665

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2*y*z)) + (y^2 / (y^2 + 2*z*x)) + (z^2 / (z^2 + 2*x*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3546_354665


namespace NUMINAMATH_CALUDE_half_jar_days_l3546_354676

/-- Represents the area of kombucha in the jar as a function of time -/
def kombucha_area (t : ℕ) : ℝ := 2^t

/-- The number of days it takes to fill the entire jar -/
def full_jar_days : ℕ := 17

theorem half_jar_days : 
  (kombucha_area full_jar_days = 2 * kombucha_area (full_jar_days - 1)) → 
  (kombucha_area (full_jar_days - 1) = (1/2) * kombucha_area full_jar_days) := by
  sorry

end NUMINAMATH_CALUDE_half_jar_days_l3546_354676


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3546_354666

theorem product_of_sums_equals_difference_of_powers : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3546_354666


namespace NUMINAMATH_CALUDE_town_distance_l3546_354656

/-- Three towns A, B, and C are equidistant from each other and are 3, 5, and 8 miles 
    respectively from a common railway station D. -/
structure TownConfiguration where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  equidistant : dist A B = dist B C ∧ dist B C = dist C A
  dist_AD : dist A D = 3
  dist_BD : dist B D = 5
  dist_CD : dist C D = 8

/-- The distance between any two towns is 7 miles. -/
theorem town_distance (config : TownConfiguration) : 
  dist config.A config.B = 7 ∧ dist config.B config.C = 7 ∧ dist config.C config.A = 7 := by
  sorry


end NUMINAMATH_CALUDE_town_distance_l3546_354656


namespace NUMINAMATH_CALUDE_number_of_gyms_l3546_354611

def number_of_bikes_per_gym : ℕ := 10
def number_of_treadmills_per_gym : ℕ := 5
def number_of_ellipticals_per_gym : ℕ := 5

def cost_of_bike : ℕ := 700
def cost_of_treadmill : ℕ := cost_of_bike + cost_of_bike / 2
def cost_of_elliptical : ℕ := 2 * cost_of_treadmill

def total_replacement_cost : ℕ := 455000

def cost_per_gym : ℕ := 
  number_of_bikes_per_gym * cost_of_bike +
  number_of_treadmills_per_gym * cost_of_treadmill +
  number_of_ellipticals_per_gym * cost_of_elliptical

theorem number_of_gyms : 
  total_replacement_cost / cost_per_gym = 20 := by sorry

end NUMINAMATH_CALUDE_number_of_gyms_l3546_354611


namespace NUMINAMATH_CALUDE_simplify_expression_l3546_354698

theorem simplify_expression (x : ℝ) (h1 : 1 < x) (h2 : x < 4) :
  Real.sqrt ((1 - x)^2) + |x - 4| = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3546_354698


namespace NUMINAMATH_CALUDE_four_solutions_implies_a_greater_than_two_l3546_354677

-- Define the equation
def equation (a x : ℝ) : Prop := |x^3 - a*x^2| = x

-- Theorem statement
theorem four_solutions_implies_a_greater_than_two (a : ℝ) :
  (∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    equation a w ∧ equation a x ∧ equation a y ∧ equation a z) →
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_four_solutions_implies_a_greater_than_two_l3546_354677


namespace NUMINAMATH_CALUDE_quick_customer_sale_l3546_354673

def chicken_problem (initial_chickens neighbor_sale remaining_chickens : ℕ) : ℕ :=
  initial_chickens - neighbor_sale - remaining_chickens

theorem quick_customer_sale :
  chicken_problem 80 12 43 = 25 := by
  sorry

end NUMINAMATH_CALUDE_quick_customer_sale_l3546_354673


namespace NUMINAMATH_CALUDE_intersection_symmetry_l3546_354616

/-- Given a line y = kx that intersects the circle (x-1)^2 + y^2 = 1 at two points
    symmetric with respect to the line x - y + b = 0, prove that k = -1 and b = -1 -/
theorem intersection_symmetry (k b : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    -- The line intersects the circle at two points
    y₁ = k * x₁ ∧ (x₁ - 1)^2 + y₁^2 = 1 ∧
    y₂ = k * x₂ ∧ (x₂ - 1)^2 + y₂^2 = 1 ∧
    -- The points are distinct
    (x₁, y₁) ≠ (x₂, y₂) ∧
    -- The points are symmetric with respect to x - y + b = 0
    ∃ x₀ y₀ : ℝ, x₀ - y₀ + b = 0 ∧
    x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2) →
  k = -1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_symmetry_l3546_354616


namespace NUMINAMATH_CALUDE_eeshas_travel_time_l3546_354640

/-- Eesha's travel time problem -/
theorem eeshas_travel_time 
  (usual_time : ℝ) 
  (usual_speed : ℝ) 
  (late_start : ℝ) 
  (late_arrival : ℝ) 
  (speed_reduction : ℝ) 
  (h1 : late_start = 30) 
  (h2 : late_arrival = 50) 
  (h3 : speed_reduction = 0.25) 
  (h4 : usual_time / (usual_time + late_arrival) = (1 - speed_reduction)) :
  usual_time = 150 := by
sorry

end NUMINAMATH_CALUDE_eeshas_travel_time_l3546_354640


namespace NUMINAMATH_CALUDE_value_of_n_l3546_354662

theorem value_of_n (x : ℝ) (n : ℝ) 
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -1)
  (h2 : Real.tan x = Real.sqrt 3)
  (h3 : Real.log (Real.sin x + Real.cos x) = (1/3) * (Real.log n - 1)) :
  n = Real.exp (3 * (-1/2 + Real.log (1 + 1 / Real.sqrt (Real.sqrt 3))) + 1) := by
sorry

end NUMINAMATH_CALUDE_value_of_n_l3546_354662


namespace NUMINAMATH_CALUDE_reflection_line_sum_l3546_354639

/-- Given a reflection of point (0,1) across line y = mx + b to point (4,5), prove m + b = 4 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 4 ∧ y = 5 ∧ 
    ((x - 0) * (x - 0) + (y - 1) * (y - 1)) / 4 = 
    ((x - 0) * (1 + y) / 2 - (y - 1) * (0 + x) / 2)^2 / ((x - 0)^2 + (y - 1)^2) ∧
    y = m * x + b) →
  m + b = 4 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l3546_354639


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3546_354625

theorem quadratic_root_problem (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x - 2 = 0 ∧ x = -2) →
  (∃ y : ℝ, y^2 + k*y - 2 = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3546_354625


namespace NUMINAMATH_CALUDE_union_cardinality_l3546_354632

def A : Finset ℕ := {1, 3, 5}
def B : Finset ℕ := {2, 3}

theorem union_cardinality : Finset.card (A ∪ B) = 4 := by
  sorry

end NUMINAMATH_CALUDE_union_cardinality_l3546_354632


namespace NUMINAMATH_CALUDE_initial_overs_played_l3546_354691

/-- Proves the number of overs played initially in a cricket game -/
theorem initial_overs_played (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (remaining_overs : ℝ) :
  target = 282 →
  initial_rate = 3.2 →
  required_rate = 8.333333333333334 →
  remaining_overs = 30 →
  ∃ (initial_overs : ℝ), 
    initial_overs * initial_rate + remaining_overs * required_rate = target ∧
    initial_overs = 10 :=
by sorry

end NUMINAMATH_CALUDE_initial_overs_played_l3546_354691


namespace NUMINAMATH_CALUDE_correct_requirements_l3546_354679

/-- A cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstInningOvers : ℕ
  runsScored : ℕ
  wicketsLost : ℕ
  runRate : ℚ
  targetScore : ℕ

/-- Calculate the required run rate and partnership score -/
def calculateRequirements (game : CricketGame) : ℚ × ℕ :=
  let remainingOvers := game.totalOvers - game.firstInningOvers
  let remainingRuns := game.targetScore - game.runsScored
  let requiredRunRate := remainingRuns / remainingOvers
  let requiredPartnership := remainingRuns
  (requiredRunRate, requiredPartnership)

/-- Theorem stating the correct calculation of requirements -/
theorem correct_requirements (game : CricketGame) 
    (h1 : game.totalOvers = 50)
    (h2 : game.firstInningOvers = 10)
    (h3 : game.runsScored = 32)
    (h4 : game.wicketsLost = 3)
    (h5 : game.runRate = 32/10)
    (h6 : game.targetScore = 282) :
    calculateRequirements game = (25/4, 250) := by
  sorry

#eval calculateRequirements {
  totalOvers := 50,
  firstInningOvers := 10,
  runsScored := 32,
  wicketsLost := 3,
  runRate := 32/10,
  targetScore := 282
}

end NUMINAMATH_CALUDE_correct_requirements_l3546_354679


namespace NUMINAMATH_CALUDE_inscribable_iff_equal_sums_l3546_354633

/-- A convex polyhedral angle -/
structure ConvexPolyhedralAngle where
  -- Add necessary fields

/-- The property of being inscribable in a cone -/
def isInscribableInCone (angle : ConvexPolyhedralAngle) : Prop :=
  sorry

/-- The property of having equal sums of opposite dihedral angles -/
def hasEqualSumsOfOppositeDihedralAngles (angle : ConvexPolyhedralAngle) : Prop :=
  sorry

/-- Theorem: A convex polyhedral angle can be inscribed in a cone if and only if 
    the sums of its opposite dihedral angles are equal -/
theorem inscribable_iff_equal_sums 
  (angle : ConvexPolyhedralAngle) : 
  isInscribableInCone angle ↔ hasEqualSumsOfOppositeDihedralAngles angle :=
sorry

end NUMINAMATH_CALUDE_inscribable_iff_equal_sums_l3546_354633


namespace NUMINAMATH_CALUDE_product_digit_sum_l3546_354608

/-- Converts a base 7 number to base 10 -/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base 7 -/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The product of 35₇ and 12₇ in base 7 -/
def product : ℕ := toBase7 (toBase10 35 * toBase10 12)

theorem product_digit_sum :
  sumOfDigitsBase7 product = 12 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3546_354608


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3546_354651

theorem consecutive_integers_sum (x y z : ℤ) (w : ℤ) : 
  y = x + 1 → 
  z = x + 2 → 
  x + y + z = 150 → 
  w = 2*z - x → 
  x + y + z + w = 203 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3546_354651


namespace NUMINAMATH_CALUDE_tangent_and_decreasing_l3546_354696

noncomputable section

def f (x : ℝ) := Real.log x

def g (a b : ℝ) (x : ℝ) := (1/2) * a * x + b

def φ (m : ℝ) (x : ℝ) := (m * (x - 1)) / (x + 1) - f x

theorem tangent_and_decreasing 
  (h1 : ∃ (a b : ℝ), ∀ x, f x = g a b x → x = 1) 
  (h2 : ∀ x ≥ 1, ∀ y > x, φ m y ≤ φ m x) :
  (∃ (a b : ℝ), ∀ x, g a b x = x - 1) ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_tangent_and_decreasing_l3546_354696


namespace NUMINAMATH_CALUDE_vasya_tolya_winning_strategy_l3546_354602

/-- Represents a player in the game -/
inductive Player : Type
| Petya : Player
| Vasya : Player
| Tolya : Player

/-- Represents a cell on the board -/
structure Cell :=
(index : Nat)

/-- Represents the game board -/
structure Board :=
(size : Nat)
(boundary_cells : Nat)

/-- Represents the game state -/
structure GameState :=
(board : Board)
(painted_cells : List Cell)
(current_player : Player)

/-- Checks if two cells are adjacent -/
def are_adjacent (c1 c2 : Cell) (board : Board) : Prop :=
  (c1.index + 1) % board.boundary_cells = c2.index ∨
  (c2.index + 1) % board.boundary_cells = c1.index

/-- Checks if two cells are symmetrical with respect to the board center -/
def are_symmetrical (c1 c2 : Cell) (board : Board) : Prop :=
  (c1.index + board.boundary_cells / 2) % board.boundary_cells = c2.index

/-- Determines if a move is valid -/
def is_valid_move (cell : Cell) (state : GameState) : Prop :=
  cell.index < state.board.boundary_cells ∧
  cell ∉ state.painted_cells ∧
  (∀ c ∈ state.painted_cells, ¬(are_adjacent cell c state.board)) ∧
  (∀ c ∈ state.painted_cells, ¬(are_symmetrical cell c state.board))

/-- Theorem: There exists a winning strategy for Vasya and Tolya -/
theorem vasya_tolya_winning_strategy :
  ∃ (strategy : GameState → Cell),
    ∀ (initial_state : GameState),
      initial_state.board.size = 100 ∧
      initial_state.board.boundary_cells = 396 ∧
      initial_state.current_player = Player.Petya →
        ∃ (final_state : GameState),
          final_state.current_player = Player.Petya ∧
          ¬∃ (move : Cell), is_valid_move move final_state :=
sorry

end NUMINAMATH_CALUDE_vasya_tolya_winning_strategy_l3546_354602


namespace NUMINAMATH_CALUDE_least_number_of_cubes_l3546_354628

/-- Represents the dimensions of a cuboidal block in centimeters -/
structure CuboidalBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the side length of a cube in centimeters -/
def CubeSideLength : ℕ := 3

/-- The given cuboidal block -/
def given_block : CuboidalBlock := ⟨18, 27, 36⟩

/-- The volume of a cuboidal block -/
def volume_cuboid (b : CuboidalBlock) : ℕ := b.length * b.width * b.height

/-- The volume of a cube -/
def volume_cube (side : ℕ) : ℕ := side * side * side

/-- The number of cubes that can be cut from a cuboidal block -/
def number_of_cubes (b : CuboidalBlock) (side : ℕ) : ℕ :=
  volume_cuboid b / volume_cube side

/-- Theorem: The least possible number of equal cubes with side lengths in a fixed ratio of 1:2:3
    that can be cut from the given cuboidal block is 648 -/
theorem least_number_of_cubes :
  number_of_cubes given_block CubeSideLength = 648 := by sorry

end NUMINAMATH_CALUDE_least_number_of_cubes_l3546_354628


namespace NUMINAMATH_CALUDE_picture_area_l3546_354630

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h_frame_area : (2 * x + 5) * (y + 4) - x * y = 84) : x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l3546_354630


namespace NUMINAMATH_CALUDE_number_of_divisors_of_30_l3546_354605

theorem number_of_divisors_of_30 : Finset.card (Nat.divisors 30) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_30_l3546_354605


namespace NUMINAMATH_CALUDE_pizza_cost_equality_l3546_354607

theorem pizza_cost_equality (total_cost : ℚ) (num_slices : ℕ) 
  (h1 : total_cost = 13)
  (h2 : num_slices = 10) :
  let cost_per_slice := total_cost / num_slices
  5 * cost_per_slice = 5 * cost_per_slice := by
sorry

end NUMINAMATH_CALUDE_pizza_cost_equality_l3546_354607


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l3546_354694

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_unique_form 
  (f : ℝ → ℝ) 
  (h_vertex : f (-1) = 4 ∧ ∀ x, f x ≤ f (-1)) 
  (h_point : f 2 = -5) :
  ∃ a b c : ℝ, f = quadratic_function a b c ∧ a = -1 ∧ b = -2 ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_form_l3546_354694


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3546_354603

/-- Theorem: Volume of a cube with surface area 150 square inches --/
theorem cube_volume_from_surface_area :
  let surface_area : ℝ := 150  -- Surface area in square inches
  let edge_length : ℝ := Real.sqrt (surface_area / 6)  -- Edge length in inches
  let volume_cubic_inches : ℝ := edge_length ^ 3  -- Volume in cubic inches
  let cubic_inches_per_cubic_foot : ℝ := 1728  -- Conversion factor
  let volume_cubic_feet : ℝ := volume_cubic_inches / cubic_inches_per_cubic_foot
  ∃ ε > 0, |volume_cubic_feet - 0.0723| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3546_354603


namespace NUMINAMATH_CALUDE_problem_statement_l3546_354618

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 + m = 0

-- Define proposition q
def q (m : ℝ) : Prop := ∃ x y : ℝ, x^2 / (1 - 2*m) + y^2 / (m + 2) = 1 ∧ (1 - 2*m) * (m + 2) < 0

theorem problem_statement :
  (∀ m : ℝ, p m ↔ (m ≤ -1 ∨ m ≥ 2)) ∧
  (∀ m : ℝ, q m ↔ (m < -2 ∨ m > 1/2)) ∧
  (∀ m : ℝ, ¬(p m ∨ q m) ↔ (-1 < m ∧ m ≤ 1/2)) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3546_354618


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l3546_354629

theorem divisible_by_eleven (n : ℕ) : ∃ k : ℤ, 3^(2*n + 2) + 2^(6*n + 1) = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l3546_354629


namespace NUMINAMATH_CALUDE_boat_speed_calculation_l3546_354682

/-- Given the downstream speed and upstream speed of a boat, 
    calculate the stream speed and the man's rowing speed. -/
theorem boat_speed_calculation (R S : ℝ) :
  ∃ (x y : ℝ), 
    (R = y + x) ∧ 
    (S = y - x) ∧ 
    (x = (R - S) / 2) ∧ 
    (y = (R + S) / 2) := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_calculation_l3546_354682


namespace NUMINAMATH_CALUDE_no_integer_solutions_specific_solution_l3546_354648

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 6*m^2 + 5*m = 27*n^3 + 9*n^2 + 9*n + 1 := by
  sorry

-- Additional fact mentioned in the problem
theorem specific_solution : (31 * 26)^3 + 6*(31*26)^2 + 5*(31*26) = 27*n^3 + 9*n^2 + 9*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_specific_solution_l3546_354648


namespace NUMINAMATH_CALUDE_die_game_first_player_win_probability_l3546_354650

def game_win_probability : ℚ := 5/11

theorem die_game_first_player_win_probability :
  let n := 6  -- number of sides on the die
  let m := 7  -- winning condition (multiple of m)
  ∀ (k : ℕ), k < m →
    let p : ℚ := game_win_probability  -- probability of winning starting from state k
    (p = n / (2*n - 1) ∧
     p = (n-1) * (1 - p) / n + 1/n) :=
by sorry

end NUMINAMATH_CALUDE_die_game_first_player_win_probability_l3546_354650


namespace NUMINAMATH_CALUDE_todds_contribution_ratio_l3546_354675

theorem todds_contribution_ratio (total_cost : ℕ) (boss_contribution : ℕ) 
  (num_employees : ℕ) (employee_contribution : ℕ) : 
  total_cost = 100 →
  boss_contribution = 15 →
  num_employees = 5 →
  employee_contribution = 11 →
  (total_cost - (boss_contribution + num_employees * employee_contribution)) / boss_contribution = 2 := by
  sorry

end NUMINAMATH_CALUDE_todds_contribution_ratio_l3546_354675


namespace NUMINAMATH_CALUDE_line_product_mb_l3546_354615

/-- Given a line passing through points (-4, -2) and (1, 3) with equation y = mx + b, 
    prove that the product mb equals 2. -/
theorem line_product_mb (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  (-2 : ℝ) = m * (-4 : ℝ) + b →  -- Point (-4, -2) satisfies the equation
  (3 : ℝ) = m * (1 : ℝ) + b →    -- Point (1, 3) satisfies the equation
  m * b = 2 := by
sorry

end NUMINAMATH_CALUDE_line_product_mb_l3546_354615


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_40_by_150_percent_l3546_354634

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial + (percentage / 100) * initial = initial * (1 + percentage / 100) := by sorry

theorem increase_40_by_150_percent :
  40 + (150 / 100) * 40 = 100 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_40_by_150_percent_l3546_354634


namespace NUMINAMATH_CALUDE_sibling_ages_l3546_354683

/-- A family with 4 siblings: Richard, David, Scott, and Jane. -/
structure Family :=
  (Richard David Scott Jane : ℕ)

/-- The conditions and question of the problem -/
theorem sibling_ages (f : Family) : 
  f.Richard = f.David + 6 →
  f.David = f.Scott + 8 →
  f.Jane = f.Richard - 5 →
  f.Richard + 8 = 2 * (f.Scott + 8) →
  f.Jane + 10 = (f.David + 10) / 2 + 4 →
  f.Scott + 12 + f.Jane + 12 = 60 →
  f.Richard - 3 + f.David - 3 + f.Scott - 3 + f.Jane - 3 = 43 := by
  sorry

#check sibling_ages

end NUMINAMATH_CALUDE_sibling_ages_l3546_354683


namespace NUMINAMATH_CALUDE_dust_storm_untouched_acres_l3546_354641

/-- The number of acres left untouched by a dust storm -/
def acres_untouched (total_acres dust_covered_acres : ℕ) : ℕ :=
  total_acres - dust_covered_acres

/-- Theorem stating that given a prairie of 65,057 acres and a dust storm covering 64,535 acres, 
    the number of acres left untouched is 522 -/
theorem dust_storm_untouched_acres : 
  acres_untouched 65057 64535 = 522 := by
  sorry

end NUMINAMATH_CALUDE_dust_storm_untouched_acres_l3546_354641


namespace NUMINAMATH_CALUDE_inequality_proof_l3546_354617

theorem inequality_proof (a b : ℝ) (h : a ≠ b) :
  a^4 + 6*a^2*b^2 + b^4 > 4*a*b*(a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3546_354617


namespace NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l3546_354667

open Real

theorem function_inequality_implies_positive_a (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ 
    a * (x₀ - 1 / x₀) - 2 * log x₀ > -a / x₀) →
  a > 0 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l3546_354667


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l3546_354631

/-- Prove that the given parametric equations are equivalent to the Cartesian equation -/
theorem parametric_to_cartesian (t : ℝ) (x y : ℝ) (h1 : t ≠ 0) (h2 : x ≠ 1) 
  (h3 : x = 1 - 1/t) (h4 : y = 1 - t^2) : 
  y = x * (x - 2) / (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l3546_354631


namespace NUMINAMATH_CALUDE_divisibility_pairs_l3546_354661

theorem divisibility_pairs : 
  ∀ m n : ℕ+, 
    (∀ k : ℕ+, k ≤ n → m.val % k = 0) ∧ 
    (m.val % (n + 1) ≠ 0) ∧ 
    (m.val % (n + 2) ≠ 0) ∧ 
    (m.val % (n + 3) ≠ 0) →
    ((n = 1 ∧ Nat.gcd m.val 6 = 1) ∨ 
     (n = 2 ∧ Nat.gcd m.val 12 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_pairs_l3546_354661


namespace NUMINAMATH_CALUDE_cost_per_square_meter_l3546_354601

def initial_land : ℝ := 300
def final_land : ℝ := 900
def total_cost : ℝ := 12000

theorem cost_per_square_meter :
  (total_cost / (final_land - initial_land)) = 20 := by sorry

end NUMINAMATH_CALUDE_cost_per_square_meter_l3546_354601


namespace NUMINAMATH_CALUDE_solution_set_f_nonnegative_range_of_a_l3546_354622

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x| - 2

-- Theorem 1: Solution set of f(x) ≥ 0
theorem solution_set_f_nonnegative :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} :=
sorry

-- Theorem 2: Range of values for a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≤ |x| + a) → a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_nonnegative_range_of_a_l3546_354622


namespace NUMINAMATH_CALUDE_product_of_differences_l3546_354644

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2010) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2009)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2010) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2009)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2010) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2009) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/2010 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l3546_354644


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l3546_354657

theorem arithmetic_sequence_squares (x : ℚ) :
  (∃ (a d : ℚ), 
    (5 + x)^2 = a - d ∧
    (7 + x)^2 = a ∧
    (10 + x)^2 = a + d ∧
    d ≠ 0) →
  x = -31/8 ∧ (∃ d : ℚ, d^2 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l3546_354657


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3546_354627

theorem sufficient_not_necessary :
  (∀ x : ℝ, x < -1 → (x < -1 ∨ x > 1)) ∧
  ¬(∀ x : ℝ, (x < -1 ∨ x > 1) → x < -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3546_354627


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_l3546_354678

theorem mark_and_carolyn_money : 
  (5 : ℚ) / 8 + (2 : ℚ) / 5 = (41 : ℚ) / 40 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_l3546_354678


namespace NUMINAMATH_CALUDE_smallest_number_l3546_354626

theorem smallest_number (a b c : ℝ) : 
  c = 2 * a →
  b = 4 * a →
  (a + b + c) / 3 = 77 →
  a = 33 ∧ a ≤ b ∧ a ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3546_354626


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l3546_354635

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ (x y z : ℝ), y = m * x + z + 2 ∧ y = (3 * m - 2) * x + z + 5) ↔ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l3546_354635


namespace NUMINAMATH_CALUDE_distance_between_towns_l3546_354692

/-- The distance between two towns given train speeds and meeting time -/
theorem distance_between_towns (express_speed : ℝ) (speed_difference : ℝ) (meeting_time : ℝ) : 
  express_speed = 80 →
  speed_difference = 30 →
  meeting_time = 3 →
  (express_speed + (express_speed - speed_difference)) * meeting_time = 390 := by
sorry

end NUMINAMATH_CALUDE_distance_between_towns_l3546_354692
