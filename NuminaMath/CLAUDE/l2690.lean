import Mathlib

namespace player_a_not_losing_probability_l2690_269007

theorem player_a_not_losing_probability 
  (p_win : ℝ) 
  (p_draw : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_draw = 0.2) : 
  p_win + p_draw = 0.6 := by
  sorry

end player_a_not_losing_probability_l2690_269007


namespace seulgi_winning_score_l2690_269053

/-- Represents a player's scores in a two-round darts game -/
structure PlayerScores where
  round1 : ℕ
  round2 : ℕ

/-- Calculates the total score for a player -/
def totalScore (scores : PlayerScores) : ℕ :=
  scores.round1 + scores.round2

/-- Theorem: Seulgi needs at least 25 points in the second round to win -/
theorem seulgi_winning_score 
  (hohyeon : PlayerScores) 
  (hyunjeong : PlayerScores)
  (seulgi_round1 : ℕ) :
  hohyeon.round1 = 23 →
  hohyeon.round2 = 28 →
  hyunjeong.round1 = 32 →
  hyunjeong.round2 = 17 →
  seulgi_round1 = 27 →
  ∀ seulgi_round2 : ℕ,
    (totalScore ⟨seulgi_round1, seulgi_round2⟩ > totalScore hohyeon ∧
     totalScore ⟨seulgi_round1, seulgi_round2⟩ > totalScore hyunjeong) →
    seulgi_round2 ≥ 25 :=
by
  sorry


end seulgi_winning_score_l2690_269053


namespace distinct_equals_odd_partitions_l2690_269057

/-- The number of partitions of n into distinct positive integers -/
def distinctPartitions (n : ℕ) : ℕ := sorry

/-- The number of partitions of n into positive odd integers -/
def oddPartitions (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of partitions of n into distinct positive integers
    equals the number of partitions of n into positive odd integers -/
theorem distinct_equals_odd_partitions (n : ℕ+) :
  distinctPartitions n = oddPartitions n := by sorry

end distinct_equals_odd_partitions_l2690_269057


namespace adult_admission_fee_if_all_receipts_from_adults_l2690_269064

/-- Proves that if all receipts came from adult tickets, the adult admission fee would be the total receipts divided by the number of adults -/
theorem adult_admission_fee_if_all_receipts_from_adults 
  (total_attendees : ℕ) 
  (total_receipts : ℚ) 
  (num_adults : ℕ) 
  (h1 : total_attendees = 578)
  (h2 : total_receipts = 985)
  (h3 : num_adults = 342)
  (h4 : num_adults ≤ total_attendees)
  (h5 : num_adults > 0) :
  let adult_fee := total_receipts / num_adults
  adult_fee * num_adults = total_receipts :=
by sorry

#eval (985 : ℚ) / 342

end adult_admission_fee_if_all_receipts_from_adults_l2690_269064


namespace complement_of_union_equals_interval_l2690_269031

-- Define the universal set U
def U : Set ℝ := {x | -5 < x ∧ x < 5}

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- State the theorem
theorem complement_of_union_equals_interval :
  (U \ (A ∪ B)) = {x | -5 < x ∧ x ≤ -2} :=
sorry

end complement_of_union_equals_interval_l2690_269031


namespace color_change_probability_l2690_269011

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDurations where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light cycle -/
def probabilityOfColorChange (d : TrafficLightDurations) : ℚ :=
  let totalCycleDuration := d.green + d.yellow + d.red
  let changeWindowDuration := 3 * d.yellow
  changeWindowDuration / totalCycleDuration

/-- The main theorem stating the probability of observing a color change -/
theorem color_change_probability (d : TrafficLightDurations) 
  (h1 : d.green = 45)
  (h2 : d.yellow = 5)
  (h3 : d.red = 45) :
  probabilityOfColorChange d = 3 / 19 := by
  sorry

#eval probabilityOfColorChange { green := 45, yellow := 5, red := 45 }

end color_change_probability_l2690_269011


namespace charlie_win_probability_l2690_269013

/-- The probability of rolling a six on a standard six-sided die -/
def probSix : ℚ := 1 / 6

/-- The probability of not rolling a six on a standard six-sided die -/
def probNotSix : ℚ := 5 / 6

/-- The number of players in the game -/
def numPlayers : ℕ := 3

/-- The probability that Charlie (the third player) wins the dice game -/
def probCharlieWins : ℚ := 125 / 546

theorem charlie_win_probability :
  probCharlieWins = probSix * (probNotSix^numPlayers / (1 - probNotSix^numPlayers)) :=
sorry

end charlie_win_probability_l2690_269013


namespace divisibility_by_eighteen_l2690_269000

theorem divisibility_by_eighteen (n : Nat) : n ≤ 9 → (315 * 10 + n) % 18 = 0 ↔ n = 0 := by
  sorry

end divisibility_by_eighteen_l2690_269000


namespace cubic_function_properties_monotonic_cubic_function_range_l2690_269043

/-- A cubic function with specified properties -/
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- Theorem for part I -/
theorem cubic_function_properties (a b c d : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c d x = -f a b c d (-x)) →  -- Symmetry about origin
  (f a b c d (1/2) = -1) →                -- Minimum value at x = 1/2
  (f' a b c (1/2) = 0) →                  -- Critical point at x = 1/2
  (f a b c d = f 4 0 (-3) 0) :=
sorry

/-- Theorem for part II -/
theorem monotonic_cubic_function_range (c : ℝ) :
  (∀ x y, x < y → (f 1 1 c 1 x < f 1 1 c 1 y) ∨ (∀ x y, x < y → f 1 1 c 1 x > f 1 1 c 1 y)) →
  c ≥ 1/3 :=
sorry

end cubic_function_properties_monotonic_cubic_function_range_l2690_269043


namespace same_color_probability_l2690_269052

/-- The probability that two remaining chairs are of the same color -/
theorem same_color_probability 
  (black_chairs : ℕ) 
  (brown_chairs : ℕ) 
  (h1 : black_chairs = 15) 
  (h2 : brown_chairs = 18) :
  (black_chairs * (black_chairs - 1) + brown_chairs * (brown_chairs - 1)) / 
  ((black_chairs + brown_chairs) * (black_chairs + brown_chairs - 1)) = 1 / 2 := by
sorry

end same_color_probability_l2690_269052


namespace union_of_A_and_B_l2690_269095

-- Define the sets A and B
def A : Set ℝ := {x | x < 3/2}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end union_of_A_and_B_l2690_269095


namespace work_efficiency_ratio_l2690_269080

/-- The work efficiency of a worker is defined as the fraction of the total work they can complete in one day -/
def work_efficiency (days : ℚ) : ℚ := 1 / days

theorem work_efficiency_ratio 
  (a_and_b_days : ℚ) 
  (b_alone_days : ℚ) 
  (h1 : a_and_b_days = 11) 
  (h2 : b_alone_days = 33) : 
  (work_efficiency a_and_b_days - work_efficiency b_alone_days) / work_efficiency b_alone_days = 2 := by
  sorry

#check work_efficiency_ratio

end work_efficiency_ratio_l2690_269080


namespace arithmetic_sequences_prime_term_l2690_269003

/-- Two arithmetic sequences with their sums -/
def ArithmeticSequences (a b : ℕ → ℕ) (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → S n / T n = (2 * n + 6 : ℚ) / (n + 1 : ℚ)

/-- The m-th term of the second sequence is prime -/
def SecondSequencePrimeTerm (b : ℕ → ℕ) (m : ℕ) : Prop :=
  m > 0 ∧ Nat.Prime (b m)

theorem arithmetic_sequences_prime_term 
  (a b : ℕ → ℕ) (S T : ℕ → ℚ) (m : ℕ) :
  ArithmeticSequences a b S T →
  SecondSequencePrimeTerm b m →
  m = 2 := by
sorry

end arithmetic_sequences_prime_term_l2690_269003


namespace remainder_problem_l2690_269023

theorem remainder_problem (n : ℤ) (h : n % 7 = 2) : (5 * n + 3) % 7 = 6 := by
  sorry

end remainder_problem_l2690_269023


namespace circle_center_and_radius_l2690_269070

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧
    radius = 2 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end circle_center_and_radius_l2690_269070


namespace monotonic_sufficient_not_necessary_l2690_269035

open Set
open Function

-- Define the concept of a monotonic function
def Monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y ∨ f y ≤ f x

-- Define the concept of having a maximum and minimum value on an interval
def HasMaxMin (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ max min : ℝ, (∀ x, a ≤ x ∧ x ≤ b → f x ≤ max) ∧
                 (∀ x, a ≤ x ∧ x ≤ b → min ≤ f x)

theorem monotonic_sufficient_not_necessary (a b : ℝ) (h : a ≤ b) :
  (∀ f : ℝ → ℝ, Monotonic f a b → HasMaxMin f a b) ∧
  (∃ f : ℝ → ℝ, HasMaxMin f a b ∧ ¬Monotonic f a b) :=
sorry

end monotonic_sufficient_not_necessary_l2690_269035


namespace unique_function_property_l2690_269020

theorem unique_function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = f x + 1)
  (h2 : ∀ x, f (x^2) = (f x)^2) :
  ∀ x, f x = x :=
by sorry

end unique_function_property_l2690_269020


namespace triangle_237_not_exists_triangle_555_exists_l2690_269036

/-- Triangle inequality theorem checker -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: A triangle with sides 2, 3, and 7 does not satisfy the triangle inequality -/
theorem triangle_237_not_exists : ¬ (satisfies_triangle_inequality 2 3 7) :=
sorry

/-- Theorem: A triangle with sides 5, 5, and 5 satisfies the triangle inequality -/
theorem triangle_555_exists : satisfies_triangle_inequality 5 5 5 :=
sorry

end triangle_237_not_exists_triangle_555_exists_l2690_269036


namespace terminating_decimals_count_l2690_269083

theorem terminating_decimals_count : 
  let n_count := Finset.filter (fun n => Nat.gcd n 420 % 3 = 0 ∧ Nat.gcd n 420 % 7 = 0) (Finset.range 419)
  Finset.card n_count = 19 := by
  sorry

end terminating_decimals_count_l2690_269083


namespace simplify_fraction_l2690_269078

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by sorry

end simplify_fraction_l2690_269078


namespace assembly_rate_after_transformation_l2690_269058

/-- Represents the factory's car assembly rate before and after transformation -/
structure AssemblyRate where
  before : ℝ
  after : ℝ

/-- The conditions of the problem -/
def problem_conditions (r : AssemblyRate) : Prop :=
  r.after = (5/3) * r.before ∧
  (40 / r.after) = (30 / r.before) - 2

/-- The theorem to prove -/
theorem assembly_rate_after_transformation (r : AssemblyRate) :
  problem_conditions r → r.after = 5 := by sorry

end assembly_rate_after_transformation_l2690_269058


namespace green_eyed_students_l2690_269087

theorem green_eyed_students (total : ℕ) (brown_green : ℕ) (neither : ℕ) 
  (h1 : total = 45)
  (h2 : brown_green = 9)
  (h3 : neither = 5) :
  ∃ (green : ℕ), 
    green = 10 ∧ 
    total = green + 3 * green - brown_green + neither :=
by
  sorry

end green_eyed_students_l2690_269087


namespace complex_addition_simplification_l2690_269045

theorem complex_addition_simplification :
  (7 - 4 * Complex.I) + (3 + 9 * Complex.I) = 10 + 5 * Complex.I :=
by sorry

end complex_addition_simplification_l2690_269045


namespace carpeting_cost_specific_room_l2690_269009

/-- Calculates the cost of carpeting a room given its dimensions and carpet specifications. -/
def carpeting_cost (room_length room_breadth carpet_width_cm carpet_cost_paise : ℚ) : ℚ :=
  let room_area : ℚ := room_length * room_breadth
  let carpet_width_m : ℚ := carpet_width_cm / 100
  let carpet_length : ℚ := room_area / carpet_width_m
  let total_cost_paise : ℚ := carpet_length * carpet_cost_paise
  total_cost_paise / 100

/-- Theorem stating that the cost of carpeting a specific room is 36 rupees. -/
theorem carpeting_cost_specific_room :
  carpeting_cost 15 6 75 30 = 36 := by
  sorry

end carpeting_cost_specific_room_l2690_269009


namespace triangle_third_side_existence_l2690_269082

theorem triangle_third_side_existence (a b c : ℝ) : 
  a = 3 → b = 5 → c = 4 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) := by
  sorry

end triangle_third_side_existence_l2690_269082


namespace remainder_of_87_pow_88_plus_7_mod_88_l2690_269037

theorem remainder_of_87_pow_88_plus_7_mod_88 : 87^88 + 7 ≡ 8 [MOD 88] := by
  sorry

end remainder_of_87_pow_88_plus_7_mod_88_l2690_269037


namespace tan_sixty_degrees_l2690_269062

theorem tan_sixty_degrees : Real.tan (60 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_sixty_degrees_l2690_269062


namespace scaled_variance_l2690_269027

-- Define a dataset type
def Dataset := List Real

-- Define the standard deviation function
noncomputable def standardDeviation (data : Dataset) : Real :=
  sorry

-- Define the variance function
noncomputable def variance (data : Dataset) : Real :=
  sorry

-- Define a function to scale a dataset
def scaleDataset (data : Dataset) (scale : Real) : Dataset :=
  data.map (· * scale)

-- Theorem statement
theorem scaled_variance 
  (data : Dataset) 
  (h : standardDeviation data = 2) : 
  variance (scaleDataset data 2) = 16 := by
  sorry

end scaled_variance_l2690_269027


namespace smallest_composite_no_small_factors_l2690_269056

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p, p < k → Prime p → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 221) ∧
  (has_no_prime_factors_less_than 221 12) ∧
  (∀ m : ℕ, m < 221 → ¬(is_composite m ∧ has_no_prime_factors_less_than m 12)) :=
sorry

end smallest_composite_no_small_factors_l2690_269056


namespace isabellas_travel_l2690_269079

/-- Proves that given the conditions of Isabella's travel and currency exchange, 
    the initial amount d is 120 U.S. dollars. -/
theorem isabellas_travel (d : ℚ) : 
  (8/5 * d - 72 = d) → d = 120 := by
  sorry

end isabellas_travel_l2690_269079


namespace combined_output_in_five_minutes_l2690_269041

/-- The rate at which Machine A fills boxes (boxes per minute) -/
def machine_a_rate : ℚ := 24 / 60

/-- The rate at which Machine B fills boxes (boxes per minute) -/
def machine_b_rate : ℚ := 36 / 60

/-- The combined rate of both machines (boxes per minute) -/
def combined_rate : ℚ := machine_a_rate + machine_b_rate

/-- The time period we're interested in (minutes) -/
def time_period : ℚ := 5

theorem combined_output_in_five_minutes :
  combined_rate * time_period = 5 := by sorry

end combined_output_in_five_minutes_l2690_269041


namespace root_value_theorem_l2690_269018

theorem root_value_theorem (a : ℝ) : 
  (a^2 - 4*a - 6 = 0) → (a^2 - 4*a + 3 = 9) := by
  sorry

end root_value_theorem_l2690_269018


namespace area_of_R_l2690_269076

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The region R -/
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | floor (p.1 ^ 2) = floor p.2 ∧ floor (p.2 ^ 2) = floor p.1}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the area of region R -/
theorem area_of_R : area R = 4 - 2 * Real.sqrt 2 := by sorry

end area_of_R_l2690_269076


namespace triangle_special_angle_l2690_269085

theorem triangle_special_angle (D E F : ℝ) : 
  D + E + F = 180 →  -- sum of angles in a triangle is 180 degrees
  D = E →            -- angles D and E are equal
  F = 2 * D →        -- angle F is twice angle D
  F = 90 :=          -- prove that F is 90 degrees
by sorry

end triangle_special_angle_l2690_269085


namespace triangle_formation_l2690_269030

/-- Triangle inequality theorem: the sum of any two sides must be greater than the third side --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 5 7 ∧
  ¬can_form_triangle 1 3 4 ∧
  ¬can_form_triangle 2 2 7 ∧
  ¬can_form_triangle 3 3 6 :=
sorry

end triangle_formation_l2690_269030


namespace sum_mod_thirteen_l2690_269034

theorem sum_mod_thirteen : (9753 + 9754 + 9755 + 9756) % 13 = 0 := by
  sorry

end sum_mod_thirteen_l2690_269034


namespace quadratic_inequality_solution_set_l2690_269010

/-- Given real numbers m and n where m < n, the quadratic inequality
    x^2 - (m + n)x + mn > 0 has the solution set (-∞, m) ∪ (n, +∞),
    and specifying a = 1 makes this representation unique. -/
theorem quadratic_inequality_solution_set (m n : ℝ) (h : m < n) :
  ∃ (a b c : ℝ), a = 1 ∧
    (∀ x, a * x^2 + b * x + c > 0 ↔ x < m ∨ x > n) ∧
    (b = -(m + n) ∧ c = m * n) :=
sorry

end quadratic_inequality_solution_set_l2690_269010


namespace no_prime_factor_congruent_to_negative_one_mod_eight_l2690_269038

theorem no_prime_factor_congruent_to_negative_one_mod_eight (n : ℕ+) (p : ℕ) 
  (h_prime : Nat.Prime p) (h_cong : p % 8 = 7) : 
  ¬(p ∣ 2^(n.val.succ.succ) + 1) := by
  sorry

end no_prime_factor_congruent_to_negative_one_mod_eight_l2690_269038


namespace intersection_A_complement_B_l2690_269088

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≤ 1}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_A_complement_B_l2690_269088


namespace PQRS_equals_negative_one_l2690_269032

theorem PQRS_equals_negative_one :
  let P : ℝ := Real.sqrt 2007 + Real.sqrt 2008
  let Q : ℝ := -Real.sqrt 2007 - Real.sqrt 2008
  let R : ℝ := Real.sqrt 2007 - Real.sqrt 2008
  let S : ℝ := -Real.sqrt 2008 + Real.sqrt 2007
  P * Q * R * S = -1 := by sorry

end PQRS_equals_negative_one_l2690_269032


namespace smallest_number_remainder_l2690_269060

theorem smallest_number_remainder (N : ℕ) : 
  N = 184 → N % 13 = 2 → N % 15 = 4 := by
sorry

end smallest_number_remainder_l2690_269060


namespace square_side_length_l2690_269042

theorem square_side_length (perimeter : ℝ) (h : perimeter = 100) :
  perimeter / 4 = 25 := by
  sorry

end square_side_length_l2690_269042


namespace parallel_line_equation_perpendicular_line_equation_l2690_269061

-- Define the point P as the intersection of two lines
def P : ℝ × ℝ := (2, 1)

-- Define line l1
def l1 (x y : ℝ) : Prop := 4 * x - y + 1 = 0

-- Define the condition that a line passes through point P
def passes_through_P (a b c : ℝ) : Prop := a * P.1 + b * P.2 + c = 0

-- Theorem for case I
theorem parallel_line_equation :
  ∀ (a b c : ℝ), passes_through_P a b c → (∀ x y, a * x + b * y + c = 0 ↔ 4 * x - y - 7 = 0) → 
  ∃ k, a = 4 * k ∧ b = -k ∧ c = -7 * k := by sorry

-- Theorem for case II
theorem perpendicular_line_equation :
  ∀ (a b c : ℝ), passes_through_P a b c → (∀ x y, a * x + b * y + c = 0 ↔ x + 4 * y - 6 = 0) → 
  ∃ k, a = k ∧ b = 4 * k ∧ c = -6 * k := by sorry

end parallel_line_equation_perpendicular_line_equation_l2690_269061


namespace mikes_typing_speed_l2690_269021

/-- Mike's original typing speed in words per minute -/
def original_speed : ℕ := 65

/-- Mike's reduced typing speed in words per minute -/
def reduced_speed : ℕ := original_speed - 20

/-- Number of words in the document -/
def document_words : ℕ := 810

/-- Time taken to type the document at reduced speed, in minutes -/
def typing_time : ℕ := 18

theorem mikes_typing_speed :
  (reduced_speed * typing_time = document_words) ∧
  (original_speed = 65) := by
  sorry

end mikes_typing_speed_l2690_269021


namespace tonya_stamps_after_trade_l2690_269012

/-- Represents the trade of matchbooks for stamps between Jimmy and Tonya --/
def matchbook_stamp_trade (stamp_match_ratio : ℕ) (matches_per_book : ℕ) (tonya_initial_stamps : ℕ) (jimmy_matchbooks : ℕ) : ℕ :=
  let jimmy_total_matches := jimmy_matchbooks * matches_per_book
  let jimmy_stamps_worth := jimmy_total_matches / stamp_match_ratio
  tonya_initial_stamps - jimmy_stamps_worth

/-- Theorem stating that Tonya will have 3 stamps left after the trade --/
theorem tonya_stamps_after_trade :
  matchbook_stamp_trade 12 24 13 5 = 3 := by
  sorry

end tonya_stamps_after_trade_l2690_269012


namespace polynomial_root_sum_l2690_269029

def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_root_sum (a b c d : ℝ) :
  g a b c d (3*I) = 0 ∧ g a b c d (1 + I) = 0 → a + b + c + d = 9 := by
  sorry

end polynomial_root_sum_l2690_269029


namespace max_product_at_12_l2690_269075

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

def product_of_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a₁^n) * (q^(n * (n - 1) / 2))

theorem max_product_at_12 (a₁ q : ℝ) (h₁ : a₁ = 1536) (h₂ : q = -1/2) :
  product_of_terms a₁ q 12 > product_of_terms a₁ q 9 ∧
  product_of_terms a₁ q 12 > product_of_terms a₁ q 13 := by
  sorry

end max_product_at_12_l2690_269075


namespace sqrt_greater_than_cube_root_l2690_269077

theorem sqrt_greater_than_cube_root (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 + y^2) > (x^3 + y^3)^(1/3) := by
  sorry

end sqrt_greater_than_cube_root_l2690_269077


namespace average_problem_l2690_269015

theorem average_problem (t b c : ℝ) :
  (t + b + c + 14 + 15) / 5 = 12 →
  (t + b + c + 29) / 4 = 15 := by
  sorry

end average_problem_l2690_269015


namespace delta_phi_solution_l2690_269063

def δ (x : ℝ) : ℝ := 4 * x + 5

def φ (x : ℝ) : ℝ := 5 * x + 4

theorem delta_phi_solution :
  ∃ x : ℝ, δ (φ x) = 4 ∧ x = -17/20 := by
  sorry

end delta_phi_solution_l2690_269063


namespace distance_to_gate_l2690_269008

theorem distance_to_gate (field_side : ℝ) (fence_length : ℝ) (gate_distance : ℝ) :
  field_side = 84 →
  fence_length = 91 →
  gate_distance^2 + field_side^2 = fence_length^2 →
  gate_distance = 35 := by
sorry

end distance_to_gate_l2690_269008


namespace partnership_profit_l2690_269017

/-- A partnership business problem -/
theorem partnership_profit (investment_ratio : ℕ) (time_ratio : ℕ) (profit_B : ℕ) : 
  investment_ratio = 5 →
  time_ratio = 3 →
  profit_B = 4000 →
  investment_ratio * time_ratio * profit_B + profit_B = 64000 :=
by
  sorry

end partnership_profit_l2690_269017


namespace mary_has_more_than_marco_l2690_269068

/-- Proves that Mary has $10 more than Marco after transactions --/
theorem mary_has_more_than_marco (marco_initial : ℕ) (mary_initial : ℕ) 
  (marco_gives : ℕ) (mary_spends : ℕ) : ℕ :=
by
  -- Define initial amounts
  have h1 : marco_initial = 24 := by sorry
  have h2 : mary_initial = 15 := by sorry
  
  -- Define amount Marco gives to Mary
  have h3 : marco_gives = marco_initial / 2 := by sorry
  
  -- Define amount Mary spends
  have h4 : mary_spends = 5 := by sorry
  
  -- Calculate final amounts
  let marco_final := marco_initial - marco_gives
  let mary_final := mary_initial + marco_gives - mary_spends
  
  -- Prove Mary has $10 more than Marco
  have h5 : mary_final - marco_final = 10 := by sorry
  
  exact 10

end mary_has_more_than_marco_l2690_269068


namespace smallest_number_with_digit_sum_10_l2690_269066

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → (d = 1 ∨ d = 2)

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_number_with_digit_sum_10 :
  ∀ n : ℕ,
    is_valid_number n ∧ digit_sum n = 10 →
    111111112 ≤ n :=
by sorry

end smallest_number_with_digit_sum_10_l2690_269066


namespace equation_represents_two_lines_l2690_269039

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop :=
  x^2 - 16*y^2 - 8*x + 16 = 0

/-- The first line represented by the equation -/
def line1 (x y : ℝ) : Prop :=
  x = 4 + 4*y

/-- The second line represented by the equation -/
def line2 (x y : ℝ) : Prop :=
  x = 4 - 4*y

/-- Theorem stating that the equation represents two lines -/
theorem equation_represents_two_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y) :=
by sorry

end equation_represents_two_lines_l2690_269039


namespace marbles_exceed_200_on_friday_l2690_269006

def marbles (k : ℕ) : ℕ := 4 * 3^k

theorem marbles_exceed_200_on_friday :
  (∀ j : ℕ, j < 4 → marbles j ≤ 200) ∧ marbles 4 > 200 :=
sorry

end marbles_exceed_200_on_friday_l2690_269006


namespace cylinder_radius_l2690_269022

theorem cylinder_radius (h : ℝ) (r : ℝ) : 
  h = 4 → 
  π * (r + 10)^2 * h = π * r^2 * (h + 10) → 
  r = 4 + 2 * Real.sqrt 14 :=
by sorry

end cylinder_radius_l2690_269022


namespace irrational_floor_congruence_l2690_269014

theorem irrational_floor_congruence (k : ℕ) (h : k ≥ 2) :
  ∃ r : ℝ, Irrational r ∧ ∀ m : ℕ, (⌊r^m⌋ : ℤ) ≡ -1 [ZMOD k] :=
sorry

end irrational_floor_congruence_l2690_269014


namespace early_arrival_time_l2690_269026

/-- 
Given a boy's usual time to reach school and his faster rate relative to his usual rate,
calculate how many minutes earlier he arrives when walking at the faster rate.
-/
theorem early_arrival_time (usual_time : ℝ) (faster_rate_ratio : ℝ) 
  (h1 : usual_time = 28)
  (h2 : faster_rate_ratio = 7/6) : 
  usual_time - (usual_time / faster_rate_ratio) = 4 := by
  sorry

end early_arrival_time_l2690_269026


namespace teachers_survey_l2690_269086

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 90)
  (h_heart_trouble : heart_trouble = 60)
  (h_both : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 20 := by
  sorry

end teachers_survey_l2690_269086


namespace bert_ernie_stamp_ratio_l2690_269001

/-- The number of stamps Peggy has -/
def peggy_stamps : ℕ := 75

/-- The number of stamps Peggy needs to add to match Bert's collection -/
def stamps_to_add : ℕ := 825

/-- The number of stamps Ernie has -/
def ernie_stamps : ℕ := 3 * peggy_stamps

/-- The number of stamps Bert has -/
def bert_stamps : ℕ := peggy_stamps + stamps_to_add

/-- The ratio of Bert's stamps to Ernie's stamps -/
def stamp_ratio : ℚ := bert_stamps / ernie_stamps

theorem bert_ernie_stamp_ratio :
  stamp_ratio = 4 / 1 := by sorry

end bert_ernie_stamp_ratio_l2690_269001


namespace neg_one_quad_residue_iff_prime_mod_four_infinite_primes_mod_four_l2690_269033

/-- For an odd prime p, -1 is a quadratic residue modulo p if and only if p ≡ 1 (mod 4) -/
theorem neg_one_quad_residue_iff_prime_mod_four (p : Nat) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∃ x, x^2 % p = (p - 1) % p) ↔ p % 4 = 1 := by sorry

/-- There are infinitely many prime numbers congruent to 1 modulo 4 -/
theorem infinite_primes_mod_four :
  ∀ n, ∃ p, p > n ∧ Nat.Prime p ∧ p % 4 = 1 := by sorry

end neg_one_quad_residue_iff_prime_mod_four_infinite_primes_mod_four_l2690_269033


namespace debate_team_boys_l2690_269097

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (boys : ℕ) : 
  girls = 46 → 
  groups = 8 → 
  group_size = 9 → 
  boys + girls = groups * group_size → 
  boys = 26 := by sorry

end debate_team_boys_l2690_269097


namespace triangle_ratio_theorem_l2690_269098

theorem triangle_ratio_theorem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  b * Real.cos C + c * Real.sin B = a →
  b = 6 →
  -- Theorem statement
  (a + 2*b) / (Real.sin A + 2 * Real.sin B) = 6 * Real.sqrt 2 := by
  sorry

end triangle_ratio_theorem_l2690_269098


namespace reflection_of_A_l2690_269067

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_of_A : reflect_x (-4, 3) = (-4, -3) := by
  sorry

end reflection_of_A_l2690_269067


namespace volleyball_team_selection_l2690_269089

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 7

theorem volleyball_team_selection :
  (Nat.choose (total_players - quadruplets) starters) +
  (quadruplets * Nat.choose (total_players - quadruplets) (starters - 1)) = 4488 := by
  sorry

end volleyball_team_selection_l2690_269089


namespace jo_age_proof_l2690_269046

theorem jo_age_proof (j d g : ℕ) : 
  (∃ (x y z : ℕ), j = 2 * x ∧ d = 2 * y ∧ g = 2 * z) →  -- ages are even
  j * d * g = 2024 →                                   -- product of ages is 2024
  j ≥ d ∧ j ≥ g →                                      -- Jo's age is the largest
  j = 46 :=                                            -- Jo's age is 46
by sorry

end jo_age_proof_l2690_269046


namespace steps_between_correct_l2690_269016

/-- The number of steps taken from the Empire State Building to Madison Square Garden -/
def steps_between (total_steps down_steps : ℕ) : ℕ :=
  total_steps - down_steps

/-- Theorem stating that the steps between buildings is the difference of total steps and steps down -/
theorem steps_between_correct (total_steps down_steps : ℕ) 
  (h1 : total_steps ≥ down_steps)
  (h2 : total_steps = 991)
  (h3 : down_steps = 676) : 
  steps_between total_steps down_steps = 315 := by
  sorry

end steps_between_correct_l2690_269016


namespace cafe_cake_division_l2690_269059

theorem cafe_cake_division (total_cake : ℚ) (tom_portion bob_portion jerry_portion : ℚ) :
  total_cake = 8/9 →
  tom_portion = 2 * bob_portion →
  tom_portion = 2 * jerry_portion →
  total_cake = tom_portion + bob_portion + jerry_portion →
  bob_portion = 2/9 :=
by sorry

end cafe_cake_division_l2690_269059


namespace x_plus_y_equals_negative_eight_l2690_269069

theorem x_plus_y_equals_negative_eight (x y : ℝ) 
  (h1 : (5 : ℝ)^x = 25^(y+2)) 
  (h2 : (16 : ℝ)^y = 4^(x+4)) : 
  x + y = -8 := by
sorry

end x_plus_y_equals_negative_eight_l2690_269069


namespace angle_BDC_is_20_l2690_269024

-- Define the angles in degrees
def angle_A : ℝ := 70
def angle_E : ℝ := 50
def angle_C : ℝ := 40

-- Theorem to prove
theorem angle_BDC_is_20 : 
  ∃ (angle_BDC : ℝ), angle_BDC = 20 := by
  sorry

end angle_BDC_is_20_l2690_269024


namespace characterization_of_M_inequality_for_M_elements_l2690_269051

-- Define the set M
def M : Set ℝ := {x : ℝ | |2*x - 1| < 1}

-- Theorem 1: Characterization of M
theorem characterization_of_M : M = {x : ℝ | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_for_M_elements (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  a * b + 1 > a + b := by sorry

end characterization_of_M_inequality_for_M_elements_l2690_269051


namespace stones_placement_theorem_l2690_269096

/-- Represents the state of the strip and bag -/
structure GameState where
  stones_in_bag : Nat
  stones_on_strip : List Nat
  deriving Repr

/-- Allowed operations in the game -/
inductive Move
  | PlaceInFirst : Move
  | RemoveFromFirst : Move
  | PlaceInNext (i : Nat) : Move
  | RemoveFromNext (i : Nat) : Move

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.PlaceInFirst => 
      { state with 
        stones_in_bag := state.stones_in_bag - 1,
        stones_on_strip := 1 :: state.stones_on_strip }
  | Move.RemoveFromFirst =>
      { state with 
        stones_in_bag := state.stones_in_bag + 1,
        stones_on_strip := state.stones_on_strip.tail }
  | Move.PlaceInNext i =>
      if i ∈ state.stones_on_strip then
        { state with 
          stones_in_bag := state.stones_in_bag - 1,
          stones_on_strip := (i + 1) :: state.stones_on_strip }
      else state
  | Move.RemoveFromNext i =>
      if i ∈ state.stones_on_strip ∧ (i + 1) ∈ state.stones_on_strip then
        { state with 
          stones_in_bag := state.stones_in_bag + 1,
          stones_on_strip := state.stones_on_strip.filter (· ≠ i + 1) }
      else state

/-- Checks if it's possible to reach a certain cell number -/
def canReachCell (n : Nat) : Prop :=
  ∃ (moves : List Move), 
    let finalState := moves.foldl applyMove { stones_in_bag := 10, stones_on_strip := [] }
    n ∈ finalState.stones_on_strip

theorem stones_placement_theorem : 
  ∀ n : Nat, n ≤ 1023 → canReachCell n :=
by sorry

end stones_placement_theorem_l2690_269096


namespace sequence_existence_l2690_269091

theorem sequence_existence : ∃ (a : ℕ → ℕ) (M : ℕ), 
  (∀ n, a n ≤ a (n + 1)) ∧ 
  (∀ k, ∃ n, a n > k) ∧
  (∀ n ≥ M, ¬(Nat.Prime (n + 1)) → 
    ∀ p, Nat.Prime p → p ∣ (Nat.factorial n + 1) → p > n + a n) := by
  sorry

end sequence_existence_l2690_269091


namespace apple_box_weight_proof_l2690_269049

/-- The number of apple boxes -/
def num_boxes : ℕ := 7

/-- The number of boxes whose initial weight equals the final weight of all boxes -/
def num_equal_boxes : ℕ := 3

/-- The amount of apples removed from each box (in kg) -/
def removed_weight : ℕ := 20

/-- The initial weight of apples in each box (in kg) -/
def initial_weight : ℕ := 35

theorem apple_box_weight_proof :
  initial_weight * num_boxes - removed_weight * num_boxes = initial_weight * num_equal_boxes :=
by sorry

end apple_box_weight_proof_l2690_269049


namespace vector_equation_sum_l2690_269040

/-- Given vectors a, b, c in R², if a = xb + yc for some real x and y, then x + y = 0 -/
theorem vector_equation_sum (a b c : Fin 2 → ℝ)
    (ha : a = ![3, -1])
    (hb : b = ![-1, 2])
    (hc : c = ![2, 1])
    (x y : ℝ)
    (h : a = x • b + y • c) :
  x + y = 0 := by
  sorry

end vector_equation_sum_l2690_269040


namespace expansion_properties_l2690_269099

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The expansion of (x^4 + 1/x)^n -/
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

theorem expansion_properties (n : ℕ) :
  (binomial n 2 - binomial n 1 = 35) →
  (n = 10 ∧ 
   ∃ (c : ℝ), c = (expansion n 1) ∧ c = 45) := by sorry

end expansion_properties_l2690_269099


namespace greatest_integer_b_for_quadratic_range_l2690_269073

theorem greatest_integer_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 5 > -25) ↔ b ≤ 10 :=
sorry

end greatest_integer_b_for_quadratic_range_l2690_269073


namespace rajans_position_l2690_269072

/-- Given a row of boys, this theorem proves Rajan's position from the left end. -/
theorem rajans_position 
  (total_boys : ℕ) 
  (vinays_position_from_right : ℕ) 
  (boys_between : ℕ) 
  (h1 : total_boys = 24) 
  (h2 : vinays_position_from_right = 10) 
  (h3 : boys_between = 8) : 
  total_boys - (vinays_position_from_right - 1 + boys_between + 1) = 6 := by
sorry

end rajans_position_l2690_269072


namespace solve_equation_l2690_269055

/-- Custom operation # -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem statement -/
theorem solve_equation (x : ℝ) (h : hash x 7 = 63) : x = 3 := by
  sorry

end solve_equation_l2690_269055


namespace perfect_div_by_three_perfect_div_by_seven_l2690_269065

/-- Definition of a perfect number -/
def isPerfect (n : ℕ) : Prop :=
  n > 0 ∧ n = (Finset.filter (· < n) (Finset.range (n + 1))).sum id

/-- Theorem for perfect numbers divisible by 3 -/
theorem perfect_div_by_three (n : ℕ) (h1 : isPerfect n) (h2 : n > 6) (h3 : 3 ∣ n) : 9 ∣ n := by
  sorry

/-- Theorem for perfect numbers divisible by 7 -/
theorem perfect_div_by_seven (n : ℕ) (h1 : isPerfect n) (h2 : n > 28) (h3 : 7 ∣ n) : 49 ∣ n := by
  sorry

end perfect_div_by_three_perfect_div_by_seven_l2690_269065


namespace tree_height_after_two_years_l2690_269071

/-- Represents the height of a tree that quadruples each year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (4 ^ years)

/-- The problem statement -/
theorem tree_height_after_two_years 
  (h : tree_height 1 4 = 256) : 
  tree_height 1 2 = 16 := by
  sorry

end tree_height_after_two_years_l2690_269071


namespace sashas_initial_questions_l2690_269081

/-- Proves that given Sasha's completion rate, work time, and remaining questions,
    the initial number of questions is 60. -/
theorem sashas_initial_questions
  (completion_rate : ℕ)
  (work_time : ℕ)
  (remaining_questions : ℕ)
  (h1 : completion_rate = 15)
  (h2 : work_time = 2)
  (h3 : remaining_questions = 30) :
  completion_rate * work_time + remaining_questions = 60 :=
by
  sorry

end sashas_initial_questions_l2690_269081


namespace smallest_number_proof_l2690_269092

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 24 →
  b = 25 →
  max a (max b c) = b + 6 →
  min a (min b c) = 16 :=
by sorry

end smallest_number_proof_l2690_269092


namespace min_sum_squares_l2690_269054

theorem min_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) 
  (heq : a^2 - 2015*a = b^2 - 2015*b) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → x^2 - 2015*x = y^2 - 2015*y → 
  a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = (2015^2) / 2 := by
sorry

end min_sum_squares_l2690_269054


namespace fruit_drink_total_volume_l2690_269093

/-- A fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_total_volume (drink : FruitDrink) 
  (h1 : drink.orange_percent = 0.15)
  (h2 : drink.watermelon_percent = 0.60)
  (h3 : drink.grape_ounces = 30) :
  (drink.grape_ounces / (1 - drink.orange_percent - drink.watermelon_percent)) = 120 := by
  sorry

end fruit_drink_total_volume_l2690_269093


namespace curve_and_line_properties_l2690_269050

-- Define the unit circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the stretched curve C₂
def C₂ (x y : ℝ) : Prop := (x / Real.sqrt 3)^2 + (y / 2)^2 = 1

-- Define the line l
def l (x y : ℝ) : Prop := 2 * x - y - 6 = 0

-- Theorem statement
theorem curve_and_line_properties :
  -- 1. Parametric equations of C₂
  (∀ φ : ℝ, C₂ (Real.sqrt 3 * Real.cos φ) (2 * Real.sin φ)) ∧
  -- 2. Point P(-3/2, 1) on C₂ has maximum distance to l
  (C₂ (-3/2) 1 ∧
   ∀ x y : ℝ, C₂ x y →
     (x + 3/2)^2 + (y - 1)^2 ≤ (2 * Real.sqrt 5)^2) ∧
  -- 3. Maximum distance from C₂ to l is 2√5
  (∃ x y : ℝ, C₂ x y ∧
    |2*x - y - 6| / Real.sqrt 5 = 2 * Real.sqrt 5 ∧
    ∀ x' y' : ℝ, C₂ x' y' →
      |2*x' - y' - 6| / Real.sqrt 5 ≤ 2 * Real.sqrt 5) := by
  sorry

end curve_and_line_properties_l2690_269050


namespace arthur_spent_fraction_l2690_269028

theorem arthur_spent_fraction (initial_amount : ℚ) (remaining_amount : ℚ) 
  (h1 : initial_amount = 200)
  (h2 : remaining_amount = 40) :
  (initial_amount - remaining_amount) / initial_amount = 4/5 := by
  sorry

end arthur_spent_fraction_l2690_269028


namespace first_box_weight_l2690_269084

theorem first_box_weight (total_weight second_weight third_weight : ℕ) 
  (h1 : total_weight = 18)
  (h2 : second_weight = 11)
  (h3 : third_weight = 5)
  : ∃ first_weight : ℕ, first_weight + second_weight + third_weight = total_weight ∧ first_weight = 2 := by
  sorry

end first_box_weight_l2690_269084


namespace arithmetic_sequence_problem_l2690_269094

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

-- State the theorem
theorem arithmetic_sequence_problem (a₁ d : ℤ) (h_d : d ≠ 0) :
  (∃ r : ℚ, (arithmetic_sequence a₁ d 2 + 1) ^ 2 = (arithmetic_sequence a₁ d 1 + 1) * (arithmetic_sequence a₁ d 4 + 1)) →
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 = -12 →
  ∀ n : ℕ, arithmetic_sequence a₁ d n = -2 * n - 1 := by
  sorry

end arithmetic_sequence_problem_l2690_269094


namespace star_equation_solution_l2690_269005

/-- The "※" operation for positive real numbers -/
def star (a b : ℝ) : ℝ := a * b + a + b^2

/-- Theorem: If 1※k = 3, then k = 1 -/
theorem star_equation_solution (k : ℝ) (hk : k > 0) (h : star 1 k = 3) : k = 1 := by
  sorry

#check star_equation_solution

end star_equation_solution_l2690_269005


namespace area_of_region_l2690_269002

-- Define the region
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 ≤ 2 ∧ abs p.2 ≤ 2 ∧ abs (abs p.1 - abs p.2) ≤ 1}

-- State the theorem
theorem area_of_region : MeasureTheory.volume R = 12 := by
  sorry

end area_of_region_l2690_269002


namespace max_product_arithmetic_sequence_l2690_269048

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem max_product_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a6 : a 6 = 4) :
  (∃ x : ℝ, a 4 * a 7 ≤ x) ∧ a 4 * a 7 ≤ 18 ∧ (∃ d : ℝ, a 4 * a 7 = 18) :=
sorry

end max_product_arithmetic_sequence_l2690_269048


namespace shirt_price_l2690_269004

/-- The cost of a pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of a shirt -/
def shirt_cost : ℝ := sorry

/-- The total cost of 3 pairs of jeans and 2 shirts is $69 -/
axiom first_purchase : 3 * jeans_cost + 2 * shirt_cost = 69

/-- The total cost of 2 pairs of jeans and 3 shirts is $86 -/
axiom second_purchase : 2 * jeans_cost + 3 * shirt_cost = 86

/-- The cost of one shirt is $24 -/
theorem shirt_price : shirt_cost = 24 := by sorry

end shirt_price_l2690_269004


namespace correct_plates_removed_l2690_269090

/-- The number of plates that need to be removed to reach the acceptable weight -/
def plates_to_remove : ℕ :=
  let initial_plates : ℕ := 38
  let plate_weight : ℕ := 10  -- in ounces
  let max_weight_lbs : ℕ := 20
  let max_weight_oz : ℕ := max_weight_lbs * 16
  let total_weight : ℕ := initial_plates * plate_weight
  let excess_weight : ℕ := total_weight - max_weight_oz
  excess_weight / plate_weight

theorem correct_plates_removed : plates_to_remove = 6 := by
  sorry

end correct_plates_removed_l2690_269090


namespace elliptical_machine_cost_l2690_269019

/-- The cost of an elliptical machine -/
def machine_cost : ℝ := 120

/-- The daily minimum payment -/
def daily_minimum_payment : ℝ := 6

/-- The number of days to pay the remaining balance -/
def payment_days : ℕ := 10

/-- Theorem stating the cost of the elliptical machine -/
theorem elliptical_machine_cost :
  (machine_cost / 2 = daily_minimum_payment * payment_days) ∧
  (machine_cost / 2 = machine_cost - machine_cost / 2) := by
  sorry

#check elliptical_machine_cost

end elliptical_machine_cost_l2690_269019


namespace positive_number_property_l2690_269044

theorem positive_number_property (x : ℝ) (h1 : x > 0) (h2 : 0.01 * x^2 + 16 = 36) : x = 20 * Real.sqrt 5 := by
  sorry

end positive_number_property_l2690_269044


namespace sufficient_not_necessary_condition_l2690_269025

theorem sufficient_not_necessary_condition :
  (∀ x y : ℝ, x > 3 ∧ y > 3 → x + y > 6) ∧
  (∃ x y : ℝ, x + y > 6 ∧ ¬(x > 3 ∧ y > 3)) :=
by sorry

end sufficient_not_necessary_condition_l2690_269025


namespace max_roses_for_680_l2690_269074

/-- Represents the pricing options for roses -/
structure RosePrices where
  individual : ℝ  -- Price of an individual rose
  dozen : ℝ       -- Price of a dozen roses
  twoDozen : ℝ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def maxRoses (budget : ℝ) (prices : RosePrices) : ℕ :=
  sorry

/-- Theorem stating that given specific pricing options and a budget of $680, the maximum number of roses that can be purchased is 325 -/
theorem max_roses_for_680 :
  let prices : RosePrices := { individual := 2.30, dozen := 36, twoDozen := 50 }
  maxRoses 680 prices = 325 := by
  sorry

end max_roses_for_680_l2690_269074


namespace visited_neither_country_l2690_269047

theorem visited_neither_country (total : ℕ) (visited_iceland : ℕ) (visited_norway : ℕ) (visited_both : ℕ)
  (h1 : total = 50)
  (h2 : visited_iceland = 25)
  (h3 : visited_norway = 23)
  (h4 : visited_both = 21) :
  total - (visited_iceland + visited_norway - visited_both) = 23 :=
by sorry

end visited_neither_country_l2690_269047
